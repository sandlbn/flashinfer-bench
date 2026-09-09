"""Per-part constants, measured and cached rather than written down.

Thresholds like "what a substitution costs" or "where the timer's floor is" are properties
of one GPU and one software stack, not of this project. Written into a default they are
wrong on every other part, and they drift on the part they came from: figures hand-measured
here had moved -- one of them by most of its value -- when re-measured on the same machine
a day later.

So nothing stores them. A caller asks for the value, it is measured once per (part, timer,
stack) and cached on disk, and a new chip gets its own answer with no edit.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import json
import logging
import math
import os
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

CACHE_VERSION = 4
"""Bumped when a stored record can no longer be trusted.

v1 records wrote an unmeasurable dispatch cost as ``0.0`` -- the claim that a substitution
is free -- so a v1 file is never read. v2 records predate ``launch_floor_us`` and
``matmul_peak_tflops``; a reader that filled those in with a default would be classifying
regimes against a number nobody measured, so a v2 file is not read either. v3 records
predate ``channel_period_bytes``; a period filled in from anywhere but the sweep would pad
weights that do not camp and miss ones that do, so a v3 file is not read.
"""

MATMUL_DTYPES = ("bfloat16", "float16", "float32")
"""The dtypes a matrix-throughput probe is attempted at, narrowed to those the part runs
natively (:meth:`Capabilities.is_native_dtype`). A closed list of what ``torch.matmul``
accepts as a real-valued operand, not a claim about any part."""


@dataclass(frozen=True)
class Calibration:
    """What this part costs, in microseconds unless stated."""

    hardware_id: str
    timer: str
    dispatch_us: Optional[float]
    """What one apply() call costs over invoking the chosen kernel the way the benchmark did.

    The part of a substitution's per-call price that no trace contains: resolving the
    definition, merging kwargs, building the key, checking dtypes, looking up the table and
    the build cache, allocating outputs for a destination-passing kernel, and returning
    through the Runnable. A kernel saving less than this loses the exchange however large
    its ratio.

    ``None`` means it could not be measured on this machine. It is never ``0.0`` from a
    failure path: a caller that reads ``None`` must treat the apply() mechanism as
    unavailable, not as free -- a gate set to zero admits every substitution.
    """
    timing_floor_us: float
    """Fixed cost a single event-timed call carries. Ratios below it measure the harness."""
    bandwidth_gbs: float
    """Contiguous read bandwidth. A kernel's own ceiling is lower; measure its access
    pattern rather than treating this as a bound (:func:`strided_read_bandwidth_gbs`)."""
    launch_floor_us: Optional[float]
    """Per-launch cost of the smallest kernel this stack can put on the device.

    Measured inside a batch of back-to-back launches under sustained load, so it is what
    one launch costs when nothing but launching is happening -- distinct from
    ``timing_floor_us``, which is what one *timed region* carries over its contents. A
    kernel whose device time sits at this floor is launch-bound: no change inside it can
    win, only removing the launch can.

    ``None`` when the rounds did not settle; never a default.
    """
    matmul_peak_tflops: Dict[str, Optional[float]]
    """Achieved matrix throughput per dtype name, from a square matmul of
    :data:`MATMUL_PROBE_SIDE` under sustained load. The denominator of ``t_cmp = flops /
    peak``. Only dtypes the part runs natively are probed; a dtype whose rounds did not
    settle maps to ``None``. What ``torch.matmul`` reaches through this stack, not a
    datasheet figure."""
    channel_period_bytes: Optional[int]
    """Distance in bytes at which row pitches put every row on the same memory channel.

    Found by sweeping the row pitch of a streaming read and taking the spacing of the
    pitches at which it is slow (:func:`measure_channel_period_bytes`). A 2-D weight whose
    row pitch is a multiple of it streams at a fraction of the bandwidth of one whose pitch
    is not, and the load-time row pad in ``flashinfer_bench.integration.weight_layout`` is
    keyed on it. The spacing is what the transform needs; whether it factors into a channel
    count times an interleave granule is an interpretation the sweep does not make.

    ``None`` when the sweep resolved no period -- its pitches did not settle, or no pitch in
    its range was slow -- and never a default: a caller that reads ``None`` must treat the
    row-pad transform as unavailable, since a period taken from another part pads the wrong
    weights and nothing downstream can tell.
    """

    @property
    def complete(self) -> bool:
        """Whether every field was measured. Only a complete record is worth caching: an
        unmeasured cost is not a result to remember, and the next process gets another go."""
        return (
            self.dispatch_us is not None
            and self.launch_floor_us is not None
            and all(v is not None for v in self.matmul_peak_tflops.values())
            and self.channel_period_bytes is not None
        )


DISPATCH_SPREAD_TOLERANCE = 0.10
"""Largest round-to-round spread (median absolute deviation over the median) a dispatch
measurement may carry and still be reported.

A property of the estimator, not of any part. On an idle box the paired rounds agree to
1-3%; a thread that is bounced between core classes or shares a core with another process
reads tens of percent, and its median is not a number about this machine but about that
moment. Below the tolerance the measurement is settled; above it, it is refused.
"""


DEVICE_SPREAD_TOLERANCE = 0.05
"""Largest window spread (median absolute deviation over the median) a device-side
measurement may carry and still be reported.

A property of the estimator, not of any part. Back-to-back regions of the same work on a
device at steady clocks agree to a percent or two; while the clock is still ramping, or
while another process shares the device, consecutive regions disagree by far more, and a
median over that describes the moment rather than the machine.
"""

SETTLE_BUDGET_S = 4.0
"""Longest a device-side measurement keeps sampling while waiting for its window to settle
before it gives up and reports ``None``. Bounds the cost of an unsettleable measurement on
a shared device; it does not set the warm-up, which ends the moment the window agrees."""

MATMUL_PROBE_SIDE = 4096
"""Side of the square matmul the matrix-throughput probe times. An estimator parameter: the
field records what this size achieved, and the docstring of ``matmul_peak_tflops`` says so."""

PATTERN_BUFFER_BYTES = 256 * 1024 * 1024
"""Size of the buffer the strided-read probe sweeps, chosen to exceed any last-level cache
this project targets so the number is memory bandwidth and not cache bandwidth."""

CHANNEL_PERIOD_PROBE_ROWS = 4096
"""Rows the channel-period probe reads at every pitch.

Fixed across the sweep so that the kernel -- its work-group count, how it splits the rows --
is identical at every pitch and only the addresses differ. A reduction whose row count
followed the pitch changes its own schedule from one pitch to the next and buries the
address effect under that.
"""

CHANNEL_PERIOD_PROBE_RUN_BYTES = 2048
"""Contiguous bytes the probe reads at the start of every row.

The column window it holds in flight across many rows at once, which is how a decode GEMM
reads a weight. It also bounds what the sweep can resolve: a period no longer than the run
is covered by every row, no pitch is slow, and the sweep reports none found.
"""

CHANNEL_PERIOD_PITCH_STEP_BYTES = 256
"""Spacing of the pitches swept. A period is resolved to a multiple of it."""

CHANNEL_PERIOD_MAX_PITCH_BYTES = 32 * 1024
"""Largest pitch swept. A period has to repeat inside the range to be one, so the largest
resolvable is half of this."""

CHANNEL_PERIOD_PROBE_CALLS = 20
"""Probe reads per timed region: enough that a region is long against the host's
synchronisation jitter, so the regions can agree to :data:`DEVICE_SPREAD_TOLERANCE`."""

CHANNEL_PERIOD_WARMUP_S = 0.3
"""Sustained load over the whole sweep before any pitch is timed, so the part has left its
gated clock before the first pitch rather than during the tenth."""

CHANNEL_PERIOD_PITCH_BUDGET_S = 0.5
"""Longest one pitch keeps sampling for its window to settle. Shorter than
:data:`SETTLE_BUDGET_S` because there are a hundred-odd pitches and the device is shared."""

CHANNEL_PERIOD_SWEEP_BUDGET_S = 10.0
"""Longest the whole sweep runs. Pitches not reached by then are unmeasured, not slow."""

CAMPING_DEFICIT = 0.20
"""How much longer than its neighbours a pitch's read must take to count as camping.

A property of the estimator, set from the shape of the curve and not from any part's
numbers. The curve is a flat baseline with isolated spikes: the spikes are a fraction of
the memory channels idling, which costs a large fraction of the read, while a pitch at a
fraction of the period and the sweep's own scatter cost a few percent. A fifth sits between
the two with room on both sides.
"""

CHANNEL_PERIOD_NEIGHBOURHOOD = 4
"""Measured pitches on each side that a pitch is compared against.

The baseline is local so that a slow drift over the sweep -- a clock still settling,
another process on the device -- is not read as a slow pitch. A resolvable period is longer
than the run, which is longer than this many steps, so a neighbourhood holds at most one
camping pitch and its median never is one.
"""


def _cache_path(hardware_id: str, timer: str) -> pathlib.Path:
    root = os.environ.get("FIB_CACHE_PATH") or os.path.expanduser("~/.cache/flashinfer_bench")
    return pathlib.Path(root) / "calibration" / f"v{CACHE_VERSION}-{hardware_id}-{timer}.json"


def _pattern_cache_path(hardware_id: str, timer: str) -> pathlib.Path:
    """Sidecar for access-pattern bandwidths: patterns are open-ended, so they are not fields."""
    return _cache_path(hardware_id, timer).with_name(
        f"v{CACHE_VERSION}-{hardware_id}-{timer}-patterns.json"
    )


def _settled_device_us(
    fn: Callable[[], object],
    sync: Callable[[], None],
    calls: int,
    window: int = 5,
    tolerance: float = DEVICE_SPREAD_TOLERANCE,
    budget_s: float = SETTLE_BUDGET_S,
    clock: Callable[[], float] = time.perf_counter,
) -> Optional[float]:
    """Per-call microseconds of `fn` once the device is in a steady state, or None.

    Warm-up is by time and open-ended rather than by count. This part gates its clock within
    tens of milliseconds of the queue draining and needs sustained load to come back, so a
    fixed number of warm-up calls measures the idle clock on a fast kernel and wastes time
    on a slow one. Regions of `calls` launches run back-to-back with no pause between them;
    the measurement is accepted the first time the last `window` regions agree within
    `tolerance`, which a ramping clock cannot do, and refused after `budget_s` if they never
    do. The first call runs before the budget starts so a one-time JIT compile is not
    charged against it.

    Never returns zero (see :func:`_settled_median`).
    """
    fn()
    sync()
    samples: List[float] = []
    deadline = clock() + budget_s
    while True:
        sync()
        start = clock()
        for _ in range(calls):
            fn()
        sync()
        samples.append((clock() - start) / calls * 1e6)
        if len(samples) >= window:
            settled = _settled_median(samples[-window:], tolerance)
            if settled is not None:
                return settled
        if clock() >= deadline:
            logger.debug(
                "device measurement did not settle in %.1fs (%d regions, last %s)",
                budget_s,
                len(samples),
                [f"{s:.1f}" for s in samples[-window:]],
            )
            return None


def _median_us(
    fn: Callable[[], object], sync: Callable[[], None], calls: int, rounds: int = 5
) -> float:
    for _ in range(max(20, calls)):
        fn()
    sync()
    samples = []
    for _ in range(rounds):
        sync()
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        sync()
        samples.append((time.perf_counter() - start) / calls * 1e6)
    return statistics.median(samples)


def _settled_median(
    deltas: Sequence[float], tolerance: float = DISPATCH_SPREAD_TOLERANCE
) -> Optional[float]:
    """The median of per-round differences, or None if the rounds do not agree.

    Refuses a non-positive median outright -- the arms are indistinguishable or the slower
    one measured faster, and either way there is no cost to report -- and refuses a positive
    one whose round-to-round spread exceeds ``tolerance`` of it. Never returns zero: zero is
    a claim that the wrapper is free, and a deploy gate set from it admits everything.
    """
    if not deltas:
        return None
    median = statistics.median(deltas)
    spread = statistics.median(abs(d - median) for d in deltas)
    if median <= 0 or spread > tolerance * median:
        logger.debug(
            "delta %.2fus with spread %.2fus over %d rounds: not settled",
            median,
            spread,
            len(deltas),
        )
        return None
    logger.debug("delta %.2fus, spread %.2fus over %d rounds", median, spread, len(deltas))
    return median


def _paired_delta_us(
    slow: Callable[[], object],
    fast: Callable[[], object],
    calls: int,
    rounds: int = 15,
    sync: Optional[Callable[[], None]] = None,
    warmup_s: float = 0.05,
    tolerance: float = DISPATCH_SPREAD_TOLERANCE,
) -> Optional[float]:
    """How much longer `slow` takes than `fast` per call, or None if that did not settle.

    Measured as a difference per round rather than as two separate medians. Whatever drifts
    over the measurement -- a CPU leaving its idle frequency, a device ramping its clocks --
    is charged to whichever arm ran first when the arms are timed one after the other, and
    a few microseconds of real difference are easily swamped or inverted. Alternating the
    arms inside each round (ABBA) cancels the drift both arms share.

    Warm-up is by time rather than by count so that a host at its idle frequency has left it
    before the first timed round, whatever one call costs.

    Returns None rather than zero when the rounds do not agree (see :func:`_settled_median`).
    """
    sync = sync or (lambda: None)
    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        for _ in range(calls):
            slow()
            fast()
    sync()

    deltas = []
    for _ in range(rounds):
        timings = []
        for fn in (fast, slow, slow, fast):
            sync()
            start = time.perf_counter()
            for _ in range(calls):
                fn()
            sync()
            timings.append((time.perf_counter() - start) / calls * 1e6)
        deltas.append((timings[1] + timings[2]) / 2 - (timings[0] + timings[3]) / 2)
    return _settled_median(deltas, tolerance)


def measure(device: str) -> Calibration:
    """Measure this part. Prefer :func:`get`, which caches."""
    import torch

    from flashinfer_bench.device import get_accelerator

    accel = get_accelerator(device)
    caps = accel.capabilities(device)
    backend = torch.xpu if device.startswith("xpu") else torch.cuda

    def sync() -> None:
        backend.synchronize()

    def _bandwidth() -> float:
        buf = torch.randn(256 * 1024 * 1024 // 2, dtype=torch.float16, device=device)
        us = _median_us(lambda: buf.sum(), sync, 10)
        return (buf.numel() * 2) / (us * 1e-6) / 1e9

    bandwidth = _bandwidth()

    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    tiny = lambda: torch.mul(x, 1.0001, out=out)  # noqa: E731
    batched = _median_us(tiny, sync, 200)
    singles = []
    for _ in range(5):
        sync()
        a, b = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
        a.record()
        tiny()
        b.record()
        sync()
        singles.append(a.elapsed_time(b) * 1000)
    floor = max(0.0, statistics.median(singles) - batched)

    launch = measure_launch_floor_us(device)
    if launch is None:
        logger.warning("Launch floor did not settle on %s; launch_floor_us is None.", device)
    peak = measure_matmul_peak_tflops(device)
    for name, value in peak.items():
        if value is None:
            logger.warning("Matrix throughput at %s did not settle on %s.", name, device)

    period = measure_channel_period_bytes(device)
    if period is None:
        logger.warning(
            "No memory channel period resolved on %s; channel_period_bytes is None. The "
            "row-pad weight transform is unavailable here, not keyed on a borrowed period.",
            device,
        )

    dispatch = measure_dispatch_us(device)
    if dispatch is None:
        logger.warning(
            "Could not measure what an apply() substitution costs on %s; dispatch_us is None. "
            "Treat the apply() mechanism as unavailable here, not as free.",
            device,
        )
    return Calibration(
        hardware_id=caps.canonical_id,
        timer=accel.make_timer(device).name,
        dispatch_us=dispatch,
        timing_floor_us=floor,
        bandwidth_gbs=bandwidth,
        launch_floor_us=launch,
        matmul_peak_tflops=peak,
        channel_period_bytes=period,
    )


def _backend(device: str):
    import torch

    return torch.xpu if device.startswith("xpu") else torch.cuda


def measure_launch_floor_us(device: str) -> Optional[float]:
    """What one launch of the smallest possible kernel costs inside a batch, or None.

    A one-element elementwise op, hundreds of times back-to-back, host-timed around the
    synced batch: per launch this is the larger of what the host takes to submit one and
    what the device takes to run one, which is exactly the floor no per-call time through
    this stack can go under. Warm-up and acceptance are :func:`_settled_device_us`'s.
    """
    import torch

    backend = _backend(device)
    x = torch.randn(1, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    return _settled_device_us(
        lambda: torch.mul(x, 1.0001, out=out), lambda: backend.synchronize(), calls=500
    )


def measure_matmul_peak_tflops(
    device: str, dtypes: Optional[Sequence[str]] = None
) -> Dict[str, Optional[float]]:
    """Achieved TFLOP/s of a :data:`MATMUL_PROBE_SIDE` square matmul per native dtype.

    Each dtype is timed under sustained load and accepted only once its regions agree
    (:func:`_settled_device_us`); a dtype that never settles maps to ``None`` rather than to
    whatever the clock ramp produced. Dtypes the part only emulates are not probed: a
    throughput measured on an emulation is a property of the emulation.
    """
    import torch

    from flashinfer_bench.device import get_accelerator

    caps = get_accelerator(device).capabilities(device)
    backend = _backend(device)
    n = MATMUL_PROBE_SIDE
    result: Dict[str, Optional[float]] = {}
    for name in dtypes or MATMUL_DTYPES:
        if not caps.is_native_dtype(name):
            continue
        dtype = getattr(torch, name)
        a = torch.randn(n, n, dtype=dtype, device=device)
        b = torch.randn(n, n, dtype=dtype, device=device)
        c = torch.empty(n, n, dtype=dtype, device=device)

        def _mm(a=a, b=b, c=c):
            return torch.matmul(a, b, out=c)

        us = _settled_device_us(_mm, lambda: backend.synchronize(), calls=3)
        result[name] = None if us is None else (2.0 * n**3) / (us * 1e-6) / 1e12
        del a, b, c
    return result


def _measure_strided_read_gbs(device: str, run_bytes: int, stride_bytes: int) -> Optional[float]:
    import torch

    backend = _backend(device)
    itemsize = 2
    buf = torch.randn(PATTERN_BUFFER_BYTES // itemsize, dtype=torch.float16, device=device)
    stride_e, run_e = stride_bytes // itemsize, run_bytes // itemsize
    rows = buf.numel() // stride_e
    view = buf[: rows * stride_e].view(rows, stride_e)[:, :run_e]
    us = _settled_device_us(lambda: view.sum(), lambda: backend.synchronize(), calls=10)
    if us is None:
        return None
    return (rows * run_bytes) / (us * 1e-6) / 1e9


def strided_read_bandwidth_gbs(
    device: str, run_bytes: int, stride_bytes: int, refresh: bool = False
) -> Optional[float]:
    """Bandwidth, in GB/s of *useful* bytes, of reading `run_bytes` out of every `stride_bytes`.

    The ``bw_pattern`` of the regime classification: ``bandwidth_gbs`` is what a contiguous
    stream reaches, and a kernel obliged to touch memory in runs shorter than that -- a
    column of a row-major matrix, one head out of an interleaved row, a gathered table --
    reaches less. ``run_bytes == stride_bytes`` is the contiguous case and should agree with
    ``bandwidth_gbs``. Measured once per (part, timer, pattern) and cached in a sidecar of
    the calibration record; ``None`` when it did not settle, and never cached as such.
    """
    itemsize = 2
    if run_bytes <= 0 or stride_bytes < run_bytes:
        raise ValueError(f"need 0 < run_bytes <= stride_bytes, got {run_bytes}:{stride_bytes}")
    if run_bytes % itemsize or stride_bytes % itemsize:
        raise ValueError(f"run and stride must be multiples of {itemsize} bytes")
    if stride_bytes > PATTERN_BUFFER_BYTES:
        raise ValueError(f"stride {stride_bytes} exceeds the probe buffer ({PATTERN_BUFFER_BYTES})")
    try:
        from flashinfer_bench.device import get_accelerator

        accel = get_accelerator(device)
        path = _pattern_cache_path(accel.canonical_id(device), accel.make_timer(device).name)
    except Exception:
        return None
    key = f"{run_bytes}:{stride_bytes}"
    table: Dict[str, float] = {}
    if path.exists():
        try:
            table = json.loads(path.read_text())
        except Exception:
            table = {}
    if not refresh and key in table:
        return table[key]
    try:
        gbs = _measure_strided_read_gbs(device, run_bytes, stride_bytes)
    except Exception as exc:
        logger.debug("strided read %s failed on %s: %s", key, device, exc)
        return None
    if gbs is None:
        return None
    table[key] = gbs
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=2) + "\n")
    except OSError:
        pass
    return gbs


AUTHORED_PROBE_BLOCKS = (1024, 2048, 4096, 8192)
"""Elements per program the authored-stream probe is swept over. An estimator parameter:
the probe reports the best of the sweep, and the sidecar records which one that was."""

AUTHORED_PROBE_WARPS = (4, 8, 16)
"""Warps (sub-groups) per program the probe is swept over, crossed with the sub-group sizes
the driver reports for the device -- the sweep the in-tree Triton kernels autotune over."""

AUTHORED_PROBE_LANGUAGE = "triton"
"""The language the probe is written in. Triton needs no compiler beyond the wheel that is
already installed beside torch, so the probe runs wherever a kernel could be authored at
all; a SYCL probe would additionally depend on a toolchain being on the box."""


def _authored_cache_path(hardware_id: str, timer: str) -> pathlib.Path:
    """Sidecar for the authored-stream probe: keyed by language, beside the record."""
    return _cache_path(hardware_id, timer).with_name(
        f"v{CACHE_VERSION}-{hardware_id}-{timer}-authored.json"
    )


def _measure_authored_stream(device: str) -> Optional[Dict[str, object]]:
    """Bandwidth a kernel written here, the plain way, streams at on this part; or None.

    The probe is a Triton copy -- one masked load, one masked store, a block per program --
    over a buffer larger than any last-level cache this project targets, swept over block
    size, warps per program and the sub-group sizes the driver reports, each configuration
    accepted only once its regions agree (:func:`_settled_device_us`). The best settled
    configuration is the answer: what an author reaches with the recipe the skill gives
    and nothing cleverer. Bytes are counted as read plus written.

    None when Triton is not importable, has no backend for this device, or no configuration
    settled; never a number from another part or from a datasheet.
    """
    try:
        import torch
        import triton
        import triton.language as tl
        from triton.runtime import driver
    except Exception as exc:
        logger.debug("authored-stream probe unavailable: %s", exc)
        return None

    backend = _backend(device)
    index = torch.device(device).index or 0
    try:
        backend.set_device(index)
        props = driver.active.utils.get_device_properties(index)
    except Exception as exc:
        logger.debug("authored-stream probe unavailable on %s: %s", device, exc)
        return None
    # Intel's driver reports the sub-group sizes the part supports and Triton takes one as
    # `warp_size`; a driver that reports none has no such knob and the sweep omits it.
    sub_group_sizes = tuple(props.get("sub_group_sizes") or ()) or (None,)

    @triton.jit
    def _copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    itemsize = 2
    n = PATTERN_BUFFER_BYTES // itemsize
    # Random contents, as for every probe here: a constant fill moves faster than data does.
    x = torch.randn(n, dtype=torch.float16, device=device)
    y = torch.empty_like(x)

    def sync() -> None:
        backend.synchronize()

    best: Optional[Tuple[float, Dict[str, object]]] = None
    for block in AUTHORED_PROBE_BLOCKS:
        grid = (triton.cdiv(n, block),)
        for warps in AUTHORED_PROBE_WARPS:
            for sg in sub_group_sizes:
                if sg is not None and warps * sg > int(props.get("max_work_group_size", 1 << 30)):
                    continue
                launch: Dict[str, object] = {"num_warps": warps}
                if sg is not None:
                    launch["warp_size"] = sg

                def fn(block=block, grid=grid, launch=launch, x=x, y=y):
                    _copy[grid](x, y, n, BLOCK=block, **launch)

                try:
                    us = _settled_device_us(fn, sync, calls=5, budget_s=1.0)
                except Exception as exc:
                    logger.debug("authored-stream config %s/%s failed: %s", block, launch, exc)
                    continue
                if us is None:
                    continue
                if best is None or us < best[0]:
                    best = (us, {"block": block, **launch})
    if best is None:
        return None
    us, config = best
    return {
        "gbs": (2 * n * itemsize) / (us * 1e-6) / 1e9,
        "language": AUTHORED_PROBE_LANGUAGE,
        "config": config,
        "probe_bytes": n * itemsize,
        "triton": getattr(triton, "__version__", None),
    }


def authored_stream_probe(device: str, refresh: bool = False) -> Optional[Dict[str, object]]:
    """The authored-stream probe's record for `device`: ``gbs`` plus how it was reached.

    What a kernel *written here* streams at, as distinct from ``bandwidth_gbs``, which is
    what the framework's own reduction reaches. The routing prices an authored kernel
    against this number: the bound such a kernel can plausibly reach on this part for a
    memory-bound op is ``bytes_min / gbs``, not ``bytes_min / bandwidth_gbs``. Measured
    once per (part, timer, language) and cached in a sidecar of the calibration record;
    ``None`` when it could not be measured, and never cached as such.
    """
    try:
        from flashinfer_bench.device import get_accelerator

        accel = get_accelerator(device)
        path = _authored_cache_path(accel.canonical_id(device), accel.make_timer(device).name)
    except Exception:
        return None
    table: Dict[str, Dict[str, object]] = {}
    if path.exists():
        try:
            table = json.loads(path.read_text())
        except Exception:
            table = {}
    cached = table.get(AUTHORED_PROBE_LANGUAGE)
    if not refresh and isinstance(cached, dict) and cached.get("gbs"):
        return cached
    try:
        record = _measure_authored_stream(device)
    except Exception as exc:
        logger.debug("authored-stream probe failed on %s: %s", device, exc)
        return None
    if record is None:
        return None
    table[AUTHORED_PROBE_LANGUAGE] = record
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=2) + "\n")
    except OSError:
        pass
    return record


def authored_stream_gbs(device: str, refresh: bool = False) -> Optional[float]:
    """``authored_stream_probe(device)["gbs"]``, or None when there is no record."""
    record = authored_stream_probe(device, refresh=refresh)
    return None if record is None else float(record["gbs"])  # type: ignore[arg-type]


def _camping_pitches(
    times: Mapping[int, Optional[float]], deficit: float, neighbourhood: int
) -> Optional[List[int]]:
    """Pitches whose read exceeds the median of their nearest measured neighbours by more
    than `deficit`, or None when too few pitches were measured to have neighbours."""
    measured = sorted((p, t) for p, t in times.items() if t is not None and t > 0)
    if len(measured) < 2 * neighbourhood + 1:
        return None
    slow = []
    for i, (pitch, t) in enumerate(measured):
        around = measured[max(0, i - neighbourhood) : i] + measured[i + 1 : i + 1 + neighbourhood]
        if t > statistics.median(q for _, q in around) * (1.0 + deficit):
            slow.append(pitch)
    return slow


def channel_period_from_sweep(
    times: Mapping[int, Optional[float]],
    deficit: float = CAMPING_DEFICIT,
    neighbourhood: int = CHANNEL_PERIOD_NEIGHBOURHOOD,
) -> Optional[int]:
    """The spacing of the slow pitches in a pitch sweep, or None.

    `times` maps a row pitch in bytes to the settled per-call time of the probe read at that
    pitch, or None where it did not settle. A pitch is slow when it exceeds the median of
    its nearest measured neighbours by more than `deficit`. The period is the greatest
    common divisor of the slow pitches, accepted only when it is itself slow and every
    measured multiple of it is slow: the first refuses a spacing assembled from unrelated
    slow pitches, the second a divisor finer than the repeat. Fewer than two slow pitches is
    no repeat and no period.

    Pure arithmetic over the sweep, so it is testable without a device. It never guesses: a
    sweep with no slow pitch returns None, the same as one that did not settle.
    """
    slow = _camping_pitches(times, deficit, neighbourhood)
    if not slow or len(slow) < 2:
        return None
    period = functools.reduce(math.gcd, slow)
    if period not in slow:
        logger.debug("slow pitches %s share %d, which is not itself slow", slow, period)
        return None
    missing = [p for p, t in times.items() if t is not None and p % period == 0 and p not in slow]
    if missing:
        logger.debug("slow pitches %s share %d, but %s are not slow", slow, period, missing)
        return None
    return period


def _measure_pitch_sweep_us(
    device: str,
    pitches: Sequence[int],
    rows: int = CHANNEL_PERIOD_PROBE_ROWS,
    run_bytes: int = CHANNEL_PERIOD_PROBE_RUN_BYTES,
    budget_s: float = CHANNEL_PERIOD_SWEEP_BUDGET_S,
) -> Dict[int, Optional[float]]:
    """Settled per-call microseconds of the probe read at each pitch; None where it did not.

    The probe is a column-wise reduction over a ``[rows, run]`` window of a big buffer whose
    row pitch is the pitch under test: many rows in flight at the same column window, the
    access a decode GEMM makes to a weight. Rows and run are the same at every pitch, so the
    kernel is the same and only the addresses move. Each pitch rotates over enough disjoint
    windows that what the rotation touches exceeds the last-level cache -- the effect does
    not exist while the window is cache-resident. The sweep runs under sustained load from
    a time-based warm-up through the last pitch, and each pitch is accepted by
    :func:`_settled_device_us`.
    """
    import torch

    from flashinfer_bench.device import get_accelerator

    backend = _backend(device)
    l2_bytes = int(get_accelerator(device).capabilities(device).l2_bytes)
    itemsize = 2
    run_e = run_bytes // itemsize
    windows = max(2, math.ceil(2 * l2_bytes / (rows * run_bytes)))
    nbytes = max(PATTERN_BUFFER_BYTES, windows * rows * max(pitches))
    # Random contents, as for every read probe here: a constant fill reads faster than data
    # does through this memory path, and the probe would then be measuring the fill.
    buf = torch.randn(nbytes // itemsize, dtype=torch.float16, device=device)
    out = torch.empty(run_e, dtype=torch.float16, device=device)

    probes: Dict[int, Callable[[], object]] = {}
    for pitch in pitches:
        stride_e = pitch // itemsize
        per_window = rows * stride_e
        count = min(windows, buf.numel() // per_window)
        views = [
            buf[j * per_window : (j + 1) * per_window].view(rows, stride_e)[:, :run_e]
            for j in range(count)
        ]
        rotation = itertools.cycle(views)
        probes[pitch] = lambda it=rotation: torch.sum(next(it), dim=0, out=out)

    def sync() -> None:
        backend.synchronize()

    deadline = time.perf_counter() + CHANNEL_PERIOD_WARMUP_S
    while time.perf_counter() < deadline:
        for fn in probes.values():
            fn()
    sync()

    end = time.perf_counter() + budget_s
    result: Dict[int, Optional[float]] = {}
    for pitch in pitches:
        left = end - time.perf_counter()
        if left <= 0:
            result[pitch] = None
            continue
        result[pitch] = _settled_device_us(
            probes[pitch],
            sync,
            calls=CHANNEL_PERIOD_PROBE_CALLS,
            budget_s=min(CHANNEL_PERIOD_PITCH_BUDGET_S, left),
        )
    return result


def measure_channel_period_bytes(device: str) -> Optional[int]:
    """Distance in bytes at which row pitches camp on one memory channel, or None.

    A streaming read is timed at every row pitch from :data:`CHANNEL_PERIOD_PROBE_RUN_BYTES`
    to :data:`CHANNEL_PERIOD_MAX_PITCH_BYTES` in steps of
    :data:`CHANNEL_PERIOD_PITCH_STEP_BYTES` (:func:`_measure_pitch_sweep_us`), and the period
    is read off the sweep as the spacing of the slow pitches
    (:func:`channel_period_from_sweep`). Nothing about a channel count or an interleave
    granule is assumed: the spacing is what a weight's pitch is tested against, and it is
    established from where the reads are slow.

    None when the sweep resolved no period -- pitches that did not settle on a shared
    device, no slow pitch within the range, or slow pitches that do not repeat at one
    spacing -- and never a value from elsewhere. Cheap enough to run at every calibration:
    a fraction of a second of device time when the pitches settle.
    """
    pitches = list(
        range(
            CHANNEL_PERIOD_PROBE_RUN_BYTES,
            CHANNEL_PERIOD_MAX_PITCH_BYTES + 1,
            CHANNEL_PERIOD_PITCH_STEP_BYTES,
        )
    )
    try:
        times = _measure_pitch_sweep_us(device, pitches)
    except Exception as exc:
        logger.debug("channel period sweep failed on %s: %s", device, exc)
        return None
    period = channel_period_from_sweep(times)
    if period is None:
        unsettled = sum(t is None for t in times.values())
        logger.debug(
            "no channel period on %s: %d of %d pitches unsettled, slow pitches %s",
            device,
            unsettled,
            len(pitches),
            _camping_pitches(times, CAMPING_DEFICIT, CHANNEL_PERIOD_NEIGHBOURHOOD),
        )
    return period


_MISS = object()
"""What the probe's fallback returns, so a miss cannot be confused with a kernel's output."""


def measure_dispatch_us(device: str, dataset: Optional[str] = None) -> Optional[float]:
    """What apply() adds per call over invoking the kernel it chose, or None if unmeasurable.

    Host-side by construction. The kernel that apply() dispatches to is replaced by a no-op
    for the duration, so the measurement compares two Python paths that end at the same
    place -- ``apply(name, kwargs=...)`` against ``runnable(*args)``, the call the benchmark
    timed when it produced the solution's trace -- and the device does no work at all.

    It used to run the real kernel in both arms. That made a few microseconds of host cost
    depend on the GPU's power state: an idle Arc B580 gates its clock to 400 MHz within
    50 ms of the queue draining and takes ~150 ms of load to come back, and at 400 MHz the
    probe kernel outlasts the host, both arms become device-bound and the difference reads
    zero or negative. The dataset load between the previous measurement and this one is
    enough of a gap, so the same code returned 6 us on one call and nothing on the next.
    A quantity that does not involve the device should not be measured with it.

    Returns None rather than a guess: a deploy gate set from an invented number is worse
    than one left off, because it silently admits or refuses the wrong kernels.
    """
    import torch

    try:
        from flashinfer_bench.apply import ApplyConfig, apply
        from flashinfer_bench.apply.runtime import ApplyRuntime
        from flashinfer_bench.data import TraceSet
    except Exception as exc:
        logger.debug("dispatch calibration unavailable: %s", exc)
        return None

    dataset = dataset or os.environ.get("FIB_DATASET_PATH") or "tmp/flashinfer-trace"
    if not pathlib.Path(dataset).exists():
        logger.debug("dispatch calibration unavailable: no dataset at %s", dataset)
        return None

    trace_set = TraceSet.from_path(dataset)
    # The probe is whichever plain RMSNorm definition in the dataset apply() can dispatch on
    # this machine. Nothing names a width: a dataset that lacks one width still calibrates
    # on another.
    probes = _rmsnorm_probes(trace_set)
    if not probes:
        logger.debug("dispatch calibration unavailable: no RMSNorm definition with a solution")
        return None

    runtime = ApplyRuntime(
        trace_set, ApplyConfig(max_atol=0.02, max_rtol=0.02, on_miss_policy="use_def_best")
    )
    runtime.start()
    try:
        for name, hidden, dtype in probes:
            x = torch.randn(64, hidden, dtype=dtype, device=device)
            w = torch.randn(hidden, dtype=dtype, device=device)
            kwargs = {"hidden_states": x, "weight": w}

            def _through(name=name, kwargs=kwargs):
                return apply(name, kwargs=kwargs, fallback=lambda **k: _MISS)

            runnable = _dispatched_runnable(runtime, _through)
            if runnable is None:
                continue
            with _kernel_stubbed(runnable, (x, w)) as bench_call:
                if _through() is _MISS:
                    continue
                delta = _paired_delta_us(_through, bench_call, calls=200)
            if delta is not None:
                logger.debug("dispatch cost %.2fus via %s on cpu %s", delta, name, _current_cpu())
                return delta
        return None
    except Exception as exc:
        logger.debug("dispatch calibration unavailable: %s", exc)
        return None
    finally:
        runtime.stop()


def _dispatched_runnable(runtime, call: Callable[[], object]):
    """The Runnable `call` dispatches to through `runtime`, or None if it fell back.

    Observed from the outside -- the runtime's build step is wrapped for one call and then
    restored -- so that which solution the table selects, and through which builder, is
    exactly what a serving process would get.
    """
    seen = []
    original = runtime._try_build

    def observe(definition, solution):
        runnable = original(definition, solution)
        seen.append(runnable)
        return runnable

    runtime._try_build = observe
    try:
        if call() is _MISS:
            return None
    finally:
        del runtime._try_build
    return next((r for r in reversed(seen) if r is not None), None)


@contextlib.contextmanager
def _kernel_stubbed(runnable, inputs: Tuple[object, ...]) -> Iterator[Callable[[], object]]:
    """Replace `runnable`'s kernel with a no-op for the block; yield the call the bench timed.

    The benchmark timed ``runnable(*inputs)`` for a value-returning solution and
    ``runnable(*inputs, *outputs)`` with pre-allocated outputs for a destination-passing
    one, so that is what apply() is measured against; anything apply() does beyond it --
    including allocating those outputs -- is cost the trace does not contain. The stub
    returns what the real kernel returned so the Runnable's return handling is unchanged.
    The real kernel is put back however the block exits: the build cache is process-wide,
    and a serving process runs calibration before it serves.
    """
    original = runnable._callable
    if runnable.metadata.destination_passing_style:
        args = (*inputs, *runnable._allocate_output_tensors(*inputs))
        result = None
    else:
        args = tuple(inputs)
        result = original(*args)
    runnable._callable = lambda *a: result
    try:
        yield lambda: runnable(*args)
    finally:
        runnable._callable = original


def _current_cpu() -> Optional[int]:
    """Which CPU this thread is on, for the log: a hybrid part's core classes differ by half."""
    try:
        return int(pathlib.Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[36])
    except Exception:
        return None


def _rmsnorm_probes(trace_set) -> List[Tuple[str, int, torch.dtype]]:
    """(definition, hidden, dtype) for every plain RMSNorm that has a solution, smallest first.

    Read from the dataset rather than named: the probe only has to be *a* definition apply()
    can dispatch here, and which widths exist is a property of the dataset in hand. Smallest
    first only because the probe's tensors are cheapest to make; the kernel does not run.
    """
    probes = []
    for name, definition in trace_set.definitions.items():
        if definition.op_type != "rmsnorm" or set(definition.inputs) != {"hidden_states", "weight"}:
            continue
        hidden = definition.const_axes.get("hidden_size")
        if not hidden or not trace_set.solutions.get(name):
            continue
        probes.append((name, int(hidden), definition.torch_input_dtypes[0]))
    return sorted(probes, key=lambda p: p[1])


def get(device: Optional[str] = None, refresh: bool = False) -> Optional[Calibration]:
    """Cached calibration for `device`, measuring once if needed.

    Returns None when it cannot be measured here, so a caller can fall back to a documented
    behaviour instead of acting on a fabricated threshold. A record with any field still
    None (``dispatch_us``, ``launch_floor_us``, a dtype's matrix throughput,
    ``channel_period_bytes``) is returned but not cached: an unmeasured cost is not a result
    to remember, and the next process gets another attempt.
    """
    try:
        from flashinfer_bench.device import default_device_type, get_accelerator

        accel = get_accelerator(device or default_device_type())
        device = device or accel.list_devices()[0]
        hardware_id = accel.canonical_id(device)
        timer = accel.make_timer(device).name
    except Exception:
        return None

    path = _cache_path(hardware_id, timer)
    if not refresh and path.exists():
        try:
            return Calibration(**json.loads(path.read_text()))
        except Exception:
            pass
    try:
        result = measure(device)
    except Exception as exc:
        logger.debug("calibration failed on %s: %s", device, exc)
        return None
    if not result.complete:
        return result
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    except OSError:
        pass
    return result
