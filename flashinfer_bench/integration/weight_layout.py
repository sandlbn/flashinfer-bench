"""Weight transforms that let a fused kernel consume a model's existing weights.

A fused GEMM epilogue often wants its operands laid out differently from how a model
stores them. The fix belongs in the weights, applied once at load time -- not in the
kernel, and not in a per-model branch of it.

Branching kernels per model layout scales badly: the cost is models x fusions, and every
branch needs its own verified reference, which is the expensive part. A weight transform
is O(1) code that serves any model of the same shape. Serving frameworks already work this
way -- vLLM's ``MergedColumnParallelLinear`` merges gate and up projections at load time,
and an interleave is the same kind of operation in the same place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

if TYPE_CHECKING:
    import torch


def interleave_gate_up(
    gate_weight: "torch.Tensor",
    up_weight: "torch.Tensor",
    norm_weight: Optional["torch.Tensor"] = None,
    *,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """Build the interleaved B matrix a fused SwiGLU/GeGLU GEMM epilogue expects.

    Models keep the gate and up projections as separate ``[d, k]`` matrices and apply the
    activation afterwards. A fused epilogue that computes ``silu(gate) * up`` in registers
    needs both operands adjacent, so the two matrices interleave column-wise into one
    ``[k, 2d]``: even columns gate, odd columns up.

    An RMSNorm applied before the projection can be folded in at the same time. It has two
    parts and they go to different places:

    - the per-channel ``gamma`` scales *input* channels, so it cannot travel through the
      GEMM as a per-row output scale -- it multiplies into the weight here;
    - the per-token ``rsqrt(mean(x^2) + eps)`` *does* commute, since ``(r*x) @ W`` equals
      ``r * (x @ W)``, so it stays a runtime per-row scale handed to the kernel.

    Parameters
    ----------
    gate_weight : torch.Tensor
        Gate projection, ``[d, k]`` as ``nn.Linear`` stores it.
    up_weight : torch.Tensor
        Up projection, ``[d, k]``. Must match ``gate_weight``.
    norm_weight : Optional[torch.Tensor]
        Per-channel RMSNorm gamma, ``[k]``. Folded into the result when given.
    dtype : Optional[torch.dtype]
        Output dtype. Defaults to ``gate_weight``'s.

    Returns
    -------
    torch.Tensor
        ``[k, 2d]`` with gate on even columns and up on odd.

    Raises
    ------
    ValueError
        If the shapes disagree.
    """
    import torch

    if gate_weight.shape != up_weight.shape:
        raise ValueError(
            f"gate and up projections must have the same shape, got "
            f"{tuple(gate_weight.shape)} and {tuple(up_weight.shape)}"
        )
    if gate_weight.ndim != 2:
        raise ValueError(f"expected 2-D projections, got {gate_weight.ndim}-D")

    d, k = gate_weight.shape
    if norm_weight is not None and norm_weight.shape != (k,):
        raise ValueError(
            f"norm weight must be [{k}] to match the projection's input dimension, "
            f"got {tuple(norm_weight.shape)}"
        )

    target_dtype = dtype if dtype is not None else gate_weight.dtype

    # Fold in float32 regardless of storage dtype: gamma and the weights are both small,
    # and multiplying them in bf16 loses precision that never comes back.
    gate = gate_weight.float()
    up = up_weight.float()
    if norm_weight is not None:
        gamma = norm_weight.float()
        gate = gate * gamma
        up = up * gamma

    out = torch.empty((k, 2 * d), device=gate_weight.device, dtype=target_dtype)
    out[:, 0::2] = gate.T.to(target_dtype)
    out[:, 1::2] = up.T.to(target_dtype)
    return out


def rms_row_scale(x: "torch.Tensor", eps: float) -> "torch.Tensor":
    """Per-token RMSNorm factor to hand a fused epilogue as its row scale.

    This is the half of RMSNorm that commutes through the GEMM. The other half, the
    per-channel gamma, belongs in the weight -- see :func:`interleave_gate_up`.

    Parameters
    ----------
    x : torch.Tensor
        Layer input, ``[..., k]``.
    eps : float
        The model's ``rms_norm_eps``.

    Returns
    -------
    torch.Tensor
        Contiguous float32 ``[m]``, one factor per token.
    """
    import torch

    flat = x.reshape(-1, x.shape[-1]).float()
    return torch.rsqrt(flat.pow(2).mean(-1) + eps).contiguous()


def deinterleave_output(out: "torch.Tensor") -> "torch.Tensor":
    """Take the meaningful half of a pairwise fused epilogue's output.

    Xe-Fuse's pairwise ops compute the same value on both lanes of each ``(gate, up)``
    pair, so the ``[m, 2d]`` result carries each answer twice. The even lanes are the
    ``[m, d]`` activation the model wants.
    """
    return out[..., 0::2]


def qwen_style_mlp_weights(
    mlp: object, norm: object, *, dtype: Optional["torch.dtype"] = None
) -> Tuple["torch.Tensor", float]:
    """Prepare a HuggingFace gated-MLP block for a fused SwiGLU GEMM.

    Works for any module exposing ``gate_proj`` and ``up_proj`` alongside an RMSNorm with
    a ``weight`` -- the LLaMA/Qwen/Mistral family shape.

    Returns
    -------
    Tuple[torch.Tensor, float]
        The interleaved, gamma-folded ``[k, 2d]`` matrix, and the norm's epsilon.
    """
    gate = getattr(mlp, "gate_proj").weight.detach()
    up = getattr(mlp, "up_proj").weight.detach()
    gamma = getattr(norm, "weight").detach()
    eps = float(getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", 1e-6))
    return interleave_gate_up(gate, up, gamma, dtype=dtype), eps


# ----------------------------------------------------------------------------- row padding
#
# A second kind of load-time transform: not a different *arrangement* of the values for a
# different kernel, but a different *address pattern* for the same kernel.
#
# Device memory is spread over several channels by address. When a weight's row pitch in
# bytes is a multiple of the distance at which the channel pattern repeats, every row starts
# on the same channel, and a GEMM that walks many rows at the same column offset -- which is
# what a skinny decode GEMM does -- queues all of its reads on that one channel while the
# others idle. Padding the pitch by a few elements breaks the coincidence; the values, the
# kernel and the numerics are unchanged (oneDNN takes the strided descriptor as-is).
#
# The distance is a property of one part's memory system. No driver interface reports it
# reliably, and a value carried over from another part pads the wrong weights without any
# sign that it did, so it is measured: ``flashinfer_bench.device.calibration`` sweeps the
# row pitch of a streaming read and takes the spacing of the pitches at which the read is
# slow, once per part, and caches it with the part's other calibrated costs.
#
# The transform is keyed on the pitch arithmetic and nothing else: not on a model, a layer
# name or a shape. Any weight of any model whose pitch lands on the period gets nominated;
# any weight that does not is left alone; and a nomination is only kept on a measured win.

PERIOD_ENV = "FIB_CHANNEL_PERIOD_BYTES"
"""Explicit channel period in bytes, for a box where the calibration sweep cannot settle --
another process on the device throughout -- and the period has been measured another way,
such as the harness pitch sweep. An operator's measurement, not a default: unset, the
period is the calibrated one, and when there is none the transform stands down."""

ROW_PAD_BYTES = 64
"""How far the pitch is moved off the period.

One cache line: every pad from 16 bytes to two kilobytes measured the same gain on the
first part, so the smallest that keeps every row cache-line aligned -- which the block
loads oneDNN's GEMM kernels issue rely on -- is the right one. The added storage is
``64 / pitch``: a fraction of a percent on any projection wide enough to hit the period.
"""

PROBE_M = 8
"""Decode-sized row count the load-time probe times the GEMM at."""

PROBE_ROUNDS = 7
PROBE_CALLS = 40
PROBE_WARMUP_S = 0.25
"""Enough sustained load to bring a clock-gated part up before either arm is timed."""

_NOISE_SIGMAS = 2.0
_MAD_TO_SIGMA = 1.4826

STREAMING_POOL_LLC_MULTIPLE = 2
"""How many last-level caches a rotating working set spans before repeated calls read from
memory rather than from the cache. A margin over the cache size, not a measured quantity:
one cache's worth of distinct lines already defeats an LRU, and twice that leaves room for
whatever else the part keeps resident."""


def streaming_pool_bytes(device: object) -> Optional[int]:
    """Bytes a rotating pool of inputs must span so that each call streams them from memory.

    The size of the part's last-level cache, from its capability record, times
    :data:`STREAMING_POOL_LLC_MULTIPLE`. This is the working set a measurement needs when
    its subject is a tensor that is not cache-resident when the model runs -- a weight at
    decode, read once per step among every other weight -- and the bound it is compared to
    is a memory bound. ``None`` when the capability record cannot be read for the device,
    or the device is host memory: the caller then has no streaming measurement, not a
    resident one relabelled.
    """
    try:
        from flashinfer_bench.device import get_accelerator
        from flashinfer_bench.device.accelerator import parse_device

        if parse_device(str(device))[0] == "cpu":
            return None
        l2 = int(get_accelerator(str(device)).capabilities(str(device)).l2_bytes)
    except Exception:
        return None
    return STREAMING_POOL_LLC_MULTIPLE * l2 if l2 > 0 else None


class Probe(NamedTuple):
    """What a load-time A/B measured: median per-round speedup and its robust scatter.

    ``win`` is the same call :func:`probe_verdict` makes, kept as a field because a caller
    that only has to decide "keep it or not" should not have to re-derive it. A caller that
    has to tell a measured *slowdown* from a measurement that said nothing wants the verdict.
    """

    win: bool
    speedup: float
    spread: float


PadProbe = Probe
"""The row pad's name for :class:`Probe`, from when it was the only measured decision."""


def probe_verdict(probe: "Probe", sigmas: float = None) -> str:
    """``"WIN"``, ``"LOSS"`` or ``"NOISE"`` -- the three-way reading of a probe.

    ``win`` collapses LOSS and NOISE into "not a win", which is the right question when the
    transform is free to decline. It is the wrong question when declining is itself a
    decision: a transform that wins in one regime and is *measured slower* in another is a
    different situation from one that wins in the first and says nothing about the second.
    """
    import math

    if sigmas is None:
        sigmas = _NOISE_SIGMAS
    if probe.speedup <= 0:
        return "NOISE"
    if abs(math.log(probe.speedup)) <= sigmas * math.log1p(max(probe.spread, 0.0)):
        return "NOISE"
    return "WIN" if probe.speedup > 1.0 else "LOSS"


class ChannelPeriodUnknown(LookupError):
    """No memory channel period is known for a device, so the pitch rule cannot be applied.

    Raised rather than answered: the rule with a guessed period pads weights that do not
    camp and misses ones that do, and nothing downstream can tell which happened. A caller
    that wants to stand down quietly asks :func:`channel_period_bytes` first.
    """


_PERIODS: Dict[str, Optional[int]] = {}
"""One resolution per device per process. The calibration record caches a measured period
on disk, but a record that could not complete is measured again by every process that asks,
and a model asks once per weight."""


def channel_period_bytes(device: object = None) -> Optional[int]:
    """Distance in bytes at which row pitches camp on one memory channel, or None.

    Measured per part by ``flashinfer_bench.device.calibration`` -- a sweep of row pitches
    under a streaming read, the period being the spacing of the pitches at which the read
    is slow -- and cached with the rest of the part's calibration. ``FIB_CHANNEL_PERIOD_BYTES``
    overrides it with a period measured another way.

    ``None`` means no period is known for this device: the sweep resolved none, or the
    tensor is in host memory, which the accelerator's GEMM does not read. A caller must
    then treat the row-pad transform as unavailable rather than pick a period. ``None`` is
    not "does not camp": it is "cannot tell", and the two must not be conflated silently.
    """
    import os

    override = os.environ.get(PERIOD_ENV)
    if override:
        period = int(override)
        if period <= 0:
            raise ValueError(f"{PERIOD_ENV} must be a positive period in bytes, got {override!r}")
        return period
    key = "" if device is None else str(device)
    if key not in _PERIODS:
        _PERIODS[key] = _calibrated_period(key or None)
    return _PERIODS[key]


def _calibrated_period(device: Optional[str]) -> Optional[int]:
    from flashinfer_bench.device import calibration
    from flashinfer_bench.device.accelerator import parse_device

    if device is not None and parse_device(device)[0] == "cpu":
        return None
    record = calibration.get(device)
    return None if record is None else record.channel_period_bytes


def _resolve_period(device: object, period_bytes: Optional[int]) -> int:
    if period_bytes is None:
        period_bytes = channel_period_bytes(device)
        if period_bytes is None:
            raise ChannelPeriodUnknown(
                f"no memory channel period is known for {device}: calibration resolved none "
                f"and {PERIOD_ENV} is unset, so the pitch rule cannot say whether a weight camps"
            )
    if period_bytes <= 0:
        raise ValueError(f"channel period must be positive bytes, got {period_bytes}")
    return period_bytes


def row_pitch_bytes(weight: "torch.Tensor") -> int:
    """Bytes from the start of one row to the start of the next."""
    return int(weight.stride(0)) * weight.element_size()


def pitch_camps_on_one_channel(pitch_bytes: int, period_bytes: int) -> bool:
    """The pitch rule itself: a row pitch puts every row on one channel exactly when it is
    a multiple of the channel period.

    The one place the arithmetic lives. :func:`camps_on_one_channel` applies it to a tensor;
    a stage that only has a recorded shape (the routing in ``scripts/bound_candidates.py``)
    applies it to the pitch a contiguous tensor of that shape would have. A period must be
    given: there is no answer without one, and this function does not look one up.
    """
    if period_bytes <= 0:
        raise ValueError(f"channel period must be positive bytes, got {period_bytes}")
    return pitch_bytes % period_bytes == 0


def camps_on_one_channel(weight: "torch.Tensor", period_bytes: Optional[int] = None) -> bool:
    """Whether every row of a row-major 2-D weight starts on the same memory channel.

    True exactly when the row pitch in bytes is a multiple of the channel period
    (:func:`pitch_camps_on_one_channel`). Only a row-major 2-D tensor can qualify; anything
    else is False. Pure arithmetic: whether the tensor lives in a memory that has channels
    is the caller's decision. With no period given, the device's calibrated one is used,
    and :class:`ChannelPeriodUnknown` is raised when there is none -- the rule has no answer
    then, and False would be one.
    """
    if weight.ndim != 2 or weight.stride(1) != 1:
        return False
    return pitch_camps_on_one_channel(
        row_pitch_bytes(weight), _resolve_period(weight.device, period_bytes)
    )


def padded_row_pitch(pitch_bytes: int, period_bytes: int, pad_bytes: int = ROW_PAD_BYTES) -> int:
    """The pitch a camping weight is moved to: ``pitch + pad``, checked to be off the period."""
    new_pitch = pitch_bytes + pad_bytes
    if new_pitch % period_bytes == 0:
        # Only reachable when pad is itself a multiple of the period, which would put the
        # rows straight back where they were.
        raise ValueError(
            f"a pad of {pad_bytes} bytes leaves pitch {new_pitch} on the {period_bytes}-byte "
            "channel period"
        )
    return new_pitch


def pad_rows_off_channel_period(
    weight: "torch.Tensor",
    period_bytes: Optional[int] = None,
    pad_bytes: int = ROW_PAD_BYTES,
) -> Optional["torch.Tensor"]:
    """A copy of ``weight`` whose rows are spread across memory channels, or None.

    Returns None when the weight does not camp -- the caller keeps what it has and nothing
    was allocated. Raises :class:`ChannelPeriodUnknown` when no period is given and none is
    known for the device. Otherwise the result is a ``[n, k]`` view into ``[n, k + pad]`` storage:
    same values, same dtype, same device, same shape; only ``stride(0)`` differs. Passing
    it where the original went is bit-identical (the GEMM reads the same numbers in the
    same order), and it costs ``pad / pitch`` extra memory on that one tensor.
    """
    if weight.ndim != 2 or weight.stride(1) != 1:
        return None
    period_bytes = _resolve_period(weight.device, period_bytes)
    if not camps_on_one_channel(weight, period_bytes):
        return None
    esize = weight.element_size()
    if pad_bytes % esize:
        raise ValueError(f"pad of {pad_bytes} bytes is not whole {weight.dtype} elements")
    new_pitch = padded_row_pitch(row_pitch_bytes(weight), period_bytes, pad_bytes)
    return _copy_with_pitch(weight, new_pitch)


def _copy_with_pitch(weight: "torch.Tensor", pitch_bytes: int) -> "torch.Tensor":
    """A copy of a 2-D ``weight`` whose rows are ``pitch_bytes`` apart: a view of wider storage."""
    import torch

    n, k = weight.shape
    storage = torch.empty(
        (n, pitch_bytes // weight.element_size()), dtype=weight.dtype, device=weight.device
    )
    view = storage[:, :k]
    view.copy_(weight)
    return view


def streaming_pad_wins(
    weight: "torch.Tensor",
    padded: "torch.Tensor",
    *,
    m: int = PROBE_M,
    rounds: int = PROBE_ROUNDS,
    calls: int = PROBE_CALLS,
    pool_bytes: Optional[int] = None,
) -> Optional[PadProbe]:
    """Time the GEMM through ``weight`` against ``padded`` as they would stream at decode.

    The pitch rule says where camping *can* happen; whether the pad pays depends on more
    than the pitch. On the first part a 1024-row weight at the camping pitch gained a third
    from the pad while a 4096-row weight at the same pitch lost three percent, and padding
    a weight that does not camp costs up to eight percent -- the pad is never free, and the
    kernel's tiling over the rows decides whether the channel spread is worth it. Nothing
    available at load time predicts that, so it is measured, once per shape.

    Both arms rotate over enough copies to exceed the last-level cache -- the effect does
    not exist while the weight is cache-resident, and at decode it never is -- and are timed
    in interleaved rounds with the order flipped, judged on the median per-round ratio
    against its own scatter, as ``scripts/kernel_trials.py`` judges. Returns None when it
    cannot measure (host memory, no room for the pools): unknown is not a win.
    """
    import math

    import torch

    if weight.device.type == "cpu":
        return None
    try:
        from flashinfer_bench.device import get_accelerator

        accel = get_accelerator(str(weight.device))
        if pool_bytes is None:
            pool_bytes = streaming_pool_bytes(weight.device)
        sync = accel.synchronize
    except Exception:
        return None
    if pool_bytes is None:
        return None

    n, k = weight.shape
    copies = max(2, math.ceil(pool_bytes / (padded.untyped_storage().nbytes() or 1)))
    pools: List[List["torch.Tensor"]] = [[weight], [padded]]
    try:
        for _ in range(copies - 1):
            pools[0].append(_copy_with_pitch(weight, row_pitch_bytes(weight)))
            pools[1].append(_copy_with_pitch(weight, row_pitch_bytes(padded)))
        x = torch.randn((m, k), dtype=weight.dtype, device=weight.device)
    except RuntimeError:  # no room for the pools; leave the weight as it is
        return None

    counters = [0, 0]

    def _call(arm: int) -> None:
        pool = pools[arm]
        torch.nn.functional.linear(x, pool[counters[arm] % len(pool)])
        counters[arm] += 1

    with torch.no_grad():
        return paired_speedup(
            lambda: _call(0),
            lambda: _call(1),
            lambda: sync(str(weight.device)),
            rounds=rounds,
            calls=calls,
        )


def paired_speedup(
    call_a,
    call_b,
    sync,
    *,
    rounds: int = PROBE_ROUNDS,
    calls: int = PROBE_CALLS,
    warmup_s: float = PROBE_WARMUP_S,
) -> Optional["Probe"]:
    """Time two ways of doing the same thing against each other, and judge the difference.

    The shape every load-time decision in this module takes: warm both arms together until
    the part's clocks have settled, then time them in interleaved rounds with the order
    flipped each round, and read the median of the per-round log-ratios against its own
    robust scatter -- the same judgement ``scripts/kernel_trials.py`` makes on a trial, so a
    decision taken at load and a decision taken in the trial loop mean the same thing.

    ``call_a`` is the arm being defended (what the stack does today) and ``call_b`` the
    challenger, so a speedup above one favours ``call_b``. ``sync`` takes no arguments and
    must drain the device. None when a round measured no time at all.
    """
    import math
    import statistics
    import time

    def _time(fn) -> float:
        sync()
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        sync()
        return time.perf_counter() - start

    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        call_a()
        call_b()
    samples: List[Tuple[float, float]] = []
    for r in range(rounds):
        # Flip the order each round so neither arm systematically inherits the other's
        # clock state.
        if r % 2 == 0:
            a, b = _time(call_a), _time(call_b)
        else:
            b, a = _time(call_b), _time(call_a)
        samples.append((a, b))
    if any(a <= 0 or b <= 0 for a, b in samples):
        return None
    ratios = [math.log(a / b) for a, b in samples]
    center = statistics.median(ratios)
    sigma = _MAD_TO_SIGMA * statistics.median(abs(v - center) for v in ratios)
    return Probe(
        win=center > 0 and center > _NOISE_SIGMAS * sigma,
        speedup=math.exp(center),
        spread=math.exp(sigma) - 1,
    )


# ------------------------------------------------------------ the mm operand and its entry
#
# A third kind of load-time transform, and the first that changes which *kernel* runs.
#
# A model stores a projection as ``[n, k]`` row-major and the stack calls ``F.linear``,
# which hands the GEMM library a B operand that is the transpose of that storage -- column-
# major, in the library's descriptor a `ba` tag. Given a `ba` B operand oneDNN selects one
# GEMM kernel; given a row-major ``[k, n]`` one it selects another, and on a large-M problem
# the second is the faster of the two. It is also the kernel the part's peak-throughput
# calibration probe runs, because that probe's operand is already row-major -- so the
# transform is what makes a projection run the kernel the part's peak was measured on.
#
# Two things have to change together and neither is sufficient alone:
#
# - the *operand*: the weight re-laid so its transpose is contiguous. A transposed **view**
#   of the original storage carries the original descriptor and changes no selection; the
#   copy is real, and it costs one weight's worth of memory traffic once, at load.
# - the *entry point*: ``aten.mm`` on that operand rather than ``F.linear`` on its
#   transpose. Measured on the same operand, the linear entry point gave the layout's gain
#   back; ``mm`` is the shallowest entry that reaches the primitive.
#
# Unlike the pitch rule there is no arithmetic here that predicts the answer: which kernel a
# library selects, and whether the selected one is faster at the M this deployment presents,
# are properties of the library and the part. So the structural test below only rejects
# weights the transform is *meaningless* for, and every weight it admits is decided by
# measuring both arms at that weight's own shape -- see :func:`mm_entry_wins`.


def mm_operand(weight: "torch.Tensor") -> "torch.Tensor":
    """The ``[k, n]`` right-hand operand ``aten.mm`` takes for an ``[n, k]`` weight.

    A view, never a copy. It is contiguous exactly when ``weight`` has been through
    :func:`to_mm_operand_layout` -- which is the whole point of that function.
    """
    return weight.t()


def mm_operand_ready(weight: "torch.Tensor") -> bool:
    """Whether ``mm_operand(weight)`` is already the contiguous ``[k, n]`` the win needs."""
    return weight.ndim == 2 and weight.t().is_contiguous()


def is_row_major(weight: "torch.Tensor") -> bool:
    """Whether a 2-D tensor is plain ``[n, k]`` row-major, as a model stores a projection."""
    return weight.ndim == 2 and weight.stride(1) == 1 and weight.stride(0) == weight.shape[1]


def to_mm_operand_layout(weight: "torch.Tensor") -> Optional["torch.Tensor"]:
    """``weight`` re-laid so that ``mm_operand`` of it is contiguous, or None.

    Same shape, dtype, device and values; only the strides differ, so everything that reads
    ``weight.shape`` -- weight loaders, tensor-parallel bookkeeping, ``F.linear`` itself --
    keeps working. None when the tensor is not a plain row-major 2-D weight: something has
    already re-laid it (the row pad, or vLLM's own N-contiguous option) and stacking a
    second layout transform on top of the first would measure neither.
    """
    if not is_row_major(weight):
        return None
    return weight.t().contiguous().t()


def mm_entry_wins(
    weight: "torch.Tensor",
    converted: "torch.Tensor",
    *,
    m: int,
    rounds: int = PROBE_ROUNDS,
    calls: int = PROBE_CALLS,
    pool_bytes: Optional[int] = None,
) -> Optional["Probe"]:
    """Time the production call against the mm entry on the converted operand, at ``m`` rows.

    Arm A is exactly what the stack does today -- ``F.linear`` on the weight as stored. Arm B
    is ``aten.mm`` on ``converted``'s transpose. Both arms rotate over enough copies to
    exceed the last-level cache, because a weight is not cache-resident when a model runs.

    ``m`` is the decision's whole context: the same lever measured at a prefill-sized M and
    at a decode-sized one does not give the same answer, and a caller that wants to know
    about both asks twice. Returns None when it cannot measure -- host memory, no capability
    record, no room for the pools. Unknown is not a win.
    """
    import math

    import torch

    if weight.device.type == "cpu" or not mm_operand_ready(converted):
        return None
    try:
        from flashinfer_bench.device import get_accelerator

        accel = get_accelerator(str(weight.device))
        if pool_bytes is None:
            pool_bytes = streaming_pool_bytes(weight.device)
        sync = accel.synchronize
    except Exception:
        return None
    if pool_bytes is None:
        return None

    k = weight.shape[1]
    bytes_each = weight.numel() * weight.element_size() or 1
    copies = max(2, math.ceil(pool_bytes / bytes_each))
    try:
        stored: List["torch.Tensor"] = [weight] + [weight.clone() for _ in range(copies - 1)]
        operands: List["torch.Tensor"] = [mm_operand(converted)] + [
            mm_operand(w.t().contiguous().t()) for w in stored[1:]
        ]
        rows: List["torch.Tensor"] = [torch.randn((m, k), dtype=weight.dtype, device=weight.device)]
    except RuntimeError:  # no room for the pools; leave the weight as it is
        return None

    counters = [0, 0]

    def _linear() -> None:
        torch.nn.functional.linear(rows[0], stored[counters[0] % len(stored)])
        counters[0] += 1

    def _mm() -> None:
        torch.ops.aten.mm.default(rows[0], operands[counters[1] % len(operands)])
        counters[1] += 1

    try:
        with torch.no_grad():
            return paired_speedup(
                _linear,
                _mm,
                lambda: sync(str(weight.device)),
                rounds=rounds,
                calls=calls,
            )
    finally:
        # Hand the pools and the arms' results back to the driver before returning. A
        # serving stack sizes its KV cache from the free memory it finds after loading, so a
        # probe that left a few hundred megabytes cached would give the measured arm a
        # smaller cache than the arm it is compared against -- and the A/B would then differ
        # by the batch size as well as by the kernel.
        stored.clear()
        operands.clear()
        rows.clear()
        accel.empty_cache()
