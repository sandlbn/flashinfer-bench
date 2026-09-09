"""Per-part constants, measured and cached rather than written down.

Thresholds like "a substitution costs ~6us" or "the timer's floor is ~30us" are properties
of one GPU and one software stack, not of this project. Written into a default they are
wrong on every other part, and they drift on the part they came from: figures hand-measured
here were already 3% and 87% off when re-measured on the same machine a day later.

So nothing stores them. A caller asks for the value, it is measured once per (part, timer,
stack) and cached on disk, and a new chip gets its own answer with no edit.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


@dataclass(frozen=True)
class Calibration:
    """What this part costs, in microseconds unless stated."""

    hardware_id: str
    timer: str
    dispatch_us: float
    """Cost of one successful apply() substitution beyond the kernel itself.

    A kernel saving less than this loses the exchange however large its ratio.
    """
    timing_floor_us: float
    """Fixed cost a single event-timed call carries. Ratios below it measure the harness."""
    bandwidth_gbs: float
    """Contiguous read bandwidth. A kernel's own ceiling is lower; measure its access
    pattern rather than treating this as a bound."""


def _cache_path(hardware_id: str, timer: str) -> pathlib.Path:
    root = os.environ.get("FIB_CACHE_PATH") or os.path.expanduser("~/.cache/flashinfer_bench")
    return pathlib.Path(root) / "calibration" / f"v{CACHE_VERSION}-{hardware_id}-{timer}.json"


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

    dispatch = _measure_dispatch(device, sync) or 0.0
    return Calibration(caps.canonical_id, accel.make_timer(device).name, dispatch, floor, bandwidth)


def _measure_dispatch(device: str, sync: Callable[[], None]) -> Optional[float]:
    """apply()'s cost over calling the same kernel directly, or None if unmeasurable.

    Returns None rather than a guess: a deploy gate set from an invented number is worse
    than one left off, because it silently refuses or admits the wrong kernels.
    """
    import torch

    try:
        import vllm_xpu_kernels._C  # noqa: F401

        from flashinfer_bench.apply import ApplyConfig, apply, enable_apply
        from flashinfer_bench.apply.runtime import ApplyRuntime
    except Exception:
        return None

    dataset = os.environ.get("FIB_DATASET_PATH", "tmp/flashinfer-trace")
    if not pathlib.Path(dataset).exists():
        return None

    hidden = 1024
    x = torch.randn(64, hidden, dtype=torch.bfloat16, device=device)
    w = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    ApplyRuntime._stack.clear()
    runtime = enable_apply(
        dataset, ApplyConfig(max_atol=0.02, max_rtol=0.02, on_miss_policy="use_def_best")
    )
    try:
        name = f"rmsnorm_h{hidden}"
        if apply(name, kwargs={"hidden_states": x, "weight": w}, fallback=lambda **k: None) is None:
            return None
        direct = _median_us(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6), sync, 300)
        through = _median_us(
            lambda: apply(
                name, kwargs={"hidden_states": x, "weight": w}, fallback=lambda **k: None
            ),
            sync,
            300,
        )
        return max(0.0, through - direct)
    except Exception as exc:
        logger.debug("dispatch calibration unavailable: %s", exc)
        return None
    finally:
        runtime.stop()


def get(device: Optional[str] = None, refresh: bool = False) -> Optional[Calibration]:
    """Cached calibration for `device`, measuring once if needed.

    Returns None when it cannot be measured here, so a caller can fall back to a documented
    behaviour instead of acting on a fabricated threshold.
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    except OSError:
        pass
    return result
