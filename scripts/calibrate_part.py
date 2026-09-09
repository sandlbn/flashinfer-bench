"""Measure the part-specific constants the tuning and deploy gates depend on.

Every threshold in this repo's Intel work is a measurement of one GPU, not a property of the
software: what a kernel substitution costs, where the timing instrument's floor sits, and
what bandwidth is actually reachable. On new silicon all three move, and a gate carrying the
old numbers is silently wrong -- it will deploy kernels that lose, or refuse kernels that
win, without saying anything.

Run this first on a new part. It prints the values, and the flags that carry them.

    python scripts/calibrate_part.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

import torch


def _sync(device: str) -> None:
    backend = torch.xpu if device.startswith("xpu") else torch.cuda
    backend.synchronize()


def _per_call_us(fn: Callable[[], object], device: str, calls: int) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / calls * 1e6


def _median_of(fn: Callable[[], object], device: str, calls: int, rounds: int) -> float:
    for _ in range(max(20, calls)):
        fn()
    _sync(device)
    return statistics.median([_per_call_us(fn, device, calls) for _ in range(rounds)])


def bandwidth(device: str, mib: int, rounds: int) -> float:
    """Achievable read bandwidth, GB/s, on a contiguous buffer larger than any cache."""
    n = mib * 1024 * 1024 // 2
    a = torch.randn(n, dtype=torch.float16, device=device)
    us = _median_of(lambda: a.sum(), device, 10, rounds)
    return (a.numel() * 2) / (us * 1e-6) / 1e9


def timing_floor(device: str, rounds: int) -> float:
    """What one event-timed call costs beyond the work, in microseconds.

    The gap between timing a call on its own and timing many in one region is the fixed cost
    the instrument adds. Below it, per-kernel ratios are measuring the harness.
    """
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    fn = lambda: torch.mul(x, 1.0001, out=out)  # noqa: E731 - tiny, deliberately

    batched = _median_of(fn, device, 200, rounds)
    backend = torch.xpu if device.startswith("xpu") else torch.cuda
    singles: List[float] = []
    for _ in range(rounds):
        _sync(device)
        start, end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync(device)
        singles.append(start.elapsed_time(end) * 1000)
    return max(0.0, statistics.median(singles) - batched)


def dispatch_cost(device: str, dataset: str, rounds: int) -> float | None:
    """What one successful apply() substitution costs on top of the kernel.

    Returns None when no definition on this dataset both matches this part and has a
    solution -- the number cannot be invented, and a gate set from a guess is worse than a
    gate left off.
    """
    from flashinfer_bench.apply import ApplyConfig, apply, enable_apply
    from flashinfer_bench.apply.runtime import ApplyRuntime

    hidden = 1024
    x = torch.randn(64, hidden, dtype=torch.bfloat16, device=device)
    w = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    ApplyRuntime._stack.clear()
    runtime = enable_apply(
        dataset, ApplyConfig(max_atol=0.02, max_rtol=0.02, on_miss_policy="use_def_best")
    )
    try:
        name = f"rmsnorm_h{hidden}"
        hit = apply(name, kwargs={"hidden_states": x, "weight": w}, fallback=lambda **k: None)
        if hit is None:
            return None
        direct = None
        try:
            import vllm_xpu_kernels._C  # noqa: F401

            out = torch.empty_like(x)
            direct = _median_of(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6), device, 300, rounds)
        except Exception:
            return None
        through = _median_of(
            lambda: apply(
                name, kwargs={"hidden_states": x, "weight": w}, fallback=lambda **k: None
            ),
            device,
            300,
            rounds,
        )
        return max(0.0, through - direct)
    finally:
        runtime.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset", default="tmp/flashinfer-trace")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--buffer-mib", type=int, default=512)
    args = ap.parse_args()

    from flashinfer_bench.device import default_device_type, get_accelerator

    accel = get_accelerator(args.device or default_device_type())
    device = args.device or accel.list_devices()[0]
    caps = accel.capabilities(device)
    print(
        f"\n  part: {caps.canonical_id}   device: {device}   timer: {accel.make_timer(device).name}"
    )
    print(
        f"  reported L2: {caps.l2_bytes / 1024 / 1024:.0f} MiB   sycl_target: {caps.sycl_target}\n"
    )

    gb = bandwidth(device, args.buffer_mib, args.rounds)
    print(f"  achievable read bandwidth   {gb:8.1f} GB/s")
    print("     the ceiling for a contiguous stream. A kernel's own ceiling is lower: read")
    print("     the bytes in the shape it must touch, and never treat that as a hardware bound.")

    floor = timing_floor(device, args.rounds)
    print(f"\n  per-call timing overhead    {floor:8.2f} us")
    print(f"     -> rank_vs_provider.py --floor-us {max(1, round(floor * 1.5))}")

    cost = dispatch_cost(device, args.dataset, args.rounds)
    if cost is None:
        print("\n  substitution cost           unavailable")
        print("     No definition here both matches this part and has a benchmarked solution.")
        print("     Benchmark one, then re-run; do not guess this number -- the deploy gate")
        print("     is built on it.")
    else:
        print(f"\n  substitution cost           {cost:8.2f} us")
        print(f"     -> ApplyConfig(min_gain_us={cost:.2f})  /  FIB_APPLY_MIN_GAIN_US={cost:.2f}")
        print("     A kernel saving less than this loses the exchange however large its ratio.")
    print()


if __name__ == "__main__":
    main()
