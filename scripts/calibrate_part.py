"""Measure the part-specific constants the tuning and deploy gates depend on.

Every threshold in this repo's Intel work is a measurement of one GPU, not a property of the
software: what a kernel substitution costs, where the timing instrument's floor sits, what
bandwidth is actually reachable, and at what row pitch a weight camps on one memory channel.
On new silicon all of them move, and a gate carrying the old numbers is silently wrong -- it
will deploy kernels that lose, refuse kernels that win, or pad the wrong weights, without
saying anything.

Run this first on a new part. It prints the values, and the flags that carry them.

    python scripts/calibrate_part.py
    python scripts/calibrate_part.py --pattern 256:4096     # bandwidth of a strided read

Every device-side number here is taken through ``flashinfer_bench.device.calibration`` so
this script and the gates that consume the record cannot disagree, and each is accepted
only once its rounds settle; a number that never settles prints as ``unavailable`` rather
than as whatever the clock ramp produced.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List, Tuple

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


def dispatch_cost(device: str, dataset: str) -> float | None:
    """What one apply() call costs on top of the kernel it dispatches to, in microseconds.

    Delegates to ``flashinfer_bench.device.calibration`` so that this script and the deploy
    gate cannot disagree about the number. The measurement is host-side -- the kernel is
    stubbed out for its duration -- so it does not depend on the GPU's clock state, which
    is what made an earlier version of it here read anywhere from zero to its true value.

    Returns None when it could not be measured: no definition on this dataset that apply()
    can dispatch here, or rounds that did not agree (another process on the box). The
    number is not invented in either case -- the deploy gate is built on it.
    """
    from flashinfer_bench.device import calibration

    return calibration.measure_dispatch_us(device, dataset)


def _parse_pattern(text: str) -> Tuple[int, int]:
    try:
        run, stride = (int(x) for x in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"pattern must be RUN_BYTES:STRIDE_BYTES, got {text!r}"
        ) from exc
    return run, stride


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset", default="tmp/flashinfer-trace")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--buffer-mib", type=int, default=512)
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        type=_parse_pattern,
        metavar="RUN_BYTES:STRIDE_BYTES",
        help="also measure the bandwidth of reading RUN contiguous bytes out of every STRIDE "
        "(a kernel's access pattern); repeatable",
    )
    args = ap.parse_args()

    from flashinfer_bench.device import calibration

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

    launch = calibration.measure_launch_floor_us(device)
    if launch is None:
        print("\n  launch floor                unavailable (rounds did not settle)")
    else:
        print(f"\n  launch floor                {launch:8.2f} us per launch")
    print("     what the smallest kernel costs inside a batch of launches. A kernel whose")
    print("     device time sits here is launch-bound: only removing the launch can win.")

    peak = calibration.measure_matmul_peak_tflops(device)
    print(f"\n  matrix throughput (square matmul, side {calibration.MATMUL_PROBE_SIDE})")
    for name in calibration.MATMUL_DTYPES:
        if name not in peak:
            print(f"     {name:10} not native on this part; not probed")
        elif peak[name] is None:
            print(f"     {name:10} unavailable (rounds did not settle)")
        else:
            print(f"     {name:10} {peak[name]:8.1f} TFLOP/s")
    print("     the denominator of t_cmp = flops / peak; what torch.matmul reaches here.")

    for run, stride in args.pattern:
        bw_pattern = calibration.strided_read_bandwidth_gbs(device, run, stride)
        if bw_pattern is None:
            print(f"\n  strided read {run}:{stride:<12} unavailable (rounds did not settle)")
        else:
            print(f"\n  strided read {run}:{stride:<12} {bw_pattern:8.1f} GB/s of useful bytes")
            print(f"     against {gb:.1f} GB/s contiguous: t_mem_pattern for a kernel obliged to")
            print("     read in runs of this length.")

    period = calibration.measure_channel_period_bytes(device)
    if period is None:
        print("\n  memory channel period       unavailable (no periodic slow pitch resolved)")
        print("     Either the pitches did not settle (another process on the device?) or no")
        print("     pitch in the sweep streamed slow. The row-pad weight transform stands down")
        print("     on this part until a sweep resolves one; FIB_CHANNEL_PERIOD_BYTES carries")
        print("     a period measured another way.")
    else:
        print(f"\n  memory channel period       {period:8d} bytes")
        print("     row pitches that are multiples of this put every row on one memory channel;")
        print("     the load-time row pad nominates weights by it and keeps a pad only on a")
        print("     measured win for that shape.")

    cost = dispatch_cost(device, args.dataset)
    if cost is None:
        print("\n  substitution cost           unavailable")
        print("     Either no definition here has a solution apply() can dispatch on this part,")
        print("     or the timed rounds did not agree (another process on the box?). Benchmark")
        print("     one / retry on an idle box; do not guess this number -- the deploy gate is")
        print("     built on it, and the vLLM sitecustomize refuses to enable apply() without it.")
    else:
        print(f"\n  substitution cost           {cost:8.2f} us")
        print(f"     -> ApplyConfig(min_gain_us={cost:.2f})  /  FIB_APPLY_MIN_GAIN_US={cost:.2f}")
        print("     A kernel saving less than this loses the exchange however large its ratio.")
    print()


if __name__ == "__main__":
    main()
