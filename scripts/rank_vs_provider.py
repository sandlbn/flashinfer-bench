"""Rank solutions against the provider kernel, not against the definition's reference.

`speedup_factor` in a trace is measured against the definition's `reference` -- a plain
PyTorch implementation that exists to decide *correctness*. It is not what a served model
runs, and it is frequently much worse: a norm reference makes several passes over memory, a
gated-MLP reference computes two separate projections where the server issues one merged
GEMM. Ratios taken against it overstate the win by whatever that gap happens to be, and
kernels have measured at parity or worse against the kernel they would actually replace
despite a large ratio against the reference.

The comparison that decides deployment is against the provider baseline (`vllm-xpu`,
`sgl-kernel-xpu`), which `add-baselines` already writes and the benchmark already times on
the same workloads. This computes it from traces on disk; it runs no kernels.

Two things the raw ratio does not tell you, and this does:

* **The saving has to clear what a substitution costs.** That cost does not shrink with the
  kernel, so a large ratio on a small kernel still loses the exchange. The figure is measured
  on this part by `flashinfer_bench.device.calibration`, not carried here.
* **The saving is bimodal.** A few microseconds at decode sizes and hundreds at prefill
  sizes, so a single median lands between the clusters and describes neither. Serving spends
  most of its calls at decode, which is why that column decides.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Dict, List, Optional, Tuple

from flashinfer_bench.apply.table import _PROVIDER_MARKERS as PROVIDERS  # noqa: E402

"""Shared with the apply table, so the ranking and the deploy gate cannot disagree about
which solutions represent the kernel a deployment would otherwise run."""

Bucket = List[Tuple[float, float]]  # (ours_ms, provider_ms)


def _is_provider(solution_name: str) -> bool:
    return any(f"__{p}" in solution_name for p in PROVIDERS)


def collect(
    dataset: pathlib.Path, hardware_id: str, timing: Optional[str] = None
) -> Dict[str, Dict[Tuple[str, Optional[int]], Dict[str, float]]]:
    """{definition: {(workload, tokens): {solution: latency_ms}}} over PASSED traces here.

    Only this part's traces, and only this timing methodology: a latency from another
    device or another timer is not comparable, and mixing either silently produces a ratio
    between two things that were never measured the same way.
    """
    out: Dict[str, Dict[Tuple[str, Optional[int]], Dict[str, float]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    for path in (dataset / "traces").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trace = json.loads(line)
            ev = trace.get("evaluation") or {}
            if ev.get("status") != "PASSED":
                continue
            env = ev.get("environment") or {}
            if env.get("hardware_id") != hardware_id:
                continue
            # Latencies from different timing methodologies are not comparable: the earlier
            # per-call event timer charged its own overhead to every call, several times a
            # short kernel's true latency, so
            # ranking a stale trace against a fresh one invents a win out of the change in
            # measurement. Traces recording no methodology are kept; most predate the field.
            recorded = (env.get("libs") or {}).get("timing")
            if timing and recorded and recorded != timing:
                continue
            latency = (ev.get("performance") or {}).get("latency_ms")
            solution = trace.get("solution")
            if not latency or not solution:
                continue
            workload = trace["workload"]
            axes = workload.get("axes") or {}
            tokens = axes.get("batch_size") or axes.get("M") or axes.get("num_tokens")
            out[trace["definition"]][(workload["uuid"], tokens)][solution] = latency
    return out


def _best_pair(by_solution: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """(ours, provider) fastest of each, or None when one side is missing."""
    provider = [v for s, v in by_solution.items() if _is_provider(s)]
    ours = [v for s, v in by_solution.items() if not _is_provider(s)]
    return (min(ours), min(provider)) if provider and ours else None


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _stats(bucket: Bucket) -> Tuple[Optional[float], Optional[float]]:
    if not bucket:
        return None, None
    return _median([p / o for o, p in bucket]), _median([p - o for o, p in bucket])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("tmp/flashinfer-trace"))
    ap.add_argument("--hardware-id", default=None, help="Defaults to this machine's part.")
    ap.add_argument("--min-ratio", type=float, default=1.05)
    ap.add_argument(
        "--dispatch-us",
        type=float,
        default=None,
        help="Cost of one successful apply() substitution. Measured on this part when "
        "omitted; 0 ignores it.",
    )
    ap.add_argument(
        "--floor-us",
        type=float,
        default=None,
        help="Drop workloads where every measurement is below this. Measured on this part "
        "when omitted (the calibration's timing floor, with margin); 0 disables it. The "
        "floor exists to hide measurements dominated by timer overhead, so applying one "
        "to traces from a timer that amortizes it discards exactly the decode-sized data "
        "that decides deployment.",
    )
    ap.add_argument(
        "--decode-max-tokens",
        type=int,
        default=1024,
        help="At or below this token count counts as decode-sized.",
    )
    ap.add_argument("--show", choices=("candidates", "all"), default="candidates")
    args = ap.parse_args()

    hardware_id = args.hardware_id
    if hardware_id is None:
        from flashinfer_bench.device import default_device_type, get_accelerator

        accel = get_accelerator(default_device_type())
        hardware_id = accel.canonical_id(accel.list_devices()[0])

    timing = None
    try:
        from flashinfer_bench.device import default_device_type, get_accelerator

        accel = get_accelerator(default_device_type())
        timing = accel.make_timer(accel.list_devices()[0]).name
    except Exception:
        pass
    # Both thresholds are properties of this part, so they are measured rather than
    # carried as constants -- a constant is wrong on the next chip and stale on this one.
    calibration = None
    if args.dispatch_us is None or args.floor_us is None:
        from flashinfer_bench.device.calibration import get as get_calibration

        calibration = get_calibration()
    if args.dispatch_us is None:
        args.dispatch_us = calibration.dispatch_us if calibration else 0.0
        if calibration is None:
            print("  note: substitution cost could not be measured here; not gating on it.")
    if args.floor_us is None and calibration is not None:
        args.floor_us = calibration.timing_floor_us * 1.5

    if args.floor_us is None:
        # The floor exists to hide measurements dominated by timer overhead. Only the old
        # per-call event timer had overhead worth hiding; keeping its floor for a timer that
        # amortizes it would throw away every decode-sized comparison as "unmeasurable".
        # No constant: the floor is a property of this part and this timer, so it is
        # measured. Only the superseded per-call methodology had a floor worth hiding, and
        # even that is measured rather than assumed.
        args.floor_us = (calibration.timing_floor_us * 1.5) if calibration else 0.0

    data = collect(args.dataset, hardware_id, timing)
    dispatch_ms = args.dispatch_us / 1000.0
    rows, no_provider, only_floor = [], [], []

    for definition, workloads in sorted(data.items()):
        decode: Bucket = []
        prefill: Bucket = []
        paired = 0
        for (_uuid, tokens), by_solution in workloads.items():
            pair = _best_pair(by_solution)
            if pair is None:
                continue
            paired += 1
            if max(pair) * 1000 < args.floor_us:
                continue
            (prefill if (tokens or 0) > args.decode_max_tokens else decode).append(pair)
        if not (decode or prefill):
            # Distinguish the two reasons: nothing to compare against, versus a comparison
            # that exists but is entirely inside the timing floor. They call for different
            # fixes -- run add-baselines, or fix the measurement.
            (only_floor if paired else no_provider).append(definition)
            continue
        d_ratio, d_saved = _stats(decode)
        p_ratio, p_saved = _stats(prefill)
        rows.append((definition, d_ratio, d_saved, p_ratio, p_saved, len(decode), len(prefill)))

    def net(saved: Optional[float]) -> Optional[float]:
        return None if saved is None else saved - dispatch_ms

    rows.sort(key=lambda r: -(net(r[2]) or -1e9))

    def cell(ratio, value):
        return "        -        " if ratio is None else f"{ratio:6.2f}x {value * 1000:8.1f}us"

    print(f"\n  hardware: {hardware_id}   ours vs the provider kernel; >1 means ours is faster\n")
    print(f"  {'definition':44} {'decode, net of dispatch':>18} {'prefill, gross':>18}   n")
    print("  " + "-" * 92)
    shown = 0
    for definition, d_ratio, d_saved, p_ratio, p_saved, n_d, n_p in rows:
        worth = d_ratio is not None and d_ratio >= args.min_ratio and (net(d_saved) or 0) > 0
        if args.show == "candidates" and not worth:
            continue
        shown += 1
        print(
            f"  {definition[:44]:44} {cell(d_ratio, net(d_saved)):>18} "
            f"{cell(p_ratio, p_saved):>18}   {n_d}/{n_p}{'  <- deploy' if worth else ''}"
        )
    if not shown:
        print("  (none)")

    worth_n = sum(
        1 for r in rows if r[1] is not None and r[1] >= args.min_ratio and (net(r[2]) or 0) > 0
    )
    print(
        f"\n  {len(rows)} comparable on this part. {worth_n} beat the provider at decode "
        f"sizes by >={args.min_ratio}x AND save more than the {args.dispatch_us:.2f}us a "
        "substitution costs."
    )
    print(
        "  n is decode/prefill workloads compared. A large prefill win with no decode win "
        "moves little in a\n  serving run, which is mostly decode."
    )
    if no_provider:
        print(
            f"  {len(no_provider)} have Intel traces but no provider baseline to compare against."
        )
    if only_floor:
        print(
            f"  {len(only_floor)} had every measurement inside the {args.floor_us:.0f}us "
            "timing floor and were dropped as unmeasurable."
        )


if __name__ == "__main__":
    main()
