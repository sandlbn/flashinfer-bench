"""Rank solutions against the provider kernel, not against the definition's reference.

`speedup_factor` in a trace is measured against the definition's `reference` -- a plain
PyTorch implementation that exists to decide *correctness*. It is not what a served model
runs, and it is frequently much worse: a norm reference makes several passes over memory, a
gated-MLP reference computes two separate projections where the server issues one merged
GEMM. Ratios taken against it overstate the win by whatever that gap happens to be, and
kernels recorded at 2-3x have measured at parity or worse against the kernel they would
actually replace.

The comparison that decides deployment is against the provider baseline (`vllm-xpu`,
`sgl-kernel-xpu`), which `add-baselines` already writes and the benchmark already times on
the same workloads. This computes it from traces on disk; it runs no kernels.

Two things the raw ratio does not tell you, and this does:

* **The saving has to clear what a substitution costs.** A 2.6x win on a 4us kernel saves
  2.5us and costs ~5.9us of dispatch to obtain -- a net loss.
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

PROVIDERS = ("vllm_xpu", "sgl_kernel_xpu")
"""Solution-name markers for the kernel a real deployment would otherwise run."""

Bucket = List[Tuple[float, float]]  # (ours_ms, provider_ms)


def _is_provider(solution_name: str) -> bool:
    return any(f"__{p}" in solution_name for p in PROVIDERS)


def collect(
    dataset: pathlib.Path, hardware_id: str
) -> Dict[str, Dict[Tuple[str, Optional[int]], Dict[str, float]]]:
    """{definition: {(workload, tokens): {solution: latency_ms}}} over PASSED traces here.

    Only this part's traces: a latency from another device is not comparable, and mixing
    them silently produces a ratio between two machines.
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
            if (ev.get("environment") or {}).get("hardware_id") != hardware_id:
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
        default=5.91,
        help="Cost of one successful apply() substitution, measured on Arc B580. "
        "Re-measure on your part; 0 ignores it.",
    )
    ap.add_argument(
        "--floor-us",
        type=float,
        default=60.0,
        help="Drop workloads where every measurement is below this: device-event "
        "timing has a fixed cost and below it the ratio measures the harness.",
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

    data = collect(args.dataset, hardware_id)
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
