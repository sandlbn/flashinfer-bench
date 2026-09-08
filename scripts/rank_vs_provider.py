"""Rank solutions against the provider kernel, not against the definition's reference.

`speedup_factor` in a trace is measured against the definition's `reference` -- a plain
PyTorch implementation that exists to decide *correctness*. It is not what a served model
runs, and it is frequently much worse: a norm reference makes several passes over memory, a
gated-MLP reference computes two separate projections where the server issues one merged
GEMM. Ratios taken against it therefore overstate the win, sometimes by the whole factor
that matters -- kernels recorded at 2-3x have measured at parity or worse against the
kernel they would actually replace.

The comparison that decides whether to deploy something is against the provider baseline
(`vllm-xpu`, `sgl-kernel-xpu`), which `add-baselines` already writes and the benchmark
already times on the same workloads. This computes that comparison from the traces on
disk; it runs no kernels.

Read the output as: "ours/provider > 1 means our solution is faster than the kernel vLLM or
SGLang would otherwise have run." Anything at or below 1.0 is not a deployment candidate,
whatever its speedup_factor says.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Dict, List, Optional, Tuple

PROVIDERS = ("vllm-xpu", "sgl-kernel-xpu")
"""Trace authors that represent the kernel a real deployment runs."""


def _provider_of(solution_name: str) -> Optional[str]:
    for p in PROVIDERS:
        if (
            solution_name.endswith(f"__{p.replace('-', '_')}")
            or f"__{p.replace('-', '_')}_" in solution_name
        ):
            return p
    return None


def collect(dataset: pathlib.Path, hardware_id: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{definition: {workload_uuid: {solution: latency_ms}}} for PASSED traces here.

    Only this part's traces are considered: a latency measured on another device is not
    comparable, and mixing them silently produces a ratio between two different machines.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(
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
            latency = (ev.get("performance") or {}).get("latency_ms")
            solution = trace.get("solution")
            if not latency or not solution:
                continue
            out[trace["definition"]][trace["workload"]["uuid"]][solution] = latency
    return out


def _split(
    by_solution: Dict[str, float],
) -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
    """Best provider entry and best non-provider entry for one workload, by latency."""
    provider = [(s, v) for s, v in by_solution.items() if _provider_of(s)]
    ours = [(s, v) for s, v in by_solution.items() if not _provider_of(s)]
    return (
        min(provider, key=lambda kv: kv[1]) if provider else None,
        min(ours, key=lambda kv: kv[1]) if ours else None,
    )


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("tmp/flashinfer-trace"))
    ap.add_argument("--hardware-id", default=None, help="Defaults to this machine's part.")
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=1.05,
        help="Ratio above which a definition counts as a deployment candidate.",
    )
    ap.add_argument(
        "--floor-us",
        type=float,
        default=60.0,
        help="Workloads whose measurements are all below this are dropped: device-event "
        "timing has a fixed cost, and below it the ratio is measuring the harness.",
    )
    ap.add_argument("--show", choices=("candidates", "all"), default="candidates")
    ap.add_argument(
        "--dispatch-us",
        type=float,
        default=5.91,
        help="What one successful apply() substitution costs, on top of the kernel. A ratio "
        "above 1 is not a deployment case unless the time saved exceeds this: measured at "
        "5.91us on Arc B580, so re-measure on your part. Set 0 to ignore.",
    )
    args = ap.parse_args()

    hardware_id = args.hardware_id
    if hardware_id is None:
        from flashinfer_bench.device import default_device_type, get_accelerator

        accel = get_accelerator(default_device_type())
        hardware_id = accel.canonical_id(accel.list_devices()[0])

    data = collect(args.dataset, hardware_id)
    rows, no_provider, only_floor = [], [], []

    for definition, workloads in sorted(data.items()):
        ratios, detail, above_floor = [], [], 0
        for uuid, by_solution in workloads.items():
            provider, ours = _split(by_solution)
            if not provider or not ours:
                continue
            if max(provider[1], ours[1]) * 1000 < args.floor_us:
                continue  # both inside the timing floor; the ratio is noise
            above_floor += 1
            ratios.append(provider[1] / ours[1])
            detail.append((uuid, ours[1], provider[1], provider[1] / ours[1]))
        if not ratios:
            (only_floor if workloads else no_provider).append(definition)
            continue
        # The saving that has to clear dispatch is the absolute time, not the ratio: a 2.6x
        # win on a 4us kernel saves 2.5us and costs 5.9us to obtain.
        saved_ms = _median([p - o for _, o, p, _ in detail])
        rows.append((definition, _median(ratios), min(ratios), max(ratios), above_floor, saved_ms))

    rows.sort(key=lambda r: -r[1])
    print(f"\n  hardware: {hardware_id}   (ours vs the provider kernel; >1 means ours is faster)\n")
    dispatch_ms = args.dispatch_us / 1000.0
    print(f"  {'definition':46} {'median':>8} {'saved':>10} {'net':>10}  n")
    print("  " + "-" * 88)
    shown = 0
    for definition, med, lo, hi, n, saved_ms in rows:
        net_ms = saved_ms - dispatch_ms
        worth = net_ms > 0 and med >= args.min_ratio
        if args.show == "candidates" and not worth:
            continue
        shown += 1
        mark = "  <- worth deploying" if worth else ""
        print(
            f"  {definition[:46]:46} {med:7.2f}x {saved_ms * 1000:8.1f}us "
            f"{net_ms * 1000:8.1f}us  {n}{mark}"
        )
    if not shown:
        print("  (none)")

    beat = [r for r in rows if r[1] >= args.min_ratio]
    worth = [r for r in beat if r[5] - dispatch_ms > 0]
    print(
        f"\n  {len(rows)} comparable here. {len(beat)} beat the provider kernel by "
        f">={args.min_ratio}x, but only {len(worth)} save more than the "
        f"{args.dispatch_us:.2f}us a substitution costs -- the rest win the kernel and lose "
        "the exchange."
    )
    if no_provider:
        print(
            f"  {len(no_provider)} have Intel traces but no provider baseline to compare against."
        )
    if only_floor:
        print(
            f"  {len(only_floor)} had every measurement inside the {args.floor_us:.0f}us timing "
            "floor and were dropped as unmeasurable."
        )


if __name__ == "__main__":
    main()
