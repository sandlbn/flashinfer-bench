"""Drop traces whose timing methodology has been superseded on the same hardware.

Latencies measured with different timers are not comparable, and the dataset validator
treats a mix on one part as an error. When a timer is corrected, the old numbers are not
merely different but wrong -- the per-call event timer reported about 8x the true latency
for a short kernel -- so there is nothing to preserve by keeping them beside the new ones.

Only (definition, hardware) groups holding *both* methodologies are touched, and only the
superseded traces in them are removed. A part measured entirely with an older timer is left
alone: its traces are self-consistent and still rank correctly among themselves.

Dry by default. Pass --write to modify the dataset, which rewrites trace files in place.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Dict, List, Set, Tuple


def _timing(trace: dict) -> str:
    env = ((trace.get("evaluation") or {}).get("environment")) or {}
    return (env.get("libs") or {}).get("timing") or ""


def _hardware(trace: dict) -> str:
    env = ((trace.get("evaluation") or {}).get("environment")) or {}
    return env.get("hardware_id") or env.get("hardware") or "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("tmp/flashinfer-trace"))
    ap.add_argument(
        "--current",
        default=None,
        help="Methodology to keep. Defaults to what this machine's timer reports.",
    )
    ap.add_argument("--write", action="store_true", help="Apply. Without it, report only.")
    args = ap.parse_args()

    current = args.current
    if current is None:
        from flashinfer_bench.device import default_device_type, get_accelerator

        accel = get_accelerator(default_device_type())
        current = accel.make_timer(accel.list_devices()[0]).name

    # Which (definition, hardware) groups hold the current methodology alongside another.
    seen: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    files = sorted((args.dataset / "traces").rglob("*.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trace = json.loads(line)
            timing = _timing(trace)
            if timing:
                seen[(trace["definition"], _hardware(trace))].add(timing)

    superseded = {k for k, v in seen.items() if current in v and len(v) > 1}
    if not superseded:
        print(f"  nothing to do: no (definition, hardware) group mixes {current} with another")
        return

    removed_total = 0
    touched: List[str] = []
    for path in files:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        kept = []
        removed = 0
        for line in lines:
            trace = json.loads(line)
            key = (trace["definition"], _hardware(trace))
            timing = _timing(trace)
            if key in superseded and timing and timing != current:
                removed += 1
                continue
            kept.append(line)
        if removed:
            removed_total += removed
            touched.append(f"{path.relative_to(args.dataset)}: -{removed} of {len(lines)}")
            if args.write:
                path.write_text("".join(line + "\n" for line in kept))

    verb = "removed" if args.write else "would remove"
    print(
        f"  {verb} {removed_total} superseded trace(s) across {len(touched)} file(s), "
        f"keeping {current}, in {len(superseded)} (definition, hardware) group(s)"
    )
    for line in touched[:20]:
        print(f"    {line}")
    if len(touched) > 20:
        print(f"    ... and {len(touched) - 20} more")
    if not args.write:
        print("\n  Re-run with --write to apply.")


if __name__ == "__main__":
    main()
