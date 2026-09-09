"""Which GEMM-epilogue fusions this model's own execution would support.

Every op the resolver classifies is routed on its own, and on this hardware that ends the
same way each time: the GEMM is already oneDNN, and each elementwise kernel is too small to
pay back what substituting it costs. Fusing the elementwise work into the GEMM's epilogue is
the one route that escapes both -- the operand is still in registers, so the saving is a
launch and a round trip through memory rather than a faster loop.

A fusion is only real if the model runs the two ops back to back, which is a property of the
edge between them and cannot be read off either op's call count. Discovery records those
edges; this reads them, and proposes only pairs that actually occurred.

The preset list is read from Xe-Fuse itself (`generate_kernel.py --list-presets`) rather
than copied here, so it cannot drift from the tool that has to generate the kernel.

    python scripts/fusion_candidates.py --report tools/kernel-harness/auto/discovered.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# What an epilogue preset's description talks about, and the op vocabulary that means the
# same thing. This is a translation between two projects' words for one operation, not a
# routing policy: the routing is decided by the edges the model ran.
_VOCABULARY = {
    "rope": ("rotary", "rope"),
    "swiglu": ("silu_and_mul", "swiglu"),
    "geglu": ("gelu_and_mul", "geglu"),
    "residual": ("fused_add", "residual"),
    "scale": ("rms_norm", "rmsnorm", "scale"),
    "gamma": ("rms_norm", "rmsnorm"),
    "dequant": ("dequant", "scaled_mm", "quant"),
}

# The op whose accumulator an epilogue would consume. Anything that resolves to a matmul
# qualifies; the resolver already establishes which ops those are.
_GEMM_HINTS = ("linear", "matmul", "mm", "addmm", "bmm", "scaled_mm")


def presets(xe_fuse: pathlib.Path) -> List[Tuple[str, str]]:
    """`(name, description)` straight from the generator, so the two cannot drift apart."""
    gen = xe_fuse / "autotune" / "generate_kernel.py"
    if not gen.is_file():
        return []
    try:
        r = subprocess.run(
            [sys.executable, str(gen), "--list-presets"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.splitlines():
        m = re.match(r"^\s{2,}(\w+)\s{2,}(\S.*)$", line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def _terms(text: str) -> set:
    lowered = text.lower()
    return {k for k, words in _VOCABULARY.items() if any(w in lowered for w in words)}


def _bare(op: str) -> str:
    return op.split(".")[1] if "." in op else op


def match(consumer: str, preset_name: str, preset_desc: str) -> bool:
    """Does this preset's epilogue do what the consuming op does?"""
    want = _terms(f"{preset_name} {preset_desc}")
    have = _terms(_bare(consumer))
    if not want or not have:
        return False
    # Every operation the preset performs must be one the consumer performs. Overlap alone
    # is far too generous: a quantized-dequant preset shares the word "scale" with a plain
    # norm and would be offered for a model that quantizes nothing.
    return want <= have


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="tools/kernel-harness/auto/discovered.json")
    ap.add_argument("--xe-fuse", default="tmp/Xe-Fuse")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.report).read_text())
    edges = data.get("edges") or []
    if not edges:
        print("No edges recorded. Re-run discovery: fusion cannot be judged from op counts.")
        return

    pres = presets(pathlib.Path(args.xe_fuse))
    if not pres:
        print(f"Xe-Fuse not readable at {args.xe_fuse}; clone it to propose epilogue presets.")
        return

    shares: Dict[str, float] = {
        op: v.get("share_pct", 0.0) for op, v in (data.get("op_share") or {}).items()
    }
    by_kernel = data.get("device_time_by_kernel") or {}
    total = float(data.get("device_time_total_us") or 0.0)

    def gemm_share() -> Optional[float]:
        # The GEMM's own time is not on the op that launched it when that op decomposes.
        us = sum(v for k, v in by_kernel.items() if "gemm" in k.lower())
        return 100.0 * us / total if us and total else None

    print(f"\nEdges this model ran (>= 4 times), against Xe-Fuse's {len(pres)} presets\n")
    print(f"{'producer -> consumer':58} {'share':>7}  preset(s)")
    print("-" * 100)

    proposed = []
    for e in edges:
        prod, cons, n = e["producer"], e["consumer"], e["count"]
        if not any(h in _bare(prod) for h in _GEMM_HINTS):
            continue
        hits = [name for name, desc in pres if match(cons, name, desc)]
        if not hits:
            continue
        sh = shares.get(cons, 0.0)
        pair = f"{_bare(prod)} -> {_bare(cons)}  (x{n})"
        print(f"{pair:58} {sh:6.2f}%  {', '.join(hits)}")
        proposed.append((pair, sh, hits))

    if not proposed:
        print("(none -- no GEMM feeds an op any preset covers in this model)")
        return

    g = gemm_share()
    reach = sum(sh for _, sh, _ in proposed)
    print("-" * 100)
    print(f"\nElementwise time reachable by fusion: {reach:.2f}% of device time")
    if g is not None:
        print(f"GEMM the epilogue would attach to:    {g:.2f}%")
    print(
        "\nA fusion removes a launch and a round trip, so its value is that reachable share\n"
        "and not a kernel ratio -- and it pays no substitution cost, because the fused GEMM\n"
        "replaces the call the stack already makes. Generate one with:\n"
        "    python tmp/Xe-Fuse/autotune/generate_kernel.py --preset <name> --m M --n N --k K \\\n"
        "        -o kernel.cpp\n"
        "then build and measure it. Xe-Fuse is marked not stable by IntelLabs: treat a\n"
        "regression as losing a contender, never as breaking the pipeline.\n"
        "Build flags and the operand-layout traps: "
        ".claude/skills/optimize-intel-kernels/xe-fuse.md"
    )


if __name__ == "__main__":
    main()
