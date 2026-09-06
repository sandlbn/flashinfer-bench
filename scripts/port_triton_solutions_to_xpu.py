"""Port CUDA-bound Triton solutions in a trace dataset so they run on any accelerator.

Triton compiles for whichever backend its tensors live on, so a Triton *kernel* written
for CUDA generally runs unchanged on Intel. What stops it is the Python wrapper around it:
every Triton solution in the dataset today refuses on XPU, and none of them refuse in the
kernel. They refuse in hand-written device management --

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This kernel requires a CUDA GPU.")

-- or by moving inputs with ``.cuda()``, or by allocating on ``torch.device("cuda")``. The
kernel underneath would have compiled and run.

This script rewrites those constructs to derive the device from the inputs, and writes the
result as a **new** solution rather than editing the original. That matters: the dataset
records which model produced which solution, and a solution that says "requires CUDA" is a
faithful record of what that model wrote. Mutating it would destroy that record and
attribute code to an author who did not write it. The port is a derived artifact, authored
separately, naming its origin.

Correctness is not assumed. A port that changes behaviour fails the benchmark's
correctness gate like any other solution, which is the point of running it afterwards:

    python scripts/port_triton_solutions_to_xpu.py --local tmp/flashinfer-trace --dry-run
    python scripts/port_triton_solutions_to_xpu.py --local tmp/flashinfer-trace
    flashinfer-bench run --local tmp/flashinfer-trace --definitions <name>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("port-triton")

PORT_AUTHOR = "xpu-port"
"""Author for ported solutions, so they never masquerade as the original."""

_DEVICE_HELPER = '''
def _fib_device(*candidates):
    """Device to run on, taken from the inputs rather than assumed.

    Replaces the vendor-pinned availability check, device constructor and tensor-move
    calls the original used. Triton itself is portable; only this wrapper was not.
    """
    import torch

    for c in candidates:
        if isinstance(c, torch.Tensor):
            return c.device
    for name in ("cuda", "xpu", "mps"):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            return torch.device(name)
    return torch.device("cpu")


def _fib_accelerator_available():
    """Whether any non-CPU accelerator is present."""
    import torch

    for name in ("cuda", "xpu", "mps"):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            return True
    return False

'''


def _rewrite(source: str) -> Tuple[str, List[str]]:
    """Rewrite vendor-pinned device management. Returns (source, applied rule names)."""
    applied: List[str] = []

    def sub(pattern: str, repl: str, name: str, text: str, flags: int = 0) -> str:
        new, n = re.subn(pattern, repl, text, flags=flags)
        if n:
            applied.append(f"{name} x{n}")
        return new

    s = source

    # A guard whose whole purpose is to refuse on non-CUDA hardware. Widened rather than
    # deleted: the intent (refuse if there is no accelerator at all) is still reasonable.
    s = sub(r"torch\.cuda\.is_available\(\)", "_fib_accelerator_available()", "is_available", s)
    s = sub(r"\btorch\.cuda\.current_device\(\)", "_fib_device()", "current_device", s)
    s = sub(r"\btorch\.cuda\.synchronize\(\)", "torch.accelerator.synchronize()", "synchronize", s)

    # Explicit device construction.
    s = sub(
        r"torch\.device\(\s*[\"']cuda(?::\d+)?[\"']\s*\)",
        "_fib_device(*_fib_inputs)",
        "torch.device(cuda)",
        s,
    )
    s = sub(
        r"device\s*=\s*[\"']cuda(?::\d+)?[\"']",
        "device=_fib_device(*_fib_inputs)",
        "device='cuda'",
        s,
    )

    # Moving a tensor onto the vendor device.
    s = sub(r"\.cuda\(\)", ".to(_fib_device(*_fib_inputs))", ".cuda()", s)

    # Vendor-specific predicates on tensors.
    s = sub(r"\.is_cuda\b", '.device.type != "cpu"', ".is_cuda", s)

    if not applied:
        return source, []

    # The rewrites reference the call's inputs; bind them at the top of `run`.
    def bind_inputs(match: re.Match) -> str:
        head = match.group(0)
        return head + "\n    _fib_inputs = tuple(args) + tuple(kwargs.values())\n"

    if re.search(r"^def run\(\*args,\s*\*\*kwargs\).*:\n", s, flags=re.M):
        s = re.sub(r"^def run\(\*args,\s*\*\*kwargs\).*:\n", bind_inputs, s, count=1, flags=re.M)
    else:
        m = re.search(r"^def run\(([^)]*)\).*:\n", s, flags=re.M)
        if m:
            names = [
                p.split(":")[0].split("=")[0].strip()
                for p in m.group(1).split(",")
                if p.strip() and not p.strip().startswith("*")
            ]
            binding = (
                "    _fib_inputs = (" + ", ".join(names) + ",)\n"
                if names
                else "    _fib_inputs = ()\n"
            )
            s = s[: m.end()] + binding + s[m.end() :]
        else:
            # No `run` to bind in: the rewrite would reference an undefined name.
            return source, []

    return _DEVICE_HELPER + "\n" + s, applied


def port_solution(path: Path, out_root: Path, dry_run: bool) -> Tuple[bool, str]:
    solution = json.loads(path.read_text())
    if solution["spec"]["language"] != "triton":
        return False, "not triton"
    if solution.get("author") == PORT_AUTHOR:
        return False, "already a port"

    rewritten, applied = [], []
    for src in solution["sources"]:
        new, rules = _rewrite(src["content"])
        rewritten.append({**src, "content": new})
        applied += rules
    if not applied:
        return False, "nothing vendor-pinned"

    original = solution["name"]
    ported = {
        **solution,
        "name": f"{original}__xpu",
        "author": PORT_AUTHOR,
        "sources": rewritten,
        "spec": {**solution["spec"], "target_hardware": ["xpu"]},
        "description": (
            f"Device-agnostic port of '{original}' (author: {solution.get('author', '?')}). "
            f"The Triton kernel is unchanged; only the wrapper's device management was "
            f"rewritten to derive the device from its inputs. Applied: {', '.join(sorted(set(applied)))}."
        ),
    }
    dest = out_root / f"{ported['name']}.json"
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(ported, indent=2) + "\n")
    return True, ", ".join(sorted(set(applied)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True, help="Trace dataset root.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--definitions",
        type=str,
        default=None,
        help="Comma-separated definition names to restrict to.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s")

    wanted = set(args.definitions.split(",")) if args.definitions else None
    ported = skipped = 0
    for path in sorted((args.local / "solutions").rglob("*.json")):
        if f"/{PORT_AUTHOR}/" in str(path):
            continue
        try:
            solution = json.loads(path.read_text())
        except Exception:
            continue
        if wanted and solution.get("definition") not in wanted:
            continue
        out_root = args.local / "solutions" / PORT_AUTHOR / solution["definition"]
        ok, note = port_solution(path, out_root, args.dry_run)
        if ok:
            ported += 1
            logger.info("%-52s %s", solution["name"], note)
        else:
            skipped += 1
    verb = "Would port" if args.dry_run else "Ported"
    logger.info("%s %d solution(s); skipped %d", verb, ported, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
