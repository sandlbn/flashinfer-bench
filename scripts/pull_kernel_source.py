"""Resolve a recorded op to the kernel that actually implements it, and locate its source.

Discovery says the model called ``_C.fused_add_rms_norm``. That is a name, not a kernel:
the dispatcher decides what runs, and the answer differs per device and per installed
provider. Optimizing before resolving it is how you end up tuning a decomposition, or
rewriting a kernel a library already provides.

This asks the dispatcher rather than a table. ``torch._C._dispatch_dump`` reports which
dispatch keys are registered and the source file each registration came from, so the
providing project falls out of the op itself. The source is then located in whatever
checkouts exist on this box.

With ``--bundle`` it writes each op's harness and its kernel's actual source into one
directory, so optimization starts from the code that runs rather than from a blank file and
a remembered ratio.

    python scripts/pull_kernel_source.py --op _C::fused_add_rms_norm
    python scripts/pull_kernel_source.py --from-harnesses tools/kernel-harness/auto \
        --bundle tools/kernel-harness/pulled
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List, Optional, Tuple

# A registration under one of these keys is not a kernel -- it is a rule for rewriting the
# op into other ops. There is nothing to pull; what runs is whatever the decomposition
# calls, so discovery has already recorded those separately.
_DECOMPOSITION_KEYS = ("CompositeImplicitAutograd", "CompositeExplicitAutograd", "Meta")


def dispatch_report(op: str) -> Tuple[List[str], List[str]]:
    """``(dispatch keys, registration source paths)`` for ``ns::name``, from the dispatcher."""
    import torch

    dump = torch._C._dispatch_dump(op)
    keys, sources = [], []
    for line in dump.splitlines():
        m = re.match(r"^(\w+(?:\[[^\]]*\])?):\s*(.*)$", line.strip())
        if m:
            keys.append(m.group(1))
        if "registered at " in line:
            sources.append(line.split("registered at ")[-1].split(":")[0].strip())
    return keys, sorted(set(sources))


def device_keys(keys: List[str], device: str) -> List[str]:
    """The keys that would actually run on this device, decompositions excluded."""
    want = device.split(":")[0].upper()
    return [k for k in keys if k.split("[")[0].upper() == want]


def search_roots(extra: Optional[List[str]] = None) -> List[pathlib.Path]:
    """Places a kernel's source might live on this box, without naming any project."""
    import site

    roots: List[pathlib.Path] = []
    tmp = pathlib.Path("tmp")
    if tmp.is_dir():
        roots += [p for p in tmp.iterdir() if p.is_dir()]
    for d in site.getsitepackages():
        p = pathlib.Path(d)
        if p.is_dir():
            roots.append(p)
    for name in extra or []:
        roots.append(pathlib.Path(name))
    return roots


_NOT_IMPLEMENTATION = re.compile(r"(^|/)(tests?|benchmarks?|examples?|docs?|\.claude)(/|$)")


def locate(op_name: str, registration_paths: List[str], extra: Optional[List[str]] = None):
    """Find the source implementing ``op_name``, keyed off where it was registered.

    A registration path is a build-time path from the providing project, so its leading
    components name a build machine while its trailing ones describe the project's own
    layout (``vllm_xpu_kernel/csrc/torch_bindings.cpp``). Matching on the tail finds the
    binding file in a local checkout, and the checkout it was found in *is* the providing
    project -- which is what makes the rest of the search precise instead of a grep across
    every repo on the box.

    Ranked, because an op name appears in far more files than implement it: the binding
    site first, then a definition of the symbol, then a mention. Tests and benchmarks are
    excluded -- they name every op and implement none.
    """
    bare = op_name.split("::")[-1]
    tails = set()
    for rp in registration_paths:
        parts = pathlib.PurePosixPath(rp).parts
        for depth in (3, 2, 1):
            if len(parts) >= depth:
                tails.add(str(pathlib.PurePosixPath(*parts[-depth:])))

    binding: List[Tuple[pathlib.Path, str]] = []
    project_roots: List[pathlib.Path] = []
    for root in search_roots(extra):
        for tail in sorted(tails, key=len, reverse=True):
            cand = root / tail
            if cand.is_file():
                binding.append((cand, f"registered here ({tail})"))
                project_roots.append(root)
                break

    # Without a binding site there is no project to scope to, and an unscoped search
    # returns every test that mentions the name. Say so rather than guessing.
    if not project_roots:
        return binding

    defines = re.compile(rf"^[\w:<>,\s\*&]*\b{re.escape(bare)}\s*\(", re.M)
    found: List[Tuple[pathlib.Path, str]] = list(binding)
    for root in project_roots:
        for f in sorted(root.rglob("*")):
            if f.suffix not in (".cpp", ".cu", ".sycl", ".h", ".hpp", ".py"):
                continue
            rel = str(f.relative_to(root))
            if _NOT_IMPLEMENTATION.search(rel) or any(f == b for b, _ in binding):
                continue
            try:
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            if defines.search(text):
                found.append((f, "defines it"))
            elif re.search(rf"\b{re.escape(bare)}\b", text):
                found.append((f, "mentions it"))
    order = {"defines it": 1, "mentions it": 2}
    found.sort(key=lambda t: (0 if t[1].startswith("registered") else order[t[1]], str(t[0])))
    return found[:6]


_PROVENANCE = """# {op}

Pulled by `scripts/pull_kernel_source.py`. Everything here describes what **actually runs**
on `{device}` -- the dispatcher was asked, not a table.

| | |
| --- | --- |
| dispatch key | `{keys}` |
| registered at | `{registered}` |
| providing project | `{project}` |
| harness | `harness.py` -- calls this op at a shape the model called it at |

## Schema

```
{schema}
```

## Source

{files}

The kernel is copied whole rather than sliced: the launcher, the functor and the dispatch
macros are all part of what you are changing, and a slice that keeps only the arithmetic
compiles into a different kernel.

## Before optimizing

Bound it. `flashinfer_bench.device.calibration.get()` gives this part's substitution cost,
timing floor and achievable bandwidth; a win smaller than what taking it costs is a loss.
See `.claude/skills/route-kernel-work/PLAN.md`.

Then: `python scripts/kernel_trials.py init --harness <this dir>/harness.py`
"""


def bundle(
    op: str,
    device: str,
    keys: List[str],
    sources: List[str],
    found: List[Tuple[pathlib.Path, str]],
    harness: Optional[pathlib.Path],
    out: pathlib.Path,
) -> Optional[pathlib.Path]:
    """Write the harness and the kernel's own source into one directory."""
    import shutil

    impl = [(f, why) for f, why in found if why != "mentions it"]
    if not impl:
        return None
    d = out / op.replace("::", "_")
    (d / "source").mkdir(parents=True, exist_ok=True)
    project = None
    copied = []
    for f, why in impl:
        for root in search_roots():
            try:
                rel = f.relative_to(root)
            except ValueError:
                continue
            project = project or root.name
            dest = d / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copied.append((str(rel), why, sum(1 for _ in open(f, errors="ignore"))))
            break
    if harness and harness.is_file():
        shutil.copy2(harness, d / "harness.py")

    import torch

    try:
        schema = str(
            getattr(torch.ops, op.split("::")[0]).__getattr__(op.split("::")[1]).default._schema
        )
    except Exception:
        schema = "(not resolvable)"
    files_md = (
        "\n".join(f"- `source/{rel}` ({n} lines) -- {why}" for rel, why, n in copied)
        or "- (none found on this box)"
    )
    (d / "PROVENANCE.md").write_text(
        _PROVENANCE.format(
            op=op,
            device=device,
            keys=", ".join(keys) or "(none)",
            registered=sources[0] if sources else "(unknown)",
            project=project or "(unknown)",
            schema=schema,
            files=files_md,
        )
    )
    return d


def report(
    op: str,
    device: str,
    extra: Optional[List[str]],
    out: Optional[pathlib.Path] = None,
    harness: Optional[pathlib.Path] = None,
) -> Dict[str, object]:
    keys, sources = dispatch_report(op)
    dev = device_keys(keys, device)
    decomp = [k for k in keys if k.split("[")[0] in _DECOMPOSITION_KEYS]
    print(f"\n=== {op} ===")
    print(f"  dispatch keys : {', '.join(keys) if keys else '(none)'}")
    if dev:
        print(f"  on {device:8}   : {', '.join(dev)}  <- a real kernel runs here")
    elif decomp:
        print(f"  on {device:8}   : none; {decomp[0]} rewrites it into other ops.")
        print("                  There is no kernel to pull -- optimize what it decomposes to,")
        print("                  which discovery recorded separately.")
    else:
        print(f"  on {device:8}   : nothing registered; this op does not run here")
    for s in sources:
        print(f"  registered at : {s}")
    found = locate(op, sources, extra) if dev else []
    seen = set()
    for path, why in found:
        if path in seen:
            continue
        seen.add(path)
        print(f"  source        : {path}  ({why})")
    if dev and not found:
        print("  source        : not on this box. Clone the providing project to tmp/ to")
        print("                  read it, or optimize against the harness alone.")
    written = bundle(op, device, dev, sources, found, harness, out) if out and dev else None
    if written:
        print(f"  bundled to    : {written}  (harness + source + PROVENANCE.md)")
    return {"op": op, "keys": keys, "device_keys": dev, "sources": sources}


def ops_from_harnesses(d: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Map each verified op to its harness, so the two steps agree on the list.

    Discovery emits one harness per (op, shape); several shapes share an op. Keep the one
    with the most calls, which is the shape the model spends its time at.
    """
    best: Dict[str, Tuple[int, pathlib.Path]] = {}
    for f in sorted(d.glob("*.py")):
        text = f.read_text()
        m = re.search(r'^OP = "(.+)"$', text, re.M)
        if not m:
            continue
        # Harnesses record `ns.name.overload`; the dispatcher wants `ns::name`.
        parts = m.group(1).split(".")
        op = f"{parts[0]}::{parts[1]}"
        c = re.search(r"^CALLS = (\d+)$", text, re.M)
        calls = int(c.group(1)) if c else 0
        if op not in best or calls > best[op][0]:
            best[op] = (calls, f)
    return {op: f for op, (_, f) in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", action="append", default=[], help="ns::name, repeatable")
    ap.add_argument("--from-harnesses", help="directory written by harness_from_model.py")
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--search", action="append", default=[], help="extra source root")
    ap.add_argument(
        "--bundle",
        help="write harness + kernel source + provenance per op into this directory",
    )
    args = ap.parse_args()

    harnesses: Dict[str, pathlib.Path] = {}
    if args.from_harnesses:
        harnesses = ops_from_harnesses(pathlib.Path(args.from_harnesses))
    ops = sorted(set(list(args.op) + list(harnesses)))
    if not ops:
        ap.error("give --op or --from-harnesses")

    import torch  # noqa: F401  (loads the dispatcher)

    out = pathlib.Path(args.bundle) if args.bundle else None
    for op in ops:
        report(op, args.device, args.search, out, harnesses.get(op))
    if out:
        print(f"\nBundles in {out}: each has the harness, the kernel's own source, and a")
        print("PROVENANCE.md saying what runs and where it came from.")
    print("\nNext: scripts/kernel_trials.py against the harness for the ops that have a")
    print("kernel here; the decompositions are a rewrite, not a kernel.")


if __name__ == "__main__":
    main()
