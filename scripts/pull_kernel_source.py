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
import os
import pathlib
import re
import json
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# A registration under one of these keys is not a kernel -- it is a rule for rewriting the
# op into other ops. There is nothing to pull; what runs is whatever the decomposition
# calls, so discovery has already recorded those separately.
_DECOMPOSITION_KEYS = ("CompositeImplicitAutograd", "CompositeExplicitAutograd", "Meta")

# A kernel registered inside PyTorch itself is not the end of the answer: for GEMM-shaped
# work ATen calls oneDNN, and the primitive oneDNN picks -- not the ATen wrapper -- is what
# runs and what has to be changed. oneDNN says which, if asked.
_PROBE = """
import json, sys, torch
spec = json.loads(sys.argv[1]); op = sys.argv[2]


def mk(a):
    if isinstance(a, list) and a and a[0] == "T":
        _, shape, dtype = a
        f = torch.randn if dtype.startswith(("float", "bfloat")) else torch.ones
        return f(shape, dtype=getattr(torch, dtype), device=sys.argv[3])
    if isinstance(a, list):
        return [mk(i) for i in a]
    return a


ns, name = op.split("::")


def _resolves():
    try:
        getattr(getattr(torch.ops, ns), name)
        return True
    except Exception:
        return False


# A custom op only exists in a process that imported the package registering it, and the
# parent hands over what it loaded rather than this script guessing a name. Import them
# one at a time and stop as soon as the op appears: importing a whole serving stack to
# reach an extension module costs more than the probe's own time budget, and a probe that
# times out reports no kernels, which reads as "this op has no share".
if not _resolves():
    for _m in sorted(json.loads(sys.argv[4]), key=lambda m: (m.count("."), len(m)), reverse=True):
        try:
            __import__(_m)
        except Exception:
            continue
        if _resolves():
            break
dev = sys.argv[3].split(":")[0]
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

acts = [ProfilerActivity.CPU, getattr(ProfilerActivity, dev.upper())]
try:
    args = [mk(a) for a in spec]
    fn = getattr(getattr(torch.ops, ns), name)
    fn(*args)
    torch.__dict__[dev].synchronize()
    with profile(activities=acts) as prof:
        fn(*args)
        torch.__dict__[dev].synchronize()
    for e in prof.key_averages():
        if e.device_type != DeviceType.CPU and (e.self_device_time_total or 0) > 0:
            print("PROBE_KERNEL", e.key)
except Exception as exc:
    print("PROBE_ERROR", type(exc).__name__, str(exc)[:120], file=sys.stderr)
"""


def _op_registering_modules() -> List[str]:
    """Top-level modules loaded here that a probe may need in order to see the op.

    Custom ops are registered as a side effect of import, so a fresh subprocess knows none
    of them. Rather than guess a package name from the op's namespace -- which is wrong as
    often as it is right, `_C` being nobody's package name -- hand over what this process
    actually loaded and let the probe import them.
    """
    import sys as _sys

    keep = []
    for name, mod in list(_sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if not ("site-packages" in f or "/Projects/" in f):
            continue
        if name.startswith("torch") or name.startswith("_"):
            continue
        # Ops are registered by the compiled extension, not by the package that wraps it:
        # importing `vllm_xpu_kernels` alone leaves `torch.ops._C` empty, and only
        # `vllm_xpu_kernels._C` fills it. Keep extension submodules for that reason.
        if "." in name and not f.endswith((".so", ".pyd")):
            continue
        keep.append(name)
    return sorted(set(keep))


def probe_library(op: str, argspec: List, device: str) -> Tuple[List[str], List[str]]:
    """Run the op, and report ``(oneDNN primitives, device kernels)`` it actually launched.

    Two things come out of running it that reading cannot give. oneDNN's verbose log names
    the primitive it selected -- the thing that runs when ATen "implements" a GEMM. And the
    profiler names the device kernels the op launched, which is what lets an op be charged
    its share of a run: a decomposition like ``aten::linear`` has no device time of its own,
    all of it belongs to the kernels it reaches, and without this link the busiest op in a
    model reports zero.

    A subprocess, because the verbose logging is enabled by an environment variable read at
    library load, and because an op that faults must not take the resolver down with it.
    """
    env = {**os.environ, "ONEDNN_VERBOSE": "1"}
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                _PROBE,
                json.dumps(argspec),
                op,
                device,
                json.dumps(_op_registering_modules()),
            ],
            capture_output=True,
            text=True,
            timeout=420,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return [], []
    out = r.stdout + r.stderr
    prims, kernels = [], []
    for line in out.splitlines():
        if ",primitive,exec," in line:
            f = line.split(",")
            # engine, primitive, implementation, ..., problem, time
            if len(f) > 8:
                prims.append(f"{f[5]} via {f[6]}  [{f[-2]}]")
        elif line.startswith("PROBE_KERNEL "):
            kernels.append(line[len("PROBE_KERNEL ") :].strip())
    return sorted(set(prims)), sorted(set(kernels))


def dispatch_report(op: str, overload: str = "") -> Tuple[List[str], List[str]]:
    """``(dispatch keys, registration source paths)`` for ``ns::name``, from the dispatcher.

    Some ops are only addressable with their overload -- ``aten::index`` dumps nothing,
    ``aten::index.Tensor`` dumps the truth -- and an empty dump is indistinguishable from an
    op the dispatcher has never heard of. Try the overload the model actually called before
    concluding anything from silence.
    """
    import torch

    dump = ""
    candidates = [op]
    if overload and overload != "default":
        candidates.insert(0, f"{op}.{overload}")
    for name in candidates:
        try:
            dump = torch._C._dispatch_dump(name)
        except RuntimeError:
            continue  # not a legal overload for this namespace
        if dump:
            break
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
        if "_meta_registrations" in rp or "RegisterMeta" in rp:
            continue  # a shape rule, not a kernel
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
                # Scope to the package, not the directory that holds every package: a
                # site-packages root turns the search below into a grep over numpy and
                # pandas for anything sharing the op's name.
                head = pathlib.PurePosixPath(tail).parts[0]
                scoped = root / head
                project_roots.append(scoped if scoped.is_dir() else root)
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


def python_op_source(op: str) -> List[str]:
    """Locate a custom op that was registered from Python rather than C++.

    ``torch.library`` registrations leave nothing in the dispatcher dump, so the namespace
    is the only handle -- and it is enough: a namespace belongs to a package, and the
    package's source declares the op by name.
    """
    import importlib

    ns, bare = op.split("::")
    try:
        mod = importlib.import_module(ns)
    except Exception:
        return []
    root = pathlib.Path(getattr(mod, "__file__", "") or "").parent
    if not root.is_dir():
        return []
    decl = re.compile(rf'(def |"|\'){re.escape(bare)}\b')
    hits = []
    for f in sorted(root.rglob("*.py")):
        try:
            if decl.search(f.read_text(errors="ignore")):
                hits.append(str(f))
        except OSError:
            continue
        if len(hits) >= 3:
            break
    return hits


def _in_torch(sources: List[str]) -> bool:
    return any("/pytorch/" in x or "/aten/" in x for x in sources)


def report(
    op: str,
    device: str,
    extra: Optional[List[str]],
    out: Optional[pathlib.Path] = None,
    harness: Optional[pathlib.Path] = None,
    argspec: Optional[List] = None,
    calls: Optional[int] = None,
    overload: str = "",
    share_of: Optional[object] = None,
) -> Dict[str, object]:
    """Say what implements ``op`` on ``device``, and where that source is."""
    keys, sources = dispatch_report(op, overload)
    dev = device_keys(keys, device)
    decomp = [k for k in keys if k.split("[")[0] in _DECOMPOSITION_KEYS]
    head = f"{op}" + (f"   ({calls} calls)" if calls else "")
    print(f"\n=== {head} ===")

    provider, where, route = "unresolved", [], ""
    share_pct: Optional[float] = None
    prims, launched = probe_library(op, argspec, device) if argspec is not None else ([], [])
    if share_of is not None and launched:
        pct = share_of(launched)
        if pct is not None:
            share_pct = pct
            print(
                f"  share         : {pct:.2f}% of device time (kernels: {', '.join(launched)[:60]})"
            )
    found = locate(op, sources, extra) if (dev or prims) else []
    in_torch = _in_torch(sources) or any("site-packages/torch/" in str(f) for f, _ in found)

    if prims:
        # oneDNN ran. That is true whether the op dispatched straight to it or a
        # decomposition reached it, and it names the primitive that has to change.
        provider = "oneDNN"
        where = [f"primitive: {x}" for x in prims]
        route = "/optimize-onednn -- the win is in the call, not a replacement kernel"
    elif not keys:
        # No dispatcher entry at all: a custom op registered from Python.
        provider = "Python-registered custom op"
        where = python_op_source(op) or ["source not located"]
        route = "/wrap-kernel-for-tuning -- it needs the stack's context to run"
    elif not dev and decomp:
        provider = "decomposition"
        where = [f"{decomp[0]} -- rewritten into ops discovery recorded separately"]
        route = "no kernel to pull; optimize what it decomposes to"
    elif not dev:
        provider = "not on this device"
        route = "not a target"
    elif in_torch and not [f for f, _ in found if "site-packages/torch/" not in str(f)]:
        provider = "ATen kernel inside PyTorch"
        where = [x for x in sources if "/aten/" in x][:2] or ["(built into torch)"]
        route = "no local source to edit; measure against the harness, or report upstream"
    elif found:
        provider = "provider kernel"
        where = [f"{f}  ({why})" for f, why in found]
        route = "/discover-model-kernels step 4 -- trial loop against harness.py"
    else:
        provider = "registered, source not on this box"
        where = sources[:2]
        route = "clone the providing project into tmp/ to read it"

    print(f"  implemented by: {provider}")
    for w in where:
        print(f"  where         : {w}")
    print(f"  route         : {route}")

    written = None
    if out and provider == "provider kernel":
        written = bundle(op, device, dev, sources, found, harness, out)
        if written:
            print(f"  bundled to    : {written}")
    return {
        "op": op,
        "provider": provider,
        "where": where,
        "route": route,
        "bundle": written,
        "share": share_pct,
    }


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


def share_lookup(path: pathlib.Path):
    """Charge an op the device time of the kernels it launches.

    Returns None when discovery recorded no timing, rather than inventing a share.
    """
    data = json.loads(path.read_text())
    by_kernel = data.get("device_time_by_kernel") or {}
    total = float(data.get("device_time_total_us") or 0.0)
    if not by_kernel or total <= 0:
        return None

    def share(kernels: List[str]) -> Optional[float]:
        us = 0.0
        for k in kernels:
            # Profiler keys are truncated in the stored table; match on the stored prefix.
            us += next((v for name, v in by_kernel.items() if name.startswith(k[:60])), 0.0)
        return 100.0 * us / total if us else None

    return share


def from_report(path: pathlib.Path) -> Tuple[Dict[str, Tuple[int, List, str]], List[Dict]]:
    """Every op discovery recorded, with the busiest shape's args, plus its Triton kernels."""
    data = json.loads(path.read_text())
    best: Dict[str, Tuple[int, List, str]] = {}
    for row in data.get("ops", []):
        parts = row["op"].split(".")
        op = f"{parts[0]}::{parts[1]}"
        overload = parts[2] if len(parts) > 2 else ""
        if op not in best or row["calls"] > best[op][0]:
            best[op] = (row["calls"], row["args"], overload)
    return best, data.get("triton", [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", action="append", default=[], help="ns::name, repeatable")
    ap.add_argument("--from-harnesses", help="directory written by harness_from_model.py")
    ap.add_argument("--from-report", help="discovered.json written by harness_from_model.py")
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
    recorded: Dict[str, Tuple[int, List, str]] = {}
    triton: List[Dict] = []
    share_of = None
    if args.from_report:
        recorded, triton = from_report(pathlib.Path(args.from_report))
        share_of = share_lookup(pathlib.Path(args.from_report))
    ops = sorted(set(list(args.op) + list(harnesses) + list(recorded)))
    if not ops and not triton:
        ap.error("give --op, --from-harnesses or --from-report")

    import torch  # noqa: F401  (loads the dispatcher)

    out = pathlib.Path(args.bundle) if args.bundle else None
    results = []
    for op in ops:
        calls, argspec, overload = recorded.get(op, (None, None, ""))
        results.append(
            report(
                op,
                args.device,
                args.search,
                out,
                harnesses.get(op),
                argspec,
                calls,
                overload,
                share_of,
            )
        )

    # Triton kernels are not dispatcher ops; the JIT knows where their source is.
    for t in triton:
        print(f"\n=== {t['kernel']}   ({t['calls']} calls) ===")
        print("  implemented by: Triton (JIT-compiled at run time)")
        print(f"  where         : {t['source']}")
        if t.get("constexprs"):
            print(f"  constexprs    : {t['constexprs']}")
        print("  route         : /wrap-kernel-for-tuning -- tune in place, no substitution")
        results.append({"op": t["kernel"], "provider": "Triton", "route": "wrap"})

    print("\n" + "=" * 86)
    print(f"{'share':>8}  {'op':42} {'implemented by':32}")
    print("-" * 86)
    # Ranked by measured share, because that is the only thing that says what an
    # optimization is worth. Ops with no share measured sort last, not first.
    for r in sorted(results, key=lambda r: (-(r.get("share") or -1), str(r["op"]))):
        sh = f"{r['share']:.2f}%" if r.get("share") is not None else "-"
        print(f"{sh:>8}  {str(r['op'])[:41]:42} {str(r['provider'])[:31]:32}")
    print("=" * 86)
    print("\nEach line is a kernel that ran. The route says which skill takes it from here;")
    print("bound it against flashinfer_bench.device.calibration before optimizing any of them.")


if __name__ == "__main__":
    main()
