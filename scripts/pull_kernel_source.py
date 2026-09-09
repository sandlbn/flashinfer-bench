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
a remembered ratio. An op that a library implements (oneDNN, for GEMM-shaped work) bundles
the same way, adapted to a library not being one kernel: the implementation it selected,
the candidates it rejected and the gate each failed at, the library version, and a script
that reproduces the selection -- see the oneDNN section below.

    python scripts/pull_kernel_source.py --op _C::fused_add_rms_norm
    python scripts/pull_kernel_source.py --from-harnesses tools/kernel-harness/auto \
        --bundle tools/kernel-harness/pulled
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple

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


def probe_library(op: str, argspec: List, device: str) -> Tuple[List[str], List[str], List[str]]:
    """Run the op, and report ``(oneDNN primitives, device kernels)`` it actually launched.

    Two things come out of running it that reading cannot give. oneDNN's verbose log names
    the primitive it selected -- the thing that runs when ATen "implements" a GEMM. And the
    profiler names the device kernels the op launched, which is what lets an op be charged
    its share of a run: a decomposition like ``aten::linear`` has no device time of its own,
    all of it belongs to the kernels it reaches, and without this link the busiest op in a
    model reports zero.

    A subprocess, because the verbose logging is enabled by an environment variable read at
    library load, and because an op that faults must not take the resolver down with it.

    ``ONEDNN_VERBOSE=all`` rather than ``1``: the plain mode names only the winner, while
    ``all`` also prints every candidate the library rejected and the gate that rejected it
    (``create:dispatch``), and the strategies its code generator scored. Those lines are the
    library's account of *why* this implementation ran, and they are returned whole as the
    third element so a bundle can keep them.
    """
    env = {**os.environ, "ONEDNN_VERBOSE": "all"}
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
        return [], [], []
    out = r.stdout + r.stderr
    prims, kernels, verbose = [], [], []
    for line in out.splitlines():
        if line.startswith("onednn_verbose,"):
            verbose.append(line.rstrip())
        if ",primitive,exec," in line:
            f = line.split(",")
            # engine, primitive, implementation, ..., problem, time
            if len(f) > 8:
                prims.append(f"{f[5]} via {f[6]}  [{f[-2]}]")
        elif line.startswith("PROBE_KERNEL "):
            kernels.append(line[len("PROBE_KERNEL ") :].strip())
    return sorted(set(prims)), sorted(set(kernels)), verbose


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

Then: `python scripts/kernel_trials.py init {series} <this-dir>/harness.py`
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

    schema = op_schema(op) or "(not resolvable)"
    files_md = (
        "\n".join(f"- `source/{rel}` ({n} lines) -- {why}" for rel, why, n in copied)
        or "- (none found on this box)"
    )
    (d / "PROVENANCE.md").write_text(
        _PROVENANCE.format(
            op=op,
            series=op.replace("::", "_"),
            device=device,
            keys=", ".join(keys) or "(none)",
            registered=sources[0] if sources else "(unknown)",
            project=project or "(unknown)",
            schema=schema,
            files=files_md,
        )
    )
    return d


# ----------------------------------------------------------------------------------------
# oneDNN: an op implemented by a library rather than by a kernel
#
# When ATen hands a GEMM to oneDNN, "the kernel" is not one file. The library walks an
# ordered list of candidate implementations, rejects each whose gates fail, takes the first
# that passes, and -- for its jit GEMM -- then scores strategies from a catalog and generates
# the kernel. Its verbose log reports every step of that, and each step names real source:
# the selected implementation is declared by name, the rejected ones cite the gate's
# ``file:line``, and the primitive kind names the list that was walked. That is what gets
# bundled: the identity of the code that ran, and the reasoning that chose it.
#
# Two grades of fact live in a bundle, and PROVENANCE.md tags every line with one:
#   [reported]        the library printed it at run time
#   [cited]           a file:line the library itself named
#   [rule]            inferred by a stated mapping rule from cited or reported facts
#   [interpretation]  a reading of a reported reason into what a caller could change
# A reader must be able to tell the first two from the last two without reading this code.
# ----------------------------------------------------------------------------------------

_ONEDNN_VERSION = re.compile(r"oneDNN v([0-9][0-9.]*)(?: \(commit ([0-9a-f]+)\))?")
_ONEDNN_CONSIDER = re.compile(r",consider:(.+),score:(\S+)\s*$")
# Column order of an exec line after ``operation``, as oneDNN v1 verbose documents it in
# its own ``template:`` header line. Used only when that header is missing from the log.
_ONEDNN_DEFAULT_COLUMNS = [
    "engine",
    "primitive",
    "implementation",
    "prop_kind",
    "memory_descriptors",
    "attributes",
    "auxiliary",
    "problem_desc",
    "exec_time",
]
# The macro whose first argument is the string the log prints as ``implementation``.
_ONEDNN_DECLARE = "DECLARE_COMMON_PD_T("
_ONEDNN_ANCHOR = pathlib.PurePosixPath("src/common/verbose.cpp")  # prints what is parsed here
# An implementation list is a sequence of ``<ENGINE>_INSTANCE<_QUALIFIER>(<class>)`` entries.
_ONEDNN_INSTANCE = re.compile(r"^\s*([A-Z][A-Z0-9_]*INSTANCE[A-Z0-9_]*)\((.+)\)\s*,?\s*$")
# A gate is a VDISPATCH_*/VCHECK*/VCONDCHECK* macro; it ends at the first ``;`` after it.
_ONEDNN_GATE = re.compile(r"\b(VDISPATCH_\w+|VCONDCHECK\w*|VCHECK\w*)\s*\(")

# The library's own rejection vocabulary (``src/common/verbose_msg.hpp``), read into the
# lever a *caller* holds. Keyed on fragments of the printed message, which is stable across
# versions in a way source lines are not. This is interpretation, tagged as such in the
# bundle; a reason matching nothing here is left unclassified rather than guessed.
_ONEDNN_LEVERS: List[Tuple[Tuple[str, ...], str, str]] = [
    (
        (
            "format tag",
            "tensor layout",
            "format kind",
            "memory stride",
            "trivial strides",
            "gemm format",
            "md flags",
            "sparse md",
        ),
        "operand layout",
        "caller-controlled: operand memory format and strides -- pass `any` for a weight so "
        "the library may pick its packed layout, pre-pack it once, or make operands "
        "contiguous",
    ),
    (
        ("datatype", "fpmath mode"),
        "data type",
        "caller-controlled where the model allows: operand and accumulation dtypes, or the "
        "fpmath mode attribute",
    ),
    (
        (
            "attribute",
            "post-ops",
            "scales",
            "zero-point",
            "bias configuration",
            "dropout",
            "accumulation mode",
        ),
        "attributes / post-ops",
        "caller-controlled: what is fused into the call (post-ops, scales, zero points, "
        "bias) and how it is described",
    ),
    (
        (
            "runtime dimension",
            "shape",
            "dimension",
            "broadcast",
            "scratchpad memory limit",
            "no elements",
            "ndims",
        ),
        "problem shape / decomposition",
        "caller-controlled through batching, splitting or padding; otherwise fixed by the "
        "model's shapes",
    ),
    (
        (
            "isa",
            "architecture",
            "device",
            "feature unavailable",
            "backend",
            "platform",
            "engine kind",
            "threadpool",
        ),
        "hardware / build",
        "fixed: not a caller lever on this part",
    ),
    (
        ("heuristic", "skipping", "fall back", "blocking", "dispatching to another"),
        "library heuristic",
        "fixed by the library's own policy for this problem; the gate says which quantity it "
        "weighed, and only a different problem (shape, layout) moves it",
    ),
]


def onednn_lever(reason: str) -> Tuple[str, str]:
    """``(lever, what a caller can do about it)`` for a rejection reason, by vocabulary."""
    low = reason.lower()
    for fragments, lever, action in _ONEDNN_LEVERS:
        if any(fr in low for fr in fragments):
            return lever, action
    return "unclassified", "read the gate: this reason is outside the vocabulary this tool knows"


@dataclasses.dataclass
class OneDNNRun:
    """What one oneDNN verbose log says about a single op, parsed and nothing more."""

    version: Optional[str] = None
    commit: Optional[str] = None
    environment: List[str] = dataclasses.field(default_factory=list)
    executed: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    created: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    rejected: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    considered: List[Tuple[str, float]] = dataclasses.field(default_factory=list)
    lines: List[str] = dataclasses.field(default_factory=list)

    def kinds(self) -> List[str]:
        """Primitive kinds that took part, in the order they appeared."""
        seen: List[str] = []
        for row in self.rejected + self.created + self.executed:
            if row["kind"] and row["kind"] not in seen:
                seen.append(row["kind"])
        return seen

    def engines(self) -> List[str]:
        """Engine kinds (``gpu``, ``cpu``) that took part."""
        seen: List[str] = []
        for row in self.rejected + self.created + self.executed:
            eng = row.get("engine", "").split(":")[0]
            if eng and eng not in seen:
                seen.append(eng)
        return seen

    def selected(self, kind: str) -> set:
        """Implementation names the log shows chosen for ``kind`` -- executed, or created as
        a nested primitive of something that executed. Both are the library's own report."""
        return {r["implementation"] for r in self.executed + self.created if r["kind"] == kind}

    def summary(self) -> Dict[str, object]:
        executed = {(e["kind"], e["implementation"]) for e in self.executed}
        return {
            "version": self.version,
            "commit": self.commit,
            "selected": [
                {k: r[k] for k in ("kind", "implementation", "problem_desc")} for r in self.executed
            ],
            "nested": [
                {k: r[k] for k in ("kind", "implementation", "problem_desc")}
                for r in self.created
                if (r["kind"], r["implementation"]) not in executed
            ],
            "rejected": [
                {k: r[k] for k in ("kind", "implementation", "reason", "path", "line")}
                for r in self.rejected
            ],
            "strategies_scored": len(self.considered),
        }


def parse_onednn_verbose(lines: Iterable[str]) -> OneDNNRun:
    """Parse the lines of an ``ONEDNN_VERBOSE=all`` log into what ran and why.

    Three kinds of line carry the answer. ``primitive,exec`` names the implementation that
    ran and the problem it ran on; ``primitive,create*`` lines name every primitive that was
    built to get there, which is how a nested primitive (a ``matmul`` that delegates to a
    ``gemm``) reports its own selection. ``primitive,create:dispatch`` is one rejected candidate:
    its name, the descriptors it was offered, the reason, and ``file:line`` of the gate --
    parsed from the end, because the reason is the only free-text field and the location
    always closes the line. ``info,...,consider:<entry>,score:<n>`` is one strategy the jit
    GEMM's selector scored; the log prints no "chosen" line, so the winner is inferred from
    the selector's own ordering (lowest score, after its alignment-fallback preference).

    Column positions come from the log's own ``template:`` header when present, so a
    version that reorders or adds a column still parses. The banner lines naming the
    library version, runtime and engines are kept: selection depends on all three.
    """
    run = OneDNNRun()
    columns = list(_ONEDNN_DEFAULT_COLUMNS)
    for raw in lines:
        line = raw.rstrip()
        if not line.startswith("onednn_verbose,"):
            continue
        run.lines.append(line)
        f = line.split(",")
        if len(f) < 4:
            continue
        if f[2] == "info":
            m = _ONEDNN_VERSION.search(line)
            if m and run.version is None:
                run.version, run.commit = m.group(1), m.group(2)
            elif ",engine," in line or "runtime:" in line:
                run.environment.append(",".join(f[3:]))
            elif (m := _ONEDNN_CONSIDER.search(line)) is not None:
                try:
                    score = float(m.group(2))
                except ValueError:
                    continue
                if (m.group(1), score) not in run.considered:
                    run.considered.append((m.group(1), score))
            continue
        if f[2] == "primitive" and f[3] == "info" and f[4].startswith("template:"):
            columns = [f[4].split(":", 1)[1]] + f[5:]
            columns = columns[1:] if columns and columns[0] == "operation" else columns
            continue
        is_create = f[3].startswith(("create:", "create_nested:")) and f[3] != "create:dispatch"
        if f[2] == "primitive" and (f[3] == "exec" or is_create):
            values = f[4:]
            row = {c: (values[i] if i < len(values) else "") for i, c in enumerate(columns)}
            executed = {
                "engine": row.get("engine", ""),
                "kind": row.get("primitive", ""),
                "implementation": row.get("implementation", ""),
                "prop_kind": row.get("prop_kind", ""),
                "memory_descriptors": row.get("memory_descriptors", ""),
                "attributes": row.get("attributes", ""),
                "problem_desc": row.get("problem_desc", ""),
            }
            # The probe runs the op twice (warm-up, then profiled); a repeated identical
            # execution is the same fact, not a second one. The log itself is kept whole.
            target = run.created if is_create else run.executed
            if executed not in target:
                target.append(executed)
            continue
        if f[2] == "primitive" and f[3] == "create:dispatch" and len(f) >= 11:
            # create:dispatch,<component>,<engine>,<kind>,<impl>,<prop>,<mds>,<attrs>,<aux>,
            #                 <problem>,<reason>,<file>:<line>
            path, _, lineno = f[-1].rpartition(":")
            if not path:
                continue
            run.rejected.append(
                {
                    "component": f[4],
                    "engine": f[5],
                    "kind": f[6],
                    "implementation": f[7],
                    "prop_kind": f[8],
                    "memory_descriptors": f[-6],
                    "attributes": f[-5],
                    "problem_desc": f[-3],
                    "reason": f[-2],
                    "path": path.strip(),
                    "line": lineno.strip(),
                }
            )
    return run


def onednn_checkout(roots: Iterable[pathlib.Path], hints: Iterable[str]) -> Optional[pathlib.Path]:
    """The oneDNN source tree among ``roots``, identified by the library's own paths.

    A dispatch line cites a path relative to the library's source root, so a root under
    which that path exists is the checkout -- the same rule ``locate`` uses for a provider's
    registration path. With no rejection to cite, the file that printed the log is the
    anchor: every oneDNN tree has it, and nothing else does.
    """
    hint_paths = [pathlib.PurePosixPath(h) for h in hints if h]
    for root in roots:
        if not root.is_dir():
            continue
        if any((root / h).is_file() for h in hint_paths) or (root / _ONEDNN_ANCHOR).is_file():
            return root
    return None


class _Tree:
    """Read-only view of a checkout, at a git revision when one is given.

    Reading from the git object at the runtime's exact version, rather than from whatever
    the working tree happens to be checked out at, is what makes a cited ``file:line`` land
    on the gate it names. Without a revision -- no ``.git``, or the runtime's commit was
    never fetched -- the working tree is used and the bundle says so.
    """

    def __init__(self, root: pathlib.Path, rev: Optional[str] = None):
        self.root = root
        self.rev = rev
        self._cache: Dict[str, Optional[bytes]] = {}

    def _git(self, *args: str, binary: bool = False):
        try:
            r = subprocess.run(
                ["git", "-C", str(self.root), *args],
                capture_output=True,
                text=not binary,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout if r.returncode == 0 else None

    def describe(self) -> str:
        if self.rev:
            return self.rev
        out = self._git("describe", "--tags", "--always", "--dirty")
        return (out or "").strip() or "(working tree; not a git checkout)"

    def exists(self, rel: str) -> bool:
        return self.read_bytes(rel) is not None

    def read_bytes(self, rel: str) -> Optional[bytes]:
        if rel not in self._cache:
            if self.rev:
                self._cache[rel] = self._git("show", f"{self.rev}:{rel}", binary=True)
            else:
                try:
                    self._cache[rel] = (self.root / rel).read_bytes()
                except OSError:
                    self._cache[rel] = None
        return self._cache[rel]

    def read_text(self, rel: str) -> Optional[str]:
        data = self.read_bytes(rel)
        return data.decode(errors="ignore") if data is not None else None

    def grep(self, literal: str, subdir: str) -> List[Tuple[str, int, str]]:
        """``(path, line, text)`` for every line under ``subdir`` containing ``literal``."""
        if self.rev:
            out = self._git("grep", "-n", "-F", "-e", literal, self.rev, "--", subdir)
            hits = []
            for row in (out or "").splitlines():
                # <rev>:<path>:<line>:<text>
                rest = row.split(":", 1)[1] if ":" in row else ""
                path, _, tail = rest.partition(":")
                lineno, _, text = tail.partition(":")
                if path and lineno.isdigit():
                    hits.append((path, int(lineno), text))
            return hits
        hits = []
        base = self.root / subdir
        if not base.is_dir():
            return hits
        for f in sorted(base.rglob("*")):
            if f.suffix not in (".hpp", ".cpp", ".h", ".in", ".cl"):
                continue
            try:
                for i, text in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if literal in text:
                        hits.append((str(f.relative_to(self.root)), i, text))
            except OSError:
                continue
        return hits

    def find(self, pattern: str, subdir: str) -> List[str]:
        """Paths under ``subdir`` whose basename matches the glob ``pattern``."""
        if self.rev:
            out = self._git("ls-tree", "-r", "--name-only", self.rev, "--", subdir)
            names = (out or "").splitlines()
        else:
            base = self.root / subdir
            names = (
                [str(f.relative_to(self.root)) for f in sorted(base.rglob("*")) if f.is_file()]
                if base.is_dir()
                else []
            )
        return [n for n in names if fnmatch.fnmatch(pathlib.PurePosixPath(n).name, pattern)]


def onednn_revision(root: pathlib.Path, run: OneDNNRun) -> Optional[str]:
    """The git revision in ``root`` matching the library that ran, if it is present.

    The commit from the verbose banner first -- exact by construction -- then the release
    tag named by the version. None means the working tree is the best available, which is
    reported rather than silently accepted.
    """
    tree = _Tree(root)
    for rev in [run.commit, f"v{run.version}" if run.version else None]:
        if rev and tree._git("cat-file", "-e", f"{rev}^{{commit}}") is not None:
            return rev
    return None


def onednn_declarations(tree: _Tree, engine: str) -> List[Dict[str, str]]:
    """Every ``DECLARE_COMMON_PD_T`` under ``src/<engine>``: path, class, and its name.

    The name is the macro's first argument. When that is a string literal it is exactly
    what the log prints; when it is an expression (``impl_name()``, ``name_.c_str()``, a
    nested primitive's ``name()``) the name is only known at run time, and the record says
    so with ``literal`` empty and the expression kept in ``expr``.
    """
    out = []
    for path, lineno, text in tree.grep(_ONEDNN_DECLARE, f"src/{engine}"):
        if "#define" in text:
            continue
        m = re.search(r"DECLARE_COMMON_PD_T\((.*)\)\s*;", text) or re.search(
            r"DECLARE_COMMON_PD_T\((.*)$", text
        )
        args = (m.group(1) if m else "").strip()
        lit = re.match(r'"([^"]*)"\s*,\s*([\w:<>]+)', args)
        cls = lit.group(2) if lit else (args.rsplit(",", 1)[-1].strip() if "," in args else "")
        out.append(
            {
                "path": path,
                "line": str(lineno),
                "class": cls,
                "literal": lit.group(1) if lit else "",
                "expr": "" if lit else args.split(",")[0].strip(),
            }
        )
    return out


def onednn_impl_sites(
    tree: _Tree, impl: str, engines: List[str], decls: Optional[Dict[str, List[Dict]]] = None
) -> List[Tuple[str, str]]:
    """Files that declare the implementation the log calls ``impl``.

    The rule: oneDNN names an implementation with the string literal in
    ``DECLARE_COMMON_PD_T("<name>", <class>)``, so a literal match on that declaration finds
    the header that declares it, and its sibling ``.cpp``/``.cl`` is the body. Scoped to
    ``src/<engine>`` because CPU and GPU reuse names such as ``ref:any``.

    The rule has a known hole, and the caller is told rather than guessed around: an
    implementation whose name is computed at run time cannot be reached by any literal
    search, and no file is attributed to it.
    """
    found: List[Tuple[str, str]] = []
    for eng in engines or ["gpu", "cpu"]:
        rows = (decls or {}).get(eng)
        if rows is None:
            rows = onednn_declarations(tree, eng)
        for d in rows:
            if d["literal"] != impl:
                continue
            found.append((d["path"], f"declares `{impl}` as `{d['class']}` (line {d['line']})"))
            stem = str(pathlib.PurePosixPath(d["path"]).with_suffix(""))
            for ext in (".cpp", ".cl"):
                if tree.exists(stem + ext):
                    found.append((stem + ext, f"implements `{d['class']}`"))
    return found


def onednn_build_conditions(tree: _Tree, engine: str) -> Dict[str, str]:
    """Which ``*_INSTANCE*`` macros expand to nothing under some build flag, and which flag.

    Read from the header that defines them: a macro defined empty in one ``#if`` branch is
    conditional on that branch's flag, and one that routes through ``DNNL_<ENGINE>_<X>_ONLY``
    is built only for vendor ``X``. Derived, so a new qualifier in a later version is picked
    up without a table here.
    """
    conds: Dict[str, str] = {}
    for hdr in tree.find(f"{engine}_impl_list.hpp", f"src/{engine}"):
        text = (tree.read_text(hdr) or "").replace("\\\n", " ")
        stack: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith(("#ifdef", "#if ")):
                stack.append(line.split(None, 1)[1] if " " in line else line)
            elif line.startswith("#ifndef"):
                stack.append("!(" + (line.split(None, 1)[1] if " " in line else line) + ")")
            elif line.startswith("#else") and stack:
                stack[-1] = "!(" + stack[-1] + ")"
            elif line.startswith("#endif") and stack:
                stack.pop()
            m = re.match(r"#define\s+([A-Z0-9_]*INSTANCE[A-Z0-9_]*)\(\.\.\.\)\s*(.*)$", line)
            if not m:
                continue
            name, body = m.group(1), m.group(2).strip()
            if body == "" and stack:
                conds[name] = f"compiled out when {stack[-1]}"
            vendor = re.search(r"DNNL_[A-Z]+_([A-Z_]+?)_ONLY", body)
            if vendor and name not in conds:
                conds[name] = f"built only for {vendor.group(1).lower().replace('_', ' ')}"
    return conds


def onednn_candidates(
    tree: _Tree, kind: str, engine: str, decls: List[Dict[str, str]], run: OneDNNRun
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """The ordered candidate list for ``kind`` with each entry's name and outcome.

    The list file is ``<engine>_<kind>_list.cpp``; each ``*_INSTANCE*(ns::cls)`` line is one
    candidate, tried in order. Its name comes from the declaration of ``cls`` under the
    directory ``ns`` maps to (``intel::gemm::gen_t`` -> ``src/gpu/intel/gemm/``), which is
    a rule and tagged as one. Outcomes: ``selected`` and ``rejected`` come from the log;
    ``not tried`` is every entry after the winner; an entry before the winner with no
    dispatch line either is compiled out (its macro says so) or was passed over without
    printing -- both are reported as what they are.
    """
    lists = tree.find(f"{engine}_{kind}_list.cpp", f"src/{engine}")
    if not lists:
        return None, []
    path = lists[0]
    text = tree.read_text(path) or ""
    conds = onednn_build_conditions(tree, engine)
    vendor_names = " ".join(run.environment).lower()
    selected = run.selected(kind)
    rejected = {r["implementation"]: r for r in run.rejected if r["kind"] == kind}
    nested = {
        r["kind"] for r in run.created if r["implementation"] in selected and r["kind"] != kind
    }

    rows: List[Dict[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        m = _ONEDNN_INSTANCE.match(raw)
        if not m:
            continue
        macro, cls_full = m.group(1), m.group(2).strip()
        cls = re.sub(r"<.*>$", "", cls_full.split("::")[-1]).strip()
        ns_dir = "/".join(cls_full.split("::")[:-1])
        decl = next(
            (
                d
                for d in decls
                if re.sub(r"<.*>$", "", d["class"].split("::")[-1]) == cls
                and (not ns_dir or f"/{ns_dir}/" in f"/{d['path']}")
            ),
            None,
        )
        name = decl["literal"] if decl and decl["literal"] else ""
        computed = decl["expr"] if decl and not decl["literal"] else ""
        rows.append(
            {
                "position": str(len(rows) + 1),
                "macro": macro,
                "class": cls_full,
                "name": name,
                "computed": computed,
                "declared_at": f"{decl['path']}:{decl['line']}" if decl else "",
                "list_line": str(lineno),
                "outcome": "",
                "detail": "",
            }
        )

    # Outcomes. The winner's position splits the list into tried and not tried.
    win = next((i for i, r in enumerate(rows) if r["name"] and r["name"] in selected), None)
    if win is None and selected:
        # The selected name matched no literal: it is carried by a computed-name entry
        # (a wrapper taking a nested primitive's name). Attribute it to the first such entry
        # that was not rejected, and say that this is inference.
        for i, r in enumerate(rows):
            if r["computed"] and r["name"] not in rejected:
                win = i
                via = (
                    f" -- the log shows a nested `{sorted(nested)[0]}` primitive of that name "
                    f"being created, so this entry is the wrapper that carries it"
                    if nested
                    else ""
                )
                r["detail"] = (
                    f"name computed at run time (`{r['computed']}`); the log's "
                    f"`{sorted(selected)[0]}` is attributed here by elimination{via}"
                )
                break

    # The vendors this build includes: proven by the macro of every entry the log shows was
    # tried, and supplemented by the engine banner. An entry qualified for another vendor
    # was never in the list the dispatcher walked.
    def vendor_of(macro: str) -> str:
        cond = conds.get(macro, "")
        return cond.split("for", 1)[1].strip() if cond.startswith("built only for") else ""

    built = {
        vendor_of(r["macro"])
        for i, r in enumerate(rows)
        if r["name"] in rejected or (win is not None and i == win)
    } - {""}
    for i, r in enumerate(rows):
        cond = conds.get(r["macro"], "")
        vendor = vendor_of(r["macro"])
        if r["name"] in rejected:
            rj = rejected[r["name"]]
            r["outcome"] = "rejected"
            r["detail"] = f"{rj['reason']} at `{rj['path']}:{rj['line']}`"
        elif win is not None and i == win:
            r["outcome"] = "selected"
        elif vendor and vendor not in built and vendor not in vendor_names:
            r["outcome"] = "not built"
            r["detail"] = cond
        elif cond.startswith("compiled out"):
            r["outcome"] = "build-conditional"
            r["detail"] = cond
        elif win is not None and i > win:
            r["outcome"] = "not tried"
            r["detail"] = "after the winner in the list"
        elif win is not None:
            r["outcome"] = "passed over"
            r["detail"] = (
                "before the winner, no dispatch line: rejected without printing, or its `init` bailed before its first gate"
            )
        else:
            r["outcome"] = "unknown"
    return path, rows


def onednn_gate(tree: _Tree, path: str, line: str) -> Dict[str, object]:
    """The gate statement at a cited ``file:line``, its position among the gates before it,
    and the same-file helpers it calls.

    Statement: from the last gate macro at or before the cited line to the first ``;`` after
    it. Position: the count of gate macros from the enclosing function's start to that
    statement, against the count in the whole function -- how many checks the candidate
    passed before this one, which is the closest thing to "how close was it" the source
    can give. Helpers: functions called in the condition that are defined in the same file,
    with their line range, so the reader knows where the real test lives. Remaining: the
    message each later gate in the function would print, so "what would make it pass" is
    read as the whole set of conditions and not only the first one that failed.
    """
    text = tree.read_text(path)
    empty: Dict[str, object] = {
        "statement": "",
        "function": "",
        "index": 0,
        "total": 0,
        "helpers": [],
        "remaining": [],
    }
    if text is None or not line.isdigit():
        return empty
    lines = text.splitlines()
    n = int(line)
    if not 1 <= n <= len(lines):
        return empty

    # Start of the statement: the nearest line at or above n that opens a gate macro.
    start = next((i for i in range(n, 0, -1) if _ONEDNN_GATE.search(lines[i - 1])), n)
    end = next(
        (i for i in range(start, min(len(lines), start + 12) + 1) if ";" in lines[i - 1]), start
    )
    statement = "\n".join(lines[start - 1 : end])

    # Enclosing function: the nearest preceding line that looks like a definition header.
    fn_re = re.compile(r"^\w[\w:<>\s\*&,]*\s([\w:~]+)\s*\([^;]*\)?\s*(const)?\s*\{?\s*$")
    fn_start, fn_name = 1, ""
    for i in range(start, 0, -1):
        m = fn_re.match(lines[i - 1])
        if m and "::" in m.group(1) or (m and lines[i - 1].rstrip().endswith("{")):
            fn_start, fn_name = i, m.group(1)
            break
    fn_end = len(lines)
    for i in range(fn_start + 1, len(lines) + 1):
        if lines[i - 1].startswith("}"):
            fn_end = i
            break
    gates_before = sum(1 for i in range(fn_start, start + 1) if _ONEDNN_GATE.search(lines[i - 1]))
    gates_total = sum(1 for i in range(fn_start, fn_end + 1) if _ONEDNN_GATE.search(lines[i - 1]))
    # What it would still have had to pass: the message each later gate would print.
    remaining: List[str] = []
    for i in range(end + 1, fn_end + 1):
        if _ONEDNN_GATE.search(lines[i - 1]):
            stmt = " ".join(lines[i - 1 : min(i + 11, fn_end)]).split(";", 1)[0]
            msg = re.search(r"\bVERBOSE_\w+", stmt)
            remaining.append(msg.group(0) if msg else stmt.strip()[:60])

    # Helpers: identifiers called in the statement that are defined in this same file.
    helpers = []
    for ident in sorted(set(re.findall(r"\b([a-z_]\w*)\s*\(", statement))):
        if _ONEDNN_GATE.match(ident + "(") or ident in ("utils", "one_of"):
            continue
        for i, row in enumerate(lines, 1):
            if re.search(rf"::{re.escape(ident)}\s*\(", row) and not row.strip().endswith(";"):
                j = next((k for k in range(i, len(lines) + 1) if lines[k - 1].startswith("}")), i)
                helpers.append({"name": ident, "from": i, "to": j})
                break
    return {
        "statement": statement,
        "function": fn_name,
        "index": gates_before,
        "total": gates_total,
        "helpers": helpers,
        "remaining": remaining,
    }


def onednn_sources(
    tree: _Tree, run: OneDNNRun
) -> Tuple[List[Tuple[str, str, str]], List[str], Dict[str, object]]:
    """Every file the log points at -- each with why and a confidence tag -- the gaps the
    rules cannot close, and the analysis (candidate lists, gates) that PROVENANCE.md renders.
    """
    files: List[Tuple[str, str, str]] = []
    gaps: List[str] = []
    engines = run.engines()
    decls = {eng: onednn_declarations(tree, eng) for eng in engines}
    analysis: Dict[str, object] = {"candidates": {}, "gates": []}

    def add(path: str, why: str, tag: str) -> None:
        if path and tree.exists(path) and path not in {p for p, _, _ in files}:
            files.append((path, why, tag))

    # The winner: the implementation the exec line names.
    for row in run.executed:
        impl = row["implementation"]
        sites = onednn_impl_sites(tree, impl, engines, decls)
        if not sites:
            gaps.append(
                f"`{impl}` (selected for `{row['kind']}`) is not a string literal in any "
                f"`{_ONEDNN_DECLARE[:-1]}` under this tree: its name is computed at run time, "
                f"so no file can be attributed to it by rule. The candidate list for "
                f"`{row['kind']}` names the classes; start there."
            )
        for path, why in sites:
            add(path, f"selected implementation -- {why}", "rule")
        if sites and not any(row["kind"] in pathlib.PurePosixPath(p).parts for p, _ in sites):
            gaps.append(
                f"`{impl}` was selected for `{row['kind']}`, but the declaration carrying that "
                f"name lives outside any `{row['kind']}` directory: the `{row['kind']}` entry "
                f"is a wrapper that takes its name from the nested primitive it delegates to, "
                f"and the nested primitive is what is bundled. The wrapper is the computed-name "
                f"entry in the `{row['kind']}` candidate list."
            )

    # The chain: the ordered candidate list the dispatcher walked, one per primitive kind,
    # with every entry's outcome.
    for kind in run.kinds():
        for eng in engines:
            path, rows = onednn_candidates(tree, kind, eng, decls.get(eng, []), run)
            if path:
                add(path, f"candidate order for `{kind}` -- walked top to bottom", "rule")
                analysis["candidates"][kind] = {"path": path, "rows": rows}

    # The rejections: the gate, at the file the library cited, and the candidate it gates.
    for row in run.rejected:
        add(
            row["path"],
            f"rejected `{row['implementation']}`: {row['reason']} (line {row['line']})",
            "cited",
        )
        for path, why in onednn_impl_sites(tree, row["implementation"], engines, decls):
            add(path, f"rejected candidate -- {why}", "rule")
        gate = onednn_gate(tree, row["path"], row["line"])
        lever, action = onednn_lever(row["reason"])
        analysis["gates"].append({**row, **gate, "lever": lever, "action": action})

    # The strategies: the file that scored them, and the catalog(s) they came from.
    if run.considered:
        emitters = [p for eng in engines for p, _, _ in tree.grep("consider:%s", f"src/{eng}")]
        for path in sorted(set(emitters)):
            add(path, "scores the generator's strategies (prints `consider:`)", "rule")
            for db in tree.find("*.db", str(pathlib.PurePosixPath(path).parent)):
                add(db, "strategy catalog the scored entries are drawn from", "rule")
        if not emitters:
            gaps.append(
                "strategies were scored but no file under this tree prints `consider:`; "
                "the catalog cannot be located by rule at this revision."
            )
    return files, gaps, analysis


_ONEDNN_PROVENANCE = """# {op}

Pulled by `scripts/pull_kernel_source.py`. Everything here describes what **actually runs**
on `{device}` -- the dispatcher was asked, not a table, and then the library was asked which
of its implementations it selected and why.

Every line below carries one of four tags. Trust them differently.

| tag | meaning |
| --- | --- |
| `[reported]` | the library printed it at run time (`verbose.log`) |
| `[cited]` | a `file:line` the library itself named |
| `[rule]` | inferred by a stated mapping rule from reported or cited facts; can be wrong when the rule's stated assumption fails |
| `[interpretation]` | a reading of a reported reason into what a caller could change; check it against the gate |

| | |
| --- | --- |
| dispatch key | `{keys}` |
| registered at | `{registered}` |
| providing project | `oneDNN` -- a library, not one kernel: it selects an implementation per call |
| library version (ran here) `[reported]` | `{runtime_version}` |
| library version (SYCL solutions link) | `{link_version}` |
| runtime and engines `[reported]` | {environment} |
| source checkout | `{checkout}` |
| source revision bundled | `{revision}` |
| harness | `harness.py` -- calls this op at a shape the model called it at |
| record of this pull | `verbose.log` (unedited), `selection.json` (machine-readable, diff two bundles with it) |

## Schema

```
{schema}
```

## 1. What ran `[reported]`

| primitive | implementation | problem | memory descriptors | attributes |
| --- | --- | --- | --- | --- |
{executed}

{selected_source}

## 2. Why that one: the dispatch chain

oneDNN walks the candidate list for the primitive kind top to bottom and takes the first
implementation whose gates all pass. `selected`/`rejected` outcomes are `[reported]`; the
list order, the names of entries that did not print, and the build conditions are `[rule]`
(entry `ns::cls` is matched to the `DECLARE_COMMON_PD_T` of `cls` under `src/<engine>/ns/`;
a computed name cannot be matched and is shown as the expression that computes it).

{candidates}

## 3. What would change the choice

{gates}

## 4. Strategy selection inside the implementation `[reported]`

{strategies}

## Source

{files}

{gaps}Files are copied whole from the checkout, at the revision above, so a `file:line` cited by
the library lands on the line it names. Read the candidate list first; the gates are in the
order they are tried.

## Reproduce

```
ONEDNN_VERBOSE=all python repro.py 2>&1 | grep '^onednn_verbose'
```

`repro.py` calls the op at the same shape and dtype the harness does; the output should match
`verbose.log` line for line, apart from timings and cache hits. `selection.json` holds the
same facts as this file in a fixed shape: regenerate the bundle after a library upgrade and
diff the two. Selection changes between versions, so a different `oneDNN v...` banner means a
different chain, not a bug in this bundle.

## Before optimizing

Bound it. `flashinfer_bench.device.calibration.get()` gives this part's substitution cost,
timing floor and achievable bandwidth; a win smaller than what taking it costs is a loss.
See `.claude/skills/route-kernel-work/PLAN.md`.

The win is usually in the call, not in a replacement kernel: `/optimize-onednn` walks the
fixes (weight layout, primitive caching, post-op fusion) against section 3 above.

Then: `python scripts/kernel_trials.py init {series} <this-dir>/harness.py`
"""

_ONEDNN_ABSENT = """No oneDNN source tree was found under any search root, so nothing is copied and sections
2 and 3 hold only what the library reported. This bundle would have held:

{would}

To obtain it, clone oneDNN **at the version that ran** into `tmp/` -- see `/clone-repos`,
"oneDNN source" -- or pass an existing checkout with `--search <dir>`, then run this pull
again. The revision to check out is `{revision}`.

"""

_REPRO = '''"""Reproduce the oneDNN selection for {op} at the shape the model called it at.

    ONEDNN_VERBOSE=all python repro.py 2>&1 | grep '^onednn_verbose'
"""

import torch

DEVICE = "{device}"
SPEC = {spec}


def mk(a):
    if isinstance(a, list) and a and a[0] == "T":
        _, shape, dtype = a
        f = torch.randn if dtype.startswith(("float", "bfloat")) else torch.ones
        return f(shape, dtype=getattr(torch, dtype), device=DEVICE)
    if isinstance(a, list):
        return [mk(i) for i in a]
    return a


# A custom op only exists once the package registering it is imported; ATen ops need nothing.
fn = getattr(getattr(torch.ops, "{ns}"), "{name}")
fn(*[mk(a) for a in SPEC])
getattr(torch, DEVICE.split(":")[0]).synchronize()
'''


def _onednn_link_version() -> Optional[str]:
    """The oneDNN a SYCL solution would link, from the existing helper; None when unknown."""
    try:
        from flashinfer_bench.integration.providers import onednn_link_version

        return onednn_link_version()
    except Exception:
        return None


def _md_candidates(analysis: Dict[str, object]) -> str:
    out = []
    for kind, info in (analysis.get("candidates") or {}).items():
        out.append(f"**`{kind}`** -- `source/{info['path']}` `[rule]`\n")
        out.append("| # | candidate | name | outcome | detail |\n| --- | --- | --- | --- | --- |")
        for r in info["rows"]:
            name = (
                f"`{r['name']}`"
                if r["name"]
                else (f"computed: `{r['computed']}`" if r["computed"] else "(no declaration found)")
            )
            tag = "`[reported]`" if r["outcome"] in ("selected", "rejected") else "`[rule]`"
            out.append(
                f"| {r['position']} | `{r['class']}` | {name} | **{r['outcome']}** {tag} | {r['detail']} |"
            )
        out.append("")
    return "\n".join(out) or "(no candidate list could be read: no source tree)"


def _md_gates(analysis: Dict[str, object], run: OneDNNRun) -> str:
    gates = analysis.get("gates") or []
    if not run.rejected:
        return (
            "No candidate was rejected: the first implementation tried was selected. There is "
            "no gate to move; only the strategy selection below and the call itself remain."
        )
    if not gates:
        out = ["No source tree, so the gates cannot be read. What the library reported:\n"]
        for r in run.rejected:
            lever, action = onednn_lever(r["reason"])
            out.append(
                f"- `{r['implementation']}` rejected: **{r['reason']}** at "
                f"`{r['path']}:{r['line']}` `[reported]` -- lever: {lever} "
                f"`[interpretation]`: {action}"
            )
        return "\n".join(out)
    out = []
    for g in gates:
        out.append(f"### `{g['implementation']}` -- {g['reason']} `[reported]`\n")
        out.append("| | |\n| --- | --- |")
        out.append(f"| gate `[cited]` | `{g['path']}:{g['line']}` |")
        if g["statement"]:
            fn = f" in `{g['function']}`" if g["function"] else ""
            out.append(
                f"| position `[rule]` | gate {g['index']} of {g['total']}{fn}: it passed "
                f"{max(g['index'] - 1, 0)} checks before this one |"
            )
        out.append(
            f"| descriptor offered `[reported]` | mds: `{g['memory_descriptors'] or '-'}`; "
            f"attrs: `{g['attributes'] or '-'}`; problem: `{g['problem_desc']}` |"
        )
        out.append(f"| lever `[interpretation]` | **{g['lever']}** -- {g['action']} |")
        if g["helpers"]:
            hs = ", ".join(
                f"`{h['name']}` (lines {h['from']}-{h['to']} of the same file)"
                for h in g["helpers"]
            )
            out.append(
                f"| the real test `[rule]` | the condition calls {hs}; read that, not the macro |"
            )
        if g.get("remaining"):
            ahead = ", ".join(f"`{m}`" for m in g["remaining"])
            out.append(
                f"| still ahead of it `[rule]` | passing this gate is necessary, not sufficient: "
                f"{len(g['remaining'])} more follow in the same function -- {ahead} |"
            )
        out.append("")
        if g["statement"]:
            out.append("```cpp\n" + g["statement"] + "\n```\n")
    return "\n".join(out)


def _md_strategies(run: OneDNNRun) -> str:
    if not run.considered:
        return "The selected implementation printed no strategy scores; it has no catalog step."
    ranked = sorted(run.considered, key=lambda t: t[1])
    rows = "\n".join(f"| {s:.6g} | `{e}` |" for e, s in ranked[:12])
    return (
        f"The selected implementation scored {len(run.considered)} distinct catalog strategies "
        "for this problem. The log prints no winner; the selector orders by score ascending "
        "after preferring non-fallback alignment (`[rule]`: read from the selector source in "
        "`source/`), so the lowest score is the likely choice.\n\n"
        "| score | strategy |\n| --- | --- |\n" + rows
    )


def bundle_onednn(
    op: str,
    device: str,
    keys: List[str],
    sources: List[str],
    run: OneDNNRun,
    harness: Optional[pathlib.Path],
    argspec: Optional[List],
    out: pathlib.Path,
    extra: Optional[List[str]] = None,
    roots: Optional[List[pathlib.Path]] = None,
) -> pathlib.Path:
    """Write a bundle for an op that oneDNN implements, shaped like a provider kernel's.

    Same layout -- ``harness.py``, ``source/``, ``PROVENANCE.md`` -- extended with what a
    library needs and a kernel does not: the selected implementation, the full candidate
    list with outcomes, each rejection's gate and the lever behind it, the strategies scored,
    the library version, and ``repro.py`` plus ``verbose.log`` and ``selection.json`` so the
    selection can be reproduced and diffed. Written even when no checkout is present,
    naming what it would have held.
    """
    import shutil

    d = out / op.replace("::", "_")
    (d / "source").mkdir(parents=True, exist_ok=True)
    # A re-pull into the same directory may be handed the bundle's own harness.
    if harness and harness.is_file() and harness.resolve() != (d / "harness.py").resolve():
        shutil.copy2(harness, d / "harness.py")
    (d / "verbose.log").write_text("\n".join(run.lines) + ("\n" if run.lines else ""))
    ns, name = op.split("::")
    (d / "repro.py").write_text(
        _REPRO.format(op=op, device=device, spec=json.dumps(argspec or []), ns=ns, name=name)
    )

    runtime_version = (
        f"{run.version}+{run.commit[:12]}"
        if run.version and run.commit
        else run.version or "(no version banner in the log)"
    )
    hints = [r["path"] for r in run.rejected]
    checkout = onednn_checkout(roots if roots is not None else search_roots(extra), hints)

    copied: List[Tuple[str, str, str, int]] = []
    gaps: List[str] = []
    analysis: Dict[str, object] = {"candidates": {}, "gates": []}
    revision = run.commit or (f"v{run.version}" if run.version else "(unknown)")
    checkout_md = "(none found)"
    if checkout is not None:
        rev = onednn_revision(checkout, run)
        tree = _Tree(checkout, rev)
        checkout_md = str(checkout)
        revision = tree.describe()
        if rev is None and run.version:
            gaps.append(
                f"the checkout does not contain the revision that ran (`{run.commit or ''}`"
                f" / `v{run.version}`), so files come from its working tree at `{revision}`;"
                f" a `file:line` cited above may be off by the drift between the two."
                f" `git -C {checkout} fetch --tags && git checkout v{run.version}` aligns them."
            )
        files, rule_gaps, analysis = onednn_sources(tree, run)
        gaps += rule_gaps
        for rel, why, tag in files:
            data = tree.read_bytes(rel)
            if data is None:
                continue
            dest = d / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            copied.append((rel, why, tag, data.count(b"\n")))

    executed_md = (
        "\n".join(
            f"| `{r['kind']}` | `{r['implementation']}` | `{r['problem_desc']}` | "
            f"`{r['memory_descriptors'] or '-'}` | `{r['attributes'] or '-'}` |"
            for r in run.executed
        )
        or "| (no exec line) | | | | |"
    )
    sel_files = [c for c in copied if c[1].startswith("selected implementation")]
    selected_source_md = (
        'Resolved to source `[rule]` -- the string literal in `DECLARE_COMMON_PD_T("<name>", '
        "<class>)` is what the log prints as the implementation, so the file holding that "
        "literal declares it and its sibling body implements it:\n\n"
        + "\n".join(f"- `source/{rel}` -- {why}" for rel, why, _, _ in sel_files)
        if sel_files
        else "Not resolved to source: see the gaps under **Source**."
    )

    files_md = (
        "\n".join(f"- `source/{rel}` ({n} lines) `[{tag}]` -- {why}" for rel, why, tag, n in copied)
        or "- (nothing copied)"
    )
    if checkout is None:
        would = "\n".join(
            [f"- the implementation declared as `{r['implementation']}`" for r in run.executed]
            + [f"- the candidate list for `{k}`" for k in run.kinds()]
            + [f"- `{r['path']}` (rejected `{r['implementation']}`)" for r in run.rejected]
        )
        files_md = _ONEDNN_ABSENT.format(would=would or "- (nothing to name)", revision=revision)
    gaps_md = ("\n".join(f"- gap: {g}" for g in gaps) + "\n\n") if gaps else ""

    (d / "selection.json").write_text(
        json.dumps(
            {
                "op": op,
                "device": device,
                "library": {"name": "oneDNN", "version": run.version, "commit": run.commit},
                "environment": run.environment,
                "source_revision": revision,
                "selected": run.executed,
                "nested": run.created,
                "rejected": run.rejected,
                "candidates": analysis.get("candidates"),
                "gates": [
                    {k: v for k, v in g.items() if k != "statement"}
                    for g in (analysis.get("gates") or [])
                ],
                "strategies": [
                    {"strategy": e, "score": s}
                    for e, s in sorted(run.considered, key=lambda t: t[1])
                ],
                "files": [
                    {"path": rel, "why": why, "confidence": tag} for rel, why, tag, _ in copied
                ],
                "gaps": gaps,
                "reproduce": "ONEDNN_VERBOSE=all python repro.py",
            },
            indent=2,
        )
        + "\n"
    )

    (d / "PROVENANCE.md").write_text(
        _ONEDNN_PROVENANCE.format(
            op=op,
            series=op.replace("::", "_"),
            device=device,
            keys=", ".join(keys) or "(none)",
            registered=sources[0] if sources else "(unknown)",
            runtime_version=runtime_version,
            link_version=_onednn_link_version() or "(no oneAPI install found)",
            environment="; ".join(f"`{e}`" for e in run.environment) or "(not in log)",
            checkout=checkout_md,
            revision=revision,
            schema=op_schema(op) or "(not resolvable)",
            executed=executed_md,
            selected_source=selected_source_md,
            candidates=_md_candidates(analysis),
            gates=_md_gates(analysis, run),
            strategies=_md_strategies(run),
            files=files_md,
            gaps=gaps_md,
        )
    )
    return d


def op_schema(op: str) -> Optional[str]:
    """The dispatcher's schema string for ``ns::name``, or None if it cannot be resolved.

    The schema is what says which arguments an op mutates (``Tensor(a!)``) and whether it
    returns a tensor -- the two facts a bytes-moved lower bound needs and the shapes alone
    do not carry.
    """
    import torch

    try:
        return str(
            getattr(torch.ops, op.split("::")[0]).__getattr__(op.split("::")[1]).default._schema
        )
    except Exception:
        return None


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
    prims, launched, verbose = (
        probe_library(op, argspec, device) if argspec is not None else ([], [], [])
    )
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
    run = parse_onednn_verbose(verbose) if prims else None
    if out and provider == "provider kernel":
        written = bundle(op, device, dev, sources, found, harness, out)
    elif out and provider == "oneDNN" and run is not None:
        written = bundle_onednn(op, device, dev, sources, run, harness, argspec, out, extra)
    if written:
        print(f"  bundled to    : {written}")
    return {
        "op": op,
        "provider": provider,
        "where": where,
        "route": route,
        "bundle": str(written) if written else None,
        "share": share_pct,
        "schema": op_schema(op) if (dev or prims or not keys) else None,
        "launched": launched,
        "primitives": prims,
        "dispatch_keys": keys,
        "onednn": run.summary() if run is not None else None,
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
    ap.add_argument(
        "--json",
        help="also write every op's resolution (class, source, schema, launched kernels) "
        "here, for scripts/bound_candidates.py",
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
        results.append(
            {
                "op": t["kernel"],
                "provider": "Triton",
                "where": [t["source"]],
                "route": "wrap",
                "bundle": None,
                "share": None,
                "schema": None,
                "launched": [t["kernel"]],
                "primitives": [],
                "dispatch_keys": [],
            }
        )

    if args.json:
        out_json = pathlib.Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"device": args.device, "ops": results}, indent=2) + "\n")

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
