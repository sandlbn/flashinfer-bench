"""Extract the ops a model actually runs, then emit and verify a harness for each.

Choosing what to optimize by naming a definition is guesswork: it picks an op because
someone remembered it, at a shape nobody checked, and it can name an op the model never
executes. The model already knows. Running it under a dispatch recorder yields the ops it
performed, with the shapes and dtypes it performed them at.

For each op worth harnessing this emits a file in the harness contract -- `Model`,
`get_inputs`, `get_init_inputs` -- whose `forward` calls the *same op the model called*,
imported from wherever it lives. Nothing is reimplemented, so there is no chance of tuning a
different function than the one in production.

Each generated harness is then verified before it is offered: it must run, return the shape
and dtype the model saw, and produce finite values. A harness that fails is reported and
discarded rather than left for someone to optimize against.

Some ops only run inside the serving stack's per-step context -- on a transformer, the
attention and the KV-cache update, which look their metadata and cache up from state the
model runner establishes around each forward. Such an op fails verification by raising from
the module that holds that state. Rather than special-case any op, the failure itself is the
trigger: the model is run once more, the state that module had during the real call is
captured, pruned to the entries the op consulted, and stored next to the harness, which
re-establishes it around each call. Nothing is fabricated; the harness calls the production
op against the state the production run gave it.

    python scripts/harness_from_model.py --model <repo_id> --out-dir tools/kernel-harness/auto
"""

from __future__ import annotations

import argparse
import collections
import copy
import io
import json
import os
import pathlib
import pickle
import sys
import types
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

RECORDS: Dict[Tuple[str, tuple], Dict[str, Any]] = {}

# Triton kernels never reach the dispatcher as ops -- they are JIT-compiled Python called
# straight from the stack -- so a dispatch recorder alone reports a model as having no
# Triton in it. Record them from the JIT entry point instead, with the source they were
# compiled from.
TRITON: Dict[str, Dict[str, Any]] = {}

# Which op consumed which op's output. Fusing an elementwise op into a GEMM's epilogue is
# only meaningful if the model actually runs them back to back, and no per-op tally can say
# whether it does -- that is a property of the edge, not of either endpoint.
EDGES: Dict[Tuple[str, str], int] = {}
_PRODUCER: "collections.OrderedDict[int, Tuple[str, tuple, str]]" = collections.OrderedDict()
_PRODUCER_MAX = 4096

# Ops that are plumbing rather than arithmetic: harnessing them measures the allocator or
# the copy engine, not a kernel anyone would optimize. Matched as `namespace.op` against
# `str(func)`, which a TorchDispatchMode reports as `aten.view.default`.
_SKIP_OPS = frozenset(
    {
        "aten.empty",
        "aten.zeros",
        "aten.ones",
        "aten.arange",
        "aten.detach",
        "aten.view",
        "aten._unsafe_view",
        "aten.expand",
        "aten.as_strided",
        "aten.slice",
        "aten.select",
        "aten.t",
        "aten.transpose",
        "aten.permute",
        "aten.reshape",
        "aten.squeeze",
        "aten.unsqueeze",
        "aten.to",
        "aten._to_copy",
        "aten.copy_",
        "aten.clone",
        "aten.contiguous",
        "aten.item",
        "aten.equal",
        "aten.fill_",
        "aten.resize_",
        "aten.set_",
        "aten.narrow",
        "aten.split",
    }
)

_PRIMITIVE = (int, float, bool, str, type(None))


def _jsonable(x: Any) -> Any:
    """Records hold tuples for hashing; JSON needs lists."""
    if isinstance(x, tuple):
        return [_jsonable(i) for i in x]
    return x


def _describe(x: Any) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        return ("T", tuple(x.shape), str(x.dtype).replace("torch.", ""))
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_describe(i) for i in x)
    return type(x).__name__


def _is_tensor_desc(d: Any) -> bool:
    return isinstance(d, tuple) and len(d) == 3 and d[0] == "T"


def _numel(desc: Any) -> int:
    n = 1
    for dim in desc[1]:
        n *= int(dim)
    return n


def device_time(llm, prompts: int, out_tokens: int, device: str) -> Dict[str, object]:
    """Per-op device time for this model on this stack, as a share of the total.

    Call counts are not share, and share is the only thing that says what an optimization
    is worth. It has to be measured per model and per stack -- the ranking one model
    produces is not the ranking the next one produces, and a remembered ranking is how you
    end up optimizing the wrong op confidently.

    A separate pass from the dispatch recorder: the two instrument the same calls, and
    running them together charges the recorder's own overhead to the kernels.
    """
    from torch.profiler import ProfilerActivity, profile
    from vllm import SamplingParams

    act = [ProfilerActivity.CPU]
    if device.startswith("xpu"):
        act.append(ProfilerActivity.XPU)
    elif device.startswith("cuda"):
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act) as prof:
        llm.generate(
            [f"Topic {i}." for i in range(prompts)],
            SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True),
        )

    # Every op appears twice: once as the host-side call that launched the work, carrying
    # the device time it is responsible for, and once as the device kernel itself. Summing
    # both halves double-counts the whole run and halves every share.
    from torch.autograd import DeviceType

    per_op: Dict[str, float] = {}
    kernels: Dict[str, float] = {}
    for e in prof.key_averages():
        us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if us <= 0:
            continue
        if e.device_type == DeviceType.CPU:
            per_op[e.key] = per_op.get(e.key, 0.0) + us
        else:
            kernels[e.key] = kernels.get(e.key, 0.0) + us
    return {
        "by_op": dict(sorted(per_op.items(), key=lambda kv: -kv[1])),
        "by_kernel": dict(sorted(kernels.items(), key=lambda kv: -kv[1])),
        # The denominator is the time kernels actually spent on the device.
        "total_us": sum(kernels.values()),
    }


def observe_triton():
    """Patch Triton's JIT entry point to record each kernel and where its source lives.

    Returns a restore callable. Absent Triton is not an error: a stack that compiles none
    is a fact about the stack, and the report should say so rather than fail.
    """
    try:
        from triton.runtime.jit import JITFunction
    except Exception:
        return lambda: None

    original = JITFunction.run

    def run(self, *args, **kwargs):
        try:
            fn = self.fn
            rec = TRITON.setdefault(
                self.__name__,
                {
                    "kernel": self.__name__,
                    "source": f"{fn.__code__.co_filename}:{fn.__code__.co_firstlineno}",
                    "calls": 0,
                    "constexprs": {},
                },
            )
            rec["calls"] += 1
            for k in ("BLOCK_SIZE", "BLOCK", "num_warps", "num_stages"):
                if k in kwargs:
                    rec["constexprs"][k] = kwargs[k]
        except Exception:
            pass  # never let observation break the run
        return original(self, *args, **kwargs)

    JITFunction.run = run
    return lambda: setattr(JITFunction, "run", original)


def _caller_modules(limit: int = 6) -> List[str]:
    """The stack's modules between the dispatcher and the model, innermost first.

    A custom op exists only in a process that imported whatever registered it, and the
    harness runs in a fresh process. The module that called the op is the best evidence of
    what to import to get it back -- in a serving stack the caller and the registrar are
    usually the same file -- and the frames above it are the fallbacks.
    """
    out: List[str] = []
    frame = sys._getframe(1)
    while frame is not None and len(out) < limit:
        name = frame.f_globals.get("__name__", "")
        if (
            name
            and name != __name__
            and not name.startswith(("torch", "importlib", "__main__", "contextlib"))
            and name not in out
        ):
            out.append(name)
        frame = frame.f_back
    return out


def _extension_modules() -> List[str]:
    """Compiled extensions this process loaded; the same list the resolver hands its probe."""
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from pull_kernel_source import _op_registering_modules

        return [m for m in _op_registering_modules() if "." in m]
    except Exception:
        return []


def record_mode():
    """A TorchDispatchMode that tallies every op with the shapes it ran on."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            import torch

            name_pre = str(func)
            # Read the edge before running: an in-place op overwrites the buffer it
            # consumed, so after the call the producer's output is already gone.
            for a in args:
                if isinstance(a, torch.Tensor):
                    prev = _PRODUCER.get(a.data_ptr())
                    # A freed allocation's address gets reused, which would invent an edge
                    # between unrelated ops. Requiring the shape and dtype to match too
                    # makes a false edge unlikely rather than merely uncommon.
                    if prev and prev[1] == tuple(a.shape) and prev[2] == str(a.dtype):
                        key = (prev[0], name_pre)
                        EDGES[key] = EDGES.get(key, 0) + 1

            out = func(*args, **(kwargs or {}))
            name = str(func)

            # A view or a reshape computes nothing; it renames what its input already
            # holds. Recording it as the producer hides the op that computed the data, and
            # a GEMM feeding an activation -- the edge a fusion needs -- is exactly the
            # pair a view tends to sit between. Carry the real producer through.
            carried = None
            if ".".join(name.split(".")[:2]) in _SKIP_OPS:
                for a in args:
                    if isinstance(a, torch.Tensor):
                        carried = _PRODUCER.get(a.data_ptr())
                        if carried:
                            break

            for t in out if isinstance(out, (list, tuple)) else [out]:
                if isinstance(t, torch.Tensor):
                    origin = carried[0] if carried else name
                    _PRODUCER[t.data_ptr()] = (origin, tuple(t.shape), str(t.dtype))
                    if len(_PRODUCER) > _PRODUCER_MAX:
                        _PRODUCER.popitem(last=False)
            # Destination-passing ops return nothing; their result is in an argument.
            if out is None:
                for a in args:
                    if isinstance(a, torch.Tensor):
                        _PRODUCER[a.data_ptr()] = (name, tuple(a.shape), str(a.dtype))
            if ".".join(name.split(".")[:2]) not in _SKIP_OPS:
                sig = tuple(_describe(a) for a in args)
                key = (name, sig)
                rec = RECORDS.get(key)
                if rec is None:
                    rec = RECORDS[key] = {
                        "op": name,
                        "args": sig,
                        "calls": 0,
                        "out": _describe(out),
                        "callers": _caller_modules(),
                    }
                rec["calls"] += 1
            return out

    return Recorder()


# ----------------------------------------------------------------------------- ambient state
#
# An op that raises because "the context is not set" reads module-level state that the
# stack establishes around each forward and clears afterwards. Nothing here names that
# module or that op: the failing frame names the module, the difference between that
# module's globals during the real call and outside it names the state, and an observing
# proxy installed for the duration of one call names the parts of it the op consulted.


def _globals_snapshot(module: types.ModuleType) -> Dict[str, Any]:
    """Module-level values that could be state: not names of code, not other modules."""
    code = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)
    return {
        k: v for k, v in vars(module).items() if not k.startswith("__") and not isinstance(v, code)
    }


def _tensors_in(
    obj: Any, path: tuple = (), depth: int = 0, seen=None
) -> Iterator[Tuple[tuple, Any]]:
    """Every tensor reachable from `obj` through dict keys, sequence indices and attributes.

    The path is how the harness will reach the same tensor again: `("attr", name)` and
    `("key", k)` steps from the root. Objects that are code rather than data are not
    entered -- a module's globals lead everywhere.
    """
    import torch

    seen = set() if seen is None else seen
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return
    if isinstance(obj, (types.ModuleType, type, types.FunctionType, types.MethodType)):
        return
    if isinstance(obj, dict):
        items = [(("key", k), v) for k, v in obj.items() if isinstance(k, (str, int))]
    elif isinstance(obj, (list, tuple)):
        items = [(("key", i), v) for i, v in enumerate(obj)]
    elif hasattr(obj, "__dict__") and not isinstance(obj, _PRIMITIVE):
        items = [(("attr", k), v) for k, v in vars(obj).items()]
    else:
        return
    for step, v in items:
        yield from _tensors_in(v, path + (step,), depth + 1, seen)


class _LogDict(dict):
    """A dict that reports which of its entries a caller looks up."""

    def __init__(self, source: dict, on_key, on_all):
        super().__init__(source)
        self._on_key, self._on_all = on_key, on_all

    def __getitem__(self, k):
        v = super().__getitem__(k)
        self._on_key(k, v)
        return v

    def get(self, k, default=None):
        return self[k] if super().__contains__(k) else default

    def __contains__(self, k):
        present = super().__contains__(k)
        if present:
            self._on_key(k, super().__getitem__(k))
        return present

    def _whole(self, method):
        def call(*a, **kw):
            self._on_all()
            return getattr(super(_LogDict, self), method)(*a, **kw)

        return call

    def __iter__(self):
        return self._whole("__iter__")()

    def items(self):
        return self._whole("items")()

    def values(self):
        return self._whole("values")()

    def keys(self):
        return self._whole("keys")()


class _AttrProxy:
    """Stands in for one state object during one call, reporting the attributes read."""

    def __init__(self, target, observed: "_Observed"):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_observed", observed)

    def __getattr__(self, name):
        value = getattr(self._target, name)
        obs = self._observed
        if isinstance(value, dict) and obs.attrs.get(name, set()) is not None:
            return _LogDict(
                value,
                lambda k, v: obs.hit(name, k, v),
                lambda: obs.hit_all(name, value),
            )
        obs.hit_all(name, value)
        return value

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


class _Observed:
    """One module-level value, watched through a single op call.

    Records which attributes and dict entries the op consulted, snapshots every tensor
    reachable from each consulted entry at the moment it is reached -- before the op has
    computed anything -- and afterwards yields the value pruned to what was consulted, with
    the tensors the call mutated identified. The snapshot is what the harness stores: the
    state the call started from, not the state it left behind.
    """

    def __init__(self, value: Any):
        self.value = value
        self.attrs: Dict[Optional[str], Optional[Set[Any]]] = {}
        self.before: Dict[int, Tuple[Any, Any]] = {}  # id(tensor) -> (tensor, pre-call copy)
        self.whole = False
        if isinstance(value, dict):
            self.proxy = _LogDict(
                value, lambda k, v: self.hit(None, k, v), lambda: self.hit_all(None, value)
            )
        elif hasattr(value, "__dict__") and not isinstance(value, _PRIMITIVE):
            self.proxy = _AttrProxy(value, self)
        else:
            self.proxy, self.whole = value, True
            self.snapshot(value)

    def snapshot(self, value: Any) -> None:
        for _, t in _tensors_in(value):
            if id(t) not in self.before:
                self.before[id(t)] = (t, t.detach().clone())

    def hit(self, attr: Optional[str], key: Any, value: Any) -> None:
        keys = self.attrs.setdefault(attr, set())
        if keys is not None:
            keys.add(key)
        self.snapshot(value)

    def hit_all(self, attr: Optional[str], value: Any) -> None:
        self.attrs[attr] = None
        self.snapshot(value)

    def _prune_dict(self, d: dict, keys: Optional[Set[Any]]) -> dict:
        if keys is None:
            return d
        pruned = copy.copy(d)
        pruned.clear()
        pruned.update((k, d[k]) for k in keys if k in d)
        return pruned

    def pruned(self) -> Any:
        if self.whole:
            return self.value
        if isinstance(self.value, dict):
            return self._prune_dict(self.value, self.attrs.get(None, set()))
        p = copy.copy(self.value)
        for name, val in list(vars(self.value).items()):
            if isinstance(val, dict):
                setattr(p, name, self._prune_dict(val, self.attrs.get(name, set())))
        return p

    def consulted(self) -> Dict[str, Any]:
        return {
            str(attr): (sorted(map(str, keys)) if keys is not None else "all")
            for attr, keys in self.attrs.items()
        }

    def mutated(self, pruned: Any) -> List[Tuple[tuple, Any]]:
        """Tensors of the pruned state whose contents the call changed.

        A weak witness on its own: a write of values already present leaves the contents
        unchanged, and a run that repeats the same prompts does exactly that. It is the
        fallback when the probe below cannot run.
        """
        import torch

        out = []
        for path, t in _tensors_in(pruned):
            snap = self.before.get(id(t))
            if snap is not None and not torch.equal(t, snap[1]):
                out.append((path, t))
        return out


def _probe_writes(func, args, kwargs, tensors: List[Tuple[Any, Any]]) -> Optional[List[Any]]:
    """Which of `tensors` the op writes, found by calling it once more and watching.

    The op's own schema cannot say -- a stack may declare no mutation at all so a compiler
    will not reorder it -- and no dispatch mode sees the ops a Python custom op issues
    inside its body. So the op is called again with its floating-point arguments replaced by
    random values of the same shape, which makes any write visible however idempotent the
    real one was, and every tensor it touched is then put back exactly as the real call had
    left it. Integer arguments are left alone: they may be indices, and random indices
    would write somewhere no call ever writes. Returns None when the probe is inconclusive.
    """
    import torch

    kept = [(path, t, t.detach().clone()) for path, t in tensors]
    probe_args = []
    for a in args:
        if isinstance(a, torch.Tensor):
            probe_args.append(torch.randn_like(a) if a.is_floating_point() else a.clone())
        else:
            probe_args.append(a)
    try:
        func(*probe_args, **kwargs)
    except Exception:
        return None
    written = []
    for path, t, snap in kept:
        if not torch.equal(t, snap):
            written.append(path)
            t.copy_(snap)
    return written


def _dump_state(path: pathlib.Path, payload: Any, before: Dict[int, Tuple[Any, Any]]) -> int:
    """Pickle `payload` with its tensors lifted out, pre-call copies substituted; return bytes.

    A plain pickle of a tensor view serializes the whole allocation behind it -- a KV-cache
    slice would drag the entire cache along -- so tensors are extracted, cloned, and stored
    beside the object graph with the device type they lived on. The harness restores them
    to its own device.
    """
    import torch

    tensors: List[Tuple[Any, str]] = []

    class Lift(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, torch.Tensor):
                src = before.get(id(obj), (None, obj))[1]
                tensors.append((src.detach().to("cpu", copy=True), obj.device.type))
                return len(tensors) - 1
            return None

    buf = io.BytesIO()
    Lift(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(payload)
    torch.save({"pickle": buf.getvalue(), "tensors": tensors}, path)
    return sum(t.numel() * t.element_size() for t, _ in tensors) + len(buf.getvalue())


def capture_state(
    llm,
    targets: Dict[Tuple[str, tuple], Set[str]],
    prompts: int,
    out_tokens: int,
    out_dir: pathlib.Path,
) -> Dict[Tuple[str, tuple], Dict[str, Any]]:
    """Run the model again and capture, for each target op, the state its modules held.

    `targets` maps a recorded (op, signature) to the modules whose accessors raised when the
    harness ran without the stack. The baseline for "what differs during the call" is taken
    now, outside any forward -- the same situation the harness found itself in.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm import SamplingParams

    modules = {m for ms in targets.values() for m in ms}
    outside = {m: _globals_snapshot(sys.modules[m]) for m in modules}
    results: Dict[Tuple[str, tuple], Dict[str, Any]] = {}

    def capture_one(key, func, args, kwargs):
        watched: Dict[str, Dict[str, _Observed]] = {}
        for mod_name in targets[key]:
            mod = sys.modules[mod_name]
            now = _globals_snapshot(mod)
            changed = {
                k: v
                for k, v in now.items()
                if k not in outside[mod_name] or outside[mod_name][k] is not v
            }
            if changed:
                watched[mod_name] = {k: _Observed(v) for k, v in changed.items()}
        if not watched:
            results[key] = {
                "error": f"no value in {sorted(targets[key])} differs between the model's "
                "call and the harness; the failure is not missing state"
            }
            return func(*args, **kwargs)
        for mod_name, obs in watched.items():
            for k, o in obs.items():
                setattr(sys.modules[mod_name], k, o.proxy)
        try:
            out = func(*args, **kwargs)
        finally:
            for mod_name, obs in watched.items():
                for k, o in obs.items():
                    setattr(sys.modules[mod_name], k, o.value)

        before: Dict[int, Tuple[Any, Any]] = {}
        state: Dict[str, Dict[str, Any]] = {}
        consulted: Dict[str, Dict[str, Any]] = {}
        changed: List[Tuple[list, Any]] = []
        reachable: List[Tuple[list, Any]] = []
        for mod_name, obs in watched.items():
            state[mod_name], consulted[mod_name] = {}, {}
            for k, o in obs.items():
                pruned = o.pruned()
                state[mod_name][k] = pruned
                consulted[mod_name][k] = o.consulted()
                before.update(o.before)
                changed += [([mod_name, k, list(p)], t) for p, t in o.mutated(pruned)]
                reachable += [([mod_name, k, list(p)], t) for p, t in _tensors_in(pruned)]
        # The state is real again (the proxies are gone), so the op can be probed in place.
        written = _probe_writes(func, args, kwargs, reachable)
        by_path = {json.dumps(p): t for p, t in reachable}
        mutated = [(p, by_path[json.dumps(p)]) for p in written] if written is not None else changed
        concrete = {
            i: a
            for i, a in enumerate(args)
            if not isinstance(a, (torch.Tensor, list, tuple) + _PRIMITIVE)
        }
        payload = {
            "modules": state,
            "args": concrete,
            "consulted": consulted,
            "mutated": [p for p, _ in mutated],
            "recorded_with": sys.executable,
        }
        path = out_dir / f"{_harness_name(RECORDS[key])}.state.pt"
        try:
            size = _dump_state(path, payload, before)
        except Exception as exc:
            path.unlink(missing_ok=True)
            results[key] = {
                "error": f"state of {sorted(watched)} is not serializable: "
                f"{type(exc).__name__}: {str(exc)[:80]}"
            }
            return out
        results[key] = {
            "file": path,
            "modules": {m: sorted(v) for m, v in state.items()},
            "consulted": consulted,
            "arg_indices": sorted(concrete),
            "result_path": mutated[0][0] if mutated else None,
            "result_desc": _describe(mutated[0][1]) if mutated else None,
            "bytes": size,
        }
        return out

    class Capturer(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            key = (str(func), tuple(_describe(a) for a in args))
            if key in targets and key not in results:
                return capture_one(key, func, args, kwargs)
            return func(*args, **kwargs)

    with Capturer():
        llm.generate(
            [f"Topic {i}." for i in range(prompts)],
            SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True),
        )
    return results


# ------------------------------------------------------------------------------------ emit

_TEMPLATE = '''"""Auto-generated harness for `{op}`.

Emitted by scripts/harness_from_model.py from a run of {model}: this op was called
{calls} time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.{state_doc}
"""

import importlib
import sys

import torch
import torch.nn as nn

OP = "{op}"
CALLS = {calls}
RECORDED_WITH = "{interpreter}"
# Modules on the stack when the model reached this op, innermost first, then the compiled
# extensions the serving process had loaded. An op is registered as a side effect of
# importing whatever provides it, so these are imported in order until the op resolves.
PROVIDERS = {providers}
{state_block}
_OP = None


def _op():
    global _OP
    if _OP is None:
        ns, name = OP.split(".")[:2]
        for provider in (None, *PROVIDERS):
            if provider is not None:
                try:
                    importlib.import_module(provider)
                except Exception:
                    continue
            try:
                _OP = getattr(getattr(torch.ops, ns), name)
                break
            except (AttributeError, RuntimeError):
                pass
        else:
            raise ImportError(
                f"{{OP}} is not registered in {{sys.executable}}; the harness was recorded "
                f"under {{RECORDED_WITH}}, which has the stack that provides it."
            )
    return _OP


def _device():
    return "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"


class Model(nn.Module):
    def forward(self, {params}):
{body}


def get_inputs():
    device = _device()
    return [
{inputs}
    ]


def get_init_inputs():
    return []
'''

_STATE_DOC = """

This op reads state the stack establishes around each forward step rather than taking it
as arguments. That state was captured from the recorded run -- the values {modules}
held during the call, pruned to the entries the op consulted, tensors as they were before
the call -- and lives in the `.state.pt` beside this file. `forward` re-establishes it for
the duration of each call and restores whatever was there before."""

_STATE_BLOCK = """
import contextlib
import io
import pathlib
import pickle

STATE_FILE = pathlib.Path(__file__).with_suffix(".state.pt")
CONSULTED = {consulted}
# Where, inside the state, the tensor this call writes its result lives (None: the result
# is the return value or an argument).
RESULT = {result}

_STATE = None
_MISSING = object()


class _Unpickler(pickle.Unpickler):
    def __init__(self, data, tensors, device):
        super().__init__(io.BytesIO(data))
        self._tensors, self._device = tensors, device

    def persistent_load(self, pid):
        tensor, was_on = self._tensors[pid]
        return tensor.to(self._device if was_on != "cpu" else "cpu")


def _state():
    global _STATE
    if _STATE is None:
        _op()  # the classes inside the state come from the stack that provides the op
        if not STATE_FILE.is_file():
            raise FileNotFoundError(f"{{STATE_FILE}} must sit beside this harness")
        blob = torch.load(STATE_FILE, weights_only=False, map_location="cpu")
        _STATE = _Unpickler(blob["pickle"], blob["tensors"], _device()).load()
        _STATE["_bound"] = [
            (importlib.import_module(m), name, value)
            for m, values in _STATE["modules"].items()
            for name, value in values.items()
        ]
    return _STATE


@contextlib.contextmanager
def _established():
    state = _state()
    saved = []
    for module, name, value in state["_bound"]:
        saved.append((module, name, getattr(module, name, _MISSING)))
        setattr(module, name, value)
    try:
        yield state
    finally:
        for module, name, previous in reversed(saved):
            if previous is _MISSING:
                delattr(module, name)
            else:
                setattr(module, name, previous)


def _ambient(state, path):
    node = state["modules"][path[0]][path[1]]
    for kind, key in path[2]:
        node = getattr(node, key) if kind == "attr" else node[key]
    return node

"""


def _written_args(op: str) -> List[int]:
    """Indices of the arguments the op's schema declares it writes into (`Tensor(a!)`)."""
    import torch

    ns, name, overload = (op.split(".") + ["default"])[:3]
    try:
        schema = getattr(getattr(getattr(torch.ops, ns), name), overload)._schema
    except Exception:
        return []
    return [
        i
        for i, a in enumerate(schema.arguments)
        if a.alias_info is not None and a.alias_info.is_write
    ]


def _providers(op: str, callers: List[str]) -> List[str]:
    """What the harness should import to get the op registered, most likely first.

    The stack modules that were on the stack when the op ran, then the compiled extensions
    that plausibly registered it: one whose module is named for the op's namespace, or one
    from a package related by name to the stack that called it. The rest of the loaded
    extensions -- parsers, serializers, a dozen numeric libraries -- register no ops and
    would only pad the list.
    """
    ns = op.split(".")[0]
    tops = {c.split(".")[0] for c in callers}
    related = []
    for ext in _extension_modules():
        top, last = ext.split(".")[0], ext.split(".")[-1]
        if last == ns or any(top.startswith(t) or t.startswith(top) for t in tops):
            related.append(ext)
    return list(dict.fromkeys(callers + related))


_NAMES: Dict[str, int] = {}


def _harness_name(record: Dict[str, Any]) -> str:
    """A file name for this record: the op and the first tensor's shape, made unique.

    Two records of one op can share a first shape and differ in the rest -- a projection
    at the same input width with different weights -- and the later would silently
    overwrite the earlier. Further shapes are appended until the name is this record's.
    """
    if "name" in record:
        return record["name"]
    safe = record["op"].replace("::", "_").replace(".", "_")
    shapes = ["x".join(str(d) for d in a[1]) for a in record["args"] if _is_tensor_desc(a)]
    for n in range(1, len(shapes) + 1):
        name = "_".join([safe, *shapes[:n]])
        if _NAMES.get(name, id(record)) == id(record):
            break
    _NAMES[name] = id(record)
    record["name"] = name
    return name


def _emit(record: Dict[str, Any], model: str, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Write a harness for one recorded op, or None when it cannot be expressed.

    Sets `record["expect"]` to the description of what the harness returns, which is what
    verification checks against: the op's own result when it returns one, else the argument
    its schema says it writes, else the ambient tensor the recorded call mutated.
    """
    op = record["op"]
    state = record.get("state")
    tensors = [(i, a) for i, a in enumerate(record["args"]) if _is_tensor_desc(a)]
    if not tensors:
        return None  # nothing to feed it; not a kernel worth a harness

    params, inputs, call_args = [], [], []
    for i, a in enumerate(record["args"]):
        if _is_tensor_desc(a):
            name = f"t{i}"
            params.append(name)
            _, shape, dtype = a
            maker = "randn" if dtype.startswith(("float", "bfloat")) else "ones"
            inputs.append(
                f"        torch.{maker}({list(shape)}, dtype=torch.{dtype}, device=device),"
            )
            call_args.append(name)
        elif state and i in state["arg_indices"]:
            call_args.append(f"state['args'][{i}]")
        else:
            call_args.append(repr(a))

    invoke = f"_op()({', '.join(call_args)})"
    out = record["out"]
    written = [i for i in _written_args(op) if _is_tensor_desc(record["args"][i])]
    # An empty tensor is "returns nothing useful": the result must be somewhere else.
    returns_result = (_is_tensor_desc(out) and _numel(out) > 0) or (
        out is not None and not _is_tensor_desc(out)
    )
    if returns_result:
        call, expect = f"return {invoke}", out
    elif written:
        # Destination-passing: the op returns nothing and writes into an argument. Return
        # that argument so the harness has a result to compare, and record which one so
        # verification checks it rather than the return value.
        call, expect = f"return {invoke} or t{written[0]}", record["args"][written[0]]
    elif state and state["result_path"]:
        # Its result is in neither the return value nor an argument: it wrote into the
        # ambient state. Return that tensor, so the trial loop compares what the op did.
        call, expect = f"{invoke}\n    return _ambient(state, RESULT)", state["result_desc"]
    else:
        call, expect = f"return {invoke} or t{tensors[0][0]}", tensors[0][1]
    record["expect"] = expect

    if state:
        body = "        with _established() as state:\n" + "\n".join(
            "            " + line for line in call.split("\n    ")
        )
        state_block = _STATE_BLOCK.format(
            consulted=repr(state["consulted"]), result=repr(state["result_path"])
        )
        modules = ", ".join(f"`{m}.{n}`" for m, names in state["modules"].items() for n in names)
        state_doc = _STATE_DOC.format(modules=modules)
    else:
        body = "        " + call
        state_block, state_doc = "", ""

    providers = _providers(op, record.get("callers", []))
    body_text = _TEMPLATE.format(
        op=op,
        model=model,
        calls=record["calls"],
        interpreter=sys.executable,
        providers=repr(providers),
        state_block=state_block,
        state_doc=state_doc,
        params=", ".join(params),
        body=body,
        inputs="\n".join(inputs),
    )
    path = out_dir / f"{_harness_name(record)}.py"
    path.write_text(body_text)
    return path


def verify(path: pathlib.Path, expected: Any) -> Tuple[bool, str, Optional[str]]:
    """Run the harness and check it reproduces the shape and dtype the model saw.

    Returns (ok, reason, raised_in): when the harness raised, `raised_in` is the module of
    the innermost frame, which is what tells a missing-context failure apart from a broken
    harness -- the former raises from the stack's own module, the latter from the harness.
    """
    import torch

    try:
        # Executed from source, not imported: the import system caches bytecode by the
        # file's mtime and size, and two harnesses written in the same second at the same
        # length would have one verified as the other.
        module = types.ModuleType(path.stem)
        module.__file__ = str(path)
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        out = module.Model()(*module.get_inputs())
    except Exception as exc:
        raised_in, tb = None, exc.__traceback__
        while tb is not None:
            raised_in, tb = tb.tb_frame.f_globals.get("__name__"), tb.tb_next
        return False, f"{type(exc).__name__}: {str(exc)[:90]}", raised_in
    if not isinstance(out, torch.Tensor):
        return False, f"returned {type(out).__name__}, not a tensor", None
    if not torch.isfinite(out).all():
        return False, "produced non-finite values", None
    if _is_tensor_desc(expected):
        _, shape, dtype = expected
        if tuple(out.shape) != shape:
            return False, f"shape {tuple(out.shape)} != {shape} seen in the model", None
        if str(out.dtype).replace("torch.", "") != dtype:
            return False, f"dtype {out.dtype} != {dtype} seen in the model", None
    return True, "ok", None


def _needs_state(raised_in: Optional[str], harness_module: str) -> bool:
    """Did the harness fail inside a module of the stack, rather than in itself or torch?"""
    return bool(
        raised_in
        and raised_in != harness_module
        and raised_in in sys.modules
        and not raised_in.startswith(("torch", "importlib", "builtins"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="tools/kernel-harness/auto")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--out-tokens", type=int, default=16)
    ap.add_argument("--top", type=int, default=10, help="Harness this many ops, by call count.")
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    warm = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    llm.generate(["warm"], warm)

    RECORDS.clear()
    TRITON.clear()
    EDGES.clear()
    _PRODUCER.clear()
    restore_triton = observe_triton()
    with record_mode():
        llm.generate(
            [f"Topic {i}." for i in range(args.prompts)],
            SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True),
        )
    restore_triton()

    # A second pass, for the number that decides what any of this is worth.
    # A failed device-time pass does not stop the harnesses being written -- they are
    # still worth having -- but it is recorded in the report and in the exit status, because
    # every later stage prices work from the shares this pass produces, and a report with
    # zero everywhere reads as "nothing is worth doing" rather than "nothing was measured".
    device_time_error = None
    try:
        profiled = device_time(llm, args.prompts, args.out_tokens, "xpu:0")
    except Exception as exc:
        device_time_error = f"{type(exc).__name__}: {exc}"
        print(f"  device-time pass failed ({device_time_error}); shares unavailable")
        profiled = {"by_op": {}, "by_kernel": {}, "total_us": 0.0}
    timed = profiled["by_op"]

    ranked = sorted(RECORDS.values(), key=lambda r: -r["calls"])
    by_op = collections.Counter(r["op"] for r in ranked)
    print(
        f"\n  {len(RECORDS)} distinct (op, shape) pairs over {len(by_op)} ops"
        f", {len(TRITON)} Triton kernel(s)\n"
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every op, not just the ones harnessed: resolving where each kernel lives is a separate
    # step, and it cannot resolve what this step did not write down.
    total_us = float(profiled["total_us"]) or 1.0

    # The profiler names ops `aten::linear`; the dispatch recorder names them
    # `aten.linear.default`. Same op, two spellings.
    def _share(op_dotted: str) -> Dict[str, float]:
        parts = op_dotted.split(".")
        us = timed.get(f"{parts[0]}::{parts[1]}", 0.0)
        return {"device_us": round(us, 1), "share_pct": round(100.0 * us / total_us, 2)}

    report = {
        "model": args.model,
        "device_time_error": device_time_error,
        "device_time_total_us": round(total_us, 1) if device_time_error is None else 0.0,
        "device_time_by_kernel": {
            k: round(v, 1) for k, v in list(profiled["by_kernel"].items())[:40]
        },
        "ops": [{"op": r["op"], "calls": r["calls"], "args": _jsonable(r["args"])} for r in ranked],
        "op_share": dict(
            sorted(
                ((o, _share(o)) for o in by_op),
                key=lambda kv: -kv[1]["share_pct"],
            )
        ),
        "op_calls": dict(
            sorted(
                collections.Counter(
                    {o: sum(r["calls"] for r in ranked if r["op"] == o) for o in by_op}
                ).items(),
                key=lambda kv: -kv[1],
            )
        ),
        "triton": sorted(TRITON.values(), key=lambda t: -t["calls"]),
        # producer -> consumer, so a fusion route can ask whether the pair it needs is one
        # this model actually runs.
        "edges": [
            {"producer": a, "consumer": b, "count": n}
            for (a, b), n in sorted(EDGES.items(), key=lambda kv: -kv[1])
            if n >= 4
        ],
    }
    (out_dir / "discovered.json").write_text(json.dumps(report, indent=2))
    print(f"  full op list -> {out_dir / 'discovered.json'}\n")

    made, failed = 0, 0
    todo = list(ranked[: args.top])
    # A harness may fail because the op reads state the stack sets around each forward.
    # Such failures are collected, the state is captured from another run of the model, and
    # the harness is emitted again with it. Bounded, because an op can need state from more
    # than one module and each round reveals one.
    for attempt in range(3):
        needs: Dict[Tuple[str, tuple], Set[str]] = {}
        for record in todo:
            key = (record["op"], record["args"])
            path = _emit(record, args.model, out_dir)
            if path is None:
                continue
            ok, why, raised_in = verify(path, record["expect"])
            state = record.get("state")
            if ok:
                made += 1
                note = ""
                if state:
                    where = ", ".join(f"{m}.{n}" for m, ns in state["modules"].items() for n in ns)
                    note = f"  (+ {state['bytes'] / 2**20:.0f} MB of state from {where})"
                print(f"  ok    {record['calls']:>6} calls  {record['op']:38} -> {path.name}{note}")
                continue
            path.unlink(missing_ok=True)
            if attempt < 2 and _needs_state(raised_in, path.stem):
                needs[key] = {raised_in} | set(state["modules"] if state else ())
                print(
                    f"  wait  {record['calls']:>6} calls  {record['op']:38} {why}"
                    f"\n        raised inside {raised_in}: capturing its state from the run"
                )
                continue
            if state:
                pathlib.Path(state["file"]).unlink(missing_ok=True)
            failed += 1
            print(f"  drop  {record['calls']:>6} calls  {record['op']:38} {why}")
        if not needs:
            break
        print()
        captured = capture_state(llm, needs, args.prompts, args.out_tokens, out_dir)
        todo = []
        for key in needs:
            record, result = RECORDS[key], captured.get(key)
            if result is None or "error" in result:
                failed += 1
                why = result["error"] if result else "op did not recur in the capture run"
                print(f"  drop  {record['calls']:>6} calls  {record['op']:38} {why}")
                continue
            record["state"] = result
            todo.append(record)
        print()
    print(f"\n  {made} harness(es) written to {out_dir}, {failed} discarded as unverifiable.")
    status, message = exit_status(made, failed, report)
    print(f"  {message}")
    raise SystemExit(status)


def exit_status(made: int, failed: int, report: Dict[str, Any]) -> Tuple[int, str]:
    """The exit code and closing line, from what the run achieved.

    The command promises verified harnesses with shares. Zero verified harnesses, or a
    report with no device time, is not that -- and an exit of 0 there is what lets a
    driver move to the next stage on nothing.
    """
    if report.get("device_time_error"):
        return 1, (
            "device-time pass failed; discovered.json carries no shares and "
            "scripts/bound_candidates.py will refuse it. Fix the profiler and re-run."
        )
    if made == 0:
        return 1, (
            f"no harness verified ({failed} discarded); nothing to optimize was produced. "
            "Read the drop reasons above."
        )
    return 0, "Next: pick one and run scripts/kernel_trials.py against it."


if __name__ == "__main__":
    main()
