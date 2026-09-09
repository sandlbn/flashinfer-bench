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

    python scripts/harness_from_model.py --model <repo_id> --out-dir tools/kernel-harness/auto
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
from typing import Any, Dict, Tuple

RECORDS: Dict[Tuple[str, tuple], Dict[str, Any]] = {}

# Triton kernels never reach the dispatcher as ops -- they are JIT-compiled Python called
# straight from the stack -- so a dispatch recorder alone reports a model as having no
# Triton in it. Record them from the JIT entry point instead, with the source they were
# compiled from.
TRITON: Dict[str, Dict[str, Any]] = {}

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


def record_mode():
    """A TorchDispatchMode that tallies every op with the shapes it ran on."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = str(func)
            if ".".join(name.split(".")[:2]) not in _SKIP_OPS:
                sig = tuple(_describe(a) for a in args)
                key = (name, sig)
                rec = RECORDS.setdefault(
                    key, {"op": name, "args": sig, "calls": 0, "out": _describe(out)}
                )
                rec["calls"] += 1
            return out

    return Recorder()


_TEMPLATE = '''"""Auto-generated harness for `{op}`.

Emitted by scripts/harness_from_model.py from a run of {model}: this op was called
{calls} time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.
"""

import torch
import torch.nn as nn

OP = "{op}"
CALLS = {calls}


class Model(nn.Module):
    def forward(self, {params}):
        return {call}


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
{inputs}
    ]


def get_init_inputs():
    return []
'''


def _emit(record: Dict[str, Any], model: str, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Write a harness for one recorded op, or None when it cannot be expressed."""
    op = record["op"]
    tensors = [
        (i, a) for i, a in enumerate(record["args"]) if isinstance(a, tuple) and a and a[0] == "T"
    ]
    if not tensors:
        return None  # nothing to feed it; not a kernel worth a harness

    params, inputs, call_args = [], [], []
    for i, a in enumerate(record["args"]):
        if isinstance(a, tuple) and a and a[0] == "T":
            name = f"t{i}"
            params.append(name)
            _, shape, dtype = a
            maker = "randn" if dtype.startswith(("float", "bfloat")) else "ones"
            inputs.append(
                f"        torch.{maker}({list(shape)}, dtype=torch.{dtype}, device=device),"
            )
            call_args.append(name)
        else:
            call_args.append(repr(a))

    ns, opname = op.split(".")[0], op.split(".")[1]
    invoke = f"torch.ops.{ns}.{opname}({', '.join(call_args)})"
    if record["out"] is None:
        # Destination-passing: the op returns nothing and writes into an argument. Return
        # that argument so the harness has a result to compare, and record which one so
        # verification checks it rather than the return value.
        dps = f"t{tensors[0][0]}"
        call = f"{invoke} or {dps}"
        record["result_is"] = tensors[0][1]
    else:
        call = invoke
    body = _TEMPLATE.format(
        op=op,
        model=model,
        calls=record["calls"],
        params=", ".join(params),
        call=call,
        inputs="\n".join(inputs),
    )
    safe = op.replace("::", "_").replace(".", "_")
    shape_tag = "x".join(str(d) for _, a in tensors[:1] for d in a[1])
    path = out_dir / f"{safe}_{shape_tag}.py"
    path.write_text(body)
    return path


def verify(path: pathlib.Path, expected_out: Any) -> Tuple[bool, str]:
    """Run the harness and check it reproduces the shape and dtype the model saw."""
    import importlib.util

    import torch

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = module.Model()(*module.get_inputs())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:90]}"
    if not isinstance(out, torch.Tensor):
        return False, f"returned {type(out).__name__}, not a tensor"
    if not torch.isfinite(out).all():
        return False, "produced non-finite values"
    if expected_out is None:
        return True, "ok (destination-passing; returns the mutated argument)"
    if isinstance(expected_out, tuple) and expected_out and expected_out[0] == "T":
        _, shape, dtype = expected_out
        if tuple(out.shape) != shape:
            return False, f"shape {tuple(out.shape)} != {shape} seen in the model"
        if str(out.dtype).replace("torch.", "") != dtype:
            return False, f"dtype {out.dtype} != {dtype} seen in the model"
    return True, "ok"


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
    restore_triton = observe_triton()
    with record_mode():
        llm.generate(
            [f"Topic {i}." for i in range(args.prompts)],
            SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True),
        )
    restore_triton()

    # A second pass, for the number that decides what any of this is worth.
    try:
        profiled = device_time(llm, args.prompts, args.out_tokens, "xpu:0")
    except Exception as exc:  # profiling is not worth failing discovery over
        print(f"  device-time pass failed ({type(exc).__name__}: {exc}); shares unavailable")
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
        "device_time_total_us": round(total_us, 1),
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
    }
    (out_dir / "discovered.json").write_text(json.dumps(report, indent=2))
    print(f"  full op list -> {out_dir / 'discovered.json'}\n")
    made, failed = 0, 0
    for record in ranked[: args.top]:
        path = _emit(record, args.model, out_dir)
        if path is None:
            continue
        ok, why = verify(path, record["out"])
        if ok:
            made += 1
            print(f"  ok    {record['calls']:>6} calls  {record['op']:38} -> {path.name}")
        else:
            failed += 1
            path.unlink(missing_ok=True)
            print(f"  drop  {record['calls']:>6} calls  {record['op']:38} {why}")
    print(f"\n  {made} harness(es) written to {out_dir}, {failed} discarded as unverifiable.")
    print("  Next: pick one and run scripts/kernel_trials.py against it.")


if __name__ == "__main__":
    main()
