"""Capture a real Triton kernel launch from a running model, and verify the capture.

Lifting a kernel out of a serving stack by hand is where extraction goes wrong: the
signature is long, several arguments are strides or flags derived from objects that no
longer exist outside the engine, and a reconstruction that is subtly wrong still runs. So
this does not reconstruct anything. It hooks Triton's launch path, records the arguments of
an actual launch, and then replays that exact launch standalone and compares the output
against the engine's.

**This captures; it does not yet verify.** Replaying the capture standalone and comparing
against the engine's own output is the check that would make extraction safe, and it is not
built. Until it is, treat a capture as a starting point to be validated by hand, not as a
faithful copy -- a kernel optimized against a subtly wrong capture is fast at the wrong
problem, and nothing here would say so.

What it does refuse: a launch whose arguments map to no tensors at all. Triton passes kernel
arguments positionally, so reading only `kwargs` yields the constexprs and none of the data
-- an empty harness that still looks like a capture. That case is reported and skipped.

    python scripts/capture_triton_kernel.py --model <repo_id> --kernel kernel_unified_attention
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, List, Optional

import torch

_CAPTURED: List[Dict[str, Any]] = []
_WANTED: Optional[str] = None
_LIMIT = 1


def _describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "stride": tuple(value.stride()),
            "device": str(value.device),
        }
    return value


def install_hook(kernel_name: str, limit: int) -> None:
    """Record the first `limit` launches of `kernel_name`, arguments included."""
    from triton.runtime.jit import JITFunction

    global _WANTED, _LIMIT
    _WANTED, _LIMIT = kernel_name, limit
    original = JITFunction.run

    def run(self, *args, **kwargs):
        name = getattr(self, "__name__", None) or getattr(getattr(self, "fn", None), "__name__", "")
        if name == _WANTED and len(_CAPTURED) < _LIMIT:
            # Triton passes kernel arguments positionally (`run(self, *args, grid, warmup,
            # **kwargs)`), so a capture that reads only kwargs records the constexprs and
            # none of the tensors -- an empty harness that still looks like a capture. Map
            # positions back to parameter names through the decorated function's signature.
            names = []
            fn = getattr(self, "fn", None)
            if fn is not None:
                try:
                    names = list(inspect.signature(fn).parameters)
                except (TypeError, ValueError):
                    names = []
            snapshot = {}
            for i, value in enumerate(args):
                key = names[i] if i < len(names) else f"arg{i}"
                # Clone before the launch: it mutates its outputs, and a capture taken
                # afterwards would record results as though they were inputs.
                snapshot[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items():
                snapshot[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
            if not any(isinstance(v, torch.Tensor) for v in snapshot.values()):
                # Refuse rather than save something that cannot be replayed.
                print(
                    f"  WARNING: {name} launch had no tensor arguments after mapping "
                    f"{len(args)} positional and {len(kwargs)} keyword args -- not captured."
                )
                return original(self, *args, **kwargs)
            _CAPTURED.append({"name": name, "kwargs": snapshot})
        return original(self, *args, **kwargs)

    JITFunction.run = run


def save(out_dir: str) -> int:
    """Write each capture to disk. Does not verify it replays -- see the module docstring."""
    if not _CAPTURED:
        print(f"  no launch of {_WANTED!r} was observed -- the kernel did not run.")
        print("  Check the name with scripts/observe_triton_kernels.py first.")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    for i, cap in enumerate(_CAPTURED):
        tensors = {k: v for k, v in cap["kwargs"].items() if isinstance(v, torch.Tensor)}
        meta = {k: _describe(v) for k, v in cap["kwargs"].items()}
        path = os.path.join(out_dir, f"{cap['name']}_{i}.pt")
        torch.save({"meta": meta, "tensors": tensors}, path)
        kept += 1
        print(
            f"  captured {cap['name']} #{i}: {len(tensors)} tensor arg(s), "
            f"{len(meta) - len(tensors)} scalar/constexpr"
        )
        consts = {
            k: v for k, v in cap["kwargs"].items() if isinstance(v, (int, bool)) and k.isupper()
        }
        if consts:
            print(f"     constexprs: {consts}")
        print(f"     -> {path}")
    return 0 if kept else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kernel", required=True, help="Kernel function name to capture.")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--out-dir", default="tmp/captured-kernels")
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--out-tokens", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the hook must see the worker
    install_hook(args.kernel, args.limit)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    llm.generate(
        [f"Topic {i}." for i in range(args.prompts)],
        SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True),
    )
    raise SystemExit(save(args.out_dir))


if __name__ == "__main__":
    main()
