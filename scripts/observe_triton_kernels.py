"""Record which Triton kernels a model actually launches, and how often.

Which kernels a serving stack uses is a property of the model, not of the library: vLLM
ships hundreds of Triton kernels and any one model touches a handful. Enumerating the
library and mapping it wholesale wastes the effort on kernels that never run; enumerating
what a model *did* run is evidence, and it is what decides where a baseline or an
optimization round is worth spending.

This hooks Triton's JIT launch path, runs a short generation, and reports every kernel that
fired with its call count and the constexpr values it was specialised on. Nothing is
patched permanently and no kernel is replaced -- it only observes.

Run it with the interpreter of the environment vLLM is installed in. vLLM's V1 engine runs
the model in a worker subprocess, so the hook is installed from `sitecustomize`-style
environment state rather than from this process; `--in-process` forces a single process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the counts come back here.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, Tuple

_LAUNCHES: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)


def install_hook() -> None:
    """Count every Triton kernel launch, keyed by kernel and specialisation."""
    from triton.runtime.jit import JITFunction

    if getattr(JITFunction, "_fib_observed", False):
        return
    original = JITFunction.run

    def run(self, *args, **kwargs):
        try:
            src = getattr(self, "fn", None)
            module = getattr(src, "__module__", "?") if src else "?"
            name = getattr(self, "__name__", None) or getattr(src, "__name__", "?")
            # constexpr values are what the kernel was specialised on -- block sizes, head
            # dims, flags. They are what a tuning round would vary, so they belong in the
            # report rather than a bare call count.
            consts = {k: v for k, v in kwargs.items() if isinstance(v, (int, bool)) and k.isupper()}
            _LAUNCHES[(module, name)][json.dumps(consts, sort_keys=True)] += 1
        except Exception:
            pass  # observation must never break the run it is watching
        return original(self, *args, **kwargs)

    JITFunction.run = run
    JITFunction._fib_observed = True


def report(top: int) -> None:
    if not _LAUNCHES:
        print("  no Triton kernel launches observed -- the hook did not see the worker.")
        print("  Re-run with --in-process, or install the hook in the worker's interpreter.")
        return
    rows = sorted(
        ((sum(spec.values()), mod, name, spec) for (mod, name), spec in _LAUNCHES.items()),
        reverse=True,
    )
    print(f"\n  {len(rows)} distinct Triton kernel(s) launched\n")
    print(f"  {'calls':>9}  {'kernel':38} module")
    print("  " + "-" * 96)
    for calls, mod, name, spec in rows[:top]:
        print(f"  {calls:>9}  {name[:38]:38} {mod}")
        if len(spec) > 1:
            print(f"  {'':>9}  ({len(spec)} distinct specialisations)")
        for blob, n in sorted(spec.items(), key=lambda kv: -kv[1])[:2]:
            consts = json.loads(blob)
            if consts:
                shown = ", ".join(f"{k}={v}" for k, v in list(consts.items())[:6])
                print(f"  {'':>9}    {n:>8} x  {shown}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--out-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="Run the engine in this process so the hook sees its launches.",
    )
    ap.add_argument("--json", type=str, help="Also write the observed kernels here.")
    args = ap.parse_args()

    if args.in_process:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    install_hook()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
    )
    llm.generate(
        [f"Describe topic {i} briefly." for i in range(args.prompts)],
        SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True),
    )
    report(args.top)

    if args.json:
        payload = [
            {
                "module": mod,
                "kernel": name,
                "calls": sum(spec.values()),
                "specialisations": dict(spec),
            }
            for (mod, name), spec in _LAUNCHES.items()
        ]
        with open(args.json, "w") as fh:
            json.dump(sorted(payload, key=lambda r: -r["calls"]), fh, indent=2)
        print(f"\n  written to {args.json}")


if __name__ == "__main__":
    main()
