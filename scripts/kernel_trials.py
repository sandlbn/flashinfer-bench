"""A trial loop for kernel optimization: propose, measure, branch, keep the best.

Optimizing a kernel is a search, and a search needs three things this repo did not have: a
record of what was tried and what came of it, a measurement the searcher cannot get wrong,
and a policy for where to go next. Without the first, a loop repeats itself; without the
second it optimizes noise -- every measurement error in this repo's Intel work came from an
ad-hoc timing script; without the third it walks forward from a regression instead of going
back to what worked.

Trials form a tree, not a chain. Each records its parent and the strategy that produced it,
so a regression branches back to the best node rather than compounding.

    kernel-trials init      <name> <baseline.py>
    kernel-trials save      <name> <candidate.py> [--parent t3] [--strategy "..."]
    kernel-trials benchmark <name> <candidate.py> [--baseline-us N]
    kernel-trials status    <name>
    kernel-trials best      <name>
    kernel-trials finalize  <name> <output.py>

`benchmark` is the only sanctioned way to get a number. It gates on correctness before it
times anything, warms every arm before timing any of them, and alternates the arms in
interleaved rounds -- a GPU that has been idle ramps its clocks, so a sequential sweep
charges the ramp to whichever case runs first and can invert the comparison outright.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Any, Dict, Optional

TRIALS_DIR = pathlib.Path("tmp/kernel-trials")


# --------------------------------------------------------------------------- store


def _rel(path: str) -> str:
    """Store paths relative to the repo when possible.

    An absolute path orphans an entire trial series the moment a file is moved or the repo
    is cloned elsewhere, and a series is meant to outlive any one layout.
    """
    resolved = pathlib.Path(path).resolve()
    try:
        return str(resolved.relative_to(pathlib.Path.cwd()))
    except ValueError:
        return str(resolved)


def _store(name: str) -> pathlib.Path:
    return TRIALS_DIR / f"{name}.json"


def _load(name: str) -> Dict[str, Any]:
    path = _store(name)
    if not path.exists():
        raise SystemExit(f"No trial series {name!r}. Run `init` first.")
    return json.loads(path.read_text())


def _save_store(name: str, data: Dict[str, Any]) -> None:
    _store(name).parent.mkdir(parents=True, exist_ok=True)
    _store(name).write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------- measure


def _load_model(path: str):
    """Import a harness file and return (Model instance, inputs)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"harness_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The harness contract is three names, checked here and nowhere else. It is deliberately
    # not an import of any benchmarking package: this loop runs the file itself, so a harness
    # stays usable if an external harness format changes or goes away. The shape happens to
    # match the KernelBench/ai-bench convention, which costs nothing and makes a harness
    # portable to those runners, but nothing here requires them.
    for required in ("Model", "get_inputs", "get_init_inputs"):
        if not hasattr(module, required):
            raise SystemExit(
                f"{path} defines no {required}(). A harness needs exactly three names: "
                "Model (nn.Module), get_inputs() -> list[Tensor], get_init_inputs() -> list. "
                "See /wrap-kernel-for-tuning."
            )
    return module.Model(*module.get_init_inputs()), module.get_inputs()


def _sync() -> None:
    import torch

    for backend in (getattr(torch, "xpu", None), getattr(torch, "cuda", None)):
        if backend is not None and backend.is_available():
            backend.synchronize()
            return


def _time_one(fn, calls: int) -> float:
    """Microseconds per call over `calls` calls inside one timed region."""
    _sync()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    _sync()
    return (time.perf_counter() - start) / calls * 1e6


def benchmark(
    baseline_path: str, candidate_path: str, rounds: int, calls: int, atol: float, rtol: float
) -> Dict[str, Any]:
    """Correctness first, then interleaved timing of both arms."""
    import torch

    base_model, inputs = _load_model(baseline_path)
    cand_model, _ = _load_model(candidate_path)

    with torch.no_grad():
        expected = base_model(*inputs)
        actual = cand_model(*inputs)

    if isinstance(expected, torch.Tensor) != isinstance(actual, torch.Tensor):
        return {"correctness": "fail", "reason": "candidate returned a different type"}
    outs = (expected, actual) if isinstance(expected, torch.Tensor) else (expected[0], actual[0])
    if outs[0].shape != outs[1].shape:
        return {
            "correctness": "fail",
            "reason": f"shape {tuple(outs[1].shape)} != {tuple(outs[0].shape)}",
        }
    err = (outs[1].float() - outs[0].float()).abs().max().item()
    tol = atol + rtol * outs[0].float().abs().max().item()
    if not torch.isfinite(outs[1]).all():
        return {"correctness": "fail", "reason": "candidate produced non-finite values"}
    if err > tol:
        return {"correctness": "fail", "reason": f"max abs error {err:.5g} > {tol:.5g}"}

    # Warm both before timing either: whichever runs first otherwise absorbs the clock ramp.
    with torch.no_grad():
        for _ in range(max(10, calls)):
            base_model(*inputs)
            cand_model(*inputs)
        _sync()

        base_samples, cand_samples = [], []
        for _ in range(rounds):
            base_samples.append(_time_one(lambda: base_model(*inputs), calls))
            cand_samples.append(_time_one(lambda: cand_model(*inputs), calls))

    base_us = statistics.median(base_samples)
    cand_us = statistics.median(cand_samples)
    spread = (max(cand_samples) - min(cand_samples)) / cand_us if cand_us else 0.0
    return {
        "correctness": "pass",
        "max_abs_error": err,
        "baseline_us": base_us,
        "candidate_us": cand_us,
        "speedup": base_us / cand_us if cand_us else 0.0,
        "spread": spread,
        "rounds": rounds,
        "calls_per_round": calls,
    }


# --------------------------------------------------------------------------- commands


def cmd_init(args) -> None:
    data = {"name": args.name, "baseline": str(pathlib.Path(args.baseline).resolve()), "trials": []}
    _save_store(args.name, data)
    print(f"  initialised {args.name} with baseline {args.baseline}")


def cmd_save(args) -> None:
    data = _load(args.name)
    trial_id = f"t{len(data['trials'])}"
    data["trials"].append(
        {
            "id": trial_id,
            "file": _rel(args.file),
            "parent": args.parent,
            "strategy": args.strategy or "",
            "result": None,
        }
    )
    _save_store(args.name, data)
    print(f"  saved {trial_id}" + (f" (from {args.parent})" if args.parent else ""))


def cmd_benchmark(args) -> None:
    data = _load(args.name)
    result = benchmark(data["baseline"], args.file, args.rounds, args.calls, args.atol, args.rtol)
    if args.trial:
        for trial in data["trials"]:
            if trial["id"] == args.trial:
                trial["result"] = result
                break
        else:
            raise SystemExit(f"no trial {args.trial!r} in {args.name}")
        _save_store(args.name, data)

    if result["correctness"] != "pass":
        print(f"  INCORRECT: {result['reason']}")
        raise SystemExit(1)
    print(f"  correct (max abs error {result['max_abs_error']:.5g})")
    print(f"  baseline  {result['baseline_us']:9.2f}us")
    print(
        f"  candidate {result['candidate_us']:9.2f}us   {result['speedup']:.2f}x"
        f"   spread {result['spread'] * 100:.1f}%"
    )
    # Only a *difference* smaller than the scatter is unmeasured. Comparing the signed gain
    # against the spread reported a 100x regression as "inside noise", which is the opposite
    # of what the check is for.
    if abs(result["speedup"] - 1) < result["spread"]:
        print(
            "  NOTE: the difference is inside the candidate's own run-to-run spread "
            f"({result['spread'] * 100:.1f}%); raise --rounds or --calls before believing "
            "it in either direction."
        )
        print(
            "  NOTE: the gain is inside the candidate's own run-to-run spread; "
            "raise --rounds or --calls before believing it."
        )


def _best(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    passing = [
        t for t in data["trials"] if t["result"] and t["result"].get("correctness") == "pass"
    ]
    return max(passing, key=lambda t: t["result"]["speedup"], default=None)


def cmd_status(args) -> None:
    data = _load(args.name)
    print(f"\n  {args.name}: {len(data['trials'])} trial(s), baseline {data['baseline']}\n")
    print(f"  {'id':5} {'parent':7} {'speedup':>9}  strategy")
    print("  " + "-" * 76)
    for trial in data["trials"]:
        res = trial["result"]
        if res is None:
            verdict = "not run"
        elif res["correctness"] != "pass":
            verdict = "INCORRECT"
        else:
            verdict = f"{res['speedup']:.2f}x"
        print(
            f"  {trial['id']:5} {trial['parent'] or '-':7} {verdict:>9}  {trial['strategy'][:48]}"
        )
    best = _best(data)
    if best:
        print(f"\n  best: {best['id']} at {best['result']['speedup']:.2f}x")
        print("  A regression should branch from here (--parent), not continue forward.")
    else:
        print("\n  no correct trial yet")


def cmd_best(args) -> None:
    best = _best(_load(args.name))
    if best is None:
        raise SystemExit("no correct trial yet")
    print(json.dumps(best, indent=2))


def cmd_finalize(args) -> None:
    import shutil

    best = _best(_load(args.name))
    if best is None:
        raise SystemExit("no correct trial to finalize")
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["file"], args.output)
    print(f"  {best['id']} ({best['result']['speedup']:.2f}x) -> {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("name")
    p.add_argument("baseline")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("save")
    p.add_argument("name")
    p.add_argument("file")
    p.add_argument("--parent")
    p.add_argument("--strategy")
    p.set_defaults(func=cmd_save)

    p = sub.add_parser("benchmark")
    p.add_argument("name")
    p.add_argument("file")
    p.add_argument("--trial", help="Record the result against this trial id.")
    p.add_argument(
        "--rounds", type=int, default=7, help="Interleaved rounds; the median of these is reported."
    )
    p.add_argument(
        "--calls",
        type=int,
        default=30,
        help="Calls inside one timed region, so per-call overhead amortizes.",
    )
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.set_defaults(func=cmd_benchmark)

    for name, fn in (("status", cmd_status), ("best", cmd_best)):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.set_defaults(func=fn)

    p = sub.add_parser("finalize")
    p.add_argument("name")
    p.add_argument("output")
    p.set_defaults(func=cmd_finalize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
