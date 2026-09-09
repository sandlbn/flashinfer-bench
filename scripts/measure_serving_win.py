"""Measure what our kernels are worth end-to-end: tokens/sec under vLLM, A/B.

A per-kernel speedup against a PyTorch reference is not a serving result. This runs the same
generation twice -- once on stock vLLM, once with our kernels substituted through `apply()`
-- and reports both the throughput delta and the dispatch counters that prove the
substitution happened.

The two arms run in separate processes because the patch installs at interpreter start, via
`sitecustomize`, and cannot be toggled inside one run. That is also why the switches are
environment variables: vLLM's V1 engine runs the model in a worker subprocess, and a launcher
that patches its own process patches a class that never sees a forward pass.

`--plain-arm` is for a subject that is not `apply()`: a provider build or a source patch.
Neither arm then carries the integration -- every `FIB_*` variable is removed from both --
and the arms differ only by `--env`, so what is measured is the patch and nothing else.

Every arm prints a digest of the tokens it generated. Under greedy decoding the two arms
must agree; a "win" whose digests differ is a correctness change wearing a throughput number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

_CHILD = "--_run-arm"


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _token_digest(sequences: Iterable[Iterable[int]]) -> str:
    """One hash over every generated sequence, in prompt order.

    Under greedy decoding two arms that compute the same thing produce the same tokens. If
    the digests differ, the faster arm was faster at a different computation, and the
    delta between them is not a kernel comparison.
    """
    h = hashlib.sha256()
    for seq in sequences:
        h.update(",".join(str(int(t)) for t in seq).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _tokens_identical(a: List[str], b: List[str]) -> Optional[bool]:
    """Whether two arms' digest sets prove they generated the same tokens.

    None when an arm reported none. False also when an arm was not stable across its own
    repeats: a comparison against a moving target is not a comparison.
    """
    if not a or not b:
        return None
    return a == b and len(a) == 1


def _bench(model: str, prompts: int, out_tokens: int, gpu_util: float, max_len: int) -> None:
    """One arm, in its own process. Only reached via the child invocation."""
    from vllm import LLM, SamplingParams

    texts = [
        f"Write a short paragraph about topic number {i} in computing history."
        for i in range(prompts)
    ]
    # ignore_eos fixes the generated length: without it the two arms can stop at different
    # points and the comparison silently measures different amounts of work.
    params = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)

    llm = LLM(
        model=model,
        max_model_len=max_len,
        enforce_eager=True,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
    )
    llm.generate(texts[:2], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))

    start = time.perf_counter()
    outputs = llm.generate(texts, params)
    elapsed = time.perf_counter() - start

    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(
        "FIB_RESULT "
        + json.dumps(
            {
                "generated_tokens": generated,
                "seconds": round(elapsed, 4),
                "tokens_per_sec": round(generated / elapsed, 2),
                "token_digest": _token_digest(o.outputs[0].token_ids for o in outputs),
            }
        ),
        flush=True,
    )


# vLLM prefixes worker output with "(EngineCore pid=NNN) " -- a space inside the parens --
# so this must not try to anchor past the prefix.
_DISPATCH = re.compile(r"(\w+): (\d+) call\(s\), (\d+) applied")


def _parse(stdout: str) -> Dict[str, Any]:
    """Throughput plus the dispatch counters, which decide whether the number means anything."""
    result: Dict[str, Any] = {"tokens_per_sec": None, "dispatch": {}, "detail": []}
    for line in stdout.splitlines():
        # Not `startswith`: vLLM and its dependencies emit warnings without a trailing
        # newline, which land glued to the front of this line. Anchoring on the start of
        # the line loses the whole measurement and reports the arm as a failure.
        marker = line.find("FIB_RESULT {")
        if marker != -1:
            result.update(json.loads(line[marker + len("FIB_RESULT ") :]))
        m = _DISPATCH.search(line)
        if m:
            family, calls, applied = m.group(1), int(m.group(2)), int(m.group(3))
            result["dispatch"][family] = (calls, applied)
        if "detail: {" in line:
            result["detail"].append(line.split("detail: ", 1)[1].strip())
    return result


def _env_overrides(args: argparse.Namespace) -> Dict[str, str]:
    """The --env KEY=VALUE pairs, which belong to the patched arm only."""
    overrides: Dict[str, str] = {}
    for item in getattr(args, "env", None) or []:
        key, _, value = item.partition("=")
        overrides[key] = value
    return overrides


def _arm_env(
    base: Mapping[str, str],
    args: argparse.Namespace,
    patched: bool,
    empty_dataset: Optional[str] = None,
) -> Dict[str, str]:
    """The environment one arm runs under. Pure, so both modes can be checked without vLLM.

    Default (apply) mode: the baseline is stock vLLM with the integration switched off; the
    patched arm switches it on, points it at the dataset, and gets --env on top.

    Plain mode (--plain-arm): neither arm carries the integration -- every ``FIB_*``
    variable is removed from both -- and the patched arm differs from the baseline only by
    --env. A provider or source patch measured with apply() in the path would otherwise be
    measured together with apply()'s own dispatch cost and substitutions.
    """
    env = dict(base)
    # Invoking the interpreter by absolute path leaves its venv's bin/ off PATH, so the
    # SYCL builder cannot find `ninja` and every SYCL solution falls back silently -- which
    # surfaces as applied: 0, indistinguishable from having no solution at all.
    bindir = str(Path(sys.executable).parent)
    if bindir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")

    if getattr(args, "plain_arm", False):
        for key in [k for k in env if k.startswith("FIB_")]:
            env.pop(key)
        if patched:
            env.update(_env_overrides(args))
        return env

    if patched:
        env["FIB_VLLM_INTEGRATION"] = "1"
        # Extra integration flags belong to the patched arm only: the baseline is stock
        # vLLM, and setting them there would make the two arms differ in more than the
        # thing under test.
        env.update(_env_overrides(args))
        env["FIB_ENABLE_APPLY"] = "1"
        # An empty dataset installs the patch and runs every interception, but nothing can
        # ever match -- which is exactly the cost of being in the path, with none of the
        # benefit. Subtracting it separates what the kernels gained from what the dispatch
        # spent, and those are routinely the same order of magnitude.
        env["FIB_DATASET_PATH"] = empty_dataset or str(Path(args.dataset).resolve())
    else:
        for key in ("FIB_VLLM_INTEGRATION", "FIB_ENABLE_APPLY", "FIB_DATASET_PATH"):
            env.pop(key, None)
    return env


def _run_arm(
    args: argparse.Namespace, patched: bool, empty_dataset: Optional[str] = None
) -> Dict[str, Any]:
    env = _arm_env(os.environ, args, patched, empty_dataset)
    cmd = [
        sys.executable,
        __file__,
        _CHILD,
        "--model",
        args.model,
        "--prompts",
        str(args.prompts),
        "--out-tokens",
        str(args.out_tokens),
        "--gpu-util",
        str(args.gpu_util),
        "--max-model-len",
        str(args.max_model_len),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    combined = proc.stdout + "\n" + proc.stderr
    parsed = _parse(combined)
    if parsed["tokens_per_sec"] is None:
        parsed["error"] = _root_cause(combined) or f"exit {proc.returncode}"
        parsed["log"] = _dump_log(args.model, patched, combined)
    return parsed


# vLLM's top-level failure is always "Engine core initialization failed. See root cause
# above" -- the cause is hundreds of lines earlier, in the worker's own traceback. A plain
# tail of the output therefore reports the symptom every time and never the reason.
# An exception line has a distinctive shape -- `SomeError: message` -- which traceback
# frames, caret rules and log prose do not. Matching that, rather than any line containing
# the word "error", is what keeps the reported cause from being a random source line.
_EXCEPTION = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception|Interrupt))\b: (.+)")

# A second pass for failures that raise no exception line at all.
_CAUSE = re.compile(
    r"(out of memory|OutOfMemory|not supported|Unsupported|No available memory)",
    re.IGNORECASE,
)

# vLLM's own wrapper never names the reason, and the dispatch counters legitimately
# contain words like "unsupported"; both outrank the real cause when the last match wins.
_NOISE = re.compile(r"See root cause above|Failed core proc|detail: \{|call\(s\), ")


def _root_cause(output: str) -> str:
    """The most specific error lines, preferring the worker's over the launcher's."""
    hits = []
    for line in output.splitlines():
        if _NOISE.search(line):
            continue
        match = _EXCEPTION.search(line)
        if match:
            hits.append(f"{match.group(1)}: {match.group(2).strip()}"[:300])
    if not hits:
        hits = [
            line.strip()[:300]
            for line in output.splitlines()
            if _CAUSE.search(line) and not _NOISE.search(line)
        ]
    seen, uniq = set(), []
    for line in hits:
        if line[:160] not in seen:
            seen.add(line[:160])
            uniq.append(line)
    return "\n".join(uniq[-4:])


def _dump_log(model: str, patched: bool, output: str) -> str:
    """Keep the whole failing run; a truncated failure is not diagnosable."""
    path = Path("tmp/serving-win") / (
        f"{model.replace('/', '-')}.{'ours' if patched else 'baseline'}.fail.log"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        return str(path)
    except OSError:
        return ""


def _supported(model: str) -> Optional[bool]:
    """Whether vLLM registers this architecture. None when it cannot be determined.

    An absent vLLM is not "undeterminable" -- it means this interpreter cannot run the
    harness at all -- so that case raises rather than returning None.
    """
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError as exc:
        raise SystemExit(
            f"vLLM is not importable here ({exc}). Run this script with the interpreter "
            "of the environment vLLM is installed in -- the same one whose sitecustomize "
            "installs the integration; the child arms inherit it via sys.executable."
        ) from exc
    try:
        from transformers import AutoConfig

        archs = (
            getattr(
                AutoConfig.from_pretrained(model, trust_remote_code=True), "architectures", None
            )
            or []
        )
        return any(a in set(ModelRegistry.get_supported_archs()) for a in archs)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--dataset", default="tmp/flashinfer-trace", help="Dataset apply() draws solutions from."
    )
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Times to run each arm. Arms alternate, and the median is reported; a single "
        "run cannot separate a kernel effect from run-to-run drift.",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="KEY=VALUE set in the patched arm only, e.g. FIB_VLLM_MLP_FUSION=1. Use to "
        "A/B an integration flag against the same baseline. Repeatable.",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--overhead-arm",
        action="store_true",
        help="Add a third arm: patched, but pointed at an empty dataset, so nothing can "
        "match. Separates what apply() costs to sit in the path from what the kernels win.",
    )
    mode.add_argument(
        "--plain-arm",
        action="store_true",
        help="Run both arms with no apply() in the path: every FIB_* variable is removed "
        "from both, and the arms differ only by --env. For a provider build or a source "
        "patch, which apply() would otherwise mask. No dispatch counters exist in this mode.",
    )
    ap.add_argument("--json", type=Path, help="Also write the result here.")
    ap.add_argument(_CHILD, dest="child", action="store_true", help=argparse.SUPPRESS)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.child:
        _bench(args.model, args.prompts, args.out_tokens, args.gpu_util, args.max_model_len)
        return

    if _supported(args.model) is False:
        raise SystemExit(
            f"vLLM does not register an architecture for {args.model}, so there is no "
            "serving number to measure. Per-kernel results are all this model can give."
        )

    # Arms alternate rather than running all baselines then all patched runs, so that
    # any thermal or clock drift over the session falls on both equally.
    arms = ["baseline", "ours"] + (["overhead"] if args.overhead_arm else [])
    runs: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    with tempfile.TemporaryDirectory(prefix="fib-empty-") as empty:
        for _ in range(max(1, args.repeats)):
            runs["baseline"].append(_run_arm(args, patched=False))
            runs["ours"].append(_run_arm(args, patched=True))
            if args.overhead_arm:
                runs["overhead"].append(_run_arm(args, patched=True, empty_dataset=empty))

    base, ours = runs["baseline"][-1], runs["ours"][-1]
    rates = {
        arm: sorted(r["tokens_per_sec"] for r in rs if r.get("tokens_per_sec"))
        for arm, rs in runs.items()
    }

    print(f"\n  model: {args.model}")
    for label in arms:
        got = rates[label]
        if not got:
            err = next((r["error"] for r in runs[label] if r.get("error")), "unknown")
            print(f"  {label:9} FAILED: {err.splitlines()[-1][:100]}")
            log = next((r.get("log") for r in runs[label] if r.get("log")), None)
            if log:
                print(f"  {'':9} full log: {log}")
            continue
        med = _median(got)
        spread = (max(got) - min(got)) / med * 100 if len(got) > 1 else 0.0
        extra = f"  [n={len(got)}, spread {spread:.1f}%]" if len(got) > 1 else ""
        print(f"  {label:9} {med:>8.2f} tok/s{extra}")

    if rates["baseline"] and rates["ours"]:
        b, o = _median(rates["baseline"]), _median(rates["ours"])
        delta = o / b - 1.0
        print(f"  delta     {delta * 100:+.2f}%")
        # A delta inside the arms' own scatter is not a result.
        noise = (
            max((max(v) - min(v)) / _median(v) for v in rates.values() if len(v) > 1)
            if any(len(v) > 1 for v in rates.values())
            else 0.0
        )
        if noise and abs(delta) <= noise:
            print(
                f"  {'':9} within run-to-run spread ({noise * 100:.1f}%) -- "
                "not distinguishable from noise"
            )
        if rates.get("overhead"):
            h = _median(rates["overhead"])
            tax = h / b - 1.0
            print(f"  dispatch  {tax * 100:+.2f}%  (patched, nothing matchable)")
            print(f"  kernels   {(o / h - 1.0) * 100:+.2f}%  (ours vs that)")

    # Same tokens, or not a comparison. Greedy decoding with a fixed length makes the two
    # arms' outputs a deterministic function of the model, so a differing digest means an
    # arm computed something else -- and a differing digest within one arm means its own
    # repeats disagree, which no delta can be read against.
    digests = {
        arm: sorted({r["token_digest"] for r in rs if r.get("token_digest")})
        for arm, rs in runs.items()
    }
    identical = _tokens_identical(digests["baseline"], digests["ours"])
    print("  tokens:")
    for label in arms:
        got = digests.get(label) or []
        flag = "" if len(got) <= 1 else "   <- not stable across its own repeats"
        print(f"    {label:9} digest {' '.join(got) or 'n/a'}{flag}")
    if identical is True:
        print("    identical across arms")
    elif identical is False:
        print(
            "    DIFFER -- the arms did not generate the same tokens under greedy decoding; "
            "the delta above includes a correctness change and is not a kernel comparison"
        )
    else:
        print("    not compared -- an arm reported no digest")

    # The validity gate. Without this the throughput number cannot be interpreted.
    if args.plain_arm:
        diff = _env_overrides(args)
        print("  substitution: n/a -- plain arm, no apply() in either arm")
        print(
            "  arms differ by: "
            + (", ".join(f"{k}={v}" for k, v in diff.items()) or "nothing (A/A: noise floor)")
        )
    elif ours["dispatch"]:
        print("  substitution:")
        for family, (calls, applied) in sorted(ours["dispatch"].items()):
            pct = 100.0 * applied / calls if calls else 0.0
            note = "" if applied else "   <- never ran; this family contributed nothing"
            print(f"    {family:20} {applied}/{calls} applied ({pct:.1f}%){note}")
        for d in ours["detail"]:
            print(f"    detail: {d}")
    else:
        print(
            "  substitution: NO dispatch counters -- the patch did not install; "
            "the timing above is not a comparison."
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "mode": "plain" if args.plain_arm else "apply",
                    "runs": runs,
                    "baseline": base,
                    "ours": ours,
                    "digests": digests,
                    "tokens_identical": identical,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
