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

The result is gated, and a failed gate halts: no throughput table is printed, the verdict
names the gate, and the exit is non-zero. The gates are that every arm ran, that every arm
generated the same tokens (and each arm the same tokens on every repeat), that something
was in fact substituted, and -- when `--bound` names the routing -- that the (candidate,
mechanism) being measured is one `scripts/bound_candidates.py` accepted for this model's
current discovery. Everything that helps fix the failure is printed; the delta is not,
because a delta that survives a failed gate is the line that gets quoted.

The output is the key contract `scripts/kernel_trials.py` prints -- one `KEY: value` per
line, `VERDICT` naming the outcome, `DONE` last -- so a driving agent reads keys and never a
sentence. Verdicts: WIN, LOSS, NOISE (valid; a NOISE delta is inside the arms' own
scatter), and the halts ARM_FAILED, TOKENS_DIFFER, TOKENS_UNSTABLE, TOKENS_UNCOMPARED,
NOT_SUBSTITUTED, ROUTING_REJECTED, STALE_INPUT, MECHANISM_MISMATCH, UNSUPPORTED_MODEL.
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


def _bench(
    model: str,
    prompts: int,
    out_tokens: int,
    gpu_util: float,
    max_len: int,
    prompt_tokens: int = 0,
) -> None:
    """One arm, in its own process. Only reached via the child invocation."""
    from vllm import LLM, SamplingParams

    # Prompt length decides which regime the run measures. A short prompt with many output
    # tokens is almost entirely decode, so a change that only helps prefill is invisible in
    # the result -- and a null reads as "the change is worth nothing" rather than "this run
    # never exercised it". The filler's content is irrelevant; its length is not.
    filler = (
        "The quick brown fox jumps over the lazy dog while the serving stack prefills "
        "the prompt and records what it ran. "
    )
    texts = [
        f"Write a short paragraph about topic number {i} in computing history. "
        + filler * max(0, prompt_tokens // 20)
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
        "--prompt-tokens",
        str(args.prompt_tokens),
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


# --------------------------------------------------------------------------- verdict


def _report(**fields: object) -> None:
    """The machine-readable contract: one key per line, fixed names, nothing else."""
    for key, value in fields.items():
        if value is not None:
            print(f"{key.upper()}: {value}")


def _halt(fields: Dict[str, Any], why: str) -> None:
    """Refuse in the contract -- the keys, the prose, DONE, exit 1 -- and print no result."""
    _report(**fields)
    print(f"  {why}")
    print("DONE")
    raise SystemExit(1)


def _bound_module():
    """scripts/bound_candidates.py, loaded by path: the routing stage owns its own reader."""
    import importlib.util

    path = Path(__file__).with_name("bound_candidates.py")
    spec = importlib.util.spec_from_file_location("bound_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through here
    spec.loader.exec_module(module)
    return module


ARM_FAILED = "ARM_FAILED"
TOKENS_DIFFER = "TOKENS_DIFFER"
TOKENS_UNSTABLE = "TOKENS_UNSTABLE"
TOKENS_UNCOMPARED = "TOKENS_UNCOMPARED"
NOT_SUBSTITUTED = "NOT_SUBSTITUTED"
MECHANISM_MISMATCH = "MECHANISM_MISMATCH"
UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"


def routing_precheck(args: argparse.Namespace, bc) -> Dict[str, Any]:
    """The routing fields for the contract, or ``bc.RoutingRefused``. Pure: no vLLM, no GPU.

    Runs before any arm is launched. Checks, in order: the routing exists and is still about
    the discovery it names; that discovery was of the model being measured; the mechanism
    named is measured in the mode that isolates it (an apply() mechanism with apply() in the
    path, anything else with `--plain-arm`); and every named candidate is an ACCEPT row for
    that mechanism. A serving number for a pair the routing rejected is a number for
    something the pipeline already priced out.
    """
    if not (args.bound and args.mechanism and args.candidate):
        raise bc.RoutingRefused(
            {"routing": bc.ROUTING_UNEVALUATED, "verdict": bc.VERDICT_ROUTING_REJECTED},
            "--bound, --mechanism and --candidate go together: the routing accepts a "
            "mechanism per candidate, and a serving run is of one such pair (or several "
            "for one mechanism). Name all three, or none to run unrouted (ROUTING: UNCHECKED).",
        )
    bound = bc.load_bound(args.bound)
    bc.check_bound_provenance(bound)
    model = bound.get("model")
    if model and model != args.model:
        raise bc.RoutingRefused(
            {
                "routing": bc.ROUTING_STALE,
                "routed_model": model,
                "run_id": bound.get("run_id"),
                "verdict": bc.VERDICT_STALE_INPUT,
            },
            f"the routing in {bound['_path']} was computed from a discovery of {model}; "
            f"this run is of {args.model}. Worth and ceilings are that model's. Run "
            "discovery and bounding on this model first.",
        )
    apply_mechanism = args.mechanism in bc.APPLY
    if args.mechanism in bc.MECHANISMS and apply_mechanism == bool(args.plain_arm):
        mode = "plain" if args.plain_arm else "apply"
        raise bc.RoutingRefused(
            {
                "routing": bc.ROUTING_REJECTED,
                "mechanism": args.mechanism,
                "mode": mode,
                "verdict": MECHANISM_MISMATCH,
            },
            (
                f"{args.mechanism} is delivered through apply(), which --plain-arm removes "
                "from both arms; drop --plain-arm."
                if apply_mechanism
                else f"{args.mechanism} puts nothing in the call path, and the routing priced "
                "it at nothing; measuring it with apply() in the path measures apply()'s "
                "dispatch and substitutions with it. Pass --plain-arm and switch the change "
                "on with --env."
            ),
        )
    candidates = bound.get("candidates") or []
    accepted: List[Dict[str, Any]] = []
    for name in args.candidate:
        ids = [c["candidate_id"] for c in candidates if name in (c["candidate_id"], c["op"])]
        if not ids:
            raise bc.RoutingRefused(
                {
                    "routing": bc.ROUTING_UNEVALUATED,
                    "candidate": name,
                    "mechanism": args.mechanism,
                    "run_id": bound.get("run_id"),
                    "verdict": bc.VERDICT_ROUTING_REJECTED,
                },
                f"run {bound.get('run_id')} has no candidate {name!r} (neither a candidate_id "
                "nor an op it priced). An op with no device time in discovery was never a "
                "worklist row.",
            )
        accepted.extend(bc.require_routed(bound, cid, args.mechanism) for cid in ids)
    return {
        "routing": bc.ROUTING_OK,
        "candidate": " ".join(a["candidate"] for a in accepted),
        "mechanism": args.mechanism,
        "ceiling_us": " ".join(str(a["ceiling_us"]) for a in accepted),
        "run_id": bound.get("run_id"),
    }


def judge(
    runs: Mapping[str, List[Dict[str, Any]]], arms: List[str], plain_arm: bool, env: bool = True
) -> Dict[str, Any]:
    """Whether the arms can be compared, and the comparison only when they can.

    Pure, so it can be checked without vLLM. Gates in the order a reader would need them:
    every arm produced a throughput; every arm generated one and the same set of tokens
    across its repeats; the arms generated the same tokens as each other; and something was
    substituted (apply mode: the counters exist and at least one call was applied; plain
    mode with `--env` and counters present: likewise). Only past all of those are the
    medians, the delta and the noise computed, and `valid` set. A failed gate leaves
    `delta` None: there is no partial result to carry.
    """
    rates = {
        arm: sorted(r["tokens_per_sec"] for r in rs if r.get("tokens_per_sec"))
        for arm, rs in runs.items()
    }
    digests = {
        arm: sorted({r["token_digest"] for r in rs if r.get("token_digest")})
        for arm, rs in runs.items()
    }
    ours = runs["ours"][-1] if runs.get("ours") else {}
    dispatch = ours.get("dispatch") or {}
    out: Dict[str, Any] = {
        "valid": False,
        "verdict": None,
        "reason": None,
        "tokens": None,
        "rates": rates,
        "digests": digests,
        "dispatch": dispatch,
        "detail": ours.get("detail") or [],
        "delta": None,
        "noise": None,
        "baseline_tok_s": None,
        "ours_tok_s": None,
        "dispatch_tax": None,
        "kernels_delta": None,
    }

    failed = [a for a in arms if not rates.get(a)]
    if failed:
        out["verdict"] = ARM_FAILED
        out["reason"] = f"arm(s) {', '.join(failed)} produced no throughput"
        return out

    unstable = [a for a in arms if len(digests.get(a) or []) > 1]
    missing = [a for a in arms if not digests.get(a)]
    if unstable:
        out["tokens"], out["verdict"] = "UNSTABLE", TOKENS_UNSTABLE
        out["reason"] = (
            f"arm(s) {', '.join(unstable)} generated different tokens across their own "
            "repeats under greedy decoding; no delta can be read against a moving target"
        )
    elif missing:
        out["tokens"], out["verdict"] = "UNCOMPARED", TOKENS_UNCOMPARED
        out["reason"] = (
            f"arm(s) {', '.join(missing)} reported no token digest, so the arms cannot be "
            "shown to have done the same work"
        )
    elif len({d for a in arms for d in digests[a]}) > 1:
        out["tokens"], out["verdict"] = "DIFFER", TOKENS_DIFFER
        out["reason"] = (
            "the arms did not generate the same tokens under greedy decoding: the faster arm "
            "was faster at a different computation, and a delta between them includes a "
            "correctness change"
        )
    else:
        out["tokens"] = "IDENTICAL"
    if out["verdict"]:
        return out

    if not plain_arm and not dispatch:
        out["verdict"] = NOT_SUBSTITUTED
        out["reason"] = (
            "no dispatch counters in the patched arm: the patch did not install, and both "
            "arms ran the stock kernels"
        )
        return out
    if dispatch and (env or not plain_arm) and not any(a for _, a in dispatch.values()):
        out["verdict"] = NOT_SUBSTITUTED
        out["reason"] = (
            "0 applied in every family: nothing was substituted, so the arms differ only by "
            "the cost of being in the path (an A/A, whatever the delta says)"
        )
        return out

    b, o = _median(rates["baseline"]), _median(rates["ours"])
    delta = o / b - 1.0
    # A delta inside the arms' own scatter is not a result.
    noise = max((max(v) - min(v)) / _median(v) for v in rates.values() if len(v) > 1)
    out.update(baseline_tok_s=b, ours_tok_s=o, delta=delta, noise=noise)
    if rates.get("overhead"):
        h = _median(rates["overhead"])
        out["dispatch_tax"] = h / b - 1.0
        out["kernels_delta"] = o / h - 1.0
    out["verdict"] = "NOISE" if abs(delta) <= noise else ("WIN" if delta > 0 else "LOSS")
    out["valid"] = True
    return out


def _pct(fraction: Optional[float]) -> Optional[str]:
    return None if fraction is None else f"{fraction * 100:+.2f}"


def _substitution(outcome: Dict[str, Any], plain_arm: bool) -> str:
    dispatch = outcome["dispatch"]
    if not dispatch:
        return "n/a (plain arm, no counters)" if plain_arm else "NONE"
    calls = sum(c for c, _ in dispatch.values())
    applied = sum(a for _, a in dispatch.values())
    return f"{applied}/{calls} applied across {len(dispatch)} family(ies)"


def _print_counters(outcome: Dict[str, Any]) -> None:
    for family, (calls, applied) in sorted(outcome["dispatch"].items()):
        pct = 100.0 * applied / calls if calls else 0.0
        note = "" if applied else "   <- never applied; this family contributed nothing"
        print(f"    {family:20} {applied}/{calls} applied ({pct:.1f}%){note}")
    for d in outcome["detail"]:
        print(f"    detail: {d}")


def _print_rates(outcome: Dict[str, Any], arms: List[str], prefix: str = "") -> None:
    for label in arms:
        got = outcome["rates"].get(label) or []
        if not got:
            continue
        med = _median(got)
        spread = (max(got) - min(got)) / med * 100 if len(got) > 1 else 0.0
        print(f"  {prefix}{label:9} {med:>8.2f} tok/s  [n={len(got)}, spread {spread:.1f}%]")
    if outcome["delta"] is not None:
        print(f"  {prefix}delta     {outcome['delta'] * 100:+.2f}%")
    elif outcome["rates"].get("baseline") and outcome["rates"].get("ours"):
        b, o = _median(outcome["rates"]["baseline"]), _median(outcome["rates"]["ours"])
        print(f"  {prefix}delta     {(o / b - 1.0) * 100:+.2f}%")
    if outcome["dispatch_tax"] is not None:
        print(
            f"  {prefix}dispatch  {outcome['dispatch_tax'] * 100:+.2f}%  (patched, nothing matchable)"
        )
        print(f"  {prefix}kernels   {outcome['kernels_delta'] * 100:+.2f}%  (ours vs that)")


def render(
    outcome: Dict[str, Any],
    args: argparse.Namespace,
    arms: List[str],
    runs: Mapping[str, List[Dict[str, Any]]],
    routing: Dict[str, Any],
) -> None:
    """The contract, then either the result or the diagnostics -- never both."""
    plain = bool(args.plain_arm)
    head: Dict[str, Any] = {
        "model": args.model,
        "mode": "plain" if plain else "apply",
        **routing,
        "repeats": args.repeats,
        "arms": " ".join(arms),
    }
    if not outcome["valid"]:
        _report(
            **head,
            tokens=outcome["tokens"],
            substitution=_substitution(outcome, plain) if outcome["tokens"] else None,
            reason=outcome["reason"],
            verdict=outcome["verdict"],
        )
        print("--- diagnostics (no throughput is reported past a failed gate) ---")
        for label in arms:
            rs = runs.get(label) or []
            got = outcome["digests"].get(label) or []
            flag = "" if len(got) <= 1 else "   <- not stable across its own repeats"
            print(f"  {label:9} runs {len(rs)}, digest {' '.join(got) or 'n/a'}{flag}")
            err = next((r["error"] for r in rs if r.get("error")), None)
            if err:
                print(f"  {'':9} FAILED: {err.splitlines()[-1][:100]}")
                log = next((r.get("log") for r in rs if r.get("log")), None)
                if log:
                    print(f"  {'':9} full log: {log}")
        if plain:
            diff = _env_overrides(args)
            print(
                "  arms differ by: "
                + (", ".join(f"{k}={v}" for k, v in diff.items()) or "nothing (A/A)")
            )
        if outcome["dispatch"]:
            print("  counters (patched arm):")
            _print_counters(outcome)
        if args.report_unvalidated:
            print(
                "--- UNVALIDATED throughput: the gate above failed; this is not a result "
                "and must not be quoted as one ---"
            )
            _print_rates(outcome, arms, prefix="UNVALIDATED ")
        print("DONE")
        return

    _report(
        **head,
        baseline_tok_s=f"{outcome['baseline_tok_s']:.2f}",
        ours_tok_s=f"{outcome['ours_tok_s']:.2f}",
        delta_pct=_pct(outcome["delta"]),
        spread_pct=f"{outcome['noise'] * 100:.1f}",
        dispatch_pct=_pct(outcome["dispatch_tax"]),
        kernels_pct=_pct(outcome["kernels_delta"]),
        tokens="IDENTICAL",
        substitution=_substitution(outcome, plain),
        verdict=outcome["verdict"],
    )
    _print_rates(outcome, arms)
    if outcome["verdict"] == "NOISE":
        print(
            f"  NOTE: the delta is within the arms' run-to-run spread ({outcome['noise'] * 100:.1f}%) "
            "and is not distinguishable from noise; widen the window or add repeats."
        )
    if plain:
        diff = _env_overrides(args)
        print(
            "  arms differ by: "
            + (", ".join(f"{k}={v}" for k, v in diff.items()) or "nothing (A/A: noise floor)")
        )
    if outcome["dispatch"]:
        print("  substitution:" if not plain else "  counters (patched arm):")
        _print_counters(outcome)
    print("DONE")


# --------------------------------------------------------------------------- cli


def _at_least(floor: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < floor:
            raise argparse.ArgumentTypeError(f"must be at least {floor}, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--dataset", default="tmp/flashinfer-trace", help="Dataset apply() draws solutions from."
    )
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help=(
            "Approximate tokens per prompt. Zero keeps the short prompt, which makes a run "
            "almost entirely decode; raise it to measure a change that acts at prefill."
        ),
    )
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument(
        "--repeats",
        type=_at_least(2),
        default=3,
        help="Times to run each arm, at least 2. Arms alternate, and the median is reported "
        "against the run-to-run spread; a single run has no spread, so it cannot separate a "
        "kernel effect from drift and is refused.",
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
    routing = ap.add_argument_group(
        "routing",
        "Tie the run to a row of scripts/bound_candidates.py. All three together; the run "
        "then refuses -- before launching anything -- a mechanism the routing rejected for "
        "the candidate, a routing computed from a discovery that has since changed or of "
        "another model, or a mechanism measured in the wrong mode. Without them the "
        "contract says ROUTING: UNCHECKED.",
    )
    routing.add_argument("--bound", help="bound.json, or the directory holding it.")
    routing.add_argument(
        "--mechanism",
        help="The mechanism under test, as bound.log names it (apply_substitution, "
        "fusion_apply, provider_patch, library_call, ...).",
    )
    routing.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate_id from the worklist, or an op name (every candidate of that op). "
        "Repeatable; each must be an ACCEPT row for --mechanism.",
    )
    ap.add_argument(
        "--report-unvalidated",
        action="store_true",
        help="UNSAFE. When the token gate fails, also print the per-arm throughput and delta, "
        "each line marked UNVALIDATED. The verdict stays the failed gate's and the exit "
        "stays non-zero: the arms did different work, so the delta is not a kernel "
        "comparison. For investigating a kernel whose numerics are expected to move the "
        "tokens, never for a result.",
    )
    ap.add_argument("--json", type=Path, help="Also write the result here.")
    ap.add_argument(_CHILD, dest="child", action="store_true", help=argparse.SUPPRESS)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.child:
        _bench(
            args.model,
            args.prompts,
            args.out_tokens,
            args.gpu_util,
            args.max_model_len,
            args.prompt_tokens,
        )
        return

    mode = "plain" if args.plain_arm else "apply"
    # Everything that can refuse without a GPU refuses first, so a rejected pair costs no
    # serving time and no partial output.
    routing: Dict[str, Any] = {"routing": "UNCHECKED"}
    if args.bound or args.mechanism or args.candidate:
        bc = _bound_module()
        try:
            routing = routing_precheck(args, bc)
        except bc.RoutingRefused as exc:
            _halt({"model": args.model, "mode": mode, **exc.fields}, exc.why)

    if _supported(args.model) is False:
        _halt(
            {"model": args.model, "mode": mode, **routing, "verdict": UNSUPPORTED_MODEL},
            f"vLLM does not register an architecture for {args.model}, so there is no "
            "serving number to measure. Per-kernel results are all this model can give.",
        )

    # Arms alternate rather than running all baselines then all patched runs, so that
    # any thermal or clock drift over the session falls on both equally.
    arms = ["baseline", "ours"] + (["overhead"] if args.overhead_arm else [])
    runs: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    with tempfile.TemporaryDirectory(prefix="fib-empty-") as empty:
        for _ in range(args.repeats):
            runs["baseline"].append(_run_arm(args, patched=False))
            runs["ours"].append(_run_arm(args, patched=True))
            if args.overhead_arm:
                runs["overhead"].append(_run_arm(args, patched=True, empty_dataset=empty))

    outcome = judge(runs, arms, args.plain_arm, env=bool(args.env))
    render(outcome, args, arms, runs, routing)

    if args.json:
        record = {
            "model": args.model,
            "mode": mode,
            "routing": routing,
            "valid": outcome["valid"],
            "verdict": outcome["verdict"],
            "reason": outcome["reason"],
            "tokens": outcome["tokens"],
            "delta": outcome["delta"],
            "noise": outcome["noise"],
            "rates": outcome["rates"] if outcome["valid"] else None,
            "digests": outcome["digests"],
            "dispatch": outcome["dispatch"],
            "runs": runs,
        }
        if not outcome["valid"] and args.report_unvalidated:
            record["unvalidated"] = {"rates": outcome["rates"]}
        args.json.write_text(json.dumps(record, indent=2, default=str) + "\n")
    raise SystemExit(0 if outcome["valid"] else 1)


if __name__ == "__main__":
    main()
