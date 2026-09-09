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
    kernel-trials benchmark <name> <candidate.py> [--trial t3]
    kernel-trials ab        <harness.py> [--env-a K=V]... [--env-b K=V]... [--harness-b other.py]
    kernel-trials status    <name>
    kernel-trials best      <name>
    kernel-trials finalize  <name> <output.py> [--no-require-win]

`benchmark` is the only sanctioned way to get a number. It gates on correctness before it
times anything, warms every arm before timing any of them, and alternates the arms in
interleaved rounds -- a GPU that has been idle ramps its clocks, so a sequential sweep
charges the ramp to whichever case runs first and can invert the comparison outright.

`ab` is `benchmark` for the case one process cannot hold: two builds of the same
`torch.ops` symbol. Each arm runs in its own process with its own environment, the
processes alternate round by round with the order flipped each round, and each flipped
pair of rounds is one paired observation, judged by the same rule and reported through the
same contract.

`finalize` refuses to copy a best trial that is not a measured win. A slower kernel must
never pass, and "measured" means the difference between the arms clears its own round-to-
round scatter: the arms are timed in pairs, and it is the scatter of the paired ratio that
is judged, not the scatter of either arm alone.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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


def _ensure_toolchain_on_path() -> None:
    """Put this interpreter's bin/ on PATH before any build runs.

    Invoking a venv's python by absolute path leaves its bin/ off PATH, so the SYCL builder
    cannot find `ninja` and every SYCL candidate fails to compile -- which reads as "the
    kernel is broken" rather than "the toolchain was not visible". Cheap to prevent, and it
    has cost real debugging time here more than once.
    """
    bindir = str(pathlib.Path(sys.executable).parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bindir not in parts:
        os.environ["PATH"] = os.pathsep.join([bindir, *parts])


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
    """Wait for whichever accelerator the harness actually used.

    Only an initialised backend is synchronised: a CPU harness must not spin up a GPU
    context on the side, and an uninitialised device has nothing in flight to wait for.
    """
    import torch

    for backend in (getattr(torch, "xpu", None), getattr(torch, "cuda", None)):
        if backend is not None and backend.is_available() and backend.is_initialized():
            backend.synchronize()
            return


def _seed(seed: int) -> None:
    """Seed every generator a harness's `get_inputs()` might draw from."""
    import torch

    torch.manual_seed(seed)
    for backend in (getattr(torch, "xpu", None), getattr(torch, "cuda", None)):
        if backend is not None and backend.is_available():
            backend.manual_seed_all(seed)


def _time_one(fn, calls: int) -> float:
    """Microseconds per call over `calls` calls inside one timed region."""
    _sync()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    _sync()
    return (time.perf_counter() - start) / calls * 1e6


# --------------------------------------------------------------------------- build output

_BUILD_LOGGER = "tvm_ffi.cpp.extension"
_BUILD_LOG_ENV = "TVM_FFI_CPP_EXTENSION_LOG_BUILD"
_BUILD_LOG_KEEP = 64 * 1024


@contextlib.contextmanager
def _capture_build_output(into: List[str]) -> Iterator[None]:
    """Collect what the compilers said while the body ran, appending it to `into`.

    Spill is reported by the compiler, at build or at first launch, and a gate that reads
    it for free on every trial has to be listening on every channel it can arrive by:

    * ``tvm_ffi.cpp.build`` runs ninja with its output captured and drops it on success
      unless ``TVM_FFI_CPP_EXTENSION_LOG_BUILD`` is set, in which case it logs it. An AOT
      SYCL build's ``spilled around N`` warning travels this way and nowhere else, so the
      variable is set and a collecting handler attached for the duration.
    * Native code writes to file descriptors 1 and 2 directly, below ``sys.stdout``: a
      driver JIT at first launch, or a compiler subprocess that inherits the descriptors
      (torch's ``cpp_extension``, ocloc under Triton). Those descriptors are redirected to
      a file for the duration.

    Nothing is echoed while the body runs; the caller decides what to show.
    """
    records: List[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    collector = _Collect(level=logging.INFO)
    build_logger = logging.getLogger(_BUILD_LOGGER)
    prior_level = build_logger.level
    if build_logger.getEffectiveLevel() > logging.INFO:
        build_logger.setLevel(logging.INFO)
    build_logger.addHandler(collector)
    prior_env = os.environ.get(_BUILD_LOG_ENV)
    os.environ[_BUILD_LOG_ENV] = "1"

    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    fd_text = ""
    try:
        with tempfile.TemporaryFile(mode="w+b") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                sink.seek(0)
                fd_text = sink.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved[0])
        os.close(saved[1])
        build_logger.removeHandler(collector)
        build_logger.setLevel(prior_level)
        if prior_env is None:
            os.environ.pop(_BUILD_LOG_ENV, None)
        else:
            os.environ[_BUILD_LOG_ENV] = prior_env
        into.append(fd_text)
        into.extend(records)


_SPILL_RE = re.compile(r"spilled around (\d+)|Spill Memory Per Thread\s*:?\s*(\d+)")

# ninja -v prints one `[k/n] <command>` line per step it actually ran. A cached build prints
# "ninja: no work to do." instead, and the compiler -- which is what reports spill -- never
# ran, so its silence says nothing.
_COMPILED_RE = re.compile(r"^\[\d+/\d+\] ", re.MULTILINE)


def _spill_from(text: str) -> Optional[int]:
    """Register spill as the compiler reported it, or None if it said nothing.

    The build log already knows. Reading it here means every trial is checked for free,
    instead of spill being something you discover later with a profiler after wondering why
    a correct kernel is many times too slow.
    """
    m = _SPILL_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def _spill_state(build_log: Optional[str]) -> Union[int, str]:
    """Three states, because two hide the failure the gate exists for.

    A number is what the compiler reported. ``none`` means a compiler ran in view and said
    nothing about spill. ``unknown`` means nothing was checked: no compiler ran (a plain
    PyTorch harness, or a build ninja found cached), or the build produced SPIR-V for the
    driver to finish at first launch, which allocates registers silently. Reporting
    ``none`` in those cases is the gate certifying what it never looked at.
    """
    text = build_log or ""
    reported = _spill_from(text)
    if reported is not None:
        return reported if reported > 0 else "none"
    if not _COMPILED_RE.search(text):
        return "unknown"
    if "-fsycl" in text and "spir64_gen" not in text:
        return "unknown"
    return "none"


def _trim_log(parts: Sequence[str], keep: int = _BUILD_LOG_KEEP) -> str:
    """Bound what a trial stores, without ever dropping the lines the gate reads."""
    text = "\n".join(p for p in parts if p).strip()
    if len(text) <= keep:
        return text
    tail = text[-keep:]
    pinned = [ln for ln in text[:-keep].splitlines() if _SPILL_RE.search(ln)]
    head = "\n".join(pinned) + "\n" if pinned else ""
    return f"{head}...[{len(text) - keep} chars truncated]...\n{tail}"


# --------------------------------------------------------------------------- correctness


def _observe(model, args: list) -> Dict[str, Any]:
    """Run once on `args` and return what there is to compare afterwards.

    The kernels worth tuning are destination-passing: they return nothing useful and write
    into an argument. Running both arms over one set of tensors therefore compares a buffer
    with itself -- the candidate reads what the baseline just wrote, both return the same
    object, and the gate reports zero error no matter what the candidate computed. Each arm
    must be given its own copies, and the comparison includes whatever the op wrote into
    its arguments.
    """
    import torch

    returned = model(*args)
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    in_args = returned is None or (
        isinstance(returned, torch.Tensor)
        and returned.data_ptr() in {t.data_ptr() for t in tensors}
    )
    if isinstance(returned, torch.Tensor):
        kind, primary = "tensor", returned
    elif isinstance(returned, (tuple, list)) and returned and isinstance(returned[0], torch.Tensor):
        kind, primary = "sequence", returned[0]
    else:
        kind, primary = type(returned).__name__, None
    return {"in_args": in_args, "kind": kind, "returned": primary, "args": tensors}


def _to_host(obs: Dict[str, Any]) -> Dict[str, Any]:
    """An observation as plain CPU float tensors, so it survives a process boundary."""

    def host(t):
        return None if t is None else t.detach().float().cpu()

    return {
        "in_args": obs["in_args"],
        "kind": obs["kind"],
        "returned": host(obs["returned"]),
        "args": [host(t) for t in obs["args"]],
    }


def _compare(
    expected: Dict[str, Any], actual: Dict[str, Any], atol: float, rtol: float
) -> Tuple[Optional[str], float]:
    """(reason the candidate fails, or None; max abs error) -- the one rule every arm meets."""
    import torch

    if expected["in_args"]:
        # The result is in the arguments; compare those, pairwise, instead of the return
        # value. Pairwise because the arguments of one op need not share a size -- an
        # attention call's query and its KV differ by the head ratio.
        if len(expected["args"]) != len(actual["args"]):
            return "candidate took a different number of tensor arguments", float("inf")
        worst = 0.0
        for i, (b, c) in enumerate(zip(expected["args"], actual["args"])):
            if b.shape != c.shape:
                return f"argument {i} has shape {tuple(c.shape)} != {tuple(b.shape)}", float("inf")
            err = (b.float() - c.float()).abs().max().item() if b.numel() else 0.0
            worst = max(worst, err)
            if not torch.allclose(b.float(), c.float(), atol=atol, rtol=rtol):
                return f"argument {i} differs after the call (max abs {err:.5g})", err
        if not all(torch.isfinite(c).all() for c in actual["args"]):
            return "candidate produced non-finite values", worst
        return None, worst

    if expected["returned"] is None:
        raise SystemExit(
            f"the baseline returned {expected['kind']}, which is nothing to compare against; "
            "a harness must return a tensor (or a tuple whose first item is one) or write "
            "its result into an argument."
        )
    if expected["kind"] != actual["kind"] or actual["returned"] is None:
        return "candidate returned a different type", float("inf")
    e, a = expected["returned"], actual["returned"]
    if e.shape != a.shape:
        return f"shape {tuple(a.shape)} != {tuple(e.shape)}", float("inf")
    err = (a.float() - e.float()).abs().max().item()
    tol = atol + rtol * e.float().abs().max().item()
    if not torch.isfinite(a).all():
        return "candidate produced non-finite values", err
    if err > tol:
        return f"max abs error {err:.5g} > {tol:.5g}", err
    return None, err


def _input_digest(inputs: list) -> str:
    """A fingerprint of the inputs, so two processes can prove they saw the same problem."""
    import torch

    h = hashlib.sha256()
    for item in inputs:
        if isinstance(item, torch.Tensor):
            flat = item.detach().flatten().contiguous().cpu()
            h.update(f"{item.dtype}{tuple(item.shape)}".encode())
            try:
                h.update(flat.view(torch.uint8).numpy().tobytes())
            except Exception:
                h.update(bytes(flat.view(torch.uint8).tolist()))
        else:
            h.update(repr(item).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- in-process


def benchmark(
    baseline_path: str,
    candidate_path: str,
    rounds: int,
    calls: int,
    atol: float,
    rtol: float,
    build_log: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Correctness first, then interleaved timing of both arms.

    `build_log`, when given, receives the compiler output captured while the candidate
    loaded and ran once, so a caller still has it if the load raises.
    """
    import torch

    if rounds < MIN_PAIRED_ROUNDS:
        raise ValueError(
            f"{rounds} round(s) cannot estimate a spread; need at least {MIN_PAIRED_ROUNDS}"
        )
    log: List[str] = [] if build_log is None else build_log
    base_model, inputs = _load_model(baseline_path)
    with _capture_build_output(log):
        cand_model, _ = _load_model(candidate_path)

    def _fresh():
        return [a.clone() if isinstance(a, torch.Tensor) else a for a in inputs]

    base_args, cand_args = _fresh(), _fresh()
    with torch.no_grad():
        expected = _observe(base_model, base_args)
        with _capture_build_output(log):
            actual = _observe(cand_model, cand_args)

    captured = _trim_log(log)
    reason, err = _compare(expected, actual, atol, rtol)
    if reason:
        return {"correctness": "fail", "reason": reason, "build_log": captured}

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

    return {
        **_summarize(base_samples, cand_samples),
        "correctness": "pass",
        "max_abs_error": err,
        "rounds": rounds,
        "calls_per_round": calls,
        "build_log": captured,
    }


MIN_PAIRED_ROUNDS = 5
"""Fewest paired observations a verdict is made from.

The spread is estimated from the observations themselves, and with three or four of them a
median absolute deviation is one or two gaps between neighbours: in simulation, identical
arms then read as a WIN or a LOSS about one time in four. Five brings that under one in
ten, seven under one in twenty-five, eleven under one in a hundred. Five is the floor and
the defaults sit above it; more rounds only ever make the gate safer.
"""

NOISE_SIGMAS = 2.0
"""How many robust sigmas of the per-observation ratio the median must clear to count.

One sigma is the natural reading of "a difference inside the scatter", and it is not
enough: the scatter is itself estimated from the same few rounds and is often under-read,
so at one sigma identical arms came through as a WIN or a LOSS one time in seven at seven
rounds, in simulation and on this machine under load. Two sigmas holds that to a few
percent at seven rounds and under one percent at eleven. It costs power against small
gains, which is the right side to err on for a gate whose job is to keep a slower or an
unmeasured kernel out. A threshold scaled by the standard error of the median was tried
and rejected: it tightens with the count exactly where the scatter estimate is weakest.
"""

_MAD_TO_SIGMA = 1.4826
"""Scales a median absolute deviation to the standard deviation it estimates for normal
scatter, so the reported spread reads as one sigma of the per-round ratio."""

PAIRED_SPREAD_KIND = "paired"
"""Marker a result carries when its `spread` is the paired statistic below. A stored result
without it was judged by an earlier rule on one arm's scatter and is not re-scored."""


def _robust_sigma(values: Sequence[float]) -> Tuple[float, float]:
    """Median and a robust one-sigma scatter (scaled median absolute deviation).

    Not max-minus-min: a range is set by its single worst sample and grows with the number
    of samples, so a gate built on it tightened or loosened with `--rounds` alone. The MAD
    is set by the bulk of the rounds and estimates the same quantity at any count.
    """
    center = statistics.median(values)
    mad = statistics.median(abs(v - center) for v in values)
    return center, _MAD_TO_SIGMA * mad


def _summarize(base_samples: Sequence[float], cand_samples: Sequence[float]) -> Dict[str, Any]:
    """The paired comparison a verdict is made of.

    Sample k of each arm was timed in the same interleaved round, so the two saw the same
    moment of the machine. The quantity judged is the per-round log-ratio
    ``ln(base_k / cand_k)``: its median is the speedup (exactly the median of the per-round
    ratios), and its scatter across rounds is the uncertainty of the *difference between the
    arms* -- the only number a verdict about that difference can rest on. Either arm's own
    scatter is beside the point: a noisy baseline against a stable candidate yields a
    difference that is pure noise, and the candidate's range reported it as a WIN.

    `spread` is one robust sigma of the per-round ratio as a fraction (``exp(sigma) - 1``),
    in log space so that a slowdown and a speedup of the same factor carry the same weight;
    `noise_floor` is the smallest speedup or slowdown that clears :data:`NOISE_SIGMAS` of
    it. The per-arm spreads are kept for diagnosis -- which arm is unsteady -- and play no
    part in the verdict.
    """
    if len(base_samples) != len(cand_samples):
        raise ValueError(
            f"arms are not paired: {len(base_samples)} baseline vs {len(cand_samples)} "
            "candidate samples"
        )
    if len(base_samples) < MIN_PAIRED_ROUNDS:
        raise ValueError(
            f"{len(base_samples)} paired round(s) cannot estimate a spread; need at least "
            f"{MIN_PAIRED_ROUNDS}"
        )
    if min(*base_samples, *cand_samples) <= 0:
        raise ValueError("a timed region measured no time at all; raise --calls")
    log_ratios = [math.log(b / c) for b, c in zip(base_samples, cand_samples)]
    center, sigma = _robust_sigma(log_ratios)
    base_us, base_sigma = _robust_sigma(base_samples)
    cand_us, cand_sigma = _robust_sigma(cand_samples)
    return {
        "baseline_us": base_us,
        "candidate_us": cand_us,
        "speedup": math.exp(center),
        "spread": math.exp(sigma) - 1,
        "noise_floor": math.exp(NOISE_SIGMAS * sigma) - 1,
        "spread_kind": PAIRED_SPREAD_KIND,
        "baseline_spread": base_sigma / base_us,
        "candidate_spread": cand_sigma / cand_us,
        "pairs": len(log_ratios),
        "baseline_samples": list(base_samples),
        "candidate_samples": list(cand_samples),
    }


def _verdict(result: Dict[str, Any]) -> str:
    """WIN, LOSS or NOISE, by the one rule used everywhere a result is judged.

    Only a *difference* inside the paired scatter is unmeasured: NOISE when the speedup, in
    whichever direction it points, is within :data:`NOISE_SIGMAS` robust sigmas of the
    per-round ratio of 1 -- the paired `spread`, never either arm's own.
    The comparison is on the factor ``max(s, 1/s)`` rather than the signed gain ``s - 1``:
    the signed form compresses every regression into (-1, 0) while a gain is unbounded, so
    a slowdown was excused as noise where the same-sized speedup was a WIN. Comparing the
    signed gain against the spread once reported a gross regression as "inside noise",
    which is the opposite of what the check is for -- a regression that clears the scatter
    is a LOSS, however the spread was arrived at.

    A result without the paired marker was scored by the earlier one-arm rule and is not a
    measured anything; it is NOISE until `benchmark --trial` re-measures it.
    """
    if result.get("spread_kind") != PAIRED_SPREAD_KIND:
        return "NOISE"
    speedup = result["speedup"]
    if speedup <= 0:
        return "LOSS"
    if math.log(max(speedup, 1 / speedup)) <= NOISE_SIGMAS * math.log1p(result["spread"]):
        return "NOISE"
    return "WIN" if speedup > 1 else "LOSS"


def _report(**fields: object) -> None:
    """Emit the machine-readable contract a driving agent parses.

    One key per line, fixed names, no prose. An agent looping over this reads the keys and
    decides; it never has to interpret a sentence, and a changed adjective cannot change
    what it concludes.
    """
    for key, value in fields.items():
        if value is not None:
            print(f"{key.upper()}: {value}")


def _report_result(result: Dict[str, Any], spills: Union[int, str], **extra: object) -> str:
    """The passing-result contract, shared by every command that produces a number."""
    verdict = _verdict(result)
    _report(
        build="OK",
        spills=spills,
        **extra,
        correct="OK",
        max_abs_error=f"{result['max_abs_error']:.5g}",
        baseline_us=f"{result['baseline_us']:.2f}",
        candidate_us=f"{result['candidate_us']:.2f}",
        speedup=f"{result['speedup']:.3f}",
        baseline_spread_pct=_pct(result.get("baseline_spread")),
        candidate_spread_pct=_pct(result.get("candidate_spread")),
        spread_pct=_pct(result["spread"]),
        noise_floor_pct=_pct(result.get("noise_floor")),
        verdict=verdict,
    )
    if isinstance(spills, int):
        print(
            "  NOTE: the compiler spilled registers. Spill produces a correct kernel that "
            "is many times too slow, so fix it before reading the timing above."
        )
    elif spills == "unknown":
        print(
            "  NOTE: spill was not checked -- no compiler ran in view (plain PyTorch, a cached "
            "build, or a SPIR-V JIT). unitrace's Kernel Properties is where to read it."
        )
    if verdict == "NOISE":
        print(
            "  NOTE: the difference between the arms is inside their paired round-to-round "
            f"scatter: SPREAD_PCT {result['spread'] * 100:.1f}% is one robust sigma of the "
            f"per-round ratio, and a speedup or slowdown of at least NOISE_FLOOR_PCT "
            f"{result['noise_floor'] * 100:.1f}% ({NOISE_SIGMAS:g} sigmas) would have "
            "counted. Raise --calls to steady each round and --rounds to steady the "
            "estimate before believing it in either direction. BASELINE_SPREAD_PCT and "
            "CANDIDATE_SPREAD_PCT say which arm is unsteady; neither enters the verdict."
        )
    return verdict


def _pct(fraction: Optional[float]) -> Optional[str]:
    return None if fraction is None else f"{fraction * 100:.1f}"


# --------------------------------------------------------------------------- routing


def _bound_module():
    """scripts/bound_candidates.py, loaded by path: the routing stage owns its own reader."""
    import importlib.util

    path = pathlib.Path(__file__).with_name("bound_candidates.py")
    spec = importlib.util.spec_from_file_location("bound_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through here
    spec.loader.exec_module(module)
    return module


def _routing_or_exit(bound: str, mechanism: str, harness: str) -> Dict[str, Any]:
    """The ACCEPT row for (this harness's candidate, mechanism), or a refusal in the contract.

    A series on a pair the routing rejected would measure what the routing already priced
    out, and its numbers would be read as evidence about the pipeline. So the check runs
    before anything is timed -- at `init`, and again on every `benchmark`, because the
    routing can be recomputed under a running series and a re-run of discovery voids it.
    The refusal names the candidate, the mechanism, the gate that rejected it and the
    arithmetic, and prints no timing.
    """
    bc = _bound_module()
    try:
        row = bc.routed_from_harness(bound, mechanism, harness)
    except bc.RoutingRefused as exc:
        _report(**exc.fields)
        print(f"  {exc.why}")
        print("DONE")
        raise SystemExit(1)
    # A row can be genuine and still be priced against a device time this interpreter
    # cannot reproduce. Two interpreters here carry different builds of the accelerator
    # library, so a ceiling measured under one is not a target under the other -- and a
    # series chasing it looks like it is failing when it is simply aimed at the wrong
    # number. Warn rather than refuse: the trial's own paired measurement is still valid,
    # it is only the comparison against the ceiling that is void.
    try:
        drift = bc.check_measured_with(bc.load_bound(bound))
    except Exception:  # a routing this old records nothing to compare; not a failure
        drift = None
    if drift:
        _report(routing_toolchain="DIFFERS")
        print(f"  NOTE: {drift}")
        print(
            "  The series is still measured correctly against its own baseline; treat "
            "ceiling_us as unpriced here, and re-run the routing under this interpreter "
            "to compare against it."
        )
    return row


def _routing_flags(args) -> Optional[Dict[str, Any]]:
    """The (bound, mechanism) pair from the CLI, or None; half a pair is refused."""
    bound, mechanism = getattr(args, "bound", None), getattr(args, "mechanism", None)
    if bool(bound) != bool(mechanism):
        raise SystemExit(
            "--bound and --mechanism go together: the routing names a mechanism per "
            "candidate, and a mechanism without the routing that priced it is unchecked."
        )
    return {"bound": bound, "mechanism": mechanism} if bound else None


# --------------------------------------------------------------------------- commands


def cmd_init(args) -> None:
    flags = _routing_flags(args)
    data: Dict[str, Any] = {
        "name": args.name,
        "baseline": str(pathlib.Path(args.baseline).resolve()),
        "trials": [],
    }
    if flags:
        row = _routing_or_exit(flags["bound"], flags["mechanism"], args.baseline)
        data["routing"] = {
            "bound": _rel(flags["bound"]),
            "mechanism": flags["mechanism"],
            **{
                k: row.get(k) for k in ("candidate", "op", "shape", "run_id", "ceiling_us", "worth")
            },
        }
    _save_store(args.name, data)
    print(f"  initialised {args.name} with baseline {args.baseline}")
    if flags:
        r = data["routing"]
        print(
            f"  routed: {r['mechanism']} for {r['op']} at {r['shape']} "
            f"(ceiling_us {r['ceiling_us']}, worth {r['worth']}) from {r['bound']}"
        )


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
    # A series opened from the routing is held to it on every benchmark: the routing may
    # have been recomputed since, and a rejected pair is refused here rather than measured.
    routing = data.get("routing")
    if routing:
        _routing_or_exit(routing["bound"], routing["mechanism"], data["baseline"])
        extra: Dict[str, Any] = {"routing": "OK", "mechanism": routing["mechanism"]}
    else:
        extra = {"routing": "UNCHECKED"}
    if args.trial and not any(t["id"] == args.trial for t in data["trials"]):
        raise SystemExit(f"no trial {args.trial!r} in {args.name}")
    # A build or load failure is not an error to report and stop on -- it is the next input
    # to the loop. Emit it in the contract, with the compiler's own diagnostics, so the
    # agent's next iteration is a fix rather than a guess.
    captured: List[str] = []
    try:
        result = benchmark(
            data["baseline"],
            args.file,
            args.rounds,
            args.calls,
            args.atol,
            args.rtol,
            build_log=captured,
        )
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        log_text = _trim_log(captured)
        _report(
            build="FAILED",
            spills=_spill_from(text + "\n" + log_text) or "unknown",
            **extra,
            verdict="BUILD_FAILED",
        )
        print("--- build/load diagnostics ---")
        print(text)
        if log_text:
            print(log_text)
        print("DONE")
        raise SystemExit(1)
    if args.trial:
        for trial in data["trials"]:
            if trial["id"] == args.trial:
                trial["result"] = result
                break
        _save_store(args.name, data)

    spills = _spill_state(result.get("build_log"))
    if result["correctness"] != "pass":
        # Correctness is reported before any timing, and no timing is reported at all.
        # A number attached to a wrong kernel is the one output that can waste a whole
        # series, because it looks like progress.
        _report(
            build="OK",
            spills=spills,
            **extra,
            correct="FAILED",
            reason=result["reason"],
            verdict="INCORRECT",
        )
        print("DONE")
        raise SystemExit(1)

    _report_result(result, spills, **extra)
    print("DONE")


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
        print(
            f"\n  best: {best['id']} at {best['result']['speedup']:.2f}x "
            f"({_verdict(best['result'])})"
        )
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
    result = best["result"]
    verdict = _verdict(result)
    fields = {
        "best": best["id"],
        "speedup": f"{result['speedup']:.3f}",
        "spread_pct": _pct(result["spread"]),
        "noise_floor_pct": _pct(result.get("noise_floor")),
        "verdict": verdict,
    }
    if args.require_win and verdict != "WIN":
        # The best of a series that never beat its baseline is still slower than the
        # baseline, and a difference inside the spread was never measured. Copying either
        # to an output path is how a regression gets shipped with a straight face.
        _report(**fields, finalize="REFUSED")
        print(
            f"  {best['id']} is the best correct trial but not a measured win ({verdict}); "
            "nothing was written. Keep searching, or pass --no-require-win to archive it "
            "as a record of the attempt -- never to deploy it."
        )
        if result.get("spread_kind") != PAIRED_SPREAD_KIND:
            print(
                f"  {best['id']} was scored under an earlier rule that read only the "
                "candidate's scatter; re-run `benchmark --trial` to measure it."
            )
        print("DONE")
        raise SystemExit(1)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["file"], out)
    failure = _import_fails(out)
    if failure:
        # A trial file may import a helper that sits beside it in the series directory.
        # Copying it verbatim to a different directory leaves those imports pointing at
        # the wrong place, and the promoted winner then fails to load -- the one file in
        # the series that has to work. Carry the directory it was written against.
        series_dir = pathlib.Path(best["file"]).resolve().parent
        _prepend_search_path(out, series_dir, series_dir.parent)
        still = _import_fails(out)
        if still:
            out.unlink()
            _report(**fields, finalize="REFUSED", import_error=still)
            print(
                f"  {best['id']} does not load from {out.parent} even with the directories "
                f"it was written against on the path: {still}. Nothing was written; make "
                "the trial file self-contained and re-run finalize."
            )
            print("DONE")
            raise SystemExit(1)
    _report(**fields, finalize="OK", output=str(out))
    print("DONE")


def _import_fails(path: pathlib.Path) -> Optional[str]:
    """Import `path` in a fresh interpreter; return the failure, or None if it loaded.

    Asking the interpreter is the only reliable test. A trial file may reach for helpers
    through a bootstrap of its own, at any depth, so no static reading of its imports
    establishes whether it will load from a different directory.
    """
    probe = (
        "import importlib.util as u,sys;"
        "s=u.spec_from_file_location('promoted',sys.argv[1]);"
        "m=u.module_from_spec(s);s.loader.exec_module(m)"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe, str(path)], capture_output=True, text=True, timeout=300
    )
    if r.returncode == 0:
        return None
    err = r.stderr or ""
    # Distinguish "cannot be found" from "refused to configure itself". Only the first is
    # finalize's business: it guarantees the promoted file is reachable from where it was
    # written to, not that a caller has supplied its settings. A file that raises its own
    # configuration error has already loaded far enough to run that check.
    if not any(k in err for k in ("ModuleNotFoundError", "ImportError")):
        return None
    last = [ln for ln in err.strip().splitlines() if ln.strip()]
    return last[-1] if last else f"exit {r.returncode}"


def _prepend_search_path(path: pathlib.Path, *origins: pathlib.Path) -> None:
    """Make `path` import what it imported where it was written, without naming a machine.

    Each inserted directory is derived from the file's own location at run time, so the
    promoted file stays valid in any checkout rather than carrying one box's layout.
    """
    rels = []
    for origin in origins:
        try:
            rels.append(os.path.relpath(origin, path.parent))
        except ValueError:
            continue
    if not rels:
        return
    header = (
        "# Added by kernel_trials finalize: this trial imported helpers from the series\n"
        "# directory it was written in. Resolved relative to this file, never absolute.\n"
        "import pathlib as _p, sys as _s\n"
        f"for _r in {rels!r}:\n"
        "    _s.path.insert(0, str((_p.Path(__file__).resolve().parent / _r).resolve()))\n"
    )
    path.write_text(header + path.read_text())


# --------------------------------------------------------------------------- ab


class _ArmFailed(Exception):
    def __init__(self, arm: str, output: str, returncode: int):
        super().__init__(f"arm {arm} exited {returncode}")
        self.arm, self.output, self.returncode = arm, output, returncode


def _parse_env(items: Sequence[str], flag: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def _run_ab_arm(
    arm: str,
    harness: str,
    overrides: Dict[str, str],
    seed: int,
    calls: int,
    inner: int,
    out: pathlib.Path,
    timeout: int,
) -> Dict[str, Any]:
    """One process: load the harness under `overrides`, observe once, time `inner` regions."""
    env = dict(os.environ)
    env.update(overrides)
    cmd = [
        sys.executable,
        __file__,
        "_ab-arm",
        harness,
        "--seed",
        str(seed),
        "--calls",
        str(calls),
        "--inner-rounds",
        str(inner),
        "--out",
        str(out),
    ]
    out.unlink(missing_ok=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0 or not out.exists():
        raise _ArmFailed(arm, combined, proc.returncode)
    import torch

    payload = torch.load(out, weights_only=True)
    payload["log"] = combined
    return payload


def cmd_ab_arm(args) -> None:
    """The child side of `ab`. Not for direct use; the parent owns the protocol."""
    import torch

    _seed(args.seed)
    log: List[str] = []
    with _capture_build_output(log):
        model, inputs = _load_model(args.file)
    fresh = [a.clone() if isinstance(a, torch.Tensor) else a for a in inputs]
    with torch.no_grad():
        with _capture_build_output(log):
            observation = _observe(model, fresh)
        for _ in range(max(10, args.calls)):
            model(*inputs)
        _sync()
        samples = [_time_one(lambda: model(*inputs), args.calls) for _ in range(args.inner_rounds)]
    torch.save(
        {
            "samples": samples,
            "input_digest": _input_digest(inputs),
            "observation": _to_host(observation),
            "build_log": _trim_log(log),
        },
        args.out,
    )


def _ab_observations(round_medians: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """One observation per flipped pair of rounds, per arm.

    The launch order of a round enters its two per-process medians with opposite sign in
    the next round, so the geometric mean of an arm's medians over the pair carries the
    arm's own time and not its place in the order; the log-ratio of the two arms' means is
    the ABBA difference of :func:`flashinfer_bench.device.calibration._paired_delta_us`.
    Pairing rounds one by one would instead put the whole order effect into the spread and
    hide a real difference behind it.
    """
    return {
        arm: [math.sqrt(m[i] * m[i + 1]) for i in range(0, len(m) - 1, 2)]
        for arm, m in round_medians.items()
    }


def _ab_arms_agree(
    first: Dict[str, Dict[str, Any]], spills: Union[int, str], atol: float, rtol: float
) -> float:
    """The max abs error between the arms' first observations; exits if they cannot be compared."""
    if first["a"]["input_digest"] != first["b"]["input_digest"]:
        _report(build="OK", spills=spills, inputs="DIFFER", verdict="UNCOMPARABLE")
        print(
            "  the two arms did not see the same inputs, so their outputs cannot be compared "
            "and their timings are of different problems. Point both arms at harnesses "
            "whose get_inputs() agree under one seed."
        )
        print("DONE")
        raise SystemExit(1)
    reason, err = _compare(first["a"]["observation"], first["b"]["observation"], atol, rtol)
    if reason:
        _report(
            build="OK",
            spills=spills,
            inputs="IDENTICAL",
            correct="FAILED",
            reason=reason,
            verdict="INCORRECT",
        )
        print("DONE")
        raise SystemExit(1)
    return err


def cmd_ab(args) -> None:
    env_a = _parse_env(args.env_a, "--env-a")
    env_b = _parse_env(args.env_b, "--env-b")
    flags = _routing_flags(args)
    routing_extra: Dict[str, Any] = {"routing": "UNCHECKED"}
    if flags:
        # Arm A is the production harness; its candidate is the one the routing priced.
        _routing_or_exit(flags["bound"], flags["mechanism"], args.harness)
        routing_extra = {"routing": "OK", "mechanism": flags["mechanism"]}
    harness_b = args.harness_b or args.harness
    if harness_b == args.harness and env_a == env_b:
        print("  NOTE: the arms are identical (A/A); the result is this harness's noise floor.")
    if args.rounds % 2 or args.rounds < 2 * MIN_PAIRED_ROUNDS:
        raise SystemExit(
            f"--rounds must be even and at least {2 * MIN_PAIRED_ROUNDS}: consecutive rounds "
            "run the arms in opposite order, one such pair is one observation, and fewer "
            f"than {MIN_PAIRED_ROUNDS} observations cannot estimate a spread."
        )

    # One observation is a flipped pair of consecutive rounds (A,B then B,A). Whatever the
    # launch order does to a process -- the clock state it inherits, the core it lands on
    # once the other arm's process has exited -- enters the two rounds with opposite sign
    # and cancels in their mean, as in an ABBA measurement. The inner samples of one process
    # share that process's placement and clock, so they are one measurement rather than
    # several pairs: each process contributes its median, and an observation pairs the
    # geometric mean of an arm's two medians against the other arm's. From there the two
    # paths are judged by one rule on one statistic.
    round_medians: Dict[str, List[float]] = {"a": [], "b": []}
    first: Dict[str, Dict[str, Any]] = {}
    spills: Union[int, str] = "unknown"
    err = 0.0
    with tempfile.TemporaryDirectory(prefix="kernel-trials-ab-") as tmp:
        try:
            for r in range(args.rounds):
                # Alternate, and flip the order each round, so neither the clock ramp nor
                # a per-process warm-up systematically favours one arm.
                order = ("a", "b") if r % 2 == 0 else ("b", "a")
                for arm in order:
                    payload = _run_ab_arm(
                        arm,
                        args.harness if arm == "a" else harness_b,
                        env_a if arm == "a" else env_b,
                        args.seed,
                        args.calls,
                        args.inner_rounds,
                        pathlib.Path(tmp) / f"{arm}.pt",
                        args.timeout,
                    )
                    round_medians[arm].append(statistics.median(payload["samples"]))
                    if arm not in first:
                        first[arm] = payload
                    elif payload["input_digest"] != first[arm]["input_digest"]:
                        _report(build="OK", inputs="UNSTABLE", verdict="UNCOMPARABLE")
                        print(
                            f"  arm {arm.upper()} built different inputs in two processes under "
                            "the same seed; get_inputs() must be deterministic for a "
                            "cross-process comparison to mean anything."
                        )
                        print("DONE")
                        raise SystemExit(1)
                if r == 0:
                    # Both arms have been observed once: settle comparability and correctness
                    # now, before the remaining launches are spent timing a wrong kernel.
                    spills = _spill_state(first["b"].get("build_log"))
                    err = _ab_arms_agree(first, spills, args.atol, args.rtol)
        except _ArmFailed as exc:
            _report(
                build="FAILED",
                arm=exc.arm.upper(),
                spills=_spill_from(exc.output) or "unknown",
                verdict="BUILD_FAILED",
            )
            print(f"--- arm {exc.arm.upper()} diagnostics (exit {exc.returncode}) ---")
            print(exc.output[-_BUILD_LOG_KEEP:])
            print("DONE")
            raise SystemExit(1)

    observations = _ab_observations(round_medians)
    result = {
        **_summarize(observations["a"], observations["b"]),
        "correctness": "pass",
        "max_abs_error": err,
        "rounds": args.rounds,
        "inner_rounds": args.inner_rounds,
        "calls_per_round": args.calls,
        "processes": 2 * args.rounds,
        "round_medians": round_medians,
        "arm_a": {"harness": args.harness, "env": env_a},
        "arm_b": {"harness": harness_b, "env": env_b},
        "build_log": first["b"].get("build_log", ""),
    }
    verdict = _report_result(
        result, spills, inputs="IDENTICAL", processes=result["processes"], **routing_extra
    )
    result["verdict"] = verdict
    if flags:
        result["routing"] = {**flags, "bound": _rel(flags["bound"])}
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    print("DONE")


# --------------------------------------------------------------------------- cli


def _add_measure_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--calls",
        type=int,
        default=30,
        help="Calls inside one timed region, so per-call overhead amortizes.",
    )
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)


def _add_routing_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bound",
        help="bound.json (or its directory) from scripts/bound_candidates.py. With "
        "--mechanism, the harness's (candidate, mechanism) pair must be an ACCEPT row of it, "
        "and the routing must still be about the discovery it names; otherwise this refuses "
        "before measuring, naming the gate that rejected the pair.",
    )
    p.add_argument(
        "--mechanism",
        help="The delivery mechanism this series exercises, as bound.log names it "
        "(provider_patch, triton_in_place, library_call, ...). Requires --bound.",
    )


def _at_least(floor: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < floor:
            raise argparse.ArgumentTypeError(f"must be at least {floor}, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("name")
    p.add_argument("baseline")
    _add_routing_flags(p)
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
        "--rounds",
        type=_at_least(MIN_PAIRED_ROUNDS),
        default=11,
        help="Interleaved rounds; the median of the per-round ratios is reported, and their "
        f"scatter is the noise floor. At least {MIN_PAIRED_ROUNDS}; more only makes the "
        "verdict safer.",
    )
    _add_measure_flags(p)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser(
        "ab",
        help="A/B two builds of one op that cannot share a process; each arm runs in its own.",
    )
    p.add_argument("harness", help="Harness both arms run (arm B may override with --harness-b).")
    p.add_argument(
        "--env-a", action="append", default=[], metavar="KEY=VALUE", help="Set in arm A only."
    )
    p.add_argument(
        "--env-b", action="append", default=[], metavar="KEY=VALUE", help="Set in arm B only."
    )
    p.add_argument("--harness-b", help="A different harness file for arm B.")
    p.add_argument(
        "--rounds",
        type=int,
        default=14,
        help="Process launches per arm, alternating A/B with the order flipped each round; "
        "each flipped pair of rounds is one observation, so this must be even and at least "
        f"{2 * MIN_PAIRED_ROUNDS}.",
    )
    p.add_argument(
        "--inner-rounds",
        type=int,
        default=3,
        help="Timed regions per process; a process contributes their median.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed both arms build inputs from.")
    p.add_argument("--timeout", type=int, default=600, help="Seconds allowed per process.")
    p.add_argument("--json", help="Also write the result here.")
    _add_routing_flags(p)
    _add_measure_flags(p)
    p.set_defaults(func=cmd_ab)

    p = sub.add_parser("_ab-arm", help=argparse.SUPPRESS)
    p.add_argument("file")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--calls", type=int, required=True)
    p.add_argument("--inner-rounds", type=int, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.set_defaults(func=cmd_ab_arm)

    for name, fn in (("status", cmd_status), ("best", cmd_best)):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.set_defaults(func=fn)

    p = sub.add_parser("finalize")
    p.add_argument("name")
    p.add_argument("output")
    p.add_argument(
        "--require-win",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse unless the best trial is a WIN: its gain clears the paired round-to-round "
        "scatter of the two arms (the rule `benchmark` reports by). On by default. --no-require-win copies a LOSS or "
        "NOISE best anyway, for record-keeping only -- archiving the least-bad attempt "
        "of an abandoned series -- and never to ship a kernel.",
    )
    p.set_defaults(func=cmd_finalize)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    _ensure_toolchain_on_path()
    args.func(args)


if __name__ == "__main__":
    main()
