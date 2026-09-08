"""Apply table for mapping workload keys to optimal solutions."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet
from flashinfer_bench.env import get_fib_cache_path

from .config import ApplyConfigRegistry
from .key import ApplyKey, ApplyKeyFactory


def _apply_table_dir() -> Path:
    """Get the directory for storing apply table cache files.

    Returns
    -------
    Path
        The apply table cache directory path.
    """
    return get_fib_cache_path() / "apply_table"


logger = logging.getLogger(__name__)

_PROVIDER_MARKERS = ("vllm_xpu", "sgl_kernel_xpu", "vllm_", "flashinfer_wrapper")
"""Solution-name markers for a kernel the serving stack would run if we declined."""


def _targets_this_machine(solution) -> bool:
    """Whether this solution declares the backend we are actually running on.

    Asked before building rather than discovered by building, because building the wrong
    one is not merely wasted work: `torch.utils.cpp_extension.load` guards its build
    directory with a `FileBaton`, and `FileBaton.wait` polls for a lock file with no
    timeout. A process killed mid-build leaves that file behind, and the next process to
    want the same extension waits on it forever -- which no exception handler can catch.
    A vLLM EngineCore hung exactly that way for ten minutes, warming a CUDA `kernel.cu`
    on an Intel GPU against a baton left by a run killed hours earlier.

    The check is deliberately strict: warm-up is an optimization, so skipping a solution
    that would in fact have built costs one lazy build later, which `ApplyRuntime` already
    handles. Building one that cannot costs a hang.
    """
    from flashinfer_bench.device import default_device_type

    try:
        backend = default_device_type()
    except Exception:
        return False
    targets = getattr(solution.spec, "target_hardware", None) or []
    return any(str(t).lower() == backend.lower() for t in targets)


def _warm(registry, definition, solution) -> None:
    """Pre-build a solution, tolerating one that cannot be built on this machine.

    Warm-up is an optimization: it moves compilation off the first call. Nothing about it
    should be able to stop the process, and a shared dataset guarantees it will meet
    solutions for other hardware -- every CUDA solution on an Intel GPU, and some that
    fail on a missing Python dependency rather than a missing device.

    Before this, warming a `cublaslt_fp4_e2m1_scaled_mm` solution that needs `torchao`
    raised out of table construction and killed a vLLM EngineCore at startup. A solution
    that cannot be pre-built is simply not pre-built; if it is ever selected, the runtime
    tries again and falls back there.
    """
    if not _targets_this_machine(solution):
        logger.debug(
            "Skipping warm-up of '%s' for '%s': targets %s, not this machine.",
            solution.name,
            definition.name,
            getattr(solution.spec, "target_hardware", None),
        )
        return
    try:
        registry.build(definition, solution)
    except Exception as e:
        logger.debug(
            "Skipping warm-up of '%s' for '%s': %s: %s",
            solution.name,
            definition.name,
            type(e).__name__,
            str(e)[:200],
        )


@dataclass
class ApplyTable:
    """Apply table for mapping workload keys to optimal solutions.

    This class manages a lookup table that maps workload characteristics (ApplyKey)
    to the best performing solution for each kernel definition. It supports caching
    to disk and ahead-of-time compilation of frequently used solutions.
    """

    digest: str
    """Hash digest identifying this table's configuration and data."""
    index: Dict[str, Dict[ApplyKey, str]] = field(default_factory=dict)
    """Mapping from definition name to (key -> solution_name) lookup."""
    def_best: Dict[str, str] = field(default_factory=dict)
    """Mapping from definition name to best overall solution name."""

    @classmethod
    def _load_from_disk(cls, digest: str) -> Optional[Dict[str, Any]]:
        """Load apply table data from cache file.

        Parameters
        ----------
        digest : str
            The digest hash identifying the cached data.

        Returns
        -------
        Optional[Dict[str, Any]]
            The cached data if exists, None otherwise.
        """
        index_path = _apply_table_dir() / f"{digest}.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)

        return None

    @classmethod
    def _save_to_disk(cls, digest: str, data: Dict[str, Any]) -> None:
        """Save apply table data to cache file.

        Parameters
        ----------
        digest : str
            The digest hash identifying the data to cache.
        data : Dict[str, Any]
            The data to save to cache.
        """
        index_path = _apply_table_dir() / f"{digest}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_or_build(
        cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry
    ) -> "ApplyTable":
        """Load an existing apply table from cache or build a new one.

        This method first attempts to load a cached apply table based on the
        digest of the trace set and configuration. If no cached version exists,
        it builds a new table from scratch and caches it for future use.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data and solutions.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry for building the apply table.

        Returns
        -------
        ApplyTable
            The loaded or newly built apply table.
        """
        digest = cls._digest(trace_set, config_registry)

        # Try to load from cache
        raw = cls._load_from_disk(digest)
        if raw is not None:
            index: Dict[str, Dict[ApplyKey, str]] = {}
            for def_name, items in raw.get("index", {}).items():
                bucket: Dict[ApplyKey, str] = {}
                for key_enc, sol_name in items.items():
                    key = ApplyKey.model_validate_json(key_enc)
                    bucket[key] = sol_name
                index[def_name] = bucket

            def_best: Dict[str, str] = raw.get("def_best", {})

            table = cls(digest=digest, index=index, def_best=def_best)
            cls._prewarm_aot(trace_set, config_registry, table)

            return table

        # Build fresh
        table = cls._build(trace_set, config_registry)
        # Persist minimal index
        to_dump: Dict[str, Any] = {"digest": table.digest, "index": {}, "def_best": {}}
        for def_name, bucket in table.index.items():
            for key, sol_name in bucket.items():
                to_dump["index"].setdefault(def_name, {})[key.model_dump_json()] = sol_name
        # Always compute and persist def_best
        for def_name, sol_name in table.def_best.items():
            to_dump["def_best"][def_name] = sol_name

        cls._save_to_disk(digest, to_dump)
        cls._prewarm_aot(trace_set, config_registry, table)

        return table

    @classmethod
    def _build(cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry) -> "ApplyTable":
        """Build a new apply table from trace set data.

        This method processes all traces in the trace set to build lookup tables
        mapping workload keys to optimal solutions for each kernel definition.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data and solutions.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry including error tolerances.

        Returns
        -------
        ApplyTable
            The newly built apply table.
        """
        digest = cls._digest(trace_set, config_registry)
        hardware_id = cls._current_hardware_id()

        index: Dict[str, Dict[ApplyKey, str]] = {}
        def_best: Dict[str, str] = {}

        for def_name, definition in trace_set.definitions.items():
            config = config_registry.get(def_name)
            if config is None:
                continue
            per_key, ranked, rejected_any = cls._sweep_def(
                trace_set,
                def_name,
                config.max_atol,
                config.max_rtol,
                hardware_id,
                config.min_gain_us / 1000.0,
            )

            # Build index
            for key, t in per_key.items():
                if not t.solution:
                    continue
                bucket = index.setdefault(def_name, {})
                bucket[key] = t.solution

            # Build def_best.
            #
            # `use_def_best` extends the overall winner to shapes that were never measured.
            # That is interpolation between measured points when every measured point was
            # worth substituting, but an unjustified guess once some were explicitly judged
            # not to be: the shapes it would cover are the ones most like the rejected ones.
            # Leaving it unset makes those calls fall back, which is what the rejection
            # asked for.
            if ranked and not rejected_any:
                def_best[def_name] = ranked[0][0]
            elif rejected_any:
                logger.info(
                    "%s: no def_best -- some keys do not clear the substitution cost, so "
                    "extending a winner to unmeasured shapes is not justified.",
                    def_name,
                )

        return cls(digest=digest, index=index, def_best=def_best)

    @classmethod
    def _provider_traces(
        cls,
        trace_set: TraceSet,
        def_name: str,
        hardware_id: Optional[str],
        timing: Optional[str],
    ) -> List[Trace]:
        """Every PASSED provider trace for this definition on this part and timer."""
        out: List[Trace] = []
        for trace in trace_set.traces.get(def_name, []):
            evaluation = trace.evaluation
            if evaluation is None or evaluation.status != EvaluationStatus.PASSED:
                continue
            solution = trace.solution or ""
            if not any(f"__{p}" in solution for p in _PROVIDER_MARKERS):
                continue
            if hardware_id and cls._trace_hardware(trace) not in (hardware_id, None):
                continue
            if timing:
                recorded = cls._trace_timing(trace)
                if recorded and recorded != timing:
                    continue
            out.append(trace)
        return out

    @classmethod
    def _drop_keys_not_worth_substituting(
        cls,
        def_name: str,
        per_key: Dict[ApplyKey, Trace],
        traces: List[Trace],
        builder: Any,
        min_gain_ms: float,
    ) -> Dict[ApplyKey, Trace]:
        """Keep only keys where our winner beats the provider by more than dispatch costs.

        The provider baseline is what runs when we decline, so it -- not the definition's
        reference -- is what a substitution has to beat. And it has to beat it by more than
        the substitution costs: a kernel three times faster than the provider's still loses
        the exchange when the kernel is 5us and dispatch is 6us.

        Keys with no provider trace are kept. There is nothing to compare against, and
        dropping them would silently disable substitution for every definition whose
        baseline has simply not been generated yet.
        """
        provider_latency: Dict[ApplyKey, float] = {}
        for t in traces:
            solution = t.solution or ""
            if not any(f"__{p}" in solution for p in _PROVIDER_MARKERS):
                continue
            latency = t.evaluation.performance.latency_ms
            if not latency:
                continue
            key = builder.build_from_workload(t.workload)
            if latency < provider_latency.get(key, float("inf")):
                provider_latency[key] = latency

        kept: Dict[ApplyKey, Trace] = {}
        dropped = uncomparable = 0
        for key, trace in per_key.items():
            provider = provider_latency.get(key)
            ours = trace.evaluation.performance.latency_ms
            if provider is None or not ours:
                # No comparator, so no evidence that substituting is worth its cost. With
                # the gate on, absence of evidence is not a licence to deploy: the burden of
                # proof belongs on the substitution, which is the thing that costs time.
                uncomparable += 1
                continue
            if (provider - ours) > min_gain_ms:
                kept[key] = trace
            else:
                dropped += 1
        if dropped or uncomparable:
            logger.info(
                "%s: %d of %d key(s) not indexed (%d lose to the provider by less than the "
                "%.2fus a substitution costs, %d have no provider baseline on this part to "
                "compare against -- run add-baselines to judge them).",
                def_name,
                dropped + uncomparable,
                len(per_key),
                dropped,
                min_gain_ms * 1000,
                uncomparable,
            )
        return kept

    @staticmethod
    def _trace_timing(trace: Trace) -> Optional[str]:
        """Which timing methodology produced this trace's latency, if recorded."""
        env = getattr(trace.evaluation, "environment", None)
        libs = getattr(env, "libs", None) if env is not None else None
        if isinstance(libs, dict):
            return libs.get("timing")
        return getattr(libs, "timing", None) if libs is not None else None

    @staticmethod
    def _current_timing(hardware_id: Optional[str]) -> Optional[str]:
        """The methodology this machine measures with, so stale traces can be excluded."""
        try:
            from flashinfer_bench.device import default_device_type, get_accelerator

            accel = get_accelerator(default_device_type())
            devices = accel.list_devices()
            return accel.make_timer(devices[0]).name if devices else None
        except Exception:
            return None

    @staticmethod
    def _trace_hardware(trace: Trace) -> Optional[str]:
        """Canonical part id a trace was measured on.

        `hardware_id` is the canonical field, but traces recorded before it existed carry
        only the raw vendor string, so fall back to canonicalizing that. Returns None when
        the trace records neither, which the caller treats as "unknown, keep" rather than
        "foreign, drop".
        """
        env = getattr(trace.evaluation, "environment", None)
        if env is None:
            return None
        recorded = getattr(env, "hardware_id", None)
        if recorded:
            return recorded
        raw = getattr(env, "hardware", None)
        if not raw:
            return None
        from flashinfer_bench.device import canonicalize_device_name

        return canonicalize_device_name(raw)

    @staticmethod
    def _current_hardware_id() -> Optional[str]:
        """Canonical id of the device this process will dispatch on, if determinable.

        Returns None when it cannot be established, in which case ranking stays unfiltered
        -- the previous behaviour -- rather than silently selecting nothing.
        """
        try:
            from flashinfer_bench.device import default_device_type, get_accelerator

            device_type = default_device_type()
            accelerator = get_accelerator(device_type)
            devices = accelerator.list_devices()
            if not devices:
                return None
            return accelerator.canonical_id(devices[0])
        except Exception:
            return None

    @classmethod
    def _sweep_def(
        cls,
        trace_set: TraceSet,
        def_name: str,
        max_atol: float,
        max_rtol: float,
        hardware_id: Optional[str] = None,
        min_gain_ms: float = 0.0,
    ) -> Tuple[Dict[ApplyKey, Trace], List[Tuple[str, int]], bool]:
        """Sweep through traces for a definition to find optimal solutions per key.

        This method processes all traces for a given kernel definition, groups them
        by workload key, and selects the best performing solution for each key.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data.
        def_name : str
            Name of the kernel definition to process.
        max_atol : float
            Maximum absolute error tolerance for filtering traces.
        max_rtol : float
            Maximum relative error tolerance for filtering traces.

        Returns
        -------
        Tuple[Dict[ApplyKey, Trace], List[Tuple[str, int]]]
            A tuple containing:
            - Dictionary mapping keys to best traces
            - List of (solution_name, win_count) pairs sorted by wins
        """
        traces = trace_set.filter_traces(def_name, max_atol, max_rtol)

        # Rank only what was measured on this machine. `speedup_factor` is relative to the
        # reference on the hardware that produced it, so comparing across parts is not a
        # comparison at all -- and the winner is then a solution that cannot run here: a
        # CUDA one fails to build and every call silently falls back, while a Triton one
        # built for another architecture builds and then raises on first use, inside the
        # serving engine. Dropping them leaves the definition with no entry, which falls
        # back cleanly, instead of selecting something unusable.
        # Latencies from different timing methodologies are not comparable -- the earlier
        # per-call event timing reported ~8x the true latency for a short kernel -- so a
        # stale trace ranked against a fresh one manufactures a win out of the measurement
        # change alone. Keep only what this machine's timer would produce; traces recording
        # no methodology are kept, since most predate the field.
        current_timing = cls._current_timing(hardware_id)
        if current_timing and traces:
            same_timing = [t for t in traces if cls._trace_timing(t) in (current_timing, None)]
            if same_timing and len(same_timing) < len(traces):
                logger.debug(
                    "%s: ignoring %d trace(s) measured with another timing methodology",
                    def_name,
                    len(traces) - len(same_timing),
                )
            if same_timing:
                traces = same_timing

        if hardware_id is not None and traces:
            # `speedup_factor` is relative to the reference on the part that produced it,
            # so ranking across parts is not a comparison -- and the winner is then a
            # solution that cannot run here: a CUDA one fails to build and every call
            # falls back silently, while a Triton one built for another architecture
            # builds and then raises on first use, inside the serving engine.
            # A trace recording no hardware at all is ambiguous, so it is kept; only one
            # that names a different part is dropped.
            same_part = [t for t in traces if cls._trace_hardware(t) in (hardware_id, None)]
            dropped = len(traces) - len(same_part)
            if dropped and same_part:
                logger.debug(
                    "%s: ignoring %d trace(s) from other hardware when ranking for %s",
                    def_name,
                    dropped,
                    hardware_id,
                )
                traces = same_part
            elif dropped:
                # Nothing was measured here at all. Refusing outright would leave the
                # definition with no entry on a dataset never benchmarked on this part, so
                # keep them and say so -- the selected solution may not run here.
                logger.warning(
                    "%s: no traces from %s; ranking %d trace(s) measured on other "
                    "hardware. The selected solution may not run here -- benchmark this "
                    "definition on this device to fix it.",
                    def_name,
                    hardware_id,
                    len(traces),
                )

        builder = ApplyKeyFactory.specialize(trace_set.definitions[def_name])

        # Pick the trace with the highest speedup_factor for each key
        per_key: Dict[ApplyKey, Trace] = {}
        for t in traces:
            key = builder.build_from_workload(t.workload)
            prev = per_key.get(key)
            if (
                prev is None
                or t.evaluation.performance.speedup_factor
                > prev.evaluation.performance.speedup_factor
            ):
                per_key[key] = t

        rejected_any = False
        if min_gain_ms > 0:
            before = len(per_key)
            per_key = cls._drop_keys_not_worth_substituting(
                def_name,
                per_key,
                # Deliberately not `traces`: that has been filtered to solutions meeting
                # *our* correctness tolerance, and several provider baselines pass with an
                # error above it. Excluding them made the comparator vanish and the key read
                # as "nothing to compare against", which the gate then let through -- so the
                # gate was inert for almost every key it was meant to judge. The provider
                # kernel runs whether or not it meets our tolerance; it is the thing being
                # replaced, so it is the thing to measure against.
                cls._provider_traces(trace_set, def_name, hardware_id, current_timing),
                builder,
                min_gain_ms,
            )
            rejected_any = len(per_key) < before

        # Count wins per solution
        win_counts: Dict[str, int] = {}
        for t in per_key.values():
            if t.solution:
                win_counts[t.solution] = win_counts.get(t.solution, 0) + 1

        ranked = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
        return per_key, ranked, rejected_any

    @classmethod
    def _prewarm_aot(
        cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry, table: "ApplyTable"
    ) -> None:
        """Perform ahead-of-time compilation of frequently used solutions.

        This method pre-compiles solutions that are used frequently according to
        the AOT ratio configuration. This reduces runtime compilation overhead.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing definitions and solutions.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry containing AOT ratio and miss policy settings.
        table : ApplyTable
            The apply table containing solution mappings.
        """
        reg = BuilderRegistry.get_instance()

        for def_name, bucket in table.index.items():
            if not bucket:
                continue

            config = config_registry.get(def_name)
            if config is None:
                continue
            if not (config.aot_ratio and config.aot_ratio > 0.0):
                continue

            win_counts = Counter(bucket.values())
            ranked = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
            cutoff = max(1, int(len(ranked) * config.aot_ratio))

            definition = trace_set.definitions.get(def_name)
            if not definition:
                continue
            for sol_name, _ in ranked[:cutoff]:
                solution = trace_set.get_solution(sol_name)
                if solution:
                    _warm(reg, definition, solution)

        # Build def_best for definitions with on_miss_policy == "use_def_best"
        for def_name, sol_name in table.def_best.items():
            config = config_registry.get(def_name)
            if config is None:
                continue
            if config.on_miss_policy == "use_def_best":
                definition = trace_set.definitions.get(def_name)
                solution = trace_set.get_solution(sol_name)
                if definition and solution:
                    _warm(reg, definition, solution)

    @classmethod
    def _digest(cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry) -> str:
        """Compute a hash digest for the trace set and configuration.

        This method creates a deterministic hash that uniquely identifies the
        combination of trace set data and apply configuration. The digest is
        used for caching apply tables.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry.

        Returns
        -------
        str
            SHA256 hash digest as a hexadecimal string.
        """
        trace_set_dict = trace_set.to_dict()
        # The table now depends on which part it was built for, so a cached one from
        # another machine (or another GPU in this one) must not be reused.
        trace_set_dict["_hardware_id"] = cls._current_hardware_id()
        for definition in trace_set_dict["definitions"].values():
            for drop in ("description", "tags", "reference", "constraints"):
                definition.pop(drop, None)
        for sol_list in trace_set_dict["solutions"].values():
            for solution in sol_list:
                spec = solution.get("spec", {}) or {}
                deps = spec.get("dependencies") or []
                spec["dependencies"] = sorted(deps)
                new_sources = []
                for sf in solution.get("sources") or []:
                    new_sources.append(
                        {
                            "path": sf["path"],
                            "sha1": hashlib.sha1(sf["content"].encode("utf-8")).hexdigest(),
                        }
                    )
                solution["sources"] = new_sources
        kept_traces: List[Dict[str, Any]] = []
        for traces in trace_set_dict["traces"].values():
            for trace in traces:
                ev = trace.get("evaluation") or {}
                perf = ev.get("performance") or {}
                corr = ev.get("correctness") or {}
                kept_traces.append(
                    {
                        "definition": trace["definition"],
                        "solution": trace.get("solution", ""),
                        "axes": sorted((trace["workload"] or {}).get("axes", {}).items()),
                        "status": ev.get("status"),
                        "max_abs_error": corr.get("max_absolute_error"),
                        "max_rel_error": corr.get("max_relative_error"),
                        "speedup": perf.get("speedup_factor"),
                    }
                )

        # Build per-definition config dict for digest
        default_cfg = config_registry.default
        per_def_cfg_dict: Dict[str, Dict[str, Any]] = {}
        for def_name in trace_set_dict["definitions"]:
            cfg = config_registry.per_definition.get(def_name)
            if cfg is not None:
                per_def_cfg_dict[def_name] = {"max_atol": cfg.max_atol, "max_rtol": cfg.max_rtol}

        payload = {
            "cfg": {
                "default": (
                    {"max_atol": default_cfg.max_atol, "max_rtol": default_cfg.max_rtol}
                    if default_cfg
                    else None
                ),
                "per_def": per_def_cfg_dict,
            },
            "definitions": trace_set_dict["definitions"],
            "solutions": trace_set_dict["solutions"],
            "traces": sorted(
                kept_traces,
                key=lambda x: (
                    x["definition"],
                    x["solution"],
                    x["axes"],
                    x["status"] or "",
                    x["speedup"] or 0.0,
                ),
            ),
        }
        str_repr = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(str_repr).hexdigest()

    def match_solution(self, def_name: str, key: ApplyKey) -> Optional[str]:
        """Find the optimal solution for a given definition and workload key.

        This method looks up the best solution for a specific kernel definition
        and workload characteristics combination.

        Parameters
        ----------
        def_name : str
            Name of the kernel definition.
        key : ApplyKey
            Workload characteristics key to match against.

        Returns
        -------
        Optional[str]
            The name of the optimal solution, or None if no match is found.
        """
        return self.index.get(def_name, {}).get(key)
