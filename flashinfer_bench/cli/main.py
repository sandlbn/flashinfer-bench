"""Main CLI entry point and command implementations."""

import argparse
import logging
from pathlib import Path
from typing import Any, List

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import TraceSet, save_json_file, save_jsonl_file

logger = logging.getLogger(__name__)
pkg_name = __name__.split(".")[0]


def cli_config_logging(args: argparse.Namespace):
    """Configure package-level logging from CLI args."""
    log_level = getattr(args, "log_level", "WARNING")
    pkg_logger = logging.getLogger(pkg_name)
    pkg_logger.setLevel(log_level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
        pkg_logger.addHandler(handler)
    pkg_logger.propagate = False


def best(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for trace_set in trace_sets:
        definitions = trace_set.definitions.keys()
        for definition in definitions:
            trace = trace_set.get_best_trace(definition)
            if not trace:
                logger.warning(f"No valid solution found for {definition}.")
                continue
            logger.info(f"Best solution for {definition}:")
            logger.info(f"- Solution: {trace.solution}")
            logger.info(f"- Speedup vs. ref: {trace.evaluation.performance.speedup_factor:.2f}×")
            logger.info(
                f"- Errors:   abs={trace.evaluation.correctness.max_absolute_error:.2e}, "
                f"rel={trace.evaluation.correctness.max_relative_error:.2e}"
            )


def summary(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for i, trace_set in enumerate(trace_sets):
        s = trace_set.summary()

        if len(trace_sets) > 1:
            logger.info("Dataset %d:", i + 1)

        logger.info("Traces: %d total, %d passed, %d failed", s.total, s.passed, s.failed)
        if s.rankings:
            logger.info("")
            logger.info("Author Rankings (by average speedup):")
            logger.info(
                "  %-4s  %-24s  %-10s  %-14s  %-14s  %-10s",
                "Rank",
                "Author",
                "Avg Speedup",
                "Success Rate",
                "Win Rate",
                "Workloads",
            )
            logger.info("  " + "-" * 92)
            for rank, (author, score) in enumerate(s.rankings, start=1):
                logger.info(
                    "  %-4d  %-24s  %-10.4f  %-14.1f  %-14.1f  %-10d",
                    rank,
                    author,
                    score.avg_speedup,
                    score.success_rate * 100,
                    score.win_rate * 100,
                    score.workloads,
                )
            logger.info("")
            logger.info("  Avg Speedup: mean of baseline_latency / author_latency")
            logger.info("  Success Rate: fraction of workloads with successful execution")
            logger.info("  Win Rate: fraction of workloads where this author beats the baseline")
        else:
            logger.info("(No author ranking data available — run with multiple solutions)")


def merge_trace_sets(trace_sets):
    """Merge multiple TraceSets into one, raising on definition conflicts."""
    if not trace_sets:
        raise ValueError("No TraceSets to merge.")
    # Start with a deep copy of the first TraceSet
    from copy import deepcopy

    merged = deepcopy(trace_sets[0])
    for trace_set in trace_sets[1:]:
        # Merge definitions
        for name, definition in trace_set.definitions.items():
            if name in merged.definitions:
                if merged.definitions[name] != definition:
                    raise ValueError(f"Definition conflict for '{name}' during merge.")
            else:
                merged.definitions[name] = definition
        # Merge solutions
        for def_name, solutions in trace_set.solutions.items():
            if def_name not in merged.solutions:
                merged.solutions[def_name] = []
            merged.solutions[def_name].extend(solutions)
        # Merge workloads
        for def_name, workloads in trace_set.workload.items():
            if def_name not in merged.workload:
                merged.workload[def_name] = []
            merged.workload[def_name].extend(workloads)
        # Merge traces
        for def_name, traces in trace_set.traces.items():
            if def_name not in merged.traces:
                merged.traces[def_name] = []
            merged.traces[def_name].extend(traces)
    return merged


def _safe_path_segment(segment: str) -> str:
    """Validate that a string is safe to use as a single path segment."""
    if not segment or "/" in segment or "\\" in segment or segment in (".", ".."):
        raise ValueError(f"Invalid path segment: {segment!r}")
    return segment


def export_trace_set(trace_set, output_dir):
    """Export a TraceSet to a directory in the expected structure."""
    output_dir = Path(output_dir)
    (output_dir / "definitions").mkdir(parents=True, exist_ok=True)
    (output_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    # Save definitions
    for definition in trace_set.definitions.values():
        out_path = output_dir / "definitions" / f"{_safe_path_segment(definition.name)}.json"
        save_json_file(definition, out_path)
    # Save solutions
    for def_name, solutions in trace_set.solutions.items():
        definition = trace_set.definitions[def_name]
        for solution in solutions:
            out_path = (
                output_dir
                / "solutions"
                / _safe_path_segment(solution.author)
                / _safe_path_segment(definition.op_type)
                / _safe_path_segment(def_name)
                / f"{_safe_path_segment(solution.name)}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_json_file(solution, out_path)
    # Save workload traces
    for def_name, workloads in trace_set.workload.items():
        if workloads:
            out_path = output_dir / "traces" / f"{_safe_path_segment(def_name)}_workloads.jsonl"
            save_jsonl_file(workloads, out_path)
    # Save regular traces
    for def_name, traces in trace_set.traces.items():
        if traces:
            out_path = output_dir / "traces" / f"{_safe_path_segment(def_name)}.jsonl"
            save_jsonl_file(traces, out_path)


def merge(args: argparse.Namespace):
    """Merge multiple TraceSets into a single one and export to output directory."""
    if not args.output:
        raise ValueError("--output <MERGED_PATH> is required for merge.")
    trace_sets = _load_traces(args)
    merged = merge_trace_sets(trace_sets)
    export_trace_set(merged, args.output)
    logger.info(f"Merged {len(trace_sets)} TraceSets and exported to {args.output}")


def visualize(args: argparse.Namespace):
    """Visualize benchmark results as a console table."""
    trace_sets = _load_traces(args)

    logger.info("FlashInfer Bench Results Visualization")
    logger.info("=" * 80)

    for i, trace_set in enumerate(trace_sets):
        if len(trace_sets) > 1:
            logger.info(f"\nDataset {i + 1}:")
            logger.info("-" * 40)

        # Print summary statistics
        summary = trace_set.summary()
        logger.info(f"Summary: {summary.passed}/{summary.total} traces passed")

        # Print detailed results table
        logger.info("\nDetailed Results:")
        logger.info("-" * 80)
        logger.info(
            f"{'Definition':<15} {'Solution':<25} {'Status':<10} {'Speedup vs. ref':<16} {'Latency(ms)':<12} {'Max Error':<15}"
        )
        logger.info("-" * 80)

        for def_name, traces in trace_set.traces.items():
            for trace in traces:
                status = trace.evaluation.get("status", "UNKNOWN")
                perf = trace.evaluation.get("performance", {})
                corr = trace.evaluation.get("correctness", {})

                speedup = perf.get("speedup_factor", "N/A")
                if isinstance(speedup, (int, float)):
                    speedup = f"{speedup:.2f}×"

                latency = perf.get("latency_ms", "N/A")
                if isinstance(latency, (int, float)):
                    latency = f"{latency:.3f}"

                max_error = corr.get("max_absolute_error", "N/A")
                if isinstance(max_error, (int, float)):
                    max_error = f"{max_error:.2e}"

                logger.info(
                    f"{def_name:<15} {trace.solution:<25} {status:<10} {speedup:<16} {latency:<12} {max_error:<15}"
                )

        # Print best solutions
        logger.info("\nBest Solutions:")
        logger.info("-" * 80)
        for def_name in trace_set.definitions.keys():
            best_trace = trace_set.get_best_op(def_name)
            if best_trace:
                perf = best_trace.evaluation.get("performance", {})
                corr = best_trace.evaluation.get("correctness", {})
                speedup = perf.get("speedup_factor", "N/A")
                if isinstance(speedup, (int, float)):
                    speedup = f"{speedup:.2f}×"
                logger.info(f"{def_name}: {best_trace.solution} (Speedup vs. ref: {speedup})")
            else:
                logger.warning(f"{def_name}: No valid solution found")


def serve(args: argparse.Namespace):
    """Start the benchmark HTTP server."""
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "uvicorn is required for the serve command. "
            "Install with: pip install flashinfer-bench[serve]"
        )

    from flashinfer_bench.bench import BenchmarkConfig
    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.serve.app import init_app
    from flashinfer_bench.serve.scheduler import Scheduler

    trace_set = TraceSet.from_path(str(args.local))

    devices = args.devices.split(",") if args.devices else None
    if devices is None:
        import flashinfer_bench.utils as fib_utils

        devices = fib_utils.list_devices()
    if not devices:
        raise RuntimeError(
            "No benchmark devices available. Set FIB_DEVICE_BACKEND to select a "
            "backend explicitly (cuda, xpu, cpu)."
        )

    config = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=args.rtol,
        atol=args.atol,
        timeout_seconds=args.timeout,
    )

    scheduler = Scheduler(trace_set=trace_set, config=config, devices=devices)
    app = init_app(scheduler)

    logger.info(f"Starting server on {args.host}:{args.port} with devices {devices}")
    uvicorn.run(app, host=args.host, port=args.port)


def run(args: argparse.Namespace):
    """Benchmark run: executes benchmarks and writes results."""
    if not args.local:
        raise ValueError("A data source is required. Please use --local <PATH>.")
    # Only support --local for now
    for path in args.local:
        trace_set = TraceSet.from_path(str(path))

        raw_cli_overrides = {
            "warmup_runs": args.warmup_runs,
            "iterations": args.iterations,
            "num_trials": args.num_trials,
            "rtol": args.rtol,
            "atol": args.atol,
            "use_isolated_runner": args.use_isolated_runner,
            "definitions": args.definitions,
            "solutions": args.solutions,
            "timeout_seconds": args.timeout,
            "required_matched_ratio": args.required_matched_ratio,
            "profile_baseline": args.profile_baseline,
        }
        cli_overrides = {k: v for k, v in raw_cli_overrides.items() if v is not None}
        config_path = getattr(args, "config", None)
        if config_path:
            config = BenchmarkConfig.from_yaml(config_path, **cli_overrides)
        else:
            config = BenchmarkConfig.default(**cli_overrides)
        benchmark = Benchmark(trace_set, config)
        logger.info(f"Running benchmark on FlashInfer Trace Dataset: {Path(path).resolve()}")
        resume = getattr(args, "resume", False)
        if resume:
            logger.info("Resume mode enabled: will skip already evaluated solutions")

        try:
            benchmark.run_all(args.save_results, resume=resume)
        except Exception:
            logger.exception("Benchmark run failed")
            raise
        finally:
            benchmark.close()

        message = "Benchmark run complete."
        if args.save_results:
            message += " Results saved."
        else:
            message += " Results not saved (use --save-results to enable saving)."
        logger.info(message)


def providers_cmd(args: argparse.Namespace) -> None:
    """Inspect or acquire the Intel kernel providers.

    Acquisition differs per provider -- a wheel, a source build that must be told the GPU
    architecture, a checkout pointed at by an environment variable, a system library -- so
    this reports all of them uniformly and installs the ones that can be installed.
    """
    from flashinfer_bench.integration import providers as prov

    if args.provider_action in (None, "list"):
        rows = prov.all_status()
        width = max(len(r["name"]) for r in rows)
        for r in rows:
            mark = "installed" if r["installed"] == "yes" else "-"
            version = f" {r['version']}" if r["version"] else ""
            where = f"  {r['location']}" if r["location"] else ""
            print(f"{r['name']:<{width}}  {r['kind']:<9} {mark}{version}{where}")
            print(f"{'':<{width}}  {r['summary']}")
            if r["installed"] != "yes":
                print(f"{'':<{width}}  cost: {r['cost']}")
        return

    if args.provider_action == "verify":
        # Registering an op that is not built for this backend costs a RUNTIME_ERROR on
        # every workload of every matching definition, and reads like a wrapper bug. This
        # calls each registered kernel once instead.
        from flashinfer_bench.integration.xpu_kernels import (
            REGISTRY,
            available_providers,
            verify_kernel,
        )

        trace_set = TraceSet.from_path(str(args.local))
        installed = set(available_providers())
        checked = failed = 0
        for kernel in REGISTRY:
            if kernel.provider not in installed:
                continue
            # Prefer a workload whose inputs are synthesizable. A workload backed by
            # safetensors needs its LFS blob present, and a missing blob fails the check
            # for a reason that has nothing to do with whether the kernel is built.
            candidates = [
                (d, t.workload)
                for d in trace_set.definitions.values()
                if kernel.matches(d)
                # workloads are stored as Traces; the Workload is a field on each.
                for t in trace_set.workloads.get(d.name, [])
            ]
            synthesizable = [
                (d, w)
                for d, w in candidates
                if all(getattr(spec, "type", None) != "safetensors" for spec in w.inputs.values())
            ]
            pair = (synthesizable or candidates or [None])[0]
            if pair is None:
                logger.info(
                    "%-16s %-34s no definition+workload to test with", kernel.provider, kernel.name
                )
                continue
            ok, note = verify_kernel(pair[0], kernel, pair[1], args.device)
            checked += 1
            failed += 0 if ok else 1
            logger.log(
                logging.INFO if ok else logging.ERROR,
                "%-16s %-34s %s  (%s)",
                kernel.provider,
                kernel.name,
                "OK" if ok else "FAIL",
                note,
            )
        logger.info("Verified %d kernel(s), %d failed", checked, failed)
        if failed:
            raise SystemExit(1)
        return

    if args.provider_action == "install":
        try:
            command, code = prov.install(
                args.name, target=args.target, device=args.device, dry_run=args.dry_run
            )
        except prov.ProviderError as e:
            # A refusal here is the useful outcome: it explains why before a multi-GiB
            # build rather than during one.
            logger.error("%s", e)
            raise SystemExit(1) from None
        if args.dry_run:
            print(" ".join(command))
            return
        if code != 0:
            logger.error("Acquisition of '%s' failed with status %d", args.name, code)
            raise SystemExit(code)
        logger.info("Installed '%s'", args.name)


def _write_sycl_solutions(args: argparse.Namespace, trace_set) -> None:
    """Expand the in-tree SYCL kernel templates over every definition they implement."""
    from flashinfer_bench.integration import intree_kernels as intree

    wanted = args.definitions.split(",") if args.definitions else None
    written = 0
    near_misses: List[str] = []
    for name, definition in sorted(trace_set.definitions.items()):
        if wanted is not None and name not in wanted:
            continue
        solutions = intree.make_intree_solutions(definition)
        if not solutions:
            near_misses.extend(f"{name}: {r}" for r in intree.explain_no_match(definition))
            continue
        out_dir = Path(args.local) / "solutions" / intree.AUTHOR / definition.op_type / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for solution in solutions:
            path = out_dir / f"{_safe_path_segment(solution.name)}.json"
            if args.dry_run:
                logger.info(
                    f"{name}: would add {solution.spec.language.value} solution '{solution.name}'"
                )
            else:
                save_json_file(solution, path)
                logger.info(
                    f"{name}: added {solution.spec.language.value} solution '{solution.name}'"
                )
            written += 1

    verb = "Would write" if args.dry_run else "Wrote"
    logger.info(f"{verb} {written} in-tree solution(s)")
    if near_misses:
        logger.warning(
            "%d definition(s) share an op_type with an in-tree kernel but do not match:",
            len(near_misses),
        )
        for line in near_misses[:10]:
            logger.warning("  %s", line)


def add_baselines(args: argparse.Namespace) -> None:
    """Add upstream vLLM / SGLang Intel kernels to a dataset as benchmark baselines.

    Beating PyTorch eager is the easy bar; the bar that matters on Intel is beating the
    SYCL kernels those projects already ship. This writes them into the dataset as ordinary
    Solutions so a normal benchmark run measures them side by side with everything else.
    """
    from flashinfer_bench.integration import available_providers, make_baseline_solutions
    from flashinfer_bench.integration.xpu_kernels import explain_no_match, registry_op_types

    trace_set = TraceSet.from_path(str(args.local))
    providers = args.providers.split(",") if args.providers else None

    if args.in_tree:
        # In-tree SYCL kernels are generated from templates rather than wrapped, so they
        # need no provider installed and are the baseline every definition can have.
        _write_sycl_solutions(args, trace_set)
        return

    installed = available_providers()
    if not installed:
        # Not an error: an Intel box with no upstream library installed is the normal
        # starting state, and the rest of the pipeline works without baselines.
        logger.warning(
            "No upstream Intel kernel library installed, so there are no baselines to "
            "add. Install vllm-xpu-kernels (pip install vllm-xpu-kernels) or "
            "sgl-kernel-xpu (github.com/sgl-project/sgl-kernel-xpu)."
        )
        return
    logger.info(f"Available kernel providers: {', '.join(installed)}")

    wanted = args.definitions.split(",") if args.definitions else None
    written = 0
    unmatched = []
    near_misses: List[str] = []

    for name, definition in sorted(trace_set.definitions.items()):
        if wanted is not None and name not in wanted:
            continue
        solutions = make_baseline_solutions(definition, providers)
        if not solutions:
            unmatched.append(name)
            near_misses.extend(f"{name}: {r}" for r in explain_no_match(definition, providers))
            continue
        out_dir = Path(args.local) / "solutions" / "baseline" / definition.op_type
        out_dir.mkdir(parents=True, exist_ok=True)
        for solution in solutions:
            path = out_dir / f"{_safe_path_segment(solution.name)}.json"
            if args.dry_run:
                logger.info(f"{name}: would add baseline '{solution.name}' -> {path}")
            else:
                save_json_file(solution, path)
                logger.info(f"{name}: added baseline '{solution.name}'")
            written += 1

    verb = "Would write" if args.dry_run else "Wrote"
    logger.info(f"{verb} {written} baseline solution(s)")

    if unmatched:
        logger.info(
            f"No upstream kernel matches {len(unmatched)} definition(s): "
            f"{', '.join(unmatched[:10])}" + (" ..." if len(unmatched) > 10 else "")
        )
    # A signature mismatch and an absent provider otherwise look identical from the
    # outside, which is how a registry that matches nothing goes unnoticed.
    if near_misses:
        logger.warning(
            "%d definition(s) share an op_type with a registered kernel but differ in signature:",
            len(near_misses),
        )
        for line in near_misses[:10]:
            logger.warning("  %s", line)
        if len(near_misses) > 10:
            logger.warning("  ... and %d more", len(near_misses) - 10)
    elif written == 0:
        served = registry_op_types(providers)
        present = sorted({d.op_type for d in trace_set.definitions.values()})
        logger.warning(
            "Nothing matched and nothing came close. Registered op_types: %s. "
            "Dataset op_types: %s.",
            ", ".join(served) or "(none)",
            ", ".join(present) or "(none)",
        )
    if written == 0:
        logger.warning(
            "Nothing written. Baselines only apply to definitions whose signature matches "
            "an upstream kernel exactly; see flashinfer_bench/integration/xpu_kernels.py."
        )


def validate_references(args: argparse.Namespace) -> None:
    """Cross-validate reference implementations on a device before benchmarking it.

    A solution's correctness is judged against the definition's reference running on the
    same device, so a reference that is wrong on new hardware would validate wrong
    solutions. This runs each reference on the target device and on the host with
    identical inputs and reports which definitions are cleared.

    Exits non-zero if any definition fails, so CI can gate on it.
    """
    import json

    from flashinfer_bench.bench import check_references
    from flashinfer_bench.device import get_accelerator, list_devices

    trace_set = TraceSet.from_path(str(args.local))

    device = args.device
    if device is None:
        devices = list_devices()
        if not devices:
            raise RuntimeError(
                "No devices available. Set FIB_DEVICE_BACKEND to select a backend "
                "explicitly (cuda, xpu, cpu)."
            )
        device = devices[0]

    raw_overrides = {
        "num_trials": args.num_trials,
        "rtol": args.rtol,
        "atol": args.atol,
        "required_matched_ratio": args.required_matched_ratio,
    }
    overrides = {k: v for k, v in raw_overrides.items() if v is not None}
    config = BenchmarkConfig.default(**overrides)

    definitions = args.definitions.split(",") if args.definitions else None

    accelerator = get_accelerator(device)
    logger.info(
        "Cross-validating references on %s (%s) against %s",
        device,
        accelerator.canonical_id(device),
        "cpu",
    )

    attestation = check_references(trace_set, device, config, definitions)

    cleared, quarantined = attestation.cleared, attestation.quarantined
    logger.info("Cleared %d definition(s) on %s", len(cleared), attestation.hardware)
    if quarantined:
        logger.error(
            "Quarantined %d definition(s) on %s -- these must not be benchmarked there:",
            len(quarantined),
            attestation.hardware,
        )
        for result in attestation.results:
            if not result.ok:
                logger.error(
                    "  %-44s %-18s %s", result.definition, result.status.value, result.detail or ""
                )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(attestation.to_dict(), indent=2) + "\n")
        logger.info("Attestation written to %s", output)

    if quarantined:
        raise SystemExit(1)


def _load_traces(args: argparse.Namespace) -> List[TraceSet]:
    trace_sets = []
    if not args.local:
        raise ValueError("A data source is required. Please use --local <PATH>.")

    if args.local:
        loaded_paths: List[Path] = args.local
        for path in loaded_paths:
            trace_sets.append(TraceSet.from_path(str(path)))
    return trace_sets


def _add_validate_run_args(parser):
    """Add shared arguments for validate / validate run."""
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset root directory. Defaults to FIB_DATASET_PATH env var.",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default=None,
        help="Comma-separated check categories: layout,definition,workload,solution,trace,baseline,benchmark",
    )
    parser.add_argument(
        "--disable-gpu", action="store_true", help="Skip benchmark checks (GPU-dependent)."
    )
    parser.add_argument(
        "--op-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter by op_type (union with --definitions).",
    )
    parser.add_argument(
        "--definitions",
        type=str,
        nargs="+",
        default=None,
        help="Filter by definition name (union with --op-types).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Report output folder. Defaults to <dataset>/reports/.",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="stdout,json,text",
        help="Comma-separated output targets: stdout,json,text (default: all).",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )


def _validate_run(args: argparse.Namespace):
    from flashinfer_bench.data.validate import validate_dataset

    checks = args.checks.split(",") if args.checks else None
    outputs = args.outputs.split(",") if args.outputs else ["stdout", "json", "text"]
    report = validate_dataset(
        dataset=args.dataset,
        op_types=args.op_types,
        definitions=args.definitions,
        checks=checks,
        disable_gpu=args.disable_gpu,
        output_folder=args.output_folder,
        outputs=outputs,
    )

    # This command is the documented pre-flight gate before opening a PR, so its exit code
    # has to mean something: discarding the report made every invocation succeed, including
    # ones that printed errors. Two distinct ways to "pass while checking nothing" are
    # failures in their own right -- a name filter that matched no definition, and a check
    # selection that left nothing to run -- because both look identical to a clean report.
    entries = getattr(report, "definitions", None) or {}
    requested = list(args.definitions or []) + list(args.op_types or [])
    if requested and not entries:
        logger.error(
            "No definition matched %s. Note that --definitions here takes space-separated "
            "names, not a comma-separated list.",
            requested,
        )
        raise SystemExit(2)

    errors = sum(1 for e in entries.values() if _entry_status(e) == "error")
    if errors:
        logger.error("%d definition(s) reported errors.", errors)
        raise SystemExit(1)


def _entry_status(entry: Any) -> str:
    """Worst level across a definition's checks, however the report models them."""
    worst = "ok"
    checks = entry.values() if isinstance(entry, dict) else vars(entry).values()
    for check in checks:
        messages = None
        if isinstance(check, dict):
            messages = check.get("messages")
        elif hasattr(check, "messages"):
            messages = check.messages
        for message in messages or []:
            level = (
                message.get("level")
                if isinstance(message, dict)
                else getattr(message, "level", None)
            )
            if level == "error":
                return "error"
            if level == "warning":
                worst = "warning"
    return worst


def _validate_render(args: argparse.Namespace):
    from flashinfer_bench.data.validate_render import render_report

    render_report(report_json=args.report_json, output=args.output)


def cli():
    parser = argparse.ArgumentParser(
        description="FlashInfer Bench CLI", formatter_class=argparse.RawTextHelpFormatter
    )

    command_subparsers = parser.add_subparsers(
        dest="command", required=True, help="Primary commands"
    )

    serve_parser = command_subparsers.add_parser("serve", help="Start the benchmark HTTP server.")
    serve_parser.add_argument(
        "--local", type=Path, required=True, help="Path to the trace set dataset."
    )
    serve_parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices (e.g. cuda:0,cuda:1 or xpu:0). Default: all available.",
    )
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--warmup-runs", type=int, default=10)
    serve_parser.add_argument("--iterations", type=int, default=50)
    serve_parser.add_argument("--num-trials", type=int, default=3)
    serve_parser.add_argument("--rtol", type=float, default=1e-2)
    serve_parser.add_argument("--atol", type=float, default=1e-2)
    serve_parser.add_argument("--timeout", type=int, default=300)
    serve_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    serve_parser.set_defaults(func=serve)

    run_parser = command_subparsers.add_parser("run", help="Execute a new benchmark run.")
    run_parser.add_argument(
        "--warmup-runs", type=int, default=None, help="Number of warmup runs before measurement"
    )
    run_parser.add_argument(
        "--iterations", type=int, default=None, help="Number of iterations for benchmarking"
    )
    run_parser.add_argument(
        "--num-trials", type=int, default=None, help="Number of trials for each benchmark"
    )
    run_parser.add_argument(
        "--rtol", type=float, default=None, help="Relative tolerance for correctness checks"
    )
    run_parser.add_argument(
        "--atol", type=float, default=None, help="Absolute tolerance for correctness checks"
    )
    run_parser.add_argument(
        "--required-matched-ratio",
        type=float,
        default=None,
        help="Required ratio of elements within tolerance. Overrides evaluator default (0.95 for low-bit).",
    )
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    run_parser.add_argument(
        "--use-isolated-runner",
        action="store_true",
        default=None,
        help="Use IsolatedRunner instead of the default PersistentRunner",
    )
    run_parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume run, skip already evaluated solution-workload jobs",
    )
    run_parser.add_argument(
        "--definitions",
        type=str,
        nargs="+",
        help="List of definition names to run. If not specified, runs all definitions.",
    )
    run_parser.add_argument(
        "--solutions",
        type=str,
        nargs="+",
        help="List of solution names to run. If not specified, runs all solutions.",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each solution evaluation (default: 300)",
    )
    run_parser.add_argument(
        "--no-profile-baseline",
        dest="profile_baseline",
        action="store_false",
        default=None,
        help="Skip profiling the reference implementation (correctness check still runs). "
        "Useful when the reference is slow (e.g. large prefill workloads).",
    )
    run_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark config YAML file. Overrides default eval config.",
    )
    run_parser.set_defaults(func=run)

    report_parser = command_subparsers.add_parser(
        "report", help="Analyze and manage existing traces."
    )
    report_subparsers = report_parser.add_subparsers(
        dest="report_subcommand", required=True, help="Report actions"
    )

    summary_parser = report_subparsers.add_parser(
        "summary", help="Prints a human-readable summary of loaded traces."
    )
    summary_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    summary_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    summary_parser.set_defaults(func=summary)

    best_parser = report_subparsers.add_parser("best", help="Find best solution for a definition.")
    best_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    best_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    best_parser.set_defaults(func=best)

    merge_parser = report_subparsers.add_parser("merge", help="Merges multiple traces.")
    merge_parser.add_argument("--output", type=Path)
    merge_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    merge_parser.set_defaults(func=merge)

    visualize_parser = report_subparsers.add_parser(
        "visualize", help="Generates a visual representation of benchmark results."
    )
    visualize_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    visualize_parser.set_defaults(func=visualize)

    # --- validate ---
    ref_parser = command_subparsers.add_parser(
        "validate-references",
        help="Cross-validate reference implementations on a device against the host.",
    )
    ref_parser.add_argument(
        "--local", type=Path, required=True, help="Path to the trace set dataset."
    )
    ref_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to validate (e.g. xpu:0). Default: first available.",
    )
    ref_parser.add_argument(
        "--definitions", type=str, default=None, help="Comma-separated definition names."
    )
    ref_parser.add_argument(
        "--output", type=str, default=None, help="Path to write the attestation JSON."
    )
    ref_parser.add_argument("--num-trials", type=int, default=None)
    ref_parser.add_argument("--rtol", type=float, default=None)
    ref_parser.add_argument("--atol", type=float, default=None)
    ref_parser.add_argument("--required-matched-ratio", type=float, default=None)
    ref_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    ref_parser.set_defaults(func=validate_references)

    baselines_parser = command_subparsers.add_parser(
        "add-baselines",
        help="Add upstream vLLM / SGLang Intel kernels to a dataset as benchmark baselines.",
    )
    baselines_parser.add_argument(
        "--local", type=Path, required=True, help="Path to the trace set dataset."
    )
    baselines_parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="Comma-separated providers (vllm-xpu, sgl-kernel-xpu). Default: all installed.",
    )
    baselines_parser.add_argument(
        "--definitions", type=str, default=None, help="Comma-separated definition names."
    )
    baselines_parser.add_argument(
        "--in-tree",
        dest="in_tree",
        action="store_true",
        help=(
            "Generate in-tree solutions (SYCL and Triton) from kernel templates instead "
            "of wrapping an upstream library. Needs no provider installed."
        ),
    )
    baselines_parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be written without writing it."
    )
    baselines_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    baselines_parser.set_defaults(func=add_baselines)

    providers_parser = command_subparsers.add_parser(
        "providers", help="List or install the Intel kernel providers."
    )
    provider_sub = providers_parser.add_subparsers(dest="provider_action")
    provider_sub.add_parser("list", help="Show every provider and whether it is present.")
    verify_parser = provider_sub.add_parser(
        "verify", help="Call every registered upstream kernel once, to prove it is really built."
    )
    verify_parser.add_argument(
        "--local", type=Path, required=True, help="Path to the trace set dataset."
    )
    verify_parser.add_argument("--device", type=str, default="xpu:0")

    install_parser = provider_sub.add_parser("install", help="Acquire one provider.")
    install_parser.add_argument("name", type=str, help="Provider name, e.g. vllm-xpu.")
    install_parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="SYCL target for a source build (bmg, cri). Default: derived from the device.",
    )
    install_parser.add_argument(
        "--device", type=str, default=None, help="Device whose architecture to build for."
    )
    install_parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without running it."
    )
    providers_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    providers_parser.set_defaults(func=providers_cmd)

    validate_parser = command_subparsers.add_parser(
        "validate", help="Validate dataset correctness and completeness."
    )
    _add_validate_run_args(validate_parser)
    validate_parser.set_defaults(func=_validate_run)

    # --- validate-render ---
    render_parser = command_subparsers.add_parser(
        "validate-render", help="Render an existing validation report JSON as text."
    )
    render_parser.add_argument("report_json", type=str, help="Path to report JSON file.")
    render_parser.add_argument(
        "--output", type=str, default=None, help="Path to write the text output."
    )
    render_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    render_parser.set_defaults(func=_validate_render)

    args = parser.parse_args()

    cli_config_logging(args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
