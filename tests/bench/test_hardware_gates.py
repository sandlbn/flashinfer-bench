"""Tests for the gates that keep multi-hardware results honest.

Three separate hazards, three gates:
  * per-hardware tolerance overrides must be explicit and reviewable, not hidden;
  * a definition the hardware cannot execute must be skipped, not benchmarked;
  * results from different devices must not be ranked against each other.
"""

from types import SimpleNamespace

import pytest

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.data import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Trace,
    TraceSet,
    Workload,
)

DEFINITION = SimpleNamespace(op_type="gemm", name="hw_gate_def")


def _trace(
    hardware_id: str,
    speedup: float,
    *,
    solution: str = "sol",
    timing: str = "event",
    raw_hardware: str = "Some Device",
) -> Trace:
    return Trace(
        definition="hw_gate_def",
        workload=Workload(axes={"n": 4}, inputs={"x": RandomInput()}, uuid=f"wl-{solution}"),
        solution=solution,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(
                hardware=raw_hardware, hardware_id=hardware_id, libs={"timing": timing}
            ),
            timestamp="2026-09-04T00:00:00Z",
            correctness=Correctness(max_absolute_error=0.0, max_relative_error=0.0),
            performance=Performance(
                latency_ms=1.0, reference_latency_ms=speedup, speedup_factor=speedup
            ),
        ),
    )


class TestHardwareEvalConfigLayer:
    def test_hardware_layer_applies_when_hardware_matches(self):
        cfg = BenchmarkConfig(hardware_config={"INTEL_ARC_B580": {"rtol": 0.05}})
        assert cfg.resolve_eval_config(DEFINITION, "INTEL_ARC_B580").rtol == 0.05

    def test_hardware_layer_is_inert_for_other_devices(self):
        cfg = BenchmarkConfig(hardware_config={"INTEL_ARC_B580": {"rtol": 0.05}})
        default_rtol = cfg.resolve_eval_config(DEFINITION).rtol
        assert cfg.resolve_eval_config(DEFINITION, "NVIDIA_H100").rtol == default_rtol

    def test_hardware_layer_beats_definition_layer(self):
        cfg = BenchmarkConfig(
            definition_config={"hw_gate_def": {"atol": 0.1}},
            hardware_config={"INTEL_ARC_B580": {"atol": 0.2}},
        )
        assert cfg.resolve_eval_config(DEFINITION, "INTEL_ARC_B580").atol == 0.2

    def test_cli_override_still_beats_hardware_layer(self):
        """A tolerance passed on the command line is never shadowed by config."""
        cfg = BenchmarkConfig(atol=0.5, hardware_config={"INTEL_ARC_B580": {"atol": 0.2}})
        assert cfg.resolve_eval_config(DEFINITION, "INTEL_ARC_B580").atol == 0.5

    def test_unrelated_fields_are_not_disturbed(self):
        cfg = BenchmarkConfig(hardware_config={"INTEL_ARC_B580": {"rtol": 0.05}})
        resolved = cfg.resolve_eval_config(DEFINITION, "INTEL_ARC_B580")
        assert resolved.atol == BenchmarkConfig().resolve_eval_config(DEFINITION).atol

    def test_hardware_extras_merge_into_evaluator_params(self):
        cfg = BenchmarkConfig(hardware_config={"INTEL_ARC_B580": {"extra": {"cold_cache": False}}})
        assert cfg.resolve_eval_config(DEFINITION, "INTEL_ARC_B580").extra["cold_cache"] is False


class TestHardwareGroupedRanking:
    def _trace_set(self, *traces: Trace) -> TraceSet:
        return TraceSet(traces={"hw_gate_def": list(traces)})

    def test_ranks_within_a_single_hardware(self):
        ts = self._trace_set(
            _trace("INTEL_ARC_B580", 1.5, solution="slow"),
            _trace("INTEL_ARC_B580", 3.0, solution="fast"),
        )
        assert ts.get_best_trace("hw_gate_def").solution == "fast"

    def test_refuses_to_rank_across_hardware(self):
        """A speedup on one device is not the same achievement as on another."""
        ts = self._trace_set(
            _trace("INTEL_ARC_B580", 4.0, solution="intel"),
            _trace("NVIDIA_H100", 2.0, solution="nvidia"),
        )
        with pytest.raises(ValueError, match="span multiple hardware"):
            ts.get_best_trace("hw_gate_def")

    def test_hardware_filter_resolves_the_ambiguity(self):
        ts = self._trace_set(
            _trace("INTEL_ARC_B580", 4.0, solution="intel"),
            _trace("NVIDIA_H100", 2.0, solution="nvidia"),
        )
        assert ts.get_best_trace("hw_gate_def", hardware="NVIDIA_H100").solution == "nvidia"
        assert ts.get_best_trace("hw_gate_def", hardware="INTEL_ARC_B580").solution == "intel"

    def test_mixing_can_be_opted_into_explicitly(self):
        ts = self._trace_set(
            _trace("INTEL_ARC_B580", 4.0, solution="intel"),
            _trace("NVIDIA_H100", 2.0, solution="nvidia"),
        )
        best = ts.get_best_trace("hw_gate_def", allow_mixed_hardware=True)
        assert best.solution == "intel"

    def test_unknown_hardware_filter_yields_nothing(self):
        ts = self._trace_set(_trace("INTEL_ARC_B580", 4.0))
        assert ts.get_best_trace("hw_gate_def", hardware="INTEL_XE3P_CRESCENT_ISLAND") is None

    def test_lists_hardware_present_for_a_definition(self):
        ts = self._trace_set(
            _trace("INTEL_ARC_B580", 1.0, solution="a"), _trace("NVIDIA_H100", 1.0, solution="b")
        )
        assert ts.hardware_ids("hw_gate_def") == ["INTEL_ARC_B580", "NVIDIA_H100"]

    def test_legacy_traces_group_by_normalized_raw_name(self):
        """Traces predating hardware_id still group, via the raw driver name."""
        legacy = _trace("", 1.0, raw_hardware="Intel(R) Arc(TM) B580 Graphics")
        legacy.evaluation.environment.hardware_id = None
        assert TraceSet.trace_hardware_id(legacy) == "INTEL_ARC_B580"


class TestUnsupportedDtypeSkip:
    """A definition the hardware cannot execute must be skipped, not benchmarked.

    Running it anyway either fails in a way that reads as a broken kernel, or lands on a
    silently emulated path and gets timed as though it were native.
    """

    def _definition(self, name: str, dtype: str):
        from flashinfer_bench.data import AxisConst, Definition, TensorSpec

        return Definition(
            name=name,
            op_type="gemm",
            axes={"M": AxisConst(value=4)},
            inputs={"A": TensorSpec(shape=["M"], dtype=dtype)},
            outputs={"B": TensorSpec(shape=["M"], dtype=dtype)},
            reference="import torch\n\n\ndef run(A):\n    return A + 1\n",
        )

    def _benchmark(self, monkeypatch, *definitions):
        from unittest.mock import MagicMock

        import flashinfer_bench.bench.benchmark as bm
        from flashinfer_bench.bench import Benchmark

        runner = MagicMock()
        runner._available_devices = ["cpu"]
        runner.run_workload.return_value = {}
        monkeypatch.setattr(bm, "PersistentRunner", lambda: runner)

        trace_set = TraceSet(
            definitions={d.name: d for d in definitions},
            solutions={d.name: [self._solution(d.name)] for d in definitions},
            workloads={
                d.name: [
                    Trace(
                        definition=d.name,
                        workload=Workload(
                            axes={"M": 4}, inputs={"A": RandomInput()}, uuid=f"wl-{d.name}"
                        ),
                    )
                ]
                for d in definitions
            },
        )
        return Benchmark(trace_set, BenchmarkConfig()), runner

    def _solution(self, definition_name: str):
        from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

        return Solution(
            name=f"{definition_name}_sol",
            definition=definition_name,
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
                destination_passing_style=False,
            ),
            sources=[SourceFile(path="main.py", content="def run(A):\n    return A + 1\n")],
        )

    def test_definition_needing_an_absent_dtype_is_not_run(self, monkeypatch, caplog):
        import logging

        definition = self._definition("gate_fp8", "float8_e4m3fn")
        benchmark, runner = self._benchmark(monkeypatch, definition)

        with caplog.at_level(logging.WARNING):
            result = benchmark.run_all(dump_traces=False)

        runner.run_workload.assert_not_called()
        assert result.traces == {}
        assert any("float8_e4m3fn" in r.message for r in caplog.records)

    def test_supported_definition_still_runs(self, monkeypatch):
        definition = self._definition("gate_fp32", "float32")
        benchmark, runner = self._benchmark(monkeypatch, definition)

        benchmark.run_all(dump_traces=False)

        runner.run_workload.assert_called_once()


class TestDeviceEvalDefaults:
    """A backend may recommend eval parameters for itself.

    Applied below every other layer, so it fills in what nothing else specified and never
    shadows an explicit choice.
    """

    def test_backend_recommendation_fills_in_an_unset_value(self):
        from flashinfer_bench.bench.config import EvalConfig

        cfg = BenchmarkConfig()
        resolved = cfg.resolve_eval_config(DEFINITION, None, EvalConfig(warmup_runs=200))
        assert resolved.warmup_runs == 200

    def test_cli_flag_beats_the_backend_recommendation(self):
        from flashinfer_bench.bench.config import EvalConfig

        cfg = BenchmarkConfig(warmup_runs=5)
        resolved = cfg.resolve_eval_config(DEFINITION, None, EvalConfig(warmup_runs=200))
        assert resolved.warmup_runs == 5

    def test_op_type_config_beats_the_backend_recommendation(self):
        from flashinfer_bench.bench.config import EvalConfig

        cfg = BenchmarkConfig(op_type_config={"gemm": {"warmup_runs": 42}})
        resolved = cfg.resolve_eval_config(DEFINITION, None, EvalConfig(warmup_runs=200))
        assert resolved.warmup_runs == 42

    def test_recommendation_does_not_disturb_other_fields(self):
        from flashinfer_bench.bench.config import EvalConfig

        cfg = BenchmarkConfig()
        resolved = cfg.resolve_eval_config(DEFINITION, None, EvalConfig(warmup_runs=200))
        assert resolved.iterations == BenchmarkConfig().resolve_eval_config(DEFINITION).iterations

    def test_cpu_recommends_nothing(self):
        from flashinfer_bench.bench.config import device_eval_defaults

        assert device_eval_defaults("cpu") is None

    def test_unknown_device_is_not_an_error(self):
        from flashinfer_bench.bench.config import device_eval_defaults

        assert device_eval_defaults("quantum:0") is None

    @pytest.mark.requires_torch_xpu
    def test_xpu_recommends_a_longer_warmup(self):
        """Intel GPUs ramp clocks from idle; the generic default is too short."""
        from flashinfer_bench.bench.config import device_eval_defaults

        defaults = device_eval_defaults("xpu:0")
        assert defaults is not None and defaults.warmup_runs == 200
