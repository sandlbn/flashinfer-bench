"""Tests for reference cross-validation (the gate that precedes any new-hardware run)."""

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.reference_check import (
    ReferenceCheckStatus,
    check_reference,
    check_references,
    required_dtypes,
    unsupported_dtypes,
)
from flashinfer_bench.data import Definition, RandomInput, Trace, TraceSet, Workload

REFERENCE = "import torch\n\n\ndef run(x, y):\n    return x + y\n"


def _definition(name: str = "refcheck_add", dtype: str = "float32") -> Definition:
    return Definition(
        name=name,
        description="reference cross-check fixture",
        op_type="gemm",
        axes={"n": {"type": "var"}},
        inputs={"x": {"shape": ["n"], "dtype": dtype}, "y": {"shape": ["n"], "dtype": dtype}},
        outputs={"out": {"shape": ["n"], "dtype": dtype}},
        reference=REFERENCE,
    )


def _workload(n: int = 8, uuid: str = "wl-refcheck") -> Workload:
    return Workload(axes={"n": n}, inputs={"x": RandomInput(), "y": RandomInput()}, uuid=uuid)


def _cfg(**overrides):
    definition = _definition()
    return BenchmarkConfig.default(num_trials=1, **overrides).resolve_eval_config(definition)


def _trace_set(*definitions: Definition) -> TraceSet:
    defs = {d.name: d for d in definitions}
    workloads = {
        d.name: [Trace(definition=d.name, workload=_workload(uuid=f"wl-{d.name}"))]
        for d in definitions
    }
    return TraceSet(definitions=defs, workloads=workloads)


class TestDtypeGating:
    def test_collects_input_and_output_dtypes(self):
        definition = _definition(dtype="bfloat16")
        assert required_dtypes(definition) == frozenset({"bfloat16"})

    def test_cpu_supports_ordinary_dtypes(self, tmp_cache_dir):
        assert unsupported_dtypes(_definition(dtype="float32"), "cpu") == frozenset()

    def test_flags_dtype_the_device_lacks(self, tmp_cache_dir):
        # The CPU capability record does not claim FP8, so an FP8 definition must be
        # reported as unsupported rather than benchmarked.
        definition = _definition(dtype="float8_e4m3fn")
        assert unsupported_dtypes(definition, "cpu") == frozenset({"float8_e4m3fn"})

    def test_unsupported_dtype_short_circuits_the_check(self, tmp_cache_dir):
        definition = _definition(dtype="float8_e4m3fn")
        result = check_reference(definition, _workload(), "cpu", _cfg())
        assert result.status is ReferenceCheckStatus.UNSUPPORTED_DTYPE
        assert not result.ok
        assert "float8_e4m3fn" in (result.detail or "")


class TestCheckReference:
    def test_matching_reference_is_cleared(self, tmp_cache_dir):
        result = check_reference(_definition(), _workload(), "cpu", _cfg())
        assert result.status is ReferenceCheckStatus.PASSED
        assert result.ok
        assert result.max_absolute_error == 0.0
        assert result.workload_uuid == "wl-refcheck"

    def test_records_the_hardware_it_ran_on(self, tmp_cache_dir):
        result = check_reference(_definition(), _workload(), "cpu", _cfg())
        assert result.hardware  # canonical id, never empty
        assert result.device == "cpu"

    def test_numerical_divergence_is_caught(self, tmp_cache_dir, monkeypatch):
        """A device whose reference computes something slightly different must not pass."""
        import flashinfer_bench.bench.reference_check as rc

        real_run = rc._run_reference
        calls = {"n": 0}

        def perturbed(runnable, definition, inputs, device):
            out = real_run(runnable, definition, inputs, device)
            calls["n"] += 1
            # Perturb only the second (device) run of each trial.
            if calls["n"] % 2 == 0:
                return [t + 1.0 for t in out]
            return out

        monkeypatch.setattr(rc, "_run_reference", perturbed)
        result = check_reference(_definition(), _workload(), "cpu", _cfg())
        assert result.status is ReferenceCheckStatus.MISMATCH
        assert not result.ok
        assert result.max_absolute_error > 0

    def test_shape_divergence_is_caught(self, tmp_cache_dir, monkeypatch):
        import flashinfer_bench.bench.reference_check as rc

        real_run = rc._run_reference
        calls = {"n": 0}

        def reshaped(runnable, definition, inputs, device):
            out = real_run(runnable, definition, inputs, device)
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return [t[:-1] for t in out]
            return out

        monkeypatch.setattr(rc, "_run_reference", reshaped)
        result = check_reference(_definition(), _workload(), "cpu", _cfg())
        assert result.status is ReferenceCheckStatus.MISMATCH
        assert "shape differs" in (result.detail or "")

    def test_device_failure_is_reported_not_swallowed(self, tmp_cache_dir, monkeypatch):
        import flashinfer_bench.bench.reference_check as rc

        real_run = rc._run_reference
        calls = {"n": 0}

        def failing(runnable, definition, inputs, device):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("operator not implemented for this backend")
            return real_run(runnable, definition, inputs, device)

        monkeypatch.setattr(rc, "_run_reference", failing)
        result = check_reference(_definition(), _workload(), "cpu", _cfg())
        assert result.status is ReferenceCheckStatus.TARGET_ERROR
        assert "not implemented" in (result.detail or "")

    def test_broken_reference_is_a_build_error(self, tmp_cache_dir):
        definition = _definition(name="refcheck_broken")
        broken = definition.model_copy(
            update={"reference": "import torch\n\n\ndef run(x, y):\n    return x + y\n    ("}
        )
        result = check_reference(broken, _workload(), "cpu", _cfg())
        assert result.status is ReferenceCheckStatus.BUILD_ERROR
        assert not result.ok


class TestCheckReferences:
    def test_attestation_separates_cleared_from_quarantined(self, tmp_cache_dir):
        good = _definition(name="refcheck_good")
        fp8 = _definition(name="refcheck_fp8", dtype="float8_e4m3fn")
        attestation = check_references(
            _trace_set(good, fp8), "cpu", BenchmarkConfig.default(num_trials=1)
        )

        assert attestation.cleared == ["refcheck_good"]
        assert attestation.quarantined == ["refcheck_fp8"]
        assert attestation.backend == "cpu"
        assert attestation.torch_version == torch.__version__

    def test_definition_without_workload_is_not_silently_cleared(self, tmp_cache_dir):
        definition = _definition(name="refcheck_no_wl")
        trace_set = TraceSet(definitions={definition.name: definition}, workloads={})
        attestation = check_references(trace_set, "cpu", BenchmarkConfig.default(num_trials=1))

        assert attestation.cleared == []
        assert attestation.quarantined == ["refcheck_no_wl"]
        assert attestation.results[0].status is ReferenceCheckStatus.NO_WORKLOAD

    def test_can_restrict_to_named_definitions(self, tmp_cache_dir):
        a = _definition(name="refcheck_a")
        b = _definition(name="refcheck_b")
        attestation = check_references(
            _trace_set(a, b),
            "cpu",
            BenchmarkConfig.default(num_trials=1),
            definitions=["refcheck_a"],
        )
        assert [r.definition for r in attestation.results] == ["refcheck_a"]

    def test_attestation_serializes_to_plain_json_types(self, tmp_cache_dir):
        import json

        attestation = check_references(
            _trace_set(_definition(name="refcheck_json")),
            "cpu",
            BenchmarkConfig.default(num_trials=1),
        )
        payload = json.loads(json.dumps(attestation.to_dict()))
        assert payload["results"][0]["status"] == "PASSED"
        assert payload["cleared"] == ["refcheck_json"]
