"""Dataset-level checks that timed traces carry enough metadata to be compared.

A latency only means something alongside the hardware it ran on and the methodology that
measured it. These checks are the backstop that makes that enforceable against
contributions, not just against our own runs.
"""

from pathlib import Path

from flashinfer_bench.data import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Trace,
    Workload,
)
from flashinfer_bench.data.validate import ScannedTrace, _check_trace_comparability


def _trace(
    *, timing: str | None = "cupti", hardware_id: str | None = "NVIDIA_H100", timed: bool = True
) -> Trace:
    libs = {"timing": timing} if timing else {}
    status = EvaluationStatus.PASSED if timed else EvaluationStatus.RUNTIME_ERROR
    return Trace(
        definition="cmp_def",
        workload=Workload(axes={"n": 4}, inputs={"x": RandomInput()}, uuid="wl-cmp"),
        solution="sol",
        evaluation=Evaluation(
            status=status,
            environment=Environment(hardware="Raw Name", hardware_id=hardware_id, libs=libs),
            timestamp="2026-09-04T00:00:00Z",
            correctness=Correctness() if timed else None,
            performance=(
                Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0)
                if timed
                else None
            ),
        ),
    )


def _entry(*traces: Trace) -> ScannedTrace:
    return ScannedTrace(
        author="tester",
        op_type="gemm",
        definition_name="cmp_def",
        path=Path("traces/gemm/cmp_def.jsonl"),
        traces=list(traces),
    )


def _levels(messages):
    return {(m.level, m.message) for m in messages}


class TestTraceComparability:
    def test_complete_traces_produce_no_messages(self):
        assert _check_trace_comparability([_entry(_trace())]) == []

    def test_missing_timing_methodology_is_flagged(self):
        messages = _check_trace_comparability([_entry(_trace(timing=None))])
        assert any(m.level == "warning" and "timing methodology" in m.message for m in messages)

    def test_missing_hardware_id_is_flagged(self):
        messages = _check_trace_comparability([_entry(_trace(hardware_id=None))])
        assert any(m.level == "warning" and "hardware id" in m.message for m in messages)

    def test_mixed_timing_methodologies_is_an_error(self):
        """CUPTI and device-event latencies measure different things."""
        messages = _check_trace_comparability(
            [_entry(_trace(timing="cupti"), _trace(timing="event"))]
        )
        errors = [m for m in messages if m.level == "error"]
        assert errors and "not comparable" in errors[0].message

    def test_multiple_hardware_is_reported_but_not_an_error(self):
        """A dataset spanning devices is normal; ranking across them is not."""
        messages = _check_trace_comparability(
            [_entry(_trace(hardware_id="NVIDIA_H100"), _trace(hardware_id="INTEL_ARC_B580"))]
        )
        assert not [m for m in messages if m.level == "error"]
        assert any(m.level == "info" and "grouped by hardware" in m.message for m in messages)

    def test_untimed_traces_are_not_examined(self):
        """A failed evaluation has no latency, so it has nothing to be comparable with."""
        assert _check_trace_comparability([_entry(_trace(timed=False, timing=None))]) == []

    def test_unparsable_entries_are_skipped(self):
        broken = ScannedTrace(
            author="tester",
            op_type="gemm",
            definition_name="cmp_def",
            path=Path("traces/gemm/cmp_def.jsonl"),
            traces=[],
            error="parse failure",
        )
        assert _check_trace_comparability([broken]) == []
