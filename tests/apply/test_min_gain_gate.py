"""A solution must beat the provider by more than substituting costs, not merely beat it.

Resolving a definition, building a key, checking dtypes and invoking the Runnable cost real
time on every call, and that cost does not shrink with the kernel. Measured at ~5.9us on
Arc B580 against a ~5us elementwise kernel, so a solution can be three times faster than the
provider's and still lose the exchange -- which is what several families measured end to end.
"""

from typing import Dict, List

import pytest

from flashinfer_bench.apply.config import ApplyConfig, ApplyConfigRegistry
from flashinfer_bench.apply.table import ApplyTable
from flashinfer_bench.data import (
    Definition,
    Evaluation,
    EvaluationStatus,
    Solution,
    Trace,
    TraceSet,
    Workload,
)

HIDDEN = 1024


def _definition() -> Definition:
    return Definition.model_validate(
        {
            "name": "norm",
            "description": "d",
            "op_type": "rmsnorm",
            "axes": {
                "batch_size": {"type": "var"},
                "hidden_size": {"type": "const", "value": HIDDEN},
            },
            "inputs": {"x": {"shape": ["batch_size", "hidden_size"], "dtype": "float16"}},
            "outputs": {"out": {"shape": ["batch_size", "hidden_size"], "dtype": "float16"}},
            "reference": "def run(x):\n    return x\n",
        }
    )


def _solution(name: str, author: str) -> Solution:
    return Solution.model_validate(
        {
            "name": name,
            "definition": "norm",
            "author": author,
            "spec": {"language": "python", "entry_point": "m.py::run", "target_hardware": ["cpu"]},
            "sources": [{"path": "m.py", "content": "def run(x):\n    return x\n"}],
        }
    )


def _trace(solution: str, batch: int, latency_ms: float) -> Trace:
    return Trace.model_validate(
        {
            "definition": "norm",
            "solution": solution,
            "workload": {
                "definition": "norm",
                "axes": {"batch_size": batch, "hidden_size": HIDDEN},
                "inputs": {"x": {"type": "random"}},
                "uuid": f"norm-b{batch}",
            },
            "evaluation": {
                "status": "PASSED",
                "timestamp": "2026-09-08T00:00:00",
                "environment": {
                    "hardware": "Fake",
                    "hardware_id": "FAKE_PART",
                    "libs": {"timing": "event-batched"},
                },
                "correctness": {"max_relative_error": 0.0, "max_absolute_error": 0.0},
                "performance": {
                    "latency_ms": latency_ms,
                    "reference_latency_ms": 1.0,
                    "speedup_factor": 1.0 / latency_ms,
                },
            },
        }
    )


def _table(traces: List[Trace], min_gain_us: float) -> ApplyTable:
    ts = TraceSet(
        root="/tmp/fake",
        definitions={"norm": _definition()},
        solutions={
            "norm": [
                _solution("norm__ours", "flashinfer-bench-intree"),
                _solution("norm__vllm_xpu_rms_norm", "vllm-xpu"),
            ]
        },
        traces={"norm": traces},
    )
    registry = ApplyConfigRegistry()
    registry.register(
        "norm",
        ApplyConfig(
            max_atol=1.0, max_rtol=1.0, on_miss_policy="use_def_best", min_gain_us=min_gain_us
        ),
    )
    return ApplyTable._build(ts, registry)


@pytest.fixture(autouse=True)
def _fake_part(monkeypatch):
    monkeypatch.setattr(ApplyTable, "_current_hardware_id", staticmethod(lambda: "FAKE_PART"))


class TestGate:
    def test_a_win_smaller_than_dispatch_is_not_indexed(self):
        """3us saved is a real win over the provider and still a net loss to take."""
        traces = [_trace("norm__ours", 64, 0.005), _trace("norm__vllm_xpu_rms_norm", 64, 0.008)]
        assert _table(traces, 0.0).index.get("norm")  # indexed with the gate off
        assert not _table(traces, 5.91).index.get("norm")  # refused with it on

    def test_a_win_larger_than_dispatch_is_kept(self):
        traces = [_trace("norm__ours", 64, 0.100), _trace("norm__vllm_xpu_rms_norm", 64, 0.400)]
        assert len(_table(traces, 5.91).index.get("norm", {})) == 1

    def test_keys_are_judged_individually(self):
        """Decline at decode sizes, still substitute at prefill sizes."""
        traces = [
            _trace("norm__ours", 64, 0.005),
            _trace("norm__vllm_xpu_rms_norm", 64, 0.008),
            _trace("norm__ours", 8192, 0.100),
            _trace("norm__vllm_xpu_rms_norm", 8192, 0.400),
        ]
        assert len(_table(traces, 5.91).index.get("norm", {})) == 1

    def test_no_provider_baseline_leaves_the_key_alone(self):
        """Nothing to compare against; refusing would disable ungenerated baselines."""
        traces = [_trace("norm__ours", 64, 0.005)]
        assert len(_table(traces, 5.91).index.get("norm", {})) == 1

    def test_def_best_is_withheld_once_any_key_is_rejected(self):
        """Otherwise a miss falls through to def_best and re-substitutes what was rejected.

        The unmeasured shapes a miss covers are the ones most like the rejected ones, so
        extending the surviving winner to them is exactly the wrong extrapolation.
        """
        traces = [
            _trace("norm__ours", 64, 0.005),
            _trace("norm__vllm_xpu_rms_norm", 64, 0.008),
            _trace("norm__ours", 8192, 0.100),
            _trace("norm__vllm_xpu_rms_norm", 8192, 0.400),
        ]
        assert _table(traces, 0.0).def_best.get("norm") is not None
        assert _table(traces, 5.91).def_best.get("norm") is None

    def test_gate_off_by_default(self):
        assert ApplyConfig().min_gain_us == 0.0
