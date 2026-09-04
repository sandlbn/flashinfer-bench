"""Random inputs must be reproducible and backend-independent.

This is a prerequisite for cross-validating a reference implementation on a new
accelerator: if two backends generate different tensors for the same workload, comparing
their reference outputs proves nothing. Generation therefore happens on the host from a
workload-derived seed, and only the finished tensors are copied to the device.
"""

import pytest
import torch

from flashinfer_bench.bench.utils import _workload_seed, gen_inputs
from flashinfer_bench.data import Definition, RandomInput, Workload


def _definition() -> Definition:
    return Definition(
        name="det_test",
        description="determinism fixture",
        op_type="gemm",
        axes={"m": {"type": "var"}, "n": {"type": "var"}},
        inputs={
            "x": {"shape": ["m", "n"], "dtype": "float32"},
            "idx": {"shape": ["m"], "dtype": "int32"},
        },
        outputs={"out": {"shape": ["m", "n"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x, idx):\n    return x\n",
    )


def _workload(m: int = 4, n: int = 8, uuid: str = "det-wl") -> Workload:
    return Workload(
        axes={"m": m, "n": n}, inputs={"x": RandomInput(), "idx": RandomInput()}, uuid=uuid
    )


class TestWorkloadSeed:
    def test_is_stable_across_calls(self):
        defn, wl = _definition(), _workload()
        assert _workload_seed(defn, wl, 0) == _workload_seed(defn, wl, 0)

    def test_differs_per_trial(self):
        defn, wl = _definition(), _workload()
        seeds = {_workload_seed(defn, wl, t) for t in range(8)}
        assert len(seeds) == 8

    def test_differs_per_workload(self):
        defn = _definition()
        assert _workload_seed(defn, _workload(4, 8), 0) != _workload_seed(defn, _workload(5, 8), 0)

    def test_differs_per_workload_identity(self):
        """Two workloads with identical axes are still distinct workloads."""
        defn = _definition()
        a = _workload(uuid="wl-a")
        b = _workload(uuid="wl-b")
        assert _workload_seed(defn, a, 0) != _workload_seed(defn, b, 0)

    def test_is_independent_of_axis_ordering(self):
        defn = _definition()
        inputs = {"x": RandomInput(), "idx": RandomInput()}
        a = Workload(axes={"m": 4, "n": 8}, inputs=inputs, uuid="det-wl")
        b = Workload(axes={"n": 8, "m": 4}, inputs=inputs, uuid="det-wl")
        assert _workload_seed(defn, a, 0) == _workload_seed(defn, b, 0)


class TestGenInputs:
    def test_same_trial_reproduces_identical_tensors(self):
        defn, wl = _definition(), _workload()
        first = gen_inputs(defn, wl, device="cpu", trial=0)
        second = gen_inputs(defn, wl, device="cpu", trial=0)
        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_different_trials_produce_different_data(self):
        defn, wl = _definition(), _workload()
        first = gen_inputs(defn, wl, device="cpu", trial=0)
        second = gen_inputs(defn, wl, device="cpu", trial=1)
        assert not torch.equal(first[0], second[0])

    def test_is_unaffected_by_global_rng_state(self):
        """Seeding is local, so an unrelated torch.rand elsewhere cannot shift inputs."""
        defn, wl = _definition(), _workload()
        torch.manual_seed(1234)
        first = gen_inputs(defn, wl, device="cpu", trial=0)
        torch.manual_seed(9999)
        torch.rand(1000)
        second = gen_inputs(defn, wl, device="cpu", trial=0)
        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_respects_declared_dtypes_and_shapes(self):
        defn, wl = _definition(), _workload(m=4, n=8)
        x, idx = gen_inputs(defn, wl, device="cpu", trial=0)
        assert x.shape == (4, 8) and x.dtype == torch.float32
        assert idx.shape == (4,) and idx.dtype == torch.int32

    @pytest.mark.requires_accelerator
    def test_matches_across_host_and_device(self):
        """The same workload must yield identical values on CPU and on the accelerator."""
        from flashinfer_bench.device import list_devices

        devices = list_devices()
        if not devices or devices[0] == "cpu":
            pytest.skip("no accelerator device available")

        defn, wl = _definition(), _workload()
        on_host = gen_inputs(defn, wl, device="cpu", trial=0)
        on_device = gen_inputs(defn, wl, device=devices[0], trial=0)
        for a, b in zip(on_host, on_device):
            assert torch.equal(a, b.cpu())
