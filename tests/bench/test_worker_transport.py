"""Tests for moving baseline tensors into a worker process.

Benchmark workers run in separate processes and receive the baseline's input and
reference-output tensors over a pipe. CUDA can hand a device tensor across directly via
IPC handles, and CPU via shared memory. Intel XPU can do neither in torch today --
pickling an XPU tensor raises ``_share_fd_: only available on CPU`` -- so those tensors
are routed through host memory instead. That routing must not apply where it is not
needed, or the CUDA path pays a copy it never used to.
"""

from typing import ClassVar, List, Optional

import pytest
import torch

from flashinfer_bench.bench.runner.runner import from_transport, needs_host_transport, to_transport
from flashinfer_bench.device import (
    Accelerator,
    Capabilities,
    register_accelerator,
    unregister_accelerator,
)
from flashinfer_bench.device.accelerator import reset_registry_cache
from flashinfer_bench.device.timer import Timer, WallClockTimer


def _make_backend(name: str, ipc: bool):
    @register_accelerator
    class _Backend(Accelerator):
        type: ClassVar[str] = name
        caps: ClassVar[Capabilities] = Capabilities(
            canonical_id=name.upper(), supports_ipc_tensors=ipc
        )

        @staticmethod
        def is_available() -> bool:
            return True

        def device_count(self) -> int:
            return 1

        def list_devices(self) -> List[str]:
            return [f"{name}:0"]

        def set_device(self, device: str) -> None: ...

        def synchronize(self, device: Optional[str] = None) -> None: ...

        def empty_cache(self) -> None: ...

        def device_name(self, device: str) -> str:
            return name

        def capabilities(self, device: str) -> Capabilities:
            return self.caps

        def make_timer(self, device: str) -> Timer:
            return WallClockTimer()

    return _Backend


@pytest.fixture(autouse=True)
def _test_backends():
    """Register the stand-in backends for this module only.

    Registration is global, so leaving them behind would change which device every later
    test considers the default.
    """
    reset_registry_cache()
    _make_backend("ipcless", ipc=False)
    _make_backend("ipcful", ipc=True)
    yield
    unregister_accelerator("ipcless")
    unregister_accelerator("ipcful")
    reset_registry_cache()


class TestNeedsHostTransport:
    def test_device_without_ipc_needs_it(self):
        assert needs_host_transport("ipcless:0")

    def test_device_with_ipc_does_not(self):
        assert not needs_host_transport("ipcful:0")

    def test_cpu_does_not(self):
        assert not needs_host_transport("cpu")

    def test_unknown_device_does_not_break_the_run(self):
        """A device we cannot classify should not fail; it just gets no rerouting."""
        assert not needs_host_transport("quantum:0")


class TestTransport:
    def test_ipc_capable_device_is_left_untouched(self):
        """No copy on a backend that can share tensors, so CUDA is unaffected."""
        tensors = [[torch.ones(4)], [torch.zeros(2)]]
        assert to_transport(tensors, "ipcful:0") is tensors

    def test_ipc_less_device_routes_tensors_through_the_host(self):
        tensors = [[torch.ones(4)]]
        moved = to_transport(tensors, "ipcless:0")
        assert moved is not tensors
        assert moved[0][0].device.type == "cpu"
        assert torch.equal(moved[0][0], tensors[0][0])

    def test_nested_structures_keep_their_shape(self):
        value = {"a": [torch.ones(2), (torch.zeros(3), 7)], "b": "not a tensor", "c": 1.5}
        moved = to_transport(value, "ipcless:0")
        assert isinstance(moved, dict)
        assert isinstance(moved["a"], list)
        assert isinstance(moved["a"][1], tuple)
        assert moved["a"][1][1] == 7
        assert moved["b"] == "not a tensor"
        assert moved["c"] == 1.5

    def test_non_tensor_values_survive_the_crossing(self):
        original = [[torch.randn(8), 3, None, "s"]]
        moved = to_transport(original, "ipcless:0")
        assert torch.equal(moved[0][0], original[0][0])
        assert moved[0][1] == 3
        assert moved[0][2] is None
        assert moved[0][3] == "s"

    def test_restore_is_a_no_op_on_an_ipc_capable_device(self):
        value = [[torch.ones(4)]]
        assert from_transport(value, "ipcful:0") is value

    def test_transport_detaches_so_graphs_do_not_cross_processes(self):
        t = torch.ones(4, requires_grad=True)
        moved = to_transport([t], "ipcless:0")
        assert not moved[0].requires_grad

    @pytest.mark.requires_torch_xpu
    def test_xpu_tensors_are_routed_through_the_host(self):
        """The real backend must declare it needs this, or workers fail at pickling."""
        assert needs_host_transport("xpu:0")
        moved = to_transport([torch.ones(4, device="xpu:0")], "xpu:0")
        assert moved[0].device.type == "cpu"
        restored = from_transport(moved, "xpu:0")
        assert restored[0].device.type == "xpu"
        assert torch.equal(restored[0].cpu(), moved[0])
