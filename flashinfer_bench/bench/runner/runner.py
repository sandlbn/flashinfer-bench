"""Abstract base class and common types for benchmark runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.data import Definition, Evaluation, Solution, Workload


class RunnerError(RuntimeError): ...


class RunnerFatalError(RunnerError): ...


class BaselineHandle(str):
    pass


@dataclass
class DeviceBaseline:
    handle: BaselineHandle
    definition: Definition
    device: str
    inputs: List[List[Any]]
    outputs: List[List[torch.Tensor]]
    mean_latency_ms: float


class Runner(ABC):
    def __init__(self, logger: logging.Logger) -> None: ...

    @abstractmethod
    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solutions: List[Solution],
        config: BenchmarkConfig,
        root: Path,
    ) -> Dict[str, Evaluation]: ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources and terminate worker processes."""
        ...


def _map_tensors(value: Any, fn: Any) -> Any:
    """Apply ``fn`` to every tensor inside a nested list/tuple/dict, preserving shape."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


def needs_host_transport(device: str) -> bool:
    """Whether tensors on ``device`` must go through host memory to reach a worker.

    CUDA has IPC handles and CPU has shared memory, so their tensors can be sent to a
    worker process directly. Intel XPU has neither in torch today -- pickling an XPU
    tensor raises ``_share_fd_: only available on CPU`` -- so its baselines are copied to
    the host for the crossing and restored on the far side.
    """
    from flashinfer_bench.device import get_accelerator

    try:
        return not get_accelerator(device).capabilities(device).supports_ipc_tensors
    except Exception:
        return False


def to_transport(value: Any, device: str) -> Any:
    """Prepare tensors for the crossing into a worker process.

    A no-op on devices whose tensors can be shared directly, so the CUDA path is
    unchanged and pays no copy.
    """
    if not needs_host_transport(device):
        return value
    return _map_tensors(value, lambda t: t.detach().to("cpu"))


def from_transport(value: Any, device: str) -> Any:
    """Restore tensors received from another process onto ``device``."""
    if not needs_host_transport(device):
        return value
    return _map_tensors(value, lambda t: t.to(device))
