"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.
"""

from __future__ import annotations

from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, List

from flashinfer_bench.compile import Runnable
from flashinfer_bench.device import get_accelerator

# Device-specific lock registry to ensure multiprocess-safe benchmarking
_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()


def _device_lock(device: str) -> LockType:
    """Get or create a multiprocessing lock for the specified device.

    This function maintains a registry of locks per device to ensure that
    benchmarking operations on the same device are serialized, preventing
    interference between concurrent measurements.

    Parameters
    ----------
    device : str
        The device identifier (e.g., "cuda:0", "xpu:0").

    Returns
    -------
    LockType
        A lock object specific to the given device.
    """
    with _registry_lock:
        lock = _device_locks.get(device)
        if lock is None:
            lock = Lock()
            _device_locks[device] = lock
        return lock


def time_runnable(fn: Runnable, args: List[Any], warmup: int, iters: int, device: str) -> float:
    """Time the execution of a value-returning style Runnable kernel.

    The timing strategy is chosen by the device's accelerator backend: CUPTI activity
    tracing on CUDA, device events on Intel XPU, a wall clock on CPU. Because those
    methodologies are not interchangeable, the one used is recorded in the trace
    environment by :func:`flashinfer_bench.utils.env_snapshot`.

    Parameters
    ----------
    fn : Runnable
        The kernel function to benchmark (must be value-returning style).
    args : List[Any]
        List of arguments in definition order.
    warmup : int
        Number of warmup iterations before timing.
    iters : int
        Number of timing iterations to average over.
    device : str
        The device to run the benchmark on.

    Returns
    -------
    float
        The median execution time in milliseconds.
    """
    accelerator = get_accelerator(device)
    timer = accelerator.make_timer(device)

    lock = _device_lock(device)
    with lock:
        return timer.time(fn, args, warmup, iters, device)
