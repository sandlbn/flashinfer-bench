"""Kernel timing strategies.

Each accelerator resolves a :class:`Timer`. Timers are interchangeable in interface but
*not* in methodology: a CUPTI trace reports device-side kernel duration, while an event
timer includes launch overhead. Because those numbers are not directly comparable, every
timer reports a :attr:`Timer.name` that is recorded into the trace environment, so a
stored latency always says how it was measured.
"""

from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, ClassVar, List, Optional, Sequence

logger = logging.getLogger(__name__)


class Timer(ABC):
    """Measures the wall-time of a callable on a specific device."""

    name: ClassVar[str]
    """Identifier for the timing methodology, recorded in the trace environment."""

    @abstractmethod
    def time_all(
        self, fn: Any, args: Sequence[Any], warmup: int, iters: int, device: str
    ) -> List[float]:
        """Return per-iteration latencies in milliseconds."""
        ...

    def time(self, fn: Any, args: Sequence[Any], warmup: int, iters: int, device: str) -> float:
        """Return the median latency in milliseconds."""
        times = self.time_all(fn, args, warmup, iters, device)
        if not times:
            raise RuntimeError(f"{type(self).__name__} produced no timing samples")
        return statistics.median(times)


@lru_cache(maxsize=1)
def cupti_timing_available() -> bool:
    """Whether FlashInfer's CUPTI-backed benchmark helper can be imported.

    Probed once and cached. Callers select a timer *before* running, so a trace never
    claims a methodology that was not actually used.
    """
    try:
        from flashinfer.testing import bench_gpu_time_with_cupti  # noqa: F401
    except Exception as e:
        logger.debug("CUPTI timing unavailable: %s", e)
        return False
    return True


class CuptiTimer(Timer):
    """CUDA timing via CUPTI activity tracing, using FlashInfer's benchmark helper.

    Reports device-side kernel duration and flushes L2 between iterations. Select it
    only when :func:`cupti_timing_available` is true -- it raises rather than silently
    degrading to a different methodology.
    """

    name: ClassVar[str] = "cupti"

    def time_all(
        self, fn: Any, args: Sequence[Any], warmup: int, iters: int, device: str
    ) -> List[float]:
        import torch

        try:
            from flashinfer.testing import bench_gpu_time_with_cupti
        except Exception as e:
            raise RuntimeError(
                "CUPTI timing requires flashinfer-python. Install it with "
                "'pip install flashinfer-bench[cuda]', or select another timer."
            ) from e

        with torch.cuda.device(device):
            return list(
                bench_gpu_time_with_cupti(
                    fn=fn,
                    dry_run_iters=warmup,
                    repeat_iters=iters,
                    input_args=tuple(args),
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                )
            )


class EventTimer(Timer):
    """Device-event timing, portable across any torch accelerator backend.

    Uses ``<backend>.Event(enable_timing=True)`` around each iteration. Latencies include
    launch overhead, which matters for very short kernels. Between iterations a scratch
    buffer is written to evict the last-level cache, so repeated calls do not read a
    warm cache and report an unrealistically low number.
    """

    name: ClassVar[str] = "event"

    def __init__(self, module: Any, l2_bytes: int = 0) -> None:
        """
        Parameters
        ----------
        module : Any
            The torch backend module providing ``Event`` and ``synchronize``
            (``torch.cuda`` or ``torch.xpu``).
        l2_bytes : int
            Last-level cache size to evict between iterations. Zero disables eviction.
        """
        self._module = module
        self._l2_bytes = l2_bytes

    def _make_flush_buffer(self, device: str) -> Optional[Any]:
        if self._l2_bytes <= 0:
            return None
        import torch

        try:
            # int8 so element count == byte count; 2x cache size to defeat any
            # replacement policy that would otherwise retain part of the buffer.
            return torch.empty(2 * self._l2_bytes, dtype=torch.int8, device=device)
        except Exception as e:  # pragma: no cover - depends on free device memory
            logger.debug("Cache-flush buffer allocation failed (%s); timing warm-cache", e)
            return None

    def time_all(
        self, fn: Any, args: Sequence[Any], warmup: int, iters: int, device: str
    ) -> List[float]:
        import torch

        flush = self._make_flush_buffer(device)

        with torch.no_grad():
            for _ in range(warmup):
                fn(*args)
            self._module.synchronize(device)

            times: List[float] = []
            for _ in range(iters):
                if flush is not None:
                    flush.zero_()
                start = self._module.Event(enable_timing=True)
                end = self._module.Event(enable_timing=True)
                start.record()
                fn(*args)
                end.record()
                self._module.synchronize(device)
                times.append(start.elapsed_time(end))

        return times


class WallClockTimer(Timer):
    """Host-side timing for backends without device events (CPU).

    Only meaningful for synchronous execution; on an asynchronous device it measures
    launch time rather than kernel time.
    """

    name: ClassVar[str] = "wallclock"

    def time_all(
        self, fn: Any, args: Sequence[Any], warmup: int, iters: int, device: str
    ) -> List[float]:
        import time

        import torch

        with torch.no_grad():
            for _ in range(warmup):
                fn(*args)

            times: List[float] = []
            for _ in range(iters):
                start = time.perf_counter()
                fn(*args)
                times.append((time.perf_counter() - start) * 1000.0)

        return times
