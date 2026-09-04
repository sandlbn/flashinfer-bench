"""NVIDIA CUDA backend.

Wraps the behavior FlashInfer-Bench had before the device abstraction existed. Changes
here are visible in published NVIDIA traces, so this module deliberately preserves the
prior semantics rather than tidying them.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import ClassVar, Dict, List, Optional

from .accelerator import Accelerator, canonicalize_device_name, parse_device, register_accelerator
from .capabilities import DEFAULT_DTYPES, FP4_DTYPES, FP8_DTYPES, Capabilities
from .timer import CuptiTimer, EventTimer, Timer, cupti_timing_available

logger = logging.getLogger(__name__)

_DEFAULT_L2_BYTES = 40 * 1024 * 1024
"""Fallback last-level cache size when the driver does not report one (A100-class)."""


@register_accelerator
class CudaAccelerator(Accelerator):
    """CUDA devices, timed with CUPTI activity tracing."""

    type: ClassVar[str] = "cuda"

    _cupti_warned: ClassVar[bool] = False

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def device_count(self) -> int:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

    def list_devices(self) -> List[str]:
        return [f"cuda:{i}" for i in range(self.device_count())]

    def _index(self, device: str) -> int:
        index = parse_device(device)[1]
        if index is not None:
            return index
        import torch

        return int(torch.cuda.current_device())

    def set_device(self, device: str) -> None:
        import torch

        torch.cuda.set_device(self._index(device))

    def synchronize(self, device: Optional[str] = None) -> None:
        import torch

        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device=device)

    def empty_cache(self) -> None:
        import torch

        torch.cuda.empty_cache()

    def device_name(self, device: str) -> str:
        import torch

        return str(torch.cuda.get_device_name(self._index(device)))

    def capabilities(self, device: str) -> Capabilities:
        return _cuda_capabilities(self._index(device))

    def make_timer(self, device: str) -> Timer:
        # CUPTI reports device-side kernel time; the event timer includes launch
        # overhead. That is a methodology change, not a detail, so it is chosen up
        # front (and recorded in the trace) rather than fallen into mid-measurement.
        if cupti_timing_available():
            return CuptiTimer()
        if not CudaAccelerator._cupti_warned:
            CudaAccelerator._cupti_warned = True
            logger.warning(
                "flashinfer-python is unavailable, so CUDA kernels are timed with device "
                "events instead of CUPTI. Latencies include launch overhead and are not "
                "comparable to CUPTI numbers; install flashinfer-bench[cuda] for parity."
            )
        import torch

        return EventTimer(torch.cuda, l2_bytes=self.capabilities(device).l2_bytes)

    def lib_versions(self) -> Dict[str, str]:
        libs: Dict[str, str] = {}
        try:
            import torch.version as tv

            if getattr(tv, "cuda", None):
                libs["cuda"] = str(tv.cuda)
        except Exception:
            pass
        return libs


@lru_cache(maxsize=None)
def _cuda_capabilities(index: int) -> Capabilities:
    import torch

    props = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)

    dtypes = set(DEFAULT_DTYPES)
    if (major, minor) >= (8, 9):  # Ada / Hopper introduced FP8 tensor cores
        dtypes |= FP8_DTYPES
    if major >= 10:  # Blackwell introduced FP4
        dtypes |= FP4_DTYPES

    return Capabilities(
        canonical_id=canonicalize_device_name(torch.cuda.get_device_name(index)),
        supported_dtypes=frozenset(dtypes),
        l2_bytes=int(getattr(props, "L2_cache_size", 0) or _DEFAULT_L2_BYTES),
        sycl_target=None,
        supports_graphs=True,
        extra={
            "compute_capability": f"{major}.{minor}",
            "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
            "total_memory_bytes": int(getattr(props, "total_memory", 0)),
        },
    )
