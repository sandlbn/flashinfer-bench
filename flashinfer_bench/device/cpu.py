"""CPU backend.

Not a benchmarking target -- it exists so the runners, evaluators and validation gates
can be exercised end to end with no GPU present, and so a reference implementation can
still run when an accelerator lacks an operator.
"""

from __future__ import annotations

import platform
from functools import lru_cache
from typing import ClassVar, List, Optional

from .accelerator import Accelerator, canonicalize_device_name, register_accelerator
from .capabilities import DEFAULT_DTYPES, Capabilities
from .timer import Timer, WallClockTimer


@register_accelerator
class CpuAccelerator(Accelerator):
    """The host CPU, timed with a wall clock."""

    type: ClassVar[str] = "cpu"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def device_count(self) -> int:
        return 1

    def list_devices(self) -> List[str]:
        return ["cpu"]

    def set_device(self, device: str) -> None:
        return None

    def synchronize(self, device: Optional[str] = None) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def device_name(self, device: str) -> str:
        return _cpu_name()

    def capabilities(self, device: str) -> Capabilities:
        return Capabilities(
            canonical_id=canonicalize_device_name(_cpu_name()),
            supported_dtypes=DEFAULT_DTYPES,
            l2_bytes=0,  # host-side timing gains nothing from a flush buffer
            sycl_target=None,
            supports_graphs=False,
            extra={"platform": platform.machine()},
        )

    def make_timer(self, device: str) -> Timer:
        return WallClockTimer()


@lru_cache(maxsize=None)
def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "CPU"
