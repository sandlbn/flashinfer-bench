"""Intel GPU (XPU) backend -- Battlemage today, Xe3P next.

Both parts share one software stack (``torch.xpu`` / SYCL / Level Zero), so they share
one accelerator. What differs between them is data, not code: :data:`_PART_PROFILES`
maps a canonical device id to its capability overrides. Enabling a new Intel part means
adding a row there.

A part with no row still works. It gets the conservative defaults below, and its SYCL
target stays ``None``, which makes native kernels compile to SPIR-V and JIT at load
time -- the path that lets an unreleased device run on the day it arrives.
"""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar, Dict, List, Optional

from .accelerator import Accelerator, canonicalize_device_name, parse_device, register_accelerator
from .capabilities import DEFAULT_DTYPES, Capabilities
from .timer import EventTimer, Timer

_MIB = 1024 * 1024

_CONSERVATIVE_L2_BYTES = 64 * _MIB
"""Cache-flush size for parts whose last-level cache size we do not know.

Overestimating costs a little benchmark time; underestimating silently produces
warm-cache latencies, so unknown parts round up.
"""

_PART_PROFILES: Dict[str, Dict[str, object]] = {
    # Battlemage (Xe2-HPG), validated hardware.
    "INTEL_ARC_B580": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "intel_gpu_bmg_g21",
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage"},
    },
    "INTEL_ARC_B570": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "intel_gpu_bmg_g21",
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage"},
    },
    "INTEL_ARC_PRO_B50": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "intel_gpu_bmg_g21",
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage"},
    },
    "INTEL_ARC_PRO_B60": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "intel_gpu_bmg_g21",
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage"},
    },
    # Xe3P "Crescent Island". Reserved so traces from early silicon carry a stable id.
    # Cache size, dtype support and the AOT target triple are all still unpublished --
    # the defaults keep it correct-but-conservative until the ISA details land.
    "INTEL_XE3P_CRESCENT_ISLAND": {
        "extra": {"architecture": "Xe3P", "codename": "Crescent Island", "status": "preliminary"}
    },
}
"""Per-part capability overrides, keyed by canonical device id."""


@register_accelerator
class XpuAccelerator(Accelerator):
    """Intel GPUs via the PyTorch XPU backend."""

    type: ClassVar[str] = "xpu"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
        try:
            return bool(xpu.is_available())
        except Exception:
            return False

    def device_count(self) -> int:
        import torch

        if not self.is_available():
            return 0
        return int(torch.xpu.device_count())

    def list_devices(self) -> List[str]:
        return [f"xpu:{i}" for i in range(self.device_count())]

    def _index(self, device: str) -> int:
        index = parse_device(device)[1]
        if index is not None:
            return index
        import torch

        return int(torch.xpu.current_device())

    def set_device(self, device: str) -> None:
        import torch

        torch.xpu.set_device(self._index(device))

    def synchronize(self, device: Optional[str] = None) -> None:
        import torch

        if device is None:
            torch.xpu.synchronize()
        else:
            torch.xpu.synchronize(device=device)

    def empty_cache(self) -> None:
        import torch

        torch.xpu.empty_cache()

    def device_name(self, device: str) -> str:
        import torch

        return str(torch.xpu.get_device_name(self._index(device)))

    def capabilities(self, device: str) -> Capabilities:
        return _xpu_capabilities(self._index(device))

    def make_timer(self, device: str) -> Timer:
        # Tier 1 of the timing plan: device events, cache-flushed between iterations.
        # Includes launch overhead, which is why every trace records its timer name.
        # Tiers 2 and 3 (kineto/PTI, then PTI directly) land once measured on hardware.
        import torch

        return EventTimer(torch.xpu, l2_bytes=self.capabilities(device).l2_bytes)

    def lib_versions(self) -> Dict[str, str]:
        libs: Dict[str, str] = {}
        try:
            import torch.version as tv

            for attr in ("xpu", "sycl"):
                value = getattr(tv, attr, None)
                if value:
                    libs[attr] = str(value)
        except Exception:
            pass
        try:
            import torch

            props = torch.xpu.get_device_properties(torch.xpu.current_device())
            driver = getattr(props, "driver_version", None)
            if driver:
                libs["level_zero_driver"] = str(driver)
        except Exception:
            pass
        return libs


@lru_cache(maxsize=None)
def _xpu_capabilities(index: int) -> Capabilities:
    import torch

    canonical = canonicalize_device_name(torch.xpu.get_device_name(index))
    profile = _PART_PROFILES.get(canonical, {})

    extra: Dict[str, object] = dict(profile.get("extra", {}))  # type: ignore[arg-type]
    try:
        props = torch.xpu.get_device_properties(index)
        for attr in ("gpu_eu_count", "max_compute_units", "total_memory", "max_work_group_size"):
            value = getattr(props, attr, None)
            if value is not None:
                extra.setdefault(attr, value)
    except Exception:
        pass
    if canonical not in _PART_PROFILES:
        extra.setdefault("profile", "unknown-part-defaults")

    return Capabilities(
        canonical_id=canonical,
        # Conservative until measured: a dtype that torch accepts but lowers to an
        # emulated path would otherwise be benchmarked as if it were native.
        supported_dtypes=frozenset(profile.get("supported_dtypes", DEFAULT_DTYPES)),  # type: ignore[arg-type]
        l2_bytes=int(profile.get("l2_bytes", _CONSERVATIVE_L2_BYTES)),  # type: ignore[arg-type]
        sycl_target=profile.get("sycl_target"),  # type: ignore[arg-type]
        supports_graphs=False,
        extra=extra,
    )
