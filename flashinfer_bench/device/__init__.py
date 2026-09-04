"""Device abstraction: one interface over CUDA, Intel XPU and CPU backends."""

from .accelerator import (
    Accelerator,
    available_types,
    canonicalize_device_name,
    default_device_type,
    device_synchronize,
    get_accelerator,
    hardware_id,
    list_devices,
    parse_device,
    register_accelerator,
    registered_types,
    reset_registry_cache,
    unregister_accelerator,
)
from .capabilities import DEFAULT_DTYPES, FP4_DTYPES, FP8_DTYPES, Capabilities
from .cpu import CpuAccelerator
from .cuda import CudaAccelerator
from .power import measurement_warnings, on_battery, platform_profile
from .timer import CuptiTimer, EventTimer, Timer, WallClockTimer
from .xpu import XpuAccelerator

__all__ = [
    "Accelerator",
    "Capabilities",
    "CpuAccelerator",
    "CudaAccelerator",
    "CuptiTimer",
    "DEFAULT_DTYPES",
    "EventTimer",
    "FP4_DTYPES",
    "FP8_DTYPES",
    "Timer",
    "WallClockTimer",
    "XpuAccelerator",
    "available_types",
    "canonicalize_device_name",
    "default_device_type",
    "device_synchronize",
    "get_accelerator",
    "hardware_id",
    "list_devices",
    "measurement_warnings",
    "on_battery",
    "platform_profile",
    "parse_device",
    "register_accelerator",
    "registered_types",
    "unregister_accelerator",
    "reset_registry_cache",
]
