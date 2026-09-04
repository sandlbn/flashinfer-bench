"""Integration layer for external frameworks like FlashInfer."""

from .xpu_kernels import (
    SGL_KERNEL_XPU,
    VLLM_XPU,
    BaselineKernel,
    available_providers,
    find_baselines,
    is_provider_available,
    make_baseline_solution,
    make_baseline_solutions,
)

__all__ = [
    "BaselineKernel",
    "SGL_KERNEL_XPU",
    "VLLM_XPU",
    "available_providers",
    "find_baselines",
    "is_provider_available",
    "make_baseline_solution",
    "make_baseline_solutions",
]
