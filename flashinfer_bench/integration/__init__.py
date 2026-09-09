"""Integration layer for external frameworks like FlashInfer."""

from .weight_layout import (
    camps_on_one_channel,
    channel_period_bytes,
    deinterleave_output,
    interleave_gate_up,
    memory_channel_count,
    pad_rows_off_channel_period,
    qwen_style_mlp_weights,
    rms_row_scale,
)
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
    "camps_on_one_channel",
    "channel_period_bytes",
    "deinterleave_output",
    "interleave_gate_up",
    "memory_channel_count",
    "pad_rows_off_channel_period",
    "qwen_style_mlp_weights",
    "rms_row_scale",
    "SGL_KERNEL_XPU",
    "VLLM_XPU",
    "available_providers",
    "find_baselines",
    "is_provider_available",
    "make_baseline_solution",
    "make_baseline_solutions",
]
