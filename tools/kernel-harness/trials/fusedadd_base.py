"""Baseline: vLLM's `_C::fused_add_rms_norm` at the shape Qwen3-0.6B calls it.

Identical to tools/kernel-harness/pulled/_C_fused_add_rms_norm/harness.py except that the
extension is imported inside forward.

Kernel that runs: vllm::fused_add_rms_norm_kernel<bfloat16, width=8, HasWeight=true>.
"""

import torch
import torch.nn as nn

OP = "_C.fused_add_rms_norm.default"
CALLS = 840


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        import vllm_xpu_kernels._C  # noqa: F401

        torch.ops._C.fused_add_rms_norm(t0, t1, t2, 1e-06)
        return t0


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([1024], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
