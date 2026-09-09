"""Baseline: vLLM's `_C::rms_norm` at the shape Qwen3-0.6B calls it (q_norm/k_norm).

Identical to tools/kernel-harness/pulled/_C_rms_norm/harness.py except that the extension
is imported inside forward -- the pulled file relies on it already being loaded, and the
op namespace does not exist otherwise.

Kernel that runs: vllm::rms_norm_multi_row_kernel<bfloat16, NUM_DIMS=3, VEC=8, ROWS=16>.
"""

import torch
import torch.nn as nn

OP = "_C.rms_norm.default"
CALLS = 420


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        import vllm_xpu_kernels._C  # noqa: F401

        torch.ops._C.rms_norm(t0, t1, t2, 1e-06)
        return t0


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.empty([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([128], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
