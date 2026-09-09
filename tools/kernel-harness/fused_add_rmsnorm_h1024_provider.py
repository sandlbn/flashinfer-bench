"""Baseline: vLLM's in-place fused add + RMSNorm, the kernel a deployment runs at h1024."""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "tools/kernel-harness")
from sycl_harness import inputs_for  # noqa: E402

DEFINITION = "fused_add_rmsnorm_residual_h1024"
BATCH = 64
EPS = 1e-6


class Model(nn.Module):
    def __init__(self, eps: float = EPS):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states, residual, weight):
        import vllm_xpu_kernels._C  # noqa: F401

        # vLLM's kernel is in place and overwrites both; clone so repeated timing runs
        # measure the same problem rather than compounding results.
        x = hidden_states.clone()
        r = residual.clone()
        torch.ops._C.fused_add_rms_norm(x, r, weight, self.eps)
        return x


get_inputs = inputs_for(DEFINITION, batch_size=BATCH)


def get_init_inputs():
    return [EPS]
