"""Baseline harness: the provider kernel a deployment runs for rmsnorm at h2560.

The baseline a SYCL trial must beat is the compiled kernel vLLM would otherwise call, not
the definition's PyTorch reference -- a reference makes several passes over memory and
flatters anything measured against it.
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "tools/kernel-harness")
from sycl_harness import inputs_for  # noqa: E402

DEFINITION = "rmsnorm_h2560"
BATCH = 4096


class Model(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states, weight):
        import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

        out = torch.empty_like(hidden_states)
        torch.ops._C.rms_norm(out, hidden_states, weight, self.eps)
        return out


get_inputs = inputs_for(DEFINITION, batch_size=BATCH)


def get_init_inputs():
    return [1e-6]
