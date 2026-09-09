"""Auto-generated harness for `_C.rms_norm.default`.

Emitted by scripts/harness_from_model.py from a run of Qwen/Qwen3-0.6B: this op was called
420 time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.
"""

import torch
import torch.nn as nn

OP = "_C.rms_norm.default"
CALLS = 420


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        return torch.ops._C.rms_norm(t0, t1, t2, 1e-06) or t0


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([128], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
