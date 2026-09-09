"""Auto-generated harness for `_C.rotary_embedding.default`.

Emitted by scripts/harness_from_model.py from a run of Qwen/Qwen3-0.6B: this op was called
420 time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.
"""

import torch
import torch.nn as nn

OP = "_C.rotary_embedding.default"
CALLS = 420


class Model(nn.Module):
    def forward(self, t0, t1, t2, t4):
        return torch.ops._C.rotary_embedding(t0, t1, t2, 128, t4, True) or t0


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.ones([4], dtype=torch.int64, device=device),
        torch.randn([4, 2048], dtype=torch.bfloat16, device=device),
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([40960, 128], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
