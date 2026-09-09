"""Correctness reference for the rms_norm trials -- NOT a timing baseline.

`kernel_trials benchmark` compares `base_model(*inputs)` against `cand_model(*inputs)` on
the same tensor list.  Both the production harness and the candidates write their result
into `t0` and return `t0`, so `expected` and `actual` are the *same object* and the gate
reads 0 error whatever the kernel did.  This file is the fix: it computes the norm in plain
torch into a fresh tensor and never touches `t0`, so a candidate that also snapshots its
output (`return t0.clone()`) is really compared.  Ratios from this series are meaningless.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        x = t1.float()
        var = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + 1e-06) * t2.float()).to(torch.bfloat16)


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.empty([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([128], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
