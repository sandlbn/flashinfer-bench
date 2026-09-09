"""Correctness reference for the fused_add_rms_norm trials -- NOT a timing baseline.

Same reason as rmsnorm_ref.py, and worse here: the production op mutates *both* t0 and t1
in place, so running the two arms in sequence feeds the candidate the baseline's output.
This reference reads the inputs and returns fresh tensors, leaving t0/t1 pristine for the
arm that runs second.  Candidates gated against it must clone before calling.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        r = (t0.float() + t1.float())
        var = r.pow(2).mean(-1, keepdim=True)
        return (r * torch.rsqrt(var + 1e-06) * t2.float()).to(torch.bfloat16)


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([1024], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
