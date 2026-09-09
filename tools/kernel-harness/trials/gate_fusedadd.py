"""Gated wrapper for the in-place fused_add candidates: clones both mutated tensors.

The op overwrites t0 (normed output) and t1 (running residual).  Cloning here keeps the
caller's tensors pristine so the reference arm and the candidate arm see the same inputs.
Timing from this wrapper is meaningless.
"""

import importlib.util
import os

import torch
import torch.nn as nn

_PATH = os.environ["CAND"]
_spec = importlib.util.spec_from_file_location("gated_cand", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _mod.Model(*_mod.get_init_inputs())

    def forward(self, t0, t1, t2):
        return self.inner(t0.clone(), t1.clone(), t2).clone()


get_inputs = _mod.get_inputs


def get_init_inputs():
    return []
