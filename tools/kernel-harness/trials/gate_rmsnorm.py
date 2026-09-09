"""Gated wrapper: runs the candidate named by $CAND and snapshots its output.

Used only with the `rmsnorm_gate` series (baseline = rmsnorm_ref.py), to get a real
correctness verdict.  Its timing is meaningless -- the clone is not part of the kernel.
"""

import importlib.util
import os
import pathlib

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
        return self.inner(t0, t1, t2).clone()


get_inputs = _mod.get_inputs


def get_init_inputs():
    return []
