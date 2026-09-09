"""Weight in [K, N] with a padded leading dimension -- the layout oneDNN picks for itself.

`od_any.py` offers the weight as `format_tag::any`; the library's answer, read off
ONEDNN_VERBOSE, is `wei bf16:a:blocked:ab:4128x1` -- the [K, N] orientation of `v_ab.py`
*plus* a leading dimension of 4128 instead of 4096, i.e. 64 bytes of row padding. This
builds that operand through torch, so the same descriptor reaches the same library through
the production entry point instead of through a hand-built primitive.

FIB_LINEAR_LD_PAD sets the padding in elements (default 32 = 64 bytes, what oneDNN chose).
"""

import importlib.util
import math
import os
import pathlib
from typing import List

import torch
import torch.nn as nn

BASE_ENV = "FIB_HARNESS_BASE"
POOL_ENV = "FIB_WEIGHT_POOL_MB"
PAD_ENV = "FIB_LINEAR_LD_PAD"

OP = "aten.linear.default"
CALLS = 28


def _base_module():
    path = os.environ[BASE_ENV]
    spec = importlib.util.spec_from_file_location("lc_pitch_base", pathlib.Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_copies(w: torch.Tensor) -> int:
    pool_mb = int(os.environ.get(POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, math.ceil(pool_mb * 2**20 / (w.numel() * w.element_size())))


def _prepare(w: torch.Tensor) -> torch.Tensor:
    """[N, K] as the model stores it -> [K, N] with leading dimension N + pad."""
    pad = int(os.environ.get(PAD_ENV, "32"))
    wt = w.t()
    k, n = wt.shape
    buf = torch.empty((k, n + pad), dtype=w.dtype, device=w.device)
    buf[:, :n] = wt
    return buf[:, :n]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._ptr = None
        self._pool: List[torch.Tensor] = []
        self._calls = 0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if w.data_ptr() != self._ptr:
            self._ptr = w.data_ptr()
            self._pool = [_prepare(w)] + [_prepare(w.clone()) for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        if len(self._pool) == 1:
            return torch.matmul(x, self._pool[0])
        operand = self._pool[self._calls % len(self._pool)]
        self._calls += 1
        return torch.matmul(x, operand)


_base = _base_module()
get_inputs = _base.get_inputs
get_init_inputs = _base.get_init_inputs
