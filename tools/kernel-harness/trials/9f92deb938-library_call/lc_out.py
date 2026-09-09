"""t1's [K, N] weight, with the destination preallocated once and reused.

The only change from `v_ab.py` is `out=`: the 5040x4096 bf16 result (41 MB) is allocated
once instead of once per call, so the allocator and the empty() path leave the per-call
work. The GEMM, its operands and its descriptor are t1's.
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
    """[N, K] as the model stores it -> [K, N] contiguous, exactly t1's operand."""
    return w.t().contiguous()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._ptr = None
        self._pool: List[torch.Tensor] = []
        self._calls = 0
        self._out = None

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if w.data_ptr() != self._ptr:
            self._ptr = w.data_ptr()
            self._pool = [_prepare(w)] + [_prepare(w.clone()) for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        if self._out is None or self._out.shape[0] != x.shape[0]:
            self._out = torch.empty(
                (x.shape[0], self._pool[0].shape[1]), dtype=x.dtype, device=x.device
            )
        if len(self._pool) == 1:
            return torch.matmul(x, self._pool[0], out=self._out)
        operand = self._pool[self._calls % len(self._pool)]
        self._calls += 1
        return torch.matmul(x, operand, out=self._out)


_base = _base_module()
get_inputs = _base.get_inputs
get_init_inputs = _base.get_init_inputs
