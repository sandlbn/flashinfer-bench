"""Problem decomposition on top of t1's [K, N] weight.

t1 established the operand orientation; this varies only how the product is cut up.
FIB_LINEAR_MODE picks the cut, FIB_LINEAR_SPLITS how many pieces:

  n   split N into `splits` column slices, one matmul each, results concatenated
  k   split K into `splits` slices, one matmul each, results summed (split-K)

The weight slices are derived once per weight, as a load-time transform would be, so the
per-call work is only the matmuls and the join.
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
MODE_ENV = "FIB_LINEAR_MODE"
SPLIT_ENV = "FIB_LINEAR_SPLITS"

OP = "aten.linear.default"
CALLS = 28


def _base_module():
    spec = importlib.util.spec_from_file_location(
        "lc_split_base", pathlib.Path(os.environ[BASE_ENV])
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_copies(w: torch.Tensor) -> int:
    pool_mb = int(os.environ.get(POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, math.ceil(pool_mb * 2**20 / (w.numel() * w.element_size())))


def _splits() -> int:
    return int(os.environ.get(SPLIT_ENV, "2"))


def _mode() -> str:
    return os.environ.get(MODE_ENV, "n")


def _prepare(w: torch.Tensor) -> List[torch.Tensor]:
    """[N, K] as stored -> the [K, N] pieces this decomposition multiplies."""
    wt = w.t().contiguous()  # [K, N], t1's operand
    dim = 1 if _mode() == "n" else 0
    return [c.contiguous() for c in wt.chunk(_splits(), dim=dim)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._ptr = None
        self._pool: List[List[torch.Tensor]] = []
        self._calls = 0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if w.data_ptr() != self._ptr:
            self._ptr = w.data_ptr()
            self._pool = [_prepare(w)] + [_prepare(w.clone()) for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        pieces = self._pool[self._calls % len(self._pool)]
        if len(self._pool) > 1:
            self._calls += 1
        if _mode() == "n":
            return torch.cat([torch.matmul(x, p) for p in pieces], dim=1)
        k = pieces[0].shape[0]
        out = torch.matmul(x[:, :k], pieces[0])
        off = k
        for p in pieces[1:]:
            out = out + torch.matmul(x[:, off : off + p.shape[0]], p)
            off += p.shape[0]
        return out


_base = _base_module()
get_inputs = _base.get_inputs
get_init_inputs = _base.get_init_inputs
