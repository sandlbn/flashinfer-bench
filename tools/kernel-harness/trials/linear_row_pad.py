"""Harness: `aten.linear` against a weight whose row stride is set from the environment.

The op and the values are the same in every arm; only the *stride* of the weight differs,
so an A/B over this harness isolates what the weight's address pattern costs. Arm it with
`scripts/kernel_trials.py ab`:

    kernel_trials ab tools/kernel-harness/trials/linear_row_pad.py \
        --env-a FIB_ROW_PAD_ELEMS=0 --env-b FIB_ROW_PAD_ELEMS=32

Environment (read at import, so each `ab` arm sees its own):

    FIB_HARNESS_M, FIB_HARNESS_N, FIB_HARNESS_K   GEMM shape ``[M,K] x [N,K]^T``
    FIB_HARNESS_DTYPE                             torch dtype name (default bfloat16)
    FIB_ROW_PAD_ELEMS                             extra elements per weight row (default 0)
    FIB_WEIGHT_POOL_MB                            rotate over enough distinct copies of the
                                                  weight to exceed this many MB, so the weight
                                                  streams from device memory as it does at
                                                  decode instead of sitting in the last-level
                                                  cache (default 0: one weight, cache-resident)
    FIB_HARNESS_OP                                ``linear`` (default) or ``rowsum``: the
                                                  same weight read by a plain row reduction
                                                  instead of the GEMM, to tell a pitch the
                                                  memory system dislikes from one only the
                                                  GEMM's access pattern dislikes

`get_inputs()` draws the weight *before* allocating the padded storage, so the values -- and
`ab`'s input digest -- are identical across arms whatever the pad.
"""

import math
import os

import torch
import torch.nn as nn


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


M = _env_int("FIB_HARNESS_M", 4)
N = _env_int("FIB_HARNESS_N", 1024)
K = _env_int("FIB_HARNESS_K", 3072)
DTYPE = getattr(torch, os.environ.get("FIB_HARNESS_DTYPE", "bfloat16"))
ROW_PAD_ELEMS = _env_int("FIB_ROW_PAD_ELEMS", 0)
WEIGHT_POOL_MB = _env_int("FIB_WEIGHT_POOL_MB", 0)
OP = os.environ.get("FIB_HARNESS_OP", "linear")


def _run(x, w):
    if OP == "rowsum":
        return torch.sum(w, dim=1)
    return torch.ops.aten.linear.default(x, w)


def _device():
    return "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"


def _with_row_stride(w: torch.Tensor, row_stride: int) -> torch.Tensor:
    """A copy of ``w`` whose rows are ``row_stride`` elements apart (a view of wider storage)."""
    n, k = w.shape
    storage = torch.empty((n, row_stride), dtype=w.dtype, device=w.device)
    view = storage[:, :k]
    view.copy_(w)
    return view


class Model(nn.Module):
    """Calls `aten.linear` on whatever weight it is handed, optionally through a rotation pool.

    A tight loop over one weight measures the last-level cache once the weight fits in it.
    Serving does not run that way: a decode step touches every weight of the model once, so
    each GEMM's operand arrives from device memory. With a pool larger than the cache, each
    call reads a different copy, laid out with the *same* stride as the weight given.
    """

    def __init__(self):
        super().__init__()
        self._pools = {}
        self._calls = 0

    def _pool(self, w: torch.Tensor):
        key = (w.data_ptr(), tuple(w.shape), tuple(w.stride()))
        pool = self._pools.get(key)
        if pool is None:
            copies = max(1, math.ceil(WEIGHT_POOL_MB * 2**20 / (w.numel() * w.element_size())))
            pool = [w] + [_with_row_stride(w, w.stride(0)) for _ in range(copies - 1)]
            self._pools[key] = pool
        return pool

    def forward(self, x, w):
        if WEIGHT_POOL_MB <= 0:
            return _run(x, w)
        pool = self._pool(w)
        w_i = pool[self._calls % len(pool)]
        self._calls += 1
        return _run(x, w_i)


def get_inputs():
    device = _device()
    x = torch.randn([M, K], dtype=DTYPE, device=device)
    w = torch.randn([N, K], dtype=DTYPE, device=device)
    if ROW_PAD_ELEMS:
        w = _with_row_stride(w, K + ROW_PAD_ELEMS)
    return [x, w]


def get_init_inputs():
    return []
