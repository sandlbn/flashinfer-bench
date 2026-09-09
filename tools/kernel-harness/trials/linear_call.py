"""Call-shape variants of one `aten.linear` harness, for the `library_call` mechanism.

oneDNN picks a strategy per problem descriptor, so the levers a caller holds are the
descriptor itself: which operand is A, how the weight is laid out, whether the problem is
split, and what post-ops ride along. Each variant here issues the *same* mathematical
product as the routed harness with a different descriptor, so a trial measures the call and
nothing else.

The shape comes from the harness the series was initialised with -- ``FIB_HARNESS_BASE``
names it -- and nothing here spells out a dimension. ``FIB_WEIGHT_POOL_MB`` rotates the
weight over enough copies to exceed the last-level cache, so the operand streams from
device memory as it does in a decode step instead of sitting in cache.

A trial file is three lines:

    from linear_call import make
    Model, get_inputs, get_init_inputs = make("tn")
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

BASE_ENV = "FIB_HARNESS_BASE"
POOL_ENV = "FIB_WEIGHT_POOL_MB"
SPLIT_ENV = "FIB_LINEAR_SPLITS"


def _base_module():
    path = os.environ.get(BASE_ENV)
    if not path:
        raise SystemExit(
            f"{BASE_ENV} must name the routed harness whose get_inputs() this variant "
            "reuses; a variant does not carry a shape of its own."
        )
    spec = importlib.util.spec_from_file_location("linear_call_base", pathlib.Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_copies(w: torch.Tensor) -> int:
    pool_mb = int(os.environ.get(POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, math.ceil(pool_mb * 2**20 / (w.numel() * w.element_size())))


class _Variant(nn.Module):
    """Holds per-weight derived operands (transposed copies, split views) built once.

    Anything derived from the weight is built on first sight of that weight and reused, as
    a load-time weight transform would be in a serving stack; the per-call work is exactly
    the GEMM call the variant is testing. The single-weight fast path is one pointer compare
    so the wrapper itself costs nothing a host-bound loop could see.
    """

    def __init__(self, prepare: Callable[[torch.Tensor], object], run: Callable):
        super().__init__()
        self._prepare, self._run = prepare, run
        self._ptr = None
        self._pool: List[object] = []
        self._calls = 0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if w.data_ptr() != self._ptr:
            self._ptr = w.data_ptr()
            self._pool = [self._prepare(w)] + [
                self._prepare(w.clone()) for _ in range(_pool_copies(w) - 1)
            ]
            self._calls = 0
        if len(self._pool) == 1:
            return self._run(x, self._pool[0])
        operand = self._pool[self._calls % len(self._pool)]
        self._calls += 1
        return self._run(x, operand)


def _splits() -> int:
    return int(os.environ.get(SPLIT_ENV, "2"))


PADM_ENV = "FIB_LINEAR_PADM_ROWS"


def _padm_rows() -> int:
    return int(os.environ.get(PADM_ENV, "16"))


def _padded(w: torch.Tensor, only_if_camping: bool) -> torch.Tensor:
    from flashinfer_bench.integration import weight_layout as wl

    if only_if_camping:
        padded = wl.pad_rows_off_channel_period(w)
        return w if padded is None else padded
    return wl._copy_with_pitch(w, wl.row_pitch_bytes(w) + wl.ROW_PAD_BYTES)


# name -> (prepare(weight) -> operand, run(x, operand) -> out)
VARIANTS: Dict[str, Tuple[Callable, Callable]] = {
    # Production: src ab, wei ba, one M x K : K x N problem.
    "ba": (lambda w: w, lambda x, w: torch.ops.aten.linear.default(x, w)),
    # What the serving stack's Python actually calls (vLLM: default_unquantized_gemm).
    "flinear": (lambda w: w, lambda x, w: torch.nn.functional.linear(x, w)),
    # The same product through shallower dispatcher entry points: aten.linear is
    # CompositeImplicit over matmul over mm, and each layer is host work per call.
    "matmul": (lambda w: w.t(), lambda x, wt: torch.matmul(x, wt)),
    "mm": (lambda w: w.t(), lambda x, wt: torch.ops.aten.mm.default(x, wt)),
    # Weight pre-transposed once to [K, N]: wei ab.
    "ab": (lambda w: w.t().contiguous(), lambda x, wt: torch.matmul(x, wt)),
    # The transposed problem N x K : K x M with the weight as A and x^T as B, written
    # through a transposed view of a row-major [M, N] output: dst ba, no copy anywhere.
    "tn": (
        lambda w: w,
        lambda x, w: torch.matmul(
            w, x.t(), out=torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device).t()
        ).t(),
    ),
    # The delivered pitch fix: rows moved off the memory-channel period, only when the
    # weight camps on it (otherwise the weight is used as is).
    "pad": (lambda w: _padded(w, only_if_camping=True), lambda x, w: torch.ops.aten.linear.default(x, w)),
    # The same pitch move applied unconditionally, to see the pitch effect at every shape.
    "pitch": (lambda w: _padded(w, only_if_camping=False), lambda x, w: torch.ops.aten.linear.default(x, w)),
    # M padded up to the GEMM's M tile with zero rows, so the selector sees a fuller tile.
    "padm": (
        lambda w: w,
        lambda x, w: torch.ops.aten.linear.default(
            torch.cat([x, x.new_zeros(_padm_rows() - x.shape[0], x.shape[1])]), w
        )[: x.shape[0]],
    ),
    # Split-K into FIB_LINEAR_SPLITS batched products plus a reduction: more work-groups
    # in flight over the weight, at the price of a second launch.
    "bmm": (
        lambda w: w.view(w.shape[0], _splits(), w.shape[1] // _splits()).permute(1, 2, 0).contiguous(),
        lambda x, wb: torch.bmm(
            x.view(x.shape[0], wb.shape[0], wb.shape[1]).transpose(0, 1), wb
        ).sum(0),
    ),
    # Split-N into FIB_LINEAR_SPLITS independent products over column slices of the weight,
    # written into slices of one output (no reduction, but one launch per slice).
    "splitn": (
        lambda w: [c.contiguous() for c in w.chunk(_splits(), dim=0)],
        lambda x, chunks: torch.cat([torch.ops.aten.linear.default(x, c) for c in chunks], dim=1),
    ),
}


def make(name: str):
    prepare, run = VARIANTS[name]
    base = _base_module()

    class Model(_Variant):
        def __init__(self):
            super().__init__(prepare, run)

    return Model, base.get_inputs, base.get_init_inputs
