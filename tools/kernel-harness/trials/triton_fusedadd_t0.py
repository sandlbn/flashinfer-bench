"""t0: Triton fused residual-add + RMSNorm, in place, in the vendor's destination-passing form.

Why destination-passing rather than returning tensors: the production op mutates both of its
tensor arguments (`input` becomes the normed output, `residual` the running sum), and the
trial loop's correctness gate compares exactly those arguments after the call. A kernel that
returned fresh tensors would fail the gate on the untouched `residual`, and its timing would
carry two allocations the vendor path never pays -- the comparison would no longer be
kernel against kernel.

Structure: one program per BLOCK_Y rows, the whole row in one block, so the row never leaves
registers between the reduction and the normalize -- the vendor kernel writes the summed
residual out and reads it back (`residual_v[id]` in both loops). The sum is rounded to the
storage dtype before the variance, as the vendor does (`temp.val[i] += res.val[i]` in
bfloat16), so the two agree to reduction order.

In-place safety: every load of a program's tile precedes its first store, and tiles are
disjoint, so no program reads what another has written.

Autotuned over Intel's sub-group width (`warp_size`), `num_warps` and rows-per-program,
all read from the driver rather than carried over from CUDA; `num_stages` stays at 1
because Intel has no async-copy pipeline for it to feed. The key includes a power-of-two
row bucket so a decode step and a prefill tune separately -- the optimum moves with how
many work-groups there are to fill the machine.
"""

import os
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_fusedadd_inputs import EPS, get_init_inputs, get_inputs  # noqa: E402, F401

_PROPS = driver.active.utils.get_device_properties(torch.xpu.current_device())
MAX_WORK_GROUP_SIZE = _PROPS["max_work_group_size"]
SUB_GROUP_SIZES = tuple(_PROPS.get("sub_group_sizes", (32,)))

_CONFIGS = [
    triton.Config({"BLOCK_Y": by, "warp_size": ws}, num_warps=nw, num_stages=1)
    for ws in SUB_GROUP_SIZES
    for nw in (1, 2, 4, 8, 16)
    for by in (1, 2, 4, 8)
    if ws * nw <= MAX_WORK_GROUP_SIZE
]


def _prune(configs, named_args, **kwargs):
    """Drop configs that cannot pay off at this call: more rows per program than there are
    rows, or more threads than elements in the tile."""
    rows_bucket = named_args["rows_bucket"]
    block_x = kwargs["BLOCK_X"]
    kept = [
        c
        for c in configs
        if c.kwargs["BLOCK_Y"] <= rows_bucket
        and c.kwargs["warp_size"] * c.num_warps <= block_x * c.kwargs["BLOCK_Y"]
    ]
    return kept or configs[:1]


@triton.autotune(
    configs=_CONFIGS,
    key=["hidden", "rows_bucket"],
    restore_value=["x_ptr", "r_ptr"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    rows,
    hidden,
    rows_bucket,
    eps,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    row0 = tl.program_id(0) * BLOCK_Y
    cols = tl.arange(0, BLOCK_X)
    rws = row0 + tl.arange(0, BLOCK_Y)
    mask = (cols[None, :] < hidden) & (rws[:, None] < rows)
    offs = rws[:, None] * hidden + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    summed = (x + r).to(r_ptr.dtype.element_ty)
    tl.store(r_ptr + offs, summed, mask=mask)

    v = summed.to(tl.float32)
    scale = tl.rsqrt(tl.sum(v * v, axis=1) / hidden + eps)[:, None]
    w = tl.load(w_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)[None, :]
    tl.store(x_ptr + offs, (v * scale * w).to(x_ptr.dtype.element_ty), mask=mask)


def run(hidden_states, residual, weight, eps):
    """Destination-passing: `hidden_states` <- normed output, `residual` <- summed residual."""
    hidden = hidden_states.shape[-1]
    rows = hidden_states.numel() // hidden
    if rows == 0 or hidden == 0:
        return
    x = hidden_states.view(rows, hidden)
    r = residual.view(rows, hidden)
    block_x = triton.next_power_of_2(hidden)
    rows_bucket = triton.next_power_of_2(rows)
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_Y"]),)  # noqa: E731
    _fused_add_rms_norm_kernel[grid](
        x, r, weight, rows, hidden, rows_bucket, eps, BLOCK_X=block_x
    )


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        run(t0, t1, t2, EPS)
        return t0
