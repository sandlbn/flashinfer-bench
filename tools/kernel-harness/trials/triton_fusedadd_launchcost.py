"""Attribution control: the vendor kernel plus one Triton launch that computes nothing.

Before a difference between the Triton candidate and the vendor op is credited to the
kernel, the call path has to be priced: a Triton launch pays a Python-side argument
binding, specialization lookup and autotuner cache probe before the device sees anything,
and the vendor op pays a `torch.ops` dispatch instead. This arm runs the production op --
so it passes the correctness gate unchanged -- and then a Triton kernel with the same
launcher, autotuner and argument list as the candidate whose body reads one element and
writes it back. Its slowdown against the plain vendor arm is the Triton call path at this
shape, and nothing else.
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
def _null_kernel(
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
    # One element in, the same element out: the launch without the work.
    row0 = tl.program_id(0) * BLOCK_Y
    if row0 < rows:
        tl.store(x_ptr + row0 * hidden, tl.load(x_ptr + row0 * hidden))


def run(hidden_states, residual, weight, eps):
    hidden = hidden_states.shape[-1]
    rows = hidden_states.numel() // hidden
    if rows == 0 or hidden == 0:
        return
    x = hidden_states.view(rows, hidden)
    r = residual.view(rows, hidden)
    block_x = triton.next_power_of_2(hidden)
    rows_bucket = triton.next_power_of_2(rows)
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_Y"]),)  # noqa: E731
    _null_kernel[grid](x, r, weight, rows, hidden, rows_bucket, eps, BLOCK_X=block_x)


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        import vllm_xpu_kernels._C  # noqa: F401  registers torch.ops._C

        torch.ops._C.fused_add_rms_norm(t0, t1, t2, EPS)
        run(t0, t1, t2, EPS)
        return t0
