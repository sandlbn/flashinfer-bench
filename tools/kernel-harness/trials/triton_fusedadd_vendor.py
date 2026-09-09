"""The vendor kernel on the trial's own input path: `_C::fused_add_rms_norm` from vllm-xpu-kernels.

Same call as the auto-generated harness, but the inputs come through
`triton_fusedadd_inputs`, so this file is the baseline arm wherever the series departs from
the harness's row count (`FIB_TRIAL_ROWS`) -- and the arm of an A/A run that reads the
harness's own noise floor.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_fusedadd_inputs import EPS, get_init_inputs, get_inputs  # noqa: E402, F401


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        import vllm_xpu_kernels._C  # noqa: F401  registers torch.ops._C

        torch.ops._C.fused_add_rms_norm(t0, t1, t2, EPS)
        return t0
