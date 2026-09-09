"""Inputs for the Triton fused_add_rms_norm trials come from the pipeline's harness, not from here.

`FIB_TRIAL_BASELINE` names the auto-generated harness
(`tools/kernel-harness/auto*/_C_fused_add_rms_norm_default_*.py`) whose `get_inputs()` builds
the tensors at the shape and dtype the model actually called the op with. Every trial in a
series imports its inputs from that one file, so a candidate cannot quietly change the
problem it is measured on.

`FIB_TRIAL_ROWS`, when set, replaces the row count of the 2-D tensors -- the one axis that
differs between a decode step and a prefill -- so the same kernel can be measured at a
prefill-sized batch against the same vendor op. Nothing else about the inputs changes.

`FIB_TRIAL_EPS` is the epsilon the harness passes to the op (it is inlined in the harness's
`forward`, where a trial cannot import it); the default mirrors that call.
"""

import importlib.util
import os

import torch

EPS = float(os.environ.get("FIB_TRIAL_EPS", "1e-6"))


def _baseline():
    path = os.environ.get("FIB_TRIAL_BASELINE")
    if not path:
        raise SystemExit(
            "FIB_TRIAL_BASELINE must name the auto-generated harness this series was "
            "initialised with, so the trial measures the shape the model ran."
        )
    spec = importlib.util.spec_from_file_location("fusedadd_baseline_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_inputs():
    inputs = _baseline().get_inputs()
    rows = os.environ.get("FIB_TRIAL_ROWS")
    if rows:
        inputs = [
            torch.randn([int(rows), t.shape[1]], dtype=t.dtype, device=t.device)
            if isinstance(t, torch.Tensor) and t.dim() == 2
            else t
            for t in inputs
        ]
    return inputs


def get_init_inputs():
    return []
