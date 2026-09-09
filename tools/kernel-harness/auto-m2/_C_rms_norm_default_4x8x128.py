"""Auto-generated harness for `_C.rms_norm.default`.

Emitted by scripts/harness_from_model.py from a run of Qwen/Qwen3-1.7B: this op was called
420 time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.
"""

import importlib
import sys

import torch
import torch.nn as nn

OP = "_C.rms_norm.default"
CALLS = 420
RECORDED_WITH = "/home/sand/Projects/vllm-xpu-venv/bin/python"
# Modules on the stack when the model reached this op, innermost first, then the compiled
# extensions the serving process had loaded. An op is registered as a side effect of
# importing whatever provides it, so these are imported in order until the op resolves.
PROVIDERS = ['vllm.kernels.vllm_c', 'vllm.ir.op', 'vllm.model_executor.layers.layernorm', 'vllm.model_executor.custom_op', 'vllm.model_executor.models.qwen3', 'vllm.model_executor.models.qwen2', 'vllm_xpu_kernels._C', 'vllm_xpu_kernels._moe_C', 'vllm_xpu_kernels._vllm_fa2_C', 'vllm_xpu_kernels._xpu_C']

_OP = None


def _op():
    global _OP
    if _OP is None:
        ns, name = OP.split(".")[:2]
        for provider in (None, *PROVIDERS):
            if provider is not None:
                try:
                    importlib.import_module(provider)
                except Exception:
                    continue
            try:
                _OP = getattr(getattr(torch.ops, ns), name)
                break
            except (AttributeError, RuntimeError):
                pass
        else:
            raise ImportError(
                f"{OP} is not registered in {sys.executable}; the harness was recorded "
                f"under {RECORDED_WITH}, which has the stack that provides it."
            )
    return _OP


def _device():
    return "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        return _op()(t0, t1, t2, 1e-06) or t0


def get_inputs():
    device = _device()
    return [
        torch.randn([4, 8, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 8, 128], dtype=torch.bfloat16, device=device),
        torch.randn([128], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
