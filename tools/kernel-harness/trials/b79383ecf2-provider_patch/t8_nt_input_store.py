"""Streaming-weight derivation of tools/kernel-harness/auto-pf2/_C_fused_add_rms_norm_default_5040x1024.py; see stream_harness.py."""
"""Auto-generated harness for `_C.fused_add_rms_norm.default`.

Emitted by scripts/harness_from_model.py from a run of Qwen/Qwen3-0.6B: this op was called
56 time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.
"""

import importlib
import sys

import torch
import torch.nn as nn

OP = "_C.fused_add_rms_norm.default"
CALLS = 56
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


VARIANT = "fused_add_rms_norm_nt2"


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        _op()  # imports whatever registers the provider's ops
        return getattr(torch.ops._C, VARIANT)(t0, t1, t2, 1e-06) or t0


def get_inputs():
    device = _device()
    return [
        torch.randn([5040, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([5040, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([1024], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []


# --- streaming pool (added by tools/kernel-harness/trials/stream_harness.py) ---------------
import math as _math
import os as _os

_POOL_ENV = "FIB_WEIGHT_POOL_MB"

_PAD_ENV = "FIB_WEIGHT_PAD"


def _prepare_weight(w):
    """The delivered pitch fix, when asked for: rows moved off the memory-channel period."""
    if _os.environ.get(_PAD_ENV, "") not in ("1", "true", "yes", "on"):
        return w
    from flashinfer_bench.integration import weight_layout as _wl

    padded = _wl.pad_rows_off_channel_period(w)
    return w if padded is None else padded


def _pool_copies(t):
    pool_mb = int(_os.environ.get(_POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, _math.ceil(pool_mb * 2**20 / (t.numel() * t.element_size())))


class _Streaming(Model):
    """The generated Model, with its largest tensor argument rotated over a pool of copies.

    The largest argument is the weight in every GEMM-shaped op; rotating it -- and only
    it -- keeps the activations resident exactly as a decode step would.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._slot, self._ptr, self._pool, self._calls = None, None, [], 0

    def forward(self, *args):
        slot = self._slot
        if slot is None or args[slot].data_ptr() != self._ptr:
            tensors = [(i, t) for i, t in enumerate(args) if hasattr(t, "data_ptr")]
            if not tensors:
                return super().forward(*args)
            slot, w = max(tensors, key=lambda it: it[1].numel() * it[1].element_size())
            self._slot, self._ptr = slot, w.data_ptr()
            self._pool = [_prepare_weight(w)] + [_prepare_weight(w.clone()) for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        pool = self._pool
        args = list(args)
        if len(pool) == 1:
            args[slot] = pool[0]
            return super().forward(*args)
        args[slot] = pool[self._calls % len(pool)]
        self._calls += 1
        return super().forward(*args)


Model = _Streaming
