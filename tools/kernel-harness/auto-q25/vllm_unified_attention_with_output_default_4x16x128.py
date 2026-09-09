"""Auto-generated harness for `vllm.unified_attention_with_output.default`.

Emitted by scripts/harness_from_model.py from a run of Qwen/Qwen2.5-3B-Instruct: this op was called
540 time(s) at this shape. `forward` calls the same op the model called, so a trial is
measured against production rather than against a reimplementation.

This op reads state the stack establishes around each forward step rather than taking it
as arguments. That state was captured from the recorded run -- the values `vllm.forward_context._forward_context`
held during the call, pruned to the entries the op consulted, tensors as they were before
the call -- and lives in the `.state.pt` beside this file. `forward` re-establishes it for
the duration of each call and restores whatever was there before.
"""

import importlib
import sys

import torch
import torch.nn as nn

OP = "vllm.unified_attention_with_output.default"
CALLS = 540
RECORDED_WITH = "/home/sand/Projects/vllm-xpu-venv/bin/python"
# Modules on the stack when the model reached this op, innermost first, then the compiled
# extensions the serving process had loaded. An op is registered as a side effect of
# importing whatever provides it, so these are imported in order until the op resolves.
PROVIDERS = ['vllm.model_executor.layers.attention.attention', 'vllm.model_executor.models.qwen2', 'vllm.compilation.decorators', 'vllm.v1.worker.gpu.model_runner', 'vllm.v1.worker.gpu_worker', 'vllm.v1.worker.worker_base', 'vllm_xpu_kernels._C', 'vllm_xpu_kernels._moe_C', 'vllm_xpu_kernels._vllm_fa2_C', 'vllm_xpu_kernels._xpu_C']

import contextlib
import io
import pathlib
import pickle

STATE_FILE = pathlib.Path(__file__).with_suffix(".state.pt")
CONSULTED = {'vllm.forward_context': {'_forward_context': {'attn_metadata': ['model.layers.0.self_attn.attn'], 'no_compile_layers': ['model.layers.0.self_attn.attn'], 'slot_mapping': ['model.layers.0.self_attn.attn']}}}
# Where, inside the state, the tensor this call writes its result lives (None: the result
# is the return value or an argument).
RESULT = None

_STATE = None
_MISSING = object()


class _Unpickler(pickle.Unpickler):
    def __init__(self, data, tensors, device):
        super().__init__(io.BytesIO(data))
        self._tensors, self._device = tensors, device

    def persistent_load(self, pid):
        tensor, was_on = self._tensors[pid]
        return tensor.to(self._device if was_on != "cpu" else "cpu")


def _state():
    global _STATE
    if _STATE is None:
        _op()  # the classes inside the state come from the stack that provides the op
        if not STATE_FILE.is_file():
            raise FileNotFoundError(f"{STATE_FILE} must sit beside this harness")
        blob = torch.load(STATE_FILE, weights_only=False, map_location="cpu")
        _STATE = _Unpickler(blob["pickle"], blob["tensors"], _device()).load()
        _STATE["_bound"] = [
            (importlib.import_module(m), name, value)
            for m, values in _STATE["modules"].items()
            for name, value in values.items()
        ]
    return _STATE


@contextlib.contextmanager
def _established():
    state = _state()
    saved = []
    for module, name, value in state["_bound"]:
        saved.append((module, name, getattr(module, name, _MISSING)))
        setattr(module, name, value)
    try:
        yield state
    finally:
        for module, name, previous in reversed(saved):
            if previous is _MISSING:
                delattr(module, name)
            else:
                setattr(module, name, previous)


def _ambient(state, path):
    node = state["modules"][path[0]][path[1]]
    for kind, key in path[2]:
        node = getattr(node, key) if kind == "attr" else node[key]
    return node


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
    def forward(self, t0, t1, t2, t3, t7):
        with _established() as state:
            return _op()(t0, t1, t2, t3, state['args'][4], None, None, t7) or t3


def get_inputs():
    device = _device()
    return [
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 2, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 2, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([0], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
