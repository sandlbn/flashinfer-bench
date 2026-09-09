"""Extract kernel Definitions and Workloads from a HuggingFace model running on Intel GPU.

The onboarding skills harvest definitions from an SGLang run through FlashInfer's tracing
hooks, which is CUDA-only. This does the equivalent for Intel directly from a HuggingFace
model: run a short generation, observe the modules that actually execute, and emit the
Definitions and Workloads for the shapes that were really used.

Definitions produced here are ordinary, hardware-agnostic FlashInfer-Trace definitions --
the same artifacts the CUDA path produces. What makes this useful on Intel is only that it
needs no CUDA-only tooling to obtain them.

Usage
-----
    python scripts/extract_model_kernels_xpu.py \\
        --model <org>/<model> \\
        --output ./<model>-intel-trace \\
        --prompt "Write a short poem." --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("extract-model-kernels")

# References follow the shipped dataset exactly: float32 accumulation, the constant
# asserted, and epsilon as a module-level EPS -- which is where the tooling reads it from,
# since it is not a declared input.
RMSNORM_REFERENCE = """import torch

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == {hidden}

    EPS = {eps!r}

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
"""

LINEAR_REFERENCE = """import torch

def run(A, B):
    C = torch.matmul(A, B.T)
    return C
"""

LINEAR_FP8_BLOCK_REFERENCE = """import torch

BLOCK_N = {block_n}
BLOCK_K = {block_k}


def run(A_fp8, A_scale, B_fp8, B_scale):
    # Block-scaled FP8 W8A8 linear, as served: both operands are fp8.
    #
    #   A_fp8   [M, K]              activations, quantized per token-group along K
    #   A_scale [M, K/BLOCK_K]      one scale per (token, K-group)
    #   B_fp8   [N, K]              weights
    #   B_scale [N/BLOCK_N, K/BLOCK_K]  one scale per (BLOCK_N x BLOCK_K) weight tile
    #
    # Dequantise both, then matmul in float32. This mirrors vLLM's own upcast reference
    # (`w8a8_triton_block_scaled_mm`, the gfx1250 fallback path) so a kernel validated
    # here is validated against what the serving stack considers correct.
    N, K = B_fp8.shape
    a = A_fp8.to(torch.float32)
    a_scale = A_scale.to(torch.float32).repeat_interleave(BLOCK_K, dim=1)[:, :K]
    b = B_fp8.to(torch.float32)
    b_scale = B_scale.to(torch.float32)
    b_scale = b_scale.repeat_interleave(BLOCK_N, 0).repeat_interleave(BLOCK_K, 1)[:N, :K]
    return torch.matmul(a * a_scale, (b * b_scale).T).to(torch.bfloat16)
"""

_DTYPE_NAMES = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    # Weight storage dtypes for quantized models. These never appear as an activation
    # dtype -- the packed weight is an input to a block-scaled GEMM, not the thing the
    # layer computes in.
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
}


def _module_eps(module: Any) -> float:
    """The module's epsilon, which is part of what the kernel computes.

    Two norms of the same width that differ in eps or dtype are different operations. The
    dataset names them by width alone, so without this they collide and one silently
    overwrites the other.
    """
    for attr in ("variance_epsilon", "eps"):
        v = getattr(module, attr, None)
        if isinstance(v, float):
            return v
    return 1e-6


_TRUST_REMOTE_CODE = False
"""Whether to execute modeling code shipped in the model repository.

Off by default, and deliberately a flag rather than an inferred default: a model with an
`auto_map` runs Python from the repo inside this process. Novel architectures almost always
need it -- there is no other way to load one transformers does not yet know -- so the flag
exists, but turning it on is the caller's decision to make knowingly.
"""


def _hf_kwargs() -> Dict[str, Any]:
    return {"trust_remote_code": True} if _TRUST_REMOTE_CODE else {}


def _dtype_name(dtype: Any) -> str:
    name = _DTYPE_NAMES.get(str(dtype))
    if name is None:
        raise ValueError(f"Unsupported dtype for extraction: {dtype}")
    return name


class ShapeRecorder:
    """Records the shapes each module was actually called with.

    Counting calls matters as much as collecting shapes: a kernel invoked once per layer
    per token is worth optimizing, and one invoked twice in total is not.
    """

    def __init__(self) -> None:
        self.rmsnorm: Counter = Counter()
        self.linear: Counter = Counter()
        self.linear_fp8: Counter = Counter()
        # Epsilon per hidden size, read off the module. Models differ (1e-5 and 1e-6 both
        # occur), and the reference bakes it in, so it must be observed, not assumed.
        self.rmsnorm_eps: Dict[int, float] = {}
        self._handles: List[Any] = []

    def attach(self, model: Any) -> None:
        import torch

        for module in model.modules():
            cls_name = type(module).__name__
            if "RMSNorm" in cls_name and hasattr(module, "weight"):
                self._handles.append(module.register_forward_hook(self._make_rmsnorm_hook(module)))
            elif isinstance(module, torch.nn.Linear) and module.bias is None:
                self._handles.append(module.register_forward_hook(self._make_linear_hook(module)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_rmsnorm_hook(self, module: Any):
        def hook(_mod: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            x = inputs[0]
            if x.dim() < 2:
                return
            tokens = int(x.numel() // x.shape[-1])
            hidden = int(x.shape[-1])
            eps = _module_eps(module)
            self.rmsnorm[(tokens, hidden, _dtype_name(x.dtype), eps)] += 1
            for attr in ("variance_epsilon", "eps"):
                value = getattr(_mod, attr, None)
                if isinstance(value, float):
                    self.rmsnorm_eps.setdefault(hidden, value)
                    break

        return hook

    def _make_linear_hook(self, module: Any):
        import torch

        def hook(_mod: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            x = inputs[0]
            if x.dim() < 2:
                return
            tokens = int(x.numel() // x.shape[-1])
            in_f, out_f = int(module.in_features), int(module.out_features)

            # A block-scaled FP8 linear is a different operation from a dense one, not the
            # same one in another dtype: it takes the packed weight and its per-block
            # scales as separate tensors and dequantises inside the kernel. Emitting it as
            # `gemm_n{N}_k{K}` would collide with the bf16 definition of the same shape and
            # describe maths the kernel does not do.
            weight = getattr(module, "weight", None)
            scale_inv = getattr(module, "weight_scale_inv", None)
            if (
                weight is not None
                and scale_inv is not None
                and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            ):
                block = getattr(module, "block_size", None) or (128, 128)
                self.linear_fp8[
                    (
                        tokens,
                        in_f,
                        out_f,
                        _dtype_name(x.dtype),
                        _dtype_name(weight.dtype),
                        int(block[0]),
                        int(block[1]),
                    )
                ] += 1
                return

            self.linear[(tokens, in_f, out_f, _dtype_name(x.dtype))] += 1

        return hook


def _resolve_dtype(model_name: str, override: str | None):
    """The dtype the model is actually served in, unless the caller names one."""
    import torch
    from transformers import AutoConfig

    if override:
        resolved = getattr(torch, override, None)
        if resolved is None:
            raise ValueError(f"Unknown --dtype {override!r}")
        return resolved
    cfg = AutoConfig.from_pretrained(model_name, **_hf_kwargs())
    # Multimodal configs (*ForConditionalGeneration) nest the language model under
    # text_config and declare no dtype at the top level, so reading only the top level
    # silently falls through to the default -- which is the wrong dtype for any model that
    # is not bfloat16, and a definition's name does not record dtype to catch it later.
    for source in (cfg, getattr(cfg, "text_config", None)):
        if source is None:
            continue
        declared = getattr(source, "torch_dtype", None) or getattr(source, "dtype", None)
        if declared is not None:
            break
    if isinstance(declared, str):
        declared = getattr(torch, declared, None)
    if declared is None:
        logger.warning("%s declares no dtype; defaulting to bfloat16", model_name)
        return torch.bfloat16
    return declared


def _rmsnorm_definition(hidden: int, dtype: str, model: str, eps: float) -> Dict[str, Any]:
    """An RMSNorm definition in the shipped dataset's conventions.

    Names, axes and tags here are not cosmetic. A definition is matched to an
    implementation by `op_type` plus its exact ordered input and output names, so a
    harvested definition that invents its own spelling -- `(x, weight, eps) -> (out)`
    rather than `(hidden_states, weight) -> (output)` -- clears cross-validation and then
    matches no kernel at all. Epsilon likewise belongs in the reference as a module-level
    `EPS`, where the dataset keeps it, not as a declared input.
    """
    return {
        "name": f"rmsnorm_h{hidden}",
        "description": f"RMSNorm over the last dimension, hidden {hidden}. Observed in {model}.",
        "op_type": "rmsnorm",
        "tags": ["stage:extracted", f"model:{model}", "fi_api:flashinfer.norm.rmsnorm"],
        "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": hidden}},
        "inputs": {
            "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": dtype},
            "weight": {"shape": ["hidden_size"], "dtype": dtype},
        },
        "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": dtype}},
        "reference": RMSNORM_REFERENCE.format(hidden=hidden, eps=eps),
    }


def _linear_definition(in_f: int, out_f: int, dtype: str, model: str) -> Dict[str, Any]:
    """A GEMM definition in the shipped dataset's conventions: (A, B) -> (C), axes M/N/K.

    `B` is `[N, K]` and the reference computes `A @ B.T`, matching how `nn.Linear` stores
    its weight -- so a harvested projection maps onto the dataset's existing GEMM
    definitions and solutions rather than forming an island of its own.
    """
    return {
        "name": f"gemm_n{out_f}_k{in_f}",
        "description": (
            f"Bias-free linear projection, {in_f} -> {out_f}. Observed in {model}. "
            "GEMM dominates decoder time, so this is the highest-value optimization target."
        ),
        "op_type": "gemm",
        "tags": ["stage:extracted", f"model:{model}"],
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": out_f},
            "K": {"type": "const", "value": in_f},
        },
        "inputs": {
            "A": {"shape": ["M", "K"], "dtype": dtype},
            "B": {"shape": ["N", "K"], "dtype": dtype},
        },
        "outputs": {"C": {"shape": ["M", "N"], "dtype": dtype}},
        "reference": LINEAR_REFERENCE,
    }


INT4_REFERENCE = """import torch

GROUP_SIZE = {group_size}


@torch.no_grad()
def run(A, B_packed, B_scale, B_zeros):
    \"\"\"W4A16 GEMM: 4-bit weights, per-group scales and zero points.

    Layout is GPTQ's, which is what Intel's `int4_gemm_w4a16` consumes:

      B_packed [K/8, N]    int32, eight weights packed along K, row k = k0*8 + i in
                           bits [4i, 4i+4)
      B_scale  [K/G, N]    one scale per (group of G rows, column)
      B_zeros  [K/G, N/8]  int32, eight zero points packed along N

    An AWQ checkpoint stores `qweight` as [K, N/8] -- packed along N, with an interleaved
    nibble order -- so serving one means repacking to this layout once at load. That is a
    weight transform, not part of the GEMM, and is deliberately outside this definition:
    including it would measure a cost real serving pays once per model, not per token.

    Dequantisation is `(weight - zero) * scale`, no offset.
    \"\"\"
    K_packed, N = B_packed.shape
    K = K_packed * 8
    shifts = (4 * torch.arange(8, device=B_packed.device, dtype=torch.int32)).view(1, 8, 1)
    w = ((B_packed.unsqueeze(1) >> shifts) & 0xF).reshape(K, N)
    z = ((B_zeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF).reshape(B_zeros.shape[0], N)
    scale = B_scale.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=0)[:K]
    zero = z.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=0)[:K]
    weight = (w.to(torch.float32) - zero) * scale
    return (A.to(torch.float32) @ weight).to(A.dtype)
"""


INT8_W8A8_REFERENCE = """import torch


@torch.no_grad()
def run(A_int8, A_scale, B_int8, B_scale):
    \"\"\"W8A8 GEMM: int8 weights and activations, symmetric per-channel/per-token scales.

      A_int8  [M, K]  int8      activations, quantized per token
      A_scale [M, 1]  float32   one scale per token
      B_int8  [N, K]  int8      weights
      B_scale [N, 1]  float32   one scale per output channel

    Scales do not vary along K, unlike a block-scaled FP8 checkpoint, so this is a single
    GEMM with two outer scalings rather than a per-K-block accumulation.
    \"\"\"
    a = A_int8.to(torch.float32) * A_scale.to(torch.float32)
    b = B_int8.to(torch.float32) * B_scale.to(torch.float32)
    return (a @ b.T).to(torch.bfloat16)
"""


def _int8_definition(in_f: int, out_f: int, model: str) -> Dict[str, Any]:
    """A W8A8 GEMM definition for one projection shape."""
    return {
        "name": f"gemm_int8_w8a8_n{out_f}_k{in_f}",
        "description": (
            f"W8A8 GEMM, {in_f} -> {out_f}, int8 weights with per-output-channel scales and "
            f"int8 activations with per-token scales. Observed in {model}."
        ),
        "op_type": "gemm",
        "tags": [
            "stage:extracted",
            f"model:{model}",
            "quantization:int8",
            "quantization:w8a8",
            "quantization:per_channel",
        ],
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": out_f},
            "K": {"type": "const", "value": in_f},
            "one": {"type": "const", "value": 1},
        },
        "inputs": {
            "A_int8": {"shape": ["M", "K"], "dtype": "int8"},
            "A_scale": {"shape": ["M", "one"], "dtype": "float32"},
            "B_int8": {"shape": ["N", "K"], "dtype": "int8"},
            "B_scale": {"shape": ["N", "one"], "dtype": "float32"},
        },
        "outputs": {"C": {"shape": ["M", "N"], "dtype": "bfloat16"}},
        "reference": INT8_W8A8_REFERENCE,
    }


def _int8_shapes_from_checkpoint(model_name: str) -> List[Tuple[int, int]]:
    """Every int8 projection's (K, N), from checkpoint metadata alone.

    A compressed-tensors W8A8 checkpoint stores `weight [N, K]` as int8 beside
    `weight_scale [N, 1]`. As with the 4-bit path, the shape is fully determined by the
    metadata, so no download and no forward pass is needed -- and `transformers` would
    need `compressed-tensors` installed to load the model at all.
    """
    from huggingface_hub import HfApi

    tensors: Dict[str, Any] = {}
    for f in HfApi().get_safetensors_metadata(model_name).files_metadata.values():
        tensors.update(f.tensors)

    shapes: Dict[Tuple[int, int], None] = {}
    for name, info in tensors.items():
        if not name.endswith(".weight_scale"):
            continue
        w = tensors.get(f"{name[: -len('.weight_scale')]}.weight")
        if w is None or str(w.dtype).upper() not in ("I8", "INT8") or len(w.shape) != 2:
            continue
        out_f, in_f = w.shape
        if in_f and out_f:
            shapes.setdefault((in_f, out_f), None)
    return list(shapes)


def _write_int8_definitions(args: Any) -> None:
    """Emit a W8A8 definition per distinct projection shape."""
    shapes = _int8_shapes_from_checkpoint(args.model)
    if not shapes:
        raise SystemExit(
            f"{args.model} declares int8 quantization but exposes no int8 "
            "`weight`/`weight_scale` pairs in its safetensors metadata."
        )
    logger.warning(
        "%s is an int8 W8A8 checkpoint. Definitions come from checkpoint metadata rather "
        "than a forward pass: shapes are exact, no token counts are observed, so workloads "
        "use a standard batch grid.",
        args.model,
    )
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    for in_f, out_f in sorted(shapes):
        definition = _int8_definition(in_f, out_f, args.model)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {"M": m, "N": out_f, "K": in_f, "one": 1},
                    "inputs": {k: {"type": "random"} for k in definition["inputs"]},
                    "uuid": f"{definition['name']}-m{m}",
                },
            }
            for m in (1, 16, 512)
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1
    logger.info(f"Wrote {written} definition(s) to {root}")


def _quantization_bits(model_name: str) -> Optional[int]:
    """Weight bit-width the checkpoint declares, or None if it is not quantized."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, **_hf_kwargs())
    except Exception:
        return None
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = getattr(quant, "to_dict", dict)()
    method = str(quant.get("quant_method", "")).lower()
    if method not in ("awq", "gptq", "compressed-tensors"):
        return None
    bits = quant.get("bits") or quant.get("weight_bits")
    if bits is not None:
        return int(bits)
    # compressed-tensors states the width per config group rather than at the top level.
    groups = quant.get("config_groups") or {}
    for group in groups.values():
        weights = (group or {}).get("weights") or {}
        if weights.get("num_bits"):
            return int(weights["num_bits"])
    return 4  # AWQ/GPTQ without an explicit width are 4-bit by convention


def _is_int4_quantized(model_name: str) -> bool:
    """Whether the checkpoint stores 4-bit weights (AWQ, GPTQ, compressed-tensors)."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, **_hf_kwargs())
    except Exception:
        return False
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = getattr(quant, "to_dict", dict)()
    method = str(quant.get("quant_method", "")).lower()
    if method not in ("awq", "gptq", "compressed-tensors"):
        return False
    bits = quant.get("bits") or quant.get("weight_bits")
    return bits is None or int(bits) == 4


def _write_int4_definitions(args: Any) -> None:
    """Emit a W4A16 definition per distinct projection shape in a 4-bit checkpoint."""
    shapes, act_dtype = _int4_shapes_from_checkpoint(args.model)
    if not shapes:
        raise SystemExit(
            f"{args.model} declares 4-bit quantization but exposes no `.scales`/`.qweight` "
            "pairs in its safetensors metadata; nothing to derive a shape from."
        )
    logger.warning(
        "%s is a 4-bit checkpoint. Definitions are derived from checkpoint metadata rather "
        "than a forward pass -- shapes are exact, and no token counts are observed, so "
        "workloads use a standard batch grid. Note the layout: AWQ stores qweight as "
        "[K, N/8] and these definitions declare GPTQ's [K/8, N], which is what Intel's "
        "kernel consumes. Serving an AWQ checkpoint repacks once at load; that is a weight "
        "transform, not part of the GEMM.",
        args.model,
    )
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    for in_f, out_f, group in sorted(shapes):
        if in_f % 8 or out_f % 8:
            logger.warning("skipping %d->%d: not divisible by 8, cannot pack", in_f, out_f)
            continue
        definition = _int4_definition(in_f, out_f, group, act_dtype, args.model)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {
                        "M": m, "N": out_f, "K": in_f,
                        "K_packed": in_f // 8,
                        "K_groups": -(-in_f // group),
                        "N_packed": out_f // 8,
                    },
                    "inputs": {k: {"type": "random"} for k in definition["inputs"]},
                    "uuid": f"{definition['name']}-m{m}",
                },
            }
            for m in (1, 16, 512)
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1
    logger.info(f"Wrote {written} definition(s) to {root}")


def _int4_definition(in_f: int, out_f: int, group: int, dtype: str, model: str) -> Dict[str, Any]:
    """A W4A16 GEMM definition for one projection shape."""
    return {
        "name": f"gemm_int4_w4a16_g{group}_n{out_f}_k{in_f}",
        "description": (
            f"W4A16 GEMM, {in_f} -> {out_f}, 4-bit weights with per-group-of-{group} scales "
            f"and zero points, {dtype} activations. Observed in {model}."
        ),
        "op_type": "gemm",
        "tags": [
            "stage:extracted",
            f"model:{model}",
            "quantization:int4",
            f"quantization:group{group}",
            "quantization:w4a16",
        ],
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": out_f},
            "K": {"type": "const", "value": in_f},
            "K_packed": {"type": "const", "value": in_f // 8},
            "K_groups": {"type": "const", "value": -(-in_f // group)},
            "N_packed": {"type": "const", "value": out_f // 8},
        },
        "inputs": {
            "A": {"shape": ["M", "K"], "dtype": dtype},
            "B_packed": {"shape": ["K_packed", "N"], "dtype": "int32"},
            "B_scale": {"shape": ["K_groups", "N"], "dtype": dtype},
            "B_zeros": {"shape": ["K_groups", "N_packed"], "dtype": "int32"},
        },
        "outputs": {"C": {"shape": ["M", "N"], "dtype": dtype}},
        "reference": INT4_REFERENCE.format(group_size=group),
    }


def _int4_shapes_from_checkpoint(model_name: str) -> Tuple[List[Tuple[int, int, int]], str]:
    """Every quantized projection's (K, N, group_size), read from checkpoint metadata.

    No forward pass, and no download: HuggingFace serves safetensors headers separately,
    so the tensor shapes come back from a metadata request.

    This deliberately does not load the model. `transformers` refuses an AWQ checkpoint
    without `gptqmodel` installed, and a 4-bit GEMM's shape does not depend on running
    anything -- `scales` is [K/G, N] and `qweight`'s first dimension is K, which is the
    whole definition. Requiring a working quantization backend just to learn a shape would
    make onboarding depend on a package the kernel does not need.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    meta = api.get_safetensors_metadata(model_name)
    tensors: Dict[str, Any] = {}
    for f in meta.files_metadata.values():
        tensors.update(f.tensors)

    shapes: Dict[Tuple[int, int, int], None] = {}
    act_dtype = "float16"
    for name, info in tensors.items():
        if not name.endswith(".scales"):
            continue
        base = name[: -len(".scales")]
        qw = tensors.get(f"{base}.qweight")
        if qw is None:
            continue
        groups, out_f = info.shape
        in_f = qw.shape[0]
        if not groups or not in_f or not out_f:
            continue
        group = -(-in_f // groups)
        shapes.setdefault((in_f, out_f, group), None)
        if str(info.dtype).upper() in ("BF16", "BFLOAT16"):
            act_dtype = "bfloat16"
    return list(shapes), act_dtype


def _compat_remote_code() -> None:
    """Bridge two transformers renames that older in-repo modeling code still uses.

    A model with an `auto_map` ships its own modeling file, pinned to whatever transformers
    version its author had. Two renames since then break such files, and both fail in ways
    that look like our bug rather than theirs:

    1. `_tied_weights_keys` became a `{tied: source}` mapping; a list raises
       `AttributeError: 'list' object has no attribute 'keys'` inside `post_init`, before
       any weight is read. Converted here by finding the parameter that *is* the input
       embedding, so the mapping is derived from the model rather than guessed -- the path
       is not always `model.embed_tokens.weight` (Spark-X2.5 uses `model.embedding`).

    2. `create_causal_mask` takes `inputs_embeds`; code passing `input_embeds` raises
       `TypeError` on the first forward pass.

    Both are shims for *published* code we do not control. A model needing them is not
    validated against this transformers version, which is worth saying in any trace that
    results.
    """
    import transformers
    from transformers import masking_utils
    from transformers.modeling_utils import PreTrainedModel

    if not getattr(PreTrainedModel, "_fib_tied_keys_shim", False):
        original = PreTrainedModel.get_expanded_tied_weights_keys

        def _expanded(self, *args, **kwargs):
            keys = getattr(type(self), "_tied_weights_keys", None)
            if isinstance(keys, (list, tuple)):
                embeddings = self.get_input_embeddings()
                weight = getattr(embeddings, "weight", None)
                source = next(
                    (n for n, p in self.named_parameters(remove_duplicate=False) if p is weight),
                    None,
                )
                if source is not None:
                    type(self)._tied_weights_keys = {k: source for k in keys}
                    logger.warning(
                        "%s declares _tied_weights_keys as a list; converted to {%s: %s}. "
                        "Its modeling code targets an older transformers.",
                        type(self).__name__, keys[0] if keys else "?", source,
                    )
            return original(self, *args, **kwargs)

        PreTrainedModel.get_expanded_tied_weights_keys = _expanded
        PreTrainedModel._fib_tied_keys_shim = True

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        fn = getattr(masking_utils, name, None)
        if fn is None or getattr(fn, "_fib_alias_shim", False):
            continue

        def _aliased(*args, _fn=fn, **kwargs):
            import inspect

            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            # Older callers also pass `cache_position`, which the current signature derives
            # from `position_ids` and `past_key_values` instead. Drop whatever the callee
            # no longer accepts rather than adding one shim per rename -- and say which,
            # because a silently dropped argument to a *mask* builder would change what the
            # model computes, not just whether it runs.
            accepted = set(inspect.signature(_fn).parameters)
            dropped = sorted(k for k in kwargs if k not in accepted)
            if dropped:
                logger.warning(
                    "remote modeling code passes %s to %s, which no longer accepts them; "
                    "dropping. Verify the model's output is coherent before trusting any "
                    "definition extracted from it.",
                    ", ".join(dropped), getattr(_fn, "__name__", "mask builder"),
                )
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            return _fn(*args, **kwargs)

        _aliased._fib_alias_shim = True
        setattr(masking_utils, name, _aliased)
        # The model file does `from ...masking_utils import create_causal_mask`, so the
        # name must also be replaced wherever transformers re-exports it.
        for mod in (transformers, getattr(transformers, "modeling_utils", None)):
            if mod is not None and hasattr(mod, name):
                setattr(mod, name, _aliased)


def _enable_fp8_on_non_cuda(model_name: str) -> None:
    """Make a `quant_method: fp8` model runnable without a CUDA-only hub kernel.

    Two separate transformers problems block an FP8 model on Intel, both upstream:

    1. ``FP8Experts._impl_tp_layer_overrides.get(impl)`` has no default and the table holds
       only ``"deepgemm_megamoe"``. Any model without ``_experts_implementation`` -- every
       dense FP8 model -- gets ``None`` back and is then indexed, raising
       ``AttributeError: 'NoneType' object has no attribute 'get'`` before a single weight
       is read. Nothing about this is Intel-specific; it fails identically on CPU.

    2. ``finegrained_fp8_linear`` dispatches unconditionally to the
       ``kernels-community/finegrained-fp8`` hub kernel, whose builds are CUDA-only, and
       raises ``ImportError`` rather than falling back. So the forward pass cannot run at
       all on any non-CUDA device.

    (2) is replaced with the maths in plain PyTorch: dequantise the packed weight by
    broadcasting each ``(block_n x block_k)`` scale over its tile, then matmul. That is the
    same computation ``LINEAR_FP8_BLOCK_REFERENCE`` describes, which is the point -- the
    reference is what a correct kernel must reproduce, so running the model through it is
    the honest baseline rather than a shortcut.

    This is a no-op for an unquantized model, and should be deleted once upstream fixes
    both. Verified on an FP8 checkpoint on XPU: coherent output, with the weights resident
    at about half their bfloat16 footprint.
    """
    import torch
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name, **_hf_kwargs())
    except Exception:
        return
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = getattr(quant, "to_dict", dict)()
    if quant.get("quant_method") != "fp8":
        return

    try:
        from transformers.integrations import finegrained_fp8 as fg
    except ImportError:
        return

    class _Defaulting(dict):
        def get(self, key, default=None):
            return super().get(key, default) or {}

    fg.FP8Experts._impl_tp_layer_overrides = _Defaulting(fg.FP8Experts._impl_tp_layer_overrides)

    def _torch_fp8_linear(
        input, weight, weight_scale_inv, block_size=None, bias=None,
        activation_scale=None, output_dtype=None, **_dispatch_only,
    ):
        # **_dispatch_only absorbs backend-selection flags (allow_deepgemm) that name
        # paths this implementation does not have.
        out_dtype = output_dtype or input.dtype
        block_n, block_k = (block_size or (128, 128))[:2]
        n, k = weight.shape
        scale = weight_scale_inv.to(torch.float32)
        scale = scale.repeat_interleave(int(block_n), dim=0)
        scale = scale.repeat_interleave(int(block_k), dim=1)[:n, :k]
        dequantized = weight.to(torch.float32) * scale
        out = torch.nn.functional.linear(input.to(torch.float32), dequantized)
        if bias is not None:
            out = out + bias.to(torch.float32)
        return out.to(out_dtype)

    fg.finegrained_fp8_linear = _torch_fp8_linear
    fg.fp8_linear = _torch_fp8_linear
    logger.warning(
        "%s is FP8-quantized; transformers has no non-CUDA kernel for it, so this run "
        "substitutes a plain-PyTorch block-scaled linear. Shapes are real, and the "
        "dequantise-then-matmul latency is not a kernel measurement.",
        model_name,
    )


def _gemm_fp8_block_definition(
    in_f: int, out_f: int, act_dtype: str, w_dtype: str, block_n: int, block_k: int, model: str
) -> Dict[str, Any]:
    """Block-scaled FP8 linear, W8A8: both operands are fp8 by the time the GEMM runs.

    The activation is quantized too, not passed in bfloat16. A checkpoint declaring
    ``activation_scheme: dynamic`` -- which the common FP8 checkpoints do -- quantizes
    each token's K-groups on the fly before the matmul, and transformers'
    ``finegrained_fp8_linear`` and vLLM's ``w8a8_triton_block_scaled_mm`` both do exactly
    that. A definition taking a bfloat16 activation describes W8A16, an operation the
    served model never performs, and no serving kernel can bind to it.

    Signature and operand layout follow vLLM's kernel so a baseline can wrap it directly:
    ``A_fp8 [M, K]``, ``A_scale [M, K/block_k]``, ``B_fp8 [N, K]``,
    ``B_scale [N/block_n, K/block_k]``.
    """
    n_blocks = -(-out_f // block_n)
    k_blocks = -(-in_f // block_k)
    return {
        "name": f"gemm_fp8_w8a8_block{block_n}x{block_k}_n{out_f}_k{in_f}",
        "description": (
            f"Block-scaled FP8 W8A8 linear projection, {in_f} -> {out_f}. Weights carry one "
            f"scale per {block_n}x{block_k} tile; activations are quantized per token-group "
            f"of {block_k} along K. Observed in {model}."
        ),
        "op_type": "gemm",
        "tags": [
            "stage:extracted",
            f"model:{model}",
            f"quantization:{w_dtype}",
            f"quantization:block{block_n}x{block_k}",
            "quantization:w8a8",
        ],
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": out_f},
            "K": {"type": "const", "value": in_f},
            "N_blocks": {"type": "const", "value": n_blocks},
            "K_blocks": {"type": "const", "value": k_blocks},
        },
        "inputs": {
            "A_fp8": {"shape": ["M", "K"], "dtype": w_dtype},
            "A_scale": {"shape": ["M", "K_blocks"], "dtype": "float32"},
            "B_fp8": {"shape": ["N", "K"], "dtype": w_dtype},
            "B_scale": {"shape": ["N_blocks", "K_blocks"], "dtype": "float32"},
        },
        # bfloat16 regardless of `act_dtype`: the activation reaching this GEMM is fp8, and
        # what the layer returns is the accumulator cast down.
        "outputs": {"C": {"shape": ["M", "N"], "dtype": "bfloat16"}},
        "reference": LINEAR_FP8_BLOCK_REFERENCE.format(block_n=block_n, block_k=block_k),
    }


def _top_shapes(counter, shape_key, limit: int):
    """Group a shape+token-count counter by shape, then keep the ``limit`` hottest shapes.

    ``--top-linear`` is a cap on distinct *projections*, not on counter entries. The
    counter is keyed on the token count as well as the shape, so applying
    ``most_common(limit)`` directly makes token counts compete with shapes for the same
    slots: one shape observed at three batch sizes consumes the entire budget, and the
    model's other projections are dropped without a word. Observed on an FP8 checkpoint,
    where it emitted three of five FP8 projections and lost the attention-output and down
    projections.

    Grouping first also keeps *every* token count for the shapes it does keep, which is
    what the workloads should carry.
    """
    totals: Dict[Any, int] = {}
    tokens_by_shape: Dict[Any, List[int]] = {}
    for key, count in counter.items():
        shape = shape_key(key)
        totals[shape] = totals.get(shape, 0) + count
        tokens_by_shape.setdefault(shape, []).append(key[0])
    hottest = sorted(totals, key=lambda s: -totals[s])[:limit]
    return {shape: tokens_by_shape[shape] for shape in hottest}


def _write(root: Path, definition: Dict[str, Any], workloads: List[Dict[str, Any]]) -> None:
    op_type = definition["op_type"]
    (root / "definitions" / op_type).mkdir(parents=True, exist_ok=True)
    (root / "workloads" / op_type).mkdir(parents=True, exist_ok=True)
    (root / "definitions" / op_type / f"{definition['name']}.json").write_text(
        json.dumps(definition, indent=2) + "\n"
    )
    (root / "workloads" / op_type / f"{definition['name']}.jsonl").write_text(
        "\n".join(json.dumps(w) for w in workloads) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace repo id or local path.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Write a short poem about silicon.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None, help="Default: first available accelerator.")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Override the extraction dtype (default: the model config's own).",
    )
    parser.add_argument(
        "--top-linear",
        type=int,
        default=8,
        help="Emit definitions for the N most-called distinct linear shapes.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute modeling code from the model repository. Required for any "
             "architecture transformers does not ship (the config declares an auto_map); "
             "off by default because it runs third-party Python in this process.",
    )
    args = parser.parse_args()
    global _TRUST_REMOTE_CODE
    _TRUST_REMOTE_CODE = args.trust_remote_code

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flashinfer_bench.device import list_devices

    device = args.device or (list_devices() or ["cpu"])[0]
    logger.info(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, **_hf_kwargs())
    # The model's own configured dtype, not a hardcoded one. A definition is named for
    # its shape (`<op>_h<width>`), so a definition extracted in the wrong precision
    # collides with the right one and there is nothing in the name to tell them apart.
    # Extracting a bfloat16 model as float16 produced an fp16 norm definition that vLLM
    # then selected for its bf16 model, returning fp16 activations into the next matmul.
    _bits = _quantization_bits(args.model)
    if _bits == 8:
        # Same reasoning as the 4-bit path: shape comes from metadata, and loading the
        # model would need `compressed-tensors` that the kernel does not require.
        return _write_int8_definitions(args)
    if _bits == 4:
        # A 4-bit checkpoint takes the metadata route: transformers refuses to load one
        # without `gptqmodel`, and nothing about a W4A16 GEMM's shape requires running the
        # model. Emitting definitions here and returning keeps onboarding independent of a
        # quantization backend the kernel itself does not need.
        return _write_int4_definitions(args)

    dtype = _resolve_dtype(args.model, args.dtype)
    logger.info("Extracting %s in %s", args.model, dtype)
    if _TRUST_REMOTE_CODE:
        _compat_remote_code()
    _enable_fp8_on_non_cuda(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **_hf_kwargs())
    model = model.to(device).eval()

    # Base models ship no chat template, and apply_chat_template raises rather than
    # degrading. The template only shapes the prompt text; the kernel shapes we are here to
    # record do not depend on it, so falling back to the raw prompt is correct rather than a
    # compromise.
    text = args.prompt
    if getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:
            logger.warning("chat template unusable (%s); using the raw prompt", exc)
    else:
        logger.info("no chat template (base model); using the raw prompt")
    encoded = tokenizer(text, return_tensors="pt").to(device)

    recorder = ShapeRecorder()
    recorder.attach(model)
    with torch.no_grad():
        model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
    recorder.detach()

    # Say what was NOT covered. This hooks RMSNorm and Linear only, so a hybrid or SSM model
    # has whole families -- Mamba state updates, convolutions, attention -- that produce no
    # definitions at all. Reporting only what was found reads as completeness and is not.
    covered = (
        "rmsnorm",
        "linear",
        "embedding",
        "dropout",
        "identity",
        "modulelist",
        "moduledict",
        "sequential",
    )
    uncovered: Dict[str, int] = {}
    for module in model.modules():
        cls = type(module).__name__
        if any(k in cls.lower() for k in covered) or not list(module.children()) == []:
            continue
        uncovered[cls] = uncovered.get(cls, 0) + 1
    if uncovered:
        top = ", ".join(
            f"{k} x{v}" for k, v in sorted(uncovered.items(), key=lambda kv: -kv[1])[:6]
        )
        logger.warning(
            "Leaf modules this extractor does not hook (no definitions emitted for them): %s. "
            "Cover them with Path B (transcribe from sources) before calling the model done.",
            top,
        )

    logger.info(
        f"Observed {sum(recorder.rmsnorm.values())} RMSNorm, "
        f"{sum(recorder.linear.values())} dense linear and "
        f"{sum(recorder.linear_fp8.values())} block-scaled FP8 linear calls"
    )

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    written = 0

    # RMSNorm: one definition per hidden size, one workload per observed token count.
    by_hidden: Dict[Tuple[int, str, float], List[int]] = {}
    for (tokens, hidden, dtype, eps), _count in recorder.rmsnorm.items():
        by_hidden.setdefault((hidden, dtype, eps), []).append(tokens)

    # A definition is named by width alone, so two norms of the same width differing in
    # dtype or eps collide -- and a plain write silently drops one. Observed on a hybrid
    # model: a bf16 and an fp32 norm of the same width both existed and only the fp32 survived.
    # Disambiguate the minority variants rather than losing them.
    seen_names: Dict[str, Tuple[str, float]] = {}
    for (hidden, dtype, eps), token_counts in sorted(by_hidden.items()):
        definition = _rmsnorm_definition(hidden, dtype, args.model, eps)
        base = definition["name"]
        if base in seen_names:
            prev_dtype, prev_eps = seen_names[base]
            suffix = f"_{dtype}" if dtype != prev_dtype else f"_eps{eps:g}"
            definition["name"] = base + suffix
            logger.warning(
                "%s already emitted as (%s, eps=%g); this one is (%s, eps=%g) -- writing it "
                "as %s. A dataset definition is named by width only, so these would "
                "otherwise overwrite each other.",
                base,
                prev_dtype,
                prev_eps,
                dtype,
                eps,
                definition["name"],
            )
        else:
            seen_names[base] = (dtype, eps)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {"batch_size": t, "hidden_size": hidden},
                    "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}},
                    "uuid": f"{definition['name']}-t{t}",
                },
            }
            for t in sorted(set(token_counts))
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1

    # Linear: the most-called shapes carry the decoder's GEMM time.
    by_shape = _top_shapes(recorder.linear, lambda k: (k[1], k[2], k[3]), args.top_linear)

    for (in_f, out_f, dtype), token_counts in sorted(by_shape.items()):
        definition = _linear_definition(in_f, out_f, dtype, args.model)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {"M": t, "N": out_f, "K": in_f},
                    "inputs": {"A": {"type": "random"}, "B": {"type": "random"}},
                    "uuid": f"{definition['name']}-t{t}",
                },
            }
            for t in sorted(set(token_counts))
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1

    # Block-scaled FP8 linears, if the model is quantized.
    by_shape_fp8 = _top_shapes(
        recorder.linear_fp8, lambda k: (k[1], k[2], k[3], k[4], k[5], k[6]), args.top_linear
    )

    for (in_f, out_f, act_dt, w_dt, bn, bk), token_counts in sorted(by_shape_fp8.items()):
        definition = _gemm_fp8_block_definition(in_f, out_f, act_dt, w_dt, bn, bk, args.model)
        n_blocks = -(-out_f // bn)
        k_blocks = -(-in_f // bk)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {
                        "M": t,
                        "N": out_f,
                        "K": in_f,
                        "N_blocks": n_blocks,
                        "K_blocks": k_blocks,
                    },
                    "inputs": {
                        "A_fp8": {"type": "random"},
                        "A_scale": {"type": "random"},
                        "B_fp8": {"type": "random"},
                        "B_scale": {"type": "random"},
                    },
                    "uuid": f"{definition['name']}-t{t}",
                },
            }
            for t in sorted(set(token_counts))
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1

    logger.info(f"Wrote {written} definition(s) to {root}")
    logger.info(
        "Next: flashinfer-bench validate-references --local "
        f"{root} --device {device}, then add-baselines and run."
    )


if __name__ == "__main__":
    main()
