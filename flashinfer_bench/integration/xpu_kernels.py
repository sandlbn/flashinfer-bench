"""Upstream Intel GPU kernels as benchmark baselines.

Beating PyTorch eager is the easy bar. The bar that matters for Intel is beating the SYCL
kernels that vLLM and SGLang already ship, since those are what a real deployment runs.
This module turns those upstream kernels into ordinary Solutions so they can be measured
against the same Definitions, with the same correctness gates, as anything else.

A baseline is expressed as a plain Python solution that calls the upstream operator. That
is deliberate: it means the existing builder, evaluator, validator and trace format all
work unchanged, and an upstream kernel appears in results beside a hand-written one rather
than in a separate report.

Two providers are supported:

- ``vllm-xpu`` -- github.com/vllm-project/vllm-xpu-kernels, registered under
  ``torch.ops._C``.
- ``sgl-kernel-xpu`` -- github.com/sgl-project/sgl-kernel-xpu, exposed as the
  ``sgl_kernel`` Python package.

Neither is a dependency. When a provider is not installed, its baselines are simply not
offered.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

logger = logging.getLogger(__name__)

VLLM_XPU = "vllm-xpu"
"""vLLM's Intel kernel library."""

SGL_KERNEL_XPU = "sgl-kernel-xpu"
"""SGLang's Intel kernel library."""

VLLM = "vllm"
"""vLLM itself, distinct from its Intel kernel package.

Its quantization kernels are Triton, not SYCL, and live in the main `vllm` distribution
rather than in `vllm-xpu-kernels`. They are portable by construction and several of them
run on XPU today, so they are a legitimate Intel baseline even though nothing about them
is Intel-specific.
"""

_PROVIDER_MODULES: Dict[str, str] = {
    VLLM_XPU: "vllm_xpu_kernels",
    SGL_KERNEL_XPU: "sgl_kernel",
    VLLM: "vllm",
}


_EPS_PATTERN = re.compile(r"^\s*EPS\s*=\s*([0-9][0-9._eE+-]*)\s*$", re.MULTILINE)
"""How a definition records its epsilon.

Definitions bake epsilon into the reference as a module-level ``EPS`` assignment rather
than declaring it as an input, while every upstream kernel takes it as an argument. The
wrapper has to bridge that, and the value is not uniform across the dataset -- most
definitions use 1e-6 but some use 1e-5 -- so it is read from the reference, never assumed.
"""

DEFAULT_EPS = 1e-6
"""Fallback when a definition's reference does not state one.

This is the RMSNorm default in both vLLM and SGLang
(``vllm/model_executor/layers/layernorm.py``, ``sglang/srt/layers/layernorm.py``), so a
definition that omits epsilon gets what the frameworks themselves would have used. Applied
with a warning, because a silently wrong epsilon produces a baseline that runs cleanly and
computes the wrong thing.
"""


def definition_eps(definition: Definition) -> float:
    """Epsilon for ``definition``, from its reference, else :data:`DEFAULT_EPS`."""
    match = _EPS_PATTERN.search(definition.reference or "")
    if match is None:
        logger.warning(
            "Definition '%s' does not state an EPS in its reference; using the "
            "vLLM/SGLang default %g. Check this if the baseline fails correctness.",
            definition.name,
            DEFAULT_EPS,
        )
        return DEFAULT_EPS
    try:
        return float(match.group(1))
    except ValueError:
        logger.warning(
            "Definition '%s' has an unparseable EPS (%r); using %g.",
            definition.name,
            match.group(1),
            DEFAULT_EPS,
        )
        return DEFAULT_EPS


def definition_is_neox(definition: Definition) -> bool:
    """Whether a rope definition uses NeoX-style rotation.

    The two styles interleave differently and produce different numbers, so this cannot
    be assumed. Read from the definition's own naming, which is where the dataset records
    it -- there is no structured field for it today.
    """
    haystack = f"{definition.name} {' '.join(definition.tags)}".lower()
    if "neox" in haystack:
        return True
    if "gptj" in haystack or "interleav" in haystack:
        return False
    logger.warning(
        "Definition '%s' does not say which rope style it uses; assuming NeoX. A wrong "
        "guess here computes the wrong thing without failing to run.",
        definition.name,
    )
    return True


def _definition_block_size(definition: Definition, axis: str) -> int:
    """Weight-quantization block extent along ``axis``, from what the definition declares.

    Read from the axes rather than assumed, because it is what the scale tensor's shape
    means: `B_scale` is `[N_blocks, K_blocks]`, so a kernel told the wrong extent reads
    the wrong scale for every block and is silently wrong rather than slow.
    """
    axes = definition.axes
    full = getattr(axes.get(axis), "value", None)
    blocks = getattr(axes.get(f"{axis}_blocks"), "value", None)
    if full and blocks:
        return -(-int(full) // int(blocks))  # blocks were formed by ceiling division
    for tag in definition.tags:
        if str(tag).startswith("quantization:block"):
            n, _, k = str(tag)[len("quantization:block") :].partition("x")
            if n.isdigit() and k.isdigit():
                return int(n) if axis == "N" else int(k)
    raise ValueError(
        f"Definition '{definition.name}' declares no {axis}_blocks axis and no "
        "quantization:block{n}x{k} tag, so the weight block size is unknown."
    )


def _definition_group_size(definition: Definition) -> int:
    """Weight-quantization group size along K, from what the definition declares.

    Derived from `K / K_groups` rather than assumed: the kernel indexes scales in units of
    this, so a wrong value reads the wrong scale for every group and is silently wrong.
    """
    axes = definition.axes
    k = getattr(axes.get("K"), "value", None)
    groups = getattr(axes.get("K_groups"), "value", None)
    if k and groups:
        return -(-int(k) // int(groups))
    for tag in definition.tags:
        if str(tag).startswith("quantization:group"):
            rest = str(tag)[len("quantization:group") :]
            if rest.isdigit():
                return int(rest)
    raise ValueError(
        f"Definition '{definition.name}' declares no K_groups axis and no "
        "quantization:group{n} tag, so the weight group size is unknown."
    )


_CONSTANT_RESOLVERS: Dict[str, Callable[[Definition], object]] = {
    "eps": definition_eps,
    "group_size": _definition_group_size,
    "is_neox": definition_is_neox,
    "block_n": lambda d: _definition_block_size(d, "N"),
    "block_k": lambda d: _definition_block_size(d, "K"),
}
"""Named constants a wrapper template may request, and how to obtain each."""


@dataclass(frozen=True)
class BaselineKernel:
    """An upstream kernel that can serve a Definition.

    Parameters
    ----------
    provider : str
        Which upstream library ships it.
    name : str
        The upstream operator name, used in the generated solution's name so results say
        exactly what was measured.
    op_type : str
        The Definition ``op_type`` this kernel implements.
    inputs : Tuple[str, ...]
        Definition input names, in order, that the wrapper expects.
    outputs : Tuple[str, ...]
        Definition output names, in order.
    source : str
        Python source template defining ``run(...)``, calling the upstream operator.
        Formatted with the constants named in ``constants`` before use.
    description : str
        What the upstream kernel does, for the solution record.
    constants : Tuple[str, ...]
        Names from :data:`_CONSTANT_RESOLVERS` that ``source`` interpolates. Resolved
        against the definition, because values like epsilon live in the definition's
        reference rather than in its declared inputs.
    fi_api : Optional[str]
        Exact operation this kernel implements, matched against the definition's
        ``fi_api:`` tag. Required wherever a signature does not identify the operation --
        every gated activation is one input and one output, so ``silu_and_mul`` and
        ``gelu_and_mul`` are indistinguishable by signature and binding the wrong one
        computes something else without failing. ``None`` means the signature is enough.
    """

    provider: str
    name: str
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    source: str
    description: str
    constants: Tuple[str, ...] = ()
    fi_api: Optional[str] = None

    def render(self, definition: Definition) -> str:
        """The wrapper source for ``definition``, with its constants substituted."""
        if not self.constants:
            return self.source
        # Only what this wrapper declares. Resolving every constant would run resolvers a
        # kernel has no use for -- epsilon warns when a definition states none, which an
        # activation definition never does.
        values = {name: _CONSTANT_RESOLVERS[name](definition) for name in self.constants}
        return self.source.format(**values)

    @property
    def solution_name(self) -> str:
        return f"{self.provider.replace('-', '_')}_{self.name}"

    def matches(self, definition: Definition) -> bool:
        """Whether this kernel implements ``definition``.

        Matching is deliberately strict -- same op_type, and exactly the same input and
        output names in the same order. A baseline that silently binds to the wrong
        definition would produce a confidently wrong comparison.
        """
        if (
            definition.op_type != self.op_type
            or tuple(definition.inputs) != self.inputs
            or tuple(definition.outputs) != self.outputs
        ):
            return False
        if self.fi_api is None:
            return True
        return f"fi_api:{self.fi_api}" in definition.tags


_RMS_NORM_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

EPS = {eps!r}


def run(hidden_states, weight):
    out = torch.empty_like(hidden_states)
    torch.ops._C.rms_norm(out, hidden_states, weight, EPS)
    return out
"""

_RMS_NORM_SGL = """import torch
import sgl_kernel

EPS = {eps!r}


def run(hidden_states, weight):
    return sgl_kernel.rmsnorm(hidden_states, weight, EPS)
"""

_FUSED_ADD_RMS_NORM_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

EPS = {eps!r}


def run(hidden_states, residual, weight):
    # The upstream op updates both tensors in place, so clone to keep the benchmark's
    # inputs reusable across trials.
    hidden = hidden_states.clone()
    res = residual.clone()
    torch.ops._C.fused_add_rms_norm(hidden, res, weight, EPS)
    # The definition declares one output: the normalized result. Upstream also returns the
    # updated residual in `res`, which this definition does not model -- returning it too
    # would not match the declared arity.
    return hidden
"""

_FUSED_ADD_RMS_NORM_SGL = """import torch
import sgl_kernel

EPS = {eps!r}


def run(hidden_states, residual, weight):
    # In-place upstream; clone so repeated trials see identical inputs.
    hidden = hidden_states.clone()
    res = residual.clone()
    sgl_kernel.fused_add_rmsnorm(hidden, res, weight, EPS)
    return hidden
"""

_SILU_AND_MUL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out
"""

_MUL_AND_SILU_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.mul_and_silu(out, x)
    return out
"""

_GELU_AND_MUL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_and_mul(out, x)
    return out
"""

_GELU_TANH_AND_MUL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_tanh_and_mul(out, x)
    return out
"""

_GELU_NEW_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    out = torch.empty_like(x)
    torch.ops._C.gelu_new(out, x)
    return out
"""

_GELU_FAST_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    out = torch.empty_like(x)
    torch.ops._C.gelu_fast(out, x)
    return out
"""

_GELU_QUICK_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    out = torch.empty_like(x)
    torch.ops._C.gelu_quick(out, x)
    return out
"""


_ROPE_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

IS_NEOX = {is_neox!r}


def run(q, k, cos_sin_cache, positions):
    # Upstream rotates in place on a flattened [num_tokens, heads * head_size] view, so
    # clone and reshape rather than mutating the benchmark's inputs.
    num_tokens = q.shape[0]
    head_size = q.shape[-1]
    q_flat = q.clone().reshape(num_tokens, -1)
    k_flat = k.clone().reshape(num_tokens, -1)
    # The definition stores cos/sin in float32 and the reference rotates in float32;
    # upstream requires the cache in the query dtype and rotates there. Measured against
    # the reference on Battlemage that costs ~0.007 relative error -- inside the default
    # 1e-2 tolerance, but it is an approximation, not an equivalence.
    torch.ops._C.rotary_embedding(
        positions, q_flat, k_flat, head_size, cos_sin_cache.to(q.dtype), IS_NEOX
    )
    return q_flat.view_as(q), k_flat.view_as(k)
"""


_W8A8_BLOCK_SCALED_MM_VLLM = """import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)

BLOCK_N = {block_n}
BLOCK_K = {block_k}


def run(A_fp8, A_scale, B_fp8, B_scale):
    # vLLM's own Triton kernel for this operation, and the one a served FP8 model runs on
    # any backend. Portable Triton, no CUDA guard, so it runs on XPU as-is.
    #
    # It asserts A is contiguous and that the scale shapes agree with the block size, so a
    # mismatch fails loudly rather than reading the wrong scales.
    return w8a8_triton_block_scaled_mm(
        A_fp8, B_fp8, A_scale, B_scale, [BLOCK_N, BLOCK_K], torch.bfloat16
    )
"""


_GQA_PAGED_DECODE_SGL = """import math
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache


def run(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    # sgl-kernel-xpu's paged attention wants a dense [batch, max_pages] page table and a
    # token count per sequence. The definition carries the FlashInfer representation --
    # a ragged page list (`kv_indices`) with row offsets (`kv_indptr`) -- so the wrapper's
    # real work is the translation, and getting it wrong runs cleanly and computes the
    # wrong thing.
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    counts = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int64)
    max_pages = int(counts.max().item()) if batch else 0

    # Scatter the ragged rows into a padded matrix. `kv_indices` is already in row order,
    # so the mask selects exactly the slots each row owns, in order.
    page_table = torch.zeros((batch, max_pages), dtype=torch.int32, device=q.device)
    slot = torch.arange(max_pages, device=q.device).unsqueeze(0)
    page_table[slot < counts.unsqueeze(1)] = kv_indices.to(torch.int32)

    # Full pages plus the partially filled last one. Using pages * page_size instead would
    # attend to uninitialised tail entries.
    seqlens = ((counts - 1) * page_size + kv_last_page_len.to(torch.int64)).to(torch.int32)

    out, lse = flash_attn_with_kvcache(
        q.unsqueeze(1),  # the kernel takes [batch, seqlen_q, heads, dim]; decode is 1
        k_cache,
        v_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        softmax_scale=float(sm_scale),
        causal=False,  # one query token per sequence, so nothing to mask
        return_softmax_lse=True,
    )
    # The kernel returns lse as [heads, tokens] in natural log; the definition declares
    # [tokens, heads] in log2. Passing it through unconverted is a silent 1.44x error.
    return out.squeeze(1), lse.float().transpose(0, 1).contiguous() / math.log(2.0)
"""


_SAMPLER_PREAMBLE = """import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

# Importing `_xpu_C` specifically is what registers `torch.ops._xpu_C`. Importing the
# package, or `._C`, leaves this op looking absent -- it raises AttributeError on the
# namespace and reads exactly like a kernel that was not built.


# Bound the log() temporary. The kernel takes logits and the definitions carry
# probabilities, so the wrapper has to materialise a second tensor the same size as
# `probs`. The sampling evaluator pads the batch to 10000 rows to measure a frequency
# distribution, which at a 128k vocabulary is 5.1 GiB for `probs` alone -- allocating
# another 5.1 GiB beside it exhausts a 12 GiB card. Converting a slice at a time keeps the
# temporary at this many rows regardless of how large the batch gets.
_LOGIT_CHUNK_ROWS = 1024


def _sample(probs, top_k, top_p):
    batch = probs.shape[0]
    out = torch.empty(batch, dtype=torch.int64, device=probs.device)
    if batch > _LOGIT_CHUNK_ROWS:
        for lo in range(0, batch, _LOGIT_CHUNK_ROWS):
            hi = min(lo + _LOGIT_CHUNK_ROWS, batch)
            out[lo:hi] = _sample(
                probs[lo:hi],
                None if top_k is None else top_k[lo:hi],
                None if top_p is None else top_p[lo:hi],
            )
        return out
    logits = probs.log()
    # A fresh seed per call: the evaluator compares sampled frequencies against the
    # reference distribution, so repeated calls have to be independent draws rather than
    # the same one. `seeds` is a CPU tensor of (seed, offset), and `k` must be int64 --
    # int32 raises "expected scalar type Long but found Int".
    seeds = torch.tensor(
        [int(torch.randint(0, 2**62, (1,)).item()), 0], dtype=torch.int64
    )
    torch.ops._xpu_C.topk_topp_sampler(
        out,
        None,
        logits,
        None if top_k is None else top_k.to(torch.int64),
        None if top_p is None else top_p.to(torch.float32),
        "raw_logprobs",
        seeds,
        1.0,
    )
    return out
"""

_SAMPLE_TOPK_TOPP_VLLM = _SAMPLER_PREAMBLE + """

def run(probs, top_k, top_p):
    return _sample(probs, top_k, top_p)
"""

_SAMPLE_TOPK_VLLM = _SAMPLER_PREAMBLE + """

def run(probs, top_k):
    return _sample(probs, top_k, None)
"""

_SAMPLE_TOPP_VLLM = _SAMPLER_PREAMBLE + """

def run(probs, top_p):
    return _sample(probs, None, top_p)
"""


_INT4_W4A16_GEMM_VLLM = """import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C)

GROUP_SIZE = {group_size}


# float16 above 65504 is inf. bfloat16 carries float32's exponent range, so a bfloat16
# activation can hold values float16 cannot -- rare in a served model (activations sit
# within single digits) but not impossible, and silently producing inf would be worse than
# being slower.
_FP16_MAX = 65504.0


def run(A, B_packed, B_scale, B_zeros):
    # oneDNN-backed W4A16 matmul. The kernel requires the packed weight in "NT format",
    # which its own check spells as `B.strides()[-2] == 1` -- column-major. A row-major
    # [K/8, N] tensor is rejected outright with "Int4 weight must be in NT format!", so
    # the transpose here is not a layout preference, it is the calling convention.
    B = B_packed.t().contiguous().t()

    # Compute in float16 even when the model is bfloat16. This kernel is only validated
    # for float16 upstream, and it shows: on Arc B580 with bfloat16 activations it matches
    # 88% of elements within (1e-2, 1e-2) against a float32 reference, and 99.5% when the
    # same inputs are run through the float16 path. bfloat16 has 7 mantissa bits to
    # float16's 10, and a dequantised 4-bit weight cannot spare three of them.
    #
    # Skipped if anything would overflow float16, which bfloat16's wider exponent allows.
    upconvert = A.dtype is torch.bfloat16 and max(
        A.abs().max().item(), B_scale.abs().max().item()
    ) < _FP16_MAX
    if upconvert:
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            A.to(torch.float16), B, None, B_scale.to(torch.float16),
            B_zeros, GROUP_SIZE, None,
        )
        return out.to(A.dtype)

    return torch.ops._xpu_C.int4_gemm_w4a16(
        A, B, None, B_scale, B_zeros, GROUP_SIZE, None
    )
"""


_GDN_DECODE_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C)


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    # Decode: one token per sequence.
    batch = q.shape[0]
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[3]
    device = q.device

    # The kernel writes into `core_attn_out` and `ssm_state`, and normalises q/k in place.
    # The benchmark reuses input tensors across trials, so every mutated argument is a
    # copy: without this, trial 2 runs on q/k the previous trial normalised.
    out = torch.empty((batch, num_v_heads, head_v_dim), dtype=q.dtype, device=device)
    ssm_state = state.clone()

    torch.ops._xpu_C.gated_delta_rule_non_spec(
        out,
        q.squeeze(1).clone(),
        k.squeeze(1).clone(),
        v.squeeze(1),
        # Raw `b`: the kernel applies sigmoid itself. Passing sigmoid(b) double-applies it.
        b.squeeze(1),
        a.squeeze(1),
        num_v_heads,
        head_v_dim,
        A_log,                      # must stay float32
        dt_bias.to(q.dtype),        # must match the output dtype, not its declared float32
        ssm_state,
        0,                          # num_prefills
        batch,                      # num_decodes
        0,                          # num_spec_decodes
        None,                       # has_initial_state: state is always provided here
        torch.arange(batch + 1, device=device, dtype=torch.int32),
        None,
        torch.arange(batch, device=device, dtype=torch.int32),
        batch,                      # num_actual_tokens
        1,                          # tp_size
    )
    return out.unsqueeze(1), ssm_state
"""


_DSA_SPARSE_ATTN_SGL = """import math
import torch
from sgl_kernel import flash_mla_sparse_fwd


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    # The kernel takes one fused query and one *ragged* kv, where the definition carries
    # them split and paged. Both joins are along the last dimension and the flattening is
    # exactly what the reference does (`cache.reshape(-1, head_dim)`), so `sparse_indices`
    # already indexes the flattened cache and needs no translation -- including its -1
    # sentinel, which the kernel documents as the invalid marker.
    q = torch.cat([q_nope, q_pe], dim=-1)
    kv = torch.cat(
        [ckv_cache.reshape(-1, ckv_cache.shape[-1]), kpe_cache.reshape(-1, kpe_cache.shape[-1])],
        dim=-1,
    ).unsqueeze(1)

    output, _max_logits, lse = flash_mla_sparse_fwd(
        q, kv, sparse_indices.unsqueeze(1), float(sm_scale), q_nope.shape[-1]
    )
    # The kernel returns lse in natural log; the definition declares log2. Passing it
    # through unconverted is a silent 1.44x error on an output nothing else checks.
    return output, lse.float() / math.log(2.0)
"""


_MLA_PAGED_DECODE_SGL = """import math
import torch
from sgl_kernel import flash_mla_with_kvcache


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # `flash_mla_with_kvcache` is the MLA op that returns lse; `flash_mla_decode` and
    # `flash_mla_prefill` return the output alone, which is why they cannot serve these
    # definitions.
    batch = q_nope.shape[0]
    device = q_nope.device

    # One fused query [B, s_q=1, H, 576] and one fused cache [num_pages, P, 1, 576].
    q = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)
    k_cache = torch.cat([ckv_cache, kpe_cache], dim=-1).unsqueeze(2)

    # Ragged page list -> dense block table. Pages per sequence are the row lengths, and
    # `kv_indices` is already in row order so the mask selects each row's slots in order.
    counts = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int64)
    max_pages = int(counts.max().item()) if batch else 0
    block_table = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    slot = torch.arange(max_pages, device=device).unsqueeze(0)
    block_table[slot < counts.unsqueeze(1)] = kv_indices.to(torch.int32)

    page_size = ckv_cache.shape[1]
    cache_seqlens = (counts * page_size).to(torch.int32)

    out, lse = flash_mla_with_kvcache(
        q, k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=q_nope.shape[-1],
        softmax_scale=float(sm_scale),
        causal=False,   # decode is one query token; the op rejects causal on XPU anyway
    )
    # out [B, s_q, H, Dv] -> [B, H, Dv]; lse [B, H, s_q] -> [B, H], natural log -> log2.
    return out.squeeze(1), lse.squeeze(-1).float() / math.log(2.0)
"""


_RAGGED_PREFILL_NO_LSE_SGL = """import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func


def _max_seqlen(indptr):
    # Host-side, and unavoidable: the kernel takes max_seqlen as an int.
    return int((indptr[1:] - indptr[:-1]).max().item()) if indptr.numel() > 1 else 0


def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    # Causal here is bottom-right aligned, matching the reference's
    # `delta = kv_len - q_len`; no adjustment is needed. Verified to 3.3e-03.
    #
    # `return_softmax_lse` is deliberately absent: asking for it alongside causal masking
    # raises on this build, which is why these definitions are the output-only variants.
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=qo_indptr,
        cu_seqlens_k=kv_indptr,
        max_seqlen_q=_max_seqlen(qo_indptr),
        max_seqlen_k=_max_seqlen(kv_indptr),
        softmax_scale=float(sm_scale),
        causal=True,
    )
    return out[0] if isinstance(out, tuple) else out
"""


REGISTRY: Tuple[BaselineKernel, ...] = (
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="flash_attn_varlen_func",
        op_type="gqa_ragged",
        inputs=("q", "k", "v", "qo_indptr", "kv_indptr", "sm_scale"),
        outputs=("output",),
        source=_RAGGED_PREFILL_NO_LSE_SGL,
        description="sgl-kernel-xpu ragged causal prefill FMHA (SYCL), output only.",
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="flash_attn_varlen_func_mla",
        op_type="mla_ragged",
        inputs=("q", "k", "v", "qo_indptr", "kv_indptr", "sm_scale"),
        outputs=("output",),
        source=_RAGGED_PREFILL_NO_LSE_SGL,
        description=(
            "sgl-kernel-xpu ragged causal prefill FMHA (SYCL), output only, at MLA's "
            "asymmetric head dims. qk=192 and vo=128 are both in the dispatch table and "
            "the kernel plumbs dv independently of d."
        ),
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="flash_mla_with_kvcache",
        op_type="mla_paged",
        inputs=("q_nope", "q_pe", "ckv_cache", "kpe_cache", "kv_indptr", "kv_indices", "sm_scale"),
        outputs=("output", "lse"),
        source=_MLA_PAGED_DECODE_SGL,
        description=(
            "sgl-kernel-xpu MLA paged decode (SYCL), via the kvcache entry point that "
            "returns lse. Fuses the split ckv/kpe query and cache, and converts the "
            "ragged page list into a dense block table."
        ),
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="flash_mla_sparse_fwd",
        op_type="dsa_paged",
        inputs=("q_nope", "q_pe", "ckv_cache", "kpe_cache", "sparse_indices", "sm_scale"),
        outputs=("output", "lse"),
        source=_DSA_SPARSE_ATTN_SGL,
        description=(
            "sgl-kernel-xpu sparse MLA attention (SYCL). The one paged-attention op on "
            "Intel that emits lse alongside the output."
        ),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gated_delta_rule_non_spec",
        op_type="gdn",
        inputs=("q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "scale"),
        outputs=("output", "new_state"),
        source=_GDN_DECODE_VLLM,
        description=(
            "vLLM XPU gated delta rule, decode path (SYCL). Fuses the q/k L2 "
            "normalisation, so it binds only to definitions whose boundary includes it."
        ),
        fi_api="fusion_l2norm_qk",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="int4_gemm_w4a16",
        op_type="gemm",
        inputs=("A", "B_packed", "B_scale", "B_zeros"),
        outputs=("C",),
        source=_INT4_W4A16_GEMM_VLLM,
        description=(
            "vLLM XPU W4A16 GEMM, oneDNN-backed. This is the s4-weights-against-f16-compute "
            "path that dominates Intel's Xe2 GEMM catalog."
        ),
        constants=("group_size",),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="topk_topp_sampler",
        op_type="sampling",
        inputs=("probs", "top_k", "top_p"),
        outputs=("samples",),
        source=_SAMPLE_TOPK_TOPP_VLLM,
        description="vLLM XPU fused top-k + top-p sampler (SYCL).",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="topk_sampler",
        op_type="sampling",
        inputs=("probs", "top_k"),
        outputs=("samples",),
        source=_SAMPLE_TOPK_VLLM,
        description="vLLM XPU sampler with top-k only (SYCL).",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="topp_sampler",
        op_type="sampling",
        inputs=("probs", "top_p"),
        outputs=("samples",),
        source=_SAMPLE_TOPP_VLLM,
        description="vLLM XPU sampler with top-p only (SYCL).",
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="flash_attn_with_kvcache",
        op_type="gqa_paged",
        inputs=("q", "k_cache", "v_cache", "kv_indptr", "kv_indices", "kv_last_page_len", "sm_scale"),
        outputs=("output", "lse"),
        source=_GQA_PAGED_DECODE_SGL,
        description=(
            "sgl-kernel-xpu paged GQA decode (SYCL FMHA). Translates the definition's "
            "ragged page list into the dense page table the kernel wants, and converts "
            "lse from natural log to log2."
        ),
    ),
    BaselineKernel(
        provider=VLLM,
        name="w8a8_triton_block_scaled_mm",
        op_type="gemm",
        inputs=("A_fp8", "A_scale", "B_fp8", "B_scale"),
        outputs=("C",),
        source=_W8A8_BLOCK_SCALED_MM_VLLM,
        description=(
            "vLLM block-scaled FP8 W8A8 GEMM (Triton). One fused kernel: the accumulator "
            "stays in registers across K-blocks, which is what a composition of oneDNN "
            "primitives cannot do."
        ),
        constants=("block_n", "block_k"),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="rotary_embedding",
        op_type="rope",
        inputs=("q", "k", "cos_sin_cache", "positions"),
        outputs=("q_out", "k_out"),
        source=_ROPE_VLLM,
        constants=("is_neox",),
        description=(
            "vLLM XPU rotary_embedding (SYCL): applies rotary position embedding to q "
            "and k in place, with partial-rotary support driven by the cache width."
        ),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="silu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_SILU_AND_MUL_VLLM,
        description="vLLM XPU silu_and_mul (SYCL): SwiGLU gate, silu(x[..., :d]) * x[..., d:].",
        fi_api="flashinfer.activation.silu_and_mul",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="mul_and_silu",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_MUL_AND_SILU_VLLM,
        description="vLLM XPU mul_and_silu (SYCL): x[..., :d] * silu(x[..., d:]).",
        fi_api="flashinfer.activation.mul_and_silu",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_AND_MUL_VLLM,
        description="vLLM XPU gelu_and_mul (SYCL): GeGLU with exact gelu.",
        fi_api="flashinfer.activation.gelu_and_mul",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_tanh_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_TANH_AND_MUL_VLLM,
        description="vLLM XPU gelu_tanh_and_mul (SYCL): GeGLU with tanh-approximate gelu.",
        fi_api="flashinfer.activation.gelu_tanh_and_mul",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_new",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_NEW_VLLM,
        description="vLLM XPU gelu_new (SYCL): tanh-approximate GELU, elementwise.",
        fi_api="flashinfer.activation.gelu_new",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_fast",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_FAST_VLLM,
        description="vLLM XPU gelu_fast (SYCL): sigmoid-approximate GELU, elementwise.",
        fi_api="flashinfer.activation.gelu_fast",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_quick",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_QUICK_VLLM,
        description="vLLM XPU gelu_quick (SYCL): x * sigmoid(1.702 x), elementwise.",
        fi_api="flashinfer.activation.gelu_quick",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="rms_norm",
        op_type="rmsnorm",
        inputs=("hidden_states", "weight"),
        outputs=("output",),
        source=_RMS_NORM_VLLM,
        description="vLLM XPU rms_norm (SYCL).",
        constants=("eps",),
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "weight"),
        outputs=("output",),
        source=_RMS_NORM_SGL,
        description="SGLang XPU rmsnorm (SYCL).",
        constants=("eps",),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="fused_add_rms_norm",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output",),
        source=_FUSED_ADD_RMS_NORM_VLLM,
        description="vLLM XPU fused_add_rms_norm (SYCL), residual add fused into the norm.",
        constants=("eps",),
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="fused_add_rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output",),
        source=_FUSED_ADD_RMS_NORM_SGL,
        description="SGLang XPU fused_add_rmsnorm (SYCL), residual add fused into the norm.",
        constants=("eps",),
    ),
)
"""Upstream kernels that can be benchmarked as baselines.

Deliberately small: each entry is a kernel whose calling convention *and semantics* have
been checked against the upstream binding.

Signature matching alone is not enough. ``gemma_rms_norm`` takes exactly the same
arguments as ``rms_norm`` but scales by ``(1 + weight)``, so registering it against a
plain RMSNorm definition produced a baseline that ran fine and computed the wrong thing --
caught by the correctness gate, but only because there was one. A kernel belongs here only
when it computes the same function as the definition's reference, not merely when the
arguments line up. Gemma's variant needs its own definition before it can be a baseline.
"""


def is_provider_available(provider: str) -> bool:
    """Whether an upstream kernel library is importable here."""
    module = _PROVIDER_MODULES.get(provider)
    if module is None:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_providers() -> List[str]:
    """Upstream kernel libraries installed in this environment."""
    return [p for p in sorted(_PROVIDER_MODULES) if is_provider_available(p)]


def find_baselines(
    definition: Definition, providers: Optional[Sequence[str]] = None
) -> List[BaselineKernel]:
    """Upstream kernels that implement ``definition``.

    Parameters
    ----------
    definition : Definition
        The definition to find baselines for.
    providers : Optional[Sequence[str]]
        Restrict to these providers. Defaults to every provider installed here.

    Returns
    -------
    List[BaselineKernel]
        Matching kernels, empty when none apply or none are installed.
    """
    allowed = set(providers) if providers is not None else set(available_providers())
    return [k for k in REGISTRY if k.provider in allowed and k.matches(definition)]


def explain_no_match(
    definition: Definition, providers: Optional[Sequence[str]] = None
) -> List[str]:
    """Why no baseline bound to ``definition``, as one line per near-miss.

    A signature mismatch and an uninstalled provider both produce zero baselines, and
    without this they produce the same message too. Only kernels sharing the op_type are
    reported: anything else is not a near-miss, it is a different operation.
    """
    allowed = set(providers) if providers is not None else set(available_providers())
    if find_baselines(definition, providers):
        return []  # Something bound; there is no absence to explain.
    reasons: List[str] = []
    for kernel in REGISTRY:
        if kernel.provider not in allowed or kernel.op_type != definition.op_type:
            continue
        got_in, got_out = tuple(definition.inputs), tuple(definition.outputs)
        if got_in != kernel.inputs:
            reasons.append(
                f"{kernel.provider}/{kernel.name}: op_type matches, inputs differ "
                f"(kernel wants {kernel.inputs}, definition has {got_in})"
            )
        elif got_out != kernel.outputs:
            reasons.append(
                f"{kernel.provider}/{kernel.name}: op_type and inputs match, outputs "
                f"differ (kernel wants {kernel.outputs}, definition has {got_out})"
            )
        elif kernel.fi_api is not None:
            reasons.append(
                f"{kernel.provider}/{kernel.name}: signature matches but this definition "
                f"is a different operation (kernel implements {kernel.fi_api}; the "
                f"definition's fi_api tag says otherwise)"
            )
    return reasons


def registry_op_types(providers: Optional[Sequence[str]] = None) -> List[str]:
    """op_types the registry can serve, for reporting against a dataset's actual set."""
    allowed = set(providers) if providers is not None else set(available_providers())
    return sorted({k.op_type for k in REGISTRY if k.provider in allowed})


def verify_kernel(
    definition: Definition, kernel: BaselineKernel, workload, device: str = "xpu:0"
) -> Tuple[bool, str]:
    """Actually call ``kernel`` once, and report whether it ran.

    Existing in the provider's Python namespace is not the same as being built for this
    backend. ``sgl-kernel``'s Python wrappers ship with the package regardless of which
    backend was compiled, and its ops are JIT-registered on first use, so neither
    ``dir(sgl_kernel)`` nor ``dir(torch.ops.sgl_kernel)`` describes what is available --
    ``sgl_kernel.rmsnorm`` works while appearing in neither. A call with *valid arguments*
    is the only reliable check; calling with none only exercises the Python signature and
    reports a wrapper that dispatches to nothing as working.

    Registering an unbuilt op costs a ``RUNTIME_ERROR`` on every workload of every matching
    definition, which reads like a wrapper bug rather than a missing kernel.
    """
    from flashinfer_bench.bench.utils import gen_inputs
    from flashinfer_bench.compile import BuilderRegistry

    try:
        solution = make_baseline_solution(definition, kernel)
        runnable = BuilderRegistry.get_instance().build(definition, solution)
        inputs = gen_inputs(definition, workload, device)
        runnable(*inputs)
        from flashinfer_bench.device import device_synchronize

        device_synchronize(device)
        return True, "ran"
    except AttributeError as e:
        if "_OpNamespace" in str(e):
            missing = str(e).split("has no attribute")[-1].strip()
            return False, f"not built for this backend: {missing}"
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def make_baseline_solution(definition: Definition, kernel: BaselineKernel) -> Solution:
    """Wrap an upstream kernel as a Solution for ``definition``.

    Raises
    ------
    ValueError
        If the kernel does not implement this definition.
    """
    if not kernel.matches(definition):
        raise ValueError(
            f"Baseline '{kernel.name}' does not implement definition '{definition.name}' "
            f"(expected inputs {kernel.inputs} and outputs {kernel.outputs})"
        )

    return Solution(
        name=f"{definition.name}__{kernel.solution_name}",
        definition=definition.name,
        author=kernel.provider,
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["xpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=kernel.render(definition))],
        description=(
            f"{kernel.description} Benchmark baseline: a solution must beat this to be "
            f"an improvement on what Intel deployments already run."
        ),
    )


def make_baseline_solutions(
    definition: Definition, providers: Optional[Sequence[str]] = None
) -> List[Solution]:
    """Every available upstream baseline for ``definition``, as Solutions."""
    return [make_baseline_solution(definition, k) for k in find_baselines(definition, providers)]
