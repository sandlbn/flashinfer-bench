# Intel kernel providers: inventories and signatures

What each provider contains, how to read a kernel's real signature, and how to prove it is
actually built. **Installation, costs and GitHub locations are in `/setup-intel-env`** — do
not duplicate them here.

Wiring a provider kernel in as a baseline is `SKILL.md` Phase 5. This file is the reference
you consult while doing it.

## What is wired up right now

Never trust a hand-written list of registered baselines — it rots. Ask the registry:

```bash
python -c "
from flashinfer_bench.integration.xpu_kernels import REGISTRY
for k in sorted(REGISTRY, key=lambda k: (k.provider, k.op_type, k.name)):
    print(f'{k.provider:16} {k.op_type:12} {k.name}')
"
```

Everything in the inventories below that the registry does not list is unwired — and the
unwired part is most of the value: attention, MLA, GEMM, quantization and KV-cache ops.

## vllm-xpu-kernels

Registered as torch custom ops under `torch.ops._C`.

| Area | Kernels | op_type |
|---|---|---|
| Norms | `rms_norm`, `fused_add_rms_norm`, `gemma_rms_norm` | `rmsnorm` |
| RoPE | `rotary_embedding`, `fused_qk_norm_rope` | `rope` |
| Activations | `silu_and_mul`, `mul_and_silu`, `gelu_and_mul`, `gelu_tanh_and_mul`, `gelu_new`, `gelu_fast`, `gelu_quick`, `fatrelu_and_mul` | `activation` |
| Quantization | fp8 / mxfp4 quant family, `awq_dequantize` | `gemm`, low-bit evaluators |
| KV cache | `reshape_and_cache`, `concat_and_cache_mla`, `gather_cache` | `gqa_paged`, `mla_paged` |
| Attention glue | `merge_attn_states`, `topk_per_row` | `gqa_paged`, `sampling` |

### Inventory and signatures

**49 ops** are registered. Enumerate them from source:

```bash
grep -ohE '"[a-z_0-9]+\(' tmp/vllm-xpu-kernels/csrc/torch_bindings.cpp \
  | sed 's/"//;s/(//' | sort -u
```

The schema is also introspectable at runtime — prefer it, and never infer from a header:

```python
import torch, vllm_xpu_kernels._C  # noqa: F401  -- registers torch.ops._C
print(torch.ops._C.rms_norm.default._schema)
# _C::rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()
```

A leading `Tensor!` means destination-passing and in-place: allocate the output in the
wrapper, and clone any input the op mutates so repeated benchmark trials see identical data.

### `gemma_rms_norm` is deliberately not registered

It takes byte-identical arguments to `rms_norm` and scales by `(1 + weight)`. Registered
against a plain RMSNorm definition it produced a baseline that ran cleanly and computed the
wrong thing. **Matching arity is not matching semantics.** Gemma's variant needs its own
definition first. The module docstring of `flashinfer_bench/integration/xpu_kernels.py`
carries the full rule.

### Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `torch.ops._C` has no such op | `_C` extension never imported | `import vllm_xpu_kernels._C` first; every registry wrapper does |
| Baseline runs but fails correctness | Semantics differ from the definition | The registry entry is wrong, not the kernel — give the variant its own definition |
| Baseline never appears | Signature mismatch | Silence is the expected symptom; `SKILL.md` Phase 5 |

## sgl-kernel-xpu

An ordinary Python package (`sgl_kernel`). Builds per architecture: `bmg` and `cri` only.

| Area | Kernels | op_type |
|---|---|---|
| Attention | FMHA prefill / decode | `gqa_paged`, `gqa_ragged` |
| MLA | decode, prefill, sparse decode / prefill | `mla_paged`, `dsa_paged` |
| GEMM | GroupGemm, W4A16, W8A16 | `gemm` |
| LoRA | SGEMM LoRA A/B forward, QKV LoRA | `gemm` |
| Linear attention | `GdnAttn` | `gdn` |
| Elementwise | rope, norms, activations | `rmsnorm`, `rope` |

The table understates it — **108 ops** are registered. Enumerate them from the one file
that registers them (note `.cc`, not `.cpp` — a `--include=*.cpp` filter finds almost
nothing):

```bash
grep -ohE '"[a-z_0-9]+\(' tmp/sgl-kernel-xpu/src/torch_extension_sycl.cc \
  | sed 's/"//;s/(//' | sort -u
```

To see which of those already have a registry entry — and therefore which are still worth
wiring — ask the registry rather than reading a list that ages:

```bash
python -c "
from flashinfer_bench.integration.xpu_kernels import REGISTRY
import collections
for op, n in sorted(collections.Counter(k.op_type for k in REGISTRY).items()):
    print(f'{op:12} {n}')
"
```

Families with no entry at that moment are the backlog. KV-cache movement
(`store_cache`, `transfer_kv_per_layer`, `transfer_kv_all_layer`) has no dataset op_type at
all, so it needs a definition before a kernel can bind to anything.

### `mha_fwd` covers the paged KV cache

It is a unified attention kernel that reads the cache directly:

```cpp
const at::Tensor& k,                            // (num_pages, page_size, h_k, d) when paged
std::optional<const at::Tensor>& page_table,    // (b_k, max_num_pages_per_seq)
at::Tensor& out, std::optional<at::Tensor>& softmax_lse
```

which is the `gqa_paged` signature the dataset models as
`(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale) -> (output, lse)`.

Do not conflate this with the standalone cache *write* (`reshape_and_cache`), which has no
definition — conflating them makes the paged path look unreachable when it is not.

The wrapper's real work is translating paging representations: FlashInfer-style
`kv_indptr`/`kv_indices` against `mha_fwd`'s `page_table` matrix plus `cu_seqlens_k`. That is
exactly the kind of impedance mismatch that runs cleanly and computes the wrong thing —
verify numerically against the definition's reference before registering it.

### What is actually built, measured on Arc B580

Probed by calling each with valid inputs on `sgl-kernel-xpu` 0.11.0. The Python surface is
shared with the CUDA build, so this is the only way to know.

| Op | On XPU |
| --- | --- |
| `flash_attn_varlen_func` (FMHA prefill) | **works** |
| `flash_attn_with_kvcache` (paged decode) | **works, page_size 64 or 128 only** |
| `silu_and_mul` | works |
| `topk_softmax` | works |
| `awq_dequantize` (INT4 weight dequant) | works |
| `causal_conv1d_fn_xpu` | works |
| `fp8_blockwise_scaled_mm` | **not built** -- no op registered |
| `top_k_top_p_sampling_from_probs` | **not built** (`top_p_sampling_from_probs` missing) |

The page-size constraint matters for wiring `gqa_paged`: 16, 32 and 256 are all rejected
with `Unsupported page size for decode attention`, and the dataset carries `ps1` and `ps64`
definitions -- only the `ps64` ones can bind.

### Paged GQA decode is wired, and it is worth a lot

`flash_attn_with_kvcache` is registered against the nine `gqa_paged_decode_*_ps64`
definitions. Measured against the definitions' own reference on real dataset workloads:
**8.6x to 45.6x**.

Three things the wrapper has to get right, none of which fail loudly:

- **Paging representation.** The definition carries FlashInfer's ragged form -- `kv_indices`
  with `kv_indptr` row offsets -- and the kernel wants a dense `[batch, max_pages]` page
  table plus a token count per sequence. The count is
  `(pages - 1) * page_size + kv_last_page_len`; using `pages * page_size` attends to
  uninitialised tail entries.
- **`lse` is natural log, the definitions are log2.** Passing it through unconverted is a
  silent 1.44x error on an output nothing else checks. Dividing by `ln(2)` matches the
  reference to 9.5e-07.
- **`lse` comes back transposed**, `[heads, tokens]` against the declared
  `[tokens, heads]`.

Causal alignment needs no adjustment: the kernel's `causal=True` is bottom-right aligned,
which is what the reference's `delta = kv_len - q_len` computes. Verified to 3.3e-03.

### The causal x lse limitation, and how it was worked around

The XPU build refuses `return_softmax_lse` together with causal masking
(`return_softmax_lse is only supported without causal/local masking`). The attention output
alone is correct; only `lse` is unavailable. Since most ragged/prefill definitions are causal
*and* declare `lse` as an output, strict signature matching left them unbindable.

The resolution is a `_no_lse` definition variant: the same operation declaring only
`output`. Those bind, and the registry now covers `gqa_ragged`, `gqa_paged`, `mla_ragged`,
`mla_paged`, `dsa_paged` and `gdn`. Definitions that still declare `lse` remain unbound by
design -- recomputing it in the wrapper means materialising the attention matrix, which
costs more than the kernel saves.

Check the current split rather than trusting a count here:

```bash
ls tmp/flashinfer-trace/definitions/<op_type>/ | wc -l          # total
ls tmp/flashinfer-trace/definitions/<op_type>/*_no_lse*.json | wc -l   # output-only
```

Still genuinely unavailable:

- **Page size 1** is rejected outright by the paged kernels.
- **`fp8_blockwise_scaled_mm`** (sgl) is not built on XPU.

This limitation is upstream, not ours: it also breaks SGLang's own DeepSeek
`mha_return_lse` path on Intel.

## Which kernels a model actually uses is a measurement, not a list

vLLM ships hundreds of Triton kernels (447 distinct `@triton.jit` functions across 218
files). Any one model touches a handful, and on Intel the handful is not the one you would
guess. Observe it rather than enumerating the library:

```bash
python scripts/observe_triton_kernels.py --model <repo_id> --in-process --json out.json
```

It hooks Triton's JIT launch path, runs a short generation, and reports every kernel that
fired with its call count and the constexpr values it was specialised on -- which are what a
tuning round would vary.

Measured on Qwen3-0.6B, Arc B580, 2026-09-08: **20 distinct Triton kernels, 305 launches, and
every one of them in `v1.worker.gpu.*`** -- scheduling, block tables, input batching,
sampling. None in attention, GEMM, norms or activations. On Intel those go through compiled
`vllm-xpu-kernels` and oneDNN via torch, so Triton is left doing orchestration. Of the
op_types the dataset defines, only `sampling` overlapped what the model launched.

The consequence for effort: mapping vLLM's Triton kernels wholesale maps mostly code that
never touches the compute path. Let the observation decide which ones get a baseline, and
re-run it per model -- the answer is a property of the model and the backend selection, not
of the library.

## oneDNN

Ships with oneAPI; discovered via `FIB_ONEDNN_DIR`. Not a baseline *provider* — it is the
library your own SYCL solutions link against, and the thing `F.linear` already calls on XPU.

Diagnosing and fixing the oneDNN GEMM call is `/optimize-onednn`.

For definitions with no upstream baseline, generate one from the in-tree templates instead:

```bash
flashinfer-bench add-baselines --local tmp/flashinfer-trace --in-tree --definitions <name>
```

Templates live in `flashinfer_bench/integration/intree_kernels.py`, one per language.

## Xe-Fuse and sycl-tla

GEMM epilogue fusion on CUTLASS-SYCL. See `../optimize-intel-kernels/xe-fuse.md` for when it
is worth using, the layout traps, and the build flags.

## Hand-written SYCL

`../optimize-intel-kernels/SKILL.md` Step 4a owns this. Worked examples in `examples/sycl/`.
