# Intel kernel providers: inventories and signatures

What each provider contains, how to read a kernel's real signature, and how to prove it is
built. Installation is `/setup-intel-env`. Wiring a provider kernel in as a baseline is
`references/wiring-baselines.md`.

## What is wired up right now

Ask the registry; never trust a written list:

```bash
python -c "
from flashinfer_bench.integration.xpu_kernels import REGISTRY
for k in sorted(REGISTRY, key=lambda k: (k.provider, k.op_type, k.name)):
    print(f'{k.provider:16} {k.op_type:12} {k.name}')
"
```

Everything in the inventories below that the registry does not list is unwired.

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

Enumerate the registered ops from source:

```bash
grep -ohE '"[a-z_0-9]+\(' tmp/vllm-xpu-kernels/csrc/torch_bindings.cpp \
  | sed 's/"//;s/(//' | sort -u
```

Read the schema at runtime rather than inferring it from a header:

```python
import torch, vllm_xpu_kernels._C  # noqa: F401  -- registers torch.ops._C
print(torch.ops._C.rms_norm.default._schema)
# _C::rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()
```

A leading `Tensor!` means destination-passing and in-place: allocate the output in the
wrapper, and clone any input the op mutates so repeated benchmark trials see identical data.

### `gemma_rms_norm` is deliberately not registered

It takes byte-identical arguments to `rms_norm` and scales by `(1 + weight)`. Matching
arity is not matching semantics; Gemma's variant needs its own definition first.

| Symptom | Fix |
|---|---|
| `torch.ops._C` has no such op | `import vllm_xpu_kernels._C` first |
| Baseline runs but fails correctness | The registry entry is wrong, not the kernel — give the variant its own definition |
| Baseline never appears | Signature mismatch — `references/wiring-baselines.md` |

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

Enumerate the registered ops from the one file that registers them (`.cc`, not `.cpp`):

```bash
grep -ohE '"[a-z_0-9]+\(' tmp/sgl-kernel-xpu/src/torch_extension_sycl.cc \
  | sed 's/"//;s/(//' | sort -u
```

Registry coverage by op_type, to see what is still unwired:

```bash
python -c "
from flashinfer_bench.integration.xpu_kernels import REGISTRY
import collections
for op, n in sorted(collections.Counter(k.op_type for k in REGISTRY).items()):
    print(f'{op:12} {n}')
"
```

KV-cache movement (`store_cache`, `transfer_kv_per_layer`, `transfer_kv_all_layer`) has no
dataset op_type, so it needs a definition before a kernel can bind.

### `mha_fwd` covers the paged KV cache

It reads the cache directly:

```cpp
const at::Tensor& k,                            // (num_pages, page_size, h_k, d) when paged
std::optional<const at::Tensor>& page_table,    // (b_k, max_num_pages_per_seq)
at::Tensor& out, std::optional<at::Tensor>& softmax_lse
```

which is the `gqa_paged` signature the dataset models as
`(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale) -> (output, lse)`. The standalone
cache *write* (`reshape_and_cache`) is a different op with no definition.

The wrapper's work is translating paging representations: FlashInfer-style
`kv_indptr`/`kv_indices` into `mha_fwd`'s `page_table` matrix plus `cu_seqlens_k`. Verify
numerically against the definition's reference before registering it.

### What is built on XPU

The Python surface is shared with the CUDA build, so the only proof is a call:
`flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0`. Known
constraints of the XPU build:

| Op | On XPU |
| --- | --- |
| `flash_attn_varlen_func` (FMHA prefill) | works |
| `flash_attn_with_kvcache` (paged decode) | works, **page_size 64 or 128 only** — `ps1` definitions cannot bind |
| `silu_and_mul`, `topk_softmax`, `awq_dequantize`, `causal_conv1d_fn_xpu` | work |
| `fp8_blockwise_scaled_mm` | not built |
| `top_k_top_p_sampling_from_probs` | not built (`top_p_sampling_from_probs` missing) |

### Paged GQA decode wrapper contract

`flash_attn_with_kvcache` is registered against the `gqa_paged_decode_*_ps64` definitions.
Three things the wrapper must get right, none of which fail loudly:

- **Paging representation.** The definition carries FlashInfer's ragged form (`kv_indices`
  with `kv_indptr` row offsets); the kernel wants a dense `[batch, max_pages]` page table
  plus a token count per sequence. The count is `(pages - 1) * page_size + kv_last_page_len`;
  `pages * page_size` attends to uninitialised tail entries.
- **`lse` is natural log; the definitions are log2.** Divide by `ln(2)`.
- **`lse` comes back transposed**, `[heads, tokens]` against the declared `[tokens, heads]`.

The kernel's `causal=True` is bottom-right aligned, which matches the reference's
`delta = kv_len - q_len`; no adjustment needed.

### The causal × lse limitation

The XPU build refuses `return_softmax_lse` together with causal masking. The attention
output is correct; only `lse` is unavailable. Definitions that are causal *and* declare
`lse` therefore cannot bind under strict signature matching.

Convention: a `_no_lse` definition variant — the same operation declaring only `output`.
Those bind, across `gqa_ragged`, `gqa_paged`, `mla_ragged`, `mla_paged`, `dsa_paged` and
`gdn`. Definitions that still declare `lse` stay unbound by design; recomputing `lse` in the
wrapper means materialising the attention matrix. Check the current split:

```bash
ls tmp/flashinfer-trace/definitions/<op_type>/ | wc -l                    # total
ls tmp/flashinfer-trace/definitions/<op_type>/*_no_lse*.json | wc -l      # output-only
```

The limitation is upstream; it also affects SGLang's own `mha_return_lse` path on Intel.

## Which Triton kernels a model actually uses is a measurement, not a list

vLLM ships many Triton kernels; a model touches a handful, and on Intel the compute path
(attention, GEMM, norms, activations) goes through compiled `vllm-xpu-kernels` and oneDNN,
leaving Triton doing orchestration (`v1.worker.gpu.*`). Observe rather than enumerate, and
re-run per model:

```bash
python scripts/observe_triton_kernels.py --model <repo_id> --in-process --json out.json
```

It hooks Triton's JIT launch path, runs a short generation, and reports every kernel that
fired with its call count and the constexpr values it was specialised on. Let the
observation decide which ones get a baseline.

## oneDNN

Ships with oneAPI; discovered via `FIB_ONEDNN_DIR`. Not a baseline provider — it is the
library your own SYCL solutions link against, and what `F.linear` already calls on XPU.
Diagnosing the call is `/optimize-onednn`.

For definitions with no upstream baseline, generate one from the in-tree templates:

```bash
flashinfer-bench add-baselines --local tmp/flashinfer-trace --in-tree --definitions <name>
```

Templates live in `flashinfer_bench/integration/intree_kernels.py`, one per language.

## Xe-Fuse and sycl-tla

GEMM epilogue fusion on CUTLASS-SYCL — `../optimize-intel-kernels/xe-fuse.md`.

## Hand-written SYCL

Writing one is `../optimize-intel-kernels/SKILL.md`, under "Write a SYCL solution".
Worked examples in `examples/sycl/`.
