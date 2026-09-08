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

Ops that map onto dataset op_types nothing is wired for:

| Op | Maps to |
| --- | --- |
| `mha_fwd` | `gqa_paged`, `gqa_ragged` |
| `flash_mla_decode`, `flash_mla_prefill` | `mla_paged`, `mla_ragged` |
| `flash_mla_sparse_decode`, `flash_mla_sparse_prefill` | `dsa_paged` |
| `gdn_attention` | `gdn` |
| `top_k_top_p_sampling_from_probs`, `top_k_renorm_probs`, `top_p_renorm_probs` | `sampling` |
| `sgemm_lora_a_fwd`, `sgemm_lora_b_fwd` | `gemm` |
| `store_cache`, `transfer_kv_per_layer`, `transfer_kv_all_layer` | KV-cache movement |

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
verify numerically against the definition's reference before registering it. **Not yet
done; open item.**

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

### What is still not wireable

- **`gqa_ragged`: all ten definitions are causal**, and the XPU build refuses
  `return_softmax_lse` with causal masking (`return_softmax_lse is only supported without
  causal/local masking`). The output alone is correct, but the definitions declare `lse` as
  an output, so nothing can bind. Recomputing `lse` in the wrapper means materialising the
  attention matrix, which costs more than the kernel saves.
- **`gqa_paged` prefill (11 defs)**: same limitation -- they are causal and declare `lse`.
- **21 `ps1` definitions**: page size 1 is rejected outright.
- **`fp8_blockwise_scaled_mm`** (sgl): not built on XPU.

### W4A16 is a float16 kernel, and bfloat16 is not merely a dtype swap

`int4_gemm_w4a16` is oneDNN-backed and passes its own vendor test exactly -- **100% of
elements within (1e-2, 1e-2)**, which `torch.testing.assert_close` enforces on every
element, under both uniform and normal inputs.

In **bfloat16 the same kernel matches only 81-92%** of elements at that tolerance. Intel's
test (`tests/test_int4_gemm_onednn.py`) parametrizes `dtype=[torch.float16]` and nothing
else, so bfloat16 is untested upstream, not merely unmeasured here.

The reason is arithmetic: bfloat16 has 7 mantissa bits against float16's 10, and the
dequantised value `(nibble - zero) * scale` therefore loses three bits -- of a weight that
only carried four. Real AWQ and GPTQ checkpoints are float16, so a W4A16 definition should
declare float16 and a bfloat16 one is asking for something the ecosystem does not ship.

Declaring it correctly took the same kernel from failing a 95% gate to **8.5x-27x** against
the definition's reference.

**The general lesson:** when a vendor kernel misses a correctness gate, read the vendor's
own test before touching the gate. It states the conditions the kernel is warranted under
-- dtype above all -- and a definition outside those conditions is the thing that is wrong.

### Contract for `int4_gemm_w4a16`

Established by measurement, since none of it is documented:

- `B_packed [K/8, N]` int32, **column-major** -- the "NT format" its error message demands
  is literally `strides[-2] == 1`, so pass `packed.t().contiguous().t()`
- `B_scale [K/G, N]`, `B_zeros [K/G, N/8]` int32, eight zero-point nibbles packed along N
- dequantisation is `(nibble - zero) * scale`, with **no** offset; the `+1` convention some
  GPTQ exporters use disagrees by ~8.5e-2 relative
- internally it uses oneDNN's grouped-scale API along K -- the one that is silently wrong
  for f8 operands (see `/optimize-onednn`); on the s4 path it is correct

### `gated_delta_rule_non_spec`: resolved, and why `gdn` still does not bind

The kernel is correct and the mapping is fully established -- **matched to 3.8e-3 on output
and 9.5e-4 on state**. It nonetheless cannot serve the dataset's `gdn_decode_*`
definitions, for a reason that is neither side's bug.

**The exact calling convention** (from `csrc/xpu/gdn_attn/gated_delta_rule.hpp:95-180`, not
from the docs):

- Pass **raw `b`**. The kernel applies `act_sigmoid` to it itself (`:143`). Passing
  `sigmoid(b)` double-applies it. The vendor test's variable is named `ref_beta` because
  its *reference* takes a sigmoid'd value, while the *kernel* takes the raw one -- reading
  the test alone gets this backwards.
- Pass **raw `a`**; the kernel computes `exp(-exp(A_log) * softplus(a + dt_bias))`.
- **`dt_bias` must be bfloat16** ("dt_bias dtype must match core_attn_out dtype"), while
  **`A_log` must stay float32**. The definitions declare both float32.
- **Clone `q` and `k` before the call** -- the kernel writes to them.
- Decode: `num_prefills=0`, `num_decodes=B`, `non_spec_query_start_loc=arange(B+1, int32)`,
  `non_spec_state_indices_tensor=arange(B, int32)`. No chunk padding on this path.
- State memory order is `h*(K*V) + v*K + k`, i.e. `[H, V, K]` k-last -- matching the
  definitions.

**Why it does not bind: a fusion-boundary mismatch.** The kernel L2-normalises `q` and `k`
inside its own loop (`:159-170`, `sum += eps=1e-6`, then `q /= sqrt(sum)` and
`q *= 1/sqrt(head_k_dim)`). The dataset's `gdn_decode_*` reference does not normalise at
all. On CUDA the normalisation is a separate op upstream; Intel folded it into this kernel,
so the two draw the operation boundary in different places.

No wrapper can bridge it. Normalisation discards `|q|` and `|k|` entirely, so no transform
of the inputs reproduces the unnormalised result -- and `k` enters the state update
quadratically, so it cannot be compensated afterwards either. Checked whether the recorded
workloads happen to be unit-norm already, which would have made the normalisation
idempotent: they are not (norms range 0.40 to 1.60).

Feeding the definition's reference L2-normalised `q`/`k` reproduces the kernel to 3.8e-3,
which is what proves the mapping is right and the boundary is the only difference.

**To use this kernel, the dataset needs a definition whose boundary includes the
normalisation.** That is a new definition, not a wrapper. Recorded rather than worked
around, because a baseline registered against the current definition would run cleanly and
compute something else -- the exact failure this registry exists to prevent.

Note also that `gdn_decode_qk16_v32_d128_k_last` has **no workloads**; only the `qk4_v8`
and `qk8_v16` variants (54 each) are benchmarkable.

### A signature does not prove the kernel is built

`inspect.signature` reads the Python wrapper, which ships regardless of which backend was
compiled. The XPU build is *partial*: on the same install `top_k_renorm_prob` and
`min_p_sampling_from_probs` work while `top_p_sampling_from_probs` does not.

Enumeration lies too — `dir(sgl_kernel)` gives ~190 names, `dir(torch.ops.sgl_kernel)` gives
4, and neither is the true surface, because ops are JIT-registered on first use.

**And the wrong import makes a built op look missing.** `vllm-xpu-kernels` registers into
two namespaces: `torch.ops._C` (norms, activations, rope) and `torch.ops._xpu_C` (the
sampler, among others). Importing `vllm_xpu_kernels._C` registers only the first, and
`torch.ops._xpu_C.topk_topp_sampler` then raises `AttributeError: '_OpNamespace' '_xpu_C'
object has no attribute ...` — indistinguishable from a kernel that was never compiled.
`import vllm_xpu_kernels._xpu_C` makes the same op work. A wrapper must import the
submodule that owns the op it calls, and a probe that concludes "not built" has to have
imported the right one first. This cost a wrong conclusion here before it was caught.

**Only a call with valid inputs proves it.** Either use the CLI:

```bash
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
```

or call it directly before registering an entry:

```python
import torch, sgl_kernel
probs = torch.rand(8, 4096, device="xpu:0"); probs /= probs.sum(-1, keepdim=True)
top_p = torch.full((8,), 0.9, device="xpu:0")
sgl_kernel.top_p_sampling_from_probs(probs, top_p)   # raises here, not at benchmark time
torch.xpu.synchronize()
```

A registry entry for an op that is not built produces `RUNTIME_ERROR` on every workload of
every matching definition — noisy, and easily mistaken for a wrapper bug.

### Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Build aborts citing memory | OOM guard fired | Reduce build parallelism; the guard is doing its job |
| Build fails selecting a target | `DPCPP_SYCL_TARGET` unset or not `bmg`/`cri` | `providers install sgl-kernel-xpu --target bmg` |
| Imports, then crashes at launch | Built for the wrong architecture | Rebuild for `capabilities().sycl_target` |
| Nothing works on an integrated part | Not a supported target | Expected — use `vllm-xpu` or `--in-tree` |
| `AttributeError` on an op that has a signature | Partial build | Call it to check; treat as not installed |

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
