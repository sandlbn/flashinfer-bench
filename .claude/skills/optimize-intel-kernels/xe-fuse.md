# Xe-Fuse reference notes

Working notes for [IntelLabs/Xe-Fuse](https://github.com/IntelLabs/Xe-Fuse) — GEMM epilogue
fusion for Intel GPUs, built on `sycl-tla` (CUTLASS with SYCL bindings).

Everything here was established by building and running it, not by reading the README.
Where something is inferred rather than measured, it says so.

## Why it matters

GEMM is ~70-85% of decoder device time and runs through oneDNN. You will not beat oneDNN
at matrix multiply. What you *can* remove is the memory traffic around it — and vLLM's MLP
today is three kernels with a full round trip in the middle:

```python
gate_up, _ = self.gate_up_proj(x)   # GEMM writes [M, 2d] to memory
x = self.act_fn(gate_up)            # silu_and_mul reads it back
x, _ = self.down_proj(x)            # GEMM
```

Even vLLM's *fused* MoE path does this — `cutlass_grouped_gemm` materializes `gemm1_output`,
then `fused_moe_activation` calls `silu_and_mul` on it. "Fused" there means fused expert
routing, not a fused epilogue. **vLLM has no GEMM-with-epilogue kernel**, which is the gap
Xe-Fuse fills.

## Building

Needs `sycl-tla`, fetched by the project's CMake. Two definitions switch CUTLASS to its
SYCL backend — without them the compile fails looking for `cuda_runtime_api.h`, an error
that has nothing to do with your kernel:

```bash
icpx -fsycl -O2 -std=c++17 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I<xe-fuse>/include -I<sycl-tla>/include \
  -I<sycl-tla>/tools/util/include -I<sycl-tla>/examples/common \
  -I<sycl-tla>/applications \
  -c kernel.cpp -o kernel.o
```

Linking additionally needs SPIR-V extensions enabled, or it fails with
`RequiresExtension: Feature requires the following SPIR-V extension`:

```bash
-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
```

`SyclBuilder` supplies all of this when a solution declares the `xe-fuse` dependency and
`FIB_XE_FUSE_DIR` is set.

**Cost:** one translation unit, a few minutes, a few GiB. Unlike `sgl-kernel-xpu`, which
instantiates hundreds of CUTLASS TUs and will exhaust a small machine's memory.

## Operand layout — verify, do not read

The generated kernel builds its B stride with `make_cute_packed_stride(StrideB{},
make_shape(N, K, L))`, which reads as though B were `[N, K]`. **It is not.** B is `[K, N]`
row-major.

Established by running the kernel and testing candidate references against its output:

| candidate | max abs err |
| --- | --- |
| `swiglu(a @ b.T)` | 0.8703 |
| **`swiglu(a @ b.reshape(K, N))`** | **0.0020** |
| `silu-split(a @ b.T)` | 0.9671 |

Do this for any new preset rather than reasoning from the stride construction.

## SwiGLU is interleaved, not split-half

Xe-Fuse's pairwise ops take **interleaved** `(gate, up)` pairs — gate on even lanes, up on
odd. vLLM's `silu_and_mul` is **split-half**: `silu(x[..., :d]) * x[..., d:]`.

Same maths, incompatible layouts. A reference written for one is wrong for the other, and
the kernel will run happily and produce garbage. This is the single easiest way to produce
a confidently wrong result here.

## The output is duplicated, and it cannot compact

`XePairwiseCompute` is an EVT *visitor*: it reads a fragment, shuffles across the
sub-group (`shfl_xor_sync(0xFFFFFFFF, bits, 1, 16)`), and returns a fragment of the **same
width**. Both lanes of a pair get the same value.

The source comment says *"only even lanes should store"* — that is advice to the caller,
not something the visitor does. CUTLASS's epilogue stores every lane, so the result is
`[M, 2d]` with each answer written twice. Taking `out[..., 0::2]` is the consumer's job.

### What that costs when chaining

For M=2048, d=4864, bf16 — `[M, 2d]` is 39.8 MB, `[M, d]` is 19.9 MB:

| path | traffic |
| --- | --- |
| unfused (write 2d, read 2d, write d) | 99.5 MB |
| **fused `k2` alone** | **39.8 MB** |
| fused `k2` + compacting copy for the next GEMM | 99.5 MB |

So `k2` standalone is a 2.5x traffic reduction, but **a compacting copy gives the entire
gain back**. Chaining `k2` into `k0a` (down projection) therefore needs one of:

1. a strided A operand on the second GEMM, reading every other column — likely blocked,
   since CuTe row-major `StrideA` carries a compile-time `_1` on the K dimension
   (*inferred from the layout tag, not tested*);
2. a compacting store in the visitor — an upstream Xe-Fuse change;
3. accepting the copy, which erases the benefit.

**Resolve this before wiring a second fused kernel into an MLP.** A two-kernel fused MLP
that reintroduces a round trip in the middle is no better than what vLLM already does.

## Presets

```bash
python autotune/generate_kernel.py --preset k2 --m M --n N --k K -o kernel.cpp
```

| Preset | Fusion | Maps to |
| --- | --- | --- |
| `k1`, `k1v2` | `D = acc * R[m]` | GEMM + RMSNorm row scale |
| `k2`, `k2v2` | `D = SwiGLU(acc * R[m])` | gate/up + norm + SwiGLU |
| `k2_geglu`, `k2v2_geglu` | `D = GeGLU(acc * R[m])` | Gemma-style gated FFN |
| `k0a` | `D = gamma[n] * (acc + residual)` | down projection + residual + next norm's gamma |
| `k3`, `k4`, `k4v2` | `D = RoPE(acc * R[m], cos_sin)` | qkv + RoPE |
| `w8a8_dequant` | `int32_acc * scale_token[m] * scale_channel[n]` | quantized GEMM |

`v2` variants use a merged (flat) visitor instead of a composed tree — same maths,
different codegen. Benchmark both.

## The EVT is the thing to optimize

```cpp
using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
using KernelConfig = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
```

Vocabulary:

- **Sources** — `Acc`, `AuxLoad<E>`, `ColBroadcast` (per-row `scale[m]`), `RowBroadcast`
  (per-column `gamma[n]`)
- **Binary** — `Mul`, `Add`, `ScaleRows`, `ScaleCols`, `AddResidual`, `BiasAdd`
- **Activations** — `GeLU`, `GeLUTanh`, `SiLU`, `ReLU`, `Sigmoid`
- **Pairwise** — `SwiGLU`, `GeGLU`, `RoPE`, `PairwiseSwap`

Two cheap search axes: the tile shape (`--tile 128x256x32`), and the tree itself.

## Adapting a model's weights — transform, don't branch

The kernel never changes per model. The *weights* transform once, at load. RMSNorm splits
into two pieces that go to different places:

- **per-channel `gamma`** scales input channels, so it cannot travel through the GEMM as a
  row scale — it multiplies into the weight;
- **per-token `rsqrt(mean(x²)+eps)`** *does* commute, since `(r·x) @ W == r·(x @ W)`, so it
  stays a runtime per-row scale.

`flashinfer_bench.integration.weight_layout` implements this:

```python
B, eps = qwen_style_mlp_weights(layer.mlp, layer.post_attention_layernorm)
kernel(x, B, rms_row_scale(x, eps), out)
result = deinterleave_output(out)
```

Verified against Qwen2.5-0.5B's real layer-0 weights: relative error 0.008 at M=512 and
M=4096, which is bf16 through an 896-deep dot product plus SiLU.

Branch on the **fusion** (SwiGLU vs GeGLU vs residual+norm), never on the model layout —
the latter costs models × fusions, and each branch needs its own verified reference.

## Measured

Intel Xe3.0 integrated GPU, K=896, N=9728 (Qwen2.5-0.5B MLP), performance power profile:

| M | fused | eager unfused | ratio |
| --- | --- | --- | --- |
| 512 | 3.03 ms | 11.97 ms | 3.95x |
| 2048 | 8.66 ms | 44.01 ms | 5.08x |
| 4096 | 15.04 ms | 87.04 ms | 5.79x |
| 8192 | 27.10 ms | 173.37 ms | 6.40x |

Monotonic in M, as expected for a tiled GEMM.

**Those numbers are against PyTorch eager and are misleading.** Measured against vLLM's
actual path — oneDNN GEMM plus their own SYCL `silu_and_mul` — the fused kernel *loses*.

## The honest comparison: fusion currently loses

Full MLP block, Qwen2.5-0.5B layer 0, same hardware. `vllm_3k` is
`norm -> gate_up GEMM -> torch.ops._C.silu_and_mul -> down GEMM`; `fused_2k` is
`k2 -> strided down GEMM`:

| M | vLLM 3-kernel | fused 2-kernel | speedup | rel err |
| --- | --- | --- | --- | --- |
| 512 | 2.97 ms | 6.95 ms | **0.43x** | 0.006 |
| 2048 | 11.20 ms | 13.87 ms | **0.81x** | 0.005 |
| 4096 | 19.92 ms | 23.63 ms | **0.84x** | 0.005 |

Correct, and slower. Four plausible reasons, in rough order of suspicion:

1. **The duplicated store.** `k2` writes `[M, 2d]` where vLLM's activation output is
   `[M, d]` — twice the write traffic, on the one axis fusion was supposed to win. This
   alone may account for the result.
2. **Tile shape.** The generated `256x256x32` is presumably tuned for discrete Arc, not a
   16-EU integrated part. Untested — `--tile` is a cheap search.
3. **oneDNN is very good.** Beating a vendor-tuned GEMM on its own hardware is the whole
   difficulty; a generic CUTLASS-SYCL instantiation need not match it.
4. **Strided down GEMM** costs ~1.2 ms at M=2048 versus contiguous (still cheaper than
   compacting, which costs ~1.3 ms — but it is not free).

**Do not present epilogue fusion as a win on this hardware until it is one.** The theory —
less memory traffic — is sound, and the arithmetic in the section above still holds. The
implementation, as generated and on this device, does not cash it in.

### Resolved: it is the GEMM, not the store

unitrace settled it. Per MLP at M=2048:

| | vLLM path | Xe-Fuse path |
| --- | --- | --- |
| GEMM(s) | `gemm_kernel` 3.95 ms x2 = **7.90 ms** | CUTLASS **7.67 ms** + `gemm_kernel` 2.98 ms |
| activation | `act_and_mul_vec_kernel` 1.17 ms | *fused in* |
| **total** | **12.76 ms** | **14.51 ms** |

oneDNN does *both* projections in the time CUTLASS takes for one. Splitting by FLOPs
(gate/up is ~2x down), oneDNN's gate/up is ~5.3 ms against CUTLASS's 7.67 ms —
**CUTLASS-SYCL is ~45% slower than oneDNN on the same GEMM.** The fusion works exactly as
designed, removing the 1.17 ms activation kernel; it just pays ~2.4 ms more on the matmul
to do it. The duplicated store never appears in the top costs.

Tile search confirms tuning cannot rescue it — best of seven shapes was 128x256x16 at
0.83x (k2 9.65 -> 8.16 ms), still well above oneDNN's ~5.3 ms.

### The approach that does win: oneDNN post-ops

Keep oneDNN's matmul and fuse with oneDNN's own post-op mechanism. Post-ops are
elementwise or binary on the GEMM output and cannot express SwiGLU's pairwise reduction,
but splitting into two matmuls makes it expressible at identical total FLOPs:

```
up  = (x @ Wu) * r[m]                              # binary_mul post-op
out = swish((x @ Wg) * r[m]) * up                  # binary_mul, eltwise_swish, binary_mul
```

Measured against vLLM's real path, same hardware, full MLP block:

| M | vLLM 3-kernel | oneDNN 2-kernel | speedup |
| --- | --- | --- | --- |
| 512 | 2.54 ms | 2.82 ms | 0.90x |
| 2048 | 11.54 ms | 9.92 ms | **1.16x** |
| 4096 | 19.56 ms | 17.88 ms | **1.09x** |

Two further advantages: gate and up stay as separate matrices, so **no weight re-layout is
needed at all** — none of the interleaving trap above applies; and torch does not expose
this (only `qlinear_pointwise` / `linear_dynamic_fp16`), so it must be called from C++ via
`dnnl::sycl_interop::make_engine/make_stream` on the framework's queue.

See `examples/sycl/onednn_gemm_swiglu.cpp`. Declare `onednn` in the solution's
dependencies; `SyclBuilder` supplies the include, lib and rpath.

**The 0.90x at M=512 is fixed overhead**: engine and stream are cached but the matmul
primitives and their descriptors are rebuilt per call. Caching those should make the win
uniform, and is the obvious next change.

What might still change the Xe-Fuse verdict: re-measuring on Battlemage, which is what it
actually targets. On this integrated Xe3.0 part, oneDNN post-ops are the better route.

`torch.compile` could not be measured as a third baseline — Inductor rejects this device
(`device architecture not recognized: 32224837632`, Xe3.0 / IP 30).

## Status

- Xe-Fuse is marked **not stable / not production-ready** by IntelLabs. Treat it as one
  solution source, never a dependency.
- `k2` is wired end to end: generated, built via `SyclBuilder`, correctness-gated, and
  verified against a real model's weights.
- The full MLP chains without a compacting copy — oneDNN reads the stride-2 view natively.
- **It is slower than vLLM's unfused path** (0.43x-0.84x). Correct, but not yet a win.
- `k0a` is not done. It is not the bottleneck; the duplicated store is.
- Nothing has been measured inside a serving stack. There is no tokens/sec number.
