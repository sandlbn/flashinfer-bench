# Xe-Fuse

Fusing a memory-bound epilogue into a GEMM on CUTLASS-SYCL, so the elementwise work happens
on data still in registers.

[Xe-Fuse](https://github.com/IntelLabs/Xe-Fuse) is marked **not stable** by IntelLabs. Treat
it as one solution source among several — if it regresses you lose a contender, not the
pipeline.

## When to use it

Try oneDNN post-ops first (`/optimize-onednn`). They fuse an epilogue into
oneDNN's own tuned GEMM with no weight re-layout and no interleaving trap.

Reach for Xe-Fuse only when **both** hold:

1. The fusion cannot be expressed as oneDNN post-ops. Post-ops are elementwise or binary on
   a *single* GEMM's output, so anything needing a lane shuffle — SwiGLU, GeGLU, RoPE on
   packed qkv — is out of reach.
2. You have an AOT `sycl_target` and are willing to search tile shapes.

Budget for a GEMM deficit against oneDNN on Battlemage even after tile and register-mode
tuning; the fusion has to save more than that deficit to be worth shipping.

## Setup

Both checkouts come from `/clone-repos`; the builder needs their paths:

```bash
export FIB_XE_FUSE_DIR=$PWD/tmp/Xe-Fuse
export FIB_SYCL_TLA_DIR=$PWD/tmp/sycl-tla

# Confirm the generator and its presets are present before generating anything:
python tmp/Xe-Fuse/autotune/generate_kernel.py --help | head -20
```

`SyclBuilder` supplies every compile and link flag when a solution declares the `xe-fuse`
dependency and those variables are set — see
`flashinfer_bench/compile/builders/sycl_builder.py`. Build by hand only when debugging
outside the builder:

```bash
icpx -fsycl -O2 -std=c++17 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I$FIB_XE_FUSE_DIR/include -I$FIB_SYCL_TLA_DIR/include \
  -I$FIB_SYCL_TLA_DIR/tools/util/include -I$FIB_SYCL_TLA_DIR/examples/common \
  -I$FIB_SYCL_TLA_DIR/applications \
  -c kernel.cpp -o kernel.o
```

Without the two `-D` flags CUTLASS defaults to its CUDA backend and the compile fails
looking for `cuda_runtime_api.h` — an error with nothing to do with your kernel. Linking
additionally needs SPIR-V extensions or it fails with `RequiresExtension`:

```
-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
```

One translation unit, a few minutes, a few GiB — unlike `sgl-kernel-xpu`, which instantiates
hundreds of CUTLASS TUs and will exhaust a small host.

## Generate, wrap, validate

```bash
# 1. Generate
python tmp/Xe-Fuse/autotune/generate_kernel.py --preset k2 --m M --n N --k K -o kernel.cpp

# 2. Wrap: replace the generated standalone main() with a TVM-FFI entry point running on
#    the framework's queue. Template: examples/sycl/xefuse_gemm_swiglu.cpp

# 3. Write the reference to match the kernel's real layout (see below), then:
flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>

# 4. Check register spill BEFORE benchmarking — optimize-intel-kernels SKILL.md, Step 5
unitrace -d -v -o prof python <script>.py   # end with current_stream().synchronize()

# 5. Benchmark
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

## Presets

| Preset | Fusion | Maps to |
| --- | --- | --- |
| `k1`, `k1v2` | `D = acc * R[m]` | GEMM + RMSNorm row scale |
| `k2`, `k2v2` | `D = SwiGLU(acc * R[m])` | gate/up + norm + SwiGLU |
| `k2_geglu`, `k2v2_geglu` | `D = GeGLU(acc * R[m])` | Gemma-style gated FFN |
| `k0a` | `D = gamma[n] * (acc + residual)` | down projection + residual + next norm's gamma |
| `k3`, `k4`, `k4v2` | `D = RoPE(acc * R[m], cos_sin)` | qkv + RoPE |
| `w8a8_dequant`, `w8a8_dequant_biased` | `int32_acc * scale_token[m] * scale_channel[n]` (+ bias) | quantized GEMM |

`v2` variants use a merged (flat) visitor instead of a composed tree — same maths, different
codegen. Benchmark both rather than assuming.

## The EVT is the thing to optimize

```cpp
using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
using KernelConfig = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
```

- **Sources** — `Acc`, `AuxLoad<E>`, `ColBroadcast` (per-row `scale[m]`), `RowBroadcast`
  (per-column `gamma[n]`)
- **Binary** — `Mul`, `Add`, `ScaleRows`, `ScaleCols`, `AddResidual`, `BiasAdd`
- **Activations** — `GeLU`, `GeLUTanh`, `SiLU`, `ReLU`, `Sigmoid`
- **Pairwise (lane shuffle)** — `SwiGLU`, `GeGLU`, `RoPE`, `PairwiseSwap`

Two cheap search axes: the tile shape (`--tile 128x256x32`) and the tree itself — composing
a fusion no preset covers.

## Three layout rules that produce silent garbage

Each of these runs cleanly and computes the wrong thing. None raises.

### B is `[K, N]` row-major, whatever the stride construction reads like

The generated kernel builds its B stride with
`make_cute_packed_stride(StrideB{}, make_shape(N, K, L))`, which reads as though B were
`[N, K]`. It is not.

**Verify any new preset by testing candidate references against the kernel's actual output**
rather than reasoning from the stride. The right candidate is unmistakable — bf16 noise
versus complete disagreement:

| candidate | max abs err |
| --- | --- |
| `swiglu(a @ b.T)` | 0.87 |
| **`swiglu(a @ b.reshape(K, N))`** | **0.002** |
| `silu-split(a @ b.T)` | 0.97 |

Anything at or above ~0.1 is the wrong layout, not a tolerance problem.

### Pairwise ops are interleaved, not split-half

Xe-Fuse's pairwise ops take **interleaved** `(gate, up)` pairs — gate on even lanes, up on
odd. vLLM's `silu_and_mul` is **split-half**: `silu(x[..., :d]) * x[..., d:]`.

Same maths, incompatible layout. A reference written for one is wrong for the other, and the
kernel produces garbage without complaint.

### The output is duplicated and the visitor cannot compact it

`XePairwiseCompute` is an EVT *visitor*: it reads a fragment, shuffles across the sub-group,
and returns a fragment of the **same width**, so both lanes of a pair get the same value.
The source comment saying "only even lanes should store" is advice to the caller — CUTLASS's
epilogue stores every lane. The result is `[M, 2d]` with each answer written twice, and
taking `out[..., 0::2]` is the consumer's job.

**This blocks kernel chaining.** A compacting copy costs exactly what the fusion saved, so
chaining `k2` into `k0a` needs one of: a strided A operand on the second GEMM (likely
blocked — CuTe row-major `StrideA` carries a compile-time `_1` on K), a compacting store in
the visitor (an upstream change), or accepting the copy and the loss. **Resolve this before
wiring a second fused kernel into an MLP** — a fused MLP that reintroduces a round trip in
the middle is no better than the unfused path.

## Adapting a model's weights: transform, don't branch

The kernel never changes per model; the weights transform once at load. An RMSNorm splits
into two pieces that go to different places:

- **per-channel `gamma`** scales input channels, so it cannot travel through the GEMM as a
  row scale — it multiplies into the weight;
- **per-token `rsqrt(mean(x²)+eps)`** does commute, since `(r·x) @ W == r·(x @ W)`, so it
  stays a runtime per-row scale.

`flashinfer_bench/integration/weight_layout.py` implements this:

```python
B, eps = qwen_style_mlp_weights(layer.mlp, layer.post_attention_layernorm)
kernel(x, B, rms_row_scale(x, eps), out)
result = deinterleave_output(out)
```

Wired correctly, expect relative error around 0.008 against an fp32 reference — that is
bf16 through a deep dot product plus SiLU, not a bug. An order of magnitude more means a
layout error, not a tolerance one.

Branch on the **fusion** (SwiGLU vs GeGLU vs residual+norm), never on the model layout: the
latter costs models × fusions, and every branch needs its own verified reference.

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| Compile wants `cuda_runtime_api.h` | CUTLASS defaulted to the CUDA backend | Add `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET` |
| `RequiresExtension: ... SPIR-V extension` | SPIR-V extensions not enabled at link | Add the `-spirv-ext` flags, or declare the `xe-fuse` dependency and let `SyclBuilder` do it |
| Correct but far slower than oneDNN | Register spill from the default tile | `optimize-intel-kernels` Step 5: large GRF **or** a smaller tile, never both |
| Output is garbage, no error | Interleaved vs split-half, or the `[K, N]` layout | Re-derive with candidate references |
| Output has the right values in the wrong places | Duplicated store not compacted | Take `out[..., 0::2]` |
| Fused MLP no faster than unfused | A compacting copy between chained kernels | See the chaining constraint |
