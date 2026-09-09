# Xe-Fuse

Fusing a memory-bound epilogue into a GEMM on CUTLASS-SYCL, so the elementwise work happens
on data still in registers.

[Xe-Fuse](https://github.com/IntelLabs/Xe-Fuse) is marked **not stable** by IntelLabs. Treat
it as one solution source among several.

## When it applies

There are two ways to express a fused epilogue on this box, and which of them *can* express
a given fusion is a fact about the epilogue, not a preference:

- oneDNN **post-ops** (`/optimize-onednn`) are elementwise or binary on a *single* GEMM's
  output. A fusion needing a lane shuffle — SwiGLU, GeGLU, RoPE on packed qkv — cannot be
  written as one.
- An Xe-Fuse **EVT** adds pairwise lane ops. It needs an AOT `sycl_target`, and the tile
  shape is a swept trial parameter, never a table lookup.

Every path that can express the fusion becomes a candidate; build each and let the benchmark
order them. Xe-Fuse replaces the GEMM as well as the epilogue, so put its plain GEMM with no
epilogue in the same harness at the same shape: that arm separates what the fused epilogue
changed from what the substituted GEMM changed.

## Setup

Both checkouts come from `/clone-repos`; the builder needs their paths:

```bash
export FIB_XE_FUSE_DIR=$PWD/tmp/Xe-Fuse
export FIB_SYCL_TLA_DIR=$PWD/tmp/sycl-tla
python tmp/Xe-Fuse/autotune/generate_kernel.py --help | head -20
```

`SyclBuilder` supplies every compile and link flag when a solution declares the `xe-fuse`
dependency and those variables are set. Build by hand only when debugging outside the
builder:

```bash
icpx -fsycl -O2 -std=c++17 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I$FIB_XE_FUSE_DIR/include -I$FIB_SYCL_TLA_DIR/include \
  -I$FIB_SYCL_TLA_DIR/tools/util/include -I$FIB_SYCL_TLA_DIR/examples/common \
  -I$FIB_SYCL_TLA_DIR/applications \
  -c kernel.cpp -o kernel.o
```

Without the two `-D` flags CUTLASS defaults to its CUDA backend. Linking additionally needs:

```
-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
```

One translation unit per kernel — unlike `sgl-kernel-xpu`, which instantiates hundreds of
CUTLASS TUs and will exhaust a small host.

## Generate, wrap, validate

```bash
# 1. Generate
python tmp/Xe-Fuse/autotune/generate_kernel.py --preset k2 --m M --n N --k K -o kernel.cpp

# 2. Wrap: replace the generated standalone main() with a TVM-FFI entry point running on
#    the framework's queue. Template: examples/sycl/xefuse_gemm_swiglu.cpp

# 3. Write the reference to match the kernel's real layout (see below), then:
flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>

# 4. Check register spill BEFORE benchmarking — SKILL.md, "check the failure mode"
unitrace -d -v -o prof python <script>.py   # end with current_stream().synchronize()

# 5. Benchmark
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

## Presets

The preset list belongs to the generator in the checkout, so read it from there rather than
from a copy here — `scripts/fusion_candidates.py` does the same:

```bash
python tmp/Xe-Fuse/autotune/generate_kernel.py --list-presets
```

Each line gives the epilogue that preset emits, in the same `D = ...` form as the EVT
below, split into GEMM epilogue presets and standalone ones. Reading an epilogue back to
the call it can replace: `acc * R[m]` is a per-row scale, which is where an RMSNorm's row
factor goes when it is folded into the GEMM; `SwiGLU` and `GeGLU` are the two gated-FFN
activations, differing only in the gate function, and fuse a gate/up projection with its
norm; `gamma[n] * (acc + residual)` is a down projection's residual add carrying the next
norm's per-column weight; `int32_acc * scale_token[m] * scale_channel[n]` is a quantized
GEMM's per-token by per-channel dequant.

A `v2` suffix is a merged (flat) visitor instead of a composed tree — same maths, different
codegen. Both are buildable for the same fusion, so measure both rather than picking one.

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

Two search axes: the tile shape (`--tile 128x256x32`) and the tree itself.

## Three layout rules that produce silent garbage

None of these raises.

### B is `[K, N]` row-major, whatever the stride construction reads like

The generated kernel builds its B stride with
`make_cute_packed_stride(StrideB{}, make_shape(N, K, L))`, which reads as though B were
`[N, K]`. It is not. Verify any new preset by testing candidate references
(`swiglu(a @ b.reshape(K, N))`, `swiglu(a @ b.T)`, split-half variants) against the
kernel's actual output: the right one agrees to bf16 noise, every wrong one disagrees by
O(1). There is no intermediate case — a large error is the wrong layout, not a tolerance
problem.

### Pairwise ops are interleaved, not split-half

Xe-Fuse's pairwise ops take **interleaved** `(gate, up)` pairs — gate on even lanes, up on
odd. vLLM's `silu_and_mul` is **split-half**: `silu(x[..., :d]) * x[..., d:]`. A reference
written for one is wrong for the other.

### The output is duplicated and the visitor cannot compact it

`XePairwiseCompute` returns a fragment of the same width, so both lanes of a pair get the
same value and CUTLASS's epilogue stores every lane. The result is `[M, 2d]` with each
answer written twice; take `out[..., 0::2]`.

This is what a second fused kernel has to consume. Chaining `k2` into `k0a` needs a strided
A operand on the second GEMM (CuTe row-major `StrideA` carries a compile-time `_1` on K —
read the stride type the generator emits before assuming it admits a non-unit K stride), a
compacting store in the visitor (an upstream change), or the compacting copy. Where the copy
is taken, benchmark the chain against the two kernels run unchained in the same harness: the
difference is what the copy costs. Resolve this before wiring a second fused kernel into an
MLP.

## Adapting a model's weights: transform, don't branch

The kernel never changes per model; the weights transform once at load. An RMSNorm splits
into two pieces:

- **per-channel `gamma`** scales input channels, so it multiplies into the weight;
- **per-token `rsqrt(mean(x²)+eps)`** commutes with the GEMM, so it stays a runtime per-row
  scale.

`flashinfer_bench/integration/weight_layout.py` implements this:

```python
B, eps = qwen_style_mlp_weights(layer.mlp, layer.post_attention_layernorm)
kernel(x, B, rms_row_scale(x, eps), out)
result = deinterleave_output(out)
```

The discriminating check: take the relative error against an fp32 reference and compare it
against bf16's spacing (`2**-7`). At that order it is the format's own rounding; an order of
magnitude more is a layout error, never a tolerance to loosen.

Branch on the **fusion** (SwiGLU vs GeGLU vs residual+norm), never on the model layout.

## Failure table

| Symptom | Fix |
| --- | --- |
| Compile wants `cuda_runtime_api.h` | Add `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET` |
| `RequiresExtension: ... SPIR-V extension` | Add the `-spirv-ext` flags, or declare the `xe-fuse` dependency and let `SyclBuilder` do it |
| Correct, and `SPILLS` reports a count | Register pressure — `optimize-intel-kernels`, "Build, and check the failure mode for your language"; each candidate measured alone |
| Output is garbage, no error | Interleaved vs split-half, or the `[K, N]` layout — re-derive with candidate references |
| Output has the right values in the wrong places | Take `out[..., 0::2]` |
| Fused MLP no faster than unfused | A compacting copy between chained kernels — see the chaining constraint |
