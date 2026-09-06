---
name: optimize-intel-kernels
description: Write, validate and benchmark SYCL kernels for Intel GPUs against existing FlashInfer-Trace definitions. Use when optimizing a kernel for Intel (Battlemage, Crescent Island), porting a CUDA/Triton solution to SYCL, or picking which Intel kernels are worth working on.
---

# Optimize Intel kernels

Drive kernel optimization for Intel GPUs: pick a target, write a SYCL solution, validate it
against the definition's reference on real hardware, and benchmark it.

## The pipeline, and how it differs from the NVIDIA one

The model-onboarding skills (`extract-kernel-definitions`, `collect-workloads`) harvest
Definitions and Workloads from an SGLang inference run using **FlashInfer's** tracing hooks.
That path is CUDA-only: on Intel, SGLang does not use FlashInfer, it uses its own
`sgl-kernel-xpu`. So the harvest mechanism does not transfer.

That does not block Intel work, because of how the data model is split:

| Artifact | Hardware-specific? | Where it comes from for Intel |
| --- | --- | --- |
| Definition | **No** | Reused as-is from the dataset (harvested on NVIDIA) |
| Workload | **No** | Reused as-is — never re-collect for Intel |
| Solution | **Yes** | Written here, in SYCL |
| Trace | **Yes** | Measured here, on Intel hardware |

So the Intel loop is: **take an existing definition → write a SYCL solution → validate →
benchmark**. You do not need SGLang running on Intel to optimize Intel kernels. You need it
only to *deploy* an optimized kernel into a live server, which is a separate integration.

```
existing Definition + Workloads  (hardware-independent, already in the dataset)
              │
              ▼
   write SYCL solution  ────────────  flashinfer_bench.SYCL_PROMPT
              │                       examples/sycl/rmsnorm_sycl.cpp
              ▼
   validate-references --device xpu:0   (reference must be correct on this device first)
              │
              ▼
   FIB_DEVICE_BACKEND=xpu flashinfer-bench run  (correctness gates performance)
              │
              ▼
   Trace tagged with hardware_id + timing methodology
```

## Prerequisites

```bash
# Intel GPU + PyTorch XPU
pip install torch --index-url https://download.pytorch.org/whl/xpu

# oneAPI DPC++ for the SYCL compiler. If installed but not on PATH:
source /opt/intel/oneapi/setvars.sh
# or point the builder straight at it:
export FIB_SYCL_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx
```

Confirm the toolchain is visible:

```python
from flashinfer_bench.compile.builders import SyclBuilder
from flashinfer_bench.device import get_accelerator, list_devices

assert SyclBuilder.is_available()          # compiler found
print(list_devices())                      # ['xpu:0']
print(get_accelerator("xpu:0").capabilities("xpu:0"))
```

**Put the host in a performance power profile before measuring anything.** On a laptop this
matters more than any benchmark parameter — a power-saving profile produced ~40-50%
run-to-run variance versus ~1% under `performance`. See `docs/start/hardware-support.mdx`.

## Step 1: Pick a target

Two upstream projects define what Intel serving actually needs. Their kernel lists are the
optimization backlog, and both are SYCL.

**`sgl-kernel-xpu`** (github.com/sgl-project/sgl-kernel-xpu) — SGLang's Intel kernels.
Supports exactly two targets: `bmg` (Battlemage, device IP 20) and `cri` (Crescent Island,
device IP 35, pre-silicon). Kernel families:

| Area | Kernels | flashinfer-bench op_type |
| --- | --- | --- |
| Attention | FMHA prefill/decode | `gqa-paged`, `gqa-ragged` |
| MLA | decode, prefill, sparse decode/prefill | `mla-paged`, `dsa-paged` |
| GEMM | GroupGemm, W4A16, W8A16 | `gemm` |
| LoRA | SGEMM LoRA A/B fwd, QKV LoRA | `gemm` |
| Linear attention | GdnAttn | `gdn` |
| Elementwise | rope, norms, activations | `rmsnorm`, `rope` |

**`vllm-xpu-kernels`** (github.com/vllm-project/vllm-xpu-kernels) — vLLM's Intel kernels.
Broader elementwise and quantization coverage:

| Area | Kernels | flashinfer-bench op_type |
| --- | --- | --- |
| Norms | `rms_norm`, `fused_add_rms_norm`, `gemma_rms_norm` | `rmsnorm` |
| RoPE | `rotary_embedding`, `fused_qk_norm_rope` | `rope` |
| Activations | `silu_and_mul`, `gelu_*`, `fatrelu_and_mul` | (fused into MLP defs) |
| Quantization | fp8 / mxfp4 quant family, `awq_dequantize` | `gemm`, lowbit evaluators |
| KV cache | `reshape_and_cache`, `concat_and_cache_mla`, `gather_cache` | `gqa-paged`, `mla-paged` |
| Attention glue | `merge_attn_states`, `topk_per_row` | `gqa-paged`, `sampling` |

Pick a definition in the dataset whose op_type appears above, preferring ones with real
workloads attached. Norm and RoPE kernels are the best starting points: small, memory-bound,
easy to verify, and they appear in every model.

## Step 2: Check the definition is valid on this device

Never optimize against a reference that is wrong on the target. Correctness is judged
against the reference *running on the same device*, so a reference that misbehaves on XPU
would validate a wrong kernel:

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace \
    --device xpu:0 --definitions <name> --output attest-xpu.json
```

`PASSED` clears it. `UNSUPPORTED_DTYPE` means the device lacks a dtype the definition needs
(Battlemage has no native FP8) — pick a different target. `MISMATCH` means the PyTorch
reference itself differs on XPU; report that upstream rather than working around it.

## Step 3: Write the SYCL solution

Load `flashinfer_bench.SYCL_PROMPT` for the full guidance. The one rule that matters:

```cpp
// Use the framework's queue. Do NOT create your own -- SYCL pointers are bound to a
// sycl::context, and these tensors belong to PyTorch's.
sycl::queue* q = static_cast<sycl::queue*>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));
```

`examples/sycl/rmsnorm_sycl.cpp` is a complete worked kernel using `nd_range`, a group
reduction, and grid-stride loops.

Solution spec:

```json
{
  "spec": {
    "language": "sycl",
    "target_hardware": ["xpu"],
    "entry_point": "kernel.cpp::my_kernel",
    "destination_passing_style": true,
    "dependencies": ["onemkl"]
  }
}
```

Arguments arrive in definition order: inputs first, then outputs. Export with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`.

**Before hand-writing a GEMM-shaped kernel, try oneMKL or oneDNN.** They are tuned per
architecture and usually win. Hand-write when fusing, or for something the libraries do not
cover — which is where most of the available speedup actually is.

## Step 4: Validate and benchmark

```bash
FIB_DEVICE_BACKEND=xpu flashinfer-bench run --local tmp/flashinfer-trace \
    --definitions <name>
```

`run` has no `--devices` flag — only `serve` does. It benchmarks whatever
`flashinfer_bench.device.list_devices()` returns, and `FIB_DEVICE_BACKEND` is what selects
the backend. On an Intel-only machine the variable is unnecessary; on a host that also has
an NVIDIA card it is mandatory, because the backend preference order is `cuda, xpu, cpu`.

Correctness gates performance: a solution that fails numerically never gets a latency. The
trace records `environment.hardware_id` and `environment.libs.timing`, so results are
grouped by device and never ranked across devices.

For a quick iteration loop without the dataset, build directly:

```python
from flashinfer_bench.compile.builders import SyclBuilder
runnable = SyclBuilder().build(definition, solution)
runnable(*inputs, *outputs)
```

## Step 5: Compare against the upstream kernel

The point of the exercise is to beat what SGLang or vLLM already ships. Beating PyTorch
eager is the easy bar and means little — eager makes several passes over memory, so almost
any fused kernel wins.

Add the upstream kernels to the dataset as baselines, then benchmark normally:

```bash
flashinfer-bench add-baselines --local tmp/flashinfer-trace
FIB_DEVICE_BACKEND=xpu flashinfer-bench run --local tmp/flashinfer-trace
```

`add-baselines` writes a Solution per matching upstream kernel into
`solutions/baseline/<op_type>/`, authored to the providing project. They then appear in
results beside your own solution, measured against the same reference with the same
correctness gates — no separate report to reconcile.

Baselines are matched **strictly**: same `op_type`, and exactly the same input and output
names in the same order. A near-miss produces no baseline rather than a confidently wrong
comparison. If your definition should have a baseline but gets none, the signature does not
line up with the upstream kernel — check `flashinfer_bench/integration/xpu_kernels.py` and
add an entry with the real upstream signature rather than forcing a match.

Currently registered: `rms_norm`, `fused_add_rms_norm`, and the activation family
(`silu_and_mul`, `mul_and_silu`, `gelu_and_mul`, `gelu_tanh_and_mul`, `gelu_new`,
`gelu_fast`, `gelu_quick`) from vLLM XPU; `rmsnorm` and `fused_add_rmsnorm` from
sgl-kernel-xpu. Nothing yet for RoPE, attention, MLA, MoE, quantization or KV-cache ops.
Adding more is a table entry plus a wrapper whose signature *and semantics* have been
checked against the upstream binding — `gemma_rms_norm` was removed after matching
`rms_norm`'s arguments exactly while computing `(1 + weight)` scaling. The step-by-step
wiring procedure is in [`onboard-model-intel`](../onboard-model-intel/SKILL.md), Phase 5.

**A solution slower than the upstream kernel is not an optimization.** Record it and say so.

### Installing the providers

Neither library is a dependency; baselines are simply not offered when they are absent.

```bash
# vLLM Intel kernels
git clone https://github.com/vllm-project/vllm-xpu-kernels.git
MAX_JOBS=2 pip install -v --no-build-isolation ./vllm-xpu-kernels

# SGLang Intel kernels (Battlemage or Crescent Island only)
git clone https://github.com/sgl-project/sgl-kernel-xpu.git
MAX_JOBS=2 pip install -v --no-build-isolation \
    --config-settings=cmake.define.DPCPP_SYCL_TARGET=bmg ./sgl-kernel-xpu
```

Both build large SYCL codebases. Cap `MAX_JOBS`: individual translation units can need
several GiB, and sgl-kernel-xpu's build has an OOM guard that stops the build rather than
letting the host thrash. Expect tens of minutes.

## Fusing into the GEMM epilogue with Xe-Fuse

> **Read [`xe-fuse.md`](xe-fuse.md) before writing an Xe-Fuse kernel.** It carries the
> build flags, the verified operand layout, the interleaved-vs-split-half trap, the
> duplicated-output constraint that blocks kernel chaining, and measured numbers. All of it
> was established by building and running, and several points contradict what the source
> reads like.

### Overview

Everything above optimizes memory-bound kernels — norms, activations, RoPE — which together
are single-digit percentages of device time. GEMM is typically ~70-85%, runs through oneDNN,
and you will not beat oneDNN by writing a better matrix multiply.

The lever is to stop treating them as separate kernels. [Xe-Fuse](https://github.com/IntelLabs/Xe-Fuse)
folds the memory-bound work into the GEMM's *epilogue*, on data still in registers, removing
both the extra launches and the round trip through memory.

### Generating a fused kernel

```bash
git clone https://github.com/IntelLabs/Xe-Fuse.git
cd Xe-Fuse/autotune
python generate_kernel.py --preset k2 --m 2048 --n 9728 --k 896 -o k2_qwen.cpp
```

| Preset | Fusion | Maps to |
| --- | --- | --- |
| `k1`, `k1v2` | `D = acc * R[m]` | GEMM + RMSNorm row scaling |
| `k2`, `k2v2` | `D = SwiGLU(acc * R[m])` | gate/up projection + norm + SwiGLU |
| `k2_geglu`, `k2v2_geglu` | `D = GeGLU(acc * R[m])` | Gemma-style gated FFN |
| `k0a` | `D = gamma[n] * (acc + residual)` | down projection + residual + norm |
| `k3`, `k4`, `k4v2` | `D = RoPE(acc * R[m], cos_sin)` | qkv projection + RoPE |
| `w8a8_dequant` | `int32_acc * scale_token[m] * scale_channel[n]` | quantized GEMM |

The `v2` variants use a merged visitor (flat tree) rather than a composed one — same maths,
different codegen. Benchmark them against each other rather than assuming.

### Building one

The generated kernel is a single translation unit including CUTLASS headers. It needs
`sycl-tla` (Intel's CUTLASS-SYCL), which the project's CMake fetches, and two definitions
that switch CUTLASS to its SYCL backend. Without them the compile fails looking for
`cuda_runtime_api.h`, which is the wrong backend entirely:

```bash
icpx -fsycl -O2 -std=c++17 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I<xe-fuse>/include \
  -I<sycl-tla>/include -I<sycl-tla>/tools/util/include \
  -I<sycl-tla>/examples/common -I<sycl-tla>/applications \
  -c k2_qwen.cpp -o k2_qwen.o
```

One TU, a few minutes. This is a different cost profile from `sgl-kernel-xpu`, which
instantiates hundreds of CUTLASS translation units and will exhaust host memory on a small
machine.

### Optimizing further

The generated kernel is a starting point, not an answer. What it emits is a composable
Epilogue Visitor Tree:

```cpp
using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
using KernelConfig = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
```

That tree is the thing to optimize, built from a small vocabulary:

- **Sources** — `Acc`, `AuxLoad<E>`, `ColBroadcast` (per-row `scale[m]`), `RowBroadcast`
  (per-column `gamma[n]`)
- **Binary** — `Mul`, `Add`, `ScaleRows`, `ScaleCols`, `AddResidual`, `BiasAdd`
- **Activations** — `GeLU`, `GeLUTanh`, `SiLU`, `ReLU`, `Sigmoid`
- **Pairwise (lane shuffle)** — `SwiGLU`, `GeGLU`, `RoPE`

Two cheap axes to search:

1. **Tile shape** — `--tile 128x256x32`, or `auto`. The same tuning that produced a
   1.69x-3.93x spread on a hand-written SYCL kernel applies here too.
2. **The tree itself** — composing a fusion no preset covers, e.g. residual add *and* a
   gated activation in one epilogue.

Wrap the result as a Solution the usual way: replace the generated standalone `main` with a
TVM-FFI entry point taking the Definition's tensors and running on the framework's queue.
It then competes in the same benchmark under the same correctness gates as everything else.

**Xe-Fuse is marked "not stable" and not production-ready by IntelLabs.** Treat it as one
solution source among several, never as a dependency — if it regresses you lose a
contender, not the pipeline.

## Per-architecture facts

[`architectures.md`](architectures.md) carries the per-part record: what Battlemage,
integrated Xe3.0 and Crescent Island differ in, the measured effect of each optimization
with the hardware it was measured on, and the traps.

Read it for the *reasons*. For the values themselves, ask the device -- a capability record
is right on a part nobody has documented yet:

```python
caps = get_accelerator("xpu:0").capabilities("xpu:0")
caps.preferred_sub_group_size   # 32 on Battlemage, read from the driver
caps.vector_width(2)            # 8 bfloat16 per 16-byte access
caps.supports_large_grf         # gates the register-mode fix below
caps.supported_dtypes           # Battlemage has no FP8
```

## Vectorize the loads before anything else

A memory-bound kernel that reads one element per work-item leaves most of the memory pipe
idle. `examples/sycl/rmsnorm_sycl.cpp` is written that way -- it is a readable
introduction to `nd_range` and group reductions, not a fast kernel -- so do not copy its
access pattern into anything you intend to benchmark.

vLLM's Intel kernels (`vllm-xpu-kernels/csrc/layernorm.cpp`) do three things this example
does not, and they are worth copying verbatim:

```cpp
// 1. Eight bfloat16 = one 16-byte access. This is the whole difference.
template <typename T, int N>
struct alignas(sizeof(T) * N) VecN { T val[N]; };
constexpr int kVecSize = (sizeof(scalar_t) == 2) ? 8 : 4;

// 2. Pin the sub-group; Battlemage offers {16, 32} and the reduction is cheapest at 32.
void operator()(sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]]

// 3. Size the work-group to the row, not to a fixed constant.
size_t wg = std::min(hidden / kVecSize, max_work_group_size);
```

Measured on Battlemage, bf16 RMSNorm at hidden=2048, against the same kernel with scalar
loads:

| batch | scalar | vectorized | |
| --- | --- | --- | --- |
| 12383 | 0.3213 ms | 0.2374 ms | **1.35x** |
| 16254 | 0.4222 ms | 0.3165 ms | **1.34x** |
| 1-79 | ~0.047 ms | ~0.047 ms | unchanged |

The win is entirely in the bandwidth-bound regime, which is exactly where a scalar kernel
loses to the upstream one. Vectorizing took the kernel from ~25% *slower* than vLLM at
prefill to level with it, while keeping a 1.1-1.3x lead at decode sizes.

Two corollaries. **Guard the wide path**: it needs `hidden % vec == 0` and 16-byte-aligned
base pointers, so check both and keep a scalar fallback rather than assuming. And **do not
reach for a register cache first** -- caching the row across the reduction to avoid the
second read measured within noise at every batch size, and vLLM's kernel does not bother
either. Vectorization is the lever; reuse is not.

## Register mode, and why a correct kernel can be 14x slow

A tile that does not fit the default 128-register file spills to scratch, and spilling is
invisible to the correctness gate -- the kernel is right, just slow. Check it in unitrace's
kernel-properties table before tuning anything else:

```
SIMD=16  GRF=128  Spill Memory Per Thread = 8576     # <- this is the problem
```

Two fixes, which do not compose:

- `FIB_SYCL_LARGE_GRF=1` -- builds AOT kernels with 256 registers per thread instead of
  128. On an Xe-Fuse `k2` at tile 256x256x32 this took spill to zero and latency from
  8.668 ms to 0.618 ms.
- a smaller tile, which avoids the spill instead of accommodating it, and keeps full
  occupancy.

Large GRF halves the threads resident per EU, so it pays only when it removes spill.
Applying it to a tile that already fits makes things slightly worse. Measure both.

## Choosing a SYCL target

Leaving `sycl_target` unset compiles to SPIR-V and JIT-compiles at load. That is portable
and is how a kernel runs on silicon released after it was written. Ahead-of-time compilation
uses `-fsycl-targets=spir64_gen` with a device name (`bmg`, `cri`); the builder derives this
from the device's capability record.

Intel identifies generations by device IP version, from
`torch.xpu.get_device_properties(i).version`:

| IP | Architecture | Target name | Notes |
| --- | --- | --- | --- |
| 20 | Xe2 (Battlemage) | `bmg` | Discrete Arc B-series |
| 30 | Xe3.0 | — | Integrated (Panther/Wildcat Lake); not an sgl-kernel-xpu target |
| 35 | Xe3.5 (Crescent Island) | `cri` | Pre-silicon |

An integrated Xe3.0 part runs SYCL and Triton solutions perfectly well through
flashinfer-bench, and is a fine kernel-development machine. It cannot run `sgl-kernel-xpu`
or SGLang, whose kernels are compiled per-architecture for `bmg` or `cri` only.

## Common mistakes

1. **Re-collecting workloads for Intel.** They are hardware-independent; the file you would
   produce is the one you already have.
2. **Benchmarking before cross-validating the reference.** A wrong reference silently
   validates a wrong kernel.
3. **Measuring under a power-saving profile.** The variance swamps the effect you are
   looking for.
4. **Comparing an Intel speedup against an NVIDIA one.** Speedup is relative to the
   reference on the same device; the two numbers are not the same quantity.
5. **Creating a `sycl::queue` inside the kernel.** Wrong context, undefined behaviour.
6. **Hand-writing a GEMM before trying oneMKL/oneDNN.**
