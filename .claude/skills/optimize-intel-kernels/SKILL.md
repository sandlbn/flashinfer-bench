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
   flashinfer-bench run --devices xpu:0  (correctness gates performance)
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
flashinfer-bench run --local tmp/flashinfer-trace \
    --devices xpu:0 --definitions <name>
```

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
flashinfer-bench run --local tmp/flashinfer-trace --devices xpu:0
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

Currently registered: `rms_norm`, `fused_add_rms_norm`, `gemma_rms_norm` (vLLM XPU);
`rmsnorm`, `fused_add_rmsnorm` (sgl-kernel-xpu). Adding more is a table entry plus a
wrapper whose signature has been checked against the upstream binding.

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
