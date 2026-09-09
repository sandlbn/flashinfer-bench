---
name: optimize-intel-kernels
description: Write, validate and benchmark SYCL or Triton solutions for existing FlashInfer-Trace definitions on Intel GPUs, including porting CUDA Triton solutions to XPU. Use when optimizing a kernel for an Intel part or deciding which Intel kernels are worth working on.
---

# Optimize Intel kernels

Take a definition that already exists, write a solution that beats the kernel the serving
stack ships, and prove it on hardware. Definitions and workloads are hardware-agnostic and
are reused as-is. You produce a **Solution** (SYCL or Triton) and a **Trace**.

## Step 0: Prepare the host

```bash
powerprofilesctl set performance          # measure under a performance profile, always
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
flashinfer-bench providers list
```

| Check fails | Do this |
| --- | --- |
| `list_devices()` has no `xpu:0` | Driver or torch-xpu problem — `docs/start/hardware-support.mdx`. Stop. |
| `SyclBuilder.is_available()` is False | `source /opt/intel/oneapi/setvars.sh`, or set `FIB_SYCL_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx` |
| A provider you need is absent | `flashinfer-bench providers install <name>` — see `onboard-model-intel/providers.md` |
| Mixed-vendor host picks the wrong backend | `export FIB_DEVICE_BACKEND=xpu` |

## Step 1: Pick a target

Run `/profile-intel` first; it ranks families by recoverable time and routes each to the
skill that owns it. Without a model, definitions that already have workloads are the ones
you can benchmark today:

```bash
ls tmp/flashinfer-trace/workloads/*/ | head -40
```

Then filter:

- **Route by op_type** using the decision table in `onboard-model-intel/SKILL.md` Phase 5.
  GEMM-shaped work goes to oneDNN before anyone hand-writes a matmul.
- **Check the dtype is native on this part**: `caps.supported_dtypes` is native ∪ emulated,
  so ask `caps.is_native_dtype()` separately. An emulated dtype runs and passes at emulated
  throughput; do not read its latency as a native number.
- **Prefer ops with an upstream baseline**, so Step 6 has something to beat.
- **Quantized checkpoints** reach different kernels from their bf16 siblings — read
  `references/quantized-gemm.md` before writing anything for an FP8 or INT4 model.

## Step 2: Validate the reference on `xpu:0`

Correctness is judged against the reference running on this device.

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace \
    --device xpu:0 --definitions <name>
```

| Status | Action |
| --- | --- |
| `PASSED` | Continue |
| `UNSUPPORTED_DTYPE` | Pick a different target |
| `MISMATCH` | The PyTorch reference itself differs on XPU. Report upstream; do not work around it |
| `NO_WORKLOAD` | Attach a workload first — `onboard-model-intel` Phase 2 |

## Step 3: Choose the language

| Situation | Write |
| --- | --- |
| GEMM-shaped, or a GEMM with an elementwise epilogue | **oneDNN via SYCL** — post-ops fuse the epilogue (`/optimize-onednn`) |
| Memory-bound elementwise or row-wise (norms, activations, RoPE) | **SYCL** for speed, **Triton** for portability |
| A CUDA Triton solution already exists for this definition | **Triton** — port it (Step 4b) |
| A fusion oneDNN post-ops cannot express | **Xe-Fuse** — see the criterion at the end |

Elementwise and row-wise definitions get **both** a SYCL and a Triton solution.
`tests/integration/test_intree_kernels.py` enforces that both exist for every in-tree
signature.

## Step 4a: Write a SYCL solution

Load `flashinfer_bench.SYCL_PROMPT` for the full contract. The rule that matters:

```cpp
// Use the framework's queue. Never create your own -- SYCL pointers are bound to a
// sycl::context and these tensors belong to PyTorch's.
sycl::queue* q = static_cast<sycl::queue*>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));
```

```json
{"spec": {"language": "sycl", "target_hardware": ["xpu"],
          "entry_point": "kernel.cpp::my_kernel",
          "destination_passing_style": true, "dependencies": ["onednn"]}}
```

Arguments arrive in definition order, inputs then outputs. Export with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`. Start from a worked example:

| Want | Copy |
| --- | --- |
| Row-wise reduction | `examples/sycl/rmsnorm_sycl.cpp` |
| Two inputs, two outputs, in-place-safe | `examples/sycl/fused_add_rmsnorm_sycl.cpp` |
| GEMM + fused epilogue via oneDNN post-ops | `examples/sycl/onednn_gemm_swiglu.cpp` |
| GEMM epilogue via CUTLASS-SYCL | `examples/sycl/xefuse_gemm_swiglu.cpp` |

Levers, in order of value:

1. **Vectorize the loads.** `caps.vector_width(itemsize)` gives the element count per
   widest access. Guard the wide path on `hidden % width == 0` and aligned base pointers,
   and keep a scalar fallback.
2. **Pack multiple rows per work-group when rows are narrow** — `architectures.md`, "Rows
   narrower than a sub-group".
3. **Do not hand-write a GEMM.** oneDNN or oneMKL first; hand-write only to fuse something
   they cannot.

## Step 4b: Write a Triton solution

```json
{"spec": {"language": "triton", "target_hardware": ["xpu"],
          "entry_point": "main.py::run", "destination_passing_style": false}}
```

Copy `_TRITON_PREAMBLE` from `flashinfer_bench/integration/intree_kernels.py`. It encodes:

1. **Autotune over `warp_size` as well as `num_warps`.** `warp_size` is an Intel knob;
   Triton accepts it in a `Config` without the kernel taking it as a parameter. Read the
   candidates from the driver:
   ```python
   from triton.runtime import driver
   props = driver.active.utils.get_device_properties(torch.xpu.current_device())
   SUB_GROUP_SIZES = tuple(props.get("sub_group_sizes", (32,)))
   ```
2. **Derive rows-per-program from `max_work_group_size`**, so a narrow row packs several
   rows into one program.
3. **Keep `num_stages` small.** It exists for CUDA's async-copy pipelining, which Intel has
   no equivalent for.

Autotune with `key=["hidden"]` (or the equivalent size axis); the optimum moves with shape.

### Porting a CUDA Triton solution

Triton solutions in the dataset refuse on Intel in their device management — a
`torch.cuda.is_available()` guard, a `.cuda()` call, or `torch.device("cuda")` — not in the
kernel.

```bash
python scripts/port_triton_solutions_to_xpu.py --local tmp/flashinfer-trace --definitions <name>
```

It rewrites those constructs to derive the device from the inputs and writes a **new**
solution under a separate author. Never edit the original. A ported kernel gives you
correctness; retune the launch config for Intel separately.

## Step 5: Build, and check the failure mode for your language

**SYCL — check for register spill before benchmarking.** Spill does not fail correctness; it
produces a correct kernel that is many times too slow.

```bash
unitrace -d -v -o prof python <script>.py   # script must end with
                                            # torch.xpu.current_stream().synchronize()
grep -A2 "Kernel Properties" prof.txt      # look for: Spill Memory Per Thread
```

Nonzero spill: apply **either** `FIB_SYCL_LARGE_GRF=1` **or** a smaller tile, measure both,
never both at once — large GRF halves the threads resident per EU.

**Triton — confirm the autotuner ran.** Clear `~/.triton/cache` when in doubt, and check
that the chosen config differs between a small and a large workload.

## Step 5b: Search, do not settle for the first draft

`/wrap-kernel-for-tuning` is the loop: candidates are timed against the kernel a deployment
would otherwise run, correctness gates timing, and trials form a tree. It takes SYCL, oneDNN
and Triton candidates through `scripts/kernel_trials.py` and compiles them with the same
builder the benchmark uses, so a winner is already a Solution.

For an unattended sweep over work-group size and sub-group width on one model:

```bash
python scripts/optimize_model_kernels_xpu.py --model <hf_repo_id> --device xpu:0 \
    --output tmp/opt-<model_slug>          # required
```

It will not find a fusion or remove a materialisation — for those see `/find-kernel-gaps`.

## Step 6: Benchmark against the upstream kernel

The bar is the kernel the serving stack ships, not PyTorch eager.

```bash
flashinfer-bench add-baselines --local tmp/flashinfer-trace --in-tree --definitions <name>
flashinfer-bench add-baselines --local tmp/flashinfer-trace \
    --providers vllm-xpu,sgl-kernel-xpu --definitions <name>
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

Always pass `--definitions`; without it `add-baselines` rewrites every baseline in the
dataset. Traces record `environment.hardware_id` and `environment.libs.timing`; results
group by device and are never ranked across devices.

**No baseline appeared?** Matching is strict — same `op_type`, same input and output names
in the same order. Check `REGISTRY` in `flashinfer_bench/integration/xpu_kernels.py` and add
an entry with the real upstream signature; verify semantics, not just arity.

A solution slower than the upstream kernel is not an optimization. Record it and say so.

## Step 7: Check the win survives the caller

Before claiming a deployment win, work through `architectures.md`, "Traps when deploying a
kernel through `apply()`", then run `/measure-serving-win`.

## Step 8: Record

Traces from `--save-results` are the deliverable. Publishing them is `onboard-model-intel`
Phase 7 and `/submit-onboarding-prs`.

## When to reach for Xe-Fuse

Only when the fusion cannot be expressed as oneDNN post-ops — post-ops act on a single
GEMM's output, so anything needing a lane shuffle (SwiGLU, GeGLU, RoPE on packed qkv) is out
of reach — **and** the fusion saves more than the GEMM deficit against oneDNN costs. Read
`xe-fuse.md` first. Xe-Fuse is marked not-stable by IntelLabs: one solution source, never a
dependency.

**Read `xe-matrix.md` before writing any DPAS-backed kernel** — GEMM, attention, or
quantized matmul. Several rules there contradict what a CUDA author would assume.

## Measure what the access pattern allows before rewriting a memory-bound kernel

Peak bandwidth is the wrong yardstick. Time three reads with the accelerator's timer: the
whole tensor contiguously, the slice in the shape the kernel is obliged to touch, and the
kernel itself. If the kernel matches the strided read, the gap to peak lives in the data
layout — a serving-stack decision, not a kernel one. Treat the strided figure as a lower
bound, not proof the kernel is optimal.

## Failure table

| Symptom | Fix |
| --- | --- |
| `cannot find -ldnnl` | `export FIB_ONEDNN_DIR=/opt/intel/oneapi/dnnl/latest` |
| Compile looks for `cuda_runtime_api.h` | Add `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`; see `xe-fuse.md` |
| Correct but many times too slow | Register spill — Step 5 |
| Triton solution refuses to run on XPU | Step 4b porting script |
| Same latency at every workload size | Triton autotuner not re-running — Step 5, clear the cache |
| No baseline in the results | Strict signature match failed — Step 6 |
| `UNSUPPORTED_DTYPE` | Step 1 dtype filter |

## Sources

- `flashinfer_bench.SYCL_PROMPT` — the SYCL solution contract
- `examples/sycl/` — worked kernels, one per pattern
- `flashinfer_bench/integration/intree_kernels.py` — canonical Intel Triton preamble
- `architectures.md` — per-part rules and `apply()` deployment traps
- `xe-matrix.md` — DPAS shapes, block-2D constraints, tile configs, schedulers, precisions
- `xe-fuse.md` — GEMM epilogue fusion
- `references/quantized-gemm.md` — FP8 / INT4 checkpoints: tune the config before writing a kernel
- `/optimize-onednn` — diagnosing and fixing the oneDNN GEMM call
- `onboard-model-intel/providers.md` — installing and verifying kernel providers
