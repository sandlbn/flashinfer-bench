---
name: optimize-intel-kernels
description: Write, validate and benchmark SYCL or Triton solutions on Intel GPUs -- for an existing FlashInfer-Trace definition, for a CUDA Triton solution being ported to XPU, or for an op the routing found nothing implements well (an authored_callsite / authored_apply row of scripts/bound_candidates.py). Use when optimizing a kernel for an Intel part, authoring one where no tuned kernel exists, or deciding which Intel kernels are worth working on.
---

# Optimize Intel kernels

Take an op that runs on the device -- a definition's kernel, a CUDA Triton solution to be
ported, or an op the routing says nothing implements well -- write a solution that beats
the kernel the serving stack runs today, and prove it on hardware. Definitions and workloads
are hardware-agnostic and are reused as-is. You produce a **Solution** (SYCL or Triton) and
a **Trace**, or, for a call-site delivery, the kernel and the diff that binds it.

## Step 0: Prepare the host

```bash
powerprofilesctl set performance          # measure under a performance profile, always
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
flashinfer-bench providers list
python scripts/calibrate_part.py          # the part's measured costs, including the authored-stream rate
```

| Check fails | Do this |
| --- | --- |
| `list_devices()` has no `xpu:0` | Driver or torch-xpu problem — `docs/start/hardware-support.mdx`. Stop. |
| `SyclBuilder.is_available()` is False | `source /opt/intel/oneapi/setvars.sh`, or set `FIB_SYCL_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx` |
| A provider you need is absent | `flashinfer-bench providers install <name>` — see `onboard-model-intel/providers.md` |
| Mixed-vendor host picks the wrong backend | `export FIB_DEVICE_BACKEND=xpu` |
| `calibrate_part.py` prints `authored stream unavailable` | Triton has no backend for this device, or the rounds did not settle on a shared box; retry idle. Until it measures, the routing prices no authored kernel |

## Step 1: Pick a target

Run `/profile-intel` first; it ranks families by recoverable time and routes each to the
skill that owns it. When a run has been through `scripts/bound_candidates.py`, its
`worklist.json` is the target list: each row names the candidate, the mechanism that
admitted it, the ceiling it was priced to, and `bound.json` carries the measured regime
that the language decision below reads. Without a model, definitions that already have
workloads are the ones you can benchmark today:

```bash
ls tmp/flashinfer-trace/workloads/*/ | head -40
```

Then filter:

- **Route by the routing's rows** when they exist; without them, the measured regime of the
  op's harness decides the language (Step 3), never a table of op types.
- **Check the dtype is native on this part**: `caps.supported_dtypes` is native ∪ emulated,
  so ask `caps.is_native_dtype()` separately. An emulated dtype runs and passes at emulated
  throughput; do not read its latency as a native number.
- **Prefer ops with an upstream baseline**, so Step 6 has something to beat.
- **Quantized checkpoints** reach different kernels from their bf16 siblings — read
  `references/quantized-gemm.md` before writing anything for an FP8 or INT4 model.

## Step 1b: Entering from the routing — a kernel for an op nothing implements well

`scripts/bound_candidates.py` prices two deliveries of a kernel that does not exist yet:
`authored_callsite` (the written kernel replaces the call where the stack makes it, or a
provider's op; nothing is added to the call path) and `authored_apply` (the written kernel
is delivered through `apply()` and pays the measured dispatch per call). The class admits
them -- `aten` (an ATen kernel inside PyTorch, no local source to patch), `decomposition`
(a sequence of plain kernels with none of its own), `python_op` (a reference, not a
kernel) -- and the gates after that are measurements: no part of the op reached a library
primitive, the authored-stream rate was measured on this part, and the op's device time
sits above what a kernel written here can reach for its bytes.

An ACCEPT row is the brief. Read, from `bound.json`, the candidate's `t_dev_us`, its
`bound_authored_us` (the time the routing priced a written kernel to reach: its bytes at
the authored-stream rate, never under the pattern bound, the compute bound or the launch
floor), the row's `ceiling_us`, and `mechanism_us`. The harness discovery emitted for the
candidate is the baseline a trial has to beat; open the series from the row so every
`benchmark` re-checks it:

```bash
python scripts/kernel_trials.py init <name> <harness.py> --bound <out-dir>/bound.json --mechanism authored_callsite
```

A trial that lands above `bound_authored_us` has not reached what the probe says this part
gives a written kernel; one that lands below it beat the probe, which is worth reporting
because the calibration's meaning moves with it.

Delivery decides the deliverable. For `authored_callsite` no definition is required to
measure or to deploy: the kernel is bound at the call site (the model's source, or the
provider's op) and `/measure-serving-win` runs its plain arm. For `authored_apply`, write
the definition -- `/extract-kernel-definitions` owns naming and axes; the reference is the
op the harness calls -- and the solution, run `flashinfer-bench run --save-results`, and
`/measure-serving-win` runs its overhead arm. The two rows carry the same kernel at two
prices; when only the free delivery is an ACCEPT, the dispatch exceeded the headroom and
`apply()` is not a route for this candidate on this part.

**Illustration (one instance): INTEL_ARC_B580 / triton 3.8.0 / an ATen argmax over a [batch, vocab-slice] float32 tensor at decode batch 4 — re-establish with: `python scripts/bound_candidates.py --report <dir>/discovered.json --resolution <dir>/resolution.json --out-dir <dir>/bound --measure --harness-dir <dir>/harness && grep argmax <dir>/bound/bound.log`**

Two models' runs through the routing after the authored pair was added. The op
`aten.argmax.default` on `[4, 149]` float32, class `aten`, 16 calls: t_dev 5.96 us
streaming, `bound_authored_us` 3.01 us (the launch floor binds; the bytes at the authored
rate come to 0.006 us), spread 0.095 us. `authored_callsite`: ACCEPT, `ceiling_us=2.96`,
`mechanism_us=0`. `authored_apply`: REJECT at `net_positive`, `ceiling_us=-2.96 >
spread_us=0.095` -- the measured dispatch of 5.92 us exceeds the headroom. The second
model's run priced the same op at `ceiling_us=5.48` with the same split. The embedding of
the same runs (`aten.embedding.default`, class `decomposition`) was rejected at `headroom`
with `headroom_us=-772` because `bytes_min` counted the whole 311 MB table; with
`--bytes-min <candidate_id>=8224` (the four rows read, plus the indices) it was accepted
through `authored_callsite` at `ceiling_us=1.06` and rejected through `authored_apply` on
the same dispatch arithmetic. The gather and index ops of both runs were rejected at
`headroom` with t_dev under the launch floor: a written kernel still launches. The
authored-stream probe on this part measured 400.5 GB/s (Triton copy; block 2048, 16 warps,
sub-group 32) against 428.4 GB/s for the framework's contiguous reduction.

End illustration.

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
| `NO_WORKLOAD` | Attach a workload first — `onboard-model-intel`, "Acquire definitions" |

For an authored kernel with no definition yet, the harness's own op is the reference, and
`scripts/kernel_trials.py benchmark` checks a trial against it before it times anything.

## Step 3: Choose the language from the candidate's measured regime

The candidate's row in `bound.json` carries the terms its bound was formed from --
`t_mem_us`, `t_mem_pattern_us`, `t_mem_authored_us`, `t_cmp_us`, `launch_floor_us` -- and
the `regime` the classifier gave it. Without a routing, establish the same quantities:
`/wrap-kernel-for-tuning` for the harness, `scripts/bound_candidates.py --measure` for its
streaming device time and spread. The term that forms the bound decides what a kernel has
to do, and that decides the language question and the measurement that settles it:

| Term forming the bound | What a kernel has to do | The language question, and what settles it |
| --- | --- | --- |
| `t_cmp_us` | keep the matrix engine fed | The library call is the bar (`/optimize-onednn`); author only for a fusion the library cannot express. Triton `tl.dot` through descriptor loads and SYCL-TLA (`xe-matrix.md`) both reach DPAS; run one draft of each through the same trial series and keep the measured one |
| `t_mem_pattern_us` or `t_mem_authored_us` | move `bytes_min` once, at the rate the part gives | Compare `calibration.authored_stream_probe()["gbs"]` (a plain Triton kernel) with `calibration.get().bandwidth_gbs` (the framework's own stream). Apply the difference to the candidate's `bytes_min`: when that time is smaller than the row's headroom, the plain Triton kernel can deliver the ceiling; when it is not, the SYCL kernel with an explicit vector width (`caps.vector_width(itemsize)`) is the draft that has to be measured against it. Row-wise definitions ship both regardless -- `tests/integration/test_intree_kernels.py` requires it |
| `launch_floor_us` | remove launches, not shorten them | No language wins; the routing rejected authoring at `headroom` and the route is a fusion row (`fusion_callsite`, `fusion_apply`) or a call-site change |
| `t_mem_pattern_us` well above `t_mem_us` | change the data's layout, not the kernel | The `layout_transform` row; `architectures.md`, "A row pitch on the memory-channel period" |
| A CUDA Triton solution exists for the definition | port it | Triton (Step 4b); a port gives correctness, and the launch configuration is retuned here |

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
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`. Copy a worked example:

| Want | Copy |
| --- | --- |
| Row-wise reduction | `examples/sycl/rmsnorm_sycl.cpp` |
| Two inputs, two outputs, in-place-safe | `examples/sycl/fused_add_rmsnorm_sycl.cpp` |
| GEMM + fused epilogue via oneDNN post-ops | `examples/sycl/onednn_gemm_swiglu.cpp` |
| GEMM epilogue via CUTLASS-SYCL | `examples/sycl/xefuse_gemm_swiglu.cpp` |

What a SYCL draft varies, each settled by the trial loop and not by belief:

- **Load width.** `caps.vector_width(itemsize)` gives the element count per widest access.
  Guard the wide path on `hidden % width == 0` and aligned base pointers, and keep a scalar
  fallback.
- **Rows per work-group when rows are narrow** — `architectures.md`, "Rows narrower than a
  sub-group".
- **Whether to write a GEMM at all.** oneDNN or oneMKL is the bar; hand-write only to fuse
  something they cannot express.

## Step 4b: Write a Triton solution

```json
{"spec": {"language": "triton", "target_hardware": ["xpu"],
          "entry_point": "main.py::run", "destination_passing_style": false}}
```

Copy `_TRITON_PREAMBLE` from `flashinfer_bench/integration/intree_kernels.py`, and read
`references/triton-xpu.md` before departing from it: it is what the Intel backend's own
checkout says about writing kernels here -- the block programming model, the launch options
that exist only on this backend (`warp_size`, `grf_mode`, what `num_stages` drives), the
descriptor loads that become 2D block loads, and the kernel shapes its tutorials establish.
The preamble encodes the part of it a row-wise kernel needs:

- **Autotune over `warp_size` as well as `num_warps`.** `warp_size` is an Intel knob;
  Triton accepts it in a `Config` without the kernel taking it as a parameter. Read the
  candidates from the driver:
  ```python
  from triton.runtime import driver
  props = driver.active.utils.get_device_properties(torch.xpu.current_device())
  SUB_GROUP_SIZES = tuple(props.get("sub_group_sizes", (32,)))
  ```
- **Derive rows-per-program from `max_work_group_size`**, so a narrow row packs several
  rows into one program.
- **Put `num_stages` in the sweep.** On this backend it is the depth of a prefetch
  pipeline pass, not CUDA's async-copy pipeline; the measurement, not a carried value,
  decides it.

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

**Triton — the compiler hides spill behind a rebuild.** With `grf_mode='default'` the
backend builds with the small register file, reads the spill out of the binary, and on any
spill rebuilds with the large mode without saying so; the kernel runs correctly at halved
occupancy. After `kernel = fn.warmup(...)`, `kernel.metadata.build_flags` names the mode the
binary was built with; a large-GRF flag there is the spill report. Then the same choice as
for SYCL: a smaller tile or the large mode, measured separately.

**Triton — confirm the autotuner ran.** Clear Triton's cache
(`${TRITON_CACHE_DIR:-$HOME/.triton/cache}`) when in doubt, and check that the chosen
config differs between a small and a large workload.

## Step 5b: Search, do not settle for the first draft

`/wrap-kernel-for-tuning` is the loop: candidates are timed against the kernel a deployment
would otherwise run, correctness gates timing, and trials form a tree. It takes SYCL, oneDNN
and Triton candidates through `scripts/kernel_trials.py` and compiles them with the same
builder the benchmark uses, so a winner is already a Solution. When the harness came out of
the pipeline, the series is opened from its routing row (`--bound`, `--mechanism`) and a
pair the routing rejected is refused before anything is timed.

For an unattended sweep over work-group size and sub-group width on one model:

```bash
python scripts/optimize_model_kernels_xpu.py --model <hf_repo_id> --device xpu:0 \
    --output tmp/opt-<model_slug>          # required
```

It sweeps launch geometry only; a fusion or a removed materialisation is `/find-kernel-gaps`'s work.

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

For an authored kernel the bar is the op the harness calls -- the ATen kernel, the
decomposition's parts, the reference -- and `kernel_trials.py benchmark` is where it is
cleared; `flashinfer-bench run` enters once a definition exists (`authored_apply`).

A solution slower than the upstream kernel is not an optimization. Record it and say so.

## Step 7: Check the win survives the caller

Before claiming a deployment win, work through `architectures.md`, "Traps when deploying a
kernel through `apply()`", then run `/measure-serving-win` -- the plain arm for a kernel
bound at its call site, the overhead arm for one delivered through `apply()`; the routing
row names which.

## Step 8: Record

Traces from `--save-results` are the deliverable. Publishing them is `onboard-model-intel`,
"Benchmark, validate, publish", and `/submit-onboarding-prs`. A call-site delivery's deliverable is the kernel, the
diff that binds it, and the serving A/B that measured it.

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
bound, not proof the kernel is optimal. The routing does this arithmetic for you when a
`--pattern` is given; the authored-stream rate adds the third yardstick, what a kernel
written here reaches on a contiguous stream.

## Failure table

| Symptom | Fix |
| --- | --- |
| `cannot find -ldnnl` | `export FIB_ONEDNN_DIR=/opt/intel/oneapi/dnnl/latest` |
| Compile looks for `cuda_runtime_api.h` | Add `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`; see `xe-fuse.md` |
| Correct but many times too slow | Register spill — Step 5 (SYCL: unitrace; Triton: `metadata.build_flags`) |
| Triton solution refuses to run on XPU | Step 4b porting script |
| Same latency at every workload size | Triton autotuner not re-running — Step 5, clear the cache |
| No baseline in the results | Strict signature match failed — Step 6 |
| `UNSUPPORTED_DTYPE` | Step 1 dtype filter |
| `kernel_trials.py benchmark` prints `ROUTING: REJECTED` | The (candidate, mechanism) pair was priced out; change what the named gate reads and re-run `scripts/bound_candidates.py`, do not measure past it |
| Routing shows `attainable_calibrated ... authored_stream_gbs=None` | The authored-stream probe did not measure; `scripts/calibrate_part.py` on an idle box |

## Sources

- `flashinfer_bench.SYCL_PROMPT` — the SYCL solution contract
- `examples/sycl/` — worked kernels, one per pattern
- `flashinfer_bench/integration/intree_kernels.py` — canonical Intel Triton preamble
- `references/triton-xpu.md` — what the Intel Triton checkout says about writing kernels here
- `architectures.md` — per-part rules and `apply()` deployment traps
- `xe-matrix.md` — DPAS shapes, block-2D constraints, tile configs, schedulers, precisions
- `xe-fuse.md` — GEMM epilogue fusion
- `references/quantized-gemm.md` — FP8 / INT4 checkpoints: tune the config before writing a kernel
- `scripts/bound_candidates.py` — the routing: mechanisms, gates, `bound.json` fields
- `flashinfer_bench/device/calibration.py` — `authored_stream_probe()` and the rest of the part's measured costs
- `/wrap-kernel-for-tuning` — the harness contract and the trial loop
- `/optimize-onednn` — diagnosing and fixing the oneDNN GEMM call
- `onboard-model-intel/providers.md` — installing and verifying kernel providers
