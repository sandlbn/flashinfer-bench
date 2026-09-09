---
name: optimize-intel-kernels
description: Write, validate and benchmark a SYCL or Triton kernel on an Intel GPU -- for an existing FlashInfer-Trace definition, for a CUDA Triton solution being ported to XPU, or for an op the routing found nothing implements well (an authored_callsite / authored_apply row of scripts/bound_candidates.py). The language is chosen from the term that formed the candidate's bound, not from the op's name. Use when authoring or tuning a kernel for an Intel part.
---

# Write a kernel for an Intel part

Take an op that runs on the device -- a definition's kernel, a CUDA Triton solution to be
ported, or an op the routing says nothing implements well -- write a kernel that beats what
the serving stack runs today, and prove it on hardware. Definitions and workloads are
hardware-agnostic and are reused as-is. You produce a **Solution** (SYCL or Triton) and a
**Trace**, or, for a call-site delivery, the kernel and the diff that binds it.

Searching the space of variants is the trial loop (`/wrap-kernel-for-tuning`); this skill is
what to write, in which language, and what has to be true before a number from it counts.

---

# Obtain

## Preflight: the host

```bash
powerprofilesctl set performance          # measure under a performance profile, always
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
flashinfer-bench providers list
python scripts/calibrate_part.py          # the part's measured costs, including the authored-stream rate
```

| Check fails | Do this |
| --- | --- |
| `list_devices()` has no `xpu:0` | Driver or torch-xpu problem -- `docs/start/hardware-support.mdx`. Stop. |
| `SyclBuilder.is_available()` is False | Source the oneAPI environment script, or set `FIB_SYCL_COMPILER` to the `icpx` the compiler package installed |
| A provider you need is absent | `flashinfer-bench providers install <name>` -- see `onboard-model-intel/providers.md` |
| Mixed-vendor host picks the wrong backend | `export FIB_DEVICE_BACKEND=xpu` |
| `calibrate_part.py` prints `authored stream unavailable` | Triton has no backend for this device, or the rounds did not settle on a shared box; retry idle. Until it measures, the routing prices no authored kernel |

## The target, and what its record already carries

A target is a routed candidate, not a name you chose. `/discover-model-kernels` walks the
model to the worklist; `/profile-intel` ranks families and routes each when there is no
worklist yet.

Where `scripts/bound_candidates.py` has run, `worklist.json` is the target list: each row
names the candidate, the mechanism that admitted it, and the ceiling it was priced to.
`bound.json` carries, per candidate, the terms its bound was formed from -- `t_dev_us`,
`bytes_min`, `t_mem_us`, `t_mem_pattern_us`, `t_mem_authored_us`, `t_cmp_us`,
`launch_floor_us`, `spread_us` -- and the `regime` the classifier gave it. Read them; they
are what "Read" below reasons from, and re-measuring what the routing measured wastes a
GPU session.

Without a routing, definitions that already have workloads are what can be benchmarked
today:

```bash
ls tmp/flashinfer-trace/workloads/*/ | head -40
```

and the same quantities have to be established by hand: `/wrap-kernel-for-tuning` for the
harness, then `scripts/bound_candidates.py --measure` for its streaming device time and
spread.

Two checks on any target before effort goes into it:

- **Is the dtype native on this part?** `caps.supported_dtypes` is native together with
  emulated, so ask `caps.is_native_dtype()` separately. An emulated dtype runs and passes at
  emulated throughput, and its latency is not a result about the format.
- **Is this a quantized checkpoint?** Those reach different kernels from their bf16
  siblings; `references/quantized-gemm.md` is what to read before writing anything for one.

## Entering from the routing: an op nothing implements well

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
`apply()` is not a route for this candidate on this part. The illustration "Two deliveries
of the same authored kernel" below is one run of exactly this arithmetic.

## Validate the reference on `xpu:0`

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
| `NO_WORKLOAD` | Attach a workload first -- `onboard-model-intel`, "Acquire definitions" |

For an authored kernel with no definition yet, the harness's own op is the reference, and
`scripts/kernel_trials.py benchmark` checks a trial against it before it times anything.

---

# Read

Classify the candidate before writing a line: the rows, their inputs and the field names to
record are `optimize-model-kernels/references/read-the-numbers.md`. A routed candidate
already carries its row in `bound.json`.

## Choose the language from the candidate's measured regime

The term that forms the bound decides what a kernel has to do, and that decides the language
question and the measurement that settles it.

| Term forming the bound | What a kernel has to do | The language question, and what settles it |
| --- | --- | --- |
| `t_cmp_us` | keep the matrix engine fed | The library call is the bar (`/optimize-onednn`); author only for a fusion the library cannot express. Triton `tl.dot` through descriptor loads and SYCL-TLA (`xe-matrix.md`) both reach the matrix unit; run one draft of each through the same trial series and keep the measured one |
| `t_mem_pattern_us` or `t_mem_authored_us` | move `bytes_min` once, at the rate the part gives | Compare `calibration.authored_stream_probe()["gbs"]` (a plain Triton kernel) with `calibration.get().bandwidth_gbs` (the framework's own stream). Apply the difference to the candidate's `bytes_min`: when that time is smaller than the row's headroom, the plain Triton kernel can deliver the ceiling; when it is not, the SYCL kernel with an explicit vector width (`caps.vector_width(itemsize)`) is the draft that has to be measured against it. Row-wise definitions ship both regardless -- `tests/integration/test_intree_kernels.py` requires it |
| `launch_floor_us` | remove launches, not shorten them | No language wins; the routing rejected authoring at `headroom` and the route is a fusion row (`fusion_callsite`, `fusion_apply`) or a call-site change |
| `t_mem_pattern_us` well above `t_mem_us` | change the data's layout, not the kernel | The `layout_transform` row; `architectures.md`, "A row pitch on the memory-channel period" |
| A CUDA Triton solution exists for the definition | port it | Triton -- "Porting a CUDA Triton solution" below; a port gives correctness, and the launch configuration is retuned here |

## Measure what the access pattern allows before rewriting a memory-bound kernel

Peak bandwidth is the wrong yardstick. Time three reads with the accelerator's timer: the
whole tensor contiguously, the slice in the shape the kernel is obliged to touch, and the
kernel itself. If the kernel matches the strided read, the gap to peak lives in the data
layout -- a serving-stack decision, not a kernel one. Treat the strided figure as a lower
bound, not proof the kernel is optimal. The routing does this arithmetic when a `--pattern`
is given; the authored-stream rate adds the third yardstick, what a kernel written here
reaches on a contiguous stream.

This probe is what supplies `bw_pattern` to the classification.

---

# Generate

What a draft varies comes from the row the candidate is in and the principles in
`optimize-model-kernels/references/mechanisms.md`, each derived per case. What follows is
the contract each language imposes and the knobs each exposes -- facts about the toolchain,
not a ranking of changes.

Worked patterns for this hardware, with the guard each carries, are indexed in
`references/xe-forge-knowledge.md`: the vendor's own corpus under `tmp/Xe-Forge/`, read for
its mechanisms and its API constraints, never for its speedup fields or its shape tables.
That file also lists where the corpus and this part's own measurements disagree.

## Write a SYCL solution

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

The device queries a SYCL draft is written against, each settled by the trial loop rather
than by belief:

- **Load width.** `caps.vector_width(itemsize)` gives the element count per widest access.
  Guard the wide path on `hidden % width == 0` and aligned base pointers, and keep a scalar
  fallback.
- **Sub-group width.** `caps.preferred_sub_group_size`, and the widths the part accepts.
- **Rows per work-group when rows are narrow** -- `architectures.md`, "Rows narrower than a
  sub-group".
- **Whether to write a GEMM at all.** A library call is the bar for a dense GEMM; a
  hand-written one has to be measured against it (`/compare-implementations`), and the case
  for writing one is a fusion the library cannot express.

Before writing a matrix kernel from scratch, read what the CUTLASS-SYCL framework already
offers -- mainloop dispatch policies, tile schedulers, and the epilogue fusion catalog are
inventoried in `sycl/xpu/cutlass_sycl_framework.yaml` of the corpus, and the DPAS, shared-
local-memory and sub-group-reduction patterns in `sycl/xpu/xetla_patterns.yaml`
(`references/xe-forge-knowledge.md` says which entries are usable and which are that
corpus's own conclusions). What the part itself constrains is `xe-matrix.md`.

## Write a Triton solution

```json
{"spec": {"language": "triton", "target_hardware": ["xpu"],
          "entry_point": "main.py::run", "destination_passing_style": false}}
```

Copy `_TRITON_PREAMBLE` from `flashinfer_bench/integration/intree_kernels.py`, and read
`references/triton-xpu.md` before departing from it: it is what the Intel backend's own
checkout says about writing kernels here -- the block programming model, the launch options
that exist only on this backend (`warp_size`, `grf_mode`, what `num_stages` drives), the
descriptor loads that become 2D block loads, and the kernel shapes its tutorials establish.

Source: `references/triton-xpu.md`, from the Intel Triton backend's own checkout

| Launch option the Intel backend exposes | What it is | How its value is settled |
| --- | --- | --- |
| `warp_size` | an Intel knob Triton accepts in a `Config` without the kernel taking it as a parameter | read the candidates from the driver, sweep them: `driver.active.utils.get_device_properties(torch.xpu.current_device())["sub_group_sizes"]` |
| `num_warps` | as elsewhere | swept with `warp_size`, since the two interact |
| rows per program | derived from `max_work_group_size`, so a narrow row packs several rows into one program | swept against the row width the definition has |
| `num_stages` | on this backend the depth of a prefetch pipeline pass, not CUDA's async-copy pipeline -- a value carried from CUDA is a value for a mechanism this backend does not have | sweep it; the measurement decides |
| `grf_mode` | which register file the binary is built with | see "Build, and check the failure mode for your language" |

```python
from triton.runtime import driver
props = driver.active.utils.get_device_properties(torch.xpu.current_device())
SUB_GROUP_SIZES = tuple(props.get("sub_group_sizes", (32,)))
```

The backend also refuses, or silently mis-compiles, a handful of constructions. They are a
closed set because their subject is an API: a parameter supplied by `triton.autotune`
carries no default in the signature; the register-file mode is a compiler option declared as
a `tl.constexpr`, not an ordinary `Config` value; a flattened program id for tile swizzling
needs a one-dimensional grid; `boundary_check` takes dimension indices rather than booleans;
block pointers and tensor descriptors are different APIs and neither supports an atomic
store, so atomic accumulation computes its pointers by hand and writes into a pre-zeroed,
masked output; a packed transpose carries its own strides; batch and stride products cast to
64-bit before they overflow a pointer; and an alignment hint applies to an offset tensor,
where a false hint is wrong code rather than slow code. Each is one entry, with a
before/after, in `references/xe-forge-knowledge.md`, "Constraints worth carrying".

Autotune with `key=["hidden"]` (or the equivalent size axis); the optimum moves with shape,
so a config chosen at one size is not a config for another.

### Porting a CUDA Triton solution

Triton solutions in the dataset refuse on Intel in their device management -- a
`torch.cuda.is_available()` guard, a `.cuda()` call, or `torch.device("cuda")` -- not in the
kernel.

```bash
python scripts/port_triton_solutions_to_xpu.py --local tmp/flashinfer-trace --definitions <name>
```

It rewrites those constructs to derive the device from the inputs and writes a **new**
solution under a separate author. Never edit the original. A ported kernel gives you
correctness; retune the launch config for Intel separately.

## Build, and check the failure mode for your language

**SYCL -- read the spill before benchmarking.** Spill does not fail correctness; it produces
a correct kernel with traffic that `bytes_min` does not account for, which is why the
`spill-limited` row of `optimize-model-kernels/references/read-the-numbers.md` blocks
every bound comparison below it.
`kernel_trials.py benchmark` reports `SPILLS` from the build log it captured. Where no
compiler ran in view it reports `unknown`, and unitrace is where the number is:

```bash
unitrace -d -v -o prof python <script>.py   # script must end with
                                            # torch.xpu.current_stream().synchronize()
grep -A2 "Kernel Properties" prof.txt      # look for: Spill Memory Per Thread
```

**Triton -- the compiler hides spill behind a rebuild.** With `grf_mode='default'` the
backend builds with the small register file, reads the spill out of the binary, and on any
spill rebuilds with the large mode without saying so; the kernel runs correctly with fewer
threads resident. After `kernel = fn.warmup(...)`, `kernel.metadata.build_flags` names the
mode the binary was built with, and a large-GRF flag there is the spill report.

Spill is a register-pressure result, and the principle is to reduce register pressure. The
candidates that exist include a smaller tile, a shorter unroll, the larger register file
(`FIB_SYCL_LARGE_GRF=1` for SYCL, `grf_mode` for Triton) which trades threads resident per
execution unit for registers, and -- before any of those change the shape of the problem --
shortening live ranges: prefetch early and load close to the use, reload an operand near a
distant second use instead of holding it across a loop, and prefer an extra load to a spill.
That last one has a guard: it assumes the reloaded data survives in cache, and with a
working set that evicts it converts cache hits into global traffic, so the reload trial is
measured against the held-value trial. The corpus entries behind these, with before/after
code, are indexed under "Generators it adds" in `references/xe-forge-knowledge.md`.

Each candidate is one trial, changed alone, with `SPILLS` and the verdict read per trial --
the trade goes different ways for different tiles, so it is measured, not decided.

**Triton -- confirm the configuration you swept is the one that ran.** The cache can return
an earlier configuration for a kernel whose autotune key did not change, so clear it
(`${TRITON_CACHE_DIR:-$HOME/.triton/cache}`) when in doubt and check that the chosen config
differs between a small and a large workload. On a kernel containing a matrix multiply,
check `kernel.metadata` before believing a sub-group sweep at all: `xe-matrix.md`,
"Sub-group width is 16 for matrix work", records that the backend overrides the requested
width there, so the sweep compiles one kernel several times.

## When a fusion needs a generated epilogue kernel

A fusion the library's post-ops can express stays inside the library the stack already calls
(`/optimize-onednn`, "Post-ops, and the shape of what they can express"). Post-ops act on a
single GEMM's output, so a fusion needing a lane shuffle -- a gated activation, RoPE on
packed qkv -- is not expressible as one and needs a generated CUTLASS-SYCL kernel instead.
Read `xe-fuse.md` before generating one: it carries the build flags, the operand-layout
traps that produce silently wrong output, and the tile parameter that must always be passed
explicitly. It is one solution source, never a dependency; its upstream marks it not stable,
so a break there loses one contender and blocks nothing.

**Read `xe-matrix.md` before writing any kernel that targets the matrix unit** -- GEMM,
attention, or quantized matmul. Several constraints there contradict what a CUDA author
would assume, and each carries the source file it came from.

## Search, do not settle for the first draft

`/wrap-kernel-for-tuning` is the loop: candidates are timed against the kernel a deployment
would otherwise run, correctness gates timing, and trials form a tree. It takes SYCL, oneDNN
and Triton candidates through `scripts/kernel_trials.py` and compiles them with the same
builder the benchmark uses, so a winner is already a Solution. When the harness came out of
the pipeline, the series is opened from its routing row (`--bound`, `--mechanism`) and a
pair the routing rejected is refused before anything is timed.

For an unattended sweep over launch geometry on one model:

```bash
python scripts/optimize_model_kernels_xpu.py --model <hf_repo_id> --device xpu:0 \
    --output tmp/opt-<model_slug>          # required
```

Read its swept space out of the script before relying on it: it sweeps launch geometry only,
from lists in its own source rather than from the device's `max_work_group_size` and
`sub_group_sizes`, so a part whose legal values differ is swept over the wrong set. A fusion
or a removed materialisation is `/find-kernel-gaps`'s work, not this script's.

---

# Tools

| Question | Instrument |
| --- | --- |
| did this draft win against what the stack runs | `scripts/kernel_trials.py benchmark`, and nothing else |
| does it spill | the `SPILLS` key on every build; unitrace Kernel Properties when it reads `unknown`; `metadata.build_flags` for Triton |
| what geometry launched, and was it the kernel I meant | unitrace `-d -v` |
| why is it short of its bound when bytes and flops say otherwise | VTune counters -- `python -m flashinfer_bench.agents.vtune --check` first |
| what does this part offer | `get_accelerator(dev).capabilities(dev)`; `architectures.md` says what to query and what each answer constrains |
| what does this part cost | `python scripts/calibrate_part.py` |
| is an alternative implementation competitive for this op class at all | `/compare-implementations` |

The full index, with the trust conditions for each, is
`optimize-model-kernels/references/tools.md`.

---

# Gates

Everything in `optimize-model-kernels/references/gates.md` applies. Three matter most here.

## The bar is the kernel the stack ships, not PyTorch eager

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

**No baseline appeared?** Matching is strict -- same `op_type`, same input and output names
in the same order. Check `REGISTRY` in `flashinfer_bench/integration/xpu_kernels.py` and add
an entry with the real upstream signature; verify semantics, not just arity.

For an authored kernel the bar is the op the harness calls -- the ATen kernel, the
decomposition's parts, the reference -- and `kernel_trials.py benchmark` is where it is
cleared; `flashinfer-bench run` enters once a definition exists (`authored_apply`).

A solution that does not beat the kernel the stack ships is not an optimization. Record it
and say so.

## The win has to survive the caller

Before claiming a deployment win, work through `architectures.md`, "Traps when deploying a
kernel through `apply()`", then run `/measure-serving-win` -- the plain arm for a kernel
bound at its call site, the overhead arm for one delivered through `apply()`; the routing
row names which.

## Spill has to be zero before a bound comparison means anything

Stated above under building; it is a gate because the `spill-limited` row makes every row
below it unevaluable, so a bound quoted for a spilling kernel is not a bound.

---

# Deliver and record

Traces from `--save-results` are the deliverable. Publishing them is `onboard-model-intel`,
"Benchmark, validate, publish", and `/submit-onboarding-prs`. A call-site delivery's
deliverable is the kernel, the diff that binds it, and the serving A/B that measured it.

---

# Illustrations

## Two deliveries of the same authored kernel

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

---

# Failure table

| Symptom | Fix |
| --- | --- |
| `cannot find -ldnnl` | declare `onednn` in the solution's dependencies and point `FIB_ONEDNN_DIR` at the prefix that holds it |
| Compile looks for `cuda_runtime_api.h` | Add `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`; see `xe-fuse.md` |
| Correct but many times too slow | Register spill -- "Build, and check the failure mode for your language" |
| Triton solution refuses to run on XPU | "Porting a CUDA Triton solution" |
| Same latency at every workload size | Triton autotuner not re-running -- clear the cache, as that same section says |
| No baseline in the results | Strict signature match failed -- "The bar is the kernel the stack ships" |
| `UNSUPPORTED_DTYPE` | the native-dtype check under "The target, and what its record already carries" |
| `kernel_trials.py benchmark` prints `ROUTING: REJECTED` | The (candidate, mechanism) pair was priced out; change what the named gate reads and re-run `scripts/bound_candidates.py`, do not measure past it |
| Routing shows `attainable_calibrated ... authored_stream_gbs=None` | The authored-stream probe did not measure; `scripts/calibrate_part.py` on an idle box |

---

# Sources

- `flashinfer_bench.SYCL_PROMPT` -- the SYCL solution contract
- `examples/sycl/` -- worked kernels, one per pattern
- `flashinfer_bench/integration/intree_kernels.py` -- canonical Intel Triton preamble
- `references/triton-xpu.md` -- what the Intel Triton checkout says about writing kernels here
- `architectures.md` -- what to query per part, and the `apply()` deployment traps
- `xe-matrix.md` -- matrix-unit shapes, block-load constraints, tile parameters, precisions
- `xe-fuse.md` -- GEMM epilogue fusion through a generated kernel
- `references/quantized-gemm.md` -- FP8 / INT4 checkpoints: tune the config before writing a kernel
- `references/xe-forge-knowledge.md` -- the vendor's pattern corpus: what to take from it, what to leave, and where it disagrees with this part
- `scripts/bound_candidates.py` -- the routing: mechanisms, gates, `bound.json` fields
- `flashinfer_bench/device/calibration.py` -- `authored_stream_probe()` and the rest of the part's measured costs
- `optimize-model-kernels/references/read-the-numbers.md` -- the classification the language decision reads
- `optimize-model-kernels/references/mechanisms.md` -- what each row admits as a change
- `optimize-model-kernels/references/gates.md` -- the contract a trial prints, and every gate
- `/wrap-kernel-for-tuning` -- the harness contract and the trial loop
- `/optimize-onednn` -- diagnosing and fixing the library GEMM call
- `onboard-model-intel/providers.md` -- installing and verifying kernel providers
