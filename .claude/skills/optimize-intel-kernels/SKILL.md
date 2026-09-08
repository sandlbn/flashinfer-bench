---
name: optimize-intel-kernels
description: Write, validate and benchmark SYCL or Triton kernels for Intel GPUs against existing FlashInfer-Trace definitions. Use when optimizing a kernel for Intel (Battlemage, Crescent Island), porting a CUDA or Triton solution to run on XPU, or deciding which Intel kernels are worth working on.
---

# Optimize Intel kernels

Take a definition that already exists, write a solution that beats the kernel the serving
stack ships today, and prove it on hardware.

Definitions and Workloads are hardware-agnostic and are reused as-is — never re-collect them
for Intel. You produce two artifacts: a **Solution** (SYCL or Triton) and a **Trace**.

Steps 0-3 and 5-8 are the same whichever language you pick. Step 4 branches.

## Step 0: Prepare the host

```bash
powerprofilesctl set performance          # see docs/start/hardware-support.mdx
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

Measuring under a power-saving profile produces run-to-run variance larger than most effects
you are chasing. Do this first or the rest of the numbers are noise.

## Step 1: Pick a target

Run `/profile-intel` first. It ranks families by **recoverable** time -- share weighted
by the best speedup achieved on this hardware -- and routes each to the skill that owns it.
Ranking by share alone sends you to the family with the least headroom.

```bash
# With a model: profile it and rank op families by device time.
# Recipe and aggregation: onboard-model-intel/SKILL.md, Phase 4.

# Without a model: definitions that already have workloads attached are the ones you can
# benchmark today.
ls tmp/flashinfer-trace/workloads/*/ | head -40
```

Then filter:

- **Route by op_type** using the provider decision table in `onboard-model-intel/SKILL.md`
  Phase 5. GEMM-shaped work goes to oneDNN before anyone hand-writes a matmul.
- **Check the dtype exists on this part**: `caps.supported_dtypes` is native ∪ emulated, so
  ask `caps.is_native_dtype()` separately — Battlemage runs FP8 emulated, not natively,
  so an FP8 definition runs and passes here — at emulated throughput, which is the
  thing to check before reading its latency as a native FP8 number.
- **Prefer ops with an upstream baseline**, so Step 6 has something to beat.

Norm, activation and RoPE kernels are the best first targets: small, memory-bound, easy to
verify, present in every model.

## Step 2: Validate the reference on `xpu:0`

Never optimize against a reference that is wrong on the target — correctness is judged
against the reference *running on this device*.

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace \
    --device xpu:0 --definitions <name>
```

| Status | Meaning | Action |
| --- | --- | --- |
| `PASSED` | Reference agrees across devices | Continue |
| `UNSUPPORTED_DTYPE` | Part lacks a dtype the definition needs | Pick a different target |
| `MISMATCH` | The PyTorch reference itself differs on XPU | Report upstream; do not work around it |
| `NO_WORKLOAD` | Nothing to run it on | Attach a workload first — `onboard-model-intel` Phase 2 |

## Step 3: Choose the language

| Situation | Write |
| --- | --- |
| GEMM-shaped, or a GEMM with an elementwise epilogue | **oneDNN via SYCL** — post-ops fuse the epilogue into oneDNN's own GEMM (`/optimize-onednn`) |
| Memory-bound elementwise or row-wise (norms, activations, RoPE) | **SYCL** for the fastest result, **Triton** for portability |
| A CUDA Triton solution already exists for this definition | **Triton** — port it (Step 4b), which is the cheapest correct baseline |
| A fusion oneDNN post-ops cannot express | **Xe-Fuse** — see the criterion at the end |

**Coverage rule: elementwise and row-wise definitions should end up with both a SYCL and a
Triton solution.** SYCL wins on speed; Triton is the one that keeps working on a part whose
AOT target does not exist yet. `tests/integration/test_intree_kernels.py` enforces that both
exist for every in-tree signature — if you add one language, add the other.

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
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`.

Start from a worked example rather than a blank file:

| Want | Copy |
| --- | --- |
| Row-wise reduction | `examples/sycl/rmsnorm_sycl.cpp` |
| Two inputs, two outputs, in-place-safe | `examples/sycl/fused_add_rmsnorm_sycl.cpp` |
| GEMM + fused epilogue via oneDNN post-ops | `examples/sycl/onednn_gemm_swiglu.cpp` |
| GEMM epilogue via CUTLASS-SYCL | `examples/sycl/xefuse_gemm_swiglu.cpp` |

Apply these in order; each is worth more than the one after it.

1. **Vectorize the loads.** Sixteen bytes is the widest single access —
   `caps.vector_width(itemsize)` gives the element count (8 for bfloat16). Guard the wide
   path on `hidden % width == 0` and aligned base pointers, and keep a scalar fallback. The
   win is entirely in the bandwidth-bound regime, which is exactly where a scalar kernel
   loses to the upstream one.
2. **Pack multiple rows per work-group when rows are narrow.** See `architectures.md`,
   "Rows narrower than a sub-group" — it explains why a `(rows, sub_group)` shape lets each
   row reduce with a plain `reduce_over_group`.
3. **Do not hand-write a GEMM.** Try oneDNN or oneMKL first; hand-write only to fuse
   something they cannot.

Do not reach for a register cache to avoid re-reading a row: it measures within noise, and
vLLM's kernel re-reads too.

## Step 4b: Write a Triton solution

Triton runs on Intel and is half of what Intel ships. The kernel body usually ports
unchanged; the launch configuration and the device plumbing do not.

```json
{"spec": {"language": "triton", "target_hardware": ["xpu"],
          "entry_point": "main.py::run", "destination_passing_style": false}}
```

The canonical Intel pattern is `_TRITON_PREAMBLE` in
`flashinfer_bench/integration/intree_kernels.py` — copy it rather than reinventing it. It
encodes three things:

1. **Autotune over `warp_size` as well as `num_warps`.** `warp_size` is an Intel knob with
   no CUDA equivalent; the device reports `{16, 32}` and which wins is not predictable from
   the shape. Triton accepts it in a `Config` without the kernel taking it as a parameter.
   Read the candidates from the driver, never hardcode:
   ```python
   from triton.runtime import driver
   props = driver.active.utils.get_device_properties(torch.xpu.current_device())
   SUB_GROUP_SIZES = tuple(props.get("sub_group_sizes", (32,)))
   ```
2. **Derive rows-per-program from `max_work_group_size`**, so a row narrower than a
   work-group packs several rows into one program instead of leaving lanes idle. This is the
   Triton form of SYCL lever 2.
3. **`num_stages=2`.** It exists for CUDA's async-copy pipelining, which Intel has no direct
   equivalent for. 2 is at or near the optimum; 3 regresses sharply at small batch.

The optimum launch config **moves with shape**, so a single baked-in constant is wrong
somewhere — autotune with `key=["hidden"]` (or the equivalent size axis) rather than
choosing once.

### Porting a CUDA Triton solution

The kernel is almost never the problem. Every Triton solution in the dataset today refuses
on Intel in hand-written device management — a `torch.cuda.is_available()` guard, a
`.cuda()` call, or `torch.device("cuda")` allocation — before the kernel is reached.

```bash
python scripts/port_triton_solutions_to_xpu.py --local tmp/flashinfer-trace --definitions <name>
```

It rewrites those constructs to derive the device from the inputs and writes a **new**
solution under a separate author. Never edit the original: the dataset records which model
wrote which solution, and a solution that says "requires CUDA" is a faithful record of that.

**A ported kernel gives you correctness, not performance.** Retuning the launch config for
Intel is a separate step, and it is the axis that pays.

## Step 5: Build, and check the failure mode for your language

**SYCL — check for register spill before benchmarking anything.** Spilling does not fail
correctness; it produces a correct kernel that is many times too slow, and only the profiler
shows it.

```bash
unitrace -d -v -o prof python <script>.py   # script must end with
                                            # torch.xpu.current_stream().synchronize()
grep -A2 "Kernel Properties" prof.txt      # look for: Spill Memory Per Thread
```

Nonzero spill: apply **either** `FIB_SYCL_LARGE_GRF=1` **or** a smaller tile, measure both,
and never both at once — large GRF halves the threads resident per EU, so applying it to a
tile that already fits is worse than not.

**Triton — confirm the autotuner actually ran.** A stale Triton cache or a `key=` that omits
the varying axis will silently reuse one config for every shape. Clear
`~/.triton/cache` when in doubt, and check that the chosen config differs between a small
and a large workload.

## Unattended alternative: search instead of hand-tuning

`scripts/optimize_model_kernels_xpu.py` runs profile → extract → baseline → verify →
**search** → report for one model, unattended and with no API key. The search is over the
parameters that actually matter on Intel — work-group size and sub-group width — benchmarking
every candidate under the same correctness gate as a hand-written solution.

```bash
python scripts/optimize_model_kernels_xpu.py --model <hf_repo_id> --device xpu:0 \
    --output tmp/opt-<model_slug>          # required: definitions and report land here
```

Use it when the target is a shape-tuning problem rather than a structural one. It will not
find a fusion or remove a materialisation — for those see `/find-kernel-gaps` and Step 4.

## Step 6: Benchmark against the upstream kernel

Beating PyTorch eager is the easy bar and means nothing — eager makes several passes over
memory, so almost any fused kernel wins. The bar is the kernel the serving stack ships.

```bash
# In-tree baselines first: they need no provider.
flashinfer-bench add-baselines --local tmp/flashinfer-trace --in-tree --definitions <name>

# Then the vendor kernels, if installed.
flashinfer-bench add-baselines --local tmp/flashinfer-trace \
    --providers vllm-xpu,sgl-kernel-xpu --definitions <name>

# Prove the vendor kernels are actually built -- an import is not proof.
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0

flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

Always pass `--definitions`; without it `add-baselines` rewrites every baseline in the
dataset.

Correctness gates performance: a solution that fails numerically never gets a latency. The
trace records `environment.hardware_id` and `environment.libs.timing`, so results group by
device and are never ranked across devices.

**No baseline appeared?** Matching is strict — same `op_type`, same input and output names
in the same order. A near-miss produces no baseline rather than a confidently wrong
comparison. Check the signature in `flashinfer_bench/integration/xpu_kernels.py` (`REGISTRY`
is the live list; do not trust a list written in prose) and add an entry with the real
upstream signature. Verify semantics, not just arity — see the `xpu_kernels.py` module
docstring for how an exactly-matching signature can still compute different maths.

**A solution slower than the upstream kernel is not an optimization. Record it and say so.**

## Step 7: Check the win survives the caller

A kernel that wins here can still lose in a server. Before claiming a deployment win, work
through the traps in `architectures.md`, "Traps when deploying a kernel through `apply()`":
the bf16 tolerance gate, dtype-vs-name collision, verifying substitution actually happened,
declaring every output the caller consumes, matching the upstream kernel's allocation
behaviour, and the dispatch-cost floor.

## Step 8: Record

Traces from `--save-results` are the deliverable. Publishing them to the dataset is
`onboard-model-intel` Phase 7 and `/submit-onboarding-prs`.

## Quantized models: tune the config before writing a kernel

An FP8 or INT4 checkpoint reaches a *different* kernel from its bf16 sibling, and on Intel
the highest-value work is usually not writing one.

**Do not model a quantized linear as a dense GEMM in another dtype.** A checkpoint
declaring `activation_scheme: dynamic` -- every FP8 Qwen3 and Llama -- quantizes the
activation too, so the operation is W8A8: four inputs (`A_fp8`, `A_scale`, `B_fp8`,
`B_scale`), not two. A definition taking a bfloat16 activation describes an operation the
served model never performs, and no serving kernel can bind to it.

**vLLM's Triton kernels run on XPU as-is.** `w8a8_triton_block_scaled_mm` is portable
Triton with no CUDA guard, is what a served FP8 model actually executes, and is registered
as a baseline (`provider="vllm"`, distinct from `vllm-xpu`, whose kernels are SYCL). It
lives in the `vllm` distribution, so benchmark from an environment that has vLLM.

**It selects its tile shape from a per-device JSON file, and ships none for any Intel
part.** Without one it falls back to `BLOCK_SIZE_M=64, num_warps=4` and logs a warning
naming the file it wanted. At decode, where M is 1, that computes a 64-row tile to keep one
row.

```bash
python scripts/tune_vllm_fp8_config.py     --shape 4096,2560 --shape 9728,2560 --shape 2560,9728     --device xpu:0 --output tmp/fp8-configs
# --install also copies into vLLM's configs/ directory, where the kernel reads it
```

Tune every distinct `(N, K)` the model uses -- extract the definitions first and read them
off. The result is a JSON file, upstreamable as data, with no kernel written.

Measured on Arc B580 over the five projections of an FP8 Qwen3-4B, against the fallback:
**1.35x to 9.96x**, biggest at decode. Two patterns held everywhere and are worth knowing
before you sweep:

- **`num_warps=16` wins against the default 4**, in 13 of 15 shape/batch points. Not
  shape-dependent, and the single largest factor.
- **`BLOCK_SIZE_M` wants 16-32 at decode**, against the default 64.

`tools/vllm-fp8-configs/` holds the B580 files and the full table. Retune per part -- the
filename carries the device, and a B580 config says nothing about Crescent Island.

**Do not build a block-scaled GEMM out of oneDNN primitives.** oneDNN can express it, but
only by decomposing over K, and that pushes the accumulator through memory once per
K-block: measured bandwidth-bound and slower than naive PyTorch. Its `groups` scale
argument is also silently wrong on this hardware. `/optimize-onednn` carries the evidence
and the constraints.

## When to reach for Xe-Fuse

Only when the fusion cannot be expressed as oneDNN post-ops — post-ops act on a single
GEMM's output, so anything needing a lane shuffle (SwiGLU, GeGLU, RoPE on packed qkv) is out
of reach — **and** you are prepared to tile-search. Expect a GEMM deficit against oneDNN
that the fusion has to more than repay.

Read `xe-fuse.md` before writing one. Xe-Fuse is marked not-stable by IntelLabs: treat it as
one solution source, never a dependency.

**Read `xe-matrix.md` before writing any DPAS-backed kernel** — GEMM, attention, or
quantized matmul. It carries the DPAS shapes, the 2D-block-copy alignment constraints, the
tile configurations Intel ships, and which precisions this part actually has. Several rules
there contradict what a CUDA author would assume: sub-group 16 rather than the 32 the device
reports, no SLM double-buffering in any Xe mainloop, no pingpong schedule, and fp8 /
block-scaled mxfp4-mxfp8 present only on Crescent Island.

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| `cannot find -ldnnl` | oneDNN not discoverable | `export FIB_ONEDNN_DIR=/opt/intel/oneapi/dnnl/latest` |
| Compile looks for `cuda_runtime_api.h` | CUTLASS defaulted to the CUDA backend | Add `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`; see `xe-fuse.md` |
| Correct but many times too slow | Register spill | Step 5 |
| Triton solution refuses to run on XPU | Hand-written CUDA guard in the wrapper | Step 4b porting script |
| Same latency at every workload size | Triton autotuner not re-running | Step 5, clear the cache |
| No baseline in the results | Strict signature match failed | Step 6 |
| `UNSUPPORTED_DTYPE` | Part lacks that dtype | Step 1 dtype filter |

## Sources

- `flashinfer_bench.SYCL_PROMPT` — the SYCL solution contract
- `examples/sycl/` — worked kernels, one per pattern
- `flashinfer_bench/integration/intree_kernels.py` — canonical Intel Triton preamble
- `architectures.md` — per-part traps, design rules, `apply()` deployment traps
- `xe-matrix.md` — DPAS shapes, block-2D constraints, tile configs, Xe schedulers,
  which precisions Battlemage has. Read before any GEMM, attention or quantized kernel
- `xe-fuse.md` — GEMM epilogue fusion
- `/optimize-onednn` — diagnosing and fixing the oneDNN GEMM call
- `onboard-model-intel/providers.md` — installing and verifying kernel providers
