---
name: profile-intel
description: Profile a model on an Intel GPU, rank kernel families by recoverable device time, and route each to the skill that fixes it. Covers unitrace and torch.profiler usage on XPU. Run before any Intel optimization work.
---

# Profile and route

Answers one question: **what does this model spend device time in, and which skill fixes
that?**

## Run it

```bash
python scripts/profile_intel.py --model <hf_repo_id> --device xpu:0 --max-new-tokens 32
```

It warms up first, then reports families ranked by share, with the route and an advice line
for each:

```
  family           share        ms   calls  route
  <family>         <pct>    <ms>    <n>   /<skill>
```

## How to choose what to optimize

**Rank by recoverable time, not by share.**

```
recoverable = share x (1 - 1/speedup)
```

`speedup` is the best result achieved for that op_type **on this hardware**, read from the
trace dataset (filtered on the current device's `canonical_id`). The script computes it.

1. **Evidence must come from the same hardware.** An Intel speedup and an NVIDIA speedup
   are not the same quantity.
2. **"none yet" means measure, not skip.** A family with no evidence has unknown headroom;
   if its share is large, add a baseline and run the benchmark before optimizing anything.
3. **Sum the recoverable column before starting.** It bounds the whole exercise; if it is
   small, further kernel work has less upside than the dispatch layer, the scheduler, or
   memory traffic.
4. **Some wins cross families.** Fusing an activation into the producing GEMM's epilogue
   recovers *elementwise* time through the *gemm* route; read the advice line.

Re-profile after every change. A change that does not move a family's share did not do
anything, whatever the microbenchmark said.

## Before routing anywhere: check which backend is live

A family's share says where the time is, not whose kernel spends it. On Intel, vLLM selects
Flash Attention by default and Triton only as a fallback. Read the `Using ... backend` line
at startup, or run `scripts/observe_triton_kernels.py`, before spending trials. Once a
target is chosen, `/wrap-kernel-for-tuning` is how it gets optimized.

## The routing table

| Family | Route to | Because |
| --- | --- | --- |
| `gemm` | `/optimize-onednn` | oneDNN is already the path `F.linear` takes. The win is in how it is *called* — layout, primitive caching, post-op fusion, not blocking the host — never in beating its matmul |
| `attention` | `/onboard-model-intel` Phase 5 | `sgl-kernel-xpu` is the only source of Intel attention kernels; wire it as a baseline before writing anything |
| `norm` | `/optimize-intel-kernels` | memory-bound; vectorized loads and multi-row work-groups |
| `elementwise` | `/optimize-intel-kernels` | **try fusing into the producing GEMM's epilogue first** (`/optimize-onednn` Fix 3) |
| `rope` | `/optimize-intel-kernels` | `vllm-xpu-kernels` ships `rotary_embedding`; benchmark against it first |
| `sampling` | `/onboard-model-intel` Phase 5 | `sgl-kernel-xpu` has the family; verify each op is actually built |
| `data movement` | no kernel to write | a large copy or concat is an avoidable materialisation; fix the layout or fuse the producer |

A `transformers` profile cannot rank attention or KV-cache the way a serving stack would —
it is not paging a KV cache. Profile vLLM-XPU or SGLang-XPU for those.

## Two profilers, and which to use

### unitrace — per-kernel timing, spill, SIMD and GRF

```bash
export PATH="<your pti-gpu checkout>/tools/unitrace/build:$PATH"
unitrace -d -v -o prof python your_script.py      # -v splits by launch shape
```

Output lands in `prof.<pid>`: per-kernel rows plus the Kernel Properties table, which
reports **register spill** (`Spill Memory Per Thread`) and register file size — nothing else
exposes spill.

**The script must end with a sync that unitrace hooks, or the report is empty:**

```python
torch.xpu.current_stream().synchronize()   # zeCommandListHostSynchronize -- hooked
# NOT torch.xpu.synchronize()              # zeDeviceSynchronize -- not hooked
```

An event sync (`ev = torch.xpu.Event(); ev.record(); ev.synchronize()`) or any host
read-back (`float(y[0,0])`) also works. One sync at the very end is enough. An empty report
contains only a `Device Timing Summary` total.

Add `--chrome-kernel-logging` for a Perfetto timeline in `<script>.<pid>.json`. Hardware
metrics (`-q`, `-k`, `--stall-sampling`) need `sudo sysctl dev.xe.observation_paranoid=0`.

### torch.profiler — what `scripts/profile_intel.py` uses

`ProfilerActivity.XPU`, no build step, no sync requirement, and it maps kernels to the
`aten` ops that launched them, which is what makes family routing possible. It does not
report spill, SIMD or GRF.

Use `profile_intel.py` to decide *what* to optimize, unitrace on a kernel you wrote to check
it is not spilling.

## A family that says "other"

An unmatched kernel means the route table has a gap. Kernel names are templated C++ — a
norm appears as `ReduceKernel<1, ReduceOp<float>>`. Read the name, add a pattern to `ROUTES`
in `scripts/profile_intel.py`, and re-run.
