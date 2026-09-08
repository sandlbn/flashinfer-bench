---
name: profile-intel
description: Profile a model on an Intel GPU, rank kernel families by share of device time, and route each to the skill that fixes it. Also covers installing unitrace and what it is and is not good for. Run this before any Intel optimization work — it decides what is worth optimizing.
---

# Profile and route

Optimizing without a profile is guessing. A 3x win on 2% of device time is worth less than a
1.15x win on 50%, and which is which is not predictable from reading the model.

This skill answers one question: **what does this model actually spend device time in, and
which skill fixes that?**

## Run it

```bash
python scripts/profile_intel.py --model <hf_repo_id> --device xpu:0 --max-new-tokens 32
```

It warms up first (so allocation and first-call compilation are not profiled), then reports
families ranked by share, with the route for each. Example, Qwen3.5-4B on Arc B580:

```
  family           share        ms   calls  route
  gemm             77.5%   492.75    6192  /optimize-onednn
  elementwise      16.3%   103.54   58336  /optimize-intel-kernels
  norm              4.0%    25.39    6361  /optimize-intel-kernels
  data movement     1.2%     7.68    1852  (no kernel to write)
```

## How to choose what to optimize

**Rank by recoverable time, not by share.** The script computes it for you; this is the
reasoning behind the number, which you need in order to argue with it.

```
recoverable = share x (1 - 1/speedup)
```

`speedup` is the best result anyone has actually achieved for that op_type **on this
hardware**, read from the trace dataset. Worked example, Qwen3.5-4B on Arc B580:

| family | share | evidence | recoverable |
| --- | --- | --- | --- |
| gemm | 77.5% | 1.08x (n=47) | **5.6%** |
| elementwise | 16.3% | 1.11x (n=12) | 1.6% |
| norm | 4.0% | 1.09x (n=131) | 0.3% |

GEMM is three-quarters of device time and still the smaller opportunity than its share
suggests, because oneDNN already runs at ~90% of the part's peak. Ranking by share would
have sent an agent to spend a week on the one family with the least headroom.

Four rules that follow:

1. **Evidence must come from the same hardware.** Traces carry `hardware_id`; an Intel
   speedup and an NVIDIA speedup are not the same quantity. Mixing them here produced a
   confident "0.42x for paged attention" that was pure cross-device noise. The script
   filters on the current device's `canonical_id`.
2. **"none yet" means measure, not skip.** A family with no evidence has unknown headroom.
   If its share is large, *establishing* the evidence — add a baseline, run the benchmark —
   is the highest-value next action, ahead of optimizing a family you already understand.
3. **Sum the recoverable column before starting.** It bounds the whole exercise. Here it is
   ~7.5%: this model is close to what the stack can currently do, and further kernel work
   has less upside than the dispatch layer, the scheduler, or memory traffic. Knowing when
   to stop is the point of measuring.
4. **Some wins cross families.** Fusing an activation into the producing GEMM's epilogue
   recovers *elementwise* time through the *gemm* route. The table cannot express that, so
   read the advice line under each family, not just the number.

Re-profile after every change. A change that does not move a family's share did not do
anything, whatever the microbenchmark said — that is how a 2-5x kernel win was found to be
worth nothing end to end, and how a "structural" MLP loss turned out to be one blocking call.

## The routing table

| Family | Route to | Because |
| --- | --- | --- |
| `gemm` | `/optimize-onednn` | oneDNN is already the path `F.linear` takes and runs at ~90% of the part's peak. The win is in how it is *called* — layout, primitive caching, post-op fusion, and not blocking the host per call — never in beating its matmul |
| `attention` | `/onboard-model-intel` Phase 5 | `sgl-kernel-xpu` is the only source of Intel attention kernels; wire it as a baseline before writing anything |
| `norm` | `/optimize-intel-kernels` | Memory-bound and genuinely winnable: vectorized loads and multi-row work-groups beat the vendor kernels here |
| `elementwise` | `/optimize-intel-kernels` | **Try fusing into the producing GEMM's epilogue first** (`/optimize-onednn` Fix 3). A separate kernel is the fallback, not the first move |
| `rope` | `/optimize-intel-kernels` | `vllm-xpu-kernels` ships `rotary_embedding`; benchmark against it before writing one |
| `sampling` | `/onboard-model-intel` Phase 5 | `sgl-kernel-xpu` has the family, but verify each op is actually built — a signature does not prove it |
| `data movement` | no kernel to write | A copy or concat that large means an avoidable materialisation; fix it with a layout or fusion change |

Two rules the ranking does not encode:

- **Rank by recoverable time, not share alone** — `share x (1 - 1/speedup)`, as above.
  GEMM at ~78% routes to a library
  already near hardware peak, so its realistic upside is fusion, not the matmul. A norm at 4%
  where the vendor kernel loses by 1.3x may be the easier win.
- **A `transformers` profile cannot rank attention or KV-cache** the way a serving stack
  would — it shows near-zero attention because it is not paging a KV cache. For those,
  profile vLLM-XPU or SGLang-XPU instead.

## Two profilers, and which to use

### unitrace — per-kernel timing, spill, SIMD and GRF

```bash
export PATH="<your pti-gpu checkout>/tools/unitrace/build:$PATH"
unitrace -d -v -o prof python your_script.py      # -v splits by launch shape
```

Output lands in `prof.<pid>`. It gives per-kernel rows *and* the Kernel Properties table —
including **register spill**, which nothing else exposes:

```
Kernel, Calls, Time (ns), Time (%), Average (ns), ...
"gemm_kernel[SIMD16 {20;1;1} {64;8;1}]", 20, 7443743, 61.83, 372187, ...

Kernel, Compiled, SIMD, ..., Spill Memory Per Thread, Register File Size Per Thread
"gemm_kernel[SIMD16 {20;1;1} {64;8;1}]", AOT, 16, ..., 0, 256
```

**The script must end with a sync that unitrace hooks, or you get an empty report.**

```python
torch.xpu.current_stream().synchronize()   # zeCommandListHostSynchronize -- hooked
# NOT torch.xpu.synchronize()              # zeDeviceSynchronize -- not hooked
```

This is the whole trick, and it costs an afternoon if you do not know it. unitrace harvests
kernel records inside hooked host-sync callbacks; `zeDeviceSynchronize` is not among them
(nothing in `pti-gpu/tools/unitrace/src/levelzero/` hooks it). Pending records then sit in
thread-local storage whose destructor runs *before* the library's exit flush, so they are
discarded and you get a 98-byte file with only:

```
=== Device Timing Summary ===
        Total Execution Time (ns):   1008344251
```

An event sync (`ev = torch.xpu.Event(); ev.record(); ev.synchronize()`) or any host
read-back (`float(y[0,0])`) works equally well. One sync at the very end is enough.

Add `--chrome-kernel-logging` for a Perfetto timeline in `<script>.<pid>.json`.

Hardware metrics (`-q`, `-k`, `--stall-sampling`) additionally need
`sudo sysctl dev.xe.observation_paranoid=0`; without it they fail with
"Failed to initialize Level Zero runtime".

### torch.profiler — what `scripts/profile_intel.py` uses

`ProfilerActivity.XPU`, no build step, no sync requirement, and it maps kernels to the
`aten` ops that launched them, which is what makes family routing possible. It does **not**
report spill, SIMD or GRF.

**Use both:** `profile_intel.py` to decide *what* to optimize, unitrace on a kernel you wrote
to check it is not spilling.

## Interpreting a family that says "other"

An unmatched kernel means the route table has a gap, not that the kernel is unimportant.
Kernel names are templated C++ and rarely contain the obvious word — a norm appears as
`ReduceKernel<1, ReduceOp<float>>`, not as "norm". Read the name, add a pattern to `ROUTES`
in `scripts/profile_intel.py`, and re-run.

## After routing

Take the highest-share family, follow its skill, and re-profile. The profile is the
before/after measurement as well as the plan — a change that does not move a family's share
did not do anything, whatever the microbenchmark said.
