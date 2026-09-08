---
name: optimize-onednn
description: Diagnose and fix a slow oneDNN GEMM on Intel GPUs — read ONEDNN_VERBOSE and dispatch output, resolve a rejection to the gate in oneDNN's source, and apply the four call-level fixes (weight layout, primitive caching, post-op fusion, no per-call host block). Use when a profile says GEMM, when F.linear is slow on XPU, or before concluding a library is at fault.
---

# Optimize a oneDNN GEMM

`F.linear` and `torch.matmul` on XPU **are** oneDNN matmuls — you cannot opt out, so you tune
the call. When a profile says GEMM, this is where the answer is.

**Do not try to beat oneDNN's matmul.** Measured on Arc B580 it reaches ~100 TFLOP/s bf16,
about 90% of the part's XMX peak. Every win available here is in *how it is called*: the four
fixes below moved plain GEMM from 0.36x to 1.01-1.66x against `torch.matmul`, and a fused
SwiGLU from 0.44x to 3.11x against its reference.

Work in order — the steps are diagnosis, and applying a fix without them is guessing:

| | | |
|---|---|---|
| 1-3 | Repro, observe, read dispatch | what ran, and what was rejected |
| Fix 1 | Weight layout | when dispatch says `unsupported format tag` |
| Fix 2 | Primitive caching | when `create:` appears in a steady-state loop |
| Fix 3 | Post-op fusion | when an elementwise op follows the GEMM |
| Fix 4 | Remove the per-call host block | **check this first if a oneDNN solution is slower than torch** |
| Fix 5 | Tile/strategy catalog entry | the call is clean and the GEMM is still slow (requires rebuilding oneDNN) |

## Step 1: Build a repro

Everything below greps this script. The definition and workload already describe the problem
exactly, so the repro is short:

```python
# repro.py
import torch
from flashinfer_bench.data import TraceSet

NAME = "<definition_name>"
ts = TraceSet.from_path("tmp/flashinfer-trace")
d  = ts.definitions[NAME]
wl = ts.workloads[NAME][0].workload          # first recorded shape

inputs = [torch.randn(*s, dtype=dt, device="xpu:0")
          for s, dt in zip(d.get_input_shapes(wl.axes), d.torch_input_dtypes)]
ns = {"torch": torch}
exec(d.reference, ns)
for _ in range(3):                            # steady state, past primitive creation
    ns["run"](*inputs)
torch.xpu.synchronize()
```

## Step 2: Observe with ONEDNN_VERBOSE

oneDNN's own variable, not a flashinfer-bench one. Prints to stderr; values compose
(`ONEDNN_VERBOSE=dispatch,exec`).

| Value | What you get | When |
|---|---|---|
| `1` | one `primitive,exec` line per execution | always, first |
| `dispatch` | one line per **rejected** implementation, with reason and source location | when the chosen implementation looks wrong |
| `profile,exec` | adds `create:cache_hit` lines | when you suspect per-call primitive rebuild |
| `all` | everything | last resort; very noisy |

```
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,
  src:f16::blocked:ab::f0 wei:f16::blocked:ba::f0 dst:f16::blocked:ab::f0,
  attr-scratchpad:user,,64x896:896x4864,0.698975
```

Fields: engine, primitive, **implementation**, prop kind, **memory descriptors**,
attributes (post-ops appear here), auxiliary, problem shape, exec time in ms.

Two fields carry the diagnosis, and most people read only the first:

- **Implementation** — `jit:gemm:any` is the generic Xe GEMM generator and is the *correct*
  implementation on Battlemage. `jit:xe_hp:gemm:any` is the Xe-HP/PVC systolic path and does
  not apply. A generic-sounding name is not evidence of a fallback.
- **Memory descriptors** — the `ab`/`ba` tag on each operand is its layout. This is where
  Intel GEMM problems actually live.

`ONEDNN_VERBOSE=1` also prints an engine table at startup. On a box with a discrete card and
integrated graphics there are two GPU engines; `gpu:0` on the exec line tells you which ran,
and `torch.xpu` may expose fewer devices than oneDNN sees.

## Step 3: Read dispatch — expected skip vs your fault

| Reason text | Reading |
|---|---|
| `skipping or dispatching to another implementation` | Architecture or heuristic gate — e.g. `xe_hp` systolic does not apply to Battlemage at all. **Expected. Do not report it.** |
| `unsupported format tag` | Rejected because of **your** memory descriptor. Actionable: change the layout and the candidate becomes eligible. |
| `unsupported datatype` / `unsupported attribute` | Same category — your primitive descriptor excluded a faster path. Check dtype and post-ops. |

Each line prints the oneDNN source file and line that gated it. When the reason is opaque,
read that location — it says exactly what was tested, and it is the fastest route to a
correct upstream bug report:

```bash
# clone once, pinned to the running version -- see /clone-repos
ONEDNN_VERBOSE=dispatch python repro.py 2>&1 | grep -oE "[a-z_/]+\.cpp:[0-9]+" | sort -u
# -> e.g. jit_xe_hp_systolic.cpp:79 ; feed that back as FILE and LINE:
FILE=jit_xe_hp_systolic.cpp; LINE=79
sed -n "$((LINE-5)),$((LINE+2))p" "$(find tmp/oneDNN/src -name "$FILE" | head -1)"
```

For example `jit_xe_hp_systolic.cpp:79` resolves to
`VDISPATCH_GEMM_SC(set_default_formats(d->a_type()), VERBOSE_UNSUPPORTED_TAG)` — the format
check, i.e. your memory descriptor, which is Fix 1.

## Quantized matmul: what oneDNN actually supports on Battlemage

Established by probing the installed library. Verified identically on **3.11.4 (oneAPI
2026.0) and a source build of 3.13.2** -- upgrading changes none of it.

| attribute | status |
| --- | --- |
| f8_e4m3 src x f8_e4m3 weights | exact |
| weights scale, per-tensor | exact |
| weights scale, per-column (`set_scales_mask`, N values) | exact |
| weights scale, **grouped** (`set_scales` with `groups`) | **silently wrong** |
| **src scales, any mask** | rejected: `unsupported scales configuration` |
| `sum` post-op, `binary_mul` post-op | exact |

### The `groups` argument to `set_scales` produces wrong results

`set_scales(DNNL_ARG_WEIGHTS, mask, {BK, BN})` returns wrong values with no error. The
primitive descriptor is created and `impl_info_str()` reports `jit:gemm:any`.

Grouped along K, the result is exactly **half** the correct value, independent of M, N, K,
BN and BK. With all-ones operands, K=512 and scale 0.5, `C[0]` is 128 where 256 is correct.
It is the scale that is wrong and not the accumulation: with per-K-block scales 1,2,3,4 the
answer is 640, which is `0.5 * BK * (1+2+3+4)` -- every block contributes at half its
scale. Partial accumulation would have given 384 or 512.

Grouped along N is **also wrong**, in a way a single-element check does not reveal: with
`groups {1, 128}` over N, 30 of 40 sampled output elements are wrong, `C[0,0]` among the
ten that happen to be right. Checking one element, or checking with a uniform scale, passes
a broken configuration -- both mistakes were made here before the sweep above was written.
**Verify a scale configuration with varying values across many output positions**, never a
constant at one corner.

### How to express block scales with only the exact primitives

A W8A8 checkpoint's weight scale is per `(N-block, K-block)`, which is what `groups` exists
to express. Since `groups` cannot be trusted, decompose over K and expand over N:

- one matmul per K-block, f8 src and f8 weights, f32 destination;
- weights scale as an **ungrouped per-column** array of N values, built by repeating that
  K-block's `NB` block scales `BN` times each -- exact, where the grouped form is not;
- `binary_mul` post-op carrying `A_scale[:, kb]` as an `[M, 1]` per-row vector, because src
  scales are rejected outright and the activation scale is per row within a K-block;
- `sum` post-op to accumulate into the running f32 destination.

Verified exact at every step: plain matmul, then each attribute added in turn, sampled
across 40 output positions with varying scales.

Two further constraints found the same way:

- **oneDNN reads scale and binary-operand memory densely, ignoring the strides in the
  memory descriptor.** A scale column taken from a `[M, K_blocks]` tensor with a strided
  desc silently reads the wrong elements -- correct only at row 0, where the stride does
  not matter. Transpose or gather into contiguous memory first.
- A SYCL queue is **out-of-order by default**, so kernels preparing scale buffers race the
  matmuls consuming them. PyTorch's XPU queue is in-order, which is why an in-tree kernel
  running on the framework's queue does not hit this; a standalone reproducer must ask for
  `sycl::property::queue::in_order`.

### But do not build a block-scaled FP8 GEMM this way

The decomposition is correct and it is the wrong shape for the problem. Composing oneDNN
primitives forces the accumulator through memory once per K-block, and that is all the
kernel does. Measured on Arc B580 at M=512, N=4096, K=2560:

| | |
| --- | --- |
| full decomposed kernel | 0.778 ms |
| `memset` of the f32 accumulator | 0.011 ms |
| building the expanded scale tables | 0.004 ms |
| one K-block matmul, x20 | 0.035 ms x 20 = **0.70 ms** |

Each K-block reads and writes the whole `[M, N]` float32 destination: 8 MiB each way in
0.035 ms is about 460 GB/s, which is this part's memory bandwidth. The decomposition is
bandwidth-bound on accumulator traffic -- 320 MiB of it per call -- and no amount of tuning
inside oneDNN removes traffic the composition itself creates.

For reference, at the same shape: this kernel 0.78 ms, a naive PyTorch
dequantise-and-matmul 0.67 ms, vLLM's Triton `w8a8_triton_block_scaled_mm` 3.24 ms with no
tuned config for the device.

**A block-scaled GEMM has to keep the accumulator in registers across K-blocks**, which
means one fused kernel -- Triton or SYCL -- not a sequence of library calls. Use oneDNN for
the dense GEMM it is unbeatable at, and reach for a fused kernel the moment scales vary
along the reduction dimension.

**Do not go into oneDNN's source to make its GEMM faster.** Measured on Arc B580 it reaches
~100 TFLOP/s bf16, roughly 90% of the part's XMX peak; there is no headroom there. The
source is for finding out *why a candidate was rejected*, which is a call-level problem you
can fix in a Solution.

## Knobs that exist, and the one that does not

There is **no supported environment variable that forces a particular GPU implementation** —
the most common wrong assumption about oneDNN tuning.

| Variable | Scope | Effect |
|---|---|---|
| `ONEDNN_VERBOSE` | observability | above |
| `ONEDNN_PRIMITIVE_CACHE_CAPACITY` | perf | primitives cached before eviction |
| `ONEDNN_DEFAULT_FPMATH_MODE` | numerics | `strict`/`bf16`/`f16`/`tf32`/`any`. Changes results — a Solution relying on it must set it in-process, and the definition's tolerance must accommodate it |
| `ONEDNN_MAX_CPU_ISA` | **CPU only** | Does nothing for the GPU engine; listed so it is not tried in confusion |

Everything else is at the API level: the descriptors you pass, the post-ops you attach, and
whether you split one primitive into two. Those are what a Solution can change.

## Fix 1: Weight layout

Same shape, same implementation, same device — only the weight's layout tag differs:

```
wei:f16::blocked:ba::f0 ... 64x896:896x4864, 0.698975
wei:f16::blocked:ab::f0 ... 64x896:896x4864, 0.218994
```

`[M,896] x [896,4864]` fp16 on Arc B580, each cell a **separate process**, 100 warmup /
300 measured:

| M | `F.linear(x, w)` `[N,K]`/`ba` | `x @ w.t().contiguous()` `[K,N]`/`ab` | ratio |
|---|---|---|---|
| 1 | 0.0137 ms | 0.0231 ms | **0.59x** |
| 64 | 0.0292 ms | 0.0224 ms | **1.30x** |
| 512 | 0.0566 ms | 0.0557 ms | 1.02x |
| 2048 | 0.1904 ms | 0.1925 ms | 0.99x |

**The win is a narrow band (roughly M=8..128) and it reverses at M=1**, where the `[N,K]`
weight `F.linear` already has is 1.7x faster. A blanket load-time transpose slows down the
case it was meant to help.

Measure it for the shapes in *your* workload file — one process per cell, because what
oneDNN has already dispatched and cached changes what the next shape costs (the same M=64
cell reads 1.17x-1.71x when swept in one process).

Ship the fix as a Solution gated on `x.shape[0]`, with the weight transposed once and cached
— never as a blanket rule. Load-time weight transforms belong in
`flashinfer_bench/integration/weight_layout.py`.

## Fix 2: Primitive caching

`create:` lines appearing in a steady-state loop mean shape churn is rebuilding primitives
per call. Cache primitive descriptors keyed on problem shape, as
`examples/sycl/onednn_gemm_swiglu.cpp` does.

## Fix 3: Post-ops — fuse without leaving oneDNN

Post-ops are elementwise or binary operations applied to the GEMM output before it leaves
registers. They cannot express a pairwise reduction across accumulator lanes (SwiGLU folds
two lanes into one), but splitting into two matmuls makes it expressible at identical FLOPs:

```
up  = (x @ Wu) * r[m]                     # binary_mul post-op
out = swish((x @ Wg) * r[m]) * up         # binary_mul, eltwise_swish, binary_mul
```

Against vLLM's real 3-kernel MLP (norm → gate_up GEMM → `silu_and_mul` → down GEMM), full
MLP block, Qwen2.5-0.5B layer 0:

Measured against vLLM's real path (Arc B580, device-event timing, 2026-09-06), with the
per-call block removed:

| M | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| fused vs vLLM | 1.09x | 0.98x | 1.16x | 1.18x | 1.17x |

**Gate around M ≥ 2048**, where the win becomes consistent. Below that it is parity, not a
regression.

Apply Fix 4 before measuring this: a per-call block moves these numbers more than the
fusion does.

Two advantages over the CUTLASS-SYCL route: gate and up stay the separate matrices the model
already ships, so **no weight re-layout at all** and none of Xe-Fuse's interleaving traps
apply.

torch exposes no post-op API for this, so call it from C++ via
`dnnl::sycl_interop::make_engine` / `make_stream` on the framework's queue. The calls are
`primitive_attr::set_post_ops`, `post_ops::append_eltwise(algorithm::eltwise_swish, ...)`,
`post_ops::append_binary(algorithm::binary_mul, md)`, then `matmul::primitive_desc` with the
attr, and at execution `DNNL_ARG_ATTR_MULTIPLE_POST_OP(n) | DNNL_ARG_SRC_1`.

Declare `onednn` in the solution's dependencies and `SyclBuilder` supplies the include,
`-ldnnl` and rpath. Worked example: `examples/sycl/onednn_gemm_swiglu.cpp`. Upstream
reference: <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_post_ops.html>.

## Fix 4: Do not block the host on every call

`ctx.stream.wait()` after `execute` is the single most expensive mistake available here, and
it looks like ordinary hygiene.

The oneDNN stream is built over PyTorch's own SYCL queue by
`dnnl::sycl_interop::make_stream(engine, *q)`, so the work is already ordered against
everything else on that queue, and the caller synchronizes when it needs the result. Waiting
inside the kernel adds a full host-side round trip to every launch — which `torch.matmul`,
your baseline, does not pay.

Measured on Arc B580, same kernel, wait removed:

| | m=1 | m=701 | m=2801 |
|---|---|---|---|
| fused SwiGLU vs reference | 0.44x → **2.21x** | 1.01x → **3.02x** | 2.03x → **3.11x** |

Across seven plain GEMM shapes it moved an explicit oneDNN call from 0.36-0.87x to
1.01-1.66x against `torch.matmul`. Check for this before concluding a oneDNN-backed solution
is structurally slower than the vendor path.

## Before trusting any GEMM comparison: check the two oneDNN versions

A GEMM definition's `reference` is `torch.matmul`, which runs the oneDNN **bundled with
torch**. A SYCL solution links whatever `FIB_ONEDNN_DIR` points at. These are frequently not
the same build:

```
onednn          3.11.4+0291f8943088 (/opt/intel/oneapi/dnnl/2026.0/lib/libdnnl.so.3.11)
onednn_runtime  3.12.3+6db5f1ba860c
```

A minor-version gap like this makes every ratio cross-library rather than a kernel result.
Check it before reporting a GEMM number, not after.

`flashinfer-bench` now records both in `environment.libs` and logs a warning when they
differ. On a mismatch, either point `FIB_ONEDNN_DIR` at a matching build or state explicitly that
the comparison spans two libraries. To build a specific release:

```bash
python scripts/build_onednn.py --list
python scripts/build_onednn.py --version v3.13.2
export FIB_ONEDNN_DIR=$HOME/.cache/flashinfer_bench/onednn/v3.13.2
```

This is also the route to **Fix 5**: a modified kernel catalog requires rebuilding oneDNN,
and this builds into its own prefix so the patched library never shadows the oneAPI one.

```bash
python -c "
from flashinfer_bench.integration.providers import onednn_link_version, onednn_runtime_version
print('link   :', onednn_link_version())
print('runtime:', onednn_runtime_version())"
```

## Fix 5: The tile/strategy the selector chose is wrong for your shape

Fixes 1-4 change how you *call* oneDNN. This one changes what kernel it *generates* — for
your shape only. Use it when the call is already clean and the GEMM is still slow.

Intel GPU GEMMs are JIT-generated by gemmstone from a **shape-gated catalog of strategy
descriptors**, not hand-written per shape. The selection is observable:

```bash
ONEDNN_VERBOSE=debuginfo=5 python repro.py 2>&1 | grep "gpu,gemm"
```

```
consider:F gemm BBS T@4N@8N 32 64 at16x2+m64@48 am32+m32@64 aB wg 8x4 ... ,score:75376269
consider:F gemm BBS T@4N@8N 64 40 at16+m64@48  am32+m32@56 aB wg 4x8 ... ,score:88189478
kernel:  gemm BB[SB] T@16N@16N 32 64 ... wg 8x4 sys xaf ...
```

- `consider:` lines are catalog entries evaluated against your problem, each scored.
  **Lowest score wins** — here 135 candidates were considered and the `wg 8x4`, 32x64 tile
  won, which is what the `kernel:` line confirms was generated.
- The descriptor reads as tile and work-group shape: `32 64` is the tile, `wg 8x4` the
  work-group, `grf256` the register mode, `sys` systolic/XMX.

**To change it**, add or adjust an entry in
`tmp/oneDNN/src/gpu/intel/gemm/jit/selector/db/kernel.db`, gated on your precision,
transpose pattern and shape range:

```
{{'C', "gemm", {"B","B","S"}, {"N","T","N"}},   // bf16 x bf16 -> fp32 acc, N/T/N
 { ... shape gates, e.g. {-1, 8, -1} ... },     // applies only in this range
 "ab2 ab8 ab l4 cab1 wg 4x4 int sr",            // strategy string
 { 8, ..., {32,16,8}, {4,4,1}, ... }},          // unroll, tiles, work-group
```

Entries are **shape-scoped**, so a specialization for your shape leaves every other shape
alone. `generator/strategy_parser.cpp` defines the strategy-string grammar.

**The cost is real: this requires rebuilding oneDNN and shipping that build.** There is no
runtime override — the catalog is compiled in (`dev_getenv` exposes only
`enable_generator_dsl` and `generator_dsl_specialize`, which are generator debug switches,
not strategy selection). So:

- Confirm by measurement first that a different tile is actually faster for your shape —
  the scoring heuristic is usually right, and 135 candidates were already weighed.
- **The framework records it for you.** `environment.libs` in every trace now carries
  `onednn` as `version+commit (resolved path)` rather than a directory, so a locally rebuilt
  library is distinguishable from a released one. Still say *what* you changed in the
  solution's `description` — the commit proves it is not stock, not what the patch was.
- Prefer sending the catalog entry upstream: it is data, shape-gated, and benefits everyone
  on that part. This is the one case where rung 4 is cheap to write.

## Decide: workaround, patch, or upstream

| Finding | Action |
|---|---|
| `unsupported format tag` on a faster candidate | Change the layout in a Solution. Ships today. |
| Right implementation and descriptors, still slower than a naive SYCL kernel | SYCL solution for those shapes; report the shape range upstream with the exec lines |
| Numerically wrong output against a `PASSED` reference | Report upstream urgently, with a minimal repro; ship a SYCL solution to unblock |
| Slow at one batch size only | Record per workload and move on — regime-dependent results are normal and can reverse |
| Expected architecture skip in dispatch output | Nothing. Do not report it |

A bug report needs: the `ONEDNN_VERBOSE=dispatch,exec` output, the version line
(`onednn_verbose,v1,info,oneDNN v...`), the engine line with the driver version, and the
measured times. File at <https://github.com/uxlfoundation/oneDNN/issues>.

## Checklist

```bash
ONEDNN_VERBOSE=1            python repro.py 2>&1 | grep primitive,exec   # what ran
ONEDNN_VERBOSE=dispatch     python repro.py 2>&1 | grep create:dispatch  # what was rejected
ONEDNN_VERBOSE=profile,exec python repro.py 2>&1 | grep create:          # per-call rebuild?
```

4. Is a naive SYCL kernel competitive? If yes, oneDNN is not the problem —
   `optimize-intel-kernels` Step 4a.
5. Does layout matter for these shapes? Time `F.linear(x, w)` against
   `x @ w.t().contiguous()` at your real batch sizes, one process per cell.
