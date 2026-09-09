---
name: optimize-onednn
description: Diagnose and fix a slow oneDNN GEMM on Intel GPUs — read ONEDNN_VERBOSE and dispatch output, resolve a rejection to the gate in oneDNN's source, and apply the call-level fixes (weight layout, primitive caching, post-op fusion, no per-call host block). Use when a profile says GEMM or F.linear is slow on XPU.
---

# Optimize a oneDNN GEMM

`F.linear` and `torch.matmul` on XPU **are** oneDNN matmuls; you tune the call, not the
library. Do not try to beat oneDNN's matmul — the wins are in how it is called.

| | | |
|---|---|---|
| 1-3 | Repro, observe, read dispatch | what ran, and what was rejected |
| Fix 1 | Weight layout | when dispatch says `unsupported format tag` |
| Fix 2 | Primitive caching | when `create:` appears in a steady-state loop |
| Fix 3 | Post-op fusion | when an elementwise op follows the GEMM |
| Fix 4 | Remove the per-call host block | **check first if a oneDNN solution is slower than torch** |
| Fix 5 | Tile/strategy catalog entry | the call is clean and the GEMM is still slow — `references/strategy-catalog.md` |

Quantized matmul (fp8, scales, what `groups` does) is `references/quantized-matmul.md`.

## Step 1: Build a repro

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

Prints to stderr; values compose (`ONEDNN_VERBOSE=dispatch,exec`).

| Value | What you get | When |
|---|---|---|
| `1` | one `primitive,exec` line per execution | always, first |
| `dispatch` | one line per **rejected** implementation, with reason and source location | when the chosen implementation looks wrong |
| `profile,exec` | adds `create:cache_hit` lines | when you suspect per-call primitive rebuild |
| `all` | everything | last resort |

```
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,
  src:f16::blocked:ab::f0 wei:f16::blocked:ba::f0 dst:f16::blocked:ab::f0,
  attr-scratchpad:user,,64x896:896x4864,<ms>
```

Fields: engine, primitive, **implementation**, prop kind, **memory descriptors**,
attributes (post-ops appear here), auxiliary, problem shape, exec time in ms.

- **Implementation** — `jit:gemm:any` is the generic Xe GEMM generator and is the correct
  implementation on Battlemage. `jit:xe_hp:gemm:any` is the Xe-HP/PVC systolic path and
  does not apply.
- **Memory descriptors** — the `ab`/`ba` tag on each operand is its layout.

`ONEDNN_VERBOSE=1` also prints an engine table at startup. On a box with a discrete card and
integrated graphics there are two GPU engines; `gpu:0` on the exec line says which ran.

## Step 3: Read dispatch — expected skip vs your fault

| Reason text | Reading |
|---|---|
| `skipping or dispatching to another implementation` | Architecture or heuristic gate. Expected; do not report it |
| `unsupported format tag` | Rejected because of **your** memory descriptor — Fix 1 |
| `unsupported datatype` / `unsupported attribute` | Your primitive descriptor excluded a faster path — check dtype and post-ops |

Each line names the oneDNN source file and line that gated it. Read it when the reason is
opaque:

```bash
# clone once, pinned to the running version -- see /clone-repos
ONEDNN_VERBOSE=dispatch python repro.py 2>&1 | grep -oE "[a-z_/]+\.cpp:[0-9]+" | sort -u
FILE=<file.cpp>; LINE=<n>
sed -n "$((LINE-5)),$((LINE+2))p" "$(find tmp/oneDNN/src -name "$FILE" | head -1)"
```

## Knobs that exist, and the one that does not

There is **no environment variable that forces a particular GPU implementation**.

| Variable | Scope | Effect |
|---|---|---|
| `ONEDNN_VERBOSE` | observability | above |
| `ONEDNN_PRIMITIVE_CACHE_CAPACITY` | perf | primitives cached before eviction |
| `ONEDNN_DEFAULT_FPMATH_MODE` | numerics | `strict`/`bf16`/`f16`/`tf32`/`any`. Changes results — a Solution relying on it must set it in-process, and the definition's tolerance must accommodate it |
| `ONEDNN_MAX_CPU_ISA` | **CPU only** | Does nothing for the GPU engine |

Everything else is at the API level: the descriptors you pass, the post-ops you attach, and
whether you split one primitive into two.

## Fix 1: Weight layout

Same shape, same implementation, same device — only the weight's layout tag differs
(`wei:f16::blocked:ba` for `F.linear`'s `[N,K]`, `ab` for `x @ w.t().contiguous()`), and
which is faster depends on M: it can reverse between M=1 and mid-range M. Measure it for the
shapes in *your* workload file, **one process per cell** — what oneDNN has already
dispatched and cached changes what the next shape costs.

Ship the fix as a Solution gated on `x.shape[0]`, with the weight transposed once and
cached — never as a blanket load-time transpose. Load-time weight transforms belong in
`flashinfer_bench/integration/weight_layout.py`.

## Fix 2: Primitive caching

`create:` lines in a steady-state loop mean shape churn is rebuilding primitives per call.
Cache primitive descriptors keyed on problem shape, as `examples/sycl/onednn_gemm_swiglu.cpp`
does.

## Fix 3: Post-ops — fuse without leaving oneDNN

Post-ops are elementwise or binary operations applied to the GEMM output before it leaves
registers. They cannot express a pairwise reduction across accumulator lanes (SwiGLU folds
two lanes into one), but splitting into two matmuls makes it expressible at identical FLOPs:

```
up  = (x @ Wu) * r[m]                     # binary_mul post-op
out = swish((x @ Wg) * r[m]) * up         # binary_mul, eltwise_swish, binary_mul
```

Gate and up stay the separate matrices the model ships, so no weight re-layout and none of
Xe-Fuse's interleaving traps apply. The fusion wins only above a token count that depends
on the part; find the crossover by sweeping M against vLLM's unfused path
(`/measure-serving-win` with `--env FIB_VLLM_MLP_MIN_TOKENS=<M>`) and gate on it. Apply
Fix 4 before measuring this.

torch exposes no post-op API, so call it from C++ via `dnnl::sycl_interop::make_engine` /
`make_stream` on the framework's queue: `primitive_attr::set_post_ops`,
`post_ops::append_eltwise(algorithm::eltwise_swish, ...)`,
`post_ops::append_binary(algorithm::binary_mul, md)`, then `matmul::primitive_desc` with the
attr, and at execution `DNNL_ARG_ATTR_MULTIPLE_POST_OP(n) | DNNL_ARG_SRC_1`.

Declare `onednn` in the solution's dependencies and `SyclBuilder` supplies the include,
`-ldnnl` and rpath. Worked example: `examples/sycl/onednn_gemm_swiglu.cpp`. Upstream
reference: <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_post_ops.html>.

## Fix 4: Do not block the host on every call

Never `ctx.stream.wait()` after `execute`. The oneDNN stream is built over PyTorch's own
SYCL queue by `dnnl::sycl_interop::make_stream(engine, *q)`, so the work is already ordered
against everything else on that queue and the caller synchronizes when it needs the result.
A wait inside the kernel adds a host round trip to every launch that `torch.matmul`, your
baseline, does not pay. Check for this before concluding a oneDNN-backed solution is
structurally slower than the vendor path.

## Before trusting any GEMM comparison: check the two oneDNN versions

A GEMM definition's `reference` is `torch.matmul`, which runs the oneDNN **bundled with
torch**. A SYCL solution links whatever `FIB_ONEDNN_DIR` points at. When they differ, the
ratio is cross-library rather than a kernel result.

```bash
python -c "
from flashinfer_bench.integration.providers import onednn_link_version, onednn_runtime_version
print('link   :', onednn_link_version())
print('runtime:', onednn_runtime_version())"
```

`flashinfer-bench` records both in `environment.libs` and warns on a mismatch. Either point
`FIB_ONEDNN_DIR` at a matching build or state that the comparison spans two libraries. To
build a specific release into its own prefix:

```bash
python scripts/build_onednn.py --list
python scripts/build_onednn.py --version <tag>
export FIB_ONEDNN_DIR=$HOME/.cache/flashinfer_bench/onednn/<tag>
```

## Decide: workaround, patch, or upstream

| Finding | Action |
|---|---|
| `unsupported format tag` on a faster candidate | Change the layout in a Solution |
| Right implementation and descriptors, still slower than a naive SYCL kernel | SYCL solution for those shapes; report the shape range upstream with the exec lines |
| Numerically wrong output against a `PASSED` reference | Report upstream with a minimal repro; ship a SYCL solution to unblock |
| Slow at one batch size only | Record per workload and move on |
| Expected architecture skip in dispatch output | Nothing |

A bug report needs: the `ONEDNN_VERBOSE=dispatch,exec` output, the version line
(`onednn_verbose,v1,info,oneDNN v...`), the engine line with the driver version, and the
measured times. File at <https://github.com/uxlfoundation/oneDNN/issues>.

## Checklist

```bash
ONEDNN_VERBOSE=1            python repro.py 2>&1 | grep primitive,exec   # what ran
ONEDNN_VERBOSE=dispatch     python repro.py 2>&1 | grep create:dispatch  # what was rejected
ONEDNN_VERBOSE=profile,exec python repro.py 2>&1 | grep create:          # per-call rebuild?
```

Then: is a naive SYCL kernel competitive? If yes, oneDNN is not the problem
(`optimize-intel-kernels` Step 4a). Does layout matter for these shapes? Time
`F.linear(x, w)` against `x @ w.t().contiguous()` at your real batch sizes, one process per
cell.
