---
name: optimize-onednn
description: Tune a library GEMM call on Intel GPUs — obtain the library's execution, dispatch and selection output for the call the stack already makes, classify the shape's regime against its bytes and flops, and generate candidates along the axes a caller controls (operand descriptors, attributes, primitive lifetime, decomposition, synchronization, implementation selection, library build). Use when a profile says GEMM or F.linear is slow on XPU, or when the resolver's class for a candidate is the library.
---

# Tune a library GEMM call

Facts this skill rests on, none of them a claim about performance:

- `F.linear` and `torch.matmul` on this backend execute oneDNN matmul primitives. The
  library's kernel is therefore the baseline **by construction** -- it is what the stack
  runs today, so measuring it costs no trial and no substitution.
- The library generates its GPU GEMM kernels from a shape-gated catalog at run time. Which
  entry it generates is a function of the problem descriptor it was given, and it reports
  what it considered and what it chose.
- A caller does not choose the kernel. A caller chooses the descriptors, the attributes, the
  primitive's lifetime, how the problem is decomposed, and when the host waits. Those are
  the axes below.

Whether there is headroom at a given shape is not stated here; it is the classification in
"Read", performed per shape, per part, per library build.

Quantized matmul -- fp8, scales, and what the `groups` argument does -- is
`references/quantized-matmul.md`, and it carries a correctness finding worth reading before
any solution that takes scales.

---

# Obtain

## A repro of the call the stack makes

```python
# repro.py
import torch
from flashinfer_bench.data import TraceSet

NAME = "<definition_name>"
ts = TraceSet.from_path("tmp/flashinfer-trace")
d  = ts.definitions[NAME]
wl = ts.workloads[NAME][0].workload          # a recorded shape, not an invented one

inputs = [torch.randn(*s, dtype=dt, device="xpu:0")
          for s, dt in zip(d.get_input_shapes(wl.axes), d.torch_input_dtypes)]
ns = {"torch": torch}
exec(d.reference, ns)
for _ in range(3):                            # steady state, past primitive creation
    ns["run"](*inputs)
torch.xpu.synchronize()
```

Coming from the pipeline instead, the harness `scripts/harness_from_model.py` wrote for the
candidate is the repro, and it calls the op the model called rather than a reference.

## The library's execution line

`ONEDNN_VERBOSE` is the library's own variable. It prints to stderr and its values compose
(`ONEDNN_VERBOSE=dispatch,exec`).

Source: oneDNN verbose documentation, and the `onednn_verbose` lines the library emits

| Value | What it adds |
| --- | --- |
| `1` | one `primitive,exec` line per execution |
| `dispatch` | one line per **rejected** implementation, with the reason and the source location that gated it |
| `profile,exec` | `create:` lines, distinguishing a cache hit from a primitive built on this call |
| `debuginfo=5` | the GEMM selector's `consider:` candidates with their scores and the `kernel:` line it generated |
| `all` | everything the library can say, at the cost of volume |

```
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,
  src:f16::blocked:ab::f0 wei:f16::blocked:ba::f0 dst:f16::blocked:ab::f0,
  attr-scratchpad:user,,64x896:896x4864,<ms>
```

Fields, in order: engine, primitive, **implementation**, propagation kind, **memory
descriptors**, attributes (post-ops and scales appear here), auxiliary, **problem shape**,
and the library's own execution time in milliseconds.

- The **implementation** field names the generator that ran. Whether another implementation
  was even a candidate for this problem is what `dispatch` answers, and the gate's source
  line says on what test it was excluded. Do not carry an implementation name as the right
  or wrong one for a part: ask `dispatch`.
- The **memory descriptor** for each operand carries its layout tag and, where it is
  strided, its pitch.
- The **problem shape** field is what "Read" needs: with the operand dtypes it gives
  `bytes_min` and `flops` for this call.

`ONEDNN_VERBOSE=1` also prints an engine table at startup. On a box carrying both a discrete
card and integrated graphics there are two GPU engines, and `gpu:0` on the exec line says
which one ran.

## What was rejected, and by which gate

Source: the `dispatch` lines the library emits, each naming its own source file and line

| Reason text | What it says |
| --- | --- |
| `skipping or dispatching to another implementation` | the implementation declined at an architecture or heuristic gate of its own; nothing in the call caused it |
| `unsupported format tag` | rejected on the memory descriptor the caller passed |
| `unsupported datatype` / `unsupported attribute` | rejected on the dtype or on an attribute in the primitive descriptor the caller built |

Each line names the oneDNN source file and line that gated it. Read it when the reason is
opaque -- it says exactly what was tested, and it is what an upstream report has to quote:

```bash
# clone once, pinned to the running version -- see /clone-repos
ONEDNN_VERBOSE=dispatch python repro.py 2>&1 | grep -oE "[a-z_/]+\.cpp:[0-9]+" | sort -u
FILE=<file.cpp>; LINE=<n>
sed -n "$((LINE-5)),$((LINE+2))p" "$(find tmp/oneDNN/src -name "$FILE" | head -1)"
```

## What the selector considered

```bash
ONEDNN_VERBOSE=debuginfo=5 python repro.py 2>&1 | grep "gpu,gemm"
```

```
consider:F gemm BBS T@4N@8N 32 64 at16x2+m64@48 am32+m32@64 aB wg 8x4 ... ,score:<n>
consider:F gemm BBS T@4N@8N 64 40 at16+m64@48  am32+m32@56 aB wg 4x8 ... ,score:<n>
kernel:  gemm BB[SB] T@16N@16N 32 64 ... wg 8x4 sys xaf ...
```

`consider:` lines are catalog entries scored against this problem, lowest score chosen; the
`kernel:` line is what was generated. The descriptor reads as tile and work-group shape:
the two bare numbers are the tile, `wg AxB` the work-group, `grf256` the register mode,
`sys` the systolic path. Two things this output settles that nothing else does: whether the
descriptor change you are considering moves the problem into a different catalog entry, and
whether the entry chosen is one the catalog even has for this shape range.
Changing the catalog itself is `references/strategy-selection.md`, and it requires
rebuilding and shipping the library.

## Device time, geometry and spill

The exec line's own millisecond figure is the library's report of its own execution. For a
device-side duration free of the host path, and for the launch geometry, run the repro under
unitrace as `optimize-model-kernels/references/tools.md` describes.

## The varying axis, one process per point

Sweep the axis the workload varies (token count, batch) across the values the stack actually
presents, **one process per point**. What the library has already dispatched and cached
changes what the next shape costs, so a sweep inside one process measures the sweep order as
much as the shapes. The illustration "The weight's layout tag against the varying axis" in
`references/illustrations.md` shows how wide that difference came out on one part.

A harness that takes its shape from the environment makes the sweep one line per point;
`tools/kernel-harness/trials/linear_row_pad.py` is the form, and its module docstring lists
the variables it reads. Each `ab` arm inherits the environment it was launched with, so the
loop varies the shape and the two arms differ only in the change under test:

```bash
for M in <the token counts the scheduler presents>; do
  FIB_HARNESS_M=$M python scripts/kernel_trials.py ab <harness.py> \
      --env-a <arm A: the call as the stack makes it> \
      --env-b <arm B: the one change> --json tmp/sweep-$M.json
done
```

## Environment variables that exist, and the one that does not

There is **no environment variable that forces a particular GPU implementation**.

Source: oneDNN runtime controls documentation

| Variable | Scope | Effect |
| --- | --- | --- |
| `ONEDNN_VERBOSE` | observability | above |
| `ONEDNN_PRIMITIVE_CACHE_CAPACITY` | performance | how many primitives are cached before eviction |
| `ONEDNN_DEFAULT_FPMATH_MODE` | numerics | `strict`/`bf16`/`f16`/`tf32`/`any`. It changes results, so a solution relying on it must set it in-process and the definition's tolerance must accommodate it |
| `ONEDNN_MAX_CPU_ISA` | **CPU only** | nothing for the GPU engine |

Everything else is at the API level, which is what "Generate" is about.

---

# Read

Classify the shape before proposing anything. The full row table, its inputs and the field
names to record are `optimize-model-kernels/references/read-the-numbers.md`; what follows is
that classification's inputs for a library GEMM call.

1. **`bytes_min` and `flops`** come from the exec line's problem-shape field and the operand
   dtypes in its memory descriptors: the two operands read once plus the destination written
   once, and the multiply-accumulate count the shape implies.
2. **The denominators** come from calibration: `bandwidth_gbs`, `matmul_peak_tflops` at this
   dtype, `timing_floor_us`, `launch_floor_us` (`python scripts/calibrate_part.py`).
3. **`t_dev`** comes from unitrace, **`t_host`** and **`spread`** from
   `scripts/kernel_trials.py benchmark` on the harness against itself.
4. Then `t_mem = bytes_min / bw`, `t_cmp = flops / peak`, and the row is selected by the
   comparisons in that reference.

What each row means for a call you do not own:

| Row | What it leaves open on a library call |
| --- | --- |
| `at-the-bound` | no call-level change moves it at this shape; the candidate is fewer bytes or fewer flops -- a fusion that removes a pass, or a decomposition that removes an intermediate |
| `compute-bound-inefficient` | the gap is between the kernel and the matrix unit. Whether a different catalog entry is reachable is `debuginfo=5` plus a descriptor change; whether one exists at all is the catalog |
| `memory-bound-inefficient` | the gap is traffic. Operand layout, the pitch the descriptor carries, and whether an intermediate is being materialised between two primitives |
| `memory-bound-layout-limited` | the data arrives in the wrong layout; the change is a load-time transform, not a call |
| `launch-bound` | the shape is too small for the kernel to be what is being timed. Removing a launch -- a post-op, a merged primitive -- is the only thing that can move it |
| `host/sync-bound` | `create:` lines in the steady-state window, or a host wait in the call path. Fix the path before reading any ratio taken here: a ratio measured against a call path with a wait in it is about the wait |

Record the row and its arithmetic in whatever consumes the result -- a trial's `--strategy`
string, a note on the worklist row -- in the form
`regime=<row>; test=<lhs>=<value> <cmp> <rhs>=<value>`.

---

# Generate

The axes below are where a caller's changes live. They are the API surface, not a list of
remedies: which change to make is derived from the row selected in "Read" and the
mechanism principles in `optimize-model-kernels/references/mechanisms.md`, and a change
nobody wrote down is the expected case.

Source: the oneDNN matmul API -- memory descriptors, `primitive_attr`, primitive lifetime, and the selector's own output

| Axis | What the caller changes | It applied when | It won when |
| --- | --- | --- | --- |
| Operand descriptors | layout tag, strides and pitch, dtype | the descriptor field of the exec line changes | `t_dev` falls at the shapes the stack presents, one process per point |
| Attributes | post-ops, scales, `fpmath` mode | the attribute field of the exec line changes | as above, with correctness re-checked -- an arithmetic mode changes results |
| Primitive lifetime | built once against built per call | `create:` lines leave the steady-state window | `t_host` falls toward `t_dev` |
| Problem decomposition | split, merge, batch, or split the reduction | the number and shape of exec lines change | the sum of `t_dev` over the new calls falls |
| Synchronization | removing a host wait the queue ordering already provides | the wait is gone from the call path | `t_host` falls toward `t_dev` |
| Implementation selection | a descriptor change that makes a different catalog entry eligible | the implementation field, or the `kernel:` line, changes | `t_dev` falls |
| The library build | which build the call links | the version helpers report the build you intended | comparisons only mean anything once both arms report the same build |

## A weight's row pitch is part of its descriptor

The pitch of a weight in bytes travels in its memory descriptor, and the library accepts a
strided descriptor for a `[N, K]` weight with no reorder. `channel_period_bytes()` is the
pitch period this part's calibration resolved from a sweep of pitches under a streaming
read, and `None` where the sweep resolved none -- the transform then stands down.
`pad_rows_off_channel_period()` moves a pitch by one cache line, bit-identically.

Whether a pad helps a given weight is a measurement, not a property of the shape: it can
only show while the weight streams from device memory, so a single cache-resident weight in
a tight loop cannot see it -- measure with a rotation pool larger than `caps.l2_bytes`. The
serving hook (`flashinfer_bench/integration/vllm/weight_layout.py`,
`FIB_VLLM_PAD_WEIGHT_ROWS=1`) keeps a pad only where a load-time streaming A/B of that shape
measures a win. To check one pitch by hand:

```bash
python scripts/kernel_trials.py ab tools/kernel-harness/trials/linear_row_pad.py \
    --env-a FIB_ROW_PAD_ELEMS=0 --env-b FIB_ROW_PAD_ELEMS=32
# with FIB_WEIGHT_POOL_MB above the cache size, over the pitches the model's projections have
```

Ship a layout change as a solution gated on the axis it was measured against, with the
transform done once and cached -- never as a blanket load-time transpose. Load-time weight
transforms belong in `flashinfer_bench/integration/weight_layout.py`.

## Post-ops, and the shape of what they can express

Post-ops are elementwise or binary operations applied to the GEMM output before it leaves
registers, so they remove a launch and a round trip through memory. They cannot express a
pairwise reduction across accumulator lanes -- a gated activation folds two lanes into one
-- but splitting into two matmuls makes it expressible at identical arithmetic:

```
up  = (x @ Wu) * r[m]                     # binary_mul post-op
out = swish((x @ Wg) * r[m]) * up         # binary_mul, eltwise_swish, binary_mul
```

Gate and up stay the separate matrices the model ships, so no weight re-layout is needed and
none of the interleaving traps of a generated epilogue kernel apply. Whether the fused form
beats the stack's own sequence, and from which token count, is a sweep over the axis the
scheduler presents -- see the illustration "Split-matmul SwiGLU through post-ops" in
`references/illustrations.md` for one instance, and gate on the crossover *this* sweep
produces. Settle the synchronization axis before this sweep is read: a wait in the
call path is timed in both arms of it, and the `host/sync-bound` row of the classification
is what says whether one is there.

torch exposes no post-op API, so the call is made from C++ via
`dnnl::sycl_interop::make_engine` / `make_stream` on the framework's queue:
`primitive_attr::set_post_ops`, `post_ops::append_eltwise(algorithm::eltwise_swish, ...)`,
`post_ops::append_binary(algorithm::binary_mul, md)`, then `matmul::primitive_desc` with the
attr, and at execution `DNNL_ARG_ATTR_MULTIPLE_POST_OP(n) | DNNL_ARG_SRC_1`.

Declare `onednn` in the solution's dependencies and `SyclBuilder` supplies the include,
`-ldnnl` and rpath. Worked example: `examples/sycl/onednn_gemm_swiglu.cpp`. Upstream
reference: <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_post_ops.html>.

## Primitive lifetime

`create:` lines in a steady-state loop mean a primitive is being built per call, which shape
churn is enough to cause. Cache primitive descriptors keyed on the problem shape, as
`examples/sycl/onednn_gemm_swiglu.cpp` does, and confirm the `create:` lines leave the
window with `ONEDNN_VERBOSE=profile,exec`.

## Synchronization

The oneDNN stream is built over the framework's own SYCL queue by
`dnnl::sycl_interop::make_stream(engine, *q)`, so work submitted to it is already ordered
against everything else on that queue and the caller synchronizes when it needs the result.
A `ctx.stream.wait()` after `execute` therefore adds a host round trip to every launch that
`torch.matmul` -- the baseline -- does not pay. Check the call path for one before reading
any ratio a oneDNN-backed solution produces; the illustration "Removing a host wait from the
call path" is one instance of what it was worth to remove.

---

# Tools, and what each is good for here

| Question | Instrument | What its numbers are good for |
| --- | --- | --- |
| what primitive ran, with which descriptors and attributes | `ONEDNN_VERBOSE=1` | identifying the call; its millisecond field is the library's own report, and is not interchangeable with a verdict |
| what was rejected, and on what test | `ONEDNN_VERBOSE=dispatch` plus the named source line | deciding whether a descriptor change makes another implementation eligible |
| whether a primitive is built per call | `ONEDNN_VERBOSE=profile,exec` | the `host/sync-bound` row |
| which catalog entries were scored, and which was generated | `ONEDNN_VERBOSE=debuginfo=5` | whether the entry is reachable by a descriptor change, or only by rebuilding |
| device time, geometry, spill | unitrace | `t_dev` for the classification; the exec line's own time is not this |
| did a change win | `scripts/kernel_trials.py benchmark` (or `ab`, for two builds that cannot share a process) | the only sanctioned verdict |
| what this part costs | `python scripts/calibrate_part.py` | every denominator in "Read" |
| is a hand-written kernel competitive for this op class at all | `/compare-implementations` | it answers that with a measurement rather than a presumption, before search effort is spent |

The full instrument index is `optimize-model-kernels/references/tools.md`.

---

# Gates

Everything in `optimize-model-kernels/references/gates.md` applies. Four are specific to a
library call:

- **The two library builds must match before any comparison.** A GEMM definition's
  `reference` is `torch.matmul`, which runs the oneDNN bundled with torch; a SYCL solution
  links whatever `FIB_ONEDNN_DIR` points at. When they differ the ratio is a comparison
  between two libraries, not a kernel result.

  ```bash
  python -c "
  from flashinfer_bench.integration.providers import onednn_link_version, onednn_runtime_version
  print('link   :', onednn_link_version())
  print('runtime:', onednn_runtime_version())"
  ```

  `flashinfer-bench` records both in `environment.libs` and warns on a mismatch. Either
  point `FIB_ONEDNN_DIR` at a matching build or state that the comparison spans two
  libraries. To build a specific release into its own prefix:

  ```bash
  python scripts/build_onednn.py --list
  python scripts/build_onednn.py --version <tag>
  export FIB_ONEDNN_DIR=<prefix>    # the script prints this line with the prefix it built into
  ```

- **One process per shape.** Stated as a probe above; it is a gate because a sweep inside
  one process measures the dispatch and cache history as well as the shape.
- **`scripts/kernel_trials.py` is the timer.** The exec line's own millisecond field
  identifies a call; it does not decide a comparison.
- **Anything delivered through `apply()` pays the measured dispatch cost per call.** Read it
  from `calibration.get().dispatch_us`; where it is `None` the mechanism is unavailable, not
  free. A call-site or load-time delivery pays none of it -- `scripts/bound_candidates.py`
  prices both, and the pair it accepted is the one to build.

---

# Decide: workaround, patch, or upstream

| Action | Selected when |
| --- | --- |
| change the descriptor in a solution or at the call site | a `dispatch` line rejected a candidate implementation on the descriptor you passed, **and** the changed descriptor measures a win at the shapes the stack presents |
| a hand-written kernel for those shapes | the call is clean -- matching builds, no per-call create, no host wait -- and an alternative implementation measures competitive at those shapes (`/compare-implementations`), while the library's own `t_dev` stays above the bound its regime names |
| a catalog entry, rebuilt and shipped | `debuginfo=5` shows the entry you want is not among the ones scored for this shape range, and a measurement shows a different tile ahead -- `references/strategy-selection.md` |
| an upstream report | the output is numerically wrong against a reference that passes, or the exec lines and the bound show a shape range the library serves far from its own bound on this part |
| record per workload and move on | the finding holds at one point of the swept axis and is inside spread at the others |
| nothing | the `dispatch` line is an implementation declining at its own architecture gate |

A bug report needs: the `ONEDNN_VERBOSE=dispatch,exec` output, the version line
(`onednn_verbose,v1,info,oneDNN v...`), the engine line with the driver version, and the
measured times with the timer that produced them. File at
<https://github.com/uxlfoundation/oneDNN/issues>.

---

# Checklist

```bash
ONEDNN_VERBOSE=1            python repro.py 2>&1 | grep primitive,exec   # what ran
ONEDNN_VERBOSE=dispatch     python repro.py 2>&1 | grep create:dispatch  # what was rejected
ONEDNN_VERBOSE=profile,exec python repro.py 2>&1 | grep create:          # built per call?
ONEDNN_VERBOSE=debuginfo=5  python repro.py 2>&1 | grep "gpu,gemm"       # what was scored
python scripts/calibrate_part.py                                          # the denominators
```

Then, before proposing anything: compute `bytes_min` and `flops` from the exec line, take
`t_dev` from unitrace and `spread` from `kernel_trials.py`, and select the row in
`optimize-model-kernels/references/read-the-numbers.md`. The row, with its arithmetic, is
what the first hypothesis is written against.

---

# Failure table

| Symptom | What it is |
| --- | --- |
| no `primitive,exec` line at all | the op did not reach the library; resolve the class again (`scripts/pull_kernel_source.py`) before tuning a call that is not being made |
| verbose output empty | it goes to stderr; redirect with `2>&1` |
| `cannot find -ldnnl` when building a solution | declare `onednn` in the solution's dependencies, and set `FIB_ONEDNN_DIR` to the prefix `scripts/build_onednn.py` printed |
| the version helpers disagree | the gate above: match the builds or state the comparison spans two |
| `create:` on every call in steady state | the primitive-lifetime axis |
| a `dispatch` reason you cannot interpret | read the source line it names, with the clone pinned to the running version -- `/clone-repos` |
| ratios that move when the sweep order changes | one process per point |
| `kernel_trials.py benchmark` prints `ROUTING: REJECTED` | the (candidate, mechanism) pair was priced out; change what the named gate reads and re-run `scripts/bound_candidates.py` rather than measuring past it |

---

# Sources

- `references/illustrations.md` -- worked findings, one instance each, with the command that re-establishes them
- `references/quantized-matmul.md` -- scales, post-ops and one silent-wrongness finding
- `references/strategy-selection.md` -- reading the selector, and changing the catalog
- `optimize-model-kernels/references/read-the-numbers.md` -- the classification
- `optimize-model-kernels/references/mechanisms.md` -- the library-call row of the generators
- `optimize-model-kernels/references/tools.md`, `optimize-model-kernels/references/gates.md`
- `examples/sycl/onednn_gemm_swiglu.cpp` -- a worked C++ solution on the framework's queue
- `flashinfer_bench/integration/weight_layout.py` -- where a load-time weight transform lives
- `scripts/build_onednn.py` -- building a release into its own prefix
- `/optimize-intel-kernels` -- writing the kernel, once a measurement says one is worth writing
- `/compare-implementations` -- whether an alternative implementation is competitive at all
