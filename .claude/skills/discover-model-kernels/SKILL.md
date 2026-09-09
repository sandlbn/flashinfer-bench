---
name: discover-model-kernels
description: Discover the ops a model actually runs under its serving stack, resolve each to the kernel that really implements it, and emit a verified harness for it. Use before choosing anything to optimize — instead of naming a definition by hand.
---

# Discover the model's kernels

Naming a target by hand picks an op because someone remembered it, at a shape nobody
checked, under a definition label that may not be what the stack looks up. The model
already knows what it runs. Ask it.

Two steps, both mechanical: **discover** what executed, then **pull** the kernel behind
each op. Neither takes a definition name as input.

## Step 1: Discover the ops

```bash
python scripts/harness_from_model.py --model <repo_id> --out-dir tools/kernel-harness/auto
```

Run it with the interpreter of the environment the **serving stack** is installed in, not
the development venv — the point is to record the ops that stack dispatches, and a plain
`transformers` run dispatches different ones.

A `TorchDispatchMode` tallies every op with the shapes, dtypes and call counts it ran at,
then emits one harness per (op, shape) whose `forward` calls **the same op the model
called**, imported from wherever it lives. Nothing is reimplemented, so a trial cannot end
up tuning a different function than production runs.

Each harness is verified before it is offered — it must run, return the shape and dtype the
model saw, and be finite — and discarded with a reason if it fails. Read the discards:

| Discard reason | What it means |
| --- | --- |
| `returned list, not a tensor` | multi-output op; harness it by hand or skip |
| `wait ... raised inside <module>: capturing its state from the run` | the op reads state the stack sets around each forward step (attention, KV-cache update). Not a discard: the model is run once more, the values that module held during the call are captured, pruned to what the op consulted, and stored in a `.state.pt` beside the harness, which re-establishes them around each call. A `drop` after this names what could not be captured |
| `shape (a) != (b) seen in the model` | the recorded args do not reproduce the call — a real defect, not a nuisance |

Ops that are plumbing (views, copies, allocation) are skipped: harnessing them measures the
allocator, not a kernel.

## Step 2: Resolve every op to the kernel that ran

An op name is not a kernel. The dispatcher decides what runs, and the answer changes with
the device and the installed providers.

```bash
python scripts/pull_kernel_source.py \
    --from-report tools/kernel-harness/auto/discovered.json \
    --from-harnesses tools/kernel-harness/auto \
    --bundle tools/kernel-harness/pulled
```

`--from-report` is what makes this cover the stack rather than a favourite kernel: it reads
**every** op discovery recorded, not the handful that got harnesses. Resolving one op tells
you nothing about where the model's time goes.

Two questions get asked, and neither consults a table:

1. **`torch._C._dispatch_dump`** — which keys are registered, and the source file each
   registration came from. The providing project falls out of the op itself.
2. **The op is run once with `ONEDNN_VERBOSE=1`** — because a registration inside PyTorch
   is not the end of the answer. For GEMM-shaped work ATen calls oneDNN, and the primitive
   oneDNN picks is what runs. Running it is the only way to know, and it reports the exact
   primitive and problem: `matmul via jit:gemm:any [4x1024:1024x4096]`.

### What it can conclude, and where each goes

| Resolved as | How it was established | Route |
| --- | --- | --- |
| **oneDNN** | the op ran and oneDNN logged a primitive | `/optimize-onednn` — the library's kernel is what the stack already runs, so it is the baseline; the descriptors, attributes, lifetime and decomposition are the axes a caller controls |
| **provider kernel** | a device key, and the binding file found in a local checkout | "Optimize the one that survives" — the trial loop against the bundle |
| **Triton** | recorded at the JIT entry point, with source file and line | `/wrap-kernel-for-tuning` — tune in place, no substitution |
| **Python-registered custom op** | no dispatcher entry at all; the namespace names the package | `/wrap-kernel-for-tuning` — it needs the stack's context to run |
| **ATen kernel inside PyTorch** | a device key registered in PyTorch's own tree | no local source to edit; measure, or report upstream |
| **decomposition** | only a composite key, and oneDNN logged nothing | optimize what it decomposes to |

The first and last rows are the ones to read twice. An op with no kernel of its own can
carry most of the device time: `aten::linear` is a decomposition, and what actually runs is
a oneDNN matmul. A replacement written for the composite would not be what the model calls,
which is why the resolver runs the op instead of stopping at "composite".

A Python-registered op — attention, in most serving stacks — leaves the dispatcher dump
empty. That reads like "does not run here" unless the resolver falls back to the namespace,
which it does: an empty dump is not evidence that the op is absent.

If no checkout of the providing project is on the box, the tool says so rather than
guessing. Clone it to `tmp/` to read the kernel, or optimize against the harness alone.

### What `--bundle` hands to the optimizer

One directory per op, so optimization starts from the code that runs rather than from a
blank file and a remembered ratio:

```
tools/kernel-harness/pulled/<ns>_<op>/
├── harness.py          the verified harness, at the shape with the most calls
├── source/             the kernel's own source, copied whole
└── PROVENANCE.md       dispatch key, schema, providing project, where each file came from
```

The source is copied whole rather than sliced. The launcher, the functor and the dispatch
macros are all part of what you are changing, and a slice that keeps only the arithmetic
compiles into a different kernel.

Read the **schema** in `PROVENANCE.md` before touching the source — it is the dispatcher's
own, so it states what is optional and what is mutated in place, which the C++ signature
alone does not. An op declared `Tensor($0! -> ) input` writes into its caller's storage,
and a replacement that returns a fresh tensor is a different operation however fast it is.

These bundles are gitignored: they are reproducible from the scripts, and vendoring a
provider's source into this repo is not intended.

## Step 2b: Ask what could be fused, not just what could be replaced

Per-op routing prices each op alone against the substitution cost from
`calibration.get()`; a GEMM that is already a library call and an elementwise kernel
smaller than that cost both fail that bound. Folding the elementwise work into the GEMM's
*epilogue* escapes both — the operand is still in registers, so the saving is a launch and
a round trip through memory, and the fused GEMM replaces the call the stack already makes,
so no substitution is paid at all.

```bash
python scripts/fusion_candidates.py --report tools/kernel-harness/auto/discovered.json
```

A fusion is a property of the **edge** between two ops, which no per-op tally can supply:
discovery records producer→consumer edges, and only pairs the model actually ran are
proposed. The preset list is read from Xe-Fuse's own generator (`--list-presets`) rather
than copied, so the two cannot drift.

Two things this gets right that a table of "norm fuses into GEMM" would not:

- **A view between two ops is not a gap.** Discovery carries the real producer through
  reshapes, or every GEMM→activation edge would be recorded as `view→activation`.
- **A real op between them is a gap.** If the model splits the projection before norming
  it — per-head q/k norms make the edge `split_with_sizes→rms_norm`, not
  `linear→rms_norm` — the epilogue cannot see that operand and the preset does not apply.
  The same architecture without per-head norms would match, which is precisely why this is
  measured per model rather than asserted per architecture.

Report the reachable share, not a ratio: a fusion is worth the elementwise time it absorbs.

## Step 3: Bound it before writing anything

Do not go from "found a kernel" to "write a faster one". A substitution costs a fixed
amount per call, and that cost does not shrink with the kernel:

```python
from flashinfer_bench.device import calibration
cal = calibration.get()      # measures this part once, caches; None when it cannot
```

Against `cal.dispatch_us`, `cal.timing_floor_us` and `cal.bandwidth_gbps`, a candidate's
ceiling is `calls x (current - max(achievable, floor)) - calls x dispatch`, and the
mechanism matters: a source rewrite, a provider swap, or tuning the kernel the stack
already launches pay **no** dispatch, while an `apply()` substitution does. A candidate can
be hopeless as a substitution and worth doing as a rewrite.

A ratio measured below the timing floor is noise, and will motivate work that returns
nothing. See `../route-kernel-work/PLAN.md`.

## Step 4: Optimize the one that survives

```bash
python scripts/kernel_trials.py init <series> tools/kernel-harness/pulled/<op>/harness.py
```

The bundle's `source/` is what the model runs, not a reference to admire. Its constants —
a required sub-group size, a block size sized to another vendor's warp, a vectorization
width — are the knobs that exist, and any of them may have been chosen for a different
part. Such a constant being *legal* on this device is not evidence
it is *right*: a part can support several sub-group sizes with one of them native. Query
the device (`torch.xpu.get_device_properties(0).sub_group_sizes`), then settle it by
measuring both in the trial loop — reading the source tells you which knobs exist, never
which value wins. `../optimize-intel-kernels/architectures.md` has the per-part traps.

The trial loop gates timing on correctness, warms both arms and interleaves them, and
branches back to the best trial on a regression. `/wrap-kernel-for-tuning` covers the case
where the kernel needs the stack around it.

The baseline is **the harness**, which is the production kernel. A ratio against a
definition's PyTorch reference is a different and much larger number about a comparison
nobody deploys.

## Step 5: Prove it end to end

`/measure-serving-win`, with the overhead arm. A per-kernel ratio is not a result.

## Sources

- `scripts/harness_from_model.py` — discovery and harness emission, with verification
- `scripts/pull_kernel_source.py` — op to kernel, from the dispatcher
- `scripts/fusion_candidates.py` — which GEMM epilogues this model's edges would support
- `../optimize-intel-kernels/xe-fuse.md` — build flags and the operand-layout traps
- `scripts/kernel_trials.py` — the optimization loop
- `../route-kernel-work/PLAN.md` — choosing among candidates by measured ceiling
- `/find-kernel-gaps` — the complementary question: time no definition covers
