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
| `Forward context is not set` | needs the stack's per-step context (attention); not reachable this way |
| `shape (a) != (b) seen in the model` | the recorded args do not reproduce the call — a real defect, not a nuisance |

Ops that are plumbing (views, copies, allocation) are skipped: harnessing them measures the
allocator, not a kernel.

## Step 2: Pull the kernel behind each op

An op name is not a kernel. The dispatcher decides what runs, and the answer changes with
the device and the installed providers.

```bash
python scripts/pull_kernel_source.py --from-harnesses tools/kernel-harness/auto \
    --bundle tools/kernel-harness/pulled
```

This asks `torch._C._dispatch_dump` rather than consulting a table, so the providing
project falls out of the op itself. The registration path is a build-time path whose tail
describes the project's own layout, which locates the binding file in a local checkout —
and the checkout it lands in *is* the providing project, which is what makes the rest of
the search precise instead of a grep across every repo on the box.

Two outcomes, and they lead different places:

| Dispatcher says | Meaning | Where to go |
| --- | --- | --- |
| a key for your device (`XPU`, `CUDA`) | a real kernel runs; its source is named | Step 3 |
| `CompositeImplicitAutograd` only | the op is *rewritten* into other ops | there is no kernel to pull; the ops it decomposes to were recorded separately — optimize those, or fix the call (`/optimize-onednn`) |
| nothing registered | the op does not run on this device | not a target |

The second row is the common surprise: the largest consumer of device time is frequently an
op with no kernel of its own. `aten::linear` is a decomposition — what actually runs is the
library the decomposition reaches, and it is that call, not a hand-written replacement, that
is worth changing.

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
python scripts/kernel_trials.py init --harness tools/kernel-harness/pulled/<op>/harness.py
```

The bundle's `source/` is the starting point, not a reference to admire. The fastest first
move is usually a constant in the existing kernel that was chosen for a different part —
a required sub-group size, a block size sized to another vendor's warp, a vectorization
width — rather than a rewrite. Such a constant being *legal* on this device is not evidence
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
- `scripts/kernel_trials.py` — the optimization loop
- `../route-kernel-work/PLAN.md` — choosing among candidates by measured ceiling
- `/find-kernel-gaps` — the complementary question: time no definition covers
