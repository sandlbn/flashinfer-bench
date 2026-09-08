---
name: measure-serving-win
description: Measure what a kernel is worth end-to-end — tokens/sec under vLLM with and without our kernels, with proof the substitution actually happened. Use before claiming any deployment win, and whenever a per-kernel speedup needs converting into a number a user would feel.
---

# Measure the serving win

A speedup measured against a PyTorch reference is not a serving result. This turns it into
one: tokens/sec, before and after, on the stack a user runs.

**The rule this enforces:** a kernel speedup is worth at most its share of device time, and
that share must come from a *serving* profile. A 2x win on a family holding 2% of device
time is worth under 2%, however good the kernel is. Run this before saying a kernel helps.

## Step 1: Can vLLM serve the model?

```python
from vllm.model_executor.models.registry import ModelRegistry
print("<Arch>ForCausalLM" in set(ModelRegistry.get_supported_archs()))
```

A novel architecture is often absent. Then there is no serving number to be had, and a
per-kernel table is all the model can give — say that, rather than presenting per-kernel
ratios as though they were throughput.

## Step 2: Run both arms

```bash
python scripts/measure_serving_win.py --model <repo_id> --dataset tmp/flashinfer-trace
```

Run it with the interpreter of the environment **vLLM** is installed in, which is not
necessarily the project venv — the child arms inherit it through `sys.executable`.

That environment needs a `sitecustomize.py` on its path, because the patch has to install in
vLLM's worker process and nothing else runs there. Create it once, in that venv's
`site-packages/`:

```python
# <vllm-venv>/lib/python3.X/site-packages/sitecustomize.py
import os

if os.environ.get("FIB_VLLM_INTEGRATION", "").lower() in ("1", "true", "yes", "on"):
    from flashinfer_bench.integration.vllm import install_vllm_integrations

    install_vllm_integrations()
    if os.environ.get("FIB_ENABLE_APPLY", "").lower() in ("1", "true", "yes", "on"):
        from flashinfer_bench.apply import ApplyConfig, enable_apply

        enable_apply(
            os.environ.get("FIB_DATASET_PATH"),
            ApplyConfig(
                max_atol=float(os.environ.get("FIB_APPLY_MAX_ATOL", "0.02")),
                max_rtol=float(os.environ.get("FIB_APPLY_MAX_RTOL", "0.02")),
                on_miss_policy="use_def_best",
            ),
        )
```

It is inert unless `FIB_VLLM_INTEGRATION` is set, so it changes nothing by existing. The
tolerances matter: `apply()`'s default `max_rtol=1e-5` rejects every correct bf16 kernel,
because bf16's worst-case relative spacing is 2**-7 = 0.0078. `0.02` is ~2.5 bf16 ULPs --
loose enough to admit a correct kernel, tight enough that a wrong one (relative error
O(0.1-1)) still fails.

Baseline and patched run in **separate processes**: the patch installs at interpreter start
and cannot be toggled within one run. Both arms must generate identical work — fix the token
count with `ignore_eos=True` rather than trusting two runs to stop at the same place.

Driving it directly:

```bash
FIB_VLLM_INTEGRATION=1 FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=<dataset> python <harness>.py ...
```

These must reach the **worker** process, which is why they are environment variables read by
`sitecustomize` and not arguments to a launcher. Patching from a launcher patches a class
that never runs a forward pass, and nothing reports an error.

**Size the run so the timed window is seconds, not a second.** The patched arm pays a
one-time SYCL build and JIT that the baseline does not, and a warm-up generate does not fully
absorb it. The same configuration measured at 0.9s and at 5s per arm reported a 2.5x
difference in the delta; only the longer one was the kernels. Raise `--prompts` and
`--out-tokens` until the arms run several seconds, and treat any delta from a sub-second
window as unmeasured.

**Run each arm several times and compare medians.** The harness alternates arms and defaults
to three repeats, reporting the spread; a delta smaller than that spread is not a result and
it says so. A single run per arm cannot tell a kernel effect from drift — treat any one-shot
delta as a hypothesis, and confirm it against the substitution counters before reporting it.
A large delta on a family that barely substituted is drift, not a win.

**To decide whether an opt-in integration flag should become the default**, set it on the
patched arm only with `--env KEY=VALUE`. Both arms then share one baseline and differ in
exactly the thing under test. This is the measurement an adapter gated behind a flag is
usually waiting for: a per-kernel win on the fused path is not by itself a reason to turn it
on, because the dispatch it adds is paid on every call whether or not the fusion helps.

## Step 3: Read the dispatch counters — the validity gate

At exit the adapters print what happened:

```
[flashinfer-bench] adapter dispatch:
  <family>: N call(s), M applied (P%)
  detail: {'<family> applied h<N>': ..., '<family> no-solution d<N>': ...}
```

**A throughput number without these is uninterpretable.** `0 applied` is indistinguishable
from "the kernel did not help" when the truth is it never ran.

| Detail string | Meaning |
| --- | --- |
| `applied h<N>` | substituted; the timing counts |
| `no-solution h<N>` | nothing matched that shape — extract it, or record the family as untested |
| `unsupported <reason>` | the adapter declined (3D input, odd width); expected |

If `applied` is 0 everywhere, fix that before reading the timing at all. A family showing
`no-solution` contributed exactly zero, and no kernel work on it could have shown up.

## Step 4: Report share and ratio together

```
<family>  <ratio>x per kernel  x  <share>% of device time  ->  expected  (measured: <x>%)
```

**The ratio must be against what the serving stack runs, not against the definition's
reference.** `flashinfer-bench run` reports speedup versus the definition's PyTorch
reference, which makes several passes over memory and launches several kernels. vLLM does
not run that; it runs its own fused kernel. A family reported at 2-3x versus the reference
can be at parity with the kernel it would actually replace, and then `share x ratio` predicts
a win that cannot exist. Watch for a reference that does *more work than production*: a
gated-MLP reference computing two separate projections where the server issues one merged
GEMM is ~2x worse before any kernel is written, and every ratio measured against it inherits
that factor. Take the ratio from a baseline solution wrapping the provider kernel
(`add-baselines --providers vllm-xpu`), not from the reference column.

**Compare the kernel's runtime against what dispatch costs, before substituting at all.**
Measured on Arc B580 (fp16 RMSNorm, 64x1536, wall-clock over 3000 calls, 2026-09-08):

| path | us/call |
| --- | --- |
| the provider kernel, called directly | 3.24 |
| the same work through a successful `apply()` | 9.15 |
| `apply()` on a definition it cannot match | 0.35 |

A successful substitution costs about 5.9us of Python -- resolve, key build, dtype check,
table and solution lookup, then the `Runnable` invocation -- which is **nearly twice the
kernel's entire runtime**. So at decode sizes no elementwise kernel can win, however fast it
is: the replacement machinery costs more than the thing being replaced. Two corollaries:

- A family is worth substituting only where its kernel time is large against ~6us. That is
  GEMM, attention and prefill-sized elementwise work -- not decode-sized norms and
  activations, whatever their per-kernel ratio says.
- A *miss* is nearly free (0.35us), which is why an overhead arm pointed at an empty dataset
  reports ~0 while a fully-substituting run loses double digits. Those two numbers measure
  different paths; do not read the first as the cost of the second.

Re-measure this on your own part before relying on it -- it is a property of the dispatch
path, not of any kernel.

**Beware the timing floor when reading per-kernel numbers.** In the same setup the benchmark
reports ~47us for a kernel that takes ~3us, because device-event timing has a fixed cost that
both arms pay. Below that floor the *ratios* are noise: two kernels an order of magnitude
apart in real work can both report ~50us. Sanity-check against bandwidth -- bytes moved
divided by the part's achievable GB/s -- before believing a small-batch ratio.

**Check the family is not launch-bound before expecting anything.** Read the recorded
latencies across the batch sweep: if latency barely moves from the smallest batch to one
several hundred times larger, the kernel is not doing measurable work at those sizes and the
time is launch overhead. Elementwise families at decode sizes sit on this floor, and no
kernel can win time that is not being spent — while `apply()`'s per-call dispatch is still
paid on every one of the tens of thousands of calls a run makes. That combination produces a
net loss at 100% substitution.

A measured result far from `share x ratio` means something is wrong — most often that the
share came from a `transformers` profile rather than a serving one. A transformers profile
has no paged attention and no `reshape_and_cache`, and its `aten::cat` time is `DynamicCache`
concatenation that no serving stack performs. Take the share from `/profile-intel` run
against the serving stack, or from this harness's own profile.

## Step 5: Decide what the number means

A small end-to-end gain is the normal outcome for elementwise families, and is not evidence
the kernel is bad. On Intel, dense GEMM is oneDNN whether or not we are involved, so when
GEMM dominates a model the reachable headroom for everything else is small by construction.

Report it as "worth +X% on this model" and follow the share: the useful next move is a family
with a larger one, not more tuning on a small one.

**Measure what the dispatch itself costs, with `--overhead-arm`.** It adds a third arm:
patched, but pointed at an empty dataset, so every interception runs and nothing can ever
match. That is the price of being in the path with none of the benefit, and on an elementwise
family called tens of thousands of times per run it is not small — a model where *nothing*
substituted can still lose double-digit throughput. Two consequences follow. A reported win
is **net** of this tax, so the kernels' own contribution is larger than the headline. And a
regression on a model with no matching definitions is not evidence about any kernel; it is
the cost of enabling `apply()` at all. Report the two separately.

**A regression with 0% substitution and a near-zero overhead arm means the lookup itself is
the cost.** When a definition exists but cannot match — wrong dtype, a shape outside the
recorded keys — every call runs the full resolve path (merge, key build, dtype check) and
then falls back, whereas a definition that does not exist at all short-circuits on the first
lookup. So "patched with a real dataset that never matches" can be materially slower than
"patched with an empty one", and the difference is not the kernels. Read `dispatch` and
`kernels` separately before concluding anything: a near-zero `dispatch` with a large negative
`kernels` at 0% applied is this, not a bad kernel.

**A regression at 100% substitution is a real result, not a broken measurement.** It is the
expected consequence of how the kernels get selected: `apply()` keys its index on the exact
axis values in the recorded workloads, and `on_miss_policy="use_def_best"` extends the winner
at those shapes to every shape the scheduler actually presents. A kernel that wins at a
recorded prefill M loses at a decode M of the batch size, and the counters still read 100%
because substitution is not correctness of choice. When this happens, do not retune the
kernel — the per-kernel number was never wrong. Either record workloads at the shapes serving
actually uses, or narrow the miss policy so the unmeasured shapes fall back. Report the
regression; a kernel family that costs throughput on the stack a user runs is the finding.

## Step 6: Close the loop — a `no-solution` line is a work item

Every `no-solution <shape>` in the counters is a shape the serving stack asked for and got
nothing for. Do not read it as a missing *kernel*. For the shape-parametric families —
activations, norms — the provider kernels and the in-tree SYCL/Triton templates already work
at any width. What is missing is the **definition**: without one, `add-baselines` has nothing
to match, so no kernel is ever sourced and none is ever optimized.

Extraction cannot supply these. `extract_model_kernels_xpu.py` hooks `nn.Module`s, and
neither family is one — a gated activation is written inline inside an MLP's `forward`, and a
fused add+norm spans two statements, so a hook on the norm sees only its half. The serving
counters are the only place these shapes are observed at all.

```bash
python scripts/fill_serving_gaps.py --from-json <result>.json          # report
python scripts/fill_serving_gaps.py --from-json <result>.json --eps <e> --write
```

It clones the widest sibling of the family, rescales every constant axis together, rewrites
the reference's asserted constants, and emits workloads over the sibling's batch sweep. Then
source and optimize as usual — `validate-references`, `add-baselines --in-tree`,
`add-baselines --providers`, `run --save-results` — and re-measure. Nothing here is a kernel
you have to write: for these families the in-tree SYCL and Triton templates plus the provider
baselines are generated per definition, so a definition is the whole cost of entry.

Mind the CLI's two spellings of `--definitions`: `validate-references` and `add-baselines`
take **one comma-separated string**, while `run` takes **space-separated names**. Comma-
joining them for `run` makes it search for a single definition whose name contains commas,
which it reports as not found and then exits 0 — the benchmark silently measures nothing.

Two things it will not do silently, because both produce a definition that benchmarks
cleanly while computing the wrong function:

- **Dtype is never inherited.** A definition whose dtype differs from the one the model
  runs in is refused by `apply()`'s dtype guard, so it never substitutes — and the counters
  report `no-solution`, which reads as "not extracted yet" rather than "extracted in the
  wrong dtype". Pass `--dtype` from the model config's `torch_dtype`; quantized checkpoints
  are commonly `float16` while the sibling definition is `bfloat16`.
- **Epsilon is never inherited.** Models differ (1e-5 and 1e-6 both occur), so a norm family
  requires `--eps` with the value that model uses (`config.json`'s `rms_norm_eps`).
- **A width already present at another epsilon gets a suffixed name.** `{family}_h{H}` does
  not encode epsilon, so two models sharing a hidden size collide on one name, and whichever
  loses is served a reference computing a slightly different function. Check for this
  whenever a family substitutes at 100% on a model you did not extract it from.

Generated definitions carry `status:unverified` and no `model:` tag: nothing has validated
them at this width yet, and the sibling's provenance is not theirs.

## Failure table

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Engine core initialization failed ... Failed core proc(s): {}` on the **patched arm only** | harness has no `if __name__ == "__main__":`; vLLM spawns the model into a subprocess and multiprocessing re-imports the module | add the guard — the real error (`_check_not_importing_main`) is several frames up and names neither vLLM nor your script |
| The same message on **both arms** | not our patch at all — the model itself did not load (memory, an unsupported quantization or config). This string is vLLM's generic top-level failure and never states the cause | read the worker's traceback, hundreds of lines earlier, not the tail. The harness extracts it and writes the full run to `tmp/serving-win/<model>.<arm>.fail.log` |
| No `patched vLLM:` line | env vars did not reach the worker | set them in the environment, not in the launcher's own process |
| `applied` is 0 for every family | dataset path wrong, or no solutions for these shapes | check `FIB_DATASET_PATH`; run `add-baselines --in-tree` for the model's definitions |
| `cannot be built here (BuildError: ... No such file or directory: 'ninja')`, then `applied` near 0 | the interpreter was invoked by absolute path, so its venv's `bin/` is off `PATH` and the SYCL builder cannot find `ninja` | the harness prepends `sys.executable`'s directory to the child's `PATH`; if driving vLLM by hand, activate the venv rather than naming its `python` |
| `NO dispatch counters` although the worker printed them | the counter parser did not match vLLM's worker prefix — output is tagged `(EngineCore pid=NNN) `, with a space inside the parens | match the counter line anywhere in the line, never anchored past a prefix |
| Throughput drops on a model where every family reports 0 applied | not a kernel result at all — this is `apply()`'s interception cost | run `--overhead-arm` to quantify it, and do not attribute it to any kernel |
| Throughput moves more than `share x ratio` | the two arms did not do equal work | `ignore_eos=True`, same prompts, same `max_tokens`, warm up before timing |

## Sources

- `scripts/measure_serving_win.py` — the harness
- `scripts/fill_serving_gaps.py` — turns its `no-solution` reports into definitions
- `/find-kernel-gaps` — the complementary detector: hot ops no definition covers
- `/profile-intel` — where the share comes from
- `../optimize-intel-kernels/architectures.md` — the `apply()` deployment traps that make a
  substitution silently not happen
