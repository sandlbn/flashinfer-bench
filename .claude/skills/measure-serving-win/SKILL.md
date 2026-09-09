---
name: measure-serving-win
description: Measure what a kernel is worth end-to-end — tokens/sec under vLLM with and without our kernels, with proof the substitution happened. Use before claiming any deployment win, and to convert a per-kernel speedup into a serving number.
---

# Measure the serving win

A speedup against a PyTorch reference is not a serving result. A kernel speedup is worth at
most its share of device time, and that share must come from a *serving* profile.

## Step 1: Can vLLM serve the model?

```python
from vllm.model_executor.models.registry import ModelRegistry
print("<Arch>ForCausalLM" in set(ModelRegistry.get_supported_archs()))
```

If not, there is no serving number to be had; report a per-kernel table and say so.

## Step 2: Run both arms

```bash
python scripts/measure_serving_win.py --model <repo_id> --dataset tmp/flashinfer-trace
```

Run it with the interpreter of the environment **vLLM** is installed in — the child arms
inherit it through `sys.executable`. That environment needs a `sitecustomize.py` on its
path, because the patch has to install in vLLM's worker process. Create it once in that
venv's `site-packages/`:

```python
# <vllm-venv>/lib/python3.X/site-packages/sitecustomize.py
import os


def _min_gain():
    override = os.environ.get("FIB_APPLY_MIN_GAIN_US")
    if override is not None:
        return float(override)
    from flashinfer_bench.device import calibration

    cal = calibration.get()  # measures once, caches; None when it cannot
    return cal.dispatch_us if cal and cal.dispatch_us else 0.0


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
                # What a substitution costs is a property of the part, so measure it
                # rather than pasting a figure from another machine.
                min_gain_us=_min_gain(),
            ),
        )
```

It is inert unless `FIB_VLLM_INTEGRATION` is set. The tolerances matter: `apply()`'s
default `max_rtol=1e-5` rejects every correct bf16 kernel (bf16 spacing is `2**-7`); `0.02`
admits a correct kernel and still fails a wrong one.

Rules for the run:

- **Baseline and patched run in separate processes.** Both arms must generate identical
  work — `ignore_eos=True`, same prompts, same `max_tokens`.
- **Env vars must reach the worker process** — set them in the environment, not in a
  launcher; a launcher patches a class that never runs a forward pass.
  `FIB_VLLM_INTEGRATION=1 FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=<dataset> python <harness>.py ...`
- **Warm every arm, then time alternatives in interleaved rounds and compare medians.**
  The harness does this and reports the spread; a delta smaller than the spread is not a
  result.
- **Size the run so each arm's timed window is several seconds.** The patched arm pays a
  one-time SYCL build and JIT. Raise `--prompts` and `--out-tokens`; confirm the delta holds
  when the window is doubled.
- **To decide whether an opt-in integration flag should become the default**, set it on
  the patched arm only with `--env KEY=VALUE`.

## Step 3: Read the dispatch counters — the validity gate

At exit the adapters print:

```
[flashinfer-bench] adapter dispatch:
  <family>: N call(s), M applied (P%)
  detail: {'<family> applied h<N>': ..., '<family> no-solution d<N>': ...}
```

| Detail string | Meaning |
| --- | --- |
| `applied h<N>` | substituted; the timing counts |
| `no-solution h<N>` | nothing matched that shape — Step 6 |
| `unsupported <reason>` | the adapter declined (3D input, odd width); expected |

A throughput number without these is uninterpretable. If `applied` is 0 everywhere, fix
that before reading the timing.

## Step 4: Report share and ratio together

```
<family>  <ratio>x per kernel  x  <share>% of device time  ->  expected  (measured: <x>%)
```

- **The ratio is against what the serving stack runs, not the definition's reference.**
  `scripts/rank_vs_provider.py` computes it from the traces on disk against the provider
  baseline (`add-baselines --providers vllm-xpu`) and subtracts the dispatch cost.
- **Compare the kernel's runtime against what dispatch costs.** A successful substitution
  costs a fixed amount of Python per call; a miss is nearly free. Read both from
  `flashinfer_bench.device.calibration.get()`. A family is worth substituting only where
  its kernel time is large against that cost — GEMM, attention, prefill-sized elementwise
  work — not decode-sized norms and activations, whatever their per-kernel ratio.
- **Mind the timing floor.** Device-event timing has a fixed cost (`timing_floor_us` in the
  calibration); below it per-kernel ratios are noise. Sanity-check a small-batch ratio
  against bytes moved divided by the calibration's bandwidth.
- **Check the family is not launch-bound.** If recorded latency barely moves across the
  batch sweep, the time is launch overhead and no kernel can win it, while `apply()`'s
  per-call dispatch is still paid.
- **The share must come from a serving profile.** A `transformers` profile has no paged
  attention and no `reshape_and_cache`, and its `aten::cat` time is `DynamicCache`
  concatenation. Take the share from `/profile-intel` against the serving stack, or from
  this harness's own profile.

## Step 5: Decide what the number means

Report "worth +X% on this model" and follow the share to the next family. On Intel, dense
GEMM is oneDNN whether or not we are involved, so when GEMM dominates the reachable
headroom for everything else is small by construction.

- **Gate deployment on the margin, not the ratio.** `ApplyConfig(min_gain_us=...)`
  (`FIB_APPLY_MIN_GAIN_US` in the `sitecustomize.py` above) indexes a shape only where the
  solution beats the provider baseline by more than a substitution costs. Default is `0.0`;
  set it from the calibration. Shapes are judged individually; once any shape is rejected,
  `def_best` is withheld for that definition. A definition with no provider baseline is
  left alone.
- **Measure the dispatch tax with `--overhead-arm`.** A third arm, patched but pointed at
  an empty dataset: every interception runs, nothing matches. A reported win is net of this
  tax; a regression on a model with no matching definitions is this tax, not a kernel
  result. Report the two separately.
- **A regression with 0% substitution and a near-zero overhead arm is the lookup itself.**
  A definition that exists but cannot match (wrong dtype, shape outside the recorded keys)
  runs the full resolve path on every call; a definition that does not exist
  short-circuits. Read `dispatch` and `kernels` separately.
- **A regression at 100% substitution is a real result.** `apply()` keys on the exact axis
  values in the recorded workloads and `use_def_best` extends the winner to every shape the
  scheduler presents; a kernel that wins at a recorded prefill M can lose at decode M. Do
  not retune the kernel: record workloads at the shapes serving uses, or narrow the miss
  policy. Report the regression.

## Step 6: A `no-solution` line is a work item

Every `no-solution <shape>` is a shape the serving stack asked for and got nothing for. For
shape-parametric families (activations, norms) the missing piece is the **definition**, not
a kernel — the in-tree templates and provider baselines are generated per definition.
Extraction cannot supply these (inline activations and fused add+norm are not
`nn.Module`s). Procedure and flags: `references/fill-serving-gaps.md`.

## Failure table

| Symptom | Fix |
| --- | --- |
| `Engine core initialization failed ... Failed core proc(s): {}` on the **patched arm only** | add `if __name__ == "__main__":` to the harness — vLLM re-imports the module in a subprocess |
| The same message on **both arms** | the model did not load; read the worker traceback in `tmp/serving-win/<model>.<arm>.fail.log`, not the tail |
| No `patched vLLM:` line | set the env vars in the environment, not in the launcher's own process |
| `applied` is 0 for every family | check `FIB_DATASET_PATH`; run `add-baselines --in-tree` for the model's definitions |
| `BuildError: ... No such file or directory: 'ninja'`, then `applied` near 0 | activate the venv rather than naming its `python` by absolute path, so `bin/` is on `PATH` |
| `NO dispatch counters` although the worker printed them | match the counter line anywhere in the line; worker output is prefixed `(EngineCore pid=NNN) ` |
| Throughput drops on a model where every family reports 0 applied | `--overhead-arm`; this is `apply()`'s interception cost, not a kernel |
| Throughput moves more than `share x ratio` | `ignore_eos=True`, same prompts, same `max_tokens`, warm up before timing |

## Sources

- `scripts/measure_serving_win.py` — the harness
- `scripts/rank_vs_provider.py` — ranks solutions against the provider kernel and subtracts
  the substitution cost; a definition absent from its output has no provider baseline
- `scripts/calibrate_part.py` — the dispatch cost, timing floor and bandwidth for this part
- `references/fill-serving-gaps.md` — turning `no-solution` reports into definitions
- `/find-kernel-gaps` — hot ops no definition covers
- `/profile-intel` — where the share comes from
- `../optimize-intel-kernels/architectures.md` — the `apply()` deployment traps
