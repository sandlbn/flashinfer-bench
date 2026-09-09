# Plan: the router

Not a skill yet. This is the design for one, written after a full pipeline run whose
outcome was decided before any kernel was written.

## The problem this exists to fix

We picked `fused_add_rmsnorm_residual_h1024` by hand, wrote a SYCL kernel that is genuinely
**2.11x** faster than vLLM's own (4.69us vs 9.90us), passed every correctness gate, and
measured **-11.99%** serving throughput on Qwen3-0.6B.

Nothing in that sequence was a mistake in execution. The kernel is good. The target was
wrong, and it was knowable that it was wrong before the first line of SYCL:

```
saves   9.90 - 4.69 = 5.21us per call
costs   6.02us per call        <- calibration.dispatch_us, measured on this part
net     -0.81us per call, at 0.4% substitution, against a 5.13% interception tax
```

A router is the component that computes those four lines *first*, from measurement, given
only the constraints. Here the constraints are **a model and a serving stack** (vLLM).

## What "no hardcoding" means, concretely

The router may not contain:

| Forbidden | Because | Comes from instead |
| --- | --- | --- |
| definition names, hidden sizes, head dims | ties the router to models we happened to try | Stage A observation |
| an op-family priority list ("norms first") | encodes yesterday's profile as tomorrow's policy | Stage B ranking |
| dispatch cost, timing floor, bandwidth as literals | wrong on the next part, silently | `calibration.get()` |
| a provider preference table | encodes a guess about coverage | Stage C capability probe |
| hardware names, SYCL target strings | breaks on new silicon | `Capabilities`, device IP |

Everything the router knows is a measurement taken on the box it is running on, of the
model it is given, under the stack it is given.

## Stage A — candidates come from the run, never from a list

Run the model under the serving stack and record what executed. Three observers exist:

- `scripts/harness_from_model.py` — TorchDispatchMode; every op with real shapes, dtypes,
  call counts, and a verified harness that calls *the same op the model called*
- `scripts/observe_triton_kernels.py` — the Triton kernels the stack actually JITs
- `/profile-intel` — device-time share

And one that is free, because it comes from the previous serving run:
`measure_serving_win.py`'s `no-solution <shape>` counters are candidates the stack asked
for **by name** and got nothing for.

A candidate row is `(op, shape, dtype, calls, device_time_share)`. Nothing named it in
advance. **Missing piece: one command that merges the observers into one table.**

## Stage B — bound every candidate before writing anything

This stage does not exist. It is the whole point.

```
achievable_us = bytes_moved(shape, dtype) / calibration.bandwidth_gbps
headroom_us   = current_us - max(achievable_us, calibration.timing_floor_us)
net_gain_us   = headroom_us - cost_of(mechanism)
net_share     = net_gain_us * calls / total_device_time
```

`net_share <= 0` means **unroutable**: no kernel, however good, wins it. Drop it and say
why. This is the check that would have cost ten minutes and saved the day above.

Two things it must get right:

- **Bound per delivery mechanism, not once.** `cost_of` is `calibration.dispatch_us` for an
  `apply()` substitution and **zero** for a source rewrite, a provider swap, or tuning the
  kernel the stack already launches. A candidate can be unroutable through `apply()` and
  routable through a rewrite. Collapsing these is how the pipeline concluded "elementwise is
  hopeless on Intel" when what is hopeless is *substituting* elementwise.
- **`max(achievable, floor)`.** Below the timing floor a per-kernel ratio is noise, and a
  ratio measured there will motivate work that returns nothing.

The general shape of the Intel answer falls out of this arithmetic rather than being
asserted: GEMM is ~82% and already oneDNN, so `current ~ achievable` and the ceiling is
thin; elementwise has ratio headroom but each call is smaller than a substitution costs.
The router recomputes that per part instead of us re-learning it per kernel.

## Stage C — route to a producer by capability probe

A probe, not a table. Each producer answers at runtime: *can you cover this signature?*

| Producer | Probe | Pays dispatch |
| --- | --- | --- |
| rewrite in place | materialised contraction? (rank>=5 `mul` feeding `sum`) | no |
| library already owns it | oneDNN/oneMKL covers this op_type | no |
| tune the kernel already launched | an observer captured a real launch | no |
| provider kernel | `find_baselines(definition)` non-empty | only via `apply()` |
| new SYCL / Triton | always available | yes |

Ordering is **not** written down. Producers that do not pay dispatch have a strictly higher
Stage B ceiling, so they rank first by arithmetic. That is the non-hardcoded form of "try
oneDNN before hand-writing a GEMM".

## Stage D — build under the trial loop

`scripts/kernel_trials.py`. Correctness gates timing; both arms warmed then interleaved;
branch back to best on regression. The baseline is the **provider harness** for that
signature, never the definition's PyTorch reference — an 8.95x against the reference and a
2.11x against the provider were the same kernel.

## Stage E — admit per shape

`min_gain_us`, read from calibration, applied per shape. `def_best` withheld once any shape
is rejected. Already implemented in `flashinfer_bench/apply/table.py`.

## Stage F — prove end to end, with the overhead arm

`measure_serving_win.py --overhead-arm` — a third arm, patched but pointed at an empty
dataset. Report `dispatch` and `kernels` separately. **This number is the router's output.**
The per-kernel ratio is not a result.

## Stage G — feed back

Serving emits `no-solution <shape>` -> Stage A candidates -> `fill_serving_gaps.py` turns
them into definitions -> next iteration. Closed loop, no human naming anything.

## Why it survives new silicon

Nothing names a part. `calibration.measure()` runs on the new box; `Capabilities` supplies
the target; leaving `sycl_target` unset covers silicon with no AOT target name. Stage B's
verdicts change on their own: where dispatch is cheaper or bandwidth lower, different
candidates become routable. Bringing up a new chip is running the same script.

## What to build

| Stage | State | Work |
| --- | --- | --- |
| A | 3 observers exist | merge them into one candidate table |
| B | **absent** | `scripts/route_kernel_work.py`: bytes-moved per op, per-mechanism bound, ranking |
| C | `find_baselines`, contraction test exist | the probe interface over them |
| D | `kernel_trials.py` | — |
| E | `apply/table.py` | — |
| F | `measure_serving_win.py` | — |
| G | `fill_serving_gaps.py` | — |

Then one entry point taking `--model` and `--stack` and nothing else, and this file becomes
`SKILL.md`.
