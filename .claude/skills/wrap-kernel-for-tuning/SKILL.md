---
name: wrap-kernel-for-tuning
description: Make a kernel that lives inside a serving stack measurable and optimizable without extracting it — a small harness that imports and calls the production kernel, the three checks that prove the harness did not change the problem, and the trial loop that searches from there. Use before optimizing any vLLM or SGLang kernel, and whenever a tuning loop needs a baseline that is the real thing rather than a copy.
---

# Wrap a production kernel for tuning

A kernel you want to optimize usually lives inside a serving engine, reached through a long
signature whose arguments derive from objects that do not exist outside it. The obvious move
is to lift it out. **Do not.** A reconstruction that is subtly wrong still compiles, still
runs, and still benchmarks — you get a number for a problem nobody has.

You do not need the source. A tuning harness needs a *callable* and *inputs*, and the
production kernel is already callable.

## The contract

Three names in one file. Nothing else, and no benchmarking package required -- the loop
below imports the file and calls it, so a harness stays usable whatever happens to any
external runner. The shape matches the KernelBench convention, which costs nothing and makes
a harness portable to those runners, but nothing here depends on them.

```python
class Model(nn.Module):
    def __init__(self, ...): ...
    def forward(self, *tensors): ...     # calls the production kernel

def get_inputs():      -> list[Tensor]   # one realistic call
def get_init_inputs()  -> list           # constructor args
```

plus a spec YAML naming the input shapes, dtypes and the dimensions they are built from.

**`forward` imports the production kernel and calls it.** Import inside `forward`, not at
module scope, so the file can be read and analysed on a machine without the engine
installed.

```python
def forward(self, q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    out = torch.empty_like(q)
    unified_attention(q=q, k=k_cache, v=v_cache, out=out, ...)
    return out
```

That file is then the baseline a trial has to beat. A trial replaces the body with its own
implementation and is measured against the real kernel, not a copy of it.

## SYCL and oneDNN go through the same loop

Nothing in the loop is Triton-specific: it imports a file exposing the three names and times
it. `tools/kernel-harness/sycl_harness.py` closes the gap from source text to a callable.

```python
from sycl_harness import build, inputs_for

SOURCE = r"""...sycl..."""

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.run = build("rmsnorm_h2560", SOURCE, entry_point="k.cpp::run")
    def forward(self, x, w):
        out = torch.empty_like(x)
        self.run(x, w, out)          # destination-passing, as every in-tree kernel expects
        return out

get_inputs = inputs_for("rmsnorm_h2560", batch_size=4096)
```

Two things this buys. `build` compiles through **the same builder the benchmark uses**, so a
trial that wins is already a Solution -- no reimplementation between tuning and deployment,
and no chance of the two disagreeing about the calling convention. And `inputs_for` takes
shapes and dtypes from the definition, so a trial cannot quietly optimize a problem the
dataset does not contain; only the variable axes are supplied.

**oneDNN needs no separate path.** It is SYCL with `dependencies=["onednn"]`, passed to
`build`. Same loop, same benchmark, same correctness gate.

## Where the shapes come from

From a definition in the dataset, not from imagination: its constant axes are the model's
real head counts, hidden sizes and page sizes, and its workloads carry batch sizes that were
observed. Name the definition in a comment so the harness can be traced back to it.

Prefer a decode-sized batch *and* a prefill-sized one. A kernel that wins at one and loses at
the other is the normal case, not the exception, and a harness at a single shape will not
show it.

## The three checks — run all of them before trusting the harness

A harness that changed the problem is the failure this procedure exists to prevent, and it
does not announce itself.

| check | what it catches |
| --- | --- |
| Output shape and `isfinite` | wiring errors, wrong argument order |
| Result vs an independent reference computed over the same inputs | a harness that computes a *different* function |
| **Latency matches the same call made through the engine's own API** | a harness that computes the right thing on the wrong-sized problem |

The third is the one people skip and the one that catches a silently rescaled workload.
Measured on Arc B580 for vLLM's unified attention at `gqa_paged_decode_h32_kv8_d128_ps64`,
batch 64, ctx 1024 (2026-09-08): the wrapper measured 1636.8us against 1634.7us through
vLLM's own call path, and matched a PyTorch attention over the same paged cache to 0.0020
absolute — bf16 resolution.

If the latencies disagree by more than a few percent, the harness is not running what you
think. Fix that before optimizing anything.

## Running the loop

Read `tools/kernel-harness/knowledge/README.md` before the first trial. It is short and it
is not a style guide: it carries the measured numbers that decide whether a direction is
worth a trial — what a substitution costs, where the instrument's floor is, which directions
have already been tried here and what came of them.

`scripts/kernel_trials.py` is the loop. Trials form a tree: each records its parent and the
strategy that produced it, so a regression branches back to the best node instead of
compounding forward.

```bash
python scripts/kernel_trials.py init      <name> <harness.py>
python scripts/kernel_trials.py save      <name> <candidate.py> --parent t2 --strategy "..."
python scripts/kernel_trials.py benchmark <name> <candidate.py> --trial t3
python scripts/kernel_trials.py status    <name>      # the tree, and which node to branch from
python scripts/kernel_trials.py finalize  <name> <output.py>
```

`benchmark` is the only sanctioned way to get a number, and it is built so the searcher
cannot get it wrong:

- **Correctness gates timing.** A candidate that is wrong, wrongly shaped, or non-finite
  never records a latency.
- **Both arms are warmed before either is timed**, then timed in interleaved rounds with the
  median reported. A GPU that has been idle ramps its clocks, so a sequential sweep charges
  the ramp to whichever case runs first -- that inverted one comparison in this repo from
  0.53x to 1.09x.
- **Many calls per timed region**, so per-call launch and event overhead amortizes rather
  than becoming the measurement. Timing one call at a time reported a 4.9us kernel as 40.8us.
- **A difference smaller than the candidate's own scatter is flagged**, in either direction.

Discipline that matters more than the search strategy:

- **Never write your own benchmark script.** Use `kernel_trials.py benchmark`. Every
  measurement error in this repo's Intel work came from ad-hoc timing, and each one was
  discovered only because something else contradicted it.
- **Run every trial the budget allows.** Stopping at a plateau ends the search where it is
  hardest, which is where the interesting rewrites are.
- **Branch, do not walk.** Improved, continue from here; regressed, go back to the best
  trial and try a different strategy; incorrect, fix in place; plateau, change the algorithm
  rather than its launch geometry.
- **One GPU means one benchmark at a time.** Concurrent measurements contend and the numbers
  are worthless.

## What a win has to clear

Optimizing a kernel is not the same as being worth deploying. Before spending trials, read
`/measure-serving-win` for the two gates: the kernel must beat the **provider** baseline
(not the definition's reference), and it must save more than a substitution costs
(~5.9us on Arc B580), or the exchange loses even when the kernel wins.

And check what the access pattern allows before assuming a gap is real — read the same bytes
in the same shape the kernel is obliged to touch. Peak bandwidth is not the ceiling; the
layout usually is.

## Failure table

| Symptom | Cause | Fix |
| --- | --- | --- |
| Harness runs, latency far below the engine's | the workload was silently rescaled — smaller batch, shorter context, fewer pages | compare against the engine's own call before anything else |
| Correct output, latency far above the engine's | an extra copy or allocation in `forward`, or a sync per call | preallocate outputs; never synchronize inside `forward` |
| Capture-based harness has no tensors | Triton passes kernel arguments positionally, so reading only `kwargs` yields constexprs and no data | map positions through the decorated function's signature; refuse a capture with no tensors |
| Trials all fail to compile | the harness imports something the trial's replacement does not provide | keep generated kernels self-contained; inline helpers |

## Sources

- `tools/kernel-harness/knowledge/README.md` — what a trial should know before it starts
- `scripts/kernel_trials.py` — the trial loop: correctness-gated, interleaved benchmark,
  trial tree with parents and strategies
- `tools/kernel-harness/vllm_unified_attention.py` — a worked wrapper, with its spec YAML beside it
- `scripts/capture_triton_kernel.py` — records a real launch when shapes must come from
  production rather than from a definition. It captures but does **not** verify the capture
  replays; treat its output as a starting point
- `/measure-serving-win` — whether the win is worth deploying
- `../optimize-intel-kernels/SKILL.md` — how to write the replacement once a gap is proven
