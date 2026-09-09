---
name: wrap-kernel-for-tuning
description: Make a kernel that lives inside a serving stack measurable and optimizable without extracting it — a harness that imports and calls the production kernel, the checks that prove the harness did not change the problem, and the trial loop that searches from there. Use before optimizing any vLLM or SGLang kernel.
---

# Wrap a production kernel for tuning

Do not lift a kernel out of its engine to tune it; a reconstruction that is subtly wrong
still benchmarks. A tuning harness needs a *callable* and *inputs*, and the production
kernel is already callable.

## The contract

Three names in one file. The loop imports the file and calls it; no benchmarking package is
required. The shape matches the KernelBench convention.

```python
class Model(nn.Module):
    def __init__(self, ...): ...
    def forward(self, *tensors): ...     # calls the production kernel

def get_inputs():      -> list[Tensor]   # one realistic call
def get_init_inputs()  -> list           # constructor args
```

plus a spec YAML naming the input shapes, dtypes and the dimensions they are built from.

Import the production kernel **inside `forward`**, not at module scope, so the file can be
read on a machine without the engine installed.

```python
def forward(self, q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    out = torch.empty_like(q)
    unified_attention(q=q, k=k_cache, v=v_cache, out=out, ...)
    return out
```

That file is the baseline a trial has to beat. A trial replaces the body with its own
implementation.

## SYCL and oneDNN go through the same loop

`tools/kernel-harness/sycl_harness.py` closes the gap from source text to a callable:

```python
from sycl_harness import build, inputs_for

SOURCE = r"""...sycl..."""

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.run = build("<definition>", SOURCE, entry_point="k.cpp::run")
    def forward(self, x, w):
        out = torch.empty_like(x)
        self.run(x, w, out)          # destination-passing, as every in-tree kernel expects
        return out

get_inputs = inputs_for("<definition>", batch_size=4096)
```

`build` compiles through the same builder the benchmark uses, so a winning trial is already
a Solution. `inputs_for` takes shapes and dtypes from the definition; only the variable
axes are supplied. oneDNN is SYCL with `dependencies=["onednn"]` passed to `build`.

## Where the shapes come from

From a definition in the dataset: its constant axes are the model's real head counts,
hidden sizes and page sizes, and its workloads carry observed batch sizes. Name the
definition in a comment. Use a decode-sized batch *and* a prefill-sized one.

## The three checks — run all of them before trusting the harness

| check | what it catches |
| --- | --- |
| Output shape and `isfinite` | wiring errors, wrong argument order |
| Result vs an independent reference computed over the same inputs | a harness that computes a different function |
| **Latency matches the same call made through the engine's own API** | a harness that computes the right thing on the wrong-sized problem |

If the latencies disagree by more than the run-to-run scatter of either, the harness is not
running what you think.

## Running the loop

Read `tools/kernel-harness/knowledge/README.md` before the first trial: the rules, and
where the per-part numbers (substitution cost, timer floor, bandwidth) come from.

`scripts/kernel_trials.py` is the loop. Trials form a tree: each records its parent and the
strategy that produced it, so a regression branches back to the best node.

```bash
python scripts/kernel_trials.py init      <name> <harness.py>
python scripts/kernel_trials.py save      <name> <candidate.py> --parent t2 --strategy "..."
python scripts/kernel_trials.py benchmark <name> <candidate.py> --trial t3
python scripts/kernel_trials.py status    <name>      # the tree, and which node to branch from
python scripts/kernel_trials.py finalize  <name> <output.py>
```

`benchmark` is the only sanctioned way to get a number. It gates timing on correctness,
warms both arms before timing either, times in interleaved rounds with the median reported,
runs many calls per timed region so launch and event overhead amortize, and flags a
difference smaller than the candidate's own scatter.

- **Never write your own benchmark script.**
- **Run every trial the budget allows.** The interesting rewrites are past the plateau.
- **Branch, do not walk.** Improved: continue from here. Regressed: go back to the best
  trial and try a different strategy. Incorrect: fix in place. Plateau: change the
  algorithm rather than its launch geometry.
- **One GPU means one benchmark at a time.**

## What a win has to clear

Before spending trials, read `/measure-serving-win`: the kernel must beat the **provider**
baseline (not the definition's reference), and it must save more than a substitution costs
(`flashinfer_bench.device.calibration.get().dispatch_us`; `scripts/calibrate_part.py`
prints it). And check what the access pattern allows before assuming a gap is real — read
the same bytes in the same shape the kernel is obliged to touch.

## Failure table

| Symptom | Fix |
| --- | --- |
| Harness runs, latency far below the engine's | the workload was rescaled — compare against the engine's own call |
| Correct output, latency far above the engine's | preallocate outputs; never synchronize inside `forward` |
| Capture-based harness has no tensors | Triton passes kernel arguments positionally — map positions through the decorated function's signature; refuse a capture with no tensors |
| Trials all fail to compile | keep generated kernels self-contained; inline helpers |

## Sources

- `tools/kernel-harness/knowledge/README.md` — what a trial should know before it starts
- `scripts/kernel_trials.py` — the trial loop
- `scripts/harness_from_model.py` — emits these wrappers from the run itself, including
  for ops that read state the stack establishes around each step; a wrapper written by
  hand is a claim about production that nothing verified
- `scripts/capture_triton_kernel.py` — records a real launch when shapes must come from
  production rather than from a definition; it does not verify the capture replays
- `/measure-serving-win` — whether the win is worth deploying
- `../optimize-intel-kernels/SKILL.md` — how to write the replacement once a gap is proven
