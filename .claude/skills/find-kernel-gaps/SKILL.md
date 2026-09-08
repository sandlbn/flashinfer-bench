---
name: find-kernel-gaps
description: Find hot operations a model spends time in that no kernel covers, decide whether each is a cheap rewrite or a real kernel, and turn the worthwhile ones into a definition plus a solution. Use after /profile-intel when a large share sits in ops the router cannot route, or on any model whose architecture is not a plain transformer.
---

# Find kernel gaps

`/profile-intel` ranks families that already have definitions. This answers the prior
question: **what is burning device time that nothing covers**, and of that, what is worth
building.

The headline result from this procedure: on Zamba2-1.2B the largest single consumer was an
`aten::sum` at 27.8% of device time, and it needed **no kernel at all** — it was a batched
GEMM written as broadcast-multiply-then-sum. Rewriting it was **107x** faster.

## Step 1: List the hot ops with their shapes

```bash
python scripts/find_kernel_gaps.py --model <hf_repo_id> --device xpu:0
```

Shapes are the whole point, so this profiles with `record_shapes=True` and reports on the
`aten` rows rather than kernel names. A Mamba2 scan and an RMSNorm produce the *same* kernel
name (`ReduceKernel<1, ReduceOp<float>>`); only operand rank tells them apart.

Each row is classified as `represented`, `no definition covers this op`, or `REWRITE: ...`.

## Step 2: Recognise the rewritable shapes

Most large gaps in eager model code are one of these, and none needs a new kernel:

| Signature | What it really is | Fix |
| --- | --- | --- |
| `sum` over a rank-5+ tensor | a contraction materialised | `torch.bmm` / `einsum` |
| `mul` on rank-5+ feeding a `sum` | the other half of that contraction | fold into the same GEMM |
| `cat` / `stack` on a hot path | a materialising concatenation | layout change, or fuse the producer |
| `copy_` / `contiguous` at high call counts | a transpose forced a copy | transform weights once at load |

The test for the first two: if the output is a sum over one axis of a broadcast product,
write the index equation. `G[b,i,j,h] = Σₛ C[b,i,h,s]·B[b,j,h,s]` is a batched matmul over
`(b,h)`, and the rank-5 intermediate exists only because nobody wrote it that way.

`cumsum` is **not** in this family despite the name — a scan is not a contraction.

## Step 3: Find the source line

The profile gives the op and shapes; the model file gives the expression.

```bash
F=$(python -c "import transformers,os;print(os.path.dirname(transformers.__file__))")
grep -n "\.sum(dim=\|einsum\|\[..., None\]" $F/models/<arch>/modeling_<arch>.py
```

Broadcast indexing (`x[:, :, None, :, :]`) next to a `.sum(dim=...)` is the tell.

## Step 4: Prove equivalence and measure the ceiling — before building anything

Two checks, in this order. Skipping either wastes days.

```python
naive, rewritten = ..., ...
assert torch.allclose(naive(), rewritten(), atol=1e-3, rtol=1e-3)   # same function?
# then time both with the accelerator's timer, one process, medians
```

If the gap is small, stop: you have learned the op is fine. If the rewrite is not exactly
equivalent, stop: you have a different function, not an optimisation.

## Step 5: Make it a definition and a solution

The rewrite belongs in the dataset so it is measured under the same gates as everything
else, and so the next model that hits it inherits the answer.

- **Definition**: the `reference` is the **naive form**, because that is what the framework
  is measured against and what the model actually does today. Constants from the model's
  `config.json`; naming per `/extract-kernel-definitions` B2.
- **Solution**: the rewrite. `language: "python"` is legitimate when the win comes from
  dispatching to an existing library rather than from new device code — a `bmm` solution
  routes to oneDNN and needs no SYCL.

Worked example in the dataset: `mamba_ssu/ssd_bc_contraction_c256_h64_s128`, with
`__bmm_no_materialise` as its solution.

## Step 6: Validate, benchmark, re-profile

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

Then re-run `/profile-intel`. If the op's share has not moved, the rewrite is not on the
path the model actually takes, whatever the benchmark said.

## What this procedure is not for

**Uncovered is not the same as important.** On the same model the extractor flagged
`Conv1d x38` and `SiLU x38` as unhooked; measured, they were **0.2%** and **0.3%** of device
time. A 10x win there returns half a percent, while the op nobody flagged was 27.8%.

Always take the share from the profile, never from what a tool happened to notice.

## The gap this procedure cannot see

Everything above ranks ops that *appear in a profile*. A family can be entirely absent from
one and still be a gap: `transformers` runs the eager form, so a fusion the serving stack
performs (a gated activation as one kernel, a fused residual-add + norm) never shows up as
the op the server will actually call.

Those surface only from a serving run's dispatch counters, as `no-solution <shape>` —
see `/measure-serving-win`, whose Step 6 turns them into definitions. Run both: this
procedure finds what is hot and uncovered, that one finds what the server asked for and was
refused.

## Sources

- `scripts/find_kernel_gaps.py` — the analysis above
- `/profile-intel` — ranks the families that *are* covered
- `/optimize-ssm-scan` — the state-space case in depth
- `/extract-kernel-definitions` B2 — naming and axis rules
