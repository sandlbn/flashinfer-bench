---
name: find-kernel-gaps
description: Find hot operations a model spends device time in that no definition covers, decide whether each is a cheap rewrite or a real kernel, and turn the worthwhile ones into a definition plus a solution. Use after /profile-intel when a large share sits in unroutable ops, or on any model that is not a plain transformer.
---

# Find kernel gaps

`/profile-intel` ranks families that already have definitions. This answers the prior
question: **what is burning device time that nothing covers**, and of that, what is worth
building. A gap that is a contraction written as broadcast-multiply-then-sum needs no
kernel; the checks below say whether yours is one.

## Step 1: List the hot ops with their shapes

```bash
python scripts/find_kernel_gaps.py --model <hf_repo_id> --device xpu:0
```

It profiles with `record_shapes=True` and reports on the `aten` rows rather than kernel
names — a Mamba2 scan and an RMSNorm produce the same kernel name
(`ReduceKernel<1, ReduceOp<float>>`); only operand rank tells them apart. Each row is
classified `represented`, `no definition covers this op`, or `REWRITE: ...`.

## Step 2: Recognise the rewritable shapes

| Signature | What it really is | Fix |
| --- | --- | --- |
| `sum` over a rank-5+ tensor | a contraction materialised | `torch.bmm` / `einsum` |
| `mul` on rank-5+ feeding a `sum` | the other half of that contraction | fold into the same GEMM |
| `cat` / `stack` on a hot path | a materialising concatenation | layout change, or fuse the producer |
| `copy_` / `contiguous` at high call counts | a transpose forced a copy | transform weights once at load |

Test for the first two: if the output is a sum over one axis of a broadcast product, write
the index equation. `G[b,i,j,h] = Σₛ C[b,i,h,s]·B[b,j,h,s]` is a batched matmul over
`(b,h)`. `cumsum` is **not** in this family — a scan is not a contraction.

## Step 3: Find the source line

```bash
F=$(python -c "import transformers,os;print(os.path.dirname(transformers.__file__))")
grep -n "\.sum(dim=\|einsum\|\[..., None\]" $F/models/<arch>/modeling_<arch>.py
```

Broadcast indexing (`x[:, :, None, :, :]`) next to a `.sum(dim=...)` is the tell.

## Step 4: Prove equivalence and measure the ceiling before building anything

```python
naive, rewritten = ..., ...
assert torch.allclose(naive(), rewritten(), atol=1e-3, rtol=1e-3)   # same function?
# then time both with the accelerator's timer, one process, medians
```

Small gap: stop, the op is fine. Not exactly equivalent: stop, it is a different function.

## Step 5: Make it a definition and a solution

- **Definition**: the `reference` is the **naive form** — what the model does today.
  Constants from the model's `config.json`; naming per `/extract-kernel-definitions` B2.
- **Solution**: the rewrite. `language: "python"` is legitimate when the win comes from
  dispatching to an existing library — a `bmm` solution routes to oneDNN.

Worked example in the dataset: `mamba_ssu/ssd_bc_contraction_c256_h64_s128`, with
`__bmm_no_materialise` as its solution.

## Step 6: Validate, benchmark, re-profile

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

Then re-run `/profile-intel`. If the op's share has not moved, the rewrite is not on the
path the model takes.

## Uncovered is not the same as important

The extractor's list of unhooked modules is not a ranking. Take the share from the
profile, never from what a tool happened to notice.

## The gap this procedure cannot see

`transformers` runs the eager form, so a fusion the serving stack performs (a gated
activation as one kernel, a fused residual-add + norm) never shows up as the op the server
will call. Those surface only from a serving run's dispatch counters, as
`no-solution <shape>` — `/measure-serving-win` Step 6. Run both.

## Sources

- `scripts/find_kernel_gaps.py` — the analysis above
- `/profile-intel` — ranks the families that *are* covered
- `/optimize-ssm-scan` — the state-space case in depth
- `/extract-kernel-definitions` B2 — naming and axis rules
