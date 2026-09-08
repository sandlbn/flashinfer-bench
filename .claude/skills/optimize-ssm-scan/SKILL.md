---
name: optimize-ssm-scan
description: Optimize state-space / SSD scan kernels on Intel GPUs — Mamba2, GDN, and the hybrid models built on them (Zamba, Granite-hybrid, Falcon-H, Jamba). Use when /profile-intel reports materialised high-rank contractions, or when a hybrid model is slow and the time is in aten::sum / aten::mul over rank-5+ tensors.
---

# Optimize a state-space scan

On a hybrid model the scan, not the GEMM, is where the time goes. Measured on Zamba2-1.2B
(Arc B580): GEMM was 23.5% of device time and **64.4% sat in materialised high-rank
contractions** — a single `aten::sum` over `[1,1,256,256,64,128]` was 27.8% in 38 calls, at
**~2 GB of intermediate per call**.

## Recognising it

`/profile-intel` reports this directly:

```
!! Materialised high-rank contractions: 64.4% of device time
   aten::sum   289.62 ms  27.8% x38   rank=6 ~2.00 GB/call  [1, 1, 256, 256, 64, 128]
   aten::mul   102.82 ms   9.9% x38   rank=6
```

**You cannot see this from kernel names.** A Mamba2 scan and an ordinary RMSNorm both
surface as `ReduceKernel<1, ReduceOp<float>>`; only the operand rank separates them, and
rank is visible only on the `aten` row. A name-based router will file the scan under "norm"
and hand you the wrong advice — vectorising loads on a 2 GB intermediate is not the fix.

The shape reads as the SSD chunked form: `[batch, ?, chunk, chunk, heads, state]`, with
`chunk = mamba_chunk_size`, `heads = n_mamba_heads`, `state = mamba_d_state` from the
model's `config.json`.

## The diagnosis is always the same

`transformers` implements the scan as a broadcast multiply followed by a sum. That
materialises the full `chunk x chunk x heads x state` intermediate, writes it to memory, and
reads it back to reduce it. A fused chunked scan never creates it.

So the target is **not** a faster reduction. It is removing the intermediate. Vectorising the
existing kernel optimises the symptom.

## Where a kernel can come from, in order

1. **`sgl-kernel-xpu` `gdn_attention`** — the only Intel state-space kernel that exists.
   Wire it as a baseline first (`/onboard-model-intel` Phase 5); it may cover your case
   outright.
   ```bash
   grep -ohE '"[a-z_0-9]*gdn[a-z_0-9]*\(' tmp/sgl-kernel-xpu/src/torch_extension_sycl.cc
   ```
   Note `gdn_attention_workspace_bytes_needed`: this kernel wants a caller-provided
   workspace, so the wrapper must allocate it.
2. **An existing definition.** The dataset already carries `gdn` and `mamba_ssu`:
   ```bash
   ls tmp/flashinfer-trace/definitions/gdn tmp/flashinfer-trace/definitions/mamba_ssu
   ```
   Naming follows `/extract-kernel-definitions` B2 —
   `mamba_ssu_decode_h{n}_d{d}_s{s}_ng{g}` and `gdn_{decode,mtp,prefill}_qk{q}_v{v}_d{d}_k_last`.
   Match your config's head/state counts against those before writing a new one.
3. **Your own SYCL kernel.** `/optimize-intel-kernels` Step 4a. The win is structural — fuse
   multiply and reduce so the intermediate stays in registers — not lane-level tuning.

## Getting a definition at all

**Path C cannot see these ops.** `scripts/extract_model_kernels_xpu.py` hooks RMSNorm and
Linear, so on a hybrid model it emits the norms and projections and nothing for the scan. It
now reports what it skipped:

```
Leaf modules this extractor does not hook: SiLU x38, Conv1d x38, GELUActivation x6.
```

Take the scan through **Path B** (`/extract-kernel-definitions` section B): read the model's
`modeling_*.py`, take the constants from `config.json` (`mamba_d_state`, `mamba_headdim`,
`n_mamba_heads`, `mamba_n_groups`, `mamba_chunk_size`), and copy the nearest sibling
definition rather than starting blank.

For the `reference`, FlashInfer's `gdn_decode.py` / `gdn_prefill.py` and `mamba/` are the
CUDA implementations to mirror — read them for the maths, not to call them.

## Do not be misled by what the extractor flags

Uncovered is not the same as important. On Zamba2 the extractor flagged `Conv1d x38` and
`SiLU x38` as unhooked; measured, they are **0.2% and 0.3%** of device time. A 10x win there
returns half a percent. The 6-D `aten::sum` it did *not* flag is 27.8%.

Always measure before writing a kernel for something the tooling merely noticed.

## Validate and benchmark

Same gates as any other kernel:

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>
flashinfer-bench add-baselines --local tmp/flashinfer-trace --providers sgl-kernel-xpu --definitions <name>
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

Attach prefill-sized workloads before believing any ratio — Path C emits decode shapes only,
and at those sizes repeated runs of the same workload have varied by 74%.

Then re-run `/profile-intel`. The contraction line should shrink or disappear; if the
family's share has not moved, the fusion did not happen whatever the microbenchmark said.

## Sources

- `tmp/flashinfer/flashinfer/gdn_decode.py`, `gdn_prefill.py`, `mamba/` — CUDA references
- `tmp/sgl-kernel-xpu/src/torch_extension_sycl.cc` — `gdn_attention` signature
- `/extract-kernel-definitions` B2 — naming and axis rules
- `/optimize-intel-kernels` — writing the SYCL or Triton kernel
