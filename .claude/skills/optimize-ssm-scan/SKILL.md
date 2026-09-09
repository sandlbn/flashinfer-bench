---
name: optimize-ssm-scan
description: Optimize state-space / SSD scan kernels on Intel GPUs — Mamba2, GDN, and the hybrid models built on them (Zamba, Granite-hybrid, Falcon-H, Jamba). Use when /profile-intel reports materialised high-rank contractions, or when a hybrid model's time is in aten::sum / aten::mul over rank-5+ tensors.
---

# Optimize a state-space scan

On a hybrid model, `/profile-intel` says whether the scan or the GEMM carries the device
time; read it before assuming either.

## Recognising it

`/profile-intel` reports it directly:

```
!! Materialised high-rank contractions: <share>% of device time
   aten::sum   <ms>  <share>% x<calls>   rank=6 <bytes>/call  [B, ?, chunk, chunk, heads, state]
   aten::mul   <ms>  <share>% x<calls>   rank=6
```

You cannot see this from kernel names: a Mamba2 scan and an RMSNorm both surface as
`ReduceKernel<1, ReduceOp<float>>`; only the operand rank on the `aten` row separates them.

The shape reads as the SSD chunked form `[batch, ?, chunk, chunk, heads, state]`, with
`chunk = mamba_chunk_size`, `heads = n_mamba_heads`, `state = mamba_d_state` from the
model's `config.json`. The intermediate's size is the product of those dims times the
itemsize.

## The diagnosis

`transformers` implements the scan as a broadcast multiply followed by a sum, materialising
the full `chunk x chunk x heads x state` intermediate. The target is **removing the
intermediate**, not a faster reduction.

## Where a kernel can come from, in order

1. **A registered provider kernel** — `sgl-kernel-xpu`'s `gdn_attention`, and vllm-xpu's
   `gated_delta_rule_non_spec` (bound to `gdn` against the `_l2norm` definition variants).
   Add them as baselines first (`/onboard-model-intel` Phase 5). Ask the registry for what
   is bound today; enumerate the source for what exists:
   ```bash
   grep -ohE '"[a-z_0-9]*gdn[a-z_0-9]*\(' tmp/sgl-kernel-xpu/src/torch_extension_sycl.cc
   ```
   `gdn_attention_workspace_bytes_needed`: this kernel wants a caller-provided workspace,
   so the wrapper must allocate it.
2. **An existing definition.** The dataset carries `gdn` and `mamba_ssu`:
   ```bash
   ls tmp/flashinfer-trace/definitions/gdn tmp/flashinfer-trace/definitions/mamba_ssu
   ```
   Naming (`/extract-kernel-definitions` B2): `mamba_ssu_decode_h{n}_d{d}_s{s}_ng{g}` and
   `gdn_{decode,mtp,prefill}_qk{q}_v{v}_d{d}_k_last`. Match your config's head/state
   counts before writing a new one.
3. **Your own SYCL kernel.** `/optimize-intel-kernels` Step 4a. Fuse multiply and reduce so
   the intermediate stays in registers.

## Getting a definition at all

Path C (`scripts/extract_model_kernels_xpu.py`) hooks RMSNorm and Linear only, so it emits
nothing for the scan; it reports the leaf modules it skipped. Take the scan through
**Path B** (`/extract-kernel-definitions` section B): read the model's `modeling_*.py`,
take the constants from `config.json` (`mamba_d_state`, `mamba_headdim`, `n_mamba_heads`,
`mamba_n_groups`, `mamba_chunk_size`), and copy the nearest sibling definition.

For the `reference`, FlashInfer's `gdn_decode.py` / `gdn_prefill.py` and `mamba/` are the
CUDA implementations to mirror — read them for the maths, not to call them.

Take the share from the profile, not from the extractor's list of unhooked modules.

## Validate and benchmark

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>
flashinfer-bench add-baselines --local tmp/flashinfer-trace --providers vllm-xpu,sgl-kernel-xpu --definitions <name>
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
```

Attach prefill-sized workloads before believing any ratio — Path C emits decode shapes
only. Then re-run `/profile-intel`; the contraction line should shrink or disappear.

## Sources

- `tmp/flashinfer/flashinfer/gdn_decode.py`, `gdn_prefill.py`, `mamba/` — CUDA references
- `tmp/sgl-kernel-xpu/src/torch_extension_sycl.cc` — `gdn_attention` signature
- `/extract-kernel-definitions` B2 — naming and axis rules
- `/optimize-intel-kernels` — writing the SYCL or Triton kernel
