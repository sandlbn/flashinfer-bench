---
name: onboard-model-intel
description: End-to-end pipeline for supporting a new LLM on Intel GPUs — acquire definitions on xpu:0 without CUDA, cross-validate references, profile, source kernels from oneDNN / vllm-xpu-kernels / sgl-kernel-xpu / Xe-Fuse / SYCL, and triage a slow or wrong op to reference, harness or provider. Use for "support model X on Intel", or when an Intel deployment is slow or wrong.
---

# Onboard a model on Intel

End state: definitions in the dataset, references proven correct on `xpu:0`, a profile that
ranks kernel families by recoverable device time, and a benchmarked solution for the ops
that rank. No CUDA anywhere in this path.

| Phase | Produces | Skip when |
| --- | --- | --- |
| 0 Bring up the box | a working toolchain | already set up |
| 1 Inventory | which definitions this model needs, and which exist | — |
| 2 Acquire definitions | definition JSON + workloads | all definitions already exist |
| 3 Cross-validate | references proven on `xpu:0` | **never skip** |
| 4 Profile and rank | ordered list of what is worth optimizing | you already know the target |
| 5 Source solutions | baselines + your own solution | — |
| 6 Triage | whose fault a slow or wrong op is | nothing is wrong |
| 7 Benchmark and publish | traces + PRs | — |

## Phase 0: Bring up the box

Run `/setup-intel-env`, then `/clone-repos` for `tmp/flashinfer-trace` and the upstream
sources Phase 5 greps. The checks this pipeline depends on:

```bash
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"   # ['xpu:0']
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
flashinfer-bench providers list
```

## Phase 1: Inventory the model's kernels

```bash
python -c "
import json, urllib.request
c = json.load(urllib.request.urlopen('https://huggingface.co/<repo_id>/raw/main/config.json'))
print('architectures', c.get('architectures'))
t = c.get('text_config', c)   # multimodal configs nest the language model under text_config
for k in ('hidden_size','intermediate_size','moe_intermediate_size','num_attention_heads',
          'num_key_value_heads','head_dim','num_hidden_layers','vocab_size',
          'num_experts','num_experts_per_tok','torch_dtype','dtype'):
    if k in t: print(f'{k:24} {t[k]}')
if 'vision_config' in c:
    print('NOTE: vision tower present -- scope it explicitly; this pipeline covers text ops')
"
ls tmp/flashinfer-trace/definitions/*/ | grep -E "h<hidden_size>|d<head_dim>"
```

Naming formulas per op_type are `/extract-kernel-definitions` section B2; `/discover-models`
does the full classification for a model new to the project. Record, per definition:
exists in the dataset, has workloads, and which sources ship a kernel for its op family
("What ships a kernel for which op family", below).

## Phase 2: Acquire definitions

Definitions are hardware-agnostic; one that already exists is reused unchanged:

```bash
ls tmp/flashinfer-trace/definitions/*/<name>.json
```

### Path C (primary on Intel): module hooks on a live `transformers` run

```bash
python scripts/extract_model_kernels_xpu.py \
    --model <repo_id> --output tmp/extract-<slug> \
    --device xpu:0 --max-new-tokens 24 --top-linear 8
```

By default it extracts in the model config's own dtype; `--dtype` overrides. Extract in the
dtype the model is served in — a definition name encodes shape, not dtype, so a wrong-dtype
extraction collides by name with the right one. It emits dataset names directly
(`rmsnorm_h{H}` with `(hidden_states, weight) -> output`, `gemm_n{N}_k{K}` with
`(A, B) -> C`). Verify before staging:

```bash
python -c "
import json,sys,pathlib
for f in sorted(pathlib.Path(sys.argv[1]).rglob('*.json')):
    d=json.load(open(f)); print(d['name'], d['op_type'], list(d['inputs']), '->', list(d['outputs']))
" tmp/extract-<slug>/definitions
```

Move each file to `tmp/flashinfer-trace/definitions/{op_type}/` — the `op_type` field
decides the subdirectory. An existing definition of the same name is reused, never
overwritten. Path C emits workloads alongside its definitions, at TP=1 shapes. It hooks
RMSNorm and Linear only; it reports the leaf modules it did not hook.

### Path B (fallback): transcribe from sources

For kernels Path C does not reach (attention, MoE, KV cache, scans) or for TP/EP variants:
`/extract-kernel-definitions` section B. The HuggingFace modeling file is the ground truth
for what the op computes; `config.json` plus the TP/EP rules give the constant axes.

### Path A (optional): the CUDA harvest

If an NVIDIA box is available, `/onboard-model`, "Generate definitions". Nothing downstream
depends on it.

### Writing the `reference`

Plain PyTorch, float32 accumulation, constants asserted, epsilon as a module-level `EPS`.
Copy the nearest sibling definition.

## Phase 3: Cross-validate the references on `xpu:0`

Correctness is judged against the reference running on this device, so a reference that
misbehaves on XPU validates a wrong kernel.

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace \
    --device xpu:0 --definitions <name>
```

| Status | Action |
| --- | --- |
| `PASSED` | Continue |
| `MISMATCH` | Reduce to a minimal repro, run `run()` on `cpu` and `xpu:0`, `torch.testing.assert_close`, report to pytorch/pytorch with `torch.__version__` and the driver version. Do not work around it |
| `UNSUPPORTED_DTYPE` | The part supports the dtype neither natively nor by emulation. Pick a different target |
| `NO_WORKLOAD` | Attach a workload (Phase 2) |
| `TARGET_ERROR` | The reference uses an op with no XPU implementation. Rewrite it in primitive ops |
| `BUILD_ERROR` / `BASELINE_ERROR` | A solution or baseline failed to build — Phase 5; not a reference problem |

## Phase 4: Profile, then rank

Run `/profile-intel` (ranks families by recoverable time and routes each), then
`/find-kernel-gaps` (what is burning time that no definition covers). Do both before
optimizing anything.

For a raw unitrace pass over the extraction script:

```bash
unitrace --device-timing --chrome-kernel-logging \
    python scripts/extract_model_kernels_xpu.py \
        --model <repo_id> --output tmp/profile-scratch --device xpu:0 --max-new-tokens 64
```

`--output` must be a directory the script can create (`/dev/null` raises). Rank by
recoverable time, `share x (1 - 1/speedup)`, which `scripts/profile_intel.py` computes.

Two validity gates: confirm the run does what you think (greedy vs sampled decode changes
the mix), and remember a `transformers` profile cannot rank attention or KV-cache kernels
the way a serving stack would — profile vLLM-XPU or SGLang-XPU for those.

## Phase 5: Source the solution

### What ships a kernel for which op family

Which source ships a kernel for an op family is a fact about what is built and registered on
this box. It carries no order: wire every source that exists as a baseline ("Add baselines"
below) and let the benchmark rank them.

Source: `providers.md` inventories, `flashinfer-bench providers verify`, and each project's
own op registration

| op_type family | Sources that ship one | Written by hand when none does |
| --- | --- | --- |
| `gemm` (dense projections) | oneDNN — the primitive `F.linear` already executes on this backend; `/optimize-onednn` owns the call | SYCL |
| GEMM + norm/activation fusion | oneDNN **post-ops** where the epilogue is elementwise or binary on one GEMM's output (`/optimize-onednn`, "Post-ops, and the shape of what they can express"); Xe-Fuse where it needs a lane shuffle (`optimize-intel-kernels/xe-fuse.md`) | SYCL |
| `rmsnorm`, `rope`, activations | `vllm-xpu-kernels`, `sgl-kernel-xpu` | SYCL — check `providers.md` for which of these are actually registered before concluding none is |
| `gqa_paged`, `gqa_ragged`, `mla_paged`, `dsa_paged` | `sgl-kernel-xpu` — the only Intel attention kernels; torch SDPA is available as a correctness floor | SYCL |
| `moe`, GroupGemm, W4A16/W8A16 | `sgl-kernel-xpu` | SYCL |
| `gdn`, `mamba_ssu`, any SSD scan | `sgl-kernel-xpu`'s `gdn_attention`; vllm-xpu's `gated_delta_rule_non_spec` | SYCL — `/optimize-ssm-scan` owns the recognition rule and the definition |
| quantization, KV-cache ops | `vllm-xpu-kernels` | SYCL |
| anything else | nothing; the definition's PyTorch reference is the only implementation | SYCL |

Source: each project's build and packaging, and IntelLabs' own stability marking

| Source | Cost | Risk |
| --- | --- | --- |
| oneDNN / oneMKL | ships with oneAPI | lowest; versioned with the toolchain |
| `vllm-xpu-kernels` | prebuilt wheel | low; plain torch custom ops |
| `sgl-kernel-xpu` | long, memory-hungry source build | medium; `bmg`/`cri` only |
| Xe-Fuse | one TU per kernel | high — IntelLabs marks it not stable |
| hand-written SYCL | one compile | yours forever |

### Add baselines

The bar is the kernel a real Intel deployment runs, not PyTorch eager.

```bash
flashinfer-bench add-baselines --local tmp/flashinfer-trace --in-tree --definitions <name>
flashinfer-bench add-baselines --local tmp/flashinfer-trace \
    --providers vllm-xpu,sgl-kernel-xpu --definitions <name>
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
```

No baseline for a kernel a provider ships → wire it: `references/wiring-baselines.md`.
Inventories and signatures: `providers.md`.

### Writing your own solution

`/optimize-intel-kernels` owns this. To search rather than write once, use
`/wrap-kernel-for-tuning`: it wraps the kernel a deployment runs as the baseline and runs
SYCL, oneDNN and Triton candidates against it under one benchmark.

## Phase 6: Triage a slow or wrong op

Three suspects — reference, harness, provider — separated **in this order**:

```
1. flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>
       MISMATCH -> the reference is wrong on XPU. Not a provider problem. Stop.
       PASSED   -> continue

2. Run the provider kernel and the reference on the SAME workload and compare:
       flashinfer-bench add-baselines --local tmp/flashinfer-trace --definitions <name>
       flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --num-trials 1
       flashinfer-bench report summary --local tmp/flashinfer-trace
       INCORRECT -> provider correctness bug -> ladder rung 3/4
       PASSED    -> continue

3. Time three things on the same workload:
       (a) the provider kernel   (b) the reference (eager)   (c) a naive SYCL kernel, untuned
       (a) ~ (c), both near bandwidth -> nobody is at fault; the op is bandwidth-bound
       (a) >> (c)                     -> the provider picked a bad implementation -> /optimize-onednn
       (a) << (c) but still slow      -> needs fusion, not tuning
```

For 3(c), copy `examples/sycl/rmsnorm_sycl.cpp`, change the body, do not tune.

### The escalation ladder

| Rung | Action | Ships |
| --- | --- | --- |
| 1 | Change the call, not the library: memory layout, fpmath mode, post-ops, splitting one primitive into two. Encoded as a Solution | immediately |
| 2 | Override with a SYCL solution for the shape range where the library loses; a solution competes per workload | immediately |
| 3 | Patch the provider locally, pinned to a commit | your box only |
| 4 | Report upstream with a minimal repro built from the definition + workload | on merge |

Do 1 or 2 **and** 4. If you patch a provider, say so in the solution's `description` and the
PR body — the trace format does not record it.

## Phase 7: Benchmark, validate, publish

```bash
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
flashinfer-bench validate --dataset tmp/flashinfer-trace --definitions <name>
```

Traces record `hardware_id` and `environment.libs.timing`; results are never ranked across
devices. Then `/submit-onboarding-prs`.

## Source map

| Question | Read |
| --- | --- |
| What do I install, and what does each package offer? | `/setup-intel-env` |
| What kernels does each provider contain, and how do I read a signature? | `providers.md` |
| How do I register a provider kernel as a baseline? | `references/wiring-baselines.md` |
| Why is this GEMM slow? | `/optimize-onednn` |
| What is hot that nothing covers? | `/find-kernel-gaps` |
| A hybrid/Mamba model is slow | `/optimize-ssm-scan` |
| How do I write a fast SYCL or Triton kernel? | `../optimize-intel-kernels/SKILL.md` |
| What is special about this Intel part? | `../optimize-intel-kernels/architectures.md` |
| GEMM epilogue fusion | `../optimize-intel-kernels/xe-fuse.md` |
| How are in-tree kernels generated per definition? | `flashinfer_bench/integration/intree_kernels.py` |
| Definition naming and axis rules | `/extract-kernel-definitions` section B2 |
