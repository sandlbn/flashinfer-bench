---
name: discover-models
description: Discover candidate LLMs and produce a kernel inventory — the definitions a model needs, classified existing/new and by which backend can supply a kernel (FlashInfer on CUDA, vllm-xpu / sgl-kernel-xpu / oneDNN on Intel). Writes the run manifest the rest of the pipeline consumes. Use as /onboard-model's "Discover and classify" phase, or /onboard-model-intel's "Inventory the model's kernels".
---

# Discover models

Answer two questions and write them down: **which definitions does this model need**, and
**for each, who can supply a kernel**. Output is the run manifest every later phase reads.

## Prerequisites

`/clone-repos` — this skill greps `tmp/sglang`, `tmp/flashinfer`, `tmp/sgl-cookbook` and
`tmp/flashinfer-trace`.

## 1a. Find candidates

Skip when a model was named explicitly.

```bash
# Day-0 SGLang additions: a brand-new model file is the strongest signal
git -C tmp/sglang log --since="30 days ago" --name-status --diff-filter=A \
    -- "python/sglang/srt/models/*.py" | awk '/^A/{print $2}'

# New sgl-cookbook entries mean a recommended serving config exists
COOKBOOK=$(ls -d tmp/sgl-cookbook/data/models/generated/* | sort -V | tail -1)
git -C tmp/sgl-cookbook log --since="30 days ago" --name-status --diff-filter=A \
    -- "${COOKBOOK#tmp/sgl-cookbook/}/*.yaml" | awk '/^A/{print $2}'
```

Always resolve the cookbook directory with `sort -V | tail -1`; a pinned version path goes
stale silently.

Drop candidates already listed in the Summary table of `docs/model_coverage.mdx`.

## 1b. Read the model config

```bash
python -c "
import json, urllib.request
c = json.load(urllib.request.urlopen('https://huggingface.co/<repo_id>/raw/main/config.json'))
print('architectures', c.get('architectures'))
# Multimodal configs (*ForConditionalGeneration) nest the language model here. Reading only
# top-level keys returns nothing for them, silently.
t = c.get('text_config', c)
for k in ('hidden_size','intermediate_size','moe_intermediate_size','num_attention_heads',
          'num_key_value_heads','head_dim','num_hidden_layers','vocab_size',
          'num_experts','num_experts_per_tok','torch_dtype','dtype'):
    if k in t: print(f'{k:24} {t[k]}')
if 'vision_config' in c:
    print('NOTE: has a vision tower -- scope it explicitly; the dataset covers text ops only')
"
```

A gated repo needs `hf auth login`. An unrecognised `architectures` value means the
model is not in SGLang yet — record it and stop; there is nothing to onboard against.

## 1c. Compute the definitions this model needs

The naming formulas and TP/EP division rules live in `/extract-kernel-definitions` section
B2 — that table is the single owner. Apply it to the config values from 1b, once per (TP, EP)
combination in the cookbook YAML.

## 1d. Existing or new

```bash
for name in <computed_names>; do
  if ls tmp/flashinfer-trace/definitions/*/"$name".json >/dev/null 2>&1; then
    echo "existing $name"
  else
    echo "new      $name"
  fi
done
```

An existing definition is reused unchanged. Also check it has workloads —
`ls tmp/flashinfer-trace/workloads/*/$name.jsonl` — because a definition with none cannot be
benchmarked and needs `/collect-workloads` even though Phase 2 can be skipped.

## 1e. Who can supply a kernel

Two independent questions. Answer both; a manifest that answers only the first is useless on
an Intel-only box.

**CUDA / FlashInfer:**

| op_type | Check in `tmp/flashinfer/flashinfer/` |
|---|---|
| `rmsnorm` | `norm/` |
| `gqa_paged` | `decode.py`, `prefill.py` |
| `gqa_ragged` | `prefill.py` |
| `mla_paged` | `mla/` |
| `mamba_ssu` | `mamba/` |
| `dsa_paged` | `sparse.py` |
| `gdn` | `gdn_decode.py`, `gdn_prefill.py` |
| `moe` | `fused_moe/` — check the specific variant |
| `sampling` | `sampling.py` |
| `rope` | `rope.py` |
| `activation` | `activation.py` |
| `gemm` | always available via PyTorch |

A kernel existing is `fi_supported`. Whether Path A can *harvest* it automatically is a
separate flag — the API must carry a trace decorator:

```bash
grep -rn "@flashinfer_api(trace=" tmp/flashinfer/flashinfer/ | grep -i "<module_or_api>"
```

Record as `fi_trace_template` true/false. False still means `fi_supported`, but Phase 2 falls
back to manual extraction.

**Intel:** ask the registry and the provider inventories rather than assuming.

```bash
python -c "
from flashinfer_bench.integration.xpu_kernels import REGISTRY
for k in sorted(REGISTRY, key=lambda k: (k.op_type, k.name)):
    print(f'{k.op_type:12} {k.provider:16} {k.name}')
"
```

Record `intel_status` per definition: `wired` (a baseline exists today), `available`
(a provider ships the kernel but nothing is registered — see
`onboard-model-intel/providers.md`), or `none` (needs a SYCL or Triton solution). Without
this the manifest cannot drive `/onboard-model-intel`, "Source the solution".

## 1f. Does SGLang route through it?

For each `fi_supported` definition, whether SGLang already calls the FlashInfer kernel drives
whether workloads can be collected at all.

```bash
grep -rn "<flashinfer_api_name>" tmp/sglang/python/sglang/srt/ | grep -v __pycache__
```

No hit means `sgl_missing`: record the file that *would* host the call, found by grepping
`tmp/sglang/python/sglang/srt/layers/` for the layer this definition belongs to, so the
SGLang PR step has a target.

## 1g. Write the manifest

```python
import json, subprocess
from pathlib import Path

def sha(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", f"tmp/{repo}", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None

path = Path("tmp/onboard_<slug>_<date>.json")
m = json.loads(path.read_text()) if path.exists() else {}
m.update({
    "model_slug": "<slug>",
    "hf_repo_id": "<repo_id>",
    "date": "<YYYY-MM-DD>",
    "repo_shas": {r: sha(r) for r in
                  ("sglang", "flashinfer", "sgl-cookbook", "flashinfer-trace")},
})
by_name = {k["definition_name"]: k for k in m.get("kernels", [])}
for k in <computed_kernel_records>:              # merge, never clobber later phases
    by_name.setdefault(k["definition_name"], {}).update(k)
m["kernels"] = list(by_name.values())
path.write_text(json.dumps(m, indent=2))
```

Merging matters: `phase2_status`, `phase3_status`, `workload_entries`, `fi_issue_url` and
`phase4` are written by later phases and must survive a re-run.

Per-kernel fields:

| Field | Values |
|---|---|
| `definition_name`, `op_type` | — |
| `phase1_status` | `existing` / `new` |
| `fi_status` | `fi_supported` / `fi_missing` |
| `fi_trace_template` | true / false |
| `sgl_status` | `sgl_integrated` / `sgl_missing` / `n/a` |
| `intel_status` | `wired` / `available` / `none` |

## 1h. Report

Print four buckets, because each routes to a different next step:

- **existing** — reuse; check workloads
- **new + fi_supported + sgl_integrated** — Path A trace-dump, then collect workloads
- **new + fi_supported + sgl_missing** — Path A possible, workloads blocked on an SGLang PR
- **new + fi_missing** — manual extraction, plus a FlashInfer kernel-request issue

On Intel the routing is `intel_status` instead: `wired` → benchmark now; `available` → wire a
baseline (`/onboard-model-intel`, "Source the solution"); `none` → write a solution
(`/optimize-intel-kernels`).
