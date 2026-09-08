---
name: extract-kernel-definitions
description: Generate Definition JSON for the flashinfer-trace dataset — by harvesting a short SGLang pass with FlashInfer's trace dumper (CUDA), by transcribing the schema from model sources and config.json, or by module hooks on Intel. Owns the definition naming and axis rules. Use when adding a model or filling gaps in the dataset.
---

# Extract kernel definitions

Produce `definitions/{op_type}/{name}.json` for every kernel a model needs. Definitions are
hardware-agnostic — one that already exists is reused unchanged, never re-derived.

| Path | How | When |
| --- | --- | --- |
| **A** | Trace-dump from a short SGLang run | An NVIDIA box is available and FlashInfer has a trace template for the op |
| **B** | Transcribe from model sources + `config.json` | No trace template, or TP/EP variants Path A cannot reach |
| **C** | Module hooks on a `transformers` run | Intel-only box — see `/onboard-model-intel` Phase 2 |

Path B owns the naming and axis rules that all three paths must agree on.

## Prerequisites

`/clone-repos` for `tmp/sglang`, `tmp/flashinfer`, `tmp/sgl-cookbook`, `tmp/flashinfer-trace`.

## Path A: trace-dump from a short SGLang pass

### A1. Pick the serving config

One pass per unique (TP, EP) combination covers every shape variant.

```bash
COOKBOOK=$(ls -d tmp/sgl-cookbook/data/models/generated/* | sort -V | tail -1)
ls "$COOKBOOK" | grep -i <model_name>
cat "$COOKBOOK"/<model_yaml>
```

No cookbook entry → default to TP=1 and skip EP.

### A2. Run the dump

```bash
export DUMP_DIR=tmp/dumps/fi_trace_<model_slug>_tp<TP>

tools/gpu-lock --gpus <TP> --exec-timeout 1800 -- python - <<'PY'
import os, shutil
from pathlib import Path

os.environ["FLASHINFER_TRACE_DUMP"] = "1"
os.environ["FLASHINFER_TRACE_DUMP_DIR"] = os.environ["DUMP_DIR"]
os.environ.setdefault("SGLANG_SKIP_CUBIN_DOWNLOAD", "1")

dump = Path(os.environ["DUMP_DIR"])
if dump.exists():
    shutil.rmtree(dump)

from sglang.srt.entrypoints.engine import Engine
engine = Engine(
    model_path="<hf_repo_id>",
    attention_backend="flashinfer",   # other backends bypass the dumper entirely
    disable_cuda_graph=True,          # cached graphs skip the Python path
    disable_radix_cache=True,
    mem_fraction_static=0.5,
    tp_size=<TP>,
    log_level="warning",
)
engine.generate(["The capital of France is"],
                {"temperature": 0.0, "max_new_tokens": 4, "top_k": 50, "top_p": 0.9})
engine.shutdown()
PY
```

Requirements that are not obvious and each cost a silent empty dump:

- **Env vars must be set before `import flashinfer` / `import sglang`.** The
  `@flashinfer_api` decorator binds at import.
- **`attention_backend="flashinfer"`** — other backends produce no dumps.
- **`disable_cuda_graph=True`** — cached graphs skip the dumper.
- **Page-size variants need separate runs.** Page size is fixed per server. Enumerate the
  ones the dataset already uses:
  `ls tmp/flashinfer-trace/definitions/gqa_paged | sed -n 's/.*_ps\([0-9]*\)\.json/\1/p' | sort -u`
- **MoE routing** — only the routing your prompts exercise will dump.
- **Quantized variants** require the model to actually load with that `--quantization`.

### A3. Stage into the dataset

The `op_type` field inside each JSON decides its subdirectory.

```bash
ls "$DUMP_DIR"

DUMP_DIR="$DUMP_DIR" python - <<'PY'
import json, os, shutil
from pathlib import Path

src = Path(os.environ["DUMP_DIR"])           # read from the environment, not by
dst_root = Path("tmp/flashinfer-trace/definitions")   # shell interpolation
for p in sorted(src.glob("*.json")):
    op_type = json.loads(p.read_text())["op_type"]
    out = dst_root / op_type
    out.mkdir(parents=True, exist_ok=True)
    if (out / p.name).exists():
        print(f"exists, skipping: {op_type}/{p.name}")
        continue
    shutil.copy2(p, out / p.name)
    print(f"staged: {op_type}/{p.name}")
PY
```

A quoted heredoc (`<<'PY'`) does **not** interpolate `$DUMP_DIR`; pass it through the
environment as above, or the script silently looks for a directory literally named
`$DUMP_DIR`.

### A4. Normalize and tag

The dumper does not emit every field the dataset requires. Add the `model:`, `tp:`/`ep:` and
`status:` tags, and reconcile field names against the schema, before validating.

### A5. Validate

```bash
flashinfer-bench validate --dataset tmp/flashinfer-trace --disable-gpu
```

Fix everything it reports as `[ERROR]` before moving on.

### A6. Gap check — what still needs Path B

```bash
comm -23 <(printf '%s\n' <expected_names> | sort) \
         <(ls "$DUMP_DIR" | sed 's/\.json$//' | sort)
```

Anything left did not dump. If FlashInfer has no trace template for it, that is B4.

## Path B: transcribe from sources

### B1. Read the model and serving config

The HuggingFace modeling file is the ground truth for what the op computes; `config.json`
gives the constants; the cookbook YAML gives TP/EP.

### B2. Naming and axis rules

**This table is the single owner of definition naming.** Other skills point here.

Some kernel types produce separate definitions per parallelism setting because parallelism
changes constant axis values; the rest are parallelism-agnostic.

| op_type | TP affects | EP affects | Naming pattern |
|---|---|---|---|
| `gqa_paged` | `q_heads/=TP`, `kv_heads/=TP` | — | `gqa_paged_{decode,prefill}_h{q}_kv{kv}_d{d}_ps{P}` |
| `gqa_ragged` | same as `gqa_paged` | — | `gqa_ragged_prefill_causal_h{q}_kv{kv}_d{d}` |
| `mla_paged` | `q_heads/=TP` | — | `mla_paged_decode_h{q}_ckv{ckv}_kpe{kpe}_ps{P}`, `mla_paged_prefill_causal_…` |
| `mla_ragged` | `q_heads/=TP` | — | `mla_ragged_prefill_causal_h{q}_qk{qk}_vo{vo}` |
| `dsa_paged` | `q_heads/=TP` | — | `dsa_sparse_attention_h{q}_ckv{ckv}_kpe{kpe}_topk{k}_ps{P}`, `dsa_topk_indexer_{quant}_h{q}_d{d}_topk{k}_ps{P}` |
| `gdn` | `q_heads/=TP`, `v_heads/=TP` | — | `gdn_{decode,mtp,prefill}_qk{q}_v{v}_d{d}_k_last` |
| `mamba_ssu` | `nheads/=TP`, `ngroups/=TP` | — | `mamba_ssu_decode_h{n}_d{d}_s{s}_ng{g}` |
| `moe` | — | `num_experts/=EP` | `moe_{quant}_{routing}_topk{k}_ng{g}_kg{kg}_e{local_e}_h{H}_i{I}`; TensorRT-LLM paths are named `trtllm_{quant}_[routed_]moe_topk{k}_…` instead |
| `rmsnorm` | — | — | `rmsnorm_h{H}` / `fused_add_rmsnorm_h{H}` |
| `gemm` | — | — | `gemm_n{N}_k{K}`, `gemm_{quant}_n{N}_k{K}` (lowercase `n`/`k`); grouped and sparse forms prefix the family: `grouped_gemm_{quant}_{layout}_g{G}_…`, `sparse_gemm_{quant}_…` |
| `rope` | — | — | `rope_with_cos_sin_cache_{neox,gptj}_style_d{d}_rd{rd}` |
| `sampling` | — | — | `top_k_…`, `top_p_…`, `top_k_top_p_sampling_from_probs_v{vocab}` — each word is its own segment, not `topk` |
| `activation` | — | — | `{silu,gelu,gelu_tanh}_and_mul_d{d}` |

For MLA, `ckv = kv_lora_rank + qk_rope_head_dim` and `kpe = qk_rope_head_dim`.

These patterns are descriptive, not normative: the dataset is the authority. Before naming
anything, list the op_type's directory in `tmp/flashinfer-trace/definitions/` and match what
is already there. A near-miss name creates a duplicate definition rather than an error.

A name encodes **shape, not dtype**. Two models of the same width but different precision
collide on one name — extract in the dtype the model is actually served in.

### B3. Write the JSON

Never start from a blank schema. Copy the nearest sibling and change the constants:

```bash
ls tmp/flashinfer-trace/definitions/{op_type}/
cp tmp/flashinfer-trace/definitions/{op_type}/{nearest}.json \
   tmp/flashinfer-trace/definitions/{op_type}/{new_name}.json
```

Then edit: `name`, the const axis values, the `assert` in the `reference`, and the tags.

The `reference` is plain PyTorch with float32 accumulation. Source it from SGLang's vanilla
forward (`tmp/sglang/python/sglang/srt/layers/...`) when FlashInfer does not have the op,
otherwise mirror FlashInfer's own test:

```bash
grep -rl "{fi_api_symbol}" tmp/flashinfer/tests/
```

Schema reference: `docs/flashinfer-trace/definition.mdx`. Validation flow:
`/add-reference-tests`.

### B4. File a missing trace template

When FlashInfer has no `@flashinfer_api(trace=...)` template for an op, Path A can never
harvest it. Check first:

```bash
grep -rn "@flashinfer_api(trace=" tmp/flashinfer/flashinfer/ | grep -i <op>
```

Then open an issue against `flashinfer-ai/flashinfer` giving the op, the wrapper, the
definition name you had to hand-write, and the model that needs it. Keep the issue body in a
file and pass `--body-file` rather than inlining it.

## Path C: Intel

`scripts/extract_model_kernels_xpu.py` records shapes from module hooks on a live
`transformers` run — no CUDA, no SGLang. Its output names are **not** dataset names and must
be transcribed through the B2 table before anything downstream matches. Procedure:
`/onboard-model-intel` Phase 2.

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| Dump directory empty | Env vars set after import, or wrong attention backend | A2 requirements |
| Only some definitions dumped | CUDA graphs, or that routing/quant path never ran | A2; the rest go to Path B |
| Staging script finds nothing | Quoted heredoc did not expand `$DUMP_DIR` | A3 — pass via the environment |
| `validate` reports axis errors | Const axis disagrees with the real tensors | The dump is ground truth; fix the definition |
| A name already exists | Definition is already in the dataset | Reuse it; never re-derive or rename |

## Sources

- `tmp/flashinfer/docs/fi_trace.rst` — the trace-dump mechanism
- `tmp/flashinfer/tests/trace/example_sglang.py` — the harness A2 mirrors
- `docs/flashinfer-trace/definition.mdx` — the full schema
- `tmp/sgl-cookbook/data/models/generated/` — serving configs per model
