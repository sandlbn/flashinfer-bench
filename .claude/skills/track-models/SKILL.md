---
name: track-models
description: Maintain docs/model_coverage.mdx — add newly discovered models, recompute which definitions exist, and refresh coverage status. Owns the coverage document's format and nothing else. Use after onboarding a model, or to refresh the table.
---

# Track models

Owns `docs/model_coverage.mdx`: which models are listed, which definitions each needs, and
which of those exist in the dataset today.

It owns the **document**, not the analysis. Candidate discovery and the definition-name
formulas belong to `/discover-models` and `/extract-kernel-definitions` section B2 — this
skill consumes their output and formats it.

## Prerequisites

`/clone-repos`, so `tmp/flashinfer-trace/definitions/` reflects the dataset.

## Step 1: Get the model list

Adding a model → run `/discover-models` for the candidate and its definition names.
Refreshing → read the models already in the Summary table of `docs/model_coverage.mdx`.

## Step 2: Check which definitions exist

```bash
for name in <expected_definition_names>; do
  if ls tmp/flashinfer-trace/definitions/*/"$name".json >/dev/null 2>&1; then
    echo "| \`$name\` | <op_type> | ✅ |"
  else
    echo "| \`$name\` | <op_type> | ❌ |"
  fi
done
```

| Mark | Meaning |
|---|---|
| ✅ | The definition JSON exists in the dataset |
| ❌ | Computed from the model config, but no JSON exists — needs creating |
| — | The module exists in the architecture but maps to no definition (unmapped) |

`—` rows are excluded from the coverage fraction; `❌` rows are not.

```
coverage = count(✅) / (count(✅) + count(❌))
```

- **✅ Fully covered** — no ❌
- **🟡 Partial** — some of each
- **❌ Not covered** — no definitions exist

## Step 3: Edit the document

**Summary table** — one row per model:

```markdown
| {Model Display Name} | {architecture description} | {coverage emoji + label} |
```

**Detail section** — one per model:

```markdown
## {Model Display Name}

**Architecture**: {N} decoder layers, {attention} attention, {ffn} FFN

Standard serving configuration: **TP={N}**.

| Definition | Op Type | Status |
|-----------|---------|:------:|
| `rmsnorm_h{hidden_size}` | rmsnorm | ✅ |
| MoE gate / topk / experts | moe | — |

**Coverage**: {N} / {M} definitions present.
```

Do **not** overwrite an existing model section unless refreshing. When refreshing, re-check
every ✅/❌, then update the coverage line **and** the Summary row together.

## Format rules

**Display names** carry size and variant, matching the Summary table exactly:
`Llama 3.1 8B`, `Qwen3 30B A3B`, `Mistral 7B v0.3`, `Gemma 3 27B`,
`DeepSeek V3 / R1` (joint entry for same-architecture variants).

**Architecture descriptions** are 5-8 words:
`{layers} decoder layers, {attention} attention, {ffn} FFN`, with a `hybrid` prefix for
mixed architectures (`48 layers, hybrid GDN+GQA attention, MoE FFN`).

**Multiple TP/EP configs** get separate rows, labelled in the Op Type column:

```markdown
| `gdn_prefill_qk16_v32_d128_k_last` | gdn TP=1 | ❌ |
| `gdn_prefill_qk8_v16_d128_k_last`  | gdn TP=2 | ✅ |
| `moe_..._e256_...`                 | moe EP=1 | ❌ |
| `moe_..._e32_...`                  | moe EP=8 | ✅ |
```

## Step 4: Verify before committing

```bash
pre-commit run --files docs/model_coverage.mdx
grep -c "✅" docs/model_coverage.mdx        # sanity-check against the coverage lines
```

Confirm each model's Summary row matches the count in its detail section.

## Failure table

| Symptom | Cause | Action |
|---|---|---|
| Config unavailable | Gated HF repo | `hf auth login`; otherwise record and skip |
| Unknown `architectures` value | Model not in SGLang yet | Record as untracked; nothing to onboard against |
| Computed name matches nothing, but the op is clearly covered | Naming drift | Recheck against `/extract-kernel-definitions` B2; **fix the computed name, never rename a dataset definition** |
| Summary and detail disagree | Partial refresh | Re-run Step 2 for that model and update both |
