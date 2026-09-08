---
name: onboard-model
description: End-to-end pipeline for onboarding a new LLM into the flashinfer-trace dataset on CUDA — discover and classify its kernels, generate definitions, collect workloads, and open the PRs. Orchestrates the per-phase skills through a run manifest. For an Intel-only box use /onboard-model-intel instead.
---

# Onboard a model

A thin orchestrator. Each phase delegates to the skill that owns it; the state passed
between them is the run manifest at `tmp/onboard_{model_slug}_{date}.json`.

This path assumes CUDA (SGLang with the FlashInfer backend). On an Intel-only box use
`/onboard-model-intel`, which acquires definitions without CUDA.

| Phase | Skill | Output |
|---|---|---|
| 0 | `/clone-repos` | upstream trees in `tmp/`, SHAs recorded |
| 1 | `/discover-models` | `kernels[]` with `phase1_status`, `fi_status`, `fi_trace_template`, `sgl_status`, `intel_status` |
| 2 | `/extract-kernel-definitions` | definition JSON in `tmp/flashinfer-trace/definitions/` |
| 2c | `/add-reference-tests` | `tests/references/test_{name}.py`, green |
| 3 | `/collect-workloads` | workloads + blobs |
| 4 | `/submit-onboarding-prs` | one HF PR + one bench PR per definition |

## Phase 0: Update local repos

```bash
/clone-repos
```

Record the SHAs it prints into the manifest's `repo_shas` — they become the provenance
fields in both PR bodies.

## Phase 1: Discover and classify

```bash
/discover-models   # takes no arguments; it reads the manifest and writes back to it
                 --manifest tmp/onboard_{model_slug}_{date}.json
```

**Exit early if every kernel is `phase1_status=existing`** — unless some lack workloads:

```bash
for n in <definition_names>; do
  ls tmp/flashinfer-trace/workloads/*/"$n".jsonl >/dev/null 2>&1 || echo "no workloads: $n"
done
```

A definition with no workloads still needs Phase 3 even though Phase 2 is skipped.

## Phase 2: Generate definitions

For each kernel with `phase1_status=new`:

**`fi_supported`** → `/extract-kernel-definitions` Path A (trace-dump) when
`fi_trace_template=true`, else Path B (manual transcription).

**`fi_missing`** → Path B, then file a kernel request:

```bash
gh issue create --repo flashinfer-ai/flashinfer \
  --title "Kernel request: {definition_name} for {model_display_name}" \
  --body-file .claude/skills/onboard-model/kernel-request-issue.md
```

Fill the placeholders in that file first. Record the URL as `fi_issue_url`.

A `status:unverified` tag means FlashInfer has no ground truth for the op — it does **not**
mean the definition skips validation.

## Phase 2c: Reference test and validation gate

A definition without a passing reference test cannot ship — `/submit-onboarding-prs`
requires the pytest output in the PR body.

```bash
/add-reference-tests --definition-name {name}
cd tmp/flashinfer-trace && pytest tests/references/test_{name}.py -v
flashinfer-bench validate --dataset tmp/flashinfer-trace --definitions {name} --disable-gpu
```

Set `phase2_status=done` only when the test is green and validate reports no `[ERROR]`.

## Phase 3: Collect workloads

**`sgl_integrated`** → `/collect-workloads` directly.

**`sgl_missing`** → SGLang does not route through the FlashInfer kernel yet, so nothing can
be captured. Open the integration PR:

```bash
gh pr create --repo sgl-project/sglang \
  --title "Route {op} through FlashInfer {api}" \
  --body-file .claude/skills/onboard-model/sglang-integration-pr.md
```

Do **not** idle until it merges: `pip install -e tmp/sglang/python` on the PR branch (the repo root has no `pyproject.toml`; the package lives under `python/`) and collect
workloads locally now. Only the dataset PR waits on the merge.

Set `phase3_status=done` when the workload JSONL is non-empty and the baseline eval is all
`PASSED`.

## Phase 4: Submit

```bash
/submit-onboarding-prs --manifest tmp/onboard_{model_slug}_{date}.json
```

Then `/track-models` to update `docs/model_coverage.mdx`.

## Run manifest

Written by `/discover-models` (Phase 1) and updated in place by each later phase. Merge,
never rewrite — later phases must not lose earlier fields.

```json
{
  "model_slug": "qwen3-235b-a22b",
  "hf_repo_id": "Qwen/Qwen3-235B-A22B",
  "date": "2026-04-27",
  "repo_shas": {"sglang": "abc1234", "flashinfer": "def5678",
                "sgl_cookbook": "ghi9012", "flashinfer_trace": "jkl3456"},
  "kernels": [
    {
      "definition_name": "gqa_paged_decode_h40_kv8_d128_ps1",
      "op_type": "gqa_paged",
      "phase1_status": "new",
      "fi_status": "fi_supported",
      "fi_trace_template": true,
      "sgl_status": "sgl_integrated",
      "intel_status": "none",
      "phase2_status": "done",
      "phase3_status": "done",
      "workload_entries": 32,
      "fi_issue_url": null,
      "phase4": {"pr1_url": null, "pr2_url": null}
    }
  ]
}
```

| Field | Written by | Values |
|---|---|---|
| `phase1_status` | Phase 1 | `existing` / `new` |
| `fi_status`, `fi_trace_template`, `sgl_status`, `intel_status` | Phase 1 | see `/discover-models` |
| `phase2_status` | Phase 2c | `done` once the test is green and validate is clean |
| `phase3_status`, `workload_entries` | Phase 3 | `done` once eval is all `PASSED` |
| `fi_issue_url` | Phase 2 | kernel-request issue for `fi_missing` |
| `phase4` | Phase 4 | the two PR URLs |

**Resuming:** do not trust a status field alone — recompute it from the filesystem, because a
run can be interrupted between doing the work and recording it.

| Phase | "done" means |
|---|---|
| 2 | `definitions/{op_type}/{name}.json` exists |
| 2c | `tests/references/test_{name}.py` exists and passes |
| 3 | `workloads/{op_type}/{name}.jsonl` is non-empty |
| 4 | both PR URLs recorded |

## Failure handling

| Situation | Action |
|---|---|
| Model config unavailable (gated repo) | `hf auth login`; otherwise stop and report |
| Architecture unknown to SGLang | Nothing to onboard against — record and stop |
| Trace dump produces nothing | `/extract-kernel-definitions` A2 requirements, then fall back to Path B |
| A definition already exists | Reuse it; never re-derive or rename |
| Reference test cannot be made green | Do not ship. Leave `status:unverified` and report the disagreement upstream |
