---
name: submit-onboarding-prs
description: Open the per-definition pair of PRs that publishes a model onboarding — PR 2 to the HuggingFace flashinfer-trace dataset (definition, reference test, baseline solution, workloads, blobs, eval traces) and PR 1 to flashinfer-bench (docs/model_coverage.mdx only). Includes the pre-flight dataset validation gate. Use as Phase 4 of /onboard-model.
---

# Submit onboarding PRs

Two atomic PRs per definition, opened in this order:

| # | Target | Content |
|---|--------|---------|
| 2 | `flashinfer-ai/flashinfer-trace` (HuggingFace) | definition JSON, reference test, baseline solution, workload JSONL, safetensors blobs, eval traces |
| 1 | `flashinfer-ai/flashinfer-bench` (GitHub) | `docs/model_coverage.mdx` only, linking to PR 2 |

**One definition = one pair of PRs.** Never batch definitions — each must be independently
reviewable and mergeable.

## Prerequisites

- `gh` authenticated for `flashinfer-ai/flashinfer-bench`; `huggingface_hub` authenticated
  for `flashinfer-ai/flashinfer-trace`.
- `pre-commit` installed.
- For each definition, staged in `tmp/flashinfer-trace/`: the definition JSON, the reference
  test, a baseline solution, the workload JSONL and its blobs, and eval traces where every
  entry is `PASSED`. (`fi_missing` definitions skip workloads, baseline and traces.)

## Step 1: Pre-flight — validate before you commit

Run this before creating any worktree. It is far cheaper than a review round.

```bash
flashinfer-bench validate --dataset tmp/flashinfer-trace \
    --definitions <name> --disable-gpu          # structure and schema
flashinfer-bench validate --dataset tmp/flashinfer-trace \
    --definitions <name>                        # adds the GPU checks
```

On Intel, select the backend for the GPU checks:
`FIB_DEVICE_BACKEND=xpu flashinfer-bench validate --dataset tmp/flashinfer-trace --definitions <name>`.

| Check | What it covers | On failure |
| --- | --- | --- |
| `layout` | Duplicate names, directory structure, path-field consistency, blob existence | Move the file to the path its `op_type` dictates; re-copy missing blobs |
| `definition` | Schema, reference Python syntax, axis references, build reference | Fix the JSON; `--definitions` re-runs just this one |
| `workload` | Schema, axes > 0, shape inference, blobs present, no solution/evaluation embedded | Re-run `/collect-workloads`; a workload must not carry an evaluation |
| `solution` | Schema, `entry_point` format, sources, path-field consistency | Fix the solution JSON |
| `trace` | Has solution + evaluation, solution exists, workload coverage | Re-run `flashinfer-bench run --save-results` |
| `baseline` | Baseline exists, builds, `PASSED`, covers the workloads | See Step 3 — a missing baseline usually means a signature mismatch |
| `benchmark` | Runs baseline + reference on GPU | Fix whichever the report names before publishing |

Reports land in `<dataset>/reports/report-YYYYMMDD-HHMMSS.json`; re-render an old one with
`flashinfer-bench validate-render <report>.json`. Full report structure:
`docs/flashinfer-trace/validate.mdx`.

Do not open a PR until this is clean.

## Step 2: Create the worktrees

```bash
DATE=$(date +%Y%m%d)

git worktree add tmp/worktrees/bench-{name} -b feat/def-{name}

git -C tmp/flashinfer-trace worktree add \
    ../worktrees/trace-{name} -b workloads-${DATE}-{name}
```

Never commit directly in `tmp/flashinfer-trace/` — it is the shared clone.

## Step 3: PR 2 — the dataset

PR 2 opens first so PR 1 can link to it.

```bash
export D=tmp/worktrees/trace-{name}     # exported: the `bash -c` below runs in a subshell
cp tmp/flashinfer-trace/definitions/{op_type}/{name}.json        $D/definitions/{op_type}/
cp tmp/flashinfer-trace/tests/references/test_{name}.py          $D/tests/references/
# Baselines exist in two layouts (flat `{name}__{provider}.json` and a per-definition
# subdir); loading is recursive, so copy whichever is present:
# `-path "*{name}*"` over-matches: for rmsnorm_h2048 it also pulls fused_add_rmsnorm_h2048.
# Anchor on the path separator so only this definition's own files are taken.
find tmp/flashinfer-trace/solutions \
     \( -path "*/{name}/*" -o -name '{name}__*.json' \) -name '*.json' -exec bash -c \
  'rel=${1#tmp/flashinfer-trace/}; mkdir -p "$D/$(dirname "$rel")"; cp "$1" "$D/$rel"' _ {} \;
cp tmp/flashinfer-trace/workloads/{op_type}/{name}.jsonl         $D/workloads/{op_type}/
cp -r tmp/flashinfer-trace/blob/workloads/{op_type}/{name}/      $D/blob/workloads/{op_type}/
# Traces are sharded by author -- `traces/{author}/{op_type}/{name}.jsonl` -- and one
# definition usually has several (a `baseline/` set plus one per solution author). Copy
# every author that has traces for it, not one guessed path.
find tmp/flashinfer-trace/traces -name '{name}.jsonl' -exec bash -c \
  'rel=${1#tmp/flashinfer-trace/}; mkdir -p "$D/$(dirname "$rel")"; cp "$1" "$D/$rel"' _ {} \;
```

`$D` must be **exported**. `find -exec bash -c` spawns a child shell, which inherits only
exported variables; with a plain `D=...` the child expands `"$D/..."` to `/...` and the
`mkdir -p` targets the filesystem root.

The baseline must wrap the **production kernel for the target backend**, never a copy of the
definition's `reference`:

| Backend | Baseline wraps |
| --- | --- |
| CUDA | the FlashInfer API (`BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, …) |
| Intel | `vllm-xpu-kernels`, `sgl-kernel-xpu`, oneDNN, or an in-tree kernel — `flashinfer-bench add-baselines --in-tree` |

Regenerate eval traces whenever you touch the baseline. A pre-existing baseline may be
carried through unchanged, but its traces must still be present and `PASSED`.

```bash
cd $D
# `git add -A` over the copied trees, rather than naming paths: baselines have two
# layouts and traces are author-sharded, so an explicit list silently drops whichever
# shape this definition happens to use.
git add -A definitions tests solutions workloads blob traces
git commit -m "Add {name}: definition + reference test + baseline solution + workloads + traces

Model: {hf_repo_id}
SGLang: {sglang_sha}
FlashInfer: {flashinfer_sha}
Workload entries: {count}
"
```

Open the PR, then push the branch **into it**. `create_pull_request` opens an empty PR and
returns a discussion; it has no `head` parameter, and pushing a named branch does not open a
PR on the Hub.

```bash
PR_NUM=$(python -c "
from huggingface_hub import HfApi
pr = HfApi().create_pull_request(
    repo_id='flashinfer-ai/flashinfer-trace',
    repo_type='dataset',
    title='Add {name}: definition + reference test + baseline solution + workloads + traces',
    description=open('/tmp/pr2-body.md').read(),
)
print(pr.num)
")
git push origin HEAD:refs/pr/$PR_NUM
```

Write `/tmp/pr2-body.md` first — checklist items 3, 8 and 9 live in that body (pytest
stdout, collection log, provenance), so an empty description fails review.

## Step 4: PR 1 — the coverage doc

The diff must touch **only** `docs/model_coverage.mdx`.

```bash
cd tmp/worktrees/bench-{name}
# Mark the {name} row ✅ for this model and bump the per-model summary count.
pre-commit run --all-files
git add docs/model_coverage.mdx
git commit -m "docs: mark {name} as covered for {model_display_name}

Tracks the dataset addition at:
{pr2_url}
"
git push origin feat/def-{name}
gh pr create --repo flashinfer-ai/flashinfer-bench \
  --title "docs: mark {name} as covered for {model_display_name}" \
  --body-file /tmp/pr1-body.md
```

## Step 5: Clean up

```bash
git worktree remove tmp/worktrees/bench-{name}
git -C tmp/flashinfer-trace worktree remove ../worktrees/trace-{name}
```

Record both PR URLs in the manifest's `phase4` block.

## Review checklist

Both PRs must pass every item. Fix in the same worktree and push a follow-up commit — never
close and reopen, and never amend a commit that has been reviewed.

**PR 1 (coverage doc)**

1. The `{name}` row shows ✅ for `{model_display_name}` and the summary count matches.
2. The diff touches only `docs/model_coverage.mdx`.
3. The body links PR 2 by full URL.
4. If `fi_missing`, the body links the FlashInfer kernel-request issue.
5. `pre-commit run --all-files` passes.

**PR 2 (dataset)**

1. `definitions/{op_type}/{name}.json` present.
2. Tags carry `status:verified` (or `status:unverified` when the upstream kernel is missing),
   plus `fi_api:*` and `tp:*`/`ep:*` where applicable.
3. `tests/references/test_{name}.py` present and green; full pytest stdout in the body.
4. `workloads/{op_type}/{name}.jsonl` present and non-empty.
5. `blob/workloads/{op_type}/{name}/*.safetensors` present.
6. Baseline wraps the production kernel for the backend, not the `reference`.
7. Every entry in `traces/{author}/{op_type}/{name}.jsonl` is `PASSED`.
8. Body contains a `## Collection Log` section with the stdout from `collect_stream.py`
   (CUDA) or the Path C extraction run (Intel). Real workloads have diverse
   `(batch_size, kv_length)` pairs; a uniform sweep is a red flag for synthetic data.
9. Body records `Model`, `SGLang` and `FlashInfer` SHAs, and the workload-entry count.

## Fixing a failed checklist item

| Item | Fix |
| --- | --- |
| PR1-1 coverage row | Edit `docs/model_coverage.mdx`, commit, push to `feat/def-{name}` |
| PR1-2 stray files | `git restore --staged --source=origin/main -- :^docs/model_coverage.mdx`; the removed content belongs in the PR 2 worktree |
| PR1-3/4 missing link | `gh pr edit {num} --body-file -` |
| PR1-5 pre-commit | Fix what it reports; never `--no-verify` |
| PR2-1 definition missing | Copy from the shared clone, commit, `git push origin HEAD:refs/pr/$PR_NUM` |
| PR2-2 tags | Edit `tags`, re-run `flashinfer-bench validate --dataset $D --definitions {name}` |
| PR2-3 test missing/red | `/add-reference-tests`, run until green, update the PR body via `HfApi().edit_discussion_comment(...)` — there is no `edit_discussion`; a PR's description is the discussion's **first comment**, so read `get_discussion_details(...).events[0].id` and pass that as `comment_id` |
| PR2-4/5 workloads or blobs | Re-run `/collect-workloads`; copy the regenerated files in |
| PR2-6 baseline copies `reference` | Replace with a real wrapper (`add-baselines`), regenerate traces |
| PR2-7 non-PASSED traces | Diagnose before regenerating — usually a baseline bug or a tolerance issue |
| PR2-8 log missing/synthetic | Re-collect with the real config; a uniform axis sweep is synthetic |
| PR2-9 provenance | Append `Model` / `SGLang` / `FlashInfer` SHAs and the entry count |

After any PR 2 fix, refresh the description — stale pytest or collection output masks the
real state.

A structural mistake (PR 1 carrying workload files, PR 2 missing the definition) is fixed by
correcting the worktrees and force-pushing **only the per-definition branch**, never the
dataset's `main`, and only after telling the reviewer.
