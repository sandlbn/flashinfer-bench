---
name: collect-workloads
description: Collect real workloads for a definition from an SGLang inference run using FlashInfer's Level-10 dump, sanitize them into the flashinfer-trace dataset, and verify they are not synthetic. Use when a definition exists but has no workloads, or when a benchmark reports NO_WORKLOAD.
---

# Collect workloads

Produce real `workloads/{op_type}/{definition_name}.jsonl` entries plus their safetensors
blobs, from shapes an actual inference run produced — not from shapes someone guessed.

**This path requires NVIDIA.** It drives SGLang with the FlashInfer attention backend, and
`tools/gpu-lock` allocates through `nvidia-smi`. On an Intel-only box, workloads come from
`onboard-model-intel` Phase 2 Path C instead (module hooks on a `transformers` run); see
"On Intel" at the end.

## Prerequisites

- `/clone-repos` has been run (`tmp/sglang`, `tmp/flashinfer`, `tmp/flashinfer-trace`).
- The definition JSON already exists in `tmp/flashinfer-trace/definitions/{op_type}/`.
- The model has an entry in `examples/sglang_bench/model_configs.json` (Step 1).
- Passwordless SSH between allocated nodes, if TP exceeds the local GPU count.

## Step 1: Confirm the model is in `model_configs.json`

```json
"<model-key>": {
    "server_flags": [
        "--trust-remote-code",
        "--attention-backend", "flashinfer",
        "--tp-size", "<N>"
    ]
}
```

The extra server flags depend on which kernel you are capturing, and getting them wrong is
the usual reason a run yields zero workloads:

| Definition kind | Add to `--extra-server-flag` |
| --- | --- |
| Paged decode | none |
| Paged prefill | `--disable-cuda-graph --enable-deterministic-inference` |
| Ragged prefill | `--disable-radix-cache --disable-piecewise-cuda-graph` |

Never pass `--enable-deterministic-inference` for ragged prefill — it suppresses the ragged
path and the run captures nothing.

## Step 2: Run collection

`scripts/collect_stream.py` does the whole loop: it runs `bench_serving.py` per batch size,
sanitizes, pushes incrementally, and deletes the dump directory before the next size.

```bash
tools/gpu-lock --gpus 8 --exec-timeout 10800 -- \
  python3 scripts/collect_stream.py \
    --def-name   gqa_paged_decode_h5_kv1_d128_ps64 \
    --model-key  llama-4-scout-ps64 \
    --model-path /path/to/model \
    --batch-sizes 64 128 \
    --pr-num <pr_num> \        # the open HF dataset PR to push the trace into
    [--extra-server-flag --disable-cuda-graph --enable-deterministic-inference] \
    [--trace-dir tmp/flashinfer-trace] \
    [--peer-node-addr <host>]
```

| Flag | Default | Use |
| --- | --- | --- |
| `--dump-count` | 500 | Dump budget per server session |
| `--workloads-per-batch` | 4 | Workloads appended per batch size |
| `--num-batches` | 2 | Inference rounds; the budget is usually hit in round 1 |
| `--replace-first` | off | Start the JSONL fresh instead of appending |
| `--no-push` | off | Collect and sanitize without uploading — use this first |
| `--no-eval` | off | Skip the eval + trace push |

Read from the definition's tags automatically: `tp:N` sets `--tp N`, and a `page_size` const
axis sets `--page-size`. To simulate TP=2 on one GPU, `CUDA_VISIBLE_DEVICES=0,0`.

**Multi-node** triggers automatically when the config's TP exceeds locally visible GPUs.
Peers come from `SLURM_JOB_ID` and workers launch over SSH; dump env vars are forwarded only
to rank 0, so only the head node dumps. Model paths must be on a filesystem mounted on every
allocated node — a local path will fail on the workers. Peer-environment details are in the
header comment of `examples/sglang_bench/bench_serving.py`.

If you drive `sanitize_dumps.py` yourself rather than through `collect_stream.py`, set
`FLASHINFER_DUMP_INCLUDE` to both `plan*` and `run*`
(`BatchDecodeWithPagedKVCacheWrapper.plan*,BatchDecodeWithPagedKVCacheWrapper.run*`).
Pairing each `run()` with its `plan()` is what lets the sanitizer skip const-axis checks on
tensors absent from the `run()` dump (`k_cache`, `v_cache`). With `run*` only, pass
`--skip-const-axis-check`.

## Step 3: Verify the workloads are real

Synthetic-looking workloads are the failure this step exists to catch. Diversity on the
varying axis is the test.

```bash
python3 -c "
import json, sys
p = '<trace_dir>/workloads/<op_type>/<def_name>.jsonl'
entries = [json.loads(l) for l in open(p)]
axes = list(entries[0]['workload']['axes'])
key = axes[0]
vals = sorted({e['workload']['axes'][key] for e in entries})
print('axes:', axes, '| entries:', len(entries))
print(f'{key} unique: {len(vals)} sample: {vals[:10]}')
sys.exit(1) if len(vals) < 5 else print('REAL workloads OK')
"
```

Fewer than 5 distinct values on the varying axis means the run did not exercise the kernel —
recheck the Step 1 flags before collecting again.

## Step 4: If sanitize reports 0 workloads with const-axis warnings

A definition constant disagrees with the tensors the model actually produced. The dump is
the ground truth, not the definition.

```python
from pathlib import Path
import json, safetensors.torch

first = sorted(Path(DUMP_DIR).iterdir())[0]
for k, v in safetensors.torch.load_file(first / "inputs.safetensors").items():
    print(f"  {k}: {tuple(v.shape)} {v.dtype}")
meta = json.loads((first / "metadata.jsonl").read_text().splitlines()[0])
print("tensor_details:", json.dumps(meta.get("tensor_details", {}), indent=2))
```

Then fix the definition, in this order — skipping a step leaves the dataset inconsistent:

1. Correct the const axis value (e.g. `head_dim: 64 → 128`).
2. Rename the definition file **and** its `name` field (`_d64_` → `_d128_`).
3. Fix the matching `assert` in the `reference`.
4. Rename the workload `.jsonl` and the blob directory.
5. Delete the old definition JSON.
6. Update sibling definitions for the same model that share the constant (`_ps1` / `_ps64`).
7. Re-run `sanitize_dumps.py` with the corrected name.

## Step 5: Baseline evaluation

```bash
flashinfer-bench run --local tmp/flashinfer-trace --definitions <def_name> --save-results
```

Every entry must be `PASSED` before the workloads ship. A non-PASSED entry means the
definition's reference disagrees with the baseline on real shapes — fix that, do not publish.

## Step 6: Hand off

`/submit-onboarding-prs` opens the dataset PR. It needs the collection log (the
`collect_stream.py` stdout) as provenance for the PR body.

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| No dumps produced | `FLASHINFER_DUMP_*` not reaching the server, or wrong wrapper name | Confirm the include pattern names the wrapper the model actually calls |
| Ragged run yields 0 workloads | `--enable-deterministic-inference` was passed | Remove it; use `--disable-radix-cache --disable-piecewise-cuda-graph` |
| Sanitize reports 0 with const-axis warnings | Definition constant is wrong | Step 4 |
| Low diversity on the varying axis | Kernel was not exercised | Recheck Step 1 flags |
| Workers dump nothing on multi-node | Expected — only rank 0 dumps | Not an error |
| Model not found on worker nodes | Model path is node-local | Move it to a shared mount |
| Eval trace has non-PASSED entries | Reference disagrees on real shapes | Fix the definition; do not publish |

## On Intel

Definitions and workloads are hardware-agnostic, so workloads collected on NVIDIA are
reused unchanged on Intel — never re-collect them per backend.

When no NVIDIA box is available, `onboard-model-intel` Phase 2 Path C emits workloads
alongside its definitions from module hooks on a `transformers` run. That gives TP=1 shapes;
TP/EP variants come from Path B arithmetic. The provenance artifact for the PR is then the
Path C run log rather than an SGLang collection log.

## Sources

- `tmp/flashinfer/docs/logging.rst` and `tmp/flashinfer/flashinfer/api_logging.py` — the
  authority on Level-10 dump env vars
- `examples/sglang_bench/bench_serving.py` header — multi-node and peer-environment details
- `docs/flashinfer-trace/workload.mdx` — the workload JSONL schema
