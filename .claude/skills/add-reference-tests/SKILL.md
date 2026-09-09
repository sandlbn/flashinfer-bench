---
name: add-reference-tests
description: Write the pytest file that validates a definition's reference implementation against FlashInfer or SGLang ground truth, at tests/references/test_{definition_name}.py in the flashinfer-trace dataset. Use when a new definition needs a test before its PR, or when a reference is suspected wrong.
---

# Add reference tests

A reference test proves that a definition's `reference` field computes the same function as
the kernel the production stack actually runs. Without it, every benchmark built on that
definition validates against a guess.

One file per definition, in the dataset clone:

```
tmp/flashinfer-trace/tests/references/test_{definition_name}.py
```

These are script-style files — `run()`, `generate_random_inputs()`, `test_correctness()`,
`main()` — runnable both under pytest and directly. There is no `conftest.py`, no fixtures,
no shared helpers, and no custom pytest flags. Do not invent them.

## Prerequisites

- `/clone-repos` has been run; `tmp/flashinfer-trace/` and `tmp/flashinfer/` exist.
- The definition JSON exists in `tmp/flashinfer-trace/definitions/{op_type}/`.
- Ground truth is importable: `python -c "import flashinfer"` (CUDA only — see "On Intel").

## Step 1: Check whether a test already exists

```bash
ls tmp/flashinfer-trace/tests/references/test_{definition_name}.py
```

Exists → run it and stop. Only rewrite it if it fails or does not exercise the definition's
own `reference`.

## Step 2: Generate it

```bash
python scripts/generate_reference_tests.py --local tmp/flashinfer-trace \
    --definitions <name>[,<name>...]        # --force to overwrite
```

It reads the definition, picks a template by the **full signature** — `(op_type, inputs,
outputs)`, not op_type alone — takes the dtype and constants from the JSON, and refuses
rather than emitting a test it cannot build correctly:

```
no template for gemm_swiglu_merged_k1024_d3072 (op_type=gemm, inputs=('x','w_gate_up'),
outputs=('up','out')) -- add one rather than hand-writing
```

Adding a template is a `build()` branch in that script keyed on the signature. Prefer that to
hand-writing, so the next definition of the same shape is free.

**Hand-writing, when no template fits:** copy the nearest const-axis match rather than
starting blank — the existing reference tests are near-identical by op_type.

```bash
ls tmp/flashinfer-trace/tests/references/test_{op_type}_*.py
```

Then update the module constants, the definition name, the shapes, and the tolerances.

## Step 3: Test the definition's own reference, not a retyped copy

Load the reference from the JSON rather than retyping it, so the test cannot validate a
function the definition does not ship:

```python
import math
from pathlib import Path
from flashinfer_bench.data import Definition, load_json_file

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

def load_definition(name: str) -> Definition:
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)      # model class first
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")

def compile_reference(reference_code: str):
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]
```

Then `run = compile_reference(load_definition(NAME).reference)` and compare *that* against
ground truth. If the file instead inlines its own `run()`, verify by hand that it matches
the JSON's `reference` before trusting the result.

## Step 4: Pick the ground truth

**FlashInfer is the primary ground truth.** Fall back to SGLang only where FlashInfer does
not implement the variant, and record which one you used in the test docstring.

| op_type | Ground truth |
| --- | --- |
| `rmsnorm`, `fused_add_rmsnorm` | `flashinfer.norm.rmsnorm`, `flashinfer.norm.fused_add_rmsnorm` |
| `gqa_paged` | `flashinfer.BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper` |
| `gqa_ragged` | `flashinfer.BatchPrefillWithRaggedKVCacheWrapper` |
| `mla_paged` | `flashinfer.mla.BatchMLAPagedAttentionWrapper` |
| `rope` | `flashinfer.apply_rope_with_cos_sin_cache_inplace` |
| `sampling` | `flashinfer.sampling.*` |
| `activation` | `flashinfer.activation.*` |
| `moe` | `flashinfer.fused_moe` where it covers the variant, else `sglang/layers/moe/fused_moe.py` |
| `gdn` | `flashinfer.gdn_decode` |
| `gemm` | `torch.nn.functional.linear` |

To find how a wrapper is actually called, read FlashInfer's own tests rather than guessing
the argument order:

```bash
grep -rl "{fi_api_symbol}" tmp/flashinfer/tests/
```

## Step 5: Choose tolerances by output dtype

| dtype | atol | rtol |
| --- | --- | --- |
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 5e-2 |
| float8_e4m3fn, nvfp4 | use a hit ratio ≥ 85% of elements within tolerance, not elementwise |

`atol=1e-2, rtol=5e-2` is what most existing bf16 tests use. Tightening below one ULP of the
dtype makes the test fail on correct kernels — bfloat16's relative spacing is `2**-7`.

## Step 6: Run it

```bash
cd tmp/flashinfer-trace
pytest tests/references/test_{definition_name}.py -v -s
python tests/references/test_{definition_name}.py     # the files have main()
```

## When it fails

| Symptom | Cause | Action |
| --- | --- | --- |
| Shape mismatch | Definition `outputs` disagrees with `reference` | Fix the JSON, re-run `flashinfer-bench validate` |
| Within ~10x of tolerance, cosine similarity > 0.999 | A scaling or accumulation detail | Check `sm_scale`, `eps`, the causal flag, and that the reference accumulates in float32 |
| Cosine similarity < 0.99 | Wrong semantics | See the trap list below |
| FlashInfer and SGLang disagree with each other | Ground truth itself is suspect | File upstream; leave the definition `status:unverified` |

Semantic traps that produce a clean run and wrong numbers:

- Gemma-style norm scales by `(1 + weight)`, not `weight`.
- RoPE: neox vs gptj interleaving.
- LSE returned in natural log vs log2.
- Gated activations split `[..., :d]` / `[..., d:]`; some fused GEMM epilogues emit
  *interleaved* pairs instead.

## Exit criteria

The test passes on GPU, and only then may the definition carry `status:verified`. Commit the
test and the definition together on the dataset feature branch — `/submit-onboarding-prs`
requires the pytest stdout in the PR body.

## On Intel

Generated tests **do** run on Intel: the generator selects the device at runtime and uses
independent ground truth that exists there — `vllm-xpu-kernels`' `rms_norm`,
`fused_add_rms_norm` and `silu_and_mul` for norms and activations, and `F.linear` against the
reference's `A @ B.T` for GEMM.

Hand-written tests copied from the existing ones do **not**: they `import flashinfer` at module
scope, so on a box without CUDA `pytest tests/references/` fails with collection errors
rather than skipping. Import ground truth inside `try/except` and select the device at
runtime, as the generated ones do.

FlashInfer and SGLang ground truth themselves are CUDA-only. Guard with `pytest.skip` rather than assuming CUDA; note that `requires_torch_cuda` is
registered in flashinfer-bench's `pyproject.toml` but **not** in the dataset, so inside the
dataset it is an unregistered marker.

Validating a reference on Intel is a different check with its own command — it compares the
reference across devices rather than against a vendor kernel:

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace \
    --device xpu:0 --definitions {definition_name}
```

That belongs to `onboard-model-intel` Phase 3.

## Sources

- `tmp/flashinfer-trace/tests/references/` — the worked examples; copy the nearest
- `tmp/flashinfer/tests/` — how each FlashInfer wrapper is really called
- `docs/flashinfer-trace/definition.mdx` — the definition schema
