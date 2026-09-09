# Turning `no-solution` reports into definitions

```bash
python scripts/fill_serving_gaps.py --from-json <result>.json          # report
python scripts/fill_serving_gaps.py --from-json <result>.json --eps <e> --write
```

It clones the widest sibling of the family, rescales every constant axis together, rewrites
the reference's asserted constants, and emits workloads over the sibling's batch sweep.
Then source and optimize as usual — `validate-references`, `add-baselines --in-tree`,
`add-baselines --providers`, `run --save-results` — and re-measure.

Flags it refuses to infer, because a wrong value produces a definition that benchmarks
cleanly while computing the wrong function:

- **`--dtype`** — from the model config's `torch_dtype`. A definition in a dtype the model
  does not run in is refused by `apply()`'s dtype guard and reports `no-solution`, which
  reads as "not extracted yet". Quantized checkpoints are commonly `float16` while the
  sibling definition is `bfloat16`.
- **`--eps`** — for norm families, `config.json`'s `rms_norm_eps` (1e-5 and 1e-6 both occur).
- **A width already present at another epsilon gets a suffixed name.** `{family}_h{H}` does
  not encode epsilon; check for this whenever a family substitutes at 100% on a model you
  did not extract it from.

Generated definitions carry `status:unverified` and no `model:` tag.

## CLI spelling of `--definitions`

`validate-references` and `add-baselines` take **one comma-separated string**; `run` takes
**space-separated names**. A comma-joined list passed to `run` is searched for as a single
name, reported as not found, and exits 0.
