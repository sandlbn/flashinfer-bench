# Tuned W8A8 block-FP8 configs for Intel Arc B580

vLLM's `w8a8_triton_block_scaled_mm` picks its tile shape from a JSON file named for the
device and the projection's `(N, K)`. It ships configs for NVIDIA and AMD parts only, so
**every Intel part falls back to one hardcoded config** and logs a warning naming the file
it wanted:

    Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!

The fallback is `BLOCK_SIZE_M=64, num_warps=4`. At decode, where M is 1, that computes a
64-row tile to keep one row.

These files are for the five projections of `Qwen/Qwen3-4B-Instruct-2507-FP8`, produced by
`scripts/tune_vllm_fp8_config.py` on an Arc B580. Measured against the fallback:

| shape (N x K) | M=1 | M=16 | M=512 |
| --- | --- | --- | --- |
| 1024 x 2560 | **9.96x** | 4.89x | 1.91x |
| 2560 x 4096 | 3.18x | 6.38x | 1.82x |
| 2560 x 9728 | 3.82x | 4.62x | 1.76x |
| 4096 x 2560 | 2.83x | 3.94x | 1.69x |
| 9728 x 2560 | 2.69x | 2.98x | 1.35x |

Two patterns hold across every shape and batch size:

- **`num_warps=16` wins, against a default of 4** -- in 13 of the 15 points above. This is
  the single largest factor and it is not shape-dependent.
- **`BLOCK_SIZE_M` wants 16-32 at decode**, against a default of 64.

`BLOCK_SIZE_N` stayed at 128 almost everywhere, so the weight block extent is already the
right N tile; `GROUP_SIZE_M` and `num_stages` are shape-dependent and worth the sweep.

## Using them

vLLM reads configs from its own package directory, so they have to be copied there. To
install the files in this directory:

```bash
cp *.json "$(python -c 'import vllm.model_executor.layers.quantization.utils.fp8_utils as f, os; print(os.path.join(os.path.dirname(f.__file__), "configs"))')"
```

To tune and install for a different model in one step:

```bash
python scripts/tune_vllm_fp8_config.py --from-definitions tmp/<model-defs> --install
```

You will know it worked because vLLM stops logging `Using default W8A8 Block FP8 kernel
config`.

## Upstreaming

This is data, not code, and it is the cheapest real contribution available for a quantized
model on a new device -- no kernel is written. Retune per part: the file name carries the
device, so configs for several devices coexist, and a config tuned on B580 says nothing
about Crescent Island.
