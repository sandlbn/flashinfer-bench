# Quantized models: tune the config before writing a kernel

An FP8 or INT4 checkpoint reaches a different kernel from its bf16 sibling.

**Model the operation the served model performs.** A checkpoint declaring
`activation_scheme: dynamic` quantizes the activation too, so the operation is W8A8: four
inputs (`A_fp8`, `A_scale`, `B_fp8`, `B_scale`), not two. A definition taking a bfloat16
activation describes an operation the served model never performs, and no serving kernel
can bind to it.

**vLLM's Triton kernels run on XPU as-is.** `w8a8_triton_block_scaled_mm` is portable Triton
with no CUDA guard, is what a served FP8 model executes, and is registered as a baseline
(`provider="vllm"`, distinct from `vllm-xpu`, whose kernels are SYCL). It lives in the
`vllm` distribution, so benchmark from an environment that has vLLM.

**It selects its tile shape from a per-device JSON file and ships none for Intel parts.**
Without one it falls back to a default tile and logs a warning naming the file it wanted.
Tune every distinct `(N, K)` the model uses — extract the definitions first and read them
off:

```bash
python scripts/tune_vllm_fp8_config.py --shape <N>,<K> [--shape ...] \
    --device xpu:0 --output tmp/fp8-configs
# --install also copies into vLLM's configs/ directory, where the kernel reads it
```

The result is a JSON file, upstreamable as data. The filename carries the device; retune
per part. `tools/vllm-fp8-configs/` holds the files produced so far.

**Do not build a block-scaled GEMM out of oneDNN primitives.** oneDNN can express it only by
decomposing over K, which pushes the accumulator through memory once per K-block; its
`groups` scale argument is also wrong on this hardware. A block-scaled GEMM keeps the
accumulator in registers across K-blocks: one fused kernel, Triton or SYCL. See
`../../optimize-onednn/references/quantized-matmul.md`.
