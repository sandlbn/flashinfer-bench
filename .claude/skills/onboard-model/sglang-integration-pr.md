## Summary

- Integrates `{fi_api}` into SGLang's FlashInfer backend
- Enables optimized `{op_type}` kernel for **{model_display_name}**
- Required by flashinfer-ai/flashinfer-bench for workload collection

## Changes

- `python/sglang/srt/layers/{path}`: add FlashInfer dispatch for {op_type}

## Test plan

- [ ] SGLang unit test passes with `--attention-backend flashinfer`
- [ ] Inference output matches non-FlashInfer baseline
- [ ] Memory usage within expected bounds
