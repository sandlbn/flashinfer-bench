## Kernel Request

**Model**: {model_display_name} ({hf_repo_id})
**Op type**: {op_type}
**Definition name**: {definition_name}

### Motivation

This kernel is required for serving **{model_display_name}** with FlashInfer.
A FlashInfer-Bench definition has been staged at:
`tmp/flashinfer-trace/definitions/{op_type}/{definition_name}.json`
(landing in the HuggingFace dataset PR for `flashinfer-ai/flashinfer-trace`)

### Kernel Parameters

{formatted parameter table from definition axes}

### Reference Implementation

A plain-PyTorch reference `run()` is available in the definition JSON above.
The SGLang implementation is at:
`python/sglang/srt/layers/{layer_path}`

### Requested Work

- [ ] CUDA/Triton kernel implementation matching the definition schema
- [ ] FlashInfer Python API (`flashinfer.{module}.{function}`)
- [ ] Unit test in `tests/test_{op_type}.py`

### Links

- HuggingFace dataset PR: (link once PR 2 is open)
- SGLang model: `tmp/sglang/python/sglang/srt/models/{model_file}`
- HuggingFace model: https://huggingface.co/{hf_repo_id}
