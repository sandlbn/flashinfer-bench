# Quantized matmul: what oneDNN supports, probed

**Illustration (one instance): Battlemage / oneDNN 3.11.x and 3.13.x / fp8 matmul with scales and post-ops — re-establish with: `tools/onednn/repro_grouped_scales.cpp`**

Read this before writing any oneDNN solution that takes scales, and re-probe on a library
version outside the range above.

| attribute | status |
| --- | --- |
| f8_e4m3 src x f8_e4m3 weights | exact |
| weights scale, per-tensor | exact |
| weights scale, per-column (`set_scales_mask`, N values) | exact |
| weights scale, **grouped** (`set_scales` with `groups`) | **silently wrong** |
| **src scales, any mask** | rejected: `unsupported scales configuration` |
| `sum` post-op, `binary_mul` post-op | exact |

## Do not use the `groups` argument to `set_scales`

`set_scales(DNNL_ARG_WEIGHTS, mask, {BK, BN})` returns wrong values with no error; the
primitive descriptor is created and reports `jit:gemm:any`. Grouped along K the result is
scaled wrongly at every block; grouped along N most output elements are wrong while some —
including `C[0,0]` — happen to be right.

**Verify any scale configuration with varying scale values across many output positions**,
never a constant scale checked at one corner.

## Constraints that hold for every scale and post-op operand

- **oneDNN reads scale and binary-operand memory densely, ignoring the strides in the memory
  descriptor.** A scale column taken from a `[M, K_blocks]` tensor with a strided desc reads
  the wrong elements (correct only at row 0). Transpose or gather into contiguous memory
  first.
- **A standalone SYCL queue is out-of-order by default**, so kernels preparing scale buffers
  race the matmuls consuming them. PyTorch's XPU queue is in-order, so an in-tree kernel on
  the framework's queue is safe; a standalone reproducer must ask for
  `sycl::property::queue::in_order`.

## Expressing block scales with only the exact primitives — and why not to

Block scales can be composed exactly: one matmul per K-block (f8 src, f8 weights, f32
destination), weights scale as an ungrouped per-column array built by repeating each
K-block's `NB` block scales `BN` times, `binary_mul` post-op carrying `A_scale[:, kb]` as an
`[M, 1]` per-row vector, and `sum` post-op to accumulate.

Do not ship it. Each K-block reads and writes the whole `[M, N]` float32 destination, so the
composition is bandwidth-bound on accumulator traffic the composition itself creates, and
no tuning inside oneDNN removes it. **A block-scaled GEMM keeps the accumulator in registers
across K-blocks** — one fused kernel, Triton or SYCL. Use oneDNN for the dense GEMM and a
fused kernel the moment scales vary along the reduction dimension.
