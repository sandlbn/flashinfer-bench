# aten::linear

Pulled by `scripts/pull_kernel_source.py`. Everything here describes what **actually runs**
on `xpu:0` -- the dispatcher was asked, not a table, and then the library was asked which
of its implementations it selected and why.

Every line below carries one of four tags. Trust them differently.

| tag | meaning |
| --- | --- |
| `[reported]` | the library printed it at run time (`verbose.log`) |
| `[cited]` | a `file:line` the library itself named |
| `[rule]` | inferred by a stated mapping rule from reported or cited facts; can be wrong when the rule's stated assumption fails |
| `[interpretation]` | a reading of a reported reason into what a caller could change; check it against the gate |

| | |
| --- | --- |
| dispatch key | `(none)` |
| registered at | `/__w/pytorch/pytorch/aten/src/ATen/autocast_mode.cpp` |
| providing project | `oneDNN` -- a library, not one kernel: it selects an implementation per call |
| library version (ran here) `[reported]` | `3.12.0+80afa71049cd` |
| library version (SYCL solutions link) | `3.11.4+0291f8943088 (/opt/intel/oneapi/dnnl/2026.0/lib/libdnnl.so.3.11)` |
| runtime and engines `[reported]` | `gpu,runtime:SYCL`; `gpu,engine,sycl gpu device count:2`; `gpu,engine,0,backend:Level Zero,name:Intel(R) Arc(TM) B580 Graphics,driver_version:1.17.39395,binary_kernels:enabled`; `gpu,engine,1,backend:Level Zero,name:Intel(R) Graphics,driver_version:1.17.39395,binary_kernels:enabled` |
| source checkout | `tmp/oneDNN` |
| source revision bundled | `v3.13.2` |
| harness | `harness.py` -- calls this op at a shape the model called it at |
| record of this pull | `verbose.log` (unedited), `selection.json` (machine-readable, diff two bundles with it) |

## Schema

```
aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
```

## 1. What ran `[reported]`

| primitive | implementation | problem | memory descriptors | attributes |
| --- | --- | --- | --- | --- |
| `matmul` | `jit:gemm:any` | `4x2048:2048x2560` | `src:bf16::blocked:ab::f0 wei:bf16::blocked:ba::f0 bia:bf16::blocked:ab::f0_mask2 dst:bf16::blocked:ab::f0` | `attr-scratchpad:user` |

Resolved to source `[rule]` -- the string literal in `DECLARE_COMMON_PD_T("<name>", <class>)` is what the log prints as the implementation, so the file holding that literal declares it and its sibling body implements it:

- `source/src/gpu/intel/gemm/jit.hpp` -- selected implementation -- declares `jit:gemm:any` as `gen_t` (line 85)
- `source/src/gpu/intel/gemm/jit.cpp` -- selected implementation -- implements `gen_t`

## 2. Why that one: the dispatch chain

oneDNN walks the candidate list for the primitive kind top to bottom and takes the first
implementation whose gates all pass. `selected`/`rejected` outcomes are `[reported]`; the
list order, the names of entries that did not print, and the build conditions are `[rule]`
(entry `ns::cls` is matched to the `DECLARE_COMMON_PD_T` of `cls` under `src/<engine>/ns/`;
a computed name cannot be matched and is shown as the expression that computes it).

**`gemm`** -- `source/src/gpu/gpu_gemm_list.cpp` `[rule]`

| # | candidate | name | outcome | detail |
| --- | --- | --- | --- | --- |
| 1 | `intel::gemm::conv_t` | `conv:ir` | **build-conditional** `[rule]` | compiled out when !(DNNL_DEV_MODE) |
| 2 | `intel::gemm::xe_hp_systolic_t` | `jit:xe_hp:gemm:any` | **rejected** `[reported]` | unsupported format tag at `src/gpu/intel/gemm/jit_xe_hp_systolic.cpp:79` |
| 3 | `intel::gemm::gen_t` | `jit:gemm:any` | **selected** `[reported]` |  |
| 4 | `intel::gemm::with_post_ops_t` | `ocl:with_po:any` | **not tried** `[rule]` | after the winner in the list |
| 5 | `intel::gemm::ref_t` | `ocl:ref:any` | **build-conditional** `[rule]` | compiled out when DNNL_DISABLE_GPU_REF_KERNELS |

**`matmul`** -- `source/src/gpu/gpu_matmul_list.cpp` `[rule]`

| # | candidate | name | outcome | detail |
| --- | --- | --- | --- | --- |
| 1 | `intel::matmul::gemm_t` | computed: `gemm_pd_ ? gemm_pd_->name() : "gemm_t"` | **selected** `[reported]` | name computed at run time (`gemm_pd_ ? gemm_pd_->name() : "gemm_t"`); the log's `jit:gemm:any` is attributed here by elimination -- the log shows a nested `gemm` primitive of that name being created, so this entry is the wrapper that carries it |
| 2 | `intel::matmul::ref_sparse_t` | `ocl:ref:any` | **not tried** `[rule]` | after the winner in the list |
| 3 | `intel::matmul::grouped_micro_gemm_t` | `grouped_gemm:micro` | **build-conditional** `[rule]` | compiled out when !(DNNL_EXPERIMENTAL_GROUPED_MEMORY) |
| 4 | `intel::matmul::ref_grouped_t` | `ocl:ref_grouped:any` | **build-conditional** `[rule]` | compiled out when !(DNNL_EXPERIMENTAL_GROUPED_MEMORY) |
| 5 | `intel::matmul::ref_t` | `ocl:ref:any` | **build-conditional** `[rule]` | compiled out when DNNL_DISABLE_GPU_REF_KERNELS |
| 6 | `nvidia::cudnn_matmul_lt_t` | `cuda:cublaslt:any` | **not built** `[rule]` | built only for nvidia |
| 7 | `nvidia::cudnn_matmul_t` | `cuda:cudnn:any` | **not built** `[rule]` | built only for nvidia |
| 8 | `amd::miopen_matmul_t` | `hip:miopen:any` | **not built** `[rule]` | built only for amd |
| 9 | `generic::sycl::ref_matmul_t` | `sycl:ref:any` | **not built** `[rule]` | built only for generic sycl |


## 3. What would change the choice

### `jit:xe_hp:gemm:any` -- unsupported format tag `[reported]`

| | |
| --- | --- |
| gate `[cited]` | `src/gpu/intel/gemm/jit_xe_hp_systolic.cpp:79` |
| position `[rule]` | gate 3 of 19 in `xe_hp_systolic_t::pd_t::init`: it passed 2 checks before this one |
| descriptor offered `[reported]` | mds: `src_a:bf16::blocked:ab::f0 src_b:bf16::blocked:ba::f0 bia:undef::undef:::_mask0 dst:bf16::blocked:ab::f0`; attrs: `-`; problem: `4x2048:2048x2560` |
| lever `[interpretation]` | **operand layout** -- caller-controlled: operand memory format and strides -- pass `any` for a weight so the library may pick its packed layout, pre-pack it once, or make operands contiguous |
| still ahead of it `[rule]` | passing this gate is necessary, not sufficient: 16 more follow in the same function -- `VERBOSE_UNSUPPORTED_TAG`, `VERBOSE_UNSUPPORTED_TAG`, `VERBOSE_SKIP_PRIMITIVE_IMPL`, `VERBOSE_RUNTIMEDIM_UNSUPPORTED`, `VERBOSE_UNSUPPORTED_DT_CFG`, `VERBOSE_UNSUPPORTED_ARCH`, `VERBOSE_UNSUPPORTED_DEVICE_FEATURE`, `VERBOSE_UNSUPPORTED_ATTR`, `VERBOSE_UNSUPPORTED_FEATURE`, `VERBOSE_UNSUPPORTED_BIAS_CFG`, `VERBOSE_SHAPE_RESTRICTION`, `VERBOSE_UNSUPPORTED_ZP_CFG`, `VERBOSE_UNSUPPORTED_ZP_CFG`, `VERBOSE_UNSUPPORTED_ZP_CFG`, `VERBOSE_UNSUPPORTED_ZP_CFG`, `VERBOSE_UNSUPPORTED_ZP_CFG` |

```cpp
    VDISPATCH_GEMM(limits_ok, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
```


## 4. Strategy selection inside the implementation `[reported]`

The selected implementation scored 80 distinct catalog strategies for this problem. The log prints no winner; the selector orders by score ascending after preferring non-fallback alignment (`[rule]`: read from the selector source in `source/`), so the lowest score is the likely choice.

| score | strategy |
| --- | --- |
| -inf | `F gemm BBS T@8N@4N 4 16 am32+C32@64 at32 aS wg 1x1x16 ikr af vav sr sb256 bk0 bm0 sys rr` |
| -inf | `F gemm HHS T@8N@4N 4 16 am32+C32@64 at32 aS wg 1x1x16 ikr af vav sr sb256 bk0 bm0 sys rr` |
| 4.69386e+06 | `F gemm BBS TNN 16 8 at16x2+m64@16 aB32+m16@32 aB wg 4x2x4 kr af vav li nmk pt sr br sb64 bk0 sm sn dm grf256 sys kv afb l4 l2d` |
| 4.71392e+06 | `F gemm HHS TNN 16 8 at16x2+m64@16 aB32+m16@32 aB wg 4x2x4 kr af vav li nmk pt sr br sb64 bk0 sm sn dm grf256 sys kv afb l4 l2d` |
| 5.0733e+06 | `F gemm HHS T@4N@8N 16 16 at16x2+m32@48 am32+m16@64 aB wg 4x2x4 kr xaf st vav hi pt sr br sb64 bk0 sm sn grf256 sys kv afb` |
| 5.07604e+06 | `F gemm BBS T@4N@8N 16 16 at16x2+m32@48 am32+m16@64 aB wg 4x2x4 kr xaf st vav hi pt sr br sb64 bk0 sm sn grf256 sys kv afb` |
| 5.55901e+06 | `F gemm BBS T@4N@8N 32 32 at32+m32@48 am32+m32@48 aB wg 4x2x4 kr xaf vav hi pt sr br sb64 bk0 sm sn grf256 kv afb sys` |
| 5.66669e+06 | `F gemm HHS TNN 16 16 at16x2+m32@48 aB32 aB wg 4x2x4 kr cb4x2 ks64 af vav hi pt sr br bk0 sm sn dm grf256 kv afb sys l4 l2d` |
| 5.69259e+06 | `F gemm BBS TNN 16 16 at16x2+m32@48 aB32 aB wg 4x2x4 kr cb4x2 ks64 af vav hi pt sr br bk0 sm sn dm grf256 kv afb sys l4 l2d` |
| 5.71431e+06 | `F gemm HHS TNN 16 1 aS32+S32@48 aB32+S16@48 aB wg 4x1 af vav nmk li pt br sr sb256 bk0 sm grf256 sys l4 kd` |
| 5.72355e+06 | `F gemm BBS TNN 16 1 aS32+S32@48 aB32+S16@48 aB wg 4x1 af vav nmk li pt br sr sb256 bk0 sm grf256 sys l4 kd` |
| 5.76787e+06 | `F gemm BBS TNN 16 64 at16+m32@16 aB32 aB wg 8x1x4 kr cb4x2 ks32 xaf st vav hi pt sr br bk0 sm sn dm grf256 kv afb sys l4 l2d` |

## Source

- `source/src/gpu/intel/gemm/jit.hpp` (687 lines) `[rule]` -- selected implementation -- declares `jit:gemm:any` as `gen_t` (line 85)
- `source/src/gpu/intel/gemm/jit.cpp` (707 lines) `[rule]` -- selected implementation -- implements `gen_t`
- `source/src/gpu/gpu_gemm_list.cpp` (58 lines) `[rule]` -- candidate order for `gemm` -- walked top to bottom
- `source/src/gpu/gpu_matmul_list.cpp` (71 lines) `[rule]` -- candidate order for `matmul` -- walked top to bottom
- `source/src/gpu/intel/gemm/jit_xe_hp_systolic.cpp` (1138 lines) `[cited]` -- rejected `jit:xe_hp:gemm:any`: unsupported format tag (line 79)
- `source/src/gpu/intel/gemm/jit_xe_hp_systolic.hpp` (226 lines) `[rule]` -- rejected candidate -- declares `jit:xe_hp:gemm:any` as `xe_hp_systolic_t` (line 38)
- `source/src/gpu/intel/gemm/jit/gen_kernel.cpp` (1059 lines) `[rule]` -- scores the generator's strategies (prints `consider:`)
- `source/src/gpu/intel/gemm/jit/selector/db/kernel.db` (1311 lines) `[rule]` -- strategy catalog the scored entries are drawn from
- `source/src/gpu/intel/gemm/jit/selector/db/ukernel_lmr.db` (21 lines) `[rule]` -- strategy catalog the scored entries are drawn from
- `source/src/gpu/intel/gemm/jit/selector/db/ukernel_mlr.db` (30 lines) `[rule]` -- strategy catalog the scored entries are drawn from
- `source/src/gpu/intel/gemm/jit/selector/db/ukernel_mmr.db` (639 lines) `[rule]` -- strategy catalog the scored entries are drawn from

- gap: the checkout does not contain the revision that ran (`80afa71049cd69a3df32adcccb623b12cd7baa22` / `v3.12.0`), so files come from its working tree at `v3.13.2`; a `file:line` cited above may be off by the drift between the two. `git -C tmp/oneDNN fetch --tags && git checkout v3.12.0` aligns them.
- gap: `jit:gemm:any` was selected for `matmul`, but the declaration carrying that name lives outside any `matmul` directory: the `matmul` entry is a wrapper that takes its name from the nested primitive it delegates to, and the nested primitive is what is bundled. The wrapper is the computed-name entry in the `matmul` candidate list.

Files are copied whole from the checkout, at the revision above, so a `file:line` cited by
the library lands on the line it names. Read the candidate list first; the gates are in the
order they are tried.

## Reproduce

```
ONEDNN_VERBOSE=all python repro.py 2>&1 | grep '^onednn_verbose'
```

`repro.py` calls the op at the same shape and dtype the harness does; the output should match
`verbose.log` line for line, apart from timings and cache hits. `selection.json` holds the
same facts as this file in a fixed shape: regenerate the bundle after a library upgrade and
diff the two. Selection changes between versions, so a different `oneDNN v...` banner means a
different chain, not a bug in this bundle.

## Before optimizing

Bound it. `flashinfer_bench.device.calibration.get()` gives this part's substitution cost,
timing floor and achievable bandwidth; a win smaller than what taking it costs is a loss.
See `.claude/skills/route-kernel-work/PLAN.md`.

The win is usually in the call, not in a replacement kernel: `/optimize-onednn` walks the
fixes (weight layout, primitive caching, post-op fusion) against section 3 above.

Then: `python scripts/kernel_trials.py init aten_linear <this-dir>/harness.py`
