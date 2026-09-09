# The vendor's kernel-optimization corpus, and how to use it here

`tmp/Xe-Forge/knowledge_base/` is Intel's own corpus for driving kernel optimization on
this hardware: YAML entries keyed by DSL (Triton, SYCL, Gluon) and by target, each carrying
a description, a rationale, a before/after code pattern, and what it applies to. It is
better provenance for *what a kernel on this hardware can do* than anything written here,
and it is not a substitute for a measurement on this part.

This file is an index, not a copy. Read entries out of the checkout:

```bash
ls tmp/Xe-Forge/knowledge_base                       # the tree
grep -rn "id: <pattern_id>" tmp/Xe-Forge/knowledge_base
python3 -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); print('\n'.join(sorted(
  (i.get('id','') + ' | ' + i.get('name','')) for k in ('patterns','constraints') for i in d.get(k) or [])))" \
  tmp/Xe-Forge/knowledge_base/triton/xpu/memory_patterns.yaml
```

If the checkout is absent nothing here breaks: every entry below is a place to read a
worked pattern, never a step in a procedure.

## What to take, and what to leave

| Take | Leave |
| --- | --- |
| the **mechanism**: `description`, `rationale`, `pattern_before` / `pattern_after`, `applies_to` -- what the change is and what has to be true for it to apply | the `expected_speedup` field on every entry. It is what their runs measured on their parts; on this part the number is `SPEEDUP` from `scripts/kernel_trials.py benchmark`, and nothing else |
| **API and toolchain constraints** -- what the compiler, the backend or the hardware refuses or silently mis-compiles. Their subject is closed and they are checkable here | entries whose claim is a preference (`optimal for`, `best for`, `prefer X over Y`) with no test attached. The equivalent claim here is a row of `../../optimize-model-kernels/references/read-the-numbers.md` plus a trial |
| **applicability guards** -- the condition under which a pattern stops helping. These are the half that decides whether the pattern applies at all, and the half a summary drops | fixed tile tables, per-part hardware specs and performance targets: query the part (`architectures.md`) and calibrate it (`scripts/calibrate_part.py`) |
| the code patterns themselves, as drafts to adapt | the shape tables (`sycl/xpu/llm_workload_shapes.yaml`, `sycl/xpu/real_model_shapes.yaml`). Shapes come from the model under test, through `scripts/harness_from_model.py`; these are corroboration at most and are never an input |
| -- | `triton/xpu/optimization_levels.yaml`. It is a ranked ladder of what to try in which order, which is exactly the shape this repo's conventions reject: the order here is the output of a classification performed on the kernel in hand |

## Where each file is worth opening

| File | What it holds | Which section of the skill it feeds |
| --- | --- | --- |
| `triton/xpu/xpu_optimizations.yaml` | the backend's API constraints, and Triton launch/geometry patterns | "Write a Triton solution" |
| `triton/xpu/memory_patterns.yaml` | access-pattern and register-liveness patterns, with the guard on each | "Build, and check the failure mode for your language"; the memory row of the mechanisms reference |
| `triton/xpu/fusion_patterns.yaml` | when a fusion removes a materialised intermediate and when it collapses occupancy instead | `/find-kernel-gaps`; the launch-bound row |
| `triton/xpu/autotuning_patterns.yaml` | what belongs in a `Config` and what does not; which axes to sweep together | "Write a Triton solution" |
| `triton/xpu/persistent_kernel_patterns.yaml` | persistent and Stream-K decompositions, and the conditions under which each is pointless | the occupancy row |
| `triton/xpu/correctness.yaml`, `common/correctness.yaml` | what makes a kernel silently wrong on this backend | the correctness constraints in the gates reference |
| `triton/xpu/algorithmic_patterns.yaml`, `common/algorithmic_patterns.yaml` | reformulations that change the operation count, not the kernel | `/find-kernel-gaps` |
| `triton/xpu/dtype_optimizations.yaml` | accumulator and I/O dtype pairings | "Write a Triton solution" |
| `triton/xpu/examples/` | complete kernels for the fusions the corpus describes, with an `index.yaml` | drafts to adapt |
| `triton/xpu/implementation_reference.md` | prose walk-through of the Triton patterns | background |
| `sycl/xpu/cutlass_sycl_framework.yaml` | what the CUTLASS-SYCL framework offers: mainloop dispatch policies, tile schedulers, and the epilogue fusion catalog | "Write a SYCL solution"; `xe-fuse.md` |
| `sycl/xpu/xetla_patterns.yaml` | DPAS, SLM, block-load and sub-group-reduction patterns, and the matrix-engine constraints | `xe-matrix.md`; "Write a SYCL solution" |
| `gluon/xpu/gluon_xpu_patterns.yaml` | the same ground in a third DSL | background |

## Constraints worth carrying, because they make a kernel wrong or unbuildable

Source: `tmp/Xe-Forge/knowledge_base/triton/xpu/{xpu_optimizations,correctness}.yaml`, `triton/xpu/memory_patterns.yaml`, `common/correctness.yaml` -- entry ids in the first column

| Entry id | What it says |
| --- | --- |
| `autotune_no_duplicate_params` | a parameter supplied by `triton.autotune` must be declared in the signature without a default; declaring both conflicts |
| `grid_must_match_swizzling` | with a flattened program id for tile swizzling, the launch grid must be one-dimensional |
| `grf_mode_is_compiler_option_not_config_kwarg` | the register-file mode is a compiler option, declared as a `tl.constexpr` in the signature and never used in kernel logic; it is not an ordinary `Config` value |
| `boundary_check_tuple` | `boundary_check` takes dimension indices, not booleans |
| `block_ptr_vs_tma`, `descriptor_no_boundary_check_arg` | block pointers and tensor descriptors are different APIs; a descriptor's `load` takes no boundary check |
| `descriptor_no_atomic_support`, `mem_atomic_fallback_from_descriptors` | descriptors and block-pointer stores support no atomics; a kernel that accumulates atomically computes its pointers by hand |
| `xpu_packed_weight_requires_stride_pass_through` | a packed transpose passed into a kernel must carry its own strides, not the original tensor's |
| `int64_cast_for_large_batch_offsets` | batch and stride products cast to 64-bit before they can overflow a pointer |
| `streamk_output_must_be_prezeroed`, `streamk_atomic_add_needs_mask` | atomic accumulation needs a zeroed output and a mask on partial tiles |
| `output_dtype_must_match_original`, `no_cpu_return_from_xpu_kernel` | the output's dtype and device are part of the contract the harness checks |
| `no_device_to_host_scalar_sync` | a scalar read back in the hot path serializes the queue; it shows up as the host-bound row, not as kernel time |
| `no_tl_multiple_of_on_python_scalars` | alignment hints apply to offset tensors, not to Python scalars or strides -- and a false hint is wrong code, not slow code |
| `sycl_fp32_accumulator_required` | float32 accumulation for half-precision matrix multiplies |
| `sycl_slm_size_limits`, `sycl_slm_bank_conflict_padding` | shared local memory is budgeted per core, and a transpose access pattern over an unpadded array conflicts by construction |

## Generators it adds that are not obvious from the hardware alone

Each is a candidate for the regime named, taken from the corpus with its guard; the
mechanisms reference carries the same principles in general form.

| Entry id | The generator | The guard the corpus states with it |
| --- | --- | --- |
| `reduce_liveness_sink_load_and_prefetch`, `reduce_liveness_duplicate_load_for_multi_use`, `avoid_long_lived_large_tiles_across_control_flow`, `trade_bandwidth_for_regs_when_spilling` | spill is a liveness problem before it is a tile-size problem: prefetch early and load close to use, reload an operand near each distant use rather than keeping it live, and prefer a re-load over a spill | `do_not_apply_if_cache_eviction_likely` -- sinking a load only helps while the data stays in cache; with a working set that evicts, it converts cache hits into global traffic |
| `fuse_only_if_intermediate_is_materialized`, `fuse_when_bandwidth_bound`, `skip_fusion_when_noop_or_redundant` | a fusion pays when the unfused path writes an intermediate and reads it back | `fusion_register_pressure_guard` -- a fusion that lengthens live ranges can cost occupancy or spill more than the intermediate cost |
| `split_lse_into_separate_reduction_kernel` | the converse move: split a full-row reduction *out* of a tiled kernel, accepting one memory round trip to keep the tile parallelism | applies when the reduction spans the axis the kernel is tiled along |
| `streamk_two_wave_decomposition`, `persistent_kernel_autotuned_num_progs` | when the tile count does not divide evenly over the part, give the remainder tiles to a fixed number of programs that split the reduction and accumulate atomically, and tile the rest one-to-one | `persistent_kernel_minimum_grid_threshold`, `persistent_kernel_not_for_reduction_outputs` -- pointless when the grid is smaller than what the part holds resident, or when the output is already one program per row. Take the resident count from the device, not from the entry |
| `sum_matmul_reorder` | a reduction over a product can move inside it, so the operation count falls out of the product's inner dimension | valid by distributivity; the equivalence has to be checked on the actual dtype, since the summation order changes |
| `weight_statistic_precomputation`, `weight_transpose_pack_once`, `version_tracked_param_cache` | a statistic or layout that depends only on a weight is computed once and cached, keyed on the parameter's version so an update invalidates it | the cache is the mechanism; whether the transform is worth it at this shape is a load-time A/B |
| `mem_skip_boundary_check_when_divisible`, `even_m_n_k_constexpr_for_divisible_shapes` | specialize away boundary checks with a compile-time flag when the shape divides the tile | only where divisibility is guaranteed, with the general variant kept |
| `xpu_hardware_subslice_query`, `xpu_work_group_size_query` | derive program counts and rows-per-program from the device rather than from a constant | the corpus queries the device for exactly the reason this repo does |
| `sigmoid_via_exp2_approximation`, `xpu_exp2_attention_scaling` | a transcendental can be rewritten onto the base the hardware implements, folding the constant into the scale | it changes the arithmetic, so the definition's tolerance decides whether it is admissible, and a trial decides whether it wins |

## Where the corpus and this part's measurements disagree

Ours wins for this part, because it was measured here. Each disagreement is worth knowing
before a sweep is set up.

- **Sweeping the sub-group width on a matrix kernel.** The corpus asks for `warp_size` to be
  swept alongside `num_warps` (`xpu_warp_size_autotune`, `xpu_warp_sweep_no_fixed_32`).
  `xe-matrix.md`, "Sub-group width is 16 for matrix work", records that the Intel Triton
  backend overrides the requested width for any kernel containing a matrix multiply, so on
  such a kernel the sweep compiles the same kernel more than once. Sweep it on elementwise
  and row-wise kernels, where the width is honoured; check `kernel.metadata` before
  believing a sub-group sweep on a matrix kernel.
- **Fusing an epilogue into a generated GEMM.** The corpus treats a GEMM epilogue fusion as
  removing the intermediate outright. `xe-fuse.md`, "Three layout rules that produce silent
  garbage", records that the generated pairwise epilogue returns a fragment of the same
  width, so the output arrives duplicated and needs a compacting copy -- which costs what
  the fusion saved, and blocks chaining a second fused kernel. Establish the output layout
  against candidate references before timing anything.
- **Trusting an autotune result.** The corpus's autotuning entries assume the swept
  configuration is what ran. Two things here defeat that: the Triton cache can return an
  earlier configuration for a kernel whose key did not change, and with a default register
  mode the backend can rebuild in the large mode on a spill without saying so. The skill's
  "Build, and check the failure mode for your language" says how to read what was actually
  built.

## Sources

- `tmp/Xe-Forge/knowledge_base/` -- the corpus itself, the authority on every entry id above
- `xe-matrix.md`, `xe-fuse.md`, `architectures.md` -- what was measured or read out of the toolchain on this part
- `../../optimize-model-kernels/references/mechanisms.md` -- the same generators in regime-indexed form
- `../../optimize-model-kernels/references/gates.md` -- the correctness constraints a kernel has to hold here
