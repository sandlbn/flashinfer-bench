# Mechanisms: generating candidates from a regime

`read-the-numbers.md` says which row a kernel is in. This file says what kind of change that
row admits. Each entry is a **generator**, not a remedy: a principle, the measurement that
says it applies here, and the measurement that says the change worked. Deriving a specific
change nobody wrote down is the expected use; finding your change absent from this file is
not evidence against it.

Nothing here is ranked. Which candidate a trial tests is a judgement about this kernel,
made from its own numbers and recorded in the trial's `--strategy` string
(`regime=<row>; principle=<entry>; change=<what>; moves=<the key or field that must move>`).
The `moves=` clause is the point: a hypothesis that names no field it will move cannot be
checked, and a win whose named field did not move is unattributed -- keep the trial, but
classify again before branching from it (`gates.md`, "A win has to be the kernel's").

## Memory-bound

Applies when the row is `memory-bound-inefficient`; the layout entry applies when it is
`memory-bound-layout-limited`.

| Principle | It applies here when | It worked when |
| --- | --- | --- |
| Fewer passes over memory: fuse a producer into its consumer, work in place, drop an intermediate, recompute a value instead of reloading it | the composite's bytes exceed the bytes its mathematics requires -- `/find-kernel-gaps` measures this | `bytes_min` falls, and `t_dev` falls toward the new `t_mem` |
| Better locality: contiguous per sub-group, an access wide enough for a block load, avoiding a stride that camps one memory channel, staging a gather through shared local memory | `t_mem_pattern` exceeds `t_mem` outside `spread` | `bw_pattern` rises when the probe is re-run against the new pattern |
| More bytes in flight: the vector width the part offers (`caps.vector_width(itemsize)`), more rows per work-item, a longer prefetch distance | `t_dev` exceeds `t_mem_pattern` outside `spread` | `t_dev` approaches `t_mem_pattern`; the instruction mix shows the wider load |
| Cache reuse: the order tiles are traversed in, the working set against `caps.l2_bytes` | the working set crosses that size while the traversal revisits it | `t_dev` falls with no change to `bytes_min` |
| The data arrives in the wrong layout, or a value derived only from a weight is recomputed per call: do it once at load time and cache it against the parameter's version | the row is `memory-bound-layout-limited`, or the profile shows a weight-sized read whose result does not depend on the input | the transform is measured at the shapes the stack presents, and kept only where that A/B is a win |

## Compute-bound

Applies when the row is `compute-bound-inefficient`.

| Principle | It applies here when | It worked when |
| --- | --- | --- |
| Reach the matrix unit: a dtype the part runs natively, the operand layout the unit requires, the sub-group width it requires (`xe-matrix.md` in `optimize-intel-kernels/` carries those constraints with the source for each) | `t_dev` exceeds `t_cmp` outside `spread` and the instruction mix shows no matrix instructions | the mix shows them, and `t_dev` falls toward `t_cmp` |
| Tile shape against the register budget | `spill > 0`, or occupancy is below what the part holds | `spill` is zero and `t_dev` falls; a tile is one trial each, and `SPILLS` is reported on every build |
| Shorter live ranges, before a smaller tile: prefetch early and load close to the use, reload an operand near a distant second use instead of holding it, prefer an extra load to a spill | `spill > 0` while the tile is one the problem needs | `spill` falls with the tile unchanged. It can go the other way when the reloaded data does not survive in cache, so the reload trial is measured against the held-value trial |
| Fewer arithmetic operations: hoist an invariant, exploit a symmetry, move a reduction inside the product it is taken over where the algebra allows | `flops` overstates what the mathematics needs | `flops` falls, so `t_cmp` falls, and `t_dev` follows; the reformulation is checked for equivalence at the actual dtype, since it changes the summation order |
| Precision only where the part runs it natively | `caps.is_native_dtype()` is False for the dtype in the kernel | the definition's tolerance still passes and `t_dev` falls |
| Create parallelism when the output is small: split the reduction dimension across work-groups | `geom` fills a fraction of the part | `geom` fills it and `t_dev` falls |

## Launch-bound

Applies when the row is `launch-bound` or `host/sync-bound`.

| Principle | It applies here when | It worked when |
| --- | --- | --- |
| Remove a launch: fold the consumer into the producer's epilogue, merge two elementwise kernels, express the pair as one library primitive with a post-op | the model ran the pair back to back as an edge, not as two separate counts (`scripts/fusion_candidates.py`) | calls per step falls, and `t_host - t_dev` falls with it |
| Batch small calls into one | many calls of the same op at the same shape per step | call count falls |
| Cache what is being rebuilt per call: a primitive descriptor, a compiled kernel, an allocation | `create:` lines or an allocation appear in steady state | those lines disappear from the steady-state window |
| Remove a host wait the call does not need | a synchronization inside the call path that the framework's queue ordering already provides | `t_host` falls toward `t_dev` |
| Capture a repeated launch sequence as a graph | `caps.supports_graphs` is true and the sequence is stable | call count per step falls |

## Occupancy-limited

| Principle | It applies here when | It worked when |
| --- | --- | --- |
| Work-group size against `max_work_group_size` and the rows the problem has | a sweep of the work-group size moves `t_dev` outside `spread` | the swept value is measured, not assumed, and recorded per shape |
| The register-file mode trade: a larger file against threads resident per execution unit | `spill > 0` at the chosen tile | measured against the smaller-tile trial separately, never both changed at once |
| Rows or elements per work-item | rows are narrower than a sub-group | `geom` changes and `t_dev` falls |
| Give the remainder tiles their own decomposition when the tile count does not divide evenly over the part: a fixed number of programs splitting the reduction and accumulating atomically, the rest tiled one to one | the tile count from `geom` leaves a partial wave, and the part's resident-program count is queried rather than assumed | `t_dev` falls with no change to `bytes_min` or `flops`. Pointless where the grid is already smaller than what the part holds, or where the output is one program per row |
| Split a full-row reduction *out* of a tiled kernel, accepting one memory round trip, when fusing it would serialize the axis the kernel is tiled along | the reduction spans the tiled axis | the tiled kernel's `geom` recovers its parallelism and the pair's summed `t_dev` falls |

## A library or provider call

Applies whenever the resolver's class for the candidate is a library primitive or a
provider's op -- the kernel is not yours, and the axes are the ones a *caller* controls.

Source: the library's own API surface -- memory descriptors, primitive attributes, primitive
lifetime, and the implementation selector's own output.

| Axis a caller controls | Confirmed to have applied when | Confirmed a win when |
| --- | --- | --- |
| Operand descriptors: layout tag, strides, dtype | the descriptor field of the library's execution line changes | `t_dev` falls at the shapes the stack presents |
| Attributes: post-ops, scales, arithmetic mode | the attribute field of that line changes | as above, with correctness re-checked -- an arithmetic mode changes results |
| Primitive lifetime: created once against created per call | `create:` lines leave the steady-state window | `t_host` falls |
| Problem decomposition: split, merge, batch, or split the reduction | the number and shape of execution lines change | `t_dev` summed over the new calls falls |
| Synchronization: no host wait the queue ordering already gives | the wait is gone from the call path | `t_host` falls toward `t_dev` |
| Which implementation the selector chose, and whether another is reachable by changing a descriptor | the implementation field changes | `t_dev` falls |
| The library build behind the call | the version helpers report the build you intended | comparisons are only meaningful once both arms report the same build |

`/optimize-onednn` is this row worked through for the library `F.linear` reaches on this
backend.

## The target is at the wrong level

| Situation | What to do instead |
| --- | --- |
| the op is a decomposition -- a sequence of plain kernels, none of its own | classify and optimize what it decomposes to, or replace the sequence with one kernel |
| the op is registered from Python and the kernel is a reference | the reference is not the thing to tune; the candidate is a kernel that does not exist yet (`/optimize-intel-kernels`, "Entering from the routing") |
| the op is an ATen kernel inside the framework with no local source | there is nothing local to patch; either a kernel is written and bound at the call site, or the finding is an upstream report |

`scripts/pull_kernel_source.py` assigns the class; `scripts/bound_candidates.py` prices what
each class admits as a delivery, and its module docstring is the authority on both.

## Sources

- `read-the-numbers.md` -- the row this file is indexed by
- `scripts/bound_candidates.py` -- the delivery mechanisms, the fact that admits each, and what each costs
- `scripts/fusion_candidates.py` -- the edges the model actually ran, against the fusion tool's live preset list
- `optimize-intel-kernels/xe-matrix.md` -- matrix-unit and block-load constraints, with the source file for each
- `optimize-intel-kernels/architectures.md` -- what to query for a part, and the traps a deployment hits
- `optimize-intel-kernels/references/xe-forge-knowledge.md` -- the vendor's pattern corpus indexed against these rows, with the guard each pattern carries and where it disagrees with what this part measured
