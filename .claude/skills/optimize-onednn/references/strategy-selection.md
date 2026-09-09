# Changing which kernel the GEMM selector generates

The axes in `../SKILL.md` change how the library is *called*. This one changes what it
*generates*, for one shape range only, and it requires rebuilding the library and shipping
that build. Reach for it when the reading in the skill's "What the selector considered"
shows the entry you want is not among the ones scored for this shape range, and a
measurement shows a different tile ahead at the shapes the stack presents.

## What the selector is

Intel GPU GEMMs are generated at run time by gemmstone from a shape-gated catalog of
strategy descriptors. Reading its decision is `ONEDNN_VERBOSE=debuginfo=5`, and the skill's
"What the selector considered" section carries the output format: `consider:` lines are
catalog entries scored against the problem, lowest score chosen, and the `kernel:` line
confirms what was generated.

## Changing an entry

Add or adjust an entry in
`tmp/oneDNN/src/gpu/intel/gemm/jit/selector/db/kernel.db`, gated on the precision, the
transpose pattern and the shape range:

```
{{'C', "gemm", {"B","B","S"}, {"N","T","N"}},   // bf16 x bf16 -> fp32 acc, N/T/N
 { ... shape gates, e.g. {-1, 8, -1} ... },     // applies only in this range
 "ab2 ab8 ab l4 cab1 wg 4x4 int sr",            // strategy string
 { 8, ..., {32,16,8}, {4,4,1}, ... }},          // unroll, tiles, work-group
```

Entries are shape-scoped, so a specialization leaves every other shape alone.
`generator/strategy_parser.cpp` defines the strategy-string grammar. There is no runtime
override: the catalog is compiled in, and `dev_getenv` exposes only generator debug
switches.

## Procedure

1. Establish by measurement that a different tile is ahead for this shape --
   `scripts/kernel_trials.py benchmark`, one process per point, with the two builds compared
   through `ab` since they cannot share a process.
2. Build with `scripts/build_onednn.py` into its own prefix, as the skill's "Gates" section
   describes, and point `FIB_ONEDNN_DIR` at it. `environment.libs` records `onednn` as
   `version+commit (resolved path)`, so a rebuilt library is distinguishable from a released
   one; still say what was changed in the solution's `description`, because the commit shows
   only that it is not stock.
3. Send the catalog entry upstream. It is shape-gated data, and it benefits everyone on that
   part.
