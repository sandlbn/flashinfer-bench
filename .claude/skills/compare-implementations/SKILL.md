---
name: compare-implementations
description: Establish, by measurement, whether an alternative implementation of an operation is competitive with the one the stack runs today — a Triton kernel against a vendor's compiled kernel, a hand-written kernel against a library call, one library version against another. Use when the routing needs that comparison as an input, or before committing search effort to a language or source it has no evidence for.
---

# Compare implementations

The routing prices a delivery mechanism against a measured ceiling. It cannot price
*authoring* without knowing whether a kernel written another way is competitive at all for
this class of operation on this part. That is an empirical question, and this is how to
answer it once so the answer can be reused.

The result is a data point about one operation class, at stated shapes, on one part, with
one toolchain version. It is not a claim about a language.

## Step 1: Establish what runs today

Resolve the operation to its current implementation and take that as the baseline —
`scripts/pull_kernel_source.py` reports the class and, where a source bundle exists, the
kernel itself. The baseline is what the serving stack executes, never a reference
implementation, and never your own reimplementation of it.

Take the shapes and dtype from the verified harness `scripts/harness_from_model.py`
emitted, so the comparison is at sizes the model actually ran.

## Step 2: Decide what a fair comparison requires

Before writing anything, decide and record:

| Question | Why it decides the result |
| --- | --- |
| Does the alternative compute the same function? | A different rounding order or a fused epilogue is a different operation; compare those separately or not at all |
| Does it use the same call path? | A wrapper's cost can exceed the difference being measured. Both arms must enter through the same path, or the difference is attributed to the wrong thing |
| Is the operand residency the same? | An operand resident in cache and one streamed from memory are different problems; state which you are measuring and size the working set to match |
| Which knobs are the alternative's, and which are the problem's? | Tuning one arm and not the other measures effort, not implementation |

## Step 3: Measure both arms through the sanctioned timer

`scripts/kernel_trials.py` is the only path that gates correctness before timing, pairs the
arms, and reports a spread with the verdict. Its `ab` subcommand runs each arm in its own
process, which is required when two implementations cannot coexist in one — a rebuilt
library, or two registrations of the same symbol.

Search both arms with comparable effort. An autotuned implementation compared against a
default-configured one measures the autotuner.

Confirm the search actually ran: a tuner that silently reused a cached configuration
produces a number that describes neither implementation. Check that the chosen
configuration differs between a small and a large problem.

## Step 4: Attribute the difference before believing it

Port the baseline's own algorithm through the alternative's call path and measure that
too. Whatever that control shows is the call path, not the implementation; only what
remains belongs to the language or the source.

## Step 5: Record it as an input, not a conclusion

State the operation class, the shapes, the dtype, the part, and the versions of both
toolchains. A comparison whose toolchain versions are unrecorded cannot be repeated, and
implementation selection changes between versions.

Feed it where the routing reads calibrated inputs; a number kept in prose is a number that
will be quoted after it stops being true.

## When not to run this

Where the routing already rejects every authoring mechanism on measured grounds, this
comparison changes nothing and costs device time. Run it when a gate needs the input, not
to satisfy curiosity about a language.

## Sources

- `scripts/pull_kernel_source.py` — what implements the operation now
- `scripts/harness_from_model.py` — the shapes to compare at
- `scripts/kernel_trials.py` — the timer, and `ab` for arms that cannot share a process
- `/optimize-intel-kernels` — writing the alternative
- `/discover-model-kernels` — where the baseline and its bundle come from
