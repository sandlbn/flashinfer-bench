# Gates: what a number has to clear before it counts

A gate is enforced by machinery, not by the agent's judgement. Each one below names what
enforces it and what a failure means. Where a gate fails, the stage halts and withholds the
result it invalidates -- a delta that survives a failed gate is the line that gets quoted,
so none is printed.

## The contract a trial prints

`scripts/kernel_trials.py benchmark` emits one `KEY: value` per line and ends with `DONE`.
A driving loop reads keys; it never interprets a sentence, so a reworded note cannot change
what it concludes.

Source: `scripts/kernel_trials.py` (`_report`, `_report_result`, `cmd_benchmark`)

| Key | Values | Printed when | Read it as |
| --- | --- | --- | --- |
| `BUILD` | `OK`, `FAILED` | always | on `FAILED`, a `--- build/load diagnostics ---` block with the compiler's own text follows, and it is the next input to reason from |
| `SPILLS` | `none`, a count, `unknown` | always | read from the build log the command captured, else from unitrace. `unknown` means no compiler ran in view -- a cached build, a framework-only change, or a driver-side compile |
| `CORRECT` | `OK`, `FAILED` | after a successful build | on `FAILED`, `REASON` follows and **no timing key is printed at all** |
| `REASON` | text | only with `CORRECT: FAILED` | which argument or output differed, its error, and the tolerance it was judged against |
| `MAX_ABS_ERROR` | number | with `CORRECT: OK` | -- |
| `BASELINE_US`, `CANDIDATE_US` | per call | with `CORRECT: OK` | the thing a deployment would run, against this trial |
| `SPEEDUP` | ratio | with `CORRECT: OK` | `BASELINE_US / CANDIDATE_US` |
| `BASELINE_SPREAD_PCT`, `CANDIDATE_SPREAD_PCT` | percent | with `CORRECT: OK` | which arm is unsteady; neither enters the verdict |
| `SPREAD_PCT`, `NOISE_FLOOR_PCT` | percent | with `CORRECT: OK` | the paired ratio's own scatter, and the gain that would have counted |
| `ROUTING` | `OK`, `REJECTED`, `STALE`, `UNEVALUATED`, `UNCHECKED` | when the series was opened against a routing | `UNCHECKED` means no routing was given, not that one passed |
| `VERDICT` | `WIN`, `LOSS`, `NOISE`, `INCORRECT`, `BUILD_FAILED`, `UNCOMPARABLE`, `ROUTING_REJECTED` | always | the only field a branch decision reads |
| `DONE` | -- | always | end of block |

`NOISE` is a verdict, not a note: a difference inside the arms' paired scatter is neither
kept nor discarded on its ratio.

## Branching on the exit key

Source: `scripts/kernel_trials.py` (`_verdict`, `cmd_benchmark`) and `scripts/bound_candidates.py` (`routed_from_harness`)

| Exit key | Next `--parent` | What the next hypothesis reads | Note |
| --- | --- | --- | --- |
| `VERDICT: BUILD_FAILED` | the same parent as the failed trial | the diagnostics block, verbatim | a build failure is an input, not an exit |
| `VERDICT: INCORRECT` | the failed trial itself | `REASON` alone -- there is no timing to be tempted by | fix in place; never loosen `--atol`/`--rtol` to pass |
| `VERDICT: NOISE` | unchanged | raise `--rounds` or `--calls` once and re-measure the same trial; if it is still `NOISE`, treat it as `LOSS` | -- |
| `VERDICT: LOSS` | `best` | `SPEEDUP`, `SPREAD_PCT`, `SPILLS` | branch back, never forward from a regression |
| `VERDICT: WIN` with `SPILLS: none` | this trial | `SPEEDUP`, `SPREAD_PCT` | -- |
| `VERDICT: WIN` with a spill count | this trial | the spill first | a spilling win is not final: while spill is nonzero the memory and compute rows of `read-the-numbers.md` cannot be evaluated, so the bound is unknown |
| `VERDICT: UNCOMPARABLE` | unchanged | which input differed | the arms were not measuring the same problem |
| `VERDICT: ROUTING_REJECTED` | -- | the gate named in the refusal | the pair was priced out; change what that gate reads and re-run the routing, do not measure past it |

## The gates themselves

| Gate | Enforced by | What passes | What a failure means |
| --- | --- | --- | --- |
| the harness calls what the model called | `scripts/harness_from_model.py` verification, plus a unitrace kernel-name match | same op, same shapes and dtypes, same kernel name as the model launched | a different function is being tuned |
| correctness before timing | `kernel_trials.py benchmark`, which exits before it times anything and compares mutated arguments for in-place ops | the error clears the tolerance derived from the definition's dtype | fix the kernel; loosening the tolerance is never the fix |
| the baseline is the deployed kernel | the harness, which imports the op from where it lives | the baseline arm is the kernel a deployment would otherwise run, never a definition's reference | a ratio against a reference is not a deployment result |
| the difference clears the noise | `benchmark`'s paired rounds | `VERDICT: WIN` | inside spread is unmeasured, not a small win |
| the win is the kernel's, not the call path's | a control trial: the unmodified algorithm ported line for line onto the trial's call path | `best` beats the control outside spread | the ratio is a call-path artefact; nothing is attributed and nothing is patched |
| the predicted mechanism moved | unitrace, or whichever field the strategy's `moves=` clause named | the field moved | the win is unattributed: classify again on this node before branching from it |
| the routing still holds | `--bound`/`--mechanism` on `init`, re-checked on every `benchmark` | `ROUTING: OK` | `STALE` means discovery has been re-run since the routing; `REJECTED` means this pair was priced out |
| a substitution is worth its delivery | `calibration.get().dispatch_us` against the candidate's headroom, priced by `scripts/bound_candidates.py` | the ceiling exceeds the delivery cost | `dispatch_us` of `None` means the mechanism is unavailable, never free |
| the best trial is a win | `kernel_trials.py finalize`, `--require-win` on by default | `best` is a measured `WIN` | `--no-require-win` archives an abandoned series for the record, never to ship a kernel |
| the arms differ only in the change | the provenance each arm records from inside itself | stack entries identical, the component under test the only difference | a stack change would otherwise be booked to a kernel |
| the serving delta survives its own checks | `scripts/measure_serving_win.py` | delta outside spread on a window that survives doubling, equal token digests, substitution counters nonzero | no throughput table is printed at all; the verdict names the gate |
| nothing was installed unasked | the owner | explicit approval per install | a procedure violation regardless of the result |

## Closing a row

Stop and record which condition fired.

| Condition, in terms of keys and tree state | What it means |
| --- | --- |
| `best` is a `WIN` against production and against the control trial, outside spread | the row delivered; go on to attribution and deployment |
| `best` device time is within `SPREAD_PCT` of the bound its regime names | at the bound: the finding is which term binds, recorded with both numbers |
| `K` consecutive children of `best` with `VERDICT` in `{LOSS, NOISE}`, `K` agreed before the series starts and recorded in its first strategy string | plateau: classify again with an instrument the earlier trials did not use, and take a different row of `read-the-numbers.md`; if the next `K` also plateau, close |
| the trial budget is spent, counting `BUILD_FAILED` trials | close with the `status` output |
| `N` consecutive `INCORRECT` with the same `REASON` | the schema is being misread; re-read the bundle's `PROVENANCE.md`. Never loosen tolerance |
| `best` spills and every child that removed the spill is a `LOSS` | record both numbers and which shape they are for; the row stays open only while a row of `read-the-numbers.md` is untried |
| the ceiling for this mechanism falls below `SPREAD_PCT` after re-measurement | close as unroutable for this mechanism; the same candidate's other accepted mechanisms, if any, are separate rows |

## Correctness constraints a kernel has to hold

- **Output buffers may alias inputs.** Destination-passing callers pass an output aliased to
  an input deliberately. That is safe only if each work-item reads every index it needs
  before writing any index another work-item might still read.
- **Match the definition's dtype exactly.** A narrower mantissa cannot carry a dequantised
  low-bit weight, and a path that passes on most elements is not passing.
- **Accumulate in float32** even where inputs and outputs are half precision.
- **Verify a scaled or masked configuration across many output positions with varying
  values**, never a constant checked at one corner.

## What the agent never does inside the loop

Write a timing script; time against the definition's reference instead of the harness;
loosen a tolerance; carry a launch constant from another vendor without measuring it here;
keep a win whose predicted mechanism did not move; run two GPU jobs at once; install
anything.

## Who decides what

| Decision | Agent | Owner | Machinery |
| --- | --- | --- | --- |
| which candidate to work on | reads the worklist order | may reorder, with the reason recorded | the routing's ranking |
| the regime classification | yes, from `read-the-numbers.md`, recorded with its arithmetic | -- | the inputs come from the instruments |
| the hypothesis and the code change | yes | -- | -- |
| whether a number counts | no | no | `benchmark` for the scatter, unitrace for identity and mechanism |
| tolerances | no | by changing the definition's dtype | derived from that dtype's spacing |
| skipping the control trial | no | no | the gate above |
| the trial budget and `K` | reads them | sets them before the series starts | -- |
| closing a row | yes, when a condition above fires, with the reason recorded | may close earlier | -- |
| rebuilding a component in place, without installing | yes | -- | an overlay build |
| installing, `sudo`, promoting, switching the stack's branch | never | approves each | -- |
| declaring a deployment win | no | reads the serving result | `scripts/measure_serving_win.py` |

## Before any timed step

The GPU idle apart from the desktop (`fuser -v /dev/dri/renderD*`), the power profile
reading `performance` (`powerprofilesctl get`), and one benchmark at a time on one GPU.
The dev environment is `source .venv/bin/activate`; anything importing the serving stack
runs under the interpreter whose `python -c "import vllm"` succeeds, found as `CLAUDE.md`
"Python Environments" says. Never `uv run` or `uv pip` in either -- it replaces this box's
accelerator build of torch.

## Sources

- `scripts/kernel_trials.py` -- the contract, the verdict rule, `finalize --require-win`
- `scripts/bound_candidates.py` -- the routing check and every delivery gate
- `scripts/measure_serving_win.py` -- the serving gates
- `tools/kernel-harness/knowledge/README.md` -- the same rules as a trial-side checklist
- `read-the-numbers.md`, `mechanisms.md`, `tools.md`
