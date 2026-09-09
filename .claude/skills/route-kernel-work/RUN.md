# End-to-end run plan

One command's worth of stages, executed in order, with nothing named by hand. Every
decision below comes from a measurement taken on this box, of this model, under this stack.

## The rule this exists to enforce

**Share is measured per model, per stack, every run.** A ranking from another model is not
evidence about this one. The same pipeline on a 0.6B at batch 4 and on a 30B at batch 256
returns different targets, and the correct behaviour is to re-measure, not to remember.

## Stages

| # | Stage | Command | Gate to pass |
| --- | --- | --- | --- |
| 1 | Discover | `harness_from_model.py --model M` | ops + shapes + **device-time share** recorded to `discovered.json`; verified harnesses emitted |
| 2 | Resolve | `pull_kernel_source.py --from-report ... --bundle ...` | every op assigned a provider and a route; source bundled where one exists |
| 3 | Rank | by measured share, not call count | anything under the noise floor is dropped with its number stated |
| 4 | Bound | `calibration.get()` per candidate, per mechanism | `net_gain <= 0` ⇒ unroutable, say why and skip |
| 5 | Attempt | the route each op was assigned | a trial that fails is recorded, not hidden |
| 6 | Prove | `measure_serving_win.py --overhead-arm` | tokens/sec A/B; per-kernel ratio is not a result |
| 7 | Feed back | `no-solution` lines → stage 1 candidates | loop |

## Routes, and what "attempt" means for each

| Resolved as | Attempt |
| --- | --- |
| oneDNN | `/optimize-onednn` — layout, fpmath mode, post-ops, splitting a primitive. **Not** a replacement GEMM |
| provider kernel | trial loop from the bundled source; the first moves are its own constants, measured not assumed |
| Python-registered op | `/wrap-kernel-for-tuning` — it needs the stack's context; no substitution |
| Triton | tune in place at the JIT site |
| ATen inside PyTorch | measure; if it is slow, that is an upstream report |
| decomposition | optimize what it decomposes to |

## Every candidate gets attempted, in share order

Not just the largest. Share and headroom are independent quantities. Ranking by
`share x reachable headroom` is the whole point of stage 4, and it cannot be shortcut by
looking at the share column alone.

Stop attempting when the remaining candidates' *bounds* are below what a substitution
costs, and say what the cutoff was.
