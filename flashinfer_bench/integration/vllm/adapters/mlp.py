"""Route vLLM's gated MLP through a fused gate/up GEMM when the batch is large enough.

vLLM's MLP is three steps: a merged gate/up projection, a separate ``silu_and_mul``
kernel, then the down projection. Folding the activation into the projection's epilogue
removes a kernel launch and a round trip through ``[M, 2d]``.

**This adapter is off by default pending an end-to-end A/B.** Set FIB_VLLM_MLP_FUSION=1,
and set FIB_VLLM_MLP_MIN_TOKENS to the batch size above which the fused path wins *on the
part it is deployed on*. There is no built-in threshold: the crossover is a measurement of
one part, one model shape and one software stack, and a stored one is wrong on the next.
With fusion enabled and no threshold set, the adapter installs, fuses nothing, and prints a
histogram of the token counts the scheduler actually hands the MLP at exit -- those are the
M values to measure the crossover at.

**Measure against what vLLM actually runs, not against the reference.** This definition's
`reference` computes the projection as *two* GEMMs; vLLM issues *one* merged GEMM and then
its own fused `silu_and_mul`, so the reference is roughly 2x worse than production before
any kernel is written and every ratio taken against it inherits that factor. The baseline
is a harness that calls vLLM's own path (`/wrap-kernel-for-tuning`), timed by
`scripts/kernel_trials.py`; `scripts/rank_vs_provider.py` reads the crossover out of traces
once both sides are recorded on this part.

Two measurement artefacts have produced wrong conclusions here before, and both matter when
reproducing the crossover:

- **A per-call host block.** An earlier revision concluded the fusion could not win because
  oneDNN post-ops need two [M,k]x[k,d] matmuls against vLLM's one [M,k]x[k,2d]. The
  solution being measured called `ctx.stream.wait()` after every execute. The oneDNN stream
  wraps PyTorch's own SYCL queue, so ordering already holds and the caller synchronizes when
  it needs the result; removing the block inverted the comparison. Structure was never the
  problem.
- **Sequential timing.** A later revision concluded the fused path loses badly at decode.
  The cases were timed in sequence rather than interleaved, so the first absorbed the GPU's
  clock ramp. Interleaving the rounds and taking medians removed it. **Time alternatives in
  interleaved rounds; a sequential sweep charges the ramp to whichever case runs first.**

The floor a perfect fusion would reach is two plain half-GEMMs with no epilogue at all;
measure that alongside the fused kernel to see what the epilogue itself costs.

The token gate bounds the risk at sizes nobody has measured; below the measured crossover
vLLM's two kernels are the safer path.

It also takes the merged ``[2d, k]`` weight exactly as vLLM stores it. Splitting it into
two ``[k, d]`` operands would mean a transposed copy of every MLP weight, which on a small
model is already hundreds of megabytes and scales with the model; the kernel reads the two
halves through strided descriptors instead.
"""

from __future__ import annotations

import atexit
import collections
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec

logger = logging.getLogger(__name__)

ENABLE_ENV = "FIB_VLLM_MLP_FUSION"
"""Opt in to MLP fusion. Off by default: an experiment, not a proven serving win."""

MIN_TOKENS_ENV = "FIB_VLLM_MLP_MIN_TOKENS"
"""Batch size at which fusion switches on, measured on the deployed part.

Unset means never fuse: the crossover is a property of the part and the shape, so there is
no default to fall back to. Measure it with ``scripts/kernel_trials.py`` at the token counts
the exit histogram reports, or read it from traces with ``scripts/rank_vs_provider.py``.
"""

# Every model whose MLP is `gate_up_proj -> act_fn -> down_proj`. Qwen3 imports Qwen2MLP
# directly, and ~47 model files in vLLM share this shape.
_MLP_PATHS = (
    "vllm.model_executor.models.qwen2.Qwen2MLP.forward",
    "vllm.model_executor.models.llama.LlamaMLP.forward",
)


_SEEN: "collections.Counter[tuple]" = collections.Counter()
"""How many tokens the scheduler actually hands the MLP, and what we did about it.

The threshold was chosen from microbenchmark batch sizes, but vLLM chunks prefill and
schedules against ``max_num_batched_tokens``, so the M this layer sees is the scheduler's
choice, not the request's. Without measuring it there is no way to tell a fused path that
never fires from one that fires and does not help.
"""


def _bucket(m: int) -> str:
    """Power-of-two bucket, so the histogram stays readable across three orders."""
    if m <= 1:
        return "1"
    hi = 1 << (m - 1).bit_length()
    return f"{hi // 2 + 1}-{hi}"


def mlp_dispatch_stats() -> Dict[str, int]:
    """Observed token counts per outcome, as ``{"fused 513-1024": n, ...}``."""
    return {f"{outcome} {bucket}": n for (outcome, bucket), n in sorted(_SEEN.items())}


def _log_stats() -> None:
    if not _SEEN:
        return
    total = sum(_SEEN.values())
    fused = sum(n for (outcome, _), n in _SEEN.items() if outcome == "fused")
    message = (
        f"[flashinfer-bench] vLLM MLP dispatch: {total} call(s), {fused} fused "
        f"({100.0 * fused / total:.1f}%); by token count: {mlp_dispatch_stats()}"
    )
    logger.info("%s", message)
    # Also straight to stderr. This runs in vLLM's EngineCore subprocess, where nothing
    # has configured a handler for this package's logger, so logger.info alone is silently
    # dropped -- which is how the first instrumented run produced no histogram at all.
    print(message, file=sys.stderr, flush=True)


atexit.register(_log_stats)


def _fusion_enabled() -> bool:
    """Whether the user asked for MLP fusion. Off unless explicitly set."""
    return os.environ.get(ENABLE_ENV, "").lower() in ("1", "true", "yes", "on")


def _min_tokens() -> Optional[int]:
    """The measured crossover, or None when none has been supplied."""
    raw = os.environ.get(MIN_TOKENS_ENV)
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer; fusing nothing", MIN_TOKENS_ENV, raw)
        return None


class GatedMLPAdapter:
    """vLLM gated-MLP ``forward`` -> fused gate/up GEMM above a token threshold."""

    def targets(self) -> List[PatchSpec]:
        if not _fusion_enabled():
            # Not shown to win end to end; see the module docstring. Patching nothing is
            # the difference between an experiment someone can opt into and a serving
            # stack that is quietly slower for installing this.
            return []
        if _min_tokens() is None:
            logger.warning(
                "%s is set but %s is not: no crossover has been measured on this part, so "
                "the MLP adapter will fuse nothing and report the token counts it saw at "
                "exit. Measure at those sizes and set %s.",
                ENABLE_ENV,
                MIN_TOKENS_ENV,
                MIN_TOKENS_ENV,
            )
        return [
            PatchSpec(
                path=path,
                kind="method",
                name=f"mlp_forward:{path.rsplit('.', 2)[-2]}",
                ctx_key="vllm_gated_mlp",
            )
            for path in _MLP_PATHS
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        threshold = _min_tokens()

        def wrapper(self_, x: torch.Tensor):
            def _fallback(**_kwargs):
                return orig(self_, x)

            if x.dim() != 2:
                _SEEN[("skipped", f"{x.dim()}d")] += 1
                return _fallback()
            tokens = int(x.shape[0])
            if threshold is None:
                # No measured crossover on this part: record what the scheduler asked for,
                # which is where to measure one, and run vLLM's own path.
                _SEEN[("unmeasured", _bucket(tokens))] += 1
                return _fallback()
            if tokens < threshold:
                # Decode and short prefills: vLLM's two kernels are faster here.
                _SEEN[("deferred", _bucket(tokens))] += 1
                return _fallback()

            gate_up = getattr(self_, "gate_up_proj", None)
            down = getattr(self_, "down_proj", None)
            weight = getattr(gate_up, "weight", None)
            if weight is None or down is None or weight.dim() != 2:
                _SEEN[("unsupported", "shape")] += 1
                return _fallback()
            # Quantised or bias-carrying projections change the maths; leave them alone.
            if getattr(gate_up, "bias", None) is not None or weight.dtype != x.dtype:
                _SEEN[("unsupported", "bias-or-dtype")] += 1
                return _fallback()

            two_d, k = weight.shape
            if two_d % 2 or k != x.shape[1]:
                _SEEN[("unsupported", "shape")] += 1
                return _fallback()
            d = two_d // 2

            up = torch.empty(x.shape[0], d, dtype=x.dtype, device=x.device)
            out = torch.empty(x.shape[0], d, dtype=x.dtype, device=x.device)
            # This call is destination-passing -- four arguments for two inputs and two
            # outputs -- and `apply` returns None on *success* in that style, writing the
            # result into `out`. So the return value cannot distinguish a hit from a miss;
            # only whether the fallback ran can. Reading it as a miss made every fused call
            # recompute the MLP through vLLM as well, which is slower than not fusing.
            missed = False

            def _miss(**_kwargs):
                nonlocal missed
                missed = True
                return None

            apply(
                f"gemm_swiglu_merged_k{k}_d{d}",
                kwargs={"x": x, "w_gate_up": weight, "up": up, "out": out},
                fallback=_miss,
            )
            if missed:
                _SEEN[("no-solution", _bucket(tokens))] += 1
                return _fallback()
            _SEEN[("fused", _bucket(tokens))] += 1
            result, _ = down(out)
            return result

        return wrapper
