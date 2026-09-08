"""Route vLLM's gated MLP through a fused gate/up GEMM when the batch is large enough.

vLLM's MLP is three steps: a merged gate/up projection, a separate ``silu_and_mul``
kernel, then the down projection. Folding the activation into the projection's epilogue
removes a kernel launch and a round trip through ``[M, 2d]``.

**This adapter is off by default pending an end-to-end A/B.** Set FIB_VLLM_MLP_FUSION=1.

Against the path vLLM actually runs on XPU -- one merged GEMM followed by its own fused
`torch.ops._C.silu_and_mul` -- on Qwen3-0.6B shapes (k=1024, d=3072, bf16, Arc B580,
device-event timing, median of 100, 2026-09-06):

    M=512  1.09x   M=1024 0.98x   M=2048 1.16x
    M=3072 1.21x   M=4096 1.18x   M=8192 1.17x

An earlier revision of this note recorded 0.35x-0.99x and concluded the fusion could not
win because oneDNN post-ops need two [M,k]x[k,d] matmuls against vLLM's one [M,k]x[k,2d].
That conclusion was wrong: the solution being measured called `ctx.stream.wait()` after
every execute, blocking the host on each call. The oneDNN stream wraps PyTorch's own SYCL
queue, so ordering already holds and the caller synchronizes when it needs the result.
Removing the block took the same kernel from 0.44x to 2.21x against the reference at m=1.
Structure was never the problem; a per-call host block was.

**Measured against what vLLM actually runs, not against the reference.** This definition's
`reference` computes the projection as *two* GEMMs; vLLM issues *one* merged GEMM and then
its own fused `silu_and_mul`, so the reference is roughly 2x worse than production before
any kernel is written and every ratio taken against it inherits that factor.

Against vLLM's real path (Arc B580, k=1024 d=3072, bf16, medians of interleaved rounds,
2026-09-08):

    M=64    ours 22.7us   vLLM 24.6us   1.09x
    M=701   ours 109.9us  vLLM 144.4us  1.31x
    M=2801  ours 464.6us  vLLM 565.2us  1.22x

An earlier revision of this note recorded 0.53x at M=64 and concluded the fused path loses
badly at decode. That was a measurement artefact: the cases were timed in sequence rather
than interleaved, so the first one absorbed the GPU's clock ramp. Interleaving the rounds
and taking medians removes it, and the two distributions then do not overlap. **Time
alternatives in interleaved rounds on this part; a sequential sweep charges the ramp to
whichever case runs first.**

There is still headroom rather than a finished kernel: two plain half-GEMMs with no epilogue
at all measure 17.8us at M=64, so the fused epilogue is costing about 5us over the floor a
perfect fusion would reach.

The token gate stays because it bounds the risk at sizes nobody has measured, not because
the fused path was shown to lose.

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
from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec

logger = logging.getLogger(__name__)

ENABLE_ENV = "FIB_VLLM_MLP_FUSION"
"""Opt in to MLP fusion. Off by default: measured slower than vLLM at every batch size."""

MIN_TOKENS_ENV = "FIB_VLLM_MLP_MIN_TOKENS"
"""Override the batch size at which fusion switches on, when fusion is enabled at all."""

DEFAULT_MIN_TOKENS = 2048
"""Where the win becomes consistent: ~1.17x at and above 2048, parity at 1024.

Below this the fused path is not a regression either, but the margin is inside measurement
noise and not worth the dispatch cost.
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


def _min_tokens() -> int:
    raw = os.environ.get(MIN_TOKENS_ENV)
    if not raw:
        return DEFAULT_MIN_TOKENS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", MIN_TOKENS_ENV, raw, DEFAULT_MIN_TOKENS)
        return DEFAULT_MIN_TOKENS


class GatedMLPAdapter:
    """vLLM gated-MLP ``forward`` -> fused gate/up GEMM above a token threshold."""

    def targets(self) -> List[PatchSpec]:
        if not _fusion_enabled():
            # Measured slower than vLLM at every batch size on Battlemage; see the module
            # docstring. Patching nothing is the difference between an experiment someone
            # can opt into and a serving stack that is quietly slower for installing this.
            return []
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
