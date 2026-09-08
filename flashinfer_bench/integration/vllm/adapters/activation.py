"""Route vLLM's SwiGLU through the best solution recorded for it.

``SiluAndMul.forward_xpu(x)`` computes ``silu(x[..., :d]) * x[..., d:]`` -- the split-half
convention, matching the ``activation`` definitions. The halves are *not* interleaved; a
fused GEMM epilogue may hand over interleaved pairs instead, which is the same maths in an
incompatible layout, so this adapter deliberately only claims the split form.
"""

from __future__ import annotations

from typing import Any, Callable, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec

from . import stats
from .naming import candidates


class SiluAndMulAdapter:
    """vLLM ``SiluAndMul.forward_xpu`` -> ``silu_and_mul_d*``."""

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="vllm.model_executor.layers.activation.SiluAndMul.forward_xpu",
                kind="method",
                name="silu_and_mul_forward_xpu",
                ctx_key="vllm_silu_and_mul",
            )
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self_, x: torch.Tensor):
            def _fallback(**_kwargs):
                return orig(self_, x)

            if x.shape[-1] % 2 != 0:
                stats.record("silu_and_mul", "unsupported", "odd-width")
                return _fallback()

            # The definitions are 2D [tokens, 2d]; vLLM also calls this with
            # [batch, seq, 2d]. Deferring on that declines most of this family's calls on
            # models that do not flatten before the layer, so flatten here. `view` aliases
            # rather than copying, and raises rather than silently copying if the layout is
            # ever not contiguous -- which the check above makes unreachable.
            shape = x.shape
            if x.dim() == 2:
                xf = x
            elif x.is_contiguous():
                xf = x.view(-1, x.shape[-1])
            else:
                stats.record("silu_and_mul", "unsupported", f"{x.dim()}d-noncontiguous")
                return _fallback()
            d = x.shape[-1] // 2
            missed = False

            def _miss(**_kwargs):
                nonlocal missed
                missed = True
                return None

            # Try the dtype-qualified definition before the bare one. The fallback records
            # the miss and returns None rather than running the original, so trying a
            # second name costs nothing and cannot double-apply.
            out = None
            for name in candidates(f"silu_and_mul_d{d}", x.dtype):
                missed = False
                out = apply(name, kwargs={"x": xf}, fallback=_miss)
                if not missed:
                    break
            if missed:
                stats.record("silu_and_mul", "no-solution", f"d{d}")
                return _fallback()
            stats.record("silu_and_mul", "applied", f"d{d}")
            # The output's last dim is d, not 2d, so restore the leading dims only.
            return out.view(*shape[:-1], d)

        return wrapper
