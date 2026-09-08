"""Route vLLM's RMSNorm through the best solution recorded for it.

``vllm.model_executor.layers.layernorm.RMSNorm`` is a ``CustomOp``: it carries one
implementation per backend and ``dispatch_forward`` picks ``forward_xpu`` when the
platform is Intel. That method is the seam -- patching it puts a benchmarked kernel in
front of the one vLLM would otherwise use, without touching how the model is built.

Two definitions live behind that one method. ``RMSNorm.forward_xpu(x, residual=None)``
computes a plain norm when ``residual`` is None and a fused residual-add norm when it is
not, and vLLM signals the difference by returning one tensor or two. The definitions model
those separately, so the adapter routes on the argument rather than on the class.
"""

from __future__ import annotations

from typing import Any, Callable, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.patch_manager import PatchSpec

from . import stats


class RMSNormAdapter:
    """vLLM ``RMSNorm.forward_xpu`` -> ``rmsnorm_h*`` / ``fused_add_rmsnorm_h*``."""

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="vllm.model_executor.layers.layernorm.RMSNorm.forward_xpu",
                kind="method",
                name="rmsnorm_forward_xpu",
                ctx_key="vllm_rmsnorm",
            )
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self_, x: torch.Tensor, residual: torch.Tensor | None = None):
            def _fallback(**_kwargs):
                return orig(self_, x, residual)

            def _defer(reason):
                stats.record("rmsnorm", "unsupported", reason)
                return _fallback()

            weight = getattr(self_, "weight", None)
            if weight is None:
                return _defer("no-weight")
            hidden = int(weight.shape[0])
            if x.shape[-1] != hidden:
                return _defer("width")
            if residual is not None and residual.shape != x.shape:
                return _defer("residual-shape")

            # A 3D activation is [batch, seq, hidden] and the definitions are 2D. Deferring
            # on that was measured to decline the overwhelming majority of this family's
            # calls on models that never flatten before the layer, so flatten here instead.
            #
            # `view`, not `reshape`: it aliases the caller's storage rather than copying,
            # which the in-place residual path below depends on, and it raises instead of
            # silently copying if the layout ever stops being contiguous. Contiguity is
            # checked first so that raise can never happen at runtime.
            shape = x.shape
            if x.dim() == 2:
                xf, rf = x, residual
            elif x.is_contiguous() and (residual is None or residual.is_contiguous()):
                xf = x.view(-1, hidden)
                rf = residual.view(-1, hidden) if residual is not None else None
            else:
                return _defer(f"{x.dim()}d-noncontiguous")

            if residual is None:
                missed = False

                def _miss(**_kwargs):
                    nonlocal missed
                    missed = True
                    return _fallback()

                out = apply(
                    f"rmsnorm_h{hidden}",
                    kwargs={"hidden_states": xf, "weight": weight},
                    fallback=_miss,
                )
                stats.record("rmsnorm", "no-solution" if missed else "applied", f"h{hidden}")
                # `_miss` returns the fallback's own result, which is already the caller's
                # shape; only a result computed on the flattened view needs restoring.
                return out if missed or out.shape == shape else out.view(shape)

            # vLLM's fused form returns (normed, new_residual) and callers rely on both.
            # Prefer the definition that returns the summed residual too. The single-output
            # form forces `x + residual` to be recomputed here, a whole extra pass over
            # [tokens, hidden] that costs more than the kernel saves: measured on Arc B580
            # at 4096x1024, the kernel alone is 54.5us against vLLM's 103.9us, and the
            # recomputation takes it to 115.8us -- a 2.6x win turned into a loss.
            missed = False

            def _miss(**_kwargs):
                nonlocal missed
                missed = True
                return _fallback()

            # In-place, exactly as vLLM's own kernel is: it overwrites `input` with the
            # norm and `residual` with the sum, and its callers are written against that.
            # Allocating a fresh pair here instead cost two [tokens, hidden] allocations on
            # every one of ~7.5k calls per run and showed up as a throughput regression
            # with no device-time cause. Aliasing is safe because the kernel reads index i
            # of both inputs before writing index i of both outputs.
            residual_out = rf
            out = xf
            apply(
                f"fused_add_rmsnorm_residual_h{hidden}",
                kwargs={
                    "hidden_states": xf,
                    "residual": rf,
                    "weight": weight,
                    "output": out,
                    "residual_out": residual_out,
                },
                fallback=_miss,
            )
            if not missed:
                stats.record("fused_add_rmsnorm", "applied", f"h{hidden}+res")
                # The writes landed in the caller's own storage through the views, so hand
                # back the originals rather than the flattened aliases.
                return x, residual

            # No two-output solution here. The single-output one is still worth trying, but
            # only where the recomputation is cheap relative to the kernel -- which it is
            # not at prefill widths, so this stays a fallback rather than the default path.
            missed = False
            out = apply(
                f"fused_add_rmsnorm_h{hidden}",
                kwargs={"hidden_states": xf, "residual": rf, "weight": weight},
                fallback=_miss,
            )
            stats.record("fused_add_rmsnorm", "no-solution" if missed else "applied", f"h{hidden}")
            if isinstance(out, tuple):
                return out
            if missed:
                return out
            return out.view(shape), (xf + rf).view(shape)

        return wrapper
