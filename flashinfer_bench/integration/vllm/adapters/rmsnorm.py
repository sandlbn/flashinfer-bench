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
from .naming import candidates


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
                    return None

                # Dtype-qualified name first; the bare one is what most definitions use.
                out = None
                for name in candidates(f"rmsnorm_h{hidden}", x.dtype):
                    missed = False
                    out = apply(
                        name, kwargs={"hidden_states": xf, "weight": weight}, fallback=_miss
                    )
                    if not missed:
                        break
                if missed:
                    stats.record("rmsnorm", "no-solution", f"h{hidden}")
                    return _fallback()
                stats.record("rmsnorm", "applied", f"h{hidden}")
                return out if out.shape == shape else out.view(shape)

            # vLLM's fused form returns (normed, new_residual) and callers rely on both.
            # Prefer the definition that returns the summed residual too. The single-output
            # form forces `x + residual` to be recomputed here, a whole extra pass over
            # [tokens, hidden] that costs more than the kernel saves: measured on Arc B580
            # at 4096x1024, the kernel alone is 54.5us against vLLM's 103.9us, and the
            # recomputation takes it to 115.8us -- a 2.6x win turned into a loss.
            # This fallback must NOT run `orig`. vLLM's fused kernel is in place: it
            # overwrites `x` with the norm and `residual` with the sum. Running it here
            # would consume both buffers, and the second attempt below would then dispatch
            # on already-normed data -- computing rmsnorm(norm(x+r) + (x+r)) and returning a
            # corrupted residual stream, while the counter reported "applied". On a double
            # miss it also ran `orig` twice per call, which is real work charged to every
            # call the family makes. Defer the single `orig` call to the end.
            missed_residual = False

            def _miss_residual(**_kwargs):
                nonlocal missed_residual
                missed_residual = True
                return None

            # In-place, exactly as vLLM's own kernel is: it overwrites `input` with the
            # norm and `residual` with the sum, and its callers are written against that.
            # Allocating a fresh pair here instead cost two [tokens, hidden] allocations on
            # every one of ~7.5k calls per run and showed up as a throughput regression
            # with no device-time cause. Aliasing is safe because the kernel reads index i
            # of both inputs before writing index i of both outputs.
            residual_out = rf
            out = xf
            for name in candidates(f"fused_add_rmsnorm_residual_h{hidden}", x.dtype):
                missed_residual = False
                apply(
                    name,
                    kwargs={
                        "hidden_states": xf,
                        "residual": rf,
                        "weight": weight,
                        "output": out,
                        "residual_out": residual_out,
                    },
                    fallback=_miss_residual,
                )
                if not missed_residual:
                    break
            if not missed_residual:
                stats.record("fused_add_rmsnorm", "applied", f"h{hidden}+res")
                # The writes landed in the caller's own storage through the views, so hand
                # back the originals rather than the flattened aliases.
                return x, residual

            # No two-output solution. The single-output one is still worth trying -- the
            # buffers are untouched, because the fallback above deliberately did nothing --
            # but only where recomputing `x + residual` is cheap relative to the kernel,
            # which it is not at prefill widths. So this stays a fallback, not the default.
            missed_single = False

            def _miss_single(**_kwargs):
                nonlocal missed_single
                missed_single = True
                return None

            out = None
            for name in candidates(f"fused_add_rmsnorm_h{hidden}", x.dtype):
                missed_single = False
                out = apply(
                    name,
                    kwargs={"hidden_states": xf, "residual": rf, "weight": weight},
                    fallback=_miss_single,
                )
                if not missed_single:
                    break
            if missed_single:
                stats.record("fused_add_rmsnorm", "no-solution", f"h{hidden}")
                return _fallback()  # the one and only call into vLLM's in-place kernel
            stats.record("fused_add_rmsnorm", "applied", f"h{hidden}")
            if isinstance(out, tuple):
                return out
            return out.view(shape), (xf + rf).view(shape)

        return wrapper
