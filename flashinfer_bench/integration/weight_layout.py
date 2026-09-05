"""Weight transforms that let a fused kernel consume a model's existing weights.

A fused GEMM epilogue often wants its operands laid out differently from how a model
stores them. The fix belongs in the weights, applied once at load time -- not in the
kernel, and not in a per-model branch of it.

Branching kernels per model layout scales badly: the cost is models x fusions, and every
branch needs its own verified reference, which is the expensive part. A weight transform
is O(1) code that serves any model of the same shape. Serving frameworks already work this
way -- vLLM's ``MergedColumnParallelLinear`` merges gate and up projections at load time,
and an interleave is the same kind of operation in the same place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import torch


def interleave_gate_up(
    gate_weight: "torch.Tensor",
    up_weight: "torch.Tensor",
    norm_weight: Optional["torch.Tensor"] = None,
    *,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """Build the interleaved B matrix a fused SwiGLU/GeGLU GEMM epilogue expects.

    Models keep the gate and up projections as separate ``[d, k]`` matrices and apply the
    activation afterwards. A fused epilogue that computes ``silu(gate) * up`` in registers
    needs both operands adjacent, so the two matrices interleave column-wise into one
    ``[k, 2d]``: even columns gate, odd columns up.

    An RMSNorm applied before the projection can be folded in at the same time. It has two
    parts and they go to different places:

    - the per-channel ``gamma`` scales *input* channels, so it cannot travel through the
      GEMM as a per-row output scale -- it multiplies into the weight here;
    - the per-token ``rsqrt(mean(x^2) + eps)`` *does* commute, since ``(r*x) @ W`` equals
      ``r * (x @ W)``, so it stays a runtime per-row scale handed to the kernel.

    Parameters
    ----------
    gate_weight : torch.Tensor
        Gate projection, ``[d, k]`` as ``nn.Linear`` stores it.
    up_weight : torch.Tensor
        Up projection, ``[d, k]``. Must match ``gate_weight``.
    norm_weight : Optional[torch.Tensor]
        Per-channel RMSNorm gamma, ``[k]``. Folded into the result when given.
    dtype : Optional[torch.dtype]
        Output dtype. Defaults to ``gate_weight``'s.

    Returns
    -------
    torch.Tensor
        ``[k, 2d]`` with gate on even columns and up on odd.

    Raises
    ------
    ValueError
        If the shapes disagree.
    """
    import torch

    if gate_weight.shape != up_weight.shape:
        raise ValueError(
            f"gate and up projections must have the same shape, got "
            f"{tuple(gate_weight.shape)} and {tuple(up_weight.shape)}"
        )
    if gate_weight.ndim != 2:
        raise ValueError(f"expected 2-D projections, got {gate_weight.ndim}-D")

    d, k = gate_weight.shape
    if norm_weight is not None and norm_weight.shape != (k,):
        raise ValueError(
            f"norm weight must be [{k}] to match the projection's input dimension, "
            f"got {tuple(norm_weight.shape)}"
        )

    target_dtype = dtype if dtype is not None else gate_weight.dtype

    # Fold in float32 regardless of storage dtype: gamma and the weights are both small,
    # and multiplying them in bf16 loses precision that never comes back.
    gate = gate_weight.float()
    up = up_weight.float()
    if norm_weight is not None:
        gamma = norm_weight.float()
        gate = gate * gamma
        up = up * gamma

    out = torch.empty((k, 2 * d), device=gate_weight.device, dtype=target_dtype)
    out[:, 0::2] = gate.T.to(target_dtype)
    out[:, 1::2] = up.T.to(target_dtype)
    return out


def rms_row_scale(x: "torch.Tensor", eps: float) -> "torch.Tensor":
    """Per-token RMSNorm factor to hand a fused epilogue as its row scale.

    This is the half of RMSNorm that commutes through the GEMM. The other half, the
    per-channel gamma, belongs in the weight -- see :func:`interleave_gate_up`.

    Parameters
    ----------
    x : torch.Tensor
        Layer input, ``[..., k]``.
    eps : float
        The model's ``rms_norm_eps``.

    Returns
    -------
    torch.Tensor
        Contiguous float32 ``[m]``, one factor per token.
    """
    import torch

    flat = x.reshape(-1, x.shape[-1]).float()
    return torch.rsqrt(flat.pow(2).mean(-1) + eps).contiguous()


def deinterleave_output(out: "torch.Tensor") -> "torch.Tensor":
    """Take the meaningful half of a pairwise fused epilogue's output.

    Xe-Fuse's pairwise ops compute the same value on both lanes of each ``(gate, up)``
    pair, so the ``[m, 2d]`` result carries each answer twice. The even lanes are the
    ``[m, d]`` activation the model wants.
    """
    return out[..., 0::2]


def qwen_style_mlp_weights(
    mlp: object, norm: object, *, dtype: Optional["torch.dtype"] = None
) -> Tuple["torch.Tensor", float]:
    """Prepare a HuggingFace gated-MLP block for a fused SwiGLU GEMM.

    Works for any module exposing ``gate_proj`` and ``up_proj`` alongside an RMSNorm with
    a ``weight`` -- the LLaMA/Qwen/Mistral family shape.

    Returns
    -------
    Tuple[torch.Tensor, float]
        The interleaved, gamma-folded ``[k, 2d]`` matrix, and the norm's epsilon.
    """
    gate = getattr(mlp, "gate_proj").weight.detach()
    up = getattr(mlp, "up_proj").weight.detach()
    gamma = getattr(norm, "weight").detach()
    eps = float(getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", 1e-6))
    return interleave_gate_up(gate, up, gamma, dtype=dtype), eps
