"""Definition names an adapter should try for a given tensor dtype.

A definition is named for its shape -- ``rmsnorm_h1536`` -- but shape does not identify
the operation on its own. Two models at the same hidden size can run in different
precisions, and ``apply()`` refuses a solution whose declared dtype differs from the
tensors it is handed (an fp16 kernel returning fp16 activations into a bf16 model kills the
next matmul). So the dataset distinguishes them with a suffix, following the precedent set
by ``rmsnorm_h4096_float32``.

The adapter therefore cannot build one name from the width alone: it has to ask for the
dtype-qualified definition first and fall back to the bare one. Without this, a correctly
generated ``*_float16`` definition is never looked up, and the counters report
``no-solution`` -- which reads as "nothing has been extracted for this shape" rather than
"the extracted one is under a name nobody asks for".
"""

from __future__ import annotations

from typing import Tuple

import torch

_SUFFIX = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def candidates(base: str, dtype: torch.dtype) -> Tuple[str, ...]:
    """``(base, base_dtype)`` -- the common name first.

    Order matters for cost, not correctness. Almost every definition is unsuffixed, so
    asking for the suffixed name first is a near-certain miss, and a miss is not cheap: the
    caller pays a full resolve, key build and dtype check before falling through. Putting
    the bare name first makes the common case one lookup instead of two.

    Trying the bare name first is still correct when a suffixed definition is the right one:
    ``apply()`` refuses a solution whose declared dtype differs from the tensors handed to
    it, so a bare bfloat16 definition presented with float16 activations falls through, and
    the suffixed name is tried next.
    """
    suffix = _SUFFIX.get(dtype)
    return (base, f"{base}_{suffix}") if suffix else (base,)
