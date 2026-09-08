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
    """``(base_dtype, base)`` -- most specific first.

    The bare name stays in the list because most definitions carry no suffix; it is only
    added where two precisions had to coexist at one width.
    """
    suffix = _SUFFIX.get(dtype)
    return (f"{base}_{suffix}", base) if suffix else (base,)
