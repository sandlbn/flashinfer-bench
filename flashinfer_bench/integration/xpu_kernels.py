"""Upstream Intel GPU kernels as benchmark baselines.

Beating PyTorch eager is the easy bar. The bar that matters for Intel is beating the SYCL
kernels that vLLM and SGLang already ship, since those are what a real deployment runs.
This module turns those upstream kernels into ordinary Solutions so they can be measured
against the same Definitions, with the same correctness gates, as anything else.

A baseline is expressed as a plain Python solution that calls the upstream operator. That
is deliberate: it means the existing builder, evaluator, validator and trace format all
work unchanged, and an upstream kernel appears in results beside a hand-written one rather
than in a separate report.

Two providers are supported:

- ``vllm-xpu`` -- github.com/vllm-project/vllm-xpu-kernels, registered under
  ``torch.ops._C``.
- ``sgl-kernel-xpu`` -- github.com/sgl-project/sgl-kernel-xpu, exposed as the
  ``sgl_kernel`` Python package.

Neither is a dependency. When a provider is not installed, its baselines are simply not
offered.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

logger = logging.getLogger(__name__)

VLLM_XPU = "vllm-xpu"
"""vLLM's Intel kernel library."""

SGL_KERNEL_XPU = "sgl-kernel-xpu"
"""SGLang's Intel kernel library."""

_PROVIDER_MODULES: Dict[str, str] = {VLLM_XPU: "vllm_xpu_kernels", SGL_KERNEL_XPU: "sgl_kernel"}


@dataclass(frozen=True)
class BaselineKernel:
    """An upstream kernel that can serve a Definition.

    Parameters
    ----------
    provider : str
        Which upstream library ships it.
    name : str
        The upstream operator name, used in the generated solution's name so results say
        exactly what was measured.
    op_type : str
        The Definition ``op_type`` this kernel implements.
    inputs : Tuple[str, ...]
        Definition input names, in order, that the wrapper expects.
    outputs : Tuple[str, ...]
        Definition output names, in order.
    source : str
        Python source defining ``run(...)``, calling the upstream operator.
    description : str
        What the upstream kernel does, for the solution record.
    """

    provider: str
    name: str
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    source: str
    description: str

    @property
    def solution_name(self) -> str:
        return f"{self.provider.replace('-', '_')}_{self.name}"

    def matches(self, definition: Definition) -> bool:
        """Whether this kernel implements ``definition``.

        Matching is deliberately strict -- same op_type, and exactly the same input and
        output names in the same order. A baseline that silently binds to the wrong
        definition would produce a confidently wrong comparison.
        """
        return (
            definition.op_type == self.op_type
            and tuple(definition.inputs) == self.inputs
            and tuple(definition.outputs) == self.outputs
        )


_RMS_NORM_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x, weight, eps):
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, float(eps))
    return out
"""

_RMS_NORM_SGL = """import torch
import sgl_kernel


def run(x, weight, eps):
    return sgl_kernel.rmsnorm(x, weight, float(eps))
"""

_FUSED_ADD_RMS_NORM_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x, residual, weight, eps):
    # The upstream op updates both tensors in place, so clone to keep the benchmark's
    # inputs reusable across trials.
    hidden = x.clone()
    res = residual.clone()
    torch.ops._C.fused_add_rms_norm(hidden, res, weight, float(eps))
    return hidden, res
"""

_FUSED_ADD_RMS_NORM_SGL = """import torch
import sgl_kernel


def run(x, residual, weight, eps):
    # In-place upstream; clone so repeated trials see identical inputs.
    hidden = x.clone()
    res = residual.clone()
    sgl_kernel.fused_add_rmsnorm(hidden, res, weight, float(eps))
    return hidden, res
"""

_SILU_AND_MUL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out
"""

_MUL_AND_SILU_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.mul_and_silu(out, x)
    return out
"""

_GELU_AND_MUL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_and_mul(out, x)
    return out
"""

_GELU_TANH_AND_MUL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_tanh_and_mul(out, x)
    return out
"""

_GELU_NEW_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    out = torch.empty_like(x)
    torch.ops._C.gelu_new(out, x)
    return out
"""

_GELU_FAST_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    out = torch.empty_like(x)
    torch.ops._C.gelu_fast(out, x)
    return out
"""

_GELU_QUICK_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)


def run(x):
    out = torch.empty_like(x)
    torch.ops._C.gelu_quick(out, x)
    return out
"""


REGISTRY: Tuple[BaselineKernel, ...] = (
    BaselineKernel(
        provider=VLLM_XPU,
        name="silu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_SILU_AND_MUL_VLLM,
        description="vLLM XPU silu_and_mul (SYCL): SwiGLU gate, silu(x[..., :d]) * x[..., d:].",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="mul_and_silu",
        op_type="activation_mul_silu",
        inputs=("x",),
        outputs=("out",),
        source=_MUL_AND_SILU_VLLM,
        description="vLLM XPU mul_and_silu (SYCL): x[..., :d] * silu(x[..., d:]).",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_and_mul",
        op_type="activation_gelu",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_AND_MUL_VLLM,
        description="vLLM XPU gelu_and_mul (SYCL): GeGLU with exact gelu.",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_tanh_and_mul",
        op_type="activation_gelu_tanh",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_TANH_AND_MUL_VLLM,
        description="vLLM XPU gelu_tanh_and_mul (SYCL): GeGLU with tanh-approximate gelu.",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_new",
        op_type="gelu_new",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_NEW_VLLM,
        description="vLLM XPU gelu_new (SYCL): tanh-approximate GELU, elementwise.",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_fast",
        op_type="gelu_fast",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_FAST_VLLM,
        description="vLLM XPU gelu_fast (SYCL): sigmoid-approximate GELU, elementwise.",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_quick",
        op_type="gelu_quick",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_QUICK_VLLM,
        description="vLLM XPU gelu_quick (SYCL): x * sigmoid(1.702 x), elementwise.",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="rms_norm",
        op_type="rmsnorm",
        inputs=("x", "weight", "eps"),
        outputs=("out",),
        source=_RMS_NORM_VLLM,
        description="vLLM XPU rms_norm (SYCL).",
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="rmsnorm",
        op_type="rmsnorm",
        inputs=("x", "weight", "eps"),
        outputs=("out",),
        source=_RMS_NORM_SGL,
        description="SGLang XPU rmsnorm (SYCL).",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="fused_add_rms_norm",
        op_type="rmsnorm",
        inputs=("x", "residual", "weight", "eps"),
        outputs=("out", "residual_out"),
        source=_FUSED_ADD_RMS_NORM_VLLM,
        description="vLLM XPU fused_add_rms_norm (SYCL), residual add fused into the norm.",
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="fused_add_rmsnorm",
        op_type="rmsnorm",
        inputs=("x", "residual", "weight", "eps"),
        outputs=("out", "residual_out"),
        source=_FUSED_ADD_RMS_NORM_SGL,
        description="SGLang XPU fused_add_rmsnorm (SYCL), residual add fused into the norm.",
    ),
)
"""Upstream kernels that can be benchmarked as baselines.

Deliberately small: each entry is a kernel whose calling convention *and semantics* have
been checked against the upstream binding.

Signature matching alone is not enough. ``gemma_rms_norm`` takes exactly the same
arguments as ``rms_norm`` but scales by ``(1 + weight)``, so registering it against a
plain RMSNorm definition produced a baseline that ran fine and computed the wrong thing --
caught by the correctness gate, but only because there was one. A kernel belongs here only
when it computes the same function as the definition's reference, not merely when the
arguments line up. Gemma's variant needs its own definition before it can be a baseline.
"""


def is_provider_available(provider: str) -> bool:
    """Whether an upstream kernel library is importable here."""
    module = _PROVIDER_MODULES.get(provider)
    if module is None:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_providers() -> List[str]:
    """Upstream kernel libraries installed in this environment."""
    return [p for p in sorted(_PROVIDER_MODULES) if is_provider_available(p)]


def find_baselines(
    definition: Definition, providers: Optional[Sequence[str]] = None
) -> List[BaselineKernel]:
    """Upstream kernels that implement ``definition``.

    Parameters
    ----------
    definition : Definition
        The definition to find baselines for.
    providers : Optional[Sequence[str]]
        Restrict to these providers. Defaults to every provider installed here.

    Returns
    -------
    List[BaselineKernel]
        Matching kernels, empty when none apply or none are installed.
    """
    allowed = set(providers) if providers is not None else set(available_providers())
    return [k for k in REGISTRY if k.provider in allowed and k.matches(definition)]


def make_baseline_solution(definition: Definition, kernel: BaselineKernel) -> Solution:
    """Wrap an upstream kernel as a Solution for ``definition``.

    Raises
    ------
    ValueError
        If the kernel does not implement this definition.
    """
    if not kernel.matches(definition):
        raise ValueError(
            f"Baseline '{kernel.name}' does not implement definition '{definition.name}' "
            f"(expected inputs {kernel.inputs} and outputs {kernel.outputs})"
        )

    return Solution(
        name=f"{definition.name}__{kernel.solution_name}",
        definition=definition.name,
        author=kernel.provider,
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["xpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=kernel.source)],
        description=(
            f"{kernel.description} Benchmark baseline: a solution must beat this to be "
            f"an improvement on what Intel deployments already run."
        ),
    )


def make_baseline_solutions(
    definition: Definition, providers: Optional[Sequence[str]] = None
) -> List[Solution]:
    """Every available upstream baseline for ``definition``, as Solutions."""
    return [make_baseline_solution(definition, k) for k in find_baselines(definition, providers)]
