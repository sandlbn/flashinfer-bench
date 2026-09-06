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
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

logger = logging.getLogger(__name__)

VLLM_XPU = "vllm-xpu"
"""vLLM's Intel kernel library."""

SGL_KERNEL_XPU = "sgl-kernel-xpu"
"""SGLang's Intel kernel library."""

_PROVIDER_MODULES: Dict[str, str] = {VLLM_XPU: "vllm_xpu_kernels", SGL_KERNEL_XPU: "sgl_kernel"}


_EPS_PATTERN = re.compile(r"^\s*EPS\s*=\s*([0-9][0-9._eE+-]*)\s*$", re.MULTILINE)
"""How a definition records its epsilon.

Definitions bake epsilon into the reference as a module-level ``EPS`` assignment rather
than declaring it as an input, while every upstream kernel takes it as an argument. The
wrapper has to bridge that, and the value is not uniform across the dataset -- most
definitions use 1e-6 but some use 1e-5 -- so it is read from the reference, never assumed.
"""

DEFAULT_EPS = 1e-6
"""Fallback when a definition's reference does not state one.

This is the RMSNorm default in both vLLM and SGLang
(``vllm/model_executor/layers/layernorm.py``, ``sglang/srt/layers/layernorm.py``), so a
definition that omits epsilon gets what the frameworks themselves would have used. Applied
with a warning, because a silently wrong epsilon produces a baseline that runs cleanly and
computes the wrong thing.
"""


def definition_eps(definition: Definition) -> float:
    """Epsilon for ``definition``, from its reference, else :data:`DEFAULT_EPS`."""
    match = _EPS_PATTERN.search(definition.reference or "")
    if match is None:
        logger.warning(
            "Definition '%s' does not state an EPS in its reference; using the "
            "vLLM/SGLang default %g. Check this if the baseline fails correctness.",
            definition.name,
            DEFAULT_EPS,
        )
        return DEFAULT_EPS
    try:
        return float(match.group(1))
    except ValueError:
        logger.warning(
            "Definition '%s' has an unparseable EPS (%r); using %g.",
            definition.name,
            match.group(1),
            DEFAULT_EPS,
        )
        return DEFAULT_EPS


def definition_is_neox(definition: Definition) -> bool:
    """Whether a rope definition uses NeoX-style rotation.

    The two styles interleave differently and produce different numbers, so this cannot
    be assumed. Read from the definition's own naming, which is where the dataset records
    it -- there is no structured field for it today.
    """
    haystack = f"{definition.name} {' '.join(definition.tags)}".lower()
    if "neox" in haystack:
        return True
    if "gptj" in haystack or "interleav" in haystack:
        return False
    logger.warning(
        "Definition '%s' does not say which rope style it uses; assuming NeoX. A wrong "
        "guess here computes the wrong thing without failing to run.",
        definition.name,
    )
    return True


_CONSTANT_RESOLVERS: Dict[str, Callable[[Definition], object]] = {
    "eps": definition_eps,
    "is_neox": definition_is_neox,
}
"""Named constants a wrapper template may request, and how to obtain each."""


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
        Python source template defining ``run(...)``, calling the upstream operator.
        Formatted with the constants named in ``constants`` before use.
    description : str
        What the upstream kernel does, for the solution record.
    constants : Tuple[str, ...]
        Names from :data:`_CONSTANT_RESOLVERS` that ``source`` interpolates. Resolved
        against the definition, because values like epsilon live in the definition's
        reference rather than in its declared inputs.
    fi_api : Optional[str]
        Exact operation this kernel implements, matched against the definition's
        ``fi_api:`` tag. Required wherever a signature does not identify the operation --
        every gated activation is one input and one output, so ``silu_and_mul`` and
        ``gelu_and_mul`` are indistinguishable by signature and binding the wrong one
        computes something else without failing. ``None`` means the signature is enough.
    """

    provider: str
    name: str
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    source: str
    description: str
    constants: Tuple[str, ...] = ()
    fi_api: Optional[str] = None

    def render(self, definition: Definition) -> str:
        """The wrapper source for ``definition``, with its constants substituted."""
        if not self.constants:
            return self.source
        # Only what this wrapper declares. Resolving every constant would run resolvers a
        # kernel has no use for -- epsilon warns when a definition states none, which an
        # activation definition never does.
        values = {name: _CONSTANT_RESOLVERS[name](definition) for name in self.constants}
        return self.source.format(**values)

    @property
    def solution_name(self) -> str:
        return f"{self.provider.replace('-', '_')}_{self.name}"

    def matches(self, definition: Definition) -> bool:
        """Whether this kernel implements ``definition``.

        Matching is deliberately strict -- same op_type, and exactly the same input and
        output names in the same order. A baseline that silently binds to the wrong
        definition would produce a confidently wrong comparison.
        """
        if (
            definition.op_type != self.op_type
            or tuple(definition.inputs) != self.inputs
            or tuple(definition.outputs) != self.outputs
        ):
            return False
        if self.fi_api is None:
            return True
        return f"fi_api:{self.fi_api}" in definition.tags


_RMS_NORM_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

EPS = {eps!r}


def run(hidden_states, weight):
    out = torch.empty_like(hidden_states)
    torch.ops._C.rms_norm(out, hidden_states, weight, EPS)
    return out
"""

_RMS_NORM_SGL = """import torch
import sgl_kernel

EPS = {eps!r}


def run(hidden_states, weight):
    return sgl_kernel.rmsnorm(hidden_states, weight, EPS)
"""

_FUSED_ADD_RMS_NORM_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

EPS = {eps!r}


def run(hidden_states, residual, weight):
    # The upstream op updates both tensors in place, so clone to keep the benchmark's
    # inputs reusable across trials.
    hidden = hidden_states.clone()
    res = residual.clone()
    torch.ops._C.fused_add_rms_norm(hidden, res, weight, EPS)
    # The definition declares one output: the normalized result. Upstream also returns the
    # updated residual in `res`, which this definition does not model -- returning it too
    # would not match the declared arity.
    return hidden
"""

_FUSED_ADD_RMS_NORM_SGL = """import torch
import sgl_kernel

EPS = {eps!r}


def run(hidden_states, residual, weight):
    # In-place upstream; clone so repeated trials see identical inputs.
    hidden = hidden_states.clone()
    res = residual.clone()
    sgl_kernel.fused_add_rmsnorm(hidden, res, weight, EPS)
    return hidden
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


_ROPE_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

IS_NEOX = {is_neox!r}


def run(q, k, cos_sin_cache, positions):
    # Upstream rotates in place on a flattened [num_tokens, heads * head_size] view, so
    # clone and reshape rather than mutating the benchmark's inputs.
    num_tokens = q.shape[0]
    head_size = q.shape[-1]
    q_flat = q.clone().reshape(num_tokens, -1)
    k_flat = k.clone().reshape(num_tokens, -1)
    # The definition stores cos/sin in float32 and the reference rotates in float32;
    # upstream requires the cache in the query dtype and rotates there. Measured against
    # the reference on Battlemage that costs ~0.007 relative error -- inside the default
    # 1e-2 tolerance, but it is an approximation, not an equivalence.
    torch.ops._C.rotary_embedding(
        positions, q_flat, k_flat, head_size, cos_sin_cache.to(q.dtype), IS_NEOX
    )
    return q_flat.view_as(q), k_flat.view_as(k)
"""


REGISTRY: Tuple[BaselineKernel, ...] = (
    BaselineKernel(
        provider=VLLM_XPU,
        name="rotary_embedding",
        op_type="rope",
        inputs=("q", "k", "cos_sin_cache", "positions"),
        outputs=("q_out", "k_out"),
        source=_ROPE_VLLM,
        constants=("is_neox",),
        description=(
            "vLLM XPU rotary_embedding (SYCL): applies rotary position embedding to q "
            "and k in place, with partial-rotary support driven by the cache width."
        ),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="silu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_SILU_AND_MUL_VLLM,
        description="vLLM XPU silu_and_mul (SYCL): SwiGLU gate, silu(x[..., :d]) * x[..., d:].",
        fi_api="flashinfer.activation.silu_and_mul",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="mul_and_silu",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_MUL_AND_SILU_VLLM,
        description="vLLM XPU mul_and_silu (SYCL): x[..., :d] * silu(x[..., d:]).",
        fi_api="flashinfer.activation.mul_and_silu",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_AND_MUL_VLLM,
        description="vLLM XPU gelu_and_mul (SYCL): GeGLU with exact gelu.",
        fi_api="flashinfer.activation.gelu_and_mul",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_tanh_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_TANH_AND_MUL_VLLM,
        description="vLLM XPU gelu_tanh_and_mul (SYCL): GeGLU with tanh-approximate gelu.",
        fi_api="flashinfer.activation.gelu_tanh_and_mul",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_new",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_NEW_VLLM,
        description="vLLM XPU gelu_new (SYCL): tanh-approximate GELU, elementwise.",
        fi_api="flashinfer.activation.gelu_new",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_fast",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_FAST_VLLM,
        description="vLLM XPU gelu_fast (SYCL): sigmoid-approximate GELU, elementwise.",
        fi_api="flashinfer.activation.gelu_fast",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="gelu_quick",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        source=_GELU_QUICK_VLLM,
        description="vLLM XPU gelu_quick (SYCL): x * sigmoid(1.702 x), elementwise.",
        fi_api="flashinfer.activation.gelu_quick",
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="rms_norm",
        op_type="rmsnorm",
        inputs=("hidden_states", "weight"),
        outputs=("output",),
        source=_RMS_NORM_VLLM,
        description="vLLM XPU rms_norm (SYCL).",
        constants=("eps",),
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "weight"),
        outputs=("output",),
        source=_RMS_NORM_SGL,
        description="SGLang XPU rmsnorm (SYCL).",
        constants=("eps",),
    ),
    BaselineKernel(
        provider=VLLM_XPU,
        name="fused_add_rms_norm",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output",),
        source=_FUSED_ADD_RMS_NORM_VLLM,
        description="vLLM XPU fused_add_rms_norm (SYCL), residual add fused into the norm.",
        constants=("eps",),
    ),
    BaselineKernel(
        provider=SGL_KERNEL_XPU,
        name="fused_add_rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output",),
        source=_FUSED_ADD_RMS_NORM_SGL,
        description="SGLang XPU fused_add_rmsnorm (SYCL), residual add fused into the norm.",
        constants=("eps",),
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


def explain_no_match(
    definition: Definition, providers: Optional[Sequence[str]] = None
) -> List[str]:
    """Why no baseline bound to ``definition``, as one line per near-miss.

    A signature mismatch and an uninstalled provider both produce zero baselines, and
    without this they produce the same message too. Only kernels sharing the op_type are
    reported: anything else is not a near-miss, it is a different operation.
    """
    allowed = set(providers) if providers is not None else set(available_providers())
    if find_baselines(definition, providers):
        return []  # Something bound; there is no absence to explain.
    reasons: List[str] = []
    for kernel in REGISTRY:
        if kernel.provider not in allowed or kernel.op_type != definition.op_type:
            continue
        got_in, got_out = tuple(definition.inputs), tuple(definition.outputs)
        if got_in != kernel.inputs:
            reasons.append(
                f"{kernel.provider}/{kernel.name}: op_type matches, inputs differ "
                f"(kernel wants {kernel.inputs}, definition has {got_in})"
            )
        elif got_out != kernel.outputs:
            reasons.append(
                f"{kernel.provider}/{kernel.name}: op_type and inputs match, outputs "
                f"differ (kernel wants {kernel.outputs}, definition has {got_out})"
            )
        elif kernel.fi_api is not None:
            reasons.append(
                f"{kernel.provider}/{kernel.name}: signature matches but this definition "
                f"is a different operation (kernel implements {kernel.fi_api}; the "
                f"definition's fi_api tag says otherwise)"
            )
    return reasons


def registry_op_types(providers: Optional[Sequence[str]] = None) -> List[str]:
    """op_types the registry can serve, for reporting against a dataset's actual set."""
    allowed = set(providers) if providers is not None else set(available_providers())
    return sorted({k.op_type for k in REGISTRY if k.provider in allowed})


def verify_kernel(
    definition: Definition, kernel: BaselineKernel, workload, device: str = "xpu:0"
) -> Tuple[bool, str]:
    """Actually call ``kernel`` once, and report whether it ran.

    Existing in the provider's Python namespace is not the same as being built for this
    backend. ``sgl-kernel``'s Python wrappers ship with the package regardless of which
    backend was compiled, and its ops are JIT-registered on first use, so neither
    ``dir(sgl_kernel)`` nor ``dir(torch.ops.sgl_kernel)`` describes what is available --
    ``sgl_kernel.rmsnorm`` works while appearing in neither. A call with *valid arguments*
    is the only reliable check; calling with none only exercises the Python signature and
    reports a wrapper that dispatches to nothing as working.

    Registering an unbuilt op costs a ``RUNTIME_ERROR`` on every workload of every matching
    definition, which reads like a wrapper bug rather than a missing kernel.
    """
    from flashinfer_bench.bench.utils import gen_inputs
    from flashinfer_bench.compile import BuilderRegistry

    try:
        solution = make_baseline_solution(definition, kernel)
        runnable = BuilderRegistry.get_instance().build(definition, solution)
        inputs = gen_inputs(definition, workload, device)
        runnable(*inputs)
        from flashinfer_bench.device import device_synchronize

        device_synchronize(device)
        return True, "ran"
    except AttributeError as e:
        if "_OpNamespace" in str(e):
            missing = str(e).split("has no attribute")[-1].strip()
            return False, f"not built for this backend: {missing}"
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


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
        sources=[SourceFile(path="main.py", content=kernel.render(definition))],
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
