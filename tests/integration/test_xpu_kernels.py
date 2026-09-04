"""Tests for benchmarking upstream vLLM / SGLang Intel kernels as baselines.

The registry is checked without either provider installed: a baseline that binds to the
wrong definition would produce a confidently wrong comparison, so matching is what these
tests are mostly about.
"""

import pytest

from flashinfer_bench.data import Definition, SupportedLanguages, TensorSpec
from flashinfer_bench.integration import xpu_kernels as xk

RMSNORM_REF = (
    "import torch\n\n\n"
    "def run(x, weight, eps):\n"
    "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight\n"
)

FUSED_REF = (
    "import torch\n\n\n"
    "def run(x, residual, weight, eps):\n"
    "    r = x + residual\n"
    "    return r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * weight, r\n"
)


def _rmsnorm_definition(name: str = "rn") -> Definition:
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"tokens": {"type": "var"}, "hidden": {"type": "const", "value": 896}},
        inputs={
            "x": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "weight": TensorSpec(shape=["hidden"], dtype="float32"),
            "eps": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={"out": TensorSpec(shape=["tokens", "hidden"], dtype="float32")},
        reference=RMSNORM_REF,
    )


def _fused_definition(name: str = "far") -> Definition:
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"tokens": {"type": "var"}, "hidden": {"type": "const", "value": 896}},
        inputs={
            "x": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "residual": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "weight": TensorSpec(shape=["hidden"], dtype="float32"),
            "eps": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "out": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "residual_out": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
        },
        reference=FUSED_REF,
    )


@pytest.fixture
def all_providers(monkeypatch):
    """Pretend both upstream libraries are installed."""
    monkeypatch.setattr(xk, "is_provider_available", lambda provider: True)


class TestProviderDetection:
    def test_unknown_provider_is_not_available(self):
        assert not xk.is_provider_available("not-a-provider")

    def test_available_providers_is_a_subset_of_known_ones(self):
        assert set(xk.available_providers()) <= set(xk._PROVIDER_MODULES)

    def test_detection_survives_a_broken_import(self, monkeypatch):
        def boom(name):
            raise ImportError("broken")

        monkeypatch.setattr(xk.importlib.util, "find_spec", boom)
        assert not xk.is_provider_available(xk.VLLM_XPU)


class TestMatching:
    def test_matches_the_definition_it_implements(self, all_providers):
        found = xk.find_baselines(_rmsnorm_definition())
        assert {k.name for k in found} == {"rms_norm", "rmsnorm"}

    def test_a_different_function_with_the_same_signature_is_not_a_baseline(self, all_providers):
        """gemma_rms_norm takes the same arguments as rms_norm but scales by (1 + weight).

        Registering it against a plain RMSNorm definition produced a baseline that ran and
        computed the wrong thing. Baselines must match semantics, not just signatures.
        """
        assert not any(k.name == "gemma_rms_norm" for k in xk.REGISTRY)

    def test_fused_variant_matches_only_the_fused_definition(self, all_providers):
        found = {k.name for k in xk.find_baselines(_fused_definition())}
        assert found == {"fused_add_rms_norm", "fused_add_rmsnorm"}

    def test_plain_and_fused_definitions_do_not_cross_match(self, all_providers):
        plain = {k.name for k in xk.find_baselines(_rmsnorm_definition())}
        fused = {k.name for k in xk.find_baselines(_fused_definition())}
        assert plain.isdisjoint(fused)

    def test_wrong_op_type_matches_nothing(self, all_providers):
        definition = _rmsnorm_definition().model_copy(update={"op_type": "gemm"})
        assert xk.find_baselines(definition) == []

    def test_extra_input_prevents_a_match(self, all_providers):
        """A near-miss signature must not silently bind to the wrong kernel."""
        definition = _rmsnorm_definition()
        inputs = dict(definition.inputs)
        inputs["scale"] = TensorSpec(shape=None, dtype="float32")
        assert xk.find_baselines(definition.model_copy(update={"inputs": inputs})) == []

    def test_can_restrict_to_one_provider(self, all_providers):
        found = xk.find_baselines(_rmsnorm_definition(), providers=[xk.VLLM_XPU])
        assert {k.provider for k in found} == {xk.VLLM_XPU}

    def test_nothing_is_offered_when_no_provider_is_installed(self, monkeypatch):
        monkeypatch.setattr(xk, "is_provider_available", lambda provider: False)
        assert xk.find_baselines(_rmsnorm_definition()) == []


class TestSolutionGeneration:
    def test_builds_a_python_solution_naming_the_upstream_kernel(self, all_providers):
        definition = _rmsnorm_definition("qwen_rmsnorm")
        solutions = xk.make_baseline_solutions(definition, providers=[xk.VLLM_XPU])
        names = {s.name for s in solutions}
        assert "qwen_rmsnorm__vllm_xpu_rms_norm" in names
        for s in solutions:
            assert s.definition == "qwen_rmsnorm"
            assert s.spec.language is SupportedLanguages.PYTHON
            assert s.spec.target_hardware == ["xpu"]
            assert not s.spec.destination_passing_style

    def test_author_records_which_project_the_kernel_came_from(self, all_providers):
        solutions = xk.make_baseline_solutions(_rmsnorm_definition(), providers=[xk.SGL_KERNEL_XPU])
        assert {s.author for s in solutions} == {xk.SGL_KERNEL_XPU}

    def test_source_calls_the_upstream_operator(self, all_providers):
        (solution,) = xk.make_baseline_solutions(_fused_definition(), providers=[xk.VLLM_XPU])
        source = solution.sources[0].content
        assert "torch.ops._C.fused_add_rms_norm" in source
        assert "def run(x, residual, weight, eps):" in source

    def test_in_place_kernels_clone_so_trials_stay_comparable(self, all_providers):
        """The upstream fused op mutates its inputs; trials must see identical data."""
        for provider in (xk.VLLM_XPU, xk.SGL_KERNEL_XPU):
            (solution,) = xk.make_baseline_solutions(_fused_definition(), providers=[provider])
            assert ".clone()" in solution.sources[0].content

    def test_mismatched_kernel_is_refused(self):
        fused = next(k for k in xk.REGISTRY if k.name == "fused_add_rms_norm")
        with pytest.raises(ValueError, match="does not implement"):
            xk.make_baseline_solution(_rmsnorm_definition(), fused)

    def test_generated_solutions_have_distinct_names(self, all_providers):
        solutions = xk.make_baseline_solutions(_rmsnorm_definition())
        assert len({s.name for s in solutions}) == len(solutions)


class TestRegistryIntegrity:
    def test_every_entry_defines_a_run_function(self):
        for kernel in xk.REGISTRY:
            assert "def run(" in kernel.source

    def test_every_entry_names_a_known_provider(self):
        for kernel in xk.REGISTRY:
            assert kernel.provider in xk._PROVIDER_MODULES

    def test_run_signature_matches_declared_inputs(self):
        """The wrapper is called positionally in definition order, so these must agree."""
        for kernel in xk.REGISTRY:
            expected = "def run(" + ", ".join(kernel.inputs) + "):"
            assert expected in kernel.source, f"{kernel.solution_name}: {expected}"
