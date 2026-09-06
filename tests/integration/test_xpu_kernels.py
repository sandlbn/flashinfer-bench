"""Tests for benchmarking upstream vLLM / SGLang Intel kernels as baselines.

The registry is checked without either provider installed: a baseline that binds to the
wrong definition would produce a confidently wrong comparison, so matching is what these
tests are mostly about.

The fixtures below deliberately mirror the signature the shipped dataset actually uses --
``(hidden_states, weight) -> (output)``, with epsilon carried in the reference rather than
declared as an input. An earlier version of these tests invented its own convention, which
matched the registry and nothing else, so every test passed while ``add-baselines`` bound
to zero of 190 real definitions. :class:`TestMatchesRealDataset` is the guard against that
happening again.
"""

import pytest

from flashinfer_bench.data import Definition, SupportedLanguages, TensorSpec
from flashinfer_bench.integration import xpu_kernels as xk

RMSNORM_REF = (
    "import torch\n\n"
    "EPS = 1e-6\n\n\n"
    "def run(hidden_states, weight):\n"
    "    x = hidden_states.to(torch.float32)\n"
    "    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight).to(\n"
    "        hidden_states.dtype\n"
    "    )\n"
)

FUSED_REF = (
    "import torch\n\n"
    "EPS = 1e-6\n\n\n"
    "def run(hidden_states, residual, weight):\n"
    "    r = hidden_states.to(torch.float32) + residual.to(torch.float32)\n"
    "    return (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + EPS) * weight).to(\n"
    "        hidden_states.dtype\n"
    "    )\n"
)


def _rmsnorm_definition(name: str = "rn") -> Definition:
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"tokens": {"type": "var"}, "hidden": {"type": "const", "value": 896}},
        inputs={
            "hidden_states": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "weight": TensorSpec(shape=["hidden"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["tokens", "hidden"], dtype="float32")},
        reference=RMSNORM_REF,
    )


def _fused_definition(name: str = "far") -> Definition:
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"tokens": {"type": "var"}, "hidden": {"type": "const", "value": 896}},
        inputs={
            "hidden_states": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "residual": TensorSpec(shape=["tokens", "hidden"], dtype="float32"),
            "weight": TensorSpec(shape=["hidden"], dtype="float32"),
        },
        # One output: the normalized result. Upstream also updates the residual in place,
        # which this definition does not model -- matching that shape is the wrapper's job.
        outputs={"output": TensorSpec(shape=["tokens", "hidden"], dtype="float32")},
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
        assert "def run(hidden_states, residual, weight):" in source

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


class TestEpsilonResolution:
    """Epsilon lives in the reference, not the inputs, and is not uniform across the set."""

    def test_reads_epsilon_from_the_reference(self):
        assert xk.definition_eps(_rmsnorm_definition()) == pytest.approx(1e-6)

    def test_a_definition_may_use_a_different_epsilon(self):
        """Two shipped RMSNorm definitions use 1e-5; assuming 1e-6 would corrupt them."""
        definition = _rmsnorm_definition().model_copy(
            update={"reference": RMSNORM_REF.replace("EPS = 1e-6", "EPS = 1e-5")}
        )
        assert xk.definition_eps(definition) == pytest.approx(1e-5)

    def test_falls_back_to_the_framework_default(self):
        """vLLM and SGLang both default RMSNorm epsilon to 1e-6."""
        definition = _rmsnorm_definition().model_copy(
            update={"reference": "import torch\n\n\ndef run(hidden_states, weight):\n    ...\n"}
        )
        assert xk.definition_eps(definition) == pytest.approx(xk.DEFAULT_EPS)

    def test_epsilon_is_baked_into_the_generated_wrapper(self, all_providers):
        definition = _rmsnorm_definition().model_copy(
            update={"reference": RMSNORM_REF.replace("EPS = 1e-6", "EPS = 1e-5")}
        )
        for solution in xk.make_baseline_solutions(definition):
            assert "EPS = 1e-05" in solution.sources[0].content


class TestNearMissDiagnostics:
    """A signature mismatch and an absent provider must not look the same."""

    def test_names_the_differing_inputs(self, all_providers):
        definition = _rmsnorm_definition().model_copy(
            update={"inputs": {"x": TensorSpec(shape=["tokens", "hidden"], dtype="float32")}}
        )
        reasons = xk.explain_no_match(definition)
        assert reasons and any("inputs differ" in r for r in reasons)

    def test_a_different_op_type_is_not_a_near_miss(self, all_providers):
        definition = _rmsnorm_definition().model_copy(update={"op_type": "gemm"})
        assert xk.explain_no_match(definition) == []

    def test_a_match_has_nothing_to_explain(self, all_providers):
        assert xk.explain_no_match(_rmsnorm_definition()) == []


class TestMatchesRealDataset:
    """The guard that the fixture-based tests above cannot provide.

    Everything else here checks the registry against definitions this file wrote, so the
    registry and the tests can agree with each other and with nothing else -- which is
    exactly what happened: eleven entries matched zero of 190 shipped definitions while
    the suite stayed green. These tests read the dataset instead, and skip when it is not
    checked out rather than asserting on a fixture that proves nothing.
    """

    @staticmethod
    def _dataset_definitions(op_type: str):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "tmp" / "flashinfer-trace"
        paths = sorted((root / "definitions" / op_type).glob("*.json"))
        if not paths:
            pytest.skip("flashinfer-trace dataset not checked out under tmp/")
        return [Definition.model_validate_json(p.read_text()) for p in paths]

    def test_some_shipped_definition_gets_a_baseline(self, all_providers):
        definitions = self._dataset_definitions("rmsnorm")
        matched = [d.name for d in definitions if xk.find_baselines(d)]
        assert matched, (
            "No shipped RMSNorm definition matches any registry entry. The registry's "
            "signatures have drifted from the dataset's; compare kernel.inputs against "
            f"{tuple(definitions[0].inputs)} -> {tuple(definitions[0].outputs)}."
        )

    def test_every_registered_op_type_exists_in_the_dataset(self, all_providers):
        """An entry whose op_type no definition uses can never match anything.

        This was an expected failure until the `activation` op_type was written: seven
        vLLM entries declared op_types (`activation`, `activation_gelu`, `gelu_quick`,
        ...) that no definition used and no schema described. They are now one op_type
        whose variants are told apart by their `fi_api` tag, because every gated
        activation shares a signature.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "tmp" / "flashinfer-trace"
        if not (root / "definitions").is_dir():
            pytest.skip("flashinfer-trace dataset not checked out under tmp/")
        present = {p.name for p in (root / "definitions").iterdir() if p.is_dir()}
        registered = {k.op_type for k in xk.REGISTRY}
        orphans = sorted(registered - present)
        assert not orphans, (
            f"Registry entries declare op_types absent from the dataset: {orphans}. "
            "They cannot match any definition, and there is no docs/op-types/ schema to "
            "author one against."
        )


ROPE_REF = "import torch\n\n\ndef run(q, k, cos_sin_cache, positions):\n    ...\n"


def _rope_definition(name: str = "rope_neox_style_d128") -> Definition:
    return Definition(
        name=name,
        op_type="rope",
        axes={
            "num_tokens": {"type": "var"},
            "num_qo_heads": {"type": "var"},
            "num_kv_heads": {"type": "var"},
            "head_size": {"type": "const", "value": 128},
            "rotary_dim": {"type": "const", "value": 64},
            "max_seq_len": {"type": "var"},
        },
        inputs={
            "q": TensorSpec(shape=["num_tokens", "num_qo_heads", "head_size"], dtype="bfloat16"),
            "k": TensorSpec(shape=["num_tokens", "num_kv_heads", "head_size"], dtype="bfloat16"),
            "cos_sin_cache": TensorSpec(shape=["max_seq_len", "rotary_dim"], dtype="float32"),
            "positions": TensorSpec(shape=["num_tokens"], dtype="int64"),
        },
        outputs={
            "q_out": TensorSpec(
                shape=["num_tokens", "num_qo_heads", "head_size"], dtype="bfloat16"
            ),
            "k_out": TensorSpec(
                shape=["num_tokens", "num_kv_heads", "head_size"], dtype="bfloat16"
            ),
        },
        reference=ROPE_REF,
    )


class TestRopeStyle:
    """The two rope styles produce different numbers, so the style is never assumed."""

    def test_neox_is_read_from_the_definition(self):
        assert xk.definition_is_neox(_rope_definition("rope_neox_style_d128")) is True

    def test_gptj_style_is_not_neox(self):
        assert xk.definition_is_neox(_rope_definition("rope_gptj_interleaved_d128")) is False

    def test_style_is_baked_into_the_wrapper(self, all_providers):
        (solution,) = xk.make_baseline_solutions(_rope_definition(), providers=[xk.VLLM_XPU])
        assert "IS_NEOX = True" in solution.sources[0].content

    def test_rope_binds_to_the_rope_definition(self, all_providers):
        found = {k.name for k in xk.find_baselines(_rope_definition())}
        assert "rotary_embedding" in found

    def test_the_wrapper_does_not_mutate_its_inputs(self, all_providers):
        """Upstream rotates in place; repeated benchmark trials must see identical data."""
        (solution,) = xk.make_baseline_solutions(_rope_definition(), providers=[xk.VLLM_XPU])
        source = solution.sources[0].content
        assert "q.clone()" in source and "k.clone()" in source
