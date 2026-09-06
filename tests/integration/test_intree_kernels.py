"""Tests for expanding in-tree SYCL kernel templates into Solutions.

The point of the template registry is that one kernel source serves every definition it
implements, so what is worth protecting is the expansion: that a template matches the
signatures it should, refuses the ones it should not, and specializes correctly per
definition. Nothing here compiles anything -- the end-to-end build is covered by
tests/compile/test_sycl_builder.py.
"""

import pytest

from flashinfer_bench.data import Definition, SupportedLanguages, TensorSpec
from flashinfer_bench.integration import intree_kernels as sk

RMSNORM_REF = (
    "import torch\n\n"
    "EPS = 1e-6\n\n\n"
    "def run(hidden_states, weight):\n"
    "    x = hidden_states.to(torch.float32)\n"
    "    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(\n"
    "        hidden_states.dtype\n"
    "    )\n"
)


def _rmsnorm(name: str = "rn", hidden: int = 2048, dtype: str = "bfloat16") -> Definition:
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"tokens": {"type": "var"}, "hidden": {"type": "const", "value": hidden}},
        inputs={
            "hidden_states": TensorSpec(shape=["tokens", "hidden"], dtype=dtype),
            "weight": TensorSpec(shape=["hidden"], dtype=dtype),
        },
        outputs={"output": TensorSpec(shape=["tokens", "hidden"], dtype=dtype)},
        reference=RMSNORM_REF,
    )


def _fused(name: str = "far", dtype: str = "bfloat16") -> Definition:
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"tokens": {"type": "var"}, "hidden": {"type": "const", "value": 2048}},
        inputs={
            "hidden_states": TensorSpec(shape=["tokens", "hidden"], dtype=dtype),
            "residual": TensorSpec(shape=["tokens", "hidden"], dtype=dtype),
            "weight": TensorSpec(shape=["hidden"], dtype=dtype),
        },
        outputs={"output": TensorSpec(shape=["tokens", "hidden"], dtype=dtype)},
        reference=RMSNORM_REF,
    )


class TestMatching:
    def test_plain_and_fused_do_not_cross_match(self):
        plain = {k.name for k in sk.find_kernels(_rmsnorm())}
        fused = {k.name for k in sk.find_kernels(_fused())}
        assert plain == {"rmsnorm"}
        assert fused == {"fused_add_rmsnorm"}

    def test_a_different_op_type_matches_nothing(self):
        assert sk.find_kernels(_rmsnorm().model_copy(update={"op_type": "gemm"})) == []

    def test_an_unsupported_dtype_matches_nothing(self):
        assert sk.find_kernels(_rmsnorm(dtype="int8")) == []

    def test_wrong_signature_is_reported_as_a_near_miss(self):
        definition = _rmsnorm().model_copy(
            update={"inputs": {"x": TensorSpec(shape=["tokens", "hidden"], dtype="bfloat16")}}
        )
        reasons = sk.explain_no_match(definition)
        assert reasons and "inputs differ" in reasons[0]

    def test_a_match_has_nothing_to_explain(self):
        assert sk.explain_no_match(_rmsnorm()) == []


class TestSpecialization:
    """One template, many definitions -- the substitution is what makes that safe."""

    def test_epsilon_comes_from_the_definition(self):
        """Every language's template gets the definition's epsilon, not a default."""
        definition = _rmsnorm().model_copy(
            update={"reference": RMSNORM_REF.replace("EPS = 1e-6", "EPS = 1e-5")}
        )
        for solution in sk.make_intree_solutions(definition):
            assert "1e-05" in solution.sources[0].content

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("bfloat16", "sycl::ext::oneapi::bfloat16"),
            ("float16", "sycl::half"),
            ("float32", "float"),
        ],
    )
    def test_scalar_type_follows_the_definition(self, dtype, expected):
        sycl = [
            s
            for s in sk.make_intree_solutions(_rmsnorm(dtype=dtype))
            if s.spec.language.value == "sycl"
        ]
        assert sycl and f"using scalar_t = {expected};" in sycl[0].sources[0].content

    @pytest.mark.parametrize("dtype,width", [("bfloat16", 8), ("float16", 8), ("float32", 4)])
    def test_vector_width_keeps_the_access_at_16_bytes(self, dtype, width):
        """The whole point of vectorizing: one SYCL access should move 16 bytes."""
        sycl = [
            s
            for s in sk.make_intree_solutions(_rmsnorm(dtype=dtype))
            if s.spec.language.value == "sycl"
        ]
        assert sycl and f"kVecSize = {width}" in sycl[0].sources[0].content

    def test_no_placeholder_survives_substitution(self):
        """An unsubstituted $name would be a compile error at benchmark time."""
        for definition in (_rmsnorm(), _fused()):
            for solution in sk.make_intree_solutions(definition):
                assert "$" not in solution.sources[0].content

    def test_the_kernel_keeps_its_scalar_fallback(self):
        """A hidden size that is not a whole number of vectors must still work."""
        sycl = [s for s in sk.make_intree_solutions(_rmsnorm()) if s.spec.language.value == "sycl"]
        assert sycl and "hidden % kVecSize == 0" in sycl[0].sources[0].content


class TestGeneratedSolutions:
    def test_solutions_target_xpu_and_name_their_definition(self):
        for solution in sk.make_intree_solutions(_rmsnorm("qwen_rmsnorm")):
            assert solution.spec.target_hardware == ["xpu"]
            assert solution.definition == "qwen_rmsnorm"
            assert solution.spec.language in (SupportedLanguages.SYCL, SupportedLanguages.TRITON)

    def test_entry_point_matches_the_exported_symbol(self):
        for definition in (_rmsnorm(), _fused()):
            for solution in sk.make_intree_solutions(definition):
                path, symbol = solution.spec.entry_point.split("::")
                src = solution.sources[0].content
                assert solution.sources[0].path == path
                if solution.spec.language == SupportedLanguages.SYCL:
                    assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({symbol}," in src
                else:
                    assert f"def {symbol}(" in src

    def test_names_are_unique_per_definition_and_language(self):
        pairs = [
            (s.name, s.spec.language.value)
            for d in (_rmsnorm("a"), _rmsnorm("b"), _fused("c"))
            for s in sk.make_intree_solutions(d)
        ]
        assert len(set(pairs)) == len(pairs) == 6

    def test_binding_the_wrong_kernel_is_refused(self):
        fused = next(k for k in sk.REGISTRY if k.name == "fused_add_rmsnorm")
        with pytest.raises(ValueError):
            sk.make_intree_solution(_rmsnorm(), fused)


class TestCoversRealDataset:
    """One template per signature should cover the shipped definitions, not a fixture."""

    def test_every_shipped_rmsnorm_definition_gets_a_kernel(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "tmp" / "flashinfer-trace"
        paths = sorted((root / "definitions" / "rmsnorm").glob("*.json"))
        if not paths:
            pytest.skip("flashinfer-trace dataset not checked out under tmp/")
        uncovered = [
            json.loads(p.read_text())["name"]
            for p in paths
            if not sk.find_kernels(Definition.model_validate_json(p.read_text()))
        ]
        assert not uncovered, f"No in-tree SYCL kernel for: {uncovered}"


class TestBothLanguages:
    """SYCL and Triton are both first-class, and the benchmark picks between them."""

    def test_every_signature_has_both_languages(self):
        from collections import defaultdict

        langs = defaultdict(set)
        for k in sk.REGISTRY:
            langs[(k.op_type, k.inputs, k.outputs)].add(k.language.value)
        for sig, present in langs.items():
            assert present == {"sycl", "triton"}, f"{sig} only has {present}"

    def test_a_definition_gets_one_solution_per_language(self):
        for definition in (_rmsnorm(), _fused()):
            langs = [s.spec.language.value for s in sk.make_intree_solutions(definition)]
            assert sorted(langs) == ["sycl", "triton"]

    def test_solution_names_do_not_collide_across_languages(self):
        names = {s.name for s in sk.make_intree_solutions(_rmsnorm("shared"))}
        assert len(names) == 2

    def test_triton_solutions_return_their_output(self):
        """Triton allocates and returns; SYCL takes a pre-allocated destination."""
        for s in sk.make_intree_solutions(_rmsnorm()):
            if s.spec.language.value == "triton":
                assert s.spec.destination_passing_style is False
            else:
                assert s.spec.destination_passing_style is True


class TestDeviceAgnostic:
    """The reason the dataset's Triton solutions do not run on Intel.

    Every Triton solution shipped in the dataset fails on XPU, and none of them fail in
    the kernel: they fail in a hand-written guard like

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This kernel requires a CUDA GPU.")

    The kernel underneath would have compiled and run. Triton targets whichever backend
    the tensors are on, so a template must never name a vendor.
    """

    FORBIDDEN = ("torch.cuda", ".cuda()", "device='cuda'", 'device="cuda"', "cuda_")

    def test_no_template_names_a_vendor(self):
        for kernel in sk.REGISTRY:
            for definition in (_rmsnorm(), _fused()):
                if not kernel.matches(definition):
                    continue
                src = kernel.render(definition)
                for token in self.FORBIDDEN:
                    assert token not in src, (
                        f"{kernel.language.value}/{kernel.name} contains {token!r}; "
                        f"a template must derive the device from its inputs"
                    )

    def test_no_generated_solution_guards_on_availability(self):
        for definition in (_rmsnorm(), _fused()):
            for s in sk.make_intree_solutions(definition):
                src = s.sources[0].content
                assert "is_available" not in src, f"{s.name} guards on device availability"
