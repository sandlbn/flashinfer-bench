"""The profiler reports what it measured and who owns it; it does not draw conclusions."""

import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "profile_intel", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "profile_intel.py"
)
pi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pi)


class TestRoutes:
    def test_a_route_is_pattern_family_and_owner_only(self):
        """No fourth field: the advice strings were verdicts printed as measurements."""
        for row in pi.ROUTES:
            assert len(row) == 3, row
            pattern, family, owner = row
            assert isinstance(pattern, str) and isinstance(family, str)
            assert owner.startswith(("/", "(")), owner  # a skill name or an explicit non-route

    def test_classify_returns_family_and_owner(self):
        assert pi.classify("gemm_kernel<bf16>") == ("gemm", "/optimize-onednn")
        assert pi.classify("ReduceKernel<1, ReduceOp<float>>")[0] == "norm"
        assert pi.classify("something_nobody_named") == ("other", "(investigate)")

    def test_quantized_gemm_is_matched_before_dense_gemm(self):
        assert pi.classify("awq_gemm_kernel")[0] == "quantized gemm"
