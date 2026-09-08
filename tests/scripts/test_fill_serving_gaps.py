"""Tests for turning serving `no-solution` reports into definitions.

The risk here is not a crash but a plausible-looking wrong definition: a width scaled
inconsistently across axes, or an epsilon inherited from an unrelated model. Either produces
a file that validates and benchmarks cleanly while describing a different function.
"""

import importlib.util
import json
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fill_serving_gaps",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fill_serving_gaps.py",
)
fsg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fsg)


SILU = {
    "name": "silu_and_mul_d4864",
    "op_type": "activation",
    "description": "SwiGLU over a gated projection of width 9728.",
    "tags": ["status:verified", "model:some-model"],
    "axes": {
        "batch_size": {"type": "var"},
        "d": {"type": "const", "value": 4864},
        "two_d": {"type": "const", "value": 9728},
    },
    "inputs": {"x": {"shape": ["batch_size", "two_d"]}},
    "outputs": {"output": {"shape": ["batch_size", "d"]}},
    "reference": "def run(x):\n    _, two_d = x.shape\n    assert two_d == 9728\n",
}

NORM = {
    "name": "fused_add_rmsnorm_h7168",
    "op_type": "rmsnorm",
    "description": "Fused residual add and RMSNorm at 7168.",
    "tags": ["status:verified", "model:deepseek-v3"],
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 7168}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"]}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"]}},
    "reference": (
        "def run(hidden_states):\n"
        "    _, hidden_size = hidden_states.shape\n"
        "    assert hidden_size == 7168\n"
        "\n"
        "    EPS = 1e-6\n"
    ),
}


class TestGapParsing:
    def test_reads_family_and_width_from_detail_strings(self):
        detail = (
            "{'fused_add_rmsnorm no-solution h2560': 48096, "
            "'rmsnorm applied h2560': 668, 'silu_and_mul no-solution d9728': 24048}"
        )
        assert fsg.gaps_from_details([detail]) == [
            ("fused_add_rmsnorm", "h", 2560),
            ("silu_and_mul", "d", 9728),
        ]

    def test_applied_families_are_not_gaps(self):
        assert fsg.gaps_from_details(["{'rmsnorm applied h2560': 59292}"]) == []

    def test_repeated_reports_yield_one_gap(self):
        d = "{'silu_and_mul no-solution d8192': 1}"
        assert fsg.gaps_from_details([d, d]) == [("silu_and_mul", "d", 8192)]


class TestReparametrize:
    def test_every_constant_axis_scales_together(self):
        """`two_d` is `2 * d` by construction; scaling only the named axis breaks that.

        A definition whose `two_d` no longer equals `2 * d` declares an input shape the
        reference cannot consume, and the failure appears far from its cause.
        """
        out = fsg._reparametrize(SILU, "silu_and_mul_d8192", 8192 / 4864)
        assert out["axes"]["d"]["value"] == 8192
        assert out["axes"]["two_d"]["value"] == 16384
        assert out["axes"]["two_d"]["value"] == 2 * out["axes"]["d"]["value"]

    def test_reference_assertions_follow_the_new_constants(self):
        out = fsg._reparametrize(SILU, "silu_and_mul_d8192", 8192 / 4864)
        assert "assert two_d == 16384" in out["reference"]
        assert "9728" not in out["reference"]

    def test_description_stops_naming_the_siblings_width(self):
        out = fsg._reparametrize(SILU, "silu_and_mul_d8192", 8192 / 4864)
        assert "16384" in out["description"]
        assert "9728" not in out["description"]

    def test_provenance_is_not_inherited(self):
        """The sibling was observed on another model and validated for another width."""
        out = fsg._reparametrize(SILU, "silu_and_mul_d8192", 8192 / 4864)
        assert not [t for t in out["tags"] if t.startswith("model:")]
        assert "status:unverified" in out["tags"]
        assert "status:verified" not in out["tags"]

    def test_a_reference_that_cannot_be_reparametrized_raises(self):
        odd = dict(SILU, reference="def run(x):\n    assert two_d == 12345\n")
        with pytest.raises(ValueError, match="cannot be reparametrized"):
            fsg._reparametrize(odd, "silu_and_mul_d8192", 8192 / 4864)


class TestEpsilon:
    def test_epsilon_is_read_and_replaced(self):
        assert fsg._reference_eps(NORM["reference"]) == "1e-6"
        assert "EPS = 1e-5" in fsg._set_eps(NORM["reference"], "1e-5")

    def test_families_without_an_epsilon_report_none(self):
        assert fsg._reference_eps(SILU["reference"]) is None

    def test_same_width_at_a_different_epsilon_is_detected(self, tmp_path):
        """`{family}_h{H}` does not encode epsilon, so two models can collide on one name.

        Whichever loses the collision is then served a reference computing a slightly
        different function than the model asks for.
        """
        d = tmp_path / "rmsnorm"
        d.mkdir()
        (d / "fused_add_rmsnorm_h2560.json").write_text(
            json.dumps(dict(NORM, reference="EPS = 1e-6\n"))
        )
        assert (
            fsg._existing_at_other_eps(tmp_path, "fused_add_rmsnorm_h2560", "1e-5")
            == "fused_add_rmsnorm_h2560"
        )
        assert fsg._existing_at_other_eps(tmp_path, "fused_add_rmsnorm_h2560", "1e-6") is None

    @pytest.mark.parametrize("eps,slug", [("1e-5", "1e5"), ("1e-6", "1e6"), ("1e-05", "1e5")])
    def test_slug_is_stable_across_spellings(self, eps, slug):
        assert fsg._eps_slug(eps) == slug


class TestSibling:
    def test_picks_the_widest_sibling(self, tmp_path):
        d = tmp_path / "activation"
        d.mkdir()
        for w in (3072, 4864):
            (d / f"silu_and_mul_d{w}.json").write_text("{}")
        assert fsg._sibling(tmp_path, "silu_and_mul", "d").stem == "silu_and_mul_d4864"

    def test_unrelated_family_is_not_matched(self, tmp_path):
        d = tmp_path / "rmsnorm"
        d.mkdir()
        (d / "fused_add_rmsnorm_h2048.json").write_text("{}")
        assert fsg._sibling(tmp_path, "rmsnorm", "h") is None
