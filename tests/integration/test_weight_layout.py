"""Tests for the weight transforms that feed fused GEMM epilogues.

The transform is what lets one kernel serve any model of the same shape, instead of a
kernel branch per model layout. These check the algebra that makes that valid.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer_bench.integration import deinterleave_output, interleave_gate_up, rms_row_scale


class TestInterleaveGateUp:
    def test_places_gate_on_even_and_up_on_odd_columns(self):
        gate = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        up = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 100
        out = interleave_gate_up(gate, up)
        assert out.shape == (2, 6)
        assert torch.equal(out[:, 0::2], gate.T)
        assert torch.equal(out[:, 1::2], up.T)

    def test_folds_the_norm_gamma_into_the_weight(self):
        """gamma scales input channels, so it multiplies into the weight, not the rows."""
        gate = torch.randn(4, 3)
        up = torch.randn(4, 3)
        gamma = torch.randn(3)
        out = interleave_gate_up(gate, up, gamma)
        assert torch.allclose(out[:, 0::2], (gate * gamma).T, atol=1e-6)
        assert torch.allclose(out[:, 1::2], (up * gamma).T, atol=1e-6)

    def test_rejects_mismatched_projections(self):
        with pytest.raises(ValueError, match="same shape"):
            interleave_gate_up(torch.randn(4, 3), torch.randn(5, 3))

    def test_rejects_wrong_norm_width(self):
        with pytest.raises(ValueError, match="input dimension"):
            interleave_gate_up(torch.randn(4, 3), torch.randn(4, 3), torch.randn(4))

    def test_rejects_non_2d_projections(self):
        with pytest.raises(ValueError, match="2-D"):
            interleave_gate_up(torch.randn(4), torch.randn(4))

    def test_honours_the_requested_dtype(self):
        out = interleave_gate_up(torch.randn(4, 3), torch.randn(4, 3), dtype=torch.bfloat16)
        assert out.dtype is torch.bfloat16

    def test_folds_in_float_even_for_low_precision_weights(self):
        """Multiplying gamma into a bf16 weight in bf16 loses precision permanently."""
        gate = torch.randn(4, 3, dtype=torch.bfloat16)
        up = torch.randn(4, 3, dtype=torch.bfloat16)
        gamma = torch.randn(3, dtype=torch.bfloat16)
        out = interleave_gate_up(gate, up, gamma, dtype=torch.float32)
        assert torch.allclose(out[:, 0::2], (gate.float() * gamma.float()).T, atol=1e-6)


class TestRmsRowScale:
    def test_matches_the_rsqrt_factor(self):
        x = torch.randn(8, 16)
        scale = rms_row_scale(x, 1e-6)
        assert torch.allclose(scale, torch.rsqrt(x.pow(2).mean(-1) + 1e-6), atol=1e-6)

    def test_is_contiguous_float32(self):
        scale = rms_row_scale(torch.randn(8, 16, dtype=torch.bfloat16), 1e-6)
        assert scale.dtype is torch.float32 and scale.is_contiguous()

    def test_flattens_leading_dimensions(self):
        assert rms_row_scale(torch.randn(2, 3, 16), 1e-6).shape == (6,)


class TestAlgebraHolds:
    def test_row_scale_commutes_through_the_gemm(self):
        """The whole fusion rests on (r*x) @ W == r*(x @ W). If that fails, so does it."""
        x = torch.randn(8, 16)
        w = torch.randn(16, 32)
        r = torch.rand(8) + 0.5
        assert torch.allclose((r[:, None] * x) @ w, r[:, None] * (x @ w), atol=1e-4)

    def test_end_to_end_algebra_reproduces_a_gated_mlp(self):
        """Transform + fused maths == the model's own norm -> project -> SwiGLU."""
        k, d, m, eps = 16, 12, 5, 1e-6
        x = torch.randn(m, k)
        gate, up, gamma = torch.randn(d, k), torch.randn(d, k), torch.randn(k)

        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma
        expected = F.silu(normed @ gate.T) * (normed @ up.T)

        B = interleave_gate_up(gate, up, gamma)
        acc = (x @ B) * rms_row_scale(x, eps)[:, None]
        got = F.silu(acc[..., 0::2]) * acc[..., 1::2]

        assert torch.allclose(got, expected, atol=1e-4)


class TestDeinterleaveOutput:
    def test_takes_the_even_lanes(self):
        out = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])
        assert torch.equal(deinterleave_output(out), torch.tensor([[1.0, 2.0, 3.0]]))
