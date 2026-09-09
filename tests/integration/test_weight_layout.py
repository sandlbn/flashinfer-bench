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


# ----------------------------------------------------------------------------- row padding

from flashinfer_bench.integration import weight_layout as wl  # noqa: E402

PERIOD = 6 * 1024
"""An explicit channel period, so these tests exercise the arithmetic and not the device."""


class TestChannelPeriodRule:
    def test_camps_exactly_when_pitch_is_a_multiple_of_the_period(self):
        """The one condition the transform keys on: pitch % period == 0, nothing else."""
        cases = (
            (3072, True),
            (6144, True),
            (1536, False),
            (1024, False),
            (2048, False),
            (3104, False),
        )
        for k, expect in cases:
            w = torch.zeros(4, k, dtype=torch.bfloat16)
            assert wl.camps_on_one_channel(w, PERIOD) is expect, k

    def test_rule_is_on_bytes_not_elements(self):
        """The same element count camps or not depending on the dtype's width."""
        assert wl.camps_on_one_channel(torch.zeros(4, 1536, dtype=torch.float32), PERIOD)
        assert not wl.camps_on_one_channel(torch.zeros(4, 1536, dtype=torch.bfloat16), PERIOD)
        assert not wl.camps_on_one_channel(torch.zeros(4, 1536, dtype=torch.int8), PERIOD)
        assert wl.camps_on_one_channel(torch.zeros(4, 6144, dtype=torch.int8), PERIOD)

    def test_a_padded_view_no_longer_camps(self):
        w = torch.zeros(4, 3072, dtype=torch.bfloat16)
        assert not wl.camps_on_one_channel(wl.pad_rows_off_channel_period(w, PERIOD), PERIOD)

    def test_only_row_major_2d_qualifies(self):
        assert not wl.camps_on_one_channel(torch.zeros(3072, dtype=torch.bfloat16), PERIOD)
        assert not wl.camps_on_one_channel(torch.zeros(2, 4, 3072, dtype=torch.bfloat16), PERIOD)
        assert not wl.camps_on_one_channel(torch.zeros(3072, 4, dtype=torch.bfloat16).t(), PERIOD)

    def test_a_different_period_makes_different_pitches_camp(self):
        """The period is whatever was measured for the device; the rule carries no number."""
        w = torch.zeros(4, 3072, dtype=torch.bfloat16)  # 6144-byte pitch
        assert wl.camps_on_one_channel(w, 5 * 1024) is False
        assert wl.camps_on_one_channel(w, 6 * 1024) is True
        assert wl.camps_on_one_channel(w, 3 * 1024) is True

    def test_period_override_is_taken_as_given(self, monkeypatch):
        monkeypatch.setenv(wl.PERIOD_ENV, "5120")
        assert wl.channel_period_bytes() == 5120
        assert wl.camps_on_one_channel(torch.zeros(4, 2560, dtype=torch.bfloat16))
        assert not wl.camps_on_one_channel(torch.zeros(4, 3072, dtype=torch.bfloat16))

    def test_rejects_a_non_positive_period_override(self, monkeypatch):
        monkeypatch.setenv(wl.PERIOD_ENV, "0")
        with pytest.raises(ValueError, match="positive"):
            wl.channel_period_bytes()

    def test_pad_never_lands_back_on_the_period(self):
        with pytest.raises(ValueError, match="channel period"):
            wl.padded_row_pitch(6144, PERIOD, pad_bytes=PERIOD)

    def test_probe_declines_host_memory(self):
        """Unknown is not a win: with nothing to stream from, the probe answers None."""
        w = torch.randn(8, 3072, dtype=torch.bfloat16)
        assert wl.streaming_pad_wins(w, wl.pad_rows_off_channel_period(w, PERIOD)) is None


class TestUnknownPeriodIsNotAnAnswer:
    """With no measured period the rule has nothing to test a pitch against. It says so;
    it does not answer False, which a caller would read as "does not camp"."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.delenv(wl.PERIOD_ENV, raising=False)
        monkeypatch.setattr(wl, "_PERIODS", {})

    def test_host_memory_has_no_period_and_asks_no_calibration(self, monkeypatch):
        from flashinfer_bench.device import calibration

        monkeypatch.setattr(
            calibration, "get", lambda *a, **k: pytest.fail("calibration measured for host memory")
        )
        assert wl.channel_period_bytes(torch.device("cpu")) is None
        assert wl.channel_period_bytes("cpu") is None

    def test_an_unresolved_calibration_is_none(self, monkeypatch):
        from flashinfer_bench.device import calibration

        monkeypatch.setattr(calibration, "get", lambda *a, **k: None)
        assert wl.channel_period_bytes("xpu:0") is None
        record = calibration.Calibration("p", "t", 1.0, 1.0, 1.0, 1.0, {}, None)
        monkeypatch.setattr(wl, "_PERIODS", {})
        monkeypatch.setattr(calibration, "get", lambda *a, **k: record)
        assert wl.channel_period_bytes("xpu:0") is None

    def test_a_resolved_calibration_is_read_once_per_device(self, monkeypatch):
        from flashinfer_bench.device import calibration

        calls = []
        record = calibration.Calibration("p", "t", 1.0, 1.0, 1.0, 1.0, {}, 5120)
        monkeypatch.setattr(calibration, "get", lambda d=None, **k: calls.append(d) or record)
        assert wl.channel_period_bytes("xpu:0") == 5120
        assert wl.channel_period_bytes(torch.device("xpu", 0)) == 5120
        assert calls == ["xpu:0"]

    def test_the_rule_refuses_rather_than_answers(self):
        w = torch.zeros(4, 3072, dtype=torch.bfloat16)  # host memory: no period
        with pytest.raises(wl.ChannelPeriodUnknown):
            wl.camps_on_one_channel(w)
        with pytest.raises(wl.ChannelPeriodUnknown):
            wl.pad_rows_off_channel_period(w)
        assert isinstance(wl.ChannelPeriodUnknown("x"), LookupError)

    def test_tensors_that_can_never_camp_need_no_period(self):
        assert wl.camps_on_one_channel(torch.zeros(3072, dtype=torch.bfloat16)) is False
        assert wl.pad_rows_off_channel_period(torch.zeros(3072, dtype=torch.bfloat16)) is None

    def test_an_explicit_period_needs_no_calibration(self, monkeypatch):
        from flashinfer_bench.device import calibration

        monkeypatch.setattr(calibration, "get", lambda *a, **k: pytest.fail("not needed"))
        w = torch.zeros(4, 3072, dtype=torch.bfloat16)
        assert wl.camps_on_one_channel(w, PERIOD) is True
        assert wl.pad_rows_off_channel_period(w, PERIOD) is not None

    def test_a_non_positive_period_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            wl.camps_on_one_channel(torch.zeros(4, 3072, dtype=torch.bfloat16), 0)


class TestPadRowsOffChannelPeriod:
    def test_leaves_a_non_camping_weight_alone(self):
        assert (
            wl.pad_rows_off_channel_period(torch.randn(8, 1024, dtype=torch.bfloat16), PERIOD)
            is None
        )

    def test_changes_only_the_row_stride(self):
        w = torch.randn(8, 3072, dtype=torch.bfloat16)
        p = wl.pad_rows_off_channel_period(w, PERIOD)
        assert p is not None
        assert p.shape == w.shape and p.dtype is w.dtype and p.device == w.device
        assert p.stride(1) == 1 and p.stride(0) == w.stride(0) + wl.ROW_PAD_BYTES // 2
        assert wl.row_pitch_bytes(p) % PERIOD == wl.ROW_PAD_BYTES
        assert torch.equal(p, w)

    def test_rows_stay_cache_line_aligned(self):
        p = wl.pad_rows_off_channel_period(torch.randn(8, 3072, dtype=torch.bfloat16), PERIOD)
        assert wl.row_pitch_bytes(p) % 64 == 0

    def test_storage_overhead_is_one_pad_per_row(self):
        w = torch.randn(8, 3072, dtype=torch.bfloat16)
        p = wl.pad_rows_off_channel_period(w, PERIOD)
        extra = p.untyped_storage().nbytes() - w.untyped_storage().nbytes()
        assert extra == 8 * wl.ROW_PAD_BYTES

    def test_rejects_a_pad_that_is_not_whole_elements(self):
        with pytest.raises(ValueError, match="whole"):
            wl.pad_rows_off_channel_period(
                torch.randn(8, 1536, dtype=torch.float32), PERIOD, pad_bytes=6
            )

    def test_gemm_is_bit_identical_through_the_padded_weight(self):
        """The whole point: the same numbers reach the GEMM, so the output is the same bits."""
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(4, 3072, dtype=dtype)
            w = torch.randn(1024, 3072, dtype=dtype)
            k_bytes = 3072 * x.element_size()
            period = k_bytes // 4  # so both dtypes camp under this test's period
            p = wl.pad_rows_off_channel_period(w, period)
            assert p is not None
            assert torch.equal(F.linear(x, w), F.linear(x, p))
