"""The vLLM adapters must handle 3D activations, not decline them.

Both adapters used to bail on `x.dim() != 2` rather than reshape, on the stated assumption
that vLLM flattens before the layer "in the paths that matter". Measured against real
serving runs that is false for models that keep [batch, seq, hidden] through the norm: the
overwhelming majority of the family's calls arrived 3D and were declined, so the family
contributed nothing however good its kernel was.

Flattening is only safe if it aliases. The residual path writes in place and its callers
read the tensors they passed in, so a reshape that copied would compute the right answer
into a buffer nobody looks at -- silently, with the counters still reporting "applied".
These tests pin the aliasing, not just the arithmetic.
"""

from typing import Any, Dict

import pytest
import torch

from flashinfer_bench.integration.vllm.adapters import activation as activation_mod
from flashinfer_bench.integration.vllm.adapters import rmsnorm as rmsnorm_mod
from flashinfer_bench.integration.vllm.adapters.activation import SiluAndMulAdapter
from flashinfer_bench.integration.vllm.adapters.rmsnorm import RMSNormAdapter

HIDDEN = 16


class _Layer:
    """Stands in for vLLM's RMSNorm/SiluAndMul module."""

    def __init__(self, weight=None):
        self.weight = weight


def _spec(adapter):
    return adapter.targets()[0]


def _ref_rmsnorm(x, weight, eps=1e-6):
    f = x.to(torch.float32)
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


@pytest.fixture
def calls(monkeypatch):
    """Record what apply() was handed, and compute the real answer for it."""
    seen: Dict[str, Any] = {}

    def fake_apply(name, kwargs=None, fallback=None, **_):
        kwargs = kwargs or {}
        seen["name"] = name
        seen["kwargs"] = kwargs
        if name.startswith("silu_and_mul"):
            x = kwargs["x"]
            d = x.shape[-1] // 2
            gate, up = x[..., :d].float(), x[..., d:].float()
            return (torch.nn.functional.silu(gate) * up).to(x.dtype)
        if name.startswith("fused_add_rmsnorm_residual"):
            x, r, w = kwargs["hidden_states"], kwargs["residual"], kwargs["weight"]
            summed = x.to(torch.float32) + r.to(torch.float32)
            kwargs["residual_out"].copy_(summed.to(x.dtype))
            kwargs["output"].copy_(_ref_rmsnorm(summed.to(x.dtype), w))
            return None
        return _ref_rmsnorm(kwargs["hidden_states"], kwargs["weight"])

    monkeypatch.setattr(rmsnorm_mod, "apply", fake_apply)
    monkeypatch.setattr(activation_mod, "apply", fake_apply)
    return seen


class TestRMSNorm3D:
    def test_three_d_input_is_flattened_and_dispatched(self, calls):
        adapter = RMSNormAdapter()
        layer = _Layer(torch.ones(HIDDEN))
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: pytest.fail("should not fall back")
        )
        x = torch.randn(2, 3, HIDDEN)
        out = wrapper(layer, x)

        # The bare name is tried first -- it is what almost every definition uses, and a
        # miss costs a full resolve, so the common case must not pay for the rare one.
        assert calls["name"] == f"rmsnorm_h{HIDDEN}"
        assert calls["kwargs"]["hidden_states"].shape == (6, HIDDEN)
        assert out.shape == x.shape
        torch.testing.assert_close(out, _ref_rmsnorm(x, layer.weight))

    def test_residual_path_writes_through_to_the_callers_tensors(self, calls):
        """The kernel writes in place; the caller reads the 3D tensors it passed in."""
        adapter = RMSNormAdapter()
        layer = _Layer(torch.ones(HIDDEN))
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: pytest.fail("should not fall back")
        )
        x = torch.randn(2, 3, HIDDEN)
        r = torch.randn(2, 3, HIDDEN)
        expect_res = x + r
        expect_out = _ref_rmsnorm(expect_res, layer.weight)

        out, res = wrapper(layer, x, r)

        assert out is x and res is r, "callers hold these objects; they must be updated"
        assert out.shape == (2, 3, HIDDEN)
        torch.testing.assert_close(res, expect_res)
        torch.testing.assert_close(out, expect_out)

    def test_noncontiguous_three_d_defers_instead_of_copying(self, calls):
        """A copy would break the aliasing the residual path depends on."""
        adapter = RMSNormAdapter()
        layer = _Layer(torch.ones(HIDDEN))
        fell_back = []
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: (fell_back.append(True), torch.zeros(1))[1]
        )
        x = torch.randn(2, HIDDEN, 3).transpose(1, 2)
        assert not x.is_contiguous()
        wrapper(layer, x)
        assert fell_back, "must defer rather than silently copy"
        assert "name" not in calls

    def test_two_d_input_is_unchanged(self, calls):
        adapter = RMSNormAdapter()
        layer = _Layer(torch.ones(HIDDEN))
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: pytest.fail("should not fall back")
        )
        x = torch.randn(5, HIDDEN)
        out = wrapper(layer, x)
        assert calls["kwargs"]["hidden_states"] is x
        assert out.shape == (5, HIDDEN)


class TestSiluAndMul3D:
    def test_three_d_input_is_flattened_and_output_reshaped(self, calls):
        """The output's last dim is d, not 2d -- only the leading dims are restored."""
        adapter = SiluAndMulAdapter()
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: pytest.fail("should not fall back")
        )
        x = torch.randn(2, 3, 2 * HIDDEN)
        out = wrapper(_Layer(), x)

        assert calls["name"] == f"silu_and_mul_d{HIDDEN}"
        assert calls["kwargs"]["x"].shape == (6, 2 * HIDDEN)
        assert out.shape == (2, 3, HIDDEN)

        gate, up = x[..., :HIDDEN].float(), x[..., HIDDEN:].float()
        torch.testing.assert_close(out, (torch.nn.functional.silu(gate) * up).to(x.dtype))

    def test_odd_width_still_declines(self, calls):
        adapter = SiluAndMulAdapter()
        fell_back = []
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: (fell_back.append(True), torch.zeros(1))[1]
        )
        wrapper(_Layer(), torch.randn(2, 3, 7))
        assert fell_back
        assert "name" not in calls


class TestDtypeQualifiedLookup:
    """A definition is named for its shape, but shape does not identify the operation.

    Two models at the same width can run in different precisions, and apply() refuses a
    solution whose dtype differs from the tensors handed to it. The dataset distinguishes
    them with a suffix, so the adapter has to ask for that name -- otherwise a correctly
    generated `*_float16` definition is never looked up and the counters say "no-solution",
    which reads as "nothing extracted for this shape".
    """

    def test_falls_through_to_the_suffixed_name_when_the_bare_one_does_not_match(self, monkeypatch):
        """A bare definition of the wrong dtype is refused by apply(), so the suffixed one
        is still reached -- which is what makes bare-first safe as well as cheaper."""
        tried = []

        def fake_apply(name, kwargs=None, fallback=None, **_):
            tried.append(name)
            if not name.endswith("_float32"):
                return fallback(**(kwargs or {}))  # bare definition declares another dtype
            return _ref_rmsnorm(kwargs["hidden_states"], kwargs["weight"])

        monkeypatch.setattr(rmsnorm_mod, "apply", fake_apply)
        adapter = RMSNormAdapter()
        wrapper = adapter.make_wrapper(
            _spec(adapter), lambda *_a, **_k: pytest.fail("should not fall back to vLLM")
        )
        out = wrapper(_Layer(torch.ones(HIDDEN)), torch.randn(4, HIDDEN))
        assert tried == [f"rmsnorm_h{HIDDEN}", f"rmsnorm_h{HIDDEN}_float32"]
        assert out.shape == (4, HIDDEN)

    def test_a_hit_on_the_bare_name_costs_only_one_lookup(self, monkeypatch):
        """Almost every definition is unsuffixed; the common path must not pay for the rare
        one, because a miss is a full resolve, key build and dtype check."""
        tried = []

        def fake_apply(name, kwargs=None, fallback=None, **_):
            tried.append(name)
            return _ref_rmsnorm(kwargs["hidden_states"], kwargs["weight"])

        monkeypatch.setattr(rmsnorm_mod, "apply", fake_apply)
        adapter = RMSNormAdapter()
        wrapper = adapter.make_wrapper(_spec(adapter), lambda *_a, **_k: pytest.fail("no"))
        wrapper(_Layer(torch.ones(HIDDEN)), torch.randn(4, HIDDEN))
        assert tried == [f"rmsnorm_h{HIDDEN}"]

    def test_the_original_runs_once_when_every_candidate_misses(self, monkeypatch):
        """Trying a second name must not re-run the fallback.

        vLLM's fused kernel is in place; running it per candidate would consume the
        buffers and corrupt the residual stream.
        """
        monkeypatch.setattr(
            rmsnorm_mod, "apply", lambda name, kwargs=None, fallback=None, **_: fallback()
        )
        calls = []

        def orig(self_, x, residual=None):
            calls.append(1)
            return (x, residual) if residual is not None else x

        adapter = RMSNormAdapter()
        wrapper = adapter.make_wrapper(_spec(adapter), orig)
        wrapper(_Layer(torch.ones(HIDDEN)), torch.randn(4, HIDDEN), torch.randn(4, HIDDEN))
        assert calls == [1], "vLLM's in-place kernel must run exactly once"
