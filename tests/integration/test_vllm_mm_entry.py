"""Tests for converting vLLM's linear weights to the mm operand and entry point.

The transform has two halves at two seams -- the load hook re-lays the weight, the apply
hook takes ``aten.mm`` to it -- and half of it is worse than none, so what is checked here
is what a serving A/B needs to trust: that a nominated weight is decided by measurement and
not by its shape, that a decided weight is counted, that the forward pass really reaches the
mm entry with the converted operand, that a declined weight is left exactly as it was, and
that the answer through the converted operand stays inside what the dtype explains.

vLLM is not importable here (see ``test_vllm_integration.py``), so stubs sit at the dotted
paths the patch resolves.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from flashinfer_bench.integration import patch_manager
from flashinfer_bench.integration import weight_layout as wl
from flashinfer_bench.integration.vllm import weight_layout as hook
from flashinfer_bench.integration.vllm.adapters import stats

ROWS = (128, 8)
"""Two regimes to decide in. Any two row counts; the rule is about how many WIN and LOSS."""


def _probe(speedup, spread=0.001):
    return wl.Probe(win=speedup > 1.0, speedup=speedup, spread=spread)


class _StubMethod:
    """Shaped like vllm.model_executor.layers.linear.UnquantizedLinearMethod."""

    def __init__(self):
        self.processed = []
        self.applied = []

    def process_weights_after_loading(self, layer):
        self.processed.append(layer)

    def apply(self, layer, x, bias=None):
        self.applied.append(layer)
        return F.linear(x, layer.weight, bias)


class _StubLinear(torch.nn.Module):
    """A vLLM linear layer as the loader leaves it: a Parameter and a quant method."""

    def __init__(self, n, k, method, dtype=torch.bfloat16, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k, dtype=dtype), requires_grad=False)
        self.bias = (
            torch.nn.Parameter(torch.randn(n, dtype=dtype), requires_grad=False) if bias else None
        )
        self.quant_method = method

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


@pytest.fixture
def stub_vllm(monkeypatch):
    root = types.ModuleType("vllm")
    me = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = _StubMethod
    for name, mod in (
        ("vllm", root),
        ("vllm.model_executor", me),
        ("vllm.model_executor.layers", layers),
        ("vllm.model_executor.layers.linear", linear),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(patch_manager, "_manager", patch_manager.PatchManager())
    hook.reset_installed()
    stats.reset()
    yield linear
    patch_manager.get_manager().unpatch_all()
    hook.reset_installed()
    stats.reset()


@pytest.fixture
def device_like(monkeypatch):
    """Let CPU tensors count as device memory, with row counts and a probe the test sets,
    so the delivery around the decision is what is exercised."""
    monkeypatch.setattr(hook, "_in_device_memory", lambda tensor: True)
    monkeypatch.setattr(hook, "probe_rows", lambda: ROWS)
    monkeypatch.setattr(hook, "_MM_DECISIONS", {})


def _wins(monkeypatch, *speedups):
    """Make the load-time measurement answer with these per-regime speedups."""
    answers = iter([_probe(s) if s is not None else None for s in speedups] * 64)
    monkeypatch.setattr(hook, "mm_entry_wins", lambda w, c, m: next(answers))


class TestOperand:
    """The shared primitives the transform is built from."""

    def test_conversion_keeps_everything_but_the_strides(self):
        w = torch.randn(6, 4, dtype=torch.bfloat16)
        converted = wl.to_mm_operand_layout(w)
        assert converted.shape == w.shape and converted.dtype == w.dtype
        assert torch.equal(converted, w)
        assert converted.stride() != w.stride()

    def test_the_operand_is_a_contiguous_view_of_the_converted_weight(self):
        w = torch.randn(6, 4, dtype=torch.bfloat16)
        converted = wl.to_mm_operand_layout(w)
        operand = wl.mm_operand(converted)
        assert operand.shape == (4, 6)
        assert operand.is_contiguous()
        assert operand.data_ptr() == converted.data_ptr(), "a view, not a copy"
        assert wl.mm_operand_ready(converted) and not wl.mm_operand_ready(w)

    def test_a_transposed_view_is_not_a_conversion(self):
        """t12 in the trial ledger: the view keeps the original descriptor and gains nothing."""
        w = torch.randn(6, 4, dtype=torch.bfloat16)
        assert not wl.mm_operand_ready(w)
        assert wl.mm_operand(w).is_contiguous() is False

    def test_an_already_re_laid_weight_is_not_converted_again(self):
        w = torch.randn(6, 4, dtype=torch.bfloat16)
        assert wl.to_mm_operand_layout(wl.to_mm_operand_layout(w)) is None
        assert wl.to_mm_operand_layout(torch.randn(4, 6).t()) is None

    def test_mm_of_the_operand_is_the_linear_the_stack_called(self):
        w = torch.randn(6, 4)
        x = torch.randn(3, 4)
        converted = wl.to_mm_operand_layout(w)
        assert torch.equal(F.linear(x, w), torch.ops.aten.mm.default(x, wl.mm_operand(converted)))


class TestRule:
    """`keep_conversion` is the whole policy; it must not privilege one regime."""

    @pytest.mark.parametrize(
        "speedups,keep",
        [
            ((1.05, 1.0), True),  # win somewhere, no loss
            ((1.0, 1.05), True),  # ... in either regime
            ((1.05, 0.9), False),  # a measured loss anywhere is not paid for a gain
            ((0.9, 1.05), False),
            ((1.0, 1.0), False),  # nothing measured is not a reason to act
            ((0.9, 0.9), False),
        ],
    )
    def test_truth_table(self, speedups, keep):
        assert hook.keep_conversion(tuple(_probe(s) for s in speedups)) is keep


class TestInstall:
    def test_inert_without_the_env_var(self, stub_vllm, monkeypatch):
        monkeypatch.delenv(hook.MM_ENV_VAR, raising=False)
        assert hook.install_mm_entry() is False
        assert stub_vllm.UnquantizedLinearMethod.apply is _StubMethod.apply

    def test_patches_both_seams_and_restores_them(self, stub_vllm, monkeypatch):
        monkeypatch.setenv(hook.MM_ENV_VAR, "1")
        load, call = _StubMethod.process_weights_after_loading, _StubMethod.apply
        assert hook.install_mm_entry() is True
        assert hook.install_mm_entry() is True  # idempotent
        assert stub_vllm.UnquantizedLinearMethod.process_weights_after_loading is not load
        assert stub_vllm.UnquantizedLinearMethod.apply is not call
        patch_manager.get_manager().unpatch_all()
        assert stub_vllm.UnquantizedLinearMethod.process_weights_after_loading is load
        assert stub_vllm.UnquantizedLinearMethod.apply is call

    def test_half_the_path_is_refused(self, stub_vllm, monkeypatch):
        """Converting the weight and leaving F.linear in place ships the copy and no gain."""
        monkeypatch.delattr(stub_vllm.UnquantizedLinearMethod, "apply")
        assert hook.install_mm_entry(force=True) is False
        assert hook.MM not in hook._LIVE

    def test_no_op_without_vllm(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "vllm", raising=False)
        monkeypatch.setattr(patch_manager, "_manager", patch_manager.PatchManager())
        hook.reset_installed()
        assert hook.install_mm_entry(force=True) is False


class TestTransformApplies:
    """These fail if the transform stops firing: the count, the strides, the entry point."""

    def test_converts_a_measured_winner_and_counts_it(self, stub_vllm, device_like, monkeypatch):
        _wins(monkeypatch, 1.05, 1.0)
        hook.install_mm_entry(force=True)
        method = _StubMethod()
        layer = _StubLinear(8, 4, method)
        before = layer.weight.data.clone()

        method.process_weights_after_loading(layer)

        assert method.processed == [layer], "vLLM's own processing still runs"
        assert wl.mm_operand_ready(layer.weight.data), "the weight was re-laid"
        assert torch.equal(layer.weight.data, before), "and only its strides changed"
        applied = [k for k in stats.dispatch_stats() if "applied" in k]
        assert len(applied) == 1 and "8x4" in applied[0] and "m128-WIN-1.050x" in applied[0]

    def test_the_forward_reaches_the_mm_entry_with_the_converted_operand(
        self, stub_vllm, device_like, monkeypatch
    ):
        _wins(monkeypatch, 1.05, 1.0)
        hook.install_mm_entry(force=True)
        method = _StubMethod()
        layer = _StubLinear(8, 4, method)
        expected = F.linear(torch.eye(4, dtype=torch.bfloat16), layer.weight.data)
        method.process_weights_after_loading(layer)

        got = layer(torch.eye(4, dtype=torch.bfloat16))
        layer(torch.eye(4, dtype=torch.bfloat16))

        assert method.applied == [], "vLLM's own gemm was bypassed, not wrapped around"
        assert torch.equal(got, expected)
        confirmations = [k for k in stats.dispatch_stats() if "forward-mm-entry" in k]
        assert confirmations == ["linear_mm_entry forward-mm-entry contiguous-4x8"]
        assert stats.dispatch_stats()[confirmations[0]] == 1, "reported once, not per call"

    def test_a_declined_weight_is_untouched_and_keeps_vllms_gemm(
        self, stub_vllm, device_like, monkeypatch
    ):
        _wins(monkeypatch, 1.05, 0.9)  # a win at one regime, a loss at the other
        hook.install_mm_entry(force=True)
        method = _StubMethod()
        layer = _StubLinear(8, 4, method)
        strides = layer.weight.data.stride()

        method.process_weights_after_loading(layer)
        layer(torch.eye(4, dtype=torch.bfloat16))

        assert layer.weight.data.stride() == strides
        assert method.applied == [layer], "the untouched layer still runs vLLM's own call"
        assert [k for k in stats.dispatch_stats() if "applied" in k] == []
        assert any("m128-WIN-1.050x-m8-LOSS-0.900x" in k for k in stats.dispatch_stats())

    def test_an_unmeasurable_weight_is_left_alone(self, stub_vllm, device_like, monkeypatch):
        _wins(monkeypatch, 1.05, None)
        hook.install_mm_entry(force=True)
        layer = _StubLinear(8, 4, _StubMethod())
        assert hook.convert_layer_to_mm_entry(layer) is False
        assert "linear_mm_entry no-op unmeasurable" in stats.dispatch_stats()

    def test_force_converts_without_measuring(self, stub_vllm, device_like, monkeypatch):
        monkeypatch.setenv(hook.MM_ENV_VAR, hook.FORCE)
        monkeypatch.setattr(
            hook, "mm_entry_wins", lambda *a, **k: pytest.fail("force must not measure")
        )
        hook.install_mm_entry()
        layer = _StubLinear(8, 4, _StubMethod())
        assert hook.convert_layer_to_mm_entry(layer) is True
        assert "linear_mm_entry applied 8x4-forced" in stats.dispatch_stats()

    def test_a_biased_projection_is_left_alone(self, stub_vllm, device_like, monkeypatch):
        """`mm` takes no bias; folding one in is `addmm`, an entry point nothing priced."""
        _wins(monkeypatch, 1.05, 1.05)
        hook.install_mm_entry(force=True)
        layer = _StubLinear(8, 4, _StubMethod(), bias=True)
        assert hook.convert_layer_to_mm_entry(layer) is False
        assert "linear_mm_entry unsupported bias" in stats.dispatch_stats()

    def test_host_weights_are_left_alone(self, stub_vllm, monkeypatch):
        monkeypatch.setattr(hook, "probe_rows", lambda: ROWS)
        hook.install_mm_entry(force=True)
        layer = _StubLinear(8, 4, _StubMethod())
        assert hook.convert_layer_to_mm_entry(layer) is False
        assert "linear_mm_entry unsupported cpu" in stats.dispatch_stats()

    def test_a_padded_weight_is_not_re_laid_on_top(self, stub_vllm, device_like, monkeypatch):
        """The two transforms are alternatives; the second declines what the first took."""
        _wins(monkeypatch, 1.05, 1.05)
        hook.install_mm_entry(force=True)
        layer = _StubLinear(8, 4, _StubMethod())
        layer.weight.data = wl._copy_with_pitch(layer.weight.data, 4 * 2 + wl.ROW_PAD_BYTES)
        assert hook.convert_layer_to_mm_entry(layer) is False
        assert "linear_mm_entry unsupported not-row-major" in stats.dispatch_stats()

    def test_no_row_counts_means_no_decision(self, stub_vllm, monkeypatch):
        monkeypatch.setattr(hook, "_in_device_memory", lambda tensor: True)
        monkeypatch.setattr(hook, "probe_rows", lambda: None)
        hook.install_mm_entry(force=True)
        layer = _StubLinear(8, 4, _StubMethod())
        assert hook.convert_layer_to_mm_entry(layer) is False
        assert "linear_mm_entry no-op batch-sizes-unknown" in stats.dispatch_stats()

    def test_the_forward_reports_a_weight_replaced_after_load(
        self, stub_vllm, device_like, monkeypatch
    ):
        _wins(monkeypatch, 1.05, 1.0)
        hook.install_mm_entry(force=True)
        method = _StubMethod()
        layer = _StubLinear(8, 4, method)
        method.process_weights_after_loading(layer)

        layer.weight.data = layer.weight.data.contiguous()  # something re-materialises it
        got = layer(torch.eye(4, dtype=torch.bfloat16))

        assert torch.equal(got, F.linear(torch.eye(4, dtype=torch.bfloat16), layer.weight.data))
        assert method.applied == [layer], "it fell back rather than reading a stale operand"
        assert (
            "linear_mm_entry forward-fell-back weight-replaced-after-load" in stats.dispatch_stats()
        )

    def test_a_failing_conversion_does_not_fail_the_load(self, stub_vllm, device_like, monkeypatch):
        def explode(*a, **k):
            raise RuntimeError("no")

        monkeypatch.setattr(hook, "mm_entry_wins", explode)
        hook.install_mm_entry(force=True)
        method = _StubMethod()
        layer = _StubLinear(8, 4, method)
        method.process_weights_after_loading(layer)  # must not raise
        assert method.processed == [layer]
        assert "linear_mm_entry unsupported error:RuntimeError" in stats.dispatch_stats()


class TestProbeRows:
    def test_the_override_is_read_as_row_counts(self, monkeypatch):
        monkeypatch.setenv(hook.MM_PROBE_TOKENS_ENV, "5040, 16")
        assert hook.probe_rows() == (5040, 16)
        monkeypatch.setenv(hook.MM_PROBE_TOKENS_ENV, "512")
        assert hook.probe_rows() == (512,)

    def test_a_nonsense_override_is_refused_rather_than_guessed(self, monkeypatch):
        monkeypatch.setenv(hook.MM_PROBE_TOKENS_ENV, "0,16")
        with pytest.raises(ValueError):
            hook.probe_rows()

    def test_it_falls_back_to_the_schedulers_own_bounds(self, monkeypatch):
        config = types.ModuleType("vllm.config")
        scheduler = types.SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=32)
        config.get_current_vllm_config_or_none = lambda: types.SimpleNamespace(
            scheduler_config=scheduler
        )
        monkeypatch.setitem(sys.modules, "vllm.config", config)
        monkeypatch.delenv(hook.MM_PROBE_TOKENS_ENV, raising=False)
        assert hook.probe_rows() == (4096, 32)

    def test_no_configuration_is_not_a_guess(self, monkeypatch):
        monkeypatch.delenv(hook.MM_PROBE_TOKENS_ENV, raising=False)
        config = types.ModuleType("vllm.config")
        config.get_current_vllm_config_or_none = lambda: None
        monkeypatch.setitem(sys.modules, "vllm.config", config)
        assert hook.probe_rows() is None


class TestPairedProbe:
    def test_it_favours_the_faster_arm(self):
        import time

        probe = wl.paired_speedup(
            lambda: time.sleep(0.004),
            lambda: time.sleep(0.002),
            lambda: None,
            rounds=4,
            calls=1,
            warmup_s=0.0,
        )
        assert wl.probe_verdict(probe) == "WIN" and probe.speedup > 1.5

    def test_it_calls_a_tie_a_tie(self):
        import time

        probe = wl.paired_speedup(
            lambda: time.sleep(0.002),
            lambda: time.sleep(0.002),
            lambda: None,
            rounds=6,
            calls=1,
            warmup_s=0.0,
        )
        assert wl.probe_verdict(probe) == "NOISE"


class TestNumerics:
    """What the entry point does to the answer, bounded by what the dtype can express.

    On the accelerator the two entry points reach different GEMM kernels, so the K reduction
    can be summed in a different order and the results are not guaranteed bit-identical. The
    bound is the dtype's own: an accumulation over K in a format whose relative spacing is
    ``eps`` cannot be pinned closer than ``eps * sqrt(K)``, and anything inside that is the
    format talking, not the kernel.
    """

    @pytest.mark.parametrize("shape", [(4096, 1024), (1024, 3072)])
    @pytest.mark.parametrize("m", [8, 512])
    @pytest.mark.requires_torch_xpu
    def test_the_converted_path_stays_inside_the_dtypes_own_spacing(self, shape, m):
        n, k = shape
        device = "xpu:0"
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        converted = wl.to_mm_operand_layout(w)

        stock = F.linear(x, w)
        ours = torch.ops.aten.mm.default(x, wl.mm_operand(converted))
        reference = F.linear(x.float(), w.float())

        scale = reference.abs().max().item()
        bound = torch.finfo(torch.bfloat16).eps * k**0.5
        gap = (stock.float() - ours.float()).abs().max().item() / scale
        assert gap <= bound, f"{gap} exceeds what bfloat16 over K={k} explains ({bound})"
        # Neither arm is the more wrong one: they straddle the same reference.
        stock_err = (stock.float() - reference).abs().max().item() / scale
        ours_err = (ours.float() - reference).abs().max().item() / scale
        assert ours_err <= max(stock_err * 2, bound)

    def test_on_one_kernel_the_answers_are_the_same_bits(self):
        """Where both entry points reach the same implementation, nothing moves at all."""
        w = torch.randn(8, 6, dtype=torch.bfloat16)
        x = torch.randn(4, 6, dtype=torch.bfloat16)
        converted = wl.to_mm_operand_layout(w)
        assert torch.equal(F.linear(x, w), torch.ops.aten.mm.default(x, wl.mm_operand(converted)))
