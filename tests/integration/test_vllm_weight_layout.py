"""Tests for padding vLLM's linear weights at load.

The hook wraps ``UnquantizedLinearMethod.process_weights_after_loading``; vLLM itself is not
importable here (see ``test_vllm_integration.py``), so a stub class sits at that dotted path.
What is checked is what a serving A/B needs to trust: the transform applies to exactly the
weights whose pitch lands on the period, it is counted, the forward pass sees the padded
tensor, and the numbers through the padded weight are the numbers through the original.
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

PERIOD = 6 * 1024


class _StubMethod:
    """Shaped like vllm.model_executor.layers.linear.UnquantizedLinearMethod."""

    def __init__(self):
        self.processed = []

    def process_weights_after_loading(self, layer):
        self.processed.append(layer)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class _StubLinear(torch.nn.Module):
    """A vLLM linear layer as the loader sees it: a Parameter and a quant method."""

    def __init__(self, n, k, method, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k, dtype=dtype), requires_grad=False)
        self.quant_method = method

    def forward(self, x):
        return self.quant_method.apply(self, x)


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
    stats.reset()
    yield linear
    patch_manager.get_manager().unpatch_all()
    stats.reset()


@pytest.fixture
def device_like(monkeypatch):
    """Make CPU tensors count as device memory with a known period and a probe that says
    the pad wins, so the rule can fire and what is tested is the delivery around it."""
    monkeypatch.setattr(hook, "_in_device_memory", lambda tensor: True)
    monkeypatch.setattr(hook, "channel_period_bytes", lambda device: PERIOD)
    monkeypatch.setattr(
        hook, "streaming_pad_wins", lambda w, p: wl.PadProbe(win=True, speedup=1.3, spread=0.01)
    )
    monkeypatch.setattr(hook, "_DECISIONS", {})


class TestInstall:
    def test_inert_without_the_env_var(self, stub_vllm, monkeypatch):
        monkeypatch.delenv(hook.ENV_VAR, raising=False)
        assert hook.install_weight_row_padding() is False
        assert stub_vllm.UnquantizedLinearMethod.process_weights_after_loading is (
            _StubMethod.process_weights_after_loading
        )

    def test_patches_the_load_seam_when_enabled(self, stub_vllm, monkeypatch):
        monkeypatch.setenv(hook.ENV_VAR, "1")
        original = _StubMethod.process_weights_after_loading
        assert hook.install_weight_row_padding() is True
        assert hook.install_weight_row_padding() is True  # idempotent
        assert stub_vllm.UnquantizedLinearMethod.process_weights_after_loading is not original
        patch_manager.get_manager().unpatch_all()
        assert stub_vllm.UnquantizedLinearMethod.process_weights_after_loading is original

    def test_no_op_without_vllm(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "vllm", raising=False)
        monkeypatch.setattr(patch_manager, "_manager", patch_manager.PatchManager())
        assert hook.install_weight_row_padding(force=True) is False


class TestTransformApplies:
    """These fail if the transform stops applying: the count, the stride, the forward check."""

    def test_pads_camping_weights_and_only_those(self, stub_vllm, device_like):
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        camping = _StubLinear(64, 3072, method)  # 6144-byte pitch: on the period
        clear = _StubLinear(64, 1024, method)  # 2048-byte pitch: off it
        before = camping.weight.data.clone()
        for layer in (camping, clear):
            method.process_weights_after_loading(layer)

        assert method.processed == [camping, clear], "vLLM's own processing still runs"
        assert camping.weight.stride(0) == 3072 + wl.ROW_PAD_BYTES // 2
        assert clear.weight.stride(0) == 1024
        assert torch.equal(camping.weight.data, before)
        assert isinstance(camping.weight, torch.nn.Parameter), "still the layer's Parameter"

        counts = stats.dispatch_stats()
        assert counts == {
            f"{hook.FAMILY} applied pitch-6144-to-{6144 + wl.ROW_PAD_BYTES}-measured-1.30x": 1,
            f"{hook.FAMILY} no-op pitch-off-period-{PERIOD}": 1,
        }
        assert f"{hook.FAMILY}: 2 call(s), 1 applied" in stats.format_report()

    def test_probe_runs_once_per_shape(self, stub_vllm, device_like, monkeypatch):
        calls = []

        def probe(w, p):
            calls.append(tuple(w.shape))
            return wl.PadProbe(win=True, speedup=1.3, spread=0.01)

        monkeypatch.setattr(hook, "streaming_pad_wins", probe)
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        for _ in range(3):
            method.process_weights_after_loading(_StubLinear(64, 3072, method))
        method.process_weights_after_loading(_StubLinear(128, 3072, method))
        assert calls == [(64, 3072), (128, 3072)]
        assert stats.format_report().count("applied") >= 1
        assert f"{hook.FAMILY}: 4 call(s), 4 applied" in stats.format_report()

    def test_a_measured_loss_keeps_the_original(self, stub_vllm, device_like, monkeypatch):
        """The pitch rule says where camping can happen; only a measured win pads."""
        monkeypatch.setattr(
            hook,
            "streaming_pad_wins",
            lambda w, p: wl.PadProbe(win=False, speedup=0.97, spread=0.002),
        )
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)
        assert layer.weight.is_contiguous()
        assert stats.dispatch_stats() == {f"{hook.FAMILY} no-op measured-0.970x-spread-0.002": 1}

    def test_an_unmeasurable_probe_keeps_the_original(self, stub_vllm, device_like, monkeypatch):
        monkeypatch.setattr(hook, "streaming_pad_wins", lambda w, p: None)
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)
        assert layer.weight.is_contiguous()
        assert stats.dispatch_stats() == {f"{hook.FAMILY} no-op unmeasurable": 1}

    def test_force_mode_pads_without_measuring(self, stub_vllm, device_like, monkeypatch):
        def never(w, p):
            raise AssertionError("force mode must not probe")

        monkeypatch.setattr(hook, "streaming_pad_wins", never)
        monkeypatch.setenv(hook.ENV_VAR, hook.FORCE)
        assert hook.install_weight_row_padding() is True
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)
        assert layer.weight.stride(0) == 3072 + wl.ROW_PAD_BYTES // 2
        assert stats.dispatch_stats() == {
            f"{hook.FAMILY} applied pitch-6144-to-{6144 + wl.ROW_PAD_BYTES}-forced": 1
        }

    def test_forward_confirms_the_padded_pitch_once(self, stub_vllm, device_like):
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)
        x = torch.randn(4, 3072, dtype=torch.bfloat16)
        layer(x)
        layer(x)
        assert stats.dispatch_stats()[f"{hook.FAMILY} forward-saw-padded-pitch"] == 1
        assert layer._forward_pre_hooks == {}, "the check removes itself after one look"

    def test_forward_reports_when_the_padding_was_lost(self, stub_vllm, device_like):
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)
        layer.weight.data = layer.weight.data.contiguous()  # something re-materialised it
        layer(torch.randn(4, 3072, dtype=torch.bfloat16))
        assert stats.dispatch_stats()[f"{hook.FAMILY} forward-lost-padding"] == 1

    def test_cpu_weights_are_left_alone(self, stub_vllm):
        hook.install_weight_row_padding(force=True)
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)
        assert layer.weight.is_contiguous()
        assert stats.dispatch_stats() == {f"{hook.FAMILY} unsupported cpu": 1}

    def test_a_failing_pad_does_not_fail_the_load(self, stub_vllm, device_like, monkeypatch):
        hook.install_weight_row_padding(force=True)

        def boom(*_a, **_k):
            raise RuntimeError("no memory")

        monkeypatch.setattr(hook, "pad_rows_off_channel_period", boom)
        method = _StubMethod()
        layer = _StubLinear(64, 3072, method)
        method.process_weights_after_loading(layer)  # does not raise
        assert layer.weight.is_contiguous()
        assert stats.dispatch_stats() == {f"{hook.FAMILY} unsupported error:RuntimeError": 1}


class TestNumerics:
    """This fails if the transform ever changes what the layer computes."""

    def test_layer_output_is_bit_identical_after_padding(self, stub_vllm, device_like):
        hook.install_weight_row_padding(force=True)
        torch.manual_seed(0)
        method = _StubMethod()
        layer = _StubLinear(256, 3072, method)
        x = torch.randn(4, 3072, dtype=torch.bfloat16)
        expected = layer(x)
        method.process_weights_after_loading(layer)
        assert layer.weight.stride(0) != 3072, "precondition: the pad applied"
        assert torch.equal(layer(x), expected)
