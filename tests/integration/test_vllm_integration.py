"""Tests for putting benchmarked kernels into a running vLLM server.

vLLM cannot be installed alongside this package -- it pins ``torch==2.13.0`` and this
environment runs 2.14 -- so these tests patch a stub that mirrors the structure the
adapters actually target: a ``CustomOp``-shaped class exposing ``forward_xpu`` at the
dotted path vLLM uses. That verifies the patching, the routing and the fallbacks. It does
**not** verify behaviour inside a real server; that needs vLLM in its own environment and
is not covered here.
"""

import collections
import sys
import types

import pytest
import torch

from flashinfer_bench.integration import patch_manager
from flashinfer_bench.integration.vllm import (
    ENV_VAR,
    install_vllm_integrations,
    uninstall_vllm_integrations,
)


class _StubRMSNorm:
    """Shaped like vllm.model_executor.layers.layernorm.RMSNorm."""

    def __init__(self, hidden: int):
        self.weight = torch.ones(hidden)
        self.variance_epsilon = 1e-6
        self.calls = 0

    def forward_xpu(self, x, residual=None):
        self.calls += 1
        if residual is None:
            return x * 2.0
        return x * 2.0, x + residual


class _StubSiluAndMul:
    """Shaped like vllm.model_executor.layers.activation.SiluAndMul."""

    def __init__(self):
        self.calls = 0

    def forward_xpu(self, x):
        self.calls += 1
        d = x.shape[-1] // 2
        return x[..., :d] * 3.0


@pytest.fixture
def stub_vllm(monkeypatch):
    """Install a module tree at the dotted paths the adapters patch."""
    root = types.ModuleType("vllm")
    me = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    layernorm = types.ModuleType("vllm.model_executor.layers.layernorm")
    activation = types.ModuleType("vllm.model_executor.layers.activation")
    layernorm.RMSNorm = _StubRMSNorm
    activation.SiluAndMul = _StubSiluAndMul
    for name, module in (
        ("vllm", root),
        ("vllm.model_executor", me),
        ("vllm.model_executor.layers", layers),
        ("vllm.model_executor.layers.layernorm", layernorm),
        ("vllm.model_executor.layers.activation", activation),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # A fresh manager per test, so one test's patches cannot leak into the next.
    monkeypatch.setattr(patch_manager, "_manager", patch_manager.PatchManager())
    yield layernorm, activation
    uninstall_vllm_integrations()


class TestOptIn:
    """A serving stack must not change behaviour because a package is importable."""

    def test_does_nothing_unless_enabled(self, stub_vllm, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert install_vllm_integrations() == []

    def test_env_var_enables_it(self, stub_vllm, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "1")
        assert set(install_vllm_integrations()) == {
            "rmsnorm_forward_xpu",
            "silu_and_mul_forward_xpu",
        }

    def test_absent_vllm_is_not_an_error(self, monkeypatch):
        """The normal case on a machine that has never installed vLLM."""
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(patch_manager, "_manager", patch_manager.PatchManager())
        assert install_vllm_integrations(force=True) == []


class TestFailsOpen:
    """Every path with no recorded solution must land on vLLM's own kernel."""

    def test_rmsnorm_falls_back_when_nothing_is_recorded(self, stub_vllm):
        layernorm, _ = stub_vllm
        install_vllm_integrations(force=True)
        layer = layernorm.RMSNorm(8)
        x = torch.randn(4, 8)
        out = layer.forward_xpu(x)
        assert torch.allclose(out, x * 2.0), "did not reach the original implementation"
        assert layer.calls == 1

    def test_fused_form_still_returns_two_tensors(self, stub_vllm):
        """Callers unpack (normed, residual); returning one value would break them."""
        layernorm, _ = stub_vllm
        install_vllm_integrations(force=True)
        layer = layernorm.RMSNorm(8)
        x, residual = torch.randn(4, 8), torch.randn(4, 8)
        out = layer.forward_xpu(x, residual)
        assert isinstance(out, tuple) and len(out) == 2

    def test_activation_falls_back(self, stub_vllm):
        _, activation = stub_vllm
        install_vllm_integrations(force=True)
        layer = activation.SiluAndMul()
        x = torch.randn(4, 16)
        assert torch.allclose(layer.forward_xpu(x), x[..., :8] * 3.0)

    @pytest.mark.parametrize(
        "shape,layer_name", [((2, 4, 8), "RMSNorm"), ((2, 4, 16), "SiluAndMul")]
    )
    def test_three_dimensional_input_defers(self, stub_vllm, shape, layer_name):
        """The definitions are 2D. Reshaping here would assume the caller's strides."""
        layernorm, activation = stub_vllm
        install_vllm_integrations(force=True)
        layer = layernorm.RMSNorm(8) if layer_name == "RMSNorm" else activation.SiluAndMul()
        layer.forward_xpu(torch.randn(*shape))
        assert layer.calls == 1, "3D input should have reached the original implementation"

    def test_odd_width_activation_defers(self, stub_vllm):
        _, activation = stub_vllm
        install_vllm_integrations(force=True)
        layer = activation.SiluAndMul()
        before = layer.calls
        layer.forward_xpu(torch.randn(4, 15))
        assert layer.calls == before + 1


class TestReversible:
    """An A/B measurement in one process needs the patch to come back off."""

    def test_uninstall_restores_the_original(self, stub_vllm):
        layernorm, _ = stub_vllm
        original = layernorm.RMSNorm.forward_xpu
        install_vllm_integrations(force=True)
        assert layernorm.RMSNorm.forward_xpu is not original
        uninstall_vllm_integrations()
        assert layernorm.RMSNorm.forward_xpu is original

    def test_installing_twice_is_idempotent(self, stub_vllm):
        install_vllm_integrations(force=True)
        layernorm, _ = stub_vllm
        once = layernorm.RMSNorm.forward_xpu
        install_vllm_integrations(force=True)
        assert layernorm.RMSNorm.forward_xpu is once


class _StubMLP:
    """Shaped like vllm's Qwen2MLP: gate_up_proj -> act_fn -> down_proj."""

    def __init__(self, k: int, d: int, bias=None):
        self.gate_up_proj = types.SimpleNamespace(
            weight=torch.ones(2 * d, k, dtype=torch.bfloat16), bias=bias
        )
        self.down_proj = lambda t: (t[:, :4], None)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x[:, :4]


@pytest.fixture
def stub_mlp(monkeypatch):
    import flashinfer_bench.integration.vllm.adapters.mlp as mlp_mod

    qwen2 = types.ModuleType("vllm.model_executor.models.qwen2")
    qwen2.Qwen2MLP = _StubMLP
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.qwen2",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen2", qwen2)
    monkeypatch.setattr(patch_manager, "_manager", patch_manager.PatchManager())
    monkeypatch.setattr(
        mlp_mod, "_MLP_PATHS", ("vllm.model_executor.models.qwen2.Qwen2MLP.forward",)
    )
    monkeypatch.setenv(mlp_mod.ENABLE_ENV, "1")
    yield qwen2, mlp_mod
    uninstall_vllm_integrations()


class TestMLPThreshold:
    """Fusion is a regression below the crossover, so it must not install unconditionally.

    Below the crossover the fused path has measured slower than vLLM's own, and decode lives
    at the small end, which is where a served model spends most of its time -- an
    unconditional patch would slow it down. The crossover itself is measured per part and
    supplied through the environment; nothing here stores one.
    """

    def test_small_batches_keep_vllms_own_path(self, stub_mlp):
        qwen2, _ = stub_mlp
        install_vllm_integrations(force=True)
        layer = qwen2.Qwen2MLP(k=8, d=8)
        layer.forward(torch.randn(1, 8, dtype=torch.bfloat16))
        layer.forward(torch.randn(64, 8, dtype=torch.bfloat16))
        assert layer.calls == 2, "decode-sized batches must reach the original"

    def test_threshold_is_configurable(self, stub_mlp, monkeypatch):
        _, mlp_mod = stub_mlp
        monkeypatch.setenv(mlp_mod.MIN_TOKENS_ENV, "128")
        assert mlp_mod._min_tokens() == 128

    def test_a_bad_threshold_fuses_nothing(self, stub_mlp, monkeypatch):
        _, mlp_mod = stub_mlp
        monkeypatch.setenv(mlp_mod.MIN_TOKENS_ENV, "not-a-number")
        assert mlp_mod._min_tokens() is None

    def test_large_batch_with_no_solution_still_falls_back(self, stub_mlp):
        """Above the threshold but with nothing recorded: vLLM's path, not a crash."""
        qwen2, _ = stub_mlp
        install_vllm_integrations(force=True)
        layer = qwen2.Qwen2MLP(k=8, d=8)
        out = layer.forward(torch.randn(1024, 8, dtype=torch.bfloat16))
        assert layer.calls == 1 and out.shape == (1024, 4)

    def test_a_biased_projection_is_left_alone(self, stub_mlp):
        """A bias changes the maths the fused kernel implements."""
        qwen2, _ = stub_mlp
        install_vllm_integrations(force=True)
        layer = qwen2.Qwen2MLP(k=8, d=8, bias=torch.zeros(16))
        layer.forward(torch.randn(1024, 8, dtype=torch.bfloat16))
        assert layer.calls == 1

    def test_dtype_mismatch_is_left_alone(self, stub_mlp):
        """A quantised weight is not what this kernel implements."""
        qwen2, _ = stub_mlp
        install_vllm_integrations(force=True)
        layer = qwen2.Qwen2MLP(k=8, d=8)
        layer.forward(torch.randn(1024, 8, dtype=torch.float32))
        assert layer.calls == 1


class TestFusedCallIsRecognised:
    """A successful fused call must be used, not recomputed.

    `apply` in destination-passing style writes into the output tensors and returns
    `None` -- the same value a miss returns. Reading the return value as the hit/miss
    signal therefore counted every success as `no-solution` and ran vLLM's MLP as well,
    doing the work twice and making fusion a slowdown. Only whether the fallback ran
    distinguishes the two, so that is what the adapter checks and what this test drives.
    """

    def test_a_destination_passing_hit_counts_as_fused(self, stub_mlp, monkeypatch):
        qwen2, mlp_mod = stub_mlp

        def fake_apply(name, kwargs=None, fallback=None):
            # A hit: fill the destination, never call the fallback, return None.
            kwargs["out"].fill_(1.0)
            return None

        monkeypatch.setattr(mlp_mod, "apply", fake_apply)
        monkeypatch.setattr(mlp_mod, "_SEEN", collections.Counter())
        # Pin the threshold: this test is about hit/miss accounting, not the crossover.
        monkeypatch.setenv(mlp_mod.MIN_TOKENS_ENV, "8")
        install_vllm_integrations(force=True)

        layer = qwen2.Qwen2MLP(k=8, d=8)
        layer.forward(torch.randn(1024, 8, dtype=torch.bfloat16))

        fused = sum(n for (kind, _), n in mlp_mod._SEEN.items() if kind == "fused")
        missed = sum(n for (kind, _), n in mlp_mod._SEEN.items() if kind == "no-solution")
        assert (fused, missed) == (1, 0), f"hit was miscounted: {dict(mlp_mod._SEEN)}"
        assert layer.calls == 0, "vLLM's own MLP ran as well as the fused kernel"

    def test_a_miss_still_falls_back_and_counts_as_such(self, stub_mlp, monkeypatch):
        qwen2, mlp_mod = stub_mlp

        def fake_apply(name, kwargs=None, fallback=None):
            return fallback()  # a miss invokes the fallback

        monkeypatch.setattr(mlp_mod, "apply", fake_apply)
        monkeypatch.setattr(mlp_mod, "_SEEN", collections.Counter())
        # Pin the threshold: this test is about hit/miss accounting, not the crossover.
        monkeypatch.setenv(mlp_mod.MIN_TOKENS_ENV, "8")
        install_vllm_integrations(force=True)

        layer = qwen2.Qwen2MLP(k=8, d=8)
        layer.forward(torch.randn(1024, 8, dtype=torch.bfloat16))

        missed = sum(n for (kind, _), n in mlp_mod._SEEN.items() if kind == "no-solution")
        assert missed == 1 and layer.calls == 1


class TestMLPFusionIsOptIn:
    """Measured against vLLM's own XPU kernels the fusion is a regression at every M.

    0.35x at M=512 through 0.99x at M=8192 (Arc B580, device-event timing, 2026-09-06):
    oneDNN post-ops cannot express SwiGLU over a single GEMM, so the fused path issues two
    [M,k]x[k,d] matmuls against vLLM's one [M,k]x[k,2d]. Installing that by default would
    make a serving stack slower for having this package on the path, so it stays off until
    asked for.
    """

    def test_it_is_not_installed_by_default(self, stub_mlp, monkeypatch):
        qwen2, mlp_mod = stub_mlp
        monkeypatch.delenv(mlp_mod.ENABLE_ENV, raising=False)
        original = qwen2.Qwen2MLP.forward
        install_vllm_integrations(force=True)
        assert qwen2.Qwen2MLP.forward is original, "MLP fusion installed without opt-in"

    def test_the_env_var_installs_it(self, stub_mlp, monkeypatch):
        qwen2, mlp_mod = stub_mlp
        monkeypatch.setenv(mlp_mod.ENABLE_ENV, "1")
        original = qwen2.Qwen2MLP.forward
        install_vllm_integrations(force=True)
        assert qwen2.Qwen2MLP.forward is not original

    def test_there_is_no_stored_threshold(self, stub_mlp, monkeypatch):
        """The crossover is a measurement of one part; unset means fuse nothing, not a guess."""
        _, mlp_mod = stub_mlp
        monkeypatch.delenv(mlp_mod.MIN_TOKENS_ENV, raising=False)
        assert mlp_mod._min_tokens() is None
        assert not hasattr(mlp_mod, "DEFAULT_MIN_TOKENS")
