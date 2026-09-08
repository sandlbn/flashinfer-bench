"""The vLLM block-scaled FP8 baseline: signature binding and block-size resolution."""

from __future__ import annotations

import pytest

from flashinfer_bench.data import AxisConst, AxisVar, Definition, TensorSpec
from flashinfer_bench.integration.xpu_kernels import REGISTRY, VLLM, _definition_block_size


def _w8a8_def(name="gemm_fp8_w8a8_block128x128_n4096_k2560", *, tags=None, axes=True):
    """A block-scaled W8A8 definition.

    ``axes=False`` names the scale axes something other than ``N_blocks``/``K_blocks``,
    which is how the dataset's existing FP8 definitions spell them (``K_a_blocks``,
    ``K_b_blocks``). The schema requires every referenced axis to be declared, so this is
    the shape a definition takes when the block extent cannot be derived from the axes and
    the ``quantization:`` tag is the only thing that states it.
    """
    n_blk, k_blk = ("N_blocks", "K_blocks") if axes else ("N_b_blocks", "K_a_blocks")
    return Definition(
        name=name,
        op_type="gemm",
        tags=tags if tags is not None else ["quantization:float8_e4m3fn", "quantization:block128x128"],
        axes={
            "M": AxisVar(),
            "N": AxisConst(value=4096),
            "K": AxisConst(value=2560),
            n_blk: AxisConst(value=32),
            k_blk: AxisConst(value=20),
        },
        inputs={
            "A_fp8": TensorSpec(shape=["M", "K"], dtype="float8_e4m3fn"),
            "A_scale": TensorSpec(shape=["M", k_blk], dtype="float32"),
            "B_fp8": TensorSpec(shape=["N", "K"], dtype="float8_e4m3fn"),
            "B_scale": TensorSpec(shape=[n_blk, k_blk], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="bfloat16")},
        reference="import torch\n\ndef run(A_fp8, A_scale, B_fp8, B_scale):\n    return A_fp8\n",
    )


@pytest.fixture
def kernel():
    matches = [k for k in REGISTRY if k.provider == VLLM and "w8a8" in k.name]
    assert len(matches) == 1, "expected exactly one vLLM W8A8 block-scaled entry"
    return matches[0]


class TestBinding:
    def test_binds_the_w8a8_signature(self, kernel):
        assert kernel.matches(_w8a8_def())

    def test_does_not_bind_a_dense_gemm(self, kernel):
        """The dense definition takes two inputs; binding it would compute nonsense."""
        dense = Definition(
            name="gemm_n4096_k2560",
            op_type="gemm",
            axes={"M": AxisVar(), "N": AxisConst(value=4096), "K": AxisConst(value=2560)},
            inputs={
                "A": TensorSpec(shape=["M", "K"], dtype="bfloat16"),
                "B": TensorSpec(shape=["N", "K"], dtype="bfloat16"),
            },
            outputs={"C": TensorSpec(shape=["M", "N"], dtype="bfloat16")},
            reference="import torch\n\ndef run(A, B):\n    return A @ B.T\n",
        )
        assert not kernel.matches(dense)


class TestBlockSizeResolution:
    def test_derived_from_the_axes(self):
        """N/N_blocks is the block extent, and it is what the scale tensor's shape means."""
        d = _w8a8_def()
        assert _definition_block_size(d, "N") == 128
        assert _definition_block_size(d, "K") == 128

    def test_falls_back_to_the_quantization_tag(self):
        d = _w8a8_def(axes=False, tags=["quantization:block64x128"])
        assert _definition_block_size(d, "N") == 64
        assert _definition_block_size(d, "K") == 128

    def test_refuses_to_guess_when_nothing_declares_it(self):
        """A wrong block size reads the wrong scale for every block, silently."""
        d = _w8a8_def(axes=False, tags=["quantization:float8_e4m3fn"])
        with pytest.raises(ValueError, match="block size is unknown"):
            _definition_block_size(d, "N")

    def test_rendered_wrapper_carries_the_resolved_block_size(self, kernel):
        src = kernel.render(_w8a8_def())
        assert "BLOCK_N = 128" in src
        assert "BLOCK_K = 128" in src
        # run() takes the definition's inputs and nothing else.
        assert "def run(A_fp8, A_scale, B_fp8, B_scale):" in src
