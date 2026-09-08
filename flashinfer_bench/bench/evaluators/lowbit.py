"""Evaluator for low-bit quantized kernels with relaxed tolerances."""

from __future__ import annotations

import traceback
from typing import Any, List, Optional, Tuple

import torch
from typing_extensions import override

from flashinfer_bench.bench.config import EvalConfig, ResolvedEvalConfig
from flashinfer_bench.bench.utils import compute_error_stats, make_eval
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import Correctness, Definition, Evaluation, EvaluationStatus
from flashinfer_bench.device import device_synchronize

from .default import DefaultEvaluator
from .utils import allocate_outputs, normalize_result


_LOW_BIT_INPUT_DTYPES = frozenset(
    {"float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float6_e2m3", "float6_e3m2"}
)
"""Storage dtypes that make a definition low-bit, whatever it is named."""


class LowBitEvaluator(DefaultEvaluator):
    DEFAULT_REQUIRED_MATCHED_RATIO = 0.95

    DEFAULT_RTOL = 2e-2
    DEFAULT_ATOL = 2e-2
    """Tolerance for a low-bit kernel, when nothing on the command line says otherwise.

    The generic default is 1e-2 for every dtype. A low-bit kernel dequantises into
    bfloat16 and is compared against a float32 reference, and bfloat16's relative spacing
    is 2**-8 = 3.9e-3, so 1e-2 leaves under three ULPs -- a budget the GEMM's own
    accumulation spends before the kernel does anything wrong. 2e-2 is about five ULPs:
    loose enough to admit a correct kernel, far tighter than the O(0.1-1) error a genuinely
    wrong one produces.

    Relaxing the ratio without relaxing the tolerance, as this evaluator did, is the worst
    of both: measured on Arc B580, `gemm_fp8_w8a8_block128x128_*` reached ~90% of elements
    within 1e-2 and ~96% within 2e-2, so the 95% gate was unreachable at the tolerance it
    was being applied with, and three correct definitions failed with nothing to indicate
    a flag would change it.

    This does loosen the MoE FP8 definitions that already reached this evaluator, from 1e-2
    to 2e-2. That is the intended correction, not a side effect: they are low-bit kernels
    measured the same way and were subject to the same mismatch.
    """

    @override
    @classmethod
    def eval_defaults(cls) -> Optional[EvalConfig]:
        """Tolerance and match ratio together -- a ratio only means something at a stated
        tolerance, so both are declared here and either can still be overridden."""
        return EvalConfig(
            rtol=cls.DEFAULT_RTOL,
            atol=cls.DEFAULT_ATOL,
            required_matched_ratio=cls.DEFAULT_REQUIRED_MATCHED_RATIO,
        )

    @override
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        """Whether this definition is low-bit, judged by what it declares.

        Selected on the definition's dtypes and quantization tags rather than a substring
        of its name. The name test this replaced was ``"moe_fp8_block_scale" in name``,
        which matched only the MoE definitions that happened to exist when it was written:
        a dense block-scaled FP8 GEMM -- what every FP8-quantized non-MoE model is built
        from -- fell through to the strict evaluator, which requires every element within
        tolerance. Such a kernel dequantises to bfloat16 and is then compared against a
        float32 reference, so it cannot pass however correct it is. Measured on Arc B580
        for `gemm_fp8_block128x128_n4096_k2560`: ~90% of elements within (1e-2, 1e-2) and
        ~96% within (2e-2, 2e-2), and applying the scales to a float32 accumulator
        instead -- the exact algorithm -- recovers only 0.4 points, which is what shows
        the residual is the bfloat16 GEMM rather than the dequantisation.

        A relaxed ratio is the point of this evaluator, so widening what reaches it does
        not weaken any check that previously applied: these definitions were not being
        checked leniently before, they were being rejected outright.
        """
        if "moe_fp8_block_scale" in definition.name:
            return True  # the original signal, kept so this is pure widening
        if any(spec.dtype in _LOW_BIT_INPUT_DTYPES for spec in definition.inputs.values()):
            return True
        return any(str(tag).startswith("quantization:") for tag in definition.tags)

    @override
    @classmethod
    def check_correctness(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_outputs: List[List[torch.Tensor]],
        cfg: ResolvedEvalConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]:
        if cfg.required_matched_ratio is None:
            cfg = cfg.model_copy(
                update={"required_matched_ratio": cls.DEFAULT_REQUIRED_MATCHED_RATIO}
            )

        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        min_matched_ratio = 1.0
        is_dps = sol_runnable.metadata.destination_passing_style

        for trial, inp in enumerate(inputs):
            try:
                if is_dps:
                    out = allocate_outputs(definition, inp, device)
                    with torch.no_grad():
                        sol_runnable(*inp, *out)
                    device_synchronize(device)
                else:
                    with torch.no_grad():
                        result = sol_runnable(*inp)
                    device_synchronize(device)
                    out = normalize_result(definition, result, device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            ref_out = ref_outputs[trial]

            for sol_tensor, ref_tensor in zip(out, ref_out):
                if tuple(sol_tensor.shape) != tuple(ref_tensor.shape):
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                if sol_tensor.dtype != ref_tensor.dtype:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_path=log_path
                    )

                non_finite_err_val: Optional[float] = None
                if torch.isinf(sol_tensor).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(sol_tensor).any().item():
                    non_finite_err_val = float("nan")

                if non_finite_err_val is not None:
                    correctness = Correctness(
                        max_relative_error=non_finite_err_val, max_absolute_error=non_finite_err_val
                    )
                    return correctness, make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_path=log_path,
                        correctness=correctness,
                    )

                abs_err, rel_err, exceeds_tol, matched_ratio = compute_error_stats(
                    sol_tensor, ref_tensor, cfg
                )

                if exceeds_tol:
                    numerical_incorrect = True

                min_matched_ratio = min(min_matched_ratio, matched_ratio)
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(
            max_relative_error=max_rel,
            max_absolute_error=max_abs,
            extra={"matched_ratio": min_matched_ratio},
        )

        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None
