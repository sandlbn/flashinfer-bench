"""Evaluator registry."""

from __future__ import annotations

from typing import List, Type

from flashinfer_bench.data import Definition

from .default import DefaultEvaluator
from .dsa_sparse_attention import DsaSparseAttentionEvaluator
from .dsa_topk_indexer import DsaTopkIndexerEvaluator
from .evaluator import Evaluator
from .lowbit import LowBitEvaluator
from .sampling import SamplingEvaluator

EvaluatorType = Type[Evaluator]

_EVALUATORS: List[EvaluatorType] = [
    SamplingEvaluator,
    LowBitEvaluator,
    DsaSparseAttentionEvaluator,
    DsaTopkIndexerEvaluator,
]
_DEFAULT_EVALUATOR: EvaluatorType = DefaultEvaluator


_TOLERANCE_POLICY_EVALUATORS: frozenset = frozenset({LowBitEvaluator})
"""Evaluators that only relax tolerance, and so lose to one that knows the operation.

``LowBitEvaluator`` claims any definition with low-bit inputs, which now includes
``dsa_topk_indexer_fp8_*`` -- a definition another evaluator understands the *semantics*
of. Both matching is not ambiguity to report, it is a policy question with an obvious
answer: keep the evaluator that knows what the operation computes.
"""


def resolve_evaluator(definition: Definition) -> EvaluatorType:
    matches = [cls for cls in _EVALUATORS if cls.can_evaluate(definition)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return _DEFAULT_EVALUATOR
    specific = [cls for cls in matches if cls not in _TOLERANCE_POLICY_EVALUATORS]
    if len(specific) == 1:
        return specific[0]
    raise ValueError(f"Multiple evaluator matches for definition '{definition.name}'")
