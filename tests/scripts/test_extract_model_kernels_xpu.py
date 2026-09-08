"""Tests for the Intel definition extractor's shape selection."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "extract_model_kernels_xpu.py"


@pytest.fixture(scope="module")
def extractor():
    spec = importlib.util.spec_from_file_location("extract_model_kernels_xpu", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _shape(key):
    """(in_features, out_features) from a (tokens, in, out, dtype) counter key."""
    return (key[1], key[2])


class TestTopShapes:
    def test_token_counts_do_not_compete_with_shapes_for_the_budget(self, extractor):
        """The cap counts distinct projections, not counter entries.

        The counter is keyed on token count as well as shape, so applying
        `most_common(limit)` directly lets one shape seen at several batch sizes consume
        the whole budget. Observed on an FP8 Qwen3, which emitted three of five
        projections and silently lost o_proj and down_proj.
        """
        counter = Counter(
            {
                (1, 2560, 4096, "bf16"): 100,
                (8, 2560, 4096, "bf16"): 90,
                (64, 2560, 4096, "bf16"): 80,
                (1, 2560, 9728, "bf16"): 70,
                (1, 4096, 2560, "bf16"): 60,
            }
        )
        kept = extractor._top_shapes(counter, _shape, 3)
        assert set(kept) == {(2560, 4096), (2560, 9728), (4096, 2560)}

    def test_keeps_every_token_count_for_a_shape_it_keeps(self, extractor):
        counter = Counter(
            {
                (1, 2560, 4096, "bf16"): 100,
                (8, 2560, 4096, "bf16"): 90,
                (64, 2560, 4096, "bf16"): 80,
            }
        )
        kept = extractor._top_shapes(counter, _shape, 1)
        assert sorted(kept[(2560, 4096)]) == [1, 8, 64]

    def test_ranks_shapes_by_total_calls_across_token_counts(self, extractor):
        """A shape called often at many batch sizes outranks one called once."""
        counter = Counter(
            {
                (1, 10, 20, "bf16"): 5,
                (2, 10, 20, "bf16"): 5,
                (4, 10, 20, "bf16"): 5,  # 15 total
                (1, 30, 40, "bf16"): 9,  # 9 total
            }
        )
        assert set(extractor._top_shapes(counter, _shape, 1)) == {(10, 20)}

    def test_limit_above_the_number_of_shapes_keeps_all(self, extractor):
        counter = Counter({(1, 10, 20, "bf16"): 1, (1, 30, 40, "bf16"): 1})
        assert len(extractor._top_shapes(counter, _shape, 99)) == 2
