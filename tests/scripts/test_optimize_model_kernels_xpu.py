"""The tuning space comes from the device, never from a table."""

import importlib.util
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "optimize_model_kernels_xpu",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "optimize_model_kernels_xpu.py",
)
om = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = om  # dataclasses resolve a class's module through sys.modules
_SPEC.loader.exec_module(om)


class TestSweepSpace:
    def test_derived_from_the_reported_limits(self):
        wg, sg = om.sweep_space({"max_work_group_size": 1024, "sub_group_sizes": [32, 16]}, "xpu:0")
        assert wg == [16, 32, 64, 128, 256, 512, 1024]
        assert sg == [16, 32]

    def test_a_non_power_of_two_maximum_is_included_as_itself(self):
        wg, _ = om.sweep_space({"max_work_group_size": 768, "sub_group_sizes": [8]}, "xpu:0")
        assert wg == [8, 16, 32, 64, 128, 256, 512, 768]

    def test_missing_limits_fail_loudly_naming_the_device(self):
        with pytest.raises(SystemExit, match="xpu:1 did not report"):
            om.sweep_space({"architecture": "x"}, "xpu:1")

    def test_degenerate_limits_fail_loudly(self):
        with pytest.raises(SystemExit):
            om.sweep_space({"max_work_group_size": 0, "sub_group_sizes": [16]}, "xpu:0")
        with pytest.raises(SystemExit):
            om.sweep_space({"max_work_group_size": 256, "sub_group_sizes": []}, "xpu:0")

    def test_no_literal_table_remains(self):
        assert not hasattr(om, "WORK_GROUP_SIZES") and not hasattr(om, "SUB_GROUP_SIZES")


class TestViableCandidates:
    def _patch(self, monkeypatch, extra):
        import flashinfer_bench.device as device

        class _Acc:
            def capabilities(self, _device):
                return type("Caps", (), {"extra": extra})()

        monkeypatch.setattr(device, "get_accelerator", lambda _d: _Acc())

    def test_every_candidate_is_launchable_on_the_device(self, monkeypatch):
        self._patch(monkeypatch, {"max_work_group_size": 512, "sub_group_sizes": [16, 32]})
        candidates = om.viable_candidates("xpu:0")
        assert candidates
        for c in candidates:
            assert (
                c.work_group <= 512 and c.sub_group in (16, 32) and c.work_group % c.sub_group == 0
            )
        assert {c.work_group for c in candidates} == {16, 32, 64, 128, 256, 512}

    def test_device_without_limits_does_not_fall_back_to_literals(self, monkeypatch):
        self._patch(monkeypatch, {})
        with pytest.raises(SystemExit, match="did not report"):
            om.viable_candidates("xpu:0")
