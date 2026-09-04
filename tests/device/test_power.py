"""Tests for detecting host power settings that would distort measurements.

Host throttling does not appear anywhere in a trace, so it has to be caught up front or
it is discovered much later, as unexplained variance in the results.
"""

from pathlib import Path

import pytest

from flashinfer_bench.device import power


class TestPlatformProfile:
    def test_reads_the_reported_profile(self, tmp_path, monkeypatch):
        profile_file = tmp_path / "platform_profile"
        profile_file.write_text("performance\n")
        monkeypatch.setattr(power, "_PLATFORM_PROFILE", profile_file)
        assert power.platform_profile() == "performance"

    def test_absent_file_is_not_an_error(self, tmp_path, monkeypatch):
        """Containers and non-Linux hosts simply do not expose this."""
        monkeypatch.setattr(power, "_PLATFORM_PROFILE", tmp_path / "missing")
        assert power.platform_profile() is None

    def test_empty_file_reads_as_unknown(self, tmp_path, monkeypatch):
        profile_file = tmp_path / "platform_profile"
        profile_file.write_text("\n")
        monkeypatch.setattr(power, "_PLATFORM_PROFILE", profile_file)
        assert power.platform_profile() is None


class TestMeasurementWarnings:
    @pytest.fixture(autouse=True)
    def _no_battery_noise(self, monkeypatch):
        monkeypatch.setattr(power, "on_battery", lambda: False)

    def _with_profile(self, monkeypatch, tmp_path: Path, value: str):
        profile_file = tmp_path / "platform_profile"
        profile_file.write_text(value)
        monkeypatch.setattr(power, "_PLATFORM_PROFILE", profile_file)

    @pytest.mark.parametrize("profile", ["low-power", "quiet", "cool", "power-saver"])
    def test_throttling_profiles_are_reported(self, profile, monkeypatch, tmp_path):
        self._with_profile(monkeypatch, tmp_path, profile)
        warnings = power.measurement_warnings()
        assert len(warnings) == 1
        assert profile in warnings[0]

    def test_performance_profile_is_silent(self, monkeypatch, tmp_path):
        self._with_profile(monkeypatch, tmp_path, "performance")
        assert power.measurement_warnings() == []

    def test_unknown_profile_is_silent(self, monkeypatch, tmp_path):
        """Only flag what is known to throttle; do not nag about unfamiliar values."""
        self._with_profile(monkeypatch, tmp_path, "custom-vendor-mode")
        assert power.measurement_warnings() == []

    def test_missing_profile_is_silent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(power, "_PLATFORM_PROFILE", tmp_path / "missing")
        assert power.measurement_warnings() == []

    def test_battery_is_reported_separately(self, monkeypatch, tmp_path):
        self._with_profile(monkeypatch, tmp_path, "performance")
        monkeypatch.setattr(power, "on_battery", lambda: True)
        warnings = power.measurement_warnings()
        assert len(warnings) == 1 and "battery" in warnings[0]

    def test_both_conditions_are_reported(self, monkeypatch, tmp_path):
        self._with_profile(monkeypatch, tmp_path, "low-power")
        monkeypatch.setattr(power, "on_battery", lambda: True)
        assert len(power.measurement_warnings()) == 2


class TestOnBattery:
    def test_undetectable_ac_state_is_none(self, monkeypatch):
        monkeypatch.setattr(power.glob, "glob", lambda pattern: [])
        assert power.on_battery() is None

    def test_no_warning_when_ac_state_is_unknown(self, monkeypatch, tmp_path):
        """An unknown state must not produce a spurious 'on battery' warning."""
        monkeypatch.setattr(power, "_PLATFORM_PROFILE", tmp_path / "missing")
        monkeypatch.setattr(power.glob, "glob", lambda pattern: [])
        assert power.measurement_warnings() == []
