"""unitrace discovery: the env var, then PATH, then the documented search, then a clear error."""

import os
import pathlib
import stat

import pytest

from flashinfer_bench.agents import unitrace as ut


def _executable(path: pathlib.Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """No unitrace anywhere: empty PATH, env unset, a search location that does not exist."""
    monkeypatch.delenv(ut.UNITRACE_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ut, "_SEARCH_RELATIVE", (pathlib.Path("nonexistent") / "unitrace",))
    return tmp_path


class TestFindUnitrace:
    def test_env_var_wins_over_path(self, nowhere, monkeypatch):
        on_path = _executable(nowhere / "bin" / "unitrace")
        monkeypatch.setenv("PATH", str(nowhere / "bin"))
        from_env = _executable(nowhere / "build" / "unitrace")
        monkeypatch.setenv(ut.UNITRACE_ENV, from_env)
        assert ut.find_unitrace() == from_env
        monkeypatch.delenv(ut.UNITRACE_ENV)
        assert ut.find_unitrace() == on_path

    def test_explicit_argument_wins_over_env_var(self, nowhere, monkeypatch):
        monkeypatch.setenv(ut.UNITRACE_ENV, _executable(nowhere / "env" / "unitrace"))
        explicit = _executable(nowhere / "arg" / "unitrace")
        assert ut.find_unitrace(explicit) == explicit

    def test_set_but_wrong_env_var_is_an_error_not_a_fallthrough(self, nowhere, monkeypatch):
        _executable(nowhere / "bin" / "unitrace")
        monkeypatch.setenv("PATH", str(nowhere / "bin"))
        monkeypatch.setenv(ut.UNITRACE_ENV, str(nowhere / "missing" / "unitrace"))
        with pytest.raises(FileNotFoundError, match=ut.UNITRACE_ENV):
            ut.find_unitrace()
        assert ut.is_unitrace_available() is False

    def test_documented_search_under_the_working_directory(self, nowhere, monkeypatch):
        monkeypatch.setattr(ut, "_SEARCH_RELATIVE", (pathlib.Path("tmp") / "pti-gpu" / "unitrace",))
        built = _executable(nowhere / "tmp" / "pti-gpu" / "unitrace")
        assert ut.find_unitrace() == built

    def test_nothing_found_is_none_and_the_message_names_the_env_var(self, nowhere):
        assert ut.find_unitrace() is None
        assert ut.is_unitrace_available() is False
        message = ut.unitrace_missing_message()
        assert ut.UNITRACE_ENV in message and "PATH" in message and "pti-gpu" in message

    def test_no_machine_specific_path_is_baked_in(self):
        for rel in ut._SEARCH_RELATIVE:
            assert not rel.is_absolute()
            assert not str(rel).startswith(("/home", "/opt", "/usr"))
            assert os.sep + "home" + os.sep not in str(rel)


class TestRunUnitraceWithoutBinary:
    def test_tool_returns_the_actionable_message_before_touching_anything(self, nowhere):
        result = ut.flashinfer_bench_run_unitrace("not-a-solution", "not-a-workload")
        assert ut.UNITRACE_ENV in result
        assert "pti-gpu" in result

    def test_tool_reports_a_misconfigured_env_var_verbatim(self, nowhere, monkeypatch):
        monkeypatch.setenv(ut.UNITRACE_ENV, str(nowhere / "nope"))
        result = ut.flashinfer_bench_run_unitrace("not-a-solution", "not-a-workload")
        assert "does not name an executable" in result and ut.UNITRACE_ENV in result
