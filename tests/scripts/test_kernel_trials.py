"""Tests for the trial loop's gates.

What matters here is what the loop refuses: a regression it will not finalize, a spill it
will not report as absent, a cross-process comparison it will not make over different
inputs. The timing arithmetic itself is a median; the gates are where a loop goes wrong.
"""

import importlib.util
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "kernel_trials.py"
_SPEC = importlib.util.spec_from_file_location("kernel_trials", SCRIPT)
kt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kt)

# A CPU harness whose behaviour is steered by the environment, so one file can be either arm
# of a cross-process A/B: KT_TEST_EXTRA adds redundant work, KT_TEST_SHIFT makes it wrong.
HARNESS = textwrap.dedent(
    """
    import os

    import torch
    import torch.nn as nn


    class Model(nn.Module):
        def forward(self, x, w):
            out = x @ w
            for _ in range(int(os.environ.get("KT_TEST_EXTRA", "0"))):
                out = out + (x @ w) * 0.0
            return out + float(os.environ.get("KT_TEST_SHIFT", "0"))


    def get_inputs():
        return [torch.randn(32, 32), torch.randn(32, 32)]


    def get_init_inputs():
        return []
    """
)

# In-place: writes into its first argument and returns nothing, like the kernels worth tuning.
INPLACE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn


    class Model(nn.Module):
        def forward(self, out, x, w):
            out.copy_(x @ w + SHIFT)


    def get_inputs():
        return [torch.empty(8, 4), torch.randn(8, 16), torch.randn(16, 4)]


    def get_init_inputs():
        return []
    """
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _result(speedup, spread):
    return {
        "correctness": "pass",
        "speedup": speedup,
        "spread": spread,
        "max_abs_error": 0.0,
        "baseline_us": 100.0,
        "candidate_us": 100.0 / speedup,
    }


class TestVerdict:
    def test_gain_outside_spread_is_a_win(self):
        assert kt._verdict(_result(1.20, 0.05)) == "WIN"

    def test_regression_outside_spread_is_a_loss(self):
        assert kt._verdict(_result(0.80, 0.05)) == "LOSS"

    def test_difference_inside_spread_is_noise_in_either_direction(self):
        assert kt._verdict(_result(1.02, 0.05)) == "NOISE"
        assert kt._verdict(_result(0.98, 0.05)) == "NOISE"

    def test_gross_regression_is_never_noise(self):
        """The signed comparison once reported a 5x slowdown as inside noise."""
        assert kt._verdict(_result(0.20, 0.50)) == "LOSS"


class TestFinalize:
    @pytest.fixture
    def series(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kt, "TRIALS_DIR", tmp_path / "trials")
        candidate = _write(tmp_path, "cand.py", HARNESS)

        def make(speedup, spread):
            kt._save_store(
                "s",
                {
                    "name": "s",
                    "baseline": candidate,
                    "trials": [
                        {
                            "id": "t0",
                            "file": candidate,
                            "parent": None,
                            "strategy": "",
                            "result": _result(speedup, spread),
                        }
                    ],
                },
            )
            return tmp_path / "out" / "final.py"

        return make

    def _finalize(self, output, require_win):
        kt.cmd_finalize(
            types.SimpleNamespace(name="s", output=str(output), require_win=require_win)
        )

    def test_default_refuses_a_regression_and_writes_nothing(self, series, capsys):
        output = series(speedup=0.9, spread=0.01)
        with pytest.raises(SystemExit) as exc:
            self._finalize(output, require_win=True)
        assert exc.value.code == 1
        assert not output.exists()
        out = capsys.readouterr().out
        assert "VERDICT: LOSS" in out
        assert "FINALIZE: REFUSED" in out
        assert out.rstrip().endswith("DONE")

    def test_default_refuses_a_gain_inside_the_spread(self, series, capsys):
        """1.02x with 5% scatter was never measured; copying it ships an unmeasured kernel."""
        output = series(speedup=1.02, spread=0.05)
        with pytest.raises(SystemExit):
            self._finalize(output, require_win=True)
        assert not output.exists()
        assert "VERDICT: NOISE" in capsys.readouterr().out

    def test_default_copies_a_measured_win(self, series, capsys):
        output = series(speedup=1.3, spread=0.02)
        self._finalize(output, require_win=True)
        assert output.exists()
        assert "FINALIZE: OK" in capsys.readouterr().out

    def test_opt_out_copies_a_regression_but_still_names_the_verdict(self, series, capsys):
        output = series(speedup=0.9, spread=0.01)
        self._finalize(output, require_win=False)
        assert output.exists()
        out = capsys.readouterr().out
        assert "VERDICT: LOSS" in out and "FINALIZE: OK" in out

    def test_require_win_is_on_by_default_in_the_cli(self):
        parser = kt.build_parser()
        assert parser.parse_args(["finalize", "s", "o.py"]).require_win is True
        assert parser.parse_args(["finalize", "s", "o.py", "--no-require-win"]).require_win is False

    def test_opt_out_help_says_what_it_is_for(self):
        text = kt.build_parser()._subparsers._group_actions[0].choices["finalize"].format_help()
        assert "record-keeping" in text
        assert "never" in text


class TestSpillState:
    NINJA_AOT = (
        "ninja build stdout:\n[1/2] /opt/x/icpx -fsycl -O3 -fsycl-targets=spir64_gen -c k.cpp\n"
        "[2/2] /opt/x/icpx cpp_0.o -shared -fsycl-targets=spir64_gen -Xs -device -Xs bmg -o k.so\n"
    )

    def test_reported_spill_is_the_number(self):
        text = (
            self.NINJA_AOT
            + "warning: kernel foo compiled SIMD16 allocated 128 regs and spilled around 172\n"
        )
        assert kt._spill_state(text) == 172

    def test_unitrace_spelling_is_read_too(self):
        assert kt._spill_state("Spill Memory Per Thread: 96") == 96
        assert kt._spill_state("Spill Memory Per Thread: 0") == "none"

    def test_aot_compile_that_said_nothing_is_none(self):
        assert kt._spill_state(self.NINJA_AOT) == "none"

    def test_nothing_captured_is_unknown_not_none(self):
        """A plain PyTorch harness compiles nothing; 'none' there certifies an unchecked kernel."""
        assert kt._spill_state("") == "unknown"
        assert kt._spill_state(None) == "unknown"
        assert kt._spill_state("some torch warning about something else") == "unknown"

    def test_cached_build_is_unknown(self):
        """ninja found nothing to do, so the compiler that reports spill never ran."""
        assert kt._spill_state("ninja build stdout:\nninja: no work to do.\n") == "unknown"

    def test_spirv_jit_build_is_unknown(self):
        """Without an AOT target, registers are allocated at first launch, silently."""
        jit = "ninja build stdout:\n[1/2] /opt/x/icpx -fsycl -O3 -c k.cpp\n[2/2] /opt/x/icpx -shared\n"
        assert kt._spill_state(jit) == "unknown"

    def test_trim_keeps_spill_lines_that_truncation_would_drop(self):
        parts = ["spilled around 300\n" + "x" * 5000]
        trimmed = kt._trim_log(parts, keep=1000)
        assert len(trimmed) < 1500
        assert kt._spill_state(trimmed) == 300


class TestCaptureBuildOutput:
    def test_collects_fd_writes_and_the_build_logger(self, monkeypatch):
        monkeypatch.delenv(kt._BUILD_LOG_ENV, raising=False)
        build_logger = logging.getLogger(kt._BUILD_LOGGER)
        handlers_before = list(build_logger.handlers)
        log = []
        with kt._capture_build_output(log):
            assert os.environ[kt._BUILD_LOG_ENV] == "1"
            # Compilers and drivers write to the descriptors, not to sys.stdout; under
            # pytest's capture sys.stdout is not fd 1, so only the descriptor path is asserted.
            os.write(2, b"native: spilled around 42\n")
            build_logger.info("ninja build stderr:\n%s", "[1/2] icpx ... spir64_gen")
        text = "\n".join(log)
        assert "spilled around 42" in text
        assert "[1/2] icpx" in text
        assert kt._spill_state(text) == 42
        # Everything it touched is put back.
        assert kt._BUILD_LOG_ENV not in os.environ
        assert build_logger.handlers == handlers_before

    def test_restores_descriptors_when_the_body_raises(self, capfd):
        log = []
        with pytest.raises(RuntimeError):
            with kt._capture_build_output(log):
                os.write(1, b"inside\n")
                raise RuntimeError("boom")
        print("after")
        assert "inside" in "".join(log)
        assert "after" in capfd.readouterr().out


class TestBenchmarkResult:
    """The result dict must carry what the spill gate reads.

    The gate once read a key nothing wrote, and a dict lookup that always misses reports
    'none' forever without a single test failing. Asserting on the key is the point.
    """

    def test_passing_result_carries_build_log(self, tmp_path):
        base = _write(tmp_path, "base.py", HARNESS)
        cand = _write(tmp_path, "cand.py", HARNESS)
        result = kt.benchmark(base, cand, rounds=1, calls=2, atol=1e-3, rtol=1e-3)
        assert result["correctness"] == "pass"
        assert "build_log" in result and isinstance(result["build_log"], str)
        assert kt._spill_state(result["build_log"]) == "unknown"

    def test_failing_result_carries_build_log_too(self, tmp_path):
        base = _write(tmp_path, "base.py", INPLACE.replace("SHIFT", "0.0"))
        cand = _write(tmp_path, "cand.py", INPLACE.replace("SHIFT", "100.0"))
        result = kt.benchmark(base, cand, rounds=1, calls=2, atol=1e-3, rtol=1e-3)
        assert result["correctness"] == "fail"
        assert "argument 0 differs" in result["reason"]
        assert "build_log" in result

    def test_caller_keeps_the_log_when_the_load_raises(self, tmp_path):
        base = _write(tmp_path, "base.py", HARNESS)
        cand = _write(
            tmp_path,
            "cand.py",
            "import os\nos.write(2, b'spilled around 9')\nraise RuntimeError('x')\n",
        )
        log = []
        with pytest.raises(RuntimeError):
            kt.benchmark(base, cand, 1, 1, 1e-3, 1e-3, build_log=log)
        assert kt._spill_from("\n".join(log)) == 9


class TestCompare:
    def test_in_place_arguments_of_unequal_sizes_compare_pairwise(self):
        import torch

        a = {
            "in_args": True,
            "kind": "NoneType",
            "returned": None,
            "args": [torch.ones(4), torch.ones(2, 3)],
        }
        b = {
            "in_args": True,
            "kind": "NoneType",
            "returned": None,
            "args": [torch.ones(4), torch.ones(2, 3)],
        }
        assert kt._compare(a, b, 1e-3, 1e-3) == (None, 0.0)
        b["args"][1][0, 0] = 5.0
        reason, err = kt._compare(a, b, 1e-3, 1e-3)
        assert reason.startswith("argument 1 differs") and err == 4.0

    def test_returned_type_mismatch_is_a_failure_not_a_crash(self):
        import torch

        a = {"in_args": False, "kind": "tensor", "returned": torch.ones(3), "args": []}
        b = {"in_args": False, "kind": "sequence", "returned": torch.ones(3), "args": []}
        assert kt._compare(a, b, 1e-3, 1e-3)[0] == "candidate returned a different type"


def _run_ab(tmp_path, harness, *extra):
    cmd = [
        sys.executable,
        str(SCRIPT),
        "ab",
        harness,
        "--rounds",
        "1",
        "--inner-rounds",
        "1",
        "--calls",
        "3",
        "--json",
        str(tmp_path / "ab.json"),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


class TestAb:
    def test_arms_run_in_separate_processes_and_report_the_contract(self, tmp_path):
        harness = _write(tmp_path, "h.py", HARNESS)
        proc = _run_ab(tmp_path, harness, "--env-b", "KT_TEST_EXTRA=2")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        out = proc.stdout
        for key in (
            "BUILD: OK",
            "SPILLS: unknown",
            "INPUTS: IDENTICAL",
            "PROCESSES: 2",
            "CORRECT: OK",
        ):
            assert key in out, out
        assert any(f"VERDICT: {v}" in out for v in ("WIN", "LOSS", "NOISE")), out
        assert out.rstrip().endswith("DONE")
        recorded = json.loads((tmp_path / "ab.json").read_text())
        assert recorded["arm_b"]["env"] == {"KT_TEST_EXTRA": "2"}
        assert recorded["arm_a"]["env"] == {}
        assert recorded["processes"] == 2

    def test_a_wrong_arm_fails_correctness_before_any_timing(self, tmp_path):
        harness = _write(tmp_path, "h.py", HARNESS)
        proc = _run_ab(tmp_path, harness, "--env-b", "KT_TEST_SHIFT=100")
        assert proc.returncode == 1
        out = proc.stdout
        assert "CORRECT: FAILED" in out and "VERDICT: INCORRECT" in out, out
        assert "SPEEDUP:" not in out

    def test_an_arm_that_cannot_load_is_a_build_failure_naming_the_arm(self, tmp_path):
        harness = _write(tmp_path, "h.py", HARNESS)
        broken = _write(tmp_path, "b.py", "x = 1\n")
        proc = _run_ab(tmp_path, harness, "--harness-b", broken)
        assert proc.returncode == 1
        assert "BUILD: FAILED" in proc.stdout and "ARM: B" in proc.stdout
        assert "defines no Model" in proc.stdout

    def test_env_flags_must_be_key_value(self):
        with pytest.raises(SystemExit, match="KEY=VALUE"):
            kt._parse_env(["NOEQUALS"], "--env-b")
