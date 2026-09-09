"""Tests for the trial loop's gates.

What matters here is what the loop refuses: a regression it will not finalize, a spill it
will not report as absent, a cross-process comparison it will not make over different
inputs. The timing arithmetic itself is a median; the gates are where a loop goes wrong.
"""

import importlib.util
import json
import logging
import math
import os
import pathlib
import statistics
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


# Does WORK matmuls per call; one file per amount of work makes an unambiguous slow or fast arm.
# Single-threaded: torch's default intra-op pool spans every core, and on a shared box a
# descheduled pool thread stalls a whole round a hundredfold, which is contention, not the
# arm being timed.
WORK = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)


    class Model(nn.Module):
        def forward(self, x, w):
            out = x @ w
            for _ in range(WORK - 1):
                out = out + (x @ w) * 0.0
            return out


    def get_inputs():
        return [torch.randn(96, 96), torch.randn(96, 96)]


    def get_init_inputs():
        return []
    """
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _result(speedup, spread, kind=kt.PAIRED_SPREAD_KIND):
    return {
        "correctness": "pass",
        "speedup": speedup,
        "spread": spread,
        "spread_kind": kind,
        "max_abs_error": 0.0,
        "baseline_us": 100.0,
        "candidate_us": 100.0 / speedup,
    }


def _old_rule(base, cand):
    """The retired statistic: ratio of medians against the candidate's own range.

    Kept here only as the thing the new rule is shown not to do.
    """
    base_us, cand_us = statistics.median(base), statistics.median(cand)
    speedup = base_us / cand_us
    spread = (max(cand) - min(cand)) / cand_us
    if abs(speedup - 1) < spread:
        return "NOISE", spread
    return ("WIN" if speedup > 1 else "LOSS"), spread


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

    def test_a_slowdown_is_judged_as_hard_as_the_same_speedup(self):
        """`speedup - 1` squeezes every regression into (-1, 0); the factor does not."""
        assert kt._verdict(_result(1.5, 0.20)) == "WIN"
        assert kt._verdict(_result(1 / 1.5, 0.20)) == "LOSS"

    def test_the_threshold_is_noise_sigmas_of_the_spread(self):
        """One sigma let identical arms through one time in seven; the gate asks for more."""
        just_under = (1 + 0.10) ** kt.NOISE_SIGMAS * 0.999
        just_over = (1 + 0.10) ** kt.NOISE_SIGMAS * 1.001
        assert kt._verdict(_result(just_under, 0.10)) == "NOISE"
        assert kt._verdict(_result(just_over, 0.10)) == "WIN"
        assert kt._verdict(_result(1 / just_over, 0.10)) == "LOSS"

    def test_a_result_scored_under_the_one_arm_rule_is_never_a_win(self):
        """Stored series predate the paired statistic; their spread is not the verdict's."""
        assert kt._verdict(_result(3.0, 0.01, kind=None)) == "NOISE"
        legacy = _result(3.0, 0.01)
        del legacy["spread_kind"]
        assert kt._verdict(legacy) == "NOISE"


class TestPairedSpread:
    """The verdict rests on the scatter of the *difference* between the arms."""

    STABLE = [100.0, 100.4, 99.6, 100.2, 99.8, 100.1, 99.9]
    NOISY = [60.0, 140.0, 55.0, 150.0, 110.0, 45.0, 160.0]  # median 110, scatter +-50%

    def test_noisy_baseline_against_stable_candidate_is_noise(self):
        """The observed hole: only the candidate's scatter was read, and it was tiny."""
        old_verdict, old_spread = _old_rule(self.NOISY, self.STABLE)
        assert old_verdict == "WIN" and old_spread < 0.01
        result = kt._summarize(self.NOISY, self.STABLE)
        assert kt._verdict(result) == "NOISE"
        assert result["spread"] > abs(result["speedup"] - 1)

    def test_reported_spread_is_the_paired_one_not_either_arm(self):
        result = kt._summarize(self.NOISY, self.STABLE)
        assert result["spread_kind"] == kt.PAIRED_SPREAD_KIND
        assert result["candidate_spread"] < 0.01
        assert result["baseline_spread"] > 0.2
        assert result["spread"] > result["candidate_spread"]
        assert result["pairs"] == 7

    def test_speedup_is_the_median_of_the_per_round_ratios(self):
        base = [100.0, 200.0, 300.0, 400.0, 500.0]
        cand = [b / 2 for b in base]
        result = kt._summarize(base, cand)
        assert result["speedup"] == pytest.approx(2.0)
        assert result["spread"] == pytest.approx(0.0)
        assert kt._verdict(result) == "WIN"

    JITTER = [1.0, 1.04, 0.97, 1.03, 0.98, 1.05, 0.96]

    def test_regression_outside_the_paired_scatter_is_a_loss(self):
        cand = [3 * b * j for b, j in zip(self.STABLE, self.JITTER)]
        result = kt._summarize(self.STABLE, cand)
        assert 0 < result["spread"] < 0.1
        assert kt._verdict(result) == "LOSS"
        assert result["speedup"] == pytest.approx(1 / 3, rel=0.05)

    def test_gain_outside_the_paired_scatter_is_a_win(self):
        cand = [b / 3 * j for b, j in zip(self.STABLE, self.JITTER)]
        result = kt._summarize(self.STABLE, cand)
        assert kt._verdict(result) == "WIN"
        assert result["speedup"] == pytest.approx(3.0, rel=0.05)

    def test_noise_floor_is_what_the_verdict_compares_against(self):
        result = kt._summarize(self.STABLE, [b * j for b, j in zip(self.STABLE, self.JITTER)])
        assert result["noise_floor"] == pytest.approx((1 + result["spread"]) ** kt.NOISE_SIGMAS - 1)
        assert kt._verdict(result) == "NOISE"

    def test_gross_regression_under_heavy_scatter_is_still_a_loss(self):
        """A noisy pair must not excuse a 5x slowdown; that is what the rule is for."""
        cand = [5 * b for b in self.NOISY]
        jitter = [1.0, 1.4, 0.7, 1.3, 0.8, 1.5, 0.6]
        base = [b * j for b, j in zip(self.NOISY, jitter)]
        result = kt._summarize(base, cand)
        assert result["spread"] > 0.2
        assert kt._verdict(result) == "LOSS"

    def test_identical_arms_are_noise_even_when_every_round_is_equal(self):
        result = kt._summarize(self.STABLE, self.STABLE)
        assert result["speedup"] == 1.0 and result["spread"] == 0.0
        assert kt._verdict(result) == "NOISE"

    def test_too_few_or_unpaired_rounds_are_refused(self):
        with pytest.raises(ValueError, match="paired round"):
            kt._summarize([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="not paired"):
            kt._summarize([1.0] * 6, [1.0] * 5)

    def test_cli_refuses_too_few_rounds(self):
        parser = kt.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark", "s", "c.py", "--rounds", "4"])
        floor = kt.MIN_PAIRED_ROUNDS
        assert parser.parse_args(["benchmark", "s", "c.py", "--rounds", str(floor)]).rounds == floor
        assert parser.parse_args(["benchmark", "s", "c.py"]).rounds > floor


class TestRoundsStability:
    """The spread estimates one quantity whatever `--rounds` is; a range did not.

    Samples are the quantiles of a fixed log-normal scatter (sigma 5%) around a fixed true
    gain, so the only thing that changes between cases is how many rounds were taken.
    """

    SIGMA, GAIN = 0.05, 0.15
    COUNTS = (5, 7, 15, 31, 101)

    def _arms(self, n):
        z = [statistics.NormalDist().inv_cdf((k + 0.5) / n) for k in range(n)]
        base = [100.0] * n
        cand = [100.0 * math.exp(-(self.GAIN + self.SIGMA * zk)) for zk in z]
        return base, cand

    def test_paired_spread_does_not_grow_with_rounds(self):
        spreads = [kt._summarize(*self._arms(n))["spread"] for n in self.COUNTS]
        true = math.exp(self.SIGMA) - 1
        # A 5- or 7-point MAD is granular, so a small count lands within a quarter of the
        # true sigma rather than on it; what it never does is drift with the count.
        assert all(abs(s / true - 1) < 0.25 for s in spreads), spreads
        assert spreads[-1] == pytest.approx(true, rel=0.05)
        assert spreads != sorted(spreads)

    def test_verdict_holds_across_rounds_where_the_range_rule_flipped(self):
        new = [kt._verdict(kt._summarize(*self._arms(n))) for n in self.COUNTS]
        assert new == ["WIN"] * len(self.COUNTS)
        old_verdicts, old_spreads = zip(*(_old_rule(*self._arms(n)) for n in self.COUNTS))
        assert old_spreads == tuple(sorted(old_spreads)) and old_spreads[-1] > 1.5 * old_spreads[0]
        assert len(set(old_verdicts)) > 1


class TestBenchmarkVerdicts:
    """End to end on CPU: identical arms are NOISE, 4x the work is LOSS, a quarter is WIN."""

    def test_identical_arms_are_noise(self, tmp_path):
        base = _write(tmp_path, "base.py", WORK.replace("WORK", "1"))
        cand = _write(tmp_path, "cand.py", WORK.replace("WORK", "1"))
        result = kt.benchmark(base, cand, rounds=21, calls=10, atol=1e-3, rtol=1e-3)
        assert kt._verdict(result) == "NOISE", result

    def test_a_slower_candidate_is_a_loss(self, tmp_path):
        base = _write(tmp_path, "base.py", WORK.replace("WORK", "1"))
        cand = _write(tmp_path, "cand.py", WORK.replace("WORK", "4"))
        result = kt.benchmark(base, cand, rounds=7, calls=10, atol=1e-3, rtol=1e-3)
        assert kt._verdict(result) == "LOSS", result

    def test_a_faster_candidate_is_a_win(self, tmp_path):
        base = _write(tmp_path, "base.py", WORK.replace("WORK", "4"))
        cand = _write(tmp_path, "cand.py", WORK.replace("WORK", "1"))
        result = kt.benchmark(base, cand, rounds=7, calls=10, atol=1e-3, rtol=1e-3)
        assert kt._verdict(result) == "WIN", result

    def test_contract_reports_the_spread_the_verdict_used(self, tmp_path, capsys):
        base = _write(tmp_path, "base.py", WORK.replace("WORK", "1"))
        cand = _write(tmp_path, "cand.py", WORK.replace("WORK", "1"))
        result = kt.benchmark(base, cand, rounds=5, calls=5, atol=1e-3, rtol=1e-3)
        kt._report_result(result, "unknown")
        out = capsys.readouterr().out
        assert f"SPREAD_PCT: {result['spread'] * 100:.1f}" in out
        assert "BASELINE_SPREAD_PCT:" in out and "CANDIDATE_SPREAD_PCT:" in out
        assert f"VERDICT: {kt._verdict(result)}" in out


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
        result = kt.benchmark(base, cand, rounds=5, calls=2, atol=1e-3, rtol=1e-3)
        assert result["correctness"] == "pass"
        assert "build_log" in result and isinstance(result["build_log"], str)
        assert kt._spill_state(result["build_log"]) == "unknown"

    def test_failing_result_carries_build_log_too(self, tmp_path):
        base = _write(tmp_path, "base.py", INPLACE.replace("SHIFT", "0.0"))
        cand = _write(tmp_path, "cand.py", INPLACE.replace("SHIFT", "100.0"))
        result = kt.benchmark(base, cand, rounds=5, calls=2, atol=1e-3, rtol=1e-3)
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
            kt.benchmark(base, cand, 5, 1, 1e-3, 1e-3, build_log=log)
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


def _run_ab(tmp_path, harness, *extra, rounds=10):
    cmd = [
        sys.executable,
        str(SCRIPT),
        "ab",
        harness,
        "--rounds",
        str(rounds),
        "--inner-rounds",
        "2",
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
            "PROCESSES: 20",
            "CORRECT: OK",
        ):
            assert key in out, out
        assert any(f"VERDICT: {v}" in out for v in ("WIN", "LOSS", "NOISE")), out
        assert out.rstrip().endswith("DONE")
        recorded = json.loads((tmp_path / "ab.json").read_text())
        assert recorded["arm_b"]["env"] == {"KT_TEST_EXTRA": "2"}
        assert recorded["arm_a"]["env"] == {}
        assert recorded["processes"] == 20
        # One flipped pair of rounds is one observation, judged by the rule `benchmark` uses.
        assert recorded["spread_kind"] == kt.PAIRED_SPREAD_KIND
        assert recorded["pairs"] == 5
        assert len(recorded["round_medians"]["a"]) == 10
        assert f"NOISE_FLOOR_PCT: {recorded['noise_floor'] * 100:.1f}" in out
        assert f"SPREAD_PCT: {recorded['spread'] * 100:.1f}" in out
        assert recorded["verdict"] == kt._verdict(recorded)

    def test_odd_or_too_few_rounds_are_refused_before_any_launch(self, tmp_path):
        harness = _write(tmp_path, "h.py", HARNESS)
        for rounds in ("8", "11"):
            cmd = [sys.executable, str(SCRIPT), "ab", harness, "--rounds", rounds]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            assert proc.returncode != 0
            assert "even and at least 10" in proc.stderr, proc.stderr
            assert "BUILD" not in proc.stdout

    def test_a_flipped_pair_of_rounds_cancels_the_launch_order(self):
        """Whichever arm launches second reads 30% slower; a real 1.2x sits underneath.

        Paired round by round, the order effect lands in the spread and the gain is NOISE;
        paired as flipped pairs it cancels and the gain is measured.
        """
        rounds = 10
        medians = {"a": [], "b": []}
        for r in range(rounds):
            a, b = 120.0, 100.0
            if r % 2 == 0:
                b *= 1.3
            else:
                a *= 1.3
            medians["a"].append(a)
            medians["b"].append(b)
        assert kt._verdict(kt._summarize(medians["a"], medians["b"])) == "NOISE"
        obs = kt._ab_observations(medians)
        result = kt._summarize(obs["a"], obs["b"])
        assert result["pairs"] == rounds // 2
        assert result["speedup"] == pytest.approx(1.2)
        assert result["spread"] == pytest.approx(0.0)
        assert kt._verdict(result) == "WIN"

    def test_a_wrong_arm_fails_correctness_before_any_timing(self, tmp_path):
        harness = _write(tmp_path, "h.py", HARNESS)
        proc = _run_ab(tmp_path, harness, "--env-b", "KT_TEST_SHIFT=100")
        assert proc.returncode == 1
        out = proc.stdout
        assert "CORRECT: FAILED" in out and "VERDICT: INCORRECT" in out, out
        assert "SPEEDUP:" not in out
        # Refused after the first round, not after every launch was spent.
        assert "PROCESSES:" not in out

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
