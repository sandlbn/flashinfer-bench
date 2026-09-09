"""Tests for the serving A/B harness.

The harness's job is to refuse to report a comparison it cannot justify, so what is worth
testing is the parsing that decides whether a run counted -- not the throughput arithmetic.
"""

import importlib.util
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_serving_win",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "measure_serving_win.py",
)
msw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(msw)


class TestParse:
    def test_reads_throughput_and_counters_through_vllm_worker_prefix(self):
        """vLLM tags worker output with ``(EngineCore pid=N) `` -- a space inside the parens.

        A parser anchored past that prefix finds no counters and the harness then reports
        the run as having no validity gate, discarding a measurement that was in fact fine.
        """
        out = "\n".join(
            [
                'FIB_RESULT {"generated_tokens": 2048, "seconds": 1.0, "tokens_per_sec": 2048.0}',
                "(EngineCore pid=596906) [flashinfer-bench] adapter dispatch:",
                "(EngineCore pid=596906)   rmsnorm: 8151 call(s), 143 applied (1.8%)",
                "(EngineCore pid=596906)   silu_and_mul: 4004 call(s), 4004 applied (100.0%)",
            ]
        )
        parsed = msw._parse(out)
        assert parsed["tokens_per_sec"] == 2048.0
        assert parsed["generated_tokens"] == 2048
        assert parsed["dispatch"] == {
            "rmsnorm": (8151, 143),
            "silu_and_mul": (4004, 4004),
        }

    def test_unprefixed_counters_parse_too(self):
        parsed = msw._parse("  rmsnorm: 10 call(s), 10 applied (100.0%)")
        assert parsed["dispatch"] == {"rmsnorm": (10, 10)}

    def test_run_without_counters_is_flagged_not_silently_accepted(self):
        parsed = msw._parse(
            'FIB_RESULT {"generated_tokens": 8, "seconds": 1.0, "tokens_per_sec": 8.0}'
        )
        assert parsed["tokens_per_sec"] == 8.0
        assert parsed["dispatch"] == {}  # caller must refuse to interpret this

    def test_failed_arm_reports_no_throughput(self):
        assert msw._parse("Engine core initialization failed")["tokens_per_sec"] is None


class TestSupported:
    def test_absent_vllm_is_fatal_not_undeterminable(self, monkeypatch):
        """A missing vLLM means this interpreter cannot run the harness at all.

        Returning None there would let the sweep report every model as "could not
        determine" while the real cause is a one-line environment mistake.
        """
        import builtins

        real_import = builtins.__import__

        def _no_vllm(name, *a, **kw):
            if name.startswith("vllm"):
                raise ImportError("No module named 'vllm'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_vllm)
        with pytest.raises(SystemExit, match="not importable"):
            msw._supported("some/model")


class TestRootCause:
    def test_reports_the_worker_error_not_vllms_generic_wrapper(self):
        """vLLM's top-level message never names the cause, and a tail shows only it.

        The real reason sits far earlier in the worker's traceback, so a failure reported
        from the last few lines is the same uninformative string for every model.
        """
        output = "\n".join(
            [
                "INFO loading model",
                "(EngineCore pid=1) torch.OutOfMemoryError: XPU out of memory. Tried to "
                "allocate 2.00 GiB",
                'File "/x/vllm/engine.py", line 12, in run',
                "    raise RuntimeError(...)",
                "RuntimeError: Engine core initialization failed. See root cause above. "
                "Failed core proc(s): {}",
            ]
        )
        cause = msw._root_cause(output)
        assert "out of memory" in cause
        assert "See root cause above" not in cause

    def test_deduplicates_and_strips_the_worker_prefix(self):
        """The cause is the exception, not the log line that carried it."""
        line = "(EngineCore pid=1) ValueError: unsupported quantization"
        assert msw._root_cause("\n".join([line] * 5)) == "ValueError: unsupported quantization"

    def test_traceback_frames_do_not_outrank_the_exception(self):
        """Frames come after the exception in the text; taking the last match picked them.

        The reported cause was then a random line of someone else's source.
        """
        output = "\n".join(
            [
                "(EngineCore pid=1) ERROR [core.py:1385] Traceback (most recent call last):",
                '(EngineCore pid=1) ERROR [core.py:1385]   File "/x/runnable.py", line 95',
                "(EngineCore pid=1) ERROR [core.py:1385]     ret = self._callable(*args)",
                "(EngineCore pid=1) ERROR [core.py:1385]           ^^^^^^^^^^^^^^^^^^^^^",
                "(EngineCore pid=1) RuntimeError: CUDA is not available; Triton kernel "
                "requires CUDA.",
                "(EngineCore pid=1)   rmsnorm: 293 call(s), 5 applied (1.7%)",
                "(EngineCore pid=1)   detail: {'rmsnorm unsupported 3d': 288}",
            ]
        )
        assert msw._root_cause(output) == (
            "RuntimeError: CUDA is not available; Triton kernel requires CUDA."
        )

    def test_no_recognisable_error_yields_empty_so_caller_falls_back(self):
        assert msw._root_cause("INFO all good\nINFO done") == ""


class TestParseGluedOutput:
    def test_result_glued_to_a_newline_less_warning_is_still_read(self):
        """Dependencies print warnings without a trailing newline.

        The result line then arrives with that warning fused to its front, and a parser
        anchored at the start of the line throws away a run that in fact succeeded --
        reporting a working arm as a failure.
        """
        line = (
            "Warning: Triton requires a CUDA-enabled GPU. Falling back to reference "
            'implementation on CPU.FIB_RESULT {"generated_tokens": 32768, '
            '"seconds": 10.9344, "tokens_per_sec": 2996.77}'
        )
        parsed = msw._parse(line)
        assert parsed["tokens_per_sec"] == 2996.77
        assert parsed["generated_tokens"] == 32768


class TestMedian:
    def test_odd_and_even_lengths(self):
        assert msw._median([3.0, 1.0, 2.0]) == 2.0
        assert msw._median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_single_run(self):
        assert msw._median([7.5]) == 7.5


class TestArmEnv:
    """Which FIB_* variables each arm carries decides what the A/B is *of*."""

    def _args(self, **overrides):
        base = {"env": [], "dataset": "ds", "plain_arm": False}
        base.update(overrides)
        return __import__("argparse").Namespace(**base)

    def test_apply_mode_patched_arm_switches_the_integration_on(self):
        env = msw._arm_env({"PATH": "/bin"}, self._args(env=["FIB_X=1"]), patched=True)
        assert env["FIB_VLLM_INTEGRATION"] == "1"
        assert env["FIB_ENABLE_APPLY"] == "1"
        assert env["FIB_DATASET_PATH"].endswith("ds")
        assert env["FIB_X"] == "1"

    def test_apply_mode_baseline_strips_the_switches_it_inherited(self):
        inherited = {"PATH": "/bin", "FIB_VLLM_INTEGRATION": "1", "FIB_ENABLE_APPLY": "1"}
        env = msw._arm_env(inherited, self._args(env=["FIB_X=1"]), patched=False)
        assert "FIB_VLLM_INTEGRATION" not in env and "FIB_ENABLE_APPLY" not in env
        assert "FIB_X" not in env  # --env belongs to the patched arm only

    def test_plain_mode_carries_no_fib_variable_in_either_arm(self):
        """A provider or source patch must be measured with no apply() in the path."""
        inherited = {
            "PATH": "/bin",
            "FIB_VLLM_INTEGRATION": "1",
            "FIB_DATASET_PATH": "/d",
            "FIB_OTHER": "x",
        }
        args = self._args(plain_arm=True, env=["MY_PATCH=1"])
        base = msw._arm_env(inherited, args, patched=False)
        ours = msw._arm_env(inherited, args, patched=True)
        assert not [k for k in base if k.startswith("FIB_")]
        assert not [k for k in ours if k.startswith("FIB_")]

    def test_plain_mode_arms_differ_only_by_env(self):
        inherited = {"PATH": "/bin", "HOME": "/h", "FIB_VLLM_INTEGRATION": "1"}
        args = self._args(plain_arm=True, env=["MY_PATCH=1", "OTHER=2"])
        base = msw._arm_env(inherited, args, patched=False)
        ours = msw._arm_env(inherited, args, patched=True)
        assert {k: ours[k] for k in ours if base.get(k) != ours[k]} == {
            "MY_PATCH": "1",
            "OTHER": "2",
        }
        assert set(base) - set(ours) == set()

    def test_plain_and_overhead_arms_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            msw.build_parser().parse_args(["--model", "m", "--plain-arm", "--overhead-arm"])
        assert msw.build_parser().parse_args(["--model", "m", "--plain-arm"]).plain_arm is True


class TestTokenDigest:
    def test_same_tokens_same_digest_and_order_matters(self):
        a = msw._token_digest([[1, 2, 3], [4, 5]])
        assert a == msw._token_digest([[1, 2, 3], [4, 5]])
        assert a != msw._token_digest([[4, 5], [1, 2, 3]])
        assert a != msw._token_digest([[1, 2, 3], [4, 6]])

    def test_digest_is_read_back_from_the_result_line(self):
        parsed = msw._parse(
            'FIB_RESULT {"generated_tokens": 8, "seconds": 1.0, "tokens_per_sec": 8.0, '
            '"token_digest": "abc123"}'
        )
        assert parsed["token_digest"] == "abc123"

    def test_identity_needs_one_stable_digest_on_each_side(self):
        assert msw._tokens_identical(["a"], ["a"]) is True
        assert msw._tokens_identical(["a"], ["b"]) is False
        assert msw._tokens_identical(["a", "b"], ["a", "b"]) is False  # unstable arms
        assert msw._tokens_identical([], ["a"]) is None


def _run(tok_s, digest="d0", dispatch=None, error=None):
    r = {"tokens_per_sec": tok_s, "dispatch": dispatch or {}, "detail": []}
    if digest is not None:
        r["token_digest"] = digest
    if error:
        r.update(tokens_per_sec=None, error=error, log="tmp/x.fail.log")
    return r


APPLIED = {"rmsnorm": (100, 100)}
ARMS = ["baseline", "ours"]


class TestJudge:
    """The gates, in order, and that no delta survives a failed one."""

    def test_identical_tokens_and_a_substitution_give_a_valid_verdict(self):
        runs = {
            "baseline": [_run(100.0), _run(101.0)],
            "ours": [_run(110.0, dispatch=APPLIED), _run(111.0, dispatch=APPLIED)],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        assert out["valid"] and out["verdict"] == "WIN"
        assert out["delta"] == pytest.approx(110.5 / 100.5 - 1)
        assert out["tokens"] == "IDENTICAL"

    def test_a_delta_inside_the_spread_is_noise_not_a_win(self):
        runs = {
            "baseline": [_run(100.0), _run(110.0)],
            "ours": [_run(104.0, dispatch=APPLIED), _run(106.0, dispatch=APPLIED)],
        }
        assert msw.judge(runs, ARMS, plain_arm=False)["verdict"] == "NOISE"

    def test_differing_digests_halt_and_carry_no_delta(self):
        runs = {
            "baseline": [_run(100.0), _run(100.0)],
            "ours": [_run(61.0, "other", APPLIED), _run(62.0, "other", APPLIED)],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        assert not out["valid"]
        assert out["verdict"] == msw.TOKENS_DIFFER and out["tokens"] == "DIFFER"
        assert out["delta"] is None and out["baseline_tok_s"] is None

    def test_an_arm_unstable_across_its_own_repeats_halts_first(self):
        runs = {
            "baseline": [_run(100.0, "a"), _run(100.0, "b")],
            "ours": [_run(100.0, "a", APPLIED), _run(100.0, "a", APPLIED)],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        assert out["verdict"] == msw.TOKENS_UNSTABLE and "baseline" in out["reason"]

    def test_a_missing_digest_is_not_read_as_agreement(self):
        runs = {
            "baseline": [_run(100.0, None), _run(100.0, None)],
            "ours": [_run(100.0, dispatch=APPLIED), _run(100.0, dispatch=APPLIED)],
        }
        assert msw.judge(runs, ARMS, plain_arm=False)["verdict"] == msw.TOKENS_UNCOMPARED

    def test_a_failed_arm_halts_before_the_token_gate(self):
        runs = {
            "baseline": [_run(100.0), _run(100.0)],
            "ours": [_run(None, error="OutOfMemoryError: x"), _run(None, error="x")],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        assert out["verdict"] == msw.ARM_FAILED and "ours" in out["reason"]
        assert out["tokens"] is None

    def test_apply_mode_without_counters_or_with_nothing_applied_is_not_substituted(self):
        runs = {"baseline": [_run(100.0)] * 2, "ours": [_run(120.0)] * 2}
        out = msw.judge(runs, ARMS, plain_arm=False)
        assert out["verdict"] == msw.NOT_SUBSTITUTED and out["delta"] is None
        none = {"rmsnorm": (100, 0)}
        runs = {"baseline": [_run(100.0)] * 2, "ours": [_run(120.0, dispatch=none)] * 2}
        out = msw.judge(runs, ARMS, plain_arm=False)
        assert out["verdict"] == msw.NOT_SUBSTITUTED and "0 applied" in out["reason"]

    def test_plain_mode_needs_no_counters_but_zero_applied_under_env_is_an_aa(self):
        runs = {"baseline": [_run(100.0)] * 2, "ours": [_run(120.0)] * 2}
        assert msw.judge(runs, ARMS, plain_arm=True, env=True)["valid"]
        none = {"weight_row_pad": (24, 0)}
        runs = {"baseline": [_run(100.0)] * 2, "ours": [_run(120.0, dispatch=none)] * 2}
        assert msw.judge(runs, ARMS, plain_arm=True, env=True)["verdict"] == msw.NOT_SUBSTITUTED
        # No --env: an A/A of the noise floor, and counters that count nothing are fine.
        assert msw.judge(runs, ARMS, plain_arm=True, env=False)["valid"]

    def test_overhead_arm_must_agree_on_tokens_too(self):
        arms = ARMS + ["overhead"]
        runs = {
            "baseline": [_run(100.0)] * 2,
            "ours": [_run(120.0, dispatch=APPLIED)] * 2,
            "overhead": [_run(95.0, "x")] * 2,
        }
        assert msw.judge(runs, arms, plain_arm=False)["verdict"] == msw.TOKENS_DIFFER
        runs["overhead"] = [_run(95.0)] * 2
        out = msw.judge(runs, arms, plain_arm=False)
        assert out["valid"] and out["dispatch_tax"] == pytest.approx(-0.05)


def _args(**over):
    base = {
        "model": "test/model",
        "plain_arm": False,
        "repeats": 2,
        "env": [],
        "report_unvalidated": False,
        "bound": None,
        "mechanism": None,
        "candidate": [],
    }
    base.update(over)
    return __import__("argparse").Namespace(**base)


class TestRender:
    """A failed gate prints the contract and the diagnostics -- and no throughput."""

    def test_digest_mismatch_prints_no_delta_and_no_rate(self, capsys):
        runs = {
            "baseline": [_run(100.0), _run(100.0)],
            "ours": [_run(61.49, "other", APPLIED), _run(61.49, "other", APPLIED)],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        msw.render(out, _args(), ARMS, runs, {"routing": "UNCHECKED"})
        text = capsys.readouterr().out
        assert "VERDICT: TOKENS_DIFFER" in text and "TOKENS: DIFFER" in text
        assert "ROUTING: UNCHECKED" in text
        assert text.rstrip().endswith("DONE")
        assert "DELTA_PCT" not in text and "\n  delta" not in text
        assert "tok/s" not in text and "61.49" not in text and "-38.51" not in text
        # The diagnostics that help fix it are all there.
        assert "digest d0" in text and "digest other" in text
        assert "rmsnorm" in text and "100/100 applied" in text

    def test_unvalidated_override_marks_every_number_and_keeps_the_verdict(self, capsys):
        runs = {
            "baseline": [_run(100.0), _run(100.0)],
            "ours": [_run(61.49, "other", APPLIED), _run(61.49, "other", APPLIED)],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        msw.render(out, _args(report_unvalidated=True), ARMS, runs, {"routing": "UNCHECKED"})
        text = capsys.readouterr().out
        assert "VERDICT: TOKENS_DIFFER" in text
        assert "UNVALIDATED delta     -38.51%" in text
        assert "UNVALIDATED ours" in text and "UNVALIDATED baseline" in text
        assert "not a result" in text
        assert "\n  delta" not in text  # never an unmarked delta line

    def test_valid_result_prints_the_contract_then_the_table(self, capsys):
        runs = {
            "baseline": [_run(100.0), _run(102.0)],
            "ours": [_run(120.0, dispatch=APPLIED), _run(121.0, dispatch=APPLIED)],
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        msw.render(out, _args(), ARMS, runs, {"routing": "UNCHECKED"})
        text = capsys.readouterr().out
        for key in (
            "MODEL: test/model",
            "MODE: apply",
            "BASELINE_TOK_S: 101.00",
            "OURS_TOK_S: 120.50",
            "DELTA_PCT: +19.31",
            "SPREAD_PCT: 2.0",
            "TOKENS: IDENTICAL",
            "SUBSTITUTION: 100/100 applied across 1 family(ies)",
            "VERDICT: WIN",
        ):
            assert key in text, text
        assert text.rstrip().endswith("DONE")

    def test_failed_arm_shows_its_cause_and_log_not_the_other_arms_rate(self, capsys):
        runs = {
            "baseline": [_run(100.0), _run(100.0)],
            "ours": [_run(None, error="torch.OutOfMemoryError: XPU out of memory")] * 2,
        }
        out = msw.judge(runs, ARMS, plain_arm=False)
        msw.render(out, _args(), ARMS, runs, {"routing": "UNCHECKED"})
        text = capsys.readouterr().out
        assert "VERDICT: ARM_FAILED" in text
        assert "FAILED: torch.OutOfMemoryError" in text and "full log: tmp/x.fail.log" in text
        assert "tok/s" not in text


class TestRepeats:
    def test_a_single_repeat_is_refused_at_the_cli(self):
        with pytest.raises(SystemExit):
            msw.build_parser().parse_args(["--model", "m", "--repeats", "1"])
        assert msw.build_parser().parse_args(["--model", "m", "--repeats", "2"]).repeats == 2


_BC_SPEC = importlib.util.spec_from_file_location(
    "bound_candidates_for_msw",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "bound_candidates.py",
)
bc = importlib.util.module_from_spec(_BC_SPEC)
sys.modules[_BC_SPEC.name] = bc
_BC_SPEC.loader.exec_module(bc)


def _routing_on_disk(tmp_path, model="test/model"):
    """One candidate, one op: provider_patch ACCEPTed, apply_substitution REJECTed at
    net_positive (headroom 3 us against a 6 us dispatch cost)."""
    import json
    from types import SimpleNamespace

    op = "_C.rms_norm.default"
    t = lambda shape: ["T", shape, "bfloat16"]  # noqa: E731
    report = {
        "model": model,
        "device_time_total_us": 10000.0,
        "device_time_by_kernel": {"vllm::norm_kernel<bf16>": 2000.0},
        "ops": [{"op": op, "calls": 400, "args": [t([4, 1024]), t([4, 1024]), t([1024]), 1e-6]}],
        "op_share": {op: {"device_us": 2000.0, "share_pct": 20.0}},
        "op_calls": {op: 400},
        "triton": [],
        "edges": [],
    }
    bundle = tmp_path / "bundle"
    (bundle / "source").mkdir(parents=True)
    (bundle / "PROVENANCE.md").write_text("# x\n")
    (bundle / "source" / "k.cpp").write_text("//\n")
    resolution = {
        "_C::rms_norm": {
            "op": "_C::rms_norm",
            "provider": "provider kernel",
            "where": [f"{bundle}/source/k.cpp"],
            "bundle": str(bundle),
            "schema": "_C::rms_norm(Tensor($0! -> ) result, Tensor input, Tensor? weight, float epsilon) -> ()",
            "launched": ["vllm::norm_kernel<bf16>"],
        }
    }
    cal = SimpleNamespace(
        hardware_id="TEST_PART",
        timer="event",
        dispatch_us=6.0,
        timing_floor_us=1.0,
        bandwidth_gbs=1000.0,
        launch_floor_us=2.0,
        matmul_peak_tflops={"bfloat16": 50.0},
    )
    cid = bc.candidate_id(op, "4x1024/4x1024/1024", "bfloat16")
    report_path = tmp_path / "discovered.json"
    report_path.write_text(json.dumps(report))
    cands, rows = bc.run(
        report,
        resolution,
        {cid: {"t_host_us": 8.0, "spread_us": 0.2}},
        cal,
        "run-1",
        definitions_matching=lambda c: ["rmsnorm_h1024"],
        cutoff=0.0,
    )
    out = tmp_path / "bound"
    bc.write_outputs(
        out,
        "run-1",
        cands,
        rows,
        cal,
        0.0005,
        str(report_path),
        report_sha1=bc.report_digest(report_path),
        model=model,
    )
    return out, report_path, cid, op


class TestRoutingPrecheck:
    """A serving run of a pair the routing rejected is refused before any arm launches."""

    def test_accepted_pair_in_the_isolating_mode_passes(self, tmp_path):
        out, _, cid, op = _routing_on_disk(tmp_path)
        args = _args(bound=str(out), mechanism="provider_patch", candidate=[cid], plain_arm=True)
        fields = msw.routing_precheck(args, bc)
        assert fields["routing"] == "OK" and fields["candidate"] == cid
        # An op name selects every candidate of that op.
        args.candidate = [op]
        assert msw.routing_precheck(args, bc)["candidate"] == cid

    def test_rejected_mechanism_is_refused_with_the_gate(self, tmp_path):
        out, _, cid, _ = _routing_on_disk(tmp_path)
        args = _args(bound=str(out), mechanism="apply_substitution", candidate=[cid])
        with pytest.raises(bc.RoutingRefused) as exc:
            msw.routing_precheck(args, bc)
        f = exc.value.fields
        assert f["routing"] == "REJECTED" and f["gate"] == "net_positive"
        assert f["arithmetic"] == "ceiling_us=-3 > spread_us=0.2"
        assert f["verdict"] == "ROUTING_REJECTED"

    def test_mechanism_measured_in_the_wrong_mode_is_refused(self, tmp_path):
        out, _, cid, _ = _routing_on_disk(tmp_path)
        args = _args(bound=str(out), mechanism="provider_patch", candidate=[cid])
        with pytest.raises(bc.RoutingRefused) as exc:
            msw.routing_precheck(args, bc)
        assert exc.value.fields["verdict"] == msw.MECHANISM_MISMATCH
        assert "--plain-arm" in exc.value.why

    def test_routing_of_another_model_or_a_rerun_discovery_is_stale(self, tmp_path):
        out, report_path, cid, _ = _routing_on_disk(tmp_path)
        args = _args(
            model="other/model",
            bound=str(out),
            mechanism="provider_patch",
            candidate=[cid],
            plain_arm=True,
        )
        with pytest.raises(bc.RoutingRefused) as exc:
            msw.routing_precheck(args, bc)
        assert exc.value.fields["verdict"] == "STALE_INPUT"
        assert exc.value.fields["routed_model"] == "test/model"
        report_path.write_text(report_path.read_text() + "\n")
        args.model = "test/model"
        with pytest.raises(bc.RoutingRefused) as exc:
            msw.routing_precheck(args, bc)
        assert exc.value.fields["routing"] == "STALE"

    def test_half_the_flags_is_refused(self, tmp_path):
        out, _, cid, _ = _routing_on_disk(tmp_path)
        with pytest.raises(bc.RoutingRefused, match="go together"):
            msw.routing_precheck(_args(bound=str(out)), bc)

    def test_cli_halts_in_the_contract_before_importing_vllm(self, tmp_path):
        """The refusal costs no serving time: it happens before vLLM is even imported."""
        import subprocess
        import sys

        out, _, cid, _ = _routing_on_disk(tmp_path)
        script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "measure_serving_win.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--model",
                "test/model",
                "--bound",
                str(out),
                "--mechanism",
                "apply_substitution",
                "--candidate",
                cid,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        for key in (
            "MODEL: test/model",
            "MODE: apply",
            "ROUTING: REJECTED",
            f"CANDIDATE: {cid}",
            "MECHANISM: apply_substitution",
            "GATE: net_positive",
            "ARITHMETIC: ceiling_us=-3 > spread_us=0.2",
            "VERDICT: ROUTING_REJECTED",
        ):
            assert key in proc.stdout, proc.stdout
        assert proc.stdout.rstrip().endswith("DONE")
        assert "tok/s" not in proc.stdout and "DELTA" not in proc.stdout
