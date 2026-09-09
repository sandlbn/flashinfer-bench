"""Halts across the pipeline's stages: a failed gate stops the stage, in the contract.

`kernel_trials.py benchmark` is the reference shape -- a failed correctness gate prints no
timing at all -- and this file holds the other stages to it, on the CPU: the trial loop
refuses a (candidate, mechanism) pair the routing rejected or a routing whose discovery has
moved on, fusion proposal exits non-zero when it examined nothing, and discovery's exit
status says whether it produced what it promises.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_halts", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bc = _load("bound_candidates")

OP = "_C.rms_norm.default"

# A CPU harness in the shape discovery emits: OP, CALLS and get_inputs shapes are what the
# routing ties it to a candidate by. It runs; the op is stood in for by arithmetic.
HARNESS = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)

    OP = "{op}"
    CALLS = {calls}


    class Model(nn.Module):
        def forward(self, t0, t1, t2):
            return t1 * t2 + t0 * 0


    def get_inputs():
        device = "cpu"
        return [
            torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
            torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
            torch.randn([1024], dtype=torch.bfloat16, device=device),
        ]


    def get_init_inputs():
        return []
    """
)


def _routing(tmp_path, model="test/model"):
    """One candidate: provider_patch ACCEPTed, apply_substitution REJECTed at net_positive."""
    t = lambda shape: ["T", shape, "bfloat16"]  # noqa: E731
    report = {
        "model": model,
        "device_time_total_us": 10000.0,
        "device_time_by_kernel": {"vllm::norm_kernel<bf16>": 2000.0},
        "ops": [{"op": OP, "calls": 400, "args": [t([4, 1024]), t([4, 1024]), t([1024]), 1e-6]}],
        "op_share": {OP: {"device_us": 2000.0, "share_pct": 20.0}},
        "op_calls": {OP: 400},
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
    cid = bc.candidate_id(OP, "4x1024/4x1024/1024", "bfloat16")
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
    harness = tmp_path / "h.py"
    harness.write_text(HARNESS.format(op=OP, calls=400))
    return out, report_path, harness, cid


def _kt(tmp_path, *argv, timeout=600):
    """kernel_trials.py as a process, with its series store under tmp_path."""
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / "kernel_trials.py"), *argv],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _keys(stdout):
    return dict(
        line.split(": ", 1) for line in stdout.splitlines() if ": " in line and line[0].isupper()
    )


class TestTrialLoopRoutingGate:
    """`init --bound --mechanism` opens a series only on an ACCEPT row, and every
    `benchmark` re-checks it; a refusal names the gate and prints no timing."""

    def test_init_refuses_a_rejected_mechanism_in_the_contract(self, tmp_path):
        out, _, harness, cid = _routing(tmp_path)
        proc = _kt(
            tmp_path,
            "init",
            "s",
            str(harness),
            "--bound",
            str(out),
            "--mechanism",
            "apply_substitution",
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        keys = _keys(proc.stdout)
        assert keys["ROUTING"] == "REJECTED"
        assert keys["CANDIDATE"] == cid and keys["MECHANISM"] == "apply_substitution"
        assert keys["GATE"] == "net_positive"
        assert keys["ARITHMETIC"] == "ceiling_us=-3 > spread_us=0.2"
        assert keys["VERDICT"] == "ROUTING_REJECTED"
        assert "NEEDS" in keys and "increase" in keys["NEEDS"]
        assert proc.stdout.rstrip().endswith("DONE")
        assert not (tmp_path / "tmp" / "kernel-trials" / "s.json").exists()

    def test_half_the_flags_is_refused(self, tmp_path):
        out, _, harness, _ = _routing(tmp_path)
        proc = _kt(tmp_path, "init", "s", str(harness), "--bound", str(out))
        assert proc.returncode != 0 and "go together" in proc.stderr

    def test_init_on_an_accepted_row_records_it_and_benchmark_carries_routing_ok(self, tmp_path):
        out, _, harness, cid = _routing(tmp_path)
        proc = _kt(
            tmp_path,
            "init",
            "s",
            str(harness),
            "--bound",
            str(out),
            "--mechanism",
            "provider_patch",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        series = json.loads((tmp_path / "tmp" / "kernel-trials" / "s.json").read_text())
        assert series["routing"]["candidate"] == cid
        assert series["routing"]["mechanism"] == "provider_patch"
        assert series["routing"]["ceiling_us"] == pytest.approx(3.0)
        proc = _kt(tmp_path, "benchmark", "s", str(harness), "--rounds", "5", "--calls", "3")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        keys = _keys(proc.stdout)
        assert keys["ROUTING"] == "OK" and keys["MECHANISM"] == "provider_patch"
        assert keys["CORRECT"] == "OK" and keys["VERDICT"] in ("WIN", "LOSS", "NOISE")

    def test_benchmark_refuses_once_discovery_has_been_rerun(self, tmp_path):
        out, report_path, harness, _ = _routing(tmp_path)
        proc = _kt(
            tmp_path,
            "init",
            "s",
            str(harness),
            "--bound",
            str(out),
            "--mechanism",
            "provider_patch",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report_path.write_text(report_path.read_text() + "\n")  # discovery re-run
        proc = _kt(tmp_path, "benchmark", "s", str(harness), "--rounds", "5", "--calls", "3")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        keys = _keys(proc.stdout)
        assert keys["ROUTING"] == "STALE" and keys["VERDICT"] == "STALE_INPUT"
        assert "REPORT_SHA1" in keys and "EXPECTED_SHA1" in keys
        assert "SPEEDUP" not in keys and "CANDIDATE_US" not in keys and "CORRECT" not in keys
        assert "re-run scripts/bound_candidates.py" in proc.stdout.lower()

    def test_benchmark_refuses_once_the_routing_is_recomputed_against_the_pair(self, tmp_path):
        out, _, harness, cid = _routing(tmp_path)
        proc = _kt(
            tmp_path,
            "init",
            "s",
            str(harness),
            "--bound",
            str(out),
            "--mechanism",
            "provider_patch",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        # Bounding re-run with the pair now rejected (the bundle's source gone).
        bound = json.loads((out / "bound.json").read_text())
        for row in bound["rows"]:
            if row["candidate_id"] == cid and row["mechanism"] == "provider_patch":
                row.update(
                    status="REJECT",
                    gate="source_present",
                    arithmetic="source_files=0 > required=0",
                    where="bound_candidates.py:0",
                    needs={
                        "quantity": "source_files",
                        "observed": 0,
                        "cmp": ">",
                        "threshold_name": "required",
                        "threshold": 0,
                        "direction": "increase",
                    },
                )
        (out / "bound.json").write_text(json.dumps(bound))
        proc = _kt(tmp_path, "benchmark", "s", str(harness), "--rounds", "5", "--calls", "3")
        assert proc.returncode == 1, proc.stdout + proc.stderr
        keys = _keys(proc.stdout)
        assert keys["ROUTING"] == "REJECTED" and keys["GATE"] == "source_present"
        assert "SPEEDUP" not in keys

    def test_unrouted_series_says_so_in_the_contract(self, tmp_path):
        _, _, harness, _ = _routing(tmp_path)
        assert _kt(tmp_path, "init", "s", str(harness)).returncode == 0
        proc = _kt(tmp_path, "benchmark", "s", str(harness), "--rounds", "5", "--calls", "3")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert _keys(proc.stdout)["ROUTING"] == "UNCHECKED"

    def test_ab_refuses_a_rejected_pair_before_launching_any_arm(self, tmp_path):
        out, _, harness, _ = _routing(tmp_path)
        proc = _kt(
            tmp_path,
            "ab",
            str(harness),
            "--bound",
            str(out),
            "--mechanism",
            "apply_substitution",
            "--rounds",
            "10",
            timeout=120,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        keys = _keys(proc.stdout)
        assert keys["ROUTING"] == "REJECTED" and keys["VERDICT"] == "ROUTING_REJECTED"
        assert "PROCESSES" not in keys and "SPEEDUP" not in keys

    def test_a_harness_of_an_op_without_share_is_not_a_worklist_row(self, tmp_path):
        out, _, harness, _ = _routing(tmp_path)
        harness.write_text(HARNESS.format(op="aten.detach.default", calls=50))
        proc = _kt(
            tmp_path,
            "init",
            "s",
            str(harness),
            "--bound",
            str(out),
            "--mechanism",
            "provider_patch",
        )
        assert proc.returncode == 1
        keys = _keys(proc.stdout)
        assert keys["ROUTING"] == "UNEVALUATED" and keys["OP"] == "aten.detach.default"


class TestFusionCandidatesExitStatus:
    """A run that examined nothing exits non-zero instead of printing a sentence."""

    def _main(self, monkeypatch, *argv):
        fc = _load("fusion_candidates")
        monkeypatch.setattr(sys, "argv", ["fusion_candidates.py", *argv])
        with pytest.raises(SystemExit) as exc:
            fc.main()
        return exc.value.code

    def test_no_edges_is_a_refusal(self, tmp_path, monkeypatch):
        report = tmp_path / "discovered.json"
        report.write_text(json.dumps({"ops": [], "edges": []}))
        code = self._main(monkeypatch, "--report", str(report), "--xe-fuse", str(tmp_path))
        assert isinstance(code, str) and "no edges" in code

    def test_unreadable_xe_fuse_is_a_refusal_not_an_empty_proposal(self, tmp_path, monkeypatch):
        report = tmp_path / "discovered.json"
        report.write_text(
            json.dumps({"edges": [{"producer": "aten.linear.default", "consumer": OP, "count": 9}]})
        )
        code = self._main(monkeypatch, "--report", str(report), "--xe-fuse", str(tmp_path / "none"))
        assert isinstance(code, str) and "not readable" in code

    def test_missing_report_is_a_refusal(self, tmp_path, monkeypatch):
        code = self._main(monkeypatch, "--report", str(tmp_path / "nope.json"))
        assert isinstance(code, str) and "no discovery report" in code

    def test_presets_distinguish_unreadable_from_empty(self, tmp_path):
        fc = _load("fusion_candidates")
        assert fc.presets(tmp_path / "absent") is None
        gen = tmp_path / "autotune" / "generate_kernel.py"
        gen.parent.mkdir(parents=True)
        gen.write_text("print('  k0a  D = gamma[n] * (acc + residual)')\n")
        assert fc.presets(tmp_path) == [("k0a", "D = gamma[n] * (acc + residual)")]
        gen.write_text("raise SystemExit(2)\n")
        assert fc.presets(tmp_path) is None


class TestDiscoveryExitStatus:
    """Discovery promises verified harnesses with shares; its exit code says whether it kept
    that promise, so a driver cannot move on from a run that measured nothing."""

    hfm = _load("harness_from_model")

    def test_a_failed_device_time_pass_exits_non_zero(self):
        code, message = self.hfm.exit_status(5, 0, {"device_time_error": "RuntimeError: x"})
        assert code == 1 and "bound_candidates.py will refuse" in message

    def test_no_verified_harness_exits_non_zero(self):
        code, message = self.hfm.exit_status(0, 4, {"device_time_error": None})
        assert code == 1 and "no harness verified" in message

    def test_partial_success_is_success_with_the_drops_reported_above(self):
        code, message = self.hfm.exit_status(3, 2, {"device_time_error": None})
        assert code == 0 and "kernel_trials" in message

    def test_bounding_refuses_the_report_discovery_marks_as_unmeasured(self, tmp_path):
        with pytest.raises(SystemExit, match="RuntimeError: x"):
            bc.check_report(
                {"device_time_error": "RuntimeError: x", "device_time_total_us": 0.0, "ops": [1]},
                tmp_path / "discovered.json",
            )
