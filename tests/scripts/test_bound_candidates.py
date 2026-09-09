"""Tests for the routing stage: the gate chain, the log contract, and the ranking.

Everything here runs on the CPU against a synthetic discovery report, a synthetic resolution
and a calibration record built in the test. What is tested is what a consumer of bound.log
and bound.json relies on: every gate's pass and reject path, that a rejected mechanism says
which arithmetic failed, that an unmeasured cost never prints as a number, that the worklist
is exactly the ACCEPT rows in worth order, and that the log reproduces the worklist.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bc = _load("bound_candidates")

T = lambda shape, dtype="bfloat16": ["T", list(shape), dtype]  # noqa: E731

RMS = "_C.rms_norm.default"
LIN = "aten.linear.default"
ARGMAX = "aten.argmax.default"
TRITON = "_tiny_kernel"
TOTAL_US = 10000.0


def _report():
    return {
        "device_time_total_us": TOTAL_US,
        "device_time_by_kernel": {
            "gemm_kernel": 7000.0,
            "vllm::norm_kernel<bf16>": 2000.0,
            "at::native::xpu::ReduceKernel<ArgMax>": 100.0,
            TRITON: 5.0,
        },
        "ops": [
            {"op": RMS, "calls": 400, "args": [T([4, 1024]), T([4, 1024]), T([1024]), 1e-6]},
            {"op": LIN, "calls": 400, "args": [T([4, 1024]), T([4096, 1024])]},
            {"op": ARGMAX, "calls": 10, "args": [T([4, 149], "float32"), -1, True]},
            {"op": "aten.detach.default", "calls": 50, "args": [T([4, 1024])]},
        ],
        "op_share": {
            RMS: {"device_us": 2000.0, "share_pct": 20.0},
            LIN: {"device_us": 7000.0, "share_pct": 70.0},
            ARGMAX: {"device_us": 100.0, "share_pct": 1.0},
            "aten.detach.default": {"device_us": 0.0, "share_pct": 0.0},
        },
        "op_calls": {RMS: 400, LIN: 400, ARGMAX: 10, "aten.detach.default": 50},
        "triton": [{"kernel": TRITON, "source": "/nonexistent/stack/file.py:12", "calls": 5}],
        "edges": [
            {"producer": LIN, "consumer": RMS, "count": 400},
            {"producer": RMS, "consumer": LIN, "count": 400},
        ],
    }


def _resolution(bundle_dir):
    return {
        "_C::rms_norm": {
            "op": "_C::rms_norm",
            "provider": "provider kernel",
            "where": [f"{bundle_dir}/source/layernorm.cpp  (defines it)"],
            "bundle": str(bundle_dir),
            "schema": "_C::rms_norm(Tensor($0! -> ) result, Tensor input, Tensor? weight, float epsilon) -> ()",
            "launched": ["vllm::norm_kernel<bf16>"],
        },
        "aten::linear": {
            "op": "aten::linear",
            "provider": "oneDNN",
            "where": ["primitive: matmul via jit:gemm:any  [mb4ic1024oc4096]"],
            "bundle": None,
            "schema": "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
            "launched": ["gemm_kernel"],
        },
        "aten::argmax": {
            "op": "aten::argmax",
            "provider": "ATen kernel inside PyTorch",
            "where": ["/__w/pytorch/pytorch/aten/src/ATen/native/ReduceOps.cpp"],
            "bundle": None,
            "schema": "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor",
            "launched": ["at::native::xpu::ReduceKernel<ArgMax>"],
            "primitives": [],
        },
    }


def _cal(**over):
    base = {
        "hardware_id": "TEST_PART",
        "timer": "event-batched",
        "dispatch_us": 6.0,
        "timing_floor_us": 1.0,
        "bandwidth_gbs": 1000.0,
        "launch_floor_us": 2.0,
        "matmul_peak_tflops": {"bfloat16": 50.0},
    }
    base.update(over)
    return SimpleNamespace(**base)


PRESETS = [
    ("k0a", "D = gamma[n] * (acc + residual)"),
    ("sa_scale_rows", "D[m,n] *= scale[m] (standalone RMSNorm scaling)"),
]


def _ids():
    return {
        "rms": bc.candidate_id(RMS, "4x1024/4x1024/1024", "bfloat16"),
        "lin": bc.candidate_id(LIN, "4x1024/4096x1024", "bfloat16"),
        "argmax": bc.candidate_id(ARGMAX, "4x149", "float32"),
        "triton": bc.candidate_id(TRITON, "-", "-"),
    }


def _measurements():
    ids = _ids()
    return {
        ids["rms"]: {"t_host_us": 8.0, "spread_us": 0.2},
        ids["lin"]: {"t_host_us": 20.0, "spread_us": 1.0},
    }


def _defs(c):
    return ["rmsnorm_h1024"] if c.op == RMS else []


@pytest.fixture
def bundle_dir(tmp_path):
    d = tmp_path / "pulled" / "_C_rms_norm"
    (d / "source").mkdir(parents=True)
    (d / "PROVENANCE.md").write_text("# _C::rms_norm\n")
    (d / "source" / "layernorm.cpp").write_text("// kernel\n")
    return d


@pytest.fixture
def outcome(bundle_dir):
    cands, rows = bc.run(
        _report(),
        _resolution(bundle_dir),
        _measurements(),
        _cal(),
        run_id="test-run",
        presets=PRESETS,
        post_op_algorithms={"relu", "gelu_erf", "swish"},
        definitions_matching=_defs,
    )
    return cands, rows


def _rows(rows, cid, mech):
    return [r for r in rows if r["candidate_id"] == cid and r["mechanism"] == mech]


def _final(rows, cid, mech):
    return _rows(rows, cid, mech)[-1]


def _by_id(cands):
    return {c.candidate_id: c for c in cands}


class TestRegimeAndDerivedInputs:
    def test_bytes_min_counts_inputs_once_and_mutated_arguments_twice(self):
        args = [T([4, 1024]), T([4, 1024]), T([1024]), 1e-6]
        schema = "_C::rms_norm(Tensor($0! -> ) result, Tensor input, Tensor? weight, float epsilon) -> ()"
        n, note = bc._bytes_min(args, schema)
        assert n == 8192 * 2 + 2048 + 8192  # result written back
        assert "lower bound" not in note

    def test_value_returning_op_is_a_stated_lower_bound(self):
        n, note = bc._bytes_min(
            [T([4, 1024]), T([4096, 1024])], "aten::linear(Tensor a, Tensor b) -> Tensor"
        )
        assert n == 8192 + 4096 * 1024 * 2
        assert "lower bound" in note

    def test_gemm_flops_from_linear_shapes(self):
        assert (
            bc._gemm_flops([((4, 1024), "bfloat16"), ((4096, 1024), "bfloat16")])
            == 2 * 4 * 4096 * 1024
        )

    def test_memory_bound_regime_and_bound_arithmetic(self, outcome):
        c = _by_id(outcome[0])[_ids()["rms"]]
        assert c.t_dev == pytest.approx(5.0)  # 2000 us over 400 calls
        assert c.t_mem == pytest.approx(26624 / 1000e9 * 1e6)
        assert c.floor_us == 0.0  # a profiler duration carries no timer overhead
        assert c.bound == pytest.approx(2.0)  # the launch floor dominates a 27 KB move
        assert c.regime == bc.R_MEM_INEFF
        assert "spread_us" in c.regime_test

    def test_gemm_gets_a_compute_time_from_the_measured_peak(self, outcome):
        c = _by_id(outcome[0])[_ids()["lin"]]
        assert c.flops == 2 * 4 * 4096 * 1024
        assert c.t_cmp == pytest.approx(c.flops / 50e12 * 1e6)

    def test_no_spread_means_no_regime_not_a_guess(self, outcome):
        c = _by_id(outcome[0])[_ids()["argmax"]]
        assert c.spread_us is None and c.regime == bc.R_UNMEASURED

    def test_launch_bound_when_host_time_dominates(self, bundle_dir):
        ids = _ids()
        meas = {ids["rms"]: {"t_host_us": 30.0, "spread_us": 0.2}}
        cands, _ = bc.run(
            _report(), _resolution(bundle_dir), meas, _cal(), "r", definitions_matching=_defs
        )
        assert _by_id(cands)[ids["rms"]].regime == bc.R_LAUNCH

    def test_launch_bound_at_the_launch_floor(self, bundle_dir):
        ids = _ids()
        cands, _ = bc.run(
            _report(), _resolution(bundle_dir), _measurements(), _cal(launch_floor_us=5.0), "r"
        )
        assert _by_id(cands)[ids["rms"]].regime == bc.R_LAUNCH

    def test_spill_invalidates_the_memory_rows(self, bundle_dir):
        ids = _ids()
        meas = {ids["rms"]: {"t_host_us": 8.0, "spread_us": 0.2, "spill": 128}}
        cands, _ = bc.run(_report(), _resolution(bundle_dir), meas, _cal(), "r")
        assert _by_id(cands)[ids["rms"]].regime == bc.R_SPILL

    def test_emulated_dtype_is_its_own_regime(self, bundle_dir):
        ids = _ids()
        cands, _ = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(),
            "r",
            native=lambda dt: False,
        )
        assert _by_id(cands)[ids["rms"]].regime == bc.R_EMULATED

    def test_layout_limited_when_the_pattern_bandwidth_explains_t_dev(self, bundle_dir):
        ids = _ids()
        # bytes_min 26624 at 1000 GB/s is 0.027 us; at 5 GB/s for the pattern it is 5.3 us,
        # which is within spread of t_dev = 5.0 -> the layout, not the kernel, is the limit.
        cands, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            {ids["rms"]: {"t_host_us": 8.0, "spread_us": 0.4}},
            _cal(timing_floor_us=0.1),
            "r",
            patterns={RMS: (64, 4096)},
            bw_pattern=lambda run, stride: 5.0,
        )
        c = _by_id(cands)[ids["rms"]]
        assert c.regime == bc.R_MEM_LAYOUT
        assert _final(rows, ids["rms"], "layout_transform")["status"] == "ACCEPT"


class TestGateChain:
    def test_provider_patch_passes_every_gate_for_a_bundled_provider_kernel(self, outcome):
        rows = _rows(outcome[1], _ids()["rms"], "provider_patch")
        assert [r["gate"] for r in rows[:-1]] == [
            "class_admits",
            "source_present",
            "measurable",
            "headroom",
            "net_positive",
            "worth_cutoff",
        ]
        assert all(r["status"] == "PASS" for r in rows[:-1])
        assert rows[-1]["status"] == "ACCEPT" and rows[-1]["mechanism_us"] == 0.0
        assert rows[-1]["ceiling_us"] == pytest.approx(3.0)  # 5.0 - max(t_mem, t_cmp, launch=2.0)
        assert rows[-1]["worth"] == pytest.approx(3.0 * 400 / TOTAL_US)

    def test_class_admits_rejects_with_the_class_named(self, outcome):
        r = _final(outcome[1], _ids()["lin"], "provider_patch")
        assert r["gate"] == "class_admits" and r["status"] == "REJECT"
        assert r["arithmetic"] == "class=onednn == admits=provider_kernel"
        assert r["needs"]["quantity"] == "class" and r["needs"]["threshold"] == "provider_kernel"

    def test_source_present_rejects_a_triton_kernel_whose_file_is_not_here(self, outcome):
        r = _final(outcome[1], _ids()["triton"], "triton_in_place")
        assert r["gate"] == "source_present" and r["status"] == "REJECT"
        assert r["lhs_value"] == 0

    def test_source_present_rejects_a_provider_kernel_without_a_bundle(self, bundle_dir):
        res = _resolution(bundle_dir)
        res["_C::rms_norm"]["bundle"] = None
        _, rows = bc.run(_report(), res, _measurements(), _cal(), "r")
        r = _final(rows, _ids()["rms"], "provider_patch")
        assert (r["gate"], r["status"]) == ("source_present", "REJECT")

    def test_definition_exists_rejects_when_no_definition_binds(self, outcome):
        r = _final(outcome[1], _ids()["lin"], "apply_substitution")
        assert (r["gate"], r["status"]) == ("definition_exists", "REJECT")
        assert r["arithmetic"] == "definitions_matching=0 > required=0"

    def test_definition_exists_passes_and_records_which(self, outcome):
        rows = _rows(outcome[1], _ids()["rms"], "apply_substitution")
        gate = next(r for r in rows if r["gate"] == "definition_exists")
        assert gate["status"] == "PASS" and gate["detail"]["definitions"] == ["rmsnorm_h1024"]

    def test_cost_calibrated_rejects_with_none_never_zero(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(dispatch_us=None),
            "r",
            presets=PRESETS,
            definitions_matching=_defs,
        )
        for mech in ("apply_substitution", "fusion_apply"):
            r = _final(rows, _ids()["rms"], mech)
            assert (r["gate"], r["status"]) == ("cost_calibrated", "REJECT")
            assert r["arithmetic"] == "dispatch_us=None is_not unmeasured=None"
            assert r["lhs_value"] is None
        text = "\n".join(bc.format_line(r) for r in rows)
        assert "dispatch_us=0" not in text

    def test_cost_calibrated_passes_and_apply_pays_the_measured_dispatch(self, outcome):
        r = _final(outcome[1], _ids()["rms"], "apply_substitution")
        # headroom 3.0 minus the 6.0 us dispatch: the exchange loses.
        assert (r["gate"], r["status"]) == ("net_positive", "REJECT")
        assert r["arithmetic"] == "ceiling_us=-3 > spread_us=0.2"
        assert r["detail"]["mechanism_us"] == 6.0
        assert r["needs"] == {
            "quantity": "ceiling_us",
            "observed": -3.0,
            "cmp": ">",
            "threshold_name": "spread_us",
            "threshold": 0.2,
            "direction": "increase",
        }

    def test_measurable_holds_an_event_timed_t_dev_to_the_timing_floor(self, bundle_dir):
        meas = {_ids()["rms"]: {"t_dev_us": 5.0, "t_dev_source": "event", "spread_us": 0.2}}
        _, rows = bc.run(_report(), _resolution(bundle_dir), meas, _cal(timing_floor_us=5.0), "r")
        r = _final(rows, _ids()["rms"], "provider_patch")
        assert (r["gate"], r["status"]) == ("measurable", "REJECT")
        assert r["arithmetic"] == "t_dev_us=5 > instrument_floor_us=5"

    def test_a_device_side_t_dev_is_not_held_to_the_timer_floor(self, outcome):
        c = _by_id(outcome[0])[_ids()["rms"]]
        assert c.t_dev_source == "profiler:op_average" and c.floor_us == 0.0
        gate = next(
            r
            for r in _rows(outcome[1], c.candidate_id, "provider_patch")
            if r["gate"] == "measurable"
        )
        assert (
            gate["status"] == "PASS" and gate["arithmetic"] == "t_dev_us=5 > instrument_floor_us=0"
        )

    def test_measurable_rejects_when_the_profiled_kernel_is_not_the_class(self, bundle_dir):
        res = _resolution(bundle_dir)
        res["_C::rms_norm"]["launched"] = ["some_other_kernel"]
        _, rows = bc.run(_report(), res, _measurements(), _cal(), "r")
        r = _final(rows, _ids()["rms"], "provider_patch")
        assert (r["gate"], r["status"]) == ("measurable", "REJECT")
        assert r["lhs"] == "kernel_matched" and r["lhs_value"] is False

    def test_headroom_rejects_when_t_dev_is_at_or_under_the_bound(self, bundle_dir):
        # 20 GB/s makes t_mem for the GEMM's 8.4 MB about 420 us, far above its 17.5 us t_dev:
        # the bytes the op is charged cannot have moved at that rate, and the line says so.
        _, rows = bc.run(
            _report(), _resolution(bundle_dir), _measurements(), _cal(bandwidth_gbs=20.0), "r"
        )
        r = _final(rows, _ids()["lin"], "library_call")
        assert (r["gate"], r["status"]) == ("headroom", "REJECT")
        assert r["lhs"] == "headroom_us" and r["lhs_value"] < 0
        assert r["arithmetic"].endswith(" > 0")

    def test_net_positive_rejects_when_spread_is_unmeasured(self, bundle_dir):
        _, rows = bc.run(_report(), _resolution(bundle_dir), {}, _cal(), "r")
        r = _final(rows, _ids()["rms"], "provider_patch")
        assert (r["gate"], r["status"]) == ("net_positive", "REJECT")
        assert r["arithmetic"] == "ceiling_us=3 > spread_us=None"

    def test_worth_cutoff_rejects_below_the_operator_cutoff(self, bundle_dir):
        _, rows = bc.run(
            _report(), _resolution(bundle_dir), _measurements(), _cal(), "r", cutoff=0.5
        )
        r = _final(rows, _ids()["rms"], "provider_patch")
        assert (r["gate"], r["status"]) == ("worth_cutoff", "REJECT")
        assert r["arithmetic"].startswith("worth=0.12 >= cutoff=0.5")

    def test_default_cutoff_is_the_smallest_resolved_kernel_share(self):
        assert bc.default_cutoff(_report()) == pytest.approx(5.0 / TOTAL_US)

    def test_edge_present_rejects_without_a_gemm_producer_edge(self, bundle_dir):
        report = _report()
        report["edges"] = [{"producer": RMS, "consumer": LIN, "count": 400}]
        _, rows = bc.run(
            report, _resolution(bundle_dir), _measurements(), _cal(), "r", presets=PRESETS
        )
        r = _final(rows, _ids()["rms"], "fusion_callsite")
        assert (r["gate"], r["status"]) == ("class_admits", "REJECT")
        assert r["arithmetic"] == "gemm_producers=0 > required=0"

    def test_edge_present_passes_and_epilogue_expressible_rejects_without_a_preset(
        self, bundle_dir
    ):
        _, rows = bc.run(
            _report(), _resolution(bundle_dir), _measurements(), _cal(), "r", presets=[]
        )
        rows_ = _rows(rows, _ids()["rms"], "fusion_callsite")
        assert [(r["gate"], r["status"]) for r in rows_] == [
            ("class_admits", "PASS"),
            ("edge_present", "PASS"),
            ("epilogue_expressible", "REJECT"),
        ]
        assert rows_[1]["arithmetic"] == "edge_count=400 >= threshold=400"
        assert rows_[2]["detail"]["post_op_algorithms"] == "unreadable"

    def test_epilogue_expressible_by_a_post_op_read_from_the_header(self, bundle_dir):
        report = _report()
        report["ops"][0]["op"] = "aten.gelu.default"
        report["op_share"]["aten.gelu.default"] = report["op_share"].pop(RMS)
        report["op_calls"]["aten.gelu.default"] = report["op_calls"].pop(RMS)
        report["edges"][0]["consumer"] = "aten.gelu.default"
        res = _resolution(bundle_dir)
        res["aten::gelu"] = dict(res.pop("_C::rms_norm"), op="aten::gelu")
        cid = bc.candidate_id("aten.gelu.default", "4x1024/4x1024/1024", "bfloat16")
        _, rows = bc.run(
            report,
            res,
            {cid: {"t_host_us": 8.0, "spread_us": 0.2}},
            _cal(),
            "r",
            presets=[],
            post_op_algorithms={"gelu_erf", "gelu_tanh", "relu"},
        )
        gate = next(
            r for r in _rows(rows, cid, "fusion_callsite") if r["gate"] == "epilogue_expressible"
        )
        assert gate["status"] == "PASS" and gate["detail"]["post_op"] == "gelu_erf"

    def test_fusion_is_priced_twice_callsite_free_and_apply_at_dispatch(self, outcome):
        ids = _ids()
        callsite = _final(outcome[1], ids["rms"], "fusion_callsite")
        through_apply = _final(outcome[1], ids["rms"], "fusion_apply")
        assert callsite["status"] == "ACCEPT" and callsite["mechanism_us"] == 0.0
        assert callsite["ceiling_us"] == pytest.approx(5.0 + 2.0)  # t_dev plus the launch removed
        assert through_apply["status"] == "ACCEPT" and through_apply["mechanism_us"] == 6.0
        assert through_apply["ceiling_us"] == pytest.approx(7.0 - 6.0)

    def test_source_rewrite_needs_the_gap_analysis(self, outcome, bundle_dir):
        r = _final(outcome[1], _ids()["lin"], "source_rewrite")
        assert (r["gate"], r["status"]) == ("class_admits", "REJECT")
        assert r["arithmetic"] == "composite_bytes_moved=None > bytes_required=None"
        gaps = {LIN: {"bytes_moved": 4e7, "bytes_required": 8.4e6}}
        _, rows = bc.run(
            _report(), _resolution(bundle_dir), _measurements(), _cal(), "r", gaps=gaps
        )
        assert _final(rows, _ids()["lin"], "source_rewrite")["status"] == "ACCEPT"

    def test_upstream_report_is_logged_but_never_a_worklist_row(self, outcome):
        rows = _rows(outcome[1], _ids()["argmax"], "upstream_report")
        assert [(r["gate"], r["status"]) for r in rows] == [
            ("class_admits", "PASS"),
            ("worklist_eligible", "REJECT"),
        ]
        assert not any(r["status"] == "ACCEPT" for r in rows)


class TestAuthoredMechanisms:
    """Writing a kernel for an op nothing implements well, priced by its delivery.

    Admission is the resolver's class -- an ATen kernel inside PyTorch, a decomposition, a
    Python-registered op -- and the bound is what a kernel *written here* reaches, from the
    authored-stream probe, never the part's peak. The callsite delivery is free; the apply()
    delivery pays the measured dispatch, exactly as the fusion pair does.
    """

    AUTHORED_CAL = {"authored_stream_gbs": 500.0}

    def _run(self, bundle_dir, cal=None, res=None, meas=None, **kw):
        ids = _ids()
        measurements = {**_measurements(), ids["argmax"]: {"t_host_us": 12.0, "spread_us": 0.3}}
        if meas is not None:
            measurements = meas
        return bc.run(
            _report(),
            res if res is not None else _resolution(bundle_dir),
            measurements,
            cal if cal is not None else _cal(**self.AUTHORED_CAL),
            "r",
            presets=PRESETS,
            definitions_matching=_defs,
            **kw,
        )

    def test_class_admits_only_the_authorable_classes(self, bundle_dir):
        _, rows = self._run(bundle_dir)
        ids = _ids()
        for cid, cls in ((ids["rms"], "provider_kernel"), (ids["lin"], "onednn")):
            for mech in bc.AUTHORED:
                r = _final(rows, cid, mech)
                assert (r["gate"], r["status"]) == ("class_admits", "REJECT"), (cid, mech)
                assert r["arithmetic"] == (
                    f"class={cls} in authorable_classes=aten|decomposition|python_op"
                )
        first = _rows(rows, ids["argmax"], "authored_callsite")[0]
        assert (first["gate"], first["status"]) == ("class_admits", "PASS")
        assert first["detail"]["launched"] == ["at::native::xpu::ReduceKernel<ArgMax>"]
        assert first["detail"]["parts_us"] == {"at::native::xpu::ReduceKernel<ArgMax>": 10.0}

    def test_callsite_delivery_is_free_and_priced_against_the_authored_stream(self, bundle_dir):
        cands, rows = self._run(bundle_dir)
        cid = _ids()["argmax"]
        chain = _rows(rows, cid, "authored_callsite")
        assert [r["gate"] for r in chain[:-1]] == [
            "class_admits",
            "parts_plain",
            "attainable_calibrated",
            "measurable",
            "headroom",
            "net_positive",
            "worth_cutoff",
        ]
        assert all(r["status"] == "PASS" for r in chain[:-1])
        accept = chain[-1]
        assert accept["status"] == "ACCEPT" and accept["mechanism_us"] == 0.0
        # t_dev 10 minus max(2384 B at 500 GB/s, launch floor 2.0): the floor binds.
        assert accept["ceiling_us"] == pytest.approx(8.0)
        assert accept["worth"] == pytest.approx(8.0 * 10 / TOTAL_US)
        c = _by_id(cands)[cid]
        assert c.t_mem_authored == pytest.approx(2384 / 500e9 * 1e6)
        assert c.bound_authored == pytest.approx(2.0)
        head = next(r for r in chain if r["gate"] == "headroom")
        assert head["detail"]["authored_stream_gbs"] == 500.0
        assert head["detail"]["bound_of"].startswith("max(t_mem_authored_us")

    def test_the_authored_rate_not_the_peak_forms_the_bound(self, bundle_dir):
        cid = _ids()["argmax"]
        # 4 MB at the part's 1000 GB/s is 4 us; at the 500 GB/s a written kernel reached
        # here it is 8 us, and that is what the authored ceiling is measured from.
        _, rows = self._run(bundle_dir, bytes_overrides={cid: 4_000_000})
        r = _final(rows, cid, "authored_callsite")
        assert r["status"] == "ACCEPT" and r["ceiling_us"] == pytest.approx(2.0)
        # A slower written rate leaves nothing: rejected at headroom, with the arithmetic.
        _, rows = self._run(
            bundle_dir, cal=_cal(authored_stream_gbs=300.0), bytes_overrides={cid: 4_000_000}
        )
        r = _final(rows, cid, "authored_callsite")
        assert (r["gate"], r["status"]) == ("headroom", "REJECT")
        assert r["lhs_value"] == pytest.approx(10.0 - 4_000_000 / 300e9 * 1e6)
        assert r["needs"]["direction"] == "increase"

    def test_apply_delivery_pays_the_measured_dispatch(self, bundle_dir):
        _, rows = self._run(bundle_dir)
        cid = _ids()["argmax"]
        chain = _rows(rows, cid, "authored_apply")
        assert [r["gate"] for r in chain[:-1]] == [
            "class_admits",
            "parts_plain",
            "definition_authorable",
            "cost_calibrated",
            "attainable_calibrated",
            "measurable",
            "headroom",
            "net_positive",
            "worth_cutoff",
        ]
        accept = chain[-1]
        assert accept["status"] == "ACCEPT" and accept["mechanism_us"] == 6.0
        assert accept["ceiling_us"] == pytest.approx(8.0 - 6.0)
        authorable = next(r for r in chain if r["gate"] == "definition_authorable")
        assert authorable["arithmetic"] == "interface_tensors=1 == tensor_args=1"
        assert authorable["detail"]["schema"].startswith("aten::argmax(")
        assert authorable["detail"]["dtypes"] == ["float32"]
        # Same kernel, other delivery: the dispatch turns the exchange negative.
        _, rows = self._run(bundle_dir, cal=_cal(dispatch_us=9.0, **self.AUTHORED_CAL))
        r = _final(rows, cid, "authored_apply")
        assert (r["gate"], r["status"]) == ("net_positive", "REJECT")
        assert r["arithmetic"] == "ceiling_us=-1 > spread_us=0.3"
        assert _final(rows, cid, "authored_callsite")["status"] == "ACCEPT"

    def test_apply_delivery_is_unavailable_without_a_dispatch_cost(self, bundle_dir):
        _, rows = self._run(bundle_dir, cal=_cal(dispatch_us=None, **self.AUTHORED_CAL))
        r = _final(rows, _ids()["argmax"], "authored_apply")
        assert (r["gate"], r["status"]) == ("cost_calibrated", "REJECT")
        assert r["arithmetic"] == "dispatch_us=None is_not unmeasured=None"

    def test_attainable_calibrated_rejects_with_none_never_a_peak(self, bundle_dir):
        _, rows = self._run(bundle_dir, cal=_cal())
        cid = _ids()["argmax"]
        for mech in bc.AUTHORED:
            r = _final(rows, cid, mech)
            assert (r["gate"], r["status"]) == ("attainable_calibrated", "REJECT"), mech
            assert r["arithmetic"] == "authored_stream_gbs=None is_not unmeasured=None"
            assert r["lhs_value"] is None
        text = "\n".join(bc.format_line(r) for r in rows)
        assert "authored_stream_gbs=0" not in text
        assert "bound_authored" not in text

    def test_a_library_primitive_among_the_parts_sends_the_op_elsewhere(self, bundle_dir):
        res = _resolution(bundle_dir)
        res["aten::argmax"]["primitives"] = ["matmul via jit:gemm:any"]
        _, rows = self._run(bundle_dir, res=res)
        for mech in bc.AUTHORED:
            r = _final(rows, _ids()["argmax"], mech)
            assert (r["gate"], r["status"]) == ("parts_plain", "REJECT"), mech
            assert r["arithmetic"] == "library_primitives=1 == required=0"
            assert r["detail"]["primitives"] == ["matmul via jit:gemm:any"]

    def test_a_resolution_without_a_primitives_record_reads_none_not_zero(self, bundle_dir):
        res = _resolution(bundle_dir)
        del res["aten::argmax"]["primitives"]
        _, rows = self._run(bundle_dir, res=res)
        r = _final(rows, _ids()["argmax"], "authored_callsite")
        assert (r["gate"], r["status"]) == ("parts_plain", "REJECT")
        assert r["arithmetic"] == "library_primitives=None == required=0"

    def test_an_unsized_interface_cannot_be_authored_as_a_definition(self, bundle_dir):
        report = _report()
        op = "aten.abs.default"
        report["ops"].append({"op": op, "calls": 10, "args": [T([4, 8], "complex64")]})
        report["op_share"][op] = {"device_us": 50.0, "share_pct": 0.5}
        report["op_calls"][op] = 10
        report["device_time_by_kernel"]["at::native::xpu::AbsKernel<complex>"] = 50.0
        res = _resolution(bundle_dir)
        res["aten::abs"] = {
            "op": "aten::abs",
            "provider": "ATen kernel inside PyTorch",
            "where": [],
            "bundle": None,
            "schema": "aten::abs(Tensor self) -> Tensor",
            "launched": ["at::native::xpu::AbsKernel<complex>"],
            "primitives": [],
        }
        _, rows = bc.run(report, res, {}, _cal(**self.AUTHORED_CAL), "r")
        cid = bc.candidate_id(op, "4x8", "complex64")
        r = _final(rows, cid, "authored_apply")
        assert (r["gate"], r["status"]) == ("definition_authorable", "REJECT")
        assert r["arithmetic"] == "interface_tensors=0 == tensor_args=1"
        # The callsite delivery needs no definition and is priced on its own gates.
        assert _final(rows, cid, "authored_callsite")["gate"] != "definition_authorable"

    def test_an_unmeasured_spread_leaves_the_ceiling_unpriced(self, bundle_dir):
        _, rows = self._run(bundle_dir, meas=_measurements())
        r = _final(rows, _ids()["argmax"], "authored_callsite")
        assert (r["gate"], r["status"]) == ("net_positive", "REJECT")
        assert r["arithmetic"] == "ceiling_us=8 > spread_us=None"

    def test_launch_bound_parts_leave_no_authored_headroom(self, bundle_dir):
        # An op whose device time is at the launch floor: a written kernel still launches.
        report = _report()
        report["op_share"][ARGMAX] = {"device_us": 20.0, "share_pct": 0.2}
        report["device_time_by_kernel"]["at::native::xpu::ReduceKernel<ArgMax>"] = 20.0
        _, rows = bc.run(report, _resolution(bundle_dir), {}, _cal(**self.AUTHORED_CAL), "r")
        r = _final(rows, _ids()["argmax"], "authored_callsite")
        assert (r["gate"], r["status"]) == ("headroom", "REJECT")
        assert r["arithmetic"] == "headroom_us=0 > 0"

    def test_the_calibration_json_may_carry_the_authored_rate(self, tmp_path, bundle_dir):
        cal = _cal(**self.AUTHORED_CAL)
        out = tmp_path / "bound"
        cands, rows = self._run(bundle_dir, cal=cal)
        bc.write_outputs(out, "r", cands, rows, cal, 0.0, "x.json")
        recorded = json.loads((out / "bound.json").read_text())["calibration"]
        assert recorded["authored_stream_gbs"] == 500.0 and recorded["authored_stream"] is None
        probe = {"gbs": 400.0, "language": "triton", "config": {"block": 2048}}
        bc.write_outputs(out, "r", cands, rows, _cal(), 0.0, "x.json", authored_stream=probe)
        recorded = json.loads((out / "bound.json").read_text())["calibration"]
        assert recorded["authored_stream_gbs"] == 400.0 and recorded["authored_stream"] == probe


class TestShareAttribution:
    def test_an_op_discovery_charged_zero_is_charged_the_kernels_it_launched(self, bundle_dir):
        report = _report()
        report["op_share"][LIN] = {"device_us": 0.0, "share_pct": 0.0}
        cands, rows = bc.run(report, _resolution(bundle_dir), _measurements(), _cal(), "r")
        lin = _by_id(cands)[_ids()["lin"]]
        assert lin.device_us_op == 7000.0 and lin.share_source == "resolution:launched"
        assert lin.share_pct == pytest.approx(70.0)
        assert lin.t_dev == pytest.approx(17.5)
        assert _final(rows, lin.candidate_id, "library_call")["status"] == "ACCEPT"

    def test_an_op_with_no_share_from_either_stage_is_not_a_candidate(self, bundle_dir):
        report = _report()
        report["op_share"][LIN] = {"device_us": 0.0, "share_pct": 0.0}
        res = _resolution(bundle_dir)
        res["aten::linear"]["launched"] = []
        cands, _ = bc.run(report, res, _measurements(), _cal(), "r")
        assert _ids()["lin"] not in _by_id(cands)


class TestUnroutable:
    def test_every_mechanism_is_rejected_with_its_gate_and_arithmetic(self, outcome):
        cid = _ids()["argmax"]
        rows = [r for r in outcome[1] if r["candidate_id"] == cid]
        assert rows[-1]["status"] == "UNROUTABLE"
        assert (
            rows[-1]["mechanisms_evaluated"]
            == len(bc.MECHANISMS)
            == rows[-1]["mechanisms_rejected"]
        )
        for mech in bc.MECHANISMS:
            last = _final(rows, cid, mech)
            assert last["status"] == "REJECT", mech
            assert (
                last["gate"]
                and last["arithmetic"]
                and last["where"].startswith("bound_candidates.py:")
            )
            assert last["needs"]["quantity"] == last["lhs"]
        reasons = {mech: _final(rows, cid, mech)["gate"] for mech in bc.MECHANISMS}
        assert reasons["apply_substitution"] == "definition_exists"
        assert reasons["upstream_report"] == "worklist_eligible"
        assert reasons["provider_patch"] == "class_admits"
        assert reasons["layout_transform"] == "class_admits"
        # The class admits authoring; with no authored-stream probe the bound a written
        # kernel could reach is unmeasured, and the mechanism is unavailable, not at peak.
        assert reasons["authored_callsite"] == "attainable_calibrated"
        assert reasons["authored_apply"] == "attainable_calibrated"

    def test_triton_kernel_with_device_time_is_a_candidate_with_lines(self, outcome):
        cid = _ids()["triton"]
        rows = [r for r in outcome[1] if r["candidate_id"] == cid]
        assert rows and rows[-1]["status"] == "UNROUTABLE"
        assert _by_id(outcome[0])[cid].t_dev == pytest.approx(1.0)


class TestLogContract:
    def test_every_candidate_with_share_has_a_line_and_share_free_ops_have_none(self, outcome):
        text = "\n".join(bc.format_line(r) for r in outcome[1])
        for cid in _ids().values():
            assert f",{cid}," in text
        assert "aten.detach.default" not in text

    def test_lines_have_thirteen_comma_free_fields_and_a_three_token_arithmetic(self, outcome):
        for r in outcome[1]:
            line = bc.format_line(r)
            fields = line.split(",")
            assert len(fields) == bc.LOG_FIELDS
            assert fields[0] == "fib_bound" and fields[1] == "v1" and fields[2] == "test-run"
            if fields[9] not in ("ACCEPT", "UNROUTABLE"):
                assert fields[10] in ("PASS", "REJECT")
                assert len(fields[11].split(" ")) == 3
                assert fields[12].startswith("bound_candidates.py:")
            elif fields[9] == "UNROUTABLE":
                assert fields[8] == "-" and fields[12] == "-"

    def test_the_three_line_forms_match_the_plan(self, outcome):
        lines = [bc.format_line(r) for r in outcome[1]]
        accept = next(line for line in lines if ",ACCEPT," in line)
        assert (
            accept.split(",")[9:12]
            == [
                "ACCEPT",
                accept.split(",")[10],
                accept.split(",")[11],
            ]
            and accept.split(",")[10].startswith("ceiling_us=")
            and accept.split(",")[11].startswith("worth=")
        )
        assert accept.split(",")[12].startswith("mechanism_us=")
        unroutable = next(line for line in lines if ",UNROUTABLE," in line)
        assert unroutable.split(",")[8:] == [
            "-",
            "UNROUTABLE",
            f"{len(bc.MECHANISMS)} mechanisms evaluated",
            f"{len(bc.MECHANISMS)} rejected",
            "-",
        ]

    def test_unmeasured_values_print_as_none_never_as_zero(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            {},
            _cal(dispatch_us=None),
            "r",
            definitions_matching=_defs,
        )
        text = "\n".join(bc.format_line(r) for r in rows)
        assert "spread_us=None" in text and "dispatch_us=None" in text
        assert "spread_us=0" not in text and "dispatch_us=0" not in text

    def test_a_comma_in_a_field_is_refused_rather_than_written(self, outcome):
        row = dict(outcome[1][0])
        row["op"] = "a,b"
        with pytest.raises(ValueError):
            bc.format_line(row)

    def test_worklist_is_exactly_the_accept_rows_in_worth_order(self, outcome):
        rows = outcome[1]
        wl = bc.worklist(rows)
        assert [r["status"] for r in wl] == ["ACCEPT"] * len(wl)
        assert len(wl) == sum(1 for r in rows if r["status"] == "ACCEPT")
        worths = [r["worth"] for r in wl]
        assert worths == sorted(worths, reverse=True)
        ids = _ids()
        # library_call on the GEMM: (17.5 - max(t_mem 8.4, t_cmp 0.67, launch 2.0)) * 400 / 10000
        assert (wl[0]["candidate_id"], wl[0]["mechanism"]) == (ids["lin"], "library_call")
        assert [(r["candidate_id"], r["mechanism"]) for r in wl[1:]] == [
            (ids["rms"], "fusion_callsite"),
            (ids["rms"], "provider_patch"),
            (ids["rms"], "fusion_apply"),
        ]

    def test_round_trip_the_log_reproduces_the_worklist(self, outcome, tmp_path):
        cands, rows = outcome
        bc.write_outputs(tmp_path, "test-run", cands, rows, _cal(), 0.0005, "discovered.json")
        log_text = (tmp_path / "bound.log").read_text()
        emitted = json.loads((tmp_path / "worklist.json").read_text())
        rebuilt = bc.worklist_from_log(log_text)
        assert [(r["candidate_id"], r["mechanism"]) for r in rebuilt] == [
            (r["candidate_id"], r["mechanism"]) for r in emitted
        ]
        for a, b in zip(rebuilt, emitted):
            assert a["worth"] == pytest.approx(b["worth"], rel=1e-5)
            assert a["ceiling_us"] == pytest.approx(b["ceiling_us"], rel=1e-5)
            assert a["mechanism_us"] == pytest.approx(b["mechanism_us"])
        records = bc.parse_log(log_text)
        assert len(records) == len(rows)
        assert {r["status"] for r in records} == {"PASS", "REJECT", "ACCEPT", "UNROUTABLE"}
        sidecar = json.loads((tmp_path / "bound.json").read_text())
        assert sidecar["authoritative"] == "bound.json"
        assert len(sidecar["rows"]) == len(rows)
        assert all("needs" in r for r in sidecar["rows"] if r["status"] == "REJECT")

    def test_parse_log_refuses_a_malformed_line(self):
        with pytest.raises(ValueError):
            bc.parse_log("fib_bound,v1,only,a,few,fields\n")


class TestDefinitionMatching:
    def _defs(self):
        D = SimpleNamespace
        return [
            D(name="silu_and_mul_d3072", op_type="activation"),
            D(name="gelu_and_mul_d3072", op_type="activation"),
            D(name="rmsnorm_h1024", op_type="rmsnorm"),
            D(name="fused_add_rmsnorm_h1024", op_type="rmsnorm"),
            D(name="gemm_n4096_k1024", op_type="gemm"),
        ]

    def test_the_ops_own_name_selects_its_definitions(self):
        names = lambda bare: [d.name for d in bc._same_operation(bare, self._defs())]  # noqa: E731
        assert names("silu_and_mul") == ["silu_and_mul_d3072"]
        assert names("rms_norm") == ["rmsnorm_h1024", "fused_add_rmsnorm_h1024"]
        assert names("fused_add_rms_norm") == ["fused_add_rmsnorm_h1024"]

    def test_falls_back_to_the_op_type_vocabulary_when_no_name_carries_the_op(self):
        names = [d.name for d in bc._same_operation("linear", self._defs())]
        assert names == ["gemm_n4096_k1024"]
        assert bc._same_operation("argmax", self._defs()) == []

    def test_binding_checks_dtype_rank_constant_axes_and_consistent_var_axes(self):
        spec = lambda shape, dtype="bfloat16": SimpleNamespace(
            shape=shape, dtype=SimpleNamespace(value=dtype)
        )  # noqa: E731
        axes = {
            "batch_size": SimpleNamespace(type="var"),
            "hidden_size": SimpleNamespace(type="const", value=1024),
        }
        d = SimpleNamespace(
            inputs={
                "hidden_states": spec(["batch_size", "hidden_size"]),
                "weight": spec(["hidden_size"]),
            },
            axes=axes,
        )
        assert bc._binds(
            d, [((4, 1024), "bfloat16"), ((4, 1024), "bfloat16"), ((1024,), "bfloat16")]
        )
        assert not bc._binds(d, [((4, 16, 128), "bfloat16"), ((128,), "bfloat16")])  # rank
        assert not bc._binds(d, [((4, 2048), "bfloat16"), ((2048,), "bfloat16")])  # const axis
        assert not bc._binds(d, [((4, 1024), "float16"), ((1024,), "float16")])  # dtype
        assert not bc._binds(d, [((4, 1024), "bfloat16")])  # too few tensors


class TestHarnessMatching:
    def test_harness_for_matches_op_and_shapes(self, tmp_path):
        (tmp_path / "h1.py").write_text(
            'OP = "_C.rms_norm.default"\nCALLS = 4\n'
            "def get_inputs():\n    return [torch.randn([4, 1024], dtype=torch.bfloat16),"
            " torch.randn([4, 1024], dtype=torch.bfloat16), torch.randn([1024], dtype=torch.bfloat16)]\n"
        )
        (tmp_path / "h2.py").write_text(
            'OP = "_C.rms_norm.default"\nCALLS = 4\n'
            "def get_inputs():\n    return [torch.randn([16, 1024]), torch.randn([16, 1024]), torch.randn([1024])]\n"
        )
        cands = bc.candidates_from_report(_report(), {}, {})
        rms = next(c for c in cands if c.op == RMS)
        assert bc.harness_for(rms, tmp_path) == tmp_path / "h1.py"

    def test_benchmark_keys_are_parsed_from_the_contract(self):
        keys = bc.parse_benchmark_keys(
            "BUILD: OK\nCANDIDATE_US: 12.50\nSPREAD_PCT: 4.0\nVERDICT: NOISE\nDONE\n"
        )
        assert keys == {
            "BUILD": "OK",
            "CANDIDATE_US": "12.50",
            "SPREAD_PCT": "4.0",
            "VERDICT": "NOISE",
        }


class TestUnreadableInputsAreNoneNeverZero:
    """An input this run could not read is unavailable, which the log renders as None.

    Rendering it as a count of zero would certify an absence nobody checked -- "no
    definition binds", "no preset matches" -- and the mechanism would be recorded as
    inexpressible rather than unevaluated.
    """

    def test_no_dataset_renders_definitions_matching_none(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(),
            "r",
            definitions_matching=lambda c: None,
        )
        r = _final(rows, _ids()["rms"], "apply_substitution")
        assert (r["gate"], r["status"]) == ("definition_exists", "REJECT")
        assert r["arithmetic"] == "definitions_matching=None > required=0"
        assert r["detail"]["dataset"] == "unreadable"

    def test_no_preset_list_and_no_post_op_catalog_renders_expressions_none(self, bundle_dir):
        _, rows = bc.run(_report(), _resolution(bundle_dir), _measurements(), _cal(), "r")
        r = _final(rows, _ids()["rms"], "fusion_callsite")
        assert (r["gate"], r["status"]) == ("epilogue_expressible", "REJECT")
        assert r["arithmetic"] == "expressions=None > required=0"
        assert r["detail"]["presets"] == "unreadable"

    def test_a_read_and_empty_preset_list_is_still_zero(self, bundle_dir):
        _, rows = bc.run(
            _report(), _resolution(bundle_dir), _measurements(), _cal(), "r", presets=[]
        )
        r = _final(rows, _ids()["rms"], "fusion_callsite")
        assert r["arithmetic"] == "expressions=0 > required=0"


def _routing(tmp_path, bundle_dir, model="test/model"):
    """A complete routing on disk: report, bound.json with provenance, and its candidates."""
    report_path = tmp_path / "discovered.json"
    report = dict(_report(), model=model)
    report_path.write_text(json.dumps(report))
    cands, rows = bc.run(
        report,
        _resolution(bundle_dir),
        _measurements(),
        _cal(),
        run_id="test-run",
        presets=PRESETS,
        post_op_algorithms={"relu", "gelu_erf", "swish"},
        definitions_matching=_defs,
    )
    out = tmp_path / "bound"
    bc.write_outputs(
        out,
        "test-run",
        cands,
        rows,
        _cal(),
        0.0005,
        str(report_path),
        report_sha1=bc.report_digest(report_path),
        model=model,
    )
    return out, report_path


def _harness(tmp_path, op=RMS, calls=400, shapes=("[4, 1024]", "[4, 1024]", "[1024]")):
    lines = "\n".join(
        f"        torch.randn({s}, dtype=torch.bfloat16, device=device)," for s in shapes
    )
    path = tmp_path / "h.py"
    path.write_text(
        f'OP = "{op}"\nCALLS = {calls}\n\ndef get_inputs():\n    device = "cpu"\n'
        f"    return [\n{lines}\n    ]\n"
    )
    return path


class TestRoutingConsumers:
    """What a later stage asks before it measures a (candidate, mechanism) pair."""

    def test_accepted_pair_returns_its_ceiling_and_worth(self, tmp_path, bundle_dir):
        out, _ = _routing(tmp_path, bundle_dir)
        bound = bc.load_bound(out)
        row = bc.require_routed(bound, _ids()["rms"], "provider_patch")
        assert row["routing"] == bc.ROUTING_OK
        assert row["ceiling_us"] == pytest.approx(3.0) and row["worth"] is not None
        assert row["run_id"] == "test-run"

    def test_rejected_pair_names_the_gate_the_arithmetic_and_what_must_move(
        self, tmp_path, bundle_dir
    ):
        out, _ = _routing(tmp_path, bundle_dir)
        bound = bc.load_bound(out)
        with pytest.raises(bc.RoutingRefused) as exc:
            bc.require_routed(bound, _ids()["rms"], "apply_substitution")
        f = exc.value.fields
        assert f["routing"] == bc.ROUTING_REJECTED
        assert f["verdict"] == bc.VERDICT_ROUTING_REJECTED
        assert f["gate"] == "net_positive"
        assert f["arithmetic"] == "ceiling_us=-3 > spread_us=0.2"
        assert "ceiling_us must increase" in f["needs"]
        assert list(f)[-1] == "verdict"  # VERDICT is the last key printed
        assert "do not measure past the gate" in exc.value.why

    def test_unknown_candidate_or_mechanism_is_unevaluated_not_accepted(self, tmp_path, bundle_dir):
        out, _ = _routing(tmp_path, bundle_dir)
        bound = bc.load_bound(out)
        with pytest.raises(bc.RoutingRefused) as exc:
            bc.require_routed(bound, "nosuchcand", "provider_patch")
        assert exc.value.fields["routing"] == bc.ROUTING_UNEVALUATED
        with pytest.raises(bc.RoutingRefused, match="not a mechanism"):
            bc.require_routed(bound, _ids()["rms"], "teleport")

    def test_harness_is_tied_to_its_candidate_by_op_and_shapes(self, tmp_path, bundle_dir):
        out, _ = _routing(tmp_path, bundle_dir)
        bound = bc.load_bound(out)
        c = bc.candidate_for_harness(bound, _harness(tmp_path))
        assert c["candidate_id"] == _ids()["rms"]
        row = bc.routed_from_harness(out, "provider_patch", _harness(tmp_path))
        assert row["routing"] == bc.ROUTING_OK and row["candidate"] == _ids()["rms"]

    def test_harness_of_another_run_or_of_an_op_without_share_is_refused(
        self, tmp_path, bundle_dir
    ):
        out, _ = _routing(tmp_path, bundle_dir)
        bound = bc.load_bound(out)
        with pytest.raises(bc.RoutingRefused) as exc:
            bc.candidate_for_harness(bound, _harness(tmp_path, shapes=("[8, 1024]",)))
        assert exc.value.fields["routing"] == bc.ROUTING_UNEVALUATED
        # Same (op, shape) but a different call count: two discovery runs, and worth was
        # computed from the other one.
        with pytest.raises(bc.RoutingRefused) as exc:
            bc.candidate_for_harness(bound, _harness(tmp_path, calls=7))
        assert exc.value.fields["verdict"] == bc.VERDICT_STALE_INPUT

    def test_a_rerun_discovery_voids_the_routing(self, tmp_path, bundle_dir):
        out, report_path = _routing(tmp_path, bundle_dir)
        bound = bc.load_bound(out)
        assert bc.check_bound_provenance(bound) == report_path
        report_path.write_text(json.dumps(dict(_report(), model="test/model", extra=1)))
        with pytest.raises(bc.RoutingRefused) as exc:
            bc.check_bound_provenance(bc.load_bound(out))
        f = exc.value.fields
        assert (f["routing"], f["verdict"]) == (bc.ROUTING_STALE, bc.VERDICT_STALE_INPUT)
        assert f["report_sha1"] != f["expected_sha1"]
        report_path.unlink()
        with pytest.raises(bc.RoutingRefused, match="not there"):
            bc.check_bound_provenance(bc.load_bound(out))

    def test_missing_or_foreign_bound_json_is_refused(self, tmp_path):
        with pytest.raises(bc.RoutingRefused, match="no routing"):
            bc.load_bound(tmp_path)
        (tmp_path / "bound.json").write_text(json.dumps({"rows": []}))
        with pytest.raises(bc.RoutingRefused, match="regenerate"):
            bc.load_bound(tmp_path)
        (tmp_path / "bound.json").write_text(
            json.dumps({"format": "fib_bound/v1", "rows": [], "run_id": "x", "report": "r"})
        )
        with pytest.raises(bc.RoutingRefused, match="no report_sha1"):
            bc.check_bound_provenance(bc.load_bound(tmp_path))


class TestMainRefusesMissingInputs:
    """The CLI halts on an input it cannot price from, instead of writing a log about it."""

    def _inputs(self, tmp_path, bundle_dir, report=None):
        report_path = tmp_path / "discovered.json"
        report_path.write_text(json.dumps(report if report is not None else _report()))
        res = tmp_path / "resolution.json"
        res.write_text(json.dumps({"ops": list(_resolution(bundle_dir).values())}))
        cal = tmp_path / "cal.json"
        cal.write_text(json.dumps(vars(_cal())))
        return report_path, res, cal

    def _argv(self, tmp_path, report, res, cal, *extra):
        return [
            "--report",
            str(report),
            "--resolution",
            str(res),
            "--out-dir",
            str(tmp_path / "out"),
            "--calibration-json",
            str(cal),
            "--dataset",
            str(tmp_path / "no-dataset"),
            "--xe-fuse",
            str(tmp_path / "no-xe-fuse"),
            "--onednn-include",
            str(tmp_path / "no-header"),
            *extra,
        ]

    def test_passing_run_writes_provenance_a_consumer_can_check(self, tmp_path, bundle_dir):
        report, res, cal = self._inputs(tmp_path, bundle_dir)
        bc.main(self._argv(tmp_path, report, res, cal))
        bound = bc.load_bound(tmp_path / "out")
        assert bound["report_sha1"] == bc.report_digest(report) == bound["run_id"]
        assert bc.check_bound_provenance(bound) == report

    def test_resolution_is_required_and_must_exist(self, tmp_path, bundle_dir):
        report, res, cal = self._inputs(tmp_path, bundle_dir)
        with pytest.raises(SystemExit):
            bc.main(["--report", str(report), "--out-dir", str(tmp_path / "o")])
        with pytest.raises(SystemExit, match="resolution not found"):
            bc.main(self._argv(tmp_path, report, tmp_path / "missing.json", cal))
        res.write_text(json.dumps({"ops": []}))
        with pytest.raises(SystemExit, match="resolves no ops"):
            bc.main(self._argv(tmp_path, report, res, cal))
        assert not (tmp_path / "out" / "bound.json").exists()

    def test_report_without_device_time_is_refused(self, tmp_path, bundle_dir):
        zero = dict(_report(), device_time_total_us=0.0)
        report, res, cal = self._inputs(tmp_path, bundle_dir, zero)
        with pytest.raises(SystemExit, match="not positive"):
            bc.main(self._argv(tmp_path, report, res, cal))
        failed = dict(_report(), device_time_error="RuntimeError: profiler unavailable")
        report, res, cal = self._inputs(tmp_path, bundle_dir, failed)
        with pytest.raises(SystemExit, match="profiler unavailable"):
            bc.main(self._argv(tmp_path, report, res, cal))
        assert not (tmp_path / "out" / "bound.json").exists()

    def test_named_but_missing_gaps_or_measurements_are_refused(self, tmp_path, bundle_dir):
        report, res, cal = self._inputs(tmp_path, bundle_dir)
        with pytest.raises(SystemExit, match="gap analysis not found"):
            bc.main(self._argv(tmp_path, report, res, cal, "--gaps", str(tmp_path / "g.json")))
        with pytest.raises(SystemExit, match="measurements not found"):
            bc.main(
                self._argv(tmp_path, report, res, cal, "--measurements", str(tmp_path / "m.json"))
            )


# --------------------------------------------------------------------------- below the bound


def _lin_measured(t_dev, source, spread=1.0, **extra):
    """A GEMM measurement: 8.4 MB at 1000 GB/s bounds it at ~8.4 us (launch floor 2, t_cmp 0.67).
    The wall time sits just over the device time, as it does for a GEMM whose call is not
    dominated by its launch."""
    ids = _ids()
    return {
        ids["lin"]: {
            "t_host_us": t_dev + 1.0,
            "spread_us": spread,
            "t_dev_us": t_dev,
            "t_dev_source": source,
            **extra,
        }
    }


class TestBelowTheBound:
    """A t_dev under the memory bound is a measurement of the wrong quantity, and is named."""

    def test_a_resident_t_dev_under_the_bound_is_below_the_bound_not_unclassified(self, bundle_dir):
        cands, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _lin_measured(5.0, "profiler:harness", working_set_bytes=8396800),
            _cal(),
            "r",
        )
        c = _by_id(cands)[_ids()["lin"]]
        assert c.bound == pytest.approx(8.3968)
        assert c.regime == bc.R_BELOW_BOUND
        assert c.regime_test.startswith("bound_us-t_dev_us=3.3968 > spread_us=1")
        note = c.notes[-1]
        assert "t_mem_us=8.3968" in note and "bytes_min=8396800" in note
        assert "last-level cache" in note and "--measure" in note
        assert "8396800 bytes" in note
        # The row carries the regime, and every bound-based gate still names its numbers.
        r = _final(rows, c.candidate_id, "library_call")
        assert r["regime"] == bc.R_BELOW_BOUND
        assert (r["gate"], r["status"]) == ("headroom", "REJECT")
        assert r["lhs_value"] == pytest.approx(5.0 - 8.3968)

    def test_a_streaming_t_dev_at_the_bound_classifies(self, bundle_dir):
        cands, _ = bc.run(
            _report(),
            _resolution(bundle_dir),
            _lin_measured(9.0, "profiler:harness-streaming", t_dev_resident_us=5.0),
            _cal(),
            "r",
        )
        c = _by_id(cands)[_ids()["lin"]]
        assert c.regime == bc.R_AT_BOUND
        assert c.t_dev == 9.0 and c.t_dev_resident == 5.0
        assert c.metrics()["t_dev_resident_us"] == 5.0

    def test_a_streaming_t_dev_over_the_bound_is_memory_bound_inefficient(self, bundle_dir):
        cands, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _lin_measured(12.0, "profiler:harness-streaming", t_dev_resident_us=5.0),
            _cal(),
            "r",
        )
        c = _by_id(cands)[_ids()["lin"]]
        assert c.regime == bc.R_MEM_INEFF
        assert _final(rows, c.candidate_id, "library_call")["status"] == "ACCEPT"

    def test_within_spread_of_the_bound_is_not_below_it(self, bundle_dir):
        cands, _ = bc.run(
            _report(), _resolution(bundle_dir), _lin_measured(7.5, "profiler:harness"), _cal(), "r"
        )
        assert _by_id(cands)[_ids()["lin"]].regime == bc.R_AT_BOUND

    def test_a_streaming_number_under_the_bound_points_at_bytes_min(self, bundle_dir):
        cands, _ = bc.run(
            _report(),
            _resolution(bundle_dir),
            _lin_measured(5.0, "profiler:harness-streaming"),
            _cal(),
            "r",
        )
        c = _by_id(cands)[_ids()["lin"]]
        assert c.regime == bc.R_BELOW_BOUND
        assert "--bytes-min" in c.notes[-1] and "last-level cache" not in c.notes[-1]

    def test_the_regime_token_is_log_safe(self, bundle_dir):
        _, rows = bc.run(
            _report(), _resolution(bundle_dir), _lin_measured(5.0, "profiler:harness"), _cal(), "r"
        )
        text = "\n".join(bc.format_line(r) for r in rows)
        assert f",{bc.R_BELOW_BOUND}," in text
        assert bc.worklist_from_log(text) == bc.worklist(rows)


class TestStreamingMeasurement:
    def test_merge_keeps_the_resident_number_and_takes_the_streaming_one(self):
        record = {"t_dev_us": 5.0, "t_dev_source": "profiler:harness", "kernels": {"k": 1.0}}
        stream = {
            "t_dev_us": 9.0,
            "t_dev_source": "profiler:harness-streaming",
            "kernels": {"k": 1.8},
            "working_set_bytes": 8396800,
            "pool_bytes": 2 * 18874368,
            "copies": 5,
            "profiled_calls": 210,
        }
        out = bc.merge_streaming_profile(record, stream)
        assert out["t_dev_us"] == 9.0 and out["t_dev_source"] == "profiler:harness-streaming"
        assert out["t_dev_resident_us"] == 5.0 and out["kernels"] == {"k": 1.0}
        assert out["working_set_bytes"] == 8396800
        assert out["stream"] == {"pool_bytes": 2 * 18874368, "copies": 5, "profiled_calls": 210}
        # Merging again does not overwrite the resident number with the streaming one.
        again = bc.merge_streaming_profile(out, stream)
        assert again["t_dev_resident_us"] == 5.0

    def test_a_failed_streaming_pass_leaves_a_resident_number_labelled_resident(self):
        record = {"t_dev_us": 5.0, "t_dev_source": "profiler:harness"}
        out = bc.merge_streaming_profile(record, {"profile_error": "no room"})
        assert out["t_dev_us"] == 5.0 and out["t_dev_source"] == "profiler:harness"
        assert out["t_dev_resident_us"] == 5.0
        assert out["stream"] == {"error": "no room"}

    def test_measure_adds_the_streaming_pass_to_an_existing_record_without_re_timing(
        self, tmp_path, monkeypatch
    ):
        ids = _ids()
        harness = _harness(tmp_path)
        out = tmp_path / "measurements.json"
        prior = {
            "harness": str(harness),
            "t_host_us": 8.0,
            "spread_us": 0.2,
            "t_dev_us": 5.0,
            "t_dev_source": "profiler:harness",
        }
        out.write_text(json.dumps({ids["rms"]: prior}))
        calls = []

        def fake_profile(h, interp, n, warm_s=None, pool_bytes=None):
            calls.append((pathlib.Path(h).name, n, pool_bytes))
            return {
                "t_dev_us": 6.0,
                "t_dev_source": "profiler:harness-streaming",
                "kernels": {},
                "profiled_calls": n,
                "working_set_bytes": 26624,
                "pool_bytes": pool_bytes,
                "copies": 3,
            }

        monkeypatch.setattr(bc, "profile_harness", fake_profile)
        monkeypatch.setattr(
            bc.subprocess, "run", lambda *a, **k: pytest.fail("the host arm was re-timed")
        )
        cands = bc.candidates_from_report(_report(), _resolution(tmp_path), {})
        rms = [c for c in cands if c.candidate_id == ids["rms"]]
        got = bc.measure_harnesses(rms, tmp_path, out, "python", rounds=2, calls=3, pool_bytes=1000)
        assert calls == [("h.py", 6, 1000)]
        rec = got[ids["rms"]]
        assert rec["t_host_us"] == 8.0 and rec["t_dev_resident_us"] == 5.0
        assert rec["t_dev_us"] == 6.0 and rec["t_dev_source"] == "profiler:harness-streaming"
        # A second --measure finds the pass recorded and does nothing.
        bc.measure_harnesses(rms, tmp_path, out, "python", rounds=2, calls=3, pool_bytes=1000)
        assert len(calls) == 1

    def test_without_a_pool_an_existing_record_is_left_alone(self, tmp_path, monkeypatch):
        ids = _ids()
        harness = _harness(tmp_path)
        out = tmp_path / "measurements.json"
        out.write_text(json.dumps({ids["rms"]: {"harness": str(harness), "t_dev_us": 5.0}}))
        monkeypatch.setattr(bc, "profile_harness", lambda *a, **k: pytest.fail("profiled"))
        cands = bc.candidates_from_report(_report(), _resolution(tmp_path), {})
        got = bc.measure_harnesses(cands, tmp_path, out, "python", rounds=1, calls=1)
        assert got[ids["rms"]] == {"harness": str(harness), "t_dev_us": 5.0}


# --------------------------------------------------------------------------- layout_transform


class TestLayoutTransformNomination:
    """The pitch rule nominates a layout transform on its own; the later gates price it."""

    # The GEMM's weight is 4096x1024 bf16: a contiguous row pitch of 2048 bytes.

    def test_a_pitch_on_the_period_admits_and_the_detail_names_the_tensors(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(channel_period_bytes=2048),
            "r",
        )
        first = _rows(rows, _ids()["lin"], "layout_transform")[0]
        assert (first["gate"], first["status"]) == ("class_admits", "PASS")
        assert first["arithmetic"] == "layout_nominations=2 > required=0"
        assert first["detail"]["channel_period_bytes"] == 2048
        assert first["detail"]["camping"] == [
            "4x1024:pitch_bytes=2048",
            "4096x1024:pitch_bytes=2048",
        ]

    def test_a_pitch_off_the_period_is_rejected_with_zero(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(channel_period_bytes=1536),
            "r",
        )
        r = _final(rows, _ids()["lin"], "layout_transform")
        assert (r["gate"], r["status"]) == ("class_admits", "REJECT")
        assert r["arithmetic"] == "layout_nominations=0 > required=0"
        assert r["detail"]["camping"] == []

    def test_an_unmeasured_period_is_none_never_no_camp(self, outcome):
        r = _final(outcome[1], _ids()["lin"], "layout_transform")
        assert (r["gate"], r["status"]) == ("class_admits", "REJECT")
        assert r["arithmetic"] == "layout_nominations=None > required=0"
        assert r["lhs_value"] is None and r["detail"]["channel_period_bytes"] is None
        assert r["detail"]["camping"] == "unmeasured period"

    def test_the_regime_still_admits_without_a_period(self, bundle_dir):
        ids = _ids()
        cands, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            {ids["rms"]: {"t_host_us": 8.0, "spread_us": 0.4}},
            _cal(timing_floor_us=0.1),
            "r",
            patterns={RMS: (64, 4096)},
            bw_pattern=lambda run, stride: 5.0,
        )
        assert _by_id(cands)[ids["rms"]].regime == bc.R_MEM_LAYOUT
        first = _rows(rows, ids["rms"], "layout_transform")[0]
        assert (
            first["status"] == "PASS" and first["arithmetic"] == "layout_nominations=1 > required=0"
        )

    def test_the_period_comes_from_the_argument_before_the_record(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(channel_period_bytes=1536),
            "r",
            channel_period_bytes=2048,
        )
        assert _rows(rows, _ids()["lin"], "layout_transform")[0]["status"] == "PASS"

    def test_admitted_and_priced_headroom_to_the_contiguous_bound(self, bundle_dir):
        # t_dev 17.5 (op average) over a bound of 8.4: the transform is worth up to 9.1 us.
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _measurements(),
            _cal(channel_period_bytes=2048),
            "r",
        )
        r = _final(rows, _ids()["lin"], "layout_transform")
        assert r["status"] == "ACCEPT"
        assert r["ceiling_us"] == pytest.approx(17.5 - 8.3968) and r["mechanism_us"] == 0.0

    def test_admitted_then_rejected_on_measured_grounds_at_the_bound(self, bundle_dir):
        # Streaming t_dev within spread of the bound: admitted, measurable, headroom 0.6 us,
        # and net_positive says that is inside the spread. The gate and both sides are named.
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _lin_measured(9.0, "profiler:harness-streaming"),
            _cal(channel_period_bytes=2048),
            "r",
        )
        chain = _rows(rows, _ids()["lin"], "layout_transform")
        assert [(g["gate"], g["status"]) for g in chain] == [
            ("class_admits", "PASS"),
            ("measurable", "PASS"),
            ("headroom", "PASS"),
            ("net_positive", "REJECT"),
        ]
        assert chain[-1]["arithmetic"] == "ceiling_us=0.6032 > spread_us=1"

    def test_admitted_then_rejected_at_headroom_when_t_dev_is_below_the_bound(self, bundle_dir):
        _, rows = bc.run(
            _report(),
            _resolution(bundle_dir),
            _lin_measured(5.0, "profiler:harness"),
            _cal(channel_period_bytes=2048),
            "r",
        )
        r = _final(rows, _ids()["lin"], "layout_transform")
        assert (r["gate"], r["status"]) == ("headroom", "REJECT")
        assert r["regime"] == bc.R_BELOW_BOUND
        assert r["arithmetic"].startswith("headroom_us=-3.3968 > 0")

    def test_bound_json_records_the_period_it_priced_with(self, tmp_path, bundle_dir):
        out, _ = _routing(tmp_path, bundle_dir)
        assert "channel_period_bytes" in json.loads((out / "bound.json").read_text())["calibration"]
