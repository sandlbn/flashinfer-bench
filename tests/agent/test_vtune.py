"""VTune wrapper: discovery, GPU selection, prerequisite checks and failure translation.

None of this needs a GPU or VTune installed; every external fact comes from a temporary
directory or a monkeypatch.
"""

import os
import pathlib
import stat

import pytest

from flashinfer_bench.agents import vtune as vt


def _executable(path: pathlib.Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """No vtune anywhere: empty PATH, env unset, vendor default pointed at nothing."""
    monkeypatch.delenv(vt.VTUNE_ENV, raising=False)
    monkeypatch.delenv("ONEAPI_ROOT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vt, "_VENDOR_DEFAULTS", (tmp_path / "nonexistent" / "vtune",))
    return tmp_path


@pytest.fixture
def fake_install(tmp_path):
    """A `<root>/bin64/vtune` with a pinruntime whose libc++ names are percent-encoded."""
    root = tmp_path / "vtune-install"
    binary = _executable(root / "bin64" / "vtune")
    runtime = root / "lib64" / "pinruntime"
    runtime.mkdir(parents=True)
    for name in ("libc%2B%2B.so", "libc%2B%2Babi.so", "libxed.so", "libc-dynamic.so"):
        (runtime / name).write_text("")
    return root, binary


class TestFindVtune:
    def test_env_var_wins_over_path(self, nowhere, monkeypatch):
        on_path = _executable(nowhere / "bin" / "vtune")
        monkeypatch.setenv("PATH", str(nowhere / "bin"))
        from_env = _executable(nowhere / "install" / "bin64" / "vtune")
        monkeypatch.setenv(vt.VTUNE_ENV, from_env)
        assert vt.find_vtune() == from_env
        monkeypatch.delenv(vt.VTUNE_ENV)
        assert vt.find_vtune() == on_path

    def test_explicit_argument_wins_over_env_var(self, nowhere, monkeypatch):
        monkeypatch.setenv(vt.VTUNE_ENV, _executable(nowhere / "env" / "vtune"))
        explicit = _executable(nowhere / "arg" / "vtune")
        assert vt.find_vtune(explicit) == explicit

    def test_set_but_wrong_env_var_is_an_error_not_a_fallthrough(self, nowhere, monkeypatch):
        _executable(nowhere / "bin" / "vtune")
        monkeypatch.setenv("PATH", str(nowhere / "bin"))
        monkeypatch.setenv(vt.VTUNE_ENV, str(nowhere / "missing" / "vtune"))
        with pytest.raises(FileNotFoundError, match=vt.VTUNE_ENV):
            vt.find_vtune()
        assert vt.is_vtune_available() is False

    def test_oneapi_root_then_vendor_default(self, nowhere, monkeypatch):
        vendor = _executable(nowhere / "vendor" / "bin64" / "vtune")
        monkeypatch.setattr(vt, "_VENDOR_DEFAULTS", (pathlib.Path(vendor),))
        assert vt.find_vtune() == vendor
        oneapi = _executable(nowhere / "oneapi" / "vtune" / "latest" / "bin64" / "vtune")
        monkeypatch.setenv("ONEAPI_ROOT", str(nowhere / "oneapi"))
        assert vt.find_vtune() == oneapi

    def test_nothing_found_is_none_and_the_message_names_the_env_var(self, nowhere):
        assert vt.find_vtune() is None
        assert vt.is_vtune_available() is False
        message = vt.vtune_missing_message()
        assert vt.VTUNE_ENV in message and "PATH" in message

    def test_no_machine_specific_path_is_baked_in(self):
        for p in vt._VENDOR_DEFAULTS:
            assert os.sep + "home" + os.sep not in str(p)
            assert "intel-vtune-bin" not in str(p)


class TestTargetGpu:
    def test_uuid_decodes_vendor_device_and_pci_address(self):
        vendor, device, bdf = vt.bdf_from_uuid("86800be2-0000-0000-0400-000000000000")
        assert vendor == 0x8086 and device == 0xE20B and bdf == (0, 4, 0, 0)
        assert vt.bdf_from_uuid("not-a-uuid") is None

    def test_format_is_unpadded_decimal(self):
        assert vt.format_bdf((0, 4, 0, 0)) == "0:4:0.0"
        assert vt.format_bdf(None) == "?"

    def test_adapter_list_parses_vtune_context_values(self):
        text = "foo: 1\ngpuAdapterNameList: 0:0:2.0|Integrated Graphics;0:4:0.0|Discrete Card;\n"
        assert vt.parse_vtune_adapters(text) == [
            ("0:0:2.0", "Integrated Graphics"),
            ("0:4:0.0", "Discrete Card"),
        ]
        assert vt.parse_vtune_adapters("nothing here") == []

    def test_match_prefers_decimal_then_hex_and_refuses_ambiguity(self):
        assert vt.match_vtune_bdf((0, 4, 0, 0), ["0:0:2.0", "0:4:0.0"]) == "0:4:0.0"
        assert vt.match_vtune_bdf((0, 10, 0, 0), ["0:10:0.0"]) == "0:10:0.0"
        assert vt.match_vtune_bdf((0, 10, 0, 0), ["0:a:0.0"]) == "0:a:0.0"
        assert vt.match_vtune_bdf((0, 3, 0, 0), ["0:0:2.0", "0:4:0.0"]) is None
        # A hex digit anywhere in VTune's list means every entry is hex: bus 10 is "a".
        assert vt.match_vtune_bdf((0, 10, 0, 0), ["0:10:0.0", "0:a:0.0"]) == "0:a:0.0"
        assert vt.match_vtune_bdf((0, 16, 0, 0), ["0:10:0.0", "0:a:0.0"]) == "0:10:0.0"

    def test_sysfs_enumeration_keeps_intel_display_functions_only(self, tmp_path):
        devices = tmp_path / "bus" / "pci" / "devices"

        def add(name, cls, vendor, device, driver):
            d = devices / name
            d.mkdir(parents=True)
            (d / "class").write_text(cls)
            (d / "vendor").write_text(vendor)
            (d / "device").write_text(device)
            if driver:
                (tmp_path / driver).mkdir(exist_ok=True)
                os.symlink(tmp_path / driver, d / "driver")

        add("0000:00:02.0", "0x030000", "0x8086", "0x7d67", "i915")
        add("0000:04:00.0", "0x030000", "0x8086", "0xe20b", "xe")
        add("0000:05:00.0", "0x030000", "0x10de", "0x2684", "nvidia")
        add("0000:00:1f.3", "0x040300", "0x8086", "0x7f50", "snd_hda_intel")
        found = vt.intel_gpus_from_sysfs(tmp_path)
        assert found == [((0, 0, 2, 0), 0x7D67, "i915"), ((0, 4, 0, 0), 0xE20B, "xe")]
        assert vt.gpu_driver_for_bdf((0, 4, 0, 0), tmp_path) == "xe"
        assert vt.gpu_driver_for_bdf((0, 9, 0, 0), tmp_path) is None

    def test_select_uses_explicit_target_and_refuses_an_unmatched_device(self, monkeypatch):
        monkeypatch.setattr(
            vt, "vtune_gpu_adapters", lambda b: [("0:0:2.0", "A"), ("0:4:0.0", "B")]
        )
        knob, bdf, how = vt.select_target_gpu("xpu:0", "vtune", target_gpu="0:4:0.0")
        assert knob == "0:4:0.0" and bdf == (0, 4, 0, 0)
        monkeypatch.setattr(vt, "device_bdf", lambda device, sys_root="/sys": (0, 4, 0, 0))
        knob, bdf, how = vt.select_target_gpu("xpu:0", "vtune")
        assert knob == "0:4:0.0" and "B" in how
        monkeypatch.setattr(vt, "device_bdf", lambda device, sys_root="/sys": (0, 7, 0, 0))
        knob, _, how = vt.select_target_gpu("xpu:0", "vtune")
        assert knob is None and "target_gpu" in how
        monkeypatch.setattr(vt, "device_bdf", lambda device, sys_root="/sys": None)
        knob, _, how = vt.select_target_gpu("xpu:0", "vtune")
        assert knob is None and "2 adapters" in how
        monkeypatch.setattr(vt, "vtune_gpu_adapters", lambda b: [("0:4:0.0", "B")])
        knob, _, _ = vt.select_target_gpu("xpu:0", "vtune")
        assert knob == "0:4:0.0"


class TestPrerequisites:
    def test_percent_encoded_pin_runtime_is_reported_with_a_symlink_per_file(self, fake_install):
        root, binary = fake_install
        check = vt.check_pin_runtime(binary)
        assert check.ok is False
        assert "libc++.so" in check.fix and "libc++abi.so" in check.fix
        assert "ln -s 'libc%2B%2B.so'" in check.fix
        assert check.needs_root is (not os.access(root / "lib64" / "pinruntime", os.W_OK))
        assert vt.NEED_PIN in check.satisfies

    def test_correctly_named_pin_runtime_passes(self, fake_install):
        root, binary = fake_install
        runtime = root / "lib64" / "pinruntime"
        for enc, dec in (("libc%2B%2B.so", "libc++.so"), ("libc%2B%2Babi.so", "libc++abi.so")):
            os.symlink(runtime / enc, runtime / dec)
        assert vt.check_pin_runtime(binary).ok is True

    def test_observation_gate_per_driver(self, tmp_path):
        proc = tmp_path / "proc"
        (proc / "sys" / "dev" / "xe").mkdir(parents=True)
        (proc / "sys" / "dev" / "i915").mkdir(parents=True)
        (proc / "sys" / "dev" / "xe" / "observation_paranoid").write_text("1\n")
        (proc / "sys" / "dev" / "i915" / "perf_stream_paranoid").write_text("0\n")
        blocked = vt.check_gpu_observation((0, 4, 0, 0), proc_root=proc, driver="xe")
        assert blocked.ok is False and blocked.needs_root is True
        assert "dev.xe.observation_paranoid=0" in blocked.fix
        assert "CAP_PERFMON" in blocked.detail
        assert vt.NEED_COUNTERS in blocked.satisfies
        open_ = vt.check_gpu_observation((0, 0, 2, 0), proc_root=proc, driver="i915")
        assert open_.ok is True
        unknown = vt.check_gpu_observation((0, 5, 0, 0), proc_root=proc, driver="nvidia")
        assert unknown.ok is True

    def test_ptrace_scope_and_sampling_driver(self, tmp_path, fake_install):
        _, binary = fake_install
        proc = tmp_path / "proc"
        (proc / "sys" / "kernel" / "yama").mkdir(parents=True)
        (proc / "sys" / "kernel" / "yama" / "ptrace_scope").write_text("1\n")
        (proc / "modules").write_text("xe 4505600 45 - Live 0x0\ni915 5087232 11 - Live 0x0\n")
        scope = vt.check_ptrace_scope(proc)
        assert scope.ok is False and "ptrace_scope=0" in scope.fix and scope.needs_root
        driver = vt.check_sampling_driver(binary, proc)
        assert driver.ok is False and "insmod-sep" in driver.fix and driver.needs_root
        (proc / "modules").write_text("pax 16384 0 - Live 0x0\nsep5 1000 0 - Live 0x0\n")
        assert vt.check_sampling_driver(binary, proc).ok is True

    def test_mode_selects_which_checks_run(self, fake_install, monkeypatch):
        _, binary = fake_install
        monkeypatch.setattr(
            vt, "check_ptrace_scope", lambda *a, **k: vt.Prerequisite("p", True, "")
        )
        monkeypatch.setattr(vt, "check_metrics_library", lambda: vt.Prerequisite("m", True, ""))
        monkeypatch.setattr(
            vt, "check_gpu_observation", lambda *a, **k: vt.Prerequisite("g", False, "", "f", True)
        )
        timing = [c.name for c in vt.check_prerequisites(binary, "timing", (0, 4, 0, 0))]
        assert timing == ["pin-runtime", "p"]
        counters = [
            c.name for c in vt.check_prerequisites(binary, "characterization", (0, 4, 0, 0))
        ]
        assert counters == ["pin-runtime", "p", "m", "g"]
        with_bw = vt.check_prerequisites(binary, "characterization", (0, 4, 0, 0), bandwidth=True)
        assert with_bw[-1].name == "sampling-driver"
        report = vt.prerequisites_report(with_bw)
        assert "NEEDS ROOT" in report and "fix:" in report


class TestCommandAndFailures:
    def test_timing_mode_avoids_the_counter_stream_and_pins_the_gpu(self):
        cmd = vt.build_vtune_command("vtune", "timing", "0:4:0.0", "/tmp/r", ["python", "x.py"])
        assert cmd[:3] == ["vtune", "-collect", "gpu-offload"]
        assert "enable-characterization-insights=false" in cmd
        assert "target-gpu=0:4:0.0" in cmd
        assert "-target-gpu" not in cmd
        assert cmd[-5:] == ["-r", "/tmp/r", "--", "python", "x.py"]

    def test_characterization_knobs_and_validation(self):
        cmd = vt.build_vtune_command(
            "vtune",
            "characterization",
            "0:4:0.0",
            "r",
            ["a"],
            metric_group="full-compute",
            bandwidth=True,
            sampling_interval_ms=0.5,
            kernels_of_interest="gemm*",
        )
        joined = " ".join(cmd)
        assert "-collect gpu-hotspots" in joined
        assert "gpu-profiling-mode=characterization" in joined
        assert "characterization-mode=full-compute" in joined
        assert "collect-memory-bandwidth=true" in joined
        assert "gpu-sampling-interval=0.5" in joined
        assert "computing-tasks-of-interest=gemm*" in joined
        with pytest.raises(ValueError):
            vt.build_vtune_command(
                "vtune", "characterization", None, "r", ["a"], metric_group="nope"
            )
        with pytest.raises(ValueError):
            vt.build_vtune_command("vtune", "timing", None, "r", ["a"], bandwidth=True)

    def test_stall_and_bb_latency_are_source_analysis(self):
        stall = " ".join(vt.build_vtune_command("vtune", "stall", None, "r", ["a"]))
        assert "source-analysis=stall-sampling" in stall
        bb = " ".join(vt.build_vtune_command("vtune", "bb-latency", None, "r", ["a"]))
        assert "source-analysis=bb-latency" in bb

    @pytest.mark.parametrize(
        "text, expect",
        [
            ("pinbin: error while loading shared libraries: libc++.so", "pinruntime"),
            ('CANNOT LINK EXECUTABLE DEPENDENCIES: library "libc++.so" not found', "pinruntime"),
            ("vtune: Error: Cannot stop collection of GPU events", "observation"),
            ("ERROR perfrun.gpu <> - OpenIoStream returned error: 42", "observation"),
            ("Failed to connect to PMU reservation service (PAX)", "sampling driver"),
            ("set /proc/sys/kernel/yama/ptrace_scope to 0", "ptrace_scope"),
            ("neither libigdmd.so nor libmd.so was found", "Metrics Discovery"),
            ("ERROR cfgmgr <> - %ThisTargetTypeNotWorking", "-knob target-gpu="),
            ("vtune: Error: 0x40000024 (No data)", "no GPU data"),
        ],
    )
    def test_failure_signatures_translate(self, text, expect):
        causes = vt.explain_vtune_failure(text)
        assert causes and any(expect in c for c in causes)
        assert vt.explain_vtune_failure("all good") == []

    def test_collection_log_reader_keeps_errors_and_exit_code(self, tmp_path):
        log = tmp_path / "log"
        log.mkdir()
        (log / "perfrun-1.log").write_text(
            "1 [42] INFO perfrun.launcher <> - hello\n"
            "2 [42] ERROR perfrun.gpu <> - OpenIoStream returned error: 42\n"
            "3 [42] INFO perfrun.launcher <> - the profiled application was terminated  with exit code = 127\n"
        )
        text = vt.read_collection_logs(tmp_path)
        assert "OpenIoStream" in text and "exit code = 127" in text and "hello" not in text
        assert vt.read_collection_logs(tmp_path / "nope") == ""

    def test_list_modes_names_every_mode(self):
        text = vt.flashinfer_bench_list_vtune_modes()
        for name in vt.MODES:
            assert name in text


class TestRunWithoutBinaryOrPrerequisites:
    def test_tool_returns_the_actionable_message_before_touching_anything(self, nowhere):
        result = vt.flashinfer_bench_run_vtune("not-a-solution", "not-a-workload")
        assert vt.VTUNE_ENV in result
        result = vt.profile_command_with_vtune(["true"])
        assert vt.VTUNE_ENV in result

    def test_tool_reports_a_misconfigured_env_var_verbatim(self, nowhere, monkeypatch):
        monkeypatch.setenv(vt.VTUNE_ENV, str(nowhere / "nope"))
        result = vt.flashinfer_bench_run_vtune("not-a-solution", "not-a-workload")
        assert "does not name an executable" in result and vt.VTUNE_ENV in result

    def test_unknown_mode_is_refused(self, nowhere, monkeypatch):
        monkeypatch.setenv(vt.VTUNE_ENV, _executable(nowhere / "bin64" / "vtune"))
        assert "Unsupported VTune mode" in vt.profile_command_with_vtune(["true"], mode="magic")

    def test_missing_prerequisite_blocks_the_launch_and_names_the_fix(
        self, nowhere, fake_install, monkeypatch
    ):
        _, binary = fake_install
        monkeypatch.setattr(vt, "vtune_gpu_adapters", lambda b: [("0:4:0.0", "Card")])
        monkeypatch.setattr(vt, "device_bdf", lambda device, sys_root="/sys": (0, 4, 0, 0))
        monkeypatch.setattr(
            vt, "check_ptrace_scope", lambda *a, **k: vt.Prerequisite("p", True, "")
        )

        def boom(*a, **k):
            raise AssertionError("vtune must not be launched when a prerequisite is missing")

        monkeypatch.setattr(vt.subprocess, "run", boom)
        result = vt.profile_command_with_vtune(["true"], mode="timing", vtune_path=binary)
        assert "cannot run on this machine yet" in result
        assert "pin-runtime" in result and "ln -s" in result
        assert "Modes that need none of the missing items: none" in result

    def test_counter_modes_are_listed_as_runnable_when_only_the_gate_is_missing(
        self, fake_install, monkeypatch
    ):
        root, binary = fake_install
        runtime = root / "lib64" / "pinruntime"
        for enc, dec in (("libc%2B%2B.so", "libc++.so"), ("libc%2B%2Babi.so", "libc++abi.so")):
            os.symlink(runtime / enc, runtime / dec)
        monkeypatch.setattr(vt, "vtune_gpu_adapters", lambda b: [("0:4:0.0", "Card")])
        monkeypatch.setattr(vt, "device_bdf", lambda device, sys_root="/sys": (0, 4, 0, 0))
        monkeypatch.setattr(
            vt, "check_ptrace_scope", lambda *a, **k: vt.Prerequisite("p", True, "")
        )
        monkeypatch.setattr(vt, "check_metrics_library", lambda: vt.Prerequisite("m", True, ""))
        monkeypatch.setattr(
            vt,
            "check_gpu_observation",
            lambda *a, **k: vt.Prerequisite(
                "g", False, "gate", "sysctl", True, frozenset({vt.NEED_COUNTERS})
            ),
        )
        monkeypatch.setattr(vt.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
        result = vt.profile_command_with_vtune(["true"], mode="characterization", vtune_path=binary)
        assert "NEEDS ROOT" in result and "sysctl" in result
        assert "timing" in result.split("Modes that need none of the missing items:")[1]
