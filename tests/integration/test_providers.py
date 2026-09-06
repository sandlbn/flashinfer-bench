"""Tests for acquiring the Intel kernel providers.

Nothing here installs anything: the commands are checked, not run. The value being
protected is that a provider's acquisition is derived rather than remembered -- especially
the GPU architecture a source build must target, which is already in the device's
capability record and should never be something the user is asked for.
"""

import sys

import pytest

from flashinfer_bench.integration import providers as prov


class TestSpecTable:
    def test_every_spec_is_addressable_by_name(self):
        for spec in prov.SPECS:
            assert prov.get_spec(spec.name) is spec

    def test_unknown_provider_lists_the_known_ones(self):
        with pytest.raises(prov.ProviderError) as e:
            prov.get_spec("does-not-exist")
        assert "vllm-xpu" in str(e.value)

    def test_every_on_disk_provider_declares_how_to_find_it(self):
        """A provider found on disk needs both a search rule and a probe.

        Without a probe, a directory that merely exists counts as an installation -- and
        for oneDNN specifically, a prefix holding only the runtime library cannot build.
        """
        for spec in prov.SPECS:
            if spec.kind in ("checkout", "system"):
                assert spec.env_var, f"{spec.name} has no env var"
                assert spec.probe, f"{spec.name} has no probe path"

    def test_installable_providers_name_their_source(self):
        for spec in prov.SPECS:
            if spec.kind in ("wheel", "source"):
                assert spec.repo or spec.distribution


class TestInstallCommands:
    def test_a_wheel_provider_is_a_plain_install(self):
        """Which installer runs depends on the environment; the package does not.

        Asserting `python -m pip` here is what hid the uv case: this virtualenv has no
        pip at all, so that spelling fails before it installs anything.
        """
        command = prov.install_command(prov.get_spec("vllm-xpu"))
        assert "install" in command
        assert "vllm-xpu-kernels" in command
        assert command[0] in (sys.executable, prov.shutil.which("uv"))

    def test_a_source_build_carries_the_architecture(self):
        command = prov.install_command(prov.get_spec("sgl-kernel-xpu"), target="bmg")
        assert "--config-settings=cmake.define.DPCPP_SYCL_TARGET=bmg" in command
        assert "--no-build-isolation" in command

    def test_a_system_library_refuses_and_says_why(self):
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("onednn"))
        assert "FIB_ONEDNN_DIR" in str(e.value)

    def test_a_checkout_refuses_and_names_the_variable(self):
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("xe-fuse"))
        assert "FIB_XE_FUSE_DIR" in str(e.value)

    def test_dry_run_does_not_execute(self, monkeypatch):
        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("dry run executed a subprocess")

        monkeypatch.setattr(prov.subprocess, "run", explode)
        command, code = prov.install("vllm-xpu", dry_run=True)
        assert code == 0 and "pip" in command


class TestArchitectureDerivation:
    """The build target comes from the device, and an unbuildable part says so first."""

    def test_a_part_without_a_target_refuses_before_building(self, monkeypatch):
        class _Caps:
            canonical_id = "INTEL_INTEGRATED_XE3"
            sycl_target = None

        class _Accel:
            def capabilities(self, device):
                return _Caps()

        monkeypatch.setattr(
            "flashinfer_bench.device.list_devices", lambda: ["xpu:0"], raising=False
        )
        monkeypatch.setattr(
            "flashinfer_bench.device.get_accelerator", lambda d: _Accel(), raising=False
        )
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("sgl-kernel-xpu"))
        assert "bmg" in str(e.value) and "cri" in str(e.value)

    def test_no_intel_gpu_explains_the_alternative(self, monkeypatch):
        monkeypatch.setattr("flashinfer_bench.device.list_devices", lambda: ["cpu"], raising=False)
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("sgl-kernel-xpu"))
        assert "--target" in str(e.value)

    def test_an_explicit_target_needs_no_device(self, monkeypatch):
        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("consulted the device despite an explicit target")

        monkeypatch.setattr("flashinfer_bench.device.list_devices", explode, raising=False)
        command = prov.install_command(prov.get_spec("sgl-kernel-xpu"), target="cri")
        assert "DPCPP_SYCL_TARGET=cri" in " ".join(command)


class TestProvenance:
    def test_status_covers_every_spec(self):
        assert len(prov.all_status()) == len(prov.SPECS)

    def test_provenance_reports_only_what_is_present(self):
        recorded = prov.provider_provenance()
        for name in recorded:
            assert prov.is_installed(prov.get_spec(name))

    def test_version_comes_from_distribution_metadata(self):
        """vllm_xpu_kernels ships no __version__, so metadata is the only source."""
        spec = prov.get_spec("vllm-xpu")
        if not prov.is_installed(spec):
            pytest.skip("vllm-xpu-kernels not installed")
        assert prov.provider_version(spec)


class TestEnvironmentHandling:
    """Both of these were real failures: the documented recipes did not run here."""

    def test_uv_is_used_when_the_environment_has_no_pip(self, monkeypatch):
        """A uv-managed virtualenv has no pip module; `python -m pip` dies there."""
        monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        command = prov.install_command(prov.get_spec("vllm-xpu"))
        assert command[0] == "/usr/bin/uv"
        # uv must be told which interpreter to install into, not left to infer it.
        assert "--python" in command and sys.executable in command

    def test_pip_is_preferred_when_present(self, monkeypatch):
        monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: object())
        command = prov.install_command(prov.get_spec("vllm-xpu"))
        assert command[:3] == [sys.executable, "-m", "pip"]

    def test_no_installer_at_all_says_so(self, monkeypatch):
        monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(prov.shutil, "which", lambda name: None)
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("vllm-xpu"))
        assert "uv" in str(e.value) and "pip" in str(e.value)

    def test_a_no_isolation_build_declares_its_backend(self):
        """--no-build-isolation means nothing installs the build backend for you."""
        spec = prov.get_spec("sgl-kernel-xpu")
        command = prov.install_command(spec, target="bmg")
        assert "--no-build-isolation" in command
        assert "scikit-build-core" in spec.build_requires


class TestDistributionNames:
    """The distribution a provider installs is not always its repo name."""

    def test_sgl_kernel_xpu_reports_a_version_when_installed(self):
        """The repo is sgl-kernel-xpu; the distribution it installs is sgl-kernel.

        Using the repo name loses the version silently -- status still says "installed",
        so nothing looks wrong, but trace provenance records no version at all.
        """
        spec = prov.get_spec("sgl-kernel-xpu")
        assert spec.distribution == "sgl-kernel"
        if prov.is_installed(spec):
            assert prov.provider_version(spec), "installed but no version resolved"
