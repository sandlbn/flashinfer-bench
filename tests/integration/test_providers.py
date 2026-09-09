"""Tests for acquiring the Intel kernel providers.

Nothing here installs anything: the commands are checked, not run. The value being
protected is that a provider's acquisition is derived rather than remembered -- especially
the GPU architecture a source build must target, which is already in the device's
capability record and should never be something the user is asked for.
"""

import sys

import pytest

from flashinfer_bench.integration import providers as prov


@pytest.fixture
def with_pip(monkeypatch):
    """Pretend this interpreter has pip, so command-shape tests see a command.

    The dev venv here has no pip, and without this every install_command call would be
    the refusal instead -- which has its own tests in TestInstallerChoice.
    """
    monkeypatch.setattr(prov, "_has_pip", lambda: True)
    monkeypatch.delenv(prov.INSTALLER_ENV_VAR, raising=False)


@pytest.fixture
def without_pip(monkeypatch):
    monkeypatch.setattr(prov, "_has_pip", lambda: False)
    monkeypatch.delenv(prov.INSTALLER_ENV_VAR, raising=False)


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
    def test_a_wheel_provider_is_a_plain_pip_install(self, with_pip):
        """The default installer is this interpreter's pip and nothing else."""
        command = prov.install_command(prov.get_spec("vllm-xpu"))
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        assert "vllm-xpu-kernels" in command

    def test_a_source_build_carries_the_architecture(self, with_pip):
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

    def test_dry_run_does_not_execute(self, with_pip, monkeypatch):
        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("dry run executed a subprocess")

        monkeypatch.setattr(prov.subprocess, "run", explode)
        command, code = prov.install("vllm-xpu", dry_run=True)
        assert code == 0 and command[:3] == [sys.executable, "-m", "pip"]

    def test_dry_run_with_uv_opt_in_shows_the_uv_command_without_running_it(
        self, without_pip, monkeypatch
    ):
        """`--installer uv --dry-run` is the sanctioned way to see the uv command."""

        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("dry run executed a subprocess")

        monkeypatch.setattr(prov.subprocess, "run", explode)
        monkeypatch.setattr(prov.shutil, "which", lambda name: f"/usr/bin/{name}")
        command, code = prov.install("vllm-xpu", dry_run=True, installer="uv")
        assert code == 0 and command[:3] == ["/usr/bin/uv", "pip", "install"]


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

    def test_an_explicit_target_needs_no_device(self, with_pip, monkeypatch):
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
            if name.startswith("env:"):
                continue  # environment facts, not a claim that a provider is installed
            assert prov.is_installed(prov.get_spec(name))

    def test_environment_keys_are_namespaced(self):
        """Provenance carries two kinds of fact and they must not be confusable.

        A provider entry asserts "this library is installed at this version". An `env:` entry
        records something about the machine -- e.g. that torch runs a different oneDNN than
        solutions link against. Reading the second as the first would have a consumer
        conclude a provider named `onednn_runtime` exists.
        """
        for name in prov.provider_provenance():
            if not name.startswith("env:"):
                prov.get_spec(name)  # must resolve; raises otherwise

    def test_version_comes_from_distribution_metadata(self):
        """vllm_xpu_kernels ships no __version__, so metadata is the only source."""
        spec = prov.get_spec("vllm-xpu")
        if not prov.is_installed(spec):
            pytest.skip("vllm-xpu-kernels not installed")
        assert prov.provider_version(spec)


class TestInstallerChoice:
    """The library never runs an installer the caller did not ask for.

    The defect this guards against was real: on a box whose venvs have no pip module,
    `install_command` used to fall back to `uv pip install` on its own. uv resolves the
    provider's dependencies against PyPI and can replace an Intel XPU torch with the
    default CUDA build -- silently, and undone only by a long manual reinstall.
    """

    def test_pip_is_used_when_present(self, with_pip):
        command = prov.install_command(prov.get_spec("vllm-xpu"))
        assert command[:3] == [sys.executable, "-m", "pip"]

    def test_no_pip_declines_rather_than_falling_back(self, without_pip, monkeypatch):
        """uv on PATH must make no difference to what happens by default."""
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("vllm-xpu"))
        msg = str(e.value)
        assert "Declined" in msg
        # The caller must not need the source to know what to run: package, environment,
        # reason, and the opt-in are all in the message.
        assert "vllm-xpu-kernels" in msg
        assert sys.prefix in msg and sys.executable in msg
        assert "CUDA" in msg and "no `pip` module" in msg
        assert "--installer uv" in msg and prov.INSTALLER_ENV_VAR in msg

    def test_declining_never_runs_anything(self, without_pip, monkeypatch):
        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("a refusal ran a subprocess")

        monkeypatch.setattr(prov.subprocess, "run", explode)
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        with pytest.raises(prov.ProviderError):
            prov.install("vllm-xpu")

    def test_source_build_prerequisites_are_also_declined(self, without_pip, monkeypatch):
        """The build-requirements pre-install is a second installer call; same rule."""
        monkeypatch.setattr(prov.shutil, "which", lambda name: f"/usr/bin/{name}")
        with pytest.raises(prov.ProviderError) as e:
            prov.install("sgl-kernel-xpu", target="bmg", dry_run=True)
        assert "Declined" in str(e.value)

    def test_uv_runs_only_when_named(self, without_pip, monkeypatch):
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        command = prov.install_command(prov.get_spec("vllm-xpu"), installer="uv")
        assert command[0] == "/usr/bin/uv"
        # uv must be told which interpreter to install into, not left to infer it.
        assert "--python" in command and sys.executable in command

    def test_the_environment_variable_is_an_explicit_opt_in_too(self, without_pip, monkeypatch):
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        monkeypatch.setenv(prov.INSTALLER_ENV_VAR, "uv")
        command = prov.install_command(prov.get_spec("vllm-xpu"))
        assert command[0] == "/usr/bin/uv"

    def test_the_argument_beats_the_environment_variable(self, with_pip, monkeypatch):
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        monkeypatch.setenv(prov.INSTALLER_ENV_VAR, "uv")
        command = prov.install_command(prov.get_spec("vllm-xpu"), installer="pip")
        assert command[:3] == [sys.executable, "-m", "pip"]

    def test_uv_requested_but_absent_says_so(self, without_pip, monkeypatch):
        monkeypatch.setattr(prov.shutil, "which", lambda name: None)
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("vllm-xpu"), installer="uv")
        assert "uv" in str(e.value) and sys.prefix in str(e.value)

    def test_an_unknown_installer_is_rejected(self, with_pip):
        with pytest.raises(prov.ProviderError) as e:
            prov.install_command(prov.get_spec("vllm-xpu"), installer="conda")
        assert "conda" in str(e.value) and "pip" in str(e.value)

    def test_a_no_isolation_build_declares_its_backend(self, with_pip):
        """--no-build-isolation means nothing installs the build backend for you."""
        spec = prov.get_spec("sgl-kernel-xpu")
        command = prov.install_command(spec, target="bmg")
        assert "--no-build-isolation" in command
        assert "scikit-build-core" in spec.build_requires


class TestCli:
    """The CLI surface: --dry-run prints, a refusal is the message and exit 1."""

    def _run(self, argv, monkeypatch, capsys):
        from flashinfer_bench.cli.main import cli

        monkeypatch.setattr(sys, "argv", ["flashinfer-bench", *argv])
        try:
            cli()
        except SystemExit as e:
            return e.code, capsys.readouterr()
        return 0, capsys.readouterr()

    def test_dry_run_prints_the_pip_command(self, with_pip, monkeypatch, capsys):
        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("dry run executed a subprocess")

        monkeypatch.setattr(prov.subprocess, "run", explode)
        code, out = self._run(
            ["providers", "install", "vllm-xpu", "--dry-run"], monkeypatch, capsys
        )
        assert code in (0, None)
        assert out.out.strip() == f"{sys.executable} -m pip install vllm-xpu-kernels"

    def test_default_without_pip_declines_and_tells_the_user_what_to_do(
        self, without_pip, monkeypatch, capsys, caplog
    ):
        def explode(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("a refusal ran a subprocess")

        monkeypatch.setattr(prov.subprocess, "run", explode)
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/usr/bin/uv")
        code, _ = self._run(["providers", "install", "vllm-xpu"], monkeypatch, capsys)
        assert code == 1
        assert "Declined to install vllm-xpu-kernels" in caplog.text
        assert "--installer uv" in caplog.text

    def test_installer_flag_is_the_only_route_to_uv(self, without_pip, monkeypatch, capsys):
        monkeypatch.setattr(prov.shutil, "which", lambda name: f"/usr/bin/{name}")
        code, out = self._run(
            ["providers", "install", "vllm-xpu", "--installer", "uv", "--dry-run"],
            monkeypatch,
            capsys,
        )
        assert code in (0, None)
        assert out.out.startswith("/usr/bin/uv pip install --python ")


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
