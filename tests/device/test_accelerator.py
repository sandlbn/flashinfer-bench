"""Tests for the accelerator abstraction.

These run without any GPU: a mock backend exercises registration, dispatch and
capability reporting, which is what lets an unreleased device be dry-run in CI before
its silicon exists.
"""

from typing import ClassVar, List, Optional

import pytest

from flashinfer_bench.device import (
    Accelerator,
    Capabilities,
    CpuAccelerator,
    CudaAccelerator,
    XpuAccelerator,
    canonicalize_device_name,
    default_device_type,
    get_accelerator,
    parse_device,
    register_accelerator,
    registered_types,
    reset_registry_cache,
)
from flashinfer_bench.device.timer import Timer, WallClockTimer


class FakeTimer(WallClockTimer):
    name: ClassVar[str] = "fake"


@register_accelerator
class FakeAccelerator(Accelerator):
    """Stand-in for hardware that is not present (or does not exist yet)."""

    type: ClassVar[str] = "fake"

    caps: ClassVar[Capabilities] = Capabilities(
        canonical_id="FAKE_XE3P",
        supported_dtypes=frozenset({"float32", "bfloat16"}),
        l2_bytes=256 * 1024 * 1024,
        sycl_target=None,
        supports_graphs=False,
        extra={"architecture": "test"},
    )

    def __init__(self) -> None:
        self.synchronized: List[Optional[str]] = []
        self.set_devices: List[str] = []

    @staticmethod
    def is_available() -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def list_devices(self) -> List[str]:
        return ["fake:0", "fake:1"]

    def set_device(self, device: str) -> None:
        self.set_devices.append(device)

    def synchronize(self, device: Optional[str] = None) -> None:
        self.synchronized.append(device)

    def empty_cache(self) -> None:
        return None

    def device_name(self, device: str) -> str:
        return "Fake(R) Xe3P(TM) Accelerator Graphics"

    def capabilities(self, device: str) -> Capabilities:
        return self.caps

    def make_timer(self, device: str) -> Timer:
        return FakeTimer()


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry_cache()
    yield
    reset_registry_cache()


class TestParseDevice:
    @pytest.mark.parametrize(
        "device,expected",
        [
            ("cuda:0", ("cuda", 0)),
            ("cuda:11", ("cuda", 11)),
            ("xpu:1", ("xpu", 1)),
            ("cpu", ("cpu", None)),
            ("cuda", ("cuda", None)),
        ],
    )
    def test_parses_device_strings(self, device, expected):
        assert parse_device(device) == expected

    def test_parses_torch_device_objects(self):
        torch = pytest.importorskip("torch")
        assert parse_device(torch.device("cuda:3")) == ("cuda", 3)
        assert parse_device(torch.device("cpu")) == ("cpu", None)

    @pytest.mark.parametrize("bad", ["", "cuda:x", "cuda:", None])
    def test_rejects_malformed_devices(self, bad):
        with pytest.raises(ValueError):
            parse_device(bad)


class TestCanonicalizeDeviceName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Intel(R) Arc(TM) B580 Graphics", "INTEL_ARC_B580"),
            ("NVIDIA H100 80GB HBM3", "NVIDIA_H100_80GB_HBM3"),
            ("Intel(R) Data Center GPU Max 1550", "INTEL_DATA_CENTER_GPU_MAX_1550"),
            ("", "UNKNOWN"),
        ],
    )
    def test_normalizes_vendor_names(self, raw, expected):
        assert canonicalize_device_name(raw) == expected

    def test_is_stable_under_repeated_application(self):
        once = canonicalize_device_name("Intel(R) Arc(TM) B580 Graphics")
        assert canonicalize_device_name(once) == once


class TestRegistry:
    def test_registers_all_shipped_backends(self):
        for expected in ("cuda", "xpu", "cpu"):
            assert expected in registered_types()

    def test_dispatches_on_device_type(self):
        assert isinstance(get_accelerator("cpu"), CpuAccelerator)
        assert isinstance(get_accelerator("cuda:0"), CudaAccelerator)
        assert isinstance(get_accelerator("xpu:0"), XpuAccelerator)
        assert isinstance(get_accelerator("fake:1"), FakeAccelerator)

    def test_memoizes_one_instance_per_type(self):
        assert get_accelerator("fake:0") is get_accelerator("fake:1")

    def test_rejects_unknown_device_type(self):
        with pytest.raises(ValueError, match="No accelerator registered"):
            get_accelerator("quantum:0")

    def test_explicit_backend_overrides_autodetection(self, monkeypatch):
        monkeypatch.setenv("FIB_DEVICE_BACKEND", "fake")
        assert default_device_type() == "fake"

    def test_unknown_configured_backend_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("FIB_DEVICE_BACKEND", "quantum")
        with pytest.raises(ValueError, match="not a known backend"):
            default_device_type()

    def test_falls_back_to_cpu_when_no_device_present(self, monkeypatch):
        monkeypatch.setenv("FIB_DEVICE_BACKEND", "auto")
        monkeypatch.setattr(FakeAccelerator, "device_count", lambda self: 0)
        monkeypatch.setattr(CudaAccelerator, "is_available", staticmethod(lambda: False))
        monkeypatch.setattr(XpuAccelerator, "is_available", staticmethod(lambda: False))
        assert default_device_type() == "cpu"


class TestAcceleratorContract:
    def test_synchronize_records_target_device(self):
        accel = get_accelerator("fake:1")
        accel.synchronize("fake:1")
        assert accel.synchronized == ["fake:1"]

    def test_device_synchronize_helper_dispatches(self):
        from flashinfer_bench.device import device_synchronize

        device_synchronize("fake:0")
        assert get_accelerator("fake:0").synchronized == ["fake:0"]

    def test_canonical_id_comes_from_capabilities(self):
        assert get_accelerator("fake:0").canonical_id("fake:0") == "FAKE_XE3P"

    def test_timer_reports_its_methodology(self):
        assert get_accelerator("fake:0").make_timer("fake:0").name == "fake"

    def test_cpu_backend_is_always_available(self):
        assert CpuAccelerator.is_available()
        assert CpuAccelerator().list_devices() == ["cpu"]


class TestCapabilities:
    def test_gates_unsupported_dtypes(self):
        caps = FakeAccelerator.caps
        assert caps.supports_dtype("bfloat16")
        assert caps.is_native_dtype("bfloat16")
        # This fixture declares no emulated dtypes, so an absent dtype is still refused.
        assert not caps.supports_dtype("float8_e4m3fn")
        assert caps.unsupported_dtypes({"float8_e4m3fn"}) == frozenset({"float8_e4m3fn"})

    def test_emulated_dtype_runs_but_is_not_native(self):
        """A dtype the runtime executes correctly but not on dedicated hardware.

        Battlemage has no FP8 DPAS, so ``torch._scaled_mm`` upconverts to fp16: the
        results are bit-exact and the throughput is not the format's. Such a dtype must
        run -- refusing it would make an FP8 model impossible to onboard on a part that
        handles it fine -- while never being reported as native.
        """
        caps = Capabilities(
            canonical_id="FAKE_EMULATED",
            supported_dtypes=frozenset({"float32", "bfloat16"}),
            emulated_dtypes=frozenset({"float8_e4m3fn"}),
        )
        assert caps.supports_dtype("float8_e4m3fn")
        assert not caps.is_native_dtype("float8_e4m3fn")
        assert caps.unsupported_dtypes({"float8_e4m3fn"}) == frozenset()
        assert caps.emulated_required_dtypes({"float8_e4m3fn"}) == frozenset(
            {"float8_e4m3fn"}
        )
        # A dtype in neither set is still a hard refusal.
        assert not caps.supports_dtype("float4_e2m1")
        assert caps.unsupported_dtypes({"float4_e2m1"}) == frozenset({"float4_e2m1"})
        # And a native dtype is not mislabelled as emulated.
        assert caps.is_native_dtype("bfloat16")
        assert caps.emulated_required_dtypes({"bfloat16"}) == frozenset()
        # A mixed request splits: nothing unsupported, only the fp8 flagged as emulated.
        mixed = ["float32", "float8_e4m3fn"]
        assert caps.unsupported_dtypes(mixed) == frozenset()
        assert caps.emulated_required_dtypes(mixed) == frozenset({"float8_e4m3fn"})

    def test_unknown_part_gets_conservative_defaults(self):
        from flashinfer_bench.device.capabilities import DEFAULT_DTYPES, Capabilities

        caps = Capabilities(canonical_id="INTEL_SOMETHING_UNRELEASED")
        # No AOT target -> native kernels compile to SPIR-V and JIT on arrival.
        assert caps.sycl_target is None
        assert caps.supported_dtypes == DEFAULT_DTYPES
        assert caps.l2_bytes > 0

    def test_capabilities_are_immutable(self):
        with pytest.raises(Exception):
            FakeAccelerator.caps.canonical_id = "MUTATED"  # type: ignore[misc]
