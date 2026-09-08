"""The event timer must not report its own overhead as the kernel's latency.

Timing one call per event pair put a fixed cost -- event record plus an unpipelined launch
-- on top of every measurement. On Arc B580 that made a 4.9us kernel report 40.8us, an 8.3x
error, and it lands on every short kernel equally: measurements an order of magnitude apart
in real work all pile up on the same floor, so ratios between them are noise. That is the
regime serving actually runs in at decode sizes, which is what made it worth fixing.

These tests use a fake backend module so the arithmetic is checkable without a GPU.
"""

from typing import ClassVar, List

import pytest

from flashinfer_bench.device.timer import EventTimer


class _FakeEvent:
    """Records a timestamp from a shared fake clock."""

    def __init__(self, clock: List[float]) -> None:
        self._clock = clock
        self.t = 0.0

    def record(self) -> None:
        self.t = self._clock[0]

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return other.t - self.t


class _FakeModule:
    """A torch backend stand-in that charges a fixed cost per timed region.

    ``per_call_ms`` is the kernel; ``region_overhead_ms`` is what the timer itself costs
    each time it opens a region. A correct timer amortizes the latter away.
    """

    def __init__(self, per_call_ms: float, region_overhead_ms: float) -> None:
        self.clock = [0.0]
        self.per_call_ms = per_call_ms
        self.region_overhead_ms = region_overhead_ms
        self.calls = 0

    def Event(self, enable_timing: bool = True) -> _FakeEvent:  # noqa: N802 - torch's name
        return _FakeEvent(self.clock)

    def synchronize(self, device: str) -> None:
        return None

    def advance(self) -> None:
        self.calls += 1
        self.clock[0] += self.per_call_ms

    def open_region(self) -> None:
        self.clock[0] += self.region_overhead_ms


class _CountingTimer(EventTimer):
    """Charges the fake module's region overhead when a region is opened."""

    name: ClassVar[str] = "fake-event"

    def __init__(self, module: _FakeModule) -> None:
        super().__init__(module, l2_bytes=0)

    def _open(self):
        self._module.open_region()


@pytest.fixture
def module():
    # A 5us kernel measured by a timer that costs 36us per region -- the Arc B580 case.
    return _FakeModule(per_call_ms=0.005, region_overhead_ms=0.036)


def _run(module, monkeypatch):
    """time_all with the region overhead charged on each Event() pair."""
    real_event = module.Event
    opened = {"n": 0}

    def counting_event(enable_timing: bool = True):
        # The first Event of a pair opens the region.
        opened["n"] += 1
        if opened["n"] % 2 == 1:
            module.open_region()
        return real_event(enable_timing)

    monkeypatch.setattr(module, "Event", counting_event)
    timer = EventTimer(module, l2_bytes=0)
    return timer.time_all(lambda: module.advance(), (), warmup=0, iters=5, device="fake:0")


class TestBatching:
    def test_reported_latency_approaches_the_true_kernel_time(self, module, monkeypatch):
        times = _run(module, monkeypatch)
        reported = sorted(times)[len(times) // 2]
        # True cost is 0.005ms. Timing one call per region would report 0.041ms.
        assert reported == pytest.approx(0.005, rel=0.10), (
            f"reported {reported:.6f}ms; the region overhead is not being amortized"
        )

    def test_a_slow_kernel_is_not_batched_unnecessarily(self, monkeypatch):
        """A kernel already long against the overhead should stay near one call per region."""
        module = _FakeModule(per_call_ms=2.0, region_overhead_ms=0.036)
        times = _run(module, monkeypatch)
        assert sorted(times)[len(times) // 2] == pytest.approx(2.0, rel=0.05)

    def test_batching_is_bounded(self, monkeypatch):
        """An immeasurably fast kernel must not run unboundedly many times."""
        module = _FakeModule(per_call_ms=0.0, region_overhead_ms=0.0)
        _run(module, monkeypatch)
        assert module.calls <= EventTimer._MAX_INNER * 6 + 64, module.calls

    def test_every_iteration_is_reported(self, module, monkeypatch):
        assert len(_run(module, monkeypatch)) == 5
