"""Detection of host power settings that distort benchmark measurements.

A power-saving platform profile can dominate every number a benchmark produces. On a
laptop under a ``low-power`` profile, an Intel integrated GPU stays pinned at its maximum
clock while measured latency sporadically jumps by a large factor for hundreds of
iterations at a time -- the GPU is not throttled, the memory path is. Run-to-run variation
of the reported median went from negligible under ``performance`` to tens of percent under
``power-saver`` on the same machine and the same kernel.

That is large enough to swamp any real difference between two solutions, and it is
invisible in the results: nothing in a trace says the host was throttling. So it is worth
one cheap check at startup rather than a long investigation later.

Everything here is advisory and best-effort. These files are Linux-specific and often
absent (containers, other platforms); their absence is not a problem to report.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

_PLATFORM_PROFILE = Path("/sys/firmware/acpi/platform_profile")

_THROTTLING_PROFILES = frozenset({"low-power", "quiet", "cool", "power-saver"})
"""Profiles that trade sustained performance for power, heat, or noise."""


def platform_profile() -> Optional[str]:
    """The host's ACPI platform profile, or ``None`` if it does not expose one."""
    try:
        return _PLATFORM_PROFILE.read_text().strip() or None
    except Exception:
        return None


def on_battery() -> Optional[bool]:
    """Whether the host is running on battery, or ``None`` if it cannot be determined."""
    supplies = glob.glob("/sys/class/power_supply/A*/online")
    if not supplies:
        return None
    for path in supplies:
        try:
            if Path(path).read_text().strip() == "1":
                return False
        except Exception:
            continue
    return True


def measurement_warnings() -> List[str]:
    """Host conditions that would make measurements untrustworthy.

    Returns
    -------
    List[str]
        Human-readable warnings, empty when nothing looks wrong.
    """
    warnings: List[str] = []

    profile = platform_profile()
    if profile is not None and profile in _THROTTLING_PROFILES:
        warnings.append(
            f"Host platform profile is '{profile}', which throttles sustained work. "
            "Measured latencies can vary by tens of percent run to run, enough to swamp "
            "real differences between solutions. Switch to 'performance' before "
            "benchmarking (for example: powerprofilesctl set performance)."
        )

    if on_battery():
        warnings.append(
            "Host is running on battery. Power limits on battery make latencies "
            "unreproducible; connect AC before benchmarking."
        )

    return warnings
