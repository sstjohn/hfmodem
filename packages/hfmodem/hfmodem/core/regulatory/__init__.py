# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Evaluate an emission against the station's configured profile.

An Emission carries its supplied audio passband and session context. Carrier
span, 99-percent power bandwidth and the -26 dB span are distinct quantities.
Callers supply the applicable edges; core.occupied holds the measured spans used
by the station. Drift allowance expands the checked interval.

These profiles apply the limits the operator chose; the operator remains
responsible for their station.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

#: Transmitters drift. A few ppm on a portable HF radio is 40–90 Hz at 14 MHz,
#: which is larger than the worst passband asymmetry among these waveforms. A
#: fail-closed gate charges the emission for it rather than assuming the dial is
#: where the display says.
DEFAULT_DRIFT_HZ = 100


class Control(StrEnum):
    """Whether an operator is at the control point when a transmission is made.

    `LOCAL` means an operator is at the control point — which covers remote
    control too, since a human is still deciding. `AUTOMATIC` means the station
    is answering on its own, which is what `listen = true` amounts to. A profile
    can apply different limits to these two settings.
    """

    LOCAL = "local"
    AUTOMATIC = "automatic"


class NotPermitted(Exception):
    """The selected profile refused the emission; the message names its limit and values."""


@dataclass(frozen=True, slots=True)
class Emission:
    """The physical facts about a transmission used by an emission profile.

    `audio_lo_hz`/`audio_hi_hz` are the passband the modulator occupies above the
    suppressed carrier — the modem knows these; nothing else does. `power_w`,
    `designator`, `technique` and `symbol_rate_bd` are optional because not every
    profile uses them, and a profile that needs one refuses when it is absent
    rather than guessing.
    """

    dial_hz: int
    audio_lo_hz: float
    audio_hi_hz: float
    #: Transmitter PEP in watts, checked against the selected profile's ceilings.
    power_w: float | None = None
    #: ITU emission designator, e.g. "2K20J2D", for profiles that use it.
    designator: str | None = None
    #: Protocol name, for profiles that use it.
    technique: str | None = None
    symbol_rate_bd: float | None = None
    #: Charged against the emission on both sides. See DEFAULT_DRIFT_HZ.
    drift_hz: float = DEFAULT_DRIFT_HZ

    @property
    def bandwidth_hz(self) -> float:
        """The supplied passband width, before any drift allowance."""
        return self.audio_hi_hz - self.audio_lo_hz

    @property
    def span(self) -> tuple[float, float]:
        """The RF span this emission occupies, drift included.

        Pure geometry: dial plus the audio passband. No sideband convention and
        no assumed centre.
        """
        return (self.dial_hz + self.audio_lo_hz - self.drift_hz,
                self.dial_hz + self.audio_hi_hz + self.drift_hz)


def centred(dial_hz: int, bandwidth_hz: float, *, centre_hz: float = 1500.0,
            **kw) -> Emission:
    """An `Emission` for a waveform that genuinely is symmetric about `centre_hz`.

    A convenience for callers that know only a nominal width — PACTOR-III is
    symmetric about 1500 Hz and this is exact for it. It is *not* right for VARA
    BW2300; that modulator should pass its real passband.
    """
    half = bandwidth_hz / 2
    return Emission(dial_hz, centre_hz - half, centre_hz + half, **kw)


@runtime_checkable
class Profile(Protocol):
    """A set of operator-selected emission checks."""

    name: str

    def check(self, emission: Emission, *, control: Control,
              responding: bool = False) -> None:
        """Return None if the profile accepts; raise `NotPermitted` naming its limit.

        `responding` says this transmission answers an interrogation from a
        station whose own transmissions are under local or remote control. The
        ARQ layer is the only thing that knows it. Returns None rather than a
        boolean deliberately: a check that returns a boolean gets used in an
        ``if`` and then ignored.
        """
        ...


@dataclass(frozen=True, slots=True)
class Unregulated:
    """No emission limits are applied; the operator remains responsible for the station.

    Selecting this profile requires a stated reason, which is recorded and
    surfaced so a station running unchecked says so in its config and logs.
    """

    because: str
    name: str = field(default="unregulated", init=False)

    def __post_init__(self):
        if not self.because or not self.because.strip():
            raise ValueError(
                "the unregulated profile requires a stated reason, e.g. "
                'because = "dummy load, no antenna connected" or '
                'because = "limits applied outside hfmodem". '
                "It is recorded and shown in status output.")

    def check(self, emission: Emission, *, control: Control,
              responding: bool = False) -> None:
        return None


def profile(name: str, **kwargs) -> Profile:
    """The named profile. Unknown names are an error rather than a fallback."""
    if name == "unregulated":
        return Unregulated(**kwargs)
    if name == "part97":
        from hfmodem.core.regulatory.part97 import Part97
        return Part97(**kwargs)
    raise ValueError(
        f"unknown regulatory profile {name!r}. "
        "Shipped: 'part97' (configured emission limits — needs licence=), "
        "'unregulated' (no emission limits applied — needs because=). "
        "There is no default: select the limits to apply.")
