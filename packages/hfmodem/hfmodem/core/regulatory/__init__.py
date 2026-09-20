# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Evaluate an emission against the station's configured profile.

An Emission carries its measured audio passband, its bandwidth measure and
session context. Carrier span, 99-percent power bandwidth and the -26 dB span
are distinct quantities. Profiles choose the applicable measure; core.occupied
calculates the measured spans. Drift allowance expands the checked interval.

These checks implement configured limits. They do not determine whether an
operator is authorised to transmit.
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
    """Who is answerable for the transmission at the moment it is made.

    `LOCAL` means an operator is at the control point — which covers remote
    control too, since a human is still deciding. `AUTOMATIC` means the station
    is answering on its own, which is what `listen = true` amounts to. Several
    regimes treat the two very differently and none treats automatic as the
    easier case.
    """

    LOCAL = "local"
    AUTOMATIC = "automatic"


class NotPermitted(Exception):
    """The emission was refused. The message names the rule and the numbers."""


@dataclass(frozen=True, slots=True)
class Emission:
    """The physical facts about a transmission, in the terms rules ask about.

    `audio_lo_hz`/`audio_hi_hz` are the passband the modulator occupies above the
    suppressed carrier — the modem knows these; nothing else does. `power_w`,
    `designator`, `technique` and `symbol_rate_bd` are optional because not every
    regime asks, and a profile that needs one refuses when it is absent rather
    than guessing.
    """

    dial_hz: int
    audio_lo_hz: float
    audio_hi_hz: float
    #: Transmitter PEP. Several regimes cap it per band; §97.313(c) caps 30 m at
    #: 200 W for everyone and several bands at 200 W for Technicians.
    power_w: float | None = None
    #: ITU emission designator, e.g. "2K20J2D". Most non-amateur authorisations
    #: are written entirely in these.
    designator: str | None = None
    #: Which protocol this is. §97.309 permits only specified digital codes, or a
    #: technique "whose technical characteristics have been documented publicly".
    technique: str | None = None
    symbol_rate_bd: float | None = None
    #: Charged against the emission on both sides. See DEFAULT_DRIFT_HZ.
    drift_hz: float = DEFAULT_DRIFT_HZ

    @property
    def bandwidth_hz(self) -> float:
        """The passband width. Not any regulator's definition of bandwidth —
        profiles derive theirs from the passband itself."""
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
    """A regulatory regime, as far as software can check one."""

    name: str

    def check(self, emission: Emission, *, control: Control,
              responding: bool = False) -> None:
        """Return None if permitted; raise `NotPermitted` naming the rule.

        `responding` says this transmission answers an interrogation from a
        station whose own transmissions are under local or remote control. The
        ARQ layer is the only thing that knows it. Returns None rather than a
        boolean deliberately: a check that returns a boolean gets used in an
        ``if`` and then ignored.
        """
        ...


@dataclass(frozen=True, slots=True)
class Unregulated:
    """No constraint modelled. The operator is answerable for every emission.

    Honest for a dummy load, a receive-only installation, an experimental
    authorisation, or any regulator hfmodem does not encode. It is deliberately
    *not* called "open" — in a project built on open protocols that word reads as
    a virtue and would be doing unearned work for what is, in effect, the
    override flag.

    Because it is the override flag, it costs something to select: a stated
    reason, which is recorded and surfaced. A station running unchecked should
    not look, in its config and its logs, like a station that made a considered
    regulatory choice.
    """

    because: str
    name: str = field(default="unregulated", init=False)

    def __post_init__(self):
        if not self.because or not self.because.strip():
            raise ValueError(
                "the unregulated profile requires a stated reason, e.g. "
                'because = "dummy load, no antenna connected" or '
                'because = "operating under a licence hfmodem does not model". '
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
        "Shipped: 'part97' (47 CFR Part 97, US amateur — needs licence=), "
        "'unregulated' (no constraint modelled, operator answerable — needs "
        "because=). There is no default: a station that has not said which rules "
        "it operates under has not said enough.")
