# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Operator-selected emission limits for the ``part97`` profile.

The profile applies configured limits; the operator remains responsible for their
station. ``licence`` selects a table of data segments and power ceilings.

The bandwidth ceiling is 2.8 kHz, checked against the bare audio passband.
``technique``, ``designator`` and ``symbol_rate_bd`` do not affect this profile's
checks. Placement uses the whole RF span, including drift on both sides.

Local control uses the selected data segments. Automatic control additionally
uses AUTOMATIC_SEGMENTS, or a response no wider than 500 Hz elsewhere in the
selected data segments. The profile refuses automatic control throughout its
60 m segment, including narrow responses. The 7.100–7.105 and 18.105–18.110 MHz
automatic segments are 5 kHz wide, leaving little placement room for a 2300 Hz
emission; 14.1005–14.112 MHz accommodates the widest waveform here.

Frequencies are integer hertz. Segment bounds are inclusive on both edges. An
emission must fit inside a single segment: straddling two adjacent ones is not
the same as being inside either. The 1 kHz gap at 14.100 MHz in the automatic
segments is preserved rather than merged away.

Carrier span, 99-percent power bandwidth and the -26 dB span are different
measures. The passbands supplied by core.occupied use outward-rounded -26 dB
edges so modulation skirts are included. Drift expands placement only, not the
2.8 kHz bandwidth check: a few ppm on a portable radio is 40–90 Hz at 14 MHz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from hfmodem.core.regulatory import Control, Emission, NotPermitted


class Licence(StrEnum):
    TECHNICIAN = "technician"
    GENERAL = "general"
    ADVANCED = "advanced"
    EXTRA = "extra"


@dataclass(frozen=True, slots=True)
class Segment:
    lo: int
    hi: int

    def contains(self, lo: float, hi: float) -> bool:
        return lo >= self.lo and hi <= self.hi


def _seg(lo_khz: float, hi_khz: float) -> Segment:
    return Segment(int(lo_khz * 1000), int(hi_khz * 1000))


#: The profile's bandwidth ceiling, applied to the bare passband.
MAX_BANDWIDTH_HZ = 2800

#: The lowest frequency covered by this profile's bandwidth model. Checked below
#: so extending DATA_SEGMENTS to 2200 m or 630 m requires revisiting that model.
BANDWIDTH_CAP_FLOOR_HZ = 1_800_000

#: Automatic-control segments without the narrow-response restriction.
#: The 14.0995/14.1005 MHz gap is deliberate.
AUTOMATIC_SEGMENTS: tuple[Segment, ...] = (
    _seg(3585, 3600),
    _seg(7100, 7105),
    _seg(10140, 10150),
    _seg(14095.0, 14099.5),
    _seg(14100.5, 14112),
    _seg(18105, 18110),
    _seg(21090, 21100),
    _seg(24925, 24930),
    _seg(28120, 28189),
)

#: Outside AUTOMATIC_SEGMENTS, the profile requires a response to a station under
#: local or remote control and a passband no wider than this ceiling.
AUTOMATIC_MAX_BANDWIDTH_HZ = 500

#: The profile's 60 m segment. 5358.5 kHz sits inside it; the separate channel
#: centres 5332.0, 5348.0, 5373.0 and 5405.0 kHz are not modelled. A channel
#: model with a fixed 2K80J2D passband and carrier 1.5 kHz below its centre would
#: need an exact-placement check, which a segment table cannot express.
SIXTY_METRE_BAND = _seg(5351.5, 5366.5)

#: Data segments selected by the profile's licence setting.
#: Advanced and General intentionally share the same data-segment table.
_HF_DATA_GENERAL = (
    _seg(1800, 2000), _seg(3525, 3600), SIXTY_METRE_BAND, _seg(7025, 7125),
    _seg(10100, 10150), _seg(14025, 14150), _seg(18068, 18110),
    _seg(21025, 21200), _seg(24890, 24930), _seg(28000, 28300),
)

DATA_SEGMENTS: dict[Licence, tuple[Segment, ...]] = {
    #: The technician setting selects only the 10 m data segment.
    Licence.TECHNICIAN: (_seg(28000, 28300),),
    Licence.GENERAL: _HF_DATA_GENERAL,
    Licence.ADVANCED: _HF_DATA_GENERAL,
    Licence.EXTRA: (
        _seg(1800, 2000), _seg(3500, 3600), SIXTY_METRE_BAND, _seg(7000, 7125),
        _seg(10100, 10150), _seg(14000, 14150), _seg(18068, 18110),
        _seg(21000, 21200), _seg(24890, 24930), _seg(28000, 28300),
    ),
}

if any(s.lo < BANDWIDTH_CAP_FLOOR_HZ for ss in DATA_SEGMENTS.values() for s in ss):
    raise ValueError(
        "DATA_SEGMENTS reaches below 160 m, outside this profile's bandwidth "
        "model. Define the profile's limits before adding 2200 m or 630 m; "
        "MAX_BANDWIDTH_HZ does not cover those bands.")

#: The profile's fallback power ceiling when no segment-specific limit applies.
MAX_POWER_W = 1500.0

#: Segment-specific ceilings: 200 W PEP on 30 m and a 9.15 W comparison on 60 m.
#: The latter models 9.15 W ERP assuming a 0 dBd half-wave dipole. ERP is PEP
#: multiplied by antenna gain relative to that dipole (in linear units). The
#: software compares power_w directly and does not know antenna gain; an operator
#: using a gain antenna must account for it to keep within their chosen ERP limit.
#: The separate 60 m channels have no power model here because they are omitted.
POWER_LIMITS: tuple[tuple[Segment, float], ...] = (
    (_seg(10100, 10150), 200.0),
    (SIXTY_METRE_BAND, 9.15),
)

#: Additional 200 W PEP ceilings selected by the technician setting.
TECHNICIAN_POWER_LIMITS: tuple[tuple[Segment, float], ...] = (
    (_seg(3525, 3600), 200.0), (_seg(7025, 7125), 200.0),
    (_seg(21025, 21200), 200.0), (_seg(28000, 28500), 200.0),
)

@dataclass(frozen=True)
class Part97:
    """The ``part97`` profile of operator-selected emission limits."""

    licence: Licence
    name: str = field(default="part97", init=False)

    def __post_init__(self):
        object.__setattr__(self, "licence", Licence(self.licence))

    def check(self, emission: Emission, *, control: Control,
              responding: bool = False) -> None:
        lo, hi = emission.span
        bw = emission.bandwidth_hz

        if bw > MAX_BANDWIDTH_HZ:
            raise NotPermitted(
                f"part97 bandwidth: {bw:.0f} Hz exceeds the {MAX_BANDWIDTH_HZ} Hz "
                "profile ceiling.")

        segments = DATA_SEGMENTS[self.licence]
        if not any(s.contains(lo, hi) for s in segments):
            raise NotPermitted(
                f"part97 placement: {lo/1e6:.6f}–{hi/1e6:.6f} MHz is outside "
                "the data segments "
                f"selected by licence={self.licence.value}. "
                f"Nearest: {_nearest(lo, hi, segments)}")

        self._check_power(emission, lo, hi)

        if control is Control.LOCAL:
            return

        # This profile excludes automatic control throughout its 60 m segment.
        if SIXTY_METRE_BAND.contains(lo, hi):
            raise NotPermitted(
                "part97 automatic control: the profile excludes 60 m, including "
                "narrow responses. Use control = \"local\" only with an operator "
                "at the control point.")

        if any(s.contains(lo, hi) for s in AUTOMATIC_SEGMENTS):
            return

        if responding and bw <= AUTOMATIC_MAX_BANDWIDTH_HZ:
            return

        why = (f"and {bw:.0f} Hz exceeds the {AUTOMATIC_MAX_BANDWIDTH_HZ} Hz limit"
               if responding else "and this is not a response to an interrogation")
        raise NotPermitted(
            f"part97 automatic control: {lo/1e6:.6f}–{hi/1e6:.6f} MHz is outside the "
            f"profile's automatic-control segments, {why}. "
            f"Use control = \"local\" only with an operator at the control point, "
            f"or place the emission within "
            f"{_nearest(lo, hi, AUTOMATIC_SEGMENTS)}.")

    def _check_power(self, emission: Emission, lo: float, hi: float) -> None:
        limits = list(POWER_LIMITS)
        if self.licence is Licence.TECHNICIAN:
            limits += list(TECHNICIAN_POWER_LIMITS)
        applicable = [w for seg, w in limits if seg.contains(lo, hi)]
        cap = min(applicable) if applicable else MAX_POWER_W
        if not applicable and emission.power_w is None:
            return          # only a band with its own limit demands a declaration
        # The 60 m comparison assumes a dipole; no antenna-gain correction is made.
        dipole = (" (the profile models ERP with a presumed 0 dBd dipole; "
                  "account for antenna gain when choosing transmitter PEP)"
                  if SIXTY_METRE_BAND.contains(lo, hi) else "")
        unit = "W ERP" if dipole else "W PEP"
        if emission.power_w is None:
            raise NotPermitted(
                f"part97 power: {lo/1e6:.6f}–{hi/1e6:.6f} MHz has a profile "
                f"ceiling of {cap:g} "
                f"{unit}{dipole} and this emission does not state its power. "
                "Declare power_w to evaluate it against the selected profile.")
        if emission.power_w > cap:
            raise NotPermitted(
                f"part97 power: {emission.power_w:g} W exceeds the {cap:g} {unit} "
                "profile limit "
                f"for {lo/1e6:.6f}–{hi/1e6:.6f} MHz{dipole}.")


def _nearest(lo: float, hi: float, segments: tuple[Segment, ...]) -> str:
    mid = (lo + hi) / 2
    s = min(segments, key=lambda s: min(abs(s.lo - mid), abs(s.hi - mid)))
    return f"{s.lo/1e6:.4f}–{s.hi/1e6:.4f} MHz"
