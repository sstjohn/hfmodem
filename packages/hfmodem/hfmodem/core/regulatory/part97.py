# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""47 CFR Part 97 — the United States amateur service.

One profile among several possible ones; see `core.regulatory` for why the regime
is pluggable rather than compiled in. Nothing outside this module knows Part 97
exists.

Four constraints matter to a station of this kind.

**Bandwidth (§97.307(f)(3)).** HF data is capped at 2.8 kHz. This is newer than
it looks: until 8 January 2024 (88 FR 85128) the rule set a 300-baud symbol-rate
limit and no numeric bandwidth cap at all. The change cuts both ways here — it is
what makes a 2750 Hz waveform legal, and it is what makes an unbounded one
illegal.

**Specified digital codes (§97.309).** The rule text is stricter than it is
usually reported to be, and this profile does not enforce it. The same sentence
that sets 2.8 kHz opens "Only a RTTY or data emission using a specified digital
code listed in §97.309(a) may be transmitted", and (a) lists three: Baudot,
AMTOR, and ASCII. (a)(4) is not a fourth entry. It is a permission attached to
the other three — a station "transmitting a RTTY or data emission using one of
the specified digital codes may use any technique whose technical
characteristics have been documented publicly, such as CLOVER, G-TOR, or
PACTOR". The unspecified digital codes of §97.309(b) are authorised on VHF and
above, and nowhere on HF.

Read plainly, that is a requirement almost nothing on the air today meets. It is
also unenforced, and there is less behind it than is usually claimed: the FCC has
never set a disclosure standard, and the widely quoted "decodable by a third
party" test comes from an ARRL website page rather than from any rule or order.

An earlier version of this profile refused any technique it could not place on
the list, which would have refused VARA. That was wrong, for a reason worth
stating: whether a technique's characteristics are documented publicly is a
judgement about the state of the world, not a computation. VARA has carried
traffic across the Winlink network daily for more than a decade without
enforcement action. Software that refuses it is not protecting an operator; it is
imposing one reading of an unenforced rule, on the one question in this module
that is not arithmetic.

So the gate checks what software can check — where the emission sits, how wide it
is, how much power it uses, and whether an unattended station may answer there —
and leaves §97.309 to the operator, who is answerable for it.

There is also a way to answer the rule rather than argue with it, and it is part
of why this project exists: a waveform whose specification is published *is*
documented publicly. hfmodem publishes normative descriptions of every protocol
it implements, which satisfies the (a)(4) limb for the whole service rather than
for this station alone. It does not touch the (a)(1)–(3) limb, and nothing here
pretends otherwise.

**Privileges (§97.301, §97.305).** Which segments carry data depends on the
licence class, so `licence` is required: guessing it wrong in the permissive
direction is the failure that matters. Technician HF data is confined to 10 m —
by §97.307(f)(9), which the §97.305(c) table attaches to the 80 m, 40 m and 15 m
data rows and not to 28.0–28.3, rather than by §97.301, which does give
Technicians those segments.

**Automatic control (§97.221).** A station answering a connect request with no
operator present is automatically controlled, which confines it to the
§97.221(b) sub-bands, or elsewhere only under the §97.221(c) conditions. The
constraint bites hardest where those sub-bands are narrow: 7.100–7.105 and
18.105–18.110 are 5 kHz wide, so a 2300 Hz emission has almost nowhere to sit,
while 14.1005–14.112 holds the widest waveform here comfortably. 60 m is out
altogether; see SIXTY_METRE_BAND.

Frequencies are integer hertz. Segment bounds are **inclusive on both edges** —
§97.301 gives closed ranges and §97.307(b) requires the emission to be *confined
to* the segment, so an emission whose upper edge lands exactly on a boundary is
legal. An emission must fit inside a *single* segment: straddling two adjacent
ones is not the same as being inside either, which is why the 1 kHz gap at
14.100 in the automatic sub-bands is preserved rather than merged away.

The span measured against those bounds carries the emission's drift allowance on
both sides. No rule asks for that: §97.3(a)(8) defines bandwidth by 26 dB
attenuation and makes no allowance for drift or Doppler, and the 2.8 kHz check
below is against the bare passband. Charging drift against *placement* is
conservative engineering — a few ppm on a portable radio is 40–90 Hz at 14 MHz —
and is not presented as a rule this profile enforces.
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


#: §97.307(f)(3) — the authorised bandwidth for HF data, "except in the 2200 m
#: band and 630 m band", where the rule sets a 300-baud symbol rate or a 1 kHz
#: mark-space shift and no width at all. 60 m arrives at the same 2.8 kHz by a
#: different route: §97.305(c)(3)(iii) sends it to §97.307(f)(14), and
#: §97.303(h)(3) caps "all 60 m spectrum".
MAX_BANDWIDTH_HZ = 2800

#: The lowest frequency MAX_BANDWIDTH_HZ governs. DATA_SEGMENTS stops at 160 m,
#: so the 2200 m / 630 m exception never arises — checked below rather than
#: trusted, because adding either band would apply the wrong rule in silence.
BANDWIDTH_CAP_FLOOR_HZ = 1_800_000

#: §97.221(b) — where an automatically controlled digital station may transmit
#: without meeting the (c) conditions. The 14.0995/14.1005 gap is deliberate.
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

#: §97.221(c)(2) — outside (b), only in response to an interrogation by a station
#: under local or remote control, and not exceeding this bandwidth.
AUTOMATIC_MAX_BANDWIDTH_HZ = 500

#: §97.303(h)(3) — 60 m, as rewritten by FCC 25-60 (91 FR 1405) effective
#: 13 February 2026. Amateur stations may transmit "in the 5351.5-5366.5 kHz band
#: and on the four center frequencies" 5332.0, 5348.0, 5373.0 and 5405.0 kHz.
#: 5358.5 kHz, one of the previous five channels, now sits inside the band.
#:
#: The band is an ordinary data segment and appears in DATA_SEGMENTS below. The
#: four channels do not: data there is 2K80J2D with the carrier set 1.5 kHz below
#: the channel centre, an exact placement a segment table cannot express.
SIXTY_METRE_BAND = _seg(5351.5, 5366.5)

#: §97.301 / §97.305(c) — RTTY-and-data segments by licence class, HF.
#: Advanced and General are identical here on purpose: §97.301(c) and (d) differ
#: only in the phone segments, and every data segment is the same.
_HF_DATA_GENERAL = (
    _seg(1800, 2000), _seg(3525, 3600), SIXTY_METRE_BAND, _seg(7025, 7125),
    _seg(10100, 10150), _seg(14025, 14150), _seg(18068, 18110),
    _seg(21025, 21200), _seg(24890, 24930), _seg(28000, 28300),
)

DATA_SEGMENTS: dict[Licence, tuple[Segment, ...]] = {
    #: §97.307(f)(9) confines Technician HF data to 10 m; the §97.305(c) table
    #: attaches it to the 80/40/15 m rows and not to 28.0–28.3. 60 m is absent
    #: for a different reason: §97.301(a) does not grant it to Technicians.
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
        "DATA_SEGMENTS reaches below 160 m, where §97.307(f)(3) sets a symbol-rate "
        "and shift limit instead of a bandwidth in hertz. Model that rule before "
        "offering 2200 m or 630 m; MAX_BANDWIDTH_HZ does not apply there.")

#: §97.313(b) — 1500 W PEP, everywhere, for everyone. The cheapest possible
#: check, and its absence meant power_w=2000 on 20 m passed.
MAX_POWER_W = 1500.0

#: §97.313(c)(1) — 200 W PEP on 30 m for everyone, and §97.313(i) — 9.15 W ERP
#: in the 60 m band (100 W ERP on the four channels, which this profile does not
#: offer, so that figure has nothing to attach to).
#:
#: The 60 m limit is the only one here written in ERP rather than PEP, and the
#: rule supplies the conversion: ERP is "the transmitter PEP … multiplied by the
#: antenna gain relative to a half-wave dipole", with a dipole presumed to be
#: 0 dBd. So 9.15 W PEP is exactly the limit into a dipole, and the operator on a
#: gain antenna must come down further — the station does not know its gain, and
#: §97.313(i) requires the licensee to keep the figure in the station records.
POWER_LIMITS: tuple[tuple[Segment, float], ...] = (
    (_seg(10100, 10150), 200.0),
    (SIXTY_METRE_BAND, 9.15),
)

#: §97.313(c)(2) — 200 W PEP for a Technician control operator on these bands.
TECHNICIAN_POWER_LIMITS: tuple[tuple[Segment, float], ...] = (
    (_seg(3525, 3600), 200.0), (_seg(7025, 7125), 200.0),
    (_seg(21025, 21200), 200.0), (_seg(28000, 28500), 200.0),
)

@dataclass(frozen=True)
class Part97:
    """The US amateur profile."""

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
                f"§97.307(f)(3): {bw:.0f} Hz exceeds the {MAX_BANDWIDTH_HZ} Hz "
                "authorised bandwidth for HF data.")

        segments = DATA_SEGMENTS[self.licence]
        if not any(s.contains(lo, hi) for s in segments):
            raise NotPermitted(
                f"§97.301: {lo/1e6:.6f}–{hi/1e6:.6f} MHz is outside the data segments "
                f"available to a {self.licence.value} licensee. "
                f"Nearest: {_nearest(lo, hi, segments)}")

        self._check_power(emission, lo, hi)

        if control is Control.LOCAL:
            return

        # §97.221(c) excepts "channels specified in §97.303(h)", and (b) never
        # listed 60 m — so the four channels are plainly out. The band is the
        # open question: since 13 February 2026 §97.303(h) specifies a band as
        # well as channels, and nothing says which of the two the word reaches.
        # Held out until something settles it. 60 m is shared with Federal users
        # on a non-interference basis, and §97.307(f)(14)(ii) asks the control
        # operator to limit the length of transmissions — neither is what an
        # unattended station is good at.
        if SIXTY_METRE_BAND.contains(lo, hi):
            raise NotPermitted(
                "§97.221(c): automatic control is not available on 60 m. The "
                "exception is written for \"channels specified in §97.303(h)\", "
                "which since 13 February 2026 specifies a band as well as four "
                "channels, and it is unsettled whether it reaches the band. "
                "Set control = \"local\" and stay at the radio.")

        if any(s.contains(lo, hi) for s in AUTOMATIC_SEGMENTS):
            return

        if responding and bw <= AUTOMATIC_MAX_BANDWIDTH_HZ:
            return

        why = (f"and {bw:.0f} Hz exceeds the {AUTOMATIC_MAX_BANDWIDTH_HZ} Hz limit"
               if responding else "and this is not a response to an interrogation")
        raise NotPermitted(
            f"§97.221: {lo/1e6:.6f}–{hi/1e6:.6f} MHz is outside the sub-bands where an "
            f"automatically controlled station may transmit, {why}. "
            f"Either set control = \"local\" and stay at the radio, or move to "
            f"{_nearest(lo, hi, AUTOMATIC_SEGMENTS)}.")

    def _check_power(self, emission: Emission, lo: float, hi: float) -> None:
        limits = list(POWER_LIMITS)
        if self.licence is Licence.TECHNICIAN:
            limits += list(TECHNICIAN_POWER_LIMITS)
        applicable = [w for seg, w in limits if seg.contains(lo, hi)]
        cap = min(applicable) if applicable else MAX_POWER_W
        if not applicable and emission.power_w is None:
            return          # only a band with its own limit demands a declaration
        # The 60 m cap is the one limit here written in ERP (§97.313(i));
        # everything else is transmitter PEP. These messages once said
        # "9.15 W PEP" flatly -- an operator on a 6 dBd beam reading that sits
        # four times over -- and a test matching that exact string is how the
        # wrong unit survived.
        dipole = (" (§97.313(i) writes this in ERP: it is the PEP limit into "
                  "the presumed 0 dBd dipole, and a gain antenna must come "
                  "down by its gain)"
                  if SIXTY_METRE_BAND.contains(lo, hi) else "")
        unit = "W ERP" if dipole else "W PEP"
        if emission.power_w is None:
            raise NotPermitted(
                f"§97.313: {lo/1e6:.6f}–{hi/1e6:.6f} MHz is capped at {cap:g} "
                f"{unit}{dipole} and this emission does not state its power. "
                "Declare it rather than transmitting an unknown amount into a "
                "limited band.")
        if emission.power_w > cap:
            raise NotPermitted(
                f"§97.313: {emission.power_w:g} W exceeds the {cap:g} {unit} limit "
                f"for {lo/1e6:.6f}–{hi/1e6:.6f} MHz{dipole}.")


def _nearest(lo: float, hi: float, segments: tuple[Segment, ...]) -> str:
    mid = (lo + hi) / 2
    s = min(segments, key=lambda s: min(abs(s.lo - mid), abs(s.hi - mid)))
    return f"{s.lo/1e6:.4f}–{s.hi/1e6:.4f} MHz"
