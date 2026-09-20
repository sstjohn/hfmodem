# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What audio passband a transmission of ours will actually fill.

One fact the tree kept restating and nowhere held. `station.link.Waveform` used to
carry its own literals, one pair per protocol, and three of the four were inside
the emission they described; `core.regulatory.Emission` is per *call* and takes
the numbers from whoever built it. Neither answered "shrike is about to key here,
what does that occupy" — which is what the channel sense, the receiver's filter
and the regulatory gate all need, and all three now read it from here.

Edges are the -26 dB span of this station's own modulators, which is the measure
§97.3(a)(8) writes bandwidth in, rounded outward. They were once the outermost
carrier plus its symbol rate, on the reasoning that a tone's main lobe ends there
and that this was the safe direction. For the FSK waveforms it was: pactor
measures 436.9-2568.6 against the 380-2620 that rule gives. For VARA it was not —
its handshake alphabet measures 612.4 Hz at the bottom where the lowest carrier
sits at 679.7, so the rule ran 68 Hz inside the emission rather than outside it.

The distinction matters because these numbers are what the regulatory gate judges
a band edge on. A sense band that is too narrow costs a wait; a declared passband
that is too narrow is a station keying outside the segment it believes it is in.

A key is answered for the whole transmission and not for its first burst. PACTOR
is where that bites: the burst a launcher can name is the PACTOR-1 connect at
1200-1800 Hz, but the ARQ that follows upshifts on its own to `max_sl`, and the
sense runs before there is a link to ask. So the answer here is the widest burst
the session can reach, and `tests/core/test_occupied.py` holds it against
`shrike.session.occupied_hz`, which computes each one from the waveform.

A CALL MAY BE NARROWER THAN ITS PROTOCOL, and then it is entitled to a narrower
sense -- but only if the narrowness is enforced rather than intended. That is the
whole of the PACTOR entry below: unqualified it answers for a session that may
climb to speed level 6, because one did, on air, on 2026-08-18
(`link upgraded to PACTOR-3 at SL3`, then `SL3 pkt` keyed); qualified with the
ceiling the call is run under it answers for that ceiling instead. A launcher may
only ask for the narrow answer by also handing the modem the flag that holds the
link there, so the declaration and the emission cannot come apart.
"""
from __future__ import annotations

from .busy import FULL_BAND

#: The passband the rig is asked for, and the one every calibration in
#: `core.busy` was recorded through -- so it is `FULL_BAND`'s top edge rather
#: than a second copy of the same number. Narrowing it to the emission is a real
#: receive gain and it is not taken here: it would move the noise reference the
#: occupancy thresholds are measured against, and no capture exists at any other
#: width. `dsp` filtering does the same job after the fact.
FILTER_HZ = int(FULL_BAND[1])

#: ``(modem, bandwidth) -> (low_hz, high_hz)``, bandwidth as the launchers spell it.
OCCUPIED_HZ = {
    ("pactor", ""): (380.0, 2620.0),         # SL6 data, 480-2520 Hz at 100 Bd
    # The highest speed level the call may reach; 1 is a link held in PACTOR-1,
    # since PACTOR-3's own level 1 cannot be entered cold (arq.ArqConfig.entry_sl)
    # and no call ever opens there. Only `--pactor1-only` reaches this row.
    ("pactor", "1"): (1200.0, 1800.0),       # the FSK connect, 1400/1600 at 100 Bd
    # The VARA rows are the -26 dB span of this station's own encoder, over the
    # handshake alphabet swept tone by tone and the DATA over: 612.4-2401.3 at
    # BW2300, 1084.7-1866.6 at BW500, 318.7-2527.3 at BW2750. The carrier
    # extremes alone said 680-2297 and 1172-1781, which is 68 Hz narrow at the
    # bottom of BW2300 and 104 at the top -- a tone's skirts are the emission as
    # much as its centre is. At BW2750 the over's bottom skirt runs 244 Hz below
    # its lowest carrier at 562.5: twenty unwindowed 512-sample bins against
    # BW2300's sixteen, and the base burst carries no cyclic prefix to shape it.
    ("vara", "500"): (1080.0, 1870.0),
    ("vara", "2300"): (610.0, 2405.0),
    ("vara", "2750"): (315.0, 2530.0),
    ("ardop", "200"): (1375.0, 1625.0),      # 4FSK 1425-1575 at 50 Bd
    ("ardop", "500"): (1250.0, 1750.0),
    ("ardop", "1000"): (1000.0, 2000.0),
    ("ardop", "2000"): (500.0, 2500.0),
    # Sabir's 56-carrier gears centered at 1500 Hz, including shaped skirts.
    # A 32-payload sweep per gear measures 147.9..2852.1 Hz at -26 dB;
    # rounded outward with margin. Carrier endpoints alone are insufficient.
    ("sabir", ""): (140.0, 2860.0),
}


def occupied_hz(modem: str, bw: str = "") -> tuple[float, float]:
    """The band ``modem`` at ``bw`` will fill, or `FULL_BAND` if unlisted.

    The fallback suits a listener and nothing else. `FULL_BAND` reaches the filter
    at the top but starts at 400, which sits *inside* the sabir row and the pactor
    row at the bottom — so an unlisted mode is answered plausibly rather than
    widely. That is fine for a sense, which costs a wait when it is wrong.
    `keyed_hz` is the gate's answer, and it refuses instead.
    """
    return OCCUPIED_HZ.get((modem.lower(), str(bw)), FULL_BAND)


def keyed_hz(modem: str, bw: str = "") -> tuple[float, float]:
    """The band to judge a transmission of ours by, before it is permitted.

    Two things separate this from `occupied_hz`. An unlisted waveform raises
    rather than defaulting, because a passband that is a guess makes the band edge
    a guess and there is no safe direction to guess in. And a call that has not
    pinned its bandwidth is answered with the union of everything that modem can
    key, on the same reasoning the pactor rows are written on: what a gate must
    judge is the widest burst the call is permitted to reach, not the narrowest it
    happens to open with.
    """
    name, want = modem.lower(), str(bw)
    if (name, want) in OCCUPIED_HZ:
        return OCCUPIED_HZ[(name, want)]
    rows = [v for (m, _), v in OCCUPIED_HZ.items() if m == name]
    if rows and not want:
        return min(lo for lo, _ in rows), max(hi for _, hi in rows)
    known = sorted(b or "(unpinned)" for m, b in OCCUPIED_HZ if m == name)
    raise ValueError(
        f"no passband on file for {modem!r} at bandwidth {want!r}, so where its "
        f"emission ends is unknown and no band edge can be checked against it. "
        + (f"Known for {modem!r}: {', '.join(known)}."
           if known else
           "Measure it from the modulator and add a row to "
           "core.occupied.OCCUPIED_HZ before keying it."))


def clipped(passband_hz: int | None, band: tuple[float, float],
            centre_hz: float = 1500.0) -> str:
    """Why ``passband_hz`` is too narrow for ``band``, or ``""`` if it is not.

    The rig reports a width and not where it sits, so this assumes the filter is
    centred on ``centre_hz`` — true of PKTUSB on this station's rig, and the one
    part of the answer CAT cannot confirm.
    """
    if passband_hz is None:
        return ("the rig did not report a filter width, so whether our own signal "
                "leaves the radio whole is unknown")
    need = 2 * max(centre_hz - band[0], band[1] - centre_hz)
    if passband_hz >= need:
        return ""
    return (f"the rig's filter is {passband_hz} Hz and this emission needs "
            f"{need:.0f} Hz about {centre_hz:.0f} — {band[0]:.0f}-{band[1]:.0f} Hz "
            f"of audio into a filter that passes "
            f"{centre_hz - passband_hz / 2:.0f}-{centre_hz + passband_hz / 2:.0f}")
