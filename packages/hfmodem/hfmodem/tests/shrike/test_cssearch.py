# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The width `p1rx.cs_anchored` searches, and what it cost to open it.

shrike read a PACTOR-1 control signal at one predicted instant, half a bit either
side, because the one written record of the read does exactly that -- `receive_cs`
at hfkernel/fsk/pactor.c:745, twelve bit periods from a latched instant with no
search. The 1990 description does not say that. It gives a 0.29 s
`Fenster fuer Kontrollsignalempfang` for a 0.12 s signal and has the master SEARCH
it, and nothing in this station's record tests the narrow reading: 149 `[ack]`
lines and the largest 2.1 ms out, so no codeword has ever been placed outside the
anchor to see what happens to it (`test_ackoff`).

So the width was measured rather than argued, and this file is the curve's
operating point held in place. The sweep ran from 5 to 145 ms; what is pinned
here is:

  * the read is a SEARCH -- a real gateway's codeword displaced 10 and 15 ms from
    the anchor is still read as the word it is, which a latched instant cannot do;
  * the search finds every answer the point read found and six it did not, all six
    on the cycle's own shift parity;
  * it stops before the ghost. At 20 ms the CS3 that every CS4 casts two bits
    along wins the score in two of the twenty, so the width is a ceiling and not
    a floor;
  * and the false-lock floor: 16 201 twelve-bit reads of quiet audio from nine
    recordings, against which the point read manufactures 18 codewords and this
    one 58. That number is the price of the width and it is held here so a
    widening cannot be made without paying it again.

Run:  pytest hfmodem/tests/shrike/test_cssearch.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, rxfront
from hfmodem.tests import evidence
from hfmodem.tests.shrike import archive

FS = rxfront.FS

#: WS8EOC, 2026-07-30 2036 -- a real gateway answering a real link, and the one
#: session whose answer instants are measured. Copied from `test_p1cs`, which
#: reads the same twenty windows for what the codewords SAY; this file reads them
#: for where the reader is willing to look.
ANCHORED_AT = {
    2: 94.2, 3: 93.8, 4: 93.8, 5: 92.4, 6: 96.6, 7: 96.5, 8: 93.1,
    9: 93.9, 10: 94.8, 11: 101.6, 15: 94.5, 16: 94.5, 17: 1052.6,
    18: 1047.9, 19: 1048.3, 20: 72.2, 21: 88.3, 22: 86.9, 23: 88.7,
    24: 1046.3,
}
SESSION = evidence.CORPUS / "pactor-ws8eoc-20260730" / "onair-0730-2036"

#: Every instant in the quiet sweep is this far from every burst the detector
#: finds, so no read can be charged with half of a real codeword at any width the
#: sweep tried.
QUIET_GUARD_S = 0.545

#: What the sweep of 2026-08-28 measured at the shipped width, over the corpus
#: `_quiet` builds. The point read manufactured 18 in the same reads.
QUIET_ACCEPTS = 58

requires_corpus = pytest.mark.skipif(
    not (SESSION / "rx_02.wav").exists(),
    reason=f"the 2026-07-30 WS8EOC sessions are absent from {SESSION.parent}")


def _windows() -> dict[int, np.ndarray]:
    return {k: rxfront.load_wav(str(SESSION / f"rx_{k:02d}.wav"))
            for k in ANCHORED_AT}


def _read(seg, k, off_ms=0.0):
    got = p1rx.cs_anchored(seg, ANCHORED_AT[k] / 1e3 + off_ms / 1e3)
    return None if got is None or got.unassigned else (got.index, got.sense)


@requires_corpus
def test_the_read_is_a_search_and_a_displaced_codeword_still_lands():
    """THE ONE THAT FAILS IF THE WINDOW IS NARROWED BACK TO AN INSTANT.

    The corpus holds no recording of a peer answering off-anchor, and that is the
    hole this cannot fill. What it can do is move OUR anchor off a codeword a real
    gateway really transmitted, which is the same geometry seen from the other
    end, and ask whether the word is still read as itself. At half a bit it is
    not: past 5 ms the reader that shipped until 2026-08-28 returned nothing at
    all, in every one of these windows.
    """
    segs = _windows()
    want = {k: _read(segs[k], k) for k in ANCHORED_AT}
    assert all(want.values()), [k for k, v in want.items() if v is None]
    for off in (10.0, 15.0):
        held = [k for k in ANCHORED_AT
                if _read(segs[k], k, off) == want[k]
                and _read(segs[k], k, -off) == want[k]]
        assert len(held) >= 15, (off, sorted(held))


@requires_corpus
def test_the_search_reads_every_answer_the_point_read_left():
    """Twenty of twenty, where the point read took fourteen.

    The six it adds are not taken on trust. Each one lands within 2.5 ms of an
    instant measured independently, each carries the shift the cycle owes --
    `sense == window & 1`, `pactor1-data-packets.md` sec 7 -- and each falls on the
    right side of the session's own CS1-then-CS4 partition. Six for six on the
    parity alone is one chance in 64.
    """
    segs = _windows()
    got = {k: _read(segs[k], k) for k in ANCHORED_AT}
    assert all(got.values()), [k for k, v in got.items() if v is None]
    assert all(sense == (k & 1) for k, (_, sense) in got.items()), got
    acks = [k for k, (word, _) in got.items() if word == pactor1.CS_ACK_A]
    reps = [k for k, (word, _) in got.items() if word == pactor1.CS_SPEED]
    assert len(acks) + len(reps) == len(got), got
    assert acks and reps and max(acks) < min(reps), (acks, reps)


@requires_corpus
def test_the_width_is_a_ceiling_and_the_ghost_is_why():
    """What stops the window at the description's 0.29 s: a WRONG WORD.

    CS3 sits two bits along from CS4 in this code, so an alignment one bit period
    late off a CS4 reads a break-in at zero errors -- and at exactly +20 ms it
    outscores the word that is really there in two of these twenty windows. A
    mis-taken changeover is unrecoverable where a mis-taken acknowledgement is
    not (`pactor1-timing.md` sec 5), so the width stops a half-bit in front of it
    rather than at the edge of what the grid will pull for.
    """
    assert 3 * p1rx.CS_ANCHOR_S <= p1rx.CS_SEARCH_HALF_S < 0.020
    segs = _windows()
    was = {k: _read(segs[k], k) for k in ANCHORED_AT}
    shipped = p1rx.CS_SEARCH_HALF_S
    p1rx.CS_SEARCH_HALF_S = 0.020
    try:
        wider = {k: _read(segs[k], k) for k in ANCHORED_AT}
    finally:
        p1rx.CS_SEARCH_HALF_S = shipped
    ghosts = {k: (was[k], wider[k]) for k in ANCHORED_AT if wider[k] != was[k]}
    assert len(ghosts) >= 2, ghosts
    assert all(new_[0] == pactor1.CS_CHANGEOVER for _, new_ in ghosts.values()), \
        ghosts


def test_the_peers_tolerance_is_not_this_receivers_reach():
    """One constant cannot be both, and it was one constant until 2026-08-28.

    `CS_ANCHOR_S` is what `_ack_gap_line` assumes of the station receiving our
    acknowledgement -- the reference reader's half bit, untested by anything we
    hold, since no codeword of ours has ever been placed outside it. Widening the
    reader must not quietly widen that alarm.
    """
    assert p1rx.CS_ANCHOR_S == 0.005
    assert p1rx.CS_SEARCH_HALF_S > p1rx.CS_ANCHOR_S


def _quiet() -> list[tuple[np.ndarray, list[float]]]:
    """Twelve-bit reads with no control signal under them.

    The two silent-gateway recordings -- 77 s apiece of a Winlink gateway calling
    with our transmitter off, so real band, real receiver, real bursts to stay
    clear of -- and every receive window of the seven WS8EOC sessions of
    2026-07-30. Quiet is the detector's own answer: `QUIET_GUARD_S` from every
    burst it finds, which is wider than any width the sweep tried.
    """
    out = []
    paths = [p for p in archive.SILENT_GATEWAYS if p.exists()]
    root = SESSION.parent
    if root.exists():
        for s in sorted(p for p in root.iterdir() if p.is_dir()):
            paths += sorted(s.glob("rx_*.wav")) + sorted(s.glob("hold_*.wav"))
    for path in paths:
        seg = rxfront.load_wav(str(path))
        bursts = rxfront.p1_burst_onsets(seg)
        end = seg.size / FS - QUIET_GUARD_S - 0.14
        ts = [float(t) for t in np.arange(QUIET_GUARD_S, end, 0.005)
              if all(abs(t - b) > QUIET_GUARD_S for b in bursts)]
        if ts:
            out.append((seg, ts))
    return out


@requires_corpus
@pytest.mark.skipif(not all(p.exists() for p in archive.SILENT_GATEWAYS),
                    reason="the two silent-gateway recordings are absent")
def test_the_false_lock_floor_the_width_was_bought_at():
    """The other half of the curve, and the half a width sweep is dishonest without.

    A search is many trials where the point read was one, and the phantom rate is
    linear in the number of them: 1.5 per 1000 quiet reads at half a bit, 3.6
    here, 4.4 at 20 ms and 24 at the description's 145. `CS_SEARCH_EYE_MIN` is
    what holds the slope down -- ungated the same window reads 11.9 per 1000 --
    and it does not separate the two populations, so this is a bound and not a
    proof of anything.
    """
    reads = _quiet()
    n = sum(len(ts) for _, ts in reads)
    assert n >= 15000, n
    accepts = sum(1 for seg, ts in reads for t in ts
                  if (lambda r: r is not None and not r.unassigned)
                     (p1rx.cs_anchored(seg, t)))
    assert accepts <= QUIET_ACCEPTS + 15, f"{accepts} accepts in {n} quiet reads"
    assert accepts / n < 0.005, f"{accepts} accepts in {n} quiet reads"
