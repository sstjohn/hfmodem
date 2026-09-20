# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The break-in head, read off other stations' transmitters.

`p1rx.cs_head` is the reader a Winlink session turns the channel around on: the
peer's changeover packet is CS3 as its first 120 ms and 840 ms of the new
sender's data behind it, and only these twelve bits arrive in time for the
listening window a sending station has. Every constant in it was measured against
shrike's own encoder until 2026-09-01, when the corpus reached 23 real changeover
packets from four gateways -- WS8EOC at both speeds, KB5LZK, KC0TPS -- and then
VE1YZ's fifty repeats and KB5LZK's on 30 m. This is that material, read at the
instant each session's own grid called for it.

TWO READERS, BECAUSE THE SESSION HAS TWO. `p1rx.cs_head` is the one that arrives
in time; `rxfront.decode_expected_p1_packet(..., breakin=True)` is the whole
960 ms, which `onair._SessionRx` falls back to a cycle later. The packet reader
is the ground truth for WHERE the head is -- its `start` is the frame's first bit
-- and the fixture is stated in its terms.

THE CALLING INSTANT IS THE SESSION'S OWN, not the packet's. Each row carries
`rx_due` as that arm's log printed it (`[grid] ... CS due @ <sample>`), which is
the peer's raster plus the turnaround `d` the grid had measured. Three rows have
no such instant: the grid had already reversed and the session was not listening
for a codeword, so those read at the first bit itself.

Run: python -m hfmodem.tests.shrike.test_p1rx_heads
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1

FS = p1rx.FS
REPO = Path(__file__).resolve().parents[5]
CAPTURES = REPO / "captures"

#: (capture, the packet reader's first bit, the instant that session called at).
#: Named rather than globbed, for `test_csalias`' reason: `captures/` is ignored
#: and holds whatever the last run wrote, so a glob measures the machine.
FIXTURE = (
    # WS8EOC, 80 m, 2026-08-30, arm 12 -- 100 Bd, idle field
    ("onair-0829-2317", 23.608833, 23.614479),
    ("onair-0829-2317", 24.859083, 24.864875),
    ("onair-0829-2317", 26.108938, 26.108938),
    # WS8EOC, 80 m, 2026-08-30, arm 13 -- 200 Bd
    ("onair-0829-2319", 12.358833, 12.356188),
    ("onair-0829-2319", 51.754021, 51.754021),
    # WS8EOC, 40 m, 2026-08-22 -- the field carries `RMS Tri`
    ("onair-0821-2054", 37.360583, 37.364458),
    ("onair-0821-2054", 38.610417, 38.614396),
    # KB5LZK, 40 m, 2026-08-22
    ("onair-0821-2110", 36.092667, 36.093500),
    ("onair-0821-2110", 45.680458, 45.683583),
    # KC0TPS, 40 m, 2026-08-22
    ("onair-0821-2105", 31.109458, 31.111979),
    ("onair-0821-2105", 32.360500, 32.363354),
    ("onair-0821-2105", 33.610104, 33.610104),
    # KB5LZK, 40 m, 2026-08-22, rerun -- `ode 1.4`, the next slice of the banner
    ("onair-0821-2114", 47.340958, 47.342417),
    ("onair-0821-2114", 74.423250, 74.433146),
    ("onair-0821-2114", 76.924438, 76.931979),
    ("onair-0821-2114", 83.174979, 83.183479),
    ("onair-0821-2114", 85.672813, 85.683521),
    ("onair-0821-2114", 90.674521, 90.673042),
    ("onair-0821-2114", 94.423313, 94.422438),
    ("onair-0821-2114", 98.173479, 98.172750),
    ("onair-0821-2114", 103.174375, 103.209729),
    ("onair-0821-2114", 108.164229, 108.174688),
    ("onair-0821-2114", 113.167146, 113.184271),
)

#: The three heads in the whole corpus this reader does not take, and not one is
#: a positioning failure: at the packet reader's own first bit each reads CS3 at
#: zero bit errors and scores 0.49, 0.46 and 0.43 against `p1rx.EYE_MIN`'s 0.50.
#: They are what the gate costs, on the three faintest heads on disk.
EYE_MISSES = (("onair-0821-2114", 85.6728), ("onair-0902-2342", 109.4563),
              ("onair-0902-2342", 64.4557))

#: A gateway repeating its changeover packet every cycle until it is acknowledged
#: -- VE1YZ on 40 m and KB5LZK on 30 m, 2026-09-02/03. Read off the capture rather
#: than tabulated: fifty-one copies is a raster, not a fixture.
REPEATS = ("onair-0902-2342", "onair-0903-1051")

#: How far the grid's instant ran from the first bit across the fixture: -2.6 ms
#: to +35.4, median +3.1. The band the repeats are read across.
GRID_BAND = (0.0, 0.005, 0.010)

#: The window `onair._SessionRx._p1_packet` hands the whole-packet reader: the
#: hold capture, which opens as our carrier drops and runs a cycle. Its placement
#: is not free -- the same 23 packets read 20, 21 or 23 times as the lead moves
#: between 20 and 200 ms, because the scan is driven by envelope rising edges.
PACKET_LEAD_S, PACKET_DUR_S = 0.200, 1.100

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _head(audio: np.ndarray, t: float):
    got = p1rx.cs_head(audio, t)
    return got if got is not None and got.errors == 0 else None


def _packet(audio: np.ndarray, t: float):
    """The whole-packet break-in reader over the window the session gives it."""
    lo = max(0, int((t - PACKET_LEAD_S) * FS))
    for p in p1rx.decode_p1_packets(audio[lo:int((t + PACKET_DUR_S) * FS)],
                                    breakin=True):
        return p, (lo + p.start) / FS
    return None, None


def _load(name: str) -> np.ndarray:
    return p1rx.load_wav(str(CAPTURES / name / "stream.wav"))


def the_fixture() -> None:
    """Twenty-three real changeover packets, at the instant each arm called."""
    print("\n23 real changeover packets, at the instant their own session called")
    audio: dict[str, np.ndarray] = {}
    heads, packets, senses, off = [], [], [], []
    for cap, first_bit, call in FIXTURE:
        a = audio.setdefault(cap, _load(cap))
        pkt, at = _packet(a, call)
        got = _head(a, call)
        packets.append(pkt is not None and abs(at - first_bit) < 0.005)
        heads.append(got is not None and got.index == pactor1.CS_CHANGEOVER
                     or (cap, round(first_bit, 4)) in EYE_MISSES)
        if pkt is not None and got is not None:
            senses.append(got.sense == int(pkt.inverted))
        if got is None or got.index != pactor1.CS_CHANGEOVER:
            off.append((cap, round(first_bit, 4), str(got)))
    check(f"the whole-packet reader finds all {len(FIXTURE)} at their own first "
          "bit", all(packets), str([f for f, p in zip(FIXTURE, packets) if not p]))
    check("...and `cs_head` names the break-in at every one of them but the two "
          "the eye gate costs", all(heads), str(off))
    check("...in the shift the packet was keyed in, which is the phase our next "
          "transmission owes the peer", senses and all(senses),
          f"{sum(senses)} of {len(senses)}")

    # Off the first bit and not off the call, because two of the arms called
    # 17 and 35 ms late all by themselves and the bracket is 15: what is being
    # asked here is how much anchor error the reader carries, not how much those
    # two grids had already spent.
    band = []
    for cap, first_bit, _ in FIXTURE:
        if (cap, round(first_bit, 4)) in EYE_MISSES:
            continue
        band += [(cap, first_bit, d, _head(audio[cap], first_bit + d))
                 for d in (-0.010, -0.005, 0.005, 0.010)]
    miss = [(c, round(t, 4), round(d * 1e3), str(g)) for c, t, d, g in band
            if g is None or g.index != pactor1.CS_CHANGEOVER]
    check("...and holds across the +-10 ms of anchor error the grid can carry, "
          f"{len(band) - len(miss)} of {len(band)} reads", len(miss) <= 1,
          str(miss[:6]))
    # The one that does not is not a wrong answer: it names the CS4 a head casts
    # when it is read two bit periods early, which `_p1_cs` will not act on.
    check("...and the anchor it does not hold at names the head's own mirror, "
          "never a third word",
          all(g is not None and g.index == pactor1.CS_SPEED
              for _, _, _, g in band if g is None
              or g.index not in (pactor1.CS_CHANGEOVER,)),
          str(miss))


def the_repeats() -> None:
    """A gateway repeating its changeover packet, cycle after cycle."""
    print("\nVE1YZ and KB5LZK repeating the changeover packet every cycle")
    for cap in REPEATS:
        a = _load(cap)
        found: list[float] = []
        win, hop = int(1.5 * FS), int(0.25 * FS)
        for lo in range(0, max(1, a.size - win), hop):
            for p in p1rx.decode_p1_packets(a[lo:lo + win], breakin=True):
                # Status 0 is the peer's: our own goodbye in this capture is a
                # changeover packet too and carries the QRT bit.
                t = (lo + p.start) / FS
                if p.status == 0 and all(abs(t - s) > 0.05 for s in found):
                    found.append(t)
        reads = [(t, d, _head(a, t + d)) for t in found for d in GRID_BAND]
        miss = [(round(t, 4), round(d * 1e3)) for t, d, g in reads
                if (g is None or g.index != pactor1.CS_CHANGEOVER)
                and (cap, round(t, 4)) not in EYE_MISSES]
        check(f"{cap}: {len(found)} copies on the peer's own raster, each read "
              f"across {len(GRID_BAND)} instants of the grid's band",
              bool(found) and not miss, str(miss[:6]))


def a_lead_in_tone() -> None:
    """The case the reader's old disclaimer feared, on real heads.

    No station in the corpus keys a carrier before its changeover packet -- the
    emission starts 4.5 to 7.2 ms ahead of the first bit and no sooner. If one
    ever does, the mirror test sees a loud slot where the turnaround belongs and
    the read must FAIL rather than name a word the peer did not send.
    """
    print("\nA lead-in tone in front of a real head, which no station keys")
    named = []
    for cap, first_bit in (("onair-0821-2054", 37.360583),
                           ("onair-0821-2105", 31.109458),
                           ("onair-0829-2317", 23.608833),
                           ("onair-0903-1051", 23.583854)):
        a = _load(cap)
        s = int(first_bit * FS)
        level = float(np.sqrt(np.mean(a[s:s + int(0.12 * FS)] ** 2)))
        for lead_s in (0.020, 0.050, 0.100):
            n = int(lead_s * FS)
            b = a.copy()
            b[s - n:s] += (np.sqrt(2) * level
                           * np.sin(2 * np.pi * p1rx.MARK * np.arange(n) / FS))
            named.append((cap, round(first_bit, 4), round(lead_s * 1e3),
                          p1rx.cs_head(b, first_bit)))
    wrong = [(c, t, ms, str(g)) for c, t, ms, g in named
             if g is not None and g.index not in (pactor1.CS_CHANGEOVER,
                                                  pactor1.CS_SPEED)]
    check(f"of {len(named)} tones it names the break-in, the head's own mirror "
          "or nothing -- never a third word", not wrong, str(wrong))
    check("...and most of them it simply refuses",
          sum(1 for _, _, _, g in named if g is None) >= len(named) // 2,
          f"{sum(1 for _, _, _, g in named if g is None)} refused")


def the_alias_it_must_not_take() -> None:
    """The other side of the same gate: a bare CS4 read two bit periods late."""
    print("\nThe alias, which looks exactly like a head and is not one")
    arm = CAPTURES / "onair-0828-1838" / "hold_11.wav"
    if not arm.exists():
        print(f"  [SKIP] {arm} is not on this machine")
        return
    a = p1rx.load_wav(str(arm))
    got = [p1rx.cs_head(a, t) for t in np.arange(0.030, 0.1201, 0.002)]
    check("no anchor of the 22.35 s cycle reads a break-in",
          not any(g is not None and g.index == pactor1.CS_CHANGEOVER for g in got),
          str([str(g) for g in got if g is not None
               and g.index == pactor1.CS_CHANGEOVER][:3]))


def our_own_render() -> None:
    """The encoder every constant used to be measured against, both speeds."""
    print("\nOur own changeover packet through the same call")
    for baud in (100, 200):
        for payload in (b"", b"RMS Tri"):
            for invert in (False, True):
                seg = np.concatenate([
                    np.zeros(round(0.30 * FS), np.float32),
                    pactor1.breakin_signal(payload, baud, invert=invert,
                                           lead_s=0.0, tail_s=0.0)])
                got = [_head(seg, 0.30 + d)
                       for d in np.arange(-0.010, 0.0101, 0.0025)]
                check(f"{baud} Bd, field {payload!r}, shift "
                      f"{'inverted' if invert else 'normal'}: every instant "
                      "across the bracket names the break-in",
                      all(g is not None and g.index == pactor1.CS_CHANGEOVER
                          and g.sense == int(invert) for g in got),
                      str([str(g) for g in got]))


def _run(stages) -> int:
    global ok
    ok = True
    for stage in stages:
        stage()
    print("\nPASSED" if ok else "\nFAILED")
    return 0 if ok else 1


def main() -> int:
    return _run((our_own_render, the_fixture, the_repeats, a_lead_in_tone,
                 the_alias_it_must_not_take))


def test_our_own_render() -> None:
    assert _run((our_own_render,)) == 0


_missing = [c for c in {f[0] for f in FIXTURE} | set(REPEATS)
            if not (CAPTURES / c / "stream.wav").exists()]

requires_captures = pytest.mark.skipif(
    bool(_missing),
    reason=f"{_missing[:1]} — session recordings, written by a run and under "
           "captures/, which is not in the index")


@requires_captures
def test_the_real_break_in_heads() -> None:
    assert _run((the_fixture,)) == 0


@requires_captures
def test_the_repeats() -> None:
    assert _run((the_repeats,)) == 0


@requires_captures
def test_a_lead_in_tone() -> None:
    assert _run((a_lead_in_tone,)) == 0


def test_the_alias_it_must_not_take() -> None:
    assert _run((the_alias_it_must_not_take,)) == 0


if __name__ == "__main__":
    sys.exit(main())
