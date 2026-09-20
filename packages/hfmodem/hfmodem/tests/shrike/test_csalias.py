# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The one control signal that casts another, and the channel it gave away.

A bare CS4 read two bit periods late IS CS3, bit for bit, in the same shift
sense. CS3 is the break-in, so that read hands the link to a station that asked
for nothing. It happened: `captures/onair-0828-1838` at 22.35 s, where the same
120 ms of air reads CS4 for any anchor at or below 70 ms into the window and CS3
for any at or above 72, the session's own anchor sat at 84, and sixteen cycles
then passed with both ends receiving and neither sending.

Two things hold it off, and they are deliberately not the same thing:

  * `p1rx.cs_anchored` refuses the specific alias -- an accepted CS3 whose
    alignment is `CS_ALIAS_BITS` behind a zero-error CS4 in the same sense;
  * `onair._SessionRx._p1_cs` refuses the whole class -- the anchored path
    positions by silence BEHIND the word, which a changeover packet cannot have,
    so a CS3 from it is a contradiction whatever cast it.

Neither is redundant. The first recovers the CS4 the alias was standing on,
which the second alone would throw away; the second covers the CS3s the first
cannot see, which are the ones whose CS4 fell outside the bracket.
`CS_ALIAS_BITS = 0` restores the reader exactly as it was, and every stage below
is run both ways against it, so the pins fail if the alias is ever let back in.

THE MIRROR IS WHY THE FIRST ONE ALSO WEIGHS THE SILENCE. A genuine changeover
head read two bit periods early -- its turnaround gap in front of it -- is a
zero-error CS4 just as exactly, so a test on the codewords alone refuses every
real break-in it is shown. `a_real_break_in_still_reads` is that direction, and
it failed before the energies either side of the word were compared.

A THIRD ONE JOINED THEM once the corpus held real changeover packets to set it
against. `cs_head` used to read the same false CS3 on arm 1's cycle -- its own
lead-silence term does not exclude the alias -- and what separates the two is the
LEAD, weighed against the loudest of what surrounds the word: the two slots ahead
of an alias are the burst's own first bits and sit above both, while a head has
the turnaround there whether or not a packet follows it. Scale-free and
threshold-free like `cs_anchored`'s, it costs none of the 23 real changeover
packets the corpus holds, and it takes a bare rendered codeword -- which weighing
the lead against the 840 ms behind a head alone did not.

Run: python -m hfmodem.tests.shrike.test_csalias
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1

FS = p1rx.FS
REPO = Path(__file__).resolve().parents[5]

#: Arm 1 of 2026-08-28, and the window its 22.35 s cycle was decided on. The
#: whole session is on disk; `hold_11` is the cycle the role reversed in.
ARM1 = REPO / "captures" / "onair-0828-1838"
ARM1_CYCLE = ARM1 / "hold_11.wav"
#: Where the session's own grid put the anchor in that window, which began at
#: 22.266 s: 84 ms in, 2.8 bit periods past the codeword.
ARM1_ANCHOR = 0.084

#: The only recording in which a gateway ever took the channel from this
#: station and sent a readable field behind the codeword, and where its head
#: sits on that stream. `test_breakin` reads the same instant.
K4MSU = REPO / "captures" / "onair-0819-2210" / "stream.wav"
K4MSU_AT = 64.855

#: Arm 2 of the same slot, and the six sessions of 2026-07-30 in which WS8EOC
#: answered every cycle. Named rather than globbed, for `test_cs_evidence`'s
#: reason: `captures/` is ignored and holds whatever the last run wrote, so a
#: glob measures the machine and these measure the checkout.
CORPUS = [REPO / "captures" / n for n in
          ("onair-0828-1838", "onair-0828-1844", "onair-0730-2036",
           "onair-0730-2037", "onair-0730-2039", "onair-0730-2047",
           "onair-0730-2049", "onair-0730-2050")]

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _sweep(audio: np.ndarray, lo: float, hi: float, step: float = 0.002):
    """What `cs_anchored` returns at every anchor from `lo` to `hi`."""
    out: dict[tuple[int, int] | None, list[float]] = {}
    for t in np.arange(lo, hi + step / 2, step):
        got = p1rx.cs_anchored(audio, float(t))
        out.setdefault(None if got is None else (got.index, got.sense),
                       []).append(round(float(t) * 1e3, 1))
    return out


def _both_ways(fn):
    """`fn` with the alias guard off and on. Off is the reader as it was."""
    saved = p1rx.CS_ALIAS_BITS
    try:
        p1rx.CS_ALIAS_BITS = 0
        was = fn()
        p1rx.CS_ALIAS_BITS = saved
        return was, fn()
    finally:
        p1rx.CS_ALIAS_BITS = saved


def _slot(word: int, invert: bool, at: float = 0.10, dur: float = 0.50,
          noise: float = 0.0, seed: int = 7) -> np.ndarray:
    """One bare control signal in a slot of silence, as a peer keys it."""
    burst = np.asarray(pactor1.control_signal(word, invert=invert), float)
    a = np.zeros(int(dur * FS))
    a[int(at * FS):int(at * FS) + burst.size] = burst
    if noise:
        a += np.random.default_rng(seed).normal(0, noise, a.size)
    return a


def _cs_line(seg: np.ndarray, t: float) -> str:
    """What the session's PACTOR-1 reader would print for the cycle at `t`.

    `_p1_cs` reads nothing off its instance -- it is a reader, not a state --
    so the class's own function is called directly rather than a session stood
    up around it.
    """
    ev = onair._SessionRx._p1_cs(seg, int(round(t * FS)))
    return "" if ev is None else ev.text


# ---------------------------------------------------------------------------


def the_algebra() -> None:
    """Every slip in +/-3 bits, every fill, both senses, all four words."""
    print("\nWhich control signals cast which, and at what slip")

    def bits(w: int) -> list[int]:
        return [(w >> i) & 1 for i in range(pactor1.CS_BITS)]   # LSB first, on air

    cast = set()
    for src in range(4):
        for sense in (0, 1):
            air = [b ^ sense for b in bits(pactor1.CONTROL_SIGNALS[src])]
            for slip in [s for s in range(-3, 4) if s]:
                keep = air[slip:] if slip > 0 else air[:slip]
                for fill in product((0, 1), repeat=pactor1.CS_BITS - len(keep)):
                    read = keep + list(fill) if slip > 0 else list(fill) + keep
                    for dst in range(4):
                        for s2 in (0, 1):
                            if read == [b ^ s2 for b in
                                        bits(pactor1.CONTROL_SIGNALS[dst])]:
                                cast.add((src, sense, slip, dst, s2))
    cast -= {c for c in cast if c[0] == c[3] and c[1] == c[4]}
    check("four cross-word aliases exist and no others", len(cast) == 4,
          str(sorted(cast)))
    check("...every one of them is CS3 against CS4",
          {(c[0], c[3]) for c in cast} == {(pactor1.CS_SPEED, pactor1.CS_CHANGEOVER),
                                           (pactor1.CS_CHANGEOVER, pactor1.CS_SPEED)},
          str(sorted({(c[0], c[3]) for c in cast})))
    check(f"...at a slip of exactly +/-{p1rx.CS_ALIAS_BITS} bit periods, which is "
          "what the constant carries",
          {abs(c[2]) for c in cast} == {p1rx.CS_ALIAS_BITS},
          str(sorted({c[2] for c in cast})))
    check("...in the SAME shift sense, so the sense is no help either",
          all(c[1] == c[4] for c in cast))
    check("...and the acknowledgement pair casts nothing at any slip, which is "
          "why this is not a general defence",
          not [c for c in cast if c[0] in (pactor1.CS_ACK_A, pactor1.CS_ACK_B)])

    # The fill is not a coincidence of the silence: `_cs_decide` splits twelve
    # slots six and six, so ten surviving bits of weight 6 or 4 FORCE the two
    # empty ones to the value that completes the other word.
    weights = {sum(b ^ s for b in bits(pactor1.CONTROL_SIGNALS[pactor1.CS_SPEED])
                   [p1rx.CS_ALIAS_BITS:]) for s in (0, 1)}
    check("the two silent slots are forced, not guessed: the ten real bits of a "
          "slipped CS4 already carry weight 6 or 4 against the split's six",
          weights == {6, 4}, str(sorted(weights)))


def a_rendered_burst() -> None:
    """A bare CS4 in a slot, which is the whole of what the peer transmitted."""
    print("\nA bare CS4 in a slot, read from every anchor around it")
    for invert in (False, True):
        for noise in (0.0, 0.02):
            a = _slot(pactor1.CS_SPEED, invert, noise=noise)
            was, now = _both_ways(lambda: _sweep(a, 0.060, 0.180))
            sense = int(invert)
            tag = f"shift {'inverted' if invert else 'normal'}, noise {noise}"
            check(f"{tag}: the reader as it was reads a break-in off it",
                  (pactor1.CS_CHANGEOVER, sense) in was,
                  str(was.get((pactor1.CS_CHANGEOVER, sense))))
            check(f"{tag}: ...20 ms past the CS4's own anchors, which is the alias",
                  min(was[(pactor1.CS_CHANGEOVER, sense)])
                  - max(was[(pactor1.CS_SPEED, sense)]) == 2.0,
                  f"CS4 to {max(was[(pactor1.CS_SPEED, sense)])} ms, "
                  f"CS3 from {min(was[(pactor1.CS_CHANGEOVER, sense)])} ms")
            check(f"{tag}: and no anchor reads a break-in now",
                  (pactor1.CS_CHANGEOVER, sense) not in now,
                  str(now.get((pactor1.CS_CHANGEOVER, sense))))
            check(f"{tag}: ...while every anchor that read the CS4 still does",
                  now.get((pactor1.CS_SPEED, sense)) == was.get((pactor1.CS_SPEED, sense)),
                  f"{len(now.get((pactor1.CS_SPEED, sense), []))} against "
                  f"{len(was.get((pactor1.CS_SPEED, sense), []))}")

    print("\n  NEGATIVE CONTROL: the acknowledgement pair, which casts nothing")
    for word in (pactor1.CS_ACK_A, pactor1.CS_ACK_B):
        for invert in (False, True):
            a = _slot(word, invert)
            was, now = _both_ways(lambda: _sweep(a, 0.060, 0.180))
            check(f"CS{word + 1} shift {'inverted' if invert else 'normal'}: "
                  "the guard changes nothing at any anchor", was == now,
                  f"{sorted(map(str, was))} -> {sorted(map(str, now))}")


def a_real_break_in_still_reads() -> None:
    """The frame this must not cost: a station taking the channel.

    Missing one is the OTHER failure mode and it has already cost a field:
    `test_breakin.the_break_in_the_anchor_could_not_be_aimed_at` holds the
    session where a gateway's whole changeover body reached the disk and nothing
    reached the host. So the packet is rendered and read back through the
    session's own reader at every anchor the head bracket covers, both senses.
    """
    print("\nA changeover packet, which is what a real CS3 arrives inside")
    for invert in (False, True):
        for lead_s in (0.05, 0.09):
            seg = np.asarray(pactor1.breakin_signal(b"BK DE K7ABC", 100,
                                                    invert=invert, lead_s=lead_s,
                                                    tail_s=0.05), float)
            lines = {round(d * 1e3): _cs_line(seg, lead_s + d)
                     for d in np.arange(-0.012, 0.0121, 0.003)}
            tag = f"shift {'inverted' if invert else 'normal'}, lead {lead_s * 1e3:.0f} ms"
            check(f"{tag}: every anchor across the bracket names the break-in",
                  all("CS3/break-in head" in v for v in lines.values()),
                  str({k: v[:24] for k, v in lines.items() if "CS3" not in v}))
            check(f"{tag}: ...and none of them comes off the anchored path, "
                  "which is where it never belonged",
                  not [v for v in lines.values() if "at anchor" in v],
                  str([v for v in lines.values() if "at anchor" in v][:1]))
            check(f"{tag}: ...in the shift the packet was keyed in",
                  all(("inverted" if invert else "normal") in v
                      for v in lines.values()),
                  str(list(lines.values())[:1]))


def the_one_real_break_in() -> None:
    """K4MSU, 3595 kHz, 2026-08-19 22:10, and nothing else like it on disk.

    Everything above is rendered by this station's own encoder, so the only
    recording in which a gateway ever took the channel from us is what says
    whether the guard costs a real one. It must read exactly as it did.
    """
    print("\nThe one changeover a real station ever sent this station")
    st = p1rx.load_wav(str(K4MSU))
    was, now = _both_ways(lambda: {
        round(d * 1e3): _cs_line(st, K4MSU_AT + d)
        for d in np.arange(-0.012, 0.0121, 0.003)})
    check("the guard changes not one anchor of it", was == now,
          str({k: v for k, v in now.items() if was[k] != v}))
    check("...and the break-in reads from every anchor at or past the head",
          all("CS3/break-in head" in v for k, v in now.items() if k >= -3),
          str({k: v[:26] for k, v in now.items() if k >= -3
               and "CS3/break-in head" not in v}))


def the_arm_that_gave_the_channel_away() -> None:
    """`captures/onair-0828-1838`, cycle 11, and the 2 ms that decided it."""
    print("\nArm 1 of 2026-08-28, the 22.35 s cycle")
    a = p1rx.load_wav(str(ARM1_CYCLE))
    was, now = _both_ways(lambda: _sweep(a, 0.030, 0.120))
    cs3, cs4 = (pactor1.CS_CHANGEOVER, 1), (pactor1.CS_SPEED, 1)
    check("the reader as it was: CS4 up to 70 ms and CS3 from 72",
          max(was[cs4]) == 70.0 and min(was[cs3]) == 72.0,
          f"CS4 {min(was[cs4])}-{max(was[cs4])} ms, "
          f"CS3 {min(was[cs3])}-{max(was[cs3])} ms")
    check(f"...and the session's own anchor, {ARM1_ANCHOR * 1e3:.0f} ms, is inside "
          "the half that reads a handover",
          ARM1_ANCHOR * 1e3 in was[cs3])
    check("no anchor of that window reads a break-in now", cs3 not in now,
          str(now.get(cs3)))
    check("...and the CS4 the alias was standing on is read from further out "
          "than it was, not from less",
          max(now[cs4]) >= max(was[cs4]) and min(now[cs4]) <= min(was[cs4]),
          f"{min(now[cs4])}-{max(now[cs4])} ms against "
          f"{min(was[cs4])}-{max(was[cs4])}")
    check("the session's reader names no break-in at the anchor path there",
          "at anchor" not in _cs_line(a, ARM1_ANCHOR),
          _cs_line(a, ARM1_ANCHOR))

    print("\n  THE GAP, AND THE LEAD THAT CLOSED IT")
    # `cs_head` weighs the two slots ahead of the codeword against the loudest of
    # what surrounds it now, so the alignment the alias stands on -- two bit
    # periods into a burst, with the burst's own first bits in front of it -- is
    # the one shape it will not take. The word this cycle really carries is
    # keyed at its own start, where the lead is the turnaround, and naming that
    # is the right answer rather than a second refusal.
    check("`cs_head` names the CS4 that is really there and never the alias",
          (p1rx.cs_head(a, ARM1_ANCHOR) or (pactor1.CS_CHANGEOVER,))[0]
          != pactor1.CS_CHANGEOVER,
          str(p1rx.cs_head(a, ARM1_ANCHOR)))
    check("...so no positioner in the receiver names this cycle a break-in",
          "break-in head" not in _cs_line(a, ARM1_ANCHOR),
          _cs_line(a, ARM1_ANCHOR) or "(nothing)")


def the_corpus() -> int:
    """Every listen window of eight named sessions, both ways."""
    print("\nThe corpus: eight sessions, every window, every anchor")
    files = [f for d in CORPUS for f in sorted(d.glob("hold_*.wav"))
             + sorted(d.glob("rx_*.wav"))]
    if not files:
        print(f"  [SKIP] no on-air sessions under {REPO / 'captures'}")
        return 2

    def tally():
        seen: dict[int, int] = {}
        for f in files:
            a = p1rx.load_wav(str(f))
            for t in np.arange(0.0, a.size / FS - 0.14, 0.005):
                got = p1rx.cs_anchored(a, float(t))
                key = -1 if got is None else got.index
                seen[key] = seen.get(key, 0) + 1
        return seen

    was, now = _both_ways(tally)
    named = {i: pactor1.CS_WORDS[i] for i in range(len(pactor1.CS_WORDS))}
    check("the sessions are all here", len(files) >= 150, f"{len(files)} windows")
    check("the break-ins the reader used to find are mostly gone",
          now.get(pactor1.CS_CHANGEOVER, 0) < was.get(pactor1.CS_CHANGEOVER, 0) / 2,
          f"{was.get(pactor1.CS_CHANGEOVER, 0)} anchors -> "
          f"{now.get(pactor1.CS_CHANGEOVER, 0)}")
    check("...and every one of them was standing on a CS4, so the CS4 count "
          "rises rather than the silence",
          now.get(pactor1.CS_SPEED, 0) > was.get(pactor1.CS_SPEED, 0),
          f"{was.get(pactor1.CS_SPEED, 0)} -> {now.get(pactor1.CS_SPEED, 0)}")
    moved = {named[i]: (was.get(i, 0), now.get(i, 0))
             for i in named
             if i not in (pactor1.CS_CHANGEOVER, pactor1.CS_SPEED)
             and was.get(i, 0) != now.get(i, 0)}
    check("NO OTHER WORD MOVES. The acknowledgements a link runs on are read "
          "from the same anchors they always were", not moved, str(moved))
    return 0


SYNTHETIC = (the_algebra, a_rendered_burst, a_real_break_in_still_reads)
RECORDED = (the_one_real_break_in, the_arm_that_gave_the_channel_away,
            the_corpus)


def _run(stages) -> int:
    global ok
    ok = True
    for stage in stages:
        if stage() == 2:
            return 2
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def main() -> int:
    return _run(SYNTHETIC + RECORDED)


def test_main() -> None:
    assert _run(SYNTHETIC) == 0


@pytest.mark.skipif(
    not K4MSU.exists(),
    reason=f"{K4MSU} is a session recording, written by a run and under "
           "captures/, which is not in the index")
def test_the_one_real_break_in() -> None:
    assert _run((the_one_real_break_in,)) == 0


@pytest.mark.skipif(
    not ARM1_CYCLE.exists(),
    reason=f"{ARM1} is a session recording, written by a run and under "
           "captures/, which is not in the index")
def test_the_arm_that_gave_the_channel_away() -> None:
    assert _run((the_arm_that_gave_the_channel_away,)) == 0


def test_the_corpus() -> None:
    rc = _run((the_corpus,))
    if rc == 2:
        pytest.skip(f"no on-air sessions under {REPO / 'captures'}")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
