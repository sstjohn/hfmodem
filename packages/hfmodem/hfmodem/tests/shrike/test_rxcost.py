# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the session's rolling receiver costs, and what it is allowed to cost.

A PACTOR station has 1.25 s to hear the peer, decide and key, and nothing in this
package had ever asserted that its receiver fits inside that. On 2026-08-02 it
stopped fitting. `rxfront.decode_events` measured 31.8 ms over a 0.75 s window and
846.7 ms over a 0.85 s one on the same recording -- a 27-fold step at the length
where a rolling buffer first holds a whole PACTOR-3 body, after which every slide
ran the envelope-anchored body scan and returned nothing. The sessions where the
peer answered early broke the listen before the buffer reached that length, ran at
cadence and completed an ARQ data phase; the sessions where it did not crossed the
step twice a cycle and stalled. Same code, same gateway, same evening.

So what is pinned here is the COST, which no test had ever looked at, and the
shape of the assertion follows the defect: the fed window may grow and the cost
per cycle may not step with it. A decode that costs more wall clock than the audio
it covers cannot be caught up with -- the next window is longer for exactly the
time the last one overran.

The decode is pinned in the one direction that matters alongside it. Taking the
body scan off the stream must not be a way of going quiet, so the PACTOR-3 packet
the corpus's corroborated recording carries has to keep coming out of the
receiver that no longer runs it, and the file-reading callers have to keep
running it.

REAL OFF-AIR AUDIO, because it is the only thing that shows this: the same
measurement on synthetic noise stays under real time at every length, which is why
nothing offline had ever caught it.

AND THE OTHER CLOCK, which is not the stream's: the read the cycle's ANSWER hangs
on. `onair._SessionRx.deep_scan` runs once per cycle in front of the key, and what
it has there is `onair.PREKEY_RESERVE_S` -- 30 ms, of which 8 stand past
`key_notice` before the settle starts paying. `test_the_cycles_own_read_fits_it`
walks every speed level at both cycle lengths through that entry point and reports
the table, because the level and the cycle length are the peer's to choose and the
cost moves by an order of magnitude across them.

Where it stood on 2026-09-03 and where it stands (this box, one process, best
of three, milliseconds; the tracked read is the second cycle, off the lock the
first one took):

        level  cycle    scanned            tracked
          SL1  short   71.0 ->  72.6     4.8 ->  2.3
          SL2  short   74.2 ->  73.4    10.6 ->  5.0
          SL3  short   75.3 ->  76.1    21.8 ->  9.5
          SL4  short   78.7 ->  79.6    34.1 -> 15.9
          SL5  short   82.1 ->  83.3    40.2 -> 13.5
          SL6  short   85.0 ->  85.2    50.5 -> 16.3
          SL2   long  210.6 -> 210.0    30.0 -> 10.0
          SL3   long  218.9 -> 217.3    63.9 -> 18.0
          SL4   long  231.0 -> 233.8   102.7 -> 30.2
          SL5   long  250.3 -> 250.8   148.4 -> 36.0
          SL6   long  266.2 -> 267.0   190.6 -> 45.3

Two things moved it and each is written down where it lives.
`rxfront.SyncedRx._level_at_lock` walks its nine alignments nearest the lock
first rather than in time order, so the alignment a held grid actually put row 0
on stops being the fifth Viterbi run of five -- 50.5 ms to 23.4 at speed level 6
short, 190.6 to 67.5 long, on its own. `rx.carrier` tabulates the mixing
exponential over the one period the 120 Hz tone grid gives it, which takes the
rest. Neither changes what is admitted: the same alignments, the same CRC.

The blind read does not fit and is not meant to: it is what a cycle with no lock
pays, and `deep_scan` moves it to the top of the cycle for exactly that reason.
It is also untouched, which is a decision rather than an oversight --
`rx.carrier`'s docstring is where it is argued, and the short of it is that the
codeword sweep in front of it reads words off audio that is half silence, so
4e-12 of arithmetic is enough to change which one it meets first.

Run:  python -m hfmodem.tests.shrike.test_rxcost
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from hfmodem.shrike import live, onair, p3rx, placement, rxfront, spec
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.shrike.test_p3level_session import _Session, _payload

FS = rxfront.FS
FIXTURES = corpora.REGRESS_FIXTURES
# The recording the step was measured on -- a Winlink dial via a remote receiver,
# carrying no PACTOR-3 that any decoder in this project or outside it has read.
COST_WAV = FIXTURES / "watch_pactor3_maryland.wav"
# ...and the one an independent decoder does read end to end. Its first packet is
# a speed level 5 field at t=6.91, which is well past any window the cost test
# uses, so the two do not share a signal.
CORROBORATED_WAV = FIXTURES / "oracle_pactor3_dl6maa.wav"
CORROBORATED_AT = 6.0

# The lengths a session actually feeds, from the shortest `RollingRx` will decode
# to a window that has overrun its cycle. 0.85 is where the step used to be.
LENGTHS = (0.50, 0.75, 0.85, 1.25, 2.00)
# How much dearer a slide may get per second of audio between adjacent lengths.
# With the body scan wired back in, the 0.75 -> 0.85 boundary measures 20.4; the
# header-anchored pass that carries the PACTOR-3 decode on its own measures 2.5
# to 3.2 there, and everything else in the front end scales with the audio. Twice
# the measurement and a third of the defect.
MAX_STEP = 6.0

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _rolling_cost(audio) -> float:
    """Seconds one session-path decode of `audio` takes, wiring included.

    Through `RollingRx` rather than through `rxfront.decode_events` directly: what
    the session runs is the rolling decoder, and a switch set in the wrong place
    is exactly the kind of regression this is here to catch. A window as long as
    the audio with nothing kept slides once and decodes once.

    Best of two, because the assertions are ratios between measurements and a
    scheduler hiccup on the cheap end of one reads as a step on the dear end.
    """
    def once() -> float:
        rx = live.RollingRx(lambda ev: None, window_s=len(audio) / FS, keep_s=0.0)
        t = time.perf_counter()
        rx.push(audio)
        return time.perf_counter() - t

    return min(once(), once())


def main() -> int:
    if not COST_WAV.exists() or not CORROBORATED_WAV.exists():
        print(f"  [SKIP] corpus fixtures absent at {FIXTURES}")
        return 2

    audio = rxfront.load_wav(str(COST_WAV))
    print(f"\n{COST_WAV.name}: one session-path decode against the fed length")
    per_s = []
    for length in LENGTHS:
        cost = _rolling_cost(audio[:int(length * FS)])
        per_s.append(cost / length)
        print(f"    {length:.2f} s -> {cost * 1e3:8.1f} ms   "
              f"{cost / length:.3f}x real time")

    check("no decode costs more wall clock than the audio it covers",
          max(per_s) < 1.0, f"worst {max(per_s):.3f}x real time")
    steps = [(b / a, LENGTHS[i + 1]) for i, (a, b) in
             enumerate(zip(per_s, per_s[1:]))]
    worst, at = max(steps)
    check("the cost per second of audio does not step with the window",
          worst <= MAX_STEP, f"{worst:.1f}x at {at:.2f} s, limit {MAX_STEP:.0f}x")

    # The other half: what came off the stream is still done by the callers that
    # read files, and `Scan.trials` says so without a clock in the assertion.
    window = audio[:int(1.25 * FS)]
    kept = p3rx.decode_p3_packets(window).trials
    dropped = p3rx.decode_p3_packets(window, envelope=False).trials
    check("a file-reading caller still runs the envelope-anchored search",
          kept > dropped, f"{kept} trials against {dropped}")

    corroborated = rxfront.load_wav(str(CORROBORATED_WAV))
    seg = corroborated[int(CORROBORATED_AT * FS):int((CORROBORATED_AT + 2.0) * FS)]
    got = [ev for ev in rxfront.decode_events(seg, p3_envelope=False)
           if ev.kind == "packet" and ev.protocol == "PACTOR-3"]
    check("the stream receiver still reads the corroborated PACTOR-3 packet",
          bool(got), got[0].text[:44] if got else "nothing decoded")

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# The cycle's own read, through the entry point a live link calls
# --------------------------------------------------------------------------- #

SESSION_SNR_DB = 20.0
PRE_KEY_S = onair.PREKEY_RESERVE_S
"""What `deep_scan` has in front of the key. Not a figure this file chose: the
hold loop closes the receive window that much before the PTT instant."""


def _cycle_window(sl: int, long_cycle: bool, seed: int = 0) -> np.ndarray:
    """One cycle's receive window with the peer's packet in it, at that level.

    The window is the CYCLE, which is the length the hold loop feeds and the
    thing a long-cycle packet has to fit inside -- 3.29 s of frame in 3.75 s of
    audio, where `onair.FLUSH_CONTEXT_S` holds 0.75 and no second reader can
    cover for a miss.
    """
    body = placement.link_packet(sl, _payload(sl, long_cycle), 0x21,
                                 long_cycle=long_cycle)
    cycle = spec.CYCLE_LONG_S if long_cycle else spec.CYCLE_SHORT_S
    win = np.zeros(round(cycle * FS))
    lead = round(0.2 * FS)
    win[lead:lead + body.size] = np.asarray(body, np.float64)[:win.size - lead]
    sigma = float(np.sqrt(np.mean(np.asarray(body, np.float64) ** 2))) \
        / 10 ** (SESSION_SNR_DB / 20)
    return (win + np.random.default_rng(seed).normal(0, sigma, win.size)
            ).astype(np.float32)


def _deep_scan_cost(sl: int, long_cycle: bool) -> tuple[float, float, bool]:
    """(blind, tracked, delivered) for one level through `_SessionRx.deep_scan`.

    Two windows: the first is scanned and leaves a lock, the second comes off
    it. Best of three apiece, because the assertion is a budget and a scheduler
    hiccup is not the receiver.
    """
    first, second = _cycle_window(sl, long_cycle), _cycle_window(sl, long_cycle,
                                                                 seed=1)

    def once(sess, audio) -> float:
        sess.rx.new_cycle()
        t = time.perf_counter()
        sess.rx.deep_scan(audio)
        return time.perf_counter() - t

    blind = min(once(_Session(), first) for _ in range(3))
    sess = _Session()
    once(sess, first)
    tracked = min(once(sess, second) for _ in range(3))
    return blind, tracked, len(sess.events) >= 2


def test_the_cycles_own_read_fits_it() -> None:
    """Every level the peer may send, at both cycle lengths, on this box.

    Synthetic, and deliberately: what is measured is the receiver's arithmetic
    against the clock, and the corpus holds no recording of a peer at every
    level in a window the hold loop's own length. The stream measurements above
    are what real audio is for.
    """
    rows = []
    print(f"\n{'level':>7} {'cycle':>6} {'scanned':>10} {'tracked':>10}")
    for long_cycle in (False, True):
        for sl in sorted(placement.SPEED_PATHS):
            if long_cycle and sl == 1:
                # `placement.link_packet` refuses to build one: no long-cycle
                # speed level 1 packet is on tape to check a renderer against.
                continue
            blind, tracked, got = _deep_scan_cost(sl, long_cycle)
            rows.append((sl, long_cycle, blind, tracked, got))
            print(f"    SL{sl} {'long' if long_cycle else 'short':>6} "
                  f"{blind * 1e3:9.1f}ms {tracked * 1e3:9.1f}ms"
                  f"{'' if got else '   NOT DELIVERED'}")

    assert all(got for *_, got in rows), \
        [(sl, long) for sl, long, _, _, got in rows if not got]

    short = [(sl, t) for sl, long, _, t, _ in rows if not long]
    assert all(t < PRE_KEY_S for _, t in short), \
        f"{max(short, key=lambda r: r[1])} against {PRE_KEY_S:.3f} s"

    long = [(sl, t) for sl, long_, _, t, _ in rows if long_]
    assert all(t < spec.CYCLE_LONG_S for _, t in long), \
        f"{max(long, key=lambda r: r[1])} against {spec.CYCLE_LONG_S:.2f} s"

    # ...and the structural half, which no clock on any box can move: the read
    # that has to find the frame from cold is the dearer one, which is why the
    # cycle that comes up empty scans at the TOP of the next one and not in
    # front of its key (`onair._SessionRx.deep_scan`).
    assert all(b > t for _, _, b, t, _ in rows), \
        [(sl, long, round(b * 1e3, 1), round(t * 1e3, 1))
         for sl, long, b, t, _ in rows if b <= t]


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"corpus fixtures not present at {FIXTURES}")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
