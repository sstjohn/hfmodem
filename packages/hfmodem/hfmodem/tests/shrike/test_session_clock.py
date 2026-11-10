# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One column, one clock: when in the session a decoded codeword happened.

`onair._summary` prints every control signal a session decoded in one column,
`{ev.t:7.2f}`, and two readers fill it. `live.RollingRx` stamps its events on the
stream clock -- `t0`, which is the session's own sample count, plus the offset
into the buffer. The grid-anchored read (`_SessionRx._p1_cs` and `_p3_cs`) and
the acquisition search stamped theirs on the SEGMENT they were handed, which is a
turnaround: it never leaves the first fifth of a second, whatever minute of the
session it belongs to.

`working/force-pactor-kb5lzk-2.log`, one session, sixteen codewords, one column:

     0.08  CS4/100Bd        <- fourteen anchored reads, "where in the cycle"
      ...
     0.08  CS3/break-in
    38.16  CS2/ack          <- two rolling-decoder reads, "where in the session"
    39.37  CS2/ack

Nothing in the column said which was which, and the verdict line above it quotes
the same figure -- "the peer alternated to CS2/ack at 0.07 s" -- in a sentence an
operator reads as a moment in the session.

The stream clock wins, because everything else in the log is already on it: the
grid prints sample indices, every capture is indexed on them, and `stream.wav` is
the session on that one axis. `seg_start` is that index for the window a codeword
was read in, and the second scene below measures it against `RollingRx`'s own
clock rather than assuming the two agree.

Run: python -m hfmodem.tests.shrike.test_session_clock
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from hfmodem.shrike import onair, rxfront, spec
from hfmodem.tests.shrike import test_grid as grid
from hfmodem.tests.shrike import test_qrtack as qrt
from hfmodem.tests.shrike import test_silence as sil

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)

#: Far enough into a session that a segment offset and a session instant cannot
#: be mistaken for each other: forty cycles is fifty seconds, and the anchored
#: read's own figure is 0.094.
LATE = 40 * SLOT_N

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def the_anchored_read_says_when_in_the_session() -> None:
    """A codeword read at the grid's instant, fifty seconds in."""
    print("\nA codeword read at the anchor, 50 s into a session")
    host = qrt.calling_station()
    rx = onair._SessionRx(host)
    rx.new_cycle()
    at = LATE + qrt.D_N
    got = rx.control_signal(qrt.cs_audio(qrt.CS1), LATE, at)
    ev = rx.cs_log[-1] if rx.cs_log else None
    check("the codeword is read", got == qrt.CS1 and ev is not None, str(got))
    check("...and reported at the session instant it arrived at, not at its "
          "offset into the window it was read in",
          ev is not None and abs(ev.t - at / FS) < 0.01,
          f"reported {ev.t:.3f} s; the window opened at {LATE / FS:.3f} s and "
          f"the codeword is {qrt.D_N / FS:.3f} s into it")


def the_two_clocks_are_one_clock() -> None:
    """`seg_start` and `RollingRx.t0`, measured against each other every cycle.

    The conversion above is only worth anything if the sample index the loop
    carries and the clock the rolling decoder stamps events with are the same
    quantity. Nothing declares that they are; the session is run and asked.
    """
    print("\nThe loop's sample index against the rolling decoder's clock")
    rows: list[tuple[float, float]] = []
    seen: list = []
    real = onair._SessionRx.control_signal

    def spy(self, seg, seg_start, at, **kwargs):
        rows.append((seg_start / FS, self.rx.t0))
        seen.append(self)
        return real(self, seg, seg_start, at, **kwargs)

    onair._SessionRx.control_signal = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "peer.wav"
            sil._peer_wav(wav, answers=12, slots=16)
            sil._run(wav, Path(tmp) / "out", hold=8)
    finally:
        onair._SessionRx.control_signal = real

    worst = max((abs(a - b) for a, b in rows), default=None)
    check("the loop's window index and the rolling decoder's clock are the same "
          "quantity, every cycle",
          bool(rows) and worst is not None and worst < 1e-6,
          f"{len(rows)} cycles, worst disagreement {worst} s")
    # ...and the column they share climbs. A session-relative column steps a
    # cycle at a time; a segment-relative one sits still whatever the session
    # does, which is what fourteen readings of 0.07-0.18 in one log look like.
    ts = [ev.t for ev in seen[-1].cs_log] if seen else []
    print(f"    {len(rows)} windows, {rows[0][0]:.3f} s to {rows[-1][0]:.3f} s; "
          f"codewords at {' '.join(f'{t:.2f}' for t in ts)}")
    check("every codeword the session logged is stamped later than the one "
          "before it", len(ts) > 2 and all(b > a for a, b in zip(ts, ts[1:])),
          f"{len(ts)} codewords")
    check("...and the column spans the session rather than one turnaround",
          len(ts) > 2 and ts[-1] - ts[0] > 2 * spec.CYCLE_SHORT_S,
          f"span {ts[-1] - ts[0]:.2f} s over {len(ts)} codewords" if ts else "none")


def a_keyed_cycle_owes_its_whole_carrier_to_the_clock() -> None:
    """...and the replay above cannot see it, because a replay never keys.

    `RadioTx._tx` drops capture from the key instant to the last sample of the
    burst -- the PTT settle as well as the audio -- and the decoder's clock has
    to step over all of it. Credited the burst alone, the two clocks part by
    one settle a cycle, and `_SessionRx.cs_at` is a capture index built out of
    the reader's: a break-in read then collects its field from an instant
    already gone, and by cycle 25 there is nothing left of the window at all.
    """
    print("\nAn armed session, where every cycle drops a carrier")
    # `quiet_after` is what ENDS it, and it is not part of what is measured here:
    # the grid has acquired by then, so the station stays on the air and every
    # cycle still drops a carrier. Before 2026-08-28 the peer's answers were
    # missed often enough for `--hold`'s idle timeout to expire on its own; with
    # `p1rx.CS_SEARCH_HALF_S` every cycle is acknowledged -- and `grid._Host`
    # queues another line of its own every cycle, so 20 bytes really do leave
    # the buffer and reach the peer in each of them. `_HoldBudget` pushes its
    # deadline on that payload, which is the rule rather than a leak: an
    # acknowledged idle packet has an empty field and subtracts nothing
    # (`tests/shrike/test_holdbudget.py`). What this scene declines to wait out
    # is the bench station's own message, not a hold that cannot expire.
    got = grid._session(cycles=6, hold=14, charge=0, peer=True, quiet_after=10)
    tx, bench = got["tx"], got["bench"]
    rx = tx.sessrx
    clock = rx.rx.t0 + len(rx.rx.buf) / FS + rx._unheard
    drift = bench.pos / FS - clock
    check("the decoder's clock is the capture stream's, after a session of "
          "keyed cycles", abs(drift) < 1e-6,
          f"{len(tx.keyed)} bursts, {drift * 1e3:+.1f} ms apart")
    check("...and the session keyed enough of them for a per-burst leak to "
          "show", len(tx.keyed) >= 10, f"{len(tx.keyed)} bursts")


SCENES = (the_anchored_read_says_when_in_the_session, the_two_clocks_are_one_clock,
          a_keyed_cycle_owes_its_whole_carrier_to_the_clock)


def main() -> int:
    global ok
    ok = True
    print("One column, one clock")
    for scene in SCENES:
        scene()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
