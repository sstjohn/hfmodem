# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What `off` can and cannot settle, benched against the record that produced it.

`off` was put forward as the discriminator: the gateway acts on an
acknowledgement whose `off` sits inside the half-bit bracket and ignores one
whose does not, and the small fraction of keyed codewords that ever move the
peer's counter is the fraction that happened to land inside it. Two readings off
the record refuse that, and both are here so the question is not reopened from
impressions.

THE BRACKET IS NEVER LEFT. Across every session log on this station's disk, 149
`[ack]` lines carry an `off` and the largest is +2.1 ms -- all of them inside
`p1rx.CS_ANCHOR_S`, with no cycle outside it to correlate anything against. That
is structural rather than lucky: `_ack_gap_line` measures the gap and the aim off
ONE tracked onset, so their difference is the transmitter's own RF-start error
against the boundary it aimed at, and no placement error can reach it. A quantity
with one value across the whole corpus cannot separate the cycles the peer acted
on from the cycles it did not.

AND A TIGHT PLACEMENT IS NOT ENOUGH. `captures/onair-0809-2202` is the strongest
negative on file: WS8EOC held the channel for 96 receive windows and this station
answered every one. Its transmit raster is exact -- 84 of its 96 window starts
sit on one phase of the 1.25 s slot, spread 82 samples, no wider than the
128-frame block the codec hands over -- so the keying error there was a third of
a bit, everywhere. Under that placement the gateway repeated packet #1 for fifty-one
consecutive cycles and then advanced to #2, once, in a cycle whose geometry is
indistinguishable from the fifty-one.

So the counter is gated on something else, and this file exists to keep the next
reading of it from starting over at `off`.

Run:  pytest hfmodem/tests/shrike/test_ackoff.py
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from hfmodem.shrike import p1rx
from hfmodem.tests import evidence

FS = 48000
SLOT_N = round(1.25 * FS)

#: Half a bit at 100 Bd: the whole of a blind reader's error budget.
HALF_BIT_MS = p1rx.CS_ANCHOR_S * 1e3

#: The 2026-08-09 hold against WS8EOC on 7101500, 96 receive windows on one
#: capture-stream timebase and no session log left beside them.
CAPTURE = evidence.CAPTURES / "onair-0809-2202"

_OFF = re.compile(r"\[ack\] .*\| off\s+([-+][\d.]+) ms")

#: What the whole record held when this was measured. A later session may add
#: acks; none of them may add one outside the bracket without this failing.
LOGGED_ACKS = 149


def _offs() -> list[float]:
    # By content, because the 2026-08-15 slot was archived twice and a log
    # counted twice would let one session stand in for the corpus.
    seen: set[bytes] = set()
    out: list[float] = []
    for log in sorted(evidence.WORKING.rglob("*.log")):
        body = log.read_bytes()
        digest = hashlib.sha1(body).digest()
        if digest in seen:
            continue
        seen.add(digest)
        out += [float(m.group(1))
                for m in _OFF.finditer(body.decode("utf-8", "replace"))]
    return out


requires_working = pytest.mark.skipif(
    not evidence.WORKING.is_dir(),
    reason=f"{evidence.WORKING} is not on this machine")

requires_capture = pytest.mark.skipif(
    not CAPTURE.is_dir(),
    reason=f"{CAPTURE} is not on this machine -- on-air captures are gitignored "
           "and live only in the checkout that recorded them")


@requires_working
def test_no_acknowledgement_in_the_record_ever_left_the_half_bit_bracket():
    offs = _offs()
    assert len(offs) >= LOGGED_ACKS, f"{len(offs)} ack lines, expected {LOGGED_ACKS}+"
    worst = max(abs(v) for v in offs)
    assert worst < HALF_BIT_MS, f"|off| reached {worst:.1f} ms"
    # Not merely inside it -- half of the bracket over, so the margin is the
    # transmitter's and not the reader's tolerance being spent.
    assert worst <= HALF_BIT_MS / 2


@requires_capture
def test_the_session_that_would_not_move_the_counter_kept_its_raster_to_a_bit():
    """The keying error `off` cannot report on, read off the capture stream.

    A window starts at our own data end, so where a window start sits on the
    1.25 s raster IS where this station keyed -- the one quantity a session with
    no log beside it still carries.
    """
    starts = np.array(
        [(lambda j: j["end_stream_sample"] - j["samples"])(
            json.loads(p.read_text()))
         for p in sorted(CAPTURE.glob("hold_*.json"))])
    phase = (starts - starts[0]) % SLOT_N
    phase = np.where(phase > SLOT_N // 2, phase - SLOT_N, phase)
    # Two phases, because the session re-aimed twice; inside the hold proper the
    # raster does not move at all.
    hold = phase[np.abs(phase - np.median(phase)) <= 128]
    assert hold.size >= 80, f"{hold.size} windows on one phase"
    assert hold.max() - hold.min() <= 128, "wider than the codec's block"
    assert (hold.max() - hold.min()) / FS * 1e3 < HALF_BIT_MS / 2


@requires_capture
def test_the_counter_moved_once_in_that_session_and_not_on_a_better_placement():
    from hfmodem.core import wav
    from hfmodem.shrike import rxfront

    seen: list[tuple[int, int, float]] = []          # (window, counter, d ms)
    for p in sorted(CAPTURE.glob("hold_*.wav"))[45:75]:
        seg = np.asarray(wav.read(p), dtype=np.float64)
        got = list(p1rx.decode_p1_packets(seg))
        ons = rxfront.p1_burst_onsets(seg)
        if got and ons:
            seen.append((int(p.stem.split("_")[1]), got[0].status & 3,
                         ons[0] * 1e3))
    counters = [c for _, c, _ in seen]
    assert set(counters) == {1, 2}, counters
    # One step, and no second one: fifty-one cycles of #1 in front of it.
    steps = [i for i, (a, b) in enumerate(zip(counters, counters[1:])) if a != b]
    assert len(steps) == 1, steps
    at = steps[0]
    before = [d for _, c, d in seen[:at + 1]]
    after = [d for _, c, d in seen[at + 1:]]
    # The turnaround the accepted cycle was answered at is inside the spread of
    # the fifty-one that were not, so nothing about the placement marks it.
    assert min(before) <= before[-1] <= max(before)
    assert abs(np.median(after) - np.median(before)) < 3 * np.std(before, ddof=1)
