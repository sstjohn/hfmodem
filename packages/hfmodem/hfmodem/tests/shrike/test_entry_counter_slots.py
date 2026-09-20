# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A PACTOR-3 ISS keys every cycle its peer answers.

VE3KPG, 2026-09-13: the entry was read, the peer answered 36 consecutive cycles
at zero bit errors, and this station keyed 16 of the 32 cycles between the first
answer and the last. Every lost cycle is a `LATE TO THE KEY` or a `SLOT IS GONE`
-- the burst arriving at the emission path a few milliseconds past the instant
its own boundary could still take, and then stepping TWO slots to keep the
polarity its samples were rendered in.

THE GEOMETRY IS WHAT MAKES IT TIGHT, and it is the protocol's rather than this
station's. A PACTOR-3 control signal is twenty symbols at 100 Bd -- 200 ms,
where PACTOR-1's is 120 -- so a cycle that keys 0.87 s of packet and reads an
answer starting a turnaround behind it has the answer ending about 1.18 s into
the 1.25 s cycle. The key instant is a settle in front of the next boundary and
the converter needs its DAC notice in front of THAT, which leaves single-digit
milliseconds between the peer's last symbol and the moment our own audio must be
enqueued. Nothing in the cycle can be trimmed to buy a slot there.

So the burst keys where it can: `KEY_CLAMP_TOL_S` is what the reader forgives,
and a carrier that comes up two milliseconds into its boundary is a carrier the
peer reads. Giving the cycle away instead is what cost half of them.

Run:  python -m pytest hfmodem/tests/shrike/test_entry_counter_slots.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import arq, onair, pactor1, placement

from hfmodem.tests.shrike.test_grid import (
    FS, _Bench, _run, _session, check)

# The arm's own line, less the device and rig arguments the bench supplies.
ARM = ("--p1-grant-only", "--p1-status-bits45", "3", "--p3-entry", "template",
       "--announce-lower", "--p3-entry-stagger", "--p3-entry-rise",
       "--no-long-cycle", "--no-p3-fallback", "--p3-traffic-sl", "3",
       "--retries", "20")

# Where the gateway starts commanding PACTOR-3, in our own carriers.
GRANT_AT = 6

# ...and how many entry packets it takes before it answers one. VE3KPG read the
# fifth on 2026-09-13 and the second on the 09:55 arm; two keeps the scene short
# and the campaign is not what this file measures.
ENTRY_READS_AT = 2

# ...and how many cycles it then answers before going silent. VE3KPG answered 36
# in a row and stopped; the scene asks for 32 keyed cycles inside that.
STINT = 36

# ...and how many of those cycles this scene asks for. One short of the peer's
# stint, so the cycle its silence lands in is not counted against the keying.
STINT_CYCLES = 32

# THE TURNAROUND, and it is the arm's: VE3KPG's PACTOR-3 answers sat 951 ms past
# our slot boundary against a packet ending at 873, so it answered 78 ms behind
# our carrier. `_Bench.D_S` is 105 -- a slower peer, and one this scene would
# then be measuring instead of the geometry.
D_S = 0.078

# The pre-key work, charged where the arm pays it: between the cycle's last read
# and the admission check. The arm's own printed figures put that interval at
# 7.0-16.8 ms over 29 keyings, median 9.1 (`40 - lead`); 10.7 ms is inside that
# band and is where the overrun this scene is about becomes deterministic rather
# than one slot in four -- `test_granted_entry_slots` measures the same interval
# a block lower, at the threshold itself.
PREKEY_N = 4 * 128

# A PACTOR-3 codeword is 200 ms of symbols, and that is the whole reason this
# scene is tight. Rendered through the production renderer so the length and the
# shape are the ones a peer actually keys.
P3_CS = onair._trim_silence(
    np.asarray(placement.historical_control_signal(arq.CS_ACK), np.float32))


def _gateway(*, grant_at: int, entry_reads_at: int, stint: int,
             answers: list[int]):
    """VE3KPG: grants, reads an entry packet, then answers every cycle.

    THE ALTERNATION IS THE SPEC'S. Once it has read the entry it answers CS1,
    and every distinct packet after that draws the other codeword -- which is
    the counter law read from this side: the entry carries counter 2 and CS1
    answers an even counter, the packet behind it carries 3 and CS2 answers an
    odd one (pactor3.md §14). A cycle we key nothing in draws nothing, so the
    two ends stay in step through a lost slot.
    """
    def make(shift):
        state = {"entries": 0, "p3": False, "cs": arq.CS_ACK, "answered": 0}

        def put(bench: _Bench, rf_end: int) -> None:
            if bench.answered:
                return
            bench.answered = True
            n = len(bench.emissions)
            at, end = bench.emissions[-1]
            keyed = end - at
            if state["p3"]:
                # Our data packets only. A codeword of our own is not a packet
                # to answer, and this peer holds the receiving role throughout.
                if keyed < FS // 2:
                    return
                state["answered"] += 1
                if state["answered"] > stint:
                    return          # ...and then it goes silent, as VE3KPG did
                burst = P3_CS if state["cs"] == arq.CS_ACK else onair._trim_silence(
                    np.asarray(placement.historical_control_signal(arq.CS_REQUEST),
                               np.float32))
                state["cs"] = (arq.CS_REQUEST if state["cs"] == arq.CS_ACK
                               else arq.CS_ACK)
            else:
                if 0.80 * FS < keyed < 0.86 * FS:
                    state["entries"] += 1
                    if state["entries"] >= entry_reads_at:
                        state["p3"] = True
                word = (pactor1.CS_SPEED if n == 1 else
                        pactor1.CS_59A if n >= grant_at else
                        (pactor1.CS_ACK_A if n % 2 else pactor1.CS_ACK_B))
                burst = onair._trim_silence(np.asarray(
                    pactor1.control_signal(word, invert=bool(shift())),
                    np.float32))
            put_at = end + bench.d_n
            stop = min(put_at + burst.size, bench.audio.size)
            if stop > put_at:
                bench.audio[put_at:stop] += burst[:stop - put_at]
                answers.append(put_at)
        return put
    return make


def _arm(tolerance: float, prekey_n: int = PREKEY_N) -> dict:
    answers: list[int] = []
    was = onair.KEY_CLAMP_TOL_S
    onair.KEY_CLAMP_TOL_S = tolerance
    try:
        got = _session(cycles=6, hold=48, charge=0, decode=True, peer=False,
                       seconds=300.0, keep_upgrade=True, prekey_n=prekey_n,
                       d=D_S, extra_argv=ARM,
                       answer=_gateway(grant_at=GRANT_AT,
                                       entry_reads_at=ENTRY_READS_AT,
                                       stint=STINT, answers=answers))
    finally:
        onair.KEY_CLAMP_TOL_S = was
    got["answers"] = answers
    return got


def _packets(got: dict) -> list[dict]:
    """The PACTOR-3 data packets that reached the air, in the order flown."""
    return [b for b in got["bursts"]
            if b["what"].startswith("SL") and "pkt" in b["what"]
            and not b["refused"]]


def _slots(bursts: list[dict]) -> list[int]:
    return [b["slot"] for b in bursts]


def _gaps(slots: list[int]) -> list[int]:
    return [b - a for a, b in zip(slots, slots[1:])]


def _why(log: str) -> str:
    return "; ".join(ln.strip() for ln in log.splitlines()
                     if onair.LATE_KEY in ln or onair.SLOT_GONE in ln) or "clean"


def an_answered_iss_keys_every_cycle() -> None:
    print("\nA PACTOR-3 ISS keys every cycle a peer is answering it in")
    got = _arm(onair.KEY_CLAMP_TOL_S)
    check("the gateway's grant was read and an entry packet answered",
          "0x59A grant" in got["log"]
          and "the peer answered the entry packet" in got["log"])

    slots = _slots(_packets(got))[:STINT_CYCLES]
    check(f"the stint keys {STINT_CYCLES} data packets",
          len(slots) == STINT_CYCLES,
          f"{len(slots)} data packets on slots {slots}")

    lost = [s for s in range(slots[0], slots[-1] + 1) if s not in slots]
    check("...and every slot in it carries one -- no cycle given away",
          not lost, f"lost {lost} of {slots[0]}-{slots[-1]}; {_why(got['log'])}")

    check("...on one cadence, a slot at a time",
          set(_gaps(slots)) == {1}, f"gaps {sorted(set(_gaps(slots)))}")

    # ...AND THE CYCLE REALLY WAS OVER ITS OWN KEY INSTANT, or the scene is
    # asking nothing: the forgiveness has to have fired for these slots to be
    # the thing under test rather than a cycle that was never late.
    check("...with the overrun forgiven rather than absent",
          "the reader forgives" in got["log"], _why(got["log"]))

    # AND THE LINK IS RUNNING, not merely transmitting: a peer answering on the
    # counter's own phase advances us, so the counters step by one every cycle.
    entries = len([b for b in got["bursts"] if "ENTRY" in b["what"]])
    counters = got["tx"].seq_sent[entries + 1:][:STINT_CYCLES]
    check("...with the packet counter advancing every cycle",
          len(counters) == STINT_CYCLES
          and all((b - a) % arq.SEQ_MOD == 1
                  for a, b in zip(counters, counters[1:])),
          f"counters {counters}")


def the_tolerance_is_what_buys_those_slots() -> None:
    """NEGATIVE CONTROL. A guard test that cannot fail is not a guard test.

    Zero tolerance is the arithmetic that flew: a burst the converter cannot
    schedule on its boundary gives the cycle up and steps to the next boundary
    its own polarity fits, which is the one after next. The stint then keys two
    slots in every four.
    """
    print("\nNEGATIVE CONTROL: the same scene, forgiving nothing")
    got = _arm(0.0)
    slots = _slots(_packets(got))[:STINT_CYCLES]
    lost = ([s for s in range(slots[0], slots[-1] + 1) if s not in slots]
            if slots else [])
    check("without the tolerance the stint loses cycles", bool(lost),
          f"packets on {slots}, gaps {sorted(set(_gaps(slots)))}, lost {lost}")
    check("...and the cadence breaks, which is the burst stepping past the "
          "boundary its own polarity does not fit",
          max(_gaps(slots)) > 1, f"gaps {sorted(set(_gaps(slots)))}")


def test_an_answered_iss_keys_every_cycle() -> None:
    assert _run(an_answered_iss_keys_every_cycle)


def test_the_tolerance_is_what_buys_those_slots() -> None:
    assert _run(the_tolerance_is_what_buys_those_slots)
