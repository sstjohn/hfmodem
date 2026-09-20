# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A granted station keys EVERY slot of its entry campaign.

WS8EOC, 2026-09-13 14:42 CDT, 40 m (`working/pactor3-header-0913/
arm-v9-A-40-ws8eoc-b.launch.log`). The gateway granted PACTOR-3 and then held
`0x59A` for the rest of the arm, so this station keyed the same entry packet
every cycle -- and the grid gave away nineteen of the fifty slots it had to key
them in. `SLOT 9 IS GONE ... by +1.6 ms`, then 11 at +5.6, 13 at +1.6, 15 at
+5.6: every other slot, on an overrun of one to five milliseconds, against a
tolerance written for exactly that and dated five rounds earlier.

THE FORGIVENESS WAS NEVER ASKED. `KEY_CLAMP_TOL_S` is `RadioTx._tx`'s rule and
`_regrid` stands a whole tick and render in front of it, deciding the same
question strictly -- so the arm printed nineteen `SLOT ... IS GONE` and not one
`KEYED INTO ITS BOUNDARY`. `onair._clamp_forgives` is that gate asking the
emission path's question at the grid's own instant, with the cycle's measured
remaining work allowed for.

AND THE ALTERNATION IS THE DECISION'S OWN ECHO. A slot handed back takes the
cadence to two, and the admission guard runs on a delivered count quantised to
the callback block: 60000 samples to the slot is 96 past a 128-frame block, so a
one-slot cadence walks that residue through four values and a two-slot cadence
through exactly two (120000 mod 128 = 64). The worse of the two is past the gate
every time it comes round, which is why one lost slot costs every other slot for
the rest of the stint. The negative control below reproduces both: one slot in
four at 10.7 ms of pre-key work, every other slot at 13.3.

Run:  python -m pytest hfmodem/tests/shrike/test_entry_slots_kept.py
"""
from __future__ import annotations

import contextlib
import re
import tempfile
from pathlib import Path

import numpy as np

from hfmodem.shrike import onair, p3acquire, pactor1

from hfmodem.tests.shrike.test_grid import _Bench, _run, _session, check
from hfmodem.tests.shrike.test_p3_breakin_timing import LATE_N, breakin_cycle

# The arm's own line, less the device, rig and mail arguments the bench supplies.
ARM = ("--p1-grant-only", "--p1-status-bits45", "3", "--p1-setup-phase", "reply",
       "--p3-entry", "template", "--announce-lower", "--p3-entry-stagger",
       "--p3-entry-rise", "--over", "--no-long-cycle", "--no-p3-fallback",
       "--p3-control-waveform", "historical", "--p3-control-placement",
       "audio-start", "--retries", "20")

# Where the gateway starts commanding PACTOR-3, in our own carriers -- and it
# never stops: WS8EOC answered `0x59A` to every entry packet of the arm and read
# none of them, which is what leaves fifty cycles of one identical keying to
# count.
GRANT_AT = 6

# How many of those cycles this scene asks for.
STINT = 50

# THE TURNAROUND, and it is the arm's own: `d 93.5-93.7 ms` on every cycle of
# the granted phase.
D_S = 0.094

# THE PEER'S MEASURED CARRIER RASTER, which the entry has to be keyed on.
# WS8EOC's PACTOR-3 carriers ran between -75 and +64 Hz of ours across five arms
# of 2026-09-13, and +60 is inside that.
OFFSET_HZ = 60.0

# THE PRE-KEY WORK, CHARGED WHERE THE ARM PAYS IT: in front of the GRID's
# admission check, which is the gate that spent these slots. `test_granted_entry
# _slots` and `test_entry_counter_slots` charge theirs in front of the EMISSION
# path's, a tick and a render later, which is why neither could fail this way.
#
# 13.3 ms, and the arm's own figures put the interval at 7.0-16.8 ms. It is the
# value at which the loss is every-other-slot rather than one-in-four, which is
# the pattern the arm printed; 10.7 ms below reproduces the sawtooth instead.
PREKEY_N = 640
SAWTOOTH_N = 512

# ...and what the same cycle looks like past the tolerance, where the slot is
# still the right answer: 16 ms of work is three symbols late and no reader
# forgives it.
PAST_TOLERANCE_N = 768

# WHAT THAT PRE-KEY WORK ACTUALLY WAS, weighed on a later arm of the same
# gateway. `onair-0913-1629` gave away fifteen of twenty-eight slots on overruns
# of 4.2-10.2 ms, and a PACTOR-3 cycle spends two `p3acquire.compensate` calls in
# front of its key -- `_p3_cs`'s and `_read_p3_packet`'s. On that arm's 2.222 s
# hold windows, 106646 samples = 2 x 41 x 1301, the transform on the window's own
# length cost 4.8-7.1 ms apiece: 9.6-14.2 ms a cycle, which is `PREKEY_N` above
# and is the whole of the charge this file was written around. On a fast
# transform length it is 2.0 ms apiece, and the cycle below is what that leaves.
SHIPPED_PREKEY_N = 192


def _gateway(answers: list[int]):
    """WS8EOC: it answers, it grants, and then it holds the grant for ever.

    It never reads an entry packet, which is the arm: fifty keyings of
    `SL1 ENTRY 0B P3 status=0x1a field=0f8f87c7c31a6689`, byte-identical every
    cycle, each one answered `0x59A` at the same turnaround.
    """
    def make(shift):
        def put(bench: _Bench, rf_end: int) -> None:
            if bench.answered:
                return
            bench.answered = True
            n = len(bench.emissions)
            _, end = bench.emissions[-1]
            word = (pactor1.CS_SPEED if n == 1 else
                    pactor1.CS_59A if n >= GRANT_AT else
                    (pactor1.CS_ACK_A if n % 2 else pactor1.CS_ACK_B))
            burst = onair._trim_silence(np.asarray(
                pactor1.control_signal(word, invert=bool(shift())), np.float32))
            at = end + bench.d_n
            stop = min(at + burst.size, bench.audio.size)
            if stop > at:
                bench.audio[at:stop] += burst[:stop - at]
                answers.append(at)
        return put
    return make


@contextlib.contextmanager
def _acquire_cut(on: bool):
    """`_SessionRx.control_collect_until`, which this arm flew without.

    It closes the final collection `P3_ANSWER_ACQUIRE_RESERVE_S` early on
    exactly this cycle -- ISS, LINKED, PACTOR-3, sending, entry pending, with
    `RadioTx._tx.record_slot`'s `_p3_keyed_reply` naming the slot it keyed --
    and so hands an entry cycle 12 ms, six times `onair.TX_ADMIT_RESERVE_S`.
    Measured on this scene: it returns the earlier deadline on all 24 of the
    campaign's PACTOR-3 emissions and on none of the PACTOR-1 cycles in front
    of the grant.

    The arm predates it, so reproducing the arm means running without it. Off
    is the budget the campaign flew on; on is the budget it flies on now, and
    the scene that asks for the 12 ms names this.
    """
    was = onair._SessionRx.control_collect_until
    if not on:
        onair._SessionRx.control_collect_until = (
            lambda self, until, now, raster: until)
    try:
        yield
    finally:
        onair._SessionRx.control_collect_until = was


@contextlib.contextmanager
def _cycle(prekey_n: int, offset_hz: float, *, forgive: bool = True):
    """The cycle's own coordinates, planted at the grid's admission check.

    `_regrid` is where the arm's overrun was measured and where its slots went,
    so the charge goes in front of it -- and so does the receive offset, because
    the bench peer keys PACTOR-1 codewords and a PACTOR-1 codeword carries no
    PACTOR-3 raster to measure. Everything downstream is production: the entry
    is rendered, shifted, admitted and keyed through the real seams.
    """
    real_regrid, real_forgive = onair._regrid, onair._clamp_forgives

    def charged(live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n):
        sessrx.p3_receive_offset_hz = offset_hz
        live.spend(prekey_n)
        return real_regrid(live, raster, tx, host, sessrx, slot, seg, seg_start,
                           settle_n)

    onair._regrid = charged
    if not forgive:
        onair._clamp_forgives = lambda *a, **k: 0
    try:
        yield
    finally:
        onair._regrid, onair._clamp_forgives = real_regrid, real_forgive


def _arm(prekey_n: int, offset_hz: float = 0.0, *, forgive: bool = True,
         keep: str = "", acquire_cut: bool = False) -> dict:
    answers: list[int] = []
    shifts: list[int] = []
    bare = p3acquire.compensate

    def counted(x, hz):
        shifts.append(len(x))
        return bare(x, hz)

    argv = ARM + (("--p3-keep-slots", keep) if keep else ())
    p3acquire.compensate = counted
    try:
        with _acquire_cut(acquire_cut), _cycle(prekey_n, offset_hz,
                                               forgive=forgive):
            got = _session(cycles=6, hold=80, charge=0, decode=True, peer=False,
                           seconds=300.0, keep_upgrade=True, d=D_S,
                           extra_argv=argv, answer=_gateway(answers))
    finally:
        p3acquire.compensate = bare
    got["answers"], got["shifts"] = answers, shifts
    return got


def _entries(got: dict) -> list[int]:
    return [b["slot"] for b in got["bursts"]
            if "ENTRY" in b["what"] and not b["refused"]][:STINT]


def _lost(slots: list[int]) -> list[int]:
    return [s for s in range(slots[0], slots[-1] + 1) if s not in slots]


def _why(log: str) -> str:
    return "; ".join(ln.strip() for ln in log.splitlines()
                     if onair.SLOT_GONE in ln or onair.LATE_KEY in ln) or "clean"


def a_granted_station_keys_every_entry_slot() -> None:
    for hz in (0.0, OFFSET_HZ):
        print(f"\nA granted PACTOR-3 entry campaign at a peer {hz:+.0f} Hz off")
        got = _arm(PREKEY_N, hz)
        check("the gateway granted and the entry campaign ran",
              "0x59A grant" in got["log"]
              and "link upgraded to PACTOR-3" in got["log"])

        slots = _entries(got)
        check(f"the campaign keys {STINT} entry packets",
              len(slots) == STINT, f"{len(slots)} entries on slots {slots}")
        check("...and every slot in the run carries one -- none given away",
              not _lost(slots),
              f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}; "
              f"{_why(got['log'])}")
        check("...on one cadence, a slot at a time",
              set(b - a for a, b in zip(slots, slots[1:])) == {1},
              f"slots {slots}")

        # ...AND THE CYCLE REALLY WAS OVER ITS KEY INSTANT, or the scene asks
        # nothing: the grid has to have forgiven these, not merely met them.
        check("...with the grid keeping the slot rather than meeting it",
              got["log"].count(onair.SLOT_KEPT) >= STINT // 2,
              f"{got['log'].count(onair.SLOT_KEPT)} kept, "
              f"{got['log'].count(onair.KEYED_LATE)} keyed into the boundary")

        check(f"...every entry keyed at {hz:+.0f} Hz",
              got["log"].count(f"TX {hz:+g} Hz") >= STINT,
              f"{got['log'].count(f'TX {hz:+g} Hz')} of {STINT}")

        # THE COMPENSATED RENDER IS BUILT ONCE, not in front of every key. The
        # entry is byte-identical every cycle, so the transform that moves it
        # onto the peer's raster belongs to the packet and not to the slot.
        packets = [n for n in got["shifts"] if n > 40000]
        check("...and the shifted entry was built once for the whole campaign",
              len(packets) == (1 if hz else 0),
              f"{len(packets)} packet-length shifts for {len(slots)} keyings")


def the_grid_is_what_was_spending_those_slots() -> None:
    """NEGATIVE CONTROL. A guard test that cannot fail is not a guard test.

    `_clamp_forgives` reverted to the answer that flew: a cycle past its own key
    instant hands the slot back whatever the overrun is. The campaign then keys
    every other slot, which is `arm-v9-A-40-ws8eoc-b` -- and one slot in four at
    the lighter charge, which is the block sawtooth before a two-slot cadence
    folds it in half.
    """
    print("\nNEGATIVE CONTROL: the same cycle, forgiving nothing")
    got = _arm(PREKEY_N, OFFSET_HZ, forgive=False)
    slots = _entries(got)
    lost = _lost(slots)
    check("without the forgiveness the campaign gives slots away", bool(lost),
          f"slots {slots}")
    check("...and it is every other slot, which is the arm",
          lost == [s for s in range(slots[0], slots[-1] + 1) if s not in slots]
          and set(b - a for a, b in zip(slots, slots[1:])) == {2},
          f"lost {lost} of {slots[0]}-{slots[-1]}")

    print("\n...and the same cycle a block lighter, which is the sawtooth")
    slots = _entries(_arm(SAWTOOTH_N, OFFSET_HZ, forgive=False))
    check("a lighter cycle loses one slot in four", bool(_lost(slots)),
          f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}")
    got = _arm(SAWTOOTH_N, OFFSET_HZ)
    check("...and the forgiveness keys all of them",
          not _lost(_entries(got)), _why(got["log"]))


def the_shipped_cycle_never_reaches_the_forgiveness() -> None:
    """The point of weighing it: at this charge there is nothing to forgive.

    The scene at the top needs `_clamp_forgives` and says so -- it keeps slots
    the cycle has already overrun. Four milliseconds of pre-key work is not over
    the key instant at all, so the campaign keys every slot with that guard
    reverted, which is the state no arm of 2026-09-13 ran in.
    """
    print("\nThe same campaign at the pre-key cost a fast transform leaves")
    got = _arm(SHIPPED_PREKEY_N, OFFSET_HZ, forgive=False)
    slots = _entries(got)
    check(f"the campaign keys {STINT} entry packets", len(slots) == STINT,
          f"{len(slots)} entries on slots {slots}")
    check("...on one cadence, with nothing forgiven",
          not _lost(slots)
          and set(b - a for a, b in zip(slots, slots[1:])) == {1},
          f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}; {_why(got['log'])}")
    check("...and the grid gave no slot away",
          onair.SLOT_GONE not in got["log"],
          f"{got['log'].count(onair.SLOT_GONE)} gone")


def past_the_tolerance_the_slot_is_still_the_answer() -> None:
    """AND IT IS A FORGIVENESS, NOT A REMOVAL.

    Half a symbol at 100 Bd is what the reference decoder forgives. A cycle
    three symbols late is a cycle the peer cannot read us in, and the grid
    hands the slot back exactly as it did -- `SLOT 47 IS GONE ... by +29.6 ms`
    is the arm's own example of one that should have gone.
    """
    print("\nAn overrun past the tolerance still spends its slot")
    got = _arm(PAST_TOLERANCE_N, OFFSET_HZ)
    check("the grid still gives a slot away past the tolerance",
          got["log"].count(onair.SLOT_GONE) >= 1,
          f"{got['log'].count(onair.SLOT_GONE)} gone, "
          f"{got['log'].count(onair.SLOT_KEPT)} kept")
    check("...and the campaign still keys its entries",
          len(_entries(got)) == STINT, f"{len(_entries(got))} entries")


def the_flag_ships_on_all_and_that_is_this_campaign() -> None:
    """`--p3-keep-slots all`: the gate as rounds 14 and 16 left it.

    The switch exists because the air disagrees with itself -- the runtime that
    keys nearly every slot is the one WS8EOC stops acknowledging at SL1 seq=1,
    and the one that gave slots away delivered the whole greeting -- so the
    first thing it has to be is free. Named or not named, the campaign is the
    same fifty keyings.
    """
    print("\n--p3-keep-slots all, which is what ships")
    got = _arm(PREKEY_N, OFFSET_HZ, keep="all")
    check("the session banner names the policy",
          "keep-slots=all" in got["log"], "banner did not carry it")

    slots = _entries(got)
    check(f"the campaign keys {STINT} entry packets", len(slots) == STINT,
          f"{len(slots)} entries on slots {slots}")
    check("...and every slot in the run carries one -- none given away",
          not _lost(slots),
          f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}; {_why(got['log'])}")
    check("...with the grid keeping the slot rather than meeting it",
          got["log"].count(onair.SLOT_KEPT) >= STINT // 2,
          f"{got['log'].count(onair.SLOT_KEPT)} kept")

    print("...and the same campaign with the flag left off the line")
    bare = _arm(PREKEY_N, OFFSET_HZ)
    check("the default is that policy, slot for slot",
          _entries(bare) == slots and "keep-slots=all" in bare["log"],
          f"{_entries(bare)} against {slots}")


def controls_spends_the_changeover_and_keeps_the_rest() -> None:
    """`--p3-keep-slots controls`: round 16's exclusion, back on its own.

    v10 gave the changeover cycle away and delivered the greeting; v12 keeps it
    and stalls. This is that one difference, switchable and nothing else with
    it -- an entry campaign is not a changeover cycle and loses no slot here.
    """
    print("\n--p3-keep-slots controls: the entry campaign is untouched")
    got = _arm(PREKEY_N, OFFSET_HZ, keep="controls")
    slots = _entries(got)
    check("the session banner names the policy",
          "keep-slots=controls" in got["log"], "banner did not carry it")
    check(f"the campaign still keys {STINT} entry packets", len(slots) == STINT,
          f"{len(slots)} entries on slots {slots}")
    check("...and still gives no slot away -- no cycle here is a changeover",
          not _lost(slots) and set(b - a for a, b in zip(slots, slots[1:])) == {1},
          f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}; {_why(got['log'])}")

    print("...and the changeover cycle of 0913-1550 hands its slot back")
    with tempfile.TemporaryDirectory() as tmp:
        _, tx, _, late = breakin_cycle(Path(tmp), LATE_N[0])
        tx.prekey_cost_n = tx.breakin_cost_n = 0
        check("the same overrun is kept under \"all\"",
              onair._clamp_forgives(tx.live, tx, late) == late,
              f"{late} samples late, and the gate returned "
              f"{onair._clamp_forgives(tx.live, tx, late)}")
        tx.p3_keep_slots = "controls"
        check("...and given away under \"controls\", which is v10",
              onair._clamp_forgives(tx.live, tx, late) == 0,
              f"the gate kept {onair._clamp_forgives(tx.live, tx, late)}")
        tx.breakin_due = False
        check("...while an ordinary control cycle of the same run is kept",
              onair._clamp_forgives(tx.live, tx, late) == late,
              f"the gate kept {onair._clamp_forgives(tx.live, tx, late)}")


def none_is_the_gate_off_and_the_arm_comes_straight_back() -> None:
    """`--p3-keep-slots none`: v9 and every arm before it.

    `arm-v9-A-40-ws8eoc-b` keyed twenty-five of its fifty granted cycles, on
    every other slot, and this is the flag reaching the same place round 14's
    negative control reaches by reverting the function.
    """
    print("\n--p3-keep-slots none: the forgiveness off")
    got = _arm(PREKEY_N, OFFSET_HZ, keep="none")
    slots = _entries(got)
    check("the session banner names the policy",
          "keep-slots=none" in got["log"], "banner did not carry it")
    check("the campaign gives slots away", bool(_lost(slots)), f"slots {slots}")
    check("...and it is every other slot, which is the arm",
          set(b - a for a, b in zip(slots, slots[1:])) == {2},
          f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}")
    check("...with the grid never keeping one",
          onair.SLOT_KEPT not in got["log"],
          f"{got['log'].count(onair.SLOT_KEPT)} kept")
    check("...and the flag lands where reverting the function lands",
          _entries(_arm(PREKEY_N, OFFSET_HZ, forgive=False)) == slots,
          f"{_entries(_arm(PREKEY_N, OFFSET_HZ, forgive=False))} against "
          f"{slots}")


def the_acquire_cut_keeps_them_without_reaching_the_forgiveness() -> None:
    """THE SHIPPED BUDGET, at the arm's own charge.

    Every scene above runs the budget the arm flew on. On the budget the
    campaign flies now, `_SessionRx.control_collect_until` closes the entry
    cycle's final collection 12 ms early -- nine times the 1.3 ms the cycle was
    over by -- so the same 13.3 ms of pre-key work never reaches the key instant
    and there is nothing left for `_clamp_forgives` to forgive. The forgiveness
    reverted, which is `the_grid_is_what_was_spending_those_slots`, this keeps
    every slot; take the cut away at the same charge and that scene loses every
    other one.
    """
    print("\nThe same campaign on the shipped budget, with the acquire cut in hand")
    got = _arm(PREKEY_N, OFFSET_HZ, forgive=False, acquire_cut=True)
    slots = _entries(got)
    check(f"the campaign keys {STINT} entry packets", len(slots) == STINT,
          f"{len(slots)} entries on slots {slots}")
    check("...on one cadence, with the forgiveness reverted and nothing to "
          "forgive", not _lost(slots)
          and set(b - a for a, b in zip(slots, slots[1:])) == {1},
          f"lost {_lost(slots)} of {slots[0]}-{slots[-1]}; {_why(got['log'])}")
    # ...AND THE CUT REACHES THE ENTRY CYCLE AND NOTHING ELSE, which is the whole
    # of why the scenes above still have an arm to reproduce: the PACTOR-1 cycles
    # in front of the grant carry the same 13.3 ms, meet no `control_collect_until`
    # -- it wants PACTOR-3, ISS and an entry pending -- and hand their slots back
    # exactly as the arm did.
    gone = [int(n) for n in re.findall(r"SLOT (\d+) " + onair.SLOT_GONE,
                                       got["log"])]
    check("...and every slot it did give away is a PACTOR-1 cycle in front of "
          "the grant, which the cut does not reach",
          bool(gone) and all(s < slots[0] for s in gone),
          f"gave away {gone}, entry run {slots[0]}-{slots[-1]}")


def test_a_granted_station_keys_every_entry_slot() -> None:
    assert _run(a_granted_station_keys_every_entry_slot)


def test_the_acquire_cut_keeps_them_without_reaching_the_forgiveness() -> None:
    assert _run(the_acquire_cut_keeps_them_without_reaching_the_forgiveness)


def test_the_grid_is_what_was_spending_those_slots() -> None:
    assert _run(the_grid_is_what_was_spending_those_slots)


def test_past_the_tolerance_the_slot_is_still_the_answer() -> None:
    assert _run(past_the_tolerance_the_slot_is_still_the_answer)


def test_the_shipped_cycle_never_reaches_the_forgiveness() -> None:
    assert _run(the_shipped_cycle_never_reaches_the_forgiveness)


def test_the_flag_ships_on_all_and_that_is_this_campaign() -> None:
    assert _run(the_flag_ships_on_all_and_that_is_this_campaign)


def test_controls_spends_the_changeover_and_keeps_the_rest() -> None:
    assert _run(controls_spends_the_changeover_and_keeps_the_rest)


def test_none_is_the_gate_off_and_the_arm_comes_straight_back() -> None:
    assert _run(none_is_the_gate_off_and_the_arm_comes_straight_back)
