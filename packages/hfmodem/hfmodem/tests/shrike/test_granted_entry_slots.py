# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A grant commands the next transmit slot, and every one after it.

WS8EOC granted PACTOR-3 on 2026-09-11 and held `0x59A` for thirteen consecutive
cycles while this station put four entry packets on slots 40, 42, 43 and 46 --
gaps of two, one and three, never two in a row, the first of them 3.75 s after
the grant commanded it. The transcript is
`working/onair-0911-1931/evening-E01-ws8eoc-p3-short.log`; the reconstruction is
`working/pactor-evening-en63bc-0911/investigation/`.

Nothing in the suite could fail that way, and the reason was one line:
`test_grid._Bench.samples` returned the bench clock, so the converter and the
block-quantised delivered count -- the two numbers the admission guard is the
difference of -- were one number. Separated, the room a cycle has between its
last read and the guard is a sawtooth of one callback block on a raster of
468.75 blocks to the slot, and a cycle whose pre-key work sits on the threshold
loses one slot in four, deterministically. That is the arm.

THE GUARD IS NOT THE FAULT AND IS NOT RELAXED HERE. It caught a burst that would
otherwise have been clamped 29.4 ms into the peer's raster (`SLOT 39 IS GONE`),
which is what it is for. What this scene asks is that the budget it polices be
one the cycle can meet: `onair.TX_ADMIT_RESERVE_S`, measured, held between the
last read and the key. The negative control sets it to zero and loses the slots
again.
"""
from __future__ import annotations

import contextlib

import numpy as np

from hfmodem.shrike import onair, pactor1

from hfmodem.tests.shrike.test_grid import (
    FS, SLOT_N, _Bench, _run, _session, check)

# The arm's own line, less the device and rig arguments the bench supplies.
ARM = ("--p1-grant-only", "--p1-status-bits45", "3", "--p3-entry", "template",
       "--announce-lower", "--p3-entry-stagger", "--p3-entry-rise",
       "--no-long-cycle", "--retries", "6", "--p3-entry-delay", "0")
# The recorded budget-failure scenes used zero entry delay. Keep that explicit
# now that normal sessions restore 4.875 ms before the entry pulse.

# Where the gateway starts commanding PACTOR-3, counted in our own carriers: far
# enough in that the link is CONNECTED and this station is the ISS, which is the
# only state `ptc.PtcHost._take_grant` acts on a grant in.
GRANT_AT = 6

# ...and the cycles it says nothing at all, which is what releases the receive
# window on a real link (`_MasterGrid.MAX_MISSES`). WS8EOC's own release landed
# between the first entry and the second -- `receive window released after 3
# cycles (nothing heard)` -- after which the anchored reader stood 145 ms early
# of every answer the session went on to measure.
QUIET = range(GRANT_AT + 1, GRANT_AT + 4)

# ...and where it takes the channel back, which is what carries this scene from
# the packets we key to the codewords we answer with. The entry budget is spent
# by then and the link has gone back to PACTOR-1 data packets.
TAKEOVER_AT = GRANT_AT + 12

# THE TURNAROUND THIS SCENE'S PEER ANSWERS IN, and it is WS8EOC's own rather than
# the nominal `_Bench.D_S` inherits from `D_NOMINAL_S`. The arm measured 93 to
# 100 ms and `PEER_TURNAROUND_S` puts the corpus median at 96. It is worth naming
# rather than defaulting: the readable band has a ceiling
# (`onair._budget`'s `latest`, 107 ms at this settle once the admission reserve
# is paid), a peer close to it is measuring that ceiling rather than this scene's
# placement, and the answer position is what the receive-window checks below turn
# on.
D_S = 0.090

# THE PRE-KEY WORK, CHARGED WHERE THE ARM PAID IT: between the cycle's last read
# and the admission check. 8.67 ms, inside the 7.0-16.8 ms the arm's own printed
# figures put there (`40 - lead` over 29 keyings of that arm's own, median
# 9.1) and above the 6.23 ms the isolated reproduction reached with none of the
# arm's real overheads.
#
# AND IT IS ONE BLOCK OF THE SAWTOOTH BELOW THE BUDGET, WHICH IS THE POINT.
# `clamp_late` is taken on the delivered count, always a multiple of `_blk`, so
# the room a cycle actually has before the guard trips is
# `4 * _blk - (boundary % _blk)` -- four values 32 samples apart on a raster of
# 60000 = 468.75 blocks. 416 exceeds the smallest of those four and no other,
# whatever phase the connect burst leaves the anchor on, so at zero reserve
# exactly one slot in four is lost and the arithmetic says which.
PREKEY_N = 4 * 128 - 96


def _gateway(*, grant_at: int, quiet: range, takeover_at: int,
             answers: list[int]):
    """WS8EOC: it answers on ITS OWN raster, grants, and then holds the grant.

    NOT REACTIVE TO OUR PACKET LENGTH, which is the whole of what this scene
    measures the receive window against. `test_grid._answer` replies `d` after
    our carrier drops, so a station that shortens its packet by 150 ms is
    answered 150 ms sooner and no anchor can ever be wrong. A real peer holds a
    free-running cycle grid: measured on this arm, seven sample-indexed answers
    across a PACTOR-1-to-PACTOR-3 upgrade, all of them 1052.0 to 1065.2 ms after
    OUR slot boundary while our own packet went from 960 ms to 810.

    So the first answer fixes the instant and every one after it is that instant
    plus whole slots. `quiet` is the fade that releases the window.
    """
    def make(shift):
        comb: list[int] = []

        def put(bench: _Bench, rf_end: int) -> None:
            if bench.answered:
                return
            bench.answered = True
            n = len(bench.emissions)
            if not comb:
                comb.append(rf_end + bench.d_n)
            while comb[0] <= rf_end:
                comb[0] += SLOT_N
            at = comb[0]
            if n in quiet:
                return
            if n >= takeover_at:
                burst = onair._trim_silence(np.asarray(pactor1.breakin_signal(
                    b"BK DE K7ABC", 100, invert=bool(shift()),
                    lead_s=0, tail_s=0), np.float32))
            else:
                word = (pactor1.CS_SPEED if n == 1 else
                        pactor1.CS_59A if n >= grant_at else
                        (pactor1.CS_ACK_A if n % 2 else pactor1.CS_ACK_B))
                burst = onair._trim_silence(np.asarray(
                    pactor1.control_signal(word, invert=bool(shift())),
                    np.float32))
            end = min(at + burst.size, bench.audio.size)
            if end > at:
                bench.audio[at:end] += burst[:end - at]
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
def _reserve(seconds: float):
    was = onair.TX_ADMIT_RESERVE_S
    onair.TX_ADMIT_RESERVE_S = seconds
    try:
        yield
    finally:
        onair.TX_ADMIT_RESERVE_S = was


def _arm(reserve: float, *, acquire_cut: bool = False) -> dict:
    answers: list[int] = []
    with _reserve(reserve), _acquire_cut(acquire_cut):
        got = _session(cycles=6, hold=40, charge=0, decode=True, peer=False,
                       seconds=300.0, keep_upgrade=True, prekey_n=PREKEY_N,
                       d=D_S, extra_argv=ARM,
                       answer=_gateway(grant_at=GRANT_AT, quiet=QUIET,
                                       takeover_at=TAKEOVER_AT,
                                       answers=answers))
    got["answers"] = answers
    return got


def _keyed(got: dict, what: str) -> list[dict]:
    """The bursts of one kind that actually reached the air, in the order flown."""
    return [b for b in got["bursts"] if what in b["what"] and not b["refused"]]


def _slots(bursts: list[dict]) -> list[int]:
    return [b["slot"] for b in bursts]


def _gaps(slots: list[int]) -> list[int]:
    return [b - a for a, b in zip(slots, slots[1:])]


def _owed(got: dict) -> int | None:
    """The slot the grant commands: one on from the packet it arrived behind.

    "Die Gegenstation sendet daraufhin im naechsten Sendeblock" -- the grant
    acknowledges the packet it answers and commands the granted station's very
    next burst (`ptc.PtcHost._take_grant`, pactor3.md §17.1), which the driver
    restates as `0x59A grant -> PACTOR-3 entry packet in the next transmit slot`.
    """
    prior = None
    for b in got["bursts"]:
        if "ENTRY" in b["what"]:
            return prior
        prior = b["slot"]
    return None


def _through_the_entries(got: dict) -> str:
    """The log down to the last entry keyed, and no further.

    What the session does with its slots forty cycles later, holding a link as
    the IRS, is `onair.REGRID_RESERVE_S`'s budget rather than this one.
    """
    lines = got["log"].splitlines()
    last = [i for i, ln in enumerate(lines) if "ENTRY" in ln and "keying" in ln]
    return "\n".join(lines[:last[-1] + 1] if last else lines)


def _why(log: str) -> str:
    return "; ".join(ln.strip() for ln in log.splitlines()
                     if onair.LATE_KEY in ln or onair.SLOT_GONE in ln
                     or onair.SHORT_OF_THE_FLOOR in ln) or "clean"


def a_granted_entry_keys_every_slot_it_is_owed() -> None:
    print("\nA granted PACTOR-3 entry keys every slot the grant commands")
    got = _arm(onair.TX_ADMIT_RESERVE_S)
    check("the gateway's grant was read", "0x59A grant" in got["log"])
    entries = _keyed(got, "ENTRY")
    slots = _slots(entries)
    owed = _owed(got)

    check("the grant is answered with an entry on the slot it commands",
          owed is not None and slots[:1] == [owed + 1],
          f"the packet the grant answered keyed on slot {owed}, "
          f"entries on {slots}")

    check("...and the entry run is CONSECUTIVE -- one slot to the next, which "
          "is the cadence the peer counts its cycles on",
          len(slots) >= 3 and set(_gaps(slots)) == {1},
          f"{len(slots)} entries on slots {slots}, gaps {_gaps(slots)}")

    check("...every one of them on its boundary, to the sample",
          bool(entries) and all(b["first"] == b["boundary"] for b in entries),
          f"{[b['first'] - b['boundary'] for b in entries]} samples")

    upto = _through_the_entries(got)
    check("...with nothing re-aimed, handed back or short of the listen floor "
          "anywhere between the call and the last entry",
          onair.LATE_KEY not in upto and onair.SLOT_GONE not in upto
          and onair.SHORT_OF_THE_FLOOR not in upto, _why(upto))

    # ...AND THE FADE THAT RELEASED THE WINDOW ACTUALLY HAPPENED, or the window
    # check above is asking nothing: the arm's reader moved 145 ms the cycle
    # after `receive window released`, and a scene the peer never faded in would
    # never reach that line.
    check("...with the receive window released mid-run, which is where the "
          "reader moved on the air",
          "receive window released" in upto,
          "released" if "receive window released" in upto
          else "the peer's fade never released it")

    # THE WINDOW IN FRONT OF EACH KEY. `rx_ref_n` is the transmission `d` was
    # measured against, and the peer above answers on its own raster rather than
    # on our packet -- so an upgrade that shortens what we key by 150 ms must not
    # move the reader. The arm's moved 145 ms the cycle after its window was
    # released, and the thirteen grants behind it were never read again.
    # Against the ENTRIES' OWN boundaries, not the session's final anchor: every
    # role reversal behind the run rotates the grid 840 ms, so the raster the
    # session ends on is not the one these cycles were keyed on.
    first, last = entries[0]["boundary"], entries[-1]["boundary"]
    heard = sorted({round((at - first) % SLOT_N / FS * 1e3)
                    for at in got["answers"] if first <= at <= last + SLOT_N})
    windows = [round((b["rx_due"] - b["boundary"]) / FS * 1e3) for b in entries]
    check("the window in front of every entry is aimed where the peer actually "
          "answers, across the release and the upgrade both",
          bool(windows) and bool(heard)
          and all(min(heard) - 40 <= w <= max(heard) + 40 for w in windows),
          f"windows {windows} ms past the boundary, answers {heard} ms")

    # DATA -> CONTROL. Every quantity above belongs to a cycle this station KEYS,
    # and the CS6 receiving-turn work never touched one -- it exercises a turn we
    # are receiving in. So the same three checks are asked again of the bursts
    # behind the entry run, where the link has gone back to PACTOR-1 data packets
    # and then to answering the peer with a control signal.
    for kind, label in (("P1 pkt#", "data packet"), ("P1 CS", "control signal")):
        after = [b for b in _keyed(got, kind)
                 if slots and b["slot"] > slots[-1]]
        if not after:
            continue
        # ONE CADENCE, WHATEVER IT IS. A station holding a link keys on a comb;
        # which increment the comb has is the peer's business and the role's, and
        # a gap that VARIES is the fault -- 2, 1, 3 is what the arm flew.
        check(f"...and the {label}s behind the entry run keep one cadence, "
              f"each on its own boundary to the sample",
              len(set(_gaps(_slots(after)))) <= 1
              and all(b["first"] == b["boundary"] for b in after),
              f"slots {_slots(after)[:8]}..., gaps {sorted(set(_gaps(_slots(after))))}, "
              f"{sorted(set(b['first'] - b['boundary'] for b in after))} samples off")


def the_reserve_is_what_buys_those_slots() -> None:
    """NEGATIVE CONTROL. A guard test that cannot fail is not a guard test.

    Zero reserve is the arithmetic that flew: the cycle's last read is scheduled
    at `boundary - settle_n` and the admission check is taken at
    `boundary - key_notice`, so everything between them has 8 ms plus whatever
    the callback phase happens to leave. Same scene, same charge, same peer.

    WHAT IT COSTS IS THE BOUNDARY ITSELF, and it used to be the slot. An overrun
    inside `onair.KEY_CLAMP_TOL_S` now keys where the converter can rather than
    stepping two slots to keep its polarity, so the run above survives -- and
    what the reserve buys is the thing the positive scene asserts to the sample:
    every entry ON its boundary. Without it they come up inside one, and the
    four-slot sawtooth is still there in WHICH of them do.

    ON THE ARM'S OWN BUDGET, which is what `_acquire_cut(False)` restores. The
    entry cycle now carries a second and larger reserve -- 12 ms returned by
    `_SessionRx.control_collect_until`'s earlier collection cut -- and with that
    in hand the 2 ms below buys nothing this scene can see. The scene after this
    one is that measurement; this one is still the arm.
    """
    print("\nNEGATIVE CONTROL: the same scene with no measured reserve")
    got = _arm(0.0)
    entries = _keyed(got, "ENTRY")
    into = {b["slot"]: b["first"] - b["boundary"] for b in entries}
    late = [s for s, off in into.items() if off]
    check("without the reserve the entries no longer key on their boundary",
          bool(late),
          f"entries on {_slots(entries)}, {sorted(set(into.values()))} samples off")
    check("...and the ones that miss share one residue mod 4, which is the "
          "128-frame callback grid beating against the 1.25 s raster",
          bool(late) and len({s % 4 for s in late}) == 1,
          f"late {late}, residues {sorted({s % 4 for s in late})}")


def the_acquire_cut_is_the_other_reserve() -> None:
    """WHAT BUYS THOSE SLOTS NOW, and it is not the 2 ms above.

    `_SessionRx.control_collect_until` closes this cycle's final collection 12 ms
    early while the entry is pending, so the same campaign at the same charge
    keys every entry on its boundary with `TX_ADMIT_RESERVE_S` set to zero --
    the run the scene above fails on its own budget. Take the cut away and the
    boundaries go, which is that scene; leave it and they stay, which is this
    one. The two together are the whole of what the cycle has in hand.
    """
    print("\nThe same scene with no measured reserve and the acquire cut in hand")
    got = _arm(0.0, acquire_cut=True)
    entries = _keyed(got, "ENTRY")
    into = sorted({b["first"] - b["boundary"] for b in entries})
    check("the 12 ms the acquire cut returns keys every entry on its boundary "
          "with no admission reserve at all",
          bool(entries) and into == [0],
          f"entries on {_slots(entries)}, {into} samples off")


def test_a_granted_entry_keys_every_slot_it_is_owed() -> None:
    assert _run(a_granted_entry_keys_every_slot_it_is_owed)


def test_the_reserve_is_what_buys_those_slots() -> None:
    assert _run(the_reserve_is_what_buys_those_slots)


def test_the_acquire_cut_is_the_other_reserve() -> None:
    assert _run(the_acquire_cut_is_the_other_reserve)
