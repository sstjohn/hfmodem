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
import tempfile
from pathlib import Path

import numpy as np

from hfmodem.shrike import onair, p3acquire, pactor1
from hfmodem.tests.shrike.entry_reply_policy import observe_entry_policy, assert_entry_policy

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
    """Production whole-reply collection, or the historical uncapped deadline."""
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
         keep: str = "", acquire_cut: bool = True) -> dict:
    answers: list[int] = []
    shifts: list[int] = []
    entry_shifts: list[int] = []
    rendering_entry = False
    bare = p3acquire.compensate
    offset = onair.RadioTx._p3_offset

    def counted(x, hz):
        shifts.append(len(x))
        if rendering_entry:
            entry_shifts.append(len(x))
        return bare(x, hz)

    def tuned(self, audio, *args, **kwargs):
        nonlocal rendering_entry
        rendering_entry = kwargs.get("ident", (None,))[0] == "entry"
        try:
            return offset(self, audio, *args, **kwargs)
        finally:
            rendering_entry = False

    argv = ARM + (("--p3-keep-slots", keep) if keep else ())
    p3acquire.compensate = counted
    onair.RadioTx._p3_offset = tuned
    try:
        with observe_entry_policy() as evidence, _acquire_cut(acquire_cut), _cycle(prekey_n, offset_hz,
                                               forgive=forgive):
            got = _session(cycles=6, hold=80, charge=0, decode=True, peer=False,
                           seconds=300.0, keep_upgrade=True, d=D_S,
                           extra_argv=argv, answer=_gateway(answers))
    finally:
        p3acquire.compensate = bare
        onair.RadioTx._p3_offset = offset
    got["answers"], got["shifts"] = answers, shifts
    got["entry_policy"] = evidence
    got["entry_shifts"] = entry_shifts
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
    """Finite owned grants, admitted keys and whole-reply listens."""
    for hz in (0.0, OFFSET_HZ):
        got = _arm(PREKEY_N, hz)
        entries = assert_entry_policy(got, bounded=True)
        first_attempt = next(b for b in got['bursts'] if 'ENTRY' in b['what'])
        assert entries[0] is first_attempt
        assert got['entry_policy']['listens']
        assert got['log'].count(f'TX {hz:+g} Hz') >= len(entries)
        packets = got['entry_shifts']
        assert len(packets) == (1 if hz else 0)


def the_grid_is_what_was_spending_those_slots() -> None:
    """Forced refusals and explicit listens each account for their slots."""
    for charge in (PREKEY_N, SAWTOOTH_N):
        strict = _arm(charge, OFFSET_HZ, forgive=False)
        kept = _arm(charge, OFFSET_HZ)
        assert_entry_policy(strict, bounded=True)
        assert_entry_policy(kept, bounded=True)
        assert strict['entry_policy']['refused']
        assert strict['entry_policy']['listens']
        assert onair.SLOT_GONE in strict['log']
        assert len(strict['entry_policy']['refused']) > len(kept['entry_policy']['refused'])


def the_shipped_cycle_never_reaches_the_forgiveness() -> None:
    """Every admitted key meets its deadline; whole-reply listens are explicit."""
    got = _arm(SHIPPED_PREKEY_N, OFFSET_HZ, forgive=False)
    assert_entry_policy(got, bounded=True)
    assert got['entry_policy']['listens']
    assert onair.SLOT_GONE not in got['log']
    assert onair.SLOT_KEPT not in got['log']


def past_the_tolerance_the_slot_is_still_the_answer() -> None:
    """Overdue attempts are refused, without losing owned-grant accounting."""
    got = _arm(PAST_TOLERANCE_N, OFFSET_HZ)
    assert_entry_policy(got, bounded=True)
    assert got['entry_policy']['refused']
    assert onair.SLOT_GONE in got['log']


def the_flag_ships_on_all_and_that_is_this_campaign() -> None:
    """Default and explicit all preserve the finite campaign and cache."""
    got = _arm(PREKEY_N, OFFSET_HZ, keep='all')
    bare = _arm(PREKEY_N, OFFSET_HZ)
    for run in (got, bare):
        assert_entry_policy(run, bounded=True)
        assert 'keep-slots=all' in run['log']
        assert len(run['entry_shifts']) == 1
    assert _entries(bare) == _entries(got)


def controls_spends_the_changeover_and_keeps_the_rest() -> None:
    """Controls preserves entry ownership and its finite campaign."""
    print("\n--p3-keep-slots controls: the entry campaign is untouched")
    got = _arm(PREKEY_N, OFFSET_HZ, keep="controls")
    assert_entry_policy(got, bounded=True)
    assert 'keep-slots=controls' in got['log']

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
    """Strict admission shares the full-reply listening policy."""
    got = _arm(PREKEY_N, OFFSET_HZ, keep='none')
    reverted = _arm(PREKEY_N, OFFSET_HZ, forgive=False)
    for run in (got, reverted):
        assert_entry_policy(run, bounded=True)
        assert run['entry_policy']['refused']
        assert run['entry_policy']['listens']
        assert onair.SLOT_KEPT not in run['log']
    assert 'keep-slots=none' in got['log']
    assert _entries(reverted) == _entries(got)


def the_acquire_cut_keeps_them_without_reaching_the_forgiveness() -> None:
    """Retained P1 rx_due + 5760 and support precede every listen."""
    got = _arm(PREKEY_N, OFFSET_HZ, forgive=False, acquire_cut=True)
    assert_entry_policy(got, bounded=True)
    assert got['entry_policy']['listens']
    for keyed, extent in got['entry_policy']['listens'].values():
        assert extent[1] - extent[0] == 5760
        assert keyed[2] <= extent[0] < extent[1] <= keyed[1] + keyed[3]


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
