# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The v23 turn: what a changeover cycle owes, and what a recovered read costs.

`arm-v23-A-40-ws8eoc` (transcript `working/onair-0914-0934`) took the whole
Winlink greeting through `CMS via WS8EOC >` and then refused eight consecutive
changeovers at +5.1 to +14.8 ms while the gateway repeated ` via WS8EOC >`
twenty-eight times. Three things in this file are what that cost:

  * the drain leaves the same interval in front of the admission check whatever
    the cycle is about to build, and a changeover cycle spends 8.2 ms in it
    where an ordinary tick and render measure 1.1 to 1.6 (`_prekey_lead`);
  * `_clamp_forgives` charged that cost a second time, so the first refusal --
    whose cost `RadioTx._tx` records before it asks the clamp -- shut the gate
    for every changeover behind it;
  * and a recovered slot's tracked read transforms the whole window it
    collected, though both tracked readers aim at one instant inside the last
    cycle (`_SessionRx.P3_TRACKED_KEEP_S`).

The fourth scene is the one the report was wrong about. It read `_collect`'s
holdback as coming OUT of the tracked read's reserve; it goes the other way,
and the arithmetic is pinned here rather than argued.
"""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench
from hfmodem.tests.shrike.test_p3_breakin_timing import (
    PAYLOAD, STATUS, SETTLE_N, _Sessrx, duplex, host_stub, transitioned)

FS, SPS = onair.FS, rxfront.SPS

# The arm's own geometry: a 0.96 s frame on the 1.25 s cycle, read at the
# tracked anchor with the peer 13.6 Hz low.
SPAN, CYCLE_N, OFFSET_HZ = 35760, 60000, -13.6

# What a changeover cycle spends between the grid's admission check and the
# emission path's own, on `captures/onair-0914-0934`: 395 samples, 8.2 ms, of
# which `changeover_packet` and its 2% trim are 1.65 and the rest is the tick
# that assembles the B2F login. An ordinary cycle's is 1.1 to 1.6 ms.
CHANGEOVER_COST_N = 395


class _Rx(_Sessrx):
    """...plus the frame scan a recovered slot runs on its own audio."""

    _p3_row0 = None
    _p3_span, _p3_cycle_n = SPAN, CYCLE_N

    def deep_scan(self, audio) -> None:
        pass


def recovered_host():
    """...and what a slot handed back asks the FSM that a kept one does not."""
    host = host_stub()
    host.arq.state = arq.State.CONNECTED
    host.arq.cycle_request = None
    host.arq.cycle_command_emitted = False
    host.arq.cycle_long = False
    return host


def turn_grid():
    """A linked station with a corroborated PACTOR-3 reply position."""
    g = transitioned()
    g.note_p3_control(2551434)
    g.note_p3_control(2611434)
    g.note_p3_packet(2568704, 38880, CYCLE_N, swapped=False,
                     identity=(True, 1, 0, b"RMS"))
    return g


def turn_cycle(tmp_path, *, owed_n, slot=44):
    """A changeover cycle standing a slot behind, as the arm's recovered ones did.

    The clock is put past the aimed slot's key instant, so `_regrid` hands the
    slot back, listens, and drains to `_prekey_lead` in front of the next one --
    which is the drain this file is about.
    """
    g = turn_grid()
    tx = duplex(g, tmp_path)
    tx._flip = lambda: False
    tx.breakin_due = True
    tx.defer_p3_cs = True
    tx.breakin_cost_n = owed_n
    tx.aim(g, slot)
    live = tx.live
    rx = _Rx()
    rx.host = recovered_host()
    # The arm's own geometry: the peer's frame ends 27 ms in front of the key,
    # which is inside the decode deadline -- so `_p3_frame_ready` returns an
    # index a whole cycle in the past, `_collect` waits for nothing, and the
    # drain below is what sets the cycle's position.
    rx._p3_row0 = tx.key_instant(g, slot) - SPAN - 1300
    live.now = live.pos = tx.key_instant(g, slot) + 480 - live.key_notice
    slot, seg, seg_start = onair._regrid(
        live, g, tx, rx.host, rx, slot,
        np.zeros(0, np.float32), 0, SETTLE_N)
    return g, tx, slot


def session_rx(**kw):
    session = _Session(role=arq.IRS, **kw)
    rx = session.rx
    rx.p3_receive_offset_hz = OFFSET_HZ
    rx._p3_row0, rx._p3_span, rx._p3_cycle_n = 3 * CYCLE_N, SPAN, CYCLE_N
    rx.sync.packet_level = 1
    return session, rx


def transforms(monkeypatch):
    """Every window `p3acquire.compensate` is handed, by length."""
    sizes: list[int] = []
    bare = p3acquire.compensate

    def counted(audio, hz):
        sizes.append(int(audio.size))
        return bare(audio, hz)

    monkeypatch.setattr(p3acquire, "compensate", counted)
    return sizes


# -- the drain owes a changeover cycle its own cost --------------------------

def test_the_drain_stands_off_what_this_cycle_still_owes():
    """`_prekey_lead`, which is one number for the read deadline and the drain.

    It was the same 7.6 ms for every cycle, and a changeover's tick and render
    measure 8.2 where a codeword's measure 1.1 to 1.6.
    """
    live = _Bench(seconds=10.0)
    flat = onair._prekey_lead(live, SETTLE_N)
    assert onair._prekey_lead(live, SETTLE_N, CHANGEOVER_COST_N) == \
        flat + CHANGEOVER_COST_N
    # A stream with no transmitter has no DAC to give notice to and owes nothing.
    live.transmit = None
    assert onair._prekey_lead(live, SETTLE_N, CHANGEOVER_COST_N) == SETTLE_N


def test_the_grids_drain_charges_the_changeover_and_an_ordinary_cycle_nothing(
        tmp_path):
    """`_regrid`'s own drain, which is what set the position on the arm.

    The recovered slot listens, collects, and then drains to this lead; the
    tick and the render follow it, and `RadioTx._tx` asks the clamp on the far
    side of both.
    """
    _, owing, slot = turn_cycle(tmp_path, owed_n=CHANGEOVER_COST_N)
    g, plain, plain_slot = turn_cycle(tmp_path, owed_n=0)
    assert slot == plain_slot
    key = owing.key_instant(g, slot)
    assert owing.live.taken[-1][1] == key - onair._prekey_lead(
        owing.live, SETTLE_N, CHANGEOVER_COST_N)
    assert plain.live.taken[-1][1] == key - onair._prekey_lead(
        plain.live, SETTLE_N)
    assert plain.live.taken[-1][1] - owing.live.taken[-1][1] == \
        CHANGEOVER_COST_N


def test_a_changeover_cycle_keys_where_the_drain_paid_for_its_render(
        tmp_path, capsys):
    """The arm's cycle, end to end: 8.2 ms of render behind the grid's check.

    The cost is `breakin_cost_n`, which the previous changeover measured. With
    the drain standing it off the cycle is not late at all and the burst keys;
    without it the placement goes by while the render is still running, which
    is `LATE TO THE KEY ... went +9.1 ms ago` eight times over.
    """
    # Twenty milliseconds, which is past the ten the drain leaves plus the five
    # the reader forgives -- the arm's 8.2 ms against a 7.6 ms interval, scaled
    # to this bench's own notice so the scene asks the gate a question.
    work_n = 960
    for owed_n, keys in ((work_n, True), (0, False)):
        _, tx, _ = turn_cycle(tmp_path, owed_n=owed_n)
        tx.live.spend(work_n)
        tx.breakin_due = True
        got = tx.send_packet(3, PAYLOAD, STATUS, breakin=True)
        assert (got != arq.REFUSED) is keys, (owed_n, got)
        assert bool(tx.live.emissions) is keys
        assert (onair.LATE_KEY in capsys.readouterr().out) is not keys


# -- ...and a refusal's own cost cannot shut the gate behind it ---------------

def test_a_refused_changeover_does_not_shut_the_gate_on_the_next(tmp_path):
    """`_tx` measures the interval BEFORE it asks the clamp, so a refusal
    records its cost too. Charged against the tolerance as well as against the
    drain, an 8.2 ms cost left `room = 240 - 395` negative and no changeover
    overrun could ever be forgiven again: the seven refusals behind the first.
    """
    g = turn_grid()
    tx = duplex(g, tmp_path)
    tx.breakin_due = True
    tx.breakin_cost_n = 10 * round(onair.BREAKIN_CLAMP_TOL_S * FS)
    tol = round(onair.BREAKIN_CLAMP_TOL_S * FS)
    assert onair._clamp_forgives(tx.live, tx, tol) == tol


# -- a tracked read transforms only the cycle it aims at ---------------------

def test_a_tracked_read_transforms_only_the_cycle_it_aims_at(monkeypatch):
    """Both tracked readers aim at ONE instant and both sit in the last cycle.

    `p3acquire.compensate` ran over the whole window instead: 0.88 ms of the
    0.95 s a one-slot cycle collects and 4.18 of the 4.71 s the third recovered
    slot of `arm-v23-A-40-ws8eoc` collected, which is what made a lost slot
    self-sustaining.
    """
    _, rx = session_rx()
    keep = round(rx.P3_TRACKED_KEEP_S * FS)
    recovered = np.zeros(round(4.71 * FS), np.float32)
    sizes = transforms(monkeypatch)
    onair._scan_frame(rx, recovered, 3 * CYCLE_N, tracked_only=True)
    assert sizes and max(sizes) == keep < recovered.size
    # The one-slot cycle is untouched: its window is already shorter than this.
    del sizes[:]
    short = np.zeros(round(.95 * FS), np.float32)
    onair._scan_frame(rx, short, 3 * CYCLE_N, tracked_only=True)
    assert sizes and max(sizes) == short.size


@pytest.mark.parametrize("state", ["cycle_long", "_cycle_command_emitted"])
def test_a_cycle_a_long_frame_may_arrive_in_keeps_its_whole_window(
        state, monkeypatch):
    """`P3_TRACKED_KEEP_S` is the SHORT cycle's number and a long frame is 3.37 s.

    Trimmed to it, `reference-long.wav`'s answer is not in the window at all and
    the comb never moves to the long geometry -- which is
    `test_longcycle_cadence`'s
    `test_the_comb_moves_to_the_long_geometry_only_on_a_crc_valid_long_frame`.
    """
    session, rx = session_rx()
    setattr(session.host.arq, state, True)
    recovered = np.zeros(round(4.71 * FS), np.float32)
    sizes = transforms(monkeypatch)
    onair._scan_frame(rx, recovered, 3 * CYCLE_N, tracked_only=True)
    assert sizes and max(sizes) == recovered.size


def test_the_blind_ladder_behind_a_miss_keeps_the_whole_window(monkeypatch):
    """It has a cycle to spend and it is what finds a changeover."""
    _, rx = session_rx()
    recovered = np.zeros(round(4.71 * FS), np.float32)
    sizes = transforms(monkeypatch)
    onair._scan_frame(rx, recovered, 3 * CYCLE_N)
    assert sizes and max(sizes) == recovered.size


def test_the_anchored_read_keeps_a_window_whose_anchor_is_not_in_the_tail(
        monkeypatch):
    """`_p3_cs`'s anchor is the grid's own codeword position, which a recovered
    slot can leave several cycles back -- and there a trimmed read is lost
    outright rather than cheap."""
    session, rx = session_rx()
    keep = round(rx.P3_TRACKED_KEEP_S * FS)
    recovered = np.zeros(round(4.71 * FS), np.float32)
    sizes = transforms(monkeypatch)
    rx.new_cycle()
    rx.control_signal(recovered, 0, recovered.size - SPS)
    assert sizes and max(sizes) == keep
    del sizes[:]
    rx.new_cycle()
    rx.control_signal(recovered, 0, SPS)
    assert sizes and max(sizes) == recovered.size


def test_both_tracked_readers_still_share_one_transform(monkeypatch):
    """Round 35's memo, which the trim must not step out from under.

    Both readers take the last `P3_TRACKED_KEEP_S` of the same parent buffer, so
    the view they hand `_corrected` is the same samples at the same address.
    """
    session, rx = session_rx()
    recovered = np.zeros(round(4.71 * FS), np.float32)
    sizes = transforms(monkeypatch)
    onair._scan_frame(rx, recovered, 3 * CYCLE_N, tracked_only=True)
    rx.control_signal(recovered, 3 * CYCLE_N, 3 * CYCLE_N + recovered.size - SPS)
    assert len(sizes) == 1


# -- what the holdback is actually on the side of ----------------------------

def test_the_decode_reserve_already_holds_the_holdback():
    """`_p3_decode_deadline`'s reserve is `P3_DECODE_RESERVE_S` PLUS the holdback.

    An earlier reading had it the other way -- "`_collect`'s read returns one
    `holdback` LATER" -- and proposed charging the holdback here. It goes the
    other way. `_collect` is `wait_until` then `read_ready`, and
    `_LiveInput.wait_until` waits on the sample's CAPTURE, "so it returns one
    input latency before that sample can be read": the audio a read hands back
    stops a holdback SHORT of the deadline, and `clamp_late` is taken on that
    same delivered count. The reserve a tracked read keeps in front of the
    admission check is therefore 19.8 ms on the arm's 660-sample holdback, not
    2.5 -- and the level-2 read that arm held is 4.70.
    """
    live = _Bench(seconds=20.0, holdback=660, lat_in=660 - 128)
    key = 10 * CYCLE_N
    until = onair._p3_decode_deadline(live, key, SETTLE_N)
    assert key - until == live.key_notice + round(onair.P3_DECODE_RESERVE_S * FS)
    live.wait_until(until)
    live.read_ready()
    # What the reader got, and what the guard is taken on, are the one count --
    # a holdback behind the deadline it waited for.
    assert live.pos == live.samples
    assert until - live.samples >= live.holdback - live._blk
    assert live.clamp_late(key) == 0
    room = key - live.key_notice - live.samples
    assert room >= round(onair.P3_DECODE_RESERVE_S * FS) + live.holdback - live._blk


@pytest.mark.parametrize("gap", [2587, 2708])
def test_charging_the_holdback_again_costs_the_frame_and_not_the_slack(gap):
    """The measurement that refuses the proposed hunk, on the arm's own spreads.

    `fixtures/prekey-read-0913` keys 2587 to 2828 samples past the frame span's
    end. The two tightest of those are what the extra 660 costs: at 2587 the
    deadline lands inside the frame, `_p3_frame_ready` floors a whole cycle back
    and no read of this cycle runs at all; at 2708 it selects the cycle and then
    stops the delivered window 41 samples short of `UNREAD_TAIL_N`. Only the
    2828 cycles survive the cut, which is why `test_prekey_read_0913` loses two
    of seven deliveries rather than all of them.
    """
    from types import SimpleNamespace
    holdback = 660
    live = SimpleNamespace(key_notice=1652, holdback=holdback)
    row0 = 10 * CYCLE_N
    rx = SimpleNamespace(_p3_row0=row0 - CYCLE_N, _p3_span=SPAN,
                         _p3_cycle_n=CYCLE_N)
    key = row0 + SPAN + gap
    need = row0 + SPAN - rxfront.UNREAD_TAIL_N
    deadline = onair._p3_decode_deadline(live, key, SETTLE_N)
    assert onair._p3_frame_ready(rx, deadline, holdback) - holdback >= need
    charged = deadline - holdback
    assert onair._p3_frame_ready(rx, charged, holdback) - holdback < need
