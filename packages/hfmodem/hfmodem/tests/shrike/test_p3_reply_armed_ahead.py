# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The reply is placed and rendered before the decode, and keys on that instant.

A reply stands on the answer slot of the packet it answers (`P3_ANSWER_S`),
read off that packet. The instant is fixed when the window is sized: the frame
this cycle decodes moves the next reply, never this one. A forced +69 ms aim without that cost two recorded
slots (VE3KPG's first changeover ACK, slot 0 -> 1; WS8EOC 20:08, 214 -> 215),
because the late re-placement moved a key the clock had already passed.
"""
import threading
import time

import numpy as np
import pytest

from hfmodem.shrike import arq, onair
from hfmodem.shrike.p3trial import ReplyClock, CYCLE
from hfmodem.tests.shrike.tx_latency_support import _Stream, BLK, LAT, SETTLE_N
from hfmodem.tests.shrike.p3_reply_support import (
    FORM,
    WIDTH,
    arm_reply,
    emit_reply,
    frame,
    mail_link,
    owe,
    receive,
    receive_packet,
    receiving,
    setup,
    top_of_cycle,
)

FS = onair.FS
#: Our control's pulse centre past its audio start (`[ack] P3 pulse centers`),
#: so a reply's audio starts this far past the answered packet's phase reference.
LEAD = 852
REPLY_N = round(onair.P3_ANSWER_S * FS) - LEAD


def test_the_reply_sits_on_the_answer_slot_and_follows_the_peer_both_ways(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    assert grid.boundary(index) == origin + (index - 1) * CYCLE + REPLY_N
    for step in (480, -480):      # the peer's raster moves 10 ms, then back
        for _ in range(12):
            index += 1
            grid.cycles = index
            frame(grid, origin + (index - 1) * CYCLE + max(step, 0), index)
            tx.aim(grid, index)
            onair._p3_place_reply(grid, tx, index)
        phase = origin + (index - 1) * CYCLE + max(step, 0)
        assert abs(grid.boundary(index) - phase - REPLY_N) <= 48


def test_a_frame_decoded_inside_the_cycle_never_moves_its_reply(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = index + 1
    tx.aim(grid, slot)
    onair._p3_place_reply(grid, tx, slot)          # the top of the cycle
    placed = grid.boundary(slot)
    # This cycle's packet reads a frame-search step late.
    grid.cycles = slot
    frame(grid, origin + (slot - 1) * CYCLE + 120, slot)
    onair._p3_place_reply(grid, tx, slot, hold=True)    # `_regrid`, the retry
    assert grid.boundary(slot) == placed
    tx._pending_p3_cs = arq.CS_ACK
    live.now = live.pos = tx.key_instant(grid, slot) - 1920
    tx.emit_pending_cs()
    assert not tx.refused and tx.tx_audio_start == placed
    # The next reply follows the reading by its marker's quarter.
    tx.aim(grid, slot + 1)
    onair._p3_place_reply(grid, tx, slot + 1)
    assert grid.boundary(slot + 1) - placed - CYCLE == 30


def test_every_word_the_reply_can_be_is_rendered_before_its_window(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    tx.p3_follow_offset = 'all'
    tx.sessrx.p3_receive_offset_hz = -8.9
    slot = index
    tail, _ = tx._p3_control_shape()
    for cs in (arq.CS_ACK, arq.CS_REQUEST, arq.CS_ACK):
        slot += 1
        tx.aim(grid, slot)
        onair._p3_place_reply(grid, tx, slot)
        tx.arm_p3_reply()
        words = {("cs", word, tail, None) for word in (arq.CS_ACK, arq.CS_REQUEST)}
        assert words <= set(tx._p3_shifted)
        rendered = tx._p3_shifted[("cs", cs, tail, None)][1]
        tx._pending_p3_cs = cs
        live.now = live.pos = tx.key_instant(grid, slot) - 1920
        tx.emit_pending_cs()
        assert not tx.refused and tx.tx_audio_start == grid.boundary(slot)
        # Selected, not rendered again.
        assert tx._p3_shifted[("cs", cs, tail, None)][1] is rendered


@pytest.mark.parametrize('room_ms, decoded', [(40.0, True), (6.0, False)])
def test_the_changeover_answer_keys_on_its_instant_whether_or_not_the_body_decoded(
        tmp_path, monkeypatch, room_ms, decoded):
    """VE3KPG's recorded reversal: with room for the body's decode the ACK keys
    on the instant placed before it; without, the decode is declined and the
    repeat request keys on that same instant instead of the slot going."""
    from hfmodem.tests.shrike.p3_cycle_support import _buffered_listen
    from hfmodem.tests.shrike.p3_changeover_support import ROWS, arm
    row = ROWS[2]
    s, tx, live, g, seg = arm(tmp_path, row)
    delivered = []
    s.host.deliver = delivered.append
    key = row['first_ack_audio'] - row['origin']
    live.samples = key - live.key_notice - round(room_ms * onair.FS / 1e3)
    seg = onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0,
                                    np.zeros(0), 0, 1920)
    assert (b''.join(delivered) == b'***') == decoded
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot, _, _, _ = onair._regrid(live, g, tx, s.host, s.rx, 0, seg, 0, 1920)
    assert slot == 0
    s.host.tick()
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used == [0]
    # The two codewords' pulses sit a sample apart in their bursts.
    assert abs(tx.tx_audio_start - key) <= 1


def test_a_settle_longer_than_the_aim_takes_the_reply_later(tmp_path):
    """An 80 ms settle must not put PTT inside the packet being answered."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    tx.settle = .080
    slot = index + 1
    tx.aim(grid, slot)
    onair._p3_place_reply(grid, tx, slot)
    end = origin + (slot - 1) * CYCLE + WIDTH
    assert grid.boundary(slot) - round(tx.settle * onair.FS) >= end + round(
        onair.P3_PTT_CLEAR_S * onair.FS)


def test_cs5_is_rendered_with_the_acknowledgements(tmp_path):
    """A failed read can reach CS5 after the words were rendered."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    tx.p3_follow_offset = 'all'
    tx.sessrx.p3_receive_offset_hz = -8.9
    tx.arm_p3_reply()
    tail, _ = tx._p3_control_shape()
    assert ("cs", arq.CS_NAK, tail, None) in tx._p3_shifted


# -- the transmitter: an armed burst keys on its own instant -----------------

class _Keyed(_Stream):
    """`_Stream` whose burst has played when the test says so, PTT recorded."""

    def __init__(self, *, ptt_in: float = 0.0) -> None:
        super().__init__()
        self._tx_done = threading.Event()
        self.edges: list[bool] = []
        self.at = self.samples + 40 * BLK + LAT
        # The PTT instant `ptt_in` seconds from now on the system clock.
        self._dac = (time.monotonic() + ptt_in + SETTLE_N / FS
                     - (self.at - LAT) / FS, 0)

    def key(self, on):
        self.edges.append(on)
        return True


def _until(done, timeout=2.0):
    end = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < end, "the keyer never got there"
        time.sleep(0.001)


def _play_once_keyed(stream, seen):
    """The DAC, playing the burst once its PTT is up."""
    def play():
        _until(lambda: stream.edges[:1] == [True])
        seen.append(stream._tx)
        stream._tx_done.set()
    threading.Thread(target=play, daemon=True).start()


def test_an_armed_burst_keys_with_nothing_waiting_on_it():
    """The caller is still decoding: no commit, and the PTT edges come anyway."""
    stream = _Keyed()
    armed = np.ones(BLK, np.float32)
    stream.arm(armed, at=stream.at, settle=SETTLE_N / FS, key=stream.key)
    _until(lambda: stream.edges == [True])
    assert stream._tx is armed and stream._tx_at == stream.at - LAT
    assert stream.slept[0] == stream._dac_time(stream.at - LAT) - SETTLE_N / FS
    stream._tx_done.set()
    _until(lambda: stream.edges == [True, False])
    # The decode's answer arrives after the burst: it is the armed one on record.
    first, end = stream.transmit(np.zeros(BLK, np.float32), at=stream.at,
                                 settle=SETTLE_N / FS)
    assert (first, end) == (stream.at, stream.at + BLK) and stream.missed
    assert stream.edges == [True, False] and stream.armed_at is None


def test_a_commit_ahead_of_the_dac_notice_takes_its_place():
    stream, seen = _Keyed(ptt_in=0.05), []
    stream.arm(np.ones(BLK, np.float32), at=stream.at, settle=SETTLE_N / FS,
               key=stream.key)
    decided = np.full(BLK, 0.5, np.float32)
    _play_once_keyed(stream, seen)
    assert stream.transmit(decided, at=stream.at, settle=SETTLE_N / FS) == (
        stream.at, stream.at + BLK)
    assert seen == [decided] and not stream.missed
    assert stream.ahead == stream.at - LAT - stream.samples - 3 * BLK > 0
    assert stream.edges == [True, False]


def test_a_commit_past_the_dac_notice_leaves_the_armed_audio_on_its_instant():
    stream, seen = _Keyed(ptt_in=0.05), []
    armed = np.ones(BLK, np.float32)
    stream.arm(armed, at=stream.at, settle=SETTLE_N / FS, key=stream.key)
    stream.samples = stream.at - LAT - 2 * BLK      # the decode ran this long
    _play_once_keyed(stream, seen)
    assert stream.transmit(np.full(BLK, 0.5, np.float32), at=stream.at,
                           settle=SETTLE_N / FS) == (stream.at, stream.at + BLK)
    assert seen == [armed] and stream.missed == BLK and stream.ahead is None
    assert stream._tx_at == stream.at - LAT


def test_a_withdrawal_before_the_ptt_keys_nothing():
    stream = _Keyed(ptt_in=5.0)
    stream.arm(np.ones(BLK, np.float32), at=stream.at, settle=SETTLE_N / FS,
               key=stream.key)
    assert stream.disarm()
    assert stream.edges == [] and stream._tx is None and stream.armed_at is None


def test_a_withdrawal_once_keyed_leaves_the_burst_to_go_out():
    stream = _Keyed()
    armed = np.ones(BLK, np.float32)
    stream.arm(armed, at=stream.at, settle=SETTLE_N / FS, key=stream.key)
    _until(lambda: stream.edges == [True])
    assert not stream.disarm() and stream._tx is armed
    stream._tx_done.set()
    stream.transmit(armed, at=stream.at, settle=SETTLE_N / FS)
    assert stream.edges == [True, False]


# -- the reply: the decode chooses the word, never the instant ---------------


def decided_against(tx):
    return arq.CS_REQUEST if tx._p3_armed.cs == arq.CS_ACK else arq.CS_ACK


def test_the_default_answer_is_in_the_stream_before_the_window(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    armed = tx._p3_armed
    assert live.armed_at == grid.boundary(slot) and armed.slot == slot
    assert armed.cs == s.host._counter_cs_for(s.host.arq._p3_error_reply(count=False))


def test_a_decode_in_time_keys_its_word_on_the_armed_instant(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    decided = decided_against(tx)
    owe(s, tx, decided)
    live.now = live.pos = grid.boundary(slot) - 5000
    tx.emit_pending_cs()
    assert not tx.refused and tx.tx_audio_start == grid.boundary(slot)
    assert tx._p3_emitted_cs == decided and tx._pending_p3_cs is None
    assert live.armed_at is None and not live.missed


@pytest.mark.parametrize('overrun', [168, 552, 6000])
def test_a_decode_past_the_dac_notice_keys_the_armed_word_on_its_instant(
        tmp_path, overrun):
    """A review of 1e285fcab: a decode admitted inside its budget that
    finished 168 samples past the DAC's notice keyed its CS 3.5 ms late, and
    552 past lost the slot. Now the armed word keys on its instant and the
    decided one is owed at the next boundary."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    armed, decided = tx._p3_armed, decided_against(tx)
    owe(s, tx, decided)
    acked = []
    s.host.arq.on_cs_emitted = acked.append
    live.now = live.pos = grid.boundary(slot) - live.key_notice + overrun
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used[-1] == slot
    assert live.emissions[-1][0] == tx.tx_audio_start == grid.boundary(slot)
    assert acked == [armed.cs] and tx._p3_emitted_cs == armed.cs
    assert tx._pending_p3_cs == decided


def test_late_speed_up_twice_keeps_unread_counter_and_retries_climb(tmp_path):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1
    assert arm_reply(tx, live, grid, 2) == arq.CS_ACK
    receive_packet(s, 1)
    assert tx._pending_p3_cs == arq.CS_SPEED_UP and a._clean_run == 0
    assert emit_reply(tx, live, late=True) == arq.CS_ACK
    # The late read gets its own ACK before that keyed word can be held.
    assert arm_reply(tx, live, grid, 3) == arq.CS_REQUEST
    assert emit_reply(tx, live, late=True) == arq.CS_REQUEST
    held = arm_reply(tx, live, grid, 4)
    assert emit_reply(tx, live) == held == arq.CS_REQUEST
    assert a.rx_seq == 1 and tx.slots_used[-3:] == [2, 3, 4]
    arm_reply(tx, live, grid, 5)
    receive_packet(s, 2)
    assert tx._pending_p3_cs == arq.CS_SPEED_UP and a._clean_run == 0
    assert emit_reply(tx, live) == arq.CS_SPEED_UP


def test_late_cycle_command_twice_keeps_unread_counter_and_request(tmp_path):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.long_cycle = True
    assert arm_reply(tx, live, grid, 2) == arq.CS_ACK
    receive_packet(s, 1, cycle_long=False, wants_long=True)
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG and a._cycle_request is True
    assert emit_reply(tx, live, late=True) == arq.CS_ACK
    # The late read gets its own ACK before that keyed word can be held.
    assert arm_reply(tx, live, grid, 3) == arq.CS_REQUEST
    assert emit_reply(tx, live, late=True) == arq.CS_REQUEST
    held = arm_reply(tx, live, grid, 4)
    assert emit_reply(tx, live) == held == arq.CS_REQUEST
    assert a.rx_seq == 1 and a._cycle_request is True
    assert not a.cycle_command_emitted and not a.cycle_long
    arm_reply(tx, live, grid, 5)
    receive_packet(s, 2, cycle_long=False, wants_long=True)
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG
    assert emit_reply(tx, live) == arq.CS_CYCLE_TOG
    assert a._cycle_request is True and a.cycle_command_emitted


def test_reconciled_late_cycle_command_keeps_next_unread_counter(tmp_path):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.long_cycle = True
    grid.cycles = 2
    tx.aim(grid, 2)
    live.now = live.pos = tx.boundary - CYCLE // 2
    receive_packet(s, 1, cycle_long=True)
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG and a._cycle_request is False
    phase = grid.boundary(2) - round(onair.P3_COMB_LONG_REPLY_S * FS)
    tx.reconcile_p3_cycle(phase, True)
    assert tx._p3_armed.cs == arq.CS_REQUEST
    assert emit_reply(tx, live, late=True) == arq.CS_REQUEST
    slot = grid.next_slot(tx.slot)
    # The unkeyed CS6 cannot replace the keyed ACK.
    held = arm_reply(tx, live, grid, slot)
    assert emit_reply(tx, live) == held == arq.CS_REQUEST
    assert a.rx_seq == 1 and a._cycle_request is False
    assert not a.cycle_command_emitted and a.cycle_long


def test_refused_repeat_gear_after_earlier_ack_preserves_unread_counter(
        tmp_path, monkeypatch):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.repeat_gear = 3
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1000
    reported = []
    notify = s.host.on_cs_emitted

    def record(emitted):
        reported.append(emitted)
        notify(emitted)

    monkeypatch.setattr(s.host, 'on_cs_emitted', record)
    for slot in (2, 3):
        arm_reply(tx, live, grid, slot)
        receive_packet(s, 1)
        assert emit_reply(tx, live) == arq.CS_REQUEST
    assert a._repeat_run == 2

    assert arm_reply(tx, live, grid, 4) == arq.CS_REQUEST
    receive_packet(s, 1)
    assert tx._pending_p3_cs == arq.CS_REQUEST
    tx.cancel_pending_cs()
    live.now = live.pos = tx.boundary - round(tx.settle * FS) + 1
    tx.emit_pending_cs()
    assert tx._p3_armed is None and live.armed_at is None
    assert tx.slots_used[-1] == 4 and live.emissions[-1][0] == tx.boundary
    assert reported == [arq.CS_REQUEST] * 3 and a._repeat_run == 3

    arm_reply(tx, live, grid, 5)
    live.now = live.pos = tx.boundary - 5000
    receive_packet(s, 1)
    assert tx._pending_p3_cs == arq.CS_SPEED_UP
    assert a.rx_seq == 1 and a._expected_seq == 2
    keyed, slots = len(live.emissions), list(tx.slots_used)
    retained = s.host.sent_total, s.host.rcvd_total, a.rx_progress
    tx.listening = True
    try:
        tx.emit_pending_cs()
    finally:
        tx.listening = False
    assert tx.refused and len(live.emissions) == keyed and tx.slots_used == slots
    assert reported == [arq.CS_REQUEST] * 3
    assert tx._p3_armed is None and live.armed_at is None
    # A refusal cannot prove the peer has not advanced on an earlier ACK.
    assert tx._pending_p3_cs == arq.CS_REQUEST
    assert arm_reply(tx, live, grid, 6) == arq.CS_REQUEST
    assert emit_reply(tx, live) == arq.CS_REQUEST
    assert reported == [arq.CS_REQUEST] * 4
    assert len(live.emissions) == keyed + 1 and tx.slots_used == slots + [6]
    assert a.rx_seq == 1 and a._expected_seq == 2
    assert a._repeat_run == 3
    assert (s.host.sent_total, s.host.rcvd_total, a.rx_progress) == retained


def test_an_answer_nothing_commits_is_withdrawn_before_its_ptt(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    tx._pending_p3_cs = None
    keyed = len(live.emissions)
    live.now = live.pos = grid.boundary(slot) - round(tx.settle * FS) - 1
    tx.emit_pending_cs()                 # the FSM owed nothing this cycle
    assert len(live.emissions) == keyed and live.armed_at is None
    assert tx._p3_armed is None


def test_an_answer_keyed_before_anything_commits_it_is_on_the_record(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    armed = tx._p3_armed
    tx._pending_p3_cs = None
    acked = []
    s.host.arq.on_cs_emitted = acked.append
    live.now = live.pos = grid.boundary(slot) - round(tx.settle * FS) + 1
    tx.emit_pending_cs()
    assert live.emissions[-1][0] == grid.boundary(slot) and tx.slots_used[-1] == slot
    assert acked == [armed.cs] and tx._p3_armed is None


def test_the_grid_never_hands_back_an_armed_slot(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    live.now = live.pos = grid.boundary(slot) - live.key_notice + 900
    kept, _, _, _ = onair._regrid(live, grid, tx, s.host, s.rx, slot,
                               np.zeros(0, np.float32), live.pos, round(tx.settle * FS))
    assert kept == slot and live.armed_at == grid.boundary(slot)


def test_a_hold_places_a_comb_no_clock_has_placed(tmp_path):
    """A review of 1e285fcab: with the reply clock unseated when the
    window was sized, a hold froze the comb the grid had left, and the reply
    keyed 95 ms past its packet or was refused inside the next one."""
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    tx.p3_control_instant = 'peer'
    receive(s, tx, live, grid, entry + 44400)
    origin = grid._p3_peer[0]
    tx.aim(grid, 2)
    assert tx.reply_clock.pulse_epoch is None
    grid.cycles = 2
    frame(grid, origin + CYCLE, 2)         # this cycle's packet, read in the window
    onair._p3_place_reply(grid, tx, 2, hold=True)
    assert grid.boundary(2) == origin + CYCLE + REPLY_N


LONG_WIDTH, LONG_CYCLE = 158160, 180000


def test_a_long_reply_sits_on_the_long_answer_slot(tmp_path):
    """3402.25 ms past the long packet it answers, where the 69.5 ms aim put
    long replies on air and K4MSU and WS8EOC took 116 of 155 of them. WS8EOC
    11-07 17:50Z refused 38 of 45 at 3399, two short cycles past the short slot."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    grid.regear(True)
    start = origin + (index + 1) * CYCLE
    for k in range(4):
        grid.cycles = index + 1 + 3 * k
        grid.note_p3_packet(start + k * LONG_CYCLE, LONG_WIDTH, LONG_CYCLE,
                            swapped=None, identity=(*FORM, k % 4, b'long'))
        # The long reply is the third slot on from its packet's.
        tx.aim(grid, grid.cycles + 3)
        slot = onair._p3_place_reply(grid, tx, grid.cycles + 3)
        assert grid.boundary(slot) + LEAD == start + k * LONG_CYCLE + round(3.40225 * FS)


def test_the_cs6_window_arms_each_owed_long_boundary(tmp_path, monkeypatch):
    from hfmodem.tests.shrike.p3_cycle_support import _walks
    armed = []
    monkeypatch.setattr(onair.RadioTx, "arm_p3_reply",
                        lambda self, **kw: armed.append((self.slot, kw)))
    s, tx, g, walks = _walks(tmp_path, monkeypatch)
    assert armed == [(3, {}), (6, {}), (9, {}), (12, {})]


def test_a_cycle_after_six_unsupported_windows_still_arms_its_answer(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    s.host.arq._dark_windows = 6
    top_of_cycle(tx, live, grid, index + 1)
    assert tx._p3_armed is not None and live.armed_at is not None


def test_an_answer_the_fsm_refuses_once_its_ptt_is_up_keys_whole(tmp_path):
    """K4MSU replay of 2da47927d: answers armed on cycles whose read made the
    proposal obsolete were cut with their PTT already up, keying the rig five
    times with no audio behind it."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    armed = tx._p3_armed
    acked = []
    s.host.arq.on_cs_emitted = acked.append
    s.host.arq._cancel_p3_proposals()         # this cycle's read obsoletes the proposal
    live.now = live.pos = grid.boundary(slot) - round(tx.settle * FS) + 1
    tx.emit_pending_cs()
    assert live.emissions[-1][0] == grid.boundary(slot) and tx.slots_used[-1] == slot
    assert acked == [armed.cs] and live.armed_at is None


def test_the_first_reply_on_a_comb_a_turn_rotated_is_armed(tmp_path):
    """K4MSU 04:51 replay of e311f3a6c: the first ACK after the peer's break-in,
    on the comb the turn rotated (the clock seated off its CS3 head), was
    decided 0.6 ms past the DAC's notice and its slot went unkeyed; the link
    never caught up with the peer after it."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    grid._p3_peer_seeded = True          # placed off a CS3 head, no decoded width
    slot = top_of_cycle(tx, live, grid, index + 1)
    placed = grid.boundary(slot)
    assert placed == origin + index * CYCLE + REPLY_N and live.armed_at == placed
    owe(s, tx, decided_against(tx))
    live.now = live.pos = placed - live.key_notice + 288
    tx.emit_pending_cs()
    assert not tx.refused and live.emissions[-1][0] == placed


def placeable(tmp_path):
    """Two controls keyed on the comb and the packets behind them: the pair
    a changeover is placed on, and the next slot armed."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    for k in (1, 2):
        slot = top_of_cycle(tx, live, grid, index + k)
        owe(s, tx, tx._p3_armed.cs)
        live.now = live.pos = grid.boundary(slot) - 5000
        tx.emit_pending_cs()
        grid.cycles = index + k + 1
        frame(grid, origin + (index + k) * CYCLE, index + k + 1)
    slot = top_of_cycle(tx, live, grid, index + 3)
    return s, tx, live, grid, slot


def test_a_changeover_decided_in_time_keys_in_the_armed_answers_place(tmp_path):
    """WS8EOC 2026-11-07 15:19: settled first, the armed word's commit took the
    changeover's placement with it, and the break-in was guarded as a control
    against the peer's next packet -- the overlap a changeover is for."""
    s, tx, live, grid, slot = placeable(tmp_path)
    live.now = live.pos = grid.boundary(slot) - 5000
    assert tx.send_packet(1, b';FW', 0, breakin=True) != arq.REFUSED
    assert tx.slots_used[-1] == slot and live.armed_at is None and not live.missed
    assert tx.tx_end - tx.tx_audio_start > onair.P3_PACKET_N // 2
    assert not grid._p3_reply_phase_invalid


def test_a_changeover_decided_past_the_notice_leaves_the_slot_to_the_armed_word(
        tmp_path):
    s, tx, live, grid, slot = placeable(tmp_path)
    armed = tx._p3_armed
    acked = []
    s.host.arq.on_cs_emitted = acked.append
    live.now = live.pos = grid.boundary(slot) - live.key_notice + 552
    assert tx.send_packet(1, b';FW', 0, breakin=True) == arq.REFUSED
    assert tx.slots_used[-1] == slot
    assert live.emissions[-1][0] == tx.tx_audio_start == grid.boundary(slot)
    assert acked == [armed.cs]
    keyed = len(live.emissions)
    tx.send_cs(arq.CS_ACK)          # the control a refused changeover falls back to
    tx.emit_pending_cs()
    assert len(live.emissions) == keyed and tx._pending_p3_cs is None


def test_a_burst_behind_a_keyed_armed_word_does_not_take_its_slot(tmp_path):
    s, tx, live, grid, slot = placeable(tmp_path)
    live.now = live.pos = grid.boundary(slot) - round(tx.settle * FS) + 1
    tx._tx(np.zeros(4800, np.float32), "a burst the FSM sent behind it")
    assert tx.refused and tx.slots_used[-1] == slot
    assert live.emissions[-1][0] == grid.boundary(slot) and tx.slot == slot


def test_the_ptt_clears_the_packet_as_read_and_not_only_its_raster(tmp_path):
    """A review of 0f2ea8fe5: a packet 15 ms later than its smoothed raster, a
    20 ms transmit correction, and the armed PTT stood 1.25 ms inside it."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    live.tx_latency_n = 960
    phase = origin + index * CYCLE + 720
    grid.cycles = index + 1
    frame(grid, phase, index + 1)
    top_of_cycle(tx, live, grid, index + 2)
    edge = (round(tx.settle * FS) + live.tx_latency_n - getattr(live, 'geometry_n', 0))
    copy = phase + (live.armed_at - phase) // CYCLE * CYCLE
    assert live.armed_at - edge >= copy + WIDTH


def test_a_packet_read_late_after_arming_warns_and_keys_its_owed_reply(tmp_path, capsys):
    """A recheck of 93a10bf23: armed on the raster, then this cycle's packet
    read 15 ms late; the final guard asked about the settle alone, and a 20 ms
    transmit correction stood the PTT 5 ms inside the packet. Only the settle
    overlaps a packet already read, so the owed ACK keys; a refusal needs our
    audio over the decoded packet, as in the WS8EOC 09-20 ISS pin."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    live.tx_latency_n = 960
    slot = top_of_cycle(tx, live, grid, index + 1)
    phase = origin + index * CYCLE + 720
    grid.cycles = index + 1
    frame(grid, phase, index + 1)
    owe(s, tx, tx._p3_armed.cs)
    live.now = live.pos = grid.boundary(slot) - 5000
    aimed = grid.boundary(slot)
    audio_n = tx._p3_armed.audio.size
    tx.emit_pending_cs()
    edge = round(tx.settle * FS) + live.tx_latency_n - getattr(live, 'geometry_n', 0)
    assert not tx.refused and tx.tx_audio_start == grid.boundary(slot)
    assert tx.tx_audio_start - edge < phase + WIDTH <= tx.tx_audio_start
    assert grid.key_refusal(aimed - edge, edge + audio_n,
                            audio_at=aimed).kind == 'projected'
    assert grid._p3_reply_phase_invalid
    said = capsys.readouterr().out
    assert 'P3 ACK GUARD -- KEYING CS1 ACK seq=0 ANYWAY' in said

def test_a_commit_racing_the_keyer_never_moves_the_audio_into_the_settle():
    """A recheck of 93a10bf23: a commit read `keying` unset, the keyer then
    took PTT, and the commit moved the audio 10 ms earlier into the settle."""
    stream = _Keyed(ptt_in=5.0)
    stream.arm(np.ones(BLK, np.float32), at=stream.at, settle=SETTLE_N / FS,
               key=stream.key)
    b, done = stream._armed, threading.Event()
    start = b.start
    with b.lock:
        threading.Thread(target=lambda: (stream._propose(
            b, np.zeros(BLK, np.float32), stream.at - 480), done.set()), daemon=True).start()
        assert not done.wait(0.1), 'the commit waits on the keyer'
        b.keying = True                     # PTT taken under the lock
    assert done.wait(1.0) and b.start == start and b.missed
    stream.abort()


class _MovedLater(_Keyed):
    """`_Keyed` whose PTT wait sees a commit move the burst 10 ms later."""

    def _sleep_until(self, deadline):
        super()._sleep_until(deadline)
        if len(self.slept) == 1:
            self._propose(self._armed, np.ones(BLK, np.float32), self._armed.at + 480)


def test_a_burst_moved_later_in_the_last_hop_moves_its_ptt_too():
    stream = _MovedLater()
    stream.arm(np.ones(BLK, np.float32), at=stream.at, settle=SETTLE_N / FS,
               key=stream.key)
    _until(lambda: stream.edges == [True])
    assert stream.slept[-1] == stream._dac_time(stream._armed.start) - SETTLE_N / FS
    assert stream._armed.start == stream.at - LAT + 480
    stream._tx_done.set()
    _until(lambda: stream.edges == [True, False])


class _Sleeping(_Keyed):
    """`_Keyed` whose keyer really sleeps to its PTT instant."""

    def _sleep_until(self, deadline):
        time.sleep(max(0.0, deadline - time.monotonic()))


def test_a_withdrawn_keyer_neither_waits_out_its_ptt_nor_clears_a_later_arm():
    """A review of 0057da7cd: withdrawn in its coarse wait, a keyer still slept
    to its old PTT instant, then cleared the output -- a later arm's audio."""
    stream = _Sleeping(ptt_in=1.5)
    stream.arm(np.ones(BLK, np.float32), at=stream.at, settle=SETTLE_N / FS,
               key=stream.key)
    first = stream._armed
    started = time.monotonic()
    assert stream.disarm()
    assert first.done.is_set() and time.monotonic() - started < 0.5
    later = np.full(BLK, 2, np.float32)
    stream.arm(later, at=stream.at + 4 * BLK, settle=SETTLE_N / FS, key=stream.key)
    time.sleep(0.1)
    assert stream._tx is not None and float(stream._tx[0]) == 2.0
    assert stream.disarm()


def test_a_late_qrt_ack_still_goes_out_before_the_link_closes(tmp_path):
    """A review of 0057da7cd: the armed word aired where a QRT's final ACK was
    decided too late for the notice, was reported as that ACK, and closed the
    link with the ACK never sent."""
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    slot = top_of_cycle(tx, live, grid, index + 1)
    a = s.host.arq
    seq = a._expected_seq
    a.on_rx_packet(1, b'', 0x80 | seq, True)
    decided = tx._pending_p3_cs
    assert a._rx_close_pending and decided is not None and decided != tx._p3_armed.cs
    live.now = live.pos = grid.boundary(slot) - live.key_notice + 552
    tx.emit_pending_cs()
    assert a._rx_close_pending and tx._pending_p3_cs == decided


@pytest.mark.parametrize('refusal', ['fresh', 'generic', 'control', 'key'])
def test_linked_p3_owed_cs1_keys_despite_projected_overlap(tmp_path, monkeypatch, capsys, refusal):
    from hfmodem.core.tx_events import read_line

    s, tx, live, grid, origin, index = mail_link(tmp_path)
    tx.aim(grid, index + 1)
    onair._p3_place_reply(grid, tx, index + 1)
    at, slot = grid.boundary(tx.slot), tx.slot
    live.now = live.pos = at - CYCLE // 2
    if refusal == 'control':
        monkeypatch.setattr(grid, 'p3_control_refusal', lambda slot: 'test clock overlap')
    else:
        monkeypatch.setattr(grid, 'key_refusal', lambda *args, **kw:
                            onair._KeyRefusal('test air overlap', 'projected'))
    if refusal != 'control':
        grid.cycles = grid._p3_peer[3] + (2 if refusal == 'generic' else 1)
    capsys.readouterr()
    if refusal in ('control', 'key'):
        tx.arm_p3_reply()
        assert tx._p3_armed is not None and live.armed_at == at
    owe(s, tx, arq.CS_ACK)
    live.now = live.pos = at - 5000
    tx.emit_pending_cs()
    said = capsys.readouterr().out
    events = [row for line in said.splitlines() if (row := read_line(line))
              and row['event'] == 'completed']
    assert any(row['outcome'] == 'sent' and row['slot'] == slot
               and row['audio_start_sample'] == at for row in events)
    diagnostic = {'fresh': 'P3 ACK GUARD -- KEYING', 'generic': 'ACK GUARD -- KEYING',
                  'control': 'P3 REPLY CLOCK -- KEYING', 'key': 'P3 ACK GUARD -- KEYING'}
    assert diagnostic[refusal] in said and 'ANYWAY' in said
    assert ('test clock overlap' if refusal == 'control' else 'test air overlap') in said
    assert tx.guard_drops == 0 and not tx.refused
    if refusal == 'fresh':
        assert grid._p3_reply_phase_invalid


@pytest.mark.parametrize('committed', [False, True])
def test_pending_short_cycle_keeps_a_committed_interior_reply(
        tmp_path, monkeypatch, capsys, committed):
    from hfmodem.tests.shrike.p3_cycle_support import scene, grant

    s, tx, grid, live = scene(tmp_path, monkeypatch, old_long=True)
    grant(s, tx, grid, live, True)
    tx.aim(grid, 1)
    tx.arm_p3_reply()
    armed, burst = tx._p3_armed, live._armed
    submitted = len(live.submitted)
    if committed:
        live.spend(grid.boundary(1) - 1000 - live.now)
        assert burst.keying and not live.disarm()
    capsys.readouterr()
    slot, _, _, _ = onair._p3_transition_window(
        live, grid, tx, s.host, s.rx, 1, live.audio[:0], live.pos, 1920)
    if committed:
        assert slot == tx.slot == 1
        assert tx._p3_armed is armed and live._armed is burst
        assert len(live.submitted) == submitted
        assert not burst.stop.is_set()
    else:
        assert slot == tx.slot == tx._p3_armed.slot == 3
        assert burst.stop.is_set() and not burst.keying
        assert 'for slot 1 withdrawn before its PTT' in capsys.readouterr().out


@pytest.mark.parametrize('state', ['withdrawable', 'missing', 'disconnected'])
def test_long_header_withdrawal_uses_the_reply_lifecycle(
        tmp_path, monkeypatch, capsys, state):
    s, tx, live, grid = receiving(tmp_path)
    arm_reply(tx, live, grid, 2)
    emitted = len(live.emissions)
    if state == 'missing':
        live._armed = None
        monkeypatch.setattr(live, 'disarm', lambda: pytest.fail('no armed burst'))
    elif state == 'disconnected':
        live.now = live.pos = grid.boundary(2) - 1000
        s.host.arq.state = arq.State.DISCONNECTED
    capsys.readouterr()
    assert onair._p3_reply_after_long(tx, 2, 3) == 3
    assert tx._p3_armed is None and live.armed_at is None
    assert len(live.emissions) == emitted
    if state == 'withdrawable':
        assert 'for slot 2 withdrawn before its PTT' in capsys.readouterr().out


@pytest.mark.parametrize('state', ['absent', 'missing', 'withdrawable', 'committed', 'disconnected'])
def test_reply_withdrawal_reports_whether_the_slot_is_free(tmp_path, state):
    s, tx, live, grid = receiving(tmp_path)
    if state != 'absent':
        arm_reply(tx, live, grid, 2)
    armed = tx._p3_armed
    if state == 'missing':
        live._armed = None
    elif state in ('committed', 'disconnected'):
        live.now = live.pos = grid.boundary(2) - 1000      # its PTT is up
    if state == 'disconnected':
        s.host.arq.state = arq.State.DISCONNECTED
    emitted = len(live.emissions)
    assert tx.withdraw_p3_reply() is (state != 'committed')
    assert tx._p3_armed is (armed if state == 'committed' else None)
    # Only a committed answer on a live link is left to air.
    assert (live.armed_at is not None) is (state == 'committed')
    assert len(live.emissions) == emitted


def test_observed_peer_overlap_cannot_arm_an_owed_control(tmp_path):
    s, tx, live, grid, origin, index = mail_link(tmp_path)
    tx.aim(grid, index + 1)
    onair._p3_place_reply(grid, tx, index + 1)
    live.now = live.pos = tx.boundary - CYCLE // 2
    grid.note_peer_codeword(tx.boundary - SETTLE_N, onair.P3_CS_N,
                            'REQ', 'WS8EOC', protocol=onair.Protocol.PACTOR3)
    tx.arm_p3_reply()
    assert tx._p3_armed is None
    assert live.armed_at is None
