# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The answer to a peer's changeover stands on our own raster, not past its head.

Measured on the 11-08 tapes, each station's packet phase and control pulse on
its own emissions so the path cancels: WS8EOC and K4MSU answer our break-ins
598.0-601.1 ms past their own last packet, short or long, copy after copy. We
answered 890 ms past the gateway's head, which on WS8EOC's and K4MSU's paths is
589-593 ms on ours, and none of 11 such first CS1s was taken. Where KY4RY's and
KB2PCN's paths put the same 886 ms at 597-598 it was taken first time.

Rows: (our last packet's leading pulse to the gateway's head, as read).
"""
import pytest

from hfmodem.shrike import onair, placement, spec
from hfmodem.shrike.p3trial import CYCLE, ROTATION, ReplyClock
from hfmodem.tests.shrike.test_p3_answer_slot_1107 import ANSWER_N, answered
from hfmodem.tests.shrike.test_p3_reply_armed_ahead import WIDTH, owe
from hfmodem.tests.shrike.test_p3_timing_trial import receive, setup

FS = onair.FS

TAPES = {
    # WS8EOC 40 m 19:20Z: ENTRY audio 483091, head 529540, CS1 lead tone 572083
    'ws8eoc-1920': 45783,
    # WS8EOC 80 m 00:19Z: ENTRY audio 602985, head 649416, CS1 pulse 692136
    'ws8eoc-0019': 45765,
    # KY4RY 20 m 23:06Z: SL1 audio 1502984, head 1549746, CS1 lead tone 1592291
    'ky4ry-2306': 46096,
}
# `receive` places a synthetic head's pulse centre; the decoder reads it this
# much later, so the head is placed that much early to read where the tape's did.
READ_LAG = 90


def turned(tmp_path, gap, waveform):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    tx.p3_control_waveform = waveform
    receive(s, tx, live, grid, entry + gap - READ_LAG)
    return s, tx, live, grid, entry


@pytest.mark.parametrize('waveform', ['historical', 'current'])
@pytest.mark.parametrize('tape', sorted(TAPES))
def test_every_changeover_copy_is_answered_on_our_raster(tmp_path, tape, waveform):
    gap = TAPES[tape]
    s, tx, live, grid, entry = turned(tmp_path, gap, waveform)
    head = grid._p3_peer[0]
    assert head == entry + gap
    for slot in range(1, 6):
        if slot > 1:
            grid.cycles = slot
            grid.note_p3_packet(head + (slot - 1) * CYCLE, WIDTH, CYCLE,
                                swapped=None, identity=(True, 1, 0, b';PM'))
        lead = answered(s, tx, live, grid, slot)
        # The two feet's leading pulses sit two samples apart.
        assert abs(lead - entry - ROTATION - slot * CYCLE) <= 2, (slot, lead - entry)
        # Past the head by whatever the path leaves: 896.2 ms at WS8EOC 19:20,
        # 889.7 at KY4RY.
        assert abs(lead - head - (slot - 1) * CYCLE - (ROTATION + CYCLE - gap)) <= 2


@pytest.mark.parametrize('tape', sorted(TAPES))
def test_the_new_isss_data_packets_are_answered_890_ms_past_them(tmp_path, tape):
    s, tx, live, grid, entry = turned(tmp_path, TAPES[tape], 'historical')
    head = grid._p3_peer[0]
    answered(s, tx, live, grid, 1)
    for slot in range(2, 5):
        grid.cycles = slot
        phase = head + (slot - 1) * CYCLE
        grid.note_p3_packet(phase, WIDTH, CYCLE, swapped=None,
                            identity=(False, 1, slot - 1, b'data'))
        lead = answered(s, tx, live, grid, slot + 1)
        assert abs(lead - (phase + CYCLE) - ANSWER_N) <= FS // 1000, slot


def long_iss_grid(pulse):
    """Our ISS stint on the long cycle, its packets' leading pulses on `pulse`."""
    grid = onair._MasterGrid(pulse - 653, 60000, 0, packet_n=38880,
                            cs_n=10080, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, True
    grid.acquired = grid.corroborated = grid.turn_accepted = True
    grid.d_n, grid.d_ref_n = 4800, 162720
    grid.regear(True)
    grid.reply_clock = ReplyClock(entry_phase=1, pulse_epoch=pulse)
    return grid


def test_a_turn_after_a_long_stint_answers_in_the_first_short_slot():
    """WS8EOC 11-09 00:19 TX[32] -> TX[33]: our SL2 LONG keyed at 2851964, its
    leading pulse 653 samples on; the gateway's head read at 3018264. The
    turn's first CS1 went out at pulse 3180457, two cycles and 20 ms early off
    stock's 3.75 s + 600 ms."""
    pulse = 2851964 + 653
    grid = long_iss_grid(pulse)
    slot = round((pulse - 653 - grid.anchor) / grid.slot_n)
    rx_gap = grid.rx_ref_n
    line = grid.reverse(to_iss=False)
    grid.regear(False)                     # every turn opens SL1 short
    assert 'transmit anchor +600 ms' in line
    assert grid._p3_yield_pulse == pulse + ROTATION
    first = slot + onair.LONG_TICKS        # the next cycle's slot, as the driver steps
    assert grid.boundary(first) + 653 == pulse + 180000 + ROTATION
    assert grid.boundary(first) + 653 - 3018264 == 43153          # 899 ms past the head
    # The receive side is unmoved: the peer's next copy is due where it was.
    assert grid.rx_ref_n == rx_gap - (onair.P3_LONG_PACKET_N - grid.cs_n)


def test_a_turn_after_a_short_stint_rotates_as_before():
    grid = long_iss_grid(1000000)
    grid.regear(False)
    before = grid.anchor
    assert 'transmit anchor +600 ms' in grid.reverse(to_iss=False)
    assert grid.anchor - before == round((placement.PACKET_S - 0.210) * FS)


@pytest.mark.parametrize('waveform', ['historical', 'current'])
def test_a_later_turn_is_answered_on_the_raster_our_packets_stood_on(tmp_path, waveform):
    """Our IRS stint, a break-in, then the gateway's changeover again: the comb
    the clock had frozen on the data answers moves once more, onto our raster."""
    s, tx, live, grid, entry = turned(tmp_path, TAPES['ws8eoc-1920'], waveform)
    head = grid._p3_peer[0]
    answered(s, tx, live, grid, 1)
    for slot in range(2, 4):
        grid.cycles = slot
        grid.note_p3_packet(head + (slot - 1) * CYCLE, WIDTH, CYCLE, swapped=None,
                            identity=(False, 1, slot - 1, b'data'))
        answered(s, tx, live, grid, slot + 1)
    ours = tx.reply_clock.pulse_epoch            # where our break-in and packets stand
    grid.reverse(to_iss=True)
    grid.turn_accepted = True
    grid.cycles += 6
    grid.reverse(to_iss=False)
    assert grid._p3_yield_pulse == ours + ROTATION
    # Its head one answer slot past our last packet, as on the tapes.
    grid.note_p3_head(ours + 7 * CYCLE + TAPES['ws8eoc-1920'], CYCLE)
    slot = round((ours + 8 * CYCLE + ROTATION - grid.anchor) / grid.slot_n)
    # Armed straight off the reversal, as `_receive_changeover` arms it.
    tx.aim(grid, slot)
    live.now = live.pos = grid.boundary(slot) - CYCLE // 2
    tx.arm_p3_reply()
    armed = live.armed_at
    owe(s, tx, tx._p3_armed.cs)
    live.now = live.pos = grid.boundary(slot) - 5000
    tx.emit_pending_cs()
    assert not tx.refused and tx.tx_audio_start == armed
    lead = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    assert abs(lead - (ours + 8 * CYCLE + ROTATION)) <= 2


def test_an_unanswered_claim_keeps_the_answer_on_our_raster():
    grid = long_iss_grid(1000000)
    grid.regear(False)
    grid.reverse(to_iss=False)
    held = grid._p3_yield_pulse
    grid.reverse(to_iss=True)                    # our break-in, never answered
    grid.turn_accepted = False
    grid.reverse(to_iss=False)                   # the peer carries on its stint
    assert grid._p3_yield_pulse == held


def test_a_placement_that_would_miss_the_slot_arms_where_the_comb_stood(tmp_path):
    """Review of 995e8a7c9: an earlier operator offset moved a later turn's
    first answer behind the clock, and the arm failed. It arms on the comb it
    found, and the next cycle's placement moves it."""
    s, tx, live, grid, entry = turned(tmp_path, TAPES['ws8eoc-1920'], 'historical')
    head = grid._p3_peer[0]
    answered(s, tx, live, grid, 1)
    ours = tx.reply_clock.pulse_epoch
    grid.reverse(to_iss=True)
    grid.turn_accepted = True
    grid.cycles += 6
    grid.reverse(to_iss=False)
    tx.p3_answer_n = round((onair.P3_ANSWER_S * 1e3 - 80) * FS / 1000)
    grid.note_p3_head(ours + 7 * CYCLE + TAPES['ws8eoc-1920'], CYCLE)
    slot = round((ours + 8 * CYCLE + ROTATION - grid.anchor) / grid.slot_n)
    tx.aim(grid, slot)
    stood = grid.boundary(slot)
    live.now = live.pos = stood - live.LAT - 3 * live._blk + live.lat_in
    tx.arm_p3_reply()
    assert live.armed_at == stood and not grid._p3_yield_placed
    owe(s, tx, tx._p3_armed.cs)
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used[-1] == slot and tx.tx_audio_start == stood
    # The next cycle places it: 80 ms early by the operator's offset, as far
    # as the PTT clears the changeover copy in front of it.
    grid.cycles += 1
    lead = answered(s, tx, live, grid, slot + 1)
    assert grid._p3_yield_placed
    assert ours + 9 * CYCLE + ROTATION - 80 * 48 <= lead < stood + CYCLE + 852
    assert tx.tx_key_up >= grid._p3_peer[0] + CYCLE + 38880
