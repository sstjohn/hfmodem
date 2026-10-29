# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A decoder marker change cannot spend milliseconds of the ACK phase."""
import pytest

from hfmodem.shrike import onair, placement
from hfmodem.shrike.p3trial import ReplyClock, CYCLE
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive


def residue(delta):
    return (delta+CYCLE//2) % CYCLE-CYCLE//2


@pytest.mark.parametrize('bias', [-240, 240, -360])
def test_marker_bias_does_not_move_emitted_reply_but_real_drift_does(bias):
    clock = ReplyClock(entry_phase=100000)
    first = clock.target(0, 140000, form=(True, 1))
    data = 200000+bias
    assert residue(clock.target(0, data, form=(False, 1))-first) == 0
    # Repeated pre-key/emit queries do not apply the filter twice.
    assert residue(clock.target(0, data, form=(False, 1))-first) == 0
    assert residue(clock.target(0, data+CYCLE+40, form=(False, 1))-first) == 10
    clock.rotate(28800)
    next_phase = data+5*CYCLE+28800+80
    gap = clock._form_gaps[(False, 1)]
    assert residue(clock.target(0, next_phase, form=(False, 1))-next_phase) == residue(gap)


def test_capture_loss_rebases_calibrated_markers_once():
    clock = ReplyClock(entry_phase=100000)
    clock.target(0, 140000, form=(True, 1))
    before = clock.target(0, 199760, form=(False, 1))
    clock.slip(768)
    after = clock.target(0, 199760+CYCLE-768, form=(False, 1))
    assert after == before+CYCLE-768
    assert clock.target(0, 199760+CYCLE-768, form=(False, 1)) == after


@pytest.mark.parametrize('period', [CYCLE, 3*CYCLE])
def test_k4msu_0546_returning_turn_does_not_reuse_old_data_marker_gap(period):
    clock = ReplyClock(entry_phase=665214)
    clock.target(0, 709326, form=(True, 1))
    clock.target(0, 769326, form=(True, 1))
    clock.target(0, 833646-4320, form=(False, 1))
    clock.target(0, 893526-4320, form=(False, 1))
    clock.rotate(28800)
    # The returning changeover seats the new turn on its actual received
    # phase. The data decoder's old calibration must not undo that placement.
    clock.target(0, 2958255, form=(True, 1))
    anchor = clock.target(0, 3018195, form=(True, 1))
    data = 3082395-4320
    assert residue(clock.target(0, data, form=(False, 1))-anchor) == 0
    assert residue(clock.target(0, data+period-120, form=(False, 1))-anchor) == -30


def test_ws8eoc_0528_marker_positions_keep_first_reply_phase(tmp_path):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    first = entry+44400
    receive(s, tx, live, grid, first)
    tx.aim(grid, 1)
    onair._p3_place_reply(grid, tx, 1)
    lead = placement.historical_control_pulse_lead(0)
    initial = grid.boundary(1)+lead
    # Recorded phase positions from d1d4a94d, first CS3 at469724.
    # Carrier orders independently fitted to the CRC-known recorded fields.
    frames = [(529484,True), (589364,False), (649484,True),
              (709484,False), (769484,True), (829364,False),
              (889364,True), (949364,False), (1009364,True)]
    moves = []
    for index, (phase, swapped) in enumerate(frames, 2):
        grid.cycles = index
        grid.note_p3_packet(first+phase-469724, 39120, CYCLE,
                            swapped=swapped, identity=(False,1,index%4,b'test'))
        tx.aim(grid, index)
        onair._p3_place_reply(grid, tx, index)
        moves.append(residue(grid.boundary(index)+lead-initial))
    # Previous cross-marker filtering advanced these replies by >5 ms.
    # The ordinary decoder alternates between positions 120 samples apart.
    # Its filter follows that quantization, without inheriting the 240-sample
    # offset between the changeover and ordinary decoder measurements.
    assert moves == [0, -30, -22, -17, -13, -40, -60, -74, -86]


def test_k4msu_0538_long_transition_keeps_reply_phase_and_body_clear(tmp_path):
    from hfmodem.tests.shrike.test_p3_reply_placement import key_ack

    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    first = entry+44400
    receive(s, tx, live, grid, first)
    tx.aim(grid, 1)
    onair._p3_place_reply(grid, tx, 1)
    initial = grid.boundary(1)
    # Capture 9ec7bbff: last CS3 at349521; short packet at469281,
    # then first long at529281. Later rows are at713481,893601,1073601.
    frames = [(469281, False), (529281, True), (709161, True),
              (889281, True), (1069281, True)]
    moves = []
    for phase, long_cycle in frames:
        phase += first-349521
        width, period = (158160, 180000) if long_cycle else (39120, CYCLE)
        grid.regear(long_cycle)
        slot = round((phase+width+4320-grid.anchor)/CYCLE)
        grid.cycles = slot
        grid.note_p3_packet(phase, width, period, swapped=False,
                            identity=(False, 1, 0x22 if long_cycle else 1, b'test'))
        tx.aim(grid, slot)
        slot = onair._p3_place_reply(grid, tx, slot)
        key_ack(tx, grid, slot)
        assert not tx.refused
        gap = (tx.tx_audio_start-phase) % period
        assert width < gap < period-11332
        moves.append(residue(tx.tx_audio_start-initial))
    # In the recording, the short ACK moved -60 samples and the first long
    # another -180. Neither movement belongs to these same-phase packets.
    assert moves == [0, 0, -30, -22, -17]
