# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A capture hole is taken at its stamps' word, and every reply slot still keys.

K4MSU 2026-11-07 04:52: one callback ran 4 ms late, the converter stamped 204
samples on the on-time one after it, and the peer's next frame sat exactly one
long cycle on -- the stamps re-anchored and no air was lost. A stamp step the
driver's silence cannot hold moves nothing. Missing air moves our keys and the
peer's raster back together by its stamped size. Either way the reply keys on
the grid that is left, and the first CRC frame past the hole checks the call:
off the raster, everything moves onto that frame -- the reply clock's pulse
epoch with it, or a CRC of a form not yet calibrated measures its gap against
the old epoch and each hole adds its error to the last (astra 11-07, round 6:
910 -> 950 -> 990 ms). No reverse gap is measured from a control before a hole
to a packet after it (round 6: 6747 -> 6027 samples on a 15 ms hole).
"""
from types import SimpleNamespace

import numpy as np

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_clockfit import BLOCK, _Status, _Times, _live_late
from hfmodem.tests.shrike.test_p3_grid_survives_capture_gap import (
    CONTROL, CYCLE, FS, HOLE, IDENTITY, PEER, WIDTH, irs_grid, reply_slot)
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import modem

AIR_N = round(.272 * FS)          # an ACK's carrier, 0.272 s (K4MSU 04:52 TX[22])
SETTLE_N = round(.040 * FS)
END = PEER + 2 * CYCLE + WIDTH    # the first packet past the hole ends here
REPLY_N = round(.910 * FS)


def carrier(audio_ms, moved=0):
    """Our carrier for a reply whose audio starts `audio_ms` past that packet."""
    return END + moved + round(audio_ms * FS / 1e3) - SETTLE_N


def stamped(g, step, silence):
    live = SimpleNamespace(lost=step, xruns=0, fs=FS, lost_at=[(HOLE, step, silence)])
    return onair._CycleEvidence(live, SimpleNamespace(host=None), g, None).input_loss('frame-scan')


def frame(g, k, moved=0, form=IDENTITY):
    g.cycles += 1
    at = PEER + k * CYCLE + moved
    g.note_p3_packet(at, WIDTH, CYCLE, identity=form, swapped=False)
    return at


def test_a_rebased_stamp_on_a_late_callback_moves_nothing_and_keeps_keying():
    """4800 samples stamped on a callback 60 ms late, every sample delivered:
    moved as air, the reply 95 ms past the packet would key 5 ms inside it."""
    g = irs_grid()
    slot = reply_slot(g)
    anchor = g.anchor
    assert not stamped(g, 4800, .0627)
    assert g.anchor == anchor and g._p3_peer[0] == PEER
    assert g.key_refusal(carrier(95), AIR_N) is None
    assert g.p3_control_refusal(slot) is None


def test_missing_air_moves_ours_and_the_peers_together_and_keeps_keying():
    g = irs_grid()
    slot = reply_slot(g)
    boundary = g.boundary(slot)
    assert stamped(g, 4800, (4800 + BLOCK) / FS)
    assert g.boundary(slot) == boundary - 4800
    assert g.p3_control_refusal(slot) is None
    assert g._peer_raster_position(PEER + 3 * CYCLE - 4800, CYCLE) == PEER + 3 * CYCLE - 4800
    assert g.key_refusal(carrier(70, -4800), AIR_N) is None
    assert g.key_refusal(carrier(-5, -4800), AIR_N) is not None


def test_stamps_never_move_the_grid_however_many_come_between_frames():
    g = irs_grid()
    slot = reply_slot(g)
    anchor, boundary = g.anchor, g.boundary(slot)
    for k in range(12):
        g.note_capture_step(480, at=HOLE + k * 1000)
    assert g.anchor == anchor and g.boundary(slot) == boundary
    assert g.p3_control_refusal(slot) is None


def test_the_first_frame_past_a_hole_lands_on_the_raster_or_moves_it():
    for truth in (0, -4800, -720):          # the stamps, or air after all
        g = irs_grid()
        stamped(g, 4800, .0027)
        at = frame(g, 3, truth)
        assert g._peer_raster_position(at, CYCLE) == at
        assert g.p3_control_refusal(reply_slot(g)) is None


def test_a_frame_from_before_a_hole_does_not_check_it():
    g = irs_grid()
    g.note_capture_step(1920, at=HOLE)
    g.note_p3_packet(PEER + 2 * CYCLE, WIDTH, CYCLE, identity=IDENTITY, swapped=False)
    assert g._p3_hole_at == HOLE
    frame(g, 3, -1920)
    assert g._p3_hole_at is None
    assert g._peer_raster_position(PEER + 3 * CYCLE - 1920, CYCLE) == PEER + 3 * CYCLE - 1920


def test_a_hole_of_unknown_place_turns_away_the_readings_taken_before_it_was_seen():
    g = irs_grid()
    g.note_capture_loss(200, seen=HOLE)
    g.cycles += 1
    g.note_p3_packet(PEER + 2 * CYCLE - 200, WIDTH, CYCLE, identity=IDENTITY, swapped=False)
    assert g._p3_peer[0] == PEER - 200


def _reply_offset(clock, at, form):
    return (clock.target(0, at, form=form[:2]) - at) % CYCLE


def test_holes_taken_wrongly_never_walk_the_reply_clock_across_forms():
    """Round 6: 910 ms past the packet; a 40 ms hole of real air taken as a
    stamp; the next CRC at another form. Re-seated with the epoch left behind,
    the new form measured its gap against it: 950 ms, then 990 at a third."""
    g = irs_grid()
    clock = g.reply_clock = ReplyClock(entry_phase=PEER - 1000)
    clock.pulse_epoch = PEER + REPLY_N
    assert _reply_offset(clock, PEER, IDENTITY) == REPLY_N
    lost = 0
    for k, form in ((3, (False, 2, 0, b'')), (6, (False, 3, 0, b''))):
        g.note_capture_step(1920, at=PEER + (k - 1) * CYCLE + WIDTH + 100)
        lost -= 1920
        at = frame(g, k, lost, form)
        assert _reply_offset(clock, at, form) == REPLY_N
    at = frame(g, 7, lost)
    assert _reply_offset(clock, at, IDENTITY) == REPLY_N


def test_a_hole_never_measures_a_reverse_gap_across_it():
    """Round 6: a real 15 ms hole taken as a stamp and a same-form CRC after it
    moved the reverse gap 6747 -> 6027 samples, inside the 20 ms corroboration
    tolerance, and the next changeover with it."""
    for truth, hole in ((0, 204), (-720, 720), (-720, None)):
        g = irs_grid()
        gap = g._p3_reply_timing().reverse_gap
        if hole is None:
            g.note_capture_loss(720, at=HOLE)     # missing air, moved
            truth = -720
        else:
            g.note_capture_step(hole, at=HOLE)
        frame(g, 3, truth)
        assert g._p3_reply_timing().reverse_gap == gap
        # A control keyed after it measures again.
        g.note_p3_control(CONTROL + 4 * CYCLE + truth)
        frame(g, 4, truth)
        assert g._p3_reply_timing().reverse_gap == gap


def test_stamps_between_every_control_and_frame_keep_the_pair_fresh():
    """A stamp moves nothing, so the controls before it still measure: ten
    cycles of control, on-time 204-sample step, frame on the raster."""
    g = irs_grid()
    gap = g._p3_reply_timing().reverse_gap
    for k in range(3, 13):
        g.note_p3_control(CONTROL + k * CYCLE)
        g.note_capture_step(204, at=PEER + (k - 1) * CYCLE + WIDTH + 100)
        frame(g, k)
    timing = g._p3_reply_timing()
    assert timing is not None and timing.reverse_gap == gap
    assert timing.packet_phase == PEER + 12 * CYCLE


def test_a_count_read_before_the_journal_takes_only_what_it_explains():
    """A callback between the count and the journal read journals a step the
    count does not hold yet. Taken together they disagreed, and the whole
    count went as air: 4800 stamped samples moved the grid back."""
    g = irs_grid()
    anchor = g.anchor
    live = SimpleNamespace(lost=4800, xruns=0, fs=FS, air_lost=0,
                           lost_at=[(HOLE, 4800, BLOCK / FS), (HOLE + 1000, 204, BLOCK / FS)])
    evidence = onair._CycleEvidence(live, SimpleNamespace(host=None), g, None)
    assert not evidence.input_loss('frame-scan')
    live.lost = 5004
    assert not evidence.input_loss('frame-scan')
    assert g.anchor == anchor


def _ack_after_a_hole(tmp_path, ahead, lost):
    from hfmodem.tests.shrike.test_p3_reply_placement import REPLY_N as PLACED_N, driver
    from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid as placement_grid
    phase = 1500000
    g = placement_grid()
    g.note_p3_packet(phase, 39120, CYCLE, swapped=False, identity=(False, 1, 2, b'>\r'))
    tx = driver(g, tmp_path)
    tx.p3_control_waveform = 'current'
    slot = onair._p3_place_reply(g, tx, round((phase + PLACED_N - g.anchor) / CYCLE)) + ahead
    tx.aim(g, slot)
    g.cycles += ahead
    boundary = tx.boundary
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - 1920 - lost
    tx.live.lost = lost
    tx.live.lost_at = [(tx.live.pos, lost, (lost + BLOCK) / FS)]
    assert tx._send_p3_control(arq.CS_ACK) is None
    return tx, phase, boundary


def test_a_hole_found_before_the_render_moves_the_aim_with_the_grid(tmp_path):
    """Review of 38ed64fbf: the grid moved and the transmitter kept the boundary
    it was aimed at before, so the ACK keyed 20.8 ms late of a 1000-sample hole."""
    tx, _, boundary = _ack_after_a_hole(tmp_path, 0, 1000)
    assert len(tx.live.emissions) == 1
    assert tx.tx_audio_start == boundary - 1000


def test_a_hole_never_leaves_the_ack_on_the_peers_next_packet(tmp_path):
    """The same two cycles on with aged evidence and 4800 samples of air: on the
    old aim the ACK ran 157 samples into the peer's next packet."""
    tx, phase, boundary = _ack_after_a_hole(tmp_path, 2, 4800)
    assert tx.tx_audio_start == boundary - 4800
    next_packet = phase + 3 * CYCLE - 4800
    assert tx.tx_end <= next_packet


def _k4msu_0452(monkeypatch):
    """The callbacks around K4MSU 11-07 sample 2382592: one entered 6.69 ms
    after the last with an on-time stamp, the next 2.6 ms later stamped 204
    samples on."""
    live, now = _live_late(monkeypatch)
    ind = np.zeros((BLOCK, 1), np.float32)
    entry = adc = 0.0
    for gap, lag in [(BLOCK / FS, 0)] * 40 + [(.00669, 0), (.0026, 204)] + [(BLOCK / FS, 0)] * 8:
        entry += gap
        adc += BLOCK / FS + lag / FS
        now["now"] = entry
        live._stream.callback(ind, BLOCK, _Times(entry, adc), _Status())
    return live


def test_k4msu_0452_rebase_after_a_late_callback_moves_nothing(monkeypatch):
    live = _k4msu_0452(monkeypatch)
    assert live.lost == 204
    (_, step, silence), = live.lost_at
    g = irs_grid()
    slot = reply_slot(g)
    anchor, boundary = g.anchor, g.boundary(slot)
    stamped(g, step, silence)
    assert g.anchor == anchor and g.boundary(slot) == boundary
    assert g.p3_control_refusal(slot) is None
    # The gateway's next frame sat exactly one cycle on, and lands on the raster.
    g.cycles += 1
    g.note_p3_packet(PEER + 3 * CYCLE, WIDTH, CYCLE, identity=IDENTITY, swapped=False)
    assert g._peer_raster_position(PEER + 3 * CYCLE, CYCLE) == PEER + 3 * CYCLE
    assert g.anchor == anchor
    assert live.air_lost == 0


def _presence(live, proof=(600000, 1)):
    a, _ = modem()
    a.begin_receive_opportunity()
    op = a._receive_opportunity
    op.capture_lost = 0
    sessrx = SimpleNamespace(
        host=SimpleNamespace(arq=a, protocol=spec.Protocol.PACTOR3),
        _p3_presence_form=(a._link_generation, 1), _p3_row0=600000, _p3_cycle_n=CYCLE,
        _p3_presence_gaps=(), _p3_repeat_changeover=True, _p3_span=38880,
        _p3_prekey_crc=proof, _p3_delivered_at=600000,
        P3_SL1_RESERVE_S=onair._SessionRx.P3_SL1_RESERVE_S,
        P3_WIDEBAND_RESERVE_S=onair._SessionRx.P3_WIDEBAND_RESERVE_S,
        _p3_acquisition_fits=lambda reserve: False,
        _p3_expected=lambda: (), _corrected=lambda x: x)
    # This packet's CRC, read before the key that answers it.
    raster = SimpleNamespace(boundary=lambda slot: 594000 + slot * CYCLE,
                             _p3_peer=(595680, 38880, CYCLE, 0))
    tx = SimpleNamespace(live=live, slot=1, keyings=(), tx_key_up=None,
                         tx_audio_start=None, tx_end=None)
    # The audio reaches the packet the slot answers; only the budget is short.
    onair._p3_qualify_presence(sessrx, raster, tx, np.zeros(48000, np.float32), 593000)
    a.on_rx_packet(1, b';PM', 0x00, crc_ok=True)
    return a, op


def test_a_capture_hole_is_one_dark_window_and_the_link_keeps_answering():
    a, op = _presence(SimpleNamespace(lost=204, air_lost=204))
    assert op.presence_checked and op.presence_source == 'unsupported'
    assert a.settle_peer_presence()
    assert a._dark_windows == 1 and a.p3_tx_allowed()


def test_a_stamp_step_is_not_a_capture_hole_for_presence():
    # This packet's CRC answers for the peer, and the stamp step that came
    # with it is not a hole across it.
    a, op = _presence(SimpleNamespace(lost=204, air_lost=0))
    assert op.presence_checked and op.presence_source == 'current CRC'
    assert a.peer_qualified


def test_a_budget_skip_without_this_packets_crc_keeps_its_verdict():
    a, op = _presence(SimpleNamespace(lost=0, air_lost=0), proof=None)
    assert op.presence_checked and op.presence_source == 'unsupported'


def test_a_stamp_step_in_the_final_drain_keys_on_its_slot(tmp_path, monkeypatch):
    """12000 samples stamped with the callbacks on time, found while draining to
    the key: the stream lost nothing, so the rendered ACK keys where it was."""
    from hfmodem.tests.shrike.test_p3_reply_placement import REPLY_N as PLACED_N, driver
    from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid as placement_grid
    phase = 1500000
    g = placement_grid()
    g.note_p3_packet(phase, 39120, CYCLE, swapped=False, identity=(False, 1, 2, b'>\r'))
    tx = driver(g, tmp_path)
    tx.p3_control_waveform = 'current'
    slot = onair._p3_place_reply(g, tx, round((phase + PLACED_N - g.anchor) / CYCLE))
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - 1920
    take = tx.live.take_until

    def splice(*args, **kwargs):
        got = take(*args, **kwargs)
        if not tx.live.lost:
            tx.live.lost = 12000
            tx.live.lost_at = [(tx.live.pos, 12000, .0027)]
        return got

    monkeypatch.setattr(tx.live, 'take_until', splice)
    boundary = g.boundary(slot)
    assert tx._send_p3_control(arq.CS_ACK) is None
    assert len(tx.live.emissions) == 1 and tx.slot == slot
    assert tx.tx_audio_start == boundary


def test_a_stamp_is_air_to_nothing():
    forgot, unread = [], [object()]
    g = irs_grid()
    host = SimpleNamespace(forget_p1_phase=lambda: forgot.append(True))
    live = SimpleNamespace(lost=60000, xruns=0, fs=FS, air_lost=0,
                           lost_at=[(HOLE, 60000, .004)])
    tx = SimpleNamespace(host=host, unread_before_key=unread)
    assert not onair._CycleEvidence(live, tx, g, None).input_loss('frame-scan')
    assert not forgot and unread and live.air_lost == 0


def test_a_stamp_moves_nothing_in_any_role():
    g = irs_grid()
    anchor = g.anchor
    g.sending = True
    g.note_capture_step(204, at=HOLE)
    assert g.anchor == anchor


def test_an_on_time_half_block_step_is_a_stamp():
    """Review of 06558b4da: `lost_step` reports a 64.25-sample increment as 64,
    and with the callback exactly on time `0 >= 0` called it air."""
    assert not onair._air_missing(64, BLOCK / FS, FS, BLOCK)
    assert onair._air_missing(64, (64 + BLOCK) / FS, FS, BLOCK)


def test_on_time_callbacks_never_journal_air(monkeypatch):
    """The same review's run: 199 on-time callbacks, each stamped 64.25
    samples past its block, journaled 4864 samples of air."""
    live, now = _live_late(monkeypatch)
    ind = np.zeros((BLOCK, 1), np.float32)
    for k in range(1, 201):
        now["now"] = k * BLOCK / FS
        live._stream.callback(ind, BLOCK, _Times(k * BLOCK / FS, (k * BLOCK + (k - 1) * 64.25) / FS),
                              _Status())
    assert live.lost > 10000 and live.air_lost == 0


def test_a_frame_that_corrects_a_hole_moves_the_aim_with_the_grid(tmp_path):
    """The same review: a stamp of 4800 that was air, corrected by the next CRC
    frame, moved the grid and left the ACK aimed where it was -- 157 samples
    into the peer's next packet."""
    from hfmodem.tests.shrike.test_p3_reply_placement import REPLY_N as PLACED_N, driver
    from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid as placement_grid
    phase = 1500000
    identity = (False, 1, 2, b'>\r')
    g = placement_grid()
    g.note_p3_packet(phase, 39120, CYCLE, swapped=False, identity=identity)
    tx = driver(g, tmp_path)
    tx.p3_control_waveform = 'current'
    slot = onair._p3_place_reply(g, tx, round((phase + PLACED_N - g.anchor) / CYCLE)) + 3
    tx.aim(g, slot)
    aimed = tx.boundary
    g.note_capture_step(4800, at=1540000)
    g.cycles += 1
    g.note_p3_packet(phase + CYCLE - 4800, 39120, CYCLE, swapped=False, identity=identity)
    assert tx.boundary == g.boundary(slot) == aimed - 4800
    g.cycles += 2
    tx.live.now = tx.live.pos = g.boundary(slot) - 1920
    assert tx._send_p3_control(arq.CS_ACK) is None
    assert tx.tx_audio_start == aimed - 4800
    assert tx.tx_end <= phase + 4 * CYCLE - 4800

