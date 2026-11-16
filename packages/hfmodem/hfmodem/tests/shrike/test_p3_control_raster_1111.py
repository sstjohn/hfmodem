# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, spec
from hfmodem.shrike.p3trial import CYCLE, ROTATION
from hfmodem.tests.shrike.p3_reply_support import WIDTH, old_clamp, owe, top_of_cycle, turned
from hfmodem.tests.evidence import require_input

FS = onair.FS
BASELINE = Path(__file__).with_name('p3_control_peer_1111.json')
WORDS = (0, 0, 1, 3, 4, 5)


def controls(directory, option, waveform, hz=0, offset=0, route='data', gap=45783):
    directory.mkdir(exist_ok=True)
    s, tx, live, grid, entry = turned(directory, gap, waveform)
    if option is not None:
        tx.p3_control_instant = option
    tx.p3_follow_offset = 'all'
    s.rx.p3_receive_offset_hz = hz
    s.rx._p3_offset_pending = s.rx._p3_carrier_pending = None
    tx.p3_answer_n += round(offset * FS / 1000)
    head = grid._p3_peer[0]
    if route == 'unseeded':
        grid._p3_yield_pulse = None
    elif route == 'bare':
        tx.reply_clock = None
    elif route == 'long':
        grid.regear(True)
    period = grid.cycle_n
    rows = []
    for i, word in enumerate(WORDS):
        slot = (i + 2) * grid.ticks
        grid.cycles = slot
        phase = head + (i + 1) * period
        grid.note_p3_packet(phase, onair.P3_LONG_PACKET_N if route == 'long' else WIDTH,
                            period, swapped=None,
                            identity=(route == 'changeover', 1, i % 4, b'data'))
        top_of_cycle(tx, live, grid, slot)
        armed = live.armed_at
        foot = tx._p3_armed.swapped if tx._p3_armed else None
        owe(s, tx, word)
        live.now = live.pos = grid.boundary(slot) - 5000
        tx.emit_pending_cs()
        assert not tx.refused and tx.slots_used[-1] == slot
        assert tx.tx_audio_start == (grid.boundary(slot) if route == 'bare' else armed)
        assert tx._p3_emitted_cs == word
        pulse = tx.tx_audio_start + min(tx.tx_pulse_offsets)
        assert grid._p3_controls[-1][0] == (tx.tx_audio_start if route == 'bare' else pulse)
        rows.append(dict(slot=slot, audio=tx.tx_audio_start, pulse=pulse,
                         clock=None if tx.reply_clock is None else tx.reply_clock.pulse_epoch,
                         foot=foot))
    return rows, (s, tx, live, grid, entry)


@pytest.mark.parametrize('waveform', ['historical', 'current'])
@pytest.mark.parametrize('hz', [-13.5, 0, 17.25])
@pytest.mark.parametrize('offset', [-13, 0, 13])
def test_short_data_controls_key_on_own_pulses(tmp_path, waveform, hz, offset):
    rows, (_, tx, _, _, entry) = controls(tmp_path, None, waveform, hz, offset)
    for row in rows:
        error = (row['pulse'] - entry - ROTATION - round(offset * FS / 1000)
                 + CYCLE // 2) % CYCLE - CYCLE // 2
        assert abs(error) <= FS // 1000, (row, error)
    assert len(tx.slots_used) == len(WORDS) + 1
    assert {r['foot'] for r in rows} == ({False, True} if waveform == 'current' else {None})


@pytest.mark.parametrize('waveform', ['historical', 'current'])
@pytest.mark.parametrize('hz', [-13.5, 0, 17.25])
@pytest.mark.parametrize('offset', [-13, 0, 13])
def test_peer_reproduces_original_sample_instants(tmp_path, waveform, hz, offset):
    rows, _ = controls(tmp_path, 'peer', waveform, hz, offset)
    expected = json.loads(require_input(BASELINE).read_text())['scenarios'][f'{waveform}:{hz}:{offset}']
    assert rows == expected


@pytest.mark.parametrize('waveform', ['historical', 'current'])
@pytest.mark.parametrize('route', ['changeover', 'long', 'unseeded', 'bare'])
def test_other_reply_routes_keep_their_instants(tmp_path, waveform, route):
    peer, _ = controls(tmp_path / 'peer', 'peer', waveform, route=route)
    raster, _ = controls(tmp_path / 'raster', 'raster', waveform, route=route)
    assert raster == peer


@pytest.mark.parametrize('waveform', ['historical', 'current'])
def test_raster_snap_does_not_nudge_the_breakin_clock(tmp_path, waveform):
    peer, _ = controls(tmp_path / 'peer', 'peer', waveform)
    raster, _ = controls(tmp_path / 'raster', 'raster', waveform)
    assert [r['clock'] for r in raster] == [r['clock'] for r in peer]
    assert [r['pulse'] for r in raster] != [r['pulse'] for r in peer]


def test_control_instant_defaults_and_modem_parser():
    parser = onair.build_parser()
    assert parser.parse_args([]).p3_control_instant == 'raster'
    assert parser.parse_args(['--p3-control-instant', 'peer']).p3_control_instant == 'peer'
    with pytest.raises(SystemExit):
        parser.parse_args(['--p3-control-instant', 'other'])


def breakin_after_controls(directory, option, waveform, offset, settle):
    _, (_, tx, live, grid, _) = controls(
        directory, option, waveform, offset=offset, gap=45900)
    old_clamp(tx)
    phase, width, period, _ = grid._p3_peer
    grid.cycles += 1
    grid.note_p3_packet(phase + period, width, period, swapped=None,
                        identity=(False, 1, 2, b'next'))
    assert grid._p3_reply_timing() is not None
    slot = tx.slot + 1
    tx.settle = settle
    tx.aim(grid, slot)
    end = grid.peer_packet_end(slot).at_slot
    reference = tx._breakin_reference(grid, slot)
    peer_read = end + grid.peer_read_gap
    floor = end + round((settle + onair.PREKEY_RESERVE_S) * FS)
    key = tx._breakin_key(grid, slot)
    assert key == max(min(reference, peer_read), floor)
    assert (key > reference) == (floor > reference)
    tx.breakin_due = True
    live.now = live.pos = key - CYCLE // 2
    tx.send_packet(1, b';FW', 0, breakin=True)
    assert not tx.refused and tx.slots_used[-1] == slot
    assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == key
    return dict(key=key, reference=reference, peer_read=peer_read, floor=floor)


@pytest.mark.parametrize('waveform', ['historical', 'current'])
@pytest.mark.parametrize('settle', [.04, .1])
def test_first_breakin_follows_measured_controls_with_floor(tmp_path, waveform, settle,
                                                          record_property):
    runs = {}
    for option, offset in [('peer', 0), ('raster', 0), ('raster', -13), ('raster', 13)]:
        runs[option, offset] = breakin_after_controls(
            tmp_path / f'{option}{offset}', option, waveform, offset, settle)
    base = runs['raster', 0]
    assert base['reference'] == runs['peer', 0]['reference']
    assert base['peer_read'] != runs['peer', 0]['peer_read']
    assert (base['floor'] > base['reference']) == (settle == .1)
    for offset in (-13, 0, 13):
        row = runs['raster', offset]
        shift = row['key'] - base['key']
        total = row['key'] - runs['peer', 0]['key']
        assert abs(shift) <= round(abs(offset) * FS / 1000)
        assert shift * offset >= 0
        record_property(f'{offset}_offset_shift_samples', shift)
        record_property(f'{offset}_total_delta_from_peer_samples', total)


@pytest.mark.parametrize('option', ['peer', 'raster'])
@pytest.mark.parametrize('offset', [-13, 0, 13])
def test_saved_retry_breakin_keeps_its_emitted_phase(tmp_path, option, offset):
    from hfmodem.tests.shrike.p3_reply_support import pending
    s, tx, grid, packet = pending(tmp_path)
    tx.p3_control_instant = option
    tx.p3_answer_n += round(offset * FS / 1000)
    grid.note_peer_codeword(7457526, 10080, 'REQ', 'WS8EOC', protocol=spec.Protocol.PACTOR3)
    assert tx._p3_retry_phase(grid, 124) == tx._breakin_key(grid, 124) == 7473433
    tx.settle = .25
    assert tx._breakin_key(grid, 124) == 7473433
    tx.settle = .04
    tx.send_packet(1, b';FW', 0, breakin=True)
    assert not tx.refused and tx.tx_audio_start + min(tx.tx_pulse_offsets) == 7473433
    assert s.host.arq._inflight is packet


@pytest.mark.parametrize('protocol,sending', [(spec.Protocol.PACTOR1, False),
                                              (spec.Protocol.PACTOR2, False),
                                              (spec.Protocol.PACTOR3, True)])
def test_control_instant_is_inert_outside_p3_irs(tmp_path, protocol, sending):
    _, tx, _, grid, _ = turned(tmp_path, 45783, 'current')
    grid.protocol, grid.sending = protocol, sending
    original = grid.anchor, tx.reply_clock.pulse_epoch
    for option in ('raster', 'peer'):
        tx.p3_control_instant = option
        assert tx.p3_reply_shift(2) is None
        assert (grid.anchor, tx.reply_clock.pulse_epoch) == original


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_timing_trial_keeps_its_instants(tmp_path, arm):
    from hfmodem.tests.shrike.p3_reply_support import (setup, receive)
    pulses = []
    for option in ('raster', 'peer'):
        directory = tmp_path / option
        directory.mkdir()
        s, tx, live, grid = setup(directory, arm, delay=234)
        tx.p3_control_instant = option
        receive(s, tx, live, grid, tx.timing_trial.entry_phase + 44400)
        tx.aim(grid, 1)
        tx._pending_p3_cs = 0
        onair._p3_place_reply(grid, tx, 1)
        tx.emit_pending_cs()
        assert not tx.refused
        pulses.append((tx.tx_audio_start, tx.tx_audio_start + min(tx.tx_pulse_offsets)))
    assert pulses[0] == pulses[1]


@pytest.mark.parametrize('waveform', ['historical', 'current'])
def test_unread_slots_keep_keying_the_held_word_on_our_raster(tmp_path, waveform):
    _, (s, tx, live, grid, entry) = controls(tmp_path, 'raster', waveform)
    for slot in range(tx.slot + 1, tx.slot + 4):
        grid.cycles = slot
        top_of_cycle(tx, live, grid, slot)
        word = tx._p3_armed.cs
        owe(s, tx, word)
        live.now = live.pos = grid.boundary(slot) - 5000
        tx.emit_pending_cs()
        assert tx.slots_used[-1] == slot and not tx.refused
        assert tx._p3_emitted_cs == word
        pulse = tx.tx_audio_start + min(tx.tx_pulse_offsets)
        assert abs((pulse - entry - ROTATION + CYCLE // 2) % CYCLE - CYCLE // 2) <= 2


@pytest.mark.parametrize('waveform', ['historical', 'current'])
def test_raster_control_respects_ptt_clearance_floor(tmp_path, waveform):
    _, (s, tx, live, grid, _) = controls(tmp_path, 'raster', waveform)
    tx.settle = .1
    slot = tx.slot + 1
    phase, width, period, _ = grid._p3_peer
    grid.cycles = slot
    grid.note_p3_packet(phase + period, width, period, swapped=None,
                        identity=(False, 1, 2, b'next'))
    top_of_cycle(tx, live, grid, slot)
    owe(s, tx, tx._p3_armed.cs)
    live.now = live.pos = grid.boundary(slot) - 6000
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used[-1] == slot
    assert tx.tx_key_up == phase + period + width + round(onair.P3_PTT_CLEAR_S * FS)
