# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded cycle changes and 17 repeat requests through the production seams."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, onair, p3acquire, p3rx, placement, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_cs6_driver import _arm, _window
from hfmodem.tests.shrike.test_cs6_stalls import _duplex
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex

FIX = Path(__file__).parent / 'fixtures/ws8eoc-long-recovery-0920'
META = json.loads((FIX / 'metadata.json').read_text())
FS, SHORT, LONG = 48000, 60000, 180000


def recording(name):
    path = FIX / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META['files'][name]['sha256']
    return rxfront.load_wav(str(path))


def test_first_long_ack_meets_its_own_slot_with_decoder_cost(tmp_path, monkeypatch):
    pcm = recording('first-long.wav')
    s, tx, g, _ = _arm(tmp_path, pcm, expected_seq=2)
    g.anchor = -13171  # Recorded reply audio epoch, shifted by crop origin.
    tx.reply_clock = g.reply_clock = ReplyClock(
        entry_phase=544493-705762, pulse_epoch=573443-705762)
    s.rx.p3_receive_offset_hz = -13.1
    s.rx.sync.packet_level = 1
    clock = _duplex(tx, s, pcm, 2612)
    clock.holdback = 660
    calls = []
    decode = p3rx.decode_at

    def charged(*args, **kwargs):
        calls.append(clock.pos)
        result = decode(*args, **kwargs)
        clock.samples += round(.025 * FS)  # Deliberately more than measured CRC cost.
        return result

    monkeypatch.setattr(p3rx, 'decode_at', charged)
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == 3 and g.ticks == 3
    assert len(s.packets) == 1 and s.packets[0].packet[2].startswith(b'ode 1.4.3.0')
    assert not clock.clamp_late(tx.key_instant(g, slot))
    assert len(calls) == 1  # No short-frame guesses inside the long body.
    tx.emit_pending_cs()
    assert tx.slots_used == [3] and len(clock.emissions) == 1
    assert not tx.refused


@pytest.mark.parametrize('damage', ['body', 'header', 'truncated'])
def test_long_header_alone_cannot_ack_or_commit_cycle(tmp_path, damage):
    pcm = recording('first-long.wav')
    if damage == 'body': pcm[12000:] = 0
    elif damage == 'header': pcm[:] = 0
    else: pcm = pcm[:100000]
    s, tx, g, _ = _arm(tmp_path, pcm, expected_seq=2)
    s.rx.p3_receive_offset_hz = -13.1
    s.rx.sync.packet_level = 1
    g.anchor = -13171
    clock = _duplex(tx, s, pcm, 2612)
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == 3
    assert not s.packets and not s.host.arq.cycle_long
    assert tx._pending_p3_cs is None
    assert not clock.emissions


def old_long_grid():
    g = onair._MasterGrid(32591, SHORT, 0, packet_n=38880, cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(True)
    g.note_p3_packet(1428642, 158160, LONG)
    g.note_p3_control(1593443)
    g.note_p3_packet(1608642, 158160, LONG)
    assert g._p3_reply_timing().reverse_gap == 5119
    return g


def emit_toggle(g, tmp_path, monkeypatch):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, settle=.04)
    tx.reply_clock = ReplyClock(entry_phase=544493, pulse_epoch=573443)
    g.reply_clock = tx.reply_clock
    tx.aim(g, 29)

    def emitted(audio, what, **kwargs):
        tx.n += 1
        tx.refused = False
        tx.tx_pulse_offsets = (852, 852)
        tx.tx_audio_start = 1773443 - min(kwargs["pulse_offsets"])

    monkeypatch.setattr(tx, '_tx', emitted)
    tx._send_p3_control(arq.CS_CYCLE_TOG)
    assert g._p3_command_row0 == 1792962
    assert g._p3_cycle_timing is not None
    return tx


@pytest.mark.parametrize('probe_first', [False, True])
def test_first_crc_short_packet_preserves_measured_turn_position(tmp_path, monkeypatch, probe_first):
    g = old_long_grid()
    tx = emit_toggle(g, tmp_path, monkeypatch)
    if probe_first:
        g.regear(False)  # Scheduler probes before the physical header resolves it.
        assert g._p3_reply_timing() is None
    s = _Session(role=arq.IRS)
    s.host.peer.raster = g
    s.host.arq._cycle_long = True
    s.host.arq._cycle_request = False
    s.host.arq._cycle_command_emitted = True
    s.rx._p3_row0 = 1612962
    s.rx._p3_cycle_n = LONG
    s.rx._p3_span = rxfront._frame_span(placement.LONG_PATHS[1])
    s.rx.sync.packet_level = 1
    s.rx.p3_receive_offset_hz = -13.1
    onair._scan_frame(s.rx, recording('return-short.wav'), 1781762,
                      tracked_only=True, p3_row0=1792962)
    assert s.packets[-1].packet[1] == 0x42
    assert not s.host.arq.cycle_long
    g.regear(False)
    assert g._p3_reply_timing() is not None
    # The return frame's header block sits a quarter symbol past the row the
    # long cycle projects, and a tracked read now takes the block's own row
    # over the lock grid's, so the packet lands at 1793082 and the gap past
    # the emitted command carries those 120 samples.
    assert g._p3_reply_timing().reverse_gap == 5239
    assert g.peer_read_gap == 5561
    tx.aim(g, 30)
    assert tx._place_breakin()
    assert tx.boundary == 1833443


@pytest.mark.parametrize('bad', ['no_command', 'stale', 'off_grid', 'reverse_gap', 'wrong_length', 'reset'])
def test_transition_never_supplies_unmeasured_placement(tmp_path, monkeypatch, bad):
    g = old_long_grid()
    emit_toggle(g, tmp_path, monkeypatch)
    phase, width = 1788642, 39120
    if bad == 'no_command': g._p3_cycle_timing = None
    elif bad == 'stale': g.cycles += onair.ONSET_MAX_CYCLES + 1
    elif bad == 'off_grid': phase += 2000
    elif bad == 'reverse_gap':
        old, control = g._p3_cycle_timing
        g._p3_cycle_timing = old, control + 2000
    elif bad == 'wrong_length': width = 55000
    elif bad == 'reset': g._clear_p3_reply_timing()
    g.regear(False)
    g.note_p3_packet(phase, width, SHORT)
    assert g._p3_reply_timing() is None


def pending_turn(tmp_path):
    g = onair._MasterGrid(32591, SHORT, 0, packet_n=38880, cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 38880
    g.reply_clock = ReplyClock(entry_phase=544493, pulse_epoch=573443)
    g.note_p3_packet(1788642, 39120, SHORT)
    g.note_p3_control(1833443)
    g.note_p3_packet(1848642, 39120, SHORT)
    s = _Session(role=arq.ISS)
    tx = duplex(g, tmp_path)
    tx.reply_clock = g.reply_clock
    s.host.peer = tx
    tx.attach(s.host)
    tx.sessrx = s.rx
    a = s.host.arq
    a._inflight = arq._Packet(status=0, payload=b';FW', sl=1, breakin=True)
    a._next_seq, a._buffer_raw = 1, 3
    a.payload_ledger.queue(a._inflight.payload)
    tx.aim(g, 31)
    tx.breakin_due = True
    tx.live.now = tx.live.pos = 1888800
    tx.send_packet(1, b';FW', 0, breakin=True)
    assert not tx.refused
    g.reverse(to_iss=True)
    g.turn_accepted = False
    s.rx.p3_receive_offset_hz = -13.1
    return s, tx, g


def test_all_seventeen_recorded_repeats_emit_on_retained_comb(tmp_path, monkeypatch):
    s, tx, g = pending_turn(tmp_path)
    pcm = recording('repeat-controls.wav')
    packet = s.host.arq._inflight
    s.rx._p3_answer_at = META['controls'][0] - SHORT
    def blind(*args, **kwargs):
        pytest.fail('packet recovery ran before honoring a valid repeat control')
    monkeypatch.setattr(onair, '_scan_frame', blind)
    for i, phase in enumerate(META['controls']):
        origin = phase - META['control_lead']
        audio = pcm[i*16800:i*16800+13400]
        slot = round((phase + 15713 - g.anchor) / SHORT)
        g.cycles = slot
        tx.aim(g, slot)
        tx.breakin_due = True
        tx.listening = tx.cede_listening = True
        s.rx.new_cycle()
        # The actual collected answer ends before DAC notice, as the repaired
        # provisional cede window now does. Emission uses the production guard.
        tx.live.now = tx.live.pos = phase + 11000
        heard, _, _, _ = onair._read_held_window(
            tx.live, g, tx, s.rx, slot, audio, origin, g.boundary(slot), cede=True)
        assert heard == arq.CS_REQUEST
        assert not tx.listening and not tx.refused
        assert tx.slots_used[-1] == slot
        assert (tx.tx_audio_start + min(tx.tx_pulse_offsets) - 573443) % SHORT == 0
        assert s.host.arq._inflight is packet and packet.payload == b';FW'
        assert s.host.arq._next_seq == 1 and s.host.arq._buffer_raw == 3
    assert len(tx.live.emissions) == 18


@pytest.mark.parametrize('automatic', [False, True])
def test_only_automatic_cede_listening_can_be_cancelled(tmp_path, automatic):
    s, tx, g = pending_turn(tmp_path)
    tx.aim(g, 37)
    tx.live.now = tx.live.pos = 2248700
    tx.listening = True
    tx.cede_listening = automatic
    phase = META['controls'][0]
    pcm = p3acquire.compensate(recording('repeat-controls.wav')[:16800], -13.1)
    ev = rxfront.SyncedRx().control_signal_at(pcm, META['control_lead'])
    assert ev.cs == arq.CS_REQUEST
    s.rx._on(replace(ev, t=phase/FS), anchored=True)
    assert tx.listening is not automatic
    assert len(tx.live.emissions) == (2 if automatic else 1)


def test_automatic_listen_still_collects_and_decodes_peer_packet(tmp_path, monkeypatch):
    s, tx, g = pending_turn(tmp_path)
    tx.aim(g, 33)
    tx.listening = tx.cede_listening = True
    s.rx.new_cycle()
    s.rx._p3_row0 = 1852962
    s.rx._p3_delivered_at = 1852962
    s.rx._p3_cycle_n = SHORT
    s.rx._p3_span = rxfront._frame_span(placement.SPEED_PATHS[1])
    s.rx.sync.packet_level = 1
    s.rx.p3_receive_offset_hz = 0
    phase = 1968642
    wave = placement.link_packet(1, b'', 3)
    start = phase - (placement.protocol_config().pulse().size - 1) // 2
    tx.live.audio[start:start+len(wave)] = wave
    origin = start - 2400
    tx.live.now = tx.live.pos = phase + 36000  # Body still incomplete.
    audio = tx.live.audio[origin:tx.live.pos].copy()
    monkeypatch.setattr(s.rx, 'control_signal_in', lambda *a: (None, None))
    before = len(tx.live.emissions)
    _, _, audio, _ = onair._read_held_window(
        tx.live, g, tx, s.rx, 33, audio, origin, g.boundary(33), cede=True)
    assert s.packets and s.packets[-1].packet == (1, 3, b'', True)
    assert len(tx.live.emissions) == before
    assert tx.listening
