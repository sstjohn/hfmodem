# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An entry ACK keeps the ISS packet clock, before any IRS reply epoch exists."""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.shrike.p3trial import CYCLE, FS, ReplyClock
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p3_entry_controls import entry_session


def entry_link(tmp_path, *, grid_parity=False):
    session = entry_session()
    session.host.arq.cfg.speed_up = 'hold'
    session.host.arq.cfg.repeat_gear = 0
    session.host.arq.cfg.long_cycle = False
    session.host.arq.cfg.entry_sl = 1
    session.host.arq.cfg.traffic_sl = 1
    grid = onair._MasterGrid(10*FS, CYCLE, 8880, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    grid.shift_slot = -int(grid_parity)
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(session.host)
    session.host.peer = tx
    live = _Bench(seconds=60)
    live.dac_audio = []
    transmit = live.transmit

    def capture(audio, **kwargs):
        live.dac_audio.append(np.array(audio, copy=True))
        return transmit(audio, **kwargs)

    live.transmit = capture
    tx.live, tx.raster, tx.sessrx = live, grid, session.rx
    tx.reply_clock = ReplyClock()
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'none'
    tx.aim(grid, 0)
    live.now = live.pos = 9*FS
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused
    assert tx.reply_clock.entry_phase == 10*FS+234+666
    assert tx.reply_clock.pulse_epoch is None
    return session, tx, live, grid


def bare_ack(session, tx, live, grid, slot):
    # A strong tracked CS1 confirms entry; following CS1s request the odd-
    # counter packet again. Decode the waveform through production RX/ARQ.
    phase = (grid.boundary(slot-1)
             + (tx.reply_clock.entry_phase-grid.anchor) % CYCLE+45000)
    raw = placement.control_signal(arq.CS_ACK)
    audio = np.pad(raw, (4800-placement.pulse_lead(raw), 0))
    live.now = live.pos = phase+10080
    tx.aim(grid, slot)
    session.rx.new_cycle()
    return session.rx.control_signal(audio, phase-4800, phase)


def check_packet(tx, entry):
    audio = np.pad(tx.live.dac_audio[-1], (4800, 4800))
    # This is the native matched-filter/CRC reader on the actually scheduled
    # DAC samples, independently of the transmitting ARQ's counter assertion.
    sync = rxfront.SyncedRx()
    sync.packet_at, sync.packet_level = 4800+666+4320, 1
    packet = sync.packet(audio, preferred_only=True)
    assert packet is not None and packet.packet[1] & 3 == 3
    assert packet.packet[2] == b'bench'
    # Driver audio has already lost its 2% skirt. Recover that cut from the
    # CRC-identified full waveform, rather than measuring the trim twice.
    raw = placement.link_packet(1, packet.packet[2], packet.packet[1],
                                swapped=packet.carrier_swapped)
    lead = placement.pulse_lead(raw)
    pulse = tx.tx_audio_start+lead
    assert (pulse-entry) % CYCLE == 0
    assert tx.reply_clock.pulse_epoch is None
    return packet, pulse


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('grid_parity', [False, True])
def test_bare_entry_ack_and_repeated_packets_keep_entry_pulse(
        tmp_path, monkeypatch, late, grid_parity):
    session, tx, live, grid = entry_link(tmp_path, grid_parity=grid_parity)
    entry = tx.reply_clock.entry_phase
    if late:
        original_tx = tx._tx

        def late_render(audio, what, **kwargs):
            # Rendering exhausts the final enqueue margin after carrier order
            # has been selected. The real backstop skips two physical slots.
            live.now = live.pos = tx.boundary-live.key_notice+480
            return original_tx(audio, what, **kwargs)

        monkeypatch.setattr(tx, '_tx', late_render)
    assert bare_ack(session, tx, live, grid, 1) == arq.CS_ACK
    assert not session.host.arq.entry_pending
    assert session.host.arq.role == arq.ISS and not tx.refused
    first, first_pulse = check_packet(tx, entry)
    assert first_pulse == entry+(3 if late else 1)*CYCLE
    # Entry is physically unswapped regardless of the P1 grid's parity.
    # Its first ordinary successor must continue that physical alternation.
    assert first.carrier_swapped == bool(((first_pulse-entry)//CYCLE) & 1)
    if late:
        monkeypatch.setattr(tx, '_tx', original_tx)
    for slot in (6, 7, 10):
        assert bare_ack(session, tx, live, grid, slot) == arq.CS_ACK
        assert not tx.refused
        packet, pulse = check_packet(tx, entry)
        assert pulse == entry+slot*CYCLE
        assert packet.carrier_swapped == (
            first.carrier_swapped ^ bool(((pulse-first_pulse)//CYCLE) & 1))

    # The entry packet's ISS reference must not prematurely create the IRS
    # epoch: the eventual first CS3 still receives E+603.125 ms replies.
    head = entry+11*CYCLE+45000
    raw = placement.changeover_packet(b'RMS', 0)
    audio = np.pad(raw, (9600-int(np.argmax(placement.protocol_config().pulse())),
                         FS//10))
    origin = head-9600
    live.audio[origin:origin+len(audio)] = audio
    live.now = live.pos = head+round(.84*FS)
    tx.aim(grid, 13)
    session.rx.new_cycle()
    assert session.rx.control_signal(audio, origin, head) == arq.CS_BREAKIN
    assert session.packets and session.packets[-1].packet[2] == b'RMS'
    assert session.host.arq.role == arq.IRS
    grid.reverse(to_iss=False)
    tx.aim(grid, 12)
    onair._p3_place_reply(grid, tx, 12)
    tx.emit_pending_cs()
    assert not tx.refused
    assert tx.reply_clock.pulse_epoch == entry+28950
    assert (tx.tx_audio_start+min(tx.tx_pulse_offsets)-entry) % CYCLE == 28950


@pytest.mark.parametrize('grid_parity', [False, True])
def test_repeated_entry_calibrates_carriers_only_when_emitted(
        tmp_path, monkeypatch, grid_parity):
    session, tx, live, grid = entry_link(tmp_path, grid_parity=grid_parity)
    entry = tx.reply_clock.entry_phase
    # The pinned repeat in the next slot replaces the physical reference.
    tx.aim(grid, 1)
    live.now = live.pos = entry+CYCLE-6000
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused
    assert tx._p3_packet_swap_bias == grid.shift(1)
    # A refused repeat in the opposite parity must not change that reference.
    bias = tx._p3_packet_swap_bias
    with monkeypatch.context() as patch:
        patch.setattr(tx, '_tx', lambda *args, **kwargs: setattr(tx, 'refused', True))
        tx.aim(grid, 2)
        tx.send_entry_packet(1, b'', 0x1a)
    assert tx.refused and tx._p3_packet_swap_bias == bias
    assert bare_ack(session, tx, live, grid, 3) == arq.CS_ACK
    assert not tx.refused
    packet, phase = check_packet(tx, entry)
    # Two cycles after the last actual entry: same physical arrangement.
    assert phase == entry+3*CYCLE
    assert not packet.carrier_swapped
