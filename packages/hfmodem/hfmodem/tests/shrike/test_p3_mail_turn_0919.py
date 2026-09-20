# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded greeting to real mail login without timing-trial termination.

This is a receiver/host/renderer seam test, not the complete onair.main loop or
its wall-clock deadline model. It replays the CRC-valid greeting windows and
final peer role request from WS8EOC on 80 m, then supplies hypothetical decoded
bare ACKs and an FF changeover through the production audio readers. Outgoing
DAC bytes are independently CRC-decoded to prove the complete login was sent.
Only an explicit fake password is used; no real credentials or RF are opened.
"""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3trial, p3acquire, p3rx, placement, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive
from hfmodem.winlink import B2FSession, MailClient

CAP = evidence.CAPTURES / 'onair-0919-2227'
FS, CYCLE = 48000, 60000


def test_recorded_prompt_and_peer_request_start_mail_instead_of_disconnect(tmp_path):
    if not (CAP/'stream.wav').exists():
        pytest.skip('local WS8EOC September 19 22:27 recording unavailable')
    fs, raw = wavfile.read(CAP/'stream.wav')
    assert fs == FS
    raw = raw.astype(np.float32)/32768
    report = json.loads((CAP/'p3-timing-trial.json').read_text())
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    live.audio = np.zeros(180*FS, np.float32)
    live.limit = len(live.audio)
    entry = tx.timing_trial.entry_phase
    s.host.arq.cfg.traffic_sl = 1
    s.host.arq._sl = 1
    tx.timing_trial = None
    tx.reply_clock = p3trial.ReplyClock()
    tx.reply_clock.entry(entry)
    mail = MailClient(B2FSession('W9SSJ', target='WS8EOC', password='offline-only'),
                      s.host.arq.on_host_data)
    s.host.app = mail
    mail.link_up()
    first = entry + (report['opened']-report['entry_phase'])
    shift = first-report['opened']
    receive(s, tx, live, grid, first)
    onair._p3_place_reply(grid, tx, 1)
    tx.emit_pending_cs()
    assert not tx.refused
    s.rx.p3_receive_offset_hz = -5.4
    packets = report['packets'][1:]
    login = None
    previous_phase = report['opened']
    for packet in packets:
        phase = packet['phase']
        grid.cycles += max(1, round((phase-previous_phase)/CYCLE))
        previous_phase = phase
        head = phase+shift
        origin = head-4800
        audio = raw[phase-4800:phase+45000].copy()
        live.now = live.pos = head+round(.84*FS)
        slot = round((head+round(.91*FS)-grid.anchor)/CYCLE)
        tx.aim(grid, slot)
        s.rx.new_cycle()
        onair._scan_frame(s.rx, audio, origin, tracked_only=True,
                          p3_row0=head+4320)
        assert s.packets[-1].packet[1:3] == (
            packet['status'], bytes.fromhex(packet['payload_hex']))
        onair._mail_app_turns(s.host, mail, wait_greeting=True)
        if login is None and mail.session.sent_text:
            login = bytes(s.host.arq._outbuf)
        onair._p3_place_reply(grid, tx, slot)
        onair._reverse_before_key(grid, s.host, tx, slot)
        s.host.tick()
        onair._grid_reversal(grid, s.host)
        tx.emit_pending_cs()
    assert mail.session.sent_text
    assert s.host.arq.state is arq.State.CONNECTED
    assert not s.host.arq._qrt_pending
    assert s.host.arq.role == arq.ISS
    assert tx.timing_trial is None
    assert s.host.arq._inflight is not None
    assert s.host.arq._inflight.breakin
    # The peer now acknowledges our CS3 in its ordinary 210 ms reply slot.
    # Two consistent control copies establish the new direction's RX clock.
    from hfmodem.tests.shrike.test_entry_answer import _answer
    from hfmodem.tests.shrike.test_iss_answer_placement import peer_answers, CS_PHASE_N
    heard = None
    for _ in range(2):
        slot += 1
        grid.cycles += 1
        tx.aim(grid, slot+1)
        audio, origin, at = peer_answers(grid, slot, _answer(arq.CS_ACK, -5.4))
        live.now = live.pos = origin+len(audio)-2000
        s.rx.new_cycle()
        heard = s.rx.control_signal(audio, origin, at+CS_PHASE_N)
        if heard is not None:
            onair._forecast_next_key(s.rx, tx, grid, origin)
        onair._grid_reversal(grid, s.host)
        s.host.tick()
    assert heard == arq.CS_ACK
    assert not s.host.arq._inflight.breakin
    assert s.host.arq._inflight.seq == 1
    assert tx.n > 1
    fs, emitted = wavfile.read(tmp_path/f'tx_{tx.n:02d}.wav')
    assert fs == FS
    decoded = p3rx.decode_p3_packets(np.pad(emitted.astype(float)/32768, (2400,2400))).packets
    assert decoded and decoded[0].status & 3 == 1
    assert decoded[0].payload
    # Continue the real ARQ with synthetic, decoded bare controls until the
    # whole login drains and the normal mail application hands the link back.
    for _ in range(40):
        if s.host.arq._inflight.status & spec.STATUS_CHANGEOVER:
            break
        counter = s.host.arq._inflight.seq
        slot += 1
        grid.cycles += 1
        tx.aim(grid, slot+1)
        audio, origin, at = peer_answers(grid, slot, _answer(counter & 1, -5.4))
        live.now = live.pos = origin+len(audio)-2000
        s.rx.new_cycle()
        assert s.rx.control_signal(audio, origin, at+CS_PHASE_N) == counter & 1
        onair._forecast_next_key(s.rx, tx, grid, origin)
        onair._grid_reversal(grid, s.host)
        onair._mail_app_turns(s.host, mail, wait_greeting=True)
        s.host.tick()
    assert s.host.arq._inflight.status & spec.STATUS_CHANGEOVER
    assert not s.host.arq._outbuf
    # A role-request packet is answered by the peer's new CS3-headed packet,
    # not by inventing a role change from an ordinary CS1/CS2 acknowledgement.
    before_epoch = tx.reply_clock.pulse_epoch
    head = tx.tx_end+4413
    burst = placement.changeover_packet(b'FF\r', 0)
    center = int(np.argmax(placement.protocol_config().pulse()))
    audio = p3acquire.compensate(np.pad(burst, (9600-center, 4800)), 5.4)
    origin = head-9600
    live.now = live.pos = head+round(.84*FS)
    slot += 1
    grid.cycles += 1
    tx.aim(grid, slot)
    s.rx.new_cycle()
    assert s.rx.control_signal(audio, origin, head) == arq.CS_BREAKIN
    assert s.host.arq.role == arq.IRS
    onair._reverse_before_key(grid, s.host, tx, slot)
    slot = int(np.ceil((head+round(.91*FS)-grid.anchor)/CYCLE))
    tx.aim(grid, slot)
    s.rx.expect_frame()
    onair._scan_frame(s.rx, audio, origin)
    assert s.packets[-1].packet[2] == b'FF\r'
    onair._p3_place_reply(grid, tx, slot)
    tx.emit_pending_cs()
    assert not tx.refused
    assert tx.reply_clock.pulse_epoch != before_epoch
    assert (tx.tx_audio_start+min(tx.tx_pulse_offsets)-tx.reply_clock.pulse_epoch) % CYCLE == 0
    assert bytes(s.host.arq._outbuf) == b'FQ\r'
    assert mail.done and s.host._txbuf == 3
    # B2F is done locally, but the IRS still owes these bytes to the peer.
    onair._mail_app_turns(s.host, mail, wait_greeting=True)
    assert s.host.arq._breakin_pending
    assert not s.host.arq._qrt_pending
    assert s.host.arq.state is arq.State.CONNECTED
    assert tx.reply_clock.pulse_epoch != entry+28950
    # Decode actual saved DAC waveforms and reconstruct each newly numbered
    # field. Re-keyed CS3 copies must not count as extra application bytes.
    sent = bytearray()
    expected_seq = 0
    for path in sorted(tmp_path.glob('tx_*.json')):
        saved = json.loads(path.read_text())
        label = saved['control']
        if not label.startswith(('P3 BREAK-IN', 'SL1 pkt')):
            continue
        fs, pcm = wavfile.read(path.with_suffix('.wav'))
        pcm = pcm.astype(float)/32768
        if label.startswith('P3 BREAK-IN'):
            field, valid = p3rx.decode_changeover(
                np.pad(pcm, (4800,4800)), 4800+min(saved['pulse_offsets']))
            assert valid
            status, payload = field[-3], field[:-3]
        else:
            got = p3rx.decode_p3_packets(np.pad(pcm, (2400,2400))).packets
            assert got
            status, payload = got[0].status, got[0].payload
        if status & 3 == expected_seq:
            sent.extend(payload)
            expected_seq = (expected_seq+1) % 4
    assert bytes(sent) == login
