# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, compress, hostmode, p3acquire, p3rx, rx, rxfront, spec, traffic
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_tx_gear_hold import control, sender
from hfmodem.tests.shrike.test_p3_ve3kpg_proposal_0924 import PROPOSAL, SID, proposal_leg
from hfmodem.winlink.lzhuf import decompress
from hfmodem.winlink.message import Message
from hfmodem.winlink.session import B2FSession, _Outbound

FIXTURE = Path(__file__).with_name('fixtures') / 'cs4-credit-1003'

# These fixtures reconstruct locally settled TX fields, not a gateway's
# delivered stream. Synthetic ACK closure and hypothetical peer models below
# cannot establish remote B2F acceptance or absence of duplicates on a retry.


@pytest.fixture(scope='module', params=['onair-1003-0033', 'onair-1002-1137'])
def arm(request):
    row = json.loads((FIXTURE / (request.param + '.json')).read_text())
    row['_packets'] = load_packets(row)
    offset = 0
    records = {record['id']: record for record in row['records']}
    for k in row['locally_settled']:
        assert records[k]['wire_offset'] == offset
        offset += len(row['_packets'][k])
    row['_wire'] = b''.join(row['_packets'][k] for k in row['locally_settled'])
    row['_expected'] = compress.Supervisor().feed(row['_wire'])
    assert len(row['_expected']) == row['expected']['bytes']
    assert hashlib.sha256(row['_expected']).hexdigest() == row['expected']['sha256']
    _, blob = blocks(row['_expected'])
    row['_original'] = decompress(blob, expected_size=row['original']['bytes'])
    assert hashlib.sha256(row['_original']).hexdigest() == row['original']['sha256']
    return row


@pytest.fixture(scope='module')
def decoded(arm):
    return arm['_packets']


def load_packets(arm):
    cap = evidence.CAPTURES / arm['arm']
    for name, row in arm['files'].items():
        path = evidence.require_input(cap / name)
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == row['sha256'], name
        if name.endswith('.json'):
            meta = json.loads(path.read_text())
            assert all(meta[k] == v for k, v in row.items() if k != 'sha256')
    packets = {}
    for row in arm['records']:
        k = row['id']
        meta = json.loads((cap / f'tx_{k}.json').read_text())
        audio = p3acquire.compensate(rxfront.load_wav(str(cap / f'tx_{k}.wav')),
                                    meta['tx_offset_hz'])
        if row['changeover']:
            field, ok, _ = p3rx.decode_changeover_details(audio, int(np.mean(meta['pulse_offsets'])))
            assert ok and field[-3] == row['status']
            payload = spec.field_payload(field[:-3])
        else:
            scan, _ = p3rx.decode_headed(audio, 0, len(audio), levels=(row['sl'],))
            assert len(scan.packets) == 1
            packet, = scan.packets
            assert (packet.sl, packet.status) == (row['sl'], row['status'])
            payload = packet.payload
        assert len(payload) == row['bytes'], k
        assert hashlib.sha256(payload).hexdigest() == row['sha256'], k
        packets[k] = payload
    return packets


def seed(arm):
    host, seam = sender(enabled=False, payload=arm['_expected'])
    host.p3_mail_active = True
    a = host.arq
    wire = compress.transparent(arm['_expected'])
    a._inflight = arq._Packet(0, wire[:3], 1, breakin=True)
    a._outbuf = bytearray(wire[3:])
    a._buffer_raw = len(wire)
    return host, seam, wire


def test_both_recordings_cs4_on_each_carrier_and_parity(arm, decoded):
    cap = evidence.CAPTURES / arm['arm']
    audio = rxfront.load_wav(str(cap / 'stream.flac'))
    pulse = rx._pulse(480)
    delay = (len(pulse) - 1) // 2
    for sample in arm['cs4_samples']:
        crop = audio[sample - 6000:sample + 15000]
        event = rxfront.SyncedRx().control_signal_tracked(crop, 6000, details=False)
        assert event is not None and event.cs == arq.CS_SPEED_UP
        for tone in rx.HEADER_TONES:
            baseband = rx._baseband(crop, tone, 48000, pulse)
            symbols = baseband[6000 + delay + np.arange(21) * 480]
            bits = (np.real(symbols[1:] * symbols[:-1].conj()) < 0).astype(np.uint8)
            ci, errors = rx.nearest_control_signal(bits)
            assert (ci, errors) == (arq.CS_SPEED_UP, 0)
            assert sum(int(b) << i for i, b in enumerate(bits)) == 0x95ac5
    if arm['lost'] == 71:
        assert arm['controls'] == arm['control_words']
        for sample, word in zip(arm['control_samples'], arm['control_words']):
            crop = audio[sample - 6000:sample + 15000]
            event = rxfront.SyncedRx().control_signal_tracked(crop, 6000, details=False)
            assert event is not None and event.cs + 1 == word
            assert '0 bit errors' in event.text
    host, _, _ = seed(arm)
    host.arq._inflight.status = 2
    for physical, meaning in [(2, arq.CS_REQUEST), (1, arq.CS_ACK), (3, arq.CS_BREAKIN)]:
        event = rxfront.Event(0, 'cs', '', protocol=spec.Protocol.PACTOR3, cs=physical - 1)
        assert host._counter_logical_cs(event) == meaning
    lost = decoded[arm['lost']]
    assert lost == (b'9SSJ PACTOR test via K4' if arm['lost'] == 71 else b'9SSJ PACTOR test via WS')
    log = traffic.TrafficLog()
    assert repr(lost) in '\n'.join(log.packet('TX', 2, 1, lost))
    assert decoded[arm['lost']] == lost


@pytest.mark.xfail(strict=True, reason='Operator decision 2026-10-03 never lose bytes: CS4 alone no longer credits a field')
def test_recorded_seam_keeps_parent_acceptance_without_a_discriminator(arm):
    host, seam, wire = seed(arm)
    a = host.arq
    control(host, 4)
    assert a._buffer_raw == len(wire) - 3
    assert a._inflight.payload == wire[3:26] and a.tx_seq == 1
    control(host, 4)
    assert a._buffer_raw == len(wire) - 26 and a.tx_seq == 2
    assert a._inflight.payload + bytes(a._outbuf) == wire[26:]
    assert seam.frames[:2] == [(2, 1, wire[3:26]), (3, 2, wire[26:85])]


def test_matching_ack_model_delivers_complete_b2f_once(arm):
    host, seam, wire = seed(arm)
    a = host.arq
    delivered = wire[:3]
    control(host, 4)
    while a._buffer_raw:
        packet = a._inflight
        delivered += packet.payload
        control(host, 1 if packet.seq % 2 == 0 else 2)
    assert delivered == wire
    assert compress.Supervisor().feed(delivered) == arm['_expected']
    assert delivered.count(wire[3:26]) == 1


def test_cs4_unanswered_tick_then_cs5_downshifts_and_preserves_suffix(arm):
    host, seam, wire = seed(arm)
    a = host.arq
    control(host, 4)
    outstanding = wire[3:]
    a._rx_this_cycle = False
    host.tick()
    control(host, 5)
    assert a.state == arq.State.CONNECTED and not a._qrt_pending
    assert a.speed_level == 1 and a.tx_seq == 1
    assert a._buffer_raw == len(outstanding)
    assert a._inflight.payload + bytes(a._outbuf) == outstanding
    assert seam.frames[-1] == (1, 1, outstanding[:5])


@pytest.mark.xfail(strict=True, reason='Operator decision 2026-10-03 never lose bytes: CS4 alone no longer credits a field')
def test_original_ve3kpg_three_cs4_accepts_distinct_fields_and_complete_sid():
    host, seam, peer = proposal_leg()
    for cs, takes in [(4, False), (4, True), (4, True)]:
        peer.answer(host, cs, takes=takes)
    assert [status & 3 for _, status, _ in seam.frames[1:]] == [1, 2, 3]
    assert peer.got.startswith(SID)
    for _ in range(len(PROPOSAL)):
        if not host.arq._buffer_raw:
            break
        peer.answer(host, 1 if host.arq.tx_seq % 2 == 0 else 2)
    assert peer.got == PROPOSAL and peer.got.count(SID) == 1


def test_ve3kpg_recorded_cs4_uses_the_same_answer_window_and_word(arm):
    ve = json.loads((FIXTURE / 've3kpg-0924-0033.json').read_text())
    cap = evidence.CAPTURES / ve['arm']
    for name, row in ve['files'].items():
        path = evidence.require_input(cap / name)
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == row['sha256'], name
        if name.startswith('tx_'):
            meta = json.loads(path.read_text())
            assert all(meta[k] == v for k, v in row.items() if k != 'sha256')
    audio = rxfront.load_wav(str(cap / 'stream.flac'))
    pulse = rx._pulse(480)
    delay = (len(pulse) - 1) // 2
    for k, sample in zip((34, 35, 37), ve['cs4_samples']):
        row = ve['files'][f'tx_{k}.json']
        assert row['ptt_up'] < row['audio_start'] < row['audio_end'] < sample
        assert 0.09 < (sample - row['audio_end']) / 48000 < 0.15
        crop = audio[sample - 6000:sample + 15000]
        event = rxfront.SyncedRx().control_signal_tracked(crop, 6000, details=False)
        assert event is not None and event.cs == arq.CS_SPEED_UP
        if k == 35:
            for tone, shift in zip(rx.HEADER_TONES, ve['tx35_cs4_carrier_phase_offsets']):
                baseband = rx._baseband(crop, tone, 48000, pulse)
                symbols = baseband[6000 + shift + delay + np.arange(21) * 480]
                bits = (np.real(symbols[1:] * symbols[:-1].conj()) < 0).astype(np.uint8)
                assert rx.nearest_control_signal(bits) == (arq.CS_SPEED_UP, 0)
                assert sum(int(b) << i for i, b in enumerate(bits)) == 0x95ac5
    for k, sample in zip((arm['first'], arm['lost']), arm['cs4_samples']):
        row = arm['files'][f'tx_{k}.json']
        assert row['ptt_up'] < row['audio_start'] < row['audio_end'] < sample
        assert 0.09 < (sample - row['audio_end']) / 48000 < 0.15


@pytest.mark.parametrize('bearing', [False, True])
def test_first_normal_refusal_spends_no_retry(arm, bearing):
    host, seam, wire = seed(arm)
    a = host.arq
    if not bearing:
        a._inflight.payload = b''
        a._buffer_raw -= 3
    seam.refuse = True
    control(host, 4)
    assert a.tx_seq == 1 and a._inflight.payload == wire[3:26]
    assert a._inflight.retries == a._inflight.sent_at_level == 0
    seam.refuse = False
    host.tick()
    assert a.tx_seq == 1 and a._inflight.retries == 0
    assert a._inflight.sent_at_level == 1


def blocks(stream):
    pos = stream.index(b'\0', stream.index(b'\0', 2) + 1) + 1
    sizes, blob = [], bytearray()
    while stream[pos] == 2:
        size = stream[pos + 1] or 256
        sizes.append(size)
        blob += stream[pos + 2:pos + 2 + size]
        pos += 2 + size
    assert stream[pos] == 4 and pos + 2 == len(stream)
    assert (sum(blob) + stream[pos + 1]) % 256 == 0
    return sizes, bytes(blob)


def test_recorded_complete_b2f_and_exact_subject_deletion(arm, decoded):
    """Reconstruct local TX bytes and the effect of deleting one field.

    This is not a recording of bytes delivered by the gateway.
    """
    expected = arm['_expected']
    original = arm['_original']
    rendered = Message.parse(original).render()
    generated = bytearray()
    B2FSession._send_body(None, _Outbound(Message.parse(original)), 0, generated)
    assert generated == expected
    wire = b''.join(decoded[k] for k in arm['locally_settled'])
    assert wire == compress.transparent(expected)
    assert len(expected) == 500 and len(wire) == (506 if arm['lost'] == 71 else 512)
    assert compress.Supervisor().feed(wire) == expected
    missing_wire = b''.join(decoded[k] for k in arm['locally_settled'] if k != arm['lost'])
    damaged = compress.Supervisor().feed(missing_wire)
    assert damaged == expected[:3] + expected[26:]
    assert damaged[2:damaged.index(b'\0', 2)] == (
        b'WMSU 2026-10-03 0533z' if arm['lost'] == 71 else b'W8EOC 2026-10-02 1637z')
    sizes, blob = blocks(expected)
    damaged_sizes, damaged_blob = blocks(damaged)
    assert sizes == damaged_sizes == [250, 195 if arm["lost"] == 71 else 194]
    assert len(blob) == (445 if arm["lost"] == 71 else 444)
    assert blob == damaged_blob
    assert decompress(blob, expected_size=len(rendered)) == rendered
    if arm['lost'] == 71:
        assert len(rendered) == 598
        assert hashlib.sha256(blob).hexdigest() == '4d5b5b5af1c82db455f9c0e0958c07c148d5f4a2ccc95c38efe711077e4df497'
    assert hashlib.sha256(Message.parse(rendered).subject.encode()).hexdigest() == arm['subject_sha256']


def test_c2_fallback_suffix_cs5_cs1_and_cs3_are_conserved(arm):
    if arm['lost'] != 71:
        return
    host, seam, wire = seed(arm)
    a = host.arq
    a._sl, a._peer_sl, a._established_sl = 3, 3, 1
    a._inflight = arq._Packet(2, wire[26:85], 3, trial_from=1)
    a._outbuf = bytearray(wire[85:])
    a._buffer_raw = len(wire) - 26
    a._next_seq = 3
    a._sent(59)
    pending = wire[26:]
    for i in range(6):
        control(host, 2)
        assert a._buffer_raw == len(pending)
        assert a._inflight.payload + bytes(a._outbuf) == pending
        if i >= 1:
            assert a.speed_level == 1 and a._inflight.payload == b'MSU 2'
            assert bytes(a._outbuf).startswith(wire[31:85])
    control(host, 5)
    assert a.speed_level == 2 and a.tx_seq == 2 and a._buffer_raw == len(pending)
    control(host, 1)
    assert a._buffer_raw == len(pending) - 5 and a.tx_seq == 3
    assert a._inflight.payload == wire[31:54]
    control(host, 3)
    assert a._buffer_raw == len(pending) - 28 and a.role == arq.IRS
    a._breakin_pending = a._breakin_armed = True
    a._start_next_packet(breakin=True)
    assert a._inflight.breakin and a.tx_seq == 0
    assert a._inflight.payload + bytes(a._outbuf) == wire[54:]


def test_operator_1003_synthetic_matching_ack_closure_of_recorded_bytes(arm):
    """Synthetic ACKs prove local byte conservation, not gateway delivery.

    Operator decision 2026-10-03: a second CS4 alone does not credit the pending field.
    """
    host, seam, wire = seed(arm)
    a = host.arq
    control(host, 4)
    pending, suffix = a._inflight, bytes(a._outbuf)
    frames = tuple(seam.frames)
    for cs in (4, 4, 1, 3, 5, 6):
        control(host, cs)
        assert a._inflight is pending and a.tx_seq == 1
        assert pending.payload == wire[3:26] and bytes(a._outbuf) == suffix
        assert a._buffer_raw == len(wire) - 3
        assert tuple(seam.frames) == frames
    control(host, 2)
    assert a._mail_cs4_wait is None and a.tx_seq == 2
    assert a._buffer_raw == len(wire) - 26
    delivered = wire[:26]
    while a._buffer_raw:
        packet = a._inflight
        delivered += packet.payload
        control(host, 1 if packet.seq % 2 == 0 else 2)
    assert delivered == wire and delivered.count(wire[3:26]) == 1
    expected = compress.Supervisor().feed(delivered)
    assert expected == arm['_expected']
    _, blob = blocks(expected)
    rendered = decompress(blob, expected_size=arm['original']['bytes'])
    assert rendered == arm['_original']
    assert hashlib.sha256(Message.parse(rendered).subject.encode()).hexdigest() == arm['subject_sha256']


@pytest.mark.parametrize('accepted', [False, True])
def test_operator_1003_bounded_wait_model_retains_recorded_field_without_retransmission(arm, accepted):
    """Both hypothetical peer states: no local reissue before/after timeout.

    This does not establish remote delivery or duplicates after an operator retry.
    Operator decision 2026-10-03: a second CS4 alone does not credit the pending field.
    """
    host, seam, wire = seed(arm)
    a = host.arq
    control(host, 4)
    packet = a._inflight
    delivered = wire[:26] if accepted else wire[:3]
    control(host, 4)
    frames = tuple(seam.frames)
    for i in range(a.cfg.p3_mail_cs4_wait_ticks - 1):
        control(host, (4, 1, 3, 5, 6)[i % 5])
        a._rx_this_cycle = False
        host.tick()
        assert a.state == arq.State.CONNECTED
        assert a._inflight is packet and tuple(seam.frames) == frames
    host.tick()
    assert a.state == arq.State.DISCONNECTED
    retained = a.transfer_unconfirmed
    assert (retained['counter'], retained['status'], retained['sl']) == (1, packet.status, 2)
    assert retained['field'] == wire[3:26]
    assert retained['field'] + retained['suffix'] == wire[3:]
    assert any('transfer unconfirmed' in line for line in host.log_lines)
    assert tuple(seam.frames) == frames
    assert delivered.count(wire[3:26]) == int(accepted)
    for cs in (1, 2, 3, 4, 5, 6):
        control(host, cs)
        host.tick()
    assert a.transfer_unconfirmed is retained and tuple(seam.frames) == frames
    a._enter_connected()
    assert not a._outbuf and a._inflight is None
    assert a.unconfirmed_transfers[-1] is retained


def test_operator_1003_ve3kpg_peer_model_three_cs4_credits_only_validating_cs2_complete_sid_once():
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam, peer = proposal_leg()
    for cs, takes in [(4, False), (4, True), (4, True)]:
        peer.answer(host, cs, takes=takes)
    assert [status & 3 for _, status, _ in seam.frames[1:]] == [1]
    assert peer.got == PROPOSAL[:23]
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == PROPOSAL
    peer.answer(host, 2)
    assert host.arq.tx_seq == 2
    peer.answer(host, 4)
    for _ in range(len(PROPOSAL)):
        if not host.arq._buffer_raw:
            break
        peer.answer(host, 1 if host.arq.tx_seq % 2 == 0 else 2)
    assert peer.got == PROPOSAL and peer.got.count(SID) == 1


@pytest.mark.parametrize('end', ['on_host_abort', 'on_host_disconnect'])
def test_operator_1003_early_end_retains_unconfirmed_field_and_suffix(arm, end):
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam, wire = seed(arm)
    control(host, 4)
    control(host, 4)
    frames = tuple(seam.frames)
    getattr(host.arq, end)()
    assert host.arq.state == arq.State.DISCONNECTED
    retained = host.arq.transfer_unconfirmed
    assert retained['field'] + retained['suffix'] == wire[3:]
    assert retained['counter'] == 1 and tuple(seam.frames) == frames


def test_operator_1003_unconfirmed_report_retains_bytes_create_only(arm, tmp_path):
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    from hfmodem.shrike.onair import _write_unconfirmed_transfer

    host, _, wire = seed(arm)
    control(host, 4)
    control(host, 4)
    host.tick(elapsed_ticks=100)
    path = _write_unconfirmed_transfer(tmp_path, host.arq)
    assert path.read_bytes() == wire[3:]
    report = json.loads((tmp_path / 'p3-transfer-unconfirmed.json').read_text())
    assert report['result'] == 'transfer unconfirmed' and report['counter'] == 1
    assert report['field']['bytes'] == 23 and report['suffix']['offset'] == 23
    assert report['sha256'] == hashlib.sha256(wire[3:]).hexdigest()
    with pytest.raises(FileExistsError):
        _write_unconfirmed_transfer(tmp_path, host.arq)
    assert path.read_bytes() == wire[3:]


@pytest.mark.parametrize('policy', ['current', 'stock'])
@pytest.mark.parametrize('gear_hold', [False, True])
def test_operator_1003_default_mail_guard_overrides_speed_and_gear_policies(policy, gear_hold):
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    from hfmodem.shrike.onair import _mail_send_callback

    host, seam = sender(enabled=gear_hold)
    _mail_send_callback(host)
    a = host.arq
    a.cfg.p3_speed_policy = policy
    assert a.cfg.p3_mail_cs4_confirm and host.p3_mail_active
    control(host, 4)
    pending = a._inflight
    frames = tuple(seam.frames)
    lines = traffic.TrafficLog().control('RX', 3, a)
    assert 'UNCONFIRMED seq=1 / WAIT CS1/CS2' in lines[0]
    control(host, 4)
    control(host, 6)
    host.tick()
    assert tuple(seam.frames) == frames and a._inflight is pending
    control(host, 2)
    assert a.tx_seq == 2 and a._mail_cs4_wait is None


def test_operator_1003_data_packet_cannot_release_an_ambiguous_field():
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam = sender()
    host.p3_mail_active = True
    control(host, 4)
    pending = host.arq._inflight
    control(host, 4)
    host.arq.on_rx_packet(1, b'peer', 0, True, breakin=True, protocol=spec.Protocol.PACTOR3)
    assert host.arq._inflight is pending and host.arq.role == arq.ISS
    host.protocol = spec.Protocol.PACTOR2
    host.arq.on_rx_cs(arq.CS_ACK)
    assert host.arq._inflight is pending and host.arq._mail_cs4_wait is not None


@pytest.mark.parametrize('ticks', [0, -1, 0.5, True])
def test_operator_1003_wait_bound_must_be_positive_integer(ticks):
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    with pytest.raises(ValueError, match='p3_mail_cs4_wait_ticks'):
        arq.ArqConfig(p3_mail_cs4_wait_ticks=ticks)


def test_operator_1003_host_ignores_lower_mode_ack_and_crc_valid_changeover(arm):
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam, wire = seed(arm)
    control(host, 4)
    packet = host.arq._inflight
    control(host, 4)
    frames = tuple(seam.frames)
    for protocol in (spec.Protocol.PACTOR1, spec.Protocol.PACTOR2):
        host.on_rx_event(rxfront.Event(0, 'cs', '', protocol=protocol, cs=1))
    host.on_rx_event(rxfront.Event(0, 'packet', '', protocol=spec.Protocol.PACTOR3,
                                  packet=(1, b'peer', 0, True), breakin=True))
    assert host.protocol == spec.Protocol.PACTOR3 and host.arq.role == arq.ISS
    assert host.arq._inflight is packet and tuple(seam.frames) == frames
    assert packet.payload + bytes(host.arq._outbuf) == wire[3:]
    control(host, 2)
    assert host.arq.tx_seq == 2 and host.arq.transfer_unconfirmed is None


def test_operator_1003_mail_guard_resets_for_next_contact_and_send_reactivates():
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    from hfmodem.shrike.onair import _mail_send_callback

    host, _ = sender()
    send = _mail_send_callback(host)
    host.arq.on_host_abort()
    assert not host.p3_mail_active
    host.arq._enter_connected()
    send(b'next mail')
    assert host.p3_mail_active and host.arq.cfg.p3_mail_cs4_confirm


def test_operator_1003_attached_mail_client_uses_default_guard():
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    from hfmodem.winlink import MailClient

    host, seam = sender()
    host.app = MailClient(B2FSession('W9SSJ', target='K7ABC'), host.arq.on_host_data)
    assert not host.p3_mail_active
    control(host, 4)
    packet, frames = host.arq._inflight, tuple(seam.frames)
    control(host, 4)
    assert host.arq._inflight is packet and tuple(seam.frames) == frames
    assert host.arq._mail_cs4_wait == 0


def test_operator_1003_ve3kpg_arm_validates_c1_before_c2_cs4_and_c3_repeat():
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam, peer = proposal_leg()
    assert host.arq._buffer_raw == len(host.arq._inflight.payload) + len(host.arq._outbuf)

    def send_long_packet(sl, payload, status):
        seam.frames.append((sl, status, payload))
        return len(payload)

    seam.send_long_packet = send_long_packet
    peer.answer(host, 4, takes=False)
    peer.answer(host, 4)
    held = tuple(seam.frames)
    peer.answer(host, 4)
    peer.answer(host, 1, takes=False)
    peer.answer(host, 6)
    assert tuple(seam.frames) == held and host.arq.tx_seq == 1
    peer.answer(host, 2)
    peer.answer(host, 4)
    assert host.arq.tx_seq == 3
    c3 = seam.frames[-1]
    suffix = bytes(host.arq._outbuf)
    peer.answer(host, 1, takes=False)
    assert host.arq.tx_seq == 3 and host.arq.speed_level == 2
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == c3[2] + suffix
    peer.answer(host, 6)
    for _ in range(12):
        if not host.arq._buffer_raw:
            break
        peer.answer(host, 1 if host.arq.tx_seq % 2 == 0 else 2)
    assert peer.got == PROPOSAL and peer.got.count(SID) == 1


def test_operator_1003_refused_field_cannot_be_validated_or_reissued():
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    seam.refuse = True
    control(host, 4)
    packet = host.arq._inflight
    outstanding = packet.payload + bytes(host.arq._outbuf)
    assert packet.sent_at_level == 0
    control(host, 4)
    control(host, 2)
    assert host.arq._inflight is packet and host.arq._mail_cs4_wait == 0
    assert traffic.TrafficLog.control_meaning('RX', 1, host.arq).startswith('UNCONFIRMED ')
    seam.refuse = False
    host.tick(elapsed_ticks=host.arq.cfg.p3_mail_cs4_wait_ticks)
    assert host.arq.state == arq.State.DISCONNECTED and not seam.frames
    retained = host.arq.transfer_unconfirmed
    assert retained['counter'] == packet.seq
    assert retained['field'] + retained['suffix'] == outstanding


@pytest.mark.parametrize('mail, policy', [(False, True), (True, False)])
def test_operator_1003_guard_scope_preserves_explicit_optout_and_nonmail(mail, policy):
    """Operator decision 2026-10-03: a second CS4 alone does not credit the pending field."""
    host, seam = sender(enabled=False)
    host.p3_mail_active = mail
    host.arq.cfg.p3_mail_cs4_confirm = policy
    control(host, 4)
    control(host, 4)
    assert host.arq.tx_seq == 2 and host.arq._mail_cs4_wait is None
    assert [status & 3 for _, status, _ in seam.frames] == [1, 2]


def exchange(host, frame):
    replies = tuple(hostmode.Decoder('master').feed(host.feed(frame)))
    assert len(replies) == 1
    return replies[0]


@pytest.mark.parametrize('end', ['timeout', 'D', 'DD'])
@pytest.mark.parametrize('gear_hold', [False, True])
def test_operator_1003_external_hostmode_no_credit_failure_event_and_retained_history(end, gear_hold):
    """Opaque external writes use the mail policy; the first terminal poll fails."""
    host, seam = sender(enabled=gear_hold, payload=b';FW', hostmode=True)
    ch = host.ptchn
    reply = exchange(host, hostmode.command(ch, 'G', ignore_counter=True))
    assert reply.code == hostmode.LINK and b'CONNECTED to' in reply.data
    assert host.app is None and not host.p3_mail_active
    a = host.arq
    first, suffix = b';FW', bytes(range(32, 127))
    a._inflight.payload = first
    a._outbuf.clear()
    a._buffer_raw = len(first)
    host.buffer(len(first))
    reply = exchange(host, hostmode.data(ch, suffix, ignore_counter=True))
    assert reply.code == hostmode.OK and host.p3_host_data_active
    control(host, 4)
    packet = a._inflight
    held = tuple(seam.frames)
    assert host.sent_total == len(first)
    control(host, 4)
    control(host, 1)  # Wrong parity must not settle c1.
    assert a._mail_cs4_wait == 0 and a._inflight is packet
    assert host._txbuf == len(suffix) and host.sent_total == len(first)
    if end == 'timeout':
        host.tick(elapsed_ticks=a.cfg.p3_mail_cs4_wait_ticks)
    else:
        assert exchange(host, hostmode.command(ch, end, ignore_counter=True)).code == hostmode.OK
    assert a.state == arq.State.DISCONNECTED and tuple(seam.frames) == held
    retained = a.transfer_unconfirmed
    assert retained['field'] + retained['suffix'] == suffix
    assert a.unconfirmed_transfers[-1] is retained
    # A queue cleared for teardown is not a peer acknowledgement. The bytes
    # survive in history rather than being queued for an automatic retry.
    assert host.sent_total == len(first) and host._txbuf == 0
    reply = exchange(host, hostmode.command(ch, 'G', ignore_counter=True))
    assert reply.code == hostmode.LINK
    assert reply.data.startswith(f'({ch}) LINK FAILURE with K0NTS: transfer unconfirmed;'.encode())
    assert f"field {len(retained['field'])}B and suffix {len(retained['suffix'])}B".encode() in reply.data
    assert b'retained in modem history; not queued for retry' in reply.data
    assert b'DISCONNECTED' not in reply.data
    assert exchange(host, hostmode.command(ch, 'G', ignore_counter=True)).code == hostmode.OK
    assert not host.p3_host_data_active
    a._enter_connected()
    assert a.transfer_unconfirmed is None and a.unconfirmed_transfers[-1] is retained
    assert not a._outbuf and a._inflight is None and tuple(seam.frames) == held


def test_operator_1003_external_hostmode_explicit_optout():
    host, seam = sender(enabled=False)
    host.feed(b'\x1bJHOST4\r')
    assert exchange(host, hostmode.data(host.ptchn, b'opaque host bytes', ignore_counter=True)).code == hostmode.OK
    host.arq.cfg.p3_mail_cs4_confirm = False
    control(host, 4)
    control(host, 4)
    assert host.arq.tx_seq == 2 and host.arq._mail_cs4_wait is None


def test_operator_1003_recorded_two_cs4_answers_leave_first_normal_field_unconfirmed(arm):
    """Only measured controls up to the branch point; no invented peer ACKs."""
    host, seam, wire = seed(arm)
    audio = rxfront.load_wav(str(evidence.CAPTURES / arm['arm'] / 'stream.flac'))
    for index, sample in enumerate(arm['cs4_samples']):
        crop = audio[sample - 6000:sample + 15000]
        event = rxfront.SyncedRx().control_signal_tracked(crop, 6000, details=False)
        assert event is not None and event.cs == arq.CS_SPEED_UP
        host.on_rx_event(event)
        assert host.arq.tx_seq == 1
        assert host.arq._buffer_raw == len(wire) - 3
        assert host.arq._inflight.payload == wire[3:26]
        assert host.arq._inflight.payload + bytes(host.arq._outbuf) == wire[3:]
        assert len(seam.frames) == 1
        assert (host.arq._mail_cs4_wait is not None) == bool(index)
    # Later recorded answers addressed the old transmitter's different state;
    # they cannot be relabelled as delivery evidence for this counterfactual.
