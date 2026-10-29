# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K4SPB source resegmentation and damaged-header chirps, 2026-10-27."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from hfmodem.shrike import p4chirp, p4chirprx, p4frame, p4robusttx
from hfmodem.shrike.p4credit import ReceiptContext
from hfmodem.shrike.p4host import P4Host
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.tests.shrike.test_p4turn import machine

FIXTURE = Path(__file__).parent/'fixtures/k4spb-1027'
EVIDENCE = json.loads((FIXTURE/'evidence.json').read_text())
FS = 48000


def frame(status, data):
    return p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                data_bytes=len(data))


def clip(index):
    with np.load(FIXTURE/'chirps.npz') as recording:
        return (resample_poly(recording['audio'][index].astype(float)/32768, 6, 1),
                int(recording['nominal_samples'][index])*6)


def test_recorded_greeting_source_is_delivered_once_across_all_formats():
    host = P4Host()
    delivered = []
    for row in EVIDENCE['frames']:
        kind = Kind(row['kind'])
        payload = host.accept(frame(row['status'], bytes.fromhex(row['data'])),
                              receipt_context=ReceiptContext(None, kind))
        if payload:
            delivered.append(payload)
    assert b''.join(delivered) == (
        b'RMS Trimode 1.4.4.1 Welcome to the K4SPB Hybrid RMS in Memphis, TN\r\n'
        b'W9SSJ has 180 daily minutes remaining with K4SPB (EM55CD)\r'
        b'Minimum allowed sustained transfer speed is 25 bytes/minute\r'
        b'{SFI = 91 On 2026-10-27 06:00 UTC}\r\n'
        b'[WL2K-5.0-B2FWIHJM$]\r;PQ: 82204462\r')


@pytest.mark.parametrize('index', range(6))
def test_recorded_sweep_protects_short_answer_and_recovers_crc(index):
    audio, nominal = clip(index)
    origin = 2*FS
    at = origin+nominal
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .33, 1.25, EVIDENCE['cfo_hz'])
    rx.expected = True
    m, _ = machine(role='irs', ladder='none', data='body', prekey_lead=.04)
    prior = at-round(1.25*FS)
    control = Observation(Kind.SHORT, FS, 0, prior, prior, .76, EVIDENCE['cfo_hz'],
                          .9, value=7, header='B', placed=True)
    m.observe(RegisteredObservation(control, Registered.NEW), now=prior+round(.81*FS))
    assert m.decide(prior+round(.88*FS)) is not None
    m.keyed()
    records = rx.read_partial(audio[:nominal+round(.81*FS)], origin, deadline=at+round(.81*FS))
    heads = [r for r in records if r.observation.kind == Kind.CHIRP]
    assert heads and heads[-1].observation.provisional
    assert rx.clock.cycle_s == 3.75
    for reg in records:
        m.observe(reg, now=at+round(.81*FS))
    early_due = at+round(.92*FS)
    assert m.decide(early_due-m.lead) is None
    m.silence(early_due-m.lead, answer_due=early_due)
    assert m.decide(early_due-m.lead) is None  # No stale ACK in the chirp.
    start = heads[-1].observation.stream_sample
    records = rx.read_chirp(audio, origin, onset=start)
    assert records is not None
    complete = [r for r in records if r.observation.frame is not None]
    assert complete
    got = complete[-1].observation.frame
    expected = EVIDENCE['chirps'][index]
    assert got.status.byte == expected['status']
    assert got.data.rstrip(b'\x1e') == bytes.fromhex(expected['data'])
    for reg in records:
        m.observe(reg, now=start+round(3.33*FS))
    reply = m.decide(start+round(3.4*FS)-m.lead)
    assert reply is not None and reply.stream_sample >= start+round(p4chirp.PACKET_S*FS)


@pytest.mark.parametrize('noise', [False, True])
def test_sweep_evidence_is_not_silence_or_broadband_noise(noise):
    x = np.zeros(FS) if not noise else np.random.default_rng(1027).normal(size=FS)
    assert p4chirprx.acquire_sweep(x, nominal_sample=4800) is None


@pytest.mark.parametrize('audio', [0., [[0.]], [1j], [np.nan]])
def test_sweep_rejects_invalid_audio(audio):
    with pytest.raises(ValueError, match='finite real vector'):
        p4chirprx.acquire_sweep(audio, nominal_sample=0)


def test_sweep_does_not_relax_crc_or_default_header_requirement():
    audio, nominal = clip(1)
    assert p4chirprx.acquire_header(audio, nominal_sample=nominal,
                                   frequency_offset=EVIDENCE['cfo_hz']) is None
    head = p4chirprx.acquire_sweep(audio, nominal_sample=nominal,
                                 frequency_offset=EVIDENCE['cfo_hz'])
    assert head is not None
    assert p4chirprx.decode_packet(audio, start_sample=head.start_sample,
                                   frequency_offset=EVIDENCE['cfo_hz']) is None
    audio[nominal+FS:] = 0
    assert p4chirprx.decode_packet(audio, start_sample=head.start_sample,
                                   frequency_offset=EVIDENCE['cfo_hz'], sweep_verified=True) is None


@pytest.mark.parametrize('value', [5, 7, 38, 59])
def test_robust_short_controls_do_not_claim_a_chirp_receive_span(value):
    x = np.r_[np.zeros(4800), p4robusttx.robust_short(value, 'B'), np.zeros(4800)]
    assert p4chirprx.acquire_sweep(x, nominal_sample=4800) is None


def test_extended_empty_changeover_does_not_kill_the_mail_stream():
    host = P4Host()
    empty = frame(0xc0, b'\x01\x01\x01'+b'\x1e'*200)
    assert host.accept(empty, receipt_context=ReceiptContext(None, Kind.LONG)) == b''
    assert host.failure is None
    assert host.accept(empty, receipt_context=ReceiptContext(None, Kind.LONG)) is None


def test_unmapped_control_repetitions_report_no_progress_and_reset_on_data():
    m, _ = machine(role='irs')
    for i in range(4):
        at = FS+i*60000
        obs = Observation(Kind.SHORT, FS, 0, at, at, .76, 0., .9,
                          value=5, header='B', placed=True)
        m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.81*FS))
        assert m.decide(at+round(.92*FS)-m.lead).word == 'X27'
        m.keyed()
    assert [e['copies'] for e in m.link_events if e['event'] == 'peer-control-no-progress'] == [1, 2, 3, 4]
    from hfmodem.tests.shrike.test_p4turn import body
    m.observe(body(8*FS), now=8*FS)
    assert m._unsupported_copies == []


def test_postdecision_deferrals_and_refused_keys_are_counted():
    from hfmodem.shrike.p4budget import decision_summary
    decision = dict(event='Decision', at=FS, reason='proposal', decision_at=2*FS,
                    now_sample=2*FS+100)
    events = [decision, dict(event='answer-deferred', at=FS, due=5*FS),
              dict(event='Refused', at=3*FS, reason='deadline')]
    heard = [dict(phase='link', onset_sample=FS, kind='long', status=0x63, data='00',
                  received_at_sample=2*FS+200)]
    summary = decision_summary(events, heard, FS)
    assert summary['late_crc_valid'] == summary['deferred_decisions'] == 1
    assert summary['deferred_frames'] == summary['overdue_decisions'] == 1
    assert summary['refused_transmissions'] == 1


@pytest.mark.parametrize('index', [1, 2, 3, 4])
@pytest.mark.parametrize('decode_ms', [0, 20, 150])
def test_live_driver_waits_for_recorded_damaged_header_chirp(monkeypatch, capsys, index, decode_ms):
    from hfmodem.shrike import onair
    from hfmodem.shrike.p4receiver import Emission, Role
    from hfmodem.shrike.p4turn import P4Turn, State
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    audio, nominal = clip(index)
    onset = 2*FS+nominal
    stream = Stream(np.r_[np.zeros(2*FS), audio, np.zeros(FS)])
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.key_notice = 1652
    stream.transmit = True
    stream.clamp_late = lambda at: max(0, stream.sample_now()+stream.key_notice-at)
    tx = Tx(stream)
    rx = tx.sessrx.p4
    tx.settle = .04
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_link_timeout = 4.2
    prior = onset-round(1.25*FS)
    ack = prior+round(.92*FS)
    tx.p4_probe.emissions = [dict(audio_start=ack, audio_end=ack+11621)]
    stream.pos = ack+11621
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    read_chirp = rx.read_chirp

    def timed(*args, **kwargs):
        records = read_chirp(*args, **kwargs)
        if records and any(r.observation.frame is not None for r in records):
            stream.wall = stream.sample_now()+round(decode_ms/1000*FS)
        return records

    monkeypatch.setattr(rx, 'read_chirp', timed)

    def seed(m, *args, **kwargs):
        m.state = State.IRS_ANSWER
        m._opening = False
        obs = Observation(Kind.SHORT, FS, 0, prior, prior, .76, EVIDENCE['cfo_hz'], .9,
                          value=7, header='B', placed=True)
        m.observe(RegisteredObservation(obs, Registered.NEW), now=prior+round(.81*FS))
        m.decide(ack-m.lead)
        m.keyed()
        rx.note_emission(Emission(Kind.WORD, Role.IRS, ack, 11621/FS))
        rx.clock = Clock(Expect.PACKET, onset, .33, 1.25, EVIDENCE['cfo_hz'])
        rx.expected = True

    monkeypatch.setattr(P4Turn, 'seed_opening', seed)
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0,
                        packet_n=36000, cs_n=12000, d_max_n=4800))
    capsys.readouterr()
    keys = [r for r in tx.p4_probe.link_events if r['event'] == 'Keyed']
    assert all(r['at'] >= onset+round(3.30*FS) for r in keys)
    if decode_ms < 100:
        assert keys
    else:
        assert not keys  # A missed keying deadline must not shift the ACK late.
    frames = [r for r in tx.p4_probe.heard if r.get('kind') == 'chirp' and 'data' in r]
    assert frames and frames[0]['status'] == EVIDENCE['chirps'][index]['status']
    if index == 1:
        assert b''.join(delivered) == b'CMS via K4SPB >\r'
