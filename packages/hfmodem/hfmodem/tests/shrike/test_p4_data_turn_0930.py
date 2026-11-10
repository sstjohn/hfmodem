# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Data turn pins: k0003 FS/hand-back and the 2327 counted-body continuation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hfmodem.shrike import p4oracle, p4robustrx
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, machine, body, word
from hfmodem.tests.shrike.test_p4_bare_y27_0929 import k0003
from hfmodem.tests.shrike.test_p4_normal_short_0928 import receiver, read_copy

T = round(.920 * FS)
CYCLE = round(1.25 * FS)


def receive(m, at, data=b'', status=0x40, responds_to=None):
    m.observe(word('Y27', at, extent=.8, responds_to=responds_to), now=at+round(.3*FS))
    m.observe(replace(body(at, data, status), responds_to=responds_to), now=at+round(.81*FS))
    return m.decide(at+T-m.lead)


def sending(payload=b'F> 3B\r', **kw):
    m, a = machine(data='body', ladder='none', **kw)
    a.on_host_data(payload)
    m._data(0)
    m.keyed()
    return m, a


@pytest.mark.parametrize('role', ['irs', 'iss'])
def test_k0003_fs_status_40_is_delivered_once_and_acked_even_during_a_request(role):
    m, a = machine(data='body', ladder='none', role=role)
    m.first_body = False
    received = []
    m.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    if role == 'irs':
        m._body(0, 0x40, State.IRS_BREAKIN)
        m.keyed()
    for at in (FS, FS+CYCLE):
        p = receive(m, at, b'FS YY\r')
        assert p.kind == Kind.WORD and p.word == 'V27'
        m.keyed()
        assert m.state == State.IRS_ANSWER
    assert received == [b'FS YY\r']


@pytest.mark.parametrize('status', [0, 0x40])
def test_y27_data_reply_commits_last_field_before_delivering_fs(status):
    m, a = sending()
    delivered = []
    def deliver(data):
        assert a._buffer_raw == 0 and m.outbox.inflight == b''
        delivered.append(data)
        a.on_host_data(b'message bytes')
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *a: None)
    p = receive(m, T, b'FS Y\r', status, responds_to=0)
    assert p.kind == Kind.WORD
    assert m.counter == 1 and m._data_packet is None
    m.keyed()
    assert delivered == [b'FS Y\r'] and bytes(a._outbuf) == b'message bytes'
    repeat = receive(m, T+CYCLE, b'FS Y\r', status)
    if status == 0:
        assert repeat.kind == Kind.BODY and repeat.state == State.IRS_BREAKIN
    else:
        assert repeat.word == p.word
    assert delivered == [b'FS Y\r']


@pytest.mark.parametrize('response', ['bare', 'idle', 'partial', 'expired', 'unrelated'])
def test_y27_without_an_associated_data_reply_never_commits_the_last_field(response):
    m, a = sending()
    if response == 'idle':
        p = receive(m, T, status=0, responds_to=0)
    elif response == 'unrelated':
        p = receive(m, T, b'FS Y\r', responds_to=-CYCLE)
    else:
        early = word('Y27', T, partial=response in ('partial', 'expired'),
                     extent=.245 if response == 'bare' else None, responds_to=0)
        m.observe(early)
        if response == 'expired':
            m.observe(replace(early, outcome=Registered.EXPIRED, replaced=early.observation))
        p = m.decide(T+T-m.lead)
    assert a._buffer_raw == len(b'F> 3B\r') and m.counter == 0
    if p:
        m.keyed()
    m.link_down('test end')
    assert bytes(a._outbuf) == b'F> 3B\r' and not m.outbox.inflight


@pytest.mark.parametrize('breakin', ['body', 'y27'])
def test_empty_outbox_keys_counted_normal_idle_until_fixed_hold_deadline(breakin):
    m, a = sending(breakin=breakin, hold_cycles=16)
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    deadline = m._hold_until
    for k in range(1, 17):
        now = k*CYCLE-m.lead
        m.silence(now)
        p = m.decide(now)
        assert (p.kind, p.payload, p.status, p.state) == (Kind.NORMAL, b'', 1, State.ISS_HOLD)
        assert p.stream_sample == k*CYCLE
        m.keyed()
        assert m._hold_until == deadline
    p = m.decide(deadline)
    assert p.status == 0x20 and p.state == State.ISS_CLOSE
    assert a._buffer_raw == 0 and m.counter == 1


@pytest.mark.parametrize('bare', [False, True])
def test_peer_breakin_during_empty_outbox_hold_yields_then_acks_fs(bare):
    m, a = sending()
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    m.keyed()
    at = m.last.stream_sample+T
    if bare:
        m.observe(word('Y27', at, extent=.245, responds_to=m.last.stream_sample))
        p = m.decide(at+T-m.lead)
    else:
        p = receive(m, at, responds_to=m.last.stream_sample)
    assert p.status == 0x40 and not p.payload and p.state == State.ISS_GRANT
    m.keyed()
    p = receive(m, p.stream_sample+T, b'FS Y\r')
    assert p.kind == Kind.WORD
    m.keyed()
    assert m.state == State.IRS_ANSWER and m._hold_until is None
    a.on_host_data(b'next turn')
    at = p.stream_sample+round(.330*FS)
    receive(m, at)
    request = m.keyed()
    p = receive(m, request.stream_sample+T, responds_to=request.stream_sample)
    assert p.payload == b'next turn' and p.status == 0


def test_k0003_tape_counter_evidence_and_fs_body_is_data():
    import numpy as np
    # N9VC before the hand-back: its normal c1 (the SID body before it is c0).
    audio = k0003(1458.06, 1459.16)
    audio = np.r_[np.zeros(FS), audio]
    r = receiver(round(1.1*FS), 3.)
    frames = [reg.observation.frame for reg in read_copy(r, audio, round(1.1*FS)) if reg.observation.frame]
    assert frames and frames[-1].status.byte == 0x41
    for onset, status, prefix in [(1468.459,0x43,b'T\rFC EM '),
                                  (1477.461,0x42,b'FS YY\r'),
                                  (1491.503,0x61,b'\x01\x15GENEVA'),
                                  (1529.906,0x42,b'FF\r')]:
        audio = k0003(onset-.1, onset+3.75)
        _, symbols, _ = p4oracle.placed_despread(audio, FS, .1, 4.8, count=373)
        result = p4robustrx.decode_symbols(symbols)
        assert result and result.frame.status.byte == status
        assert result.frame.data.startswith(prefix)
    audio = k0003(1473.608, 1474.708)
    heads = p4oracle.word_scan(audio, FS)
    frames = [p4oracle.body_after(audio, FS, 0, h) for h in heads if h.word == 'Y27']
    frame = next(f for f in frames if f is not None)
    assert frame.status.byte == 0x40 and frame.data.rstrip(b'\x1e') == b'FS YY\r'
    m, a = machine(role='irs', data='body', ladder='none')
    m.first_body = False
    p = receive(m, FS, frame.data.rstrip(b'\x1e'), frame.status.byte)
    assert p.word == 'V27' and p.kind == Kind.WORD


@pytest.mark.parametrize('fs_reply', ['last-body', 'hold', 'normal-grant'])
def test_2327_recording_then_real_b2f_round_trip(monkeypatch, fs_reply):
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    import numpy as np
    from hfmodem.shrike import arq, compress, onair, p4breakintx, p4shorttx
    from hfmodem.tests.shrike import p4_stock_peer as stock
    from hfmodem.shrike.ptc import PtcHost, SimPeer
    from hfmodem.shrike.p4turn import P4Turn
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    from hfmodem.tests.shrike.test_p4_mail_0927 import options
    from hfmodem.tests.shrike.test_p4_breakin_v27_0929 import X27_2327, GRID
    from hfmodem.winlink import B2FSession, MailClient, compose

    root = Path(__file__).resolve().parents[5]
    capture = root/'captures/onair-0929-2327'
    if not (capture/'stream.flac').exists():
        pytest.skip('local 2327 recording absent')
    recorded = onair.session.load_wav(str(capture/'stream.flac'), FS)
    probe = json.loads((capture/'p4-probe.json').read_text())
    cutoff = X27_2327+round(.300*FS)
    audio = np.zeros(300*FS)
    audio[:cutoff] = recorded[:cutoff]
    for e in probe['emissions']:
        audio[e['audio_start']:e['audio_end']] = 0
    stream = Stream(audio)
    stream.pos = probe['emissions'][0]['audio_end']
    tx = Tx(stream)
    tx.settle = .044
    tx.p4_probe.emissions = probe['emissions'][:1]
    tx.p4_probe.handoff_audio = stream.pos-FS, audio[stream.pos-FS:stream.pos].copy()
    tx.p4_link_options = options(monkeypatch, '--mail-fetch', '--mail-wait-greeting',
        '--p4-ladder', 'none', '--p4-hold-cycles', '16', '--p4-link-timeout', '270')
    host = PtcHost(SimPeer(), mycall='W9SSJ')
    host.arq.state = arq.State.CONNECTED
    host.arq.role = arq.IRS
    message = compose('W9SSJ', 'SMTP:offline@example.net', 'P4 round trip',
        b'First message across a P4 data turn.\r\n', mid='P4ROUND0930',
        date=datetime(2026, 9, 30, tzinfo=timezone.utc))
    session = B2FSession('W9SSJ', target='WS8EOC', password='offline-test',
                         client_sid='Pat-1.0.0', comment=False, outbox=[message])
    mail = MailClient(session, host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    tx.p4_link_host, tx.p4_link_mail = host, mail
    # 2327 read the first grant after its decision, then answered its repeat.
    # Preserve that recorded deadline independently of decoder CPU speed.
    read_window = tx.sessrx.p4.read_window
    late_until = 0
    stream.sample_now = lambda: max(stream.pos, late_until)
    def recorded_delay(*args, **kwargs):
        nonlocal late_until
        records = read_window(*args, **kwargs)
        for reg in records:
            obs = reg.observation
            if obs.kind == Kind.BODY and abs(obs.stream_sample-5716972) < FS//100:
                late_until = obs.stream_sample+T-round(.044*FS)+1
        return records
    monkeypatch.setattr(tx.sessrx.p4, 'read_window', recorded_delay)
    emitted, settled, received, waves = [], [], [], []
    original_settle, original_deliver = host.arq.settle_bytes, host.deliver
    def settle(data):
        settled.append(data)
        original_settle(data)
        assert host._txbuf == host.arq._buffer_raw == len(host.arq._outbuf)+len(host.arq._p4_lent)
    def deliver(data):
        received.append(data)
        original_deliver(data)
    monkeypatch.setattr(host.arq, 'settle_bytes', settle)
    monkeypatch.setattr(host, 'deliver', deliver)
    transmit = tx._tx
    def transmit_audio(wave, label):
        waves.append(wave.copy())
        transmit(wave, label)
        stream.pos = tx.tx_end
    tx._tx = transmit_audio
    # The captured Y27 is near -1.85 Hz: keep its body carrier near -3.25 Hz.
    peer_cfo = -3.25
    phase = 'proposal'
    idle_count = 0
    idle_acks = 0
    keyed = P4Turn.keyed
    def place(at, wave):
        assert at >= cutoff
        audio[at:at+len(wave)] = wave
    def peer(turn, at=None):
        nonlocal phase, idle_count, idle_acks
        p = keyed(turn, at)
        emitted.append(p)
        if p.stream_sample < X27_2327:
            return p
        answer = p.stream_sample+T
        if p.kind in (Kind.BODY, Kind.NORMAL) and p.payload:
            np.testing.assert_array_equal(waves[-1], (p4breakintx.changeover_packet if p.kind == Kind.BODY else p4shorttx.short_packet)(p.payload, p.status))
            last = not host.arq._outbuf
            if phase == 'proposal' and last and fs_reply == 'last-body':
                phase = 'fs'
                place(answer, stock.body(b'FS Y\r', 0x40, cfo=peer_cfo))
            else:
                place(answer, stock.word('V27' if p.status & 1 else 'X27', cfo=peer_cfo))
                if last:
                    phase = {'proposal': 'wait-fs', 'message': 'wait-ff', 'closing': 'done'}[phase]
        elif p.kind == Kind.NORMAL or p.kind == Kind.BODY and p.status == 0x40:
            if p.kind == Kind.NORMAL:
                idle_acks += 1
            if p.state == State.IRS_BREAKIN and phase in ('fs', 'ff'):
                phase = 'message' if phase == 'fs' else 'closing'
                place(answer, stock.body(b'', 0x40, cfo=peer_cfo))
            elif phase == 'wait-fs':
                idle_count += 1
                if fs_reply == 'normal-grant' and p.kind == Kind.NORMAL:
                    phase = 'grant-fs'
                    place(answer, stock.body(b'', 0x40, cfo=peer_cfo))
                elif idle_count == 3:
                    phase = 'fs'
                    place(answer, stock.body(b'FS Y\r', 0x40, cfo=peer_cfo))
                else:
                    place(answer, stock.word('V27' if p.status & 1 else 'X27', cfo=peer_cfo))
            elif phase == 'grant-fs':
                phase = 'fs'
                place(answer, stock.normal(b'FS Y\r', 0, cfo=peer_cfo))
            elif phase == 'wait-ff':
                phase = 'ff'
                place(answer, stock.body(b'FF\r', 0x40, cfo=peer_cfo))
            elif phase in ('message', 'closing'):
                place(answer, stock.body(b'', 0x40, cfo=peer_cfo))
        elif p.kind == Kind.WORD:
            if phase in ('fs', 'ff'):
                phase = 'message' if phase == 'fs' else 'closing'
                place(p.stream_sample+round(.330*FS), stock.body(b'', 0x40, cfo=peer_cfo))
        elif p.kind == Kind.SHORT and p.value == 37:
            place(answer, stock.word('V27', cfo=peer_cfo))
        return p
    monkeypatch.setattr(P4Turn, 'keyed', peer)
    onair._run_p4_link(tx, stream, GRID)
    assert session.done and not session.failure, (session.stage, session.failure, phase)
    assert phase == 'done' and tx.p4_probe.reason == 'close acknowledged'
    assert session.sent_mids == [message.mid]
    assert host._txbuf == host.arq._buffer_raw == 0 and not host.arq._p4_lent
    assert received.count(b'FS Y\r') == 1 and received.count(b'FF\r') == 1
    plain = compress.Supervisor().feed(b''.join(settled))
    assert plain.count(b'F> ') == 1 and plain.count(b'FQ\r') == 1
    blob = session._out[0].blob
    title = message.subject.encode()
    expected = bytes((1, len(title)+3))+title+b'\0' + b'0\0'
    for i in range(0, len(blob), 250):
        block = blob[i:i+250]
        expected += bytes((2, len(block)))+block
    expected += bytes((4, (-sum(blob)) & 255))
    assert expected in plain
    assert any(r.get('word') == 'X27' and not r['provisional']
               and abs(r['onset_sample']-X27_2327) < FS//50 for r in tx.p4_probe.heard)
    assert any(r.get('status') == (0 if fs_reply == 'normal-grant' else 0x40) and bytes.fromhex(r.get('data','')).startswith(b'FS Y\r')
               for r in tx.p4_probe.heard)
    assert next(p.payload for p in emitted if p.payload) == b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;'
    if fs_reply == 'hold':
        assert idle_count == 3
    elif fs_reply == 'last-body':
        assert any(e['event'] == 'data-reply-ack' for e in tx.p4_probe.link_events)


def test_empty_grant_replies_do_not_extend_the_hold_backstop():
    m, a = sending(hold_cycles=2)
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    idle = m.keyed()
    at = idle.stream_sample+T
    deadline = m._hold_until
    p = receive(m, at, responds_to=idle.stream_sample)
    assert p.state == State.ISS_GRANT
    m.keyed()
    p = receive(m, p.stream_sample+T)
    assert p.state == State.ISS_CLOSE and p.status == 0x20
    assert m._hold_until == deadline


def test_late_close_word_after_a37_repeat_closes_the_link():
    m, a = machine(data='body', ladder='none', app_done=True)
    m._data(0)
    m.keyed()
    m.silence(CYCLE-m.lead)
    short = m.keyed()
    early = word('V27', short.stream_sample+T, partial=True, responds_to=short.stream_sample)
    m.observe(early)
    m.silence(short.stream_sample+CYCLE-m.lead)
    repeat = m.keyed()
    m.observe(word('V27', early.observation.stream_sample, outcome=Registered.CONFIRMED,
                   replaced=early.observation, answered=repeat.stream_sample,
                   responds_to=short.stream_sample), now=repeat.stream_sample+100)
    assert m.state == State.DOWN and m.reason == 'close acknowledged'


def test_checked_fs_ends_hold_even_when_its_ack_must_be_deferred():
    m, a = sending(hold_cycles=2)
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    m.keyed()
    deadline = m._hold_until
    at = deadline-T-round(.010*FS)
    m.observe(word('Y27', at, extent=.8), now=at+round(.3*FS))
    m.observe(body(at, b'FS Y\r', 0x40), now=deadline-1)
    assert m._deferred is not None and m._hold_until is None
    assert m.decide(deadline) is None
    p = m.decide(at+T+CYCLE-m.lead)
    assert p.kind == Kind.WORD and p.word in ('V27', 'X27')
