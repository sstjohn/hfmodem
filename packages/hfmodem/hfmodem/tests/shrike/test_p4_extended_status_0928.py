# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from copy import deepcopy
from pathlib import Path

import pytest

from hfmodem.shrike import arq, p4frame
from hfmodem.shrike.p4host import P4Host, P4StreamFailure
from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import body, machine

PREFIX = b'\x01\x01\x01'
TAIL = b'306\rCMS via WS8EOC >\r'
DATA = (PREFIX + TAIL).ljust(40, b'\x1e')
ROOT = Path(__file__).resolve().parents[5]


def frame(data=DATA, status=0x83):
    return p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                data_bytes=len(data))


@pytest.mark.parametrize('status', [0x83, 0xe3])
def test_observed_extension_delivers_only_text_and_deduplicates(status):
    adapter = P4Host()
    recorded = frame(status=status)
    assert adapter.accept(recorded) == TAIL
    assert adapter.accept(recorded) is None
    assert recorded.data == DATA
    assert adapter.last == (status, DATA)


def test_late_extension_keeps_raw_prefix_event_without_a_late_ack():
    m, _ = machine(role='irs')
    obs = replace(body(48000).observation, kind=Kind.NORMAL, frame=frame())
    m.observe(RegisteredObservation(obs, Registered.ENRICHED), now=92160)
    assert m.decide(92160) is None
    events = [e for e in m.link_events if e['event'] not in ('source-accept', 'source-replay', 'source-unclassified')]
    assert events == [dict(event='source-extension', at=48000, status=0x83, prefix='010101'),
                             dict(event='answer-deferred', at=48000, due=48000+44160+60000, kind=Kind.NORMAL)]
    assert m.adapter.last == (0x83, DATA)


@pytest.mark.parametrize('status', [3, 0x83])
def test_decided_normal_enriches_and_answers_the_accepted_frame(status):
    from types import SimpleNamespace
    m, _ = machine(role='irs')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    first = replace(body(48000).observation, kind=Kind.NORMAL, frame=None)
    full = replace(first, frame=frame(status=status))
    decision = first.stream_sample + 44160 - m.lead
    m.observe(RegisteredObservation(first, Registered.NEW), now=first.stream_sample+38880)
    assert not m.decide(decision).current_slot_accepted
    m.observe(RegisteredObservation(full, Registered.ENRICHED, replaced=first), now=decision)
    assert delivered == [TAIL if status & 0x80 else PREFIX + TAIL]
    answer = m.decide(decision)
    assert answer.purpose is None and answer.current_slot_accepted
    assert answer.ack_identity == (status, full.frame.data)
    m.keyed()
    repeat = replace(full, stream_sample=full.stream_sample+60000)
    m.observe(RegisteredObservation(repeat, Registered.NEW), now=repeat.stream_sample+38880)
    pending = [sample for sample in m.pending if sample not in m._decided]
    assert pending == [repeat.stream_sample]
    assert m._deferred is None
    proposal = m.decide(repeat.stream_sample+44160-m.lead)
    assert proposal.word == 'V27' and proposal.stream_sample == repeat.stream_sample+44160
    m.keyed()
    assert delivered == [TAIL if status & 0x80 else PREFIX + TAIL]
    assert m.decide(proposal.stream_sample) is None


@pytest.mark.parametrize('prefix', [b'', b'\x01', b'\x01\x01', b'\x01\x00',
                                    b'\x01\x02\x01\x01', b'\x02\x01\x01',
                                    b'\x01\x01\x02', b'\x01\xff\x01'])
@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('status', [0x83, 0xe3])
def test_unknown_or_truncated_extension_refuses_atomically(prefix, terminal, status):
    adapter = P4Host(terminal=terminal)
    adapter.accept(frame(b'hello\x1c', 0))
    before = adapter.last, deepcopy(vars(adapter.supervisor))
    with pytest.raises(P4StreamFailure):
        adapter.accept(frame(prefix, status))
    assert (adapter.last, vars(adapter.supervisor)) == before
    assert bool(adapter.failure) == terminal
    if not terminal:
        assert adapter.accept(frame(b'`', 1)) == b'\x00'


@pytest.mark.parametrize('status', [0x87, 0xa3, 0xc3, 0xe7])
def test_observed_prefix_does_not_authorize_other_status_modes(status):
    with pytest.raises(P4StreamFailure):
        P4Host().accept(frame(status=status))


def test_extension_without_e_stays_application_data_and_crc_is_required():
    assert P4Host().accept(frame(status=3)) == PREFIX + TAIL
    wire = bytearray(p4frame.encode_field(DATA, p4frame.Status.from_byte(0x83)))
    wire[3] ^= 1
    assert p4frame.decode_field(bytes(wire), data_bytes=40) is None


def test_normal_decode_budget_event_is_bounded_and_reports_actual_attempt(monkeypatch):
    from hfmodem.shrike import p4short, p4shorttx
    from hfmodem.shrike.p4hold import BreakinHold
    hold = BreakinHold(48000)
    hold.acked_counter = 2
    clock = [0.]
    hold.now = lambda: clock[0]
    decode = p4short.decode_scores

    hold.quick_decode_times = [1.]

    def timed(scores, **kw):
        clock[0] += .051
        return decode(scores, **kw)

    monkeypatch.setattr(p4short, 'decode_scores', timed)
    fields = 2*(1.-2*p4shorttx.coded_bits(p4frame.encode_field(DATA, frame().status)).astype(float))
    assert hold.normal(48000, 0, 0., .7, fields, None, deadline=.030) is None
    assert hold._normal_decode(.030) is None
    assert [e['reason'] for e in hold.events if e['event'] == 'normal-decode' and e['iterations'] == 8] == ['budget']
    assert hold._normal_decode(.100) == frame()
    events = [e for e in hold.events if e['event'] == 'normal-decode' and e['iterations'] == 8]
    assert [e['reason'] for e in events] == ['budget', 'accepted']
    assert events[0]['crc_valid'] is None and events[0]['allowance_ms'] == 30.
    assert events[1]['crc_valid'] and events[1]['elapsed_ms'] == pytest.approx(51.)
    assert events[1]['stage'] == 'sum' and events[1]['copies'] == 1
    assert hold._normal_decode(.100) == frame()
    assert len([e for e in hold.events if e['event'] == 'normal-decode' and e['iterations'] == 8]) == 2


@pytest.mark.parametrize('association', ['latest', 'none', 'older', 'supervisor'])
def test_empty_peer_grant_leaves_iss_grant_for_data_whatever_it_answered(association):
    m, queue = machine(role='irs', data='body')
    queue.on_host_data(b'FC test')
    m._body(0, 0x40, State.ISS_GRANT)
    m.keyed()
    m._body(60000, 0x40, State.ISS_GRANT)
    m.keyed()
    incoming = body(104160, b'\x1cA' if association == 'supervisor' else b'', 0x40)
    incoming = replace(incoming, responds_to={'latest': 60000, 'none': None,
                                              'older': 0, 'supervisor': 60000}[association])
    m.observe(incoming, now=104160)
    p = m.decide(148320-m.lead)
    assert p is not None
    if association != 'supervisor':
        assert p.payload == b'FC test' and p.state == State.ISS_DATA_ACK
        assert m.outbox.inflight == b'FC test'
        assert queue._buffer_raw == 7 and bytes(queue._p4_lent) == b'FC test'
        m.refused('deadline')
        assert m.outbox.inflight == b'FC test' and queue._buffer_raw == 7
        m.link_down('test')
        assert bytes(queue._outbuf) == b'FC test' and not queue._p4_lent
    else:
        assert p.payload == b'' and p.state == State.IRS_ANSWER
        assert p.kind == Kind.WORD
        assert queue._buffer_raw == 7 and not queue._p4_lent


@pytest.mark.parametrize('data', [b'', b'\x1cA\x1e'])
def test_association_does_not_turn_a_repeated_request_into_a_new_grant(data):
    m, queue = machine(role='irs', data='body')
    queue.on_host_data(b'FC test')
    first = body(48000, data, 0x40)
    m.observe(first, now=48000)
    p = m.decide(92160-m.lead)
    m.keyed()
    repeat = replace(body(136320, data, 0x40), responds_to=p.stream_sample)
    m.observe(repeat, now=136320)
    p = m.decide(180480-m.lead)
    assert p.state == (State.IRS_ANSWER if data else State.ISS_DATA_ACK)
    assert p.payload == (b'' if data else b'FC test')
    assert queue._buffer_raw == 7
    assert bool(queue._p4_lent) == (not data)


@pytest.mark.parametrize('at', [1307772, 1367772])
def test_recorded_extension_single_copy_crc_and_adapter(at):
    from hfmodem.shrike import onair, p4rx, p4short
    path = ROOT/'captures/onair-0928-2237/stream.flac'
    if not path.exists():
        pytest.skip('local WS8EOC recording absent')
    audio = onair.session.load_wav(str(path), 48000)
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    placed = max((p4short.place_short(z, 1440+p4short.GEOM, cfo)
                  for cfo in (-9., -3., 3., 9.)), key=lambda x: x[0])
    scores = p4short.short_scores(z, first_training_start=placed[1],
                                  frequency_offset=placed[2], refine=0)
    received, polarity = p4short.decode_scores(scores[2])
    assert received == frame() and polarity == 1 and placed[0] > .68
    assert P4Host().accept(received) == TAIL


def test_recorded_prompt_queues_b2f_login_and_leaves_through_driver(monkeypatch):
    from datetime import datetime, timezone
    import json
    from hfmodem.tests.shrike import p4_stock_peer as stock

    import numpy as np
    from hfmodem.shrike import compress, onair, p4breakintx
    from hfmodem.shrike.p4turn import P4Turn
    from hfmodem.shrike.ptc import PtcHost, SimPeer
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx, FS
    from hfmodem.tests.shrike.test_p4_mail_0927 import options
    from hfmodem.winlink import B2FSession, MailClient, compose
    from hfmodem.winlink.session import secure_login_response

    path = ROOT/'captures/onair-0928-2237/stream.flac'
    if not path.exists():
        pytest.skip('local WS8EOC recording absent')
    args = options(monkeypatch, '--mail-fetch', '--mail-wait-greeting',
                   '--p4-link-timeout', '90', '--p4-entry-timeout', '90')
    audio = onair.session.load_wav(str(path), FS)
    stream = Stream(np.r_[audio, np.zeros(90*FS)])
    tx = Tx(stream)
    tx.settle = .044
    transmit = tx._tx
    waves = []

    def flushing(audio, what):
        waves.append(audio.copy())
        transmit(audio, what)
        stream.pos = tx.tx_end

    tx._tx = flushing
    original = json.loads(path.with_name('p4-probe.json').read_text())
    tx.p4_probe.emissions = original['emissions'][:1]
    tx.p4_probe.values = (7, 38)
    stream.pos = original['emissions'][0]['audio_end']
    tx.p4_probe.timeout = 90
    tx.p4_link_options = args
    host = PtcHost(SimPeer(), mycall='W9SSJ')
    host.arq.state, host.arq.role = arq.State.CONNECTED, arq.IRS
    message = compose('W9SSJ', 'SMTP:offline@example.net', 'P4 test', b'hello\r\n',
                      mid='P4EXT0928', date=datetime(2026, 9, 28, tzinfo=timezone.utc))
    session = B2FSession('W9SSJ', target='WS8EOC', password='offline-test-password',
                         comment=False, outbox=[message])
    mail = MailClient(session, host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    tx.p4_link_host, tx.p4_link_mail = host, mail
    sent, settled, counts = [], [], []
    keyed = P4Turn.keyed
    settle = host.arq.settle_bytes
    phase = ['capture']

    def check():
        counts.append(host._txbuf)
        assert host._txbuf == host.arq._buffer_raw
        assert host._txbuf == len(host.arq._outbuf) + len(host.arq._p4_lent)

    def settling(wire):
        settled.append(wire)
        settle(wire)
        check()

    monkeypatch.setattr(host.arq, 'settle_bytes', settling)

    def place(at, wave):
        stream.audio[at:at+len(wave)] = wave

    def peer(turn, at=None):
        p = keyed(turn, at)
        if p.kind == Kind.BODY:
            np.testing.assert_array_equal(waves[-1], p4breakintx.changeover_packet(p.payload, p.status))
        sent.append(p)
        check()
        if session.sent_text and p.state == State.IRS_BREAKIN:
            phase[0] = 'grant'
            stream.audio[p.stream_sample+round(.920*FS):] = 0
            place(p.stream_sample+round(.920*FS), stock.word('V27'))
        elif phase[0] == 'capture' and session.sent_text and p.kind == Kind.WORD:
            assert session.challenge == '31859306'
            phase[0] = 'grant'
            stream.audio[p.stream_sample+round(.330*FS):] = 0
            place(p.stream_sample+round(.330*FS), stock.body(b'', 0x43))
        elif p.kind == Kind.BODY and p.state == State.ISS_GRANT:
            place(p.stream_sample+round(.920*FS), stock.body(b'', 0x40))
        elif p.kind in (Kind.BODY, Kind.NORMAL) and p.payload:
            name = 'V27' if p.status & 1 else 'X27'
            place(p.stream_sample+round(.920*FS), stock.word(name))
        elif p.kind == Kind.SHORT:
            name = 'X27' if p.value == 13 else 'U27' if p.value == 60 else 'V27'
            place(p.stream_sample+round(.920*FS), stock.word(name))
        if phase[0] != 'capture' and host._txbuf == 0:
            turn.finish_application(p.stream_sample)
        return p

    monkeypatch.setattr(P4Turn, 'keyed', peer)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    wire = b''.join(settled)
    plain = compress.Supervisor().feed(wire)
    assert session.remote_sid == '[WL2K-5.0-B2FWIHJM$]'
    assert session.challenge == '31859306' and session.sent_text
    assert not session.failure
    assert plain.count(session.sid.encode()) == 1
    assert (';PR: '+secure_login_response('31859306', 'offline-test-password')).encode() in plain
    assert b'FC EM ' in plain
    assert counts and host._txbuf == 0 and host.sent_total == len(wire)
    assert all(chunk in [p.payload for p in sent if p.kind in (Kind.BODY, Kind.NORMAL)] for chunk in settled)
    assert any(p.state == State.IRS_BREAKIN for p in sent)
    assert any(p.state == State.ISS_DATA and p.payload for p in sent)
    assert not host.arq._p4_lent
    assert any(e.get('event') == 'source-extension' and e['prefix'] == '010101'
               for e in tx.p4_probe.link_events)
    assert any(r.get('status') == 0x83 and r['data'] == DATA.hex() for r in tx.p4_probe.heard)


def test_recorded_prompt_with_deferred_normal_decode_still_sends_login(monkeypatch, capsys):
    from hfmodem.shrike.p4hold import BreakinHold
    monkeypatch.setattr(BreakinHold, 'decode_budget', property(lambda self: .080))
    monkeypatch.setattr(BreakinHold, 'quick_decode_budget', property(lambda self: .080))
    test_recorded_prompt_queues_b2f_login_and_leaves_through_driver(monkeypatch)
    output = capsys.readouterr().out
    assert any('normal-decode at 27.245' in line and 'reason=budget' in line
               for line in output.splitlines())
    assert 'Refused deadline' not in output
