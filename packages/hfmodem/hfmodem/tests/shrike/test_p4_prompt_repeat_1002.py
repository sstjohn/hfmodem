# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Repeats of a packet we acked are acked again; idle and invitations give queued mail a turn."""

from pathlib import Path
from dataclasses import replace

import pytest

from hfmodem.core.wav import stored
from hfmodem.shrike import onair, p4frame, p4turn
from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.tests.evidence import CAPTURES, WORKING, checked_record
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal
from hfmodem.tests.shrike.test_p4turn import machine
from hfmodem.tests.shrike.p4_nak_qam_support import neutral_keys


def test_2142_repeated_chirp_prompt_is_acked_again(monkeypatch):
    path = stored(CAPTURES / 'onair-1002-2142' / 'stream.flac')
    manifest = Path(__file__).with_name('fixtures') / 'prompt-repeat-1002' / 'external.json'
    checked_record(path, manifest)
    checked_record(WORKING / 'onair-1002-2142' /
                   'pactor-p4-bench-ws8eoc-80-force-20261003T024200Z-p4-mail.log', manifest)
    # Log :303, :313, :324, :335: the identical c3 prompt on four 3.75 s
    # cycles. The arm's actual modem options were body/none/parity, :24-25.
    # Its mail stage at :297 was already awaiting proposal answer; :790-791
    # gives these exact queued proposal bytes. Seed only that outbound suffix
    # and its open greeting gate, without inventing a password response or
    # a peer grant. This isolates the necessary first step toward transmission.
    proposal = b'FC EM SZ9GL0ORVGOI 646 477 0\rF> 47\r'
    first = 2208752
    origin = first - FS // 2
    audio = onair.session.load_wav(str(path), FS)[origin:first + 15 * FS].copy()
    stream = Stream(audio)
    tx = Tx(stream)
    tx.settle = .0435
    tx.p4_probe.emissions.clear()
    options = tx.p4_link_options
    options.p4_data, options.p4_normal_ack = 'body', 'parity'
    options.p4_hold_cycles, options.p4_breakin, options.p4_chirp_ack = 16, 'body', 'toggle'
    options.p4_link_timeout = len(audio) / FS
    tx.p4_link_host.arq.on_host_data(proposal)
    receiver = tx.sessrx.p4
    receiver.chirp = receiver.expected = True
    receiver.cfo_hz = -.02
    receiver.clock = Clock(Expect.PACKET, FS // 2, .330, 3.75, -.02)
    constructor = p4turn.P4Turn
    machines = []

    def seeded(*args, **kwargs):
        machine = constructor(*args, **dict(kwargs, role='irs'))
        machine.first_body = False
        machine.allow_breakin = True
        machines.append(machine)
        return machine

    monkeypatch.setattr(p4turn, 'P4Turn', seeded)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    heard = [h for h in tx.p4_probe.heard
             if h['kind'] == Kind.CHIRP and h.get('status') == 0x03]
    assert len(heard) >= 2, 'the reproduction must decode the initial and repeated prompts'
    frames = [o.frame for o in receiver.observations
              if o.kind == Kind.CHIRP and o.frame is not None]
    assert all(f.status.byte == 0x03 and f.data == b' >\r' + b'\x1e' * 19
               for f in frames), 'all decoded fields must be the identical recorded c3 prompt'
    # A break-in on a repeat we had acked went unanswered (WS8EOC 02:06Z,
    # 04:48Z). On this arm the gateway gave the turn itself after the prompt
    # repeats, with an 0x40 long at 151.012 (log :610).
    sent = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert sent and all((e['kind'], e.get('state'), e.get('word')) == (Kind.WORD, p4turn.State.IRS_ANSWER, 'X27')
                        for e in sent), sent
    assert sum(e['event'] == 'acked-repeat' for e in machines[0].link_events) == len(sent) - 1


def answer(m, at, counter, data=b'peer bytes', protocol=0, *, valid=True):
    obs = normal(at, counter | protocol << 5, data)
    if not valid:
        obs = replace(obs, frame=None)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.810*FS))
    return m.decide(at+round(.920*FS)-m.lead)


@pytest.mark.parametrize('queued,allowed', [(True, True), (False, True), (True, False)])
def test_counter_repeat_is_acked_again_with_or_without_mail(queued, allowed):
    m, a = machine(role='irs', data='body')
    m.first_body = False
    m.allow_breakin = allowed
    if queued:
        a.on_host_data(b'outbound')
    first = answer(m, FS, 0, b'first bytes')
    assert first.kind == Kind.WORD and first.state == p4turn.State.IRS_ANSWER
    m.keyed()
    second = answer(m, FS+round(1.25*FS), 0, b'different bytes')
    # WS8EOC 02:06Z, 04:48Z: a break-in on a repeat we had acked went unanswered.
    assert (second.kind, second.word, second.state) == (Kind.WORD, first.word, p4turn.State.IRS_ANSWER)


def test_advancing_counter_and_continuation_defer_request():
    m, a = machine(role='irs', data='body')
    m.first_body = False
    a.on_host_data(b'outbound')
    cycle = round(1.25*FS)
    for slot, (i, protocol) in enumerate([(0, 0), (1, 0), (1, 3), (1, 0), (2, 0)]):
        p = answer(m, FS+slot*cycle, i, b'peer bytes', protocol)
        assert p.kind == Kind.WORD and p.state == p4turn.State.IRS_ANSWER
        m.keyed()
    p = answer(m, FS+5*cycle, 2, b'other bytes')
    assert p.kind == Kind.WORD and p.state == p4turn.State.IRS_ANSWER
    p = answer(m, FS+6*cycle, 3, b'\x1e'*40)
    assert p.kind == Kind.BODY and p.state == p4turn.State.IRS_BREAKIN


@pytest.mark.parametrize('protocol', [0, 2])
@pytest.mark.parametrize('queued,allowed', [(True, True), (False, True), (True, False)])
def test_link_idle_or_turn_invitation_requests_immediately(protocol, queued, allowed):
    m, a = machine(role='irs', data='body')
    m.allow_breakin = allowed
    if queued:
        a.on_host_data(b'outbound')
    p = answer(m, FS, 0, b'\x1e'*40, protocol)
    if queued and allowed:
        assert p.kind == Kind.BODY and p.status == 0x40
        assert p.state == p4turn.State.IRS_BREAKIN
    elif protocol == 2:
        assert p.kind == Kind.BODY and p.state == p4turn.State.ISS_GRANT
    else:
        assert p.kind == Kind.WORD and p.state == p4turn.State.IRS_ANSWER
    assert p.stream_sample == FS+round(.920*FS)


def test_explicit_y27_request_uses_bare_word():
    m, a = machine(role='irs', data='body',  breakin='y27')
    a.on_host_data(b'outbound')
    first = answer(m, FS, 0)
    assert first.kind == Kind.WORD
    m.keyed()
    p = answer(m, FS+round(1.25*FS), 1, b'\x1e'*40)
    assert p.kind == Kind.WORD and p.word == 'Y27'
    assert p.state == p4turn.State.IRS_BREAKIN


@pytest.mark.parametrize('data', [b'F>\r', b'body\x1e'])
def test_source_bytes_and_trailing_padding_do_not_classify_idle(data):
    m, a = machine(role='irs', data='body')
    a.on_host_data(b'outbound')
    p = answer(m, FS, 0, data)
    assert p.kind == Kind.WORD and p.state == p4turn.State.IRS_ANSWER


def test_unread_or_crc_bad_packet_cannot_start_repeat_request():
    m, a = machine(role='irs', data='body')
    a.on_host_data(b'outbound')
    wire = bytearray(p4frame.encode_field(b'peer bytes'.ljust(40, b'\x1e'),
                                            p4frame.Status.from_byte(0)))
    wire[-1] ^= 1
    assert p4frame.decode_field(bytes(wire), data_bytes=40) is None
    history_names=('_ack_chain','acked_counter','_last_ack','_last_body_ack',
                   '_held_answer','_ack_chain')
    queued=(bytes(a._outbuf),a._buffer_raw,a._inflight,m.outbox.inflight,m.counter)
    def unread(at,held):
        history=tuple(getattr(m,n) for n in history_names)
        request=answer(m,at,0,valid=False)
        word = held.word if held else 'V27'
        assert (request.kind,request.state,request.word,request.purpose)==(
            Kind.WORD,p4turn.State.IRS_ANSWER,word,'repeat-request')
        assert request.payload==b'' and request.ack_identity is None
        assert request.ack_reason is None and request.responds_to is None
        assert not request.current_slot_accepted and request.current_slot==at
        assert tuple(getattr(m,n) for n in history_names)==history
        with neutral_keys() as neutral:
            m.keyed()
        assert len(neutral)==1
        assert tuple(getattr(m,n) for n in history_names)==history
        assert (bytes(a._outbuf),a._buffer_raw,a._inflight,m.outbox.inflight,m.counter)==queued
        assert m.state==p4turn.State.IRS_ANSWER and m.breakins==m.grants==0
    unread(FS,None)
    first = answer(m, FS+round(1.25*FS), 0)
    assert first.kind == Kind.WORD and first.state == p4turn.State.IRS_ANSWER
    assert first.purpose is None and first.ack_identity is not None
    m.keyed()
    unread(FS+2*round(1.25*FS),first)


@pytest.mark.parametrize('accepted_first',[False,True])
def test_refused_unread_request_cannot_establish_ack_or_mail_turn(accepted_first):
    m,a=machine(role='irs',data='body',)
    a.on_host_data(b'queued refusal control')
    if accepted_first:
        ordinary=answer(m,FS,0)
        assert ordinary.purpose is None and ordinary.ack_identity is not None
        m.keyed()
    last=m.last
    names=('_ack_chain','acked_counter','_last_ack','_last_body_ack','_held_answer',
           '_ack_chain','counter','breakins','grants','first_body')
    history=tuple(getattr(m,n) for n in names)
    queued=(bytes(a._outbuf),a._buffer_raw,a._inflight,m.outbox.inflight)
    request=answer(m,FS+round(1.25*FS),0,valid=False)
    assert request.purpose == 'repeat-request'
    assert request.state==p4turn.State.IRS_ANSWER
    assert request.kind==Kind.WORD and request.payload==b''
    assert not request.current_slot_accepted
    assert request.word==(ordinary.word if accepted_first else 'V27')
    assert request.ack_identity is None
    assert request.ack_reason is None
    m.refused('deadline')
    assert m.last is last and m.state==p4turn.State.IRS_ANSWER
    assert tuple(getattr(m,n) for n in names)==history
    assert (bytes(a._outbuf),a._buffer_raw,a._inflight,m.outbox.inflight)==queued


def test_repeat_requests_are_idempotent_and_bounded():
    m, a = machine(role='irs', data='body')
    a.on_host_data(b'outbound')
    cycle = round(1.25*FS)
    first = answer(m, FS, 0)
    assert first.kind == Kind.WORD
    m.keyed()
    for i in range(1, 5):
        at = FS+i*cycle
        p = answer(m, at, 1, b'\x1e'*40)
        assert p is m.decide(at+round(.920*FS)-m.lead)
        assert p.kind == Kind.BODY and p.status == 0x40
        m.keyed()
        assert m.breakins == i
    answer(m, FS+5*cycle, 1, b'\x1e'*40)
    assert m.state == p4turn.State.DOWN
    assert m.breakins == 4
