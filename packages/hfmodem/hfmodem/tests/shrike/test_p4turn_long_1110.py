# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Caller LONG ownership, fixed-level retries, and stock/Trimode admission."""
from dataclasses import replace

import pytest

from hfmodem.shrike import p4frame
from hfmodem.shrike.p4turn import Kind, State, LONG_LADDERS
from hfmodem.tests.shrike.test_p4turn import machine, word, body, propose_ack, FS


def reply(m, name, *, partial=False, **kwargs):
    at = m.last.stream_sample + round((3.4 if m.last.kind == Kind.LONG else .92) * FS)
    m.observe(word(name, at, partial, responds_to=m.last.stream_sample, **kwargs), now=at + round(.25*FS))
    return m._next


def caller(payload, **kwargs):
    m, a = machine(data='long', **kwargs)
    a.on_host_data(b'A'*31 + payload)
    m._begin_data_turn(0, State.ISS_DATA_ACK)
    first = m.keyed()
    assert (first.kind, first.status, first.payload) == (Kind.BODY, 0, b'A'*31)
    reply(m, 'X27')
    assert a._buffer_raw == len(payload)
    return m, a


def ladder(m, admission='U27'):
    for i, value in enumerate(m.ladder):
        assert m._next.value == value and m._next.header == 'A'
        m.keyed()
        reply(m, admission if i == len(m.ladder)-1 else 'X27')
    assert m._next.kind == Kind.LONG
    return m.keyed()


@pytest.mark.parametrize('level', (2, 3, 4))
@pytest.mark.parametrize('admit', ('u27', 'u27,x27'))
@pytest.mark.parametrize('last_word', ('U27', 'X27'))
def test_complete_ladders(level, admit, last_word):
    m, a = caller(b'payload', long_level=level, ladder_admit=admit)
    assert m.ladder == LONG_LADDERS[level]
    for i, value in enumerate(m.ladder):
        assert m._next.value == value
        m.keyed()
        if i < len(m.ladder)-1:
            reply(m, 'U27')
            assert m._next.value == value and m._next.header == 'B'
            m.keyed()
            reply(m, 'X27')
        else:
            reply(m, last_word)
    if last_word.lower() not in admit.split(','):
        assert m._next.value == m.ladder[-1] and m._next.header == 'B'
        m.keyed()
        reply(m, 'U27')
    packet = m.keyed()
    assert (packet.level, packet.status, packet.final) == (level, 0x41, True)
    assert m._expected_data_ack() == 'X27'
    reply(m, 'X27')
    assert a._buffer_raw == 0 and not m.outbox.inflight
    assert m.state == State.IRS_ANSWER and m._next is None


def test_long_override_does_not_inherit_chirp_automatic_rung_advance():
    m, _ = caller(b'payload', ladder_values=(13, 49, 53))
    for _ in range(2):
        m.keyed()
        reply(m, 'W27')
        assert m._next.value == 13 and m._next.header == 'B'
        event = m.link_events[-1]
        assert (event['word'], event['expected'], event['verdict']) == ('W27', 'X27', 'repeat')
    m.keyed()
    reply(m, 'X27')
    assert m._next.value == 49


def test_weak_ladder_answer_does_not_reset_retry_budget():
    m, _ = caller(b'payload')
    m.keyed()
    reply(m, 'X27')
    m.keyed()
    for _ in range(3):
        m.silence(m.last.stream_sample+FS)
        m.keyed()
    assert m.repeats == 3
    at = m.last.stream_sample+round(.92*FS)
    reg = word('X27', at, responds_to=m.last.stream_sample)
    m.observe(replace(reg, observation=replace(reg.observation, weak=True)), now=at+round(.25*FS))
    assert m.repeats == 3
    event = m.link_events[-1]
    assert (event['expected'], event['verdict']) == ('U27,X27', 'ambiguous')


@pytest.mark.parametrize('size,statuses', [(69, [0x41]), (70, [0x61, 0x42]), (138, [0x61, 0x42])])
def test_boundaries_and_post_handover_output(size, statuses):
    m, a = caller(b'B'*size)
    packet = ladder(m)
    for status in statuses:
        assert packet.status == status
        reply(m, m._expected_data_ack())
        if m._next:
            packet = m.keyed()
    assert not m.outbox.inflight and a._buffer_raw == 0
    assert m.state == State.IRS_ANSWER
    a.on_host_data(b'next turn')
    assert m.decide(m.last.stream_sample + 4*FS) is None
    assert bytes(a._outbuf) == b'next turn'
    m._begin_data_turn(m.last.stream_sample+5*FS)
    assert m._next.kind == Kind.BODY and m._next.status == 0


@pytest.mark.parametrize('spug,expected', [('queued', [0x61, 0x62, 0x43]), ('off', [1, 2, 0x43])])
def test_spug(spug, expected):
    m, a = caller(b'B'*139, spug=spug)
    p = ladder(m)
    for status in expected:
        assert p.status == status
        reply(m, m._expected_data_ack())
        if m._next:
            p = m.keyed()
    assert a._buffer_raw == 0


@pytest.mark.parametrize('policy', ['repeat', 'ack'])
def test_u27_and_accepting_word_chain(policy):
    m, a = caller(b'B'*70, iss_u27=policy)
    first = ladder(m)
    reply(m, 'U27')
    event = [e for e in m.link_events if e['event'] == 'IssAck'][-1]
    assert event['expected'] == 'X27' and event['verdict'] == policy
    if policy == 'repeat':
        assert m._next.repeat and m._next.payload == first.payload
        assert a._buffer_raw == 70
        m.keyed()
        reply(m, 'X27')
    else:
        assert a._buffer_raw == 1
    m.keyed()
    assert m._expected_data_ack() == ('X27' if policy == 'ack' else 'V27')


@pytest.mark.parametrize('ack,partner', [('toggle','V27'), ('toggle-t27','T27')])
def test_long_ack_pair(ack, partner):
    m, _ = caller(b'B'*70, long_ack=ack)
    ladder(m)
    reply(m, 'X27')
    m.keyed()
    assert m._expected_data_ack() == partner
    reply(m, 'X27')
    assert m._next.repeat
    m.keyed()
    reply(m, partner)
    assert m.state == State.IRS_ANSWER


@pytest.mark.parametrize('level,size', [(2, 40), (3, 69), (4, 141)])
def test_long_counter_even_t27_survives_retry_and_wrap(level, size):
    m, _ = caller(b'B' * (4 * size), long_ack='counter-even-t27', long_level=level)
    ladder(m)
    for counter, expected in [(1, 'X27'), (2, 'T27'), (3, 'X27'), (0, 'T27')]:
        assert m.last.status & 3 == counter
        assert m._expected_data_ack() == expected
        reply(m, 'U27')
        assert m._next.repeat
        m.keyed()
        assert m._expected_data_ack() == expected
        reply(m, expected)
        if m._next is not None:
            m.keyed()
    assert m.state == State.IRS_ANSWER


def test_seven_identical_retries_restart_once_then_close_with_bytes_requeued():
    m, a = caller(b'final')
    ladder(m)
    a.on_host_data(b'later')
    for attempt in range(2):
        for count in range(1, 8):
            m.silence(m.last.stream_sample + round(3.7*FS))
            p = m.keyed()
            assert (p.level, p.payload, p.status, p.final, p.repeat) == (3, b'final', 0x41, True, True)
            assert m.data_repeats == count
        m.silence(m.last.stream_sample + round(3.7*FS))
        assert not m.outbox.inflight and bytes(a._outbuf) == b'finallater'
        if attempt == 0:
            assert m._next.value == 13
            ladder(m)
        else:
            assert m._next.state == State.ISS_CLOSE and m._next.status == 0x20
    assert a._buffer_raw == 10


def test_two_refusals_restart_once_then_close():
    m, a = caller(b'final')
    ladder(m)
    for attempt in range(2):
        reply(m, 'Z27')
        assert m._next.repeat
        m.keyed()
        reply(m, 'Z27')
        assert not m.outbox.inflight
        if not attempt:
            ladder(m)
        else:
            assert m._next.state == State.ISS_CLOSE
    assert bytes(a._outbuf) == b'final'
    assert [e['verdict'] for e in m.link_events
            if e['event'] == 'IssAck' and e['word'] == 'Z27'] == ['refused']*4


@pytest.mark.parametrize('restart', ['exhaustion', 'refusals'])
@pytest.mark.parametrize('policy', ['repeat', 'ack'])
@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('reception', ['ack', 'unread', 'lost-ack'])
def test_c2_restart_preserves_ack_phase(restart, policy, final, reception):
    tail = b'' if final else b'tail'
    m, a = caller(b'B'*69 + b'C'*69 + tail, iss_u27=policy)
    peer, _ = machine(role='irs')

    def peer_answer(packet, *, unread=False):
        frame = None if unread else p4frame.decode_field(p4frame.encode_field(
            packet.payload, p4frame.Status.from_byte(packet.status)), data_bytes=69)
        reg = body(packet.stream_sample)
        reg = replace(reg, observation=replace(reg.observation, kind=Kind.LONG,
                                              frame=frame, speedlevel='3'))
        answer = propose_ack(peer, reg)
        peer.keyed()
        return answer.word

    first = ladder(m)
    assert peer_answer(first) == 'X27'
    reply(m, 'X27')
    packet = m.keyed()
    assert packet.status & 3 == 2 and packet.final == final
    assert m._expected_data_ack() == 'V27'
    if reception == 'lost-ack':
        assert peer_answer(packet) == 'V27'

    if restart == 'exhaustion':
        for _ in range(7):
            m.silence(m.last.stream_sample + round(3.7*FS))
            m.keyed()
        m.silence(m.last.stream_sample + round(3.7*FS))
    else:
        reply(m, 'Z27')
        m.keyed()
        reply(m, 'Z27')
    assert not m.outbox.inflight and bytes(a._outbuf) == packet.payload + tail
    restored = ladder(m)
    assert restored == replace(packet, stream_sample=restored.stream_sample, repeat=True)
    assert m._expected_data_ack() == 'V27'
    assert m._last_data_ack == 'X27'

    if reception == 'unread':
        assert peer_answer(restored, unread=True) == 'X27'
        reply(m, 'X27')
        assert m.outbox.inflight == packet.payload and a._buffer_raw == 69 + len(tail)
        assert m.state == State.ISS_DATA and m._next.repeat
        assert not any(e['event'] == 'data-turn-ended' for e in m.link_events)
        restored = m.keyed()
    assert peer_answer(restored) == 'V27'
    reply(m, 'V27')
    assert m._last_data_ack == 'V27' and a._buffer_raw == len(tail)
    assert m.state == (State.IRS_ANSWER if final else State.ISS_DATA)
    if final:
        assert m._next is None and not m.outbox.inflight
    else:
        m.keyed()
        assert m._expected_data_ack() == 'X27'


def test_unacked_c0_restarts_once_then_closes_with_loan_requeued():
    m, a = machine(data='long')
    payload = b'A'*31 + b'later'
    a.on_host_data(payload)
    m._begin_data_turn(0, State.ISS_DATA_ACK)
    first = m.keyed()
    for attempt in range(2):
        for count in range(1, 8):
            m.silence(m.last.stream_sample + FS)
            p = m.keyed()
            assert (p.kind, p.status, p.payload, p.repeat) == (Kind.BODY, 0, b'A'*31, True)
            assert m.data_repeats == count
        m.silence(m.last.stream_sample + FS)
        assert not m.outbox.inflight and bytes(a._outbuf) == payload
        if attempt == 0:
            for i in range(len(m.ladder)):
                m.keyed()
                reply(m, 'U27' if i == len(m.ladder)-1 else 'X27')
            restored = m.keyed()
            assert restored == replace(first, stream_sample=restored.stream_sample, repeat=True)
            assert m.state == State.ISS_DATA_ACK and m._expected_data_ack() == 'X27'
        else:
            assert m._next.state == State.ISS_CLOSE and m._next.status == 0x20
    assert m._ladder_returned and a._buffer_raw == len(payload)
    assert m.close_context.inflight == first.payload
    assert m.close_context.queued == b'later'


def test_final_repeat_keeps_status_when_more_output_arrives():
    m, a = caller(b'final')
    p = ladder(m)
    a.on_host_data(b'future')
    reply(m, 'U27')
    assert m._next.status == p.status and m._next.final
    m.keyed()
    reply(m, 'X27')
    assert m.state == State.IRS_ANSWER and bytes(a._outbuf) == b'future'


def test_final_checked_peer_body_accepts_long_and_starts_receive_chain():
    m, a = caller(b'final')
    p = ladder(m)
    at = p.stream_sample + round(3.4*FS)
    m.observe(word('Y27', at, extent=.8, responds_to=p.stream_sample))
    m.observe(replace(body(at, b'FS YY\r', 0x40), responds_to=p.stream_sample), now=at+round(.8*FS))
    assert a._buffer_raw == 0 and m.state == State.IRS_ANSWER
    answer = m.decide(at+round(.846*FS))
    assert answer.word == 'X27'


def test_data_forms():
    for form, values in [('none', ()), ('body', ()), ('chirp', (13,49,53))]:
        m, _ = machine(data=form)
        assert m.ladder == values
    m, _ = machine(data='long', ladder_values=(61,))
    assert m.ladder == (61,)


def fixture(name):
    import json
    from pathlib import Path
    from hfmodem.tests.evidence import require_input
    return json.loads(require_input(Path(__file__).parent/'fixtures'/name).read_text())


def test_k0003_stock_caller_order_with_labelled_trimode_phase_injection():
    trace = fixture('p4-iss-k0003.json')
    rows = trace['timeline']
    assert trace['divergence'] == dict(at_s=1460.934, stock_answer='V27',
                                     injected_answer='X27', reason='D S7.3: Trimode turn-start phase')
    final_payload = rows[4]['payload'].encode('ascii')
    m, a = machine(data='long')
    a.on_host_data(b'A'*31+b'B'*69+final_payload)
    m._begin_data_turn(round(rows[0]['at_s']*FS), State.ISS_DATA_ACK)
    first = m.keyed()
    assert first.kind == Kind.BODY and first.status == rows[0]['status']
    reply(m, trace['divergence']['injected_answer'])
    for row in rows[1:3]:
        p = m.keyed()
        assert p.kind == Kind.SHORT and p.value == row['value']
        reply(m, row['answer'])
    p = m.keyed()
    assert p.kind == Kind.LONG and p.level == 3 and p.status == 0x61
    reply(m, rows[3]['answer'])
    p = m.keyed()
    assert p.level == rows[4]['level'] and p.payload == final_payload
    assert p.status & 0x60 == rows[4]['status'] & 0x60 == 0x40
    at = p.stream_sample + round(3.4*FS)
    m.observe(word('Y27', at, extent=.8, responds_to=p.stream_sample))
    m.observe(replace(body(at, b'', 0x40), responds_to=p.stream_sample), now=at+round(.8*FS))
    assert a._buffer_raw == 0 and m.state == State.IRS_ANSWER
    m.decide(at+round(.846*FS))
    m.keyed()
    peer = round(rows[5]['at_s']*FS)
    m.observe(word('Y27', peer, extent=.8))
    m.observe(body(peer, rows[5]['payload'].encode(), rows[5]['status']))
    assert m.decide(peer+round(.846*FS)).word is not None


@pytest.mark.parametrize('inject_x27', [False, True])
def test_ws8eoc_recorded_words_and_separate_x27_admission_policy(inject_x27):
    trace = fixture('p4-iss-ws8eoc-0929-2327.json')
    m, a = machine(data='long', prekey_lead=2276/FS)
    payload = bytes.fromhex(trace['emissions'][0]['payload'])
    a.on_host_data(payload+b'next long')
    m._begin_data_turn(trace['emissions'][0]['audio_start'], State.ISS_DATA_ACK)
    for i, row in enumerate(trace['words']):
        p = m.keyed()
        assert p.stream_sample == row['responds_to']
        name = 'X27' if inject_x27 and i == 2 else row['word']
        m.observe(word(name, row['onset_sample'], responds_to=p.stream_sample),
                  now=row['onset_sample']+round(.24*FS))
    if inject_x27:
        assert m._next.kind == Kind.LONG and m._next.status == 0x41
    else:
        assert m._next.value == 60 and m._next.header == 'B'
    assert a._buffer_raw == len(b'next long')


def test_long_turn_grant_by_word_still_starts_with_body_c0():
    m, a = machine(data='long')
    a.on_host_data(b'A'*100)
    m._begin_data_turn(0, word_yield=True)
    p = m.keyed()
    assert p.kind == Kind.BODY and p.status == 0
    reply(m, 'X27')
    assert m._next.value == 13


def test_none_does_not_lend_queued_data_or_run_a_ladder():
    m, a = machine(data='none')
    a.on_host_data(b'not sent')
    m.start(0)
    assert m._next.kind == Kind.NORMAL and not m._next.payload
    assert bytes(a._outbuf) == b'not sent' and not m.outbox.inflight
    for name in ('X27', 'V27'):
        m.keyed()
        reply(m, name)
        assert m._next.kind == Kind.NORMAL and not m._next.payload
        assert bytes(a._outbuf) == b'not sent' and not m.outbox.inflight


@pytest.mark.parametrize('field', ['weak', 'unexpected', 'provisional', 'expired', 'unassociated'])
def test_uncertain_or_unassociated_long_word_cannot_commit(field):
    from hfmodem.shrike.p4observation import Registered
    m, a = caller(b'final')
    p = ladder(m)
    at = p.stream_sample+round(3.4*FS)
    reg = word('X27', at, responds_to=p.stream_sample)
    if field in ('weak','unexpected','provisional'):
        reg = replace(reg, observation=replace(reg.observation, **{field:True}))
    elif field == 'expired':
        reg = replace(reg, outcome=Registered.EXPIRED)
    else:
        reg = replace(reg, responds_to=None)
    m.observe(reg, now=at+round(.25*FS))
    assert a._buffer_raw == 5 and m.outbox.inflight == b'final'
    assert m.state != State.IRS_ANSWER


def test_counter_wrap_is_independent_of_long_ack_toggle():
    m, a = caller(b'B'*(69*6))
    p = ladder(m)
    for counter, expected in [(1,'X27'),(2,'V27'),(3,'X27'),(0,'V27'),(1,'X27'),(2,'V27')]:
        assert p.status & 3 == counter and m._expected_data_ack() == expected
        reply(m, expected)
        if m._next:
            p = m.keyed()
    assert a._buffer_raw == 0 and m.state == State.IRS_ANSWER


@pytest.mark.parametrize('level', [2,3,4])
def test_last_rung_silence_and_other_words_repeat_b_until_budget(level):
    m, a = caller(b'final', long_level=level)
    while m._next.index < len(m.ladder)-1:
        m.keyed()
        reply(m, 'X27')
    m.keyed()
    for i in range(3 if level == 2 else 7):
        if i % 2:
            reply(m, 'W27')
        else:
            m.silence(m.last.stream_sample+FS)
        p = m.keyed()
        assert p.value == m.ladder[-1] and p.header == 'B'
    m.silence(m.last.stream_sample+FS)
    assert m._next.state == State.ISS_CLOSE
    assert a._buffer_raw == 5 and bytes(a._outbuf) == b'final'


def test_final_ack_then_gateway_idle_and_fs_keeps_new_output_for_next_turn():
    delivered = []
    from types import SimpleNamespace
    m, a = caller(b'final')
    p = ladder(m)
    reply(m, 'X27')
    a.on_host_data(b'future')
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *args: None)
    for i, payload in enumerate((b'', b'FS YY\r')):
        at = p.stream_sample+round((4.5+1.25*i)*FS)
        response = p.stream_sample if i == 0 else m.last.stream_sample
        m.observe(word('Y27', at, extent=.8, responds_to=response))
        m.observe(replace(body(at,payload,0x40),responds_to=response), now=at+round(.81*FS))
        q = m.decide(at+round(.846*FS))
        assert q.kind == Kind.WORD and q.word == 'X27'
        m.keyed()
        assert m.state == State.IRS_ANSWER and bytes(a._outbuf) == b'future'
    assert delivered == [b'FS YY\r']
