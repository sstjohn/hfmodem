# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""V27 hand-back and normal ack policy pins."""

from dataclasses import replace
import json

import numpy as np
import pytest

from hfmodem.shrike import p4robusttx
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Emission, P4Receiver, Role
from hfmodem.shrike.p4turn import P4Turn, Proposal, State
from hfmodem.tests.shrike.p4_chirp_deadline_support import FS, replay
from hfmodem.tests.shrike.test_p4_close_refused_0930 import answered
from hfmodem.tests.shrike.test_p4turn import ack_frame, machine, propose_ack, word
from hfmodem.tests.evidence import CAPTURES


def heard_bare_v27(heard, at):
    return any(e.get('word') == 'V27' and not e['provisional']
               and abs(e['onset_sample']-at) <= 24 for e in heard)


def handback(result, latest):
    bodies = [e for e in result['events'] if e['event'] == 'Keyed'
              and e.get('kind') == Kind.BODY and e.get('state') == 'ISS_HANDBACK'
              and e.get('status') == 0x40 and not e.get('payload')]
    assert bodies and bodies[0]['at'] < latest
    assert not [e for e in result['events'] if e['event'] == 'Refused']
    assert result['buffered'] == 0
    return bodies[0]


def test_1013_chirp_handback_on_confirmed_train():
    result = replay('onair-1002-1013', stop=50*FS)
    body = handback(result, round(49.5*FS))
    assert body['at'] > 2297472
    assert not [e for e in result['events'] if e['event'] == 'Decision'
                and e['at'] >= 46*FS
                and e['reason'] == 'packet observed; no response selected']
    assert heard_bare_v27(result['heard'], 2237472)
    recorded = json.loads((CAPTURES/'onair-1002-1013'/'p4-probe.json').read_text())
    assert all(heard_bare_v27(recorded['heard'], at) for at in (2237472, 2297472))


def test_1013_pin_rejects_chirp_gate(monkeypatch):
    seed = P4Turn._handback_seed

    def gated(self, reg, now):
        if self.data_kind == Kind.CHIRP and self.state == 'IRS_ANSWER':
            return False
        return seed(self, reg, now)

    with monkeypatch.context() as mutation:
        mutation.setattr(P4Turn, '_handback_seed', gated)
        result = replay('onair-1002-1013', stop=50*FS)
    with pytest.raises(AssertionError):
        handback(result, round(49.5*FS))
    late = [e for e in result['events'] if e['event'] == 'Keyed' and e['at'] >= 46*FS]
    # B1: unread slots repeat the held counter-1 ACK; neither slot is accepted.
    assert [(e['at'], e['kind'], e['word'], e['purpose'], e['ack_reason'], e['state'],
             e['current_slot_accepted']) for e in late] == [
                (2312834, Kind.WORD, 'V27', None, 'counter 1', State.IRS_ANSWER, False),
                (2372834, Kind.WORD, 'V27', None, 'counter 1', State.IRS_ANSWER, False)]
    assert not [e for e in result['events'] if e['event'] == 'Keyed'
                and (e.get('kind') in (Kind.BODY, Kind.CHIRP, Kind.LONG)
                     or e.get('word') == 'Y27' or e.get('payload'))]
    assert not [e for e in result['events'] if e['event'] in ('turn', 'data-reply-ack')]
    assert result['buffered'] == 0 and result['reason'] == 'session ceiling'


def test_1027_protocol_zero_chirp_handback():
    result = replay('onair-1002-1027', stop=35*FS)
    handback(result, round(34.5*FS))
    assert heard_bare_v27(result['heard'], 1517328)
    recorded = json.loads((CAPTURES/'onair-1002-1027'/'p4-probe.json').read_text())
    assert all(heard_bare_v27(recorded['heard'], at) for at in (1517328, 1577328))


def test_2104_seeded_hold_closes_within_four_cycles():
    result = replay('onair-1001-2104', stop=38*FS)
    body = handback(result, 31*FS)
    closes = [e for e in result['events'] if e['event'] == 'Keyed'
              and e.get('kind') == Kind.BODY and e.get('status') == 0x20]
    assert closes and closes[0]['at'] <= body['at']+4*round(1.25*FS)


@pytest.mark.parametrize('data,allow_breakin,handback_expected', [
    ('chirp', False, True),
    ('chirp', True, False),
    ('hold', False, False),
])
def test_queued_bytes_obey_the_greeting_gate(data, allow_breakin, handback_expected):
    m, a = machine(role='irs', data=data, ladder='chirp', normal_ack='parity')
    sent = propose_ack(m, ack_frame(FS, b'greeting', 0x61))
    m.keyed()
    a.on_host_data(b'waiting')
    m.allow_breakin = allow_breakin
    at = sent.stream_sample+round(.935*FS)
    m.observe(word('V27', at, extent=.245), now=at+round(.28*FS))
    p = m.decide(at+round(.30*FS))
    assert bool(p is not None and p.kind == Kind.BODY and p.status == 0x40
                and p.state == 'ISS_HANDBACK') == handback_expected
    assert a._buffer_raw == 7


@pytest.mark.parametrize('policy,words', [
    ('parity', ('V27', 'X27')),
    ('x27', ('X27', 'X27')),
    ('x27-u27', ('U27', 'X27')),
])
def test_protocol_three_normal_uses_counter_policy(policy, words):
    m, _ = machine(role='irs', normal_ack=policy)
    for status, expected in zip((0x61, 0x62), words):
        p = propose_ack(m, ack_frame(48000+status*60000, b'data', status))
        assert p.kind == Kind.WORD and p.word == expected
        m.keyed()


@pytest.mark.parametrize('policy,words', [
    ('parity', ('X27', 'V27', 'X27', 'V27')),
    ('x27', ('X27',)*4),
    ('x27-u27', ('X27', 'U27', 'X27', 'U27')),
])
def test_iss_expectation_matches_normal_policy(policy, words):
    m, _ = machine(normal_ack=policy)
    for counter, expected in enumerate(words):
        m._data_packet = Proposal(Kind.NORMAL, 0, status=counter)
        assert m._expected_data_ack() == expected
        assert m._normal_ack((counter, b''), None)[0] == expected
    if policy == 'x27-u27':
        assert m._normal_ack((1, b''), replace(Proposal(Kind.WORD, 0, word='U27'),
                                                ack_reason='counter 1'))[0] == 'U27'


def test_x27_u27_receiver_admits_odd_counter_reply():
    m, a = machine(normal_ack='x27-u27', data='body')
    a.on_host_data(b'mail')
    m.counter = 1
    m._first_data = False
    m._data(FS, State.ISS_DATA, Kind.NORMAL)
    sent = m.keyed()
    assert sent.status == 1 and sent.payload == b'mail'
    assert 'U27' in m.expected_words()
    assert m._expected_data_ack() == 'U27'
    at = sent.stream_sample+round(.935*FS)
    rx = P4Receiver(FS, admitted=True)
    rx.expected_words = m.expected_words()
    rx.note_emission(Emission(Kind.NORMAL, Role.ISS, sent.stream_sample,
                              .805, packet_reply=True))
    wave = p4robusttx.robust_word('U27')
    audio = np.zeros(at+len(wave)+FS//10, np.float32)
    audio[at:at+len(wave)] = wave
    heard = rx.read_window(audio, 0, whole=True)
    replies = [reg for reg in heard if reg.observation.word == 'U27'
               and not reg.observation.unexpected and not reg.observation.provisional]
    assert replies
    for reg in replies:
        m.observe(reg, now=at+round(.28*FS))
    assert m._data_packet is None and a._buffer_raw == 0


def test_x27_u27_does_not_admit_u27_after_first_grant():
    m, _ = machine(normal_ack='x27-u27', data='body')
    m._body(FS, 0x40, State.ISS_GRANT)
    grant = m.keyed()
    assert 'U27' not in m.expected_words()
    at = grant.stream_sample+round(.935*FS)
    rx = P4Receiver(FS, admitted=True)
    rx.expected_words = m.expected_words()
    rx.note_emission(Emission(Kind.BODY, Role.ISS, grant.stream_sample,
                              .805, packet_reply=True))
    wave = p4robusttx.robust_word('U27')
    audio = np.zeros(at+len(wave)+FS//10, np.float32)
    audio[at:at+len(wave)] = wave
    heard = rx.read_window(audio, 0, whole=True)
    assert any(reg.observation.word == 'U27' and reg.observation.unexpected
               for reg in heard)
    for reg in heard:
        m.observe(reg, now=at+round(.30*FS))
    assert m.decide(at+round(.30*FS)) is None


def seeded_handback(*, allow_breakin, managed=False):
    m, a = machine(data='chirp', ladder='chirp', hold_cycles=16,
                   managed_application=managed)
    m._idle(FS)
    idle = m.keyed()
    if not allow_breakin:
        a.on_host_data(b'queued mail')
        m.allow_breakin = False
    at = idle.stream_sample+round(.935*FS)
    m.observe(word('V27', at, extent=.245, responds_to=idle.stream_sample),
              now=at+round(.30*FS))
    offer = m.decide(at+round(.30*FS))
    assert (offer.kind, offer.status, offer.state) == (Kind.BODY, 0x40, State.ISS_HANDBACK)
    m.keyed()
    deadline = m._hold_until
    at = offer.stream_sample+round(.935*FS)
    m.observe(word('V27', at, extent=.245, responds_to=offer.stream_sample),
              now=at+round(.30*FS))
    idle = m.decide(at+round(.30*FS))
    assert (idle.kind, idle.state) == (Kind.NORMAL, State.ISS_HOLD)
    m.keyed()
    return m, a, idle, deadline


@pytest.mark.parametrize('managed', [False, True], ids=['probe', 'managed-mail'])
def test_gated_mail_stays_queued_after_handback_seed_and_idle(managed):
    m, a, idle, deadline = seeded_handback(allow_breakin=False, managed=managed)
    assert m._hold_until == deadline
    p = m.decide(idle.stream_sample+round(1.25*FS)-m.lead)
    assert p.kind == Kind.NORMAL and not p.payload
    assert a._buffer_raw == len(b'queued mail')
    m.keyed()
    assert m._hold_until == deadline
    while m.last.stream_sample < deadline-round(1.25*FS):
        p = m.decide(m.last.stream_sample+round(1.25*FS)-m.lead)
        assert p.kind == Kind.NORMAL and not p.payload
        m.keyed()
        assert m._hold_until == deadline
    p = m.decide(m.last.stream_sample+round(1.25*FS)-m.lead)
    if managed:
        assert (p.kind, p.state) == (Kind.NORMAL, State.ISS_HOLD)
        assert m.close_context is None
        m.finish_application(p.stream_sample, failure='mail expired')
        p = m.decide(p.stream_sample)
    assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
    assert p.stream_sample <= deadline + (round(1.25*FS) if managed else 0)
    assert a._buffer_raw == len(b'queued mail')


def test_eligible_mail_after_handback_seed_starts_word_yield_ladder():
    m, a, idle, _ = seeded_handback(allow_breakin=True)
    a.on_host_data(b'eligible mail')
    p = m.decide(idle.stream_sample+round(1.25*FS)-m.lead)
    assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
    assert a._buffer_raw == len(b'eligible mail')
    m.keyed()
    assert answered(m, 'X27').value == 49
    assert answered(m, 'X27').value == 53
    chirp = answered(m, 'X27')
    assert (chirp.kind, chirp.status, chirp.payload) == (
        Kind.CHIRP, 0x60, b'eligible mail')


def test_handback_replaces_existing_hold_deadline():
    m, _ = machine(data='chirp', hold_cycles=16)
    m._idle(FS)
    m.keyed()
    m._hold_until = FS + round(16*1.25*FS)
    at = FS+round(.935*FS)
    m.observe(word('V27', at, extent=.245, responds_to=FS),
              now=at+round(.28*FS))
    p = m.decide(at+round(.30*FS))
    assert p is not None and p.state == 'ISS_HANDBACK'
    assert m._hold_until == p.stream_sample + round(4*1.25*FS)
