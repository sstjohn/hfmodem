# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Every hand-back entry and reply exit obeys the mail gate and chirp ladder."""

import pytest

from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4turn import Proposal, State, counter_ack
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, machine, word
from hfmodem.tests.shrike.test_p4_data_turn_0930 import receive

CYCLE = round(1.25 * FS)
T = round(.920 * FS)
MAIL = b'queued mail in two chirp fields for the hand-back test'


def offer(entry, *, ladder='chirp', data='chirp', managed=False):
    m, arq = machine(role='irs', ladder=ladder, data=data, hold_cycles=16,
                     managed_application=managed)
    if entry == 'irs-word':
        m._next = Proposal(Kind.WORD, 0, word='X27', state=State.IRS_ANSWER)
    else:
        m._idle(0)
    m.keyed()
    m.observe(word('V27', T, responds_to=m.last.stream_sample), now=T+round(.3*FS))
    proposed = m.decide(T+round(.3*FS))
    assert (proposed.kind, proposed.status, proposed.state) == (Kind.BODY, 0x40,
                                                               State.ISS_HANDBACK)
    m.keyed()
    return m, arq, m._hold_until


def reply(m, name, *, provisional=False):
    at = m.last.stream_sample + T
    m.observe(word(name, at, partial=provisional,
                   responds_to=m.last.stream_sample), now=at+round(.3*FS))
    return m.decide(at+round(.3*FS))


@pytest.mark.parametrize('entry', ['irs-word', 'iss-idle'])
@pytest.mark.parametrize('exit', ['x27', 'x27-confirmed', 'x27-replaced-v27',
                                  'v27', 'v27-confirmed', 'z27-twice',
                                  'silence', 'peer-body', 'peer-normal-0',
                                  'peer-normal-3', 'peer-body-data', 'peer-long',
                                  'peer-chirp', 'peer-grant', 'peer-y27',
                                  'peer-close', 'peer-body-grant'])
@pytest.mark.parametrize('eligible', [False, True], ids=['gated-mail', 'eligible-mail'])
@pytest.mark.parametrize('managed', [False, True], ids=['probe', 'managed-mail'])
def test_handback_entry_exit_mail_policy(entry, exit, eligible, managed):
    m, arq, deadline = offer(entry, managed=managed)
    arq.on_host_data(MAIL)
    m.allow_breakin = eligible

    if exit == 'x27':
        p = reply(m, 'X27')
    elif exit == 'x27-confirmed':
        at = m.last.stream_sample + T
        early = word('X27', at, partial=True, responds_to=m.last.stream_sample)
        m.observe(early, now=at+round(.25*FS))
        p = m.decide(at+round(.3*FS))
        m.observe(word('X27', at, outcome=Registered.CONFIRMED,
                       replaced=early.observation, responds_to=m.last.stream_sample),
                  now=at+round(.31*FS))
        p = m.decide(at+round(.31*FS))
    elif exit == 'x27-replaced-v27':
        at = m.last.stream_sample + T
        early = word('V27', at, partial=True, responds_to=m.last.stream_sample)
        m.observe(early, now=at+round(.25*FS))
        m.observe(word('X27', at, outcome=Registered.CONTRADICTED,
                       replaced=early.observation, responds_to=m.last.stream_sample),
                  now=at+round(.3*FS))
        p = m.decide(at+round(.3*FS))
    elif exit in ('v27', 'v27-confirmed'):
        at = m.last.stream_sample + T
        if exit == 'v27-confirmed':
            early = word('V27', at, partial=True, responds_to=m.last.stream_sample)
            m.observe(early, now=at+round(.25*FS))
            m.observe(word('V27', at, outcome=Registered.CONFIRMED,
                           replaced=early.observation, responds_to=m.last.stream_sample),
                      now=at+round(.3*FS))
            p = m.decide(at+round(.3*FS))
        else:
            p = reply(m, 'V27')
    elif exit == 'z27-twice':
        reply(m, 'Z27')
        m.silence(m.last.stream_sample+CYCLE-m.lead)
        m.keyed()
        p = reply(m, 'Z27')
    elif exit == 'silence':
        for _ in range(4):
            m.silence(m.last.stream_sample+CYCLE-m.lead)
            p = m.decide(m.last.stream_sample+CYCLE-m.lead)
            if p.state == State.ISS_CLOSE:
                break
            m.keyed()
    elif exit == 'peer-body':
        at = m.last.stream_sample + T
        p = receive(m, at, b'FS Y\r', 0x40, responds_to=m.last.stream_sample)
    elif exit == 'peer-body-grant':
        at = m.last.stream_sample + T
        m.observe(ack_frame(at, b'', 0x40, kind=Kind.BODY), now=at)
        p = m.decide(at+round(m.t[Kind.BODY]*FS)-m.lead)
    elif exit == 'peer-y27':
        at = m.last.stream_sample + T
        m.observe(word('Y27', at, extent=.245, responds_to=m.last.stream_sample), now=at)
        p = m.decide(at+T-m.lead)
    else:
        at = m.last.stream_sample + T
        status = {'peer-normal-0': 0, 'peer-normal-3': 0x61,
                  'peer-body-data': 0x40, 'peer-long': 0, 'peer-chirp': 0x61,
                  'peer-grant': 0x40, 'peer-close': 0x20}[exit]
        kind = {'peer-body-data': Kind.BODY, 'peer-long': Kind.LONG,
                'peer-chirp': Kind.CHIRP}.get(exit, Kind.NORMAL)
        data = b'' if exit == 'peer-grant' else b'greeting'
        m.observe(ack_frame(at, data, status, kind=kind), now=at)
        p = m.decide(at+round(m.t[kind]*FS)-m.lead)

    assert arq._buffer_raw == len(MAIL)
    if exit.startswith('x27') or exit.startswith('v27'):
        if eligible:
            if exit.startswith('v27'):
                assert p.kind == Kind.NORMAL and not p.payload
                m.keyed()
                p = m.decide(m.last.stream_sample+CYCLE-m.lead)
            assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
        else:
            assert (p.kind, p.payload, p.state) == (Kind.NORMAL, b'', State.ISS_HOLD)
            m.keyed()
            assert m._hold_until == deadline
            while m.last.stream_sample + CYCLE <= deadline:
                m.silence(m.last.stream_sample+CYCLE-m.lead)
                p = m.decide(m.last.stream_sample+CYCLE-m.lead)
                if p.state == State.ISS_CLOSE:
                    break
                m.keyed()
            if managed:
                assert (p.kind, p.state) == (Kind.NORMAL, State.ISS_HOLD)
                assert m.close_context is None
            else:
                assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
                assert p.stream_sample <= deadline
            assert arq._buffer_raw == len(MAIL)
    elif exit in ('z27-twice', 'silence'):
        if managed:
            assert (p.kind, p.state) == (Kind.NORMAL, State.ISS_HOLD)
            assert m.close_context is None
        else:
            assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
            assert p.stream_sample <= deadline
    elif exit == 'peer-body':
        assert (p.kind, p.word, p.state) == (Kind.WORD, 'X27', State.IRS_ANSWER)
    elif exit in ('peer-grant', 'peer-y27'):
        assert (p.kind, p.status, p.state) == (Kind.BODY, 0x40, State.ISS_GRANT)
    elif exit == 'peer-body-grant':
        if eligible:
            assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
        else:
            assert (p.kind, p.status, p.state) == (Kind.BODY, 0x40, State.ISS_GRANT)
            m.keyed()
            assert m._hold_until == deadline
            while p.state != State.ISS_CLOSE and (not managed or p.stream_sample < deadline):
                now = min(m.last.stream_sample+CYCLE, deadline)-m.lead
                m.silence(now)
                p = m.decide(now)
                if p.state != State.ISS_CLOSE:
                    m.keyed()
                    if not managed:
                        assert m._hold_until == deadline
            if managed:
                assert p.state != State.ISS_CLOSE and m.close_context is None
            else:
                assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
                assert p.stream_sample <= deadline
            assert arq._buffer_raw == len(MAIL)
    else:
        assert p.kind == Kind.WORD and p.state == (State.IRS_CLOSE if exit == 'peer-close'
                                                   else State.IRS_ANSWER)


@pytest.mark.parametrize('entry', ['irs-word', 'iss-idle'])
@pytest.mark.parametrize('exit', ['x27', 'x27-replaced-v27', 'v27'])
def test_eligible_handback_mail_walks_entire_chirp_ladder(entry, exit):
    m, arq, _ = offer(entry)
    arq.on_host_data(MAIL)
    if exit == 'x27-replaced-v27':
        at = m.last.stream_sample+T
        early = word('V27', at, partial=True, responds_to=m.last.stream_sample)
        m.observe(early)
        m.observe(word('X27', at, outcome=Registered.CONTRADICTED,
                       replaced=early.observation, responds_to=m.last.stream_sample))
        p = m.decide(at+round(.3*FS))
    else:
        p = reply(m, 'X27' if exit == 'x27' else 'V27')
    if exit == 'v27':
        m.keyed()
        p = m.decide(m.last.stream_sample+CYCLE-m.lead)
    assert p.value == 13
    for value, answer in ((49, 'X27'), (53, 'X27'), (None, 'U27')):
        m.keyed()
        p = reply(m, answer)
        assert p.value == value
        assert p.kind == (Kind.CHIRP if value is None else Kind.SHORT)
    assert p.status == 0x60 and p.payload == MAIL[:22]
    m.keyed()
    at = m.last.stream_sample+round(3.4*FS)
    m.observe(word('X27', at, responds_to=m.last.stream_sample), now=at+round(.25*FS))
    p = m.decide(at+round(.25*FS))
    assert (p.kind, p.status, p.payload) == (Kind.CHIRP, 0x61, MAIL[22:44])


@pytest.mark.parametrize('entry', ['irs-word', 'iss-idle'])
@pytest.mark.parametrize('resolved', ['X27', 'V27'])
@pytest.mark.parametrize('eligible', [False, True], ids=['gated-mail', 'eligible-mail'])
def test_keyed_speculative_idle_resolves_without_losing_mail_policy(entry, resolved,
                                                                     eligible):
    m, arq, deadline = offer(entry)
    arq.on_host_data(MAIL)
    m.allow_breakin = eligible
    offer_at = m.last.stream_sample
    at = offer_at + T
    early = word('V27', at, partial=True, responds_to=offer_at)
    m.observe(early, now=at+round(.25*FS))
    speculative = m.decide(offer_at+CYCLE-m.lead)
    assert (speculative.kind, speculative.status, speculative.payload) == (Kind.NORMAL, 0, b'')
    m.keyed()
    now = m.last.stream_sample+round(.05*FS)
    m.observe(word(resolved, at, outcome=Registered.CONFIRMED if resolved == 'V27'
                   else Registered.CONTRADICTED, replaced=early.observation,
                   responds_to=offer_at), now=now)
    p = m.decide(now)
    if eligible:
        assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
    else:
        assert (p.kind, p.payload, p.state) == (Kind.NORMAL, b'', State.ISS_HOLD)
        m.keyed()
        assert m._hold_until == deadline
    assert arq._buffer_raw == len(MAIL)


@pytest.mark.parametrize('entry', ['irs-word', 'iss-idle'])
@pytest.mark.parametrize('seed', ['v27', 'x27'])
@pytest.mark.parametrize('ack', ['word', 'checked-body'])
@pytest.mark.parametrize('eligible', [False, True], ids=['gated-mail', 'eligible-mail'])
def test_counted_idle_ack_obeys_handback_mail_policy(entry, seed, ack, eligible):
    m, arq, deadline = offer(entry)
    p = reply(m, seed.upper())
    assert (p.kind, p.state) == (Kind.NORMAL, State.ISS_HOLD)
    m.keyed()
    arq.on_host_data(MAIL)
    m.allow_breakin = eligible
    at = m.last.stream_sample + T
    if ack == 'word':
        m.observe(word(counter_ack(m.last.status), at,
                       responds_to=m.last.stream_sample), now=at+round(.3*FS))
        p = m.decide(at+round(.3*FS))
    else:
        m.observe(ack_frame(at, b'', 0x40, kind=Kind.BODY), now=at)
        p = m.decide(at+round(m.t[Kind.BODY]*FS)-m.lead)
    assert arq._buffer_raw == len(MAIL)
    if eligible:
        assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
        for value, answer in ((49, 'X27'), (53, 'X27'), (None, 'U27')):
            m.keyed()
            p = reply(m, answer)
            assert p.value == value
        assert (p.kind, p.status, p.payload) == (Kind.CHIRP, 0x60, MAIL[:22])
    else:
        if p.state != State.ISS_CLOSE:
            assert (p.kind, p.payload, p.state) == (Kind.NORMAL, b'', State.ISS_HOLD)
            m.keyed()
            assert m._hold_until == deadline
            while m.last.stream_sample + CYCLE <= deadline:
                m.silence(m.last.stream_sample+CYCLE-m.lead)
                p = m.decide(m.last.stream_sample+CYCLE-m.lead)
                if p.state == State.ISS_CLOSE:
                    break
                m.keyed()
        assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
        assert p.stream_sample <= deadline
        assert arq._buffer_raw == len(MAIL)


@pytest.mark.parametrize('entry', ['irs-word', 'iss-idle'])
@pytest.mark.parametrize('resolution', ['expired', 'refuted'])
@pytest.mark.parametrize('eligible', [False, True], ids=['gated-mail', 'eligible-mail'])
def test_refuted_speculative_v27_retries_offer_before_mail(entry, resolution, eligible):
    m, arq, deadline = offer(entry)
    arq.on_host_data(MAIL)
    m.allow_breakin = eligible
    offer_at = m.last.stream_sample
    at = offer_at + T
    early = word('V27', at, partial=True, responds_to=offer_at)
    m.observe(early, now=at+round(.25*FS))
    speculative = m.decide(offer_at+CYCLE-m.lead)
    assert (speculative.kind, speculative.payload) == (Kind.NORMAL, b'')
    m.keyed()
    now = m.last.stream_sample+round(.05*FS)
    m.observe(word('V27' if resolution == 'expired' else 'Z27', at,
                   outcome=Registered.EXPIRED if resolution == 'expired'
                   else Registered.CONTRADICTED, replaced=early.observation,
                   responds_to=offer_at), now=now)
    p = m.decide(now)
    assert (p.kind, p.status, p.state) == (Kind.BODY, 0x40, State.ISS_HANDBACK)
    assert arq._buffer_raw == len(MAIL)
    assert m._hold_until == deadline
    m.keyed()
    p = reply(m, 'X27')
    if eligible:
        assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
    else:
        assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
        assert p.stream_sample <= deadline
        assert arq._buffer_raw == len(MAIL)


@pytest.mark.parametrize('entry', ['irs-word', 'iss-idle'])
@pytest.mark.parametrize('exit', ['x27', 'counted-idle'])
@pytest.mark.parametrize('eligible', [False, True], ids=['gated-mail', 'eligible-mail'])
def test_body_mode_retains_counted_continuation_and_mail_gate(entry, exit, eligible):
    m, arq, deadline = offer(entry, ladder='none', data='body')
    arq.on_host_data(MAIL)
    m.allow_breakin = eligible
    if exit == 'x27':
        p = reply(m, 'X27')
    else:
        p = reply(m, 'V27')
        assert (p.kind, p.state) == (Kind.NORMAL, State.ISS_HOLD)
        m.keyed()
        at = m.last.stream_sample+T
        m.observe(word(counter_ack(m.last.status), at,
                       responds_to=m.last.stream_sample), now=at+round(.3*FS))
        p = m.decide(at+round(.3*FS))
    if eligible:
        assert (p.kind, p.status, p.payload) == (Kind.NORMAL, 1, MAIL[:40])
    else:
        while p.state == State.ISS_HOLD:
            assert (p.kind, p.payload) == (Kind.NORMAL, b'')
            m.keyed()
            assert m._hold_until == deadline
            p = m.decide(m.last.stream_sample+CYCLE-m.lead)
        assert (p.kind, p.status, p.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
        assert p.stream_sample <= deadline
        assert arq._buffer_raw == len(MAIL)
