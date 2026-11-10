# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Modeled observation/key contracts; no recording or RF-success claim."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import body, machine, word

CYCLE = 60000
ANSWER = 44160
DATA = bytes(range(65, 91)) * 4


def opening(**options):
    m, a = machine(role='irs', data='body', ladder='none',
                   wall_clock=lambda: 0., **options)
    a.on_host_data(DATA)
    m._body(0, 0x40, State.IRS_BREAKIN)
    grant = m.keyed()
    m.observe(word('V27', ANSWER, responds_to=grant.stream_sample))
    p = m.keyed()
    assert (p.kind, p.status, m.counter) == (Kind.BODY, 0x40, 0)
    return m, a, p


def answer(m, p, name, **changes):
    reg = word(name, p.stream_sample + ANSWER, responds_to=p.stream_sample)
    if changes:
        reg = replace(reg, **changes)
    m.observe(reg)
    return reg


def silent_proposal(m):
    decision = m.last.stream_sample + CYCLE - m.lead
    m.silence(decision)
    return m.decide(decision)


def exhaust_after_first_answer(m, p):
    answer(m, p, 'V27')
    evidence = m._held_word_yield
    assert evidence is not None
    first_generation = m._peer_generation
    keys = [p]
    keys.append(m.keyed())
    while len(keys) < 8:
        assert silent_proposal(m) is not None
        keys.append(m.keyed())
    assert len(keys) == 8 and m.data_repeats == 7
    assert all((q.kind, q.status, q.payload) == (p.kind, p.status, p.payload) for q in keys)
    assert m._peer_generation > first_generation
    assert m._held_word_yield is evidence
    close = silent_proposal(m)
    assert close.state == State.ISS_CLOSE
    assert m.close_context.held_word_yield is evidence
    return evidence, close


def test_early_seed_survives_silent_retries_refused_close_and_close_keys():
    m, a, p = opening()
    evidence, close = exhaust_after_first_answer(m, p)
    assert a._buffer_raw == len(DATA) and bytes(a._outbuf) == DATA
    assert m.counter == 0 and not m.outbox.inflight
    context = m.close_context
    last = m.last
    for _ in range(2):
        m.refused('modeled final pre-key refusal')
        assert m.last is last and m.close_context is context
        assert context.held_word_yield is evidence
        assert a._buffer_raw == len(DATA) and m.counter == 0
    close = m.keyed()
    assert m._close_held_v27 and m.state == State.ISS_CLOSE
    answer(m, close, 'V27')
    assert silent_proposal(m).value == 37
    short = m.keyed()
    answer(m, short, 'V27')
    assert m.state != State.DOWN and m.reason != 'close acknowledged'
    assert context.held_word_yield is evidence
    assert a._buffer_raw == len(DATA)


@pytest.mark.parametrize('name', ['V27', 'T27', 'U27'])
def test_unaccepted_answers_keep_budget_and_late_real_ack_commits_once(name):
    m, a, first = opening()
    p = first
    for _ in range(4):
        answer(m, p, name)
        p = m.keyed()
        assert (p.kind, p.status, p.payload) == (first.kind, first.status, first.payload)
        assert a._buffer_raw == len(DATA) and m.counter == 0
        assert not m._data_refusals and m.close_context is None
    reg = answer(m, p, 'X27')
    assert m.counter == 1 and a._buffer_raw == len(DATA)-len(first.payload)
    assert m._held_word_yield is None and m._data_episode is None
    m.observe(replace(reg, outcome=Registered.DUPLICATE))
    assert m.counter == 1 and a._buffer_raw == len(DATA)-len(first.payload)
    q = m.keyed()
    assert q.kind == Kind.NORMAL and q.status == 1


@pytest.mark.parametrize('invalid', ['weak', 'provisional', 'unexpected', 'expired',
                                     'duplicate', 'unrelated', 'stale-generation'])
def test_invalid_seed_never_creates_episode_evidence(invalid):
    m, a, p = opening()
    reg = word('V27', p.stream_sample+ANSWER, responds_to=p.stream_sample)
    if invalid in ('weak', 'provisional', 'unexpected'):
        reg = replace(reg, observation=replace(reg.observation, **{invalid: True}))
    elif invalid in ('expired', 'duplicate'):
        reg = replace(reg, outcome=Registered.EXPIRED if invalid == 'expired' else Registered.DUPLICATE)
    elif invalid == 'unrelated':
        reg = replace(reg, responds_to=None)
    else:
        owner = SimpleNamespace(
            train_lease=SimpleNamespace(generation=12, active=p.stream_sample),
            _answered_response=lambda *_: False)
        reg = replace(reg, response_generation=11, response_owner=owner)
    m.observe(reg)
    assert m._held_word_yield is None and not m._data_refusals
    assert m.counter == 0 and a._buffer_raw == len(DATA)


def test_carried_seed_does_not_admit_fresh_stale_ack():
    m, a, p = opening()
    answer(m, p, 'V27')
    held = m._held_word_yield
    q = m.keyed()
    owner = SimpleNamespace(
        train_lease=SimpleNamespace(generation=14, active=q.stream_sample),
        _answered_response=lambda *_: False)
    answer(m, q, 'X27', response_generation=13, response_owner=owner)
    assert m._held_word_yield is held
    assert m.counter == 0 and a._buffer_raw == len(DATA)
    assert silent_proposal(m).payload == p.payload


def test_new_turn_with_same_bytes_is_not_the_old_episode():
    m, a, p = opening()
    answer(m, p, 'V27')
    old = m._held_word_yield
    m._begin_data_turn(p.stream_sample+2*CYCLE, word_yield=True)
    q = m.keyed()
    assert (q.kind, q.status, q.payload) == (p.kind, p.status, p.payload)
    assert m._held_word_yield is None and m._data_episode != old.episode
    assert m._data_episode.serial > old.episode.serial
    assert a._buffer_raw == len(DATA) and m.counter == 0


@pytest.mark.parametrize('change', ['status', 'field', 'form'])
def test_different_actual_data_identity_retires_seed(change):
    m, _, p = opening()
    answer(m, p, 'V27')
    q = m._next
    if change == 'status':
        q = replace(q, status=1)
    elif change == 'field':
        q = replace(q, payload=bytes(reversed(q.payload)))
    else:
        q = replace(q, kind=Kind.NORMAL)
    m._next = q
    m.keyed()
    assert m._held_word_yield is None


def test_changed_qualified_word_retires_seed_before_close():
    m, a, p = opening()
    answer(m, p, 'V27')
    p = m.keyed()
    answer(m, p, 'T27')
    assert m._held_word_yield is None
    while m.data_repeats < 7:
        p = m.keyed()
        if m.data_repeats < 7:
            silent_proposal(m)
    close = silent_proposal(m)
    assert close.state == State.ISS_CLOSE and m.close_context.held_word_yield is None
    close = m.keyed()
    assert not m._close_held_v27
    assert a._buffer_raw == len(DATA)


def test_mixed_words_only_count_z27_as_accelerated_refusal():
    m, a, p = opening()
    for name, refusals in [('Z27', 1), ('V27', 1), ('T27', 1), ('U27', 1)]:
        answer(m, p, name)
        assert len(m._data_refusals) == refusals
        p = m.keyed()
        assert p.state != State.ISS_CLOSE
    answer(m, p, 'Z27')
    close = m.keyed()
    assert close.state == State.ISS_CLOSE and len(m._data_refusals) == 2
    assert m.counter == 0 and a._buffer_raw == len(DATA)


def test_refused_retry_does_not_spend_key_credit_or_change_evidence():
    m, a, p = opening()
    answer(m, p, 'V27')
    held = m._held_word_yield
    due = m._next.stream_sample
    generation = m._peer_generation
    m.refused('modeled final pre-key refusal')
    assert m.last is p and m.data_repeats == 0 and m._peer_generation == generation
    assert m._next.stream_sample > due and m._held_word_yield is held
    assert m.counter == 0 and a._buffer_raw == len(DATA)
    q = m.keyed()
    assert m.data_repeats == 1 and q.payload == p.payload


def test_checked_body_reply_clears_episode_and_keeps_exact_credit():
    m, a, p = opening()
    answer(m, p, 'V27')
    q = m.keyed()
    reg = replace(body(q.stream_sample+ANSWER, status=0x40), responds_to=q.stream_sample)
    m.observe(reg)
    assert m._held_word_yield is None and m._data_episode is None
    assert m.counter == 1 and a._buffer_raw == len(DATA)-len(p.payload)
    m.observe(replace(reg, outcome=Registered.DUPLICATE))
    assert a._buffer_raw == len(DATA)-len(p.payload)


def test_link_end_retires_seed_without_credit():
    m, a, p = opening()
    answer(m, p, 'V27')
    m.link_down('modeled session ceiling')
    assert m._held_word_yield is None and m._data_episode is None
    assert m.state == State.DOWN and a._buffer_raw == len(DATA)
    assert bytes(a._outbuf) == DATA and m.counter == 0


def test_ladder_key_retires_data_seed_without_extending_its_scope():
    m, a, p = opening()
    answer(m, p, 'V27')
    # Supply the existing ladder only for this distinct semantic transition.
    m.ladder = (13, 60)
    m._short(p.stream_sample+CYCLE, 0)
    q = m.keyed()
    assert q.kind == Kind.SHORT
    assert m._held_word_yield is None and m._data_episode is None
    assert m.counter == 0 and a._buffer_raw == len(DATA)
