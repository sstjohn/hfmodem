# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""IRS answers name the latest accepted target while unresolved slots stay unresolved."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Emission, P4Receiver, Role
from hfmodem.shrike.p4turn import Proposal, State
from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, body, machine, word
from hfmodem.tests.shrike.p4_chirp_deadline_support import replay


def unread(kind, at=FS):
    if kind == Kind.BODY:
        obs = body(at).observation
    else:
        obs = ack_frame(at, b'data', 0, kind).observation
    return RegisteredObservation(replace(obs, frame=None), Registered.NEW)


def test_long_ack_becomes_the_request_word():
    m, _ = machine(role='irs', normal_ack='parity')
    m.require_long_crc = True
    m.observe(ack_frame(FS, b'long', 0, Kind.LONG))
    due = FS + round(m.t[Kind.LONG]*FS)
    assert m.decide(due-m.lead).word == 'X27'
    m.keyed()
    assert m._last_body_ack[1] == 'X27' and m.ack_word is None
    m.observe(unread(Kind.LONG, FS+round(3.75*FS)))
    request = m.decide(FS+round(3.75*FS)+round(m.t[Kind.LONG]*FS)-m.lead)
    assert (request.word, request.purpose) == ('X27', None)


def test_body_ack_after_opening_updates_request_word():
    m, _ = machine(role='irs', normal_ack='parity')
    m.observe(body(FS, b'first'))
    assert m.decide(FS+round(m.t[Kind.BODY]*FS)-m.lead).word == 'X27'
    m.keyed()
    second = FS+round(1.25*FS)
    m.observe(body(second, b'second'))
    assert m.decide(second+round(m.t[Kind.BODY]*FS)-m.lead).word == 'V27'
    m.keyed()
    assert m._last_body_ack[1] == 'V27' and m.ack_word == 'X27'
    third = second+round(1.25*FS)
    m.observe(unread(Kind.BODY, third))
    request = m.decide(third+round(m.t[Kind.BODY]*FS)-m.lead)
    assert (request.word, request.purpose) == ('V27', None)


@pytest.mark.parametrize('normal_ack', ['v27', 'x27', 'parity', 'alternate-packet'])
def test_protocol3_body_request_uses_last_keyed_u27_without_changing_ack_history(normal_ack):
    m, _ = machine(role='irs', normal_ack=normal_ack, chirp_ack='x27', ack_53='U27')
    m.observe(body(FS, b'gateway', 0x61), now=FS+round(.810*FS))
    assert m.decide(FS+round(.920*FS)-m.lead).word == 'U27'
    m.keyed()
    history = m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack
    assert m._last_body_ack[1] == 'U27'
    at = FS+round(1.25*FS)
    m.observe(unread(Kind.BODY, at))
    request = m.decide(at+round(m.t[Kind.BODY]*FS)-m.lead)
    assert (request.word, request.purpose) == ('U27', None)
    m.keyed()
    assert (m.ack_word, m.acked_counter, m._last_ack[0], m._last_body_ack) == (history[0], history[1], history[2][0], history[3])


@pytest.mark.parametrize('normal_ack', ['v27', 'x27', 'parity', 'alternate-packet'])
def test_v53_u27_overrides_older_body_ack_for_repeat_request(normal_ack):
    m, _ = machine(role='irs', normal_ack=normal_ack, ack_53='U27')
    m.observe(body(FS, b'prior'))
    assert m.decide(FS+round(.920*FS)-m.lead).word == 'X27'
    m.keyed()
    at = FS+round(1.25*FS)
    control = replace(body(at).observation, kind=Kind.SHORT, frame=None, value=53)
    m.observe(RegisteredObservation(control, Registered.NEW))
    answer = m.decide(at+round(m.t[Kind.SHORT]*FS)-m.lead)
    assert (answer.word, answer.peer_control) == ('U27', 53)
    m.keyed()
    history = m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack
    assert m._last_body_ack[1] == 'X27'

    missed = at+round(1.25*FS)
    m.observe(unread(Kind.BODY, missed))
    request = m.decide(missed+round(m.t[Kind.BODY]*FS)-m.lead)
    assert (request.word, request.purpose) == ('U27', None)
    m.keyed()
    assert (m.ack_word, m.acked_counter, m._last_ack[0], m._last_body_ack) == (history[0], history[1], history[2][0], history[3])


def test_unread_normal_keeps_breakin_state_until_decode():
    m, _ = machine(role='irs')
    m.state = State.IRS_BREAKIN
    m.observe(unread(Kind.NORMAL))
    assert m.state == State.IRS_BREAKIN
    assert m.decide(FS+round(m.t[Kind.NORMAL]*FS)-m.lead) is None
    assert m.state == State.IRS_BREAKIN


def test_late_provisional_long_cancels_proposed_empty_request():
    m, _ = machine(role='irs')
    m.require_long_crc = True
    m.silence(round(3.096*FS))
    request = m.decide(round(3.096*FS))
    assert request is not None and request.purpose == None
    onset = round(2.25*FS)
    long = replace(unread(Kind.LONG, onset).observation, provisional=True)
    m.observe(RegisteredObservation(long, Registered.NEW), now=round(3.10*FS))
    assert m.decide(round(3.17*FS)) is None
    due = onset+round(m.t[Kind.LONG]*FS)
    request = m.decide(due-m.lead)
    assert (request.stream_sample, request.purpose) == (due, None)


def test_closed_unread_long_replaces_earlier_empty_request_with_own_answer():
    m, _ = machine(role='irs')
    m.require_long_crc = True
    m.observe(body(FS, b'prior'))
    assert m.decide(FS+round(.920*FS)-m.lead).word == 'X27'
    m.keyed()
    onset, now = round(2.25*FS), round(5.55*FS)
    m.silence(now)
    old = m.decide(now)
    assert old.purpose == None and old.stream_sample == round(5.67*FS)
    long = replace(unread(Kind.LONG, onset).observation,
                   provisional=False, duration_s=3.28)
    m.observe(RegisteredObservation(long, Registered.NEW), now=now)
    due = onset+round(m.t[Kind.LONG]*FS)
    request = m.decide(due-m.lead)
    assert (request.stream_sample, request.purpose) == (due, None)
    assert request.stream_sample < old.stream_sample
    sent = m.keyed()
    assert (sent.stream_sample, sent.word) == (due, 'X27')
    assert m.last is sent and m._next is None


def test_contradicted_word_closed_long_replaces_earlier_empty_request():
    m, _ = machine(role='irs')
    m.require_long_crc = True
    rx = P4Receiver(FS)
    m.observe(body(FS, b'prior'))
    assert m.decide(FS+round(.920*FS)-m.lead).word == 'X27'
    m.keyed()
    onset, now = round(2.25*FS), round(5.55*FS)
    head = replace(word('V27', onset, partial=True).observation, provisional=True)
    m.silence(now)
    old = m.decide(now)
    assert old.purpose == None and old.stream_sample == round(5.67*FS)
    m.observe(rx.register(head), now=now)
    closed = replace(unread(Kind.LONG, onset).observation,
                     provisional=False, duration_s=3.28)
    reg = rx.register(closed)
    assert reg.outcome == Registered.CONTRADICTED and reg.replaced == head
    m.observe(reg, now=now)
    due = onset+round(m.t[Kind.LONG]*FS)
    request = m.decide(due-m.lead)
    assert (request.stream_sample, request.purpose) == (due, None)
    assert request.stream_sample < old.stream_sample
    assert m.keyed() == request


@pytest.mark.parametrize('prior_offset', [0, -round(.030*FS)])
def test_decoded_late_long_replaces_queued_empty_request_and_defers_ack(prior_offset):
    m, _ = machine(role='irs', normal_ack='parity', prekey_lead=.049)
    m.require_long_crc = True
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append,
                                     supervisor=lambda *args: None)
    m.observe(body(FS+prior_offset, b'prior'))
    assert m.decide(FS+prior_offset+round(.920*FS)-m.lead).word == 'X27'
    m.keyed()

    onset = round(2.25*FS)
    due = onset+round(m.t[Kind.LONG]*FS)
    m.silence(round(5.55*FS))
    old = m.decide(round(5.55*FS))
    assert (old.stream_sample, old.purpose) == (round(5.67*FS)+prior_offset, None)

    now = due-m.lead+144
    assert due-now == 2208 and due-now >= 1796
    if prior_offset:
        assert old.stream_sample-now == 1728 < 1796
    m.observe(ack_frame(onset, b'long', 1, Kind.LONG), now=now)
    assert delivered == [b'prior', b'long']
    request = m.decide(now)
    assert (request.stream_sample, request.word, request.purpose) == (
        due, 'X27', None)
    assert request.stream_sample != old.stream_sample
    assert m._deferred is None
    assert request.ack_identity is not None
    assert m.keyed() == request
    assert m._next is None and m._deferred is None
    assert delivered == [b'prior', b'long']
    assert m.decide(due+round(3.75*FS)-m.lead) is None


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.BODY, Kind.LONG])
@pytest.mark.parametrize('form', ['provisional', 'closed', 'contradicted', 'decoded'])
@pytest.mark.parametrize('old_at', [5.64, 5.65, 5.67])
@pytest.mark.parametrize('decode_offset', ['early', 144, -1, 0, 1])
def test_registration_owns_queued_request_slot_and_device_notice(kind, form, old_at,
                                                                  decode_offset):
    m, _ = machine(role='irs', normal_ack='parity', prekey_lead=.049)
    m.require_long_crc = True
    m.observe(body(FS, b'prior'))
    m.decide(FS+round(.920*FS)-m.lead)
    m.keyed()
    due = round(5.65*FS)
    onset = due-round(m.t[kind]*FS)
    m._next = Proposal(Kind.WORD, round(old_at*FS), word='X27',
                       state=State.IRS_ANSWER, current_slot_accepted=False)
    now = (due-m.lead-1 if decode_offset == 'early' else
           due-m.lead+decode_offset if decode_offset == 144 else
           due-1796+decode_offset)
    if form == 'decoded':
        reg = body(onset, b'late') if kind == Kind.BODY else ack_frame(onset, b'late', 1, kind)
    else:
        obs = replace(unread(kind, onset).observation,
                      provisional=form == 'provisional')
        if form == 'contradicted':
            replaced = replace(word('V27', onset, partial=True).observation,
                               provisional=True)
            reg = RegisteredObservation(obs, Registered.CONTRADICTED, replaced)
        else:
            reg = RegisteredObservation(obs, Registered.NEW)
    m.observe(reg, now=now)
    decision = max(now, due-m.lead)
    proposal = m.decide(decision)
    assert proposal is not None
    expected_word = ('V27' if form == 'decoded' and kind != Kind.LONG else 'X27')
    assert (proposal.stream_sample, proposal.word) == (due, expected_word)
    assert proposal.purpose is None
    assert proposal.current_slot_accepted == (form == 'decoded')
    live = SimpleNamespace(pos=decision,
                           clamp_late=lambda at: max(0, decision+1796-at))
    grid = onair._MasterGrid(due, round(1.25*FS), 0,
                             packet_n=round(.244*FS), cs_n=round(.244*FS),
                             d_max_n=round(.1*FS))
    admitted = onair._keyable_slot(live, grid, 0, 0, listen=False, shift=False) == 0
    assert admitted == (due-decision >= 1796)
    if admitted:
        sent = m.keyed()
        assert (sent.stream_sample, sent.word) == (due, expected_word)
    else:
        history = m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack
        m.refused('deadline')
        assert m.last.stream_sample != due
        assert (m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack) == history
        if form == 'decoded':
            assert m._deferred[1]['due'] == due+round((3.75 if kind == Kind.LONG else 1.25)*FS)
        else:
            assert m._deferred is None


def test_closed_decoded_long_answer_preempts_queued_empty_request():
    m, _ = machine(role='irs')
    m.require_long_crc = True
    m.observe(body(FS, b'prior'))
    assert m.decide(FS+round(.920*FS)-m.lead).word == 'X27'
    m.keyed()
    onset, now = round(2.25*FS), round(5.55*FS)
    m.silence(now)
    old = m.decide(now)
    assert (old.stream_sample, old.purpose) == (round(5.67*FS), None)

    m.observe(ack_frame(onset, b'long', 1, Kind.LONG), now=now)
    due = onset+round(m.t[Kind.LONG]*FS)
    answer = m.decide(due-m.lead)
    assert (answer.stream_sample, answer.word, answer.purpose) == (due, 'X27', None)
    assert answer.ack_identity is not None
    assert m.keyed() == answer
    assert m._next is None and m.last.stream_sample != old.stream_sample


def test_request_does_not_mark_provisional_receive_as_acknowledged():
    rx = P4Receiver(FS)
    ack = Emission(Kind.WORD, Role.IRS, 0, .244)
    rx.note_emission(ack)
    provisional = replace(unread(Kind.NORMAL).observation, provisional=True)
    rx.observations.append(provisional)
    request = Emission(Kind.WORD, Role.IRS, FS+round(.920*FS), .244,
                       current_slot_accepted=False)
    rx.note_emission(request)
    assert provisional.stream_sample not in rx._answered
    assert rx.last_emission is request
    following = replace(provisional, stream_sample=FS+round(1.250*FS))
    assert rx._response_target(following) is None


def test_driver_empty_long_cycle_uses_its_due_not_a_later_raster():
    m, _ = machine(role='irs')
    key_at = round(43.150208333*FS)
    m.observe(body(key_at-round(.920*FS), b'held'))
    m.decide(key_at-m.lead)
    m.keyed(key_at)
    m._peer_cycle = 3.75
    decision = round(50.607854167*FS)
    due = decision + m.lead
    assert m.next_raster(decision) > due + round(3*FS)
    m.silence(decision, answer_due=due)
    request = m.decide(decision)
    assert (request.stream_sample, request.word, request.purpose) == (
        due, 'X27', None)


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.BODY, Kind.LONG])
@pytest.mark.parametrize('prior', [None, 'X27'])
def test_unaccepted_packet_requests_last_keyed_word_without_ack(kind, prior):
    m, _ = machine(role='irs', data='body', normal_ack='parity')
    if prior:
        m.observe(body(0, b'held'))
        m.decide(round(.920*FS)-m.lead)
        m.keyed()
    m.observe(unread(kind))
    due = FS + round(m.t[kind]*FS)
    p = m.decide(due-m.lead)
    assert p is not None and (p.kind, p.word, p.stream_sample) == (Kind.WORD, prior or 'V27', due)
    assert p.state == State.IRS_ANSWER and p.purpose is None
    assert not p.current_slot_accepted
    assert p.accepted_target == (0 if prior else None)
    assert m.unread == 1
    old = m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack, m.first_body
    m.keyed()
    assert (m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack, m.first_body) == old


@pytest.mark.parametrize('state', [State.ISS_LADDER, State.IRS_BREAKIN, State.IRS_CLOSE,
                                    State.ISS_HOLD])
def test_unaccepted_packet_does_not_request_outside_irs_answer(state):
    m, _ = machine(role='irs')
    m.state = state
    obs = unread(Kind.NORMAL).observation
    m.pending[FS] = dict(word=None, accepted=False, text=False, body=obs)
    assert m.decide(FS+round(.920*FS)-m.lead) is None


def test_empty_irs_cycle_repeats_last_word_on_peer_raster():
    m, _ = machine(role='irs')
    p = m.decide(0)
    assert p is None
    m.silence(0)
    p = m.decide(0)
    assert (p.word, p.purpose) == ('V27', None)
    m.keyed()
    at = p.stream_sample+FS
    m.observe(body(at, b'held'))
    m.decide(at+round(.920*FS)-m.lead)
    held = m.keyed()
    m._peer_cycle = 3.75
    due = held.stream_sample+round(3.75*FS)
    m.silence(due-m.lead, answer_due=due)
    q = m.decide(due-m.lead)
    assert (q.stream_sample, q.word, q.purpose) == (due, 'X27', None)


def test_provisional_long_owns_its_full_slot():
    m, _ = machine(role='irs', data='body')
    m.observe(unread(Kind.LONG))
    m.silence(FS+round(.920*FS)-m.lead)
    assert m.decide(FS+round(.920*FS)-m.lead) is None
    p = m.decide(FS+round(3.400*FS)-m.lead)
    assert (p.stream_sample, p.purpose) == (FS+round(3.400*FS), None)


def test_unread_y27_head_does_not_block_later_empty_cycles():
    m, _ = machine(role='irs')
    m.observe(body(0, b'held'))
    m.decide(round(.920*FS)-m.lead)
    m.keyed()
    m._blocked_redue = True
    m.pending[FS] = dict(word=None, accepted=False, text=False)
    m._decided.add(FS)
    first = FS+round(.920*FS)-m.lead
    m.silence(first)
    assert m.decide(first) is None
    following = first+round(1.25*FS)
    m.silence(following, answer_due=following+m.lead)
    p = m.decide(following)
    assert (p.word, p.stream_sample, p.purpose) == ('X27', following+m.lead, None)


def test_request_does_not_advance_parity_before_next_normal():
    m, _ = machine(role='irs', normal_ack='parity')
    first = ack_frame(FS, b'first', 0)
    m.observe(first)
    m.decide(FS+round(.920*FS)-m.lead)
    m.keyed()
    assert m.ack_word == 'X27'
    m.observe(unread(Kind.NORMAL, 2*FS))
    m.decide(2*FS+round(.920*FS)-m.lead)
    m.keyed()
    assert m.ack_word == 'X27'
    second = ack_frame(3*FS, b'second', 1)
    m.observe(second)
    p = m.decide(3*FS+round(.920*FS)-m.lead)
    assert (p.word, p.ack_reason) == ('V27', 'counter 1')


def test_irs_silent_experiment_still_keys_nothing():
    m, _ = machine(role='irs', irs_silent_cycles=8)
    m.observe(body(FS))
    at = FS+round(.920*FS)-m.lead
    assert m.decide(at) is None
    m.silence(at)
    assert m.decide(at) is None


def test_late_accepted_packet_answers_its_current_slot():
    m, _ = machine(role='irs', normal_ack='parity')
    frame = ack_frame(FS, b'first', 0)
    m.observe(frame)
    due = FS+round(.920*FS)
    answer = m.decide(due-m.lead+1)
    assert (answer.word, answer.purpose, answer.ack_identity) == (
        'X27', None, (frame.observation.frame.status.byte, frame.observation.frame.data))
    assert m._deferred is None and m.ack_word is None
    m.keyed()
    assert m.ack_word == 'X27'
    assert m.decide(due+round(1.25*FS)-m.lead) is None


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.BODY, Kind.LONG])
@pytest.mark.parametrize('prior_unread', [False, True])
@pytest.mark.parametrize('prior_ack,overrun', [(False, 144), (True, 144), (True, 528)])
@pytest.mark.parametrize('refuse', [False, True])
def test_feasible_late_packet_answers_before_driver_refusal(
        kind, prior_unread, prior_ack, overrun, refuse):
    m, _ = machine(role='irs', data='body', normal_ack='parity', prekey_lead=.049)
    m.require_long_crc = True
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *args: None)
    if prior_ack:
        m.observe(body(FS, b'prior'))
        assert m.decide(FS+round(.920*FS)-m.lead).word == 'X27'
        m.keyed()
    at = round(2.25*FS)
    due = at+round(m.t[kind]*FS)
    now = due-m.lead+overrun
    assert due-now >= 1796
    previous = unread(kind, at) if prior_unread else None
    if previous is not None:
        m.observe(previous, now=at)
    decoded = body(at, b'late') if kind == Kind.BODY else ack_frame(at, b'late', 1, kind)
    if previous is not None:
        decoded = RegisteredObservation(decoded.observation, Registered.ENRICHED, previous.observation)
    m.observe(decoded, now=now)
    expected_delivery = [b'prior', b'late'] if prior_ack else [b'late']
    assert delivered == expected_delivery
    history = m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack
    answer = m.decide(now)
    expected_word = 'X27' if kind == Kind.LONG or kind == Kind.BODY and not prior_ack else 'V27'
    identity = decoded.observation.frame.status.byte, decoded.observation.frame.data
    assert (answer.kind, answer.stream_sample, answer.word, answer.purpose, answer.ack_identity) == (
        Kind.WORD, due, expected_word, None, identity)
    assert m._deferred is None
    assert (m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack) == history
    next_due = due+round((3.75 if kind == Kind.LONG else 1.25)*FS)
    if refuse:
        m.refused('deadline')
        assert m._deferred[0] == at and m._deferred[1]['due'] == next_due
        assert (m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack) == history
        answer = m.decide(next_due-m.lead)
        assert (answer.stream_sample, answer.word, answer.ack_identity) == (next_due, expected_word, identity)
    assert m.keyed() == answer
    assert m._last_ack[0] == identity
    assert m.decide(next_due+round(1.25*FS)) is None
    assert delivered == expected_delivery


def test_k4msu_1104_keys_silent_cycles_unread_copies_then_c1_ack():
    result = replay('onair-1002-1104', stop=54*FS)
    keyed = [e for e in result['events'] if e['event'] == 'Keyed' and e['kind'] == Kind.WORD]
    for onset in (29.76+1.25*i for i in range(10)):
        word, = [e for e in keyed if abs(e['at']/FS-(onset+.920)) < .005]
        assert (word['word'], word['purpose']) == ('X27', None)
    for at in (43.178, 44.428):
        word, = [e for e in keyed if abs(e['at']/FS-at) < .005]
        assert (word['word'], word['purpose']) == ('X27', None)
    ack, = [e for e in keyed if abs(e['at']/FS-45.678) < .005]
    assert (ack['word'], ack['purpose']) == ('V27', None)
    answers = [e for e in result['events'] if e['event'] == 'normal-answer'
               and abs(e['at']/FS-44.758) < .01]
    assert answers and answers[-1]['previous'] == 'X27'
    assert not [e for e in result['events'] if e['event'] == 'Refused'
                and 29.76*FS <= e['at'] <= 45.68*FS]


def test_k4msu_0411_keys_each_unread_header_b_copy():
    result = replay('onair-1001-0411', stop=75*FS)
    keys = [e for e in result['events'] if e['event'] == 'Keyed' and e['kind'] == Kind.WORD
            and 28.49*FS <= e['at'] <= 73.17*FS]
    assert len(keys) == 36
    for i, keyed_word in enumerate(keys):
        assert abs(keyed_word['at']/FS-(29.413+1.25*i)) < .005
        assert (keyed_word['word'], keyed_word['purpose']) == ('X27', None)
    unread = [e for e in result['events'] if e['event'] == 'BREAKIN_UNREAD'
              and 28.49*FS <= e['at'] <= 72.25*FS]
    assert len(unread) == 36 and unread[-1]['count']-unread[0]['count'] == 35
    assert not [e for e in result['events'] if e['event'] == 'Refused'
                and 28.49*FS <= e['at'] <= 73.17*FS]
