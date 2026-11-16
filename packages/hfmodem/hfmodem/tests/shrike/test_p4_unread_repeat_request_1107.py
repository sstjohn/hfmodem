# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An unread counted packet slot keys the repeat-request word on grid, never silence.

WS8EOC 40 m, 2026-11-07: the gateway's greeting broke into our opening ladder
as Y27 + body at 17.242 s. Three copies were unread at their decisions and
three reply slots (18.18, 23.17, 24.42 s) went out with no key.
"""
from dataclasses import replace
import json

from hfmodem.tests.evidence import RECORD, require_input

import pytest

from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, body, machine, word
def short(at, value, header='B'):
    from hfmodem.shrike.p4observation import Observation
    return RegisteredObservation(Observation(
        Kind.SHORT, FS, 0, at, at, .758, 0., .9,
        value=value, header=header, walsh_peak=64), Registered.NEW)


GREETING = b'RMS Trimode 1.4.4.0\r\nW9SSJ has '
V38 = 60000
CYCLE = 60000


def opening():
    m, _ = machine(data='body', managed_application=True)
    m.seed_opening([dict(audio_start=0, value=7), dict(audio_start=V38, value=38)], [], values=(7, 38))
    return m


def head(m, at, first=False):
    m.observe(word('Y27', at, extent=.8, responds_to=V38 if first else 'none'), now=at+round(.30*FS))


def decide(m, at):
    decision = at + round(.92*FS) - m.lead
    p = m.decide(decision)
    if p is None:
        m.silence(decision, answer_due=at+round(.92*FS))
        p = m.decide(decision)
    return p


def test_unread_greeting_copies_each_key_v27_at_their_answer_instant():
    m = opening()
    first = V38 + round(.92*FS)
    for k in range(3):
        at = first + k*CYCLE
        head(m, at, first=k == 0)
        p = decide(m, at)
        assert p is not None, f'copy {k} left its reply slot silent'
        assert (p.word, p.stream_sample, p.purpose) == ('V27', at+round(.92*FS), 'repeat-request')
        assert p.ack_identity is None and not p.current_slot_accepted
        m.keyed()
        # The unread body decides the role once read; the opening stands.
        assert m.state == State.ISS_LADDER and m._opening
    assert m.unread == 3 and m.first_body


def test_a_copy_read_after_its_v27_is_acked_x27_on_the_repeat():
    m = opening()
    at = V38 + round(.92*FS)
    head(m, at, first=True)
    assert decide(m, at).word == 'V27'
    m.keyed()
    m.observe(body(at, GREETING), now=at+round(1.0*FS))
    repeat = at + CYCLE
    head(m, repeat)
    m.observe(body(repeat, GREETING), now=repeat+round(.5*FS))
    p = decide(m, repeat)
    assert (p.word, p.ack_reason, p.stream_sample) == ('X27', 'chain start', repeat+round(.92*FS))
    m.keyed()
    assert m.state == State.IRS_ANSWER and not m._opening


def test_a_changeover_request_read_after_its_v27_draws_our_grant():
    # 1002-1018: the opening's Y27 + body was an empty 0x40 request, read a cycle late.
    m = opening()
    at = V38 + round(.92*FS)
    head(m, at, first=True)
    assert decide(m, at).word == 'V27'
    m.keyed()
    repeat = at + CYCLE
    head(m, repeat)
    m.observe(body(repeat, b'', 0x40), now=repeat+round(.5*FS))
    p = decide(m, repeat)
    assert (p.kind, p.status, p.state, p.stream_sample) == (
        Kind.BODY, 0x40, State.ISS_GRANT, repeat+round(.92*FS))


def test_after_a_ladder_answer_with_no_counter_keys_the_chain_start_repeat_word():
    # No counted packet was acknowledged, so X27 accepts and V27 requests a repeat.
    m, _ = machine(role='irs')
    m.observe(short(FS, 13, 'A'), now=FS+round(.81*FS))
    assert m.decide(FS+round(.92*FS)-m.lead).word == 'X27'
    m.keyed()
    at = FS + CYCLE
    m.observe(word('Y27', at, extent=.8, responds_to='none'), now=at+round(.30*FS))
    unread = replace(body(at).observation, frame=None)
    m.observe(RegisteredObservation(unread, Registered.ENRICHED), now=at+round(.81*FS))
    p = decide(m, at)
    assert p is not None and p.word == 'V27'
    assert p.ack_identity is None and not p.current_slot_accepted
    m.keyed()
    assert m._ack_chain is None


@pytest.mark.parametrize('unread_body', [False, True])
@pytest.mark.parametrize('status', [0, 0x61])
def test_an_unread_y27_in_a_long_peer_cycle_keys_nothing_at_the_short_offset(status, unread_body):
    m, _ = machine(role='irs')
    m.require_long_crc = True
    m.observe(ack_frame(FS, b'prior', status, Kind.LONG if status == 0 else Kind.CHIRP), now=206400)
    m.decide(207648)
    m.keyed()
    m.observe(word('Y27', 228000, extent=.8, responds_to='none'), now=242400)
    if unread_body:
        m.observe(RegisteredObservation(replace(body(228000).observation, frame=None), Registered.ENRICHED),
                  now=266880)
    p = m.decide(268608)
    assert p is None or p.stream_sample != 228000 + round(.92*FS)


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
def test_a_later_long_registration_withdraws_the_opening_repeat_request(kind):
    m = opening()
    m.observe(word('Y27', 104880, extent=.8, responds_to=V38), now=104880+round(.30*FS))
    p = m.decide(145488)
    assert (p.word, p.stream_sample) == ('V27', 149040)
    long = replace(body(96000).observation, kind=kind, frame=None, provisional=True)
    m.observe(RegisteredObservation(long, Registered.NEW), now=145489)
    again = m.decide(145489)
    assert again is None or again.stream_sample != 149040
    assert not m.can_key(p)


def test_zero_lead_keys_the_repeat_request_at_its_answer_instant():
    m, _ = machine(data='body', managed_application=True, prekey_lead=0)
    m.seed_opening([dict(audio_start=0, value=7), dict(audio_start=V38, value=38)], [], values=(7, 38))
    at = V38 + round(.92*FS)
    head(m, at, first=True)
    p = m.decide(at+round(.92*FS))
    assert p is not None and (p.word, p.stream_sample) == ('V27', at+round(.92*FS))


def v53_held(**options):
    from hfmodem.tests.shrike.test_p4_v53_0929 import control
    m, _ = machine(role='irs', **options)
    m.state = State.IRS_ANSWER
    for at in (48000,):
        m.observe(RegisteredObservation(control(at), Registered.NEW), now=at+round(.81*FS))
        m.decide(at+44160-m.lead)
        m.keyed()
    return m


def test_an_expired_chirp_hypothesis_keeps_the_opening_repeat_request():
    m = opening()
    m.observe(word('Y27', 104880, extent=.8, responds_to=V38), now=119280)
    p = m.decide(145488)
    assert (p.word, p.stream_sample) == ('V27', 149040)
    obs = replace(body(96000).observation, kind=Kind.CHIRP, frame=None, provisional=True)
    m.observe(RegisteredObservation(obs, Registered.EXPIRED, replaced=obs), now=145489)
    assert m._next is p and m.can_key(p)


def test_a_held_u27_still_answers_an_unread_chirp():
    m = v53_held(peer_ack={53: 'U27'})
    chirp = replace(body(108000).observation, kind=Kind.CHIRP, frame=None)
    m.observe(RegisteredObservation(chirp, Registered.NEW), now=108000+round(3.3*FS))
    p = m.decide(267648)
    if p is None:
        m.silence(267648, answer_due=271200)
        p = m.decide(267648)
    assert p is not None and (p.word, p.stream_sample) == ('V27', 271200)
    assert p.purpose == 'repeat-request' and p.ack_identity is None
    assert p.peer_control is None


@pytest.mark.parametrize('first,expected', [('x27', 'V27'), ('v27', 'X27')])
def test_opening_unread_uses_opposite_chain_start_word(first, expected):
    probe = RECORD / 'operator-data/captures/onair-20261107T152917427998Z-64c4fbe2-3ac9-4306-899e-2b68aca47cf6/p4-probe.json'
    require_input(probe)
    recorded = json.loads(probe.read_text())
    emission = recorded['emissions'][0]
    assert emission['value'] == 7
    greeting = next(r for r in recorded['heard'] if r.get('kind') == 'body')
    assert bytes.fromhex(greeting['data']) == GREETING and greeting['status'] == 0
    at = next(r['onset_sample'] for r in recorded['heard']
              if r.get('word') == 'Y27' and not r['provisional'])
    m, _ = machine(data='body', managed_application=True, first_ack=first)
    m.seed_opening([emission], [], values=(7, 38))
    m.observe(word('Y27', at, extent=.8, responds_to=emission['audio_start']),
              now=at+round(.30*FS))
    proposal = decide(m, at)
    assert (proposal.word, proposal.purpose) == (expected, 'repeat-request')
    assert proposal.ack_identity is None and not proposal.current_slot_accepted
    m.keyed()
    assert m._ack_chain is None and m.state == State.ISS_LADDER


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('kind,policy', [(Kind.CHIRP, 'chirp_ack'), (Kind.NORMAL, 'normal_ack')])
@pytest.mark.parametrize('held', ['X27', 'V27'])
def test_parity_held_v53_keys_on_grid_repeat_request(late, kind, policy, held):
    m = v53_held(peer_ack={53: held}, **{policy: 'parity'})
    at = FS + CYCLE
    due = at + round(m.t[kind]*FS)
    if late:
        m.silence(due-m.lead, answer_due=due)
        previous = m.decide(due-m.lead)
        assert previous is not None and previous.word == ('V27' if kind == Kind.CHIRP else 'U27')
    unread = replace(body(at).observation, kind=kind, frame=None)
    m.observe(RegisteredObservation(unread, Registered.NEW), now=due-m.lead+int(late))
    proposal = m.decide(due-m.lead+int(late))
    assert proposal is not None and (proposal.word, proposal.stream_sample) == ('U27', due)
    assert proposal.purpose == 'repeat-request' and proposal.peer_control is None
    assert proposal.ack_identity is None and not proposal.current_slot_accepted
    m.observe(RegisteredObservation(unread, Registered.ENRICHED), now=due-m.lead+2)
    assert m.can_key(proposal)
    assert m.decide(due-m.lead+2).word == 'U27'
    if late:
        assert m.can_key(previous) is (previous.purpose == 'repeat-request')
    assert m.keyed().word == 'U27'


@pytest.mark.parametrize('kind,policy', [(Kind.CHIRP, 'chirp_ack'), (Kind.NORMAL, 'normal_ack')])
def test_parity_control_replay_is_replaced_by_unread_repeat_request(kind, policy):
    m = v53_held(peer_ack={53: 'X27'}, **{policy: 'parity'})
    at = FS + CYCLE
    due = at + round(m.t[kind]*FS)
    replay = replace(m.last, stream_sample=due, current_slot_accepted=False)
    m._next = replay
    unread = replace(body(at).observation, kind=kind, frame=None)
    m.observe(RegisteredObservation(unread, Registered.NEW), now=due-m.lead)
    assert not m.can_key(replay)
    proposal = m.decide(due-m.lead)
    assert proposal is not None and (proposal.word, proposal.stream_sample) == ('U27', due)
    assert m.can_key(proposal)
    assert m.keyed().word == 'U27'


@pytest.mark.parametrize('kind,policy,counter', [(Kind.CHIRP, 'chirp_ack', 1),
                                                (Kind.NORMAL, 'normal_ack', 0)])
@pytest.mark.parametrize('previous', ['X27', 'V27'])
def test_parity_control_mid_chain_keeps_last_word(kind, policy, counter, previous):
    from hfmodem.tests.shrike.test_p4turn import propose_ack
    m, _ = machine(role='irs', **{policy: 'parity'})
    counter ^= previous == 'V27'
    assert propose_ack(m, ack_frame(FS, b'packet', counter, kind)).word == previous
    m.keyed()
    control_at = 6*FS
    assert propose_ack(m, short(control_at, 13)).word == 'X27'
    m.keyed()
    at = control_at+CYCLE
    due = at+round(m.t[kind]*FS)
    unread = replace(body(at).observation, kind=kind, frame=None)
    m.observe(RegisteredObservation(unread, Registered.NEW), now=due-m.lead)
    proposal = m.decide(due-m.lead)
    assert proposal is not None and (proposal.word, proposal.stream_sample) == (previous, due)
    assert proposal.purpose == 'repeat-request' and proposal.peer_control is None
    assert m.keyed().word == previous
    assert m._ack_chain == (counter, previous)


@pytest.mark.parametrize('kind,policy,form,history', [
    (kind, policy, form, history)
    for kind, policy, form in [(Kind.CHIRP, 'chirp_ack', None),
                              (Kind.NORMAL, 'normal_ack', None),
                              (Kind.LONG, 'long_ack', None),
                              (Kind.LONG, 'long_ack', 'SL5-long')]
    for history in ('control', 'form', 'turn', 'none')
    if kind != Kind.NORMAL or history != 'form'])
@pytest.mark.parametrize('counter', [0, 1])
def test_parity_chain_start_requests_unread_data_with_u27(kind, policy, form, counter, history):
    from hfmodem.tests.shrike.test_p4turn import propose_ack
    options = {policy: 'counter-even-t27' if kind == Kind.LONG else 'parity'}
    m, _ = machine(role='irs', **options)
    m.require_long_crc = True
    if history in ('form', 'turn'):
        prior_kind = Kind.NORMAL if history == 'form' else kind
        propose_ack(m, ack_frame(FS, b'prior', 2, prior_kind))
        m.keyed()
        if history == 'turn':
            m._open_receive_epoch(6*FS)
    if history == 'control':
        assert propose_ack(m, short(6*FS, 13)).word == 'X27'
        m.keyed()
    at = 8*FS
    packet = ack_frame(at, b'packet', 0x60 | counter, kind)
    packet = replace(packet, observation=replace(packet.observation, speedlevel=form))
    unread = replace(packet.observation, frame=None)
    due = at+round(m.t[kind]*FS)
    m.observe(RegisteredObservation(unread, Registered.NEW), now=due-m.lead)
    proposal = m.decide(due-m.lead)
    assert proposal is not None and (proposal.word, proposal.stream_sample) == ('U27', due)
    assert proposal.ack_identity is None and not proposal.current_slot_accepted
    m.keyed()
    again = replace(packet, observation=replace(packet.observation,
                    stream_sample=at+round((3.75 if kind != Kind.NORMAL else 1.25)*FS)))
    ack = propose_ack(m, again)
    expected = ('T27' if counter == 0 else 'X27') if kind == Kind.LONG else (
        'V27' if counter == 0 else 'X27') if kind == Kind.CHIRP else (
        'X27' if counter == 0 else 'V27')
    assert ack is not None and ack.word == expected and ack.word != proposal.word
