# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""`--p4-normal-ack x27`: every new body or normal short is acked X27; an unread slot asks V27.

WS8EOC, 2026-11-13 02:34Z: c0 advanced after X27; c1 was read cleanly and acked
V27 under the toggle 14 times, and the gateway resent the identical c1 each time.
"""
from dataclasses import replace

import pytest

from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import (
    FS, ack_frame, ack_greeting, body, machine, propose_ack, word)
from hfmodem.tests.shrike.test_p4_unread_repeat_request_1107 import V38, decide


def key_packet(turn, at, status, kind=Kind.NORMAL):
    answer = propose_ack(turn, ack_frame(at, b'packet', status, kind))
    assert answer is not None
    turn.keyed()
    return answer


@pytest.mark.parametrize('first', ['x27', 'v27'])
def test_ws8eoc_c1_copies_are_each_acked_x27(first):
    turn, _ = machine(role='irs', normal_ack='x27', first_ack=first)
    assert key_packet(turn, FS, 0, Kind.BODY).word == 'X27'
    for k in range(14):
        answer = key_packet(turn, 2*FS + k*60000, 1)
        assert (answer.word, answer.ack_reason) == ('X27', 'x27 every new packet')
    assert key_packet(turn, 2*FS + 14*60000, 2).word == 'X27'


@pytest.mark.parametrize('arm', ['1151', '2200'])
def test_recorded_greeting_is_all_x27(arm):
    turn, _ = machine(role='irs', normal_ack='x27')
    words = []
    for frame in ack_greeting(arm):
        words.append(propose_ack(turn, frame).word)
        turn.keyed()
    assert words == ['X27'] * len(words)


@pytest.mark.parametrize('read', [0, 1, 2])
def test_unread_slot_asks_v27_not_a_held_x27(read):
    turn, _ = machine(role='irs', normal_ack='x27')
    for counter in range(read):
        key_packet(turn, FS + counter*60000, counter, Kind.BODY if counter == 0 else Kind.NORMAL)
    at = 480000
    turn.observe(RegisteredObservation(replace(body(at).observation, kind=Kind.NORMAL, frame=None),
                                       Registered.NEW))
    reply = turn.decide(at + round(.92*FS) - turn.lead)
    assert (reply.word, reply.purpose) == ('V27', 'repeat-request')
    assert reply.ack_identity is None and not reply.current_slot_accepted


def test_a_packet_read_after_its_v27_is_acked_x27():
    turn, _ = machine(role='irs', normal_ack='x27')
    key_packet(turn, FS, 0, Kind.BODY)
    at = 300000
    turn.observe(RegisteredObservation(replace(body(at).observation, kind=Kind.NORMAL, frame=None),
                                       Registered.NEW))
    assert turn.decide(at + round(.92*FS) - turn.lead).word == 'V27'
    turn.keyed()
    assert key_packet(turn, at + 60000, 1).word == 'X27'


@pytest.mark.parametrize('first', ['x27', 'v27'])
def test_opening_unread_greeting_asks_v27(first):
    m, _ = machine(data='body', managed_application=True, normal_ack='x27', first_ack=first)
    m.seed_opening([dict(audio_start=0, value=7), dict(audio_start=V38, value=38)], [],
                   values=(7, 38))
    at = V38 + round(.92*FS)
    m.observe(word('Y27', at, extent=.8, responds_to=V38), now=at + round(.30*FS))
    proposal = decide(m, at)
    assert (proposal.word, proposal.purpose) == ('V27', 'repeat-request')
    m.keyed()
    assert m.state == State.ISS_LADDER


def test_iss_still_expects_the_peer_toggle():
    toggle, _ = machine(data='body', normal_ack='toggle')
    x27, _ = machine(data='body', normal_ack='x27')
    for turn in (toggle, x27):
        turn._data_packet = replace(body(FS).observation, kind=Kind.BODY)
        turn._last_data_ack, turn._last_data_form = 'X27', 'normal short'
    assert x27._expected_data_ack() == toggle._expected_data_ack() == 'V27'


def test_parser_accepts_x27():
    from hfmodem.shrike import onair
    assert onair.build_parser().parse_args(['--p4-normal-ack', 'x27']).p4_normal_ack == 'x27'
