# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K0NTS 11-02: the CMS prompt arrived as an extended changeover, status 0xc3."""

from dataclasses import replace

import pytest

from hfmodem.shrike import spec
from hfmodem.shrike.p4host import P4Host, P4StreamFailure
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_greeting_resegment_1010 import packet, receive, recording, stream
from hfmodem.tests.shrike.test_p4turn import FS

K0NTS = [
    (Kind.BODY, 53.519, 0x00, bytes.fromhex('547279696e67206563322d332d3230382d3132302d3235352e636f6d707574')),
    (Kind.NORMAL, 54.769, 0x01, bytes.fromhex('652d312e616d617a6f6e6177732e636f6d0d2a2a2a20573953534a20436f6e6e656374656420746f')),
    (Kind.NORMAL, 56.019, 0x02, bytes.fromhex('20434d530d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a2031373338373436360d')),
    (Kind.NORMAL, 57.269, 0xc3, bytes.fromhex('010101434d5320766961204b304e5453203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
]
PROMPT = b'CMS via K0NTS >\r'
TEXT = (b'Trying ec2-3-208-120-255.compute-1.amazonaws.com\r*** W9SSJ Connected to CMS\r'
        b'[WL2K-5.0-B2FWIHJM$]\r;PQ: 17387466\r' + PROMPT)


def test_recorded_extended_changeover_delivers_the_prompt_once_and_grants():
    m, events = recording()
    answers = [receive(m, *row) for row in K0NTS]
    assert stream(events) == TEXT
    accepted = [e for e in m.link_events if e['event'] == 'source-accept']
    assert accepted[-1]['counter'] == 3 and accepted[-1]['new_source_bytes'] == len(PROMPT)
    assert dict(event='source-extension', at=round(57.269*FS), status=0xc3,
                prefix='010101') in m.link_events
    assert not m.adapter.failure and not m._closing_failure
    # The stock peer handed over with the prompt; answer as for the plain 0x43 prompt.
    assert answers[-1].kind == Kind.BODY and answers[-1].status == 0x40
    assert m.state == State.ISS_GRANT
    m.link_down('peer silent')
    assert stream(events) == TEXT


def test_extended_changeover_matches_the_plain_changeover_answer():
    def run(status, data):
        m, events = recording()
        answers = [receive(m, *row) for row in K0NTS[:3]]
        answers.append(receive(m, Kind.NORMAL, 57.269, status, data))
        return [(a.kind, a.status, a.word) for a in answers], m.state, stream(events)

    *_, (_, _, _, data) = K0NTS
    plain = data[3:]+b'\x1e'*3
    assert run(0xc3, data) == run(0x43, plain)


@pytest.mark.parametrize('queued', [b'', b';PR: 1\rFF\r'])
@pytest.mark.parametrize('status', [0xc3, 0x43])
def test_changeover_repeated_after_the_grant_is_not_delivered_again(status, queued):
    m, events = recording()
    data = K0NTS[3][3] if status & 0x80 else K0NTS[3][3][3:] + b'\x1e'*3
    for row in K0NTS[:3]:
        receive(m, *row)
    first = receive(m, Kind.NORMAL, 57.269, status, data)
    m.outbox.arq.on_host_data(queued)
    for at in (58.519, 59.769):
        again = receive(m, Kind.NORMAL, at, status, data)
        assert (again.kind, again.status, again.payload) == (first.kind, first.status, b'')
        assert m.state == State.ISS_GRANT
    assert stream(events) == TEXT


@pytest.mark.parametrize('first,again', [(0xc3, 0x43), (0x43, 0xc3), (0xc3, 0x03)])
def test_changeover_repeated_in_another_form_after_the_grant_is_not_delivered_again(first, again):
    m, events = recording()
    prompt = K0NTS[3][3]
    for row in K0NTS[:3]:
        receive(m, *row)
    def form(status):
        return prompt if status & 0x80 else prompt[3:] + b'\x1e'*3
    receive(m, Kind.NORMAL, 57.269, first, form(first))
    receive(m, Kind.NORMAL, 58.519, again, form(again))
    assert stream(events) == TEXT


def granted():
    m, events = recording()
    for row in K0NTS:
        receive(m, *row)
    return m, events


def test_repeat_window_survives_an_acked_empty_control():
    m, events = granted()
    receive(m, Kind.NORMAL, 58.519, 0x83, b'\x01\x01\x01'.ljust(40, b'\x1e'))
    receive(m, Kind.NORMAL, 59.769, *K0NTS[3][2:])
    assert stream(events) == TEXT


@pytest.mark.parametrize('status,data', [(0x03, b'CMS via K0NTS >\rFC'),
                                         (0x02, PROMPT)])
def test_new_source_or_counter_after_the_grant_is_delivered(status, data):
    m, events = granted()
    receive(m, Kind.NORMAL, 58.519, status, data.ljust(40, b'\x1e'))
    assert stream(events)[len(TEXT):]
    assert not m.adapter.failure


def test_resumed_counted_source_ends_the_repeat_window():
    m, events = granted()
    sent = b''
    for i, (counter, line) in enumerate([(0, b'A\r'), (1, b'B\r'), (2, b'C\r'), (3, PROMPT)]):
        sent += line
        answer = receive(m, Kind.NORMAL, 58.519+1.25*i, counter, line.ljust(40, b'\x1e'))
        assert answer.ack_identity == (counter, line.ljust(40, b'\x1e'))
    assert stream(events) == TEXT + sent


@pytest.mark.parametrize('status,data', [(0x47, PROMPT), (0xc3, PROMPT)])
def test_unsupported_repeat_after_the_grant_still_reaches_host_validation(status, data):
    m, events = granted()
    receive(m, Kind.NORMAL, 58.519, status, data.ljust(40, b'\x1e'))
    assert m.adapter.failure and stream(events) == TEXT


def answered(m, reg, responds_to):
    at = reg.observation.stream_sample
    m.observe(replace(reg, responds_to=responds_to), now=at)
    proposal = m.decide(at+round(m.t[Kind.BODY]*FS)-m.lead)
    m.keyed()
    return proposal


def test_extended_changeover_body_after_the_grant_is_a_handoff_and_ends_the_repeat_window():
    m, events = recording()
    for row in K0NTS:
        grant = receive(m, *row)
    answered(m, packet(Kind.BODY, 58.519, 0xc0, b'\x01\x01\x01;FW: W9SSJ\r', 31),
             grant.stream_sample)
    assert stream(events) == TEXT + b';FW: W9SSJ\r'
    assert m._handoff_ticket is None
    receive(m, *K0NTS[3][:1], 61.019, *K0NTS[3][2:])
    assert stream(events) == TEXT + b';FW: W9SSJ\r' + PROMPT


@pytest.mark.parametrize('extended', [False, True])
def test_empty_extended_grant_reply_starts_our_data_turn_like_the_plain_one(extended):
    def run(status, data):
        m, events = recording()
        for row in K0NTS:
            grant = receive(m, *row)
        m.outbox.arq.on_host_data(b';PR: 1\rFF\r')
        grant = receive(m, Kind.NORMAL, 58.519, *K0NTS[3][2:])
        reply = answered(m, packet(Kind.BODY, 59.769, status, data, 31), grant.stream_sample)
        return (reply.kind, reply.state, reply.payload), m.state, stream(events)

    plain = run(0x40, b'')
    assert plain[0][1] == State.ISS_DATA_ACK and plain[0][2]
    assert run(0xc0 if extended else 0x40, b'\x01\x01\x01' if extended else b'') == plain


def test_recorded_extended_changeover_repeat_is_a_duplicate():
    adapter = P4Host()
    for kind, _, status, data in K0NTS[1:3]:
        adapter.accept(packet(kind, 0, status, data).observation.frame, counted=True)
    frame = packet(Kind.NORMAL, 0, 0xc3, K0NTS[3][3]).observation.frame
    assert adapter.accept(frame, counted=True) == PROMPT
    assert adapter.result.new_source_bytes == len(PROMPT)
    assert adapter.accept(frame, counted=True) is None
    assert adapter.result.duplicate and adapter.result.replayed_source_bytes == len(PROMPT)


@pytest.mark.parametrize('status', [0xa0, 0xa3, 0xc7, 0xe7])
def test_extended_close_and_source_modes_stay_refused(status):
    with pytest.raises(P4StreamFailure):
        P4Host().accept(packet(Kind.NORMAL, 0, status, K0NTS[3][3]).observation.frame)


@pytest.mark.parametrize('prefix', [b'\x01\x01', b'\x01\x01\x02', b'\x02\x01\x01'])
def test_extended_changeover_needs_the_exact_prefix(prefix):
    data = (prefix + PROMPT).ljust(40, bytes([spec.IDLE]))
    with pytest.raises(P4StreamFailure):
        P4Host().accept(packet(Kind.NORMAL, 0, 0xc3, data).observation.frame)
