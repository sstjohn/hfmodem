# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.shrike import p4frame, p4robustrx, spec
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import Proposal, State, T
from hfmodem.tests.shrike.test_p4turn import FS, machine
from hfmodem.tests.shrike.test_p4robustrx import recorded
from hfmodem.winlink import B2FSession

ROOT = Path(__file__).resolve().parents[5]
GREETING = ROOT/'working/p4-normal-budget-0929/greeting-fields.json'
if not (GREETING).exists():
    pytest.skip('working/p4-normal-budget-0929/greeting-fields.json is not in this tree', allow_module_level=True)
ROWS = json.loads(GREETING.read_text())['frames']
PREFIX = bytes.fromhex(ROWS[0]['field_hex'])


def packet(kind, at, data, status=0, width=None):
    width = width or (31 if kind == Kind.BODY else 40)
    data = data.ljust(width, b'\x1e')
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                 data_bytes=len(data))
    obs = Observation(kind, FS, 0, at, at, .796 if kind == Kind.BODY else 3.28,
                      0., .8, frame=frame)
    return RegisteredObservation(obs, Registered.ENRICHED if kind == Kind.BODY else Registered.NEW)


def receive(m, reg, *, key=True):
    at = reg.observation.stream_sample
    due = at+round(T[reg.observation.kind]*FS)
    m.observe(reg, now=due-m.lead-480)
    proposal = m.decide(due-m.lead)
    if key and proposal is not None:
        m.keyed()
    return proposal


def negotiate(m, value=49, *, key=True):
    control = Observation(Kind.SHORT, FS, 0, 3*FS+value, 3*FS+value, .758, 0., .8,
                          value=value, header='B', walsh_peak=64)
    m.observe(RegisteredObservation(control, Registered.NEW), now=3*FS+38000)
    proposal = m.decide(control.stream_sample+44160-m.lead)
    assert proposal.word == ('U27' if value in (53, 60, 61) else 'X27')
    if key:
        m.keyed()


def opening(prefix=PREFIX, *, negotiated=True, status=0, width=31, turn=None):
    m = turn or machine(data='body')[0]
    received = []
    m.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    first = packet(Kind.BODY, FS, prefix, status, width)
    receive(m, first)
    if negotiated:
        negotiate(m)
    return m, received, first


def test_recorded_opening_recut_reaches_clean_real_b2f_lines_and_original_ack():
    m, received, first = opening()
    receive(m, replace(first, observation=replace(first.observation, stream_sample=4*FS)))
    for index, row in enumerate(ROWS[1:], 1):
        incoming = packet(Kind.LONG, (5+4*index)*FS, bytes.fromhex(row['field_hex']),
                          row['status'], len(bytes.fromhex(row['field_hex'])))
        proposal = receive(m, incoming)
        assert proposal.ack_identity == (row['status'], incoming.observation.frame.data)
        assert m.adapter.last == proposal.ack_identity
        if index == 1:
            repeated = replace(incoming, observation=replace(incoming.observation,
                                stream_sample=incoming.observation.stream_sample+round(3.75*FS)))
            assert receive(m, repeated).ack_identity == proposal.ack_identity
    mail = B2FSession('W9SSJ', target='VE3KPG', password='offline-test-password')
    for payload in received:
        mail.feed(payload)
    assert [line for line in mail.log_lines if line.startswith('banner:')] == [
        'banner: RMS Trimode 1.4.4.0',
        'banner: W9SSJ has 103 daily minutes remaining with VE3KPG (FN04VE)',
    ]
    assert mail.challenge == '25286779' and mail.sent_text and not mail.failure
    credit, = [event for event in m.link_events if event['event'] == 'opening-prefix-credit']
    assert credit['credited_source_bytes'] == 31 and credit['remaining_source_bytes'] == 9
    assert credit['prefix'] == PREFIX.hex() and credit['reason'] == 'opening negotiated long'


@pytest.mark.parametrize('status,width', [(0x41, 40), (0x61, 40), (0x61, 69)])
def test_opening_recut_preserves_supervisor_escape_across_credit_boundary(status, width):
    prefix = b'A'*30+b'\x1c'
    m, received, _ = opening(prefix)
    receive(m, packet(Kind.LONG, 5*FS, prefix+b'\x7eB', status, width))
    assert received == [b'A'*30, b'\x1eB']


@pytest.mark.parametrize('clear', ['normal', 'unrelated-long', 'role', 'close'])
def test_opening_credit_does_not_remove_later_legitimate_repeated_prefix(clear):
    m, received, _ = opening()
    if clear == 'normal':
        receive(m, packet(Kind.NORMAL, 5*FS, b'103 daily', 1, 22))
    elif clear == 'unrelated-long':
        receive(m, packet(Kind.LONG, 5*FS, b'unrelated field', 0x61))
    else:
        m._next = Proposal(Kind.BODY, 5*FS, status=0x40 if clear == 'role' else 0x20,
                           state=State.ISS_GRANT if clear == 'role' else State.ISS_CLOSE)
        m.keyed()
        m.state = State.IRS_ANSWER
    before = len(received)
    receive(m, packet(Kind.LONG, 9*FS, PREFIX+b'again', 0x61))
    assert received[before:] == [PREFIX+b'again']


@pytest.mark.parametrize('negotiated,status,prefix', [
    (False, 0, PREFIX), (True, 1, PREFIX), (True, 0, b'short body'),
])
def test_opening_credit_requires_initial_full_counter_zero_body_and_negotiation(negotiated, status, prefix):
    m, received, _ = opening(prefix, negotiated=negotiated, status=status)
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+b'again', 0x61))
    assert received[-1] == PREFIX+b'again'


def test_consumed_opening_credit_does_not_rearm_on_later_body_or_counter_wrap():
    m, received, _ = opening()
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+b'first', 0x61))
    assert received[-1] == b'first'
    receive(m, packet(Kind.BODY, 9*FS, PREFIX))
    receive(m, packet(Kind.LONG, 13*FS, PREFIX+b'later', 0x61))
    assert received[-1] == PREFIX+b'later'


@pytest.mark.parametrize('status', [0x65, 0xe1, 0x81])
def test_unsupported_first_long_is_validated_before_opening_credit(status):
    m, received, _ = opening()
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+b'tail', status))
    assert received == [PREFIX]
    assert m.adapter.failure and m.state == State.IRS_CLOSE


@pytest.mark.parametrize('name', ['n9vc_0', 've3kpg_0'])
def test_stock_and_gateway_decoded_long_repeat_only_the_opening_prefix(name):
    decoded = p4robustrx.decode_symbols(recorded(name))
    assert decoded is not None
    if name == 'n9vc_0':
        meta = json.loads((Path(__file__).parent/'fixtures/p4-breakin/n9vc-sid.json').read_text())
        prefix = bytes.fromhex(meta['data'])
    else:
        prefix = PREFIX
    m, received, _ = opening(prefix)
    frame = decoded.frame
    incoming = packet(Kind.LONG, 5*FS, frame.data, frame.status.byte, len(frame.data))
    proposal = receive(m, incoming)
    assert b''.join(received) == spec.field_payload(frame.data)
    assert proposal.ack_identity == (frame.status.byte, frame.data)


def test_stock_normal_vc_continuation_is_not_credited():
    meta = json.loads((Path(__file__).parent/'fixtures/p4-breakin/n9vc-sid.json').read_text())
    prefix = bytes.fromhex(meta['data'])
    m, received, _ = opening(prefix)
    receive(m, packet(Kind.NORMAL, 5*FS, b'VC>\r', 0x41, 40))
    assert b''.join(received) == b'[BPQ-6.0.25.1-B2FWIHJM$]\r\rde N9VC>\r'
    assert not [e for e in m.link_events if e['event'] == 'opening-prefix-credit']


@pytest.mark.parametrize('mode', ['proposed', 'refused', '13-only', '49-then-53'])
def test_unkeyed_or_incompatible_negotiation_cannot_credit_long(mode):
    m, received, _ = opening(negotiated=False)
    negotiate(m, 13 if mode == '13-only' else 49, key=mode in ('13-only', '49-then-53'))
    if mode == 'refused':
        m.refused('deadline')
    if mode == '49-then-53':
        negotiate(m, 53)
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+b'again', 0x61))
    assert received[-1] == PREFIX+b'again'


@pytest.mark.parametrize('state,payload', [(State.ISS_GRANT, b''), (State.ISS_CLOSE, b''),
                                         (State.ISS_DATA_ACK, b'own source')])
def test_prior_role_or_source_progress_prevents_late_body_arming_credit(state, payload):
    m, _ = machine(data='body')
    m._next = Proposal(Kind.BODY, 0, status=0x40, payload=payload, state=state)
    m.keyed()
    m.state = State.IRS_ANSWER
    m, received, _ = opening(turn=m)
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+b'again', 0x61))
    assert received[-1] == PREFIX+b'again'


def test_credited_suffix_is_not_reclassified_as_a_complete_idle_field():
    m, received, _ = opening()
    suffix = spec.field_fill(9)
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+suffix, 0x61))
    assert received == [PREFIX, suffix]
    assert m.link_events[-1]['remaining_source_bytes'] == len(suffix)


def test_nonterminal_unsupported_first_data_discards_opening_credit():
    m, received, _ = opening()
    m.adapter.terminal = False
    receive(m, packet(Kind.LONG, 5*FS, PREFIX+b'tail', 0x65))
    receive(m, packet(Kind.LONG, 9*FS, PREFIX+b'again', 0x61))
    assert received == [PREFIX, PREFIX+b'again']


def test_recut_with_only_trailing_idle_has_no_new_host_bytes():
    m, received, _ = opening()
    frame = packet(Kind.LONG, 5*FS, PREFIX, 0x61)
    proposal = receive(m, frame)
    assert received == [PREFIX]
    assert m.adapter.last == proposal.ack_identity
    assert m.link_events[-1]['remaining_source_bytes'] == 0
