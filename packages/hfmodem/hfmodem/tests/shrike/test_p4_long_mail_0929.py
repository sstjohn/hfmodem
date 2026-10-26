# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from types import SimpleNamespace

import pytest
from hfmodem.shrike import p4frame
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, machine


def long(at=FS, status=0, *, valid=True, partial=False):
    data = b'gateway long data'.ljust(69, b'\x1e')
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)), data_bytes=69)
    return RegisteredObservation(Observation(Kind.LONG, FS, 0, at, at, 3.28, 0., .8,
                                 provisional=partial, frame=frame if valid else None), Registered.NEW)


@pytest.mark.parametrize('queued', [False, True])
def test_unread_long_in_data_mode_requests_repeat_without_turn(queued):
    m, queue = machine(role='irs', data='body')
    if queued:
        queue.on_host_data(b'FC waiting')
    incoming = long(valid=False)
    m.observe(incoming, now=FS+round(3.31*FS))
    p = m.decide(FS+round(3.4*FS)-m.lead)
    assert (p.kind, p.word, p.state, p.purpose) == (
        Kind.WORD, 'V27', State.IRS_ANSWER, None)
    assert queue._buffer_raw == (10 if queued else 0)


@pytest.mark.parametrize('status', [0, 0x61])
def test_checked_long_delivers_once_and_answers_each_copy_on_long_deadline(status):
    m, queue = machine(role='irs', data='body')
    received = []
    m.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    for at in [FS, FS+round(3.75*FS)]:
        m.observe(long(at, status), now=at+round(3.30*FS))
        p = m.decide(at+round(3.4*FS)-m.lead)
        assert p.word == 'X27' and p.stream_sample == at+round(3.4*FS)
        m.keyed()
    assert received == [b'gateway long data']
    assert queue._buffer_raw == 0


@pytest.mark.parametrize('status,kind,word,state', [
    (0x40, Kind.WORD, 'X27', State.IRS_ANSWER),
    (0x20, Kind.WORD, 'V27', State.IRS_CLOSE),
])
def test_checked_long_control_uses_its_status_and_long_deadline(status, kind, word, state):
    m, queue = machine(role='irs', data='body')
    m.observe(long(status=status), now=FS+round(3.30*FS))
    p = m.decide(FS+round(3.4*FS)-m.lead)
    assert p.kind == kind and p.word == word and p.state == state
    assert p.stream_sample == FS+round(3.4*FS)
    if status == 0x40:
        epoch = m._receive_epoch
        assert not p.payload and not m.outbox.inflight and queue.tx_acked_bytes == 0
        m.keyed()
        assert m._receive_epoch == epoch and m.grants == 0 and queue._buffer_raw == 0
        assert not [e for e in m.link_events if e['event'] == 'turn']


def test_unsupported_long_source_is_terminal_without_delivering_bytes():
    m, _ = machine(role='irs', data='body')
    received = []
    m.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    m.observe(long(status=4), now=FS+round(3.30*FS))
    assert m.adapter.failure == 'unsupported P4 source status 0x04'
    assert not received
    assert m.state == State.IRS_CLOSE


def test_long_family_only_hold_experiment_retains_its_existing_answer():
    m, _ = machine(role='irs', data='hold')
    m.observe(long(valid=False), now=FS+round(3.30*FS))
    assert m.decide(FS+round(3.4*FS)-m.lead).word == 'X27'


@pytest.mark.parametrize('already_decided', [False, True])
def test_shifted_long_resolution_replaces_the_short_pending_identity(already_decided):
    m, _ = machine(role='irs', data='body')
    old = replace(long(valid=False).observation, kind=Kind.NORMAL, provisional=True)
    m.observe(RegisteredObservation(old, Registered.NEW), now=FS+round(.81*FS))
    if already_decided:
        assert m.decide(FS+round(.92*FS)-m.lead).purpose is None
    full = long(FS+2)
    m.observe(replace(full, outcome=Registered.CONTRADICTED, replaced=old), now=FS+round(3.30*FS))
    p = m.decide(FS+2+round(3.4*FS)-m.lead)
    assert p is not None and p.word == 'X27'
    assert p.stream_sample == FS+2+round(3.4*FS)


@pytest.mark.parametrize('late_samples', [0, 3552])
def test_late_checked_long_delivers_once_but_waits_for_a_timely_repeat(late_samples):
    m, _ = machine(role='irs', data='body')
    received = []
    m.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    m.observe(long(valid=False), now=FS+round(3.30*FS))
    due = FS+round(3.4*FS)
    assert m.decide(due-m.lead).purpose is None
    now = due-m.lead+late_samples
    m.observe(replace(long(), outcome=Registered.ENRICHED), now=now)
    answer = m.decide(now)
    if now >= due:
        assert answer is None and m._deferred is not None
    else:
        assert answer.word == 'X27' and answer.purpose is None
        assert answer.current_slot_accepted and m._deferred is None
        m.keyed()
    at = FS+round(3.75*FS)
    m.observe(long(at), now=at+round(3.30*FS))
    p = m.decide(at+round(3.4*FS)-m.lead)
    assert p.word == 'X27' and p.stream_sample == at+round(3.4*FS)
    assert received == [b'gateway long data']
