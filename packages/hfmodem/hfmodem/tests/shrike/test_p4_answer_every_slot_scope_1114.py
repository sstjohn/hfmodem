# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keep the 1114 unread-slot fixes within their receive context."""

from dataclasses import replace

import pytest

from hfmodem.shrike.p4receiver import Kind, Registered
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, body, machine, word


@pytest.mark.parametrize('extent', [None, .796])
@pytest.mark.parametrize('normal_ack', ['toggle', 'parity'])
def test_opening_unread_form_controls_gap_answers(extent, normal_ack):
    m, _ = machine(normal_ack=normal_ack)
    m.seed_opening([dict(audio_start=0, value=7)], [], values=(7, 38))
    at = round(.920 * FS)
    due = at + round(.920 * FS)
    m.observe(word('Y27', at, extent=extent), now=due - m.lead)
    assert m.decide(due - m.lead).purpose == 'repeat-request'
    m.keyed()
    unknown = extent is None and normal_ack == 'toggle'
    assert m._unread_opening_breakin() is unknown
    due += 60000
    m.silence(due - m.lead, answer_due=due)
    answer = m.decide(due - m.lead)
    if unknown:
        assert (answer.stream_sample, answer.word, answer.state) == (due, 'V27', None)
    else:
        assert answer is None
    assert m.state == State.ISS_LADDER


@pytest.mark.parametrize('resolution', ['extent', 'body'])
def test_opening_form_resolved_after_decision_preserves_the_body_decision(resolution):
    m, _ = machine()
    m.seed_opening([dict(audio_start=0, value=7)], [], values=(7, 38))
    at = round(.920 * FS)
    due = at + round(.920 * FS)
    head = word('Y27', at)
    m.observe(head, now=due - m.lead)
    assert m.decide(due - m.lead).purpose == 'repeat-request'
    m.keyed()
    if resolution == 'extent':
        m.observe(replace(head, observation=replace(head.observation, extent_s=.796),
                          outcome=Registered.EXTENDED, replaced=head.observation),
                  now=due - m.lead + 1)
    else:
        m.observe(body(at, b'', 0x40), now=due + 1)
    assert not m._unread_opening_breakin()
    due += 60000
    m.silence(due - m.lead, answer_due=due)
    answer = m.decide(due - m.lead)
    if resolution == 'extent':
        assert answer is None
    else:
        assert (answer.stream_sample, answer.kind, answer.status, answer.state) == (
            due, Kind.BODY, 0x40, State.ISS_GRANT)


@pytest.mark.parametrize('values', [(7, 38), (7, 38, 7)])
def test_completed_opening_does_not_start_unknown_form_gap_answers(values):
    m, _ = machine()
    m.seed_opening([dict(audio_start=i * 60000, value=value)
                    for i, value in enumerate(values)], [], values=(7, 38))
    at = (len(values) - 1) * 60000 + round(.920 * FS)
    due = at + round(.920 * FS)
    m.observe(word('Y27', at), now=due - m.lead)
    assert m.decide(due - m.lead).purpose == 'repeat-request'
    m.keyed()
    due += 60000
    m.silence(due - m.lead, answer_due=due)
    assert m.decide(due - m.lead) is None


@pytest.mark.parametrize('held_kind', [None, Kind.BODY, Kind.NORMAL])
def test_head_only_answer_requires_a_held_body(held_kind):
    m, _ = machine(role='irs')
    if held_kind is not None:
        held = body(FS)
        m.observe(replace(held, observation=replace(held.observation, kind=held_kind)))
        assert m.decide(FS + round(.920 * FS) - m.lead).ack_identity is not None
        m.keyed()
    at = FS + 60000
    due = at + round(.920 * FS)
    m.observe(word('Y27', at, extent=.796), now=due - m.lead)
    answer = m.decide(due - m.lead)
    if held_kind == Kind.BODY:
        assert (answer.stream_sample, answer.current_slot) == (due, at)
        assert answer.purpose == 'repeat-request' and answer.ack_identity is None
    else:
        assert answer is None


@pytest.mark.parametrize('flag', ['provisional', 'weak'])
def test_uncertain_head_cannot_repeat_a_held_body(flag):
    m, _ = machine(role='irs')
    m.observe(body(FS))
    m.decide(FS + round(.920 * FS) - m.lead)
    m.keyed()
    at = FS + 60000
    due = at + round(.920 * FS)
    head = word('Y27', at, extent=.796)
    m.observe(replace(head, observation=replace(head.observation, **{flag: True})),
              now=due - m.lead)
    assert m.decide(due - m.lead) is None
