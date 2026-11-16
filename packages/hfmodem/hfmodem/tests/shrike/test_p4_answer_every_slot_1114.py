# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""WS8EOC 1114 break-in slots and a bounded close after an unchanged v53 hold."""

from dataclasses import replace
import json

import pytest

from hfmodem.shrike.p4receiver import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.tests.shrike.test_p4_ack_toggle_1110 import registered
from hfmodem.tests.shrike.test_p4_arm_replay_1112 import CAPTURES, replay
from hfmodem.tests.shrike.test_p4_v53_hold_1107 import copy, irs, words
from hfmodem.tests.shrike.test_p4turn import FS, body, machine, word


def every_slot(probe, start, end):
    decisions = [e for e in probe.link_events if e['event'] == 'Decision'
                 and start <= e['at'] / FS < end]
    keys = [e for e in probe.link_events if e['event'] == 'Keyed']
    assert decisions
    for decision in decisions:
        assert decision['proposed_at'] is not None, decision
        keyed, = [e for e in keys if e['at'] == decision['proposed_at']]
        assert keyed['kind'] == Kind.WORD
        assert keyed['ack_identity'] is not None or keyed['purpose'] == 'repeat-request'
        assert abs(keyed['at'] - decision['at'] - round(.920 * FS)) <= 48
    return decisions


def test_breakin_normal_packets_answer_on_every_peer_slot():
    probe = replay('ws8eoc-1114-breakin')
    decisions = every_slot(probe, 19.7, 40.)
    assert len(decisions) == 17
    normals = [h for h in probe.heard if h['kind'] == Kind.NORMAL
               and h.get('status') == 1 and h.get('data')]
    assert any(abs(h['onset_sample'] / FS - 21.012) < .01 for h in normals)
    keys = [e for e in probe.link_events if e['event'] == 'Keyed' and e.get('ack_identity')]
    assert any(e['state'] == State.IRS_ANSWER and e['ack_identity'][0] == 1 for e in keys)


def test_breakin_normal_packets_answer_on_every_peer_slot_under_ack_x27():
    probe = replay('ws8eoc-1114-breakin', p4_normal_ack='x27')
    decisions = every_slot(probe, 19.7, 40.)
    assert len(decisions) == 17
    keys = {e['at']: e for e in probe.link_events if e['event'] == 'Keyed'}
    unread = [keys[d['proposed_at']] for d in decisions
              if d['reason'] == 'body unread at decision']
    assert unread and all(k['word'] == 'V27' for k in unread)
    acks = [e for e in keys.values() if e.get('ack_identity') and e['ack_identity'][0] == 1]
    assert acks and all(k['word'] == 'X27' for k in acks)


def test_irs_answer_tape_keys_both_live_unread_slots():
    every_slot(replay('ws8eoc-1114-hold'), 28.5, 30.)


@pytest.mark.parametrize('next_slot', ['gap', 'normal-unread', 'y27-unread'])
def test_unread_opening_breakin_keeps_answering_until_a_body_is_read(next_slot):
    capture = RECORD / 'operator-data/captures' / CAPTURES['ws8eoc-1114-breakin']
    tape = json.loads(require_input(capture / 'p4-probe.json').read_text())
    row = next(h for h in tape['heard'] if h['kind'] == 'word' and h['word'] == 'Y27'
               and not h['provisional'])
    m, _ = machine()
    m.seed_opening(tape['emissions'][:1], [], values=(7, 38))
    m.observe(word('Y27', row['onset_sample'], responds_to=row['responds_to']),
              now=row['onset_sample'] + round(.81 * FS))
    due = row['onset_sample'] + round(.920 * FS)
    assert m.decide(due - m.lead).purpose == 'repeat-request'
    m.keyed()
    at = row['onset_sample'] + 60000
    due = at + round(.920 * FS)
    if next_slot != 'gap':
        obs = replace(word('Y27', at).observation,
                      kind=Kind.NORMAL if next_slot == 'normal-unread' else Kind.WORD,
                      word=None if next_slot == 'normal-unread' else 'Y27')
        m.observe(RegisteredObservation(obs, Registered.NEW), now=at + round(.81 * FS))
        answer = m.decide(due - m.lead)
    else:
        m.silence(due - m.lead, answer_due=due)
        answer = m.decide(due - m.lead)
    assert answer.stream_sample == due and answer.word == 'V27'
    assert answer.purpose == 'repeat-request' and answer.ack_identity is None
    m.keyed()
    assert m.state == State.ISS_LADDER


def test_irs_answer_recorded_head_only_slots_repeat_the_held_answer():
    capture = RECORD / 'operator-data/captures' / CAPTURES['ws8eoc-1114-hold']
    tape = json.loads(require_input(capture / 'p4-probe.json').read_text())
    held = next(h for h in tape['heard'] if h['kind'] == 'body'
                and h['onset_sample'] == 1308483)
    m, _ = machine(role='irs')
    m.observe(registered(held), now=held['onset_sample'] + round(.81 * FS))
    due = held['onset_sample'] + round(.920 * FS)
    assert m.decide(due - m.lead).ack_identity is not None
    m.keyed()
    for at in (1368580, 1428478):
        rows = [h for h in tape['heard'] if h['kind'] == 'word' and h['onset_sample'] == at
                and h['outcome'] in ('new', 'confirmed')]
        assert len(rows) == 2
        previous = None
        for row in rows:
            obs = replace(word(row['word'], at).observation,
                          provisional=row['provisional'], extent_s=row['extent_s'])
            reg = RegisteredObservation(obs, Registered(row['outcome']), replaced=previous,
                                        responds_to=row['responds_to'])
            m.observe(reg, now=at + round(.83 * FS))
            previous = obs
        assert m._held_answer.onset == held['onset_sample']
        assert 'body' not in m.pending[at]
        due = at + round(.920 * FS)
        answer = m.decide(due - m.lead)
        assert (answer.stream_sample, answer.kind, answer.word) == (due, Kind.WORD, 'X27')
        assert answer.purpose == 'repeat-request' and answer.ack_identity is None
        assert answer.current_slot == at and not answer.current_slot_accepted
        m.keyed()
        assert m.state == State.IRS_ANSWER


@pytest.mark.parametrize('breakin', ['body', 'y27'])
def test_v53_hold_close_is_bounded_when_breakin_is_unanswered(breakin):
    m, _ = irs(v53_hold=2, breakin=breakin)
    words(m, 1)
    proposal = copy(m, 1)
    assert m.state == State.IRS_CLOSE and m._closing_failure and m.app_done
    assert proposal.state == State.IRS_BREAKIN
    for _ in range(4):
        assert m._next.state == State.IRS_BREAKIN
        sent = m.keyed()
        m.silence(sent.stream_sample + 60000 - m.lead)
    assert m.state == State.DOWN and m.reason == 'break-in unanswered'


@pytest.mark.parametrize('data', ['none', 'body', 'long'])
@pytest.mark.parametrize('breakin', ['body', 'y27'])
def test_v53_hold_granted_close_keys_body_then_v37(data, breakin):
    m, _ = irs(v53_hold=2, breakin=breakin, data=data)
    words(m, 1)
    assert copy(m, 1).state == State.IRS_BREAKIN
    assert m.reason == 'no progress after 2 peer control 53 copies'
    request = m.keyed()
    if breakin == 'y27':
        at = request.stream_sample + round(.330 * FS)
        m.observe(word('V27', at, responds_to=request.stream_sample), now=at + round(.25 * FS))
        close = m.decide(m.next_raster(at) - m.lead)
    else:
        at = request.stream_sample + round(.920 * FS)
        grant = replace(body(at, b'', 0x40), responds_to=request.stream_sample)
        m.observe(grant, now=at + round(.81 * FS))
        close = m.decide(at + round(.920 * FS) - m.lead)
    assert (close.kind, close.status, close.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
    assert m.close_context.reason == 'no progress after 2 peer control 53 copies'
    sent = m.keyed()
    m.silence(sent.stream_sample + 60000 - m.lead)
    robust = m.decide(sent.stream_sample + 60000 - m.lead)
    assert (robust.kind, robust.value, robust.state) == (Kind.SHORT, 37, State.ISS_CLOSE)
    sent = m.keyed()
    at = sent.stream_sample + round(.920 * FS)
    m.observe(word('V27', at, responds_to=sent.stream_sample), now=at + round(.25 * FS))
    assert m.state == State.DOWN and m.reason == 'close acknowledged'
