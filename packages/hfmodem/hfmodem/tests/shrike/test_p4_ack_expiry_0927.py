# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace

import pytest

from hfmodem.shrike.p4observation import Registered
from hfmodem.shrike.p4turn import CYCLE, Kind, State
from hfmodem.tests.shrike.test_p4turn import FS, machine, word


def waiting(step):
    turn, queue = machine(data='chirp' if step == 'chirp' else 'body')
    queue.on_host_data(b'A' * 31)
    if step == 'ladder':
        turn._short(0, 1)
        name = 'U27'
    else:
        turn._data(0)
        name = 'X27'
    sent = turn.keyed()
    onset = round((3.4 if step == 'chirp' else .920) * FS)
    early = word(name, onset, partial=True, responds_to=sent.stream_sample)
    turn.observe(early, now=onset + round(.210 * FS))
    return turn, queue, sent, early


@pytest.mark.parametrize('step', ['ladder', 'data', 'chirp'])
def test_missing_read_expires_once_and_rolls_back_without_accepting_a_late_ack(step):
    turn, queue, sent, early = waiting(step)
    limit = early.observation.stream_sample + round((.275 + CYCLE[sent.kind]) * FS)
    turn.silence(limit - 1)
    assert turn.decide(limit - 1) is None
    turn.silence(limit)
    expired = [r for r in turn.events if r.outcome == Registered.EXPIRED]
    assert len(expired) == 1 and expired[0].replaced == early.observation
    proposal = turn.decide(limit)
    assert proposal is not None and proposal.repeat
    assert proposal.stream_sample >= limit + turn.lead
    lent = turn.outbox.inflight
    turn.observe(word(early.observation.word, early.observation.stream_sample,
                      outcome=Registered.CONFIRMED, replaced=early.observation,
                      responds_to=sent.stream_sample),
                 now=limit + 1)
    assert turn.outbox.inflight == lent and queue._buffer_raw == 31
    assert turn.decide(limit + 1) == proposal
    closing = []
    for _ in range(12):
        if turn.state == State.DOWN:
            break
        sent = turn.keyed()
        if sent.state == State.ISS_CLOSE:
            closing.append(sent)
        at = sent.stream_sample + round(CYCLE[sent.kind] * FS) - turn.lead
        turn.silence(at)
    assert turn.state == State.DOWN
    assert turn.reason == ('retry budget exhausted' if step == 'ladder' else 'close silence')
    if step != 'ladder':
        assert closing[0].kind == Kind.BODY and closing[0].status == 0x20
        assert [p.value for p in closing[1:]] == [37, 37, 37]
    assert queue._buffer_raw == len(queue._outbuf) == 31 and not queue._p4_lent
    assert sum(r.outcome == Registered.EXPIRED for r in turn.events) == 1


@pytest.mark.parametrize('state,reason', [(State.IRS_CLOSE, 'peer closed'),
                                         (State.ISS_CLOSE, 'close silence')])
def test_close_silence_is_not_suppressed_by_a_pending_word(state, reason):
    turn, queue, sent, early = waiting('data')
    turn.state = state
    if state == State.ISS_CLOSE:
        turn.last = replace(sent, kind=Kind.SHORT, value=37)
        turn.repeats = 2
    turn.silence(early.observation.stream_sample + round(.330 * FS))
    assert turn.state == State.DOWN and turn.reason == reason
    assert turn._waiting_ack is None
    assert queue._buffer_raw == len(queue._outbuf) == 31 and not queue._p4_lent
