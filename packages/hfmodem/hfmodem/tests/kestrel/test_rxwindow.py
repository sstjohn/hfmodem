# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""DESIGN §§2–6,10,10a: pure decisions, independent peer/channel/witness."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import zlib

import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300
from hfmodem.kestrel.arq.rxwindow import (
    Block, Close, Commit, Emit, FilesHold, FilesParse, Frame,
    Key, Prepared, Ready, ReceiveWindow, Replay, Report, ResetDecoder,
)
from hfmodem.kestrel.vara.compression import decode_text, encode_text
from .rxwindow_reference import DecisionHistory


class Harness:
    def __init__(self, mode='OFF', high_speed=True):
        self.rx = ReceiveWindow(mode, high_speed=high_speed)
        self.host = b''
        self.actions = []
        self.keys = []
        self.closes = []
        self.erased = []
        self.parser_pending = b''

    def take(self, actions, air=True):
        self.actions.extend(actions)
        self.host += b''.join(a.data for a in actions if isinstance(a, Emit))
        for a in actions:
            if isinstance(a, Close):
                self.closes.append(a.report)
            if isinstance(a, Key):
                self.keys.append(a)
                self.rx.sent(a, air)
        return actions

    def event(self, kind, air=True, **kw):
        return self.take(self.rx.event(kind, **kw), air)

    def commit(self, values, air=True, **kw):
        prepared = tuple(v if isinstance(v, Prepared) else Prepared(v) for v in values)
        parsed = None
        if self.rx.mode == 'FILES':
            p = self.rx.placement(prepared)
            terminal = not self.rx.W.partial and self.rx.W.frames[-1].last
            parsed = parse_preview(self.parser_pending, p.new, terminal)
        actions = self.rx.commit(prepared, files=parsed, **kw)
        if parsed and any(isinstance(a, Commit) for a in actions):
            self.parser_pending = parsed.pending
        return self.take(actions, air)

    def window(self, *values, verdict=None, last=False, air=True, hint=6,
               level=1, compressed=False, envelope=None, **kw):
        verdict = verdict or ('CERTAIN' if len(values) == 2 else 'NONE')
        envelope = envelope or ('double' if len(values) == 2 else 'single')
        for i, value in enumerate(values):
            data = value.data if isinstance(value, Prepared) else value
            frame = Frame(data, last=last and i == len(values)-1,
                          compressed=compressed, hint=hint)
            self.event('F', frame=frame, verdict=verdict, level=level, envelope=envelope)
        self.event('E', single=True)
        return self.commit(values, air=air, **kw)

    def query(self, *, phase=None, inside=False, air=True, **kw):
        return self.event('Q', phase=phase, inside=inside, air=air, **kw)

    def arm(self, air=True):
        self.query(phaseless=True, inside=False, air=air)
        assert self.rx.A.kind == 'nak'

    def erase(self, *payloads):
        assert self.rx.state == 'IDLE'
        self.erased.append(tuple(payloads))


def stored(data):
    return len(data).to_bytes(2, 'big') + data


def deflated(data):
    d = zlib.compressobj(wbits=-15)
    body = d.compress(data) + d.flush()
    return (0x8000 | len(body)).to_bytes(2, 'big') + body


def parse_preview(pending, new, terminal):
    """Adapter surrogate: parse wire bytes, never touch committed codec state."""
    buf = pending + new
    blocks = []
    error = None
    while len(buf) >= 2:
        header = int.from_bytes(buf[:2], 'big')
        n = header & 0x7fff
        if not header & 0x8000 and n > 32500:
            error = 'stored block exceeds 32500'
            break
        if len(buf) < n+2:
            break
        body, buf = buf[2:2+n], buf[2+n:]
        if header & 0x8000:
            try:
                d = zlib.decompressobj(-15)
                data = d.decompress(body, 32501)
                if not d.eof or d.unused_data or d.unconsumed_tail or len(data) > 32500:
                    raise ValueError('bad deflate extent')
            except (zlib.error, ValueError):
                error = 'invalid deflate'
                break
        else:
            data = body
        blocks.append(Block(data, not bool(header & 0x8000)))
    if terminal and buf:
        error = error or 'delivery ends inside a block'
    return FilesParse(tuple(blocks), buf, error)


def test_row01_turn_drops_window_and_replay_keeps_only_final():
    h = Harness()
    h.window(b'AB')
    h.arm()
    h.event('F', frame=Frame(b'A'))
    h.event('T')
    assert h.rx.W is h.rx.T is h.rx.A is h.rx.K is None
    assert h.rx.state == 'IDLE' and h.rx.alt == 0 and not h.rx.trusted
    h.window(b'Z', last=True, air=False)
    a = h.rx.A
    h.event('T')
    assert h.rx.A is a and not h.rx.trusted
    h.query(phase=0)
    assert h.keys[-1].kind == 'final' and h.rx.trusted


def test_row02_grant_owes_unread_even_without_data():
    h = Harness()
    h.event('G')
    assert h.rx.state == 'IDLE' and h.rx.A.phase == 0 and not h.rx.A.sent
    h.query(phase=0)
    assert h.keys[-1].kind == 'nak' and not h.host


@pytest.mark.parametrize('kind', ['F', 'X'])
@pytest.mark.parametrize('verdict,expected', [('NONE', {1}), ('CERTAIN', {2}), ('POSSIBLE', {1, 2})])
def test_row03_open(kind, verdict, expected):
    h = Harness()
    h.rx.alt = 1
    h.event(kind, frame=Frame(b'A'), verdict=verdict, level=7)
    assert h.rx.state == 'OPEN' and h.rx.W.phase == 1
    assert h.rx.W.expected == expected and h.rx.W.level == 7
    assert h.rx.W.frames == ((Frame(b'A'),) if kind == 'F' else (None,))


def test_row04_append_expected_companion():
    h = Harness()
    h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
    h.event('F', frame=Frame(b'B'))
    assert h.rx.state == 'OPEN' and len(h.rx.W.frames) == 2
    assert not h.host and not h.keys


def test_row05_located_failure_stays_in_window():
    h = Harness()
    h.event('F', frame=Frame(b'A'), verdict='POSSIBLE')
    h.event('X')
    assert h.rx.W.frames == (Frame(b'A'), None)
    assert h.rx.state == 'OPEN'
    h.event('E', single=True)
    assert h.keys[-1].kind == 'nak' and not h.host


@pytest.mark.parametrize('verdict,values', [('NONE', (b'A',)), ('CERTAIN', (b'A', b'B')),
                                           ('POSSIBLE', (b'A',)), ('POSSIBLE', (b'A', b'B'))])
def test_row06_ended_complete(verdict, values):
    h = Harness()
    for value in values:
        h.event('F', frame=Frame(value), verdict=verdict)
    out = h.event('E', single=True)
    assert out == [Ready()] and h.rx.state == 'ENDED-COMPLETE'
    assert h.rx.R == h.host == b''


@pytest.mark.parametrize('shape', ['missing-companion', 'failed-head', 'failed-tail', 'certain-terminal'])
def test_row07_ended_incomplete(shape):
    h = Harness()
    if shape == 'failed-head':
        h.event('X', verdict='CERTAIN')
        h.event('F', frame=Frame(b'B'))
    else:
        h.event('F', frame=Frame(b'A', last=shape == 'certain-terminal'), verdict='CERTAIN')
        if shape == 'failed-tail':
            h.event('X')
    h.event('E')
    assert h.rx.state == 'IDLE' and h.rx.W is None
    assert h.keys[-1].kind == 'nak' and h.rx.K is None and not h.host


@pytest.mark.parametrize('verdict,ready', [('CERTAIN', False), ('POSSIBLE', True), ('NONE', True)])
def test_row08_query_is_end_evidence_first(verdict, ready):
    h = Harness()
    h.event('F', frame=Frame(b'A'), verdict=verdict)
    out = h.query(single=True, phase=1)
    assert isinstance(out[0], Ready if ready else Key)
    if ready:
        h.commit((b'A',))
        assert h.host == b'A'
    else:
        assert h.keys[-1].kind == 'nak'


def test_row09_open_ceiling_is_incomplete():
    h = Harness()
    h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
    h.event('C', peer_keying=True)
    assert h.rx.A.kind == 'nak' and not h.rx.A.sent and not h.keys
    h.query()
    assert h.keys[-1].kind == 'nak' and not h.host


def held(h):
    h.event('F', frame=Frame(b'A'), verdict='POSSIBLE')
    h.event('E', single=True, peer_keying=True)
    assert h.rx.state == 'ENDED-COMPLETE' and h.rx.W.held


def test_row10_held_companion_reopens():
    h = Harness()
    held(h)
    h.event('F', frame=Frame(b'B'))
    assert h.rx.state == 'OPEN' and not h.host and not h.keys
    h.event('E')
    h.commit((b'A', b'B'))
    assert h.host == b'AB'


def test_row11_held_companion_failure_naks_whole():
    h = Harness()
    held(h)
    h.event('X', peer_keying=True)
    assert h.rx.A.kind == 'nak' and not h.rx.A.sent and not h.host
    h.query()
    assert h.keys[-1].kind == 'nak'


def test_row12_defer_complete_while_peer_keys():
    h = Harness()
    held(h)
    h.event('E', peer_keying=True)
    assert not h.host and not h.keys
    with pytest.raises(ValueError, match='ready'):
        h.rx.commit((Prepared(b'A'),))


@pytest.mark.parametrize('event', ['E', 'Q', 'C'])
@pytest.mark.parametrize('answer_before', [False, True])
def test_row13_commit_preflight_and_order(event, answer_before):
    h = Harness()
    held(h)
    h.event(event)
    out = h.commit((b'A',), answer_before=answer_before)
    assert h.rx.state == 'IDLE' and h.rx.W is None
    assert h.host == h.rx.R == b'A'
    assert (h.rx.K.start, h.rx.K.end) == (0, 1)
    assert h.keys[-1].kind == 'continue'
    assert isinstance(out[0 if answer_before else -1], Key)


def partial(h, air=True):
    h.event('F', frame=Frame(b'A'), verdict='CERTAIN', limited_pair=True)
    h.event('X')
    assert h.event('E') == [Ready(True)]
    h.commit((b'A',), air=air)


def test_row14_bw500_frame10_head_only():
    h = Harness()
    partial(h)
    assert h.host == b'A' and h.keys[-1].kind == 'frame10'
    assert h.rx.K.frames == (Frame(b'A'),) and not h.rx.trusted
    h.window(b'B', last=True)
    assert h.host == b'AB'


def test_row15_nak_owed_never_into_peer_data():
    h = Harness()
    h.window(b'K')
    k = h.rx.K
    h.event('X')
    h.event('E', peer_keying=True)
    assert h.rx.A.kind == 'nak' and not h.rx.A.sent
    assert h.rx.K == k and h.rx.T is None and h.host == b'K'
    h.query(air=False)
    assert not h.rx.A.sent and h.rx.trusted
    h.query()
    assert h.rx.A.sent and not h.rx.trusted and h.rx.T is None


@pytest.mark.parametrize('event', ['Q', 'R'])
@pytest.mark.parametrize('phase', [0, 1, None])
def test_row16_final_precedes_phase_and_bound(event, phase):
    h = Harness()
    h.window(b'A', last=True)
    h.event(event, phase=phase, inside=False)
    assert h.keys[-1].kind == 'final' and h.keys[-1].basis == 'final'
    assert h.rx.T is None and h.host == b'A'


@pytest.mark.parametrize('air', [False, True])
@pytest.mark.parametrize('event', ['Q', 'R'])
def test_row17_frame10_is_never_upgraded(event, air):
    h = Harness()
    partial(h, air=air)
    for _ in range(5):
        h.event(event, phase=0, inside=True)
        assert h.keys[-1].kind == h.keys[-1].form == 'frame10'
        assert h.host == b'A'
    assert h.rx.alt == 1


def test_row18_proving_phase_reanswers_late_without_bound():
    h = Harness()
    h.window(b'A')
    h.window(b'B')
    assert h.rx.A.phase == 1
    h.query(phase=1, inside=False)
    assert h.keys[-1].basis == 'phase' and h.keys[-1].form == 'query32'
    assert h.host == b'AB' and h.rx.alt == 0


@pytest.mark.parametrize('a_phase,q', [(0, 1), (1, 0)])
def test_row19_proving_next_is_unread_nak(a_phase, q):
    h = Harness()
    if a_phase:
        h.window(b'0')
    h.window(b'A')
    h.query(phase=q, inside=False)
    assert h.keys[-1].kind == 'nak' and h.rx.T is None
    assert not h.rx.trusted


@pytest.mark.parametrize('reason', ['zero-zero', 'idle', 'untrusted', 'bw500-poll'])
def test_row20_nonproof_inside_bound_reanswers(reason):
    h = Harness()
    h.window(b'A')
    if reason == 'bw500-poll':
        h.window(b'B')
    if reason == 'untrusted':
        h.rx.trusted = False
    h.query(phase=1 if reason == 'bw500-poll' else 0, inside=True,
            phaseless=reason == 'idle', bw500_poll=reason == 'bw500-poll')
    assert h.keys[-1].basis == 'bound' and h.keys[-1].kind == 'continue'


def test_row21_late_phaseless_nak_effect_waits_for_air():
    h = Harness()
    h.window(b'AB')
    k = h.rx.K
    h.arm(air=False)
    assert h.rx.T is None and h.rx.A.effect == (0, 2)
    assert h.rx.K == k and h.rx.trusted
    h.query(air=False)
    assert h.rx.T is None
    h.query()
    assert h.rx.T == Replay(0, 2) and not h.rx.trusted
    assert h.rx.alt == 0


def test_row22_late_frame10_arms_on_head():
    h = Harness()
    partial(h)
    h.arm()
    assert h.rx.T == Replay(0, 1)
    h.window(b'A')
    h.window(b'B', last=True)
    assert h.host == b'AB'


def test_row23_nak_debt_repeats_same_effect_and_phase():
    h = Harness()
    h.window(b'AB')
    h.arm()
    a, t, alt = h.rx.A, h.rx.T, h.rx.alt
    for phase in (0, 1, None):
        h.query(phase=phase)
        assert h.rx.A is a and h.rx.T == t and h.rx.alt == alt
        assert h.keys[-1].kind == 'nak'


@pytest.mark.parametrize('high_speed', [False, True])
def test_row24_identified_idle_ladder_and_explicit_query_form(high_speed):
    h = Harness(high_speed=high_speed)
    h.window(b'A', step=4, original='burst8')
    h.event('R', inside=True, explicit=False)
    assert h.keys[-1].form == 'burst8' and h.keys[-1].step == 4
    h.event('R', inside=True, explicit=False)
    assert h.keys[-1].form == 'query32' and h.keys[-1].step == (4 if high_speed else 0)
    h.query(inside=True)
    assert h.keys[-1].form == 'query32'
    h.event('R', inside=True, budget=False)
    assert h.closes[-1].code == 'budget'


def test_row25_silent_cadence_never_authorizes_positive():
    h = Harness()
    h.window(b'P')
    before = len(h.keys)
    assert h.event('R', recognized=False, inside=True) == []
    assert len(h.keys) == before
    h.event('R', recognized=False, inside=False)
    assert h.keys[-1].kind == 'nak' and h.rx.T == Replay(0, 1)


@pytest.mark.parametrize('kind', ['continue', 'frame10', 'final', 'nak'])
@pytest.mark.parametrize('inside', [True, False])
def test_silent_timer_all_saved_forms(kind, inside):
    h = Harness()
    if kind == 'frame10':
        partial(h)
    elif kind == 'nak':
        h.event('G')
    else:
        h.window(b'A', last=kind == 'final')
    anchor, a = h.rx.K, h.rx.A
    actions = h.event('R', recognized=False, inside=inside)
    keys = [v for v in actions if isinstance(v, Key)]
    if kind == 'final':
        assert len(keys) == 1 and keys[0].kind == 'final'
        assert keys[0].basis == 'final' and h.rx.A is a
        assert h.rx.T is None
    elif kind == 'nak' or not inside:
        assert len(keys) == 1 and keys[0].kind == 'nak'
        if anchor:
            assert h.rx.T == Replay(anchor.start, anchor.end)
    else:
        assert not keys and h.rx.A is a
    assert h.host == (b'' if kind == 'nak' else b'A')


@pytest.mark.parametrize('air', [False, True])
@pytest.mark.parametrize('inside', [False, True])
def test_silent_timer_keeps_closing_double_final_recovery(air, inside):
    h = Harness()
    h.window(b'head', b'close', last=True, air=air)
    saved, stream = h.rx.A, h.rx.R
    for _ in range(3):
        actions = h.event('R', recognized=False, inside=inside, phase=1)
        assert [a.kind for a in actions if isinstance(a, Key)] == ['final']
        assert not any(isinstance(a, (Commit, Emit, Close)) for a in actions)
        assert h.rx.A is saved and h.rx.T is None and h.rx.R == stream
    assert h.host == b'headclose'


def test_refused_timer_nak_debt_precedes_proving_next_query():
    h = Harness()
    h.window(b'P')
    h.window(b'K')
    assert h.rx.A.phase == 1 and h.rx.alt == 0 and h.rx.trusted
    h.event('R', recognized=False, inside=False, air=False)
    debt = h.rx.A
    assert debt.effect == (1, 2) and not debt.sent
    h.query(phase=1, air=False)
    assert h.rx.A is debt and h.rx.A.effect == (1, 2)
    h.query(phase=1)
    assert h.rx.A is debt and h.rx.T == Replay(1, 2)
    h.window(b'K')
    assert h.host == b'PK' and h.rx.T is None


@pytest.mark.parametrize('located', [False, True])
@pytest.mark.parametrize('end', ['E', 'Q', 'C'])
@pytest.mark.parametrize('air', [False, True])
def test_limited_pair_absent_or_failed_tail_frame10(located, end, air):
    h = Harness()
    h.event('F', frame=Frame(b'head'), limited_pair=True, verdict='CERTAIN')
    if located:
        h.event('X')
    assert h.event(end) == [Ready(True)]
    h.commit((b'head',), air=air)
    assert h.host == b'head' and h.keys[-1].kind == 'frame10'
    h.query(inside=True)
    assert h.host == b'head' and h.keys[-1].kind == 'frame10'


def test_row26_sent_result_flips_only_first_actual_answer():
    h = Harness()
    h.window(b'A', air=False)
    assert h.rx.alt == 0 and not h.rx.A.sent and h.host == b'A'
    h.query(phase=0, inside=True, air=False)
    assert h.rx.alt == 0
    h.query(phase=0, inside=True)
    assert h.rx.alt == 1 and h.rx.A.sent
    h.query(phase=0, inside=True)
    assert h.rx.alt == 1 and h.host == b'A'
    h.arm()
    assert h.rx.alt == 0 and not h.rx.trusted
    h.query()
    assert h.rx.alt == 0


def test_short_head_and_compressed_overrides_and_positioned_tail():
    h = Harness()
    h.event('F', frame=Frame(b'A', full=False), verdict='NONE')
    assert h.rx.W.expected == {2}
    h.event('E', single=True)
    assert h.keys[-1].kind == 'nak'
    h.event('F', frame=Frame(b'text', full=False, compressed=True), verdict='CERTAIN')
    assert h.rx.W.expected == {1} and h.rx.W.verdict == 'NONE'
    h.event('E')
    h.commit((b'plaintext',))
    assert h.host == b'plaintext'


def test_previews_are_nonmutating_and_refused_commit_is_permanent():
    h = Harness()
    h.event('F', frame=Frame(b'A'))
    h.event('E')
    for _ in range(3):
        assert h.rx.placement((Prepared(b'A'),)).new == b'A'
        assert h.rx.R == h.host == b'' and h.rx.K is None
    h.commit((b'A',), air=False)
    assert h.host == b'A' and h.rx.K.end == 1
    h.query(inside=True)
    assert h.host == b'A'


def test_key_results_are_serial_and_not_peer_decode_results():
    rx = ReceiveWindow()
    rx.event('G')
    key, = rx.event('Q')
    with pytest.raises(ValueError, match='keying result'):
        rx.event('Q')
    with pytest.raises(ValueError, match='pending decision'):
        rx.sent(Key(999, 'nak', 'nak', 0, 'unread'), True)
    rx.sent(key, False)
    assert rx.trusted and not rx.A.sent


SCRIPT_CASES = (
    'pass2-double-lost-ack-erased-next',
    'pass3-4-PPQZ-erased-second-P', 'pass3-4-PPQZ-lost-first-answer',
    'pass3-xara-two-failed-retries', 'pass3-erased-peer-final-phase0',
    'pass3-erased-peer-final-phase1', 'pass3-possible-invisible-replay-tail',
    'pass3-certain-missing-replay-tail', 'pass4-19-20-lower-singleton-retry',
    'pass4-19-20-floor-singleton-retry', 'closing-double-final-lost-phase0',
    'closing-double-final-lost-phase1', 'incomplete-closing-double-whole-retry',
    'pass20-invisible-head-answer-heard', 'pass20-invisible-head-answer-lost',
    'pass19-invisible-head-late-zero-zero', 'pass4-frame10-lost-refused-rungs',
    'pass5-short-equal-final-next-delivery', 'pass5-text-fault-replay',
    'pass5-final-query-phase0', 'pass6-resolved-replay-rearm',
    'pass6-silent-cadence-erased-next', 'pass7-89-byte-progressing-replay',
    'pass7-repeated-first-chunk-nested', 'pass7-frame10-late-query',
    'pass8-equal-head-distinct-continuation', 'pass9-12-erased-short-final',
    'pass9-12-closing-double-failed-tail', 'pass10-four-lost-chunk-answers',
    'pass13-frame10-tail-repacked-erased-745', 'pass15-abort-odd-first-erased',
    'pass15-abort-even-first-erased', 'pass15-abort-odd-second-erased',
    'pass15-abort-even-second-erased', 'pass16-PP-new-PC-nested',
    'pass17-PQ-nested-repeated-PQ-final', 'pass17-invisible-replay-head',
    'pass18-repeated-invisible-replay-head', 'unchanged-geometry-changed-hint',
    'refused-nak-owed-keyed-on-query', 'grant-first-attempt-erased',
    'held-located-failure', 'held-open-ceiling',
)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('case', SCRIPT_CASES)
def test_section10_fixed_orderings(case, role):
    h = Harness('TEXT' if case == 'pass5-text-fault-replay' else 'OFF')
    expected = b''
    closed = False
    if case == 'pass2-double-lost-ack-erased-next':
        h.window(b'P', b'Q', air=True)
        h.arm()
        h.window(b'P')
        h.window(b'Q')
        h.erase(b'Z')
        h.query(phaseless=True)
        assert h.keys[-1].kind == 'nak'
        h.window(b'Z', last=True)
        expected = b'PQZ'
    elif case.startswith('pass3-4-PPQZ'):
        h.window(b'P'*9 + b'P'*9 + b'Q'*9 + b'Z'*20, air=True)
        h.arm()
        h.window(b'P'*9)
        if case.endswith('second-P'):
            h.erase(b'P'*9)
        h.query(phaseless=True)
        assert h.keys[-1].kind == 'nak' and h.rx.T.alternative == 9
        n = len(h.keys)
        h.window(b'P'*9)
        assert len(h.keys) == n
        expected = b'P'*18 + b'Q'*9 + b'Z'*20
        closed = True
    elif case == 'pass3-xara-two-failed-retries':
        h.window(b'K', air=True)
        h.arm()
        for _ in range(2):
            h.event('X')
            h.event('E')
            assert h.keys[-1].kind == 'nak'
        h.window(b'K')
        h.erase(b'Z')
        h.query(phase=1, inside=False)
        assert h.keys[-1].kind == 'nak' and not h.rx.trusted
        h.window(b'Z', last=True)
        expected = b'KZ'
    elif case.startswith('pass3-erased-peer-final'):
        if case.endswith('phase1'):
            h.window(b'0')
        h.window(b'A')
        expected = h.host + b'Z'
        h.erase(b'Z')
        h.query(phase=0, inside=False)
        assert h.keys[-1].kind == 'nak'
        h.window(b'Z', last=True)
    elif case in ('pass3-possible-invisible-replay-tail',
                  'pass17-invisible-replay-head', 'pass18-repeated-invisible-replay-head'):
        a, b = ((b'U'+b'Z'*35, b'Z'*36) if case.startswith('pass18') else (b'A', b'B'))
        h.window(a, b, air=True)
        h.arm()
        before = len(h.keys)
        h.window(b if 'head' in case else a, verdict='POSSIBLE')
        assert len(h.keys) == before
        expected = a+b
        closed = True
    elif case == 'pass3-certain-missing-replay-tail':
        h.window(b'A', b'B', air=True)
        h.arm()
        h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
        h.event('E')
        assert h.keys[-1].kind == 'nak' and h.rx.T == Replay(0, 2)
        h.window(b'A')
        h.window(b'B')
        h.window(b'C', last=True)
        expected = b'ABC'
    elif case in ('pass4-19-20-lower-singleton-retry', 'pass4-19-20-floor-singleton-retry'):
        # DESIGN §3, second amendment: floor retry is NONE after our aired NAK.
        h.window(b'P', b'Q', level=1 if 'floor' in case else 2, air=True)
        h.arm()
        h.window(b'P', level=1)
        assert h.rx.T.cursor == 1
        h.window(b'Q', level=1)
        assert h.rx.T is None
        h.window(b'Z', last=True)
        expected = b'PQZ'
    elif case.startswith('closing-double-final-lost'):
        h.window(b'A', b'B', last=True, air=True)
        h.query(phase=int(case[-1]))
        assert h.keys[-1].kind == 'final' and h.rx.T is None
        expected = b'AB'
    elif case == 'incomplete-closing-double-whole-retry':
        h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
        h.event('X')
        h.event('E')
        assert h.host == b'' and h.rx.T is None
        h.window(b'A', b'B', last=True)
        expected = b'AB'
    elif case.startswith(('pass20-invisible', 'pass19-invisible')):
        heard = case.endswith('heard')
        h.window(b'B', verdict='POSSIBLE')
        if heard:
            h.window(b'C', last=True)
            expected = b'BC'
        else:
            h.query(phase=0, inside=False)
            expected, closed = b'B', True
    elif case == 'pass4-frame10-lost-refused-rungs':
        partial(h, air=True)
        for air in (False, True, True, True):
            h.query(phase=0, inside=True, air=air)
            assert h.keys[-1].form == 'frame10' and h.host == b'A'
        h.window(b'B', last=True)
        expected = b'AB'
    elif case == 'pass5-short-equal-final-next-delivery':
        h.window(b'PQ')
        h.arm()
        # R1: a genuinely new short final agrees with K's prefix and is omitted.
        h.window(b'P', last=True)
        assert h.rx.T is None and h.host == b'PQ'
        h.event('T')
        h.event('G')
        h.window(b'Z', last=True)
        expected = b'PQZ'
    elif case == 'pass5-text-fault-replay':
        h.window(b'PQ', air=True)
        h.arm()
        h.window(Prepared(b'PQZ', 'damaged suffix'))
        expected, closed = b'PQZ', True
        assert h.closes[-1].code == 'fault'
    elif case == 'pass5-final-query-phase0':
        h.window(b'P', last=True, air=True)
        h.query(phase=0, inside=False)
        assert h.keys[-1].basis == 'final'
        expected = b'P'
    elif case == 'pass6-resolved-replay-rearm':
        h.window(b'PQ', air=True)
        h.arm()
        h.window(b'P')
        h.window(b'Q', air=True)
        assert h.rx.T is None and h.rx.K.start == 1
        h.arm()
        assert h.rx.T.start == 1
        h.window(b'Q')
        h.window(b'Z', last=True)
        expected = b'PQZ'
    elif case == 'pass6-silent-cadence-erased-next':
        h.window(b'P')
        h.erase(b'Q')
        h.event('R', recognized=False, inside=False)
        assert h.keys[-1].kind == 'nak'
        h.window(b'Q', last=True)
        expected = b'PQ'
    elif case == 'pass7-89-byte-progressing-replay':
        data = bytes(range(89))
        h.window(data, air=True)
        h.arm()
        for pos in range(0, len(data), 9):
            out = h.window(data[pos:pos+9], air=True)
            assert next(a for a in out if isinstance(a, Commit)).progressed
            h.query(inside=True)
        h.window(b'Z', last=True)
        expected = data+b'Z'
    elif case in ('pass7-repeated-first-chunk-nested', 'pass10-four-lost-chunk-answers',
                  'pass16-PP-new-PC-nested', 'pass17-PQ-nested-repeated-PQ-final'):
        data = b'PP' if case.startswith('pass16') else b'PQRS' if case.startswith('pass10') else b'PQ'
        h.window(data, air=True)
        if case.startswith('pass16'):
            h.erase(b'P', b'C')
        h.arm()
        h.window(b'P', air=True)
        h.query(phaseless=True)
        assert not h.closes and h.rx.T == Replay(0, len(data), alternative=1)
        if case.startswith('pass16'):
            before = len(h.keys)
            h.window(b'P')
            assert len(h.keys) == before
            expected, closed = data, True
        elif case.startswith('pass7'):
            for _ in range(3):
                out = h.window(b'P')
                assert not next(a for a in out if isinstance(a, Commit)).progressed
                h.query(phaseless=True)
            h.event('R', budget=False)
            assert h.closes[-1].code == 'budget'
            expected, closed = data, True
        elif case.startswith('pass10'):
            for value in (b'Q', b'R', b'S'):
                h.window(value)
                h.query(phaseless=True)
            h.window(b'S')
            h.window(b'Z', last=True)
            expected = data+b'Z'
        else:
            h.window(b'P')
            h.window(b'Q')
            h.window(b'Z', last=True)
            expected = data+b'Z'
    elif case == 'pass7-frame10-late-query':
        partial(h)
        h.arm()
        h.window(b'A')
        h.window(b'B', last=True)
        expected = b'AB'
    elif case == 'pass8-equal-head-distinct-continuation':
        h.window(b'A', b'B')
        h.erase(b'A', b'C')
        h.arm()
        h.window(b'A')
        before = len(h.keys)
        h.window(b'C')
        assert len(h.keys) == before
        expected, closed = b'AB', True
    elif case.startswith('pass9-12-'):
        h.window(b'K')
        if case.endswith('failed-tail'):
            h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
            h.event('X')
            h.event('E')
        else:
            h.erase(b'A')
            h.query(phase=0, inside=False)
        assert not h.rx.trusted
        h.window(b'A')
        h.erase(b'B')
        h.query(phase=h.rx.A.phase, inside=False)
        assert h.keys[-1].kind == 'nak'
        h.window(b'B', last=True)
        expected = b'KAB'
    elif case == 'pass13-frame10-tail-repacked-erased-745':
        partial(h)
        h.window(b'B')
        h.erase(b'C')
        h.query(phase=1, inside=False, bw500_poll=True)
        assert h.keys[-1].kind == 'nak' and not h.rx.trusted
        h.window(b'C', last=True)
        expected = b'ABC'
    elif case.startswith('pass15-abort'):
        h.window(b'P')
        if '-even-' in case:
            h.window(b'Q')
        h.event('T')
        h.event('G')
        if '-second-' in case:
            h.window(b'R')
        expected = h.host+b'Z'
        h.erase(b'Z')
        h.query(phase=h.rx.A.phase, inside=False)
        assert h.keys[-1].kind == 'nak'
        h.window(b'Z', last=True)
    elif case == 'unchanged-geometry-changed-hint':
        h.window(b'A', hint=2, air=True)
        h.arm()
        h.window(b'A', hint=6)
        h.window(b'Z', last=True)
        expected = b'AZ'
    elif case == 'refused-nak-owed-keyed-on-query':
        h.window(b'PQ', air=True)
        h.arm(air=False)
        assert h.rx.T is None
        h.query()
        assert h.rx.T == Replay(0, 2)
        h.window(b'PQ')
        h.window(b'Z', last=True)
        expected = b'PQZ'
    elif case == 'grant-first-attempt-erased':
        h.event('G')
        h.erase(b'A')
        h.query(phase=0)
        assert h.keys[-1].kind == 'nak'
        h.window(b'A', last=True)
        expected = b'A'
    elif case == 'held-located-failure':
        held(h)
        h.event('X')
        assert h.keys[-1].kind == 'nak' and not h.host
        h.window(b'A', b'B', last=True)
        expected = b'AB'
    elif case == 'held-open-ceiling':
        h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
        h.event('C')
        assert h.keys[-1].kind == 'nak' and not h.host
        h.window(b'A', b'B', last=True)
        expected = b'AB'
    else:
        pytest.fail(f'unimplemented fixed script: {case} / {role}')
    assert h.host == expected, (case, role, h.host)
    assert h.rx.closed == closed
    if closed:
        assert h.closes[-1].reason and h.closes[-1].frontier == len(h.rx.R)
        n = len(h.keys)
        h.event('F', frame=Frame(b'WRONG'))
        h.query()
        assert h.host == expected and len(h.keys) == n


@pytest.mark.parametrize('answer_before', [False, True])
@pytest.mark.parametrize('replay', [False, True])
def test_text_fault_preflight_both_turn_orders(answer_before, replay):
    h = Harness('TEXT')
    if replay:
        h.window(b'PQ')
        h.arm()
    before = len(h.keys)
    h.window(Prepared(b'PQZ', 'invalid TEXT suffix'), answer_before=answer_before)
    assert h.host == b'PQZ' and len(h.keys) == before
    assert h.closes[-1].code == 'fault'
    assert h.closes[-1].rejected == b'PQZ'


@pytest.mark.parametrize('caller,raw', [('W1AGC', 0), ('W0AEB', 0), ('W1AEU', 0x14), ('W0AGT', 0x14)])
def test_pass4_marker_ordering_prepared_interface(caller, raw):
    # §10a excludes parsing: this pins the prepared-input boundary; §6a adapter pins remain owed.
    body = b'payload\x14\x01\x00\x00\x80'
    payload = body[:body.index(b'\x14\x01')]
    h = Harness()
    h.event('F', frame=Frame(body, full=False), verdict='NONE')
    h.event('F', frame=Frame(b'end', last=True))
    h.event('E')
    h.commit((payload, b'end'))
    assert h.host == b'payloadend', (caller, raw)


FILES_CASES = ('stored-hold', 'stored-then-deflate', 'later-fault', 'terminal-partial',
               'R1-stored-fault', 'R9-stored-realignment', 'R9-deflate-realignment',
               'R10-exhaustion', 'replenishment', 'abort-inside-block', 'abort-after-held-block',
               'long-32500-intermittent-loss', 'deflate-releases-at-validation')


@pytest.mark.parametrize('answer_before', [False, True])
@pytest.mark.parametrize('case', FILES_CASES)
def test_pass6_11_13_15_files_orderings(case, answer_before):
    h = Harness('FILES')
    if case == 'stored-hold':
        h.window(stored(b'abc'), answer_before=answer_before)
        assert not h.host
        h.window(b'', last=True)
        assert h.host == b'abc'
    elif case == 'stored-then-deflate':
        h.window(stored(b'abc'))
        h.window(deflated(b'hello'*100))
        assert not h.host
        h.window(b'', last=True)
        assert h.host == b'abc'+b'hello'*100
    elif case in ('later-fault', 'terminal-partial'):
        h.window(stored(b'held'))
        n = len(h.keys)
        wire = b'\x80\x03BAD' if case == 'later-fault' else b'\x00\x05ab'
        h.window(wire, last=True, answer_before=answer_before)
        assert not h.host and len(h.keys) == n
        assert h.closes[-1].code == 'fault'
        assert h.closes[-1].blocks[0].data == b'held'
    elif case == 'R1-stored-fault':
        prefix = b'\x00\x06'
        h.window(prefix)
        h.arm()
        # R1 removes two genuinely new body bytes equal to K, leaving the stored block short.
        h.window(prefix)
        h.window(b'ABCD', last=True, answer_before=answer_before)
        assert not h.host and h.closes[-1].code == 'fault'
    elif case == 'R9-stored-realignment':
        prefix = b'\x00\x10' + b'P'*7
        h.window(prefix)
        h.arm()
        # Pass 13 R9: omitted nine-byte body suffix consumes a whole following stored record.
        h.window(prefix)
        h.window(stored(b'U'*7)+deflated(b'Z'), last=True)
        original = prefix+prefix+stored(b'U'*7)+deflated(b'Z')
        edited = prefix+stored(b'U'*7)+deflated(b'Z')
        assert witness_files(original, True)[1] is None
        assert h.host == witness_files(edited, True)[0] == b'P'*7+b'\x00\x07'+b'U'*7+b'Z'
        assert not h.closes
    elif case == 'R9-deflate-realignment':
        # Pass 14 construction: a final DEFLATE stored sub-block has no checksum.
        p = b'ABCDEFGHI'
        d = zlib.compressobj(wbits=-15)
        prefix = d.compress(b'Z'*500)+d.flush(zlib.Z_SYNC_FLUSH)
        tail = b'v'*8+p+p+b'q'*20
        raw = prefix+b'\x01'+len(tail).to_bytes(2, 'little')+(65535-len(tail)).to_bytes(2, 'little')+tail
        wire = (0x8000 | len(raw)).to_bytes(2, 'big')+raw
        assert len(raw) == 63 and wire[27:45] == p+p
        following = stored(b'U'*7)
        witness = StreamWitness(wire+following)
        witness.omit('R1', 3, 36, 45)
        assert witness_files(wire+following, True)[1] is None
        h.window(wire[:27])
        h.window(p)
        h.arm()
        h.window(p)
        h.window(wire[45:]+following, last=True)
        assert h.host == witness_files(witness.edited(), True)[0]
        assert h.host == b'Z'*500+b'v'*8+p+b'q'*20+following
        assert not h.closes
    elif case == 'R10-exhaustion':
        for _ in range(4):
            h.window(stored(b'x'*32500))
        assert not h.host and not h.closes
        n = len(h.keys)
        h.window(stored(b'y'*1073), answer_before=answer_before)
        assert len(h.keys) == n and not h.host
        assert h.closes[-1].code == 'R10'
        assert sum(len(b.data) for b in h.closes[-1].blocks) == 131073
    elif case == 'replenishment':
        for _ in range(3):
            for _ in range(4):
                h.window(stored(b'x'*32500))
            assert not h.closes
            h.window(b'', last=True)
        assert h.host == b'x'*(32500*12)
    elif case.startswith('abort-'):
        wire = stored(b'held')
        h.window(wire[:3] if case == 'abort-inside-block' else wire)
        out = h.event('T')
        reports = [a for a in out if isinstance(a, Report)]
        assert reports and (reports[0].pending or reports[0].blocks)
        assert any(isinstance(a, ResetDecoder) for a in out) and not h.host
        h.parser_pending = b''
        h.event('G')
        h.window(stored(b'new'), last=True)
        assert h.host == b'new' and not h.closes
    elif case == 'long-32500-intermittent-loss':
        data = bytes(i % 251 for i in range(32500))
        wire = stored(data)
        for i in range(0, len(wire), 89):
            out = h.window(wire[i:i+89])
            assert next(a for a in out if isinstance(a, Commit)).progressed
            if i % 7 == 0:
                h.query(inside=True)
            assert not h.host
        h.window(b'', last=True)
        assert h.host == data and not h.closes
    elif case == 'deflate-releases-at-validation':
        wire = deflated(b'A'*32500)
        h.window(wire[:4])
        assert not h.host
        h.window(wire[4:])
        assert h.host == b'A'*32500
        h.window(b'', last=True)
    else:
        pytest.fail(case)


def test_files_preflight_is_pure_and_requires_parser_preview():
    hold = FilesHold((Block(b'held', True),), b'\x00')
    before = hold
    after, data, report = hold.preview(FilesParse((Block(b'valid', False),), b'partial'), True)
    assert hold == before == after and data == b'' and report.code == 'fault'
    h = Harness('FILES')
    h.event('F', frame=Frame(b'wire'))
    h.event('E')
    with pytest.raises(ValueError, match='parser preflight'):
        h.rx.commit((Prepared(b'wire'),))
    assert not h.rx.R and not h.keys


SEEDS = (0, 1, 2, 3, 0x430, 0x1031, 0x5EED, 0xC0FFEE)
WALK_RESULTS = []
WALK_FAILURES = []
CHANNEL_PROFILES = json.loads((Path(__file__).parent / 'fixtures/rxwindow-channel.json').read_text())['profiles']


@dataclass(frozen=True)
class Residual:
    code: str
    step: int
    start: int
    end: int
    data: bytes = b''


class StreamWitness:
    """History-owned edit map; equality alone cannot label a genuine retry R1."""
    def __init__(self, wire):
        self.wire = wire
        self.edits = []
        self.sources = {}

    def admitted(self, attempt, ordinal, source):
        assert source == 'peer', ('P-DATA-SPAN/source', attempt, ordinal, source)
        self.sources[attempt, ordinal] = source

    def credited(self, attempt, ordinals):
        assert all(self.sources.get((attempt, i)) == 'peer' for i in ordinals), (
            'P-DATA-SPAN/credit', attempt, ordinals, self.sources)

    def omit(self, code, step, start, end):
        assert code in ('R1', 'R2', 'R7') and start < end
        assert not any(max(start, e.start) < min(end, e.end) for e in self.edits if not e.data)
        self.edits.append(Residual(code, step, start, end))

    def insert(self, step, start, data):
        self.edits.append(Residual('R5', step, start, start, data))

    def omitted(self, offset):
        return any(not e.data and e.start <= offset < e.end for e in self.edits)

    def mapped(self, offset):
        dropped = sum(max(0, min(offset, e.end)-e.start) for e in self.edits if not e.data)
        inserted = sum(len(e.data) for e in self.edits if e.data and e.start <= offset)
        return offset-dropped+inserted

    def edited(self, through=None):
        through = len(self.wire) if through is None else through
        signature = through, tuple(self.edits)
        cached = getattr(self, '_edited_cache', None)
        if cached is not None and cached[0] == signature:
            return cached[1]
        out = bytearray()
        for i in range(through+1):
            for e in self.edits:
                if e.data and e.start == i:
                    out += e.data
            if i < through and not any(not e.data and e.start <= i < e.end for e in self.edits):
                out.append(self.wire[i])
        result = bytes(out)
        self._edited_cache = signature, result
        return result

    def classify_placement(self, step, attempt_start, visible_start, placement, old_s, replay):
        mapped = self.mapped(visible_start)
        if placement.start == mapped:
            return
        if self.omitted(visible_start):
            assert all(self.omitted(visible_start+i) or
                       self.mapped(visible_start+i) == placement.start+i
                       for i in range(len(placement.data)))
            return
        # R1 needs genuinely new bytes after an ambiguous NAK, not a legitimate retransmission.
        assert replay is not None and replay.alternative is None
        assert self.mapped(attempt_start) == old_s, (step, self.edits, old_s, attempt_start)
        assert placement.start == (replay.start if replay.cursor is None else replay.cursor)
        n = min(len(placement.data), old_s-placement.start)
        self.omit('R1', step, visible_start, visible_start+n)
        assert self.mapped(visible_start) == old_s


def witness_files(wire, terminal=False):
    """Independent whole-prefix codec witness; stored output waits for clean EOF."""
    pos, output, held, blocked = 0, b'', b'', False
    while pos+2 <= len(wire):
        value = wire[pos]*256 + wire[pos+1]
        n, packed = value % 32768, value >= 32768
        if not packed and n > 32500:
            return output, 'fault'
        if pos+n+2 > len(wire):
            break
        body = wire[pos+2:pos+2+n]
        if packed:
            try:
                data = zlib.decompress(body, -15)
                check = zlib.decompressobj(-15)
                check.decompress(body)
                if not check.eof or check.unused_data or len(data) > 32500:
                    return output, 'fault'
            except zlib.error:
                return output, 'fault'
        else:
            data = body
            blocked = True
        if blocked:
            held += data
        else:
            output += data
        if len(held) > 128*1024 and not terminal:
            return output, 'R10'
        pos += n+2
    if terminal:
        if pos != len(wire):
            return output, 'fault'
        output += held
    return output, None


@dataclass(frozen=True)
class PeerFrame:
    offset: int
    replay: bytes
    wire: bytes
    full: bool
    last: bool
    hint: int
    compressed: bool

    def model_frame(self):
        return Frame(self.wire, self.full, self.last, self.compressed, self.hint)

    def prepare(self):
        if self.compressed:
            decoded = decode_text(self.wire)
            assert not decoded.error and decoded.data == self.replay
            return Prepared(decoded.data)
        return Prepared(self.wire)


@dataclass(frozen=True)
class Attempt:
    number: int
    start: int
    phase: int
    frames: tuple[PeerFrame, ...]
    level: int
    bw: str = '2300'

    @property
    def end(self):
        return self.frames[-1].offset+len(self.frames[-1].replay)


class ModelPeer:
    def __init__(self, variant, role, mode, app, *, bw='2300', level=4):
        self.variant, self.role, self.mode, self.app = variant, role, mode, app
        self.bw, self.level = bw, level
        if mode == 'FILES':
            pieces = [app[i:i+257] for i in range(0, len(app), 257)]
            self.wire = b''.join(stored(p) if i % 2 == 0 else deflated(p)
                                 for i, p in enumerate(pieces))
        else:
            self.wire = app
        self.acked = self.alt = self.over_phase = self.stays = self.number = 0
        self.whole_retry = None
        self.history = []

    def capacity(self):
        return phy.body_size(self.bw, phy.level_key(self.bw, self.level))-1

    def key_data(self):
        assert self.acked < len(self.wire)
        cap = self.capacity()
        count = 2 if self.stays > 5 and self.mode != 'TEXT' else 1
        if self.whole_retry is not None:
            frames, level = self.whole_retry
            self.whole_retry = None
        else:
            level, frames, pos = self.level, [], self.acked
            for _ in range(count):
                data = self.wire[pos:pos+cap]
                compressed = self.mode == 'TEXT' and self.number % 3 != 1
                if compressed:
                    while len(encode_text(data)) > cap:
                        data = data[:-1]
                    assert data
                    encoded = encode_text(data)
                else:
                    encoded = data
                last = pos+len(data) == len(self.wire)
                full = len(encoded) == cap
                rest = len(self.wire)-(pos+len(data))
                hint = phy.fit_code(self.bw, level, rest, max(1, level-1),
                                    min(phy.TOP[self.bw], level+1))
                frames.append(PeerFrame(pos, data, encoded, full, last, hint, compressed))
                pos += len(data)
                if last:
                    break
            frames = tuple(frames)
        phase = self.alt if self.variant == 'xara' else self.over_phase
        if self.variant == 'reference':
            if frames[-1].last:
                phase = self.over_phase = 0
            else:
                self.over_phase ^= 1
        attempt = Attempt(self.number, self.acked, phase, frames, level, self.bw)
        self.number += 1
        self.history.append(attempt)
        return attempt

    def query_form(self, attempt, unphased=False):
        if unphased:
            return '288-idle', None
        if (self.bw == '500' and self.variant == 'reference' and self.role == 'initiator'
                and attempt.phase == 1):
            return 'frame12-poll', 1
        if self.variant == 'xara' or self.role == 'responder' or attempt.level >= 7:
            return 'frame11' if attempt.phase == 0 else 'frame12', attempt.phase
        return '60/683' if attempt.phase == 0 else '60/745', attempt.phase

    def hear(self, key, attempt):
        self.alt ^= 1
        if key.kind == 'nak':
            self.stays = 0
            if self.variant == 'reference' and len(attempt.frames) == 2 and attempt.frames[-1].last:
                self.whole_retry = attempt.frames, attempt.level
            else:
                self.level = max(1, self.level-1)
        elif key.kind == 'frame10':
            self.acked = attempt.frames[0].offset+len(attempt.frames[0].replay)
            self.stays = 0
            self.level = max(1, self.level-1)
        else:
            self.acked = attempt.end
            if key.kind == 'final':
                self.stays = self.alt = self.over_phase = 0
            elif key.step == 0:
                self.stays = 0 if self.variant == 'reference' and attempt.frames[-1].hint == 2 else self.stays+1
            else:
                self.stays = 0
                self.level = min(phy.TOP[self.bw], max(1, self.level+({1: 1, 2: -1}.get(key.step, 0))))


@dataclass(frozen=True)
class UIInterference:
    number: int
    level: int
    packets: tuple[bytes | None, ...]
    first: int
    shape: str
    timing: str


class ModelChannel:
    def __init__(self, rng, residuals=False, *, completion=False, generator='harsh'):
        self.rng, self.residuals = rng, residuals
        self.completion, self.generator = completion, generator
        self.data_failures = self.answer_failures = 0
        self.warmup = 0
        self.invisible_first_double = False
        self.injected_double = False
        self.priming_attempt = None
        self.ui_serial = 0
        self.ui_schedule = []
        self.ui_loss_limit = None
        self.ui_counts = dict.fromkeys(('groups', 'packets', 'holes'), 0)
        self.counts = dict.fromkeys(('held', 'held_companion', 'end_query', 'ceiling', 'inside_timer',
                                    'erased', 'X', 'head', 'tail', 'both', 'invisible',
                                    'answer_loss', 'query_loss', 'refusal', 'early', 'late', 'timer'), 0)

    def ui_group(self, count, level, body, *, shape, timing, identical):
        assert 1 <= count <= 18 and level in (1, 2, 4)
        self.ui_serial += 1
        packets = [body if identical else self.rng.randbytes(len(body)) for _ in range(count)]
        if shape == 'damaged':
            packets[count // 2] = None
        elif shape == 'truncated':
            packets[-1] = None
        self.ui_counts['groups'] += 1
        self.ui_counts['packets'] += count
        return UIInterference(self.ui_serial, level, tuple(packets),
                              1000000 * self.ui_serial, shape, timing)

    def ui_loss(self):
        # A collision consumes retry budget even when UI bytes equal peer bytes.
        self.ui_counts['holes'] += 1
        self.counts['X'] += 1
        self.data_failures += 1

    def schedule_ui(self, count, level, contest, *, at=0, attempts=1, shape='intact',
                    timing='data-first', identical=True, reset=None):
        self.ui_schedule.append(dict(count=count, level=level, contest=contest, at=at,
            attempts=attempts, shape=shape, timing=timing, identical=identical, reset=reset))

    @property
    def ui_exhausted(self):
        return self.ui_loss_limit is not None and self.data_failures > self.ui_loss_limit

    def observations(self, attempt, statuses, context):
        from hfmodem.kestrel.rx.broadcast import BODY_SAMPLES, PERIOD
        from hfmodem.kestrel.rx.observation import LocatedFrame
        from hfmodem.kestrel.rx.ui_spans import UISpan

        start = (attempt.number + 1) * 10000000
        plans = [p for p in self.ui_schedule
                 if p['at'] <= attempt.number < p['at'] + p['attempts']
                 and p['level'] == attempt.level and p['contest'] < len(attempt.frames)]
        anchor = plans[0]['contest'] if plans else 0
        stride = {1: 253952, 2: 233472, 4: 201728}.get(attempt.level, 300000)
        observations = [LocatedFrame(i, context.origin(start + (i - anchor) * stride,
            start + (i - anchor) * stride + BODY_SAMPLES.get(attempt.level, 300000),
            attempt.level, int(attempt.bw)),
            f.wire if status == 'read' else None)
            for i, (f, status) in enumerate(zip(attempt.frames, statuses))]
        sources = ['peer'] * len(observations)
        groups = []
        for plan in plans:
            i, level = plan['contest'], plan['level']
            group = self.ui_group(plan['count'], level, attempt.frames[i].wire,
                shape=plan['shape'], timing=plan['timing'], identical=plan['identical'])
            slot = len(group.packets) - 1 if i == 0 else 0
            # Distinct accepted groups may predict the same colliding slot.
            group = replace(group, first=start - slot * PERIOD[level] + 8 * (group.number - 1))
            span = UISpan(group.number, context.tune_epoch, level, group.first - 67072,
                possible_packet_count=len(group.packets), ambiguous_2_or_18=len(group.packets) == 18)
            observations[i] = LocatedFrame(i, context.origin(
                start, start + BODY_SAMPLES[level], level, 500), group.packets[slot])
            sources[i] = ('ui', group.number, slot)
            groups.append((group, span, plan['reset']))
        return tuple(observations), tuple(sources), groups

    def profile(self, level=None):
        source = self.rng.choice(sorted({p['source'] for p in CHANNEL_PROFILES}))
        candidates = [p for p in CHANNEL_PROFILES if p['source'] == source]
        matched = [p for p in candidates if p['level'].split('/L')[-1] == str(level)]
        return self.rng.choice(matched or candidates)

    def frames(self, attempt, recovery, *, account=True):
        out = []
        profile = self.profile(attempt.level) if self.generator == 'calibrated' else None
        for _ in attempt.frames:
            if self.generator == 'calibrated':
                rate = profile['data_loss']
                fraction = profile['located']/profile['failed'] if profile['located'] is not None and profile['failed'] else 0.5
                status = self.rng.choices(('X', 'erased', 'read'),
                    (rate*fraction, rate*(1-fraction), 1-rate))[0]
            else:
                roll = self.rng.randrange(24)
                status = 'X' if roll < 2 else 'erased' if roll < 4 else 'read'
            out.append(status)
        planned_invisible = self.invisible_first_double and not self.injected_double and len(out) == 2
        if self.invisible_first_double and self.priming_attempt is None and len(out) == 2:
            out = ['read', 'read']
            self.priming_attempt = attempt.number
        elif planned_invisible:
            out = ['read', 'invisible']
            self.injected_double = True
        elif self.residuals and len(out) == 2 and self.rng.randrange(5) == 0:
            out = ['read', 'read']
            out[self.rng.randrange(2)] = 'invisible'
        if attempt.number < self.warmup:
            out = ['read']*len(out)
        if self.completion and attempt.number == 0:
            out = ['read']*len(out)
        if self.completion and all(v == 'erased' for v in out):
            out[0] = 'X'
        if self.completion and recovery:
            out = ['read']*len(out)
        if self.data_failures >= 2:
            out = ['read']*len(out)
        if account:
            self.account_frames(out)
        return out

    def account_frames(self, out):
        self.counts['invisible'] += sum(v == 'invisible' for v in out)
        self.data_failures = self.data_failures+1 if any(s != 'read' for s in out) else 0
        if all(v == 'erased' for v in out):
            self.counts['both'] += 1
        for i, status in enumerate(out):
            if status in ('X', 'erased'):
                self.counts[status] += 1
                self.counts['head' if i == 0 else 'tail'] += 1

    @staticmethod
    def envelope(attempt, statuses):
        """Erasure/below-detection carries no envelope energy; CRC failures do."""
        lit = [i for i, s in enumerate(statuses) if s not in ('erased', 'invisible')]
        lengths = phy.envelope_lengths(attempt.bw, attempt.level)
        if lengths is None or not lit:
            return 'inconclusive'
        if len(lit) == 2:
            return 'double'
        if lit == [1] and abs(2*lengths[0]-lengths[1]) > 4096:
            return 'inconclusive'
        return 'single'

    def answer(self, *, force=False, replay=False):
        force = force or self.answer_failures == 2 or (self.completion and replay)
        if self.generator == 'calibrated':
            profile = self.profile()
            if profile['answer_loss'] is None:
                profile = self.rng.choice([p for p in CHANNEL_PROFILES if p['answer_loss'] is not None])
            air = True
            heard = force or self.rng.random() >= profile['answer_loss']
        else:
            air = force or self.rng.randrange(12) != 0
            heard = air and (force or self.rng.randrange(5) != 0)
        self.counts['refusal'] += not air
        self.counts['answer_loss'] += air and not heard
        self.answer_failures = 0 if heard else self.answer_failures+1
        return air, heard


class PropertyWalk:
    def __init__(self, seed, variant, role, mode, content, residuals=False, bw='2300', level=4,
                 cohort='adversarial', generator='harsh', partition='high_speed', progressing=False):
        self.seed, self.cohort, self.generator = seed, cohort, generator
        self.partition, self.progressing, self.content = partition, progressing, content
        self.forced_replay = False
        self.model = DecisionHistory()
        self.clock = self.unkey = 0.0
        self.context = None
        self.rng = random.Random(seed)
        random_bytes = self.rng.randbytes(3600)
        app = {'random': random_bytes, 'zeros': b'\0'*3600,
               'blocks': bytes(range(90))*40,
               'head': b''.join(b'RECURRING'+random_bytes[i:i+31] for i in range(0, 2800, 31))}[content]
        self.peer = ModelPeer(variant, role, mode, app, bw=bw, level=level)
        self.h = Harness(mode)
        self.witness = StreamWitness(self.peer.wire)
        self.channel = ModelChannel(self.rng, residuals, completion=cohort == 'completion', generator=generator)
        self.channel.warmup = 6 if cohort == 'adversarial' and mode != 'TEXT' else 0
        self.channel.invisible_first_double = (cohort == 'adversarial' and bw == '500'
                                               and partition == 'high_speed' and mode != 'TEXT')
        self.step = 0
        self.rx_peer = self.rx_any = 0
        self.rx_nak_sent = False
        self.observed_level = None
        self.read_by_attempt = {}
        self.checked = set()
        self.trace = []
        self.terminal = False
        self.aborted = False
        self.replay_commits = 0
        self.ui_station = None
        self.observations_by_attempt = {}
        self.h.event('G')

    def admit_channel_observations(self, attempt, statuses):
        from hfmodem.kestrel.rx.broadcast import BODY_SAMPLES, PERIOD
        from .test_ui_window_admission import receiver

        if self.ui_station is None:
            self.ui_station, _ = receiver(mode=self.peer.mode)
        hs = self.ui_station
        context = hs.receive_context
        observations, sources, groups = self.channel.observations(
            attempt, statuses, hs.capture_context())
        self.observations_by_attempt[attempt.number] = tuple(zip(observations, sources))
        ledger = context.ledger
        def resolve(group, span):
            if group.shape == 'intact':
                ledger.resolve(span.id, 'singleton' if len(group.packets) == 1 else 'group',
                    count=len(group.packets), validated_end=group.first +
                    (len(group.packets) - 1) * PERIOD[group.level] + BODY_SAMPLES[group.level])
            else:
                ledger.resolve(span.id, 'failed' if group.shape == 'damaged' else 'truncated')
        for group, span, _ in groups:
            ledger.publish(span)
            if group.timing == 'ui-first':
                resolve(group, span)
        decisions = []
        def release():
            decisions.extend(hs._admit_receive_observation(o) for o in observations)
        before = self.h.host, self.h.rx.R, self.h.rx.K
        lag = any(g.timing == 'lag' for g, _, _ in groups)
        context.watermark = 0 if lag else max(o.origin.end_sample for o in observations) + 1
        gated = hs._gate_receive_attempt(observations, release)
        if lag:
            assert gated is None and hs._pending_receive is not None
            assert (self.h.host, self.h.rx.R, self.h.rx.K) == before
            reset = next((reset for _, _, reset in groups if reset is not None), None)
            if reset:
                assert reset in ('decoder', 'tune')
                hs.rx_discontinuity() if reset == 'decoder' else hs.retuned()
                hs._release_receive_attempt()
                assert hs._pending_receive is None and not decisions
                self.trace.append(('UI-reset', attempt.number, reset))
                return ['erased'] * len(statuses), sources
            context.watermark = max(o.origin.end_sample for o in observations) + 1
            hs._release_receive_attempt()
        else:
            assert gated is not None
            decisions.extend(gated)
        assert len(decisions) == len(statuses)
        for i, (decision, source) in enumerate(zip(decisions, sources)):
            if source != 'peer':
                assert decision == 'X', ('P-DATA-SPAN', source, observations[i])
                statuses[i] = 'X'
                self.channel.ui_counts['holes'] += 1
            elif statuses[i] == 'read':
                assert decision == 'F'
        for group, span, _ in groups:
            if group.timing != 'ui-first':
                resolve(group, span)
        if groups:
            self.trace.append(('UI', attempt.number, tuple(sources), tuple(decisions)))
            self.checked.add('P-DATA-SPAN')
        return statuses, sources

    def check_stream(self):
        h = self.h
        if h.rx.mode == 'FILES':
            expected, fault = witness_files(h.rx.R, self.terminal)
            assert not fault
        else:
            expected = h.rx.R
        assert h.host == expected, ('P-stream', self.trace)
        edited = self.witness.edited()
        assert edited.startswith(h.rx.R), ('P-stream/map', self.trace, self.witness.edits)
        self.checked.add('P-stream')

    def consume(self, actions, attempt, statuses, *, air=True):
        before = self.h.host
        model = self.model
        keys = [a for a in actions if isinstance(a, Key)]
        closes = [a for a in actions if isinstance(a, Close)]
        if closes:
            assert not keys
            model.verify_close(closes[0].report)
        else:
            assert model.close is None, ('missing required close', model.close, self.trace)
            assert [k.kind for k in keys] == ([model.expected] if model.expected else [])
        for key in keys:
            assert key.basis == model.basis, ('P-phase/authorization', key, self.context)
            if key.kind != 'nak':
                source = model.saved.source
                assert source == attempt.number, ('P-decide/credit', source, attempt, self.trace)
                credited = attempt.frames[:1] if key.kind == 'frame10' else attempt.frames
                read = self.read_by_attempt.get(attempt.number, set())
                for f in credited:
                    assert f.offset in read or (self.channel.residuals and
                        statuses[attempt.frames.index(f)] == 'invisible' and
                        any(e.code == 'R2' and e.start == f.offset for e in self.witness.edits)), (
                            'P-decide/unread-credit-before-transport', key, self.trace)
                if key.basis == 'bound':
                    assert self.context and self.context['inside'] and not self.context['silent']
                    onset, unkey, bound = self.timing
                    assert 0 <= onset-unkey < bound
                if key.basis == 'phase':
                    c = self.context
                    assert model.trusted and not c['phaseless'] and not c['silent']
                    assert not (c['poll'] and model.saved.phase == 1)
                    assert (model.saved.phase, c['phase']) == (1, 1)
                    self.checked.add('P-phase')
            self.checked.add('P-decide')
        for commit in (a for a in actions if isinstance(a, Commit)):
            assert (commit.start, commit.end, commit.new) == (*model.span, model.new)
        for key in keys:
            model.sent(air)
            if air:
                self.unkey = self.clock
        self.h.take(actions, air)
        emitted = b''.join(a.data for a in actions if isinstance(a, Emit))
        assert self.h.host == before+emitted
        if emitted:
            assert any(isinstance(a, Commit) for a in actions), ('P-emit', self.trace)
        self.checked.add('P-emit')
        assert self.h.rx.R == model.stream
        assert (self.h.rx.alt, self.h.rx.trusted) == (model.alt, model.trusted)
        self.check_stream()

    def note_sent(self, key, air, first, attempt):
        if not air or not first:
            return
        if key.kind == 'nak':
            self.rx_nak_sent = True
        elif key.kind == 'frame10' or key.step != 0:
            self.rx_peer = 0
        elif key.kind == 'continue':
            last_read = next(f for f in reversed(attempt.frames) if f.offset in self.read_by_attempt[attempt.number])
            if last_read.compressed:
                self.rx_peer = self.rx_any = 0
            else:
                self.rx_peer = 0 if last_read.hint == 2 else self.rx_peer+1
                self.rx_any += 1
        elif key.kind == 'final':
            self.rx_peer = self.rx_any = 0

    def apply(self, key, attempt, statuses):
        if key.kind != 'nak':
            credited = attempt.frames[:1] if key.kind == 'frame10' else attempt.frames
            read = self.read_by_attempt.get(attempt.number, set())
            for f in credited:
                if f.offset not in read:
                    assert self.channel.residuals and statuses[attempt.frames.index(f)] == 'invisible'
                    assert any(e.code == 'R2' and e.start == f.offset for e in self.witness.edits)
            self.checked.add('P-read')
        self.peer.hear(key, attempt)
        assert self.witness.mapped(self.peer.acked) <= self.h.rx.S, ('P-place/credit', self.trace)
        self.checked.add('P-place')

    def interleave_ui(self, count, level, contest, *, shape='intact', timing='data-first',
                      identical=True, reset=None):
        from types import SimpleNamespace
        from hfmodem.kestrel.rx.broadcast import BODY_SAMPLES, PERIOD
        from hfmodem.kestrel.rx.observation import LocatedFrame
        from hfmodem.kestrel.rx.ui_spans import UISpan, UISpanLedger
        from .test_ui_window_admission import receiver

        h, model = self.h, self.model
        h.rx.high_speed = self.partition != 'limited'
        payloads = (b'A' * 12, b'A' * 12) if identical else (b'head source', b'tail source')
        frames = tuple(PeerFrame(i * len(payloads[0]), data,
            encode_text(data) if self.peer.mode == 'TEXT' else data,
            True, False, 6, self.peer.mode == 'TEXT') for i, data in enumerate(payloads))
        attempt = Attempt(0, 0, 0, frames, level, '500')
        self.witness = StreamWitness(b''.join(payloads))
        group = self.channel.ui_group(count, level, frames[contest].wire,
            shape=shape, timing=timing, identical=identical)
        hs, io = receiver(mode=self.peer.mode)
        ledger = UISpanLedger(hs._tune_epoch)
        hs.receive_context = SimpleNamespace(ledger=ledger, watermark=0, input_clock=0)
        hs.receive_context.retune = lambda epoch, start: ledger.retune(epoch)
        span = UISpan(group.number, hs._tune_epoch, level, group.first - 67072,
                      possible_packet_count=count, ambiguous_2_or_18=count == 18)
        ledger.publish(span)
        slot = self.rng.randrange(count)
        training = group.first + slot * PERIOD[level]
        observations, sources = [], []
        for i, f in enumerate(frames):
            at = training if i == contest else training + (1 if i else -1) * PERIOD[level] // 2
            raw = group.packets[slot] if i == contest else None
            source = ('ui', group.number, slot) if raw is not None else 'peer'
            observations.append(LocatedFrame(i, hs.capture_context().origin(
                at, at + BODY_SAMPLES[level], level, 500), raw if raw is not None else f.wire))
            sources.append(source)
        self.trace.append(('UI', count, level, contest, shape, timing, identical, reset, slot))
        expected = ['read', 'read']
        expected[contest] = 'X'
        self.channel.ui_loss()
        self.channel.counts['head' if contest == 0 else 'tail'] += 1
        before = h.host, h.rx.R, h.rx.K

        def resolve():
            if shape == 'intact':
                ledger.resolve(span.id, 'singleton' if count == 1 else 'group', count=count,
                    validated_end=group.first + (count - 1) * PERIOD[level] + BODY_SAMPLES[level])
            else:
                ledger.resolve(span.id, 'failed' if shape == 'damaged' else 'truncated')

        def admit():
            decisions = [hs._admit_receive_observation(o) for o in observations]
            assert decisions == ['X' if i == contest else 'F' for i in range(2)]
            limited = self.partition == 'limited' and self.peer.mode != 'TEXT'
            verdict = 'NONE' if self.peer.mode == 'TEXT' else 'CERTAIN'
            model.window(attempt, expected, verdict, limited, single=True, envelope='double')
            for i, (f, observation, decision) in enumerate(zip(frames, observations, decisions)):
                if decision == 'F':
                    self.witness.admitted(attempt.number, i, sources[i])
                    assert observation.body == f.wire
                h.event(decision, frame=f.model_frame() if decision == 'F' else None,
                        verdict=verdict, level=level, envelope='double', limited_pair=limited)
            actions = h.rx.event('E', single=True)
            if any(isinstance(a, Ready) for a in actions):
                assert limited and contest == 1 and h.rx.W.partial
                self.witness.credited(attempt.number, (0,))
                actions = h.rx.commit((frames[0].prepare(),))
                model.committed()
            assert [a.kind for a in actions if isinstance(a, Key)] == [model.expected]
            key = next(a for a in actions if isinstance(a, Key))
            assert key.kind == ('frame10' if limited and contest == 1 else 'nak')
            model.sent(True)
            h.take(actions)
            assert h.rx.R == model.stream and h.host == model.stream

        if timing == 'ui-first':
            resolve()
        if timing != 'lag':
            hs.receive_context.watermark = max(o.origin.end_sample for o in observations) + 1
        decisions = hs._gate_receive_attempt(observations, admit)
        if timing == 'lag':
            assert decisions is None and hs._pending_receive is not None
            assert (h.host, h.rx.R, h.rx.K) == before
            if reset is not None:
                if reset == 'T':
                    hs._receive_window_event('T')
                elif reset == 'decoder':
                    hs.rx_discontinuity()
                else:
                    hs.retuned()
                h.event('T')
                hs._release_receive_attempt()
                assert hs._pending_receive is None
                assert h.rx.W is None and h.rx.K is None and h.rx.T is None
                assert h.host == before[0] and not self.witness.sources
                assert not io.host and not io.warnings
                return self
            hs.receive_context.watermark = max(o.origin.end_sample for o in observations) + 1
            hs._release_receive_attempt()
        else:
            assert decisions is not None
            admit()
        after = h.host, h.rx.R, h.rx.K, len(h.keys)
        if timing != 'ui-first':
            resolve()
        hs._release_receive_attempt()
        assert (h.host, h.rx.R, h.rx.K, len(h.keys)) == after
        assert not h.closes and not io.host and not io.warnings
        assert h.host in (b'', payloads[0])
        assert all(source == 'peer' for source in self.witness.sources.values())
        self.checked.update(('P-DATA-SPAN', 'P-stream', 'P-decide', 'P-emit'))
        return self

    def run(self):
        peer, h = self.peer, self.h
        while peer.acked < len(peer.wire) and not h.rx.closed:
            assert self.step < 2400, ('P-bound', self.seed, self.trace)
            self.step += 1
            attempt = peer.key_data()
            replay = h.rx.T
            true_start = self.witness.mapped(attempt.start)
            allowed = (replay.start, replay.alternative if replay.alternative is not None else replay.frontier) if replay and replay.cursor is None else (
                (replay.cursor,) if replay else (true_start,))
            truthful = true_start in allowed
            ui = bool(self.channel.ui_schedule)
            statuses = self.channel.frames(attempt, bool(replay), account=not ui)
            sources = ['peer'] * len(statuses)
            if ui:
                statuses, sources = self.admit_channel_observations(attempt, statuses)
                self.channel.account_frames(statuses)
            envelope = self.channel.envelope(attempt, statuses)
            floor_single_retry = (self.model.retry_double and
                self.model.retry_level == attempt.level == 1 and envelope == "single")
            self.clock += 8.0
            self.context = None
            force_late = False
            self.model.expected = self.model.basis = None
            self.model.close = None
            self.trace.append((self.step, attempt.start, attempt.level, attempt.phase, tuple(statuses), envelope))
            if all(s == 'erased' for s in statuses):
                actions = []
            else:
                first_read = next((f for f, s in zip(attempt.frames, statuses) if s == 'read'), None)
                if first_read is not None:
                    if self.rx_nak_sent:
                        self.rx_peer = self.rx_any = 0
                        self.rx_nak_sent = False
                    elif self.observed_level != attempt.level:
                        self.rx_peer = self.rx_any = 0
                    self.observed_level = attempt.level
                verdict = ('CERTAIN' if self.rx_peer >= 6 and
                           (peer.bw != '500' or attempt.level == phy.TOP[peer.bw]) else
                           'POSSIBLE' if self.rx_any >= 6 else 'NONE')
                limited = peer.bw == '500' and self.partition == 'limited' and self.rx_peer >= 6
                if limited:
                    verdict = 'CERTAIN'
                if self.model.retry_double and attempt.level <= self.model.retry_level:
                    verdict = ('NONE' if attempt.level < self.model.retry_level or
                               floor_single_retry else 'POSSIBLE')
                if attempt.frames[0].compressed:
                    verdict, limited = 'NONE', False
                visible = [i for i, s in enumerate(statuses) if s == 'read']
                for i, (f, status) in enumerate(zip(attempt.frames, statuses)):
                    if (i == 1 and verdict == 'POSSIBLE' and statuses == ['read', 'read']
                            and attempt.frames[0].full and (attempt.number == self.channel.priming_attempt
                                                          or self.rng.randrange(2) == 0)):
                        held_head = h.rx.event('E', single=True, peer_keying=True, envelope='inconclusive')
                        assert not held_head
                        self.channel.counts['held_companion'] += 1
                    if status == 'invisible':
                        if peer.bw == '500' and attempt.level <= 4 and i == 0:
                            h.event('X', verdict=verdict, level=attempt.level, envelope=envelope)
                        continue
                    kind = 'F' if status == 'read' else 'X'
                    if kind == 'F':
                        self.witness.admitted(attempt.number, i, sources[i])
                    h.event(kind, frame=f.model_frame(), verdict=verdict, level=attempt.level,
                            limited_pair=limited, envelope=envelope)
                    self.check_stream()
                self.read_by_attempt[attempt.number] = {attempt.frames[i].offset for i in visible}
                end_kind = 'E' if attempt.number == self.channel.priming_attempt or attempt.number < self.channel.warmup or self.progressing and not self.forced_replay else self.rng.choice(('E', 'E', 'Q', 'C'))
                held = (end_kind != 'C' and all(v == 'read' for v in statuses)
                        and not (verdict == 'CERTAIN' and len(statuses) == 1)
                        and self.rng.randrange(3) == 0)
                self.channel.counts['held'] += held
                self.channel.counts['end_query'] += end_kind == 'Q'
                self.channel.counts['ceiling'] += end_kind == 'C'
                if held:
                    held_actions = h.rx.event('E', single=True, peer_keying=True)
                    assert not held_actions
                    end_kind = self.rng.choice(('E', 'Q', 'C'))
                model_statuses = list(statuses)
                if peer.bw == '500' and attempt.level <= 4 and statuses[0] == 'invisible':
                    model_statuses[0] = 'X'
                self.model.window(attempt, model_statuses, verdict, limited, single=True,
                                  ceiling=end_kind == 'C' and not held, envelope=envelope)
                end = h.rx.event(end_kind, single=True, envelope=envelope)
                if any(isinstance(a, Ready) for a in end):
                    w = h.rx.W
                    assert self.model.span is not None or self.model.close is not None
                    assert w.partial == self.model.partial
                    self.checked.add('P-decide')
                    credited_visible = visible[:1] if w.partial else visible
                    self.witness.credited(attempt.number, credited_visible)
                    prepared = tuple(attempt.frames[i].prepare() for i in credited_visible)
                    p = h.rx.placement(prepared)
                    if not p.error:
                        if not self.model.partial and any(s == 'invisible' for s in statuses):
                            assert p.start == h.rx.S and (verdict == 'POSSIBLE' or
                                verdict == 'NONE' and floor_single_retry), (
                                'invisible credit requires POSSIBLE or floor single-envelope retry at frontier', self.trace)
                            for f, s in zip(attempt.frames, statuses):
                                if s == 'invisible':
                                    self.witness.omit('R2', self.step, f.offset, f.offset+len(f.replay))
                        self.witness.classify_placement(self.step, attempt.start,
                            attempt.frames[credited_visible[0]].offset, p, h.rx.S, replay)
                        mapped = self.witness.mapped(attempt.frames[credited_visible[0]].offset)
                        is_r1 = any(e.code == 'R1' and e.step == self.step for e in self.witness.edits)
                        start_omitted = self.witness.omitted(attempt.start)
                        assert p.start == mapped or is_r1 or start_omitted, ('P-place', self.trace)
                        if not truthful:
                            # An R1-omitted interval has no image in R; subsequent surviving bytes still must place exactly.
                            assert is_r1 or start_omitted, ('P-truth', self.trace, allowed, true_start)
                        for i in range(len(p.data)):
                            offset = attempt.frames[credited_visible[0]].offset+i
                            if not any(not e.data and e.start <= offset < e.end for e in self.witness.edits):
                                assert self.witness.mapped(offset) == p.start+i, ('P-place/byte', self.trace)
                    self.checked.add('P-truth')
                    terminal = not self.model.partial and attempt.frames[credited_visible[-1]].last
                    parsed = parse_preview(h.parser_pending, p.new, terminal) if peer.mode == 'FILES' else None
                    if not self.model.close and parsed:
                        _, fault = witness_files(self.model.stream+self.model.new, terminal)
                        if fault:
                            self.model.reject(parsed.error or 'FILES hold exhausted', self.model.new, fault,
                                blocks=self.model.blocks+parsed.blocks, pending=parsed.pending)
                    step = ((1 if attempt.level < phy.TOP[peer.bw] else 2) if self.rx_any >= 3 else 0) if self.cohort == 'completion' else 0 if attempt.number == self.channel.priming_attempt or attempt.number < self.channel.warmup else self.rng.choices((0, 1, 2), (16, 1, 1))[0]
                    if attempt.number < self.channel.warmup:
                        step = 0
                    actions = h.rx.commit(prepared, files=parsed, step=step)
                    if parsed and any(isinstance(a, Commit) for a in actions):
                        h.parser_pending = parsed.pending
                    if any(isinstance(a, Commit) for a in actions):
                        self.replay_commits += bool(replay)
                        self.model.committed(parsed=parsed)
                    if any(isinstance(a, Commit) for a in actions) and terminal:
                        self.terminal = True
                else:
                    actions = end
            key = next((a for a in actions if isinstance(a, Key)), None)
            if key:
                air, heard = self.channel.answer(replay=bool(replay), force=attempt.number == self.channel.priming_attempt or attempt.number < self.channel.warmup)
                if self.progressing and not self.forced_replay and key.kind == 'continue':
                    air, heard = True, False
                    self.forced_replay = True
                    force_late = True
                else:
                    force_late = False
                first = not h.rx.A.sent
                self.consume(actions, attempt, statuses, air=air)
                self.note_sent(key, air, first, attempt)
                if self.channel.ui_exhausted:
                    assert key.kind == 'nak'
                    self.model.reject('re-answer budget exhausted', code='budget')
                    self.consume(h.rx.event('R', recognized=False, budget=False),
                                 attempt, statuses)
                    assert h.rx.closed and h.rx.W is None and h.rx.T is None
                    break
                if heard:
                    self.apply(key, attempt, statuses)
                    continue
            elif actions:
                self.consume(actions, attempt, statuses)
            if h.rx.closed:
                break
            for recovery in range(3):
                # Onset is sampled on the channel clock, independently of A's ownership.
                delay = self.rng.choice((0.2, 0.7, 6.4))
                if self.cohort == 'completion':
                    delay = 0.2
                if key and force_late:
                    delay = 6.4
                self.clock += delay
                onset = self.clock
                inside = onset-self.unkey < 3.672
                self.channel.counts['early' if inside else 'late'] += 1
                if self.generator == 'harsh' and recovery < 2 and self.rng.randrange(8) == 0:
                    self.channel.counts['query_loss'] += 1
                    self.check_stream()
                    continue
                unphased = (self.channel.residuals and self.rng.randrange(3) == 0) or (
                    self.progressing and key and force_late)
                form, phase = peer.query_form(attempt, unphased)
                timer = self.cohort != 'completion' and recovery < 2 and self.rng.randrange(6) == 0
                if timer:
                    self.channel.counts['timer'] += 1
                    self.channel.counts['inside_timer'] += inside
                self.timing = onset, self.unkey, 3.672
                self.context = dict(phase=phase, inside=inside, silent=timer,
                    phaseless=unphased, poll=form == 'frame12-poll')
                self.model.recovery(**self.context)
                self.trace.append(('R' if timer else 'Q', phase, onset, self.unkey, inside, form))
                actions = h.rx.event('R' if timer else 'Q', phase=phase, inside=inside,
                    recognized=not timer, phaseless=unphased, bw500_poll=form == 'frame12-poll',
                    explicit=form != '288-idle')
                key = next((a for a in actions if isinstance(a, Key)), None)
                air, heard = self.channel.answer(force=recovery == 2, replay=bool(self.model.replay))
                first = h.rx.A is not None and not h.rx.A.sent
                self.consume(actions, attempt, statuses, air=air)
                if key:
                    self.note_sent(key, air, first, attempt)
                if h.rx.closed:
                    break
                if key and heard:
                    self.apply(key, attempt, statuses)
                    break
            else:
                pytest.fail(f'P-bound recovery failed: seed={self.seed} trace={self.trace}')
        if peer.acked == len(peer.wire) and not self.terminal and not h.rx.closed:
            assert self.cohort == 'adversarial' and statuses[-1] == 'invisible'
            assert any(e.code == 'R2' and e.end == len(peer.wire) for e in self.witness.edits)
            before = h.host
            actions = h.rx.event('T')
            if peer.mode == 'FILES':
                start = self.model.anchor[0] if self.model.anchor else len(self.model.stream)
                expected = Report('abort', 'delivery aborted', self.model.blocks,
                                  self.model.pending, start, len(self.model.stream))
                assert [a for a in actions if isinstance(a, Report)] == [expected]
                assert any(isinstance(a, ResetDecoder) for a in actions)
            assert not any(isinstance(a, (Key, Emit, Close)) for a in actions)
            h.take(actions)
            assert h.host == before and h.rx.K is None and h.rx.T is None
            assert h.rx.alt == 0 and not h.rx.trusted
            self.trace.append(('T', 'R2 erased the terminal frame; no clean FILES flush'))
            self.aborted = True
        if h.rx.closed:
            self.model.verify_close(h.closes[-1])
            assert self.cohort != 'completion', ('completion cohort closed', self.trace)
        elif self.aborted:
            expected = witness_files(h.rx.R, False)[0] if peer.mode == 'FILES' else self.witness.edited()
            assert h.host == expected
        else:
            assert peer.acked == len(peer.wire) and self.terminal
            edited = self.witness.edited()
            expected = witness_files(edited, True)[0] if peer.mode == 'FILES' else edited
            assert h.host == expected
            if not self.witness.edits:
                assert h.host == peer.app
        if self.progressing:
            assert self.forced_replay and self.replay_commits >= 2
        self.checked.add('P-bound')
        assert {'P-stream', 'P-decide', 'P-place', 'P-emit', 'P-bound'} <= self.checked
        WALK_RESULTS.append(dict(seed=self.seed, variant=peer.variant, role=peer.role,
            mode=peer.mode, content=self.content, bw=peer.bw, level=peer.history[0].level, cohort=self.cohort,
            generator=self.generator, progressing=self.progressing, replay_commits=self.replay_commits, trace=self.trace,
            steps=self.step, outcome=h.closes[-1].code if h.rx.closed else 'abort' if self.aborted else 'complete',
            checked=sorted(self.checked), channel=self.channel.counts,
            residuals=[e.code for e in self.witness.edits]))
        return self


def adversarial_disposition(config):
    mode, seed = config['mode'], config['seed']
    h = Harness(mode)
    rng = random.Random(seed)
    trace = []
    def payload(n):
        raw = rng.randbytes(n)
        if config['content'] == 'zeros':
            return b'\0'*n
        if config['content'] == 'blocks':
            return (bytes(range(90))*(n//90+1))[:n]
        if config['content'] == 'head':
            return ((b'RECURRING'+raw[:31])*(n//40+1))[:n]
        return raw
    if mode == 'OFF':
        prefix = payload(37)
        h.window(prefix)
        h.event('T')
        assert h.rx.alt == 0 and not h.rx.trusted
        h.event('G')
        h.event('R', recognized=False, budget=False)
        expected = Report('budget', 're-answer budget exhausted', start=len(prefix), frontier=len(prefix))
        assert h.closes == [expected] and h.host == prefix
        trace = [('commit', prefix.hex()), ('T',), ('G',), ('R', 'budget exhausted')]
        outcome = 'budget'
    elif mode == 'TEXT':
        wire, prefix = payload(89), payload(23)
        h.event('F', frame=Frame(wire, compressed=True))
        h.event('E')
        h.commit((Prepared(prefix, 'injected TEXT fault'),))
        expected = Report('fault', 'injected TEXT fault', start=0, frontier=len(prefix), rejected=prefix)
        assert h.closes == [expected] and not h.keys and h.host == prefix
        trace = [('F', wire.hex()), ('prepared-fault-prefix', prefix.hex())]
        outcome = 'fault'
    else:
        peer = ModelPeer(config['variant'], config['role'], 'FILES', payload(5*32500),
                         level=phy.TOP['2300'])
        frontier, anchor, pending, blocks = 0, 0, b'', ()
        while not h.closes:
            attempt = peer.key_data()
            data = b''.join(f.replay for f in attempt.frames)
            parsed = parse_preview(pending, data, False)
            proposed = blocks+parsed.blocks
            expected_close = sum(len(b.data) for b in proposed) > 128*1024
            trace.append(('DATA', attempt.start, [f.replay.hex() for f in attempt.frames]))
            h.window(*(f.replay for f in attempt.frames), level=attempt.level)
            if expected_close:
                expected = Report('R10', 'FILES hold exhausted', proposed, parsed.pending,
                                  anchor, frontier, data)
                assert h.closes == [expected] and not h.host
            else:
                assert not h.closes and not h.host
                blocks, pending, anchor = proposed, parsed.pending, frontier
                frontier += len(data)
                peer.hear(h.keys[-1], attempt)
        outcome = 'R10'
    before = h.host
    assert h.event('F', frame=Frame(b'after close')) == [] and h.host == before
    WALK_RESULTS.append(dict(seed=seed, variant=config['variant'], role=config['role'],
        mode=mode, content=config['content'], bw='2300', level=config['level'], cohort='adversarial',
        generator=config['generator'], trace=trace, steps=len(trace), outcome=outcome,
        checked=['P-stream', 'P-decide', 'P-bound', 'P-emit'],
        channel=dict.fromkeys(ModelChannel(rng).counts, 0), residuals=[]))


def walk_configuration(index, cohort, generator):
    seed = SEEDS[index % len(SEEDS)]
    variant = ('reference', 'xara')[(index // 2) % 2]
    role = ('initiator', 'responder')[index % 2]
    mode = ('OFF', 'TEXT', 'FILES')[(index // 4) % 3]
    content = ('random', 'zeros', 'blocks', 'head')[(index // 12) % 4]
    bw, level, partition = ('2300', 4, 'high_speed')
    if index % 5 == 0:
        bw, level = '500', 3
        partition = 'limited' if index % 10 == 0 else 'high_speed'
    elif index % 5 == 1:
        level = 7
    progressing = cohort == 'completion' and (index // 48 + index // 12) % 2 == 0
    if progressing:
        bw, level, partition = '2300', 4, 'high_speed'
    return dict(seed=seed, variant=variant, role=role, mode=mode, content=content,
        residuals=cohort == 'adversarial', bw=bw, level=level, partition=partition,
        cohort=cohort, generator=generator, progressing=progressing)


@pytest.mark.parametrize('generator', ['harsh', 'calibrated'])
@pytest.mark.parametrize('cohort', ['completion', 'adversarial'])
@pytest.mark.parametrize('index', range(240))
def test_section10a_seeded_peer_channel_walk(index, cohort, generator):
    config = walk_configuration(index, cohort, generator)
    walk = None
    try:
        if cohort == 'adversarial' and index >= 216:
            adversarial_disposition(config)
            return
        walk = PropertyWalk(**config)
        walk.run()
        if cohort == 'completion':
            assert not walk.witness.edits and not walk.h.closes
            assert walk.h.host == walk.peer.app
    except AssertionError as e:
        trace = walk.trace if walk else []
        WALK_FAILURES.append(dict(index=index, configuration=config, failure=str(e), trace=trace))
        pytest.fail(f'configuration={config}: {e}; full_trace={json.dumps(trace)}')


@pytest.mark.parametrize('a_phase', [0, 1])
def test_bw500_frame12_poll_phase_proof_partition(a_phase):
    h = Harness()
    if a_phase:
        h.window(b'0')
    h.window(b'A')
    h.query(phase=1, bw500_poll=True, inside=False)
    assert h.keys[-1].kind == 'nak'
    assert (h.rx.T is not None) == bool(a_phase)


@pytest.mark.parametrize('verdict,companion,outcome', [
    ('NONE', False, 'final'), ('POSSIBLE', False, 'final'),
    ('CERTAIN', False, 'nak'), ('POSSIBLE', True, 'final'),
    ('CERTAIN', True, 'final'),
])
def test_caller_key_collision_waits_for_window_evidence(verdict, companion, outcome):
    h = Harness()
    h.event('F', frame=Frame(b'..\x14\x80', full=False, ambiguous=True), verdict=verdict)
    assert not h.keys and not h.host
    if companion:
        h.event('F', frame=Frame(b'close', last=True))
        assert not h.rx.W.frames[0].last
    out = h.event('E', single=True)
    if isinstance(out[0], Ready):
        if not companion:
            assert h.rx.W.frames[0].last
        h.commit((b'head', b'tail') if companion else (b'head',))
    assert h.keys[-1].kind == outcome
    assert h.host == (b'headtail' if companion else b'head' if outcome == 'final' else b'')


@pytest.mark.parametrize('seed', SEEDS, ids=lambda n: f'seed-{n:08x}')
@pytest.mark.parametrize('mode', ['OFF', 'TEXT', 'FILES'])
def test_opt_in_R5_false_nak_and_R7_marker_edits(seed, mode):
    rng = random.Random(seed)
    app = rng.randbytes(40)
    if mode == 'FILES':
        first, rest = stored(app[:20]), stored(app[20:])
    else:
        first, rest = app[:20], app[20:]
    witness = StreamWitness(first+rest)
    h = Harness(mode)
    h.window(first)
    # R5 is enabled by a peer false decode; no NAK from the receive machine arms T.
    witness.insert(2, len(first), first)
    h.window(first)
    assert h.rx.T is None
    h.window(rest, last=True)
    expected = witness_files(witness.edited(), True)[0] if mode == 'FILES' else witness.edited()
    assert h.host == expected
    assert [e.code for e in witness.edits] == ['R5']
    if mode == 'FILES':
        wire = stored(b'A\x14')
        marker_offset = len(wire)-1
    else:
        wire = b'A\x14'
        marker_offset = 1
    r7 = StreamWitness(wire)
    # §6a's raw-0x14 collision is a parser input edit, not permission to exempt other bytes.
    r7.omit('R7', 1, marker_offset, marker_offset+1)
    h = Harness(mode)
    h.window(r7.edited(), last=True)
    expected, error = witness_files(r7.edited(), True) if mode == 'FILES' else (r7.edited(), None)
    assert h.host == expected
    assert bool(h.closes) == bool(error)
    if error:
        assert h.closes[-1].code == 'fault'


@pytest.mark.parametrize('seed', SEEDS, ids=lambda n: f'seed-{n:08x}')
@pytest.mark.parametrize('new_session', [False, True])
def test_seeded_aborted_files_decoder_and_next_delivery(seed, new_session):
    rng = random.Random(seed)
    h = Harness('FILES')
    app = rng.randbytes(100)
    wire = stored(app)
    split = rng.randrange(1, len(wire)+1)
    for i in range(0, split, 9):
        h.window(wire[i:min(i+9, split)])
    assert not h.host
    out = h.event('T', new_session=new_session)
    report, = (a for a in out if isinstance(a, Report))
    assert report.code == 'abort' and (report.pending or report.blocks)
    assert h.rx.trusted == new_session
    assert h.rx.alt == 0 and h.rx.T is None
    h.parser_pending = b''
    h.event('G')
    h.window(deflated(b'fresh'), last=True)
    assert h.host == b'fresh'


def test_replay_oracle_never_labels_a_misplaced_genuine_retry_R1():
    from hfmodem.kestrel.arq.rxwindow import Placement
    witness = StreamWitness(b'PQ')
    with pytest.raises(AssertionError):
        witness.classify_placement(3, 0, 0, Placement(1, b'P', b'', None), 2, Replay(0, 2))
    assert not witness.edits


def test_files_rejected_preflight_does_not_install_decoder_or_emit():
    h = Harness('FILES')
    h.window(stored(b'held'))
    before = h.rx.R, h.parser_pending
    h.window(deflated(b'valid') + b'\x00\x04bad', last=True, answer_before=True)
    assert (h.rx.R, h.parser_pending) == before and not h.host
    assert h.closes[-1].code == 'fault'
    assert b'held' in b''.join(b.data for b in h.closes[-1].blocks)


def test_deflate_without_stored_hold_is_not_charged_to_R10():
    blocks = tuple(Block(b'a'*32500, False) for _ in range(5))
    hold, data, report = FilesHold().preview(FilesParse(blocks), False)
    assert not report and not hold.blocks and len(data) == 162500


def test_seed_corpus_property_and_channel_coverage(record_testsuite_property):
    if len(WALK_RESULTS) < 960:
        pytest.skip('coverage aggregation requires the complete seed corpus in this process')
    checked = set().union(*(set(r['checked']) for r in WALK_RESULTS))
    assert checked == {'P-stream', 'P-decide', 'P-read', 'P-place', 'P-truth',
                       'P-emit', 'P-phase', 'P-bound'}
    channels = {k: sum(r['channel'][k] for r in WALK_RESULTS) for k in WALK_RESULTS[0]['channel']}
    assert all(channels.values()), channels
    for generator in ('harsh', 'calibrated'):
        complete = [r for r in WALK_RESULTS if r['generator'] == generator and r['cohort'] == 'completion']
        adversarial = [r for r in WALK_RESULTS if r['generator'] == generator and r['cohort'] == 'adversarial']
        assert len(complete) == len(adversarial) == 240
        for cohort in (complete, adversarial):
            assert all(sum(r['variant'] == v for r in cohort) == 120 for v in ('reference', 'xara'))
            assert all(sum(r['role'] == v for r in cohort) == 120 for v in ('initiator', 'responder'))
            assert all(sum(r['mode'] == v for r in cohort) == 80 for v in ('OFF', 'TEXT', 'FILES'))
            assert all(sum(r['content'] == v for r in cohort) == 60 for v in ('random', 'zeros', 'blocks', 'head'))
        assert all(r['outcome'] == 'complete' for r in complete)
        assert any(r['outcome'] == 'R8' for r in adversarial)
    assert {e for r in WALK_RESULTS for e in r['residuals']} >= {'R1', 'R2'}
    assert {r['seed'] for r in WALK_RESULTS} == set(SEEDS)
    outcomes = {k: sum(r['outcome'] == k for r in WALK_RESULTS)
                for k in sorted({r['outcome'] for r in WALK_RESULTS})}
    by_generator = {g: {c: {o: sum(r['generator'] == g and r['cohort'] == c and r['outcome'] == o for r in WALK_RESULTS) for o in outcomes} for c in ('completion', 'adversarial')} for g in ('harsh', 'calibrated')}
    summary = dict(seeds=[f'{s:#x}' for s in SEEDS], walks=len(WALK_RESULTS), by_generator=by_generator,
                   outcomes=outcomes, properties=sorted(checked), channel=channels,
                   residuals={k: sum(r['residuals'].count(k) for r in WALK_RESULTS) for k in ('R1', 'R2')})
    record_testsuite_property('rxwindow_seed_corpus', json.dumps(summary, sort_keys=True))


@pytest.mark.parametrize('air', [False, True])
def test_decide_oracle_rejects_whole_credit_for_partial_before_transport(air):
    from dataclasses import replace
    walk = PropertyWalk(0, 'reference', 'responder', 'OFF', 'random', bw='500', level=3)
    peer, h = walk.peer, walk.h
    peer.stays = 6
    attempt = peer.key_data()
    statuses = ['read', 'X']
    walk.read_by_attempt[attempt.number] = {attempt.frames[0].offset}
    walk.model.window(attempt, statuses, 'CERTAIN', True, single=False)
    h.event('F', frame=attempt.frames[0].model_frame(), verdict='CERTAIN', limited_pair=True)
    h.event('X')
    h.event('E')
    actions = h.rx.commit((attempt.frames[0].prepare(),))
    walk.model.committed()
    unsafe = [replace(a, kind='continue') if isinstance(a, Key) else a for a in actions]
    with pytest.raises(AssertionError):
        walk.consume(unsafe, attempt, statuses, air=air)


def test_close_oracle_rejects_plausibly_labelled_but_unjustified_R8():
    model = DecisionHistory()
    with pytest.raises(AssertionError, match='unjustified close'):
        model.verify_close(Report('R8', 'nested phaseless NAK', start=0, frontier=1))


@pytest.mark.parametrize('invisible', [0, 1])
@pytest.mark.parametrize('content', ['distinct', 'repeated'])
def test_peer_counts_drive_invisible_double_inside_progressing_replay(invisible, content):
    data = random.Random(18).randbytes(5000) if content == 'distinct' else b'Z'*5000
    peer = ModelPeer('xara', 'responder', 'OFF', data, bw='500', level=4)
    h = Harness()
    epoch_holds = any_holds = 0
    for _ in range(6):
        a = peer.key_data()
        h.window(*(f.replay for f in a.frames), level=a.level)
        peer.hear(h.keys[-1], a)
        epoch_holds = 0 if a.frames[-1].hint == 2 else epoch_holds+1
        any_holds += 1
    verdict = 'POSSIBLE' if any_holds >= 6 else 'NONE'
    a = peer.key_data()
    assert len(a.frames) == 2 and verdict == 'POSSIBLE'
    h.window(*(f.replay for f in a.frames), verdict=verdict, level=4)
    h.arm()
    peer.hear(h.keys[-1], a)
    for _ in range(2):
        retry = peer.key_data()
        h.event('X', level=retry.level)
        h.event('E')
        peer.hear(h.keys[-1], retry)
    assert peer.level == 1
    epoch_holds = any_holds = 0
    cursor = a.start
    for _ in range(6):
        retry = peer.key_data()
        assert len(retry.frames) == 1 and retry.start == cursor
        h.window(retry.frames[0].replay, verdict='NONE', level=1)
        peer.hear(h.keys[-1], retry)
        epoch_holds = 0 if retry.frames[-1].hint == 2 else epoch_holds+1
        any_holds += 1
        cursor = retry.end
    assert h.rx.T and cursor < a.end and peer.stays == 6
    # High speed capable BW500 below top cannot be CERTAIN, even with six peer holds (§3).
    verdict = 'CERTAIN' if epoch_holds >= 6 and peer.level == phy.TOP[peer.bw] else 'POSSIBLE' if any_holds >= 6 else 'NONE'
    retry = peer.key_data()
    assert len(retry.frames) == 2 and verdict == 'POSSIBLE'
    statuses = ['read', 'read']
    statuses[invisible] = 'invisible'
    observed = retry.frames[1-invisible]
    before, keys = h.host, len(h.keys)
    if invisible == 0:
        h.event('X', verdict=verdict, level=1)
    h.event('F', frame=observed.model_frame(), level=1, verdict=verdict)
    h.event('E', single=True)
    if invisible == 0:
        assert not h.closes and h.keys[-1].kind == 'nak' and len(h.keys) == keys+1
    else:
        model = DecisionHistory()
        model.stream, model.anchor = h.rx.R, (a.start, a.end, False)
        model.replay = a.start, a.end, cursor, None
        model.window(retry, statuses, verdict, False, single=True)
        h.commit((observed.replay,))
        model.verify_close(h.closes[-1])
        assert len(h.keys) == keys
    assert h.host == before


def test_files_frame10_unread_terminal_tail_does_not_flush_head():
    peer = ModelPeer('reference', 'responder', 'FILES', b'head-and-tail', bw='500', level=1)
    peer.stays = 6
    attempt = peer.key_data()
    assert len(attempt.frames) == 2 and attempt.frames[-1].last
    h = Harness('FILES')
    h.event('F', frame=attempt.frames[0].model_frame(), verdict='CERTAIN', limited_pair=True)
    h.event('X')
    h.event('E')
    h.commit((attempt.frames[0].replay,))
    assert h.keys[-1].kind == 'frame10' and not h.closes and not h.host
    assert h.rx.files.pending == attempt.frames[0].replay
    peer.hear(h.keys[-1], attempt)
    tail = peer.key_data()
    assert len(tail.frames) == 1 and tail.frames[0].last
    h.window(tail.frames[0].replay, last=True)
    assert h.host == peer.app and not h.closes


@pytest.mark.parametrize('seed', ['1031', 'c0ffee'])
def test_base_comparison_history_amended_counts(seed):
    """Operator decision B: both base histories complete with unconditional count resets."""
    case = json.loads((Path(__file__).parent/f'fixtures/rxwindow-base-{seed}.json').read_text())
    peer = ModelPeer(case['variant'], case['role'], 'OFF', random.Random(case['seed']).randbytes(256))
    h, model = Harness(), DecisionHistory()
    h.event('G')
    epoch_holds = any_holds = 0
    nak_sent = False
    observed_level = None
    now = unkey = 0.0
    attempt = None
    for e in case['trace']:
        if e['event'] == 'DATA':
            model.expected = model.basis = model.close = None
            attempt = peer.key_data()
            assert (attempt.start, attempt.level, attempt.phase) == (e['offset'], e['level'], e['phase'])
            assert [f.replay.hex() for f in attempt.frames] == e['payloads']
            record = varahf2300.RECORDS[
                phy.level_key('2300', attempt.level)]
            now += (record.lead+len(attempt.frames)*record.ncols)*record.dw50/48000
            statuses = e['status']
            envelope = ModelChannel.envelope(attempt, statuses)
            if all(v == 'erased' for v in statuses):
                actions = []
                continue
            first_read = next((f for f, s in zip(attempt.frames, statuses) if s == 'read'), None)
            if first_read is not None:
                if nak_sent:
                    epoch_holds = any_holds = 0
                    nak_sent = False
                elif observed_level != attempt.level:
                    epoch_holds = any_holds = 0
                observed_level = attempt.level
            verdict = 'CERTAIN' if epoch_holds >= 6 else 'POSSIBLE' if any_holds >= 6 else 'NONE'
            model.window(attempt, statuses, verdict, False, single=True, envelope=envelope)
            for f, status in zip(attempt.frames, statuses):
                h.event('F' if status == 'read' else 'X', frame=f.model_frame(),
                        level=attempt.level, verdict=verdict, envelope=envelope)
            actions = h.rx.event('E', single=True)
            if any(isinstance(a, Ready) for a in actions):
                actions = h.rx.commit(tuple(f.prepare() for f, s in zip(attempt.frames, statuses) if s == 'read'))
                if any(isinstance(a, Commit) for a in actions):
                    model.committed()
            continue
        if e['event'] == 'Q':
            now += e['delay']
            inside = now-unkey < 3.672
            model.recovery(phase=e['phase'], inside=inside, silent=False,
                           phaseless=e['unphased'], poll=False)
            actions = h.rx.event('Q', phase=e['phase'], inside=inside, phaseless=e['unphased'])
        keys = [a for a in actions if isinstance(a, Key)]
        assert [k.kind for k in keys] == ([model.expected] if model.expected else []), (e, model.replay, h.rx.T)
        assert (keys[0].kind if keys else None) == e.get('kind', e.get('answer'))
        closes = [a.report for a in actions if isinstance(a, Close)]
        if closes:
            model.verify_close(closes[0])
        first = not model.saved.sent
        h.take(actions, air=e['air'])
        if keys:
            key = keys[0]
            model.sent(e['air'])
            if e['air']:
                now += 0.85
                unkey = now
            if first and e['air']:
                if key.kind == 'nak':
                    nak_sent = True
                elif key.kind == 'frame10':
                    epoch_holds = 0
                elif key.kind == 'continue':
                    epoch_holds = 0 if attempt.frames[-1].hint == 2 else epoch_holds+1
                    any_holds += 1
            if e['heard']:
                peer.hear(key, attempt)
        assert h.host == model.stream and peer.app.startswith(h.host)
    assert case['base_outcome'] == 'exact'
    assert not h.closes and h.host == peer.app
    assert peer.acked == len(peer.app)


def test_completion_channel_does_not_lose_a_replay_answer():
    channel = ModelChannel(random.Random(0), completion=True)
    for _ in range(100):
        assert channel.answer(replay=True) == (True, True)
    assert channel.counts['answer_loss'] == channel.counts['refusal'] == 0


@pytest.mark.parametrize('data,payload,start,reason', [
    (b'PQR', b'P', 0, None),
    (b'PQR', b'Q', 1, None),
    (b'PQR', b'QRZ', 1, None),
    (b'PPQ', b'P', None, 'nested candidates both agree'),
    (b'PQR', b'X', None, 'nested candidates neither agrees'),
])
@pytest.mark.parametrize('air', [False, True])
def test_nested_candidates_wait_for_air_then_require_one_match(data, payload, start, reason, air):
    h = Harness()
    h.window(data)
    h.arm()
    h.window(b'P')
    previous = h.rx.T
    h.query(phaseless=True, air=air)
    assert h.keys[-1].kind == 'nak' and h.rx.A.effect == (0, 1)
    if not air:
        assert h.rx.T == previous
        h.query(phase=1)
    assert h.rx.T == Replay(0, len(data), alternative=1)
    before = len(h.keys)
    actions = h.window(payload)
    if reason:
        assert len(h.keys) == before
        assert h.closes == [Report('R8', reason, start=0, frontier=len(data), rejected=payload)]
    else:
        commit = next(a for a in actions if isinstance(a, Commit))
        assert commit.start == start
        assert commit.new == (b'Z' if payload == b'QRZ' else b'')
        assert commit.progressed == (start+len(payload) > 1)
        assert not h.closes
    assert h.host == data+(b'Z' if payload == b'QRZ' else b'')


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('variant,invisible', [('reference', None), ('reference', 0),
                                              ('reference', 1), ('xara', None)])
@pytest.mark.parametrize('first_air', [False, True])
def test_nak_of_closing_double_resend(role, variant, invisible, first_air):
    """DESIGN §3 exception: whole peer retry exact or R2; Xara single retry exact."""
    peer = ModelPeer(variant, role, 'OFF', b'', level=7)
    cap = peer.capacity()
    peer.app = peer.wire = random.Random(0x430).randbytes(7*cap+20)
    h, model = Harness(), DecisionHistory()
    witness = StreamWitness(peer.wire)
    for _ in range(6):
        attempt = peer.key_data()
        assert len(attempt.frames) == 1
        model.window(attempt, ['read'], 'NONE', False, single=True)
        h.window(attempt.frames[0].replay)
        model.committed()
        model.sent(True)
        peer.hear(h.keys[-1], attempt)
    assert peer.stays == 6 and h.host == peer.app[:6*cap]
    attempt = peer.key_data()
    assert len(attempt.frames) == 2 and attempt.frames[-1].last
    model.window(attempt, ['read', 'X'], 'CERTAIN', False, single=True)
    h.event('F', frame=attempt.frames[0].model_frame(), verdict='CERTAIN', level=attempt.level)
    h.event('X')
    h.event('E', air=first_air)
    model.sent(first_air)
    assert h.keys[-1].kind == 'nak' and not h.rx.T
    assert h.host == peer.app[:6*cap]
    if not first_air:
        assert not model.retry_double
        model.recovery(phase=1, inside=False, silent=False, phaseless=False, poll=False)
        h.query(phase=1)
        model.sent(True)
    assert model.retry_double
    peer.hear(h.keys[-1], attempt)
    assert peer.stays == 0
    retry = peer.key_data()
    assert retry.start == attempt.start
    assert len(retry.frames) == (2 if variant == 'reference' else 1)
    statuses = ['read']*len(retry.frames)
    if invisible is not None:
        channel = ModelChannel(random.Random(0), residuals=True)
        assert channel.residuals
        statuses[invisible] = 'invisible'
    envelope = ModelChannel.envelope(retry, statuses)
    model.window(retry, statuses, 'NONE', False, single=True, envelope=envelope)
    for f, status in zip(retry.frames, statuses):
        if status == 'invisible':
            witness.omit('R2', retry.number, f.offset, f.offset+len(f.replay))
        else:
            h.event('F', frame=f.model_frame(), verdict='NONE', level=retry.level, envelope=envelope)
    assert h.rx.W.verdict == ('POSSIBLE' if variant == 'reference' else 'NONE')
    assert h.event('E', single=True) == [Ready()]
    out = h.commit(tuple(f.replay for f, status in zip(retry.frames, statuses) if status == 'read'))
    assert next(a.start for a in out if isinstance(a, Commit)) == 6*cap
    model.committed()
    model.sent(True)
    peer.hear(h.keys[-1], retry)
    assert h.rx.R == model.stream and not h.closes
    while peer.acked < len(peer.wire):
        retry = peer.key_data()
        assert len(retry.frames) == 1
        h.event('F', frame=retry.frames[0].model_frame(), verdict='NONE', level=retry.level, envelope='single')
        assert h.rx.W.verdict == 'NONE'
        h.event('E', single=True)
        h.commit((retry.frames[0].replay,))
        peer.hear(h.keys[-1], retry)
    assert h.host == witness.edited()
    if invisible is None:
        assert not witness.edits and h.host == peer.app
    else:
        f = attempt.frames[invisible]
        assert witness.edits == [Residual('R2', 7, f.offset, f.offset+len(f.replay))]
        assert h.host == peer.app[:f.offset]+peer.app[f.offset+len(f.replay):]
    assert not h.closes and peer.acked == len(peer.app)


@pytest.mark.parametrize('shape', ['missing-tail', 'hole-head', 'hole-tail', 'ceiling'])
@pytest.mark.parametrize('next_verdict', ['NONE', 'CERTAIN'])
def test_incomplete_double_retry_floor_and_window_lifetime(shape, next_verdict):
    h = Harness()
    h.event('X' if shape == 'hole-head' else 'F', frame=Frame(b'A'), verdict='CERTAIN')
    if shape in ('hole-head', 'hole-tail', 'ceiling'):
        h.event('X' if shape == 'hole-tail' else 'F', frame=Frame(b'B'))
    h.event('C' if shape == 'ceiling' else 'E')
    assert h.keys[-1].kind == 'nak'
    h.query(phaseless=True)
    h.event('F', frame=Frame(b'C'), verdict=next_verdict)
    assert h.rx.W.verdict == 'POSSIBLE'
    h.event('F', frame=Frame(b'D'))
    h.event('E')
    h.commit((b'C', b'D'))
    h.event('F', frame=Frame(b'E'))
    assert h.rx.W.verdict == 'NONE'


@pytest.mark.parametrize('override', ['compressed', 'turn', 'session'])
def test_incomplete_double_retry_floor_respects_reset_and_compression(override):
    h = Harness()
    h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
    h.event('E')
    if override != 'compressed':
        h.event('T', new_session=override == 'session')
    h.event('F', frame=Frame(b'B', compressed=override == 'compressed'))
    assert h.rx.W.verdict == 'NONE'


@pytest.mark.parametrize('level,next_level,envelope,verdict', [
    (7, 6, 'double', 'NONE'), (7, 7, 'single', 'POSSIBLE'),
    (1, 1, 'single', 'NONE'), (1, 1, 'double', 'POSSIBLE'),
    (1, 1, 'inconclusive', 'POSSIBLE'),
])
@pytest.mark.parametrize('replay', [False, True])
def test_retry_level_then_envelope_placement(level, next_level, envelope, verdict, replay):
    h = Harness()
    if replay:
        h.window(b'A', b'B', level=level)
        h.arm()
    h.event('F', frame=Frame(b'A'), verdict='CERTAIN', level=level)
    h.event('E')
    h.event('F', frame=Frame(b'A'), level=next_level, envelope=envelope)
    assert h.rx.W.verdict == verdict
    h.event('E', single=True)
    if envelope == 'double' and verdict == 'POSSIBLE':
        assert h.keys[-1].kind == 'nak' and not h.closes
        assert h.host == (b'AB' if replay else b'')
        return
    h.commit((b'A',))
    if replay and verdict == 'POSSIBLE':
        assert h.closes[-1].reason == 'POSSIBLE single inside replay'
    else:
        assert not h.closes and h.host == (b'AB' if replay else b'A')
        assert h.rx.K.possible_single == (verdict == 'POSSIBLE')


@pytest.mark.parametrize('invisible', [0, 1])
def test_floor_invisible_retry_is_single_extent_none_r2(invisible):
    peer = ModelPeer('reference', 'initiator', 'OFF', b'A'*18, level=1)
    peer.stays = 6
    attempt = peer.key_data()
    assert len(attempt.frames) == 2
    h, model = Harness(), DecisionHistory()
    model.window(attempt, ['read', 'X'], 'CERTAIN', False, single=True)
    h.event('F', frame=attempt.frames[0].model_frame(), level=1, verdict='CERTAIN')
    h.event('X')
    h.event('E')
    model.sent(True)
    peer.hear(h.keys[-1], attempt)
    retry = peer.key_data()
    statuses = ['read', 'read']
    statuses[invisible] = 'invisible'
    envelope = ModelChannel.envelope(retry, statuses)
    assert envelope == ('single' if invisible == 1 else 'inconclusive')
    model.window(retry, statuses, 'NONE', False, single=True, envelope=envelope)
    f = retry.frames[1-invisible]
    h.event('F', frame=f.model_frame(), level=1, envelope=envelope)
    assert h.rx.W.verdict == ('NONE' if invisible == 1 else 'POSSIBLE')
    h.event('E', single=True)
    h.commit((f.replay,))
    model.committed()
    witness = StreamWitness(peer.wire)
    missing = retry.frames[invisible]
    witness.omit('R2', retry.number, missing.offset, missing.offset+len(missing.replay))
    assert h.host == model.stream == witness.edited() and not h.closes


@pytest.mark.parametrize('envelope,want', [('single', 'NONE'), ('double', 'POSSIBLE'),
                                         ('inconclusive', 'POSSIBLE')])
def test_floor_retry_refines_envelope_when_burst_ends(envelope, want):
    h = Harness()
    h.window(b'A', b'B')
    h.arm()
    h.event('F', frame=Frame(b'A'), verdict='CERTAIN')
    h.event('E')
    h.event('F', frame=Frame(b'A'))
    assert h.rx.W.verdict == 'POSSIBLE'
    out = h.event('E', single=True, envelope=envelope)
    if envelope == 'double':
        assert h.keys[-1].kind == 'nak' and h.host == b'AB'
    else:
        assert out == [Ready()] and h.rx.W.verdict == want
        h.commit((b'A',))
        assert bool(h.closes) == (envelope == 'inconclusive')


@pytest.mark.parametrize('end', ['E', 'Q', 'C'])
@pytest.mark.parametrize('peer_keying', [False, True])
def test_ui_x_head_with_clear_tail_is_never_a_partial_commit(end, peer_keying):
    h = Harness(high_speed=False)
    h.event('X', level=4, verdict='CERTAIN', envelope='double', limited_pair=True)
    h.event('F', frame=Frame(b'clear tail'), level=4)
    h.event(end, peer_keying=peer_keying)
    assert not any(isinstance(a, Ready) for a in h.actions)
    assert h.rx.W is None and h.rx.K is None and h.rx.T is None
    assert h.rx.A.kind == 'nak' and h.rx.A.sent == (not peer_keying)
    assert h.host == h.rx.R == b''


@pytest.mark.parametrize('end', ['E', 'Q', 'C'])
def test_ui_x_tail_preserves_only_limited_pair_head(end):
    h = Harness(high_speed=False)
    h.event('F', frame=Frame(b'clear head'), level=4,
            verdict='CERTAIN', envelope='double', limited_pair=True)
    h.event('X', level=4)
    assert h.event(end) == [Ready(True)]
    h.commit((b'clear head',))
    assert h.host == b'clear head' and h.rx.K.frames == (Frame(b'clear head'),)
    assert h.keys[-1].kind == 'frame10'


def test_ui_x_complete_attempt_ceiling_stays_negative_after_refusal_and_query():
    h = Harness()
    h.event('X', level=2, verdict='CERTAIN', envelope='double')
    h.event('X', level=2)
    h.event('C', air=False)
    assert h.rx.A.kind == 'nak' and not h.rx.A.sent
    h.query()
    assert h.keys[-1].kind == 'nak' and h.rx.A.sent
    assert not h.host and h.rx.K is None and h.rx.T is None


@pytest.mark.parametrize('count', range(1, 19))
@pytest.mark.parametrize('level', [1, 2, 4])
@pytest.mark.parametrize('contest', [0, 1], ids=['head', 'tail'])
@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
def test_property_ui_interleaver_source_and_safe_loss(count, level, contest, mode):
    shape = ('intact', 'damaged', 'truncated')[(count + level + contest) % 3]
    timing = ('ui-first', 'data-first', 'lag')[(count + contest) % 3]
    walk = PropertyWalk(seed=0x5350414E + count * 10 + level, variant='reference',
        role='responder', mode=mode, content='zeros', bw='500', level=level,
        cohort='completion', partition='limited' if count % 2 else 'high_speed')
    walk.interleave_ui(count, level, contest, shape=shape, timing=timing,
                       identical=count % 2 == 0)
    assert 'P-DATA-SPAN' in walk.checked
    assert walk.channel.ui_counts['groups'] == 1
    assert walk.channel.ui_counts['packets'] == count
    assert walk.channel.ui_counts['holes'] == walk.channel.data_failures == 1


@pytest.mark.parametrize('reset', ['T', 'decoder', 'tune'])
@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
@pytest.mark.parametrize('contest', [0, 1])
def test_property_ui_interleaver_reset_drops_pending_sources(reset, mode, contest):
    walk = PropertyWalk(seed=0x1031, variant='xara', role='initiator', mode=mode,
                        content='head', bw='500', level=4)
    walk.interleave_ui(18, 4, contest, timing='lag', reset=reset)
    assert not walk.h.host and not walk.witness.sources


def test_source_witness_rejects_identical_ui_before_credit():
    witness = StreamWitness(b'A' * 12)
    with pytest.raises(AssertionError, match='P-DATA-SPAN/source'):
        witness.admitted(0, 0, ('ui', 1, 0))
    with pytest.raises(AssertionError, match='P-DATA-SPAN/credit'):
        witness.credited(0, (0,))
    witness.admitted(0, 0, 'peer')
    witness.credited(0, (0,))


def test_completion_channel_counts_ui_beyond_its_data_retry_bound():
    channel = ModelChannel(random.Random(0), completion=True)
    for _ in range(18):
        channel.ui_group(18, 1, b'A' * 9, shape='truncated', timing='data-first', identical=True)
        channel.ui_loss()
    assert channel.data_failures == channel.ui_counts['holes'] == 18


@pytest.mark.parametrize('count', range(1, 19))
@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
def test_property_ui_channel_walk_retries_and_recovers(count, mode):
    level = (1, 2, 4)[(count - 1) % 3]
    contest = count % 2 if mode == 'OFF' else 0
    at = 6 if mode == 'OFF' else 0
    shape = ('intact', 'damaged', 'truncated')[(count // 3) % 3]
    timing = ('ui-first', 'data-first', 'lag')[count % 3]
    walk = PropertyWalk(seed=0x5350414E + count, variant='reference', role='responder',
        mode=mode, content='zeros', bw='500', level=level, cohort='completion',
        partition='limited' if count % 2 else 'high_speed')
    walk.channel.warmup = 7 if mode == 'OFF' else 0
    walk.channel.schedule_ui(count, level, contest, at=at, shape=shape,
                             timing=timing, identical=count % 2 == 0)
    witness = walk.witness
    walk.run()
    assert walk.witness is witness and 'P-DATA-SPAN' in walk.checked
    assert walk.channel.ui_counts == dict(groups=1, packets=count, holes=1)
    observation, source = walk.observations_by_attempt[at][contest]
    assert source[0] == 'ui' and (at, contest) not in witness.sources
    if count % 2 == 0 and shape == 'intact':
        assert observation.body == walk.peer.history[at].frames[contest].wire
    offset = walk.peer.history[at].frames[contest].offset
    assert any(a.number > at and any(f.offset == offset for f in a.frames)
               for a in walk.peer.history)
    assert any(number > at for number, _ in witness.sources)
    assert all(source == 'peer' for source in witness.sources.values())
    assert walk.channel.data_failures == 0
    assert walk.h.host == walk.peer.app and not walk.h.closes and not witness.edits


@pytest.mark.parametrize('mode', ['OFF', 'TEXT', 'FILES'])
def test_property_ui_channel_walk_overlapping_groups_answer_loss_and_replay(monkeypatch, mode):
    walk = PropertyWalk(seed=0x5350414E, variant='xara', role='initiator', mode=mode,
        content='blocks', bw='500', level=4, cohort='completion', progressing=True)
    walk.channel.schedule_ui(18, 4, 0, shape='truncated', timing='lag')
    walk.channel.schedule_ui(2, 4, 0, shape='intact', timing='ui-first')
    answer = walk.channel.answer
    lost = []
    def lose_first_answer(**kwargs):
        if not lost:
            lost.append(walk.peer.number - 1)
            walk.channel.counts['answer_loss'] += 1
            walk.channel.answer_failures += 1
            return True, False
        return answer(**kwargs)
    monkeypatch.setattr(walk.channel, 'answer', lose_first_answer)
    witness = walk.witness
    walk.run()
    assert lost == [0] and walk.channel.counts['answer_loss'] >= 1
    assert walk.channel.ui_counts == dict(groups=2, packets=20, holes=1)
    assert len(walk.ui_station.receive_context.ledger.spans) == 2
    assert any(row[0] in ('R', 'Q') for row in walk.trace)
    assert walk.forced_replay and walk.replay_commits >= 2
    assert walk.witness is witness and all(s == 'peer' for s in witness.sources.values())
    assert 'P-DATA-SPAN' in walk.checked
    assert walk.h.host == walk.peer.app and not walk.h.closes and not witness.edits


@pytest.mark.parametrize('reset', ['decoder', 'tune'])
@pytest.mark.parametrize('mode', ['OFF', 'TEXT'])
def test_property_ui_channel_walk_recovers_after_held_decoder_reset(reset, mode):
    walk = PropertyWalk(seed=0x1031, variant='reference', role='responder', mode=mode,
        content='head', bw='500', level=2, cohort='completion')
    walk.channel.schedule_ui(18, 2, 0, timing='lag', reset=reset)
    witness = walk.witness
    walk.run()
    assert ('UI-reset', 0, reset) in walk.trace
    assert (0, 0) not in witness.sources and any(n > 0 for n, _ in witness.sources)
    assert walk.witness is witness and all(s == 'peer' for s in witness.sources.values())
    assert walk.h.host == walk.peer.app and not walk.h.closes and not witness.edits


@pytest.mark.parametrize('mode', ['OFF', 'TEXT', 'FILES'])
def test_property_ui_channel_walk_exhausted_loss_bound_closes_safely(mode):
    walk = PropertyWalk(seed=0x1031, variant='reference', role='responder', mode=mode,
        content='head', bw='500', level=1, partition='limited')
    walk.channel.warmup = 0
    walk.channel.ui_loss_limit = 2
    walk.channel.schedule_ui(18, 1, 0, attempts=18, shape='truncated', timing='lag')
    witness = walk.witness
    walk.run()
    assert walk.step == 3 and walk.channel.data_failures == 3
    assert walk.channel.ui_counts == dict(groups=3, packets=54, holes=3)
    assert walk.channel.ui_exhausted and walk.h.rx.closed
    assert walk.h.closes[-1].code == 'budget'
    assert walk.h.rx.W is None and walk.h.rx.K is None and walk.h.rx.T is None
    assert walk.witness is witness and not witness.sources and not witness.edits
    assert not walk.h.host and walk.peer.acked == 0
    assert all(k.kind == 'nak' for k in walk.h.keys)
