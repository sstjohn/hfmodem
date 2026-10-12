# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from contextlib import redirect_stdout
from os import devnull
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hfmodem.shrike import p4host, p4long, p4oracle, p4robustrx, p4rx, p4short, session
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.p4_chirp_deadline_support import FS, replay


AT = 1428445


def receiver():
    rx = P4Receiver(FS, admitted=True, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, 2.62)
    return rx


@pytest.mark.parametrize('coherence', [.2, .7], ids=['weak-carrier', 'crc-miss'])
@pytest.mark.parametrize('partial', [False, True])
def test_failed_robust_long_cannot_be_reused_as_a_normal_placement(monkeypatch, coherence, partial):
    rx = receiver()
    decode = Mock(return_value=None)
    monkeypatch.setattr(p4oracle, 'placed_despread',
                        lambda *a, **kw: (coherence, np.zeros(373), 0))
    monkeypatch.setattr(p4robustrx, 'decode_symbols', decode)
    assert rx._long(np.zeros(round(3.3*FS)), AT) == (None, coherence)
    assert decode.call_count == int(coherence >= p4oracle.ROBUST_FRACTION_KNEE)
    entry = rx._robust_long_cache[(AT, rx.clock.cfo_hz)]
    place = Mock(return_value=(.1, p4short.GEOM, rx.clock.cfo_hz))
    monkeypatch.setattr(p4short, 'place_long', place)
    assert rx._long(np.zeros(round(3.355*FS)), AT, placed=True,
                    partial=partial) == (None, .1)
    assert rx._robust_long_cache[(AT, rx.clock.cfo_hz)] is entry
    assert not rx._long_cache
    assert not rx.observations and not rx.registrations
    place.assert_called_once()


def test_failed_robust_long_still_retries_when_more_audio_arrives(monkeypatch):
    rx = receiver()
    despread = Mock(return_value=(.2, np.zeros(373), 0))
    monkeypatch.setattr(p4oracle, 'placed_despread', despread)
    assert rx._long(np.zeros(FS), AT) == (None, .2)
    assert rx._long(np.zeros(2*FS), AT) == (None, .2)
    assert despread.call_count == 2
    assert rx._long(np.zeros(2*FS), AT) == (None, .2)
    assert despread.call_count == 2


def test_rejected_full_normal_long_stays_unplaced_on_later_reads(monkeypatch):
    rx = receiver()
    monkeypatch.setattr(p4rx, 'audio_to_baseband', lambda *a: np.zeros(100000))
    nominal = p4short.GEOM
    place = Mock(side_effect=[(.7, nominal, 2.62), (.2, nominal, 2.62)])
    monkeypatch.setattr(p4short, 'place_long', place)
    pending, _ = rx._long(np.zeros(FS), AT, placed=True, partial=True)
    assert pending is not None and pending.provisional
    assert rx._long(np.zeros(round(3.355*FS)), AT, placed=True) == (None, .2)
    assert rx._long_cache[(AT, 2.62)]['obs'] is None
    for partial in (True, False):
        obs, _ = rx._long(np.zeros(4*FS), AT, placed=True, partial=partial)
        assert obs is None
    assert place.call_count == 2


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('missing_key', [False, True])
def test_missing_normal_long_observation_is_unplaced(monkeypatch, complete, partial, missing_key):
    rx = receiver()
    entry = dict(kind='normal', complete=complete, end=AT+FS, coherence=.2)
    if not missing_key:
        entry['obs'] = None
    rx._long_cache[(AT, 2.62)] = entry
    monkeypatch.setattr(p4rx, 'audio_to_baseband',
                        Mock(side_effect=AssertionError('missing observation reached acquisition')))
    assert rx._long(np.zeros(4*FS), AT, placed=True, partial=partial) == (None, .2)


@pytest.mark.parametrize('reason', ['weak-carrier', 'incomplete-scores'])
def test_unplaced_normal_short_is_safe_to_read_and_refine(monkeypatch, reason):
    rx = receiver()
    rx.clock.cycle_s = 1.25
    monkeypatch.setattr(p4rx, 'audio_to_baseband', lambda *a: np.zeros(100))
    coherence = .2 if reason == 'weak-carrier' else .8
    monkeypatch.setattr(p4short, 'place_short', lambda *a, **kw: (coherence, 0, 2.62))
    monkeypatch.setattr(p4short, 'short_scores', lambda *a, **kw: None)
    assert rx._normal(np.zeros(FS), AT) is None
    assert rx._normal_cache[AT]['obs'] is None
    assert rx._normal(np.zeros(2*FS), AT) is None
    assert rx.refine() == []


@pytest.mark.parametrize('missing_key', [False, True])
@pytest.mark.parametrize('kind', ['short', 'long'])
def test_refinement_skips_a_missing_observation(monkeypatch, kind, missing_key):
    rx = receiver()
    decode = Mock(side_effect=AssertionError('missing observation reached decode'))
    entry = dict(kind='normal', scores=np.ones(4), decoder=SimpleNamespace(done=False, decode=decode))
    if not missing_key:
        entry['obs'] = None
    if kind == 'short':
        rx._normal_cache[AT] = entry
        rx.hold.kind, rx.hold.last = 'normal', AT
        monkeypatch.setattr(rx.hold, 'refine', lambda **kw: object())
    else:
        rx._long_cache[(AT, 2.62)] = entry
        monkeypatch.setattr(p4long, 'decode_scores', decode)
    assert rx.refine() == []
    assert not rx.observations and not rx.registrations
    decode.assert_not_called()


def test_1112_replay_deduplicates_c2_and_requests_queued_turn(monkeypatch):
    arm = 'onair-1003-1112'
    require_input(CAPTURES/arm/'stream.flac')
    probe = json.loads((CAPTURES/arm/'p4-probe.json').read_text())
    decisions = [e for e in probe['link_events'] if e['event'] == 'Decision']

    def cost(method):
        def work(at, args):
            row = min(decisions, key=lambda e: abs(e['at']-at))
            return max(0., row.get('work_ms', {}).get(method, 4)/1000-.003)
        return work

    accepted = []
    accept = p4host.P4Host.accept

    def tracked(self, frame, **kwargs):
        text = accept(self, frame, **kwargs)
        if frame.status.byte == 2:
            accepted.append((bool(text), self.result.duplicate))
        return text

    monkeypatch.setattr(p4host.P4Host, 'accept', tracked)
    with open(devnull, 'w') as sink, redirect_stdout(sink):
        result = replay(arm, stop=round(37.8*FS),
                        work_costs={m: cost(m) for m in ('read_partial', 'read_window')})

    assert max(e.get('cycle_end_sample', 0) for e in result['events']) > round(36.8*FS)
    c2 = [e for e in result['heard'] if e.get('status') == 2 and not e['provisional']]
    assert len(c2) >= 2
    assert [e['onset_sample'] for e in c2] == [1068446, 1248445]
    assert len({e['data'] for e in c2}) == 1
    assert accepted[0] == (True, False)
    assert all(row == (False, True) for row in accepted[1:])
    # The approved link-layer repeat rule now requests at c2's answer slot.
    # It avoids the later live crash boundary; that boundary is pinned below.
    keys = [e for e in result['events'] if e['event'] == 'Keyed']
    first_answer, = [e for e in keys if e['at'] == 1231646]
    assert first_answer['word'] == 'X27' and first_answer['state'] == 'IRS_ANSWER'
    first_decision, = [e for e in result['events']
                       if e['event'] == 'Decision' and e['at'] == 1068446]
    assert first_decision['now_sample'] <= first_decision['decision_at']
    assert first_decision['proposed_at'] == 1231646
    requests = [e for e in keys if e.get('state') == 'IRS_BREAKIN']
    assert requests and abs(requests[0]['at']-1411645) < 10
    assert all(e.get('kind') == 'body' and not e.get('payload') for e in requests)
    assert all(not e.get('payload') for e in keys)
    assert result['buffered'] == len(bytes.fromhex(result['source']))
    assert not [e for e in result['events'] if e['event'] == 'Refused']


def test_recorded_1112_crash_boundary_still_survives_without_a_turn_request():
    # The actual failed robust read and extended normal retry from the triage.
    # This remains necessary even when the queued-turn rule avoids that path.
    path = CAPTURES/'onair-1003-1112/stream.flac'
    require_input(path)
    audio = session.load_wav(str(path), FS)
    rx = receiver()
    rx.clock = Clock(Expect.PACKET, 1428444, .330, 3.75, 2.6225948105757824)
    origin = 1426044
    assert rx._long(audio[origin:1587852], origin)[0] is None
    assert rx._long(audio[origin:1809536], origin, placed=True)[0] is None
    assert not rx.observations and not rx.registrations


@pytest.mark.parametrize('normal_first', [False, True])
def test_recorded_normal_long_survives_a_robust_miss(normal_first):
    # 1049 c2 independently passes normal CRC; the robust projection is a miss.
    path = CAPTURES/'onair-1002-1049/stream.flac'
    require_input(path)
    audio = session.load_wav(str(path), FS)
    at, origin = 2148660, 2146260
    window = audio[origin:at+round(3.355*FS)]
    rx = receiver()
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, -6.94)
    if normal_first:
        first, _ = rx._long(window, origin, placed=True)
        assert first is not None and first.frame is not None
    robust, _ = rx._long(window, origin)
    assert robust is None
    normal, _ = rx._long(window, origin, placed=True)
    assert normal is not None and normal.frame is not None
    assert normal.frame.status.byte == 2
    isolated = receiver()
    isolated.clock = Clock(Expect.PACKET, at, .330, 3.75, -6.94)
    expected, _ = isolated._long(window, origin, placed=True)
    assert normal.frame == expected.frame
    extended, _ = rx._long(audio[origin:at+round(3.6*FS)], origin, placed=True)
    assert extended.frame == normal.frame
    assert rx._long(window, origin)[0] is None


def test_recorded_robust_long_survives_extended_normal_attempt():
    path = CAPTURES/'onair-1003-1112/stream.flac'
    require_input(path)
    audio = session.load_wav(str(path), FS)
    at, origin = 708446, 706046
    rx = receiver()
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, 2.6225948105757824)
    window = audio[origin:at+round(3.355*FS)]
    robust, _ = rx._long(window, origin)
    assert robust is not None and robust.frame.status.byte == 0x61
    normal, _ = rx._long(audio[origin:at+round(3.6*FS)], origin, placed=True)
    assert normal is None or normal.frame is None
    repeated, _ = rx._long(window, origin)
    assert repeated is robust
