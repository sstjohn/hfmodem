# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from contextlib import redirect_stdout
from itertools import count
from os import devnull
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hfmodem.shrike import p4frame, p4host, p4long, p4longdecode, p4longrates, p4longtx, p4oracle, p4robustrx, p4rx, p4short, p4turn, session
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
    # Receiver integration constructs a retained shared-engine job; it no
    # longer calls the standalone compatibility decoder. A CRC miss must keep
    # that robust job separate from normal placement just as before.
    g = p4longdecode.geometry(3)
    members = Mock(return_value=(p4longdecode.LongMember(g, np.ones(g.serialized_size)),))
    advance = Mock(return_value=None)
    monkeypatch.setattr(p4oracle, 'placed_despread',
                        lambda *a, **kw: (coherence, np.zeros(373), 0))
    monkeypatch.setattr(p4robustrx, 'build_members', members)
    monkeypatch.setattr(rx, '_advance_long', advance)
    assert rx._long(np.zeros(round(3.3*FS)), AT) == (None, coherence)
    # Coherence ranks work; it no longer excludes parity recovery.
    members.assert_called_once()
    advance.assert_called_once()
    entry = rx._robust_long_cache[(AT, rx.clock.cfo_hz)]
    place = Mock(return_value=(.1, p4short.GEOM, rx.clock.cfo_hz))
    monkeypatch.setattr(p4short, 'place_long', place)
    assert rx._long(np.zeros(round(3.355*FS)), AT, placed=True,
                    partial=partial) == (None, .1)
    assert rx._robust_long_cache[(AT, rx.clock.cfo_hz)] is entry
    assert not rx._long_cache
    job, = rx._long_jobs.values()
    assert job['obs'].frame is None and job['search'].members == members.return_value
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


@pytest.mark.parametrize('valid', [False, True])
def test_weak_full_normal_long_observation_requires_crc(monkeypatch, valid):
    rx = receiver()
    monkeypatch.setattr(p4rx, 'audio_to_baseband', lambda *a: np.zeros(100000))
    place = Mock(side_effect=[(.7, p4short.GEOM, 2.62), (.2, p4short.GEOM, 2.62)])
    monkeypatch.setattr(p4short, 'place_long', place)
    observe = Mock(return_value=object())
    monkeypatch.setattr(p4longrates, 'observe_body', observe)
    wire = p4frame.encode_field(bytes(range(203)), p4frame.Status.from_byte(2))
    fields = 8*(1-2.0*p4longtx.coded_bits(wire)) if valid else np.ones(4968)
    monkeypatch.setattr(p4longrates, 'build_members', lambda observed:
                        (p4longdecode.LongMember(p4longdecode.geometry(5), fields),))
    pending, _ = rx._long(np.zeros(FS), AT, placed=True, partial=True)
    # Prefix readiness alone cannot claim complete-body carrier liveness.
    assert pending.provisional and not pending.placed and pending.frame is None
    full, coherence = rx._long(np.zeros(round(3.355*FS)), AT, placed=True)
    assert full is not None and coherence == .2
    assert full.placed == valid and full.provisional != valid
    assert (full.frame is not None) == valid
    if valid:
        assert full.frame == p4frame.decode_field(wire, data_bytes=203)
    assert rx._long(np.zeros(4*FS), AT, placed=True)[0] is full
    observe.assert_called_once()
    assert place.call_count == 2
    assert not rx.observations and not rx.registrations


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


def test_1112_replay_deduplicates_c2_and_acks_each_repeat_again(monkeypatch):
    # A different reader may already have allocated many ownership nonces.
    monkeypatch.setattr(p4turn, '_peer_watch_generations', count(1000))
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
    turns, expired = [], []
    observe = p4turn.P4Turn.observe

    def observed(self, reg, **kwargs):
        if not turns:
            turns.append(self)
        assert self is turns[0]
        return observe(self, reg, **kwargs)

    monkeypatch.setattr(p4turn.P4Turn, 'observe', observed)
    work_expired = P4Receiver._long_work_expired

    def checked_expiry(self, onset, **kwargs):
        result = work_expired(self, onset, **kwargs)
        if abs(onset-1608445) < FS//20:
            limits = self.long_deadline(onset, sf8=True)
            expired.append(dict(onset=onset, expired=result,
                now=limits.work.sample_now(), publishable=limits.publishable(),
                watch=self.peer_watch,
                jobs=[key for key in self._long_jobs if abs(key[0]-onset) < FS//20]))
        return result

    monkeypatch.setattr(P4Receiver, '_long_work_expired', checked_expiry)
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
    # The causal 369-symbol fit moves the second onset by one sample. Each
    # repeat is acked again (WS8EOC 04:48Z), never broken into, so the watch
    # reads every copy.
    assert [e['onset_sample'] for e in c2] == [1068446, 1248444, 1428449, 1608444]
    assert len({e['data'] for e in c2}) == 1
    assert accepted == [(True, False)] + [(False, True)]*3
    keys = [e for e in result['events'] if e['event'] == 'Keyed']
    first_answer, = [e for e in keys if e['at'] == 1231646]
    assert first_answer['word'] == 'V27' and first_answer['state'] == 'IRS_ANSWER'
    first_decision, = [e for e in result['events']
                       if e['event'] == 'Decision' and e['at'] == 1068446]
    assert first_decision['now_sample'] <= first_decision['decision_at']
    assert first_decision['proposed_at'] == 1231646
    assert not [e for e in keys if e.get('state') == 'IRS_BREAKIN']
    acks = [e for e in keys if e['at'] > 1231646]
    assert [e['at'] for e in acks] == [1411644, 1591649, 1771644]
    assert all(e.get('kind') == 'word' and e.get('word') == 'V27'
               and e.get('state') == 'IRS_ANSWER' for e in acks)
    for emission, protected in zip(
            [e for e in result['emissions'] if e.get('word') == 'V27' and e['audio_start'] > 1231646],
            [1428444, 1608444, 1788444]):
        assert emission['audio_end'] < protected
    assert expired
    watch_cycle, = [e for e in result['events']
                    if e['event'] == 'PeerWatchCycle' and e['at'] == 1608444]
    assert watch_cycle['support_end'] == 1767213
    # The watched prefix is prepared while input arrives. Those
    # checks precede the parent's V27 and may not construct/publish a body job.
    # Complete-body work still waits behind that actual key and stays expired.
    prefix = [snapshot for snapshot in expired if snapshot['now'] < 1771644]
    complete = [snapshot for snapshot in expired if snapshot['now'] >= 1771644]
    assert prefix and complete
    assert any(snapshot['now'] == 1608449+round(.810*FS) for snapshot in prefix)
    assert any(snapshot['now'] == watch_cycle['support_end'] for snapshot in prefix)
    for snapshot in prefix:
        assert snapshot['onset'] == 1608449
        assert 1608449+round(.810*FS) <= snapshot['now'] <= watch_cycle['support_end']
        assert not snapshot['expired'] and snapshot['publishable'] and not snapshot['jobs']
        assert snapshot['watch'] is not None
        assert snapshot['watch'].generation == watch_cycle['generation']
    for snapshot in complete:
        assert snapshot['onset'] == 1608444 and snapshot['expired']
        assert snapshot['now'] > 1771644 and not snapshot['publishable']
        assert not snapshot['jobs']
        assert snapshot['watch'] is not None
        assert snapshot['watch'].generation == watch_cycle['generation']
    turn, = turns
    assert not turn.outbox.inflight and turn.outbox.arq.tx_acked_bytes == 0
    assert turn._deferred is None
    assert turn._held_answer is None or turn._held_answer.onset == 1248444
    assert not [e for e in keys if e['at'] >= 1951644]
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
