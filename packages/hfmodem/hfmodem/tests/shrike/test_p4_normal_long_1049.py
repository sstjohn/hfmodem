# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Normal-long receive on the recorded VE3KPG 3.75 s raster."""
import hashlib
import json
from contextlib import redirect_stdout
from os import devnull

import numpy as np
import pytest

from hfmodem.shrike import p4frame, p4long, p4rx, p4short, p4turn, session
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.p4_chirp_deadline_support import FS, prerender, replay
from pathlib import Path
from hfmodem.tests.shrike.p4_nak_qam_support import (
    neutral_keys, waveform_keys, without_neutral_requests,
)


FIRST = 2148660
SECOND = 2328660
PREFIX = b'(FN04VE)\r[WL2K-5.0-'


def recorded(arm, stop, **kwargs):
    with open(devnull, 'w') as sink, redirect_stdout(sink):
        return replay(arm, stop=stop, **kwargs)


def assert_long_exchange(result):
    heard = result['heard']
    c1 = [row for row in heard if row['kind'] == Kind.NORMAL
          and abs(row['onset_sample']-2088660) <= 1680 and not row['provisional']]
    assert c1 and c1[-1].get('status') == 1
    for at in (FIRST, SECOND):
        copies = [row for row in heard if row['kind'] == Kind.LONG
                  and abs(row['onset_sample']-at) <= 1680 and not row['provisional']]
        assert copies, f'no framed long near {at}'
        row = copies[-1]
        assert row.get('status') == 2
        assert bytes.fromhex(row['data']).startswith(PREFIX)
        assert abs(row['cfo_hz']+7.1) <= .5
    keyed = [row['at'] for row in result['events'] if row['event'] == 'Keyed'
             and row.get('kind') == Kind.WORD and row.get('word') == 'X27'
             and row.get('state') == 'IRS_ANSWER'
             and row['at'] >= FIRST]
    assert any(abs(at-expected) <= 1680 for at in keyed
               for expected in (2311860, 2491860)), keyed
    assert not [e for e in result['events'] if e['event'] == 'Refused']
    assert not [e for e in result['events'] if e['event'] == 'Decision'
                and e['at'] >= round(45.5*FS)
                and 'no live packet' in e.get('reason', '')]
    assert all(count <= 1 for at, count in result['cycles'] if at >= round(45.574*FS))
    return keyed[0]


def test_1049_long_replay_acknowledges_the_first_or_next_copy():
    keyed = assert_long_exchange(recorded('onair-1002-1049', 52*FS,
                                          work_costs={
                                              'read_partial': lambda at, _: .015 if at >= FIRST else .003,
                                              'read_window': lambda at, args: (
                                                  .045 if at >= FIRST and args[1]+len(args[0]) >= at+3*FS
                                                  else .008 if at >= FIRST else .004),
                                          }))
    assert abs(keyed-2311860) <= 1680 or abs(keyed-2491860) <= 1680


def test_1049_pin_fails_without_the_lattice_candidate(monkeypatch):
    original = P4Receiver._long

    def despread_only(self, audio, origin, **kwargs):
        if kwargs.get('placed'):
            return None, 0.
        return original(self, audio, origin, **kwargs)

    with monkeypatch.context() as mutation:
        mutation.setattr(P4Receiver, '_long', despread_only)
        result = recorded('onair-1002-1049', 52*FS)
        with pytest.raises(AssertionError, match='no framed long'):
            assert_long_exchange(result)
        assert not [row for row in result['heard'] if row['kind'] == Kind.LONG
                    and row['onset_sample'] >= FIRST]
        assert any(count >= 2 for at, count in result['cycles'] if at >= round(45.574*FS))


@pytest.mark.realtime
def test_1049_sample_clock_receive_cost():
    result = recorded('onair-1002-1049', 52*FS, measured_work=True)
    decisions = [e for e in result['events'] if e['event'] == 'Decision'
                 and e['at'] >= round(45.5*FS)]
    assert decisions
    for row in decisions:
        work = row['work_ms']
        assert work.get('read_window', 0)+work.get('read_partial', 0) <= 100
        assert (row['decision_at']-row['now_sample'])*1000/FS >= -40
    for at in (FIRST, SECOND):
        quick = [e for e in result['events'] if e['event'] == 'long-decode'
                 and abs(e['at']-at) <= 1680 and e['iterations'] == 1
                 and e['copies'] == 1 and e['crc_valid']]
        assert quick and quick[0]['elapsed_ms'] <= 40


def receiver(at=FIRST):
    rx = P4Receiver(FS, admitted=True, cfo_hz=-6.94, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, -6.94)
    return rx


def audio():
    return session.load_wav(str(require_input(CAPTURES/'onair-1002-1049'/'stream.flac')), FS)


def test_recorded_long_quick_decode_and_held_repeat():
    tape = audio()
    rx = receiver()
    start = FIRST-round(.050*FS)
    partial, _ = rx._long(tape[start:FIRST+round(.810*FS)], start,
                          placed=True, partial=True)
    assert partial is not None and partial.provisional
    assert rx.clock.cycle_s == 3.75 and rx.clock.misses == 0
    full, _ = rx._long(tape[start:FIRST+round(3.355*FS)], start, placed=True)
    assert full is not None and full.frame is not None and full.frame.status.byte == 2
    assert full.frame.data.startswith(PREFIX)
    assert any(e['event'] == 'long-decode' and e['iterations'] == 1
               and e['polarity'] == 1 and e['crc_valid'] for e in rx.live_events)
    assert any(e['event'] == 'long-header' for e in rx.live_events)
    rx.clock.next_due = SECOND
    start = SECOND-round(.050*FS)
    repeat, _ = rx._long(tape[start:SECOND+round(.810*FS)], start,
                         placed=True, partial=True)
    assert repeat is not None and not repeat.provisional and repeat.frame == full.frame
    assert any(e['event'] == 'long-repeat' and e['agreement'] >= .70
               for e in rx.live_events)


def test_pending_long_stops_silence_and_late_frame_reanchors_clock():
    rx = receiver()
    pending = Observation(Kind.LONG, FS, 0, FIRST, FIRST, .810, -7.1, .7,
                          provisional=True, placed=True)
    rx.register(pending)
    rx.silent_cycles = 2
    assert rx.finish_cycle(FIRST) == 0
    frame = p4frame.decode_field(p4frame.encode_field(
        b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    rx.clock.next_due = FIRST+round(2*1.25*FS)
    confirmed = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                            frame=frame, placed=True)
    rx._emit(confirmed, [])
    assert rx.clock.cycle_s == 3.75
    assert rx.clock.next_due == FIRST+round(3.75*FS)


def test_expired_crc_miss_retains_placed_long_carrier_liveness():
    rx = receiver()
    rx.peer_onset = FIRST-round(1.25*FS)
    rx.clock.cycle_s = 3.75
    for cycle in range(5):
        at = FIRST+cycle*round(3.75*FS)
        if cycle:
            rx.new_cycle(at)
        pending = Observation(Kind.LONG, FS, 0, at, at, .810, -7.1, .75,
                              provisional=True, placed=True)
        rx.register(pending)
        rx.observations.remove(pending)
        rx._live(RegisteredObservation(pending, Registered.EXPIRED, pending))
        assert rx.finish_cycle(at) == 0
    assert len(rx.live_onsets) == 5


def test_second_and_third_1049_long_crc_misses_keep_carrier_live(monkeypatch):
    tape = audio()
    rx = receiver()
    rx.peer_onset = FIRST-round(1.25*FS)  # last short before the long raster
    first_start = FIRST-round(.050*FS)
    first, _ = rx._long(tape[first_start:FIRST+round(.810*FS)], first_start,
                        placed=True, partial=True)
    assert first is not None and first.provisional
    rx.register(first)
    assert rx.clock.cycle_s == 3.75
    assert rx.finish_cycle(FIRST) == 0

    monkeypatch.setattr(p4long, 'decode_scores', lambda *args, **kwargs: None)
    for at in (SECOND, SECOND+round(3.75*FS)):
        rx.new_cycle(at)
        start = at-round(.050*FS)
        pending, _ = rx._long(tape[start:at+round(.810*FS)], start,
                              placed=True, partial=True)
        assert pending is not None and pending.provisional and pending.placed
        rx.register(pending)
        missed, _ = rx._long(tape[start:at+round(3.355*FS)], start, placed=True)
        assert missed is not None and missed.frame is None
        assert any(e['event'] == 'long-decode' and abs(e['at']-at) <= 1680
                   and not e['crc_valid'] for e in rx.live_events)
        rx.observations.remove(pending)
        rx._live(RegisteredObservation(pending, Registered.EXPIRED, pending))
        assert rx.finish_cycle(at) == 0
    assert rx.peer_onset == FIRST-round(1.25*FS)


def test_new_long_frame_can_contradict_an_early_repeat():
    rx = receiver()
    first = p4frame.decode_field(p4frame.encode_field(
        b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    changed = p4frame.decode_field(p4frame.encode_field(
        b'y'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    early = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                        frame=first, placed=True)
    late = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                       frame=changed, placed=True)
    rx.register(early)
    result = rx.register(late)
    assert result.outcome == Registered.CONTRADICTED
    assert rx.observations[-1].frame == changed
    assert any(e['event'] == 'long-repeat-contradicted' for e in rx.live_events)


def test_failed_quick_decode_gets_eight_iteration_refine(monkeypatch):
    from hfmodem.shrike import p4long
    tape = audio()
    rx = receiver()
    original = p4long.decode_scores

    def slow(scores, *, iterations=8, polarity=1):
        return None if iterations == 1 else original(scores, iterations=iterations,
                                                      polarity=polarity)

    monkeypatch.setattr(p4long, 'decode_scores', slow)
    start = FIRST-round(.050*FS)
    rx._long(tape[start:FIRST+round(.810*FS)], start, placed=True, partial=True)
    pending, _ = rx._long(tape[start:FIRST+round(3.355*FS)], start, placed=True)
    assert pending is not None and pending.frame is None
    records = rx.refine()
    assert records and records[0].observation.frame is not None
    assert records[0].observation.frame.status.byte == 2
    assert any(e['event'] == 'long-decode' and e['iterations'] == 8
               and e['crc_valid'] for e in rx.live_events)


def test_failed_single_copies_get_one_iteration_sum(monkeypatch):
    from hfmodem.shrike import p4long
    tape = audio()
    rx = receiver()
    original = p4long.decode_scores
    calls = 0

    def first_two_fail(scores, *, iterations=8, polarity=1):
        nonlocal calls
        calls += 1
        return None if calls <= 2 else original(scores, iterations=iterations,
                                                polarity=polarity)

    monkeypatch.setattr(p4long, 'decode_scores', first_two_fail)
    for at in (FIRST, SECOND):
        rx.clock.next_due = at
        start = at-round(.050*FS)
        rx._long(tape[start:at+round(.810*FS)], start, placed=True, partial=True)
        obs, _ = rx._long(tape[start:at+round(3.355*FS)], start, placed=True)
    assert obs is not None and obs.frame is not None
    assert calls == 3
    assert any(e['event'] == 'long-decode' and e['copies'] == 2
               and e['iterations'] == 1 and e['crc_valid'] for e in rx.live_events)


@pytest.mark.parametrize('winner', ('single', 'sum'))
def test_failed_sum_keeps_single_copy_for_eight_iteration_refine(monkeypatch, winner):
    tape = audio()
    rx = receiver()
    wanted = p4frame.decode_field(p4frame.encode_field(
        b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)
    original_scores = p4long.body_scores
    singles = []

    def capture_scores(*args, **kwargs):
        scores = original_scores(*args, **kwargs)
        singles.append(scores.copy())
        return scores

    monkeypatch.setattr(p4long, 'body_scores', capture_scores)
    monkeypatch.setattr(p4long, 'decode_scores', lambda *a, **kw: None)
    for at in (FIRST, SECOND):
        rx.clock.next_due = at
        start = at-round(.050*FS)
        rx._long(tape[start:at+round(.810*FS)], start, placed=True, partial=True)
        rx._long(tape[start:at+round(3.355*FS)], start, placed=True)
    entries = sorted((e for e in rx._long_cache.values() if e.get('kind') == 'normal'),
                     key=lambda e: e['obs'].stream_sample)
    entries[0]['refined'] = True
    single = singles[-1]
    assert np.array_equal(entries[1]['scores'], single)
    combined = singles[0]+single
    calls = []

    def eight(scores, *, iterations=8, polarity=1):
        source = 'single' if np.array_equal(scores, single) else (
            'sum' if np.array_equal(scores, combined) else 'other')
        calls.append((iterations, polarity, source))
        return wanted if iterations == 8 and polarity == 1 and source == winner else None

    monkeypatch.setattr(p4long, 'decode_scores', eight)
    records = rx.refine()
    assert records and records[0].observation.frame == wanted
    assert calls[0] == (8, 1, 'single')
    if winner == 'sum':
        assert calls[-1] == (8, 1, 'sum')
    assert len(entries[1]['score_sets']) == 2


def test_refine_admits_measured_pass_and_resumes_negative_polarity(monkeypatch):
    from hfmodem.shrike import p4receiver
    rx = receiver()
    pending = Observation(Kind.LONG, FS, 0, FIRST, FIRST, 3.282, -7.1, .7,
                          provisional=True, placed=True)
    entry = dict(kind='normal', obs=pending, scores=np.zeros(3*p4long.SYSTEMATIC_BITS))
    rx._long_cache[(FIRST, -6.94)] = entry
    now = [10.]
    calls = []
    wanted = p4frame.decode_field(p4frame.encode_field(
        b'x'*203, p4frame.Status.from_byte(2)), data_bytes=203)

    def decode(scores, *, iterations, polarity):
        calls.append(polarity)
        now[0] += .052
        return wanted if polarity == -1 else None

    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    monkeypatch.setattr(p4long, 'decode_scores', decode)
    assert rx.refine(deadline=now[0]+.040) == []
    assert calls == []
    assert rx.refine(deadline=now[0]+.060) == []
    assert calls == [1]
    assert rx.refine(deadline=now[0]+.060)[0].observation.frame == wanted
    assert calls == [1, -1]


def test_six_hz_step_uses_crc_to_discard_strong_training_alias():
    tape = audio()
    rx = receiver()
    start = FIRST-round(.050*FS)
    nominal = round((FIRST-start)*28800/FS)+p4short.GEOM
    for end, partial in ((FIRST+round(.810*FS), True),
                         (FIRST+round(3.355*FS), False)):
        z = p4rx.audio_to_baseband(tape[start:end], FS)
        z *= np.exp(-2j*np.pi*6*np.arange(len(z))/28800)
        rx._normal_baseband = z, start, nominal
        obs, _ = rx._long(tape[start:end], start, placed=True, partial=partial, alias=True)
    assert obs is not None and obs.frame is not None
    assert obs.frame.status.byte == 2 and obs.frame.data.startswith(PREFIX)
    assert -13.5 < obs.cfo_hz < -12.5


def test_alias_retry_requires_both_short_and_robust_failure(monkeypatch):
    rx = receiver()
    calls = []

    def placed(z, nominal, cfo, *, offsets, trainings):
        calls.append(cfo)
        return (.61 if cfo > 0 else .30), nominal, cfo

    monkeypatch.setattr(p4short, 'place_long', placed)
    silence = np.zeros(FIRST+round(.810*FS), np.float32)
    obs, _ = rx._long(silence[FIRST-round(.050*FS):], FIRST-round(.050*FS),
                      placed=True, partial=True, alias=False)
    assert obs is None and calls == [-6.94]
    obs, _ = rx._long(silence[FIRST-round(.050*FS):], FIRST-round(.050*FS),
                      placed=True, partial=True, alias=True)
    assert obs is not None and obs.provisional
    assert calls[1:] == pytest.approx([-6.94, -6.94-1800/239, -6.94+1800/239])


NEGATIVE_CONTROLS = [
    ('onair-1002-1013', 60, '59e22aa1f5b5b431872e3494f95436fde6dedff1d575f5ba4efbc60a1c8ffd6d'),
    ('onair-1001-1045', 77, 'ea7eac9ed69647c980eb2a45c300018f9ad2166bd8eb291faa370fb52afb83bc'),
]


def trace_digest(value):
    return hashlib.sha256(json.dumps(value,default=str,separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('arm,stop,digest', NEGATIVE_CONTROLS)
def test_recorded_negative_controls_keep_their_decisions(arm, stop, digest, monkeypatch):
    if arm == 'onair-1002-1013':
        monkeypatch.setattr(p4turn.P4Turn, '_handback_seed', lambda self, reg, now: False)
    with neutral_keys() as neutral, waveform_keys() as waves:
        result = recorded(arm, stop*FS)
    assert not [row for row in result['heard'] if row['kind'] == Kind.LONG]
    rows = [(e['event'], e['at'], e.get('kind'), e.get('word'), e.get('state'), e.get('reason'))
            for e in result['events'] if e['event'] in ('Keyed', 'Decision')]
    pins=json.loads((Path(__file__).with_name('fixtures')/
                     'nak-stack1003e-controls.json').read_text())[arm]
    # Keep the immutable old digest and separately pin the complete active
    # event trace, rather than discarding neutral keys or changed decisions.
    assert pins['without_requests_decision_digest']==digest
    assert trace_digest(rows)==pins['decision_digest']
    assert trace_digest(result['heard'])==pins['heard_digest']
    assert neutral==pins['neutral']
    neutral_at={e['at'] for e in neutral}
    ordinary=[e for e in prerender(result['emissions']) if e['audio_start'] not in neutral_at]
    ordinary_waves=[e for e in waves if e['at'] not in neutral_at]
    assert len(ordinary)==pins['ordinary_emission_count']
    assert trace_digest(ordinary)==pins['ordinary_emission_digest']
    assert len(ordinary_waves)==pins['ordinary_waveform_count']
    assert trace_digest(ordinary_waves)==pins['ordinary_waveform_digest']


@pytest.mark.parametrize('arm,stop,digest', NEGATIVE_CONTROLS)
def test_recorded_negatives_without_neutral_requests_keep_original_digest(arm,stop,digest,monkeypatch):
    if arm=='onair-1002-1013':
        monkeypatch.setattr(p4turn.P4Turn,'_handback_seed',lambda self,reg,now:False)
    without_neutral_requests(monkeypatch)
    with neutral_keys() as neutral, waveform_keys() as waves:
        result=recorded(arm,stop*FS)
    assert neutral==[]
    assert not [row for row in result['heard'] if row['kind']==Kind.LONG]
    rows=[(e['event'],e['at'],e.get('kind'),e.get('word'),e.get('state'),e.get('reason'))
          for e in result['events'] if e['event'] in ('Keyed','Decision')]
    assert trace_digest(rows)==digest
    pins=json.loads((Path(__file__).with_name('fixtures')/
                     'nak-stack1003e-controls.json').read_text())[arm]
    assert trace_digest(result['heard'])==pins['without_requests_heard_digest']
    assert len(result['emissions'])==pins['ordinary_emission_count']
    assert trace_digest(prerender(result['emissions']))==pins['ordinary_emission_digest']
    assert len(waves)==pins['ordinary_waveform_count']
    assert trace_digest(waves)==pins['ordinary_waveform_digest']
