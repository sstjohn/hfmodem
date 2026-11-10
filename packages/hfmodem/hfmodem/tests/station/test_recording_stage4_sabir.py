# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Sabir retains acquired sense and submitted output rather than rendered drafts."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import audio, claim
from hfmodem.sabir import onair, radio
from hfmodem.tests.station.recording_stage4_support import (
    audio_files, one_session, finish, lane_audio, flush_all, LOCAL,
)


@pytest.fixture
def sabir(stage4, monkeypatch):
    monkeypatch.setattr(onair, 'FS', 80)
    monkeypatch.setattr(radio, 'FS', 80)
    monkeypatch.setattr(audio, 'CARD_RATE_HZ', 80)
    monkeypatch.setattr(audio, '_WARMED', set())
    monkeypatch.setattr(onair, 'find_device', lambda *a, **k: 'synthetic')
    monkeypatch.setattr(onair, 'arming_refusal', lambda *a: None)
    monkeypatch.setattr(onair, '_install_signals', lambda *a: None)
    monkeypatch.setattr(claim, 'received', lambda **k: NS(claim=None, opened=lambda: None, failed=lambda: None))
    monkeypatch.setattr(radio, 'time', NS(sleep=lambda *a: None))
    burst = onair.Burst(np.array([0., 1., -.5, .25] * 60, np.float32), 'synthetic beacon', 500, 2500)
    monkeypatch.setattr(onair, '_burst', lambda args: burst)
    state = NS(writes=[], attempts=0, fault=None, busy=False, key=True, fail_write=None,
               opened=[], rig_events=[], acquired=[], signal_writes=0, fail_signal=None,
               capture_opens=0)

    class Rig:
        mode, settle, retired = 'PKTUSB', 0., False
        def __init__(self, *a, **k): pass
        def identify(self): return 'synthetic'
        def transmitting(self): return False
        def qsy(self, hz): return hz
        def set_mode(self): pass
        def arm(self): state.rig_events.append('arm')
        def key(self, *a):
            state.rig_events.append('key')
            return state.key
        def unkey(self, *a): state.rig_events.append('unkey')
        def retire(self): state.rig_events.append('retire')
        def hand_back(self): state.rig_events.append('release')

    class Output:
        def __init__(self, **kwargs): state.opened.append(kwargs)
        def start(self): pass
        def write(self, block):
            flush_all(stage4)
            state.attempts += 1
            if np.any(block):
                state.signal_writes += 1
            if state.attempts == state.fail_write or (np.any(block) and state.signal_writes == state.fail_signal):
                raise OSError('synthetic uncertain output write')
            state.writes.append(np.asarray(block).ravel().copy())
            return False
        def stop(self): pass
        close = abort = stop

    def capture(n, **kwargs):
        state.acquired.append(n)
        return np.full((n, 1), .125, np.float32)

    class Input:
        def __init__(self, **kwargs):
            state.capture_opens += 1
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count):
            flush_all(stage4)
            return capture(count), False

    monkeypatch.setattr(onair.radio, 'Rig', Rig)
    monkeypatch.setattr(sys.modules['sounddevice'], 'rec', capture)
    monkeypatch.setattr(sys.modules['sounddevice'], 'InputStream', Input)
    monkeypatch.setattr(sys.modules['sounddevice'], 'OutputStream', Output)
    monkeypatch.setattr(onair, 'is_busy', lambda *a: (state.busy, [], 2. if state.busy else -2.))
    monkeypatch.setattr(onair, 'level_range', lambda *a: (-25., -35.))
    monkeypatch.setattr(onair, 'receiver_fault', lambda *a, **k: state.fault)

    def run(policy='everything', extra=()):
        return onair.main(['--mycall', LOCAL, '--dial', '7105000', '--transmit',
            '--line-ptt', '--ptt-device', 'synthetic', '--rigctld', 'synthetic.invalid:4532',
            '--regulatory', 'unregulated', '--because', 'synthetic PCM', '--listen', '1',
            '--audio-in', 'synthetic', '--audio-out', 'synthetic', '--gain', '.2',
            '--run-dir', str(stage4.root), '--recording', policy, *extra])
    state.run, state.burst = run, burst
    return state


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('outcome', ('sent', 'busy', 'deaf', 'intermittent', 'key-refused'))
def test_sabir_operating_policy_sense_and_actual_submission(sabir, stage4, monkeypatch, policy, outcome):
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything' if policy != 'everything' else 'off')
    sabir.busy = outcome == 'busy'
    if outcome in ('deaf', 'intermittent'):
        sabir.fault = NS(kind=outcome, reason='synthetic receiver fault')
    sabir.key = outcome != 'key-refused'
    rc = sabir.run(policy)
    assert rc == {'sent': 0, 'busy': 1, 'deaf': 2, 'intermittent': 2, 'key-refused': 1}[outcome]
    assert sabir.acquired == [80]
    if outcome in ('busy', 'deaf', 'intermittent'):
        assert 'key' not in sabir.rig_events
    else:
        assert 'unkey' in sabir.rig_events and sabir.rig_events[-1] == 'release'
    finish(stage4)
    assert bool(audio_files(stage4.root)) == (policy == 'everything')
    assert not (stage4.root / 'beacon.wav').exists()
    rec = one_session(stage4, policy)
    assert rec.index['contacts'] == {}, 'one-way beacon establishes no contact'
    sense = [lane for lane in rec.lanes.values() if 'sense' in lane.name or 'rx' in lane.name]
    assert len(sense) == 1 and sense[0].position == 80
    if policy == 'everything':
        np.testing.assert_array_equal(lane_audio(rec, sense[0]), np.full(80, 4096, '<i2'))
        if sabir.writes:
            tx = next(lane for lane in rec.lanes.values() if 'tx' in lane.name)
            expected = np.concatenate(sabir.writes)
            np.testing.assert_array_equal(lane_audio(rec, tx), np.round(expected * 32768).astype('<i2'))
            assert all(len(block) <= 4 for block in sabir.writes), 'submission writes must stay bounded'


@pytest.mark.parametrize('failure', ('admission', 'write'))
def test_sabir_refusal_and_uncertain_write_never_count_unsent_tail(sabir, stage4, monkeypatch, failure):
    if failure == 'admission':
        def refuse(*a, **k): raise claim.ClaimError('synthetic admission refused')
        monkeypatch.setattr(claim, 'received', refuse)
        assert sabir.run() == 2
        assert sabir.writes == []
    else:
        sabir.fail_signal = 2
        with pytest.raises(OSError, match='uncertain output'):
            sabir.run()
        assert 'unkey' in sabir.rig_events and sabir.rig_events[-1] == 'release'
        finish(stage4)
        rec = one_session(stage4, 'everything')
        tx = next(lane for lane in rec.lanes.values() if 'tx' in lane.name)
        assert tx.position <= sum(map(len, sabir.writes)) + 4
        assert tx.row['gaps'] or tx.row.get('uncertain'), 'failed write needs explicit loss metadata'
        retained = lane_audio(rec, tx)
        assert len(retained) == sum(map(len, sabir.writes))
        assert len(retained) < len(sabir.burst.audio)


def test_sabir_sense_acquisition_is_bounded(sabir, stage4):
    sabir.run(extra=('--listen', '5'))
    assert sabir.acquired
    assert all(n <= 2 * 80 for n in sabir.acquired)
    rec = one_session(stage4, 'everything')
    assert rec.lanes


def test_sabir_sense_uses_one_continuous_capture(sabir, stage4):
    assert sabir.run(extra=('--listen', '5')) == 0
    assert sabir.capture_opens == 1
    assert sum(sabir.acquired) == 5 * 80


def test_explicit_sabir_offline_render_remains_available(sabir, stage4):
    path = stage4.root / 'offline.wav'
    assert onair.main(['--mycall', LOCAL, '--recording', 'off', '--wav', str(path)]) == 0
    assert path.is_file()
    assert sabir.writes == [] and sabir.rig_events == []


def test_play_drained_submits_only_successful_bounded_writes(sabir):
    submitted = []
    sabir.fail_write = 3
    with pytest.raises(OSError, match='uncertain output'):
        audio.play_drained(np.arange(20, dtype=np.float32) / 100, 80, 'synthetic',
            submitted=lambda at, block: submitted.append((at, block.copy())))
    assert [at for at, _ in submitted] == [0, 4]
    np.testing.assert_array_equal(np.concatenate([b for _, b in submitted]),
                                  np.arange(8, dtype=np.float32) / 100)
    assert all(len(block) <= 4 for _, block in submitted)


def test_warm_output_exposes_same_submission_hook(sabir):
    submitted = []
    try:
        audio.warm_output('synthetic', submitted=lambda at, block: submitted.append((at, block.copy())))
    except TypeError as exc:
        pytest.fail(f'warm_output has no submission hook: {exc}')
    assert submitted
    assert sum(len(block) for _, block in submitted) == int(audio.WARM_S * 80)
    assert all(not np.any(block) for _, block in submitted)


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('busy', (False, True))
def test_operating_sabir_never_persists_generated_render_before_admission(sabir, stage4, policy, busy):
    sabir.busy = busy
    generated = stage4.root / 'beacon.wav'
    assert sabir.run(policy, extra=('--wav', str(generated))) == int(busy)
    assert not generated.exists(), 'an operating render is not acquired sense or submitted TX'
    finish(stage4)
    assert bool(audio_files(stage4.root)) == (policy == 'everything')


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_sabir_run_override_wins_over_contrary_station_policy(sabir, stage4, monkeypatch, policy):
    from hfmodem.tests.station.test_recording_launchers import station_file
    monkeypatch.delenv('HFMODEM_RECORDING', raising=False)
    station = station_file(stage4.root, 'off' if policy == 'everything' else 'everything')
    monkeypatch.setenv('HFMODEM_STATION', str(station))
    assert sabir.run(policy) == 0
    finish(stage4)
    rec = one_session(stage4, policy)
    assert rec.index['source'] == 'explicit'


def test_sabir_successful_underrun_marks_submission_uncertain(sabir, stage4, monkeypatch):
    output = sys.modules['sounddevice'].OutputStream
    write = output.write
    def underrun(self, block):
        write(self, block)
        return True
    monkeypatch.setattr(output, 'write', underrun)
    assert sabir.run() == 0
    finish(stage4)
    tx = one_session(stage4, 'everything').lanes['submitted-tx']
    assert tx.row.get('uncertain'), 'successful writes reporting underrun need uncertainty'
    assert len(lane_audio(stage4.sessions[0], tx)) == sum(map(len, sabir.writes))
    assert sabir.rig_events[-2:] == ['unkey', 'release']
