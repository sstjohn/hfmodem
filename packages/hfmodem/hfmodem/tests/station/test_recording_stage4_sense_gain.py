# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Operating measurements remain independent of audio persistence."""
from contextlib import nullcontext
import importlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_stage4_support import (
    assert_idle, audio_files, one_session, finish, lane_audio,
)
from hfmodem.tests.station.recording_support import bounded, pcm

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('busy', (False, True))
def test_python_sense_every_window_uses_shared_lane(stage4, monkeypatch, policy, busy):
    module = importlib.import_module('onair_session')
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(module, 'FS', 8)
    monkeypatch.setattr(module, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(module, 'device_opened', lambda: None)
    monkeypatch.setattr(module, 'sd', NS(rec=lambda n, **k: pcm(0, n)[:, None],
                                        wait=lambda: None, stop=lambda: None))
    monkeypatch.setattr(module, 'is_busy', lambda *a: (busy, [], 3. if busy else -3.))
    monkeypatch.setattr(module, 'scores', lambda *a: {'live_s': 2.})
    monkeypatch.setattr(module, 'level_range', lambda *a: (-30., -40.))
    result = module.listen(2, 'synthetic', stage4.root / 'sense.wav', policy=policy)
    assert tuple(result) == (busy, 3. if busy else -3., -30., -40., True)
    assert_idle(stage4, policy, 16)


@pytest.mark.posix_only('sources tools/lib/rig.sh under bash; Windows senses through consolelib.launch')
@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('busy', (False, True))
def test_shell_sense_keeps_all_windows_without_contact(stage4, shell_guard, policy, busy):
    shim = stage4.root / 'sense-python'
    shim.write_text(f'''#!{sys.executable}
import sys, types
from pathlib import Path
import numpy as np
from hfmodem.core import claim, devices, busy, wav
from hfmodem.shrike import rxfront
wav.SEGMENT_SAMPLES = 48000
wav._flac = lambda *a: (_ for _ in ()).throw(OSError('encoder unavailable'))
claim.received = lambda **k: types.SimpleNamespace(active=False, opened=lambda: None, failed=lambda: None)
devices.find_device = lambda *a, **k: 0
busy.scores = lambda *a: dict(burst=0., shape=0., tone=0., tone_hz=1500., live_s=8.)
busy.dominant = lambda s: ('tone', {3. if busy else -3.})
rxfront.call_b_frames = lambda *a: []
class Input:
    def __init__(self, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self, n): return (np.full((n, 1), .125, np.float32), False)
sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=Input)
assert sys.argv[1:3] == ['-m', 'consolelib.sense'], sys.argv
import functools, os
from consolelib import sense
sense.sense_channel = functools.partial(sense.sense_channel, root=Path(os.environ['REPO']))
sys.exit(sense.main(sys.argv[3:]))
''')
    shim.chmod(0o755)
    env = dict(shell_guard, HFMODEM_PYTHON=str(shim), HFMODEM_RECORDING=policy,
               REPO=str(stage4.root), AUDIO='synthetic',
               PYTHONPATH=f'{ROOT}/tools:{ROOT}/packages/hfmodem:{ROOT}/packages/hfhost')
    script = f'''source {shlex.quote(str(ROOT / 'tools/lib/rig.sh'))}
tune() {{ :; }}
claim_pending() {{ :; }}
claim_worker() {{ "$@"; }}
_sense_channel 7105000 vara 500 0
'''
    result = subprocess.run(['/bin/bash', '-c', script], env=env, text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == int(busy), result.stdout + result.stderr
    assert ('OCCUPIED' if busy else 'clear') in result.stdout
    assert bool(audio_files(stage4.root)) == (policy == 'everything')
    if policy == 'off':
        return
    indexes = list(stage4.root.rglob('recording.json'))
    assert len(indexes) == 1, 'shell sense must feed a shared sense lane'
    facts = json.loads(indexes[0].read_text())
    assert facts['contacts'] == {}
    assert sum(l['position'] for l in facts['lanes'].values()) == 8 * 48000


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_gain_probe_uses_sample_measurements_without_read_or_unlink(stage4, monkeypatch, policy):
    station = importlib.import_module('consolelib.station')
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(station, 'default_selection', lambda: NS(AUDIO_DEVICE='synthetic'))
    monkeypatch.setattr(station, 'ROOT', stage4.root)
    calls = []
    block = np.array([.25, -.25, .5, -.5], np.float32)
    peak = 20 * math.log10(.5)
    rms = 20 * math.log10(float(np.sqrt(np.mean(block ** 2))))

    def child(argv, **kwargs):
        calls.append((argv, kwargs))
        return NS(returncode=0, stdout=json.dumps(dict(samples=4, rate=8,
                  peak=.5, rms=float(np.sqrt(np.mean(block ** 2))), peak_dbfs=peak, rms_dbfs=rms)), stderr='')

    monkeypatch.setattr(station.subprocess, 'run', child)
    def no_read(*a, **k):
        raise AssertionError('gain probe must consume measurements without reading an audio file')
    monkeypatch.setattr(wav, 'read', no_read)
    try:
        result = station.listen(.5, 'synthetic')
    except RuntimeError as exc:
        pytest.fail(f'gain probe requires a file despite successful sample measurements: {exc}')
    assert tuple(result)[:2] == pytest.approx((peak, rms))
    count = getattr(result, 'samples', None)
    assert (count if count is not None else result[2]) == 4
    assert calls


@pytest.mark.posix_only('the gain report reads codec gain, which Windows leaves to its Sound settings')
@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_gain_report_advice_is_policy_independent(stage4, monkeypatch, capsys, policy):
    station = importlib.import_module('consolelib.station')
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(station, 'default_selection', lambda: NS(AUDIO_DEVICE='synthetic', RIG_NAME='ft891'))
    monkeypatch.setattr(station, 'read_codec_gain', lambda *a: .5)
    monkeypatch.setattr(station, 'listen', lambda *a: (-12., -40.))
    assert station.gain_report(1, False, 'synthetic') == 0
    assert 'gain reads 0.500' in capsys.readouterr().out


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_gain_probe_child_retains_probe_only_in_run_root(stage4, monkeypatch, policy):
    station = importlib.import_module('consolelib.station')
    tap = importlib.import_module('txwitness')
    from hfmodem.core import devices
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(station, 'default_selection', lambda: NS(AUDIO_DEVICE='synthetic'))
    monkeypatch.setattr(station, 'ROOT', stage4.root)
    monkeypatch.setattr(tap, 'FS', 8)
    monkeypatch.setattr(tap, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(tap, 'device_opened', lambda: None)
    monkeypatch.setattr(tap.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(devices, 'find_device', lambda *a, **k: 0)
    block = np.array([.25, -.25, .5, -.5], np.float32)
    class Input:
        def __init__(self, **kwargs): self.callback = kwargs['callback']
        def __enter__(self):
            self.callback(block[:, None], 4, NS(), NS(input_overflow=False))
            return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(sys.modules['sounddevice'], 'InputStream', Input)
    monkeypatch.setattr(sys.modules['sounddevice'], 'query_devices', lambda *a: [{'name': 'synthetic'}])
    def child(argv, **kwargs):
        import contextlib
        import io
        captured = io.StringIO()
        assert 'record' in argv, 'probe child must use the scripted capture entry point'
        at = argv.index('record')
        with contextlib.redirect_stdout(captured), monkeypatch.context() as child_env:
            for key, value in kwargs.get('env', {}).items():
                child_env.setenv(key, value)
            rc = tap.record(float(argv[at + 1]), argv[at + 2], 'synthetic')
        return NS(returncode=rc, stdout=captured.getvalue(), stderr='')
    monkeypatch.setattr(station.subprocess, 'run', child)
    peak, rms, *count = station.listen(.5, 'synthetic')
    assert (peak, rms) == pytest.approx((20 * math.log10(.5), 20 * math.log10(np.sqrt(.15625))))
    finish(stage4)
    assert bool(audio_files(stage4.root)) == (policy == 'everything'), 'everything must retain probe audio instead of unlinking it'
    rec = one_session(stage4, policy)
    assert not rec.index['contacts']


def test_gain_probe_silence_preserves_sample_count(stage4, monkeypatch):
    station = importlib.import_module('consolelib.station')
    monkeypatch.setenv('HFMODEM_RECORDING', 'off')
    monkeypatch.setattr(station, 'default_selection', lambda: NS(AUDIO_DEVICE='synthetic'))
    monkeypatch.setattr(station.subprocess, 'run', lambda *a, **k: NS(
        returncode=0, stderr='', stdout=json.dumps(dict(samples=4, rate=8, peak=0., rms=0.,
                                                       peak_dbfs=-120., rms_dbfs=-120.))))
    assert station.listen(.5, 'synthetic') == (None, None, 4)


def test_acquired_windows_advance_while_storage_is_blocked(sessions, monkeypatch):
    import threading
    from hfmodem.core import recording
    from hfmodem.tests.station.recording_support import bounded, flush
    rec, _ = sessions('everything')
    lane = rec.stream('sense', rate=8, clock='card')
    entered, release = threading.Event(), threading.Event()
    def blocked():
        entered.set()
        assert release.wait(5)
    rec.after_pending(blocked)
    assert entered.wait(5)
    # Only the blocked phase runs on the short timeout: closing waits on a real
    # storage worker join, which a loaded Windows box does not finish in 10 ms.
    with monkeypatch.context() as patch:
        patch.setattr(recording, '_CLOSE_TIMEOUT', .01)
        try:
            bounded(lambda: recording.record_window(lane, pcm(0, 48)))
            assert lane.position == 48
            bounded(lambda: recording.record_window(lane, pcm(48, 64)))
            assert lane.position == 64
        finally:
            release.set()
    flush(rec)
    assert lane.consumed == 64
    bounded(rec.close)
    assert sum(d['samples'] for d in rec.index['segments']) + sum(b-a for a,b in lane.gaps) == 64


def block_storage(rec, release):
    entered = threading.Event()
    def blocked():
        entered.set()
        assert release.wait(5)
    rec.after_pending(blocked)
    assert entered.wait(5)


@pytest.mark.parametrize('stalled', (False, True))
def test_record_window_keeps_five_seconds_at_production_rate(sessions, stalled):
    from hfmodem.core import recording
    rec, _ = sessions('everything')
    lane = rec.stream('sense', rate=48000, clock='card')
    audio = pcm(0, 5 * lane.rate)
    expected = np.round(audio * 32768).astype('<i2')
    release = threading.Event()
    try:
        if stalled:
            block_storage(rec, release)
        with rec._state_lock:
            bounded(lambda: recording.record_window(lane, audio))
            assert lane.position == len(audio)
            audio[:] = 0
    finally:
        release.set()
    bounded(rec.close)
    saved = lane_audio(rec, lane)
    assert len(saved) == len(expected)
    np.testing.assert_array_equal(saved, expected)
    assert lane.gaps == []


def test_record_window_leaves_callback_capacity_bounded(sessions):
    from hfmodem.core import recording
    rec, _ = sessions('everything')
    lane = rec.stream('sense', rate=48000, clock='card')
    with rec._state_lock:
        bounded(lambda: recording.record_window(lane, pcm(0, 5 * lane.rate)))
        bounded(lambda: lane.push(lane.position, pcm(5 * lane.rate, 7 * lane.rate)))
        assert lane.queued == 2 * lane.rate
        bounded(lambda: lane.push(lane.position, pcm(7 * lane.rate, 8 * lane.rate)))
        assert lane.queued == 2 * lane.rate
        assert lane.position == 8 * lane.rate
    bounded(rec.close)
    np.testing.assert_array_equal(lane_audio(rec, lane),
                                 np.round(pcm(0, 7 * lane.rate) * 32768).astype('<i2'))
    assert lane.gaps == [[7 * lane.rate, 8 * lane.rate]]


def test_repeated_acquired_windows_with_stalled_drainer_are_bounded(sessions):
    from hfmodem.core import recording
    rec, _ = sessions('everything')
    lane = rec.stream('sense', rate=48000, clock='card')
    with rec._state_lock:
        for n in range(20):
            bounded(lambda: recording.record_window(lane, pcm(0, 5 * lane.rate)))
        assert lane.position == 100 * lane.rate
        queued = list(rec._pcm.queue)
        assert sum(len(item[3]) for item in queued if item[0] == 'window') <= 5 * lane.rate
    bounded(rec.close)
    assert lane.consumed == 100 * lane.rate
    assert lane.gaps == [[5 * lane.rate, 100 * lane.rate]]
    assert len(lane_audio(rec, lane)) == 5 * lane.rate


def test_record_window_drainer_failure_preserves_absolute_gap(sessions, monkeypatch):
    from hfmodem.core import recording
    from hfmodem.tests.station.recording_support import flush
    rec, _ = sessions('everything')
    lane = rec.stream('sense', rate=48000, clock='card')
    def failed(*args):
        raise OSError('synthetic persistence failure')
    with monkeypatch.context() as patch:
        patch.setattr(rec, '_consume', failed)
        bounded(lambda: recording.record_window(lane, pcm(0, 5 * lane.rate)))
        flush(rec)
    bounded(lambda: recording.record_window(lane, pcm(5 * lane.rate, 10 * lane.rate)))
    bounded(rec.close)
    assert lane.position == lane.consumed == 10 * lane.rate
    assert lane.gaps == [[0, 5 * lane.rate]]
    np.testing.assert_array_equal(lane_audio(rec, lane),
                                 np.round(pcm(5 * lane.rate, 10 * lane.rate) * 32768).astype('<i2'))


@pytest.mark.parametrize('stalled', (False, True))
@pytest.mark.parametrize('busy', (False, True))
def test_python_listen_keeps_five_seconds_and_verdict(stage4, monkeypatch, stalled, busy, capsys):
    from hfmodem.core import recording
    module = importlib.import_module('onair_session')
    assert module.FS == 48000
    audio = pcm(0, 5 * module.FS)
    monkeypatch.setattr(module, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(module, 'device_opened', lambda: None)
    monkeypatch.setattr(module, 'sd', NS(rec=lambda n, **k: audio[:, None],
                                        wait=lambda: None, stop=lambda: None))
    monkeypatch.setattr(module, 'is_busy', lambda *a: (busy, [], 3. if busy else -3.))
    monkeypatch.setattr(module, 'scores', lambda *a: {'live_s': 5.})
    monkeypatch.setattr(module, 'level_range', lambda *a: (-30., -40.))
    release = threading.Event()
    stream = recording.RecordingSession.stream
    record_window = recording.record_window

    def opened(self, *args, **kwargs):
        lane = stream(self, *args, **kwargs)
        if stalled:
            block_storage(self, release)
        return lane

    def acquired(lane, samples):
        with lane.session._state_lock:
            record_window(lane, samples)

    results = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(recording.RecordingSession, 'stream', opened)
            patch.setattr(recording, 'record_window', acquired)
            if stalled:
                patch.setattr(recording, '_CLOSE_TIMEOUT', .01)
            bounded(lambda: results.append(module.listen(
                5, 'synthetic', stage4.root / 'sense.wav', policy='everything')))
        assert tuple(results[0]) == (busy, 3. if busy else -3., -30., -40., True)
        if stalled:
            assert 'Recording sense close:' in capsys.readouterr().err
    finally:
        release.set()
    finish(stage4)
    rec = one_session(stage4, 'everything')
    lane = rec.lanes['sense']
    assert lane.position == len(audio)
    saved = lane_audio(rec, lane)
    assert len(saved) == len(audio)
    np.testing.assert_array_equal(saved, np.round(audio * 32768).astype('<i2'))
    assert lane.gaps == []


@pytest.mark.parametrize('failure', ('stream', 'close'))
def test_python_sense_persistence_failure_preserves_verdict(stage4, monkeypatch, failure):
    from hfmodem.core import recording
    module = importlib.import_module('onair_session')
    monkeypatch.setattr(module, 'FS', 8)
    monkeypatch.setattr(module, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(module, 'device_opened', lambda: None)
    monkeypatch.setattr(module, 'sd', NS(rec=lambda n, **k: pcm(0, n)[:, None],
                                        wait=lambda: None, stop=lambda: None))
    monkeypatch.setattr(module, 'is_busy', lambda *a: (False, [], -3.))
    monkeypatch.setattr(module, 'scores', lambda *a: {'live_s': 2.})
    monkeypatch.setattr(module, 'level_range', lambda *a: (-30., -40.))
    original = getattr(recording.RecordingSession, failure)
    def failed(self, *a, **k):
        if failure == 'close': original(self, *a, **k)
        raise TimeoutError('synthetic recording failure')
    with monkeypatch.context() as patch:
        patch.setattr(recording.RecordingSession, failure, failed)
        result = module.listen(2, 'synthetic', stage4.root / 'sense.wav', policy='everything')
    assert tuple(result) == (False, -3., -30., -40., True)


def test_gain_probe_default_retention_uses_persistent_log_root(stage4, monkeypatch, capsys):
    station = importlib.import_module('consolelib.station')
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything')
    monkeypatch.setattr(station, 'ROOT', stage4.root)
    monkeypatch.setattr(station, 'default_selection', lambda: NS(AUDIO_DEVICE='synthetic'))
    paths = []
    def child(argv, **kwargs):
        paths.append(Path(argv[argv.index('record') + 2]))
        return NS(returncode=0, stderr='', stdout=json.dumps(dict(
            samples=4, peak=.5, rms=.25, peak_dbfs=-6., rms_dbfs=-12.)))
    monkeypatch.setattr(station.subprocess, 'run', child)
    station.listen(.5)
    station.listen(.5)
    assert all(path.is_relative_to(stage4.root / 'logs') for path in paths)
    assert paths[0].parent != paths[1].parent
    assert str(paths[0].parent) in capsys.readouterr().out
