# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Attached witness launch and capture clocks use the owner's policy."""
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.tests.station.recording_stage4_support import (
    assert_idle, audio_files, one_session, finish, flush_all, PEER,
)

ROOT = Path(__file__).resolve().parents[5]
SHELL = pytest.mark.posix_only(
    'runs witness_start/witness_stop from tools/onair.sh under bash; Windows uses consolelib.launch_capture')


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('kind', ('tap', 'kiwi'))
def test_console_attached_child_policy_and_control_pipe(stage4, monkeypatch, policy, kind):
    station = importlib.import_module('consolelib.station')
    operator = importlib.import_module('consolelib.operator')
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setenv('HFMODEM_TAP_DEVICE', 'synthetic')
    inventory = stage4.root / 'candidates.csv'
    inventory.write_text('host,port\nreceiver.invalid,8073\n')
    monkeypatch.setattr(operator, 'witness_inventory', lambda: inventory)
    monkeypatch.setattr(station, 'selection_for_plan', lambda p: NS(RIG_NAME='ft891'))
    monkeypatch.setattr(station, 'station_grid', lambda p: 'AA00aa')
    monkeypatch.setattr(station.claim, 'inherited', lambda *a, **k: NS(failed=lambda: None))
    monkeypatch.setattr(station.claim, 'process_started', lambda *a: 'synthetic')
    monkeypatch.setattr(station.coexist, 'event', lambda *a, **k: None)
    calls, said = [], []

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        out = kwargs['stdout']
        out.write('CLAIM ADMITTED\nWITNESS PLACED: receiver.invalid, 600 mi from us\n')
        out.flush()
        return NS(pid=123456, poll=lambda: None, terminate=lambda: None,
                  wait=lambda **k: 0, kill=lambda: None)

    monkeypatch.setattr(station.subprocess, 'Popen', spawn)
    # The tap goes through the claim's launch; the stand-in admission has no descriptor.
    monkeypatch.setattr(station.claim, 'spawn', lambda descriptor, argv, *, role, **kwargs:
                        station.claim.Launch(spawn(argv, **kwargs)))
    from consolelib import claim as console_claim
    monkeypatch.setattr(console_claim, 'spawn', lambda descriptor, argv, *, role, **kwargs: spawn(argv, **kwargs))
    plan = dict(call=PEER, dial_hz=7105000, env={'HFMODEM_RECORDING': policy},
                adjustments=dict(witness_seconds=1, prepare_radio=False, kiwi=''))
    child = (station.start_tap if kind == 'tap' else station.place_kiwi)(
        plan, stage4.root, say=said.append)
    if policy == 'off':
        assert child is None and calls == [], 'off must not start a recorder-only child'
        assert any('off' in line.lower() for line in said), 'intentional absence must be visible'
    else:
        assert child is not None and len(calls) == 1
        argv, kwargs = calls[0]
        assert kwargs['env']['HFMODEM_RECORDING'] == policy
        fd = kwargs['env'].get('HFMODEM_RECORDING_CONTROL_FD')
        assert fd is not None, 'attached child needs the shared inherited boundary pipe'
        if os.name == 'nt':
            assert fd in kwargs.get('handles', {}), 'control handle must reach the child as a launch extra'
        else:
            assert int(fd) in kwargs.get('pass_fds', ()), 'control descriptor must survive exec'
    station.close_attached_controls(plan)


@SHELL
@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_shell_witness_start_and_stop_capture_states(stage4, shell_guard, policy):
    source = (ROOT / 'tools/onair.sh').read_text()
    functions = source[source.index('witness_start() {'):source.index('unkey_and_exit() {')]
    shim = stage4.root / 'witness-child'
    # The recorder reports contact-only coverage under contacts and fails under
    # everything; as any other interpreter call it answers nothing.
    shim.write_text('#!/bin/sh\ncase "$*" in *kiwi_witness.py*) ;; *) exit 0 ;; esac\n'
                    'printf "%s\\n" "$*" >> "$STAGE4_CHILDREN"\n'
                    '[ "$HFMODEM_RECORDING" = contacts ] || exit 1\n'
                    'printf "WITNESS COVERAGE: contact-only; retained 0 contact(s)\\n"\n')
    shim.chmod(0o755)
    childlog = stage4.root / 'children'
    env = dict(shell_guard, HFMODEM_RECORDING=policy, HFMODEM_PYTHON=str(shim),
               STAGE4_CHILDREN=str(childlog))
    script = f'''{functions}
SESSION={shlex.quote(str(stage4.root))}
REPO={shlex.quote(str(ROOT))}
WITNESS=1
WITNESS_MAX_S=1
CAPTURE_PREPARED=0
claim_record() {{ :; }}
claim_exit() {{ :; }}
claim_processes() {{ :; }}
recording_available() {{ return 1; }}
witness_start 7105000 W0QXR
witness_stop
'''
    result = subprocess.run(['/bin/bash', '-c', script], env=env, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    children = childlog.read_text() if childlog.exists() else ''
    if policy == 'off':
        assert not children, 'shell started a recorder-only child under off'
        assert 'off' in result.stdout.lower(), 'off teardown must report intentional absence'
    elif policy == 'contacts':
        assert 'no retained contact' in result.stdout.lower() or 'contact-only' in result.stdout.lower()
        assert '!! NO WITNESS' not in result.stdout
    else:
        assert 'unavailable' in result.stdout.lower(), 'capture failure must be reported as unavailable'


@pytest.fixture
def kiwi_capture(stage4, monkeypatch):
    kiwi = importlib.import_module('kiwi_witness')
    monkeypatch.setattr(kiwi.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(kiwi, 'candidates', lambda *a: [kiwi.Cand('receiver.invalid', 8073, 'test', 600, None)])
    monkeypatch.setattr(kiwi, 'peer_latlon', lambda *a: ((0., 0.), PEER))
    inventory = stage4.root / 'candidates.csv'
    inventory.write_text('synthetic')
    closed = []
    receiver = NS(sample_rate=8.25, close=lambda: closed.append(True))
    probe = [np.full(4, .125, np.float32)]

    def run(policy='everything', loss=False, operating=True, controls=(), stalled=False):
        monkeypatch.setenv('HFMODEM_RECORDING', policy)
        readfd, writefd = os.pipe()
        if operating:
            monkeypatch.setenv('HFMODEM_RECORDING_CONTROL_FD', str(readfd))
        else:
            monkeypatch.delenv('HFMODEM_RECORDING_CONTROL_FD', raising=False)
        if controls:
            os.write(writefd, b''.join((json.dumps(row) + '\n').encode() for row in controls))
        def blocks():
            rec = stage4.sessions[0] if stalled else None
            with rec._state_lock if stalled else nullcontext():
                for n in range(12):
                    if not stalled:
                        flush_all(stage4)
                    stage4.clock.now += .5
                    yield NS(samples=np.full(4, .25 if n % 2 else .125, np.float32))
                    if stalled:
                        pending = list(rec._pcm.queue)
                        assert sum(len(item[3]) for item in pending
                                   if item[0] in ('window', 'pcm')) <= 2 * receiver.sample_rate
            if loss:
                raise OSError('synthetic receiver loss')
        monkeypatch.setattr(kiwi, 'acquire', lambda *a: kiwi.Take(
            receiver, kiwi.Cand('receiver.invalid', 8073, 'test', 600, None),
            probe, 1000., blocks()))
        monkeypatch.setattr(kiwi.time, 'time', stage4.clock)
        try:
            return kiwi.main(['--dial', '7105', '--seconds', '100', '--grid', 'AA00aa',
                '--out', str(stage4.root / 'witness.wav'), '--peer', PEER,
                '--candidates', str(inventory)], client=NS())
        finally:
            os.close(writefd)
            os.close(readfd)
    return kiwi, run, closed


def test_kiwi_continuous_blocks_obey_callback_bound_with_stalled_drainer(kiwi_capture, stage4):
    _, run, _ = kiwi_capture
    assert run(stalled=True) == 0
    rec = one_session(stage4, 'everything')
    lane = next(iter(rec.lanes.values()))
    assert lane.position == lane.consumed == 52
    assert lane.gaps


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('loss', (False, True))
def test_kiwi_operating_capture_fixed_scale_fractional_clock(kiwi_capture, stage4, policy, loss):
    kiwi, run, closed = kiwi_capture
    assert run(policy, loss) == 0
    assert closed
    session = assert_idle(stage4, policy, 52)
    if policy == 'everything':
        lane = next(iter(session.lanes.values()))
        assert lane.rate == 8
        encoded = np.concatenate([np.frombuffer(wav.read_pcm(p)[1], '<i2') for p in audio_files(stage4.root)])
        assert encoded.max() == 8192, 'operating witnesses must use fixed scaling'
        metadata = '\n'.join(p.read_text() for p in stage4.root.rglob('*.json'))
        assert '8.25' in metadata, 'measured receiver rate must remain separate from header rate'
        assert lane.row.get('gaps', []) == []


def test_kiwi_operating_capture_never_assembles_whole_capture(kiwi_capture, monkeypatch):
    kiwi, run, _ = kiwi_capture
    concatenate = np.concatenate
    def bounded(parts, *a, **k):
        parts = list(parts)
        assert sum(np.asarray(part).size for part in parts) <= 16, 'Kiwi capture grows with total duration'
        return concatenate(parts, *a, **k)
    monkeypatch.setattr(kiwi.np, 'concatenate', bounded)
    run('everything')


def test_explicit_kiwi_capture_remains_available(kiwi_capture, stage4):
    _, run, _ = kiwi_capture
    assert run('off', operating=False) == 0
    assert audio_files(stage4.root), 'explicit standalone witness capture remains an export'


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('stalled', (False, True))
@pytest.mark.parametrize('operating', (False, True))
def test_attached_tap_uses_samples_for_completion_and_stall(stage4, monkeypatch, policy, stalled, operating):
    tap = importlib.import_module('txwitness')
    from hfmodem.core import devices
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    rd, wr = os.pipe()
    if operating:
        monkeypatch.setenv('HFMODEM_RECORDING_CONTROL_FD', str(rd))
    else:
        monkeypatch.delenv('HFMODEM_RECORDING_CONTROL_FD', raising=False)
    monkeypatch.setattr(tap, 'FS', 8)
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(tap, 'STALL_S', 1.)
    monkeypatch.setattr(tap, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(tap, 'device_opened', lambda: None)
    monkeypatch.setattr(tap.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(devices, 'find_device', lambda *a, **k: 0)
    # The tap's own clock only: patching the shared time module also moves the
    # recording workers' close deadlines, which a slow host then overruns.
    monkeypatch.setattr(tap, 'time', NS(monotonic=stage4.clock,
        sleep=lambda seconds: setattr(stage4.clock, 'now', stage4.clock.now + seconds)))
    sd = sys.modules['sounddevice']
    monkeypatch.setattr(sd, 'query_devices', lambda *a: [{'name': 'synthetic'}])
    class Input:
        def __init__(self, **kwargs): self.callback = kwargs['callback']
        def __enter__(self):
            if not stalled:
                self.callback(np.full((16, 1), .125, np.float32), 16, NS(), NS(input_overflow=False))
            return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(sd, 'InputStream', Input)
    try:
        assert tap.record(2, str(stage4.root / 'tap.wav'), 'synthetic') == int(stalled)
        if not operating:
            assert audio_files(stage4.root), 'explicit tap capture remains independent of operating policy'
            assert sum(wav.read_pcm(path)[0].nframes for path in audio_files(stage4.root)) == (0 if stalled else 16)
        elif not stalled:
            assert_idle(stage4, policy, 16)
        else:
            if policy != 'everything':
                assert audio_files(stage4.root) == []
            else:
                assert all(wav.read_pcm(path)[0].nframes == 0 for path in audio_files(stage4.root))
            one_session(stage4, policy)
    finally:
        os.close(wr)
        os.close(rd)


@pytest.mark.parametrize('provisional,keep', [(False, True), (True, True), (True, False)])
def test_kiwi_attached_contact_controls_select_parent_interval(kiwi_capture, stage4, provisional, keep):
    _, run, _ = kiwi_capture
    controls = [dict(op='begin', contact_id='parent-c1', positions={'kiwi-1': 8}, anchors={},
                     provisional=provisional),
                dict(op='end', contact_id='parent-c1', positions={'kiwi-1': 16}, anchors={},
                     keep=keep, reason='operator')]
    assert run('contacts', controls=controls) == 0
    finish(stage4)
    rec = one_session(stage4, 'contacts')
    row = rec.index['contacts']['parent-c1']
    assert row['positions'] == {'kiwi-1': 8}
    assert row['end_positions'] == {'kiwi-1': 16}
    assert bool(audio_files(stage4.root)) == keep
    if keep:
        lane = next(iter(rec.lanes.values()))
        assert row['lanes'][lane.name]['actual'] == [0, 52]
        assert row['lanes'][lane.name]['missing_post_roll'] == 4


@SHELL
@pytest.mark.parametrize('policy,line,expected', [
    ('everything', 'WITNESS COVERAGE: everything; retained 16 samples', 'continuous coverage'),
    ('contacts', 'WITNESS COVERAGE: contact-only; retained 1 contact(s)', 'contact-only coverage'),
    ('contacts', 'WITNESS SHORT: receiver unavailable', 'capture failed'),
])
def test_shell_witness_stop_reports_capture_coverage(stage4, shell_guard, policy, line, expected):
    source = (ROOT / 'tools/onair.sh').read_text()
    functions = source[source.index('witness_start() {'):source.index('unkey_and_exit() {')]
    (stage4.root / 'witness.log').write_text(line + '\n')
    script = f'''{functions}
SESSION={shlex.quote(str(stage4.root))}
WITNESS=1
/bin/sh -c 'exit 0' &
WITNESS_PID=$!
WITNESS_WAV="$SESSION/witness.wav"
claim_processes() {{ :; }}
recording_available() {{ return 1; }}
witness_stop
'''
    result = subprocess.run(['/bin/bash', '-c', script],
                            env=dict(shell_guard, HFMODEM_RECORDING=policy), text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout.lower()


def test_tap_capture_failure_publishes_partial_state(stage4, monkeypatch):
    tap = importlib.import_module('txwitness')
    from hfmodem.core import devices
    monkeypatch.setenv('HFMODEM_RECORDING', 'contacts')
    monkeypatch.setenv('HFMODEM_RECORDING_PROBE', '1')
    monkeypatch.setattr(tap, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(tap.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(devices, 'find_device', lambda *a, **k: 0)
    sd = sys.modules['sounddevice']
    monkeypatch.setattr(sd, 'query_devices', lambda *a: [{'name': 'synthetic'}])
    class Input:
        def __init__(self, **kwargs): pass
        def __enter__(self): raise OSError('synthetic capture failure')
        def __exit__(self, *args): pass
    monkeypatch.setattr(sd, 'InputStream', Input)
    with pytest.raises(OSError, match='synthetic capture failure'):
        tap.record(1, str(stage4.root / 'tap.wav'), 'synthetic')
    assert json.loads((stage4.root / 'tap.json').read_text())['capture_state'] == 'partial'


@SHELL
@pytest.mark.parametrize('completion,status,intentional', [
    ('Traceback: synthetic child failure', 1, False), ('', 143, False),
    ('', 0, False), ('WITNESS COVERAGE: contact-only; retained 0 interval(s)', 0, True),
    ('WITNESS COVERAGE: contact-only; retained 0 interval(s)', 1, False),
])
def test_shell_witness_requires_positive_success_for_intentional_absence(
        stage4, shell_guard, completion, status, intentional):
    source = (ROOT / 'tools/onair.sh').read_text()
    functions = source[source.index('witness_stop() {'):source.index('unkey_and_exit() {')]
    (stage4.root / 'witness.log').write_text(completion + '\n')
    script = f'''{functions}
SESSION={shlex.quote(str(stage4.root))}
WITNESS=1
WITNESS_WAV="$SESSION/witness.wav"
/bin/sh -c 'exit {status}' &
WITNESS_PID=$!
claim_processes() {{ :; }}
recording_available() {{ return 1; }}
witness_stop
'''
    result = subprocess.run(['/bin/bash', '-c', script], env=dict(shell_guard, HFMODEM_RECORDING='contacts'),
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert ('no retained contact' in result.stdout) == intentional
    assert ('capture failed' in result.stdout) != intentional


@pytest.mark.parametrize('timestamped', (False, True))
def test_tap_adc_loss_preserves_absolute_positions(stage4, monkeypatch, timestamped):
    tap = importlib.import_module('txwitness')
    from hfmodem.core import devices
    from hfmodem.tests.station.recording_support import flush
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything')
    monkeypatch.setenv('HFMODEM_RECORDING_PROBE', '1')
    monkeypatch.setattr(tap, 'FS', 8)
    monkeypatch.setattr(tap, 'device_claim', lambda *a, **k: nullcontext())
    monkeypatch.setattr(tap, 'device_opened', lambda: None)
    monkeypatch.setattr(tap.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(devices, 'find_device', lambda *a, **k: 0)
    sd = sys.modules['sounddevice']
    monkeypatch.setattr(sd, 'query_devices', lambda *a: [{'name': 'synthetic'}])
    class Input:
        def __init__(self, **kwargs): self.callback = kwargs['callback']
        def __enter__(self):
            for adc, overflow in [(100., False), (102., True)]:
                times = NS(inputBufferAdcTime=adc, currentTime=adc) if timestamped else NS()
                self.callback(np.full((4, 1), .125, np.float32), 4, times, NS(input_overflow=overflow))
                flush(stage4.sessions[0])
            return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(sd, 'InputStream', Input)
    assert tap.record(1, str(stage4.root / 'tap.wav'), 'synthetic') == 0
    lane = one_session(stage4, 'everything').lanes['tap']
    assert lane.position == (20 if timestamped else 9)
    assert lane.gaps == [[4, 16 if timestamped else 5]]
    if not timestamped:
        assert lane.row.get('uncertain')


@SHELL
def test_shell_witness_failure_remains_visible_with_retained_audio(stage4, shell_guard):
    source = (ROOT / 'tools/onair.sh').read_text()
    functions = source[source.index('witness_stop() {'):source.index('unkey_and_exit() {')]
    (stage4.root / 'witness.log').write_text('Traceback: synthetic capture failure\n')
    script = f'''{functions}
SESSION={shlex.quote(str(stage4.root))}
WITNESS=1
WITNESS_WAV="$SESSION/witness.wav"
/bin/sh -c 'exit 1' &
WITNESS_PID=$!
claim_processes() {{ :; }}
recording_available() {{ return 0; }}
timeout() {{ :; }}
witness_stop
'''
    result = subprocess.run(['/bin/bash', '-c', script], env=dict(shell_guard, HFMODEM_RECORDING='contacts'),
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'capture failed' in result.stdout
