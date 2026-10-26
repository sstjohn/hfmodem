# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import ast
import errno
import fcntl
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from hfmodem.core import claim, native
from hfmodem.core.cat import YaesuCat
from hfmodem.tests.core.fakeyaesu import PtySerial
from hfmodem.tests.hermetic import fake_claim_audio_metadata  # noqa: F401

from . import shim
from .conftest import TREE
from .ffmpeg_stub import parse_args
from .observer import (IN_CLOSE, IN_OPEN, IN_Q_OVERFLOW, Inotify, LineSampler,
                       LinuxObserver, MACOS_PTY_GAP, MACOS_TRACE_GAP, MacOSObserver,
                       ObservationError, Snapshot, TracedProcess, clock_pair, fdinfo,
                       linux_participants, macos_participants, modem_bits,
                       parse_trace, probe_claim, set_modem_bits)
from .oracle import (Evidence, OracleFailure, assert_claim_possession, assert_released, assert_untraced_bound,
                     pulse_upper_bound, read_events, validate)


def completed(process, expected=0):
    out, err = process.communicate(timeout=10)
    assert process.returncode == expected, (out, err)
    return out, err


def event(run, pid=11, **fields):
    return dict(event='claim_acquired', run=run, pid=pid, start='start',
                time_ns=1_000_000_000, monotonic_ns=1, **fields)


def trace_text(path, pid=11, *, ending=True):
    lines = [f'{pid} 1.000000 openat(AT_FDCWD, "{path}", O_RDWR) = 9<{path}>',
             f'{pid} 1.001000 close(9<{path}>) = 0']
    if ending:
        lines += [f'{pid} 1.002000 +++ exited with 0 +++']
    return '\n'.join(lines) + '\n'


def evidence(path, *, platform='linux'):
    return Evidence('run', {str(path)}, platform, events=[event('run')],
                    trace=parse_trace(trace_text(path)), tracer_returncode=0, tree_finished=True)


def required_sounddevice_api(roots):
    required = {'Stream', 'InputStream', 'OutputStream', 'RawInputStream', 'RawOutputStream',
                'rec', 'play', 'wait', 'sleep', 'stop', 'query_devices', 'query_hostapis',
                'check_input_settings', 'check_output_settings', 'default', 'PortAudioError'}
    for root in roots:
        for path in root.rglob('*.py'):
            if any((root / parent / 'pyvenv.cfg').is_file()
                   for parent in path.relative_to(root).parents):
                continue
            if 'tests' in path.parts or path.name.startswith('test_'):
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            aliases = {'sd', 'sounddevice'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    aliases.update(n.asname or n.name for n in node.names if n.name == 'sounddevice')
                elif isinstance(node, ast.ImportFrom) and node.module == 'sounddevice':
                    required.update(n.name for n in node.names)
            required.update(node.attr for node in ast.walk(tree)
                            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                            and node.value.id in aliases)
    return required


def assert_shim_api(roots):
    required = required_sounddevice_api(roots)
    assert not (required - set(dir(shim)))


def test_shim_api_ratchet():
    assert_shim_api((TREE / 'packages', TREE / 'tools'))


def test_shim_api_ratchet_ignores_virtualenvs_but_retains_private_calls(tmp_path):
    environment = tmp_path / '.venv'
    environment.mkdir()
    (environment / 'pyvenv.cfg').write_text('home = /python\n')
    (environment / 'dependency.py').write_text('sd.dtype\n')
    assert_shim_api((tmp_path,))
    instrument = tmp_path / 'private_instrument.py'
    instrument.write_text('import sounddevice as audio\naudio.FutureAudioAPI()\n')
    required = required_sounddevice_api((tmp_path,))
    assert 'dtype' not in required
    assert 'FutureAudioAPI' in required
    with pytest.raises(AssertionError, match='FutureAudioAPI'):
        assert_shim_api((tmp_path,))


@pytest.mark.parametrize('enabled', [None, '0', '1'])
def test_pth_is_gated_and_precedes_prepended_pythonpath(harness, enabled):
    rival = harness.root / 'rival'
    rival.mkdir()
    (rival / 'sounddevice.py').write_text('SOURCE = "rival"\n')
    env = dict(harness.env, PYTHONPATH=str(rival) + os.pathsep + harness.env['PYTHONPATH'])
    if enabled is None:
        env.pop('HFMODEM_COEXIST_HARNESS')
    else:
        env['HFMODEM_COEXIST_HARNESS'] = enabled
    out, _ = completed(harness.python('import sounddevice; print(sounddevice.__file__)', env=env))
    assert Path(out.decode().strip()).resolve() == (
        Path(shim.__file__).resolve() if enabled == '1' else rival / 'sounddevice.py')


@pytest.mark.parametrize('shim_path', [None, ''])
def test_pth_missing_shim_is_a_quiet_noop(harness, shim_path):
    env = dict(harness.env)
    if shim_path is None:
        env.pop('HFMODEM_COEXIST_SHIM')
    else:
        env['HFMODEM_COEXIST_SHIM'] = shim_path
    out, err = completed(harness.python(
        'import sys; assert all(isinstance(p, str) for p in sys.path); print("ready")', env=env))
    assert out.strip() == b'ready'
    assert not err


@pytest.mark.parametrize('kind', ['Stream', 'InputStream', 'OutputStream', 'RawInputStream', 'RawOutputStream'])
def test_shim_real_descriptor_lifetime_and_callbacks(harness, kind):
    seen = threading.Event()
    buffers = []

    def callback(*args):
        buffers.append(args)
        seen.set()
        raise shim.CallbackStop

    stream = getattr(shim, kind)(samplerate=48000, channels=1, callback=callback)
    fds = list(stream._fds)
    assert all(os.fstat(fd).st_ino == harness.audio.stat().st_ino for fd in fds)
    stream.start()
    assert seen.wait(2)
    stream.close()
    assert buffers and not buffers[0][-1]
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    events = harness.events()
    assert sum(e['event'] == 'device_open' for e in events) == len(fds)
    assert sum(e['event'] == 'device_close' for e in events) == len(fds)


def test_shim_blocking_streams_and_convenience_calls(harness):
    with shim.RawInputStream(dtype='int16', channels=2) as stream:
        data, overflow = stream.read(12)
        assert len(data) == 48 and not overflow
    with shim.RawOutputStream(dtype='int16') as stream:
        assert stream.write(b'\0' * 24) is False
    data = shim.rec(128, channels=2, blocking=False)
    shim.wait()
    assert data.shape == (128, 2) and not data.any()
    shim.play(np.zeros(128), blocking=True)
    shim.stop()
    shim.sleep(0)
    assert shim.query_devices(0)['name'] == 'Coexist fake audio'
    assert shim.query_hostapis(0)['devices'] == [0]
    shim.check_input_settings(channels=1, samplerate=48000)
    shim.check_output_settings(channels=2)
    with pytest.raises(shim.PortAudioError):
        shim.check_input_settings(channels=3)
    assert not shim._streams


def test_shim_refuses_ungated_or_nonfake_device(harness, monkeypatch):
    monkeypatch.setenv('HFMODEM_COEXIST_AUDIO', '/dev/null')
    with pytest.raises(shim.PortAudioError, match='existing file'):
        shim.InputStream()
    monkeypatch.delenv('HFMODEM_COEXIST_HARNESS')
    with pytest.raises(shim.PortAudioError, match='HARNESS=1'):
        shim.InputStream()


@pytest.mark.parametrize('backend,device', [('avfoundation', ':0'), ('pulse', 'default'),
                                          ('alsa', 'hw:0'), ('dshow', 'audio=fake')])
def test_ffmpeg_pipe_forms_and_logged_open(harness, backend, device):
    process = harness.spawn(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', backend,
                             '-i', device, '-ac', '1', '-ar', '48000', '-t', '.01', '-f', 's16le', '-'])
    out, _ = completed(process)
    assert out == bytes(480 * 2)
    opens = [e for e in harness.events() if e['event'] == 'device_open']
    assert len(opens) == 1 and opens[0]['pid'] == process.pid
    assert opens[0]['path'] == str(harness.audio)


def test_ffmpeg_header_precedes_frames_at_fault_barrier(harness):
    path = harness.root / 'capture.wav'
    harness.pin('ffmpeg_before_output')
    process = harness.spawn(['ffmpeg', '-y', '-f', 'alsa', '-i', 'default', '-t', '.01',
                             '-ac', '1', '-ar', '48000', '-c:a', 'pcm_s16le', str(path)])
    reached = harness.reached('ffmpeg_before_output')
    header = path.read_bytes()
    assert len(header) == 78
    assert header[36:40] == b'LIST'
    assert struct.unpack_from('<I', header, 40)[0] == 26
    assert header[44:52] == b'INFOISFT'
    assert struct.unpack_from('<I', header, 52)[0] == 13
    assert header[56:70] == b'hfmodem stub\0\0'
    assert header[70:74] == b'data'
    assert struct.unpack_from('<I', header, 74)[0] == 0
    assert struct.unpack_from('<I', header, 4)[0] == len(header) - 8
    assert next(e for e in harness.events() if e['event'] == 'ffmpeg_header')['size'] == len(header)
    assert reached['pid'] == process.pid and process.poll() is None
    harness.release('ffmpeg_before_output')
    completed(process)
    final = path.read_bytes()
    assert len(final) == 78 + 960
    assert struct.unpack_from('<I', final, 4)[0] == len(final) - 8
    assert struct.unpack_from('<I', final, 74)[0] == 960
    with wave.open(str(path), 'rb') as wav:
        assert wav.getnframes() == 480
        assert wav.readframes(480) == bytes(960)


def test_ffmpeg_delay_diagnostics_listing_and_refusal(harness):
    process = harness.spawn(['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''])
    _, err = completed(process, 1)
    assert b'AVFoundation audio devices:' in err and b'[0] Coexist fake audio' in err
    assert not harness.events()
    env = dict(harness.env, HFMODEM_COEXIST_FFMPEG_DELAY='30', HFMODEM_COEXIST_FFMPEG_STDERR='empty capture')
    process = harness.spawn(['ffmpeg', '-f', 'pulse', '-i', 'default', '-t', '0', '-'], env=env)
    deadline = time.monotonic() + 5
    while not harness.events():
        assert time.monotonic() < deadline
        time.sleep(.005)
    process.terminate()
    out, err = completed(process)
    assert not out and b'empty capture' in err
    env.pop('HFMODEM_COEXIST_HARNESS')
    completed(harness.spawn(['ffmpeg', '-f', 'alsa', '-i', 'default', '-'], env=env), 2)


def test_ffmpeg_unknown_forms_fail_closed():
    with pytest.raises(ValueError, match='unsupported'):
        parse_args(['-f', 'alsa', '-i', 'default', '-new-flag', '-'])


def test_fake_yaesu_uses_existing_cat_transport(fake_yaesu):
    driver = YaesuCat(fake_yaesu.port, 'ft891', serial_factory=PtySerial)
    try:
        driver.open()
        assert driver.freq() == 7100000
        driver.set_freq(7101000)
        assert driver.freq() == 7101000
    finally:
        driver.close()


@pytest.mark.skipif(sys.platform != 'linux', reason=MACOS_PTY_GAP)
def test_pty_kernel_exclusion(fake_yaesu):
    driver = YaesuCat(fake_yaesu.port, 'ft891', serial_factory=PtySerial)
    try:
        driver.open()
        with pytest.raises(OSError) as raised:
            fd = os.open(fake_yaesu.port, os.O_RDWR | os.O_NOCTTY)
            os.close(fd)
        assert raised.value.errno == errno.EBUSY
    finally:
        driver.close()


def test_tty0tty_validation(tty0tty_pair):
    near, far = tty0tty_pair
    watch = Inotify(tty0tty_pair)
    a = b = None
    initial = None
    try:
        a = os.open(near, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        b = os.open(far, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        initial = modem_bits(a)
        for bits in (0, termios.TIOCM_RTS, termios.TIOCM_DTR, 0):
            set_modem_bits(a, bits)
            observed = modem_bits(b)
            assert bool(observed & termios.TIOCM_CTS) == bool(bits & termios.TIOCM_RTS)
            assert bool(observed & termios.TIOCM_DSR) == bool(bits & termios.TIOCM_DTR)
        fcntl.ioctl(a, termios.TIOCEXCL)
        with pytest.raises(OSError) as raised:
            unexpected = os.open(near, os.O_RDWR | os.O_NOCTTY)
            os.close(unexpected)
        assert raised.value.errno == errno.EBUSY
    finally:
        if a is not None:
            if initial is not None:
                set_modem_bits(a, initial)
            fcntl.ioctl(a, termios.TIOCNXCL)
            os.close(a)
        if b is not None:
            os.close(b)
        events = watch.drain()
        watch.close()
    for path in tty0tty_pair:
        assert any(e['path'] == path and e['mask'] & IN_OPEN for e in events)
        assert any(e['path'] == path and e['mask'] & IN_CLOSE for e in events)
    assert LineSampler.label == 'non-HUPCL'


def test_state_namespace_and_partial_holder_retry(harness, monkeypatch):
    assert claim.state_dir() == harness.root
    claim.bootstrap()
    fd = os.open(claim.claim_path(), os.O_RDWR)
    try:
        os.pwrite(fd, b'partial', 0)
        with pytest.raises(claim.ClaimError, match='unknown'):
            claim.holder(fd, attempts=2)
        assert any(e['event'] == 'holder_retry' for e in harness.events())
    finally:
        os.close(fd)
    monkeypatch.delenv('HFMODEM_COEXIST_HARNESS')
    with pytest.raises(claim.ClaimError, match='refusing'):
        claim.state_dir()


def test_real_claim_lifetime_after_sigkill_uses_probe_not_close_log(harness):
    harness.pin('holding')
    process = harness.python('''
from hfmodem.core import claim, coexist
claim.process_started = lambda pid: 'fake-start'
claim._command = lambda pid: 'fake-process'
descriptor = claim.acquire('test', 'observer')
coexist.checkpoint('holding')
''')
    harness.reached('holding')
    path = harness.root / 'rig.claim'
    assert not probe_claim(path, quiescent=True)
    process.kill()
    completed(process, -signal.SIGKILL)
    assert not any(e['event'] in ('claim_closed', 'claim_released') for e in harness.events())
    assert probe_claim(path, quiescent=True)
    snapshot = Snapshot(clock_pair('dead'), {}, {}, claim_free=True)
    assert_released(snapshot, paths=[harness.audio], claim_path=path)


def test_probe_only_at_quiescent_barrier(tmp_path):
    path = tmp_path / 'claim'
    path.touch()
    with pytest.raises(ObservationError, match='quiescent'):
        probe_claim(path, quiescent=False)


def make_proc(root, pid, *, run='run', start=123, target=None):
    directory = root / str(pid)
    (directory / 'fd').mkdir(parents=True)
    (directory / 'fdinfo').mkdir()
    (directory / 'stat').write_text(f'{pid} (name with ) parens) S ' + '0 ' * 18 + f'{start} 0\n')
    (directory / 'status').write_text('Uid:\t' + '\t'.join([str(os.getuid())] * 4) + '\n')
    (directory / 'environ').write_bytes(f'X=1\0HFMODEM_COEXIST_RUN={run}\0'.encode())
    (directory / 'cmdline').write_bytes(b'python\0fake\0')
    if target:
        (directory / 'fd/9').symlink_to(target)
        (directory / 'fdinfo/9').write_text('pos:\t17\nlock:\t1: FLOCK ADVISORY WRITE 41 00:01:2 0 EOF\n')
    return directory


def test_linux_inventory_uses_marker_not_process_group(tmp_path):
    target = tmp_path / 'claim'
    target.touch()
    make_proc(tmp_path, 41, target=target)
    make_proc(tmp_path, 42, run='run-suffix')
    (tmp_path / 'locks').write_text('')
    assert set(linux_participants('run', tmp_path)) == {41}
    snapshot = LinuxObserver('run', proc=tmp_path).barrier('pinned', 1234)
    assert snapshot.participants[41]['start'] == '123'
    assert snapshot.descriptors[41][0]['pos'] == 17
    assert snapshot.descriptors[41][0]['locks']
    assert snapshot.clock['participant_monotonic_ns'] == 1234
    with pytest.raises(ObservationError, match='pos'):
        fdinfo('flags: 02\n')


def test_unlogged_setsid_child_is_found_in_live_linux_inventory(harness):
    if sys.platform != 'linux':
        pytest.skip('live /proc run-marker discovery requires Linux; macOS has a separate ps/lsof test')
    harness.pin('detached')
    process = harness.python('''
import os, fcntl
from hfmodem.core import coexist
fd = os.open(os.path.join(os.environ['HFMODEM_STATE_DIR'], 'rig.claim'), os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
coexist.checkpoint('detached')
''', start_new_session=True)
    harness.reached('detached')
    snapshot = LinuxObserver(harness.run_id, started_after=harness.process_start_floor).capture()
    assert process.pid in snapshot.participants
    proof = Evidence(harness.run_id, {str(harness.root / 'rig.claim')}, 'linux', snapshots=[snapshot])
    with pytest.raises(OracleFailure, match='unlogged sampled holder'):
        validate(proof)
    harness.release('detached')
    completed(process)


@pytest.mark.skipif(sys.platform != 'darwin', reason='real ps -E/lsof observation requires macOS')
def test_unlogged_raw_audio_holder_is_found_by_live_macos_observer(harness):
    harness.pin('raw_audio_held')
    process = harness.python('''
import os
from hfmodem.core import coexist
fd = os.open(os.environ['HFMODEM_COEXIST_AUDIO'], os.O_RDONLY)
try:
    coexist.checkpoint('raw_audio_held')
finally:
    os.close(fd)
''', start_new_session=True)
    reached = harness.reached('raw_audio_held')

    # Probe the real tools separately so observer/parser regressions still fail.
    def available_output(argv):
        reason = 'real macOS ps -E/lsof unavailable; conductor must rerun outside the sandbox'
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as exc:
            pytest.skip(f'{reason}: {argv[0]}: {exc}')
        if result.returncode or result.stderr.strip():
            pytest.skip(f'{reason}: {argv[0]} exited {result.returncode}: {result.stderr.strip()}')
        return result.stdout

    ps = available_output(['ps', '-E', '-axo', 'pid=,lstart=,command='])
    rows = [line for line in ps.splitlines() if line.split()[:1] == [str(process.pid)]]
    if not rows or f'HFMODEM_COEXIST_RUN={harness.run_id}' not in rows[0].split():
        pytest.skip('real macOS ps -E cannot expose the run-marked child; '
                    'conductor must rerun outside the sandbox')
    lsof = available_output(['lsof', '-nP', '-o', '-a', '-p', str(process.pid), '-FpcfDino'])
    if f'p{process.pid}' not in lsof.splitlines():
        pytest.skip('real macOS lsof cannot expose the pinned child; '
                    'conductor must rerun outside the sandbox')

    observer = MacOSObserver(harness.run_id)
    snapshot = observer.barrier('raw_audio_held', participant_monotonic_ns=reached['monotonic_ns'])
    assert process.poll() is None
    assert process.pid in snapshot.participants
    assert any(Path(fd.get('path', '')).resolve() == harness.audio.resolve()
               for fd in snapshot.descriptors[process.pid])
    proof = Evidence(harness.run_id, {str(harness.audio)}, 'darwin',
                     events=harness.events(), snapshots=[snapshot])
    assert [e['event'] for e in proof.events] == ['barrier']
    with pytest.raises(OracleFailure, match=f'unlogged sampled holder: {process.pid}'):
        validate(proof)
    harness.release('raw_audio_held')
    completed(process)


def test_macos_barriers_offsets_and_one_second_polling(tmp_path):
    calls = []
    path = tmp_path / 'claim'
    path.touch()

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'ps':
            out = ' 41 Sat Oct 17 10:00:00 2026 python HFMODEM_COEXIST_RUN=run\n'
            out += '42 Sat Oct 17 10:00:00 2026 python HFMODEM_COEXIST_RUN=run2\n'
        else:
            out = f'p41\nf9\nD0x1\ni12\no0t17\nn{path}\nf10\no0x20\nn/fake\n'
        return SimpleNamespace(returncode=0, stdout=out, stderr='')

    observer = MacOSObserver('run', runner=runner)
    assert observer.interval == 1.0
    assert observer.capture().claim_free is None
    snapshot = observer.barrier('pinned', claim_path=path, quiescent=True, participant_monotonic_ns=15)
    assert snapshot.claim_free is True
    assert set(snapshot.participants) == {41}
    assert [fd['pos'] for fd in snapshot.descriptors[41]] == [17, 32]
    assert '-o' in next(call for call in calls if call[0] == 'lsof')
    assert MACOS_TRACE_GAP in observer.trace_gap


def test_discovery_failures_are_not_empty_inventory():
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout='', stderr='sandbox denied')
    with pytest.raises(ObservationError, match='failed'):
        macos_participants('run', runner)


def test_lsof_cannot_omit_a_live_marked_process():
    def runner(argv, **kwargs):
        if argv[0] == 'ps':
            return SimpleNamespace(returncode=0, stderr='', stdout=
                                   '41 Sat Oct 17 10:00:00 2026 python HFMODEM_COEXIST_RUN=run\n')
        return SimpleNamespace(returncode=1, stdout='', stderr='')
    with pytest.raises(ObservationError, match='omitted live'):
        MacOSObserver('run', runner=runner).capture()


@pytest.mark.parametrize('code,raw,error', [(0, 'complete', None),
                                         (-9, 'complete', 'signal'),
                                         (0, 'incomplete', 'before its tree'),
                                         (0, 'truncated', 'truncated')])
def test_tracer_lifecycle_requires_whole_tree(tmp_path, monkeypatch, code, raw, error):
    from . import observer

    def popen(argv, **kwargs):
        assert '-f' in argv and '--seccomp-bpf' in argv and '-yy' in argv
        assert 'trace=%process,%desc,%file,flock,ioctl' in argv
        path = Path(argv[argv.index('-o') + 1])
        data = trace_text(tmp_path / 'device', ending=raw != 'incomplete')
        path.write_text(data[:-3] if raw == 'truncated' else data)
        return SimpleNamespace(wait=lambda timeout: code)

    monkeypatch.setattr(observer.sys, 'platform', 'linux')
    monkeypatch.setattr(observer.subprocess, 'Popen', popen)
    child = TracedProcess(['fake'], directory=tmp_path / 'trace-run', env={})
    if error:
        with pytest.raises(ObservationError, match=error):
            child.finish()
    else:
        assert child.finish().pids == {11}


def test_oracle_accepts_complete_trace_and_rejects_short_unlogged_openers(tmp_path):
    path = tmp_path / 'device'
    proof = evidence(path)
    assert validate(proof) == []
    proof.trace = parse_trace(trace_text(path) + trace_text(path, 12))
    with pytest.raises(OracleFailure, match='unlogged traced opener: 12'):
        validate(proof)
    proof.events = []
    with pytest.raises(OracleFailure, match='unlogged traced opener: 11'):
        validate(proof)


def test_oracle_rejects_marker_removed_and_unlogged_setsid_holder(tmp_path):
    proof = evidence(tmp_path / 'device')
    proof.trace = parse_trace(trace_text(tmp_path / 'device', 12))
    with pytest.raises(OracleFailure, match='unlogged traced opener'):
        validate(proof)
    proof.snapshots = [Snapshot({}, {12: dict(start='detached')},
                                {12: [dict(path=str(tmp_path / 'device'), fd=9)]})]
    with pytest.raises(OracleFailure, match='unlogged sampled holder'):
        validate(proof)


def test_mac_trace_absence_is_explicit_not_completeness_pass(tmp_path):
    proof = evidence(tmp_path / 'device', platform='darwin')
    proof.trace = None
    assert validate(proof) == [MACOS_TRACE_GAP]


def test_oracle_rejects_unclaimed_shim_open_between_barriers(harness):
    harness.pin('before_audio')
    harness.pin('after_audio')
    process = harness.python('''
import sounddevice
from hfmodem.core import coexist
coexist.checkpoint('before_audio')
with sounddevice.RawInputStream():
    pass
coexist.checkpoint('after_audio')
''')
    harness.reached('before_audio')
    harness.release('before_audio')
    harness.reached('after_audio')
    events = harness.events()
    assert [e['event'] for e in events] == ['barrier', 'device_open', 'device_close', 'barrier']
    proof = Evidence(harness.run_id, {str(harness.audio)}, 'darwin', events=events)
    with pytest.raises(OracleFailure, match=f'unregistered device opener: {process.pid}'):
        validate(proof)
    harness.release('after_audio')
    completed(process)


def test_lost_participant_event_fails_the_oracle(harness):
    harness.observe([harness.audio])
    process = harness.python('''
import os
from hfmodem.core import coexist
log = os.environ['HFMODEM_COEXIST_LOG']
os.chmod(log, 0)
try:
    coexist.fault('lost_point')
finally:
    os.chmod(log, 0o600)
''', env=dict(harness.env, HFMODEM_COEXIST_FAULT='lost_point'))
    completed(process, 1)
    proof = harness.collect()
    assert not any(e['event'] == 'fault' for e in proof.events)
    with pytest.raises(OracleFailure, match='event write was lost'):
        validate(proof)


@pytest.mark.usefixtures('fake_claim_audio_metadata')
def test_oracle_registers_real_native_child(harness):
    try:
        probe = subprocess.run(['ps', '-p', str(os.getpid()), '-o', 'lstart='],
                               capture_output=True, text=True, timeout=5)
    except PermissionError:
        pytest.skip('sandbox blocks ps required by native.spawn_native; '
                    'conductor must rerun outside the sandbox')
    if probe.returncode and any(reason in probe.stderr.lower()
                                for reason in ('operation not permitted', 'permission denied')):
        pytest.skip('sandbox blocks ps required by native.spawn_native; '
                    'conductor must rerun outside the sandbox')
    assert probe.returncode == 0 and probe.stdout.strip(), probe.stderr

    owned = claim.acquire('test', 'native-oracle')
    try:
        child = native.spawn_native([sys.executable, '-c',
            'from hfmodem.tests.coexist import shim; '
            'stream = shim.RawInputStream(); stream.close()'], claim_fd=owned)
        with child._process as process:
            completed(process)
        events = harness.events()
        spawned = next(e for e in events if e['event'] == 'native_spawn')
        assert spawned['child_pid'] == child.pid
        assert spawned['child_started'] == child.started
        assert any(e['event'] == 'device_open' and e['pid'] == child.pid
                   and e['path'] == str(harness.audio) for e in events)
        proof = Evidence(harness.run_id, {str(harness.audio)}, 'darwin', events=events,
                         snapshots=[Snapshot({}, {child.pid: dict(event_start=child.started)},
                                             {child.pid: [dict(path=str(harness.audio))]})])
        assert validate(proof) == [MACOS_TRACE_GAP]
        proof.events = [e for e in events if e is not spawned]
        with pytest.raises(OracleFailure, match=f'unregistered device opener: {child.pid}'):
            validate(proof)
    finally:
        claim.release(owned)


@pytest.mark.parametrize('registration', ['claim_acquired', 'inherited', 'native_spawn', 'native_admitted'])
def test_device_open_requires_registration_only_for_evidence_paths(tmp_path, registration):
    path = tmp_path / 'device'
    path.touch()
    alias = tmp_path / 'alias'
    alias.symlink_to(path)
    opened = dict(event='device_open', run='run', pid=12, path=str(alias))
    proof = Evidence('run', {str(path)}, 'darwin', events=[opened, event('run')])
    with pytest.raises(OracleFailure, match='unregistered device opener: 12'):
        validate(proof)
    registered = dict(event=registration, run='run', pid=12, start='start', verdict=True)
    if registration in ('native_spawn', 'native_admitted'):
        registered.update(pid=11, child_pid=12, child_started='child')
    proof.events.append(registered)
    assert validate(proof) == [MACOS_TRACE_GAP]
    proof.events.pop()
    opened['path'] = str(tmp_path / 'unwatched')
    assert validate(proof) == [MACOS_TRACE_GAP]


@pytest.mark.parametrize('failure', ['overflow', 'unfinished', 'tracer_killed', 'not_waited', 'no_trace', 'error'])
def test_observer_negative_fixtures_fail_oracle(tmp_path, failure):
    proof = evidence(tmp_path / 'device')
    if failure == 'overflow':
        proof.inotify = Inotify.decode(struct.pack('iIII', -1, IN_Q_OVERFLOW, 0, 0))
    elif failure == 'unfinished':
        proof.trace = parse_trace(trace_text(tmp_path / 'device', ending=False))
    elif failure == 'tracer_killed':
        proof.tracer_returncode = -signal.SIGKILL
    elif failure == 'not_waited':
        proof.tree_finished = False
    elif failure == 'no_trace':
        proof.trace = None
    else:
        proof.errors.append('observer could not read process metadata')
    with pytest.raises(OracleFailure):
        validate(proof)


@pytest.mark.parametrize('raw', ['', '11 1.0 openat(', 'strace: Process 11 detached\n',
                                  '11 1.0 openat(AT_FDCWD, "x", O_RDONLY <unfinished ...>\n',
                                  '11 1.0 <... openat resumed>) = 9\n'])
def test_truncated_or_unattributed_trace_refused(raw):
    with pytest.raises(ObservationError):
        parse_trace(raw)


def test_trace_resumption_forks_threads_and_recorded_native_child(tmp_path):
    path = tmp_path / 'device'
    raw = ('11 1.0 clone(child_stack=NULL, flags=CLONE_THREAD) = 12\n'
           f'12 1.1 openat(AT_FDCWD, "{path}", O_RDWR <unfinished ...>\n'
           '11 1.2 clone(child_stack=NULL, flags=SIGCHLD) = 13\n'
           f'12 1.3 <... openat resumed>) = 9<{path}>\n'
           '12 1.4 +++ exited with 0 +++\n'
           '11 1.5 +++ exited with 0 +++\n') + trace_text(path, 13)
    proof = evidence(path)
    proof.trace = parse_trace(raw)
    proof.events += [dict(event='native_spawn', pid=11, run='run', child_pid=13, child_started='child')]
    assert validate(proof) == []


@pytest.mark.parametrize('logged,simultaneous,unset', [(False, False, False), (False, True, False),
                                                     (False, False, True), (True, False, False)])
def test_linux_short_process_trace(harness, logged, simultaneous, unset):
    if sys.platform != 'linux':
        pytest.skip(MACOS_TRACE_GAP)
    if not shutil.which('strace'):
        pytest.skip('strace 6.1 is required in hfmodem-ci for process trace validation')
    worker = '''
import os
from hfmodem.core import coexist
if LOGGED:
    coexist.event('claim_acquired', start='fake')
if UNSET:
    os.unsetenv('HFMODEM_COEXIST_RUN')
fd = os.open(os.environ['HFMODEM_COEXIST_AUDIO'], os.O_RDONLY)
os.close(fd)
'''.replace('LOGGED', repr(logged)).replace('UNSET', repr(unset))
    code = worker
    if simultaneous:
        logged_worker = worker.replace('if False:', 'if True:', 1)
        code = f'''import subprocess, sys
a = subprocess.Popen([sys.executable, '-c', {worker!r}])
b = subprocess.Popen([sys.executable, '-c', {logged_worker!r}])
assert a.wait() == b.wait() == 0
'''
    traced = TracedProcess([sys.executable, '-c', code], directory=harness.root / 'trace-run', env=harness.env)
    harness.children.append(traced.process)
    try:
        trace = traced.finish()
    except ObservationError:
        diagnostic = (harness.root / 'trace-run/stderr').read_text()
        if 'Operation not permitted' in diagnostic or 'Permission denied' in diagnostic:
            pytest.skip('sandbox blocks ptrace; conductor must rerun Linux trace rows outside')
        raise
    proof = Evidence(harness.run_id, {str(harness.audio)}, 'linux', events=harness.events(),
                     trace=trace, tracer_returncode=traced.process.returncode, tree_finished=True)
    if logged:
        assert validate(proof) == []
    else:
        with pytest.raises(OracleFailure, match='unlogged traced opener'):
            validate(proof)


def test_inotify_live_regular_fake_device(harness):
    if sys.platform != 'linux':
        pytest.skip('inotify is Linux-only; macOS records lsof at barriers')
    watch = Inotify([harness.audio])
    try:
        fd = os.open(harness.audio, os.O_RDONLY)
        os.close(fd)
        events = watch.drain()
        assert any(e['mask'] & IN_OPEN for e in events)
        assert any(e['mask'] & IN_CLOSE for e in events)
    finally:
        watch.close()


def test_release_uses_kernel_and_descriptors_not_participant_close(tmp_path):
    path = tmp_path / 'claim'
    path.touch()
    snapshot = Snapshot({}, {}, {}, locks='')
    assert_released(snapshot, paths=[tmp_path / 'device'], claim_path=path)
    info = path.stat()
    snapshot.locks = f'1: FLOCK ADVISORY WRITE -1 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n'
    with pytest.raises(OracleFailure, match='still locked'):
        assert_released(snapshot, paths=[], claim_path=path)
    snapshot.locks = ''
    snapshot.descriptors = {11: [dict(path=str(tmp_path / 'device'))]}
    with pytest.raises(OracleFailure, match='surviving'):
        assert_released(snapshot, paths=[tmp_path / 'device'], claim_path=path)
    snapshot.descriptors = {}
    with pytest.raises(OracleFailure, match='confirmed low'):
        assert_released(snapshot, paths=[], claim_path=path, require_line=True)


def test_forged_nonce_without_kernel_lock_is_refused(tmp_path):
    path = tmp_path / 'claim'
    descriptor = dict(path=str(path), pos=17, locks=[])
    snapshot = Snapshot({}, {11: dict(start='start')}, {11: [descriptor]}, locks='')
    with pytest.raises(OracleFailure, match='does not hold'):
        assert_claim_possession(snapshot, pid=11, claim_path=path, nonce=17)
    descriptor['locks'] = ['1: FLOCK ADVISORY WRITE 11 00:01:2 0 EOF']
    assert_claim_possession(snapshot, pid=11, claim_path=path, nonce=17)
    with pytest.raises(OracleFailure, match='nonce'):
        assert_claim_possession(snapshot, pid=11, claim_path=path, nonce=18)


def test_sampler_records_read_window_and_unknown_is_not_low():
    seen = threading.Event()

    def sense():
        seen.set()
        return None

    sampler = LineSampler(sense).start()
    assert seen.wait(2)
    samples = sampler.stop()
    assert samples and samples[0][1] >= samples[0][0] and samples[0][2] is None


def samples(levels, gap=1_000_000):
    return [(index * gap, index * gap + 100, level) for index, level in enumerate(levels)]


def test_pulse_upper_bounds_include_brackets_and_asymmetry():
    data = samples([False, True, True, False])
    assert pulse_upper_bound(data, asymmetry_ns=1000, shortest_pulse_ns=1_000_000) == 3_001_100
    with pytest.raises(OracleFailure, match='exceeds'):
        pulse_upper_bound(data, limit_ns=3_000_000, asymmetry_ns=0, shortest_pulse_ns=1_000_000)
    lows = samples([False] * 3)
    assert pulse_upper_bound(lows, blindness_ns=2000, reporting_delay_ns=3000,
                             shortest_pulse_ns=1000) == 1_005_100


@pytest.mark.parametrize('levels', [[False, True, False], [False, False, False]])
def test_pulse_resolution_is_required_even_when_high_is_detected(levels):
    with pytest.raises(OracleFailure, match='witness resolution is insufficient'):
        pulse_upper_bound(samples(levels), asymmetry_ns=0, blindness_ns=0, reporting_delay_ns=0)


@pytest.mark.parametrize('case', ['gap', 'unknown', 'delay', 'filtered', 'lowered', 'unbracketed'])
def test_witness_negative_fixtures_are_unresolved(case):
    data = samples([False, True, False])
    kwargs = dict(asymmetry_ns=0, shortest_pulse_ns=1_000_000)
    if case == 'gap':
        data = samples([False, True, False], gap=2_000_001)
    elif case == 'unknown':
        data = samples([False, None, False])
    elif case == 'delay':
        kwargs['asymmetry_ns'] = None
    elif case == 'filtered':
        data = samples([False] * 3)
    elif case == 'lowered':
        kwargs['limit_ns'] = 500_000
    else:
        data = samples([True, False])
    with pytest.raises(OracleFailure):
        pulse_upper_bound(data, **kwargs)


def test_tight_bounds_are_never_from_trace():
    with pytest.raises(OracleFailure, match='untraced'):
        assert_untraced_bound(traced=True, bound_ns=20_000_000)
    assert_untraced_bound(traced=False, bound_ns=20_000_000)


def test_partial_event_log_is_not_accepted(tmp_path):
    path = tmp_path / 'log'
    path.write_text('{"pid": 1}')
    with pytest.raises(OracleFailure, match='truncated'):
        read_events(path)
