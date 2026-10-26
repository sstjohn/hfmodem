# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
import ctypes
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import termios
import textwrap
from types import SimpleNamespace

import pytest

from hfmodem.core.native import KILL_WAIT_S, REAP_GRACE_S

from .conftest import TREE
from .observer import MACOS_TRACE_GAP, MacOSObserver, modem_bits, probe_claim, set_modem_bits
from .oracle import OracleFailure, assert_claim_possession, assert_released, validate


METADATA = '''
import os
from hfmodem.core import claim, coexist
if os.environ.get('ROW_FAKE_METADATA') == '1':
    claim.process_started = lambda pid: 'Sat_Oct_17_12:00:00_2026'
    claim._command = lambda pid: 'coexist-participant'
'''


@pytest.fixture
def proof_line(harness, tty0tty_pair):
    near, far = tty0tty_pair
    fd = os.open(near, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    witness = os.open(far, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        for bits in (0, termios.TIOCM_RTS, 0):
            set_modem_bits(fd, bits)
            assert bool(modem_bits(witness) & termios.TIOCM_CTS) == bool(bits)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        with pytest.raises(OSError):
            unexpected = os.open(near, os.O_RDWR | os.O_NOCTTY)
            os.close(unexpected)
    finally:
        fcntl.ioctl(fd, termios.TIOCNXCL)
        os.close(fd)
    harness.env['ROW_PTT'] = near
    try:
        yield near, witness
    finally:
        os.close(witness)


def line_low(proof_line):
    return not bool(modem_bits(proof_line[1]) & termios.TIOCM_CTS)


HARNESS_KEYS = ('HFMODEM_COEXIST_', 'HFMODEM_STATE_DIR', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')


def launch(harness, source, **kwargs):
    return command(harness, [sys.executable, '-c', METADATA + textwrap.dedent(source)], **kwargs)


def shell_argv(argv, env):
    if argv[0] in ('bash', '/bin/bash') or str(argv[0]).endswith('.sh'):
        # hermetic.sh resets the environment to its allowlist. Restore what the
        # harness and the row set, never what this process inherited: its tokens
        # and agent sockets would reach every row shell and its failure output.
        restored = '\n'.join('export ' + key + '=' + shlex.quote(value)
                             for key, value in env.items()
                             if key != 'PATH' and not key.startswith('HERMETIC_')
                             and (key == 'HFMODEM_QUALIFICATION_PHASE'
                                  or key.startswith(HARNESS_KEYS) or os.environ.get(key) != value))
        script = (f'source {shlex.quote(str(TREE / "tools/tests/hermetic.sh"))}\n'
                  + restored + '\nexport PATH=' + shlex.quote(env['HAMLIB_BIN'])
                  + ':"$PATH"\n' + shlex.join(argv) + '\n')
        argv = ['/bin/bash', '-c', script]
    return argv


def command(harness, argv, **kwargs):
    argv = shell_argv(argv, kwargs.get('env', harness.env))
    if harness.observer is not None:
        snapshot = harness.observer.capture('before-launch')
        (harness.root / 'observer-fixture.json').write_text(json.dumps(dict(
            participants=snapshot.participants, descriptors=snapshot.descriptors)))
    child = harness.launch(argv, **kwargs)
    if sys.platform == 'linux':
        child.row_output = harness.root / f'trace-{len(harness.traced) - 1}'
    return child


def completed(child, code=0, *, targets=0):
    out, err = child.communicate(timeout=15 + targets * (REAP_GRACE_S + KILL_WAIT_S))
    if hasattr(child, 'row_output'):
        out = (child.row_output / 'stdout').read_bytes()
        err = (child.row_output / 'stderr').read_bytes()
    if child.returncode == 98 and b'no free ports for the test' in err and b'PermissionError' in err:
        pytest.skip('sandbox blocks hermetic.sh loopback allocation; conductor reruns outside')
    assert child.returncode == code, (out.decode(), err.decode())
    return out.decode(), err.decode()


def observe(harness, paths=()):
    (harness.root / 'rig.claim').touch(exist_ok=True)
    (harness.root / 'rig.claim.cleanup').touch(exist_ok=True)
    home = harness.root / 'home'
    home.mkdir(exist_ok=True)
    harness.env.update(HOME=str(home), CAT_PORT=str(harness.root / 'absent-cat'),
                       PTT_PORT=str(harness.root / 'absent-ptt'),
                       HFMODEM_DATA_DIR=str(harness.root / 'operator-data'),
                       HFMODEM_PYTHON=sys.executable, HAMLIB_BIN=str(harness.root / 'bin'))
    harness.env['PYTHONPATH'] += os.pathsep + str(TREE / 'tools')
    guard = TREE / 'packages/hfmodem/hfmodem/tests/hermetic_py'
    harness.env['PYTHONPATH'] = str(guard) + os.pathsep + harness.env['PYTHONPATH']
    for key in ('RIG_CLAIMED', 'HFMODEM_STATION', 'HFMODEM_RIG'):
        harness.env.pop(key, None)
    for name in ('rigctl', 'rigctld', 'timeout', 'nc', 'pgrep', 'pkill', 'killall'):
        binary(harness, name, 'raise SystemExit("unexpected external binary")')
    binary(harness, 'lsof', '''
        import json, os, sys
        from pathlib import Path
        path = Path(os.environ['HFMODEM_COEXIST_LOG']).with_name('observer-fixture.json')
        snapshot = json.loads(path.read_text()) if path.exists() else dict(participants={}, descriptors={})
        if sys.platform == 'linux':
            from hfmodem.tests.coexist.fixtures.live_openers import linux_openers
            rows = linux_openers(snapshot['participants'], snapshot['descriptors'])
        else:
            rows = snapshot['descriptors']
        for pid, descriptors in rows.items():
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                continue
            for fd in descriptors:
                if fd.get('path') in sys.argv:
                    print('p' + pid + '\\ncfixture-participant\\nf' + str(fd['fd'])
                          + '\\no0t' + str(fd.get('pos', 0)))
    ''')
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(['ps', '-E', '-axo', 'pid=,lstart=,command='],
                                    capture_output=True, text=True, timeout=5)
        except PermissionError:
            result = SimpleNamespace(returncode=1)
        if result.returncode:
            probe = subprocess.run(['lsof', '-nP', '-o', '-a', '-p', str(os.getpid()),
                                    '-FpcfDino'], capture_output=True, text=True, timeout=5)
            if probe.returncode or probe.stderr.strip():
                pytest.skip('sandbox blocks independent lsof observation; conductor reruns outside')
            harness.env['ROW_FAKE_METADATA'] = '1'
            libproc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            binary(harness, 'ps', '''
                import ctypes, os, sys
                pid = int(sys.argv[sys.argv.index('-p') + 1])
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    sys.exit(1)
                info = (ctypes.c_uint32 * 34)()
                lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
                size = lib.proc_pidinfo(pid, 3, 0, info, ctypes.sizeof(info))
                state = 'Z' if size == ctypes.sizeof(info) and info[1] == 5 else 'S'
                print('Sat Oct 17 12:00:00 2026' if 'lstart=' in sys.argv
                      else state if 'stat=' in sys.argv else 'coexist-participant')
            ''')

            def runner(argv, **kwargs):
                if argv[0] != 'ps':
                    return subprocess.run(argv, **kwargs)
                pids = {child.pid for child in harness.children if child.poll() is None}
                for event in harness.events():
                    pids.add(event['pid'])
                    if 'child_pid' in event:
                        pids.add(event['child_pid'])
                live = set()
                for pid in pids:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        continue
                    info = (ctypes.c_uint32 * 34)()
                    size = libproc.proc_pidinfo(pid, 3, 0, info, ctypes.sizeof(info))
                    if size == ctypes.sizeof(info) and info[1] == 5:
                        continue
                    ctypes.set_errno(0)
                    descriptors = libproc.proc_pidinfo(pid, 1, 0, None, 0)
                    if descriptors == 0 and ctypes.get_errno() in (0, 3):
                        continue
                    live.add(pid)
                rows = [f'{pid} Sat Oct 17 12:00:00 2026 coexist-participant '
                        f'HFMODEM_COEXIST_RUN={harness.run_id}'
                        for pid in live]
                return SimpleNamespace(returncode=0, stdout='\n'.join(rows), stderr='')

            harness.paths = set(map(str, [harness.audio, *paths]))
            harness.observer = MacOSObserver(harness.run_id, runner=runner)
            harness.observer.start()
            return
    harness.observe([harness.audio, *paths])


def binary(harness, name, source):
    path = harness.root / 'bin' / name
    path.write_text('#!' + sys.executable + '\n' + textwrap.dedent(source))
    path.chmod(0o700)
    return path


def held(harness, source="owned = claim.acquire('legacy', 'row')", point='held'):
    harness.pin(point)
    child = launch(harness, source + f'\ncoexist.checkpoint({point!r})\nclaim.release(owned)')
    event = harness.reached(point, quiescent=True)
    nonce = int((harness.root / 'rig.claim').read_text().split()[4])
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'],
                            claim_path=harness.root / 'rig.claim', nonce=nonce)
    return child


def finish(harness):
    evidence = harness.collect()
    assert validate(evidence) == ([] if sys.platform == 'linux' else [MACOS_TRACE_GAP])
    snapshot = evidence.snapshots[-1]
    path = harness.root / 'rig.claim'
    if path.exists():
        if sys.platform != 'linux':
            snapshot.claim_free = probe_claim(path, quiescent=True)
        assert_released(snapshot, paths=evidence.paths, claim_path=path)
    return evidence


def _assert_p1_cli_ratchet(harness, entry, args):
    observe(harness)
    target = (['-m', entry.removeprefix('module:')] if entry.startswith('module:')
              else [str(TREE / entry)])
    child = command(harness, [sys.executable, *target, '--mycall', 'K1ABC', *args])
    out, err = completed(child, 2)
    assert 'usage:' in err.lower(), (out, err)
    assert args[0] in err
    evidence = finish(harness)
    assert not any(e['event'] in ('device_open', 'claim_acquired', 'native_spawn')
                   for e in evidence.events)


@pytest.mark.parametrize('entry,args', [
    ('module:hfmodem.shrike.onair', ['--ptt-type', 'RIG']),
    ('tools/onair_session.py', ['--no-line-ptt']),
])
def test_p1_cli_ratchet(harness, entry, args):
    _assert_p1_cli_ratchet(harness, entry, args)


def test_a_row_shell_carries_nothing_this_process_inherited(harness, monkeypatch):
    secrets = {('CLA' + 'UDE_CODE_ROW_SENTINEL'): 'sentinel-token', 'SSH_AUTH_SOCK': '/nonexistent/agent'}
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    env = dict(harness.env, **secrets, CAT_PORT=str(harness.root / 'row-cat'),
               HAMLIB_BIN=str(harness.root / 'bin'))
    out, _ = completed(harness.spawn(shell_argv(['/bin/bash', '-c', 'env -0'], env)))
    shell = dict(entry.split('=', 1) for entry in out.split('\0') if entry)
    allowed = {'HOME', 'USER', 'LOGNAME', 'SHELL', 'TERM', 'TMPDIR', 'LANG', 'TZ', 'PATH',
               'PWD', 'SHLVL', '_'}
    leaked = {key for key, value in shell.items() if os.environ.get(key) == value
              and key not in allowed | {'HFMODEM_QUALIFICATION_PHASE'}
              and not key.startswith(('LC_', *HARNESS_KEYS))}
    assert not leaked and not any(key.startswith('CLA' + 'UDE_') for key in shell), sorted(leaked)
    assert shell['CAT_PORT'] == env['CAT_PORT']
    assert shell['HFMODEM_STATE_DIR'] == env['HFMODEM_STATE_DIR']


def _assert_no_removed_ptt_implementations(paths):
    for path in paths:
        tree = ast.parse(path.read_text())
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        assert not names.intersection({'OneShotRigctl', '_RigctldPtt'}), path


def test_p1_cli_ratchet_removed_implementations():
    paths = [TREE / 'packages/hfmodem/hfmodem/core/ptt.py',
             TREE / 'tools/onair_session.py']
    _assert_no_removed_ptt_implementations(paths)
    source = paths[0].read_text()
    assert 'class Keyer:' in source
    assert 'self.transport' in source
    assert 'subprocess' not in source


@pytest.mark.parametrize('point', [
    'after_mutex', 'before_claim_lock', 'after_claim_flock', 'before_holder_line',
    'after_holder_line', 'after_device_check',
])
@pytest.mark.parametrize('failure', ['raise', 'kill'])
def test_inherited_descriptor_proven_acquisition_unwind(harness, point, failure):
    observe(harness)
    env = dict(harness.env, HFMODEM_COEXIST_FAULT=f'{point}:{failure}')
    source = '''
        claim.acquire('legacy', 'acquisition-fault')
        raise AssertionError('fault did not fire')
    '''
    expected = -signal.SIGKILL if failure == 'kill' else 1
    if sys.platform == 'linux':
        source = f'''
            import subprocess, sys
            child = subprocess.run([sys.executable, '-c',
                                    {METADATA + textwrap.dedent(source)!r}])
            assert child.returncode == {expected}
        '''
        expected = 0
    child = launch(harness, source, env=env)
    completed(child, expected)
    events = harness.events()
    assert any(e['event'] == 'fault' and e['point'] == point for e in events)
    assert not any(e['event'] == 'device_open' for e in events)
    successor = launch(harness, '''
        owned = claim.acquire('legacy', 'successor')
        claim.release(owned)
    ''')
    completed(successor)
    assert (harness.root / 'rig.claim').read_bytes() == b''
    finish(harness)


@pytest.mark.parametrize('nonce', [1, 2**40])
def test_inherited_descriptor_proven_nonce_limits(harness, nonce):
    observe(harness)
    harness.pin('nonce-held')
    child = launch(harness, f'''
        claim.secrets.randbelow = lambda limit: {nonce - 1}
        owned = claim.acquire('legacy', 'nonce-limit')
        assert owned.offset() == {nonce}
        assert claim.inherited(owned).offset() == {nonce}
        coexist.checkpoint('nonce-held')
        claim.release(owned)
    ''')
    reached = harness.reached('nonce-held', quiescent=True)
    snapshot = harness.observer.snapshots[-1]
    assert_claim_possession(snapshot, pid=reached['pid'],
                            claim_path=harness.root / 'rig.claim', nonce=nonce)
    harness.release('nonce-held')
    completed(child)
    finish(harness)


@pytest.mark.parametrize('case', [
    'inherited', 'dead-root', 'reopened-live', 'reopened-dead', 'released-parent',
    'free-file', 'forged-environment', 'missing-fd', 'publication-failed',
])
def test_inherited_descriptor_proven(harness, case):
    observe(harness)
    harness.pin('descriptor-checked')
    source = '''
        import subprocess, sys
        owned = claim.acquire('legacy', 'descriptor-parent')
        descriptor = owned.fd
        if CASE in ('reopened-live', 'reopened-dead', 'publication-failed',
                    'free-file', 'forged-environment'):
            descriptor = os.open(claim.claim_path(), os.O_RDWR)
        if CASE == 'free-file':
            claim.release(owned)
        env = dict(os.environ, RIG_CLAIMED=str(claim.claim_path()))
        child_source = METADATA + ''' + repr(textwrap.dedent('''
            import sys
            fd, case = int(sys.argv[1]), sys.argv[2]
            if case in ('dead-root', 'reopened-dead', 'publication-failed', 'released-parent'):
                sys.stdin.buffer.read(1)
            try:
                inherited = claim.inherited(fd)
            except (OSError, claim.ClaimError) as exc:
                assert case not in ('inherited', 'dead-root'), str(exc)
                print('refused', flush=True)
            else:
                assert case in ('inherited', 'dead-root'), case
                print(inherited.offset(), flush=True)
            coexist.checkpoint('descriptor-checked')
        ''')) + '''
        passed = () if CASE == 'missing-fd' else (descriptor,)
        child = subprocess.Popen([sys.executable, '-c', child_source, str(descriptor), CASE],
                                 pass_fds=passed, stdin=subprocess.PIPE, env=env)
        if CASE == 'released-parent':
            claim.release(owned)
            child.stdin.close()
        if CASE in ('dead-root', 'reopened-dead', 'publication-failed'):
            claim.close(owned)
            child.stdin.close()
        assert child.wait(timeout=10) == 0
        if CASE not in ('dead-root', 'reopened-dead', 'publication-failed',
                        'released-parent', 'free-file'):
            claim.release(owned)
        if descriptor != owned.fd:
            os.close(descriptor)
    '''
    # The dead publisher must really die; its inheritor owns the last description.
    if case in ('dead-root', 'reopened-dead'):
        source = source.replace('claim.close(owned)\n            child.stdin.close()',
                                'os._exit(0)')
    child = launch(harness, 'CASE = ' + repr(case) + '\nMETADATA = ' + repr(METADATA)
                   + '\n' + textwrap.dedent(source))
    event = harness.reached('descriptor-checked', quiescent=True)
    snapshot = harness.observer.snapshots[-1]
    path = harness.root / 'rig.claim'
    if case in ('inherited', 'dead-root'):
        assert_claim_possession(snapshot, pid=event['pid'], claim_path=path,
                                nonce=int(path.read_text().split()[4]))
    else:
        assert any(e['event'] == 'inherited' and e['verdict'] is False
                   and e['pid'] == event['pid'] for e in harness.events())
    harness.release('descriptor-checked')
    completed(child)
    finish(harness)


@pytest.mark.parametrize('point', ['inherited_after_inode', 'after_launch_validation',
                                  'before_acknowledgement', 'after_acknowledgement'])
@pytest.mark.parametrize('failure', ['raise', 'kill'])
def test_inherited_descriptor_proven_launch_faults(harness, point, failure):
    observe(harness)
    child_code = METADATA + textwrap.dedent('''
        admission = claim.inherited(int(os.environ['ROW_FD']), launch=True)
        admission.opened()
        raise AssertionError('fault did not fire')
    ''')
    expected = -signal.SIGKILL if failure == 'kill' else 1
    child = launch(harness, f'''
        import subprocess, sys
        owned = claim.acquire('legacy', 'launch-fault')
        env = dict(os.environ, ROW_FD=str(owned.fd), HFMODEM_COEXIST_FAULT={point + ':' + failure!r})
        launched = subprocess.run([sys.executable, '-c', {child_code!r}],
                                  pass_fds=(owned.fd,), env=env)
        assert launched.returncode == {expected}
        mutex = claim.mutex(timeout=0)
        claim.mutex_release(mutex)
        claim.release(owned)
    ''')
    completed(child)
    evidence = finish(harness)
    assert any(e['event'] == 'fault' and e['point'] == point for e in evidence.events)
    assert not any(e['event'] == 'device_open' for e in evidence.events)


@pytest.mark.parametrize('case', ['reject-seek', 'after-publication'])
def test_inherited_descriptor_proven_failed_publication(harness, case):
    observe(harness)
    child = launch(harness, f'''
        real_seek = claim.os.lseek
        def seek(fd, offset, whence):
            if offset and whence == os.SEEK_SET:
                raise OSError('injected rejected nonce')
            return real_seek(fd, offset, whence)
        if {case!r} == 'reject-seek':
            claim.os.lseek = seek
        else:
            os.environ['HFMODEM_COEXIST_FAULT'] = 'after_holder_line'
        try:
            claim.acquire('legacy', 'rejected-publication')
        except (OSError, RuntimeError):
            pass
        else:
            raise AssertionError('acquisition succeeded')
        assert claim.claim_path().read_bytes() == b''
        claim.os.lseek = real_seek
        os.environ.pop('HFMODEM_COEXIST_FAULT', None)
        owned = claim.acquire('legacy', 'successor')
        claim.release(owned)
    ''')
    completed(child)
    finish(harness)


@pytest.mark.parametrize('case', ['shared-admission', 'two-cleanups', 'paused-cleanup'])
def test_cleanup_mutex_is_atomic(harness, case):
    observe(harness)
    point = 'after_mutex' if case == 'shared-admission' else 'mutex-held'
    harness.pin(point)
    owner = launch(harness, ('''
        owned = claim.acquire('legacy', 'admitted')
        claim.release(owned)
    ''' if case == 'shared-admission' else '''
        mutex = claim.mutex(timeout=0)
        coexist.checkpoint('mutex-held')
        claim.mutex_release(mutex)
    '''))
    harness.reached(point, quiescent=True)
    contender = launch(harness, f'''
        try:
            {'claim.acquire("station", "own")' if case == 'paused-cleanup' else 'claim.mutex(timeout=0)'}
        except claim.ClaimBusy as exc:
            print(str(exc), flush=True)
        else:
            raise AssertionError('contender admitted')
    ''')
    out, _ = completed(contender)
    assert ('admission in progress' if case == 'shared-admission'
            else 'cleanup in progress') in out
    assert not any(e['event'] == 'device_open' for e in harness.events())
    harness.release(point)
    completed(owner)
    finish(harness)


@pytest.mark.parametrize('point', ['after_mutex', 'after_claim_flock', 'after_device_check',
                                  'before_release', 'after_release'])
def test_cleanup_acquires_under_own_mutex(harness, point):
    observe(harness)
    harness.pin(point)
    owner = launch(harness, '''
        mutex = claim.mutex(timeout=0)
        owned = claim.acquire('cleanup', 'down', cleanup=mutex)
        claim.release(owned)
        claim.mutex_release(mutex)
    ''')
    harness.reached(point, quiescent=True)
    probe = launch(harness, '''
        import fcntl
        fd = os.open(claim.mutex_path(), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('cleanup lost exclusive authority')
        os.close(fd)
    ''')
    completed(probe)
    harness.release(point)
    completed(owner)
    evidence = finish(harness)
    assert not any(e['event'] == 'admission_acquired' for e in evidence.events)


@pytest.mark.parametrize('kind', ['one-shot', 'daemon', 'interactive', 'direct-serial', 'audio'])
def test_native_spawn_admission_during_cleanup(harness, kind):
    observe(harness)
    child = launch(harness, f'''
        import sys
        from hfmodem.core.native import spawn_native
        owned = claim.acquire('legacy', 'dead-root-modem')
        mutex = claim.mutex(timeout=0)
        try:
            spawn_native(['ffmpeg', '-f', 'pulse', '-i', 'default', '-']
                         if {kind!r} == 'audio' else [sys.executable, '-c', 'raise SystemExit(99)'],
                         claim_fd=owned, long_lived={kind not in ('one-shot', 'audio')!r},
                         device=os.environ['HFMODEM_COEXIST_AUDIO'])
        except claim.ClaimBusy as exc:
            assert 'cleanup in progress' in str(exc)
        else:
            raise AssertionError('spawned during cleanup')
        claim.mutex_release(mutex)
        claim.release(owned)
    ''')
    completed(child)
    evidence = finish(harness)
    assert not any(e['event'] in ('native_spawn', 'device_open') for e in evidence.events)


@pytest.mark.parametrize('backend,output', [('avfoundation', 'file'), ('pulse', 'pipe'),
                                          ('alsa', 'pipe')])
@pytest.mark.parametrize('point', ['ffmpeg_before_open', 'ffmpeg_before_output'])
def test_native_audio_admission(harness, backend, output, point):
    observe(harness)
    harness.pin(point)
    harness.pin('capture-done')
    target = str(harness.root / 'capture.wav') if output == 'file' else '-'
    child = launch(harness, f'''
        from hfmodem.core import native
        owned = claim.acquire('legacy', 'capture')
        argv = ['ffmpeg', '-f', {backend!r}, '-i', 'default', '-ac', '1', '-ar', '48000',
                '-t', '.1', '-f', {'wav' if output == 'file' else 's16le'!r}, {target!r}]
        capture = native.spawn_native(argv, claim_fd=owned)
        data = capture.stdout.read()
        assert capture.wait_gone(5)
        assert {output == 'file'!r} or data == bytes(4800 * 2), len(data)
        for stream in (capture.stdin, capture.stdout, capture.stderr):
            stream.close()
        coexist.checkpoint('capture-done')
        claim.release(owned)
    ''')
    event = harness.reached(point, quiescent=True)
    path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                            nonce=int(path.read_text().split()[4]))
    contender = launch(harness, '''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            assert 'admission in progress' in str(exc)
        else:
            raise AssertionError('cleanup overtook unopened capture')
    ''')
    completed(contender)
    assert not any(e['event'] == 'native_admitted' for e in harness.events())
    harness.release(point)
    harness.reached('capture-done', quiescent=True)
    if output == 'file':
        import wave
        with wave.open(target) as wav:
            assert wav.getnframes() == 4800
            assert wav.readframes(4800) == bytes(9600)
    harness.release('capture-done')
    completed(child)
    evidence = finish(harness)
    assert any(e['event'] == 'device_open' and e['pid'] == event['pid'] for e in evidence.events)


@pytest.mark.parametrize('case', ['partial', 'stale'])
def test_native_diagnostic_records_never_admit(harness, case):
    observe(harness)
    directory = harness.root / 'rig.claim.native.d'
    directory.mkdir()
    record = directory / '999999-dead'
    record.write_text('{' if case == 'partial' else json.dumps(
        dict(pid=999999, started='dead', child_pid=999998, child_started='dead')))
    harness.pin('diagnostic-held')
    owner = launch(harness, '''
        mutex = claim.mutex(timeout=0)
        coexist.checkpoint('diagnostic-held')
        claim.mutex_release(mutex)
    ''')
    harness.reached('diagnostic-held', quiescent=True)
    contender = launch(harness, '''
        try:
            claim.acquire('legacy', 'diagnostic-contender')
        except claim.ClaimBusy as exc:
            assert 'cleanup in progress' in str(exc)
            assert '999999' not in str(exc) and '999998' not in str(exc)
        else:
            raise AssertionError('diagnostic record admitted contender')
    ''')
    completed(contender)
    harness.release('diagnostic-held')
    completed(owner)
    finish(harness)


@pytest.mark.proof
@pytest.mark.parametrize('kind,death', [
    ('one-shot', 'spawner'),
    *((kind, death) for kind in ('daemon', 'interactive', 'direct-serial')
      for death in ('spawner', 'supervisor', 'both')),
])
def test_native_spawn_admission_surviving_child(harness, proof_line, fake_yaesu, kind, death):
    observe(harness, [proof_line[0], fake_yaesu.port])
    for point in ('native-child-before-open', 'native-child-open', 'native-owner-finished'):
        harness.pin(point)
    executable = binary(harness, 'rigctl' if kind == 'one-shot' else 'rigctld', f'''
        import os
        from hfmodem.core import claim, coexist
        claim.inherited(9)
        coexist.checkpoint('native-child-before-open')
        fd = os.open({fake_yaesu.port!r}, os.O_RDWR | os.O_NOCTTY)
        coexist.event('device_open', path={fake_yaesu.port!r}, fd=fd)
        coexist.checkpoint('native-child-open')
        os.close(fd)
        coexist.event('device_close', path={fake_yaesu.port!r}, fd=fd)
    ''')
    inner = METADATA + textwrap.dedent(f'''
        from hfmodem.core import native
        owned = claim.acquire('legacy', 'native-{kind}', prove_line=True,
                              ptt_factory=claim.line_factory(os.environ['ROW_PTT']))
        coexist.event('row_owner')
        try:
            child = native.spawn_native([{str(executable)!r}], claim_fd=owned,
                                        long_lived={kind != 'one-shot'!r}, device={fake_yaesu.port!r})
            assert child.wait_gone(10)
        except native.NativeSpawnError as exc:
            print(str(exc), flush=True)
        finally:
            claim.close(owned)
    ''')
    guardian = launch(harness, f'''
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', {inner!r}])
        code = child.wait(timeout=20)
        assert code in (0, -9), code
        coexist.checkpoint('native-owner-finished')
    ''')
    event = harness.reached('native-child-before-open', quiescent=True)
    events = harness.events()
    owner_pid = next(e['pid'] for e in events if e['event'] == 'row_owner')
    supervisor = next((e['pid'] for e in events if e['event'] == 'native_supervisor'), None)
    try:
        if death in ('spawner', 'both'):
            os.kill(owner_pid, signal.SIGKILL)
        if death in ('supervisor', 'both'):
            os.kill(supervisor, signal.SIGKILL)
        refused = launch(harness, '''
            try:
                claim.mutex(timeout=0)
            except claim.ClaimBusy as exc:
                assert 'admission in progress' in str(exc), str(exc)
            else:
                raise AssertionError('cleanup overtook native first open')
        ''')
        completed(refused)
        harness.release('native-child-before-open')
        harness.reached('native-child-open', quiescent=True)
        path = harness.root / 'rig.claim'
        assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                                nonce=int(path.read_text().split()[4]))
        assert any(fd['path'] == fake_yaesu.port
                   for fd in harness.observer.snapshots[-1].descriptors[event['pid']])
        assert line_low(proof_line)
    finally:
        harness.release('native-child-before-open')
        harness.release('native-child-open')
        harness.release('native-owner-finished')
    completed(guardian)
    finish(harness)


@pytest.mark.proof
@pytest.mark.parametrize('point', ['native_before_spawn', 'native_before_child_publication',
                                  'native_after_child_publication', 'native_before_acknowledgement',
                                  'native_after_acknowledgement'])
@pytest.mark.parametrize('failure', ['raise', 'kill'])
def test_native_audio_admission_supervisor_faults(harness, proof_line, point, failure):
    observe(harness, [proof_line[0]])
    harness.pin(point)
    child = launch(harness, f'''
        import signal
        from hfmodem.core import native
        owned = claim.acquire('legacy', 'capture-fault', prove_line=True,
                              ptt_factory=claim.line_factory(os.environ['ROW_PTT']))
        os.environ['HFMODEM_COEXIST_FAULT'] = {point + ':' + failure!r}
        try:
            native.spawn_native(['ffmpeg', '-f', 'pulse', '-i', 'default', '-'], claim_fd=owned)
        except native.NativeSpawnError as exc:
            print(str(exc), flush=True)
        else:
            raise AssertionError('fault did not fail capture startup')
        claim.close(owned)
    ''')
    harness.reached(point, quiescent=True)
    spawned = [e for e in harness.events() if e['event'] == 'native_spawn']
    harness.release(point)
    completed(child)
    if failure == 'kill' and spawned:
        survivor = spawned[-1]
        contender = launch(harness, f'''
            from hfmodem.core.native import NativeChild, _end
            child = NativeChild({survivor['child_pid']}, {survivor['child_started']!r}, None, None, None)
            try:
                mutex = claim.mutex(timeout=0)
            except claim.ClaimBusy as exc:
                assert {point == 'native_after_acknowledgement'!r} is False
                assert 'admission in progress' in str(exc)
            else:
                assert {point == 'native_after_acknowledgement'!r} or child.wait_gone(5)
                claim.mutex_release(mutex)
            assert _end(child)
        ''')
        completed(contender)
    assert line_low(proof_line)
    finish(harness)


@pytest.mark.proof
@pytest.mark.parametrize('misuse', ['unlock', 'move-offset', 'close-with-device', 'forge-nonce'])
def test_child_misuses_shared_claim(harness, proof_line, fake_yaesu, misuse):
    observe(harness, [proof_line[0], fake_yaesu.port])
    harness.pin('misused')
    inner = METADATA + textwrap.dedent(f'''
        import fcntl, sys
        fd = int(sys.argv[1])
        inherited = claim.inherited(fd)
        if {misuse!r} == 'unlock':
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif {misuse!r} == 'move-offset':
            os.lseek(fd, 0, os.SEEK_SET)
        elif {misuse!r} == 'forge-nonce':
            nonce = inherited.offset()
            claim.close(inherited)
            fd = os.open(claim.claim_path(), os.O_RDWR)
            os.lseek(fd, nonce, os.SEEK_SET)
        else:
            device = os.open({fake_yaesu.port!r}, os.O_RDWR | os.O_NOCTTY)
            coexist.event('device_open', path={fake_yaesu.port!r}, fd=device)
            claim.close(inherited)
        coexist.checkpoint('misused')
        if {misuse!r} == 'close-with-device':
            os.close(device)
            coexist.event('device_close', path={fake_yaesu.port!r}, fd=device)
    ''')
    parent = launch(harness, f'''
        import subprocess, sys
        owned = claim.acquire('legacy', 'misuse', prove_line=True,
                              ptt_factory=claim.line_factory(os.environ['ROW_PTT']))
        child = subprocess.Popen([sys.executable, '-c', {inner!r}, str(owned.fd)],
                                 pass_fds=(owned.fd,))
        assert child.wait(timeout=10) == 0
        claim.close(owned)
    ''')
    event = harness.reached('misused', quiescent=True)
    path = harness.root / 'rig.claim'
    snapshot = harness.observer.snapshots[-1]
    try:
        with pytest.raises(OracleFailure, match='claim descriptor|does not hold'):
            assert_claim_possession(snapshot, pid=event['pid'], claim_path=path,
                                    nonce=int(path.read_text().split()[4]))
        if misuse == 'close-with-device':
            assert any(fd['path'] == fake_yaesu.port for fd in snapshot.descriptors[event['pid']])
        assert line_low(proof_line)
    finally:
        harness.release('misused')
    completed(parent)
    finish(harness)


@pytest.mark.parametrize('case', ['direct', 'server', 'different', 'defaults', 'unresolvable', 'macos'])
def test_audio_endpoints(harness, case):
    observe(harness)
    node = harness.root / 'fake-audio-node'
    node.touch()
    harness.pin('endpoint-held')
    child = launch(harness, f'''
        from types import SimpleNamespace
        fd = os.open({str(node)!r}, os.O_RDWR)
        coexist.checkpoint('endpoint-held')
        calls = []
        def query(endpoint, direction):
            calls.append((endpoint, direction))
            return {{'name': 'unresolved' if {case!r} == 'unresolvable' else
                    'fake card (hw:99999,0)' if direction == 'input' else 'fake card (hw:99998,0)'}}
        sd = SimpleNamespace(default=SimpleNamespace(device=(17, 18)), query_devices=query)
        def discovery(paths):
            if {case!r} not in ('direct', 'server'):
                return ''
            return 'p' + str(os.getpid()) + '\\nc' + (
                'pipewire' if {case!r} == 'server' else 'uncooperative') + f'\\nf{{fd}}\\no0t0\\n'
        devices = claim.Devices(capture=None if {case!r} == 'defaults' else 17, playback=18)
        platform = 'darwin' if {case!r} == 'macos' else 'linux'
        try:
            verdicts = claim.check_devices(devices, sounddevice=sd, platform=platform,
                lsof=discovery, fdinfo=lambda pid, fd: 'pos:\\t0\\n')
        except claim.ClaimBusy as exc:
            assert {case!r} == 'direct' and str(os.getpid()) in str(exc)
        else:
            assert {case!r} != 'direct'
            audio = [v for v in verdicts if v.endpoint in ('capture', 'playback')]
            assert len(audio) == 2 and all(v.verdict == 'cooperative' for v in audio)
            if {case!r} == 'server':
                assert all(v.pids == (os.getpid(),) for v in audio)
            elif {case!r} in ('unresolvable', 'macos'):
                assert all(not v.nodes for v in audio)
            else:
                assert audio[0].nodes != audio[1].nodes
            if {case!r} == 'defaults':
                assert calls == [(17, 'input'), (18, 'output')]
        os.close(fd)
        owned = claim.acquire('legacy', 'cooperative-audio')
        import sounddevice
        with sounddevice.InputStream():
            pass
        claim.release(owned)
    ''')
    event = harness.reached('endpoint-held', quiescent=True)
    assert any(fd.get('path') == str(node)
               for fd in harness.observer.snapshots[-1].descriptors[event['pid']])
    harness.release('endpoint-held')
    completed(child)
    evidence = finish(harness)
    assert any(e['event'] == 'device_open' for e in evidence.events)


@pytest.mark.parametrize('case', ['unlocked-opener', 'shared-admission', 'stale-cleanup-line',
                                'partial-publication', 'discovery-failure'])
def test_mutex_holder_identified(harness, case):
    observe(harness)
    harness.pin('mutex-published')
    owner = launch(harness, f'''
        import fcntl
        if {case!r} in ('shared-admission', 'stale-cleanup-line'):
            fd = claim._shared()
            if {case!r} == 'stale-cleanup-line':
                os.pwrite(fd, b'999999 dead 123\\n', 0)
        else:
            fd = claim.mutex(timeout=0)
            if {case!r} == 'partial-publication':
                os.ftruncate(fd, 0)
                os.pwrite(fd, b'partial', 0)
        coexist.checkpoint('mutex-published')
        os.close(fd)
    ''')
    event = harness.reached('mutex-published', quiescent=True)
    pid = event['pid']
    descriptors = [fd for fd in harness.observer.snapshots[-1].descriptors[pid]
                   if fd.get('path') == str(harness.root / 'rig.claim.cleanup')]
    assert len(descriptors) == 1
    descriptor = descriptors[0]
    contender = launch(harness, f'''
        from pathlib import Path
        def discover(paths):
            if {case!r} == 'discovery-failure':
                raise OSError('injected discovery failure')
            return 'p{pid}\\ncfixture-holder\\nf{descriptor['fd']}\\no0t{descriptor['pos']}\\n'
        options = dict(lsof=discover, platform='darwin')
        extra = os.open(claim.mutex_path(), os.O_RDWR)
        assert os.lseek(extra, 0, os.SEEK_CUR) == 0
        try:
            claim.mutex(timeout=0, **options)
        except claim.ClaimBusy as exc:
            message = str(exc)
            if {case!r} in ('shared-admission', 'stale-cleanup-line'):
                assert 'admission in progress' in message and '{pid}' in message
                assert '999999' not in message
            elif {case!r} in ('discovery-failure', 'partial-publication'):
                assert 'holder unknown' in message
            else:
                assert 'cleanup in progress' in message and '{pid}' in message
        else:
            raise AssertionError('mutex admitted contender')
        os.close(extra)
    ''')
    try:
        completed(contender)
    finally:
        harness.release('mutex-published')
        completed(owner)
    finish(harness)


@pytest.mark.noline
@pytest.mark.parametrize('count', [1, 3], ids=['one-survivor', 'several-survivors'])
def test_mutex_holder_identified_dead_publisher(harness, count):
    observe(harness)
    for index in range(count):
        harness.pin(f'mutex-survivor-{index}')
    child_code = METADATA + '''
import sys
mutex = int(sys.argv[1])
assert claim.verify_cleanup(mutex) == mutex
coexist.checkpoint('mutex-survivor-' + sys.argv[2])
os.close(mutex)
'''
    publisher = launch(harness, f'''
        import subprocess, sys
        mutex = claim.mutex(timeout=0)
        for index in range({count}):
            subprocess.Popen([sys.executable, '-c', {child_code!r}, str(mutex), str(index)],
                             pass_fds=(mutex,))
        os._exit(0)
    ''')
    events = [harness.reached(f'mutex-survivor-{i}', quiescent=True) for i in range(count)]
    published = next(e for e in harness.events() if e['event'] == 'cleanup_acquired')
    snapshot = harness.observer.snapshots[-1]
    for event in events:
        descriptors = snapshot.descriptors[event['pid']]
        assert any(fd.get('path') == str(harness.root / 'rig.claim.cleanup')
                   and fd.get('pos') == published['nonce'] for fd in descriptors)
    contender = launch(harness, f'''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            message = str(exc)
            assert 'cleanup in progress' in message
            for pid in {[e['pid'] for e in events]!r}:
                assert str(pid) in message, message
            assert str({published['pid']}) not in message, message
        else:
            raise AssertionError('dead publisher lost surviving cleanup authority')
    ''')
    try:
        completed(contender)
    finally:
        for index in range(count):
            harness.release(f'mutex-survivor-{index}')
    completed(publisher)
    finish(harness)


@pytest.mark.noline
@pytest.mark.parametrize('fate', ['alive', 'killed', 'publication-failed'])
def test_inherited_descriptor_proven_reopened_at_offset_barrier(harness, fate):
    observe(harness)
    harness.pin('inherited_after_inode')
    harness.pin('reopen-parent-held')
    harness.pin('reopen-refused')
    child_code = METADATA + '''
fd = os.open(claim.claim_path(), os.O_RDWR)
os.dup2(fd, 9)
if fd != 9:
    os.close(fd)
try:
    claim.inherited(9)
except claim.ClaimError as exc:
    assert 'offset' in str(exc) or 'holder' in str(exc), str(exc)
else:
    raise AssertionError('reopened descriptor acquired possession')
coexist.checkpoint('reopen-refused')
os.close(9)
'''
    parent_code = METADATA + textwrap.dedent(f'''
        import subprocess, sys
        owned = claim.acquire('legacy', 'reopen-parent')
        coexist.event('reopen_parent')
        child = subprocess.Popen([sys.executable, '-c', {child_code!r}])
        coexist.checkpoint('reopen-parent-held')
        if {fate!r} == 'publication-failed':
            claim.release(owned)
        assert child.wait(timeout=15) == 0
        if {fate!r} != 'publication-failed':
            claim.release(owned)
    ''')
    if fate == 'publication-failed':
        harness.pin('publication-unwound')
        parent_code = METADATA + textwrap.dedent(f'''
            import subprocess, sys
            children = []
            def failed(point):
                if point == 'after_holder_line':
                    children.append(subprocess.Popen([sys.executable, '-c', {child_code!r}]))
                    coexist.checkpoint('reopen-parent-held')
                    raise RuntimeError('injected acquisition failure after publication')
            coexist.fault = failed
            try:
                claim.acquire('legacy', 'reopen-parent')
            except RuntimeError as exc:
                assert str(exc) == 'injected acquisition failure after publication'
            else:
                raise AssertionError('acquisition failure did not fire')
            assert claim.claim_path().read_bytes() == b''
            coexist.checkpoint('publication-unwound')
            assert children[0].wait(timeout=15) == 0
        ''')
    guardian = launch(harness, f'''
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', {parent_code!r}])
        assert child.wait(timeout=20) == {-9 if fate == 'killed' else 0}
    ''')
    event = harness.reached('inherited_after_inode', quiescent=True)
    parent = harness.reached('reopen-parent-held', quiescent=True)['pid']
    if fate == 'killed':
        os.kill(parent, signal.SIGKILL)
    harness.release('reopen-parent-held')
    if fate == 'publication-failed':
        harness.reached('publication-unwound', quiescent=True)
        harness.release('publication-unwound')
    harness.release('inherited_after_inode')
    harness.reached('reopen-refused', quiescent=True)
    snapshot = harness.observer.snapshots[-1]
    assert any(fd.get('fd') == 9 and fd.get('pos') == 0
               for fd in snapshot.descriptors[event['pid']])
    harness.release('reopen-refused')
    completed(guardian)
    evidence = finish(harness)
    if evidence.trace:
        assert not any(record['pid'] == event['pid'] and record['syscall'] == 'flock'
                       for record in evidence.trace.records)


AUDIO_ENTRIES = {
    'rx_levels': '''
        from rx_levels import DeviceSource
        samples = DeviceSource('Coexist fake audio').capture(20, 'coexist admission')
        assert len(samples) == 20 * 48000 and not samples.any()
    ''',
    'vara_monitor': '''
        from vara_monitor import device_source
        source = device_source('Coexist fake audio')
        try:
            offset, samples = next(source)
            assert offset == 0 and len(samples) and not samples.any()
        finally:
            source.close()
    ''',
    'sabir-monitor': '''
        from hfmodem.sabir.monitor import device_chunks
        source = device_chunks('Coexist fake audio')
        try:
            samples = next(source)
            assert len(samples) and not samples.any()
        finally:
            source.close()
    ''',
}

LINE_CASES = [pytest.param(False, marks=pytest.mark.noline, id='receive-only'),
              pytest.param(True, marks=pytest.mark.proof, id='line-proof')]


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('entry', AUDIO_ENTRIES)
@pytest.mark.parametrize('point', ['ffmpeg_before_open', 'ffmpeg_before_output'])
def test_native_audio_admission_entry_points(harness, request, with_line, entry, point):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    harness.env['TMPDIR'] = str(harness.root)
    harness.pin(point)
    harness.pin('capture-finished')
    child = launch(harness, f'''
owned = claim.acquire('legacy', 'capture-parent', prove_line={with_line!r},
                      ptt_factory=claim.line_factory(os.environ['ROW_PTT']) if {with_line!r} else None)
os.dup2(owned.fd, 9)
if owned.fd != 9:
    claim.close(owned)
owned = claim.ClaimDescriptor(9)
os.environ['RIG_CLAIMED'] = str(claim.claim_path())
''' + textwrap.dedent(AUDIO_ENTRIES[entry]) + '''
coexist.checkpoint('capture-finished')
claim.release(owned)
''')
    event = harness.reached(point, quiescent=True)
    path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                            nonce=int(path.read_text().split()[4]))
    contender = launch(harness, '''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            assert 'admission in progress' in str(exc)
        else:
            raise AssertionError('cleanup admitted ahead of capture readiness')
    ''')
    completed(contender)
    assert not any(e['event'] == 'native_admitted' for e in harness.events())
    harness.release(point)
    try:
        harness.reached('capture-finished', timeout=40 if entry == 'rx_levels' else 15, quiescent=True)
    except TimeoutError:
        completed(child)
        raise
    harness.release('capture-finished')
    completed(child)
    evidence = finish(harness)
    opened = next(e for e in evidence.events if e['event'] == 'device_open'
                  and e['path'] == str(harness.audio))
    admitted = next(e for e in evidence.events if e['event'] == 'native_admitted')
    assert opened['pid'] == admitted['child_pid'] == event['pid']
    assert opened['monotonic_ns'] < admitted['monotonic_ns']
    if line:
        assert line_low(line)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('entry', AUDIO_ENTRIES)
@pytest.mark.parametrize('point', ['ffmpeg_before_open', 'ffmpeg_before_output'])
@pytest.mark.parametrize('death', ['spawner', 'supervisor', 'both'])
def test_native_audio_admission_entry_death(harness, request, with_line, entry, point, death):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    harness.env['TMPDIR'] = str(harness.root)
    harness.pin(point)
    harness.pin('capture-parent-ended')
    inner = METADATA + f'''
owned = claim.acquire('legacy', 'capture-parent', prove_line={with_line!r},
                      ptt_factory=claim.line_factory(os.environ['ROW_PTT']) if {with_line!r} else None)
os.dup2(owned.fd, 9)
if owned.fd != 9:
    claim.close(owned)
os.environ['RIG_CLAIMED'] = str(claim.claim_path())
coexist.event('capture_parent')
''' + textwrap.dedent(AUDIO_ENTRIES[entry])
    guardian = launch(harness, f'''
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', {inner!r}])
        assert child.wait(timeout=20) in (-9, 1)
        coexist.checkpoint('capture-parent-ended')
    ''')
    event = harness.reached(point, timeout=10, quiescent=True)
    events = harness.events()
    parent = next(e['pid'] for e in events if e['event'] == 'capture_parent')
    supervisor_event = next(e for e in events if e['event'] == 'native_supervisor')
    supervisor = supervisor_event['pid']
    if death in ('spawner', 'both'):
        os.kill(parent, signal.SIGKILL)
    if death in ('supervisor', 'both'):
        os.kill(supervisor, signal.SIGKILL)
    contender = launch(harness, '''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            assert 'admission in progress' in str(exc)
        else:
            raise AssertionError('capture death exposed an unopened child to cleanup')
    ''')
    try:
        completed(contender)
        snapshot = harness.observer.capture('capture-survived')
        if sys.platform != 'linux':
            snapshot.claim_free = probe_claim(harness.root / 'rig.claim', quiescent=True)
        assert_claim_possession(snapshot, pid=event['pid'], claim_path=harness.root / 'rig.claim',
                                nonce=int((harness.root / 'rig.claim').read_text().split()[4]))
    finally:
        os.kill(event['pid'], signal.SIGKILL)
        harness.release(point)
        harness.release('capture-parent-ended')
    completed(guardian)
    reaped = launch(harness, f'''
        from hfmodem.core.native import NativeChild
        supervisor = NativeChild({supervisor}, {supervisor_event['started']!r}, None, None, None)
        assert supervisor.wait_gone(5), 'capture supervisor did not terminate'
    ''')
    completed(reaped)
    if line:
        assert line_low(line)
    finish(harness)


@pytest.mark.noline
def test_native_diagnostic_records_concurrent_supervisors(harness):
    observe(harness)
    for point in ('first-capture', 'second-capture', 'first-completed'):
        harness.pin(point)
    binary(harness, 'ffmpeg', '''
        import os, sys
        from hfmodem.core import claim, coexist
        from hfmodem.tests.coexist import shim
        claim.inherited(9)
        slot = sys.argv[sys.argv.index('-i') + 1]
        with shim.RawInputStream() as stream:
            coexist.checkpoint(slot + '-capture')
            sys.stdout.buffer.write(bytes(512))
            sys.stdout.buffer.flush()
            sys.stdin.buffer.read()
    ''')
    owner = launch(harness, '''
        from concurrent.futures import ThreadPoolExecutor
        from hfmodem.core import native
        owned = claim.acquire('legacy', 'parallel-supervisors')
        def start(slot):
            return native.spawn_native(['ffmpeg', '-f', 'pulse', '-i', slot, '-'], claim_fd=owned)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(start, 'first')
            second = pool.submit(start, 'second')
            one = first.result(timeout=10)
            assert native._end(one)
            for stream in (one.stdin, one.stdout, one.stderr):
                stream.close()
            coexist.checkpoint('first-completed')
            two = second.result(timeout=10)
            assert one.pid != two.pid
            assert native._end(two)
            for stream in (two.stdin, two.stdout, two.stderr):
                stream.close()
        claim.release(owned)
    ''')
    harness.reached('first-capture', quiescent=True)
    second = harness.reached('second-capture', quiescent=True)
    harness.release('first-capture')
    harness.reached('first-completed', quiescent=True)
    directory = harness.root / 'rig.claim.native.d'
    records = [json.loads(path.read_text()) for path in directory.iterdir() if path.suffix != '.tmp']
    assert len(records) == 1 and records[0]['child_pid'] == second['pid'], records
    contender = launch(harness, '''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            assert 'admission in progress' in str(exc)
        else:
            raise AssertionError('one completed supervisor admitted over the other')
    ''')
    try:
        completed(contender)
    finally:
        harness.release('second-capture')
        harness.release('first-completed')
    completed(owner)
    assert not list(directory.iterdir())
    evidence = finish(harness)
    assert len([e for e in evidence.events if e['event'] == 'native_admitted']) == 2


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('case', ['ordinary-close', 'forced-close', 'broken-pipe',
                                'child-exits', 'supervisor-failure'])
def test_native_spawn_admission_shrike_handle(harness, request, with_line, fake_yaesu, case):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [fake_yaesu.port, *([line[0]] if line else [])])
    harness.env['CAT_PORT'] = fake_yaesu.port
    harness.pin('interactive-open')
    harness.pin('handle-ready')
    if case == 'supervisor-failure':
        harness.pin('native_before_acknowledgement')
    binary(harness, 'rigctl', f'''
        import os, signal, sys
        from hfmodem.core import claim, coexist
        claim.inherited(9)
        assert sys.argv[sys.argv.index('-P') + 1] == 'NONE'
        if {case!r} in ('forced-close', 'broken-pipe'):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if {case!r} == 'broken-pipe':
            os.close(0)
        fd = os.open(os.environ['CAT_PORT'], os.O_RDWR | os.O_NOCTTY)
        coexist.event('device_open', path=os.environ['CAT_PORT'], fd=fd)
        try:
            coexist.checkpoint('interactive-open')
            if {case!r} in ('broken-pipe', 'forced-close'):
                while True:
                    signal.pause()
            elif {case!r} != 'child-exits':
                for line in sys.stdin:
                    if line.strip() == 'q':
                        break
        finally:
            os.close(fd)
            coexist.event('device_close', path=os.environ['CAT_PORT'], fd=fd)
    ''')
    owner = launch(harness, f'''
        from hfmodem.shrike.ota import Rig
        owned = claim.acquire('legacy', 'interactive', prove_line={with_line!r},
                              ptt_factory=claim.line_factory(os.environ['ROW_PTT']) if {with_line!r} else None)
        os.dup2(owned.fd, 9)
        if owned.fd != 9:
            claim.close(owned)
        owned = claim.ClaimDescriptor(9)
        rig = Rig(1, os.environ['CAT_PORT'], 38400)
        if {case!r} == 'supervisor-failure':
            os.environ['HFMODEM_COEXIST_FAULT'] = 'native_before_acknowledgement:raise'
            assert rig._open() is None and rig._sick
        else:
            handle = rig._open()
            assert handle is not None and handle.alive()
            assert rig._open() is handle
            if {case!r} == 'broken-pipe':
                assert not rig._cmd('f')
                assert rig._proc is handle and handle.alive()
        coexist.checkpoint('handle-ready')
        if {case!r} == 'child-exits':
            assert handle.wait_gone(5)
        rig._close()
        assert rig._proc is None
        claim.release(owned)
    ''')
    event = harness.reached('interactive-open', quiescent=True)
    path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                            nonce=int(path.read_text().split()[4]))
    if case == 'supervisor-failure':
        harness.reached('native_before_acknowledgement', quiescent=True)
        harness.release('native_before_acknowledgement')
    harness.reached('handle-ready', timeout=10, quiescent=True)
    harness.release('interactive-open')
    harness.release('handle-ready')
    completed(owner)
    evidence = finish(harness)
    assert sum(e['event'] == 'native_spawn' for e in evidence.events) == 1
    if line:
        assert line_low(line)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('point', ['launch-validated', 'before-spawn', 'child-startup', 'modem-before-open'])
@pytest.mark.parametrize('kill_launcher', [False, True], ids=['launcher-live', 'launcher-killed'])
@pytest.mark.parametrize('kill_root', [False, True], ids=['root-live', 'root-killed'])
def test_inherited_launch_admitted_before_cleanup(harness, request, with_line, point,
                                                 kill_launcher, kill_root):
    from .test_rows_ownership import down, inventory, shell_setup, station_file

    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    path = station_file(harness)
    harness.env['HFMODEM_STATION'] = str(path)
    harness.env['PTT_PORT'] = line[0] if line else ''
    setup = shell_setup(harness)
    setup += '''
claim_twin() {
  if [ "$1" = mutex ]; then
    "$HFMODEM_PYTHON" -m hfmodem.core.claim "$@" --timeout 0
  else
    "$HFMODEM_PYTHON" -m hfmodem.core.claim "$@"
  fi
}
'''
    barriers = (point, 'campaign-admitted', 'modem-active', 'campaign-root-done',
                'campaign-guardian-wait')
    if not kill_launcher:
        barriers += ('campaign-root-wait',)
    for barrier in barriers:
        harness.pin(barrier)
    modem = binary(harness, 'campaign-modem', '''
        import os
        import sounddevice
        from hfmodem.core import claim, coexist
        coexist.checkpoint('child-startup')
        admission = claim.inherited(9, launch=True)
        try:
            coexist.checkpoint('modem-before-open')
            with sounddevice.RawInputStream():
                admission.opened()
                coexist.checkpoint('modem-active')
        finally:
            admission.failed()
            claim.close(9)
    ''')
    checkpoint = (shlex.quote(sys.executable) + ' -c '
                  + shlex.quote('from hfmodem.core import coexist; import sys; '
                                'coexist.checkpoint(sys.argv[1])'))
    identity = shlex.quote('import os; from hfmodem.core import coexist; '
                          "coexist.event('campaign_launcher', launcher=os.getppid())")
    script = (setup + '\nclaim_rig campaign\n' + shlex.quote(sys.executable) + ' -c ' + identity
              + '\n' + checkpoint + ' launch-validated 7>&- 8>&- 9>&-\n'
              + checkpoint + ' before-spawn 7>&- 8>&- 9>&-\n'
              + 'claim_start ' + shlex.quote(str(modem)) + '\n'
              + checkpoint + ' campaign-admitted 7>&- 8>&- 9>&-\nclaim_wait\n')
    argv = shell_argv(['/bin/bash', '-c', script],
                      dict(harness.env, RIG_CLAIMED=str(harness.root / 'rig.claim')))
    root_code = METADATA + textwrap.dedent(f'''
        import subprocess
        owned = claim.acquire('shell', 'campaign-root')
        os.dup2(owned.fd, 9)
        if owned.fd != 9:
            claim.close(owned)
        coexist.event('campaign_root')
        env = dict(os.environ, RIG_CLAIMED=str(claim.claim_path()))
        child = subprocess.Popen({argv!r}, pass_fds=(9,), env=env)
        if not {kill_launcher!r}:
            coexist.checkpoint('campaign-root-wait')
        child.wait(timeout=25)
        coexist.checkpoint('campaign-root-done')
        claim.release(9)
    ''')
    guardian = launch(harness, f'''
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', {root_code!r}])
        coexist.checkpoint('campaign-guardian-wait')
        assert child.wait(timeout=30) == {-9 if kill_root else 0}
    ''')
    barrier_event = harness.reached(point, timeout=12, quiescent=True)
    events = harness.events()
    root_pid = next(e['pid'] for e in events if e['event'] == 'campaign_root')
    launcher_pid = next(e['launcher'] for e in events if e['event'] == 'campaign_launcher')
    if kill_launcher:
        os.kill(launcher_pid, signal.SIGKILL)
    if kill_root:
        os.kill(root_pid, signal.SIGKILL)
        # Let the guardian reap the killed root while cleanup continues.
        harness.release('campaign-guardian-wait')

    def refresh():
        snapshot = harness.observer.capture('campaign-inventory')
        inventory(harness, {pid: 'coexist-campaign-participant' for pid in snapshot.participants})
        return len(snapshot.participants)

    try:
        if kill_launcher and not kill_root:
            harness.reached('campaign-root-done', timeout=10, quiescent=True)
        targets = refresh()
        admission_held = not kill_launcher or point == 'modem-before-open'
        expected = 2 if admission_held else (0 if kill_root else 7)
        out, err = completed(down(harness, setup, line=line[0] if line else ''), expected,
                             targets=targets)
        if admission_held:
            assert 'admission in progress' in out + err
            assert 'CLEANUP COMPLETE' not in out and 'PTT LINE' not in out
        elif kill_root:
            assert 'CLEANUP COMPLETE' in out
        else:
            assert 'holds the claim' in out
        assert not any(e['event'] == 'device_open' and e['path'] == str(harness.audio)
                       for e in harness.events())
        harness.release(point)
        if not kill_launcher:
            harness.reached('modem-active', timeout=10, quiescent=True)
            harness.reached('campaign-admitted', timeout=10, quiescent=True)
            targets = refresh()
            out, err = completed(down(harness, setup, line=line[0] if line else ''),
                                 0 if kill_root else 7, targets=targets)
            assert ('CLEANUP COMPLETE' in out) is kill_root, (out, err)
        elif point == 'modem-before-open':
            starter = barrier_event['pid']
            started = next(e['start'] for e in harness.events()
                           if e['pid'] == starter and e['event'] == 'inherited' and e['verdict'])
            completed(launch(harness, f'''
                from hfmodem.core.native import NativeChild
                assert NativeChild({starter}, {started!r}, None, None, None).wait_gone(5)
            '''))
            targets = refresh()
            out, err = completed(down(harness, setup, line=line[0] if line else ''),
                                 0 if kill_root else 7, targets=targets)
            assert ('CLEANUP COMPLETE' in out) is kill_root, (out, err)
    finally:
        # These waits bound termination after the scenario's intentional holds;
        # traced shell startup and cleanup must not spend those holds' budget.
        # A killed launcher's root must still report done before this finally.
        for barrier in barriers:
            harness.release(barrier)
    completed(guardian)
    evidence = finish(harness)
    for event in evidence.events:
        if event['event'] == 'device_open' and event['path'] == str(harness.audio):
            validated = [e for e in evidence.events if e['event'] == 'inherited'
                         and e.get('verdict') and e['pid'] == event['pid']]
            assert validated and min(e['monotonic_ns'] for e in validated) < event['monotonic_ns']
    if line:
        assert line_low(line)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('failure', ['raise', 'kill'])
def test_inherited_launch_admitted_before_cleanup_onair_entry(harness, request, fake_yaesu,
                                                            with_line, failure):
    from .test_rows_ownership import shell_setup, station_file

    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [fake_yaesu.port, *([line[0]] if line else [])])
    path = station_file(harness)
    shell_setup(harness)
    for name in ('CAT_PORT', 'PTT_PORT'):
        alias = harness.root / name.lower()
        alias.symlink_to(fake_yaesu.port if name == 'CAT_PORT' or line is None else line[0])
        harness.env[name] = str(alias)
    harness.env.update(HFMODEM_STATION=str(path), MYCALL='K1ABC',
                       HFMODEM_CAPTURE_DIR=str(harness.root / 'capture'))
    harness.pin('after_launch_validation')
    argv = shell_argv(['/bin/bash', str(TREE / 'tools/onair.sh'), 'pactor', 'K1XYZ', '7101500'],
        dict(harness.env, RIG_CLAIMED=str(harness.root / 'rig.claim'),
             HFMODEM_COEXIST_FAULT='after_launch_validation:' + failure))
    owner = launch(harness, f'''
        import subprocess
        owned = claim.acquire('shell', 'onair-parent')
        os.dup2(owned.fd, 9)
        if owned.fd != 9:
            claim.close(owned)
        owned = claim.ClaimDescriptor(9)
        published = owned.pread()
        child = subprocess.run({argv!r}, pass_fds=(9,), capture_output=True, text=True)
        assert child.returncode != 0, (child.stdout, child.stderr)
        assert 'CLAIM NOT ADMITTED' in child.stdout, (child.stdout, child.stderr)
        assert 'CLEANUP COMPLETE' not in child.stdout
        assert owned.pread() == published
        assert claim.inherited(9).offset() == owned.offset()
        claim.release(owned)
    ''')
    harness.reached('after_launch_validation', timeout=15, quiescent=True)
    contender = launch(harness, '''
        try:
            claim.mutex(timeout=0)
        except claim.ClaimBusy as exc:
            assert 'admission in progress' in str(exc)
        else:
            raise AssertionError('onair inherited launch failed to hold admission')
    ''')
    try:
        completed(contender)
    finally:
        harness.release('after_launch_validation')
    completed(owner)
    evidence = finish(harness)
    assert not any(e['event'] == 'device_open' for e in evidence.events)
    assert any(e['event'] == 'fault' and e['point'] == 'after_launch_validation'
               for e in evidence.events)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('point', ['down-inventory', 'down-between-signals'])
@pytest.mark.parametrize('bypass', [False, True], ids=['compliant', 'injected-bypass'])
def test_inherited_launch_during_cleanup(harness, request, with_line, point, bypass):
    from .test_rows_ownership import down, inventory, shell_setup, station_file

    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    station_file(harness)
    setup = shell_setup(harness)
    bootstrap = harness.root / 'bootstrap/sitecustomize.py'
    with bootstrap.open('a') as stream:
        stream.write(textwrap.dedent('''
            import signal
            from hfmodem.core import native, coexist
            original_signal = native.NativeChild._signal
            def checked_signal(self, sig):
                if sig == signal.SIGTERM:
                    coexist.checkpoint('down-inventory')
                result = original_signal(self, sig)
                if sig == signal.SIGTERM:
                    coexist.checkpoint('down-between-signals')
                return result
            native.NativeChild._signal = checked_signal
        '''))
    for barrier in ('modem-ready', point, 'late-launch-done', 'late-child-open'):
        harness.pin(barrier)
    child_code = METADATA + '''
import sounddevice
claim.inherited(9)
with sounddevice.RawInputStream():
    coexist.checkpoint('late-child-open')
'''
    modem_code = METADATA + textwrap.dedent(f'''
        import signal, subprocess, sys
        claim.inherited(9)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        coexist.checkpoint('modem-ready')
        if {bypass!r}:
            child = subprocess.Popen([sys.executable, '-c', {child_code!r}], pass_fds=(9,))
            coexist.event('native_spawn', child_pid=child.pid,
                          child_started=claim.process_started(child.pid), argv=['injected-late-child'])
        else:
            try:
                claim.inherited(9, launch=True)
            except claim.ClaimBusy as exc:
                assert 'cleanup in progress' in str(exc)
                coexist.event('late_launch_refused')
            else:
                raise AssertionError('modem launched during cleanup')
        coexist.checkpoint('late-launch-done')
        while True:
            signal.pause()
    ''')
    root_code = METADATA + textwrap.dedent(f'''
        import subprocess, sys
        owned = claim.acquire('shell', 'dead-autopilot')
        os.dup2(owned.fd, 9)
        if owned.fd != 9:
            claim.close(owned)
        subprocess.Popen([sys.executable, '-c', {modem_code!r}], pass_fds=(9,))
        os._exit(0)
    ''')
    guardian = launch(harness, f'''
        import subprocess, sys
        assert subprocess.run([sys.executable, '-c', {root_code!r}]).returncode == 0
    ''')
    modem = harness.reached('modem-ready', quiescent=True)['pid']
    inventory(harness, {modem: 'coexist-dead-root-modem'})
    cleanup = down(harness, setup, line=line[0] if line else '')
    try:
        harness.reached(point, timeout=10, quiescent=True)
        harness.release('modem-ready')
        harness.reached('late-launch-done', quiescent=True)
        if bypass:
            newcomer = harness.reached('late-child-open', quiescent=True)['pid']
            snapshot = harness.observer.snapshots[-1]
            mutex = harness.root / 'rig.claim.cleanup'
            with pytest.raises(OracleFailure, match='claim descriptor or nonce not observed'):
                assert_claim_possession(snapshot, pid=newcomer, claim_path=mutex,
                                        nonce=int(mutex.read_text().split()[2]))
            opened = next(e for e in harness.events() if e['event'] == 'device_open'
                          and e['pid'] == newcomer)
            exclusive = next(e for e in harness.events() if e['event'] == 'cleanup_acquired')
            assert opened['monotonic_ns'] > exclusive['monotonic_ns']
            inventory(harness, {modem: 'coexist-dead-root-modem', newcomer: 'coexist-new-child'})
            path = harness.root / 'inventory.json'
            rows = json.loads(path.read_text())
            rows[str(newcomer)]['parent'] = modem
            path.write_text(json.dumps(rows))
        else:
            assert any(e['event'] == 'late_launch_refused' for e in harness.events())
            assert not any(e['event'] == 'native_spawn' for e in harness.events())
        harness.release(point)
        out, err = completed(cleanup, targets=1 + bypass)
        assert 'CLEANUP COMPLETE' in out, err
        assert f'pid {modem} -- SIGKILL' in out
        if bypass:
            assert f'pid {newcomer} -- SIGTERM' in out
    finally:
        for barrier in ('modem-ready', point, 'late-launch-done', 'late-child-open'):
            harness.release(barrier)
    completed(guardian)
    finish(harness)
    if line:
        assert line_low(line)


@pytest.mark.parametrize('with_line', LINE_CASES)
@pytest.mark.parametrize('output', ['pipe', 'file'])
def test_native_audio_admission_silent_deadline(harness, request, with_line, output):
    line = request.getfixturevalue('proof_line') if with_line else None
    observe(harness, [line[0]] if line else ())
    harness.env['HFMODEM_COEXIST_FFMPEG_DELAY'] = '60'
    harness.pin('ffmpeg_before_output')
    harness.pin('silent-capture-ended')
    target = str(harness.root / 'silent.wav') if output == 'file' else '-'
    parent = launch(harness, f'''
        from hfmodem.core import native
        owned = claim.acquire('legacy', 'silent-capture')
        try:
            native.spawn_native(['ffmpeg', '-f', 'pulse', '-i', 'default',
                '-ac', '1', '-ar', '48000', '-f', {'wav' if output == 'file' else 's16le'!r},
                {target!r}], claim_fd=owned)
        except native.NativeSpawnError as exc:
            assert 'native open timed out after 12 s' in str(exc), str(exc)
            assert exc.child is not None and not exc.child.alive()
            for stream in (exc.child.stdin, exc.child.stdout, exc.child.stderr):
                stream.close()
        else:
            raise AssertionError('silent capture was admitted')
        coexist.checkpoint('silent-capture-ended')
        claim.release(owned)
    ''')
    event = harness.reached('ffmpeg_before_output', quiescent=True)
    path = harness.root / 'rig.claim'
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'], claim_path=path,
                            nonce=int(path.read_text().split()[4]))
    if output == 'file':
        assert any(e['event'] == 'ffmpeg_header' for e in harness.events())
    assert not any(e['event'] == 'native_admitted' for e in harness.events())
    harness.release('ffmpeg_before_output')
    harness.reached('silent-capture-ended', timeout=20, quiescent=True)
    assert event['pid'] not in harness.observer.snapshots[-1].participants
    harness.release('silent-capture-ended')
    completed(parent)
    evidence = finish(harness)
    assert not any(e['event'] == 'native_admitted' for e in evidence.events)
    if line:
        assert line_low(line)


@pytest.mark.noline
def test_native_audio_admission_empty_sabir_diagnostics(harness):
    observe(harness)
    harness.env['HFMODEM_COEXIST_FFMPEG_STDERR'] = 'coexist empty capture diagnostic'
    binary(harness, 'ffmpeg', '''
        import sys
        from hfmodem.tests.coexist.ffmpeg_stub import main
        raise SystemExit(main([*sys.argv[1:], '-t', '0']))
    ''')
    child = launch(harness, '''
        from hfmodem.sabir.monitor import device_chunks
        try:
            next(device_chunks('Coexist fake audio'))
        except SystemExit as exc:
            assert 'coexist empty capture diagnostic' in str(exc), str(exc)
        else:
            raise AssertionError('empty capture yielded audio')
    ''')
    completed(child)
    evidence = finish(harness)
    assert any(e['event'] == 'device_open' for e in evidence.events)
    assert not any(e['event'] == 'native_admitted' for e in evidence.events)


@pytest.mark.noline
def test_native_audio_admission_node_without_output(harness):
    if sys.platform != 'linux':
        pytest.skip('direct ALSA-node admission is a Linux scope; conductor runs hfmodem-ci')
    observe(harness)
    for point in ('ffmpeg_before_output', 'node-admitted'):
        harness.pin(point)
    child = launch(harness, f'''
        import array, fcntl, termios
        from hfmodem.core import native
        owned = claim.acquire('legacy', 'node-capture')
        native._observation = lambda argv, device: (({str(harness.audio)!r},), None, True)
        capture = native.spawn_native(['ffmpeg', '-f', 'alsa', '-i', 'Coexist fake audio',
            '-ac', '1', '-ar', '48000', '-t', '.1', '-f', 's16le', '-'], claim_fd=owned)
        count = array.array('i', [0])
        fcntl.ioctl(capture.stdout.fileno(), termios.FIONREAD, count, True)
        assert count[0] == 0
        coexist.checkpoint('node-admitted')
        assert capture.stdout.read() == bytes(9600)
        assert capture.wait_gone(5)
        for stream in (capture.stdin, capture.stdout, capture.stderr):
            stream.close()
        claim.release(owned)
    ''')
    event = harness.reached('ffmpeg_before_output', quiescent=True)
    harness.reached('node-admitted', quiescent=True)
    assert_claim_possession(harness.observer.snapshots[-1], pid=event['pid'],
        claim_path=harness.root / 'rig.claim',
        nonce=int((harness.root / 'rig.claim').read_text().split()[4]))
    assert any(e['event'] == 'native_admitted' for e in harness.events())
    harness.release('ffmpeg_before_output')
    harness.release('node-admitted')
    completed(child)
    finish(harness)
