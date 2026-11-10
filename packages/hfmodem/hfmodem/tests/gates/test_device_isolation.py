# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import errno
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from hfmodem.core import claim, winclaim
from hfmodem.tests import hermetic

POSIX_SERIAL = [pytest.param(port, marks=pytest.mark.posix_only('POSIX serial paths'))
                for port in ('/dev/tty', '/dev/ttys000', '/dev/ttyUSB0',
                             '/dev/cu.usbserial-radio', '/dev/pts/2')]
COM_SERIAL = ['COM1', 'com41', r'\\.\COM42']


@pytest.fixture
def device_guard(tmp_path, monkeypatch):
    path = hermetic.GUARD / 'hermetic_guard.py'
    spec = importlib.util.spec_from_file_location('device_guard_test', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.armed = True
    monkeypatch.setattr(guard, 'trip', lambda what: (_ for _ in ()).throw(PermissionError(what)))
    monkeypatch.setenv('HERMETIC_WORK', str(tmp_path))
    monkeypatch.setattr(claim, 'process_started', lambda pid: 'test incarnation')
    monkeypatch.setattr(winclaim, 'process_info', lambda pid, handle=None: ('test incarnation', False))
    yield guard
    guard.disarm()


def _owner_ended(guard, monkeypatch):
    if os.name == 'nt':
        # Ended, though a retained handle still names the same process.
        monkeypatch.setattr(winclaim, 'process_info', lambda pid, handle=None: ('test incarnation', True))
    else:
        guard.disarm()  # releases the lock that says the owner is alive


@pytest.mark.parametrize('port', POSIX_SERIAL + COM_SERIAL)
def test_unowned_serial_open_fails_before_any_device_access(device_guard, port):
    with pytest.raises(PermissionError, match='not owned by this test'):
        device_guard.guard('open', (port, 'r', 0))


@pytest.mark.skipif(os.name != 'nt', reason='Windows named pipe')
def test_guarded_createfile_leaves_a_single_instance_pipe_to_its_client(device_guard, monkeypatch):
    from hfmodem.core import winproc
    winproc._api()
    device_guard.install_serial_guards(patch=monkeypatch.setattr)
    reader, writer = winproc.pipe_pair()
    reader.close()
    winclaim.close_handle(writer)


@pytest.mark.posix_only('POSIX symlink to a terminal')
def test_symlink_to_foreign_terminal_is_refused(device_guard, tmp_path):
    alias = tmp_path / 'cat'
    alias.symlink_to('/dev/ttys000')
    with pytest.raises(PermissionError, match='/dev/ttys000'):
        device_guard.guard('open', (alias, 'r', 0))


@pytest.mark.posix_only('POSIX directory-relative terminal open')
@pytest.mark.parametrize('path', ['ttys000', b'ttys000'])
def test_directory_relative_foreign_terminal_is_refused(monkeypatch, path):
    directory = os.open('/dev', os.O_RDONLY | os.O_DIRECTORY)
    try:
        with monkeypatch.context() as local:
            original_open = os.open
            def reached(path, flags, mode=0o777, *, dir_fd=None):
                if dir_fd == directory or os.path.realpath(path) == '/dev/ttys000':
                    raise AssertionError('unguarded OS open')
                return original_open(path, flags, mode, dir_fd=dir_fd)
            local.setattr(hermetic.guard, 'trip',
                          lambda what: (_ for _ in ()).throw(PermissionError(what)))
            local.setattr(os, 'open', reached)
            hermetic.guard.install_serial_guards(local.setattr)
            with pytest.raises(PermissionError, match='/dev/ttys000'):
                os.open(path, os.O_RDONLY, dir_fd=directory)
    finally:
        os.close(directory)


@pytest.mark.posix_only('POSIX directory-relative owned PTY open')
@pytest.mark.parametrize('as_bytes', [False, True])
def test_directory_relative_owned_pty_requires_live_ownership(monkeypatch, as_bytes):
    master, slave = os.openpty()
    directory = None
    try:
        port = Path(os.ttyname(slave))
        # /dev itself, not Linux's /dev/pts: the guard refuses opening any /dev entry it does not own.
        directory = os.open('/dev', os.O_RDONLY | os.O_DIRECTORY)
        with monkeypatch.context() as local:
            original_open = os.open
            owned = True
            def reached(path, flags, mode=0o777, *, dir_fd=None):
                if not owned and (dir_fd == directory or os.path.realpath(path) == str(port)):
                    raise AssertionError('unguarded OS open')
                return original_open(path, flags, mode, dir_fd=dir_fd)
            local.setattr(hermetic.guard, 'trip',
                          lambda what: (_ for _ in ()).throw(PermissionError(what)))
            local.setattr(os, 'open', reached)
            hermetic.guard.install_serial_guards(local.setattr)
            relative = str(port.relative_to('/dev'))
            path = os.fsencode(relative) if as_bytes else relative
            opened = os.open(path, os.O_RDONLY | os.O_NOCTTY, dir_fd=directory)
            try:
                assert os.ttyname(opened) == str(port)
            finally:
                os.close(opened)
            hermetic.guard.closed_pty(master)
            hermetic.guard.closed_pty(slave)
            owned = False
            with pytest.raises(PermissionError, match='not owned by this test'):
                os.open(path, os.O_RDONLY | os.O_NOCTTY, dir_fd=directory)
    finally:
        if directory is not None:
            os.close(directory)
        os.close(master)
        os.close(slave)


@pytest.mark.posix_only('mocked POSIX PTY allocation')
def test_only_created_pty_is_allowed_until_both_ends_close(device_guard, monkeypatch):
    port = '/dev/ttys003'
    monkeypatch.setattr(device_guard.os, 'ttyname', lambda fd: port)
    monkeypatch.setattr(device_guard, '_device_identity', lambda path: [1, 2, 3, 4])
    device_guard.own_pty(100, 101)
    device_guard.guard('open', (port, 'r', 0))
    device_guard.closed_pty(101)
    device_guard.guard('open', (port, 'r', 0))
    device_guard.closed_pty(100)
    with pytest.raises(PermissionError):
        device_guard.guard('open', (port, 'r', 0))


@pytest.mark.posix_only('mocked POSIX PTY allocation')
def test_child_guard_reads_only_current_tests_device_manifest(device_guard, monkeypatch, tmp_path):
    port = '/dev/ttys003'
    monkeypatch.setattr(device_guard.os, 'ttyname', lambda fd: port)
    monkeypatch.setattr(device_guard, '_device_identity', lambda path: [1, 2, 3, 4])
    device_guard.own_pty(100, 101)
    device_guard._devices.clear()  # another interpreter has no local registration
    device_guard.check_serial(port)
    different_test = tmp_path / 'next-test'
    different_test.mkdir()
    monkeypatch.setenv('HERMETIC_WORK', str(different_test))
    with pytest.raises(PermissionError):
        device_guard.check_serial(port)


@pytest.mark.parametrize('change', ['node', 'owner'])
@pytest.mark.posix_only('mocked POSIX PTY allocation')
def test_reused_terminal_or_owner_does_not_inherit_permission(device_guard, monkeypatch, change):
    monkeypatch.setattr(device_guard.os, 'ttyname', lambda fd: '/dev/ttys003')
    identity = [1, 2, 3, 4]
    monkeypatch.setattr(device_guard, '_device_identity', lambda path: identity.copy())
    device_guard.own_pty(100, 101)
    if change == 'node':
        identity[-1] += 1
    else:
        _owner_ended(device_guard, monkeypatch)
        device_guard.armed = True
    with pytest.raises(PermissionError):
        device_guard.check_serial('/dev/ttys003')


def test_only_verified_software_com_port_is_allowed(device_guard, monkeypatch):
    monkeypatch.setattr(winclaim, 'serial_target', lambda port: r'\Device\com0com-test')
    device_guard.own_com('COM41', winclaim.serial_target('COM41'))
    device_guard.guard('open', (r'\\.\com41', 'r', 0))
    device_guard.check_command([r'C:\Hamlib\rigctl.exe', '-rCOM41'])
    with pytest.raises(ValueError, match='not a software serial device'):
        device_guard.own_com('COM1', r'\Device\Serial0')
    monkeypatch.setattr(winclaim, 'serial_target', lambda port: r'\Device\Serial0')
    with pytest.raises(PermissionError):
        device_guard.check_serial('COM41')


def test_win32_open_guard_preserves_ctypes_binding_and_refuses_before_driver(device_guard):
    calls = []
    def create_file(path, *args):
        calls.append(path)
        return 42
    checked = device_guard.SerialOpen(create_file)
    checked.argtypes, checked.restype = ['path'], 'handle'
    assert create_file.argtypes == checked.argtypes == ['path']
    assert create_file.restype == checked.restype == 'handle'
    with pytest.raises(PermissionError):
        checked(r'\\.\COM1', 0)
    assert calls == []
    assert checked('test-file', 0) == 42
    assert calls == ['test-file']


@pytest.mark.parametrize('event', ['subprocess.Popen', 'os.exec', 'os.posix_spawn'])
@pytest.mark.parametrize('trampoline', [False, True])
def test_native_cat_is_refused_before_process_creation(device_guard, event, trampoline):
    port = 'COM3' if os.name == 'nt' else '/dev/ttys000'
    argv = ['rigctl', '-P', 'NONE', '-m', '1036', '-r', port,
            'F', '7100000', 'M', 'PKTUSB']
    if trampoline:
        argv = ['python', '-m', 'hfmodem.core.native', '--exec', json.dumps(dict(argv=argv))]
    with pytest.raises(PermissionError, match=port):
        device_guard.guard(event, (argv[0], argv, None, None) if event == 'subprocess.Popen'
                           else (argv[0], argv, None))


@pytest.mark.parametrize('program', ['rigctl', 'rigctld', r'C:\Hamlib\bin\rigctl.exe'])
@pytest.mark.parametrize('port', POSIX_SERIAL + COM_SERIAL)
def test_native_cat_command_checks_posix_and_windows_devices(device_guard, program, port):
    with pytest.raises(PermissionError):
        device_guard.check_command([program, '-r', port])


@pytest.mark.posix_only('POSIX symlink to a terminal')
def test_native_cat_resolves_temp_alias_before_allowing_device(device_guard, tmp_path):
    alias = tmp_path / 'cat'
    alias.symlink_to('/dev/ttys000')
    with pytest.raises(PermissionError):
        device_guard.check_command(['rigctl', '--rig-file=' + str(alias)])


@pytest.mark.parametrize('command', [
    pytest.param(['sh', '-c', 'rigctl -r /dev/ttys000 2 0.05 7100000 PKTUSB'],
                 marks=pytest.mark.posix_only('POSIX shell and terminal path')),
    ['cmd.exe', '/c', r'rigctld.exe -r \\.\COM4'],
    r'"C:\Hamlib\rigctl.exe" -r COM4',
])
def test_shell_and_windows_command_line_cannot_hide_foreign_cat(device_guard, command):
    with pytest.raises(PermissionError):
        device_guard.check_command(command)


@pytest.mark.posix_only('mocked POSIX PTY allocation')
def test_native_cat_accepts_own_pty(device_guard, monkeypatch):
    monkeypatch.setattr(device_guard.os, 'ttyname', lambda fd: '/dev/ttys003')
    monkeypatch.setattr(device_guard, '_device_identity', lambda path: [1, 2, 3, 4])
    device_guard.own_pty(100, 101)
    device_guard.check_command(['rigctl', '-r', '/dev/ttys003'])


def test_native_cat_accepts_temporary_files(device_guard, tmp_path):
    fake = tmp_path / 'fake-cat'
    fake.touch()
    device_guard.check_command(['rigctld', '-r', str(fake)])
    fake = tmp_path / 'fake-COM41.dat'
    fake.touch()
    device_guard.check_command(['rigctl', '-r', str(fake)])


@pytest.mark.parametrize('endpoint', [
    pytest.param('-r/dev/ttys000', marks=pytest.mark.posix_only('POSIX terminal path')),
    '-rCOM3', '--rig-file=COM3'])
def test_attached_native_device_options_are_checked(device_guard, endpoint):
    with pytest.raises(PermissionError):
        device_guard.check_command(['rigctl', endpoint])


@pytest.mark.parametrize('library', ['kernel32.dll', r'C:\Windows\System32\kernel32.dll'])
def test_windows_install_checks_pyserial_and_cached_and_new_win32_apis(device_guard, monkeypatch, library):
    import ctypes
    calls = []
    class Serial:
        port = 'COM3'
        def open(self):
            calls.append('serial driver')
    def create_file(path):
        calls.append(path)
        return 42
    api = NS(CreateFileW=create_file, CreateFileA=create_file)
    monkeypatch.setattr(device_guard, 'os', NS(**dict(vars(os), name='nt')))
    monkeypatch.setitem(sys.modules, 'serial', NS(Serial=Serial))
    monkeypatch.setitem(sys.modules, 'hfmodem.core.winproc', NS(_api_cache=api))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: api, raising=False)
    device_guard.install_serial_guards(monkeypatch.setattr)
    with pytest.raises(PermissionError):
        Serial().open()
    with pytest.raises(PermissionError):
        api.CreateFileW(r'\\.\COM3')
    new_api = ctypes.WinDLL(library)
    with pytest.raises(PermissionError):
        new_api.CreateFileA(rb'\\.\COM3')
    assert calls == []
    assert new_api.CreateFileW('temporary-file') == 42
    assert calls == ['temporary-file']
    unconfigured = Serial()
    unconfigured.port = None
    unconfigured.open()
    assert calls == ['temporary-file', 'serial driver']


@pytest.mark.parametrize('program', ['rigctl', 'rigctld'])
@pytest.mark.parametrize('changed_cwd', [False, True])
def test_python_child_inherits_guard_and_refuses_native_spawn(tmp_path, monkeypatch, program, changed_cwd):
    # Run only Python. If the audit guard vanishes, a sentinel at the OS spawn
    # boundary fails the child; no Hamlib program or device is ever started.
    code = r'''
import os, subprocess, sys
def reached_os(*args, **kwargs):
    raise AssertionError('unguarded native process creation')
if os.name == 'nt':
    subprocess._winapi.CreateProcess = reached_os
else:
    subprocess._fork_exec = reached_os
    subprocess.Popen._posix_spawn = reached_os
try:
    port = 'COM3' if os.name == 'nt' else '/dev/ttys000'
    subprocess.Popen([sys.argv[1], '-P', 'NONE', '-m', '1036',
                      '-r', port, '2', '0.05', '7100000', 'PKTUSB'])
except PermissionError as exc:
    assert 'not owned by this test' in str(exc), str(exc)
    print('native spawn refused before OS boundary')
else:
    raise AssertionError('unguarded native process creation')
'''
    work = tmp_path / 'child-guard'
    work.mkdir()
    monkeypatch.setenv('HERMETIC_WORK', str(work))
    environment = dict(os.environ)
    # Keep this tree's guard even if the surrounding run changes PYTHONPATH.
    environment['PYTHONPATH'] = os.pathsep.join([str(hermetic.GUARD),
                                               str(Path(hermetic.__file__).parents[2])])
    result = subprocess.run([sys.executable, '-c', code, program], env=environment,
                            cwd=tmp_path if changed_cwd else None,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == 'native spawn refused before OS boundary'
    assert 'serial device is not owned by this test' in (work / 'trips').read_text()


def test_pty_wrapper_registers_only_fds_returned_by_allocator(monkeypatch):
    events = []
    monkeypatch.setattr(os, 'openpty', lambda: (100, 101), raising=False)
    monkeypatch.setattr(os, 'close', lambda fd: events.append(('close', fd)))
    monkeypatch.setattr(hermetic.guard, 'own_pty', lambda *fds: events.append(('own', fds)))
    monkeypatch.setattr(hermetic.guard, 'closed_pty', lambda fd: events.append(('closed', fd)))
    hermetic._serial_tripwires(monkeypatch)
    assert os.openpty() == (100, 101)
    os.close(100)
    assert events == [('own', (100, 101)), ('close', 100), ('closed', 100)]


def test_station_exports_written_during_test_do_not_leak(monkeypatch, tmp_path):
    # Exercise the fixture directly without installing a second audit hook.
    with monkeypatch.context() as local:
        local.setattr(hermetic, '_plant', lambda patch: set())
        local.setattr(hermetic, '_serial_tripwires', lambda patch: None)
        local.setattr(hermetic, 'guard', NS(armed=False, arm=lambda: None, disarm=lambda: None,
                                          admit=lambda *a: None, sweep=lambda *a: None))
        for key in hermetic.STATION:
            local.delenv(key, raising=False)
        fixture = hermetic.station_kept_out.__wrapped__(
            NS(node=NS(get_closest_marker=lambda *a: None), getfixturevalue=lambda name: tmp_path), local)
        next(fixture)
        os.environ.update(CAT_PORT='/dev/ttys000', PTT_PORT='COM1', AUDIO_IN='real card')
        with pytest.raises(StopIteration):
            next(fixture)
        assert not any(key in os.environ for key in hermetic.STATION)


def _os_spawn_sentinel(monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError('unguarded OS process creation')
    if os.name == 'nt':
        monkeypatch.setattr(subprocess._winapi, 'CreateProcess', reached)
    else:
        monkeypatch.setattr(subprocess, '_fork_exec', reached)
        monkeypatch.setattr(subprocess.Popen, '_posix_spawn', reached)


@pytest.mark.posix_only('POSIX relative terminal path')
@pytest.mark.parametrize('trampoline', [False, True])
def test_native_cat_relative_device_uses_child_cwd(monkeypatch, trampoline):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    argv = ['rigctl', '-r', 'ttys000']
    if trampoline:
        argv = [sys.executable, '-m', 'hfmodem.core.native', '--exec',
                json.dumps(dict(argv=argv))]
    with pytest.raises(PermissionError, match='/dev/ttys000'):
        subprocess.Popen(argv, cwd='/dev')


@pytest.mark.posix_only('POSIX relative symlink to a terminal')
@pytest.mark.parametrize('option', ['-r', '-r{}', '--rig-file={}'])
def test_native_cat_relative_alias_uses_child_cwd(monkeypatch, tmp_path, option):
    (tmp_path / 'cat').symlink_to('/dev/ttys000')
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    args = ['-r', 'cat'] if option == '-r' else [option.format('cat')]
    with pytest.raises(PermissionError, match='/dev/ttys000'):
        subprocess.Popen(['rigctl', *args], cwd=tmp_path)


@pytest.mark.posix_only('POSIX owned PTY at the spawn boundary')
def test_native_cat_relative_owned_pty_uses_child_cwd(monkeypatch):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    master, slave = os.openpty()
    try:
        port = Path(os.ttyname(slave))
        # From /dev: every native argv word is checked as an endpoint, and in
        # Linux's /dev/pts even '-r' names a would-be terminal.
        argv = ['rigctl', '-r', str(port.relative_to('/dev'))]
        with pytest.raises(AssertionError, match='unguarded OS process creation'):
            subprocess.Popen(argv, cwd='/dev')
        hermetic.guard.closed_pty(master)
        hermetic.guard.closed_pty(slave)
        with pytest.raises(PermissionError, match='not owned by this test'):
            subprocess.Popen(argv, cwd='/dev')
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.posix_only('POSIX executable lookup in the child directory')
@pytest.mark.parametrize('executable,path', [('./rigctl', ''), ('rigctl', '.'),
                                           ('rigctl', 'bin')])
def test_native_cat_relative_executable_uses_child_cwd(monkeypatch, tmp_path, executable, path):
    directory = tmp_path / path
    directory.mkdir(exist_ok=True)
    script = directory / 'rigctl'
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(0o700)
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen([executable], cwd=tmp_path, env=dict(os.environ, PATH=path))


@pytest.mark.parametrize('command', [
    ['sh', '-c', 'exec rigctl -r /dev/ttys000'],
    ['sh', '-c', 'env rigctl -r /dev/ttys000'],
    ['sh', '-c', 'rigctl -r "$CAT_PORT"'],
    ['bash', 'launcher with spaces.sh'],
    ['env', 'rigctl', '-r', '/dev/ttys000'],
    ['cmd.exe', '/c', 'rigctl -r COM3'],
    ['powershell.exe', '-Command', 'rigctl -r COM3'],
])
def test_opaque_wrappers_refused_at_actual_spawn_boundary(monkeypatch, command):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(command)


@pytest.mark.posix_only('POSIX shell syntax check at the spawn boundary')
@pytest.mark.parametrize('shell', ['/bin/bash', '/bin/sh'])
def test_shell_noexec_file_check_reaches_os_spawn(monkeypatch, tmp_path, shell):
    script = tmp_path / 'syntax only.sh'
    script.write_text('exec rigctl -r /dev/ttys000\n')
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(AssertionError, match='unguarded OS process creation'):
        subprocess.Popen([shell, '-n', str(script)])


@pytest.mark.posix_only('POSIX shell syntax input must not be a device')
@pytest.mark.parametrize('port', ['/dev/ttys000', '/dev/snd/pcmC0D0c'])
@pytest.mark.parametrize('alias', [False, True])
def test_shell_noexec_check_refuses_device_input(monkeypatch, tmp_path, port, alias):
    path = port
    if alias:
        (tmp_path / 'script').symlink_to(port)
        path = str(tmp_path / 'script')
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(['/bin/bash', '-n', path], cwd=tmp_path)


@pytest.mark.posix_only('POSIX shell script lookup through PATH stays refused')
def test_shell_noexec_exception_refuses_path_only_device_alias(monkeypatch, tmp_path):
    directory = tmp_path / 'bin'
    directory.mkdir()
    (directory / 'foreign.sh').symlink_to('/dev/ttys000')
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    environment = dict(os.environ, PATH=str(directory))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(['/bin/bash', '-n', 'foreign.sh'], cwd=tmp_path, env=environment)


@pytest.mark.posix_only('POSIX syntax checks require an absolute regular script')
@pytest.mark.parametrize('operand', ['relative', 'missing', 'directory'])
def test_shell_noexec_exception_refuses_other_operands(monkeypatch, tmp_path, operand):
    script = tmp_path / 'script.sh'
    script.write_text('exit 0\n')
    path = {'relative': script.name, 'missing': str(tmp_path / 'missing.sh'),
            'directory': str(tmp_path)}[operand]
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(['/bin/bash', '-n', path], cwd=tmp_path)


@pytest.mark.posix_only('POSIX syntax checks cannot select a login shell')
def test_shell_noexec_exception_refuses_login_argv(monkeypatch):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(['-bash', '-n', 'foreign.sh'], executable='/bin/bash')


@pytest.mark.posix_only('POSIX shell execution stays refused')
@pytest.mark.parametrize('arguments', [
    ['-c', 'exec rigctl -r /dev/ttys000'],
    ['-nc', 'exec rigctl -r /dev/ttys000'],
    ['-n', '-c', 'exec rigctl -r /dev/ttys000'],
    ['-c', 'exec rigctl -r /dev/ttys000', '-n'],
    ['-n', '+n'],
    ['-n', '+n', 'foreign.sh'],
    ['-n', 'foreign.sh', '-c', 'exec rigctl -r /dev/ttys000'],
])
def test_shell_noexec_exception_does_not_admit_execution(monkeypatch, arguments):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(['/bin/bash', *arguments])


@pytest.mark.posix_only('POSIX shell startup files stay refused')
@pytest.mark.parametrize('variable', ['BASH_ENV', 'ENV'])
def test_shell_noexec_exception_refuses_startup_files(monkeypatch, tmp_path, variable):
    script = tmp_path / 'script.sh'
    script.write_text('exit 0\n')
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    environment = dict(os.environ, **{variable: '/foreign/startup.sh'})
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen(['/bin/bash', '-n', str(script)], env=environment)


@pytest.mark.posix_only('POSIX shell binary required for syntax checks')
def test_shell_noexec_exception_refuses_a_python_script(monkeypatch, tmp_path):
    shell = tmp_path / 'bash'
    shell.write_text(f'#!{sys.executable}\nraise AssertionError("executed fake shell")\n')
    shell.chmod(0o700)
    script = tmp_path / 'script.sh'
    script.write_text('exit 0\n')
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        subprocess.Popen([str(shell), '-n', str(script)])


@pytest.mark.parametrize('name,body', [
    pytest.param('rigctl', '#!/bin/sh\nexec env rigctl -r /dev/ttys000\n',
                 marks=pytest.mark.posix_only('Windows runs no extensionless #! script')),
    pytest.param('rigctl.cmd', '@rigctl -r COM3\r\n',
                 marks=pytest.mark.skipif(os.name != 'nt', reason='PATHEXT resolves a bare name to a batch file')),
])
def test_native_script_cannot_masquerade_as_supported_tool(device_guard, tmp_path, name, body):
    script = tmp_path / name
    script.write_text(body)
    script.chmod(0o700)
    launched = str(tmp_path / 'rigctl')
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.check_launch(launched, [launched], os.environ)


@pytest.mark.parametrize('escape', ['env', 'path', 'work', 'tag', '-I', '-E', '-S', '-sES'])
def test_unprotected_python_startup_refused_before_os_spawn(monkeypatch, escape):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    environment = dict(os.environ)
    argv = [sys.executable]
    if escape == 'env':
        environment = {}
    elif escape in ('path', 'work', 'tag'):
        environment.pop({'path': 'PYTHONPATH', 'work': 'HERMETIC_WORK', 'tag': 'HERMETIC_TAG'}[escape])
    else:
        argv.append(escape)
    argv.extend(['-c', 'raise AssertionError("unprotected child started")'])
    with pytest.raises(PermissionError, match='unprotected Python'):
        subprocess.Popen(argv, env=environment)


def test_explicit_site_free_probe_admission_requires_exact_argv_and_devices(device_guard):
    command = [sys.executable, '-S', 'tools/console.py']
    environment = {'CAT_PORT': '', 'PTT_PORT': ''}
    with pytest.raises(PermissionError, match='unprotected Python startup'):
        device_guard.check_launch(sys.executable, command, environment)
    device_guard.admit(command, environment)
    device_guard.check_launch(sys.executable, command, environment)
    for argv, env, executable in [
            (command + ['extra'], environment, sys.executable),
            (command[:2] + ['tools/other.py'], environment, sys.executable),
            ([sys.executable, '-I', command[-1]], environment, sys.executable),
            (command, dict(environment, CAT_PORT='/dev/ttys000'), sys.executable),
            (command, dict(environment, PTT_PORT='COM3'), sys.executable),
            (command, environment, '/other/python')]:
        with pytest.raises(PermissionError):
            device_guard.check_launch(executable, argv, env)


def test_exact_admission_refuses_a_longer_argv_that_a_prefix_admits(device_guard):
    environment = {'CAT_PORT': '', 'PTT_PORT': ''}
    device_guard.admit(['ldd', '/bin/rigctld'], environment, exact=True)
    device_guard.admit(['lsb_release'], environment)
    device_guard.check_launch('ldd', ['ldd', '/bin/rigctld'], environment)
    device_guard.check_launch('lsb_release', ['lsb_release', '-a'], environment)
    for argv in (['ldd', '/bin/rigctld', '/dev/ttyUSB0'], ['ldd']):
        with pytest.raises(PermissionError, match='opaque native launch refused'):
            device_guard.check_launch('ldd', argv, environment)


def _strace(tmp_path, *argv, options=()):
    return ['strace', '-f', '-ttt', '-yy', '-s', '4096', '--seccomp-bpf',
            '-e', 'trace=%process,%desc,%file,flock,ioctl', *options,
            '-o', str(tmp_path / 'trace'), '--', *argv]


@pytest.mark.posix_only('the Linux coexist observer launches through strace')
def test_strace_launch_is_checked_as_the_command_it_runs(device_guard, tmp_path):
    device_guard.check_launch('strace', _strace(tmp_path, sys.executable, '-c', 'pass'), os.environ)
    stripped = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    for argv, env, refusal in [
            ([sys.executable, '-c', 'pass'], stripped, 'unprotected Python environment'),
            ([sys.executable, '-I', '-c', 'pass'], os.environ, 'unprotected Python startup'),
            (['/bin/bash', '-c', 'true'], os.environ, 'opaque native launch refused: bash'),
            (['rigctl', '-r', '/dev/ttys000'], os.environ, 'not owned by this test'),
            (_strace(tmp_path, '/bin/sh', '-c', 'true'), os.environ, 'opaque native launch refused: sh')]:
        with pytest.raises(PermissionError, match=refusal):
            device_guard.check_launch('strace', _strace(tmp_path, *argv), env)
    device_guard.admit(['/bin/bash', '-c', 'true'], os.environ)
    device_guard.check_launch('strace', _strace(tmp_path, '/bin/bash', '-c', 'true'), os.environ)


@pytest.mark.posix_only('the Linux coexist observer launches through strace')
@pytest.mark.parametrize('options,refusal', [
    (['-E', 'PYTHONPATH='], 'strace option refused: -E'),
    (['-u', 'root'], 'strace option refused: -u'),
    (['-p', '1'], 'strace option refused: -p'),
    (['-e', 'inject=open:retval=0'], 'strace option refused: -e'),
    (['-o', '|sh'], 'strace option refused: -o'),
    (['-o', '!sh'], 'strace option refused: -o'),
    (['-o', '/dev/ttys000'], 'not owned by this test'),
])
def test_strace_options_that_change_the_traced_launch_are_refused(device_guard, tmp_path, options, refusal):
    argv = _strace(tmp_path, sys.executable, '-c', 'pass')
    argv[-4:-4] = options
    with pytest.raises(PermissionError, match=refusal):
        device_guard.check_launch('strace', argv, os.environ)


@pytest.mark.posix_only('the Linux coexist observer launches through strace')
def test_strace_needs_a_command_and_a_real_strace(device_guard, tmp_path):
    for argv in (['strace', '-f', sys.executable, '-c', 'pass'], ['strace', '-f', '--']):
        with pytest.raises(PermissionError, match='strace'):
            device_guard.check_launch('strace', argv, os.environ)
    script = tmp_path / 'strace'
    script.write_text('#!/bin/sh\nexec "$@"\n')
    script.chmod(0o700)
    argv = _strace(tmp_path, sys.executable, '-c', 'pass')
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.check_launch(str(script), [str(script), *argv[1:]], os.environ)


@pytest.mark.posix_only('POSIX interpreter lines')
def test_tool_stand_in_script_runs_only_when_its_real_path_is_admitted(device_guard, tmp_path):
    stand_in = tmp_path / 'bin' / 'ffmpeg'
    stand_in.parent.mkdir()
    stand_in.write_text('#!/bin/sh\nexit 0\n')
    stand_in.chmod(0o700)
    env = dict(os.environ, PATH=str(stand_in.parent))
    argv = ['ffmpeg', '-f', 'alsa', '-i', 'default', '-']
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.check_launch('ffmpeg', argv, env)
    device_guard.admit(['ffmpeg'], env)
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.check_launch('ffmpeg', argv, env)
    device_guard.admit([os.path.realpath(stand_in)], env)
    device_guard.check_launch('ffmpeg', argv, env)
    with pytest.raises(PermissionError, match='not owned by this test'):
        device_guard.check_launch('ffmpeg', ['ffmpeg', '-i', '/dev/ttys000'], env)
    rigctl = stand_in.with_name('rigctl')
    rigctl.write_text('#!/bin/sh\nexit 0\n')
    rigctl.chmod(0o700)
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.check_launch('rigctl', ['rigctl'], env)


def test_exactly_admitted_python_child_may_start_without_the_guard_path(device_guard):
    command = [sys.executable, '-c', 'import sounddevice']
    environment = {'PYTHONPATH': '/elsewhere', 'CAT_PORT': '', 'PTT_PORT': ''}
    with pytest.raises(PermissionError, match='unprotected Python environment'):
        device_guard.check_launch(sys.executable, command, environment)
    device_guard.admit(command, environment)
    device_guard.check_launch(sys.executable, command, environment)
    for argv, env, executable, refusal in [
            (command + ['extra'], environment, sys.executable, 'unprotected Python environment'),
            ([sys.executable, '-c', 'pass'], environment, sys.executable, 'unprotected Python environment'),
            ([sys.executable, '-I', *command[1:]], environment, sys.executable, 'unprotected Python startup'),
            (command, dict(environment, CAT_PORT='/dev/ttys000'), sys.executable, 'unprotected Python environment'),
            (command, environment, '/other/python', 'unprotected Python interpreter')]:
        with pytest.raises(PermissionError, match=refusal):
            device_guard.check_launch(executable, argv, env)


@pytest.mark.parametrize('relative', ['startup', 'ahead', 'empty'])
def test_relative_startup_paths_refused_before_spawn_with_changed_cwd(monkeypatch, tmp_path, relative):
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    startup = str(hermetic.GUARD)
    if relative == 'startup':
        paths = [os.path.relpath(startup)]
    else:
        ahead = tmp_path / 'packages'
        ahead.mkdir()
        paths = [os.path.relpath(ahead) if relative == 'ahead' else '', startup]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    with pytest.raises(PermissionError, match='unprotected Python environment'):
        subprocess.Popen([sys.executable, '-c', 'pass'], env=environment, cwd=tmp_path)


@pytest.mark.parametrize('event', ['subprocess.Popen', 'os.exec', 'os.posix_spawn'])
def test_python_environment_is_checked_on_every_exec_boundary(device_guard, event):
    argv = [sys.executable, '-c', 'pass']
    args = (sys.executable, argv, None, {}) if event == 'subprocess.Popen' else (sys.executable, argv, {})
    with pytest.raises(PermissionError, match='unprotected Python environment'):
        device_guard.guard(event, args)


@pytest.mark.parametrize('options', [['-X', 'utf8', '-I'], ['-W', 'ignore', '-S']])
def test_startup_option_values_cannot_hide_isolation_flags(device_guard, options):
    with pytest.raises(PermissionError, match='unprotected Python startup'):
        device_guard.check_launch(sys.executable, [sys.executable, *options, '-c', 'pass'], os.environ)


def test_quoted_python_arguments_and_paths_are_preserved(device_guard, tmp_path):
    argv = [sys.executable, '-c', 'print("quoted value")', str(tmp_path / 'path with spaces')]
    device_guard.check_launch(sys.executable, argv, os.environ)
    # Windows Popen's audit event leaves executable=None for normal argv calls.
    device_guard.check_launch(None, argv, os.environ)
    # Flags in program arguments do not control interpreter initialization.
    device_guard.check_launch(sys.executable, [sys.executable, '-c', 'pass', '-I'], os.environ)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('entry,port', [('CreateFileW', r'\\.\COM3'),
                                       ('CreateFileA', rb'\\.\COM3')])
def test_windll_loader_and_cached_functions_refuse_before_driver(
        device_guard, monkeypatch, cached, entry, port):
    import ctypes
    calls = []
    def create_file(path):
        calls.append(path)
        return 42
    class Loader:
        def __init__(self):
            self._dlltype = lambda *a, **kw: NS(CreateFileW=create_file, CreateFileA=create_file)
        def __getattr__(self, name):
            api = self._dlltype(name)
            setattr(self, name, api)
            return api
        def __getitem__(self, name):
            return self._dlltype(name)
    loader = Loader()
    if cached:
        loader.kernel32
    monkeypatch.setattr(device_guard, 'os', NS(**dict(vars(os), name='nt')))
    monkeypatch.setitem(sys.modules, 'serial', NS(Serial=type('Serial', (), {'open': lambda self: None})))
    monkeypatch.setattr(ctypes, 'WinDLL', loader._dlltype, raising=False)
    monkeypatch.setattr(ctypes, 'windll', loader, raising=False)
    device_guard.install_serial_guards(monkeypatch.setattr)
    for api in (loader.kernel32, loader['kernel32'], ctypes.WinDLL('kernel32')):
        function = getattr(api, entry)
        function.argtypes, function.restype = ['path'], 'handle'
        with pytest.raises(PermissionError):
            function(port)
        assert function('temporary file') == 42
    assert calls == ['temporary file'] * 3


@pytest.mark.parametrize('cached', ['new', 'windll', 'winclaim', 'winproc'])
@pytest.mark.parametrize('lookup', ['str', 'bytes', 'str-nul', 'bytes-nul', 'ordinal'])
@pytest.mark.parametrize('entry,port', [('CreateFileW', r'\\.\COM3'),
                                       ('CreateFileA', rb'\\.\COM3')])
def test_indexed_windows_api_lookup_refuses_before_driver(device_guard, monkeypatch, cached, lookup, entry, port):
    import ctypes
    calls = []
    class API:
        def __init__(self):
            self.CreateFileW = self['CreateFileW']
            self.CreateFileA = self['CreateFileA']
        def __getitem__(self, name):
            # Like ctypes, indexed lookup constructs a fresh function pointer.
            def function(path):
                calls.append((name, path))
                return 42
            return function
    api = API()
    monkeypatch.setattr(device_guard, 'os', NS(**dict(vars(os), name='nt')))
    monkeypatch.setitem(sys.modules, 'serial', NS(Serial=type('Serial', (), {'open': lambda self: None})))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: API(), raising=False)
    monkeypatch.setattr(ctypes, 'windll', NS(_dlltype=ctypes.WinDLL), raising=False)
    if cached == 'windll':
        ctypes.windll.kernel32 = api
    elif cached in ('winclaim', 'winproc'):
        monkeypatch.setitem(sys.modules, f'hfmodem.core.{cached}', NS(_api_cache=api))
    device_guard.install_serial_guards(monkeypatch.setattr)
    if cached == 'new':
        api = ctypes.WinDLL('kernel32')
    name = entry.encode() if lookup.startswith('bytes') else entry
    if lookup.endswith('-nul'):
        name += b'\0suffix' if isinstance(name, bytes) else '\0suffix'
    elif lookup == 'ordinal':
        name = 123
    for function in (getattr(api, entry), api[name], api[name]):
        function.argtypes, function.restype = ['path'], 'handle'
        assert function.argtypes == ['path'] and function.restype == 'handle'
        with pytest.raises(PermissionError):
            function(port)
        assert calls == []
    assert api[name]('temporary file') == 42
    assert calls == [(name, 'temporary file')]
    assert api['UnrelatedAPI']('ordinary argument') == 42


def test_windows_command_line_uses_os_parser_and_releases_argv(device_guard, monkeypatch):
    import ctypes
    words = [r'C:\Program Files\Python\python.exe', '-c', 'print("quoted value")',
             r'C:\fake device\cat']
    released = []
    def parse(command, count):
        assert command == 'opaque Windows command line'
        count._obj.value = len(words)
        return words
    def free(result):
        released.append(result)
    monkeypatch.setattr(device_guard, 'os', NS(**dict(vars(os), name='nt')))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda name, **kw: (
        NS(CommandLineToArgvW=parse) if name == 'shell32' else NS(LocalFree=free)), raising=False)
    assert device_guard._command_words('opaque Windows command line') == words
    assert released == [words]


@pytest.mark.posix_only('mocked POSIX PTY allocation')
def test_shell_launch_runs_only_as_admitted_with_owned_devices(device_guard, monkeypatch):
    owned = '/dev/ttys003'
    monkeypatch.setattr(device_guard.os, 'ttyname', lambda fd: owned)
    monkeypatch.setattr(device_guard, '_device_identity', lambda path: [1, 2, 3])
    device_guard.own_pty(100, 101)
    command = ['/bin/bash', 'tools/onair.sh', 'check']
    environment = dict(os.environ, CAT_PORT=owned, PTT_PORT='')
    def launch(argv=command, env=environment, executable=None):
        device_guard.guard('subprocess.Popen', (executable or argv[0], argv, None, env))
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        launch()
    with pytest.raises(PermissionError, match='/dev/ttys000'):
        device_guard.admit(command[:2], dict(environment, CAT_PORT='/dev/ttys000'))
    device_guard.admit(command[:2], environment)
    launch()
    launch([*command, '--capture-prepared'])
    for argv, env, executable in [
            (['/bin/bash', 'tools/other.sh', 'check'], environment, None),
            (['/bin/bash', '-c', 'rigctl -r /dev/ttys000'], environment, None),
            (command[:1], environment, None),
            (command, dict(environment, CAT_PORT='/dev/ttys000'), None),
            (command, dict(environment, PTT_PORT=owned), None),
            (command, environment, '/bin/sh')]:
        with pytest.raises(PermissionError):
            launch(argv, env, executable)
    device_guard.closed_pty(100)
    device_guard.closed_pty(101)
    with pytest.raises(PermissionError, match='not owned by this test'):
        launch()


@pytest.mark.skipif(os.name != 'nt', reason='Windows Popen audits a command-line string')
def test_admitted_launch_runs_from_the_windows_command_line(device_guard, tmp_path):
    command = [str(tmp_path / 'hfmodem.exe'), 'console']
    environment = dict(os.environ, CAT_PORT='', PTT_PORT='')
    line = subprocess.list2cmdline(command)
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.guard('subprocess.Popen', (None, line, None, environment))
    device_guard.admit(command, environment)
    device_guard.guard('subprocess.Popen', (None, line, None, environment))


@pytest.mark.posix_only('the native trampoline execs only on POSIX')
def test_trampoline_admits_only_the_program_its_exec_will_name(device_guard, tmp_path):
    def trampoline(argv):
        return [sys.executable, '-m', 'hfmodem.core.native', '--exec', json.dumps(dict(argv=argv))]
    stand_in = tmp_path / 'capture'
    stand_in.write_text('#!/bin/sh\n')
    stand_in.chmod(0o700)
    for argv in (['true'], [str(stand_in), '-i', 'fake']):
        device_guard.admit(argv, os.environ)
    with pytest.raises(PermissionError, match='opaque native launch refused'):
        device_guard.check_command(trampoline(['true']))
    device_guard.check_command(trampoline([str(stand_in), '-i', 'fake']))
    device_guard.guard('os.exec', (str(stand_in), [str(stand_in), '-i', 'fake'], os.environ))


@pytest.mark.posix_only('POSIX shell at the spawn boundary')
def test_admitted_shell_reaches_os_spawn_only_for_its_test(monkeypatch, tmp_path):
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    master, slave = os.openpty()
    try:
        port = os.ttyname(slave)
        command = ['/bin/sh', '-c', 'exit 0']
        hermetic.guard.admit(command, {'CAT_PORT': port})
        _os_spawn_sentinel(monkeypatch)
        with pytest.raises(AssertionError, match='unguarded OS process creation'):
            subprocess.Popen(command, env=dict(os.environ, CAT_PORT=port))
        with pytest.raises(PermissionError, match='opaque native launch refused'):
            subprocess.Popen(['/bin/sh', '-c', 'exit 1'], env=dict(os.environ, CAT_PORT=port))
        # Another test's work directory holds neither the admission nor the device.
        monkeypatch.setenv('HERMETIC_WORK', str(tmp_path))
        with pytest.raises(PermissionError):
            subprocess.Popen(command, env=dict(os.environ, CAT_PORT=port))
    finally:
        os.close(master)
        os.close(slave)


def test_venv_base_interpreter_is_this_interpreter(device_guard, monkeypatch, tmp_path):
    base = tmp_path / 'python-base.exe'
    base.write_bytes(b'\x7fELF')
    base.chmod(0o700)
    monkeypatch.setattr(sys, '_base_executable', str(base))
    device_guard.check_launch(str(base), [str(base), '-c', 'pass'], os.environ)
    other = tmp_path / 'python3.exe'
    other.write_bytes(b'\x7fELF')
    other.chmod(0o700)
    with pytest.raises(PermissionError, match='unprotected Python interpreter'):
        device_guard.check_launch(str(other), [str(other), '-c', 'pass'], os.environ)


@pytest.mark.posix_only('POSIX interpreter lines')
@pytest.mark.parametrize('line,refused', [
    ('#!{python}\n', False), ('#!{python} -I\n', True), ('#!/bin/sh\n', True)])
def test_script_runs_only_under_this_interpreter_and_guard(device_guard, tmp_path, line, refused):
    script = tmp_path / 'lsof'
    script.write_text(line.format(python=sys.executable) + 'pass\n')
    script.chmod(0o700)
    if refused:
        with pytest.raises(PermissionError, match='opaque native launch refused'):
            device_guard.check_launch(str(script), [str(script)], os.environ)
        return
    device_guard.check_launch(str(script), [str(script), '-t', 'x'], os.environ)
    with pytest.raises(PermissionError, match='unprotected Python environment'):
        device_guard.check_launch(str(script), [str(script)], {'PATH': os.environ['PATH']})


def _socket_driver(device_guard, monkeypatch, *, fail=False):
    class Socket:
        def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, *, dual=False):
            self.family, self.type, self.dual = family, type, dual
        def getsockopt(self, level, option):
            if level == socket.SOL_SOCKET and option == socket.SO_TYPE:
                return self.type
            assert (level, option) == (socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
            return int(not self.dual)
        def bind(self, address):
            # The real audit event occurs BEFORE the OS accepts or rejects bind.
            device_guard.guard('socket.bind', (self, address))
            if fail:
                raise OSError(errno.EADDRINUSE, 'address already in use')
            self.address = (address[0], address[1] or 8515)
        def getsockname(self):
            return self.address
        def close(self):
            pass
        def detach(self):
            return 100
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    monkeypatch.setattr(socket, 'socket', Socket)
    device_guard.install_serial_guards(monkeypatch.setattr)
    return Socket()


@pytest.mark.parametrize('family,transport,host,dual', [
    (socket.AF_INET, socket.SOCK_DGRAM, '127.0.0.1', False),
    (socket.AF_INET, socket.SOCK_DGRAM, '0.0.0.0', False),
    (socket.AF_INET6, socket.SOCK_STREAM, '::1', False),
    (socket.AF_INET6, socket.SOCK_STREAM, '::', False),
    (socket.AF_INET6, socket.SOCK_STREAM, '::1', True),
    (socket.AF_INET, socket.SOCK_STREAM, '127.0.0.2', False),
])
def test_other_endpoint_on_same_port_does_not_authorize_tcp_ipv4(
        device_guard, monkeypatch, family, transport, host, dual):
    monkeypatch.setattr(device_guard, '_listened', lambda *a: True)
    _socket_driver(device_guard, monkeypatch)
    with socket.socket(family, transport, dual=dual) as server:
        server.bind((host, 4532))
        client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
        with pytest.raises(PermissionError):
            device_guard.guard('socket.connect', (client, ('127.0.0.1', 4532)))


@pytest.mark.parametrize('family,host,dual,destination', [
    (socket.AF_INET, '127.0.0.1', False, '127.0.0.1'),
    (socket.AF_INET, '0.0.0.0', False, '127.0.0.1'),
    (socket.AF_INET6, '::1', False, '::1'),
    (socket.AF_INET6, '::', False, '::1'),
    (socket.AF_INET6, '::', True, '127.0.0.1'),
    (socket.AF_INET6, '::ffff:127.0.0.1', True, '127.0.0.1'),
])
def test_owned_listener_endpoint_stays_reachable(
        device_guard, monkeypatch, family, host, dual, destination):
    monkeypatch.setattr(device_guard, '_listened', lambda *a: True)
    _socket_driver(device_guard, monkeypatch)
    with socket.socket(family, dual=dual) as server:
        server.bind((host, 4532))
        client_family = socket.AF_INET6 if destination == '::1' else socket.AF_INET
        client = NS(family=client_family, type=socket.SOCK_STREAM)
        device_guard.guard('socket.connect', (client, (destination, 4532)))


def test_failed_bind_does_not_grant_access_to_foreign_listener(device_guard, monkeypatch):
    monkeypatch.setattr(device_guard, '_listened', lambda *a: True)
    address = ('127.0.0.1', 4532)
    server = _socket_driver(device_guard, monkeypatch, fail=True)
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    with pytest.raises(PermissionError):
        device_guard.guard('socket.connect', (client, address))
    with pytest.raises(OSError) as failed:
        server.bind(address)
    assert failed.value.errno == errno.EADDRINUSE
    with pytest.raises(PermissionError):
        device_guard.guard('socket.connect', (client, address))
    assert not device_guard._served(client, address)


@pytest.mark.parametrize('ending', ['close', 'detach', 'collect'])
def test_closed_listener_does_not_grant_access_to_replacement(device_guard, monkeypatch, ending):
    monkeypatch.setattr(device_guard, '_listened', lambda *a: True)
    address = ('127.0.0.1', 8515)
    server = _socket_driver(device_guard, monkeypatch)
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    server.bind(address)
    device_guard.guard('socket.connect', (client, address))
    if ending == 'collect':
        import gc
        del server
        gc.collect()
    else:
        getattr(server, ending)()
        server.close()  # repeated close must not remove another socket's grant
    # The owning process is still alive, but a foreign listener replaced ours.
    with pytest.raises(PermissionError):
        device_guard.guard('socket.connect', (client, address))
    assert not device_guard._served(client, address)


def test_ephemeral_bind_ownership_ends_with_socket_context(device_guard, monkeypatch):
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    with _socket_driver(device_guard, monkeypatch) as server:
        server.bind(('127.0.0.1', 0))
        address = server.getsockname()
        assert device_guard._served(client, address)
    assert not device_guard._served(client, address)


def test_closing_one_socket_keeps_only_the_other_sockets_port_grant(device_guard, monkeypatch):
    first = _socket_driver(device_guard, monkeypatch)
    second = socket.socket()
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    address = ('127.0.0.1', 8515)
    first.bind(address)
    second.bind(address)
    first.close()
    first.close()
    assert device_guard._served(client, address)
    second.close()
    assert not device_guard._served(client, address)


@pytest.mark.parametrize('event', ['os.exec', 'subprocess.Popen'])
def test_owned_rigctld_registers_only_when_this_process_becomes_it(device_guard, monkeypatch, event):
    argv = ['rigctld', '-m', '1', '-t', '4532']
    monkeypatch.setattr(device_guard, 'check_launch', lambda *args: None)
    monkeypatch.setattr(device_guard, '_listened', lambda *address: True)
    device_guard.guard(event, (argv[0], argv, None, None) if event == 'subprocess.Popen' else (argv[0], argv, None))
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    connect = lambda: device_guard.guard('socket.connect', (client, ('127.0.0.1', 4532)))
    if event == 'os.exec':
        connect()
    else:
        with pytest.raises(PermissionError, match='connected to 127.0.0.1:4532'):
            connect()


def _windows_creation(device_guard, monkeypatch, command, create_process, environment=None, **api):
    import ctypes
    if os.name != 'nt':
        # Only the Windows install path is faked here; its command-line parser is not.
        command = command.split()
        monkeypatch.setattr(device_guard, 'os', NS(**dict(vars(os), name='nt')))
        monkeypatch.setitem(sys.modules, 'serial', NS(Serial=type('Serial', (), {'open': lambda self: None})))
        monkeypatch.setitem(sys.modules, 'hfmodem.core.winproc', NS(_api_cache=None))
        monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: NS(), raising=False)
    monkeypatch.setattr(subprocess, '_winapi', NS(CreateProcess=create_process, **api), raising=False)
    device_guard.install_serial_guards(monkeypatch.setattr)
    return lambda: subprocess._winapi.CreateProcess(None, command, None, None, 0, 4, environment, None, None)


@pytest.mark.parametrize('command,owned', [
    ('rigctld.exe -m 1 -t 4532', True), ('rigctld.exe -m 1 -t 4533', False),
    ('rigctld.exe -m 1 -t4532 --port=4533', False), ('rigctld.exe -m 1 -T 127.0.0.1 -T 10.0.0.1', False),
    ('rigctl.exe -m 1', False)])
def test_windows_created_rigctld_endpoint_lives_as_long_as_that_process(device_guard, monkeypatch, command, owned):
    # The native supervisor that creates rigctld exits once it runs; the
    # endpoint must follow the rigctld process, not its creator.
    running = {4242}
    monkeypatch.setattr(winclaim, 'process_info',
                        lambda pid, handle=None: (f'win:{pid}', pid not in running))
    monkeypatch.setattr(device_guard, '_listened', lambda *address: True)
    create = _windows_creation(device_guard, monkeypatch, command, lambda *args: (7, 8, 4242, 9))
    assert create()[2] == 4242
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    connect = lambda: device_guard.guard('socket.connect', (client, ('127.0.0.1', 4532)))
    if owned:
        connect()
    else:
        with pytest.raises(PermissionError, match='connected to 127.0.0.1:4532'):
            connect()
    # Exited, though a retained handle still names the same process.
    running.clear()
    with pytest.raises(PermissionError, match='connected to 127.0.0.1:4532'):
        connect()


@pytest.mark.parametrize('argv,host', [
    (['rigctld', '-m', '1', '-T', '127.0.0.1', '-t', '4532'], '127.0.0.1'),
    (['rigctld', '-r', '/dev/ttys000', '-vt4532'], '127.0.0.1'),
    (['rigctld', '-t4533', '--port=4532'], '127.0.0.1'),
    (['rigctld', '--listen=::1'], '::1'),
    (['rigctld', '--', '-t4533'], '127.0.0.1'),
    (['rigctld', '-m', '1', '-t4532', '-vt4533'], None),
    (['rigctld', '--por=4533'], None),
    (['rigctld', '-vT192.0.2.1'], '192.0.2.1'),
    (['rigctld', '--p=4532'], None),
    (['rigctld', '-b'], None),
    (['rigctld', '-q'], None),
    (['rigctld', '-t'], None),
    (['rigctld', '-t4533', 'stray', '-t4532'], '127.0.0.1'),
    (['rigctld', '-T192.0.2.1', 'stray', '-T127.0.0.1'], '127.0.0.1'),
    (['rigctld', '--list'], None),
    (['rigctld', '--li'], None),
    (['rigctld', '-l'], None),
    (['rigctld', '--dump-caps', '-m', '1'], None),
    (['rigctld', '-m1', '-u'], None),
    (['rigctld', '-vu'], None),
    (['rigctld', '--dump'], None),
    (['rigctld', '-h'], None),
    (['rigctld', '--he'], None),
    (['rigctld', '-V'], None),
    (['rigctld', '--vers'], None),
    (['rigctld', '-m', '1', '-L'], None),
    (['rigctld', '-vL'], None),
    (['rigctld', '--show-conf'], None),
    (['rigctld', '--show'], None)])
def test_owned_rigctld_endpoint_is_read_as_rigctld_reads_its_options(device_guard, argv, host):
    endpoint = device_guard._owned_rigctld(None, argv, {})
    assert (endpoint and endpoint[2]) == host


@pytest.mark.parametrize('argv,host', [
    (['rigctld', '-t4533', 'stray', '-t4532'], None),
    (['rigctld', '-t4532', 'stray', '-t4533'], '127.0.0.1'),
    (['rigctld', '-T192.0.2.1', 'stray', '-T127.0.0.1'], '192.0.2.1'),
    (['rigctld', 'stray', '--list'], '127.0.0.1'),
    (['rigctld', '-', '-b'], '127.0.0.1')])
def test_owned_rigctld_options_end_at_the_first_operand_when_posixly_correct(device_guard, argv, host):
    # With POSIXLY_CORRECT, rigctld's getopt_long stops at the first operand.
    endpoint = device_guard._owned_rigctld(None, argv, {'POSIXLY_CORRECT': '1'})
    assert (endpoint and endpoint[2]) == host


@pytest.mark.parametrize('event,environment,owned', [
    ('os.exec', {}, True), ('os.exec', {'POSIXLY_CORRECT': ''}, False),
    ('os.exec', {'posixly_correct': '1'}, True),
    ('CreateProcess', {}, True), ('CreateProcess', {'POSIXLY_CORRECT': ''}, False),
    ('CreateProcess', {'posixly_Correct': '1'}, False)])
def test_owned_rigctld_reads_its_options_in_the_childs_environment(device_guard, monkeypatch,
                                                                   event, environment, owned):
    # The parent has no POSIXLY_CORRECT; the child's own environment decides,
    # with Windows reading its keys without regard to case.
    monkeypatch.delenv('POSIXLY_CORRECT', raising=False)
    monkeypatch.setattr(device_guard, '_listened', lambda *address: True)
    command = 'rigctld.exe -m 1 -t4533 stray -t4532'
    if event == 'os.exec':
        if os.name == 'nt':
            pytest.skip('Windows has no exec')
        monkeypatch.setattr(device_guard, 'check_launch', lambda *args: None)
        argv = command.split()
        device_guard.guard(event, (argv[0], argv, environment))
    else:
        monkeypatch.setattr(winclaim, 'process_info', lambda pid, handle=None: (f'win:{pid}', False))
        _windows_creation(device_guard, monkeypatch, command, lambda *args: (7, 8, 4242, 9),
                          environment)()
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    connect = lambda: device_guard.guard('socket.connect', (client, ('127.0.0.1', 4532)))
    if owned:
        connect()
    else:
        with pytest.raises(PermissionError, match='connected to 127.0.0.1:4532'):
            connect()


@pytest.mark.parametrize('interruption', [OSError, KeyboardInterrupt])
@pytest.mark.parametrize('ended', [True, False])
@pytest.mark.parametrize('failing', [None, 'terminate', 'wait', 'close'])
def test_windows_rigctld_left_unregistered_is_ended_before_the_error_returns(device_guard, monkeypatch,
                                                                            ended, failing, interruption):
    calls = []
    def call(name, handle, result=None):
        calls.append((name, handle))
        if name == failing:
            raise interruption(f'{name} failed')
        return result
    def unregistered(*registration):
        raise OSError('manifest write failed')
    monkeypatch.setattr(device_guard, '_created', unregistered)
    create = _windows_creation(
        device_guard, monkeypatch, 'rigctld.exe -m 1 -t 4532', lambda *args: (7, 8, 4242, 9),
        TerminateProcess=lambda handle, code: call('terminate', handle),
        WaitForSingleObject=lambda handle, ms: call('wait', handle, 0 if ended else 258),
        CloseHandle=lambda handle: call('close', handle))
    resolved = ended and failing != 'wait'
    interrupted = interruption is KeyboardInterrupt and failing is not None
    if not resolved:
        expected, match = RuntimeError, 'pid 4242: termination unresolved'
    elif interrupted:
        expected, match = KeyboardInterrupt, f'{failing} failed'
    else:
        expected, match = OSError, 'manifest write failed'
    with pytest.raises(expected, match=match) as raised:
        create()
    assert calls == [('terminate', 7), ('wait', 7), ('close', 8), ('close', 7)]
    chain, error = [], raised.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    assert 'manifest write failed' in chain
    assert interrupted == (f'{failing} failed' in chain)


@pytest.mark.posix_only('mocked POSIX PTY allocation')
@pytest.mark.parametrize('port,served,reached', [
    (4532, None, False),
    (4532, 'nothing listening', True),
    (4532, ['rigctld', '-m', '1036', '-r', '/dev/ttys003'], True),
    (4532, ['rigctld', '-m', '1036', '-r', '/dev/ttys003', '-t', '4533'], False),
    (4532, ['rigctld', '-r', '/dev/ttys003', '-T', '127.0.0.1', '-t', '4532'], True),
    (4532, ['rigctld', '-r', '/dev/ttys003', '--port=4532'], True),
    (4532, ['rigctld', '-r', '/dev/ttys003', '-t4533'], False),
    (4532, ['rigctld', '-m', '1036', '-r', '/dev/ttys000'], False),
    (4532, 'bind', True),
    (8515, 'bind', True),
    (8515, ['rigctld', '-m', '1036', '-r', '/dev/ttys003'], False)])
def test_rig_and_modem_ports_reached_only_while_this_test_serves_them(
        device_guard, monkeypatch, tmp_path, port, served, reached):
    monkeypatch.setattr(device_guard.os, 'ttyname', lambda fd: '/dev/ttys003')
    monkeypatch.setattr(device_guard, '_device_identity', lambda path: [1, 2, 3])
    device_guard.own_pty(100, 101)
    listening = served != 'nothing listening'
    probes = []
    monkeypatch.setattr(device_guard, '_listened', lambda *address: probes.append(address) or listening)
    if served == 'bind':
        server = _socket_driver(device_guard, monkeypatch)
        server.bind(('127.0.0.1', port))
    elif listening and served:
        try:
            device_guard.guard('os.exec', (served[0], served, None))
        except PermissionError:
            assert not reached
    client = NS(family=socket.AF_INET, type=socket.SOCK_STREAM)
    connect = lambda host: device_guard.guard('socket.connect', (client, (host, port)))
    if reached:
        connect('127.0.0.1')
    else:
        with pytest.raises(PermissionError):
            connect('127.0.0.1')
    with pytest.raises(PermissionError, match=f'connected to 192.0.2.1:{port}'):
        connect('192.0.2.1')
    assert ('192.0.2.1', port) not in probes
    if not listening:
        return
    other = tmp_path / 'another-test'
    other.mkdir()
    with monkeypatch.context() as elsewhere:
        elsewhere.setenv('HERMETIC_WORK', str(other))
        with pytest.raises(PermissionError):
            connect('127.0.0.1')
    # A server that has ended serves nothing.
    _owner_ended(device_guard, monkeypatch)
    device_guard.armed = True
    with pytest.raises(PermissionError):
        connect('127.0.0.1')


@pytest.mark.parametrize('shadow,refused', [(None, False), ('sitecustomize.py', True),
                                            ('hermetic_guard.py', True), ('sounddevice', True),
                                            ('sounddevice.py', True)])
def test_entries_ahead_of_the_startup_may_not_shadow_it(
        device_guard, monkeypatch, tmp_path, shadow, refused):
    work = tmp_path / 'test-work'
    work.mkdir()
    monkeypatch.setenv('HERMETIC_WORK', str(work))
    ahead = tmp_path / 'launcher-packages'
    ahead.mkdir()
    if shadow:
        (ahead / shadow).mkdir() if '.' not in shadow else (ahead / shadow).touch()
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ahead), os.environ['PYTHONPATH']]))
    argv = [sys.executable, '-c', 'pass']
    if refused:
        with pytest.raises(PermissionError, match='unprotected Python environment'):
            device_guard.check_launch(sys.executable, argv, environment)
    else:
        device_guard.check_launch(sys.executable, argv, environment)


def test_fixture_audio_metadata_keeps_device_opens_guarded(monkeypatch, tmp_path):
    work = tmp_path / 'child-audio-guard'
    fake = work / 'scripted-audio'
    fake.mkdir(parents=True)
    (fake / 'sounddevice.py').write_text(
        "default = type('Defaults', (), {'device': (0, 1)})()\n"
        "def query_devices(*args):\n    return {'name': 'fake server'}\n"
        "def InputStream(port):\n    return open(port, 'rb')\n")
    monkeypatch.setenv('HERMETIC_WORK', str(work))
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(fake), str(hermetic.GUARD), str(Path(hermetic.__file__).parents[2])]))
    code = r'''
import os, sys
import hermetic_guard
import sounddevice
assert hermetic_guard.armed
assert os.path.realpath(sounddevice.__file__) == os.path.realpath(sys.argv[1])
assert sounddevice.default.device == (0, 1)
assert sounddevice.query_devices(0, 'input') == {'name': 'fake server'}
port = 'COM3' if os.name == 'nt' else '/dev/snd/pcmC0D0c'
def reached_device(event, args):
    if event == 'open' and args[0] == port:
        raise AssertionError('unguarded audio device open')
sys.addaudithook(reached_device)
try:
    sounddevice.InputStream(port)
except PermissionError:
    print('scripted audio device open refused')
else:
    raise AssertionError('unguarded audio device open')
'''
    result = subprocess.run([sys.executable, '-c', code, str(fake / 'sounddevice.py')],
                            env=environment, cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == 'scripted audio device open refused'
    assert 'opened ' in (work / 'trips').read_text()


@pytest.mark.parametrize('shadow', [
    pytest.param('sounddevice.py', marks=pytest.mark.posix_only('POSIX foreign audio symlink')),
    'sitecustomize.py', 'hermetic_guard.py', 'sounddevice'])
def test_fixture_audio_exception_does_not_admit_foreign_or_startup_modules(
        monkeypatch, tmp_path, shadow):
    work = tmp_path / 'child-audio-guard'
    fake = work / 'scripted-audio'
    fake.mkdir(parents=True)
    target = fake / shadow
    if shadow == 'sounddevice.py':
        foreign = tmp_path / 'foreign-sounddevice.py'
        foreign.write_text('raise AssertionError("foreign audio imported")\n')
        target.symlink_to(foreign)
    elif shadow == 'sounddevice':
        target.mkdir()
        (target / '__init__.py').touch()
    else:
        target.touch()
    monkeypatch.setenv('HERMETIC_WORK', str(work))
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(fake), str(hermetic.GUARD)]))
    _os_spawn_sentinel(monkeypatch)
    monkeypatch.setattr(hermetic.guard, 'trip',
                        lambda what: (_ for _ in ()).throw(PermissionError(what)))
    with pytest.raises(PermissionError, match='unprotected Python environment'):
        subprocess.Popen([sys.executable, '-c', 'import sounddevice'], env=environment)



def test_listener_probe_is_silent_and_unguarded(device_guard):
    import socket
    server = socket.socket()
    server.bind(('127.0.0.1', 0))
    server.listen()
    host, port = server.getsockname()
    try:
        assert device_guard._listened(host, port)
        connection, _ = server.accept()
        assert connection.recv(16) == b''
        connection.close()
    finally:
        server.close()
    assert not device_guard._listened(host, port)
