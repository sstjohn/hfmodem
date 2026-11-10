# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import os
import socket
import subprocess
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import hamlib


@pytest.fixture
def location(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    default = home / 'src/radio/hamlib/_install/bin'
    default.mkdir(parents=True)
    configured = tmp_path / 'configured'
    configured.mkdir()
    monkeypatch.setattr(hamlib, 'os', SimpleNamespace(**{**vars(os), 'name': 'posix'}))
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.setattr(hamlib.shutil, 'which', lambda name, **kwargs: None)
    return configured, default


def test_configured_beats_path(location, monkeypatch):
    configured, default = location
    (configured / 'rigctld').touch(mode=0o755)
    monkeypatch.setattr(hamlib.shutil, 'which', lambda name, **kwargs: pytest.fail('PATH before HAMLIB_BIN'))
    assert hamlib.rigctld_path({'HAMLIB_BIN': str(configured)}) == str(configured / 'rigctld')


def test_path_beats_default(location, monkeypatch):
    configured, default = location
    (default / 'rigctld').touch(mode=0o755)
    monkeypatch.setattr(hamlib.shutil, 'which', lambda name, **kwargs: '/path/rigctld')
    assert hamlib.rigctld_path({'HAMLIB_BIN': str(configured)}) == '/path/rigctld'


@pytest.mark.posix_only('Windows has no execute bit to withhold')
def test_a_file_that_cannot_run_is_passed_over(location, monkeypatch):
    configured, default = location
    (configured / 'rigctld').touch(mode=0o644)
    monkeypatch.setattr(hamlib.shutil, 'which', lambda name, **kwargs: '/path/rigctld')
    assert hamlib.rigctld_path({'HAMLIB_BIN': str(configured)}) == '/path/rigctld'


def test_station_default(location):
    configured, default = location
    (default / 'rigctld').touch(mode=0o755)
    assert hamlib.rigctld_path({}) == str(default / 'rigctld')


def test_windows_executable_name(location, monkeypatch):
    configured, default = location
    monkeypatch.setattr(hamlib.os, 'name', 'nt')
    (configured / 'rigctld.exe').touch(mode=0o755)
    assert hamlib.rigctld_path({'HAMLIB_BIN': str(configured)}) == str(configured / 'rigctld.exe')
    (configured / 'rigctld.exe').unlink()
    (default / 'rigctld.exe').touch(mode=0o755)
    assert hamlib.rigctld_path({'HAMLIB_BIN': str(configured)}) == str(default / 'rigctld.exe')


@pytest.mark.parametrize('configured_env', [True, False])
def test_missing_lists_every_location(location, configured_env):
    configured, default = location
    env = {'HAMLIB_BIN': str(configured)} if configured_env else {}
    with pytest.raises(FileNotFoundError) as error:
        hamlib.rigctld_path(env)
    message = str(error.value)
    assert (str(configured / 'rigctld') if configured_env else 'HAMLIB_BIN (unset)') in message
    assert 'PATH' in message and str(default / 'rigctld') in message
    assert 'install hamlib' in message and 'export HAMLIB_BIN' in message


@pytest.mark.parametrize('name', ['nt', 'posix'])
def test_spawn_platform_flags(monkeypatch, name):
    monkeypatch.setattr(hamlib, 'os', SimpleNamespace(name=name))
    monkeypatch.setattr(hamlib, 'rigctld_path', lambda: 'fake-rigctld')
    monkeypatch.setattr(hamlib.subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200, raising=False)
    seen = []
    proc = object()

    def popen(argv, **kwargs):
        seen.append((argv, kwargs))
        return proc

    monkeypatch.setattr(hamlib.subprocess, 'Popen', popen)
    kwargs = {'stdout': subprocess.DEVNULL}
    if name == 'nt':
        kwargs['creationflags'] = 0x08000000
    assert hamlib.spawn_rigctld(['-m', '1'], **kwargs) is proc
    assert seen == [(['fake-rigctld', '-m', '1'], {'stdout': subprocess.DEVNULL,
        **({'creationflags': 0x08000200} if name == 'nt' else {'start_new_session': True})})]


class Process:
    def __init__(self, *, stuck=False, returncode=None):
        self.stuck, self.returncode = stuck, returncode
        self.events = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.events.append(('signal', sig))

    def terminate(self):
        self.events.append(('terminate',))

    def kill(self):
        self.events.append(('kill',))

    def wait(self, timeout=None):
        self.events.append(('wait', timeout))
        if self.stuck and timeout is not None:
            raise subprocess.TimeoutExpired('rigctld', timeout)
        self.returncode = 0 if timeout is not None else -9
        return self.returncode


@pytest.mark.parametrize('name', ['nt', 'posix'])
@pytest.mark.parametrize('stuck', [False, True])
def test_stop_platform_signal_and_kill(monkeypatch, name, stuck):
    monkeypatch.setattr(hamlib, 'os', SimpleNamespace(name=name))
    monkeypatch.setattr(hamlib.signal, 'CTRL_BREAK_EVENT', 123, raising=False)
    proc = Process(stuck=stuck)
    assert hamlib.stop_rigctld(proc, .5) == (-9 if stuck else 0)
    assert proc.events == ([('signal', 123)] if name == 'nt' else [('terminate',)]) + [
        ('wait', .5)] + ([('kill',), ('wait', None)] if stuck else [])


def test_stop_kills_when_the_break_cannot_be_delivered(monkeypatch):
    monkeypatch.setattr(hamlib, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(hamlib.signal, 'CTRL_BREAK_EVENT', 123, raising=False)
    proc = Process()

    def refuse(sig):
        raise OSError(6, 'The handle is invalid')

    proc.send_signal = refuse
    assert hamlib.stop_rigctld(proc, .5) == -9
    assert proc.events == [('kill',), ('wait', None)]


def test_stop_already_exited():
    proc = Process(returncode=7)
    assert hamlib.stop_rigctld(proc, .5) == 7
    assert not proc.events


def test_real_dummy_rigctld():
    try:
        hamlib.rigctld_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    proc = hamlib.spawn_rigctld(['-m', '1', '-t', str(port), '-T', '127.0.0.1'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while True:
            assert proc.poll() is None, f'rigctld exited with {proc.returncode}'
            try:
                sock = socket.create_connection(('127.0.0.1', port), timeout=.5)
                break
            except OSError:
                assert time.monotonic() < deadline, 'dummy rigctld did not listen'
                time.sleep(.02)
        with sock, sock.makefile('rb') as replies:
            sock.sendall(b'f\n')
            assert int(replies.readline()) > 0
    finally:
        hamlib.stop_rigctld(proc, 2)
    assert proc.poll() is not None


@pytest.mark.parametrize('name', ['rigctl', 'rigctld'])
@pytest.mark.parametrize('source', ['HAMLIB_BIN', 'PATH'])
def test_windows_resolver_uses_pathext(location, monkeypatch, name, source):
    configured, default = location
    monkeypatch.setattr(hamlib.os, 'name', 'nt')
    executable = configured / (name + '.exe')
    executable.touch(mode=0o644)
    env = {source: str(configured), 'PATHEXT': '.EXE;.CMD', 'PATH': str(configured)}
    if source == 'HAMLIB_BIN':
        env['PATH'] = ''
    assert hamlib.executable_path(name, env) == str(executable)


@pytest.mark.parametrize('name', ['rigctl', 'rigctld'])
@pytest.mark.parametrize('source', ['HAMLIB_BIN', 'PATH', 'local'])
def test_windows_wrappers_cannot_shadow_native_executable(location, monkeypatch, name, source):
    configured, default = location
    monkeypatch.setattr(hamlib.os, 'name', 'nt')
    for directory in (configured, default):
        for suffix in ('.bat', '.cmd'):
            (directory / (name + suffix)).touch()
    directory = default if source == 'local' else configured
    executable = directory / (name + '.exe')
    executable.touch()
    env = {'PATH': str(configured), 'PATHEXT': '.BAT;.CMD;.EXE'}
    if source == 'HAMLIB_BIN':
        env['HAMLIB_BIN'] = str(configured)
    assert hamlib.executable_path(name, env) == str(executable)
    executable.unlink()
    assert hamlib.executable_path(name, env, required=False) is None
    with pytest.raises(FileNotFoundError):
        hamlib.executable_path(name, env)


@pytest.mark.parametrize('name', ['rigctl', 'rigctld'])
def test_windows_missing_executable(location, monkeypatch, name):
    configured, default = location
    monkeypatch.setattr(hamlib.os, 'name', 'nt')
    env = {'HAMLIB_BIN': str(configured), 'PATH': '', 'PATHEXT': '.EXE'}
    assert hamlib.executable_path(name, env, required=False) is None
    with pytest.raises(FileNotFoundError, match=name + r'\.exe'):
        hamlib.executable_path(name, env)
