# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import os
import subprocess
from types import SimpleNamespace as NS

import pytest

from hfmodem.core import claim, down, native


def argv(program='rigctld'):
    return [program, '-P', 'NONE', '-m', '1', '--set-conf=ptt_type=None',
            '-T', '127.0.0.1', '-t', '4532']


@pytest.mark.parametrize('platform', ['posix', 'nt'])
@pytest.mark.parametrize('extra', [
    ['-vm1036', '-vrCOM41'], ['--set-conf=rig_pathname=COM41'],
    ['--set-conf', 'rig_pathname=COM41'], ['-C', 'rig_pathname=COM41'],
    ['-Crig_pathname=COM41'], ['--set-conf=ptt_pathname=COM41'],
    ['--set-conf=ptt_type=None,rig_pathname=COM41'], ['-rCOM41'],
    ['--rig-file=COM41'], ['-pCOM41'], ['-v'],
])
def test_ambiguous_portless_argv_cannot_acknowledge_listener(monkeypatch, platform, extra):
    monkeypatch.setattr(native, 'os', NS(**(vars(os) | {'name': platform})))
    monkeypatch.setattr(claim, 'serial_nodes', lambda device: ('device',))
    from hfmodem.core import winclaim
    monkeypatch.setattr(winclaim, 'serial_target', lambda device: 'device')
    command = argv('rigctld.exe' if platform == 'nt' else 'rigctld') + extra
    try:
        observation = native._observation(command, None)
    except claim.ClaimError:
        pass
    else:
        assert observation.listener is None and observation.nodes == ('device',)
    # Even when a caller supplies a device, extra options cannot choose listener evidence.
    assert native._observation(command, 'COM41').listener is None


@pytest.mark.parametrize('platform', ['posix', 'nt'])
def test_only_launcher_portless_form_uses_listener(monkeypatch, platform):
    monkeypatch.setattr(native, 'os', NS(**(vars(os) | {'name': platform})))
    assert native._observation(argv(), None).listener == ('127.0.0.1', 4532)
    with pytest.raises(claim.ClaimError):
        native._observation(['rigctld', '-m1'], None)


@pytest.mark.parametrize('platform', ['posix', 'nt'])
@pytest.mark.parametrize('owners', [set(), {999}, {123, 999}, {123}])
def test_listener_ownership_inventory_is_bound_to_child(monkeypatch, platform, owners):
    monkeypatch.setattr(native, 'os', NS(**(vars(os) | {'name': platform})))
    monkeypatch.setattr(native.time, 'monotonic', lambda: 10.)
    def listeners(port, host, **options):
        assert (host, port) == ('127.0.0.1', 4532)
        assert options['deadline'] == 12 and not options['cancelled']()
        return owners
    monkeypatch.setattr(down, 'listeners', listeners)
    def run(command, **options):
        assert command == ['lsof', '-t', '-nP', '-iTCP:4532', '-sTCP:LISTEN']
        assert 0 < options['timeout'] <= 2
        return subprocess.CompletedProcess(command, 0, '\n'.join(map(str, owners)), '')
    monkeypatch.setattr(native.subprocess, 'run', run)
    child = NS(pid=123, alive=lambda: True)
    assert native._listener_owned(child, ('127.0.0.1', 4532), deadline=12,
                                  cancelled=lambda: False) == (owners == {123})


@pytest.mark.parametrize('platform', ['posix', 'windows'])
@pytest.mark.parametrize('ownership', ['foreign', 'lost-during-reply', 'own'])
def test_live_starting_child_never_acknowledges_occupied_endpoint(monkeypatch, platform, ownership):
    now, inspections = [0.], []
    monkeypatch.setattr(native, 'time', NS(monotonic=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    replies = []
    def owned(*args, **kwargs):
        inspections.append(True)
        return ownership == 'own' or ownership == 'lost-during-reply' and len(inspections) == 1
    monkeypatch.setattr(native, '_listener_owned', owned)
    monkeypatch.setattr(native, '_listener_ready', lambda *a, **kw: replies.append(True) or True)
    child = NS(pid=123, alive=lambda: True, _process=None)
    config = dict(argv=argv(), audio=False, nodes=(), listener=('127.0.0.1', 4532))
    observe = native._observe_windows if platform == 'windows' else native._observe_posix
    if ownership == 'own':
        observe(child, config, lambda: False)
        assert len(inspections) == 2 and len(replies) == 1
    else:
        with pytest.raises(native.NativeSpawnError, match='timed out|never answered'):
            observe(child, config, lambda: False)
        assert now[0] <= native.OPEN_TIMEOUT_S
        assert len(replies) == (1 if ownership == 'lost-during-reply' else 0)


@pytest.mark.parametrize('host', ['localhost', '0.0.0.0', '127.0.0.2', '::1'])
def test_listener_address_must_match_launcher(monkeypatch, host):
    command = argv()
    command[7] = host
    with pytest.raises(claim.ClaimError, match='requires a device'):
        native._observation(command, None)


@pytest.mark.parametrize('ending', ['deadline', 'interrupt'])
def test_owner_inventory_does_not_start_outside_budget(monkeypatch, ending):
    monkeypatch.setattr(native.time, 'monotonic', lambda: 12.)
    monkeypatch.setattr(native.subprocess, 'run', lambda *a, **kw: pytest.fail('late inventory'))
    child = NS(pid=123, alive=lambda: True)
    if ending == 'interrupt':
        with pytest.raises(native.NativeSpawnError, match='interrupted'):
            native._listener_owned(child, ('127.0.0.1', 4532), deadline=20, cancelled=lambda: True)
    else:
        assert not native._listener_owned(child, ('127.0.0.1', 4532), deadline=12, cancelled=lambda: False)


def test_windows_owner_table_filters_endpoint_and_returns_owner_pids(monkeypatch):
    import ctypes as C
    import socket
    calls = []
    class Query:
        def __call__(self, data, size, ordered, family, table, reserved):
            calls.append((family, table))
            assert table == 3  # TCP_TABLE_OWNER_PID_LISTENER
            rows = []
            if family == socket.AF_INET:
                for host, port, pid in [('127.0.0.1', 4532, 123), ('127.0.0.2', 4532, 999),
                                        ('0.0.0.0', 4532, 123), ('127.0.0.1', 5555, 888)]:
                    row = down._Tcp4()
                    row.address = int.from_bytes(socket.inet_pton(family, host), 'little')
                    row.port, row.pid = socket.htons(port), pid
                    rows.append(bytes(row))
            # down's table uses the Windows DWORD header (4 bytes).
            payload = len(rows).to_bytes(4, 'little') + b''.join(rows)
            if data is None:
                size._obj.value = max(C.sizeof(down.W.DWORD), len(payload))
                return 122
            C.memmove(data, payload, len(payload))
            return 0
    query = Query()
    monkeypatch.setattr(down.C, 'WinDLL', lambda *a, **kw: NS(GetExtendedTcpTable=query), raising=False)
    monkeypatch.setattr(down.socket, 'getaddrinfo', lambda host, port, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, '', (host, port))])
    assert down.listeners(4532, '127.0.0.1', deadline=native.time.monotonic() + 1) == {123}
    assert calls == [(socket.AF_INET, 3)] * 2 + [(socket.AF_INET6, 3)] * 2


@pytest.mark.parametrize('ending', ['deadline', 'interrupt'])
def test_windows_table_resize_obeys_observation_budget(monkeypatch, ending):
    now, interrupted, calls = [0.], [False], []
    monkeypatch.setattr(down.time, 'monotonic', lambda: now[0])
    class Query:
        def __call__(self, data, size, *args):
            calls.append(True)
            size._obj.value = 128
            if data is not None:
                now[0] = 12
                interrupted[0] = ending == 'interrupt'
            return 122
    monkeypatch.setattr(down.C, 'WinDLL', lambda *a, **kw: NS(GetExtendedTcpTable=Query()), raising=False)
    with pytest.raises(claim.ClaimError, match='interrupted or timed out'):
        down.listeners(4532, deadline=12, cancelled=lambda: interrupted[0])
    assert len(calls) == 2
