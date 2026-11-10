# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import contextmanager, nullcontext
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.core import claim, native, posix, winclaim, windeadman, winproc
from hfmodem.tests import hermetic

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows process handles and channels')
PYTHON = getattr(sys, '_base_executable', None) or sys.executable
PRELUDE = '''
import ctypes as C, json, os, signal, sys, threading, time
from hfmodem.core import claim, native, posix, winclaim, windeadman, winproc
'''


def test_full_api_table_binds_on_windows(monkeypatch):
    monkeypatch.setattr(winclaim, '_api_cache', None)
    monkeypatch.setattr(winproc, '_api_cache', None)
    api = winproc._api()
    assert api is winproc._api()
    assert api.DuplicateHandle.argtypes == [W.HANDLE, W.HANDLE, W.HANDLE,
                                           C.POINTER(W.HANDLE), W.DWORD, W.BOOL, W.DWORD]


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(hermetic.GUARD), *filter(None, sys.path)]))
    monkeypatch.setenv('__PYVENV_LAUNCHER__', sys.executable)
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_STATE_DIR', str(tmp_path / 'state'))
    for name in ('HFMODEM_COEXIST_LOG', 'HFMODEM_COEXIST_FAULT', 'HFMODEM_COEXIST_BARRIERS',
                 'RIG_CLAIMED'):
        monkeypatch.delenv(name, raising=False)


def spawn_claim(descriptor, argv, **options):
    admission = claim.inherited(descriptor, launch=True)
    try:
        return claim.spawn(admission, argv, role='python', **options)
    finally:
        admission.failed()


class Child:
    def __init__(self, process):
        self.process = process
        self.lines = queue.Queue()
        def read():
            for line in process.stdout:
                self.lines.put(line.rstrip('\n'))
            self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def line(self, timeout=10):
        value = self.lines.get(timeout=timeout)
        if value is None:
            self.process.wait(timeout=5)
            pytest.fail(f'child exited {self.process.returncode}: {self.process.stderr.read()}')
        return value

    def send(self, value='go'):
        self.process.stdin.write(value + '\n')
        self.process.stdin.flush()

    def entry(self, rights=winproc.END):
        return winproc.open(self.process.pid, rights)

    def native(self, *, stdin=False):
        with self.entry() as entry:
            return native.NativeChild(entry.pid, entry.started,
                                      self.process.stdin if stdin else None,
                                      None, None, self.process)


@pytest.fixture
def children():
    running = []
    def start(code, *args, flags=0, handles=(), env=None):
        options = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=dict(os.environ, **(env or {})), close_fds=True,
                       creationflags=flags)
        if handles:
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {'handle_list': list(handles)}
            options['startupinfo'] = startup
        process = subprocess.Popen([PYTHON, '-c', PRELUDE + code, *map(str, args)], **options)
        child = Child(process)
        running.append(child)
        return child
    yield start
    for child in reversed(running):
        if child.process.poll() is None:
            child.process.kill()
        child.process.wait(timeout=10)
        child.reader.join(5)
        for stream in (child.process.stdin, child.process.stdout, child.process.stderr):
            stream.close()


@contextmanager
def entries(*pids):
    pinned, unverified = winproc.pinned(pids)
    try:
        assert not unverified
        yield {entry.pid: entry for entry in pinned}
    finally:
        for entry in pinned:
            entry.close()


def test_pinning_lineage_rejects_younger_parent_and_remembers_dead_intermediate(children, monkeypatch):
    older = children("print('ready', flush=True); input()")
    assert older.line() == 'ready'
    younger = children("print('ready', flush=True); input()")
    assert younger.line() == 'ready'
    root, a, b = os.getpid(), older.process.pid, younger.process.pid
    with entries(root, a, b) as pinned:
        assert pinned[a].creation < pinned[b].creation
        real_snapshot = winproc._parents
        def snapshot():
            parents = real_snapshot()
            parents[a] = b
            parents[b] = root
            return parents
        with monkeypatch.context() as patch:
            patch.setattr(winproc, '_parents', snapshot)
            lineage = winproc.Lineage().update(pinned.values())
        assert pinned[a].identity not in lineage.links
        assert lineage.ancestors(pinned[b]) == {pinned[root].identity}
        with monkeypatch.context() as patch:
            patch.setattr(winproc, '_parents', lambda: {a: root, b: a})
            lineage.update(pinned.values())
        older.send()
        assert older.process.wait(timeout=5) == 0
        with monkeypatch.context() as patch:
            patch.setattr(winproc, '_parents', lambda: {b: a})
            lineage.update([pinned[root], pinned[b]])
        assert lineage.ancestors(pinned[b]) == {pinned[a].identity, pinned[root].identity}
        assert lineage.descendants(pinned[root]) == {pinned[a].identity, pinned[b].identity}


@pytest.mark.parametrize('rights,action', [(winproc.IDENTIFY, 'identify'),
                                         (winproc.END, 'end'), (winproc.INSPECT, 'inspect')])
def test_access_denied_is_distinct_and_unverified(monkeypatch, rights, action):
    def denied(*args):
        C.set_last_error(5)
        return None
    monkeypatch.setattr(winproc._api(), 'OpenProcess', denied)
    with pytest.raises(winproc.ProcessAccessError, match=f'cannot {action}'):
        winproc.open(1234, rights)
    assert winproc.pinned([1234]) == ([], {1234})


def test_probe_grows_to_cap_and_never_turns_failure_into_empty_inventory():
    sizes = []
    def grow(data, size, needed):
        sizes.append(size)
        C.cast(needed, C.POINTER(W.DWORD))[0] = 1024
        return winproc._MISMATCH if size < 1024 else 0
    assert len(winproc._probe('test', grow)) == 1024
    assert sizes == [256, 1024]
    with pytest.raises(winproc.InventoryUnavailable) as failure:
        winproc._probe('injected', lambda *args: 0xc0000022)
    assert failure.value.name == 'injected'
    assert failure.value.status == 0xc0000022
    sizes.clear()
    def huge(data, size, needed):
        sizes.append(size)
        C.cast(needed, C.POINTER(W.DWORD))[0] = winproc._MAX_INVENTORY + 1
        return winproc._MISMATCH
    with pytest.raises(winproc.InventoryUnavailable):
        winproc._probe('too large', huge)
    assert sizes == [256, 64 * 1024 * 1024]


def test_command_line_probe_on_this_windows_build():
    with winproc.open(os.getpid()) as entry:
        assert Path(sys.executable).name.lower() in winproc.command_line(entry).lower()


def test_launch_replace_with_open_reader_and_phases(children):
    child = children("print('ready', flush=True); input()")
    assert child.line() == 'ready'
    with child.entry() as entry:
        record = winproc.create_launch(entry, role='native', device='COM41',
                                       handles={'claim': 12, 'owner': 16}, grace=15)
        assert record['child'] == winproc._identity(entry)
        assert record['launcher']['pid'] == os.getpid()
        assert record['phase'] == 'created'
        path = winproc._path('launch', entry)
        assert path.name == f'{entry.pid}-{entry.creation}.json'
        reader = winclaim.open_handle(path, 1, 7)
        errors = []
        def replace():
            try:
                winproc.advance_launch(entry, 'resuming', handles={'claim': 24})
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=replace)
        thread.start()
        thread.join(2)
        try:
            assert not thread.is_alive()
            assert errors == []
            data, count = C.create_string_buffer(16384), W.DWORD()
            assert winproc._api().ReadFile(reader, data, len(data), C.byref(count), None)
            assert json.loads(data.raw[:count.value])['phase'] == 'created'
            for phase in ('acknowledged', 'exited'):
                assert winproc.advance_launch(entry, phase)['phase'] == phase
            assert winproc.read_launch(entry)['handles'] == {'claim': 24}
            with pytest.raises(ValueError):
                winproc.advance_launch(entry, 'unresolved')
        finally:
            winclaim.close_handle(reader)
        winproc.create_launch(entry, role='inspector')
        assert winproc.advance_launch(entry, 'unresolved')['phase'] == 'unresolved'
    with winproc.open(os.getpid()) as own:
        winproc.create_launch(own, role='python')
        assert winproc.read_launch() == winproc.read_launch(own)


@pytest.mark.wall_clock
def test_launch_replace_retries_sharing_violation(monkeypatch):
    api = winproc._api()
    replace, attempts = api.SetFileInformationByHandle, []
    def busy(handle, kind, data, size):
        assert kind == 22
        info = winproc._FILE_RENAME_INFO.from_buffer(data)
        assert info.Flags == 3
        attempts.append(time.monotonic())
        if len(attempts) <= 2:
            C.set_last_error(32)
            return False
        return replace(handle, kind, data, size)
    monkeypatch.setattr(api, 'SetFileInformationByHandle', busy)
    with winproc.open(os.getpid()) as own:
        winproc.create_launch(own, role='python')
        assert winproc.read_launch()['phase'] == 'created'
    assert len(attempts) == 3
    assert attempts[-1] - attempts[0] >= .015


@pytest.mark.wall_clock
def test_grace_publication_serializes_with_acknowledgement(monkeypatch):
    read = winproc.read_launch
    captured, release, acknowledging, acknowledged = [threading.Event() for _ in range(4)]
    errors = []
    def paused_read(entry):
        record = read(entry)
        if threading.current_thread().name == 'grace-publication':
            captured.set()
            assert release.wait(5)
        return record
    def run(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)
    with winproc.open(os.getpid()) as own:
        winproc.create_launch(own, role='python')
        winproc.advance_launch(own, 'resuming')
        channels = winproc._Channels.__new__(winproc._Channels)
        channels.entry, channels.stop = own, None
        monkeypatch.setattr(winproc, '_own_grace', lambda: 15)
        monkeypatch.setattr(winproc, 'read_launch', paused_read)
        def acknowledge():
            acknowledging.set()
            winproc.advance_launch(own, 'acknowledged', handles={'claim': 24})
            acknowledged.set()
        publisher = threading.Thread(target=lambda: run(channels.publish), name='grace-publication')
        writer = threading.Thread(target=lambda: run(acknowledge))
        publisher.start()
        try:
            assert captured.wait(5)
            writer.start()
            assert acknowledging.wait(5)
            assert not acknowledged.wait(.1)
        finally:
            release.set()
            publisher.join(5)
            if writer.ident is not None:
                writer.join(5)
        assert not publisher.is_alive() and not writer.is_alive()
        assert errors == []
        assert acknowledged.is_set()
        channels.publish()
        record = read(own)
        assert record['phase'] == 'acknowledged'
        assert record['grace'] == 15
        assert record['handles'] == {'claim': 24}


@pytest.mark.parametrize('reason', ['gone', 'reused', 'signalled', 'denied'])
def test_record_pruning(children, monkeypatch, reason):
    child = children("print('ready', flush=True); input()")
    assert child.line() == 'ready'
    with child.entry() as entry:
        if reason == 'reused':
            entry.started = f'win:{entry.creation - 1}'
        record = winproc.create_launch(entry, role='python')
        path = winproc._path('launch', entry)
        if reason == 'signalled':
            child.send()
            assert child.process.wait(timeout=5) == 0
            assert int(child.process._handle)
            assert entry.wait()
        elif reason == 'gone':
            monkeypatch.setattr(winproc, 'open', lambda *args: None)
        elif reason == 'denied':
            def denied(*args):
                raise winproc.ProcessAccessError(entry.pid, winproc.IDENTIFY)
            monkeypatch.setattr(winproc, 'open', denied)
        records, unresolved = winproc.live_launches()
        assert records == ([record] if reason == 'denied' else [])
        assert unresolved == ({entry.pid} if reason == 'denied' else set())
        assert path.exists() == (reason == 'denied')


@pytest.mark.parametrize('kind', ['launch', 'ports'])
def test_record_inventory_refuses_json_null(kind):
    with winproc._own() as own:
        winproc._write_json(winproc._path(kind, own), None)
    with pytest.raises(claim.ClaimError, match='invalid record'):
        list(winproc.launch_records() if kind == 'launch' else winproc.recorded_devices([], None))


@pytest.mark.parametrize('kind', ['launch', 'ports'])
def test_concurrent_pruning_preserves_open_readers(children, kind):
    holder = children("print('ready', flush=True); input()")
    assert holder.line() == 'ready'
    with holder.entry() as entry:
        path = winproc._path(kind, entry)
        record = winproc.create_launch(entry, role='python') if kind == 'launch' else dict(
            process=winproc._identity(entry), ports=['COM41'])
        if kind == 'ports':
            winproc._write_json(path, record)
        reader = winclaim.open_handle(path, 1, 7)
        try:
            holder.send()
            assert holder.process.wait(timeout=5) == 0
            pruners = [children('''
print('ready', flush=True)
input()
for _ in range(10):
    winproc.live_launches()
    list(winproc.recorded_devices([], None))
print('pruned', flush=True)
''') for _ in range(2)]
            for pruner in pruners:
                assert pruner.line() == 'ready'
            for pruner in pruners:
                pruner.send()
            for pruner in pruners:
                assert pruner.line() == 'pruned'
                assert pruner.process.wait(timeout=5) == 0
            assert winproc._read_json(path) is None
            if kind == 'launch':
                assert winproc.advance_launch(entry, 'exited') is None
            assert not path.exists()
            data, count = C.create_string_buffer(16384), W.DWORD()
            assert winproc._api().ReadFile(reader, data, len(data), C.byref(count), None)
            assert json.loads(data.raw[:count.value]) == record
        finally:
            winclaim.close_handle(reader)


def test_pruning_cannot_race_launch_read_modify_replace(children):
    holder = children("print('ready', flush=True); input()")
    assert holder.line() == 'ready'
    with holder.entry() as entry:
        winproc.create_launch(entry, role='python')
        path = winproc._path('launch', entry)
        holder.send()
        assert holder.process.wait(timeout=5) == 0
        updater = children('''
read = winproc.read_launch
def paused(entry):
    record = read(entry)
    print('read', flush=True)
    input()
    return record
winproc.read_launch = paused
with winproc.open(int(sys.argv[1])) as entry:
    assert winproc.advance_launch(entry, 'exited')['phase'] == 'exited'
print('updated', flush=True)
''', entry.pid)
        assert updater.line() == 'read'
        pruner = children('''
from pathlib import Path
path = Path(sys.argv[1])
try:
    h = winclaim.open_handle(path.with_suffix('.lock'), 1 | 2, 0, winclaim.OPEN_ALWAYS)
except OSError as exc:
    assert exc.winerror == 32
else:
    winclaim.close_handle(h)
    raise AssertionError('update did not lock its read-modify-replace')
print('blocked', flush=True)
winproc._prune_record(path)
print('pruned', flush=True)
''', path)
        assert pruner.line() == 'blocked'
        updater.send()
        assert updater.line() == 'updated'
        assert pruner.line() == 'pruned'
        assert updater.process.wait(timeout=5) == pruner.process.wait(timeout=5) == 0
        assert winproc.advance_launch(entry, 'exited') is None
        assert winproc.read_launch(entry) is None


CHANNELS = '''
signal.signal(signal.SIGTERM, lambda *args: None)
channels = winproc.install_channels()
assert winproc.install_channels() is channels
print('ready', flush=True)
'''


def test_channel_names_use_creation_and_missing_is_reported(children):
    child = children(CHANNELS + 'threading.Event().wait()')
    assert child.line() == 'ready'
    with child.entry() as entry:
        assert winproc.event_name('stop', entry) == f'Local\\hfmodem.stop.{entry.pid}.{entry.creation}'
        assert winproc.event_name('int', entry) == f'Local\\hfmodem.int.{entry.pid}.{entry.creation}'
        reused = winproc.Pinned(entry.handle, entry.pid, f'win:{entry.creation + 1}', entry.image)
        assert winproc.request_stop(reused) is None
        assert winproc.request_interrupt(reused) is None
        assert child.process.poll() is None
        event = winproc._api().OpenEventW(0x100000, False, winproc.event_name('stop', entry))
        assert event
        try:
            assert not winproc._wait(event, 0)
        finally:
            winclaim.close_handle(event)


def test_event_dacl_grants_only_the_current_user(children):
    child = children(CHANNELS + 'threading.Event().wait()')
    assert child.line() == 'ready'
    api = winproc._api()
    advapi = C.WinDLL('advapi32', use_last_error=True)
    convert = advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [C.c_void_p, W.DWORD, W.DWORD, C.POINTER(W.LPWSTR), C.c_void_p]
    convert.restype = W.BOOL
    token, user = W.HANDLE(), W.LPWSTR()
    assert api.OpenProcessToken(api.GetCurrentProcess(), 8, C.byref(token))
    try:
        assert api.ConvertSidToStringSidW(winclaim._token_sid(token, 1), C.byref(user))
        user_sid = user.value
    finally:
        winclaim.close_handle(token)
        api.LocalFree(C.cast(user, C.c_void_p))
    with child.entry() as entry:
        for kind in ('stop', 'int'):
            event = api.OpenEventW(0x20000, False, winproc.event_name(kind, entry))
            assert event
            security, text = C.c_void_p(), W.LPWSTR()
            try:
                assert api.GetSecurityInfo(event, 6, 4, None, None, None, None,
                                           C.byref(security)) == 0
                assert convert(security, 1, 4, C.byref(text), None)
                prefix, permissions, suffix = text.value.split(';', 2)
                assert prefix == 'D:P(A'
                assert permissions == ''
                mask, suffix = suffix.split(';', 1)
                assert mask == 'GA' or int(mask, 16) == 0x1f0003
                assert suffix == f';;{user_sid})'
            finally:
                if text:
                    api.LocalFree(C.cast(text, C.c_void_p))
                if security:
                    api.LocalFree(security)
                winclaim.close_handle(event)


def test_console_handler_passes_ctrl_c_and_break_through():
    channels = winproc._Channels.__new__(winproc._Channels)
    assert channels.console_close(0) is False
    assert channels.console_close(1) is False


def fail_thread_start(monkeypatch, failure):
    entered, threads = threading.Event(), []
    class Thread(threading.Thread):
        def __init__(self, *args, **kwargs):
            if failure == 'construct':
                raise RuntimeError('thread construction failed')
            super().__init__(*args, **kwargs)
            threads.append(self)

        def run(self):
            entered.set()
            super().run()

        def start(self):
            if failure == 'start':
                raise RuntimeError('thread startup failed')
            super().start()
            assert entered.wait(5)
            raise KeyboardInterrupt('thread startup interrupted')
    monkeypatch.setattr(winproc, 'threading', SimpleNamespace(
        Thread=Thread, Event=threading.Event, Lock=threading.Lock))
    return threads


@pytest.mark.parametrize('failure', ['construct', 'start', 'started'])
def test_channel_startup_rolls_back_and_retry_installs_one_worker(monkeypatch, failure):
    handles, closed, callbacks, waits, stops = set(), [], [], [], []
    entered, release = threading.Event(), threading.Event()
    next_handle = iter(range(100, 120))
    def new_handle():
        handle = next(next_handle)
        handles.add(handle)
        return handle
    def close(handle):
        handles.remove(handle)
        closed.append(handle)
    def handler(callback, add):
        if add:
            callbacks.append(callback)
        else:
            callbacks.remove(callback)
        return True
    def wait(count, values, *args):
        waits.append(list(values))
        assert all(handle in handles for handle in values)
        entered.set()
        assert release.wait(5)
        return 0
    monkeypatch.setattr(winproc, '_channels', None)
    monkeypatch.setattr(winproc, '_console_handlers', [])
    monkeypatch.setattr(winproc, '_own', lambda: winproc.Pinned(new_handle(), 1, 'win:10', 'test'))
    monkeypatch.setattr(winproc, '_user_security', lambda: None)
    monkeypatch.setattr(winproc._Channels, 'publish', lambda self: setattr(self, 'grace', 2))
    monkeypatch.setattr(C, 'WINFUNCTYPE', C.CFUNCTYPE, raising=False)
    monkeypatch.setattr(winproc, '_api', lambda: SimpleNamespace(
        CreateEventW=lambda *args: new_handle(), SetConsoleCtrlHandler=handler,
        WaitForMultipleObjects=wait))
    monkeypatch.setattr(winclaim, 'close_handle', close)
    monkeypatch.setattr(posix, '_windows_deadman', SimpleNamespace(stop=stops.append))
    monkeypatch.setattr(posix, '_run_on_stop', lambda: None)
    monkeypatch.setattr(winproc._thread, 'interrupt_main', lambda sig: None)
    with monkeypatch.context() as patch:
        threads = fail_thread_start(patch, failure)
        try:
            error = KeyboardInterrupt if failure == 'started' else RuntimeError
            with pytest.raises(error):
                winproc.install_channels()
        finally:
            for thread in threads:
                if thread.ident is not None:
                    thread.join(5)
                    assert not thread.is_alive()
    assert winproc._channels is None
    assert handles == set() and len(closed) == 3
    assert callbacks == winproc._console_handlers == []
    assert waits == []
    channels = winproc.install_channels()
    try:
        assert entered.wait(5)
        assert winproc.install_channels() is channels
        assert len(callbacks) == len(winproc._console_handlers) == 1
        assert len(handles) == 3
        assert waits == [[channels.stop, channels.interrupt]]
    finally:
        release.set()
        channels.thread.join(5)
        assert not channels.thread.is_alive()
        channels.rollback()
    assert stops == [2]
    assert handles == set()
    assert callbacks == winproc._console_handlers == []


def test_grace_publication_and_keying_registration_hooks(children):
    child = children('''
winproc.declare_grace(15)
channels = winproc.install_channels()
assert channels.grace == 15
key = windeadman.register_handle(123456)
assert channels.grace == 2
print(winproc._target_grace(channels.entry), flush=True)
windeadman.unregister_handle(key)
assert channels.grace == 15
print(winproc._target_grace(channels.entry), flush=True)
input()
''')
    assert child.line() == '2.0'
    assert child.line() == '15.0'
    holder = children('''
d = claim.acquire('test', 'channels')
assert winproc._channels is not None
print(winproc._target_grace(winproc._channels.entry), flush=True)
input()
claim.release(d)
''')
    assert holder.line() == '2.0'
    holder.send()
    assert holder.process.wait(timeout=5) == 0


def test_channel_registration_failure_closes_ptt_port_and_removes_registration(monkeypatch):
    from hfmodem.core import ptt, serialline
    closed, attempts = [], []
    ser = SimpleNamespace(_port_handle=123456, close=lambda: closed.append(True))
    monkeypatch.setattr(windeadman, '_handles', ())
    monkeypatch.setattr(winproc, '_channels', object())
    monkeypatch.setattr(ptt, 'require_char_device', lambda port: None)
    monkeypatch.setattr(serialline, 'open_port', lambda *args, **kwargs: ser)
    def fail():
        attempts.append(True)
        raise claim.ClaimError('publication failed')
    monkeypatch.setattr(winproc, 'install_channels', fail)
    line = ptt.RtsPtt('COM41')
    with pytest.raises(ptt.PttError, match='cannot install stop channels.*publication failed'):
        line._open_windows()
    assert closed == [True]
    assert windeadman._handles == ()
    assert line._serial is None
    assert len(attempts) >= 2


def test_channel_publication_failure_does_not_escape_ptt_release(monkeypatch):
    from hfmodem.core import ptt, serialline
    closed = []
    monkeypatch.setattr(windeadman, '_handles', ())
    monkeypatch.setattr(winproc, '_channels', object())
    monkeypatch.setattr(winproc, 'install_channels', lambda: None)
    token = windeadman.register_handle(123456)
    line = ptt.RtsPtt('COM41')
    line._serial = SimpleNamespace(close=lambda: closed.append(True))
    line._deadman_token = token
    monkeypatch.setattr(serialline, 'drive', lambda *args: None)
    monkeypatch.setattr(ptt, '_settled_outputs', lambda *args: 0)
    def fail():
        raise claim.ClaimError('publication failed')
    monkeypatch.setattr(winproc, 'install_channels', fail)
    line.release()
    assert closed == [True]
    assert windeadman._handles == ()
    assert line._serial is None and line._deadman_token is None
    assert line.released_low is True


def test_adoption_installs_channels():
    descriptor = claim.acquire('test', 'adoption-channels')
    try:
        launch = spawn_claim(descriptor, [PYTHON, '-c', PRELUDE + '''
d = claim.received()
assert winproc._channels is not None
assert claim.adopt() == d
print(winproc._target_grace(winproc._channels.entry), flush=True)
'''], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process = launch.process
        try:
            out, err = process.communicate(timeout=10)
            assert process.returncode == 0, err
            assert out == '2.0\n'
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            launch.child.close()
    finally:
        claim.release(descriptor)


def test_cached_adoption_survives_channel_publication_failure():
    descriptor = claim.acquire('test', 'adoption-publication')
    try:
        launch = spawn_claim(descriptor, [PYTHON, '-c', PRELUDE + '''
d = claim.received()
cached = claim._adopted
assert 'claim' not in winclaim.received_record()['handles']
install = winproc.install_channels
def fail():
    raise claim.ClaimError('publication failed')
winproc.install_channels = fail
try:
    try:
        claim.adopt()
    except claim.ClaimError as exc:
        assert 'publication failed' in str(exc)
    else:
        raise AssertionError('publication did not fail')
    assert claim._adopted == cached
    assert claim.inherited(d) == d
finally:
    winproc.install_channels = install
assert claim.adopt() == d
assert claim._adopted == cached
'''], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process = launch.process
        try:
            out, err = process.communicate(timeout=10)
            assert process.returncode == 0, err
            assert out == ''
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            launch.child.close()
    finally:
        claim.release(descriptor)


def test_cached_adoption_retries_publication_without_another_token(monkeypatch):
    descriptor = claim.ClaimDescriptor(123)
    identity = (456, 'win:10', 'nonce')
    cached = (descriptor.fd, identity)
    validated, published = [], []
    monkeypatch.setattr(claim, 'os', SimpleNamespace(name='nt', environ={}))
    monkeypatch.setattr(claim, '_adopted', cached)
    monkeypatch.setattr(winclaim, '_received_loaded', True)
    monkeypatch.setattr(winclaim, '_received_record', None)
    monkeypatch.setattr(winclaim, 'entries', lambda: {123: SimpleNamespace(role='adopted')})
    def inherited(fd):
        validated.append(fd)
        return descriptor
    monkeypatch.setattr(claim, 'inherited', inherited)
    monkeypatch.setattr(claim, 'holder', lambda d: SimpleNamespace(
        pid=identity[0], started=identity[1], nonce=identity[2]))
    def publish():
        published.append(True)
        if len(published) == 1:
            raise claim.ClaimError('publication failed')
    monkeypatch.setattr(winproc, 'install_channels', publish)
    with pytest.raises(claim.ClaimError, match='publication failed'):
        claim.adopt()
    assert claim._adopted is cached
    assert claim.adopt() is descriptor
    assert claim._adopted is cached
    assert validated == [123, 123]
    assert published == [True, True]


def test_deadline_word_is_earliest_and_stop_is_irrevocable():
    watchdog = windeadman.Watchdog.__new__(windeadman.Watchdog)
    watchdog._word = C.c_uint64()
    watchdog._record = (None, None)
    watchdog._stop_deadline = watchdog._timer_deadline = 0
    watchdog._clock = lambda: 100_000_000
    watchdog.arm(20)
    watchdog.stop(2)
    deadline = watchdog._word.value
    assert deadline == 120_000_000
    watchdog.cancel()
    assert watchdog._word.value == deadline
    watchdog.reschedule(40)
    assert watchdog._word.value == deadline
    watchdog.arm(100, b'later', None)
    assert watchdog._word.value == deadline
    watchdog.stop(50)
    assert watchdog._word.value == deadline
    watchdog.arm(.5)
    assert watchdog._word.value == 105_000_000
    watchdog.cancel()
    assert watchdog._word.value == deadline


@pytest.mark.wall_clock
@pytest.mark.parametrize('blocked', ['wait', 'handler'])
def test_stop_deadline_ends_blocked_main_and_survives_later_arms(children, blocked):
    child = children('''
def stopped(*args):
    print('term', flush=True)
    if sys.argv[1] == 'handler':
        threading.Event().wait()
signal.signal(signal.SIGTERM, stopped)
def wake():
    posix.cancel_itimer()
    posix.set_itimer(60)
    posix._windows_deadman.arm(60)
    print('callback', flush=True)
posix.on_stop(wake)
winproc.install_channels()
print('ready', flush=True)
if sys.argv[1] == 'handler':
    while True: pass
else:
    threading.Event().wait()
''', blocked)
    assert child.line() == 'ready'
    with child.entry() as entry:
        before = time.monotonic()
        assert winproc.request_stop(entry) == 2
        assert child.line() == 'callback'
        assert entry.wait(4)
        elapsed = time.monotonic() - before
    assert child.process.wait(timeout=5) == 1
    assert 1.5 < elapsed < 3.5
    assert 'stop deadline expired' in child.process.stderr.read()


@pytest.mark.wall_clock
def test_itimer_can_expire_before_stop_deadline(children):
    child = children(CHANNELS + '''
posix.set_itimer(.5)
print('armed', flush=True)
threading.Event().wait()
''')
    assert child.line() == 'ready'
    assert child.line() == 'armed'
    with child.entry() as entry:
        before = time.monotonic()
        assert winproc.request_stop(entry) == 2
        assert entry.wait(1.5)
        assert time.monotonic() - before < 1.5
    assert child.process.wait(timeout=5) == 1


@pytest.mark.wall_clock
def test_interrupt_dispatch_sleep_then_stop_latches(children):
    child = children('''
class Wake(Exception): pass
count = 0
def interrupt(*args):
    global count
    count += 1
    print(f'int{count}', flush=True)
    raise Wake()
def stop(*args):
    print('term', flush=True)
signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, stop)
winproc.install_channels()
print('ready', flush=True)
while True:
    try:
        time.sleep(100 if count < 2 else .01)
    except Wake:
        pass
''')
    assert child.line() == 'ready'
    with child.entry() as entry:
        wrapped = child.native()
        for number in (1, 2):
            before = time.monotonic()
            assert wrapped.interrupt().status == 'requested'
            assert child.line(timeout=1) == f'int{number}'
            assert time.monotonic() - before < 1
            with pytest.raises(queue.Empty):
                child.lines.get(timeout=.15)
        assert winproc.request_stop(entry) == 2
        assert child.line(timeout=1) == 'term'
        for _ in range(3):
            assert winproc.request_interrupt(entry) == 2
        with pytest.raises(queue.Empty):
            child.lines.get(timeout=.25)
        assert entry.wait(4)
    assert child.process.wait(timeout=5) == 1


@pytest.mark.wall_clock
def test_stopper_honours_declared_fifteen_second_grace(children):
    child = children('''
winproc.declare_grace(15)
def finish():
    time.sleep(13.25)
    os._exit(0)
posix.on_stop(lambda: threading.Thread(target=finish, daemon=True).start())
''' + CHANNELS + 'threading.Event().wait()')
    assert child.line() == 'ready'
    with child.entry() as entry:
        stopper = winproc.Stopper(entry)
        assert stopper.grace == 15
        assert stopper.deadline == stopper.requested_at + 16
        result = stopper.wait()
        assert result.status == 'stopped'
        assert stopper.wait() is result
        assert time.monotonic() - stopper.requested_at > winproc.REAP_GRACE_S
    assert child.process.wait(timeout=5) == 0


@pytest.mark.parametrize('signalled', [True, False])
def test_stopper_waits_after_terminate_access_denied(monkeypatch, signalled):
    clock, requests, waits = [100], [], []
    entry = winproc.Pinned(99, 123, 'win:10', 'test.exe')
    def terminate(handle, code):
        requests.append((handle, code))
        C.set_last_error(5)
        return False
    def wait(handle, timeout):
        assert handle == entry.handle
        waits.append(timeout)
        clock[0] += min(timeout, max(0, 114 - clock[0])) if signalled else timeout
        return signalled and clock[0] >= 114
    monkeypatch.setattr(winproc, '_api', lambda: SimpleNamespace(TerminateProcess=terminate))
    monkeypatch.setattr(winproc, '_wait', wait)
    monkeypatch.setattr(winproc, 'request_stop', lambda target: 2)
    monkeypatch.setattr(winproc, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    stopper = winproc.Stopper(entry)
    clock[0] = stopper.deadline
    assert stopper.poll() is None
    assert stopper.kill_deadline == 117
    assert stopper.poll() is None
    assert clock[0] == 112
    result = stopper.wait()
    assert result.status == ('terminated' if signalled else 'unresolved')
    assert ('cannot end pid 123: access denied' in result.detail) == (not signalled)
    assert bool(result) == signalled
    assert clock[0] == (114 if signalled else 117)
    assert sum(waits) == winproc.KILL_WAIT_S
    assert stopper.wait() is result
    assert requests == [(99, 1)]


def test_stopper_reports_no_channel_before_termination_and_keeps_one_deadline(monkeypatch):
    events, clock = [], [100]
    class Entry:
        terminated = False

        def wait(self, timeout=0):
            clock[0] += timeout
            events.append(('wait', timeout))
            return self.terminated

        def terminate(self):
            assert events[0] == 'no stop channel'
            events.append(('terminate',))
            self.terminated = True
    monkeypatch.setattr(winproc, 'request_stop', lambda entry: None)
    monkeypatch.setattr(winproc, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    stopper = winproc.Stopper(Entry(), report=events.append)
    assert stopper.deadline == 112
    assert stopper.wait() == winproc.StopResult('terminated', 'no stop channel; TerminateProcess')
    before = list(events)
    assert stopper.wait().status == 'terminated'
    assert events == before
    entry = Entry()
    entry.wait = lambda timeout=0: False
    clock[0] = 200
    stopper = winproc.Stopper(entry, report=events.append)
    clock[0] = 212
    assert stopper.poll() is None
    assert stopper.kill_deadline == 217
    assert stopper.poll() is None
    clock[0] = 217
    assert stopper.poll().status == 'unresolved'
    before = list(events)
    clock[0] = 300
    assert stopper.poll().status == 'unresolved'
    assert events == before


@pytest.mark.wall_clock
def test_overlapped_reader_deadline_with_silent_foreign_writer(children):
    reader, writer = winproc.pipe_pair()
    inherited = winclaim.duplicate(writer, inheritable=True)
    try:
        child = children("print(winproc._api().GetFileType(int(sys.argv[1])), flush=True); input()",
                         inherited, handles=[inherited])
        assert child.line() == '3'
    finally:
        winclaim.close_handle(inherited)
        winclaim.close_handle(writer)
    handle = reader.handle
    before = time.monotonic()
    result = reader.read(1024, deadline=before + .2)
    assert result.status == 'deadline'
    assert result.confirmed
    assert .15 < time.monotonic() - before < 1.5
    assert reader._pending is None
    assert reader.close()
    with pytest.raises(OSError):
        os.get_handle_inheritable(handle)
    assert child.process.poll() is None


def test_overlapped_reader_data_eof_and_exit_event():
    reader, writer = winproc.pipe_pair()
    exit_event = winproc._api().CreateEventW(None, True, True, None)
    try:
        result = reader.read(10, deadline=time.monotonic() + 10, exit_event=exit_event)
        assert result == winproc.IOResult('cancelled')
        data, count = C.create_string_buffer(b'hello'), W.DWORD()
        assert winproc._api().WriteFile(writer, data, 5, C.byref(count), None)
        assert reader.read(10, deadline=time.monotonic() + 1) == winproc.IOResult('data', b'hello')
        winclaim.close_handle(writer)
        writer = None
        assert reader.read(10, deadline=time.monotonic() + 1).status == 'eof'
    finally:
        reader.close()
        winclaim.close_handle(exit_event)
        if writer is not None:
            winclaim.close_handle(writer)


def test_pipe_reader_reserves_ownership_before_creating_event(monkeypatch):
    reader, writer = winproc.pipe_pair()
    api = winproc._api()
    exit_event = api.CreateEventW(None, True, True, None)
    create = api.CreateEventW
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []
    def paused_create(*args):
        entered.set()
        assert release.wait(5)
        return create(*args)
    monkeypatch.setattr(api, 'CreateEventW', paused_create)
    def read():
        try:
            results.append(reader.read(10, deadline=time.monotonic() + 10, exit_event=exit_event))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=read)
    handle = reader.handle
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(claim.ClaimError, match='busy'):
            reader.read(10, deadline=time.monotonic() + 1)
        assert not reader.close()
        assert reader.handle == handle
    finally:
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        winclaim.close_handle(writer)
        winclaim.close_handle(exit_event)
        assert reader.close()
    assert errors == []
    assert results == [winproc.IOResult('cancelled')]
    assert reader._pending is None
    with pytest.raises(OSError):
        os.get_handle_inheritable(handle)


@pytest.mark.wall_clock
def test_full_pipe_q_write_cancels_and_joins_writer():
    reader, writer = winproc.pipe_pair()
    fd = winclaim._crt(writer, os.O_WRONLY | os.O_BINARY | os.O_NOINHERIT, role='pipe')
    with os.fdopen(fd, 'wb', buffering=0) as stream:
        try:
            setter = C.WinDLL('kernel32', use_last_error=True).SetNamedPipeHandleState
            setter.argtypes = [W.HANDLE, C.POINTER(W.DWORD), C.c_void_p, C.c_void_p]
            setter.restype = W.BOOL
            mode, count = W.DWORD(1), W.DWORD()
            assert setter(writer, C.byref(mode), None, None)
            fill = C.create_string_buffer(b'x' * 65536)
            chunk, buffered = 65536, 0
            while chunk:
                assert winproc._api().WriteFile(writer, fill, chunk, C.byref(count), None)
                assert 0 <= count.value <= chunk
                buffered += count.value
                assert buffered <= 16 * 1024 * 1024
                if count.value == 0:
                    chunk //= 2
            assert buffered > 0
            assert count.value == 0
            mode.value = 0
            assert setter(writer, C.byref(mode), None, None)
            before = time.monotonic()
            result = winproc.bounded_write(stream)
            assert result == winproc.IOResult('cancelled')
            assert .9 < time.monotonic() - before < 2.5
            assert not any(t.name == 'windows-pipe-writer' and t.is_alive()
                           for t in threading.enumerate())
        finally:
            winclaim.forget(fd)
            assert reader.close()


@pytest.mark.parametrize('failure', ['construct', 'start', 'started'])
def test_writer_startup_failure_revokes_worker_access_before_closing(monkeypatch, failure):
    closed, writes, duplicates = [], [], []
    release = threading.Event()
    def duplicate_thread(source, thread, target, result, *args):
        C.cast(result, C.POINTER(W.HANDLE))[0] = 202
        duplicates.append(202)
        return True
    def write(handle, buffer, size, count, overlapped):
        writes.append(handle)
        assert release.wait(5)
        C.cast(count, C.POINTER(W.DWORD))[0] = size
        return True
    monkeypatch.setattr(winproc, '_api', lambda: SimpleNamespace(
        GetCurrentProcess=lambda: 1, GetCurrentThread=lambda: 2,
        DuplicateHandle=duplicate_thread, WriteFile=write))
    monkeypatch.setattr(winclaim, 'handle', lambda fd: 100)
    monkeypatch.setattr(winclaim, 'duplicate', lambda handle: 101)
    monkeypatch.setattr(winclaim, 'close_handle', closed.append)
    monkeypatch.setattr(winproc, '_unresolved_io', [])
    threads = fail_thread_start(monkeypatch, failure)
    try:
        error = KeyboardInterrupt if failure == 'started' else RuntimeError
        with pytest.raises(error):
            winproc.bounded_write(SimpleNamespace(fileno=lambda: 3))
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
                assert not thread.is_alive()
    assert closed == [101]
    assert writes == duplicates == []
    assert winproc._unresolved_io == []


@pytest.mark.wall_clock
def test_unconfirmed_read_keeps_handle_buffer_and_event(monkeypatch):
    reader, writer = winproc.pipe_pair()
    release, joined = threading.Event(), threading.Event()
    finish = winproc._finish_read
    def delayed(*args):
        result = finish(*args)
        release.wait(10)
        joined.set()
        return result
    monkeypatch.setattr(winproc, '_finish_read', delayed)
    start = len(winproc._unresolved_io)
    try:
        result = reader.read(10, deadline=time.monotonic() + .01)
        assert result == winproc.IOResult('unresolved', confirmed=False)
        assert not reader.close()
        assert reader._pending is not None
        assert len(winproc._unresolved_io) > start
        assert winproc._api().GetFileType(reader.handle) == 3
    finally:
        release.set()
        assert joined.wait(2)
        _, _, event = reader._pending
        reader._pending = None
        reader.close()
        winclaim.close_handle(event)
        winclaim.close_handle(writer)
        del winproc._unresolved_io[start:]


@pytest.mark.wall_clock
def test_unconfirmed_writer_retains_duplicate_and_storage(monkeypatch):
    reader, writer = winproc.pipe_pair()
    fd = winclaim._crt(writer, os.O_WRONLY | os.O_BINARY | os.O_NOINHERIT, role='pipe')
    release = threading.Event()
    real_write = winproc._api().WriteFile
    def delayed(*args):
        release.wait(10)
        return real_write(*args)
    monkeypatch.setattr(winproc._api(), 'WriteFile', delayed)
    start = len(winproc._unresolved_io)
    with os.fdopen(fd, 'wb', buffering=0) as stream:
        try:
            result = winproc.bounded_write(stream, timeout=.01)
            assert result == winproc.IOResult('unresolved', confirmed=False)
            handle, buffer, thread, state, kept_stream = winproc._unresolved_io[-1]
            assert kept_stream is stream and buffer.raw.startswith(b'q\n')
            assert winproc._api().GetFileType(handle) == 3
            assert thread.is_alive()
        finally:
            release.set()
            thread.join(2)
            assert not thread.is_alive()
            winclaim.close_handle(handle)
            winclaim.close_handle(state['thread_handle'])
            winclaim.forget(fd)
            del winproc._unresolved_io[start:]
            reader.close()


@pytest.mark.wall_clock
@pytest.mark.parametrize('route', ['q', 'stop', 'kill', 'fallback-stop', 'fallback-int'])
def test_native_child_uses_handles_and_reports_graceful_or_forced_path(children, monkeypatch, route):
    code = "print('ready', flush=True); input()"
    if route == 'stop':
        code = '''
wake = threading.Event()
signal.signal(signal.SIGTERM, lambda *args: None)
posix.on_stop(wake.set)
winproc.install_channels()
print('ready', flush=True)
wake.wait()
'''
    child = children(code)
    assert child.line() == 'ready'
    wrapped = child.native(stdin=route == 'q')
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('os.kill on Windows'))
    monkeypatch.setattr(native, '_process_state', lambda *args: pytest.fail('ps on Windows'))
    assert wrapped.alive()
    assert not wrapped.wait_gone(.01)
    if route == 'kill':
        result = wrapped.kill()
    elif route == 'fallback-int':
        assert winproc.request_interrupt(wrapped._win_entry) is None
        result = wrapped.interrupt()
    else:
        if route == 'fallback-stop':
            assert winproc.request_stop(wrapped._win_entry) is None
        result = wrapped.terminate()
    assert wrapped.wait_gone(5)
    assert not wrapped.alive()
    if route in ('q', 'stop'):
        assert result.status == 'stopped'
        assert ('stdin q' if route == 'q' else 'stop channel') in result.detail
        assert child.process.wait(timeout=5) == 0
    else:
        assert result.status == 'terminated'
        assert 'TerminateProcess' in result.detail
        if route.startswith('fallback'):
            assert 'no ' in result.detail and 'channel' in result.detail


def test_nt_ratchet_never_reaches_posix_signal_paths(children, monkeypatch):
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('os.kill is fatal on Windows'))
    monkeypatch.delattr(native.signal, 'SIGKILL', raising=False)
    for operation in ('alive', 'wait_gone', 'interrupt', 'terminate', 'kill'):
        child = children("print('ready', flush=True); input()")
        assert child.line() == 'ready'
        wrapped = child.native()
        if operation == 'wait_gone':
            assert not wrapped.wait_gone(0)
        else:
            getattr(wrapped, operation)()
        wrapped.kill()
        assert wrapped.wait_gone(5)
    with winproc.open(os.getpid()) as entry:
        assert not entry.wait()
        assert winproc.request_interrupt(winproc.Pinned(entry.handle, entry.pid,
                        f'win:{entry.creation + 1}', entry.image)) is None


def test_native_child_pins_without_popen_and_refuses_a_changed_creation(children, monkeypatch):
    child = children("print('ready', flush=True); input()")
    assert child.line() == 'ready'
    with child.entry() as entry:
        wrong = native.NativeChild(entry.pid, f'win:{entry.creation + 1}', None, None, None)
        assert not wrong.alive()
        assert wrong.kill().status == 'stopped'
        assert child.process.poll() is None
        wrapped = native.NativeChild(entry.pid, entry.started, None, None, None)
    monkeypatch.setattr(winproc, 'open', lambda *args: pytest.fail('reopened a pid'))
    assert wrapped.alive()
    assert not wrapped.wait_gone(0)
    assert wrapped.kill().status == 'terminated'
    assert wrapped.wait_gone(5)


@pytest.mark.wall_clock
def test_console_close_clears_rts_before_exit_while_handle_is_held(children, tmp_path):
    from hfmodem.tests.hermetic import allow_virtual_serial
    serial = pytest.importorskip('serial')
    for port in ('COM41', 'COM42'):
        try:
            target = winclaim.serial_target(port)
        except claim.ClaimError:
            pytest.skip(f'com0com {port} is absent')
        if 'cnc' not in target.casefold() and 'com0com' not in target.casefold():
            pytest.skip(f'{port} is not a com0com device')
    allow_virtual_serial('COM41', 'COM42')
    sense = serial.Serial()
    sense.port, sense.rts, sense.dtr = 'COM42', False, False
    sense.open()
    log, held, release = [tmp_path / name for name in ('events.jsonl', 'held', 'release')]
    try:
        child = children('''
import serial
from pathlib import Path
s = serial.Serial()
s.port, s.rts, s.dtr = 'COM41', False, False
s.open()
windeadman.register_handle(s._port_handle)
signal.signal(signal.SIGTERM, lambda *args: threading.Event().wait())
clear = windeadman.drop_lines
def observed_clear():
    clear()
    from hfmodem.core import coexist
    coexist.event('deadman_clear')
windeadman.drop_lines = observed_clear
exit_process = os._exit
def hold_exit(code):
    posix._windows_deadman._word.value = 0
    Path(sys.argv[1]).touch()
    while not Path(sys.argv[2]).exists(): time.sleep(.005)
    exit_process(code)
os._exit = hold_exit
s.rts = True
k = C.WinDLL('kernel32', use_last_error=True)
k.GetConsoleWindow.restype = C.c_void_p
print(k.GetConsoleWindow(), flush=True)
threading.Event().wait()
''', held, release, flags=subprocess.CREATE_NEW_CONSOLE, env={'HFMODEM_COEXIST_LOG': str(log)})
        hwnd = int(child.line())
        assert hwnd
        deadline = time.monotonic() + 2
        while not sense.cts and time.monotonic() < deadline:
            time.sleep(.005)
        assert sense.cts
        user = C.WinDLL('user32', use_last_error=True)
        user.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
        user.PostMessageW.restype = W.BOOL
        assert user.PostMessageW(hwnd, 0x0010, 0, 0)
        deadline = time.monotonic() + 3
        while not held.exists() and time.monotonic() < deadline:
            time.sleep(.005)
        assert held.exists()
        assert child.process.poll() is None
        assert not sense.cts
        with pytest.raises(serial.SerialException):
            serial.Serial('COM41')
        events = [json.loads(line) for line in log.read_text().splitlines()]
        assert [event['event'] for event in events] == ['console_close', 'deadman_clear']
        assert not Path(str(log) + '.lost').exists()
        release.touch()
        assert child.process.wait(timeout=5) == 1
    finally:
        release.touch()
        sense.close()


def test_process_handle_probe_on_this_build_and_growth(monkeypatch):
    own = winproc._api().GetCurrentProcess()
    handle = winclaim.open_handle(os.devnull, 0, 7)
    try:
        rows = winproc.process_handles(own)
        assert any(value == handle for value, index in rows)
        assert winproc._file_type_index() == dict(rows)[handle]
    finally:
        winclaim.close_handle(handle)
    sizes = []
    def query(process, information, data, size, needed):
        assert process == own and information == 51
        sizes.append(size)
        C.cast(needed, C.POINTER(W.DWORD))[0] = 4096
        if size < 4096:
            return winproc._MISMATCH
        C.c_size_t.from_buffer(data).value = 0
        return 0
    monkeypatch.setattr(winproc._api(), 'NtQueryInformationProcess', query)
    assert winproc.process_handles(own) == []
    assert sizes == [256, 4096]
    monkeypatch.setattr(winproc._api(), 'NtQueryInformationProcess', lambda *args: 0xc0000022)
    with pytest.raises(winproc.InventoryUnavailable, match='ProcessHandleInformation'):
        winproc.process_handles(own)


@pytest.mark.wall_clock
def test_python_suspended_runs_in_venv_and_push_is_not_inheritable(tmp_path):
    marker, reservation = tmp_path / 'ran', tmp_path / 'reservation'
    handle = winclaim.open_handle(reservation, 1 | 2, 0, winclaim.OPEN_ALWAYS)
    process = winproc.python_suspended(['-c', '''
import json, os, sys
from pathlib import Path
from hfmodem.core import winproc
record = winproc.read_launch()
h = record['handles']['test']
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), sys.prefix, sys.executable,
                                       os.get_handle_inheritable(h)]))
''', str(marker)], creationflags=subprocess.CREATE_NO_WINDOW)
    child = winproc._pin(winclaim.duplicate(int(process._handle)), process.pid, winproc.END)
    try:
        winproc.create_launch(child, role='inspector')
        pushed = winproc.push_handle(child, handle)
        winproc.advance_launch(child, 'resuming', handles={'test': pushed})
        time.sleep(.2)
        assert not marker.exists()
        assert process.poll() is None
        winproc.resume(child)
        assert process.wait(timeout=10) == 0
        pid, prefix, executable, inheritable = json.loads(marker.read_text())
        assert pid == process.pid
        assert Path(prefix) == Path(sys.prefix)
        assert Path(executable) == Path(sys.executable)
        assert inheritable is False
    finally:
        if not child.wait():
            child.terminate()
        assert child.wait(5)
        child.close()
        winclaim.close_handle(handle)


@pytest.mark.parametrize('failure', ['count', 'owner', 'zero', 'nested', 'failed'])
def test_resume_protocol_fault_terminates_never_run_child(tmp_path, monkeypatch, failure):
    marker = tmp_path / 'ran'
    process = winproc.python_suspended(['-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()',
                                        str(marker)])
    child = winproc._pin(winclaim.duplicate(int(process._handle)), process.pid, winproc.END)
    try:
        api = winproc._api()
        if failure == 'count':
            def no_threads(*args):
                C.set_last_error(18)
                return False
            monkeypatch.setattr(api, 'Thread32First', no_threads)
        elif failure == 'owner':
            monkeypatch.setattr(api, 'GetProcessIdOfThread', lambda *args: child.pid + 1)
        else:
            monkeypatch.setattr(api, 'ResumeThread', lambda *args: {'zero': 0, 'nested': 2,
                                                                 'failed': 0xffffffff}[failure])
        with pytest.raises(claim.ClaimError, match='cannot resume'):
            winproc.resume(child)
        assert child.wait(5)
        assert not marker.exists()
    finally:
        if not child.wait():
            child.terminate()
        child.close()
        process.wait(timeout=5)


@pytest.fixture
def com41():
    from hfmodem.tests.hermetic import allow_virtual_serial
    serial = pytest.importorskip('serial')
    try:
        target = winclaim.serial_target('COM41')
    except claim.ClaimError:
        pytest.skip('com0com COM41 is absent')
    if not any(name in target.casefold() for name in ('cnc', 'com0com')):
        pytest.skip('COM41 is not a com0com device')
    allow_virtual_serial('COM41')
    return serial


@pytest.fixture
def com41_com42(com41):
    from hfmodem.tests.hermetic import allow_virtual_serial
    try:
        target = winclaim.serial_target('COM42')
    except claim.ClaimError:
        pytest.skip('com0com COM42 is absent')
    if not any(name in target.casefold() for name in ('cnc', 'com0com')):
        pytest.skip('COM42 is not a com0com device')
    allow_virtual_serial('COM42')
    return com41


SERIAL_HOLDER = r'''
import serial
from ctypes import wintypes as W
from pathlib import Path
release = Path(sys.argv[1])
api = winclaim._api()
kernel = C.WinDLL('kernel32', use_last_error=True)
class COMMTIMEOUTS(C.Structure):
    _fields_ = [(name, W.DWORD) for name in ('ReadIntervalTimeout',
        'ReadTotalTimeoutMultiplier', 'ReadTotalTimeoutConstant',
        'WriteTotalTimeoutMultiplier', 'WriteTotalTimeoutConstant')]
kernel.SetCommTimeouts.argtypes = [W.HANDLE, C.POINTER(COMMTIMEOUTS)]
kernel.SetCommTimeouts.restype = W.BOOL
count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
def blocked_serial():
    handle = api.CreateFileW(r'\\.\COM42', 0xc0000000, 0, None, 3, 0, None)
    assert handle not in (None, winclaim.INVALID_HANDLE_VALUE), C.get_last_error()
    assert kernel.SetCommTimeouts(handle, C.byref(COMMTIMEOUTS())), C.get_last_error()
    handles = [handle] + [winclaim.duplicate(handle) for _ in range(count - 1)]
    workers, results = [], []
    for handle in handles:
        entered = threading.Event()
        def read(handle=handle, entered=entered):
            data, size = C.create_string_buffer(1), W.DWORD()
            entered.set()
            ok = api.ReadFile(handle, data, 1, C.byref(size), None)
            results.append((handle, bool(ok), size.value, data.raw))
        worker = threading.Thread(target=read)
        worker.start()
        assert entered.wait(5)
        workers.append(worker)
        if len(workers) == 1:
            time.sleep(.1)
            assert worker.is_alive() and not results
    time.sleep(.1)
    assert all(worker.is_alive() for worker in workers) and not results
    return handles, workers, results
if sys.argv[2] == 'before': handles, workers, results = blocked_serial()
s = serial.Serial()
s.port, s.rts, s.dtr = 'COM41', False, False
s.open()
if sys.argv[2] == 'after': handles, workers, results = blocked_serial()
print(json.dumps([handles, s._port_handle]), flush=True)
while not release.exists(): time.sleep(.005)
assert s.write(b'x') == 1
workers[0].join(5)
assert not workers[0].is_alive()
assert results == [(handles[0], True, 1, b'x')]
assert all(worker.is_alive() for worker in workers[1:])
if count > 1:
    assert s.write(b'x' * (count - 1)) == count - 1
for worker in workers:
    worker.join(5)
    assert not worker.is_alive()
assert sorted(results) == sorted((handle, True, 1, b'x') for handle in handles)
for handle in handles: winclaim.close_handle(handle)
s.close()
print('released', flush=True)
'''


PIPE_HOLDER = '''
import serial
from pathlib import Path
release = Path(sys.argv[1])
def pipe():
    r, w = os.pipe()
    entered = threading.Event()
    def read():
        entered.set()
        os.read(r, 1)
    worker = threading.Thread(target=read)
    worker.start()
    assert entered.wait(5)
    return r, w, worker
if sys.argv[2] == 'before': r, w, worker = pipe()
s = serial.Serial()
s.port, s.rts, s.dtr = 'COM41', False, False
s.open()
if sys.argv[2] == 'after': r, w, worker = pipe()
time.sleep(.1)
assert worker.is_alive()
print(json.dumps([winclaim.handle(r), s._port_handle]), flush=True)
while not release.exists(): time.sleep(.005)
os.write(w, b'x')
worker.join(5)
assert not worker.is_alive()
os.close(r)
os.close(w)
s.close()
print('released', flush=True)
'''


def await_condition(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.005)
    pytest.fail('condition did not become true')


def inspector_records():
    records = [winproc._read_json(path) for path in
               (claim.state_dir() / 'rig.claim.launch.d').glob('*.json')]
    return [record for record in records if record is not None and record['role'] == 'inspector']


def slot_free():
    try:
        handle = winclaim.open_handle(claim.state_dir() / 'rig.claim.inspect', 1 | 2, 0,
                                      winclaim.OPEN_ALWAYS)
    except OSError as exc:
        assert exc.winerror == 32
        return False
    winclaim.close_handle(handle)
    return True


def walk_thread(pinned, *, deadline=None, cancelled=None):
    results = []
    thread = threading.Thread(target=lambda: results.append(winproc.device_holders(
        pinned, {'COM41'}, deadline=deadline or time.monotonic() + 15, cancelled=cancelled)))
    thread.start()
    return thread, results


@contextmanager
def pending_inspector_exit(monkeypatch, inspector):
    end, wait = winproc._end_inspector, winproc.Pinned.wait
    waiting, released = threading.Event(), threading.Event()
    def pending_end(entry):
        # Keep the slot alive while modelling a driver that delays process exit.
        if entry.identity != inspector.identity:
            end(entry)
    def pending_wait(entry, timeout=0):
        if entry.identity == inspector.identity and timeout == winproc.KILL_WAIT_S:
            waiting.set()
            if not released.wait(timeout):
                return False
        return wait(entry, timeout)
    with monkeypatch.context() as patch:
        patch.setattr(winproc, '_end_inspector', pending_end)
        patch.setattr(winproc.Pinned, 'wait', pending_wait)
        try:
            yield waiting
        finally:
            end(inspector)
            released.set()
            assert wait(inspector, 5)


@contextmanager
def foreign_handle(process, handle):
    api, duplicate = winproc._api(), W.HANDLE()
    assert api.DuplicateHandle(process, handle, api.GetCurrentProcess(),
                               C.byref(duplicate), 0, False, 2), C.get_last_error()
    try:
        yield duplicate.value
    finally:
        winclaim.close_handle(duplicate)


@pytest.mark.wall_clock
@pytest.mark.parametrize('order', ['before', 'after'])
def test_n9_pending_pipe_is_other_and_serial_is_found(children, com41, tmp_path, order):
    release = tmp_path / 'release'
    holder = children(PIPE_HOLDER, release, order)
    pipe, port = json.loads(holder.line())
    try:
        with holder.entry() as entry:
            result = winproc.device_holders([entry], {'COM41'}, deadline=time.monotonic() + 10)
            walk = result.entries[entry.pid]
            assert walk.devices == {'COM41'}
            assert walk.handles[pipe] == 'other'
            assert walk.handles[port] == 'found'
            assert walk.counts['blocked'] == 0
            assert walk.complete and result.status == 'complete'
    finally:
        release.touch()
    assert holder.line() == 'released'
    assert holder.process.wait(timeout=5) == 0
    await_condition(slot_free)


@pytest.mark.wall_clock
@pytest.mark.parametrize('order', ['before', 'after'])
def test_n9_blocked_serial_does_not_hide_serial_and_only_abandoned_duplicates_remain(
        children, com41_com42, tmp_path, monkeypatch, order):
    release, barriers, log = tmp_path / 'release', tmp_path / 'barriers', tmp_path / 'events'
    barriers.mkdir()
    for point in ('inspector_after_walk', 'inspector_before_reap'):
        (barriers / point).touch()
    monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
    monkeypatch.setenv('HFMODEM_COEXIST_LOG', str(log))
    holder = children(SERIAL_HOLDER, release, order)
    (blocked,), port = json.loads(holder.line())
    with holder.entry(rights=winproc.INSPECT) as entry:
        thread, result = walk_thread([entry])
        try:
            await_condition(lambda: (barriers / 'inspector_before_reap.reached').exists())
            record, = inspector_records()
            with winproc.open(record['child']['pid'], winproc.INSPECT) as inspector:
                events = [json.loads(line) for line in log.read_text().splitlines()]
                counts = {event['stage']: event for event in events
                          if event['event'] == 'inspector_handles'}
                assert counts['final']['abandoned'] == 1
                api, file_index = winproc._api(), winproc._file_type_index()
                duplicates, serial_handles = [], []
                with foreign_handle(entry.handle, blocked) as reference:
                    for handle, index in winproc.process_handles(inspector.handle):
                        if index != file_index:
                            continue
                        with foreign_handle(inspector.handle, handle) as duplicate:
                            if api.CompareObjectHandles(reference, duplicate):
                                duplicates.append(handle)
                            iosb, device = winclaim.IO_STATUS_BLOCK(), (W.ULONG * 2)()
                            status = api.NtQueryVolumeInformationFile(duplicate, C.byref(iosb),
                                                                     device, C.sizeof(device), 4)
                            assert status == 0, f'FileFsDeviceInformation: {status:#x}'
                            if device[0] == 0x1b:
                                serial_handles.append(handle)
                assert len(duplicates) == 1
                assert serial_handles == duplicates
            (barriers / 'inspector_before_reap.release').touch()
            thread.join(3)
            assert not thread.is_alive()
            walk = result[0].entries[entry.pid]
            assert walk.devices == {'COM41'}
            assert walk.handles[port] == 'found'
            assert walk.handles[blocked] == 'blocked'
            assert walk.counts['blocked'] == 1
            assert not walk.complete and result[0].status == 'incomplete'
            await_condition(lambda: any(r['phase'] == 'exited' for r in inspector_records()))
        finally:
            (barriers / 'inspector_before_reap.release').touch()
            (barriers / 'inspector_after_walk.release').touch()
            release.touch()
            thread.join(5)
    assert holder.line() == 'released'
    assert holder.process.wait(timeout=5) == 0
    await_condition(slot_free)
    serial = com41_com42.Serial()
    serial.port, serial.rts, serial.dtr = 'COM41', False, False
    serial.open()
    serial.close()


@pytest.mark.wall_clock
def test_inspector_with_blocked_serial_worker_exits_on_terminate(
        children, com41_com42, tmp_path, monkeypatch):
    release, barriers = tmp_path / 'release', tmp_path / 'barriers'
    barriers.mkdir()
    for point in ('inspector_after_walk', 'inspector_before_reap'):
        (barriers / point).touch()
    monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
    holder = children(SERIAL_HOLDER, release, 'before')
    (blocked,), _ = json.loads(holder.line())
    with holder.entry() as entry:
        thread, results = walk_thread([entry])
        try:
            await_condition(lambda: (barriers / 'inspector_before_reap.reached').exists())
            record, = inspector_records()
            with winproc.open(record['child']['pid']) as inspector:
                assert not inspector.wait()
                at = time.monotonic()
                (barriers / 'inspector_before_reap.release').touch()
                assert inspector.wait(1)
                assert time.monotonic() - at <= 1
            thread.join(1)
            assert not thread.is_alive()
            assert results[0].entries[entry.pid].handles[blocked] == 'blocked'
            await_condition(lambda: any(r['phase'] == 'exited' for r in inspector_records()))
            assert slot_free()
            assert holder.process.poll() is None
        finally:
            (barriers / 'inspector_before_reap.release').touch()
            (barriers / 'inspector_after_walk.release').touch()
            release.touch()
            thread.join(5)
    assert holder.line() == 'released'
    assert holder.process.wait(timeout=5) == 0


@pytest.mark.wall_clock
@pytest.mark.parametrize('reason', ['deadline', 'cancelled'])
def test_inspection_returns_on_callers_bound_and_reaps_off_path(
        children, com41_com42, tmp_path, monkeypatch, reason):
    release, barriers = tmp_path / 'release', tmp_path / 'barriers'
    barriers.mkdir()
    for point in ('inspector_before_read', 'inspector_after_walk'):
        (barriers / point).touch()
    monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
    holder = children(SERIAL_HOLDER, release, 'before')
    holder.line()
    cancelled = threading.Event()
    with holder.entry() as entry:
        # Hold the coordinator after launch so startup speed cannot decide this test.
        deadline = time.monotonic() + 2
        thread, results = walk_thread([entry], deadline=deadline, cancelled=cancelled)
        try:
            await_condition(lambda: (barriers / 'inspector_before_read.reached').exists(), timeout=1.5)
            await_condition(lambda: (barriers / 'inspector_after_walk.reached').exists(),
                            timeout=max(0, deadline - time.monotonic() - .1))
            record, = inspector_records()
            if reason == 'cancelled':
                at = time.monotonic()
                cancelled.set()
                thread.join(.075)
                assert time.monotonic() - at < .075
            else:
                thread.join(max(0, deadline - time.monotonic()) + .05)
                assert time.monotonic() < deadline + .05
            assert not thread.is_alive()
            assert results[0].status == reason
            assert not results[0].entries[entry.pid].complete
            assert not slot_free()
            # Reclamation is still pending while the decision has already returned.
            with winproc.open(record['child']['pid'], winproc.END) as inspector:
                assert not inspector.wait()
                with pending_inspector_exit(monkeypatch, inspector) as waiting:
                    (barriers / 'inspector_before_read.release').touch()
                    assert waiting.wait(1)
                    await_condition(lambda: any(r['phase'] == 'unresolved' for r in inspector_records()),
                                    timeout=8)
                    assert not inspector.wait()
                    assert not slot_free()
                    unavailable = winproc.device_holders([entry], {'COM41'}, deadline=time.monotonic() + 2)
                    assert unavailable.status == 'unavailable'
                    assert unavailable.message == (f'inspection unavailable: pid {inspector.pid} '
                                                   '(inspector) has not exited')
            await_condition(slot_free)
        finally:
            (barriers / 'inspector_before_read.release').touch()
            (barriers / 'inspector_after_walk.release').touch()
            release.touch()
            thread.join(5)
    assert holder.line() == 'released'
    await_condition(slot_free)


@pytest.mark.wall_clock
def test_two_callers_share_one_inspector_slot(children, com41, tmp_path, monkeypatch):
    barriers = tmp_path / 'barriers'
    barriers.mkdir()
    (barriers / 'before_inspector_resume').touch()
    monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
    target = children("print('ready', flush=True); time.sleep(60)")
    assert target.line() == 'ready'
    with target.entry() as own:
        thread, result = walk_thread([own])
        try:
            await_condition(lambda: (barriers / 'before_inspector_resume.reached').exists())
            record, = inspector_records()
            second = winproc.device_holders([own], {'COM41'}, deadline=time.monotonic() + 3)
            assert second.status == 'busy' and second.message == 'inspection busy'
            assert [r['child']['pid'] for r in inspector_records()] == [record['child']['pid']]
        finally:
            (barriers / 'before_inspector_resume.release').touch()
            thread.join(10)
    assert not thread.is_alive()
    assert len(result) == 1
    await_condition(slot_free)


@pytest.mark.wall_clock
@pytest.mark.parametrize('point', ['after_inspector_created', 'after_inspector_slot', 'before_inspector_resume',
                                  'inspector_before_read', 'inspector_before_reap'])
def test_n9_dead_launcher_is_recovered_in_every_phase(children, com41_com42, tmp_path, monkeypatch, point):
    release, barriers = tmp_path / 'release', tmp_path / 'barriers'
    barriers.mkdir()
    # Keep a resumed orphan alive after its last line, before it can call _exit.
    (barriers / 'inspector_after_walk').touch()
    if point == 'inspector_before_read':
        (barriers / point).touch()
    holder = children(SERIAL_HOLDER, release, 'before')
    holder.line()
    caller = children('''
with winproc.open(int(sys.argv[1])) as target:
    winproc.device_holders([target], {'COM41'}, deadline=time.monotonic() + 20)
''', holder.process.pid, env={'HFMODEM_COEXIST_FAULT': point + ':kill',
                             'HFMODEM_COEXIST_BARRIERS': str(barriers)})
    try:
        if point == 'inspector_before_read':
            await_condition(lambda: (barriers / 'inspector_after_walk.reached').exists())
            (barriers / (point + '.release')).touch()
        assert caller.process.wait(timeout=10) == 1
        record, = inspector_records()
        assert record['phase'] == ('created' if point in ('after_inspector_created', 'after_inspector_slot')
                                   else 'resuming')
        assert slot_free() == (point == 'after_inspector_created')
        with winproc.open(record['child']['pid'], winproc.END) as inspector:
            assert not inspector.wait()
            pending = point in ('inspector_before_read', 'inspector_before_reap')
            reclamation = pending_inspector_exit(monkeypatch, inspector) if pending else nullcontext()
            with reclamation as waiting:
                with holder.entry() as entry:
                    deadline = time.monotonic() + (.25 if point == 'inspector_before_read' else 10)
                    recovered = winproc.device_holders([entry], {'COM41'}, deadline=deadline)
                    if point == 'inspector_before_read':
                        assert time.monotonic() <= deadline + .05
                        assert recovered.status == 'deadline'
                        assert waiting.wait(1)
                        assert not slot_free()
                        await_condition(lambda: any(r['phase'] == 'unresolved' for r in inspector_records()),
                                        timeout=8)
                    if pending:
                        assert recovered.status == ('deadline' if point == 'inspector_before_read'
                                                     else 'unavailable')
                        recovered = winproc.device_holders([entry], {'COM41'}, deadline=time.monotonic() + 2)
                if pending:
                    assert waiting.is_set()
                    assert recovered.status == 'unavailable'
                    assert recovered.message == (f'inspection unavailable: pid {inspector.pid} '
                                                  '(inspector) has not exited')
                    assert any(r['phase'] == 'unresolved' for r in inspector_records())
                    assert not inspector.wait()
                    assert not slot_free()
                else:
                    assert inspector.wait()
                    assert recovered.entries[holder.process.pid].devices == {'COM41'}
        await_condition(slot_free)
    finally:
        (barriers / 'inspector_before_read.release').touch()
        (barriers / 'inspector_after_walk.release').touch()
        release.touch()
    assert holder.line() == 'released'
    await_condition(slot_free)


@pytest.mark.parametrize('reason', ['alive', 'denied', 'gone', 'reused', 'signalled'])
def test_inspector_recovery_denied_end_requires_confirmed_exit(tmp_path, monkeypatch, reason):
    directory = tmp_path / 'rig.claim.launch.d'
    directory.mkdir()
    path = directory / 'inspector.json'
    identity = {'pid': 1234, 'started': 'win:100'}
    record = {'role': 'inspector', 'phase': 'created', 'child': identity}
    path.write_text(json.dumps(record))
    monkeypatch.setattr(winproc, '_read_json', lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(winproc, '_prune_record', lambda path: path.unlink(missing_ok=True))
    requested, closed = [], []
    entry = winproc.Pinned(99, identity['pid'], 'win:101' if reason == 'reused' else 'win:100',
                           'inspector.exe')
    monkeypatch.setattr(winproc, '_wait', lambda handle, timeout: reason == 'signalled')
    monkeypatch.setattr(winclaim, 'close_handle', closed.append)
    def denied(pid, rights=winproc.IDENTIFY):
        assert pid == identity['pid']
        requested.append(rights)
        if rights == winproc.END or reason == 'denied':
            raise winproc.ProcessAccessError(pid, rights)
        return None if reason == 'gone' else entry
    monkeypatch.setattr(winproc, 'open', denied)
    gone = reason in ('gone', 'reused', 'signalled')
    assert winproc._recover_inspectors(tmp_path) == (False, None if gone else identity['pid'])
    assert requested == [winproc.END, winproc.IDENTIFY]
    assert path.exists() == (not gone)
    assert closed == ([] if reason in ('denied', 'gone') else [99])


@pytest.mark.wall_clock
@pytest.mark.parametrize('identify_denied', [False, True])
def test_inspector_recovery_denied_end_cannot_launch_another(children, monkeypatch, identify_denied):
    caller = children('''
winproc.device_holders([], (), deadline=time.monotonic() + 20)
''', env={'HFMODEM_COEXIST_FAULT': 'after_inspector_created:kill'})
    assert caller.process.wait(timeout=10) == 1
    record, = inspector_records()
    assert record['phase'] == 'created' and not record['handles']
    with winproc.open(record['child']['pid'], winproc.END) as inspector:
        try:
            assert slot_free()
            assert not inspector.wait()
            real_open, requested = winproc.open, []
            def denied(pid, rights=winproc.IDENTIFY):
                if pid == inspector.pid:
                    requested.append(rights)
                    if rights == winproc.END or identify_denied:
                        raise winproc.ProcessAccessError(pid, rights)
                return real_open(pid, rights)
            def no_launch(*args, **kwargs):
                pytest.fail('cannot launch before the recorded inspector is confirmed gone')
            with monkeypatch.context() as patch:
                patch.setattr(winproc, 'open', denied)
                patch.setattr(winproc, 'python_suspended', no_launch)
                result = winproc.device_holders([], (), deadline=time.monotonic() + 3)
            assert result.status == 'unavailable'
            assert result.message == (f'inspection unavailable: pid {inspector.pid} '
                                      '(inspector) has not exited')
            assert requested == [winproc.END, winproc.IDENTIFY]
            assert not inspector.wait()
            assert inspector_records() == [record]
            await_condition(slot_free)
        finally:
            winproc._end_inspector(inspector)
            assert inspector.wait(5)


@pytest.mark.wall_clock
@pytest.mark.parametrize('finish', ['release', 'kill'])
def test_contested_inspector_recovery_cannot_launch_another(children, tmp_path, finish):
    barriers = tmp_path / 'barriers'
    barriers.mkdir()
    point = 'after_inspector_reap_guard'
    (barriers / point).touch()
    caller = children('''
winproc.device_holders([], (), deadline=time.monotonic() + 20)
''', env={'HFMODEM_COEXIST_FAULT': 'after_inspector_created:kill'})
    assert caller.process.wait(timeout=10) == 1
    record, = inspector_records()
    assert record['phase'] == 'created' and not record['handles']
    with winproc.open(record['child']['pid'], winproc.END) as inspector:
        slot = winclaim.open_handle(claim.state_dir() / 'rig.claim.inspect', 1 | 2, 0,
                                    winclaim.OPEN_ALWAYS)
        recoverer = None
        try:
            # Make A enter recovery through the busy-slot path before B takes the slot.
            recoverer = children('''
result = winproc.device_holders([], (), deadline=time.monotonic() + 20)
print(result.status, flush=True)
''', env={'HFMODEM_COEXIST_BARRIERS': str(barriers)})
            await_condition(lambda: (barriers / (point + '.reached')).exists())
            assert not inspector.wait()
            winclaim.close_handle(slot)
            slot = None
            assert slot_free()
            second = winproc.device_holders([], (), deadline=time.monotonic() + 3)
            assert second.status == 'unavailable'
            assert second.message == (f'inspection unavailable: pid {inspector.pid} '
                                      '(inspector) has not exited')
            assert not inspector.wait()
            assert [r['child']['pid'] for r in inspector_records()] == [inspector.pid]
            if finish == 'release':
                (barriers / (point + '.release')).touch()
                assert recoverer.line() == 'complete'
                assert recoverer.process.wait(timeout=10) == 0
            else:
                recoverer.process.kill()
                recoverer.process.wait(timeout=5)
                assert not inspector.wait()
                recovered = winproc.device_holders([], (), deadline=time.monotonic() + 10)
                assert recovered.status == 'complete'
            assert inspector.wait(1)
            await_condition(slot_free)
        finally:
            if slot is not None:
                winclaim.close_handle(slot)
            if recoverer is not None:
                if recoverer.process.poll() is None:
                    recoverer.process.kill()
                recoverer.process.wait(timeout=5)
            winproc._end_inspector(inspector)
            assert inspector.wait(5)


def test_inspect_access_comes_from_pinned_handle(children, monkeypatch):
    target = children("print('ready', flush=True); threading.Event().wait()")
    assert target.line() == 'ready'
    api = winproc._api()
    duplicate, open_process = api.DuplicateHandle, api.OpenProcess
    checked = []
    with target.entry(rights=winproc.IDENTIFY) as entry:
        def duplicate_handle(source, handle, destination, output, access, inherit, options):
            ok = duplicate(source, handle, destination, output, access, inherit, options)
            if access == 0x0040 | 0x0400:
                assert source == destination == api.GetCurrentProcess()
                assert not inherit and options == 0
                assert ok
                value = C.cast(output, C.POINTER(W.HANDLE))[0]
                assert api.CompareObjectHandles(entry.handle, handle)
                assert api.CompareObjectHandles(entry.handle, value)
                assert not os.get_handle_inheritable(value)
                winproc.process_handles(value)
                checked.append(value)
            return ok
        def no_reopen(access, inherit, pid):
            assert pid != entry.pid
            return open_process(access, inherit, pid)
        monkeypatch.setattr(api, 'DuplicateHandle', duplicate_handle)
        monkeypatch.setattr(api, 'OpenProcess', no_reopen)
        result = winproc.device_holders([entry], (), deadline=time.monotonic() + 5)
        assert len(checked) == 1
        assert result.status == 'complete'
        assert result.entries[entry.pid].complete
    await_condition(slot_free)


def test_cannot_inspect_keeps_negative_evidence_incomplete(children, monkeypatch):
    target = children("print('ready', flush=True); input()")
    assert target.line() == 'ready'
    duplicate = winproc._api().DuplicateHandle
    def denied(source, handle, destination, output, access, inherit, options):
        if access == 0x0040 | 0x0400:
            C.set_last_error(5)
            return False
        return duplicate(source, handle, destination, output, access, inherit, options)
    monkeypatch.setattr(winproc._api(), 'DuplicateHandle', denied)
    with target.entry() as entry:
        result = winproc.device_holders([entry], (), deadline=time.monotonic() + 5)
        walk = result.entries[entry.pid]
        assert walk.message == f'cannot inspect pid {entry.pid} ({Path(entry.image).name})'
        assert not walk.complete and not walk.devices
        assert result.status == 'incomplete'
    await_condition(slot_free)


@pytest.mark.wall_clock
@pytest.mark.parametrize('reason', ['deadline', 'cancelled', 'abandoned'])
def test_live_walk_deadline_cancellation_and_abandoned_limit(
        children, com41_com42, tmp_path, monkeypatch, reason):
    release = tmp_path / 'release'
    holder = children(SERIAL_HOLDER, release, 'before', 10)
    blocked, _ = json.loads(holder.line())
    cancelled, seen = threading.Event(), []
    line = winproc._Inspection.line
    def observe(self, row):
        result = line(self, row)
        if row['status'] == 'blocked' and not seen:
            seen.append(time.monotonic())
            if reason == 'cancelled':
                cancelled.set()
        return result
    monkeypatch.setattr(winproc._Inspection, 'line', observe)
    with holder.entry() as entry:
        deadline = time.monotonic() + (1.2 if reason == 'deadline' else 10)
        try:
            result = winproc.device_holders([entry], {'COM41'}, deadline=deadline, cancelled=cancelled)
            returned = time.monotonic()
            assert seen, 'inspector must reach the pending read before the bound fires'
            if reason == 'cancelled':
                assert returned - seen[0] <= .075
            elif reason == 'deadline':
                assert deadline <= returned <= deadline + .05
            else:
                walk = result.entries[entry.pid]
                assert walk.counts['blocked'] == 9
                assert sum(walk.handles.get(handle) == 'blocked' for handle in blocked) == 9
            assert result.status == ('incomplete' if reason == 'abandoned' else reason)
            assert not result.entries[entry.pid].complete
        finally:
            release.touch()
    assert holder.line() == 'released'
    await_condition(slot_free)


@pytest.mark.parametrize('outcome', ['pass', 'fail'])
def test_filtered_test_environment_and_pipe_report(
        tmp_path, tmp_path_factory, monkeypatch, request, capsys, outcome):
    if os.environ.get('HFMODEM_FILTERED_TEST'):
        assert sys.prefix == os.environ['HFMODEM_EXPECTED_PREFIX']
        assert Path.cwd() == request.config.rootpath
        basetemp = tmp_path_factory.getbasetemp()
        assert tmp_path.is_relative_to(basetemp)
        assert any(basetemp.glob('claim-state*'))
        inherited = basetemp.parent / 'inherited'
        inherited.mkdir()
        advapi = C.WinDLL('advapi32', use_last_error=True)
        P = C.c_void_p
        advapi.GetNamedSecurityInfoW.argtypes = [W.LPWSTR, C.c_int, W.DWORD,
                                                P, P, P, P, C.POINTER(P)]
        advapi.GetNamedSecurityInfoW.restype = W.DWORD
        advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
            P, W.DWORD, W.DWORD, C.POINTER(W.LPWSTR), P]
        advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = W.BOOL
        api = winproc._api()
        token, user_text = W.HANDLE(), W.LPWSTR()
        try:
            assert api.OpenProcessToken(api.GetCurrentProcess(), 8, C.byref(token))
            assert api.ConvertSidToStringSidW(winclaim._token_sid(token, 1),
                                              C.byref(user_text))
            user = user_text.value
        finally:
            if user_text:
                api.LocalFree(C.cast(user_text, P))
            if token:
                winclaim.close_handle(token)
        for path in (basetemp.parent, inherited):
            descriptor, text = P(), W.LPWSTR()
            try:
                error = advapi.GetNamedSecurityInfoW(str(path), 1, 0x4, None, None,
                                                     None, None, C.byref(descriptor))
                assert error == 0, C.WinError(error)
                assert advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                    descriptor, 1, 0x4, C.byref(text), None), C.WinError(C.get_last_error())
                aces = [ace.split(';') for ace in text.value.split('(')[1:]]
                for trustee in (user, 'SY'):
                    assert any(ace[0] == 'A' and 'OI' in ace[1] and 'CI' in ace[1]
                               and ace[2] == 'FA' and ace[5] == trustee + ')'
                               for ace in aces), text.value
            finally:
                for memory in (descriptor, C.cast(text, P)):
                    if memory:
                        winproc._api().LocalFree(memory)
        (inherited / 'writable.txt').write_text('inherited access', encoding='utf-8')
        state = Path(os.environ['HFMODEM_STATE_DIR'])
        assert state.is_relative_to(basetemp)
        state.mkdir(parents=True)
        (state / 'medium.txt').write_text('writable', encoding='utf-8')
        nested = state / 'nested'
        nested.mkdir(mode=0o700)
        (nested / 'user.txt').write_text('writable', encoding='utf-8')
        print('x' * 65536)
        print('filtered stderr report', file=sys.stderr)
        if outcome == 'fail':
            pytest.fail('filtered failure report')
        return
    from hfmodem.tests.core.win_token import filtered_test
    monkeypatch.setenv('HFMODEM_EXPECTED_PREFIX', sys.prefix)
    monkeypatch.delenv('__PYVENV_LAUNCHER__', raising=False)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    directory = tmp_path / 'filtered'
    monkeypatch.chdir(tmp_path)
    if outcome == 'fail':
        with pytest.raises(AssertionError, match='filtered failure report') as failure:
            filtered_test(request, directory)
        report = str(failure.value)
    else:
        filtered_test(request, directory)
        report = capsys.readouterr().out
    assert 'x' * 65536 in report
    assert 'filtered stderr report' in report


@pytest.mark.parametrize('token', ['elevated', 'filtered'])
@pytest.mark.parametrize('case', ['U1-opened', 'U2-inherited', 'U3-pushed',
                                 'U4-orphan-duplicate', 'U5-twenty', 'U6-attributes'])
def test_using_every_holder(children, tmp_path, monkeypatch, request, token, case):
    if token == 'filtered' and not os.environ.get('HFMODEM_FILTERED_TEST'):
        from hfmodem.tests.core.win_token import filtered_test
        filtered_test(request, tmp_path / 'filtered')
        return
    path = tmp_path / 'users.bin'
    path.touch()
    assert os.getpid() not in dict(winproc.using(path))
    print(f'{token}: query-only handle excludes pid {os.getpid()}')
    holders = []
    source = None
    if case in ('U1-opened', 'U5-twenty', 'U6-attributes'):
        for _ in range(20 if case == 'U5-twenty' else 1):
            child = children('''
h = winclaim.open_handle(sys.argv[1], int(sys.argv[2]), 7)
print('ready', flush=True)
input()
winclaim.close_handle(h)
''', path, 0x80 if case == 'U6-attributes' else 1 | 2)
            assert child.line() == 'ready'
            holders.append(child.process.pid)
    elif case == 'U2-inherited':
        source = winclaim.open_handle(path, 1 | 2, 7)
        try:
            os.set_handle_inheritable(source, True)
            child = children('''
h = int(sys.argv[1])
assert winclaim.file_id(h)
print('ready', flush=True)
input()
winclaim.close_handle(h)
''', source, handles=[source])
            assert child.line() == 'ready'
            holders.append(child.process.pid)
        finally:
            winclaim.close_handle(source)
    else:
        opener = None
        if case == 'U4-orphan-duplicate':
            opener = children('''
h = winclaim.open_handle(sys.argv[1], 1 | 2, 7)
print(h, flush=True)
input()
winclaim.close_handle(h)
''', path)
            source = int(opener.line())
        else:
            source = winclaim.open_handle(path, 1 | 2, 7)
        try:
            child = children("print('ready', flush=True); input()", flags=4)
            with winproc.open(child.process.pid, winproc.INSPECT | winproc.END) as target:
                if opener is None:
                    winproc.push_handle(target, source)
                else:
                    with winproc.open(opener.process.pid, winproc.INSPECT) as owner:
                        duplicate = W.HANDLE()
                        assert winproc._api().DuplicateHandle(owner.handle, source, target.handle,
                            C.byref(duplicate), 0, False, 2), C.get_last_error()
                winproc.resume(target)
            assert child.line() == 'ready'
            holders.append(child.process.pid)
        finally:
            if opener is None:
                winclaim.close_handle(source)
            else:
                opener.send()
                assert opener.process.wait(timeout=5) == 0
    sizes = []
    query = winproc._api().NtQueryInformationFile
    def measured(handle, iosb, data, size, kind):
        if kind == 47:
            sizes.append(size)
        return query(handle, iosb, data, size, kind)
    monkeypatch.setattr(winproc._api(), 'NtQueryInformationFile', measured)
    rows = dict(winproc.using(path))
    print(f'{token} {case}: expected {holders}, listed {rows}, buffers {sizes}')
    assert set(holders) <= rows.keys()
    assert all(Path(rows[pid]).name.lower() == Path(PYTHON).name.lower() for pid in holders)
    assert os.getpid() not in rows
    if case == 'U5-twenty':
        assert len(sizes) > 1
        assert sizes[0] == 64 and sizes[-1] >= 8 + 20 * C.sizeof(C.c_size_t)
    for pid in holders:
        with winproc.open(pid) as entry:
            assert not entry.wait()


@pytest.mark.parametrize('status', [0xc0000022, 0x80000005])
def test_using_probe_failure_names_call(tmp_path, monkeypatch, status):
    path = tmp_path / 'users.bin'
    path.touch()
    monkeypatch.setattr(winproc._api(), 'NtQueryInformationFile', lambda *args: status)
    with pytest.raises(winproc.InventoryUnavailable, match='FileProcessIdsUsingFileInformation'):
        winproc.using(path)


def test_using_keeps_pid_when_image_cannot_be_read(children, tmp_path, monkeypatch):
    path = tmp_path / 'users.bin'
    path.touch()
    child = children('''
h = winclaim.open_handle(sys.argv[1], 1, 7)
print('ready', flush=True)
input()
''', path)
    assert child.line() == 'ready'
    def denied(pid, *args):
        raise winproc.ProcessAccessError(pid, winproc.IDENTIFY)
    monkeypatch.setattr(winproc, 'open', denied)
    assert dict(winproc.using(path))[child.process.pid] == ''


COM_HOLDER = '''
import serial
s = serial.Serial()
s.port, s.rts, s.dtr = 'COM41', False, False
s.open()
print('ready', flush=True)
input()
s.close()
'''


def refused_com41(serial):
    from hfmodem.core import serialline
    with pytest.raises(serial.SerialException) as caught:
        serialline.open_port(serial.Serial, 'COM41', baudrate=9600,
                             rts=False, dtr=False, write_timeout=.25)
    return str(caught.value)


@pytest.mark.parametrize('ownership', ['launched', 'publisher', 'unclaimed'])
@pytest.mark.parametrize('opener', ['serialline', 'ptt', 'cat'])
def test_refused_com_names_python_holder_without_walk(children, com41_com42, monkeypatch,
                                                    ownership, opener):
    from hfmodem.tests.core.fakeyaesu import SerialFakeYaesu
    peer = SerialFakeYaesu('COM42') if opener == 'cat' else None
    code = '''
from hfmodem.core import serialline
if sys.argv[1] == 'launched':
    admission = claim.received(launch=True)
elif sys.argv[1] == 'publisher':
    descriptor = claim.acquire('station', 'test')
if sys.argv[2] == 'ptt':
    from hfmodem.core.ptt import RtsPtt
    port = RtsPtt('COM41')
    port.open()
    close = port.release
elif sys.argv[2] == 'cat':
    from hfmodem.core.cat import YaesuCat
    port = YaesuCat('COM41')
    port.open()
    close = port.close
else:
    import serial
    port = serialline.open_port(serial.Serial, 'COM41', baudrate=9600,
                               rts=False, dtr=False, write_timeout=.25)
    close = port.close
if sys.argv[1] == 'launched':
    admission.opened()
print('ready', flush=True)
input()
close()
print('closed', flush=True)
input()
if sys.argv[1] == 'launched':
    claim.close(admission.claim)
elif sys.argv[1] == 'publisher':
    claim.release(descriptor)
'''
    descriptor = launch = None
    try:
        if ownership == 'launched':
            descriptor = claim.acquire('station', 'test')
            launch = spawn_claim(descriptor, [PYTHON, '-c', PRELUDE + code, 'launched', opener],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child = Child(launch.process)
            assert child.line() == 'CLAIM ADMITTED'
        else:
            child = children(code, ownership, opener)
        assert child.line() == 'ready'
        with child.entry() as entry:
            assert winproc._read_json(winproc._path('ports', entry)) == dict(
                process=winproc._identity(entry), ports=[r'\\.\COM41'])
        records, _ = winproc.live_launches()
        assert all('devices' not in record for record in records)
        calls = []
        monkeypatch.setattr(winproc, '_census', lambda: calls.append('census') or set())
        monkeypatch.setattr(winproc, 'device_holders', lambda *a, **k: calls.append('inspector'))
        label = {'launched': 'modem', 'publisher': 'station', 'unclaimed': 'process'}[ownership]
        assert refused_com41(com41_com42) == f'COM41 is held by your hfmodem {label} (pid {child.process.pid})'
        assert calls == []
        child.send()
        assert child.line() == 'closed'
        records, _ = winproc.live_launches()
        evidence = list(winproc.recorded_devices(records, None))
        assert not any(devices for identity, _, devices in evidence if identity[0] == child.process.pid)
        if ownership != 'launched':
            assert records == []
        child.send()
        assert child.process.wait(timeout=10) == 0, child.process.stderr.read()
    finally:
        if launch is not None:
            if launch.process.poll() is None:
                launch.process.terminate()
                launch.process.wait(timeout=5)
            launch.child.close()
            for stream in (launch.process.stdin, launch.process.stdout, launch.process.stderr):
                stream.close()
        if descriptor is not None:
            claim.release(descriptor)
        if peer is not None:
            peer.close()


def test_claim_publisher_tracks_each_open_port(com41_com42):
    from hfmodem.core import serialline
    descriptor = claim.acquire('station', 'test')
    opened = []
    try:
        before = claim.holder(descriptor)
        for port in ('COM41', 'COM42'):
            opened.append(serialline.open_port(com41_com42.Serial, port, baudrate=9600,
                rts=False, dtr=False, write_timeout=.25))
        published = claim.holder(descriptor)
        assert published == before
        with winproc._own() as own:
            path = winproc._path('ports', own)
            assert path.name == f'{own.pid}-{own.creation}.json'
            assert winproc._read_json(path) == dict(process=winproc._identity(own),
                ports=[r'\\.\COM41', r'\\.\COM42'])
            reader = winclaim.open_handle(path, 1, 7)
            try:
                opened[0].close()
                assert winproc._read_json(path)['ports'] == [r'\\.\COM42']
                data, count = C.create_string_buffer(16384), W.DWORD()
                assert winproc._api().ReadFile(reader, data, len(data), C.byref(count), None)
                assert json.loads(data.raw[:count.value])['ports'] == [r'\\.\COM41', r'\\.\COM42']
            finally:
                winclaim.close_handle(reader)
            opened[1].close()
            assert winproc._read_json(path)['ports'] == []
        assert claim.holder(descriptor) == before
        assert claim.inherited(descriptor) == descriptor
    finally:
        for port in opened:
            port.close()
        claim.release(descriptor)


def test_refused_com_names_native_record_without_walk(children, com41, monkeypatch):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    with child.entry() as entry:
        record = winproc.create_launch(entry, role='native', device=r'\\.\com41')
        record['child']['image'] = r'C:\hamlib\rigctld.exe'
        winproc._write_json(winproc._path('launch', entry), record)
        winproc.advance_launch(entry, 'acknowledged')
        calls = []
        monkeypatch.setattr(winproc, '_census', lambda: calls.append('census') or set())
        monkeypatch.setattr(winproc, 'device_holders', lambda *a, **k: calls.append('inspector'))
        assert refused_com41(com41) == f'COM41 is held by your hfmodem rigctld (pid {entry.pid})'
        assert calls == []


@pytest.mark.wall_clock
def test_refused_com_names_unrecorded_holder_within_bound(children, com41, monkeypatch):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    calls = []
    inspect = winproc.device_holders
    def walk(entries, devices, *, deadline):
        calls.append(deadline)
        return inspect(entries, devices, deadline=deadline)
    monkeypatch.setattr(winproc, 'device_holders', walk)
    start = time.monotonic()
    assert refused_com41(com41) == f'COM41 is open in pid {child.process.pid} ({Path(PYTHON).name})'
    assert len(calls) == 1
    assert start < calls[0] <= start + 1.1
    assert time.monotonic() - start < 1.5


def test_refused_com_cannot_inspect_stays_generic(children, com41, monkeypatch):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    monkeypatch.setattr(winproc, '_census', lambda: {child.process.pid})
    real_open = winproc.open
    def denied(pid, rights=winproc.IDENTIFY):
        if rights & 0x0040:
            raise winproc.ProcessAccessError(pid, rights)
        return real_open(pid, rights)
    monkeypatch.setattr(winproc, 'open', denied)
    assert refused_com41(com41) == 'COM41 is open in another process'


@pytest.mark.wall_clock
def test_refused_com_slow_walk_cannot_extend_bound(children, com41, monkeypatch):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    monkeypatch.setattr(winproc, '_census', lambda: {child.process.pid})
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def slow(entries, devices, *, deadline):
        entered.set()
        try:
            release.wait(5)
            return winproc.DeviceHolders({}, 'deadline')
        finally:
            finished.set()
    monkeypatch.setattr(winproc, 'device_holders', slow)
    start = time.monotonic()
    try:
        assert refused_com41(com41) == 'COM41 is open in another process'
        assert entered.is_set()
        assert .9 <= time.monotonic() - start < 1.5
    finally:
        release.set()
        assert finished.wait(5)


@pytest.mark.parametrize('condition', ['identity', 'device', 'phase', 'unverified'])
def test_refused_com_rejects_unproven_record(children, com41, monkeypatch, condition):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    with child.entry() as entry:
        record = dict(child=winproc._identity(entry), role='python', device='COM41',
                      phase='acknowledged')
        unresolved = set()
        if condition == 'identity':
            record['child']['started'] = 'win:1'
        elif condition == 'device':
            record['device'] = 'COM42'
        elif condition == 'phase':
            record['phase'] = 'created'
        else:
            unresolved.add(entry.pid)
        monkeypatch.setattr(winproc, 'live_launches', lambda: ([record], unresolved))
        monkeypatch.setattr(winproc, '_census', lambda: {entry.pid})
        calls = []
        def walk(entries, devices, *, deadline):
            calls.append([process.pid for process in entries])
            return winproc.DeviceHolders({entry.pid: winproc.DeviceWalk(frozenset({'COM41'}))}, 'incomplete')
        monkeypatch.setattr(winproc, 'device_holders', walk)
        assert refused_com41(com41) == f'COM41 is open in pid {entry.pid} ({Path(PYTHON).name})'
        assert calls == [[entry.pid]]


@pytest.mark.parametrize('status,found', [('busy', True), ('unavailable', True),
                                        ('incomplete', False), ('deadline', False)])
def test_refused_com_missing_evidence_stays_generic(children, com41, monkeypatch, status, found):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    monkeypatch.setattr(winproc, '_census', lambda: {child.process.pid})
    def walk(entries, devices, *, deadline):
        evidence = frozenset({'COM41'}) if found else frozenset()
        return winproc.DeviceHolders({child.process.pid: winproc.DeviceWalk(evidence)}, status)
    monkeypatch.setattr(winproc, 'device_holders', walk)
    assert refused_com41(com41) == 'COM41 is open in another process'


def test_refused_com_other_user_is_not_inspected(children, com41, monkeypatch):
    child = children(COM_HOLDER)
    assert child.line() == 'ready'
    monkeypatch.setattr(winproc, '_census', lambda: {child.process.pid})
    checked = []
    def other_user(entry, sid):
        checked.append(entry.pid)
        return False
    monkeypatch.setattr(winproc, '_same_user', other_user)
    inspected, finished = [], threading.Event()
    monkeypatch.setattr(winproc, 'device_holders', lambda *a, **k: inspected.append(True))
    lookup = winproc._port_holder
    def tracked(*args):
        try:
            return lookup(*args)
        finally:
            finished.set()
    monkeypatch.setattr(winproc, '_port_holder', tracked)
    assert refused_com41(com41) == 'COM41 is open in another process'
    assert finished.wait(5)
    assert checked == [child.process.pid]
    assert inspected == []


@pytest.mark.parametrize('death_operation', ['open', 'close'])
@pytest.mark.parametrize('death_boundary', ['write', 'replace', 'published'])
def test_port_publication_never_refuses_surviving_child_claim(children, com41, tmp_path,
                                                            death_operation, death_boundary):
    report, stop = tmp_path / 'validation.json', tmp_path / 'stop'
    validator = PRELUDE + '''
from pathlib import Path
admission = claim.received(launch=True)
admission.opened()
report, stop = map(Path, sys.argv[1:])
count = 0
try:
    while not stop.exists():
        for _ in range(10):
            claim.inherited(admission.claim)
            count += 1
        winproc._write_json(report, dict(count=count))
except BaseException as exc:
    winproc._write_json(report, dict(count=count, error=str(exc)))
    raise
finally:
    claim.close(admission.claim)
'''
    publisher = children('''
import subprocess
from hfmodem.core import serialline
import serial
descriptor = claim.acquire('station', 'test')
admission = claim.inherited(descriptor, launch=True)
try:
    launched = claim.spawn(admission, [sys.executable, '-c', sys.argv[1], *sys.argv[2:]],
        role='python', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
finally:
    admission.failed()
print(json.dumps(winproc._identity(launched.child)), flush=True)
assert launched.process.stdout.readline().strip() == 'CLAIM ADMITTED'
operation = ''
def boundary(name):
    print(operation + ':' + name, flush=True)
    input()
write, replace = winproc._write_json, winproc._replace
def publishing(path, record):
    if path.parent.name == 'rig.claim.ports.d':
        boundary('write')
    return write(path, record)
def replacing(source, target):
    if target.parent.name == 'rig.claim.ports.d':
        boundary('replace')
        replace(source, target)
        boundary('published')
    else:
        replace(source, target)
winproc._write_json, winproc._replace = publishing, replacing
while True:
    operation = 'open'
    port = serialline.open_port(serial.Serial, 'COM41', baudrate=9600,
        rts=False, dtr=False, write_timeout=.25)
    operation = 'close'
    port.close()
''', validator, report, stop)
    identity = json.loads(publisher.line())
    survivor = winproc.open(identity['pid'], winproc.END)
    assert survivor is not None
    with survivor:
        try:
            assert survivor.started == identity['started']
            count = 0
            def progressing():
                nonlocal count
                assert not survivor.wait(), 'claim validator exited'
                if not report.exists():
                    return False
                data = winproc._read_json(report)
                assert 'error' not in data, data
                if data['count'] <= count:
                    return False
                count = data['count']
                return True
            sequence = [f'{op}:{boundary}' for op in ('open', 'close')
                        for boundary in ('write', 'replace', 'published')]
            for boundary in sequence:
                assert publisher.line() == boundary
                await_condition(progressing)
                publisher.send()
            for boundary in sequence:
                assert publisher.line() == boundary
                await_condition(progressing)
                if boundary == f'{death_operation}:{death_boundary}':
                    publisher.process.kill()
                    publisher.process.wait(timeout=5)
                    break
                publisher.send()
            count = winproc._read_json(report)['count']
            await_condition(progressing)
            stop.write_text('stop')
            assert survivor.wait(5)
            assert 'error' not in winproc._read_json(report)
            records, _ = winproc.live_launches()
            assert not any(ports for identity, _, ports in
                           winproc.recorded_devices(records, claim.holder(None))
                           if identity[0] == publisher.process.pid)
            assert not list((claim.state_dir() / 'rig.claim.ports.d').glob('*.json'))
        finally:
            stop.touch()
            if not survivor.wait(5):
                survivor.terminate()
                assert survivor.wait(5)


def test_old_com_close_preserves_replacement_holder(com41, monkeypatch):
    from hfmodem.core import serialline
    released, resume = threading.Event(), threading.Event()
    errors, calls = [], []

    class Serial(com41.Serial):
        pause = False

        def close(self):
            super().close()
            if self.pause:
                self.pause = False
                released.set()
                assert resume.wait(5)

    old = serialline.open_port(Serial, 'COM41', baudrate=9600,
                              rts=False, dtr=False, write_timeout=.25)
    old.pause = True
    def close():
        try:
            old.close()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=close)
    replacement = None
    thread.start()
    try:
        assert released.wait(5)
        replacement = serialline.open_port(com41.Serial, 'COM41', baudrate=9600,
            rts=False, dtr=False, write_timeout=.25)
        resume.set()
        thread.join(5)
        assert not thread.is_alive() and errors == []
        monkeypatch.setattr(winproc, '_census', lambda: calls.append('census') or set())
        monkeypatch.setattr(winproc, 'device_holders', lambda *a, **k: calls.append('inspector'))
        assert refused_com41(com41) == f'COM41 is held by your hfmodem process (pid {os.getpid()})'
        assert calls == []
    finally:
        resume.set()
        thread.join(5)
        if replacement is not None:
            replacement.close()
        old.close()
    assert not any(ports for identity, _, ports in winproc.recorded_devices([], None)
                   if identity[0] == os.getpid())
