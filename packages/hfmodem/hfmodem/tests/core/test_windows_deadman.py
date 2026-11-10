# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The Windows deadline and cancellation seam without terminating pytest."""
import ctypes
import os
import signal
import threading
import time
import warnings
from types import SimpleNamespace

import pytest

from hfmodem.core import posix, rig, windeadman


class _Retired(BaseException):
    pass


class _RetiredWord:
    @property
    def value(self):
        raise _Retired


def retire(watchdog):
    # A started watchdog polls forever; left running, it spins under any later
    # test in this process that patches time.sleep away.
    if not watchdog.thread.is_alive():
        return
    hook = threading.excepthook
    threading.excepthook = lambda args: None if args.exc_type is _Retired else hook(args)
    try:
        watchdog._word = _RetiredWord()
        watchdog.thread.join(1)
    finally:
        threading.excepthook = hook
    assert not watchdog.thread.is_alive()


@pytest.fixture(autouse=True)
def retire_watchdogs(monkeypatch):
    started = []
    init = windeadman.Watchdog.__init__
    def tracked(self):
        started.append(self)
        init(self)
    monkeypatch.setattr(windeadman.Watchdog, '__init__', tracked)
    yield
    for watchdog in started:
        if hasattr(watchdog, 'thread'):
            retire(watchdog)


@pytest.fixture(scope='module')
def worker():
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(windeadman, '_native_backstop', lambda: None)
        watchdog = windeadman.Watchdog()
    yield watchdog
    retire(watchdog)


@pytest.fixture
def windows(worker, monkeypatch):
    worker.cancel()
    monkeypatch.setattr(windeadman, '_handles', ())
    platform = SimpleNamespace(**{**vars(os), 'name': 'nt'})
    monkeypatch.setattr(posix, 'os', platform)
    monkeypatch.setattr(rig, 'os', platform)
    monkeypatch.setattr(posix, '_windows_deadman', worker)
    exited, calls = threading.Event(), []
    def exit(code):
        calls.append((code, time.monotonic()))
        exited.set()
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(_exit=exit))
    yield exited, calls
    worker.cancel()


def test_fires_at_deadline_and_writes_message(windows, monkeypatch):
    exited, calls = windows
    messages = []
    monkeypatch.setattr(rig.os, 'get_blocking', lambda fd: True)
    monkeypatch.setattr(rig.os, 'set_blocking', lambda fd, value: None)
    monkeypatch.setattr(rig.os, 'write', lambda fd, msg: messages.append((msg, time.monotonic())))
    start = time.monotonic()
    rig.arm_deadman(.05, message=b'deadman exhausted\n')
    assert exited.wait(1)
    assert len(calls) == 1 and calls[0][0] == 1
    assert .05 <= calls[0][1] - start < 1
    assert messages[0][0] == b'deadman exhausted\n'
    assert messages[0][1] <= calls[0][1]


def test_cancelled_deadman_does_not_fire(windows):
    exited, calls = windows
    rig.arm_deadman(.03)
    assert 0 < posix.itimer_left() <= .03
    posix.cancel_itimer()
    assert posix.itimer_left() == 0
    assert not exited.wait(.08)
    assert calls == []


def test_arm_starts_no_thread_and_takes_no_thread_lock(windows, monkeypatch):
    exited, _ = windows
    count = threading.active_count()
    def forbidden(*args, **kwargs):
        pytest.fail('arming used a threading primitive')
    with monkeypatch.context() as patcher:
        patcher.setattr(threading, 'Thread', forbidden)
        patcher.setattr(threading, 'Lock', forbidden)
        patcher.setattr(threading, 'Event', forbidden)
        rig.arm_deadman(.03)
        assert threading.active_count() == count
    assert exited.wait(1)


def test_replacing_deadline_and_rescheduling_preserves_message(windows, monkeypatch):
    exited, calls = windows
    messages = []
    monkeypatch.setattr(rig, 'fatal_write', messages.append)
    rig.arm_deadman(.03, message=b'old')
    rig.arm_deadman(.2, message=b'new')
    assert not exited.wait(.06)
    assert 0 < posix.itimer_left() <= .2
    posix.set_itimer(.03)
    assert exited.wait(1)
    assert len(calls) == 1 and messages == [b'new']


def test_zero_reschedule_cancels(windows):
    exited, _ = windows
    rig.arm_deadman(.03)
    posix.set_itimer(0)
    assert not exited.wait(.08)
    assert posix.itimer_left() == 0


def test_reschedule_before_first_arm_still_terminates(windows):
    exited, calls = windows
    posix.set_itimer(.03)
    assert exited.wait(1)
    assert calls[0][0] == 1


def test_register_unregister_and_lock_free_drop(monkeypatch):
    monkeypatch.setattr(windeadman, '_handles', ())
    calls = []
    def clear(handle, operation):
        calls.append((handle, operation))
        if (handle, operation) == (123, 4):
            raise OSError('closed handle')
        return False
    monkeypatch.setattr(windeadman, '_escape', clear)
    first = windeadman.register_handle(123)
    other = windeadman.register_handle(456)
    reused = windeadman.register_handle(123)
    assert first is not reused
    snapshot = windeadman._handles
    assert snapshot == ((first, 123), (other, 456), (reused, 123))
    windeadman.drop_lines()
    assert calls == [(123, 4), (123, 6), (456, 4), (456, 6), (123, 4), (123, 6)]
    calls.clear()
    windeadman.unregister_handle(first)
    windeadman.unregister_handle(other)
    windeadman.unregister_handle(object())
    assert snapshot == ((first, 123), (other, 456), (reused, 123))
    assert windeadman._handles == ((reused, 123),)
    class ForbiddenLock:
        def __enter__(self):
            pytest.fail('drop took the registry lock')
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(windeadman, '_handles_lock', ForbiddenLock())
    windeadman.drop_lines()
    assert calls == [(123, 4), (123, 6)]


@pytest.mark.parametrize('platform', ['darwin', 'win32'])
def test_handle_channels_use_platform_with_patched_os(monkeypatch, platform):
    from hfmodem.core import winproc

    calls = []
    monkeypatch.setattr(windeadman, '_handles', ())
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace())
    monkeypatch.setattr(windeadman, 'sys', SimpleNamespace(platform=platform))
    monkeypatch.setattr(winproc, '_channels', object())
    monkeypatch.setattr(winproc, 'install_channels', lambda: calls.append(windeadman._handles))
    token = windeadman.register_handle(123)
    windeadman.unregister_handle(token)
    assert windeadman._handles == ()
    assert calls == ([((token, 123),), ()] if platform == 'win32' else [])


def test_firing_drops_every_registered_handle_before_exit(windows, monkeypatch):
    exited, calls = windows
    clears = []
    monkeypatch.setattr(windeadman, '_escape',
                        lambda *args: clears.append((*args, time.monotonic())) or False)
    tokens = [windeadman.register_handle(handle) for handle in (123, 456, 789)]
    windeadman.unregister_handle(tokens[1])
    rig.arm_deadman(.03)
    assert exited.wait(1)
    assert [(h, op) for h, op, _ in clears] == [(123, 4), (123, 6), (789, 4), (789, 6)]
    assert all(at <= calls[0][1] for _, _, at in clears)


def test_blocked_driver_cannot_hold_deadman_exit(windows, worker, monkeypatch):
    exited, calls = windows
    entered, unblock = threading.Event(), threading.Event()
    clearer_threads = []
    def clear(handle, operation):
        clearer_threads.append(threading.current_thread())
        entered.set()
        unblock.wait()
    monkeypatch.setattr(windeadman, '_escape', clear)
    windeadman.register_handle(123)
    start = time.monotonic()
    rig.arm_deadman(.03)
    try:
        assert entered.wait(1)
        assert exited.wait(.5)
        assert calls[0][0] == 1
        assert .1 <= calls[0][1] - start < .5
        assert clearer_threads == [worker._clearer]
        assert worker._clearer.daemon
        assert not worker._clear_done.is_set()
    finally:
        unblock.set()
        assert worker._clear_done.wait(1)
        assert exited.wait(1)


@pytest.mark.parametrize('handler', [lambda signum, frame: None, signal.SIG_DFL])
def test_station_worker_drops_and_publishes_windows_fatal_request(windows, monkeypatch, handler):
    from hfmodem.station import process

    calls = []
    class Handoff(BaseException):
        pass
    def sleep(seconds):
        assert seconds == .1
        calls.append(('sleep',))
        raise Handoff
    def forbidden(*args, **kwargs):
        pytest.fail('Windows worker used os.kill or a threading primitive')
    def arm(seconds, *, message):
        calls.append(('arm', seconds, message))
        rig.arm_deadman(seconds, message=message)
    def getsignal(signum):
        calls.append(('getsignal', signum))
        return handler
    monkeypatch.setattr(process, 'os', SimpleNamespace(name='nt', kill=forbidden))
    monkeypatch.setattr(process, 'time', SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(process, 'signal', SimpleNamespace(
        SIGTERM=signal.SIGTERM, getsignal=getsignal,
        raise_signal=lambda signum: calls.append(('raise', signum))))
    monkeypatch.setattr(process, 'arm_deadman', arm)
    monkeypatch.setattr(process, 'threading', SimpleNamespace(
        current_thread=threading.current_thread, main_thread=threading.main_thread,
        Thread=forbidden, Lock=forbidden, Event=forbidden))
    monkeypatch.setattr(windeadman, '_escape', lambda *args: calls.append(('clear', *args)))
    windeadman.register_handle(123)
    class WorkerStation(process.Station):
        @property
        def _signals(self):
            pytest.fail('worker touched _signals')
        @_signals.setter
        def _signals(self, value):
            pytest.fail('worker touched _signals')
    station = object.__new__(WorkerStation)
    def fail():
        try:
            station.fatal_shutdown('worker failure')
        except Handoff:
            pass
    thread = threading.Thread(target=fail)
    thread.start()
    thread.join(1)
    assert not thread.is_alive()
    message = b'fatal_shutdown confirmed=false reason=deadman budget exhausted\n'
    assert calls == [('arm', 5.0, message), ('getsignal', signal.SIGTERM),
                     *([('raise', signal.SIGTERM)] if callable(handler) else []),
                     ('clear', 123, 4), ('clear', 123, 6), ('sleep',)]
    assert 4 < posix.itimer_left() <= 5
    assert station._fatal_reason == 'worker failure'
    assert posix._windows_deadman._record[0] == message


def test_station_worker_publishes_before_blocked_drop(windows, monkeypatch):
    from hfmodem.station import process

    entered, unblock, published = threading.Event(), threading.Event(), threading.Event()
    calls = []
    class Handoff(BaseException):
        pass
    def arm(seconds, *, message):
        calls.append(('arm', seconds, message))
        rig.arm_deadman(seconds, message=message)
    def publish(signum):
        calls.append(('raise', signum))
        published.set()
    def drop():
        calls.append(('drop',))
        entered.set()
        unblock.wait()
    def sleep(seconds):
        raise Handoff
    def forbidden(*args):
        pytest.fail('Windows worker used os.kill')
    monkeypatch.setattr(process, 'os', SimpleNamespace(name='nt', kill=forbidden))
    monkeypatch.setattr(process, 'arm_deadman', arm)
    monkeypatch.setattr(process, 'signal', SimpleNamespace(
        SIGTERM=signal.SIGTERM, getsignal=lambda signum: lambda *args: None,
        raise_signal=publish))
    monkeypatch.setattr(process, 'time', SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(windeadman, 'drop_lines', drop)
    station = object.__new__(process.Station)
    station._signals = 0
    def fail():
        try:
            station.fatal_shutdown('worker failure')
        except Handoff:
            pass
    thread = threading.Thread(target=fail, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        assert thread.is_alive() and published.is_set()
        assert calls == [
            ('arm', 5.0, b'fatal_shutdown confirmed=false reason=deadman budget exhausted\n'),
            ('raise', signal.SIGTERM), ('drop',)]
        assert 4 < posix.itimer_left() <= 5
        assert station._signals == 0
    finally:
        unblock.set()
        thread.join(1)
        assert not thread.is_alive()


@pytest.mark.parametrize('after', [None, .2])
def test_ota_hold_reschedules_or_cancels_windows_deadline(windows, monkeypatch, after):
    from hfmodem.shrike import ota

    monkeypatch.setattr(ota, '_pending', None)
    monkeypatch.setattr(ota, '_depth', 0)
    rig.arm_deadman(.5)
    with ota.signals_held(after_s=after):
        assert posix.itimer_left() > 0
    if after is None:
        assert posix.itimer_left() == 0
    else:
        assert 0 < posix.itimer_left() <= after


def test_an_expired_deadline_still_reads_as_armed():
    """The worker polls, so a deadline can be past and not yet acted on. Read as
    zero there, it looked unarmed, and the unwind armed a fresh budget over it."""
    watchdog = SimpleNamespace(_word=ctypes.c_uint64(1), _clock=lambda: 10_000_001)
    assert windeadman.Watchdog.left(watchdog) > 0


class FakeKernel32:
    address = 0x123456789ABCDEF0

    def __init__(self, failure=None):
        self.calls = []
        self.pages = []
        self.failure = failure
        self.TerminateProcess = ctypes.c_void_p(self.address)
        for name in ('VirtualAlloc', 'VirtualProtect', 'FlushInstructionCache',
                     'CreateThreadpool', 'SetThreadpoolThreadMaximum',
                     'SetThreadpoolThreadMinimum', 'CreateThreadpoolTimer', 'SetThreadpoolTimer'):
            setattr(self, name, self.function(name))

    def function(self, name):
        def call(*args):
            self.calls.append((name, *args))
            if self.failure == name:
                return 0
            if name == 'VirtualAlloc':
                self.pages.append(ctypes.create_string_buffer(4096))
                return ctypes.addressof(self.pages[-1])
            if name == 'CreateThreadpool':
                return 456
            if name == 'CreateThreadpoolTimer':
                return 123
            if name == 'SetThreadpoolTimer' and args[1] is not None:
                due = ctypes.cast(args[1], ctypes.POINTER(windeadman._FILETIME)).contents
                ticks = ctypes.c_int64(due.dwLowDateTime | due.dwHighDateTime << 32).value
                self.calls[-1] = (name, args[0], ticks, *args[2:])
            return 1
        return call

    @property
    def page(self):
        return ctypes.addressof(self.pages[1])

    def write(self, page, data, size):
        self.calls.append(('write', page, data, size))


def test_native_thunk_protection_and_timer_creation():
    kernel = FakeKernel32()
    backstop = windeadman._NativeBackstop(kernel, memmove=kernel.write)
    calls = kernel.calls
    thunk = (bytes.fromhex('48 ba 08 00 fe 7f 00 00 00 00 48 8b 12 48 b8')
             + ctypes.addressof(backstop.word).to_bytes(8, 'little')
             + bytes.fromhex('48 8b 00 48 85 c0 74 23 48 05 a0 25 26 00 '
                             '48 39 c2 72 18 '
                             '48 c7 c1 ff ff ff ff ba 01 00 00 00 48 b8 '
                             'f0 de bc 9a 78 56 34 12 ff e0 c3'))
    assert len(thunk) == 67
    assert thunk == windeadman._thunk(ctypes.addressof(backstop.word), kernel.address)
    assert 29 + 2 + thunk[30] == 66
    assert 40 + 2 + thunk[41] == 66
    assert thunk[66:] == b'\xc3'
    assert backstop.word.value == 0
    assert ctypes.addressof(backstop.word) % 8 == 0
    assert [call[0] for call in calls] == [
        'VirtualAlloc', 'VirtualAlloc', 'write', 'VirtualProtect', 'FlushInstructionCache',
        'CreateThreadpool', 'SetThreadpoolThreadMaximum',
        'SetThreadpoolThreadMinimum', 'CreateThreadpoolTimer', 'SetThreadpoolTimer']
    assert calls[0] == calls[1] == ('VirtualAlloc', None, 4096, 0x3000, 0x04)
    assert calls[-1] == ('SetThreadpoolTimer', 123, -200000, 20, 0)
    calls = calls[1:-1]
    assert calls[1] == ('write', kernel.page, thunk, len(thunk))
    assert calls[2][1:4] == (kernel.page, 4096, 0x20)
    assert ctypes.cast(calls[2][4], ctypes.POINTER(ctypes.c_uint32))[0] == 0
    assert calls[3] == ('FlushInstructionCache', -1, kernel.page, len(thunk))
    assert calls[4:7] == [('CreateThreadpool', None),
                          ('SetThreadpoolThreadMaximum', 456, 1),
                          ('SetThreadpoolThreadMinimum', 456, 1)]
    assert calls[7][1:3] == (kernel.page, None)
    env = ctypes.cast(calls[7][3], ctypes.POINTER(windeadman._TP_CALLBACK_ENVIRON_V3)).contents
    assert ctypes.sizeof(env) == 72 and ctypes.alignment(env) == 8
    assert [(name, getattr(type(env), name).offset) for name, _ in env._fields_] == [
        ('Version', 0), ('Pool', 8), ('CleanupGroup', 16),
        ('CleanupGroupCancelCallback', 24), ('RaceDll', 32), ('ActivationContext', 40),
        ('FinalizationCallback', 48), ('u', 56), ('CallbackPriority', 60), ('Size', 64)]
    assert (env.Version, env.Pool, env.u.Flags, env.CallbackPriority, env.Size) == (3, 456, 0, 1, 72)
    assert all(getattr(env, name) is None for name in (
        'CleanupGroup', 'CleanupGroupCancelCallback', 'RaceDll',
        'ActivationContext', 'FinalizationCallback'))
    assert backstop._timer == 123
    pointer, dword = ctypes.c_void_p, ctypes.c_uint32
    signatures = {
        'VirtualAlloc': ([pointer, ctypes.c_size_t, dword, dword], pointer),
        'VirtualProtect': ([pointer, ctypes.c_size_t, dword, ctypes.POINTER(dword)], ctypes.c_int),
        'FlushInstructionCache': ([pointer, pointer, ctypes.c_size_t], ctypes.c_int),
        'CreateThreadpool': ([pointer], pointer),
        'SetThreadpoolThreadMaximum': ([pointer, dword], None),
        'SetThreadpoolThreadMinimum': ([pointer, dword], ctypes.c_int),
        'CreateThreadpoolTimer': ([pointer, pointer,
                                  ctypes.POINTER(windeadman._TP_CALLBACK_ENVIRON_V3)], pointer),
        'SetThreadpoolTimer': ([pointer, ctypes.POINTER(windeadman._FILETIME), dword, dword], None),
    }
    for name, (args, result) in signatures.items():
        assert getattr(kernel, name).argtypes == args
        assert getattr(kernel, name).restype is result


def unstarted_worker(kernel=None, clock=lambda: 100_000_000):
    worker = object.__new__(windeadman.Watchdog)
    worker._backstop = (windeadman._NativeBackstop(kernel, memmove=kernel.write)
                        if kernel else None)
    worker._word = worker._backstop.word if kernel else ctypes.c_uint64()
    worker._clock = clock
    worker._record = (None, None)
    worker._fired = None
    return worker


@pytest.mark.parametrize('stop,expected', [
    (0, [100_501_000, 101_000_000, 100_251_000, 0, 0, 0, 0]),
    (.04, [100_400_000, 100_400_000, 100_251_000,
           100_400_000, 100_400_000, 100_400_000, 100_400_000]),
    (.075, [100_501_000, 100_750_000, 100_251_000,
            100_750_000, 100_750_000, 100_750_000, 100_750_000]),
    (.2, [100_501_000, 101_000_000, 100_251_000,
          102_000_000, 102_000_000, 102_000_000, 102_000_000]),
])
def test_native_arm_rearm_cancel_and_reschedule_are_single_stores(monkeypatch, stop, expected):
    kernel = FakeKernel32()
    worker = unstarted_worker(kernel)
    word, stores = worker._word, []
    class Word:
        @property
        def value(self):
            return word.value
        @value.setter
        def value(self, value):
            stores.append(value)
            word.value = value
    worker._word = Word()
    kernel.calls.clear()
    def forbidden(*args, **kwargs):
        pytest.fail('arming used a threading primitive')
    for name in ('Thread', 'Lock', 'Event'):
        monkeypatch.setattr(threading, name, forbidden)
    if stop:
        worker.stop(stop)
        assert stores == [100_000_000 + round(stop * windeadman._TICKS_PER_SECOND)]
        stores.clear()
    worker.arm(.0501, 'message', print)
    assert worker.left() == pytest.approx(min(.0501, stop) if stop else .0501)
    worker.arm(.1, 'replacement', print)
    worker.reschedule(.0251)
    assert worker._record == ('replacement', print)
    worker.cancel()
    assert worker.left() == stop
    worker.arm(0)
    worker.arm(None)
    worker.reschedule(0)
    assert stores == expected
    assert word.value == expected[-1] and kernel.calls == []
    assert worker.native
    with pytest.raises(AttributeError):
        worker.native = False


def test_deadline_defaults_are_independent_between_workers():
    stopped = unstarted_worker()
    timer = unstarted_worker()
    stopped.stop(.2)
    assert stopped._word.value == 102_000_000
    assert stopped._timer_deadline == 0
    timer.arm(.3)
    assert timer._word.value == 103_000_000
    assert timer._stop_deadline == 0
    timer.cancel()
    assert timer._word.value == 0
    assert stopped._word.value == 102_000_000


@pytest.mark.parametrize('seconds,ticks', [(.00000001, 1), (.00000011, 2), (.3, 3000000)])
def test_deadline_rounds_up(seconds, ticks):
    worker = unstarted_worker(clock=lambda: 1)
    worker.arm(seconds)
    assert worker._word.value == 1 + ticks


@pytest.mark.parametrize('inner', [None, .2])
def test_arm_from_reentrant_clock_publishes_one_state(inner):
    kernel = FakeKernel32()
    worker = unstarted_worker(kernel)
    entered = False
    def clock():
        nonlocal entered
        if not entered:
            entered = True
            worker.arm(inner, 'inner', print)
        return 100_000_000
    worker._clock = clock
    kernel.calls.clear()
    worker.arm(.1, 'outer', print)
    assert worker._word.value == worker._backstop.word.value == 101_000_000
    assert worker._record == ('outer', print)
    assert kernel.calls == []


def test_worker_reads_word_fires_once_and_can_rearm_in_callback(monkeypatch):
    worker = unstarted_worker(clock=lambda: 100_000_000)
    worker._word.value = 99_000_000
    calls = []
    def write(message):
        calls.append(message)
        worker.arm(.1, 'new', calls.append)
    worker._record = ('old', write)
    worker._clear_done = SimpleNamespace(clear=lambda: None, wait=lambda timeout: None)
    worker._clear_requested = SimpleNamespace(set=lambda: None)
    class Stop(BaseException):
        pass
    sleeps = 0
    def sleep(seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            worker._clock = lambda: 102_000_000
        if sleeps == 5:
            raise Stop
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(_exit=lambda code: calls.append(code)))
    monkeypatch.setattr(windeadman, 'time', SimpleNamespace(sleep=sleep))
    with pytest.raises(Stop):
        worker._run()
    assert calls == ['old', 1, 'new', 1]
    assert worker._word.value == 101_000_000
    assert worker._record == ('new', calls.append)


def test_clock_reads_precise_interrupt_time_or_monotonic_ticks(monkeypatch):
    def precise(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64))[0] = 123456
    monkeypatch.setattr(windeadman, '_precise', precise)
    assert windeadman._clock() == 123456
    monkeypatch.setattr(windeadman, '_precise', None)
    monkeypatch.setattr(windeadman, 'time', SimpleNamespace(monotonic_ns=lambda: 12345678))
    assert windeadman._clock() == 123456


@pytest.mark.parametrize('failure', ['VirtualAlloc', 'VirtualProtect',
                                   'FlushInstructionCache', 'CreateThreadpool',
                                   'SetThreadpoolThreadMinimum', 'CreateThreadpoolTimer'])
def test_native_construction_failure_warns_once_and_python_still_fires(monkeypatch, failure):
    native = windeadman._NativeBackstop
    kernel = FakeKernel32(failure)
    exited, codes = threading.Event(), []
    def exit(code):
        codes.append(code)
        exited.set()
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(name='nt', _exit=exit))
    monkeypatch.setattr(windeadman.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **k: kernel, raising=False)
    monkeypatch.setattr(windeadman, '_NativeBackstop',
                        lambda k: native(k, memmove=kernel.write))
    with pytest.warns(RuntimeWarning, match=f'{failure} failed') as warnings:
        worker = windeadman.Watchdog()
    assert len(warnings) == 1 and not worker.native
    try:
        worker.arm(.02)
        assert exited.wait(1) and codes == [1]
    finally:
        worker.cancel()


def test_non_amd64_warns_without_loading_kernel32(monkeypatch):
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(windeadman.platform, 'machine', lambda: 'ARM64')
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported architecture touched native APIs')
    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    with pytest.warns(RuntimeWarning, match='requires Windows AMD64') as warnings:
        worker = windeadman.Watchdog()
    assert len(warnings) == 1 and not worker.native


def test_off_windows_never_loads_native_apis(monkeypatch):
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(name='posix'))
    def forbidden(*args, **kwargs):
        pytest.fail('POSIX touched native APIs')
    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    assert windeadman.Watchdog()._backstop is None


def test_warning_as_error_does_not_prevent_python_fallback(monkeypatch):
    exited = threading.Event()
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(name='nt', _exit=lambda code: exited.set()))
    monkeypatch.setattr(windeadman.platform, 'machine', lambda: 'ARM64')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        worker = windeadman.Watchdog()
    assert not worker.native
    try:
        worker.arm(.02)
        assert exited.wait(1)
    finally:
        worker.cancel()


def test_amd64_constructs_native_timer_before_workers_start(monkeypatch):
    native = windeadman._NativeBackstop
    kernel = FakeKernel32()
    monkeypatch.setattr(windeadman, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(windeadman.platform, 'machine', lambda: 'AMD64')
    loads = []
    def load(name, **kwargs):
        loads.append((name, kwargs))
        return kernel
    monkeypatch.setattr(ctypes, 'WinDLL', load, raising=False)
    monkeypatch.setattr(windeadman, '_NativeBackstop',
                        lambda k: native(k, memmove=kernel.write))
    def start(thread):
        assert kernel.calls[-1] == ('SetThreadpoolTimer', 123, -200000, 20, 0)
    monkeypatch.setattr(threading.Thread, 'start', start)
    worker = windeadman.Watchdog()
    assert worker.native
    assert loads == [('kernel32', {'use_last_error': True})]


@pytest.mark.parametrize('replacement', [0, 20_000_000])
def test_native_snapshot_cancel_or_extend_between_loads(replacement):
    word_address, terminate = 0x1000, 0x2000
    code = windeadman._thunk(word_address, terminate)
    memory = {windeadman._INTERRUPT_TIME: 12_000_000, word_address: 10_500_000}
    registers = {'rax': 0, 'rdx': 0}
    pc, zero, below, suspended = 0, False, False, False
    while True:
        op = code[pc:pc + 3]
        if code[pc:pc + 2] in (b'\x48\xba', b'\x48\xb8'):
            reg = 'rdx' if code[pc + 1] == 0xba else 'rax'
            registers[reg] = int.from_bytes(code[pc + 2:pc + 10], 'little')
            pc += 10
        elif op in (b'\x48\x8b\x12', b'\x48\x8b\x00'):
            reg = 'rdx' if op[-1] == 0x12 else 'rax'
            registers[reg] = memory[registers[reg]]
            pc += 3
            if not suspended:
                suspended = True
                memory[word_address] = replacement
                memory[windeadman._INTERRUPT_TIME] = 14_000_000
        elif op == b'\x48\x85\xc0':
            zero = registers['rax'] == 0
            pc += 3
        elif code[pc:pc + 2] == b'\x48\x05':
            registers['rax'] += int.from_bytes(code[pc + 2:pc + 6], 'little')
            pc += 6
        elif op == b'\x48\x39\xc2':
            below = registers['rdx'] < registers['rax']
            pc += 3
        elif code[pc] in (0x74, 0x72):
            taken = zero if code[pc] == 0x74 else below
            pc += 2 + (code[pc + 1] if taken else 0)
        else:
            assert suspended and code[pc:] == b'\xc3', 'stale snapshot reached termination'
            break


def test_precise_clock_binding_uses_pydll(monkeypatch):
    precise = SimpleNamespace()
    loads = []
    def load(name):
        loads.append(name)
        return SimpleNamespace(QueryInterruptTimePrecise=precise)
    monkeypatch.setattr(ctypes, 'PyDLL', load)
    assert windeadman._bind_precise() is precise
    assert loads == ['kernelbase']
    assert precise.argtypes == [ctypes.POINTER(ctypes.c_uint64)]
    assert precise.restype is None


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows kernelbase')
def test_precise_clock_binding_keeps_gil():
    assert windeadman._bind_precise()._flags_ & ctypes._FUNCFLAG_PYTHONAPI


@pytest.mark.parametrize('offset', [0, 1, 2_500_000, 2**64])
def test_deadline_overflow_preserves_existing_publication(offset):
    grace = int(windeadman._BACKSTOP_GRACE * windeadman._TICKS_PER_SECOND)
    worker = unstarted_worker(clock=lambda: 2**64 - grace - 10 + offset)
    worker._word.value = 123
    worker._record = ('existing', print)
    with pytest.raises(ValueError, match='uint64'):
        worker.arm(.000001, 'replacement', None)
    assert worker._word.value == 123
    assert worker._record == ('existing', print)


def test_deadline_with_last_available_grace_tick_is_accepted():
    grace = int(windeadman._BACKSTOP_GRACE * windeadman._TICKS_PER_SECOND)
    worker = unstarted_worker(clock=lambda: 2**64 - grace - 11)
    worker.arm(.000001)
    assert worker._word.value == 2**64 - grace - 1
