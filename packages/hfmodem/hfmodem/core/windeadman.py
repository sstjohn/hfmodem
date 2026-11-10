# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A process deadman prepared before Windows can touch a keying line.

Arming publishes one 64-bit interrupt-time deadline shared by both deadmen;
the arm path takes no Python lock, starts no thread and keeps the GIL through
its one native call, the clock read.
Polling keeps Event locks out of the possibly interrupted arming caller. It
wakes every 50 ms, or at the deadline when that is sooner: a budget is seconds,
so noticing an arm late by 50 ms costs nothing, and a modem's decoder does not
pay a GIL handoff every few milliseconds for a timer that is almost never armed.
On Windows AMD64 a private native pool checks that word at a nominal 20 ms
interval, with another 0.25 s of grace, even during finalization or while an
extension holds the GIL. The backstop skips line clears: the lines fall to the
driver's handle-close behaviour. com0com lowers CTS about 1 ms after
TerminateProcess; the FT-891's CP2105 (driver 11.2.0.167) returned TX0 after
12.5–13.9 ms in the keyed run on 2026-10-30.
FTDI adapters are unmeasured.
If the backstop is unavailable, only the Python worker remains.
A timer whose expiry has already committed cannot be revoked: it is ending
the process, even if a caller subsequently cancels or replaces the deadline.

This ends our process; it does not supervise children. No job object is used:
claim-holding children survive their launcher and keep the claim, as on POSIX.
The phase-3 launcher boundary is deferred.
"""
from __future__ import annotations

import os
import ctypes
import math
import platform
import sys
import threading
import time
import warnings


_handles = ()
_handles_lock = threading.Lock()
_CLEAR_TIMEOUT = 0.1
_BACKSTOP_GRACE = 0.25
_PERIOD_MS = 20
_TICKS_PER_SECOND = 10_000_000
_INTERRUPT_TIME = 0x7FFE0008


def _bind_precise():
    precise = ctypes.PyDLL('kernelbase').QueryInterruptTimePrecise
    precise.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    precise.restype = None
    return precise


_escape = _precise = None
if os.name == 'nt':
    _escape = ctypes.WinDLL('kernel32', use_last_error=True).EscapeCommFunction
    _escape.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _escape.restype = ctypes.c_int
    # The same timebase as the shared word the native check reads, but interpolated:
    # that word advances a clock tick at a time and would fire Python up to a tick early.
    _precise = _bind_precise()


def _clock():
    if _precise is None:
        return time.monotonic_ns() // 100
    now = ctypes.c_uint64()
    _precise(ctypes.byref(now))
    return now.value


def _thunk(word, terminate):
    # The clock is read before the word, and x86 does not reorder loads: a cancel or
    # extension that lands while the callback is suspended can only meet an expiry
    # that was already due when the word was read.
    pieces = [
        ('clock', b'\x48\xba' + _INTERRUPT_TIME.to_bytes(8, 'little')),
        ('sample', b'\x48\x8b\x12'),
        ('deadline', b'\x48\xb8' + word.to_bytes(8, 'little')),
        ('load', b'\x48\x8b\x00'),
        ('test', b'\x48\x85\xc0'),
        ('disarmed', b'\x74\x00'),
        ('grace', b'\x48\x05' + int(_BACKSTOP_GRACE * _TICKS_PER_SECOND).to_bytes(4, 'little')),
        # Interrupt time and deadlines are unsigned counts, not signed intervals.
        ('compare', b'\x48\x39\xc2'),
        ('early', b'\x72\x00'),
        ('process', b'\x48\xc7\xc1\xff\xff\xff\xff'),
        ('status', b'\xba\x01\x00\x00\x00'),
        ('terminate', b'\x48\xb8' + terminate.to_bytes(8, 'little')),
        ('jump', b'\xff\xe0'),
        # RSP is untouched, so ret returns to the pool worker's original frame.
        ('done', b'\xc3'),
    ]
    code, offsets = bytearray(), {}
    for name, data in pieces:
        offsets[name] = len(code)
        code.extend(data)
    for name in ('disarmed', 'early'):
        end = offsets[name] + 2
        code[end - 1] = offsets['done'] - end
    return bytes(code)


class _CallbackFlags(ctypes.Union):
    _fields_ = [('Flags', ctypes.c_uint32)]


class _TP_CALLBACK_ENVIRON_V3(ctypes.Structure):
    # winnt.h's inline initializers are not exports; mirror their x64 ABI.
    _fields_ = [
        ('Version', ctypes.c_uint32),
        ('Pool', ctypes.c_void_p),
        ('CleanupGroup', ctypes.c_void_p),
        ('CleanupGroupCancelCallback', ctypes.c_void_p),
        ('RaceDll', ctypes.c_void_p),
        ('ActivationContext', ctypes.c_void_p),
        ('FinalizationCallback', ctypes.c_void_p),
        ('u', _CallbackFlags),
        ('CallbackPriority', ctypes.c_int32),
        ('Size', ctypes.c_uint32),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [('dwLowDateTime', ctypes.c_uint32), ('dwHighDateTime', ctypes.c_uint32)]


class _NativeBackstop:
    def __init__(self, kernel32, *, memmove=ctypes.memmove):
        assert ctypes.sizeof(_TP_CALLBACK_ENVIRON_V3) == 72
        pointer, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
        for name, args, result in (
                ('VirtualAlloc', [pointer, ctypes.c_size_t, dword, dword], pointer),
                ('VirtualProtect', [pointer, ctypes.c_size_t, dword, ctypes.POINTER(dword)], boolean),
                ('FlushInstructionCache', [pointer, pointer, ctypes.c_size_t], boolean),
                ('CreateThreadpool', [pointer], pointer),
                ('SetThreadpoolThreadMaximum', [pointer, dword], None),
                ('SetThreadpoolThreadMinimum', [pointer, dword], boolean),
                ('CreateThreadpoolTimer', [pointer, pointer,
                                          ctypes.POINTER(_TP_CALLBACK_ENVIRON_V3)], pointer),
                ('SetThreadpoolTimer', [pointer, ctypes.POINTER(_FILETIME), dword, dword], None)):
            function = getattr(kernel32, name)
            function.argtypes, function.restype = args, result
        address = ctypes.cast(kernel32.TerminateProcess, pointer).value
        # VirtualAlloc storage survives Python finalization and is never freed.
        word_page = kernel32.VirtualAlloc(None, 4096, 0x3000, 0x04)
        if not word_page:
            raise OSError('VirtualAlloc failed')
        self.word = ctypes.c_uint64.from_address(word_page)
        thunk = _thunk(word_page, address)
        page = kernel32.VirtualAlloc(None, 4096, 0x3000, 0x04)
        if not page:
            raise OSError('VirtualAlloc failed')
        memmove(page, thunk, len(thunk))
        old = dword()
        if not kernel32.VirtualProtect(page, 4096, 0x20, ctypes.byref(old)):
            raise OSError('VirtualProtect failed')
        if not kernel32.FlushInstructionCache(-1, page, len(thunk)):
            raise OSError('FlushInstructionCache failed')
        self._pool = kernel32.CreateThreadpool(None)
        if not self._pool:
            raise OSError('CreateThreadpool failed')
        kernel32.SetThreadpoolThreadMaximum(self._pool, 1)
        if not kernel32.SetThreadpoolThreadMinimum(self._pool, 1):
            raise OSError('SetThreadpoolThreadMinimum failed')
        self._environ = _TP_CALLBACK_ENVIRON_V3(
            Version=3, Pool=self._pool, CallbackPriority=1,
            Size=ctypes.sizeof(_TP_CALLBACK_ENVIRON_V3))
        self._timer = kernel32.CreateThreadpoolTimer(page, None, ctypes.byref(self._environ))
        if not self._timer:
            raise OSError('CreateThreadpoolTimer failed')
        ticks = -_PERIOD_MS * 10_000
        due = _FILETIME(ticks & 0xFFFFFFFF, (ticks >> 32) & 0xFFFFFFFF)
        # Native wakes need no GIL; 20 ms is a nominal interval, not a scheduling bound.
        kernel32.SetThreadpoolTimer(self._timer, ctypes.byref(due), _PERIOD_MS, 0)


def _native_backstop():
    if os.name != 'nt':
        return None
    try:
        if platform.machine().upper() not in ('AMD64', 'X86_64') or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise OSError('native deadman requires Windows AMD64')
        return _NativeBackstop(ctypes.WinDLL('kernel32', use_last_error=True))
    except Exception as exc:
        try:
            warnings.warn(f'Windows native deadman unavailable: {exc}; using Python alone',
                          RuntimeWarning, stacklevel=2)
        except RuntimeWarning:
            # A warnings-as-errors policy must not disable the Python fallback.
            pass
        return None


def register_handle(handle):
    global _handles
    token = object()
    with _handles_lock:
        _handles = (*_handles, (token, handle))
    if sys.platform == 'win32':
        from . import winproc
        try:
            winproc.install_channels()
        except BaseException as exc:
            unregister_handle(token)
            if isinstance(exc, Exception):
                raise OSError(f'cannot install stop channels: {exc}') from exc
            raise
    return token


def unregister_handle(token):
    global _handles
    with _handles_lock:
        _handles = tuple(entry for entry in _handles if entry[0] is not token)
    if sys.platform == 'win32':
        from . import winproc
        if winproc._channels is not None:
            try:
                winproc.install_channels()
            except Exception:
                # Keeping the shorter published grace is safe; release must finish cleanup.
                pass


def drop_lines():
    """Best-effort clears on a lock-free snapshot of the keying handles."""
    # A registered number may have been reused; clearing serial outputs is safe
    # and fails on any other handle.
    for _, handle in _handles:
        for operation in (4, 6):  # CLRRTS, CLRDTR
            try:
                _escape(handle, operation)
            except Exception:
                pass


def _stop_write(message):
    try:
        blocking = os.get_blocking(2)
        try:
            os.set_blocking(2, False)
            os.write(2, message)
        finally:
            os.set_blocking(2, blocking)
    except OSError:
        pass


class Watchdog:
    _stop_deadline = 0
    _timer_deadline = 0

    def __init__(self):
        self._record = (None, None)
        self._fired = None
        self._backstop = _native_backstop()
        self._word = self._backstop.word if self._backstop is not None else ctypes.c_uint64()
        self._clock = _clock
        self._clear_requested = threading.Event()
        self._clear_done = threading.Event()
        self._clearer = threading.Thread(target=self._clear, name="windows-deadman-clearer",
                                        daemon=True)
        self._clearer.start()
        self.thread = threading.Thread(target=self._run, name="windows-deadman", daemon=True)
        self.thread.start()

    def arm(self, seconds, message=None, write=None):
        deadline = self._clock() + math.ceil(seconds * _TICKS_PER_SECOND) if seconds else 0
        if deadline < 0 or deadline + int(_BACKSTOP_GRACE * _TICKS_PER_SECOND) >= 2**64:
            raise ValueError("deadline must fit in uint64 with room for grace")
        # Racing arms can mix messages, which are cosmetic; only the word arms either deadman.
        self._record = (message, write)
        self._timer_deadline = deadline
        self._publish()

    def _publish(self):
        timer, stop = self._timer_deadline, self._stop_deadline
        # No calls or backward edges: CPython retains the GIL through this publication.
        self._word.value = stop if stop and (not timer or stop < timer) else timer

    def stop(self, seconds):
        deadline = self._clock() + math.ceil(seconds * _TICKS_PER_SECOND)
        if not 0 < deadline < 2**64 - int(_BACKSTOP_GRACE * _TICKS_PER_SECOND):
            raise ValueError("deadline must fit in uint64 with room for grace")
        if not self._stop_deadline or deadline < self._stop_deadline:
            self._stop_deadline = deadline
        self._publish()

    @property
    def native(self):
        """Whether construction provided the periodic native backstop."""
        return self._backstop is not None

    def left(self):
        """Remaining seconds; an expired deadline the worker has not yet acted on
        still reads as armed, so no caller mistakes it for none and renews it."""
        deadline = self._word.value
        return max(1e-6, (deadline - self._clock()) / _TICKS_PER_SECOND) if deadline else 0.0

    def reschedule(self, seconds):
        self.arm(seconds, *(self._record if self._word.value else (None, None)))

    def cancel(self):
        self.arm(None)

    def _clear(self):
        while True:
            self._clear_requested.wait()
            self._clear_requested.clear()
            try:
                drop_lines()
            finally:
                self._clear_done.set()

    def _run(self):
        while True:
            deadline = self._word.value
            if deadline and deadline != self._fired and self._clock() >= deadline:
                self._fired = deadline
                if self._word.value == deadline:
                    message, write = self._record
                    if self._stop_deadline and deadline == self._stop_deadline:
                        message = b"hfmodem: stop deadline expired\n"
                        write = _stop_write
                    try:
                        if message is not None:
                            write(message)
                    finally:
                        try:
                            # Only the watchdog and clearer use these locks, outside signal handlers.
                            self._clear_done.clear()
                            self._clear_requested.set()
                            self._clear_done.wait(_CLEAR_TIMEOUT)
                        finally:
                            os._exit(1)
            time.sleep(0.05 if not deadline else min(0.05, max(0.0,
                       (deadline - self._clock()) / _TICKS_PER_SECOND)))
