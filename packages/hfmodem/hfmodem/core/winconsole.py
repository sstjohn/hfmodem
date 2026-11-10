# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Console handles independent of the launcher's redirected CRT streams."""
from __future__ import annotations

from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import os
import sys
from types import SimpleNamespace

def utf8_console() -> None:
    """Windows writes a pipe in the ANSI code page, which has no em dash or degree
    sign. The children this starts get UTF-8 mode, which is what POSIX gives them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")


STD_INPUT, STD_OUTPUT, STD_ERROR = -10, -11, -12
PROCESSED_OUTPUT, VT_OUTPUT = 1, 4
CONSOLE_READ_NOWAIT = 2
SLICE_MS = 50


class Char(C.Union):
    _fields_ = [('UnicodeChar', W.WCHAR), ('AsciiChar', C.c_char)]


class Key(C.Structure):
    _fields_ = [('down', W.BOOL), ('repeat', W.WORD), ('vk', W.WORD),
                ('scan', W.WORD), ('char', Char), ('control', W.DWORD)]


class Event(C.Union):
    _fields_ = [('key', Key), ('padding', C.c_byte * 16)]


class InputRecord(C.Structure):
    _fields_ = [('kind', W.WORD), ('event', Event)]


_api_cache = None
_bound = 0


def api():
    global _api_cache
    if _api_cache is None:
        dll = C.WinDLL('kernel32', use_last_error=True)
        H, D, P = W.HANDLE, W.DWORD, C.c_void_p
        PD = C.POINTER(D)
        result = SimpleNamespace()
        for name, restype, args in (
            ('CreateFileW', H, [W.LPCWSTR, D, D, P, D, D, H]),
            ('CloseHandle', W.BOOL, [H]),
            ('GetStdHandle', H, [D]),
            ('SetStdHandle', W.BOOL, [D, H]),
            ('GetConsoleMode', W.BOOL, [H, PD]),
            ('SetConsoleMode', W.BOOL, [H, D]),
            ('WriteConsoleW', W.BOOL, [H, W.LPCWSTR, D, PD, P]),
            ('CreateEventW', H, [P, W.BOOL, W.BOOL, W.LPCWSTR]),
            ('SetEvent', W.BOOL, [H]),
            ('WaitForMultipleObjects', D, [D, C.POINTER(H), W.BOOL, D]),
            ('ReadConsoleInputExW', W.BOOL,
             [H, C.POINTER(InputRecord), D, PD, W.WORD]),
        ):
            fn = getattr(dll, name, None)
            if fn is not None:
                fn.restype, fn.argtypes = restype, args
            setattr(result, name, fn)
        _api_cache = result
    return _api_cache


def checked(ok, name):
    if not ok:
        raise C.WinError(C.get_last_error(), name)
    return ok


def mode(handle):
    value = W.DWORD()
    checked(api().GetConsoleMode(handle, C.byref(value)), 'GetConsoleMode')
    return value.value


def child_streams(streams):
    # GetStdHandle points at PDCurses, but children must inherit the CRT log pipe.
    return tuple(fd if stream is None and _bound else stream
                 for fd, stream in enumerate(streams))


class Console:
    def __init__(self, *, bind=False):
        global _bound
        self.api = api()
        self.held = ExitStack()
        try:
            self.input = self._open('CONIN$')
            self.output = self._open('CONOUT$')
            self.input_mode, self.output_mode = mode(self.input), mode(self.output)
            self.held.callback(self._restore_mode, self.input, self.input_mode)
            self.held.callback(self._restore_mode, self.output, self.output_mode)
            if bind:
                for which, handle in ((STD_INPUT, self.input), (STD_OUTPUT, self.output)):
                    saved = self.api.GetStdHandle(which)
                    self.held.callback(self._restore_std, which, saved)
                    checked(self.api.SetStdHandle(which, handle), 'SetStdHandle')
                _bound += 1
                self.held.callback(self._unbind)
        except BaseException:
            self.held.close()
            raise

    def _open(self, name):
        handle = self.api.CreateFileW(name, 0xc0000000, 3, None, 3, 0, None)
        if handle == C.c_void_p(-1).value or handle is None:
            raise C.WinError(C.get_last_error(), name)
        self.held.callback(self.api.CloseHandle, handle)
        return handle

    def _restore_mode(self, handle, saved):
        checked(self.api.SetConsoleMode(handle, saved), 'restore console mode')

    def _restore_std(self, which, saved):
        checked(self.api.SetStdHandle(which, saved), 'restore standard handle')

    def _unbind(self):
        global _bound
        _bound -= 1

    def enable_vt(self):
        if self.api.SetConsoleMode(self.output, self.output_mode | PROCESSED_OUTPUT | VT_OUTPUT):
            return True
        checked(self.api.SetConsoleMode(self.output, (self.output_mode | PROCESSED_OUTPUT) & ~VT_OUTPUT),
                'enable processed console output')
        return False

    def lost(self):
        return not self.api.GetConsoleMode(self.input, C.byref(W.DWORD()))

    def write(self, text):
        # WriteConsoleW counts UTF-16 code units, not Python characters.
        data = text.encode('utf-16-le', errors='surrogatepass')
        while data:
            chunk = data[:16384]
            buffer = C.create_string_buffer(chunk)
            written = W.DWORD()
            checked(self.api.WriteConsoleW(self.output, C.cast(buffer, W.LPCWSTR),
                                           len(chunk) // 2, C.byref(written), None),
                    'WriteConsoleW')
            if not written.value:
                raise OSError('WriteConsoleW wrote no characters')
            data = data[written.value * 2:]

    def flush(self):
        pass

    def close(self):
        self.held.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
