# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Serial byte I/O without exposing platform handles to CAT.

On Windows `rts` and `dtr` are set before `open()` so pyserial's first
SetCommState writes them disabled. CreateFile precedes that call, and what the
driver does with the lines in between is its own. A CP210x or FTDI adapter
is unmeasured until the rig-attached open test.
"""
from __future__ import annotations

import os
import re
import select
import time
import weakref

from . import posix

fcntl = posix.module('fcntl')
termios = posix.module('termios')


def device_name(port):
    if os.name == 'nt':
        match = re.fullmatch(r'(?:\\\\\.\\)?COM([0-9]+)', port, re.IGNORECASE)
        if match:
            return '\\\\.\\COM' + match[1]
    return port


def same_port(a, b):
    if os.name == 'nt':
        return device_name(a).casefold() == device_name(b).casefold()
    return os.path.realpath(a) == os.path.realpath(b)


def open_port(factory, port, *, baudrate, rts, dtr, write_timeout, stopbits=1):
    options = dict(rtscts=False, dsrdtr=False) if os.name == 'nt' else {}
    ser = factory(port=None, baudrate=baudrate, bytesize=8, parity='N',
                  stopbits=stopbits, timeout=0, write_timeout=write_timeout, **options)
    try:
        ser.rts, ser.dtr = rts, dtr
        ser.port = device_name(port)
        try:
            ser.open()
        except OSError as exc:
            if os.name == 'nt' and (getattr(exc, 'winerror', None) == 5
                    or 'PermissionError(13,' in str(exc)):
                deadline = time.monotonic() + 1
                from . import winproc
                raise type(exc)(winproc.port_refusal(port, deadline=deadline)) from exc
            raise
        if os.name == 'posix':
            fcntl.ioctl(ser.fileno(), termios.TIOCEXCL)
            os.set_blocking(ser.fileno(), False)
        else:
            from . import winproc
            try:
                retire = winproc.record_port(ser.port)
            except winproc.ClaimError as exc:
                raise OSError(f'cannot record {port} holder: {exc}') from exc
            if retire is not None:
                finalized = weakref.finalize(ser, retire)
                close = ser.close
                def close_port():
                    close()
                    finalized()
                ser.close = close_port
    except BaseException:
        if os.name == 'nt':
            try:
                ser.close()
            except Exception:
                pass
        else:
            ser.close()
        raise
    return ser


_ESCAPE = {('rts', True): 3, ('rts', False): 4, ('dtr', True): 5, ('dtr', False): 6}
IOCTL_SERIAL_GET_DTRRTS = 0x1B0078
SERIAL_DTR_STATE, SERIAL_RTS_STATE = 1, 2
_pending_readbacks = []


class _ReadbackRequest:
    __slots__ = ('overlapped', 'state', 'returned', 'abandoned')

    def __init__(self, overlapped, state, returned):
        self.overlapped = overlapped
        self.state = state
        self.returned = returned
        self.abandoned = False


def drive(ser, line, on):
    """Set RTS or DTR, raising when the driver refuses.

    pyserial's Windows setters discard EscapeCommFunction's result, so a refused
    clear would pass for an accepted one. Windows only; POSIX keying is ptt.py's."""
    import ctypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    if not kernel32.EscapeCommFunction(ctypes.c_void_p(ser._port_handle), _ESCAPE[line, bool(on)]):
        raise ctypes.WinError(ctypes.get_last_error())
    setattr(ser, '_' + line + '_state', bool(on))


def read_outputs(ser, wait_s=0.25):
    """Windows driver's DTR/RTS mask, or None when it cannot be read within `wait_s`."""
    for request in tuple(_pending_readbacks):
        if request.abandoned and request.overlapped.Internal != 0x103:  # STATUS_PENDING
            try:
                _pending_readbacks.remove(request)
            except ValueError:
                pass
    handle = getattr(ser, '_port_handle', None)
    if not handle or handle == -1:
        return None
    import ctypes
    from serial.win32 import OVERLAPPED

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    pointer, ulong = ctypes.c_void_p, ctypes.c_uint32
    kernel32.CreateEventW.argtypes = [pointer, ctypes.c_int, ctypes.c_int, pointer]
    kernel32.CreateEventW.restype = pointer
    kernel32.DeviceIoControl.argtypes = [pointer, ulong, pointer, ulong, pointer,
                                        ulong, pointer, pointer]
    kernel32.GetOverlappedResult.argtypes = [pointer, pointer, pointer, ctypes.c_int]
    kernel32.WaitForSingleObject.argtypes = [pointer, ulong]
    kernel32.WaitForSingleObject.restype = ulong
    kernel32.CancelIoEx.argtypes = [pointer, pointer]
    kernel32.CloseHandle.argtypes = [pointer]
    overlapped, state, returned = OVERLAPPED(), ulong(), ulong()
    request = _ReadbackRequest(overlapped, state, returned)
    event, pending = None, False
    try:
        event = kernel32.CreateEventW(None, True, False, None)
        if not event:
            return None
        overlapped.hEvent = event
        _pending_readbacks.append(request)
        pending = True
        if not kernel32.DeviceIoControl(handle, IOCTL_SERIAL_GET_DTRRTS, None, 0,
                                        ctypes.byref(state), 4, ctypes.byref(returned),
                                        ctypes.byref(overlapped)):
            if ctypes.get_last_error() != 997:  # ERROR_IO_PENDING
                pending = False
                _pending_readbacks.remove(request)
                return None
            if kernel32.WaitForSingleObject(event, max(0, int(wait_s * 1000))) != 0:
                return None
        completed = kernel32.GetOverlappedResult(handle, ctypes.byref(overlapped),
                                                ctypes.byref(returned), False)
        pending = False
        _pending_readbacks.remove(request)
        if not completed:
            return None
        return state.value if returned.value == 4 else None
    except OSError:
        return None
    finally:
        if pending:
            # Cancellation requests completion; it does not wait for driver writes.
            try:
                kernel32.CancelIoEx(handle, ctypes.byref(overlapped))
            except OSError:
                pass
            request.abandoned = True
        if event:
            try:
                kernel32.CloseHandle(event)
            except OSError:
                pass


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('CAT deadline expired')
    return remaining


def write(ser, raw, deadline):
    if os.name == 'nt':
        remaining = _remaining(deadline)
        if ser.write_timeout != remaining:
            ser.write_timeout = remaining
        if ser.write(raw) != len(raw):
            raise OSError('CAT short write')
        return
    while raw:
        _remaining(deadline)
        _, writable, _ = select.select([], [ser.fileno()], [], min(0.01, _remaining(deadline)))
        if not writable:
            continue
        try:
            count = os.write(ser.fileno(), raw)
        except BlockingIOError:
            continue
        if count <= 0:
            raise OSError('CAT write made no progress')
        raw = raw[count:]


def read(ser, wait):
    wait = max(0, wait)
    if os.name == 'nt':
        if ser.timeout != wait:
            ser.timeout = wait
        chunk = ser.read(1)
        if not chunk:
            return b''
        waiting = ser.in_waiting
        if waiting:
            if ser.timeout != 0:
                ser.timeout = 0
            chunk += ser.read(waiting)
        return chunk
    readable, _, _ = select.select([ser.fileno()], [], [], wait)
    if not readable:
        return b''
    chunk = os.read(ser.fileno(), 4096)
    if not chunk:
        raise EOFError('CAT serial disconnected')
    return chunk
