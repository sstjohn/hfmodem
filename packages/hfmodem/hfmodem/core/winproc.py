# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import _thread
import ctypes as C
from contextlib import contextmanager
from contextvars import ContextVar
from ctypes import wintypes as W
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import uuid

from . import winclaim
from .claim import ClaimError

IDENTIFY = 0x1000 | 0x100000
END = IDENTIFY | 0x0001
INSPECT = IDENTIFY | 0x0040 | 0x0400
STOP_GRACE_S = 2.0
REAP_GRACE_S = 12.0
KILL_WAIT_S = 5.0
WRITE_CANCEL_WAIT_S = 1.0
_MAX_INVENTORY = 64 * 1024 * 1024
_MISMATCH = 0xc0000004
_PENDING = 997
_ABORTED = 995
_NOT_FOUND = 1168
_BROKEN_PIPE = 109
_INVALID = winclaim.INVALID_HANDLE_VALUE
_FILE_RENAME_INFO_EX = 22
_FILE_RENAME_FLAG_REPLACE_IF_EXISTS = 0x1
_FILE_RENAME_FLAG_POSIX_SEMANTICS = 0x2
_FILE_DISPOSITION_INFO_EX = 21
_FILE_DISPOSITION_FLAG_DELETE = 0x1
_FILE_DISPOSITION_FLAG_POSIX_SEMANTICS = 0x2
THREAD_PRIORITY_BELOW_NORMAL = -1


class ProcessAccessError(ClaimError):
    def __init__(self, pid, rights, code=5):
        self.pid, self.rights, self.winerror = pid, rights, code
        action = 'inspect' if rights & 0x0040 else 'end' if rights & 1 else 'identify'
        super().__init__(f"cannot {action} pid {pid}: access denied")


class InventoryUnavailable(ClaimError):
    def __init__(self, name, status):
        self.name, self.status = name, status & 0xffffffff
        super().__init__(f"{name} unavailable: NTSTATUS {self.status:#010x}")


class _PROCESSENTRY32W(C.Structure):
    _fields_ = [('dwSize', W.DWORD), ('cntUsage', W.DWORD), ('th32ProcessID', W.DWORD),
                ('th32DefaultHeapID', C.c_size_t), ('th32ModuleID', W.DWORD),
                ('cntThreads', W.DWORD), ('th32ParentProcessID', W.DWORD),
                ('pcPriClassBase', W.LONG), ('dwFlags', W.DWORD), ('szExeFile', W.WCHAR * 260)]


class _SECURITY_ATTRIBUTES(C.Structure):
    _fields_ = [('nLength', W.DWORD), ('lpSecurityDescriptor', C.c_void_p),
                ('bInheritHandle', W.BOOL)]


class _THREADENTRY32(C.Structure):
    _fields_ = [('dwSize', W.DWORD), ('cntUsage', W.DWORD), ('th32ThreadID', W.DWORD),
                ('th32OwnerProcessID', W.DWORD), ('tpBasePri', W.LONG),
                ('tpDeltaPri', W.LONG), ('dwFlags', W.DWORD)]


class _PROCESS_HANDLE_ENTRY(C.Structure):
    _fields_ = [('HandleValue', W.HANDLE), ('HandleCount', C.c_size_t),
                ('PointerCount', C.c_size_t), ('GrantedAccess', W.ULONG),
                ('ObjectTypeIndex', W.ULONG), ('HandleAttributes', W.ULONG),
                ('Reserved', W.ULONG)]


class _FILE_PROCESS_IDS(C.Structure):
    _fields_ = [('NumberOfProcessIdsInList', W.ULONG), ('ProcessIdList', C.c_size_t * 1)]


class _OVERLAPPED(C.Structure):
    _fields_ = [('Internal', C.c_size_t), ('InternalHigh', C.c_size_t),
                ('Offset', W.DWORD), ('OffsetHigh', W.DWORD), ('hEvent', W.HANDLE)]


class _UNICODE_STRING(C.Structure):
    _fields_ = [('Length', W.USHORT), ('MaximumLength', W.USHORT), ('Buffer', C.c_void_p)]


class _FILE_RENAME_INFO(C.Structure):
    _fields_ = [('Flags', W.DWORD), ('RootDirectory', W.HANDLE),
                ('FileNameLength', W.DWORD), ('FileName', W.WCHAR * 1)]


class _JOB_LIMITS(C.Structure):
    _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64),
                ('flags', W.DWORD), ('minimum', C.c_size_t), ('maximum', C.c_size_t),
                ('active', W.DWORD), ('affinity', C.c_size_t),
                ('priority', W.DWORD), ('scheduling', W.DWORD)]


class _JOB_EXTENDED_LIMITS(C.Structure):
    _fields_ = [('basic', _JOB_LIMITS), ('io', C.c_uint64 * 6),
                ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
                ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]


_api_cache = None


def _api():
    global _api_cache
    if _api_cache is None:
        base = winclaim._api()
        api = SimpleNamespace(**vars(base))
        dlls = {name: C.WinDLL(name, use_last_error=True) for name in
                ('kernel32', 'advapi32', 'ntdll')}
        P, D, H = C.c_void_p, W.DWORD, W.HANDLE
        PD = C.POINTER(D)
        for dll, name, result, args in (
            ('kernel32', 'QueryFullProcessImageNameW', W.BOOL, [H, D, W.LPWSTR, PD]),
            ('kernel32', 'K32EnumProcesses', W.BOOL, [P, D, PD]),
            ('kernel32', 'CreateToolhelp32Snapshot', H, [D, D]),
            ('kernel32', 'Process32FirstW', W.BOOL, [H, C.POINTER(_PROCESSENTRY32W)]),
            ('kernel32', 'Process32NextW', W.BOOL, [H, C.POINTER(_PROCESSENTRY32W)]),
            ('kernel32', 'Thread32First', W.BOOL, [H, C.POINTER(_THREADENTRY32)]),
            ('kernel32', 'Thread32Next', W.BOOL, [H, C.POINTER(_THREADENTRY32)]),
            ('kernel32', 'OpenThread', H, [D, W.BOOL, D]),
            ('kernel32', 'CreateThread', H, [P, C.c_size_t, P, P, D, PD]),
            ('kernel32', 'GetProcessIdOfThread', D, [H]),
            ('kernel32', 'ResumeThread', D, [H]),
            ('kernel32', 'TerminateProcess', W.BOOL, [H, W.UINT]),
            ('kernel32', 'CreateJobObjectW', H, [P, W.LPCWSTR]),
            ('kernel32', 'SetInformationJobObject', W.BOOL, [H, C.c_int, P, D]),
            ('kernel32', 'AssignProcessToJobObject', W.BOOL, [H, H]),
            ('kernel32', 'TerminateJobObject', W.BOOL, [H, W.UINT]),
            ('kernel32', 'CreateEventW', H, [P, W.BOOL, W.BOOL, W.LPCWSTR]),
            ('kernel32', 'OpenEventW', H, [D, W.BOOL, W.LPCWSTR]),
            ('kernel32', 'SetEvent', W.BOOL, [H]),
            ('kernel32', 'WaitForMultipleObjects', D, [D, C.POINTER(H), W.BOOL, D]),
            ('kernel32', 'SetConsoleCtrlHandler', W.BOOL, [P, W.BOOL]),
            ('kernel32', 'GetCurrentThread', H, []),
            ('kernel32', 'SetThreadPriority', W.BOOL, [H, C.c_int]),
            ('kernel32', 'GetThreadPriority', C.c_int, [H]),
            ('kernel32', 'CancelSynchronousIo', W.BOOL, [H]),
            ('kernel32', 'CreateNamedPipeW', H, [W.LPCWSTR, D, D, D, D, D, D, P]),
            ('kernel32', 'CancelIoEx', W.BOOL, [H, P]),
            ('kernel32', 'GetOverlappedResult', W.BOOL, [H, P, PD, W.BOOL]),
            ('kernel32', 'WriteFile', W.BOOL, [H, P, D, PD, P]),
            ('kernel32', 'PeekNamedPipe', W.BOOL, [H, P, D, PD, PD, PD]),
            ('kernel32', 'SetFileInformationByHandle', W.BOOL, [H, C.c_int, P, D]),
            ('advapi32', 'ConvertSidToStringSidW', W.BOOL, [P, C.POINTER(W.LPWSTR)]),
            ('advapi32', 'ConvertStringSecurityDescriptorToSecurityDescriptorW', W.BOOL,
             [W.LPCWSTR, D, C.POINTER(P), PD]),
            ('ntdll', 'NtQueryInformationProcess', C.c_int32, [H, C.c_int, P, W.ULONG, PD]),
            ('ntdll', 'NtQueryObject', C.c_int32, [H, C.c_int, P, W.ULONG, PD]),
            ('ntdll', 'NtQueryVolumeInformationFile', C.c_int32,
             [H, C.POINTER(winclaim.IO_STATUS_BLOCK), P, W.ULONG, C.c_int]),
        ):
            fn = getattr(dlls[dll], name)
            fn.restype, fn.argtypes = result, args
            setattr(api, name, fn)
        _api_cache = api
    return _api_cache


def lower_thread_priority():
    api = _api()
    if not api.SetThreadPriority(api.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL):
        raise C.WinError(C.get_last_error())


class Job:
    """Contain a suspended child and its descendants until the job is closed."""
    def __init__(self):
        api = _api()
        self.handle = api.CreateJobObjectW(None, None)
        if not self.handle:
            raise winclaim._error('CreateJobObjectW')
        limits = _JOB_EXTENDED_LIMITS()
        limits.basic.flags = 0x2000
        try:
            if not api.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)):
                raise winclaim._error('SetInformationJobObject')
        except BaseException:
            self.close()
            raise

    def assign(self, process):
        if not _api().AssignProcessToJobObject(self.handle, int(process._handle)):
            raise winclaim._error('AssignProcessToJobObject')

    def terminate(self):
        if not _api().TerminateJobObject(self.handle, 1):
            raise winclaim._error('TerminateJobObject')

    def close(self):
        if self.handle is not None:
            winclaim.close_handle(self.handle)
            self.handle = None


def _milliseconds(seconds):
    return min(0xfffffffe, max(0, math.ceil(seconds * 1000)))


def _wait(handle, timeout):
    result = _api().WaitForSingleObject(handle, _milliseconds(timeout))
    if result not in (0, 258):
        raise winclaim._error('WaitForSingleObject')
    return result == 0


@dataclass
class Pinned:
    handle: int
    pid: int
    started: str
    image: str

    @property
    def identity(self):
        return self.pid, self.started

    @property
    def creation(self):
        return int(self.started.removeprefix('win:'))

    def wait(self, timeout=0):
        return _wait(self.handle, timeout)

    def terminate(self):
        if not _api().TerminateProcess(self.handle, 1):
            code = C.get_last_error()
            if self.wait():
                return
            if code == 5:
                raise ProcessAccessError(self.pid, END)
            raise winclaim._error(f'TerminateProcess pid {self.pid}', code)

    def close(self):
        if self.handle is not None:
            winclaim.close_handle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pin(handle, pid, rights=IDENTIFY):
    try:
        started, exited = winclaim.process_info(pid, handle=handle)
    except ClaimError:
        if C.get_last_error() == 5:
            raise ProcessAccessError(pid, rights) from None
        raise
    image, size = C.create_unicode_buffer(32768), W.DWORD(32768)
    if not _api().QueryFullProcessImageNameW(handle, 0, image, C.byref(size)) and not exited:
        code = C.get_last_error()
        if code == 5:
            raise ProcessAccessError(pid, rights)
        raise winclaim._error(f'cannot identify image for pid {pid}', code)
    return Pinned(handle, pid, started, image.value)


def open(pid, rights=IDENTIFY):
    handle = _api().OpenProcess(rights, False, pid)
    if not handle:
        code = C.get_last_error()
        if code == 87:
            return None
        if code == 5:
            raise ProcessAccessError(pid, rights)
        raise winclaim._error(f'OpenProcess pid {pid}', code)
    try:
        return _pin(handle, pid, rights)
    except BaseException:
        winclaim.close_handle(handle)
        raise


def creation_flags(role, *, python=False, own_session=False):
    if role == 'terminal':
        return 0
    if role in ('native', 'supervisor') or (own_session and not python):
        return 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    return 0x00000200 if own_session else 0


def python_command(args, *, env=None):
    environment = dict(os.environ if env is None else env)
    environment['__PYVENV_LAUNCHER__'] = sys.executable
    return [sys._base_executable, *args], environment


def python_suspended(args, *, creationflags=0, env=None, **options):
    from .winconsole import child_streams
    for name, stream in zip(('stdin', 'stdout', 'stderr'),
                            child_streams(tuple(options.get(k) for k in ('stdin', 'stdout', 'stderr')))):
        if stream is not None:
            options[name] = stream
    command, environment = python_command(args, env=env)
    return subprocess.Popen(command, env=environment, close_fds=True,
                            creationflags=4 | creationflags, **options)


def push_handle(child, handle):
    value = W.HANDLE()
    api = _api()
    if not api.DuplicateHandle(api.GetCurrentProcess(), handle, child.handle,
                               C.byref(value), 0, False, 2):
        raise winclaim._error(f'DuplicateHandle into pid {child.pid}')
    return value.value


def resume(child):
    api = _api()
    snapshot = thread = None
    try:
        snapshot = api.CreateToolhelp32Snapshot(4, 0)
        if snapshot == _INVALID:
            snapshot = None
            raise winclaim._error('CreateToolhelp32Snapshot threads')
        row, tids = _THREADENTRY32(dwSize=C.sizeof(_THREADENTRY32)), []
        ok = api.Thread32First(snapshot, C.byref(row))
        while ok:
            if row.th32OwnerProcessID == child.pid:
                tids.append(row.th32ThreadID)
            row.dwSize = C.sizeof(row)
            ok = api.Thread32Next(snapshot, C.byref(row))
        if C.get_last_error() != 18:
            raise winclaim._error('Thread32Next')
        if len(tids) != 1:
            raise ClaimError(f'cannot resume pid {child.pid}: expected one thread, got {len(tids)}')
        thread = api.OpenThread(0x0002 | 0x0800, False, tids[0])
        if not thread:
            raise winclaim._error('OpenThread')
        if api.GetProcessIdOfThread(thread) != child.pid:
            raise ClaimError(f'cannot resume pid {child.pid}: thread owner changed')
        count = api.ResumeThread(thread)
        if count != 1:
            raise ClaimError(f'cannot resume pid {child.pid}: suspend count {count}')
    except BaseException:
        child.terminate()
        raise
    finally:
        for handle in (thread, snapshot):
            if handle is not None:
                winclaim.close_handle(handle)


def _census():
    size = 1024
    while True:
        data, needed = (W.DWORD * size)(), W.DWORD()
        if not _api().K32EnumProcesses(data, C.sizeof(data), C.byref(needed)):
            raise winclaim._error('EnumProcesses')
        if needed.value < C.sizeof(data):
            return set(data[:needed.value // C.sizeof(W.DWORD)])
        size *= 2


def pinned(pids=None, *, rights=IDENTIFY):
    entries, unverified = [], set()
    try:
        for pid in sorted(_census() if pids is None else set(pids)):
            try:
                entry = open(pid, rights)
            except ClaimError:
                unverified.add(pid)
                continue
            if entry is not None:
                entries.append(entry)
        return entries, unverified
    except BaseException:
        for entry in entries:
            entry.close()
        raise


def _parents():
    api = _api()
    snapshot = api.CreateToolhelp32Snapshot(2, 0)
    if snapshot == _INVALID:
        raise winclaim._error('CreateToolhelp32Snapshot')
    try:
        row = _PROCESSENTRY32W(dwSize=C.sizeof(_PROCESSENTRY32W))
        result, parents = api.Process32FirstW(snapshot, C.byref(row)), {}
        while result:
            parents[row.th32ProcessID] = row.th32ParentProcessID
            result = api.Process32NextW(snapshot, C.byref(row))
        if C.get_last_error() != 18:
            raise winclaim._error('Process32NextW')
        return parents
    finally:
        winclaim.close_handle(snapshot)


class Lineage:
    def __init__(self):
        self.links = {}

    def update(self, entries):
        entries = {entry.pid: entry for entry in entries}
        # Pinning precedes Toolhelp, so a recycled parent must be younger than its child.
        for child, parent in _parents().items():
            c, p = entries.get(child), entries.get(parent)
            if c is not None and p is not None and p.creation < c.creation:
                self.links[c.identity] = p.identity
        return self

    def ancestors(self, entry):
        current = entry.identity if isinstance(entry, Pinned) else entry
        result = set()
        while current in self.links and self.links[current] not in result:
            current = self.links[current]
            result.add(current)
        return result

    def descendants(self, entry):
        identity = entry.identity if isinstance(entry, Pinned) else entry
        return {child for child in self.links if identity in self.ancestors(child)}


def _probe(name, query, *, size=256):
    while True:
        data, needed = C.create_string_buffer(size), W.DWORD()
        status = query(data, size, C.byref(needed)) & 0xffffffff
        if status == 0:
            return data
        if status != _MISMATCH or size >= _MAX_INVENTORY:
            raise InventoryUnavailable(name, status)
        size = min(_MAX_INVENTORY, max(size * 2, needed.value))


def using(path):
    # The query excludes its own handle but includes attributes-only openers in other processes.
    name = 'FileProcessIdsUsingFileInformation'
    handle = winclaim.open_handle(path, 0x80, 7)
    try:
        iosb = winclaim.IO_STATUS_BLOCK()
        data = _probe(name, lambda data, size, needed:
                      _api().NtQueryInformationFile(handle, C.byref(iosb), data, size, 47),
                      size=64)
        offset = _FILE_PROCESS_IDS.ProcessIdList.offset
        count = W.ULONG.from_buffer(data).value
        if count > (len(data) - offset) // C.sizeof(C.c_size_t):
            raise InventoryUnavailable(name, 0xc000000d)
        pids = (C.c_size_t * count).from_buffer(data, offset)
        rows = []
        for pid in sorted(set(pids)):
            image = ''
            try:
                entry = open(pid)
                if entry is not None:
                    with entry:
                        image = entry.image
            except ClaimError:
                pass
            rows.append((pid, image))
        return rows
    finally:
        winclaim.close_handle(handle)


def _same_user(entry, sid):
    token = W.HANDLE()
    api = _api()
    if not api.OpenProcessToken(entry.handle, 8, C.byref(token)):
        return False
    try:
        return bool(api.EqualSid(sid, winclaim._token_sid(token, 1)))
    finally:
        winclaim.close_handle(token)


def _port_role(record):
    child = record['child']
    image = Path(child['image']).stem.casefold()
    if image in ('rigctld', 'rigctl'):
        return image
    return {'python': 'modem', 'supervisor': 'modem', 'native': image,
            'one-shot': image}.get(record['role'], record['role'])


_ports_lock = threading.RLock()
_ports = {}


def record_port(port):
    token = object()

    def publish():
        with _own() as own:
            path = _path('ports', own)
            with _record_lock(path, create=True):
                _write_json(path, dict(process=_identity(own),
                            ports=sorted(set(_ports.values()))))

    with _ports_lock:
        _ports[token] = port
        try:
            publish()
        except BaseException:
            del _ports[token]
            raise

    def retire():
        with _ports_lock:
            if token in _ports:
                del _ports[token]
                publish()

    return retire


def recorded_devices(records, publisher):
    from . import claim
    roles = {}
    for record in records:
        child = record['child']
        identity = child['pid'], child['started']
        role = roles[identity] = _port_role(record)
        if record['phase'] not in ('resuming', 'acknowledged'):
            continue
        devices = [record['device']] if (record['phase'] == 'acknowledged'
                   and isinstance(record.get('device'), str)) else []
        yield identity, role, devices
    if publisher is not None:
        roles.setdefault((publisher.pid, publisher.started), publisher.kind)
    directory = claim.state_dir() / 'rig.claim.ports.d'
    try:
        paths = sorted(path for path in directory.iterdir() if path.suffix == '.json')
    except FileNotFoundError:
        return
    for path in paths:
        record = _read_json(path)
        if record is None:
            continue
        process = record.get('process') if isinstance(record, dict) else None
        if (not isinstance(process, dict) or type(process.get('pid')) is not int
                or process['pid'] <= 0 or not isinstance(process.get('started'), str)
                or not process['started'].startswith('win:')
                or not process['started'][4:].isdecimal()
                or not isinstance(process.get('image'), str)
                or not isinstance(record.get('ports'), list)
                or not all(isinstance(port, str) and port for port in record['ports'])):
            raise ClaimError(f'invalid port record: {path}')
        try:
            entry = open(process['pid'])
        except ClaimError:
            continue
        if entry is None:
            _prune_record(path)
            continue
        with entry:
            if entry.started != process['started'] or entry.wait():
                _prune_record(path)
                continue
            yield entry.identity, roles.get(entry.identity, 'process'), record['ports']


def _port_holder(port, deadline):
    from . import claim, serialline
    try:
        publisher = claim.holder(None, attempts=1)
    except ClaimError:
        publisher = None
    records, unresolved = live_launches()
    for identity, role, devices in recorded_devices(records, publisher):
        if time.monotonic() >= deadline:
            return None
        if identity[0] in unresolved or not any(serialline.same_port(port, p) for p in devices):
            continue
        entry = open(identity[0])
        if entry is not None:
            with entry:
                if entry.identity == identity and not entry.wait():
                    return f'{port} is held by your hfmodem {role} (pid {entry.pid})'
    if time.monotonic() >= deadline:
        return None
    api, token = _api(), W.HANDLE()
    if not api.OpenProcessToken(api.GetCurrentProcess(), 8, C.byref(token)):
        return None
    entries = []
    try:
        sid = winclaim._token_sid(token, 1)
        for pid in sorted(_census()):
            if time.monotonic() >= deadline:
                return None
            try:
                entry = open(pid, INSPECT)
                if entry is None:
                    continue
                try:
                    if not entry.wait() and _same_user(entry, sid):
                        entries.append(entry)
                        entry = None
                finally:
                    if entry is not None:
                        entry.close()
            except (OSError, ClaimError):
                continue
        if not entries or time.monotonic() >= deadline:
            return None
        result = device_holders(entries, {port}, deadline=deadline)
        if result.status in ('busy', 'unavailable'):
            return None
        for entry in entries:
            walk = result.entries.get(entry.pid)
            if walk is not None and port in walk.devices and not entry.wait():
                return f'{port} is open in pid {entry.pid} ({Path(entry.image).name})'
    finally:
        for entry in entries:
            entry.close()
        winclaim.close_handle(token)
    return None


_port_refusals = ContextVar('port_refusals', default=None)


@contextmanager
def port_refusals(evidence):
    token = _port_refusals.set(dict(evidence))
    try:
        yield
    finally:
        _port_refusals.reset(token)


def port_refusal(port, *, deadline):
    fallback = f'{port} is open in another process'
    # Cleanup already inspected its candidates; proof must not start a wider walk.
    evidence = _port_refusals.get()
    if evidence is not None:
        from .serialline import same_port
        return next((message for name, message in evidence.items()
                     if same_port(name, port)), fallback)
    result, done = [], threading.Event()
    def lookup():
        try:
            result.append(_port_holder(port, deadline))
        except Exception:
            pass
        finally:
            done.set()
    if time.monotonic() >= deadline:
        return fallback
    try:
        threading.Thread(target=lookup, name='windows-port-holder', daemon=True).start()
    except (OSError, RuntimeError):
        return fallback
    # Let the deadline-bound inspector publish evidence collected before its deadline.
    if done.wait(max(0, deadline - time.monotonic()) + .05) and result:
        return result[0] or fallback
    return fallback


def command_line(entry):
    data = _probe('ProcessCommandLineInformation', lambda data, size, needed:
                  _api().NtQueryInformationProcess(entry.handle, 60, data, size, needed))
    return _unicode_value(data, 'ProcessCommandLineInformation')


def _unicode_value(data, name):
    value = _UNICODE_STRING.from_buffer(data)
    address, end = C.addressof(data), C.addressof(data) + len(data)
    if (value.Length % 2 or value.Length > value.MaximumLength or
            value.Length and (not value.Buffer or value.Buffer < address or
                              value.Buffer + value.Length > end)):
        raise InventoryUnavailable(name, 0xc000000d)
    return C.wstring_at(value.Buffer, value.Length // 2) if value.Length else ''


def process_handles(handle):
    data = _probe('ProcessHandleInformation', lambda data, size, needed:
                  _api().NtQueryInformationProcess(handle, 51, data, size, needed))
    offset = 2 * C.sizeof(C.c_size_t)
    count = C.c_size_t.from_buffer(data).value
    if count > (len(data) - offset) // C.sizeof(_PROCESS_HANDLE_ENTRY):
        raise InventoryUnavailable('ProcessHandleInformation', 0xc000000d)
    return [(row.HandleValue, row.ObjectTypeIndex) for row in
            (_PROCESS_HANDLE_ENTRY * count).from_buffer(data, offset)]


def _identity(entry):
    return dict(pid=entry.pid, started=entry.started, image=entry.image)


def _own():
    return open(os.getpid())


def _path(kind, entry):
    from . import claim
    return claim.state_dir() / f'rig.claim.{kind}.d' / f'{entry.pid}-{entry.creation}.json'


def _replace(source, target):
    name = str(Path(target).resolve()).encode('utf-16-le')
    data = C.create_string_buffer(C.sizeof(_FILE_RENAME_INFO) + len(name))
    info = _FILE_RENAME_INFO.from_buffer(data)
    info.Flags = _FILE_RENAME_FLAG_REPLACE_IF_EXISTS | _FILE_RENAME_FLAG_POSIX_SEMANTICS
    info.FileNameLength = len(name)
    C.memmove(C.addressof(data) + _FILE_RENAME_INFO.FileName.offset, name, len(name))
    handle = winclaim.open_handle(source, 0x10000, 7)
    try:
        # POSIX replacement leaves existing readers attached to the old file.
        if not _api().SetFileInformationByHandle(handle, _FILE_RENAME_INFO_EX, data, len(data)):
            raise C.WinError(C.get_last_error())
    finally:
        winclaim.close_handle(handle)


@contextmanager
def _record_lock(path, *, create=False):
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + KILL_WAIT_S
    while True:
        try:
            handle = winclaim.open_handle(path.with_suffix('.lock'), 1 | 2, 0,
                                          winclaim.OPEN_ALWAYS)
            break
        except FileNotFoundError:
            if create:
                raise
            yield False
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) != 32 or time.monotonic() >= deadline:
                raise
            time.sleep(.01)
    try:
        yield True
    finally:
        # Removing the sidecar would let waiters lock different file objects.
        winclaim.close_handle(handle)


def _prune_record(path):
    with _record_lock(path) as locked:
        if not locked:
            return
        try:
            handle = winclaim.open_handle(path, 0x10000, 7)
        except FileNotFoundError:
            return
        try:
            flags = W.DWORD(_FILE_DISPOSITION_FLAG_DELETE | _FILE_DISPOSITION_FLAG_POSIX_SEMANTICS)
            if not _api().SetFileInformationByHandle(handle, _FILE_DISPOSITION_INFO_EX,
                                                    C.byref(flags), C.sizeof(flags)):
                raise C.WinError(C.get_last_error())
        except FileNotFoundError:
            pass
        finally:
            winclaim.close_handle(handle)


def _write_json(path, record):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write('\n')
        deadline = time.monotonic() + 1
        while True:
            try:
                _replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, 'winerror', None) != 32 or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    try:
        h = winclaim.open_handle(path, 1, 7)
    except FileNotFoundError:
        return None
    try:
        chunks = []
        while True:
            data, count = C.create_string_buffer(65536), W.DWORD()
            if not _api().ReadFile(h, data, len(data), C.byref(count), None):
                raise winclaim._error(f'ReadFile {path}')
            if not count.value:
                break
            chunks.append(data.raw[:count.value])
            if sum(map(len, chunks)) > 1024 * 1024:
                raise ClaimError(f'record too large: {path}')
        record = json.loads(b''.join(chunks))
        if record is None:
            raise ClaimError(f'invalid record: {path}')
        return record
    finally:
        winclaim.close_handle(h)


_ROLES = {'python', 'supervisor', 'native', 'one-shot', 'cleanup', 'inspector', 'recorder'}
_PHASES = {'created', 'resuming', 'acknowledged', 'exited', 'unresolved'}
_launch_lock = threading.RLock()


def create_launch(entry, *, role, device=None, grace=STOP_GRACE_S, handles=None, launcher=None):
    if role not in _ROLES:
        raise ValueError('unknown launch role')
    grace = _grace_value(grace)
    if launcher is None:
        with _own() as own:
            launcher_identity = _identity(own)
            parent_record = read_launch(own)
    else:
        launcher_identity = _identity(launcher)
        parent_record = read_launch(launcher)
    record = dict(child=_identity(entry), launcher=launcher_identity, role=role,
                  ancestry=[launcher_identity, *(launch_ancestors(parent_record)
                                                if parent_record is not None else [])],
                  device=device, grace=grace, handles=dict(handles or {}), phase='created')
    with _launch_lock, _record_lock(_path('launch', entry), create=True):
        _write_json(_path('launch', entry), record)
    return record


def read_launch(entry=None):
    if entry is None:
        with _own() as own:
            return read_launch(own)
    record = _read_json(_path('launch', entry))
    if record is not None and (record['child']['pid'], record['child']['started']) != entry.identity:
        raise ClaimError('launch record identity mismatch')
    if record is not None:
        launch_ancestors(record)
    return record


def _launch_identity(identity):
    return (isinstance(identity, dict) and type(identity.get('pid')) is int and
            identity['pid'] > 0 and isinstance(identity.get('started'), str) and
            identity['started'].startswith('win:') and identity['started'][4:].isdecimal() and
            isinstance(identity.get('image'), str))


def launch_ancestors(record):
    """Recorded identities only; callers must match them to their pinned roots."""
    launcher = record.get('launcher')
    ancestry = record.get('ancestry', [launcher])
    if (not _launch_identity(record.get('child')) or not _launch_identity(launcher) or
            not isinstance(ancestry, list) or not ancestry or
            ancestry[0] != launcher or not all(map(_launch_identity, ancestry))):
        raise ClaimError('invalid launch ancestry')
    return ancestry


def advance_launch(entry, phase, **fields):
    if phase not in _PHASES:
        raise ValueError('invalid launch phase')
    record = _update_launch(entry, phase, **fields)
    if record is None and phase != 'exited':
        raise ClaimError(f'missing launch record for pid {entry.pid}')
    return record


def _update_launch(entry, phase=None, **fields):
    if fields.keys() - {'handles', 'grace'}:
        raise ValueError('only handles and grace may change with phase')
    if 'grace' in fields:
        fields['grace'] = _grace_value(fields['grace'])
    with _launch_lock, _record_lock(_path('launch', entry)) as locked:
        if not locked:
            return None
        record = read_launch(entry)
        if record is None:
            return None
        if phase == 'unresolved' and record['role'] != 'inspector':
            raise ValueError('invalid launch phase')
        if record['phase'] == 'exited':
            return record
        record.update(fields)
        if phase is not None:
            record['phase'] = phase
        _write_json(_path('launch', entry), record)
        return record


def launch_records():
    from . import claim
    try:
        paths = sorted(path for path in (claim.state_dir() / 'rig.claim.launch.d').iterdir()
                       if path.suffix == '.json')
    except FileNotFoundError:
        return
    for path in paths:
        record = _read_json(path)
        if record is None:
            continue
        if (not isinstance(record, dict) or record.get('role') not in _ROLES or
                record.get('phase') not in _PHASES):
            raise ClaimError(f'invalid launch record: {path}')
        child = record['child']
        if not _launch_identity(child):
            raise ClaimError(f'invalid launch identity: {path}')
        launch_ancestors(record)
        yield path, record


def live_launches(*, identify=None):
    records, unresolved = [], set()
    for path, record in launch_records():
        child = record['child']
        try:
            entry = open(child['pid']) if identify is None else identify(child)
        except ClaimError:
            unresolved.add(child['pid'])
        else:
            if entry is None:
                _prune_record(path)
                continue
            with entry:
                if entry.started != child['started'] or entry.wait():
                    _prune_record(path)
                    continue
        records.append(record)
    return records, unresolved


def event_name(kind, entry):
    if kind not in ('stop', 'int'):
        raise ValueError('unknown channel')
    return f'Local\\hfmodem.{kind}.{entry.pid}.{entry.creation}'


def _user_security():
    api, token = _api(), W.HANDLE()
    if not api.OpenProcessToken(api.GetCurrentProcess(), 8, C.byref(token)):
        raise winclaim._error('OpenProcessToken')
    sid_text, security = W.LPWSTR(), C.c_void_p()
    try:
        needed = W.DWORD()
        api.GetTokenInformation(token, 1, None, 0, C.byref(needed))
        data = C.create_string_buffer(needed.value)
        if not api.GetTokenInformation(token, 1, data, len(data), C.byref(needed)):
            raise winclaim._error('GetTokenInformation')
        sid = C.c_void_p.from_buffer(data)
        if not api.ConvertSidToStringSidW(sid, C.byref(sid_text)):
            raise winclaim._error('ConvertSidToStringSidW')
        if not api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                f'D:P(A;;GA;;;{sid_text.value})', 1, C.byref(security), None):
            raise winclaim._error('ConvertStringSecurityDescriptorToSecurityDescriptorW')
        return security
    finally:
        if sid_text:
            api.LocalFree(C.cast(sid_text, C.c_void_p))
        winclaim.close_handle(token)


def _grace_value(seconds):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('grace must be positive and finite')
    return seconds


_channels = None
_channels_lock = threading.RLock()
_declared_grace = STOP_GRACE_S
_console_handlers = []


def declare_grace(seconds):
    global _declared_grace
    with _channels_lock:
        _declared_grace = _grace_value(seconds)
        if _channels is not None:
            _channels.publish()


def _own_grace():
    from . import windeadman
    return STOP_GRACE_S if windeadman._handles else _declared_grace


class _Channels:
    def __init__(self):
        self.entry = _own()
        self.stop = self.interrupt = None
        self.handler = None
        self.installed = False
        security = None
        try:
            security = _user_security()
            attributes = _SECURITY_ATTRIBUTES(C.sizeof(_SECURITY_ATTRIBUTES), security, False)
            self.stop = _api().CreateEventW(C.byref(attributes), True, False,
                                          event_name('stop', self.entry))
            if not self.stop:
                raise winclaim._error('CreateEventW stop')
            self.interrupt = _api().CreateEventW(C.byref(attributes), False, False,
                                               event_name('int', self.entry))
            if not self.interrupt:
                raise winclaim._error('CreateEventW interrupt')
            self.publish()
            handler = C.WINFUNCTYPE(W.BOOL, W.DWORD)(self.console_close)
            if not _api().SetConsoleCtrlHandler(handler, True):
                raise winclaim._error('SetConsoleCtrlHandler')
            self.handler = handler
            _console_handlers.append(self.handler)
        except BaseException:
            self.rollback()
            raise
        finally:
            if security is not None:
                _api().LocalFree(security)

    def rollback(self):
        self.installed = False
        if self.handler is not None:
            if not _api().SetConsoleCtrlHandler(self.handler, False):
                raise winclaim._error('SetConsoleCtrlHandler remove')
            _console_handlers.remove(self.handler)
            self.handler = None
        for handle in (self.stop, self.interrupt):
            if handle:
                winclaim.close_handle(handle)
        self.stop = self.interrupt = None
        self.entry.close()

    def publish(self):
        self.grace = _own_grace()
        record = _update_launch(self.entry, grace=self.grace)
        if record is None:
            _write_json(_path('stop', self.entry), dict(**_identity(self.entry), grace=self.grace))
        if self.stop and _wait(self.stop, 0):
            from . import posix
            posix._windows_deadman.stop(self.grace)

    def run(self):
        # Installation owns the lock until startup commits or rolls back, even
        # when Thread.start raises after creating its native thread.
        with _channels_lock:
            if not self.installed:
                return
        from . import posix
        handles = (W.HANDLE * 2)(self.stop, self.interrupt)
        while True:
            result = _api().WaitForMultipleObjects(2, handles, False, 0xffffffff)
            if result == 0:
                posix._windows_deadman.stop(self.grace)
                posix._run_on_stop()
                _thread.interrupt_main(signal.SIGTERM)
                return
            if result != 1:
                raise winclaim._error('WaitForMultipleObjects channels')
            # Stop wins even if it latched just after the multiple-object wait.
            if _wait(self.stop, 0):
                continue
            signal.raise_signal(signal.SIGINT)

    def console_close(self, control):
        if control not in (2, 5, 6):
            return False
        with _channels_lock:
            if not self.installed:
                return False
        from . import coexist
        try:
            _api().SetEvent(self.stop)
            try:
                coexist.event('console_close')
            except Exception:
                coexist._lost()
        finally:
            try:
                self.entry.wait(min(self.grace + 0.5, 4.5))
            except Exception:
                pass
        return False


def install_channels():
    global _channels
    with _channels_lock:
        if _channels is None:
            channels = _Channels()
            try:
                channels.thread = threading.Thread(target=channels.run,
                                                    name='windows-stop', daemon=True)
                channels.thread.start()
                channels.installed = True
                _channels = channels
            except BaseException:
                _channels = None
                channels.rollback()
                raise
        else:
            _channels.publish()
        return _channels


def _target_grace(entry):
    record = read_launch(entry)
    if record is None:
        record = _read_json(_path('stop', entry))
        if record is not None and (record['pid'], record['started']) != entry.identity:
            raise ClaimError('stop record identity mismatch')
    return _grace_value(record['grace']) if record is not None else STOP_GRACE_S


def _request(kind, entry):
    api = _api()
    event = api.OpenEventW(2, False, event_name(kind, entry))
    if not event:
        if C.get_last_error() == 2:
            return None
        raise winclaim._error(f'cannot open {kind} channel for pid {entry.pid}')
    try:
        grace = _target_grace(entry)
        if not api.SetEvent(event):
            raise winclaim._error(f'cannot set {kind} channel for pid {entry.pid}')
        return grace
    finally:
        winclaim.close_handle(event)


def request_stop(entry):
    return _request('stop', entry)


def request_interrupt(entry):
    return _request('int', entry)


@dataclass(frozen=True)
class StopResult:
    status: str
    detail: str

    def __bool__(self):
        return self.status != 'unresolved'


class Stopper:
    def __init__(self, entry, *, stdin=None, reap_grace=REAP_GRACE_S, report=None):
        self.entry = entry
        self.requested_at = time.monotonic()
        self.grace = request_stop(entry)
        self.channel = 'stop channel' if self.grace is not None else 'no stop channel'
        if self.grace is None and stdin is not None:
            result = bounded_write(stdin, b'q\n')
            self.grace = _target_grace(entry)
            self.channel = f'stdin q ({result.status})'
        self.deadline = self.requested_at + max(reap_grace, (self.grace or 0) + 1)
        self.kill_deadline = None
        self.kill_error = None
        self.result = None
        if report is not None:
            report(self.channel)

    @property
    def next_deadline(self):
        return self.kill_deadline if self.kill_deadline is not None else self.deadline

    def poll(self):
        if self.result is not None:
            return self.result
        detail = self.channel + ('; TerminateProcess' if self.kill_deadline is not None else '')
        if self.entry.wait():
            self.result = StopResult('stopped' if self.kill_deadline is None else 'terminated', detail)
        elif time.monotonic() >= self.next_deadline:
            if self.kill_deadline is not None:
                self.result = StopResult('unresolved',
                    f'{detail}; {self.kill_error}' if self.kill_error is not None else detail)
            else:
                self.kill_deadline = time.monotonic() + KILL_WAIT_S
                try:
                    self.entry.terminate()
                except ProcessAccessError as exc:
                    self.kill_error = exc
                except ClaimError as exc:
                    self.result = StopResult('unresolved', f'{self.channel}; {exc}')
        return self.result

    def wait(self):
        while self.poll() is None:
            self.entry.wait(max(0, self.next_deadline - time.monotonic()))
        return self.result


@dataclass(frozen=True)
class IOResult:
    status: str
    data: bytes = b''
    confirmed: bool = True


# Pending kernel operations own their storage even if their caller has given up.
_unresolved_io = []


def _finish_read(handle, overlapped):
    count = W.DWORD()
    ok = _api().GetOverlappedResult(handle, C.byref(overlapped), C.byref(count), True)
    return bool(ok), count.value, 0 if ok else C.get_last_error()


class PipeReader:
    def __init__(self, handle):
        self.handle = handle
        self._pending = None
        self._lock = threading.Lock()

    def read(self, size, *, deadline, exit_event=None):
        if size <= 0:
            raise ValueError('read size must be positive')
        with self._lock:
            if self.handle is None or self._pending is not None:
                raise ClaimError('pipe reader closed or busy')
            if time.monotonic() >= deadline:
                return IOResult('deadline')
            self._pending = ()
            handle = self.handle
        event, submitted, confirmed = None, False, False
        try:
            api = _api()
            event = api.CreateEventW(None, True, False, None)
            if not event:
                raise winclaim._error('CreateEventW read')
            overlapped, buffer = _OVERLAPPED(hEvent=event), C.create_string_buffer(size)
            with self._lock:
                self._pending = (buffer, overlapped, event)
            reason = 'data'
            submitted = True
            ok = api.ReadFile(handle, buffer, size, None, C.byref(overlapped))
            code = 0 if ok else C.get_last_error()
            if not ok and code != _PENDING:
                confirmed = True
                if code == _BROKEN_PIPE:
                    return IOResult('eof')
                raise winclaim._error('ReadFile pipe', code)
            if not ok:
                handles = (W.HANDLE * (1 if exit_event is None else 2))(
                    event, *(() if exit_event is None else (exit_event,)))
                result = api.WaitForMultipleObjects(len(handles), handles, False,
                                                   _milliseconds(deadline - time.monotonic()))
                if result != 0:
                    reason = 'cancelled' if result == 1 else 'deadline' if result == 258 else 'error'
                    # ERROR_NOT_FOUND means completion won the race against cancellation.
                    if not api.CancelIoEx(handle, C.byref(overlapped)):
                        cancel_error = C.get_last_error()
                        if cancel_error != _NOT_FOUND:
                            reason = 'error'
            completed = []
            def finish():
                completed.append(_finish_read(handle, overlapped))
            thread = threading.Thread(target=finish, name='windows-pipe-completion', daemon=True)
            thread.start()
            thread.join(1)
            if thread.is_alive() or not completed:
                _unresolved_io.append((self, self._pending, thread))
                return IOResult('unresolved', confirmed=False)
            ok, count, code = completed[0]
            if not ok and code not in (_ABORTED, _BROKEN_PIPE):
                return IOResult('unresolved', confirmed=False)
            confirmed = True
            if ok:
                return IOResult('data' if count else 'eof', buffer.raw[:count])
            if code in (_ABORTED, _BROKEN_PIPE):
                return IOResult('eof' if code == _BROKEN_PIPE else reason)
        finally:
            with self._lock:
                if confirmed or not submitted:
                    if event:
                        winclaim.close_handle(event)
                    self._pending = None
                else:
                    _unresolved_io.append((self, self._pending))

    def close(self):
        with self._lock:
            if self._pending is not None:
                return False
            if self.handle is not None:
                winclaim.close_handle(self.handle)
                self.handle = None
            return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pipe_pair():
    api = _api()
    name = f'\\\\.\\pipe\\hfmodem.{os.getpid()}.{uuid.uuid4().hex}'
    security = _user_security()
    attributes = _SECURITY_ATTRIBUTES(C.sizeof(_SECURITY_ATTRIBUTES), security, False)
    server = client = None
    try:
        server = api.CreateNamedPipeW(name, 1 | 0x40000000 | 0x00080000, 0,
                                      1, 65536, 65536, 0, C.byref(attributes))
        if server == _INVALID:
            server = None
            raise winclaim._error('CreateNamedPipeW')
        client = api.CreateFileW(name, 0x40000000, 0, None, 3, 0, None)
        if client == _INVALID:
            client = None
            raise winclaim._error('CreateFileW pipe writer')
        # CreateFile has connected the client; there is no pending connect to retire.
        return PipeReader(server), client
    except BaseException:
        for handle in (client, server):
            if handle is not None:
                winclaim.close_handle(handle)
        raise
    finally:
        api.LocalFree(security)


def pipe_available(fd):
    return pipe_state(fd)[0]


def pipe_state(fd):
    count = W.DWORD()
    if not _api().PeekNamedPipe(winclaim.handle(fd), None, 0, None, C.byref(count), None):
        if C.get_last_error() == _BROKEN_PIPE:
            return 0, True
        raise winclaim._error('PeekNamedPipe')
    return count.value, False


def bounded_write(stream, data=b'q\n', *, timeout=1.0):
    api = _api()
    buffer = C.create_string_buffer(data)
    state, ready, finished = {}, threading.Event(), threading.Event()
    ownership = threading.Lock()
    thread = None
    handle = winclaim.duplicate(winclaim.handle(stream.fileno()))
    def write():
        try:
            with ownership:
                if state.get('cancelled'):
                    return
                state['active'] = True
            thread_handle = W.HANDLE()
            if not api.DuplicateHandle(api.GetCurrentProcess(), api.GetCurrentThread(),
                                       api.GetCurrentProcess(), C.byref(thread_handle), 0, False, 2):
                state['error'] = C.get_last_error()
                ready.set()
                return
            state['thread_handle'] = thread_handle.value
            ready.set()
            count = W.DWORD()
            ok = api.WriteFile(handle, buffer, len(data), C.byref(count), None)
            state.update(ok=bool(ok), count=count.value, error=0 if ok else C.get_last_error())
        finally:
            finished.set()
    try:
        # A native thread can exist even when start raises. Keep it out of the
        # operation until startup succeeds, or revoke its access before cleanup.
        with ownership:
            try:
                thread = threading.Thread(target=write, name='windows-pipe-writer', daemon=True)
                thread.start()
            except BaseException:
                state['cancelled'] = True
                raise
        thread.join(timeout)
        if not finished.is_set():
            if ready.is_set() and 'thread_handle' in state:
                api.CancelSynchronousIo(state['thread_handle'])
            thread.join(WRITE_CANCEL_WAIT_S)
    except BaseException:
        # An interrupted join can mark a Python Thread stopped before its native I/O returns.
        with ownership:
            state['cancelled'] = True
            confirmed = not state.get('active') or finished.is_set()
        if not confirmed:
            _unresolved_io.append((handle, buffer, thread, state, stream))
        else:
            winclaim.close_handle(handle)
            if 'thread_handle' in state:
                winclaim.close_handle(state['thread_handle'])
        raise
    if not finished.is_set() or thread.is_alive():
        _unresolved_io.append((handle, buffer, thread, state, stream))
        return IOResult('unresolved', confirmed=False)
    try:
        if state.get('ok') and state['count'] == len(data):
            return IOResult('written')
        if state.get('error') == _ABORTED:
            return IOResult('cancelled')
        return IOResult('failed')
    finally:
        winclaim.close_handle(handle)
        if 'thread_handle' in state:
            winclaim.close_handle(state['thread_handle'])


@dataclass(frozen=True)
class DeviceWalk:
    devices: frozenset[str] = frozenset()
    handles: dict[int, str] = field(default_factory=dict)
    complete: bool = False
    message: str = ''

    @property
    def counts(self):
        return {status: sum(value == status for value in self.handles.values())
                for status in ('found', 'other', 'blocked')}


@dataclass(frozen=True)
class DeviceHolders:
    entries: dict[int, DeviceWalk]
    status: str
    message: str = ''


def _inspect_checkpoint(point):
    from . import coexist
    if not coexist.enabled():
        return
    # The shared harness's :kill uses os.kill, which has no Windows signal semantics.
    if os.environ.get('HFMODEM_COEXIST_FAULT') == point + ':kill':
        coexist.barrier(point)
        coexist._note('fault', point=point)
        if not _api().TerminateProcess(_api().GetCurrentProcess(), 1):
            raise winclaim._error('TerminateProcess self')
    else:
        coexist.checkpoint(point)


def _handle_count(handle):
    count = W.DWORD()
    if not _api().GetProcessHandleCount(handle, C.byref(count)):
        raise winclaim._error('GetProcessHandleCount')
    return count.value


def _file_type_index():
    api = _api()
    handle = winclaim.open_handle(os.devnull, 0, 7)
    try:
        for value, index in process_handles(api.GetCurrentProcess()):
            if value == handle:
                return index
        raise InventoryUnavailable('File object type', 0xc000000d)
    finally:
        winclaim.close_handle(handle)


def _inspect_file(process, handle, result):
    api, duplicate = _api(), W.HANDLE()
    try:
        if not api.DuplicateHandle(process, handle, api.GetCurrentProcess(),
                                   C.byref(duplicate), 0, False, 2):
            raise winclaim._error('DuplicateHandle for inspection')
        iosb, device = winclaim.IO_STATUS_BLOCK(), (W.ULONG * 2)()
        status = api.NtQueryVolumeInformationFile(duplicate, C.byref(iosb), device,
                                                 C.sizeof(device), 4)
        if status:
            raise InventoryUnavailable('FileFsDeviceInformation', status)
        if device[0] == 0x1b:
            data = _probe('ObjectNameInformation', lambda data, size, needed:
                          api.NtQueryObject(duplicate, 1, data, size, needed))
            name = _unicode_value(data, 'ObjectNameInformation')
            if not name:
                raise InventoryUnavailable('ObjectNameInformation', 0xc000000d)
            result.update(status='found', device=name)
        else:
            result['status'] = 'other'
    except Exception:
        # A vanished handle or failed query cannot support negative evidence.
        result['status'] = 'blocked'
    finally:
        if duplicate.value is not None:
            winclaim.close_handle(duplicate)


def _inspect_worker(process, handle, deadline, callbacks):
    result = {}
    def query(_):
        try:
            _inspect_file(process, handle, result)
        except BaseException:
            result['status'] = 'blocked'
        return 0
    callback = C.WINFUNCTYPE(W.DWORD, C.c_void_p)(query)
    callbacks.append(callback)
    thread = _api().CreateThread(None, 0, callback, None, 0, None)
    if not thread:
        raise winclaim._error('CreateThread inspector worker')
    try:
        complete = _wait(thread, min(.2, max(0, deadline - time.monotonic())))
    finally:
        # threading.Thread retains a native handle until a blocked worker exits.
        winclaim.close_handle(thread)
    if complete:
        callbacks.remove(callback)
    return (result if complete else dict(status='blocked')), not complete


def _inspect(deadline):
    from . import coexist
    output = None
    code = 0
    try:
        deadline = min(deadline, time.monotonic() + 5)
        record = read_launch()
        if record is None or record['role'] != 'inspector' or record['phase'] != 'resuming':
            raise ClaimError('inspector has no resuming launch record')
        output = record['handles']['output']
        def emit(**line):
            data = (json.dumps(line, separators=(',', ':')) + '\n').encode('utf-8')
            count = W.DWORD()
            if not _api().WriteFile(output, data, len(data), C.byref(count), None) or count.value != len(data):
                raise ClaimError('inspector output closed')
        file_index = _file_type_index()
        coexist._note('inspector_handles', stage='baseline', count=_handle_count(_api().GetCurrentProcess()))
        abandoned, stopped, callbacks = 0, False, []
        for key, process in record['handles'].items():
            if not key.startswith('process:'):
                continue
            pid = int(key.split(':')[1])
            if time.monotonic() >= deadline:
                stopped = True
                break
            try:
                handles = process_handles(process)
            except InventoryUnavailable as exc:
                emit(pid=pid, status='unavailable', message=str(exc))
                continue
            complete = True
            for handle, index in handles:
                if index != file_index:
                    continue
                if time.monotonic() >= deadline:
                    stopped = True
                    break
                line, late = _inspect_worker(process, handle, deadline, callbacks)
                if late:
                    abandoned += 1
                if line.get('status') == 'blocked':
                    complete = False
                emit(pid=pid, handle=handle, **line)
                if abandoned > 8:
                    stopped = True
                    break
            emit(pid=pid, status='complete', complete=complete and not stopped)
            if stopped:
                break
        coexist._note('inspector_handles', stage='final', count=_handle_count(_api().GetCurrentProcess()),
                      abandoned=abandoned)
        emit(status='done', complete=not stopped)
        _inspect_checkpoint('inspector_after_walk')
    except Exception as exc:
        code = 1
        if output is not None:
            try:
                emit(status='error', message=str(exc))
            except Exception:
                pass
    finally:
        os._exit(code)


def _reap_inspector(entry, path):
    phase = 'exited' if entry.wait(KILL_WAIT_S) else 'unresolved'
    if path is None:
        return phase
    with _launch_lock, _record_lock(path) as locked:
        if not locked:
            return 'exited'
        record = _read_json(path)
        if record is None:
            return 'exited'
        if (record['child']['pid'], record['child']['started']) == entry.identity:
            if record['phase'] == 'exited':
                return 'exited'
            record['phase'] = phase
            _write_json(path, record)
    return phase


def _end_inspector(entry):
    try:
        entry.terminate()
    except ClaimError:
        # An exit already pending on driver I/O can reject another termination.
        pass


def _confirmed_exit(identity):
    try:
        entry = open(identity['pid'])
    except ClaimError:
        return False
    if entry is None:
        return True
    with entry:
        return entry.started != identity['started'] or entry.wait()


def _recover_inspectors(directory):
    recovering = False
    for path in sorted((directory / 'rig.claim.launch.d').glob('*.json')):
        record = _read_json(path)
        if record is None or record['role'] != 'inspector':
            continue
        identity = record['child']
        try:
            entry = open(identity['pid'], END)
        except ClaimError:
            if _confirmed_exit(identity):
                _prune_record(path)
                continue
            return recovering, identity['pid']
        if entry is None:
            _prune_record(path)
            continue
        try:
            if entry.started != identity['started'] or entry.wait():
                _prune_record(path)
                continue
            if record['phase'] == 'unresolved':
                if _confirmed_exit(record['launcher']):
                    _end_inspector(entry)
                return recovering, identity['pid']
            if _confirmed_exit(record['launcher']):
                recovering = True
                try:
                    guard = winclaim.open_handle(path.with_suffix('.reap'), 1 | 2, 0,
                                                 winclaim.OPEN_ALWAYS)
                except OSError as exc:
                    if exc.winerror != winclaim.ERROR_SHARING_VIOLATION:
                        raise
                    # The guard owner may not have terminated the inspector yet.
                    return recovering, identity['pid']
                else:
                    try:
                        _inspect_checkpoint('after_inspector_reap_guard')
                        # Several callers may recover the same dead launcher at once.
                        _end_inspector(entry)
                        if _reap_inspector(entry, path) == 'unresolved':
                            return recovering, identity['pid']
                        continue
                    finally:
                        winclaim.close_handle(guard)
        finally:
            if entry is not None:
                entry.close()
    return recovering, None


class _Inspection:
    def __init__(self, entries, devices, deadline, directory):
        self.entries = entries
        self.devices = devices
        self.deadline = deadline
        self.directory = directory
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.done = threading.Event()
        self.walks = {entry.pid: DeviceWalk() for entry in entries}
        self.status, self.message = 'incomplete', ''

    def active(self):
        return not self.stop.is_set() and time.monotonic() < self.deadline

    def outcome(self, status, message=''):
        with self.lock:
            self.status, self.message = status, message

    def slot(self):
        path = self.directory / 'rig.claim.inspect'
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        limit = time.monotonic() + 1
        recovered = False
        while self.active():
            try:
                return winclaim.open_handle(path, 1 | 2, 0, winclaim.OPEN_ALWAYS)
            except OSError as exc:
                if exc.winerror != winclaim.ERROR_SHARING_VIOLATION:
                    raise
            recovering, unavailable = _recover_inspectors(self.directory)
            if unavailable is not None:
                self.outcome('unavailable', f'inspection unavailable: pid {unavailable} (inspector) has not exited')
                return None
            if recovering and not recovered:
                limit, recovered = time.monotonic() + KILL_WAIT_S, True
            if time.monotonic() >= limit:
                self.outcome('busy', 'inspection busy')
                return None
            self.stop.wait(min(.05, max(0, min(limit, self.deadline) - time.monotonic())))
        return None

    def line(self, line):
        status = line['status']
        if status == 'done':
            with self.lock:
                self.status = 'complete' if line['complete'] and all(
                    walk.complete for walk in self.walks.values()) else 'incomplete'
            return False
        if status == 'error':
            self.outcome('incomplete', line['message'])
            return False
        pid = line['pid']
        with self.lock:
            walk = self.walks[pid]
            if status == 'unavailable':
                image = next(entry.image for entry in self.entries if entry.pid == pid)
                walk = replace(walk, message=f'cannot inspect pid {pid} ({Path(image).name})')
            elif status == 'complete':
                walk = replace(walk, complete=bool(line['complete']) and not walk.counts['blocked'])
            else:
                if status not in ('found', 'other', 'blocked'):
                    raise ClaimError('invalid inspector response')
                handles = dict(walk.handles)
                handles[line['handle']] = status
                found = {port for port, target in self.devices.items()
                         if status == 'found' and target.casefold() == line['device'].casefold()}
                walk = replace(walk, handles=handles, devices=walk.devices | found)
            self.walks[pid] = walk
        return True

    def run(self):
        slot = reader = writer = child = process = path = None
        try:
            with self.lock:
                if not self.active():
                    return
            if not isinstance(self.devices, dict):
                self.devices = {port: winclaim.serial_target(port) for port in self.devices}
            slot = self.slot()
            if slot is None or not self.active():
                return
            _, unavailable = _recover_inspectors(self.directory)
            if unavailable is not None:
                self.outcome('unavailable', f'inspection unavailable: pid {unavailable} (inspector) has not exited')
                return
            if not self.active():
                return
            reader, writer = pipe_pair()
            if not self.active():
                return
            process = python_suspended(['-m', 'hfmodem.core.winproc', 'inspect', str(self.deadline)],
                                       creationflags=subprocess.CREATE_NO_WINDOW,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            handle = winclaim.duplicate(int(process._handle))
            try:
                child = _pin(handle, process.pid, END)
            except BaseException:
                _api().TerminateProcess(handle, 1)
                winclaim.close_handle(handle)
                raise
            path = _path('launch', child)
            create_launch(child, role='inspector')
            _inspect_checkpoint('after_inspector_created')
            handles = {'slot': push_handle(child, slot)}
            advance_launch(child, 'created', handles=handles)
            _inspect_checkpoint('after_inspector_slot')
            if not self.active():
                return
            handles['output'] = push_handle(child, writer)
            for entry in self.entries:
                if not self.active():
                    return
                api, handle = _api(), W.HANDLE()
                if not api.DuplicateHandle(api.GetCurrentProcess(), entry.handle,
                                           api.GetCurrentProcess(), C.byref(handle),
                                           0x0040 | 0x0400, False, 0):
                    with self.lock:
                        self.walks[entry.pid] = DeviceWalk(message=
                            f'cannot inspect pid {entry.pid} ({Path(entry.image).name})')
                    continue
                try:
                    handles[f'process:{entry.pid}'] = push_handle(child, handle.value)
                finally:
                    winclaim.close_handle(handle.value)
                advance_launch(child, 'created', handles=handles)
                _inspect_checkpoint('after_inspector_push')
            if not self.active():
                return
            advance_launch(child, 'resuming', handles=handles)
            _inspect_checkpoint('before_inspector_resume')
            if not self.active():
                return
            resume(child)
            _inspect_checkpoint('after_inspector_resume')
            winclaim.close_handle(writer)
            writer = None
            _inspect_checkpoint('inspector_before_read')
            pending = b''
            while self.active():
                result = reader.read(65536, deadline=min(self.deadline, time.monotonic() + .05))
                if result.status == 'deadline':
                    continue
                if result.status != 'data':
                    break
                pending += result.data
                if len(pending) > 1024 * 1024:
                    raise ClaimError('inspector response too large')
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if not self.line(json.loads(line)):
                        _inspect_checkpoint('inspector_before_reap')
                        return
        except Exception as exc:
            self.outcome('incomplete', str(exc))
        finally:
            # Publish evidence before the potentially stuck termination and I/O retirement.
            self.done.set()
            try:
                if child is not None:
                    _end_inspector(child)
                elif process is not None:
                    process.terminate()
            finally:
                try:
                    for handle in (slot, writer):
                        if handle is not None:
                            winclaim.close_handle(handle)
                    if reader is not None:
                        reader.close()
                    for entry in self.entries:
                        entry.close()
                    if child is not None:
                        _reap_inspector(child, path)
                finally:
                    if child is not None:
                        child.close()


def device_holders(entries, devices, *, deadline, cancelled=None):
    from . import claim
    deadline = float(deadline)
    if not math.isfinite(deadline):
        raise ValueError('deadline must be finite')
    cancel = (lambda: False) if cancelled is None else (
        cancelled.is_set if hasattr(cancelled, 'is_set') else cancelled)
    owned = []
    originals = list(entries)
    targets = dict(devices) if isinstance(devices, dict) else tuple(devices)
    state = _Inspection(owned, targets, deadline, claim.state_dir())
    try:
        for entry in originals:
            if cancel() or time.monotonic() >= deadline:
                break
            owned.append(replace(entry, handle=winclaim.duplicate(entry.handle)))
        state.walks = {entry.pid: DeviceWalk() for entry in originals}
        if len(owned) == len(originals) and not cancel() and time.monotonic() < deadline:
            thread = threading.Thread(target=state.run, name='windows-inspection', daemon=True)
            with state.lock:
                try:
                    thread.start()
                except BaseException:
                    state.entries = []
                    state.stop.set()
                    raise
                owned = []
            while not state.done.is_set() and not cancel():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                state.done.wait(min(.05, remaining))
        with state.lock:
            status = 'cancelled' if cancel() else 'deadline' if time.monotonic() >= deadline else state.status
            walks = dict(state.walks)
            if status in ('cancelled', 'deadline'):
                walks = {pid: replace(walk, complete=False) for pid, walk in walks.items()}
            return DeviceHolders(walks, status, state.message)
    finally:
        state.stop.set()
        for entry in owned:
            entry.close()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'inspect' or os.name != 'nt':
        raise SystemExit('usage: python -m hfmodem.core.winproc inspect DEADLINE')
    _inspect(float(sys.argv[2]))
