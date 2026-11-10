# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

FILE_READ_DATA = 0x1
FILE_WRITE_DATA = 0x2
FILE_APPEND_DATA = 0x4
FILE_READ_ATTRIBUTES = 0x80
READ_CONTROL = 0x20000
SYNCHRONIZE = 0x100000
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
FILE_SHARE_DELETE = 4
OPEN_EXISTING = 3
OPEN_ALWAYS = 4
ERROR_SHARING_VIOLATION = 32
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
WAIT_FAILED = 0xffffffff
INVALID_HANDLE_VALUE = C.c_void_p(-1).value
COMMON_ACCESS = FILE_READ_ATTRIBUTES | READ_CONTROL | SYNCHRONIZE
ROLES = {
    "holder": (COMMON_ACCESS | FILE_READ_DATA | FILE_WRITE_DATA, FILE_SHARE_READ, OPEN_ALWAYS),
    "cleanup": (COMMON_ACCESS | FILE_READ_DATA | FILE_WRITE_DATA, FILE_SHARE_READ, OPEN_ALWAYS),
    "admission": (COMMON_ACCESS | FILE_READ_DATA, FILE_SHARE_READ, OPEN_ALWAYS),
    "reader": (FILE_READ_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
               FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING),
    "identity": (FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                 OPEN_EXISTING),
    "bootstrap": (COMMON_ACCESS | FILE_READ_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_ALWAYS),
    "event": (FILE_APPEND_DATA | SYNCHRONIZE, FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_ALWAYS),
}


class FILE_BASIC_INFO(C.Structure):
    _fields_ = [(name, C.c_longlong) for name in
                ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")] + [
                    ("FileAttributes", W.DWORD)]


class FILE_ID_INFO(C.Structure):
    _fields_ = [("VolumeSerialNumber", C.c_ulonglong), ("FileId", C.c_ubyte * 16)]


class IO_STATUS_BLOCK(C.Structure):
    _fields_ = [("Status", C.c_void_p), ("Information", C.c_size_t)]


class RM_UNIQUE_PROCESS(C.Structure):
    _fields_ = [("dwProcessId", W.DWORD), ("ProcessStartTime", W.FILETIME)]


class RM_PROCESS_INFO(C.Structure):
    _fields_ = [("Process", RM_UNIQUE_PROCESS), ("strAppName", W.WCHAR * 256),
                ("strServiceShortName", W.WCHAR * 64), ("ApplicationType", C.c_int),
                ("AppStatus", W.ULONG), ("TSSessionId", W.DWORD), ("bRestartable", W.BOOL)]


_api_cache = None
lock = threading.RLock()


def _api():
    global _api_cache
    if _api_cache is None:
        if os.name != "nt":
            raise RuntimeError("Windows claim backend requires Windows")
        import msvcrt
        libraries = {name: C.WinDLL(name, use_last_error=True) for name in
                     ("kernel32", "kernelbase", "advapi32", "ntdll")}
        api = SimpleNamespace(msvcrt=msvcrt)

        def bind(dll, name, result, *args):
            fn = getattr(libraries[dll], name)
            fn.restype, fn.argtypes = result, list(args)
            setattr(api, name, fn)

        P = C.c_void_p
        PD = C.POINTER(W.DWORD)
        PH = C.POINTER(W.HANDLE)
        bind("kernel32", "CreateFileW", W.HANDLE, W.LPCWSTR, W.DWORD, W.DWORD, P,
             W.DWORD, W.DWORD, W.HANDLE)
        bind("kernel32", "CloseHandle", W.BOOL, W.HANDLE)
        bind("kernel32", "GetFileType", W.DWORD, W.HANDLE)
        bind("kernel32", "GetFileInformationByHandleEx", W.BOOL, W.HANDLE, C.c_int, P, W.DWORD)
        bind("kernel32", "ReadFile", W.BOOL, W.HANDLE, P, W.DWORD, PD, P)
        bind("kernel32", "SetFilePointerEx", W.BOOL, W.HANDLE, C.c_longlong,
             C.POINTER(C.c_longlong), W.DWORD)
        bind("kernel32", "GetCurrentProcess", W.HANDLE)
        bind("kernel32", "DuplicateHandle", W.BOOL, W.HANDLE, W.HANDLE, W.HANDLE, PH,
             W.DWORD, W.BOOL, W.DWORD)
        bind("kernel32", "GetStdHandle", W.HANDLE, W.DWORD)
        bind("kernel32", "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
        bind("kernel32", "GetProcessTimes", W.BOOL, W.HANDLE, C.POINTER(W.FILETIME),
             C.POINTER(W.FILETIME), C.POINTER(W.FILETIME), C.POINTER(W.FILETIME))
        bind("kernel32", "WaitForSingleObject", W.DWORD, W.HANDLE, W.DWORD)
        bind("kernel32", "GetCommandLineW", W.LPWSTR)
        bind("kernel32", "QueryDosDeviceW", W.DWORD, W.LPCWSTR, W.LPWSTR, W.DWORD)
        bind("kernel32", "GetProcessHandleCount", W.BOOL, W.HANDLE, PD)
        bind("kernel32", "LocalFree", P, P)
        bind("kernelbase", "CompareObjectHandles", W.BOOL, W.HANDLE, W.HANDLE)
        bind("advapi32", "GetSecurityInfo", W.DWORD, W.HANDLE, C.c_int, W.DWORD,
             C.POINTER(P), C.POINTER(P), C.POINTER(P), C.POINTER(P), C.POINTER(P))
        bind("advapi32", "OpenProcessToken", W.BOOL, W.HANDLE, W.DWORD, PH)
        bind("advapi32", "GetTokenInformation", W.BOOL, W.HANDLE, C.c_int, P, W.DWORD, PD)
        bind("advapi32", "EqualSid", W.BOOL, P, P)
        bind("ntdll", "NtQueryInformationFile", C.c_int32, W.HANDLE,
             C.POINTER(IO_STATUS_BLOCK), P, W.ULONG, C.c_int)
        _api_cache = api
    return _api_cache


_rm_cache = None


def _rm_api():
    global _rm_cache
    if _rm_cache is None:
        api = C.WinDLL("rstrtmgr", use_last_error=True)
        signatures = {
            "RmStartSession": [C.POINTER(W.DWORD), W.DWORD, W.LPWSTR],
            "RmRegisterResources": [W.DWORD, W.UINT, C.POINTER(W.LPCWSTR), W.UINT,
                                    C.POINTER(RM_UNIQUE_PROCESS), W.UINT, C.POINTER(W.LPCWSTR)],
            "RmGetList": [W.DWORD, C.POINTER(W.UINT), C.POINTER(W.UINT),
                          C.POINTER(RM_PROCESS_INFO), C.POINTER(W.DWORD)],
            "RmEndSession": [W.DWORD],
        }
        for name, args in signatures.items():
            function = getattr(api, name)
            function.restype, function.argtypes = W.DWORD, args
        _rm_cache = api
    return _rm_cache


def _error(message, code=None):
    from .claim import ClaimError
    code = C.get_last_error() if code is None else code
    return ClaimError(f"{message}: {C.WinError(code)}")


def handle(fd):
    return _api().msvcrt.get_osfhandle(fd)


def close_handle(h):
    if not _api().CloseHandle(h):
        raise _error("CloseHandle")


def _open(path, role):
    access, share, disposition = ROLES[role]
    return open_handle(path, access, share, disposition)


def open_handle(path, access, share, disposition=OPEN_EXISTING):
    h = _api().CreateFileW(os.fspath(path), access, share, None, disposition,
                           0x80 | 0x00200000, None)
    if h == INVALID_HANDLE_VALUE:
        raise C.WinError(C.get_last_error())
    return h


@dataclass(frozen=True)
class Entry:
    handle: int
    role: str


_registry = {}


def entries():
    with lock:
        for fd, entry in list(_registry.items()):
            try:
                live = handle(fd) == entry.handle
            except OSError:
                live = False
            if not live:
                _registry.pop(fd, None)
        return dict(_registry)


def set_role(fd, role):
    with lock:
        _registry[fd] = Entry(handle(fd), role)


def forget(fd):
    with lock:
        _registry.pop(fd, None)


def _crt(h, flags, *, role="holder"):
    # UCRT conversion into slots 0-2 also sets the process standard handles.
    # Even a non-inheritable claim there can be copied by concurrent Popen.
    # Reserve those slots with harmless handles before transferring ownership.
    with lock:
        reservations, fd = [], None
        try:
            for standard in range(3):
                try:
                    os.fstat(standard)
                except OSError:
                    reservations.append(os.open(os.devnull, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT))
            fd = _api().msvcrt.open_osfhandle(h, flags)
            os.set_inheritable(fd, False)
            set_role(fd, role)
        except BaseException:
            if fd is None:
                close_handle(h)
            else:
                forget(fd)
                os.close(fd)
            raise
        finally:
            for reserved in reservations:
                os.close(reserved)
        return fd


def open_role(path, role):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        h = _open(path, role)
    except OSError as exc:
        # The best-effort coexist logger catches OSError to mark lost events
        # without suppressing its barriers or injected faults.
        if role == "event" or exc.winerror == ERROR_SHARING_VIOLATION:
            raise
        from .claim import ClaimError
        raise ClaimError(f"cannot open {path}: {exc}") from exc
    flags = os.O_RDONLY if role in ("admission", "bootstrap") else os.O_WRONLY if role == "event" else os.O_RDWR
    # Cleanup authority exists only after its record has been published.
    return _crt(h, flags | os.O_BINARY | os.O_NOINHERIT,
                role="pending" if role == "cleanup" else role)


def bootstrap(path):
    from . import claim
    fd = open_role(path, "bootstrap")
    try:
        claim._check_inode(fd, path)
    finally:
        forget(fd)
        os.close(fd)


def _info(h, kind, structure):
    result = structure()
    if not _api().GetFileInformationByHandleEx(h, kind, C.byref(result), C.sizeof(result)):
        raise _error("GetFileInformationByHandleEx")
    return result


def file_id(h):
    info = _info(h, 18, FILE_ID_INFO)
    return info.VolumeSerialNumber, bytes(info.FileId)


def _token_sid(token, kind):
    api = _api()
    needed = W.DWORD()
    api.GetTokenInformation(token, kind, None, 0, C.byref(needed))
    if not needed.value:
        raise _error("GetTokenInformation")
    data = C.create_string_buffer(needed.value)
    if not api.GetTokenInformation(token, kind, data, len(data), C.byref(needed)):
        raise _error("GetTokenInformation")
    return C.c_void_p.from_buffer(data)


def owned_regular(h):
    api = _api()
    if api.GetFileType(h) != 1:
        return False
    if _info(h, 0, FILE_BASIC_INFO).FileAttributes & (0x400 | 0x10):
        return False
    owner, security = C.c_void_p(), C.c_void_p()
    code = api.GetSecurityInfo(h, 1, 1, C.byref(owner), None, None, None, C.byref(security))
    if code:
        raise _error("GetSecurityInfo", code)
    token = W.HANDLE()
    try:
        if not api.OpenProcessToken(api.GetCurrentProcess(), 8, C.byref(token)):
            raise _error("OpenProcessToken")
        try:
            return any(api.EqualSid(owner, _token_sid(token, kind)) for kind in (1, 4))
        finally:
            close_handle(token)
    finally:
        api.LocalFree(security)


def names(h, path):
    probe = _open(path, "identity")
    try:
        return _api().GetFileType(h) == 1 and file_id(h) == file_id(probe)
    finally:
        close_handle(probe)


def check_inode(fd, path):
    from .claim import ClaimError
    try:
        h = handle(fd)
        if not owned_regular(h) or not names(h, path):
            raise ClaimError(f"descriptor does not name {path}")
        return os.fstat(fd)
    except (OSError, ClaimError) as exc:
        raise ClaimError(f"descriptor does not name {path}: {exc}") from None


def granted_access(h):
    access, status = W.ULONG(), IO_STATUS_BLOCK()
    result = _api().NtQueryInformationFile(h, C.byref(status), C.byref(access), C.sizeof(access), 8)
    if result < 0:
        from .claim import ClaimError
        raise ClaimError(f"cannot query claim access: NTSTATUS {result & 0xffffffff:#x}")
    return access.value


def check_access(fd):
    from .claim import ClaimError
    access = granted_access(handle(fd))
    if access & FILE_APPEND_DATA and not access & FILE_WRITE_DATA:
        raise ClaimError("claim refuses an append-only handle")
    if access & (FILE_READ_DATA | FILE_WRITE_DATA) != FILE_READ_DATA | FILE_WRITE_DATA:
        raise ClaimError("claim requires read and write data access")


def read_record(path, *, fd=None, size=16384, offset=0):
    from .claim import ClaimError
    h = None
    try:
        h = _open(path, "reader")
        if fd is not None and file_id(h) != file_id(handle(fd)):
            raise ClaimError(f"descriptor does not name {path}")
        if offset and not _api().SetFilePointerEx(h, offset, None, 0):
            raise _error("reader seek")
        data, count = C.create_string_buffer(size), W.DWORD()
        if not _api().ReadFile(h, data, size, C.byref(count), None):
            raise _error("ReadFile")
        return data.raw[:count.value]
    except OSError as exc:
        raise ClaimError(f"holder unknown: {exc}") from None
    finally:
        if h is not None:
            close_handle(h)


_write = os.write


def publish(fd, data, nonce):
    from .claim import ClaimError
    os.ftruncate(fd, 0)
    if os.lseek(fd, 0, os.SEEK_SET) != 0:
        raise ClaimError("holder nonce seek failed")
    if _write(fd, data) != len(data):
        raise ClaimError("short holder line write")
    if os.lseek(fd, nonce, os.SEEK_SET) != nonce:
        raise ClaimError("holder nonce seek failed")


def offset(fd):
    return os.lseek(fd, 0, os.SEEK_CUR)


def process_info(pid, *, handle=None):
    api = _api()
    h = handle if handle is not None else api.OpenProcess(0x1000 | SYNCHRONIZE, False, pid)
    if not h:
        code = C.get_last_error()
        if code == 87:
            return None, True
        raise _error(f"cannot identify pid {pid}", code)
    try:
        creation, exit_time, kernel, user = (W.FILETIME() for _ in range(4))
        if not api.GetProcessTimes(h, C.byref(creation), C.byref(exit_time),
                                   C.byref(kernel), C.byref(user)):
            raise _error(f"cannot identify pid {pid}")
        started = f"win:{(creation.dwHighDateTime << 32) | creation.dwLowDateTime}"
        wait = api.WaitForSingleObject(h, 0)
        if wait not in (WAIT_OBJECT_0, WAIT_TIMEOUT):
            raise _error(f"cannot wait for pid {pid}")
        return started, wait == WAIT_OBJECT_0
    finally:
        if handle is None:
            close_handle(h)


def process_started(pid):
    from .claim import ClaimError
    started, _ = process_info(pid)
    if started is None:
        raise ClaimError(f"cannot identify pid {pid}")
    return started


def command():
    return _api().GetCommandLineW()


def serial_target(port):
    name = re.sub(r"^\\\\\.\\", "", port)
    data = C.create_unicode_buffer(32768)
    if not _api().QueryDosDeviceW(name, data, len(data)):
        raise _error(f"cannot resolve {port}")
    return data.value


def restart_manager(path):
    api = _rm_api()
    session, key = W.DWORD(), C.create_unicode_buffer(33)
    code = api.RmStartSession(C.byref(session), 0, key)
    if code:
        raise _error("RmStartSession", code)
    try:
        files = (W.LPCWSTR * 1)(os.path.abspath(path))
        code = api.RmRegisterResources(session, 1, files, 0, None, 0, None)
        if code:
            raise _error("RmRegisterResources", code)
        needed, count, reasons = W.UINT(), W.UINT(), W.DWORD()
        data = None
        while True:
            code = api.RmGetList(session, C.byref(needed), C.byref(count), data, C.byref(reasons))
            if code == 0:
                return [(row.Process.dwProcessId, row.strAppName) for row in
                        (data[:count.value] if data is not None else [])]
            if code != 234:
                raise _error("RmGetList", code)
            count.value = needed.value
            data = (RM_PROCESS_INFO * count.value)()
    finally:
        code = api.RmEndSession(session)
        if code:
            raise _error("RmEndSession", code)


def busy_claim():
    from . import claim
    try:
        published = claim.holder(None)
        description = str(published)
        try:
            started, exited = process_info(published.pid)
            if exited or started != published.started:
                description += " (publisher exited; claim remains busy; current holder unknown)"
        except claim.ClaimError:
            pass
    except claim.ClaimError:
        description = "holder unknown"
    try:
        rows = restart_manager(claim.claim_path())
        listing = ", ".join(f"pid {pid} {name}" for pid, name in rows) or "no processes reported"
        description += "; rig.claim open in " + listing
    except (OSError, claim.ClaimError):
        description += "; rig.claim open process list unavailable"
    return "rig claimed: " + description


def gate_busy(*, admission=False):
    from . import claim
    if not admission:
        try:
            h = _open(claim.mutex_path(), "admission")
        except OSError as exc:
            if exc.winerror != ERROR_SHARING_VIOLATION:
                return True, "cleanup in progress (holder unknown)"
        else:
            close_handle(h)
            return False, "admission in progress (holder unknown)"
    try:
        published = claim.holder(None, cleanup=True)
        started, exited = process_info(published.pid)
        if not exited and started == published.started:
            return True, f"cleanup in progress (pids {published.pid})"
    except claim.ClaimError:
        pass
    return True, "cleanup in progress (holder unknown)"


def shared(fd=None):
    from . import claim
    if fd is not None:
        claim._close(claim._fd(fd))
        raise claim.ClaimError("admission requires its own share-mode open on Windows")
    try:
        fd = open_role(claim.mutex_path(), "admission")
    except OSError as exc:
        if exc.winerror != ERROR_SHARING_VIOLATION:
            raise
        raise claim.ClaimBusy(gate_busy(admission=True)[1]) from None
    try:
        claim._check_inode(fd, claim.mutex_path())
        claim._event("admission_acquired", fd, 0)
        return fd
    except BaseException:
        claim._close(fd)
        raise


def mutex(fd=None, *, timeout, pid=None, **discovery):
    from . import claim, coexist
    if fd is not None:
        claim._close(claim._fd(fd))
        raise claim.ClaimError("cleanup requires its own share-mode open on Windows")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = open_role(claim.mutex_path(), "cleanup")
        except OSError as exc:
            if exc.winerror != ERROR_SHARING_VIOLATION:
                raise
            exclusive, message = discovery.get('busy', gate_busy)()
            if exclusive or time.monotonic() >= deadline:
                raise claim.ClaimBusy(message) from None
        else:
            try:
                message = launch_admissions()
            except BaseException:
                claim._close(fd)
                raise
            if message is None:
                break
            claim._close(fd)
            if time.monotonic() >= deadline:
                raise claim.ClaimBusy(message)
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    try:
        claim._check_inode(fd, claim.mutex_path())
        nonce = claim._publish(fd, pid=pid, cleanup=True)
        set_role(fd, "cleanup")
        claim._event("cleanup_acquired", fd, nonce)
        coexist.checkpoint("after_mutex")
        return fd
    except BaseException:
        try:
            os.ftruncate(fd, 0)
            claim._event("cleanup_closed", fd, offset(fd), failed=True)
        finally:
            claim._close(fd)
        raise


def verify_cleanup(fd):
    from . import claim
    claim._check_inode(fd, claim.mutex_path())
    entry = entries().get(fd)
    if entry is None or entry.role != "cleanup":
        raise claim.ClaimError("cleanup mutex belongs to another description")
    try:
        h = _open(claim.mutex_path(), "admission")
    except OSError as exc:
        if exc.winerror != ERROR_SHARING_VIOLATION:
            raise claim.ClaimError(f"cannot verify cleanup mutex: {exc}") from exc
    else:
        close_handle(h)
        raise claim.ClaimError("cleanup mutex is not held exclusively")
    if offset(fd) != claim.holder(fd, cleanup=True).nonce:
        raise claim.ClaimError("cleanup mutex belongs to another description")
    return fd


def acquire_open(path):
    from . import claim
    try:
        return open_role(path, "holder")
    except OSError as exc:
        if exc.winerror != ERROR_SHARING_VIOLATION:
            raise
        raise claim.ClaimBusy(busy_claim()) from None


def duplicate(h, *, inheritable=False, access=None):
    api = _api()
    copied = W.HANDLE()
    process = api.GetCurrentProcess()
    if not api.DuplicateHandle(process, h, process, C.byref(copied),
                               0 if access is None else access, inheritable,
                               2 if access is None else 0):
        raise _error("DuplicateHandle")
    return copied.value


def token_handle(token):
    from .claim import ClaimError
    try:
        if not re.fullmatch(r"[0-9]+", token):
            raise ValueError
        h = int(token)
        if not 0 < h < 1 << (C.sizeof(W.HANDLE) * 8):
            raise ValueError
        return h
    except ValueError:
        raise ClaimError("missing inherited station claim handle") from None


def managed(h):
    api = _api()
    standards = {api.GetStdHandle(n & 0xffffffff) for n in (-10, -11, -12)}
    for fd in range(3):
        try:
            standards.add(handle(fd))
        except OSError:
            pass
    return h in standards or any(entry.handle == h for entry in entries().values())


_received_record = None
_received_loaded = False
_received_admission = None
_extras = {}


def received_record():
    global _received_record, _received_loaded
    from . import winproc
    with lock:
        if not _received_loaded:
            _received_loaded = True
            _received_record = winproc.read_launch()
        return _received_record


def take_handle(name):
    record = received_record()
    return record['handles'].pop(name, None) if record is not None else None


def received_admission(descriptor):
    global _received_admission
    from . import claim, winproc
    with lock:
        if _received_admission is None:
            with winproc._own() as own:
                record = received_record()
                if record is None or record['phase'] != 'resuming':
                    raise claim.ClaimError('launch record is not admitting')
                _received_admission = claim.Admission(descriptor, None, record=own)
        return _received_admission


def extra(name, *, optional=False):
    from . import claim
    with lock:
        if name == 'claim':
            raise claim.ClaimError('claim is not a launch extra')
        if name in _extras:
            fd, original = _extras[name]
            if fd is None or entries().get(fd) != original:
                raise claim.ClaimError(f'launch extra {name} is no longer live')
            return fd
        value = take_handle(name)
        if value is None and optional:
            return None
        if value is None:
            raise claim.ClaimError(f'missing launch extra {name}')
        _extras[name] = (None, None)
        h = token_handle(str(value))
        if managed(h):
            raise claim.ClaimError(f'launch extra {name} is already managed')
        fd = _crt(h, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT, role=f'extra:{name}')
        _extras[name] = (fd, entries()[fd])
        return fd


def launch_admissions():
    from . import claim, winproc
    try:
        records, unresolved = winproc.live_launches()
        admitting = [record['child'] for record in records
                     if record['role'] not in ('cleanup', 'inspector', 'recorder') and
                     (record['phase'] in ('created', 'resuming') or
                      record['child']['pid'] in unresolved)]
        if admitting:
            listing = ', '.join(f"{child['pid']} ({child['image']}" +
                                ('; cannot identify)' if child['pid'] in unresolved else ')')
                                for child in admitting)
            return f'admission in progress (pids {listing})'
    except (OSError, claim.ClaimError, ValueError, KeyError, TypeError) as exc:
        return f'admission in progress (unreadable launch records: {exc})'
    return None


def _launch_admission(descriptor):
    from . import claim, winproc
    admission = descriptor if isinstance(descriptor, claim.Admission) else _received_admission
    if admission is not None and admission.record is not None and admission.active:
        record = winproc.read_launch(admission.record)
        if record is not None and record['phase'] == 'resuming':
            # A record has no gate handle; bridge the cleanup open/scan boundary.
            return claim._shared()
    for fd, entry in entries().items():
        if entry.role == 'admission':
            claim._check_inode(fd, claim.mutex_path())
            # Reused fd and handle numbers cannot prove the original share mode.
            return claim._shared()
    raise claim.ClaimError('launch requires a live share-mode admission')


def spawn(descriptor, argv, *, role, handles, **options):
    from . import claim, winproc
    if role not in ('python', 'supervisor', 'one-shot', 'cleanup', 'inspector', 'recorder'):
        raise claim.ClaimError('claim.spawn requires a Python launch role')
    argv = list(map(os.fspath, argv))
    if not argv or os.path.normcase(os.path.abspath(argv[0])) not in {
            os.path.normcase(os.path.abspath(path)) for path in (sys.executable, sys._base_executable)}:
        raise claim.ClaimError('claim.spawn requires the Python interpreter')
    if options.pop('shell', False) or 'executable' in options or options.pop('pass_fds', ()):
        raise claim.ClaimError('claim.spawn requires a direct interpreter launch')
    if not options.pop('close_fds', True):
        raise claim.ClaimError('claim.spawn requires close_fds=True')
    if 'claim' in handles or any(not isinstance(name, str) or not name or
                               Path(name).name != name for name in handles):
        raise claim.ClaimError('invalid launch extra name')
    own_session = options.pop('own_session', False)
    own_session = options.pop('start_new_session', False) or own_session
    flags = options.pop('creationflags', 0) | winproc.creation_flags(
        role, python=True, own_session=own_session)
    bridge = child = process = None
    with lock:
        try:
            if role not in ('cleanup', 'inspector', 'recorder'):
                bridge = _launch_admission(descriptor)
            source = descriptor.claim if isinstance(descriptor, claim.Admission) else descriptor
            source = claim.inherited(source) if role != 'recorder' else None
            environment = options.pop('env', None)
            environment = dict(os.environ if environment is None else environment)
            if source is None:
                environment.pop('RIG_CLAIMED', None)
            else:
                environment['RIG_CLAIMED'] = str(claim.claim_path())
            process = winproc.python_suspended(argv[1:], creationflags=flags,
                                               env=environment, **options)
            from .winproc import _inspect_checkpoint as checkpoint
            copied = duplicate(int(process._handle))
            try:
                child = winproc._pin(copied, process.pid, winproc.END)
            except BaseException:
                close_handle(copied)
                raise
            from . import coexist
            coexist.event('launch_created', child_pid=child.pid, child_started=child.started)
            checkpoint('launch_after_creation')
            winproc.create_launch(child, role=role)
            checkpoint('launch_after_record')
            pushed = {}
            sources = ({'claim': source.fd} if source is not None else {}) | dict(handles)
            for name, fd in sources.items():
                pushed[name] = winproc.push_handle(child, handle(fd))
                winproc.advance_launch(child, 'created', handles=pushed)
                checkpoint(f'launch_after_push_{name}')
            winproc.advance_launch(child, 'resuming')
            checkpoint('launch_after_resuming')
            winproc.resume(child)
            checkpoint('launch_after_resume')
            return claim.Launch(process, child)
        except BaseException:
            try:
                if process is not None:
                    process.terminate()
                    process.wait(timeout=winproc.KILL_WAIT_S)
            finally:
                if child is not None:
                    child.close()
                if process is not None:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()
            raise
        finally:
            if bridge is not None:
                claim.Admission(claim.ClaimDescriptor(-1), bridge).failed()
