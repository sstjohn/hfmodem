# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ctypes as C
from ctypes import wintypes as W
import os
import subprocess
import sys
import time

from hfmodem.core import winclaim, winproc


class _SID_AND_ATTRIBUTES(C.Structure):
    _fields_ = [('Sid', W.HANDLE), ('Attributes', W.DWORD)]


class _TOKEN_MANDATORY_LABEL(C.Structure):
    _fields_ = [('Label', _SID_AND_ATTRIBUTES)]


class _PROCESS_INFORMATION(C.Structure):
    _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE),
                ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]


def filtered_test(request, directory):
    api = winproc._api()
    advapi = C.WinDLL('advapi32', use_last_error=True)
    P, H, D = C.c_void_p, W.HANDLE, W.DWORD
    for name, args in (
        ('CreateRestrictedToken', [H, D, D, P, D, P, D, P, C.POINTER(H)]),
        ('ConvertStringSidToSidW', [W.LPCWSTR, C.POINTER(P)]),
        ('SetTokenInformation', [H, C.c_int, P, D]),
        ('GetSecurityDescriptorDacl', [P, C.POINTER(W.BOOL), C.POINTER(P), C.POINTER(W.BOOL)]),
        ('GetSecurityDescriptorSacl', [P, C.POINTER(W.BOOL), C.POINTER(P), C.POINTER(W.BOOL)]),
        ('CreateProcessAsUserW', [H, W.LPCWSTR, W.LPWSTR, P, P, W.BOOL, D, P,
                                W.LPCWSTR, P,
                                C.POINTER(_PROCESS_INFORMATION)]),
    ):
        fn = getattr(advapi, name)
        fn.argtypes, fn.restype = args, W.BOOL
    advapi.SetNamedSecurityInfoW.argtypes = [W.LPWSTR, C.c_int, D, P, P, P, P]
    advapi.SetNamedSecurityInfoW.restype = D
    kernel = C.WinDLL('kernel32', use_last_error=True)
    for name, result, args in (
        ('InitializeProcThreadAttributeList', W.BOOL, [P, D, D, C.POINTER(C.c_size_t)]),
        ('UpdateProcThreadAttribute', W.BOOL, [P, D, C.c_size_t, P, C.c_size_t, P, P]),
        ('DeleteProcThreadAttributeList', None, [P]),
        ('GetExitCodeProcess', W.BOOL, [H, C.POINTER(D)]),
    ):
        fn = getattr(kernel, name)
        fn.argtypes, fn.restype = args, result
    class Startup(C.Structure):
        _fields_ = [('cb', D), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR),
                    ('title', W.LPWSTR)] + [(name, D) for name in
                    ('x', 'y', 'xsize', 'ysize', 'xchars', 'ychars', 'fill', 'flags')] + [
                    ('show', W.WORD), ('reserved_size', W.WORD), ('reserved2', P),
                    ('stdin', H), ('stdout', H), ('stderr', H)]
    class StartupEx(C.Structure):
        _fields_ = [('startup', Startup), ('attributes', P)]
    token, restricted, sid, security, medium, text = H(), H(), P(), P(), P(), W.LPWSTR()
    process = _PROCESS_INFORMATION()
    attributes, fds = None, []
    def check(ok):
        if not ok:
            raise C.WinError(C.get_last_error())
    try:
        check(api.OpenProcessToken(api.GetCurrentProcess(), 0x1ab, C.byref(token)))
        check(advapi.CreateRestrictedToken(token, 0x1 | 0x4, 0, None, 0, None, 0,
                                          None, C.byref(restricted)))
        check(advapi.ConvertStringSidToSidW('S-1-16-8192', C.byref(sid)))
        label = _TOKEN_MANDATORY_LABEL(_SID_AND_ATTRIBUTES(sid.value, 0x20))
        check(advapi.SetTokenInformation(restricted, 25, C.byref(label), C.sizeof(label) + C.sizeof(H)))
        user = winclaim._token_sid(restricted, 1)
        owner = (H * 1)(user.value)
        check(advapi.SetTokenInformation(restricted, 4, owner, C.sizeof(owner)))
        check(api.ConvertSidToStringSidW(user, C.byref(text)))
        # The token's default DACL protects the processes and threads the child creates,
        # so it grants generic rights, not file rights, as lua.py does.
        check(api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f'D:(A;;GA;;;{text.value})(A;;GA;;;SY)', 1, C.byref(security), None))
        present, defaulted, dacl = W.BOOL(), W.BOOL(), P()
        check(advapi.GetSecurityDescriptorDacl(security, C.byref(present), C.byref(dacl), C.byref(defaulted)))
        default_dacl = (H * 1)(dacl.value)
        check(advapi.SetTokenInformation(restricted, 6, default_dacl, C.sizeof(default_dacl)))

        # The elevated creator may grant only deny-only Administrators; name the user.
        check(api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f'D:(A;OICI;FA;;;{text.value})(A;OICI;FA;;;SY)S:(ML;OICI;NW;;;ME)',
            1, C.byref(medium), None))
        directory_dacl = P()
        check(advapi.GetSecurityDescriptorDacl(medium, C.byref(present),
                                             C.byref(directory_dacl), C.byref(defaulted)))
        sacl = P()
        check(advapi.GetSecurityDescriptorSacl(medium, C.byref(present), C.byref(sacl),
                                             C.byref(defaulted)))
        def prepare(path):
            path.mkdir()
            error = advapi.SetNamedSecurityInfoW(str(path), 1, 0x4 | 0x10,
                                                 None, None, directory_dacl, sacl)
            if error:
                raise C.WinError(error)

        prepare(directory)
        # Pytest replaces --basetemp; its replacement inherits from directory.
        basetemp = directory / 'pytest'
        prepare(basetemp)
        environment = dict(os.environ, __PYVENV_LAUNCHER__=sys.executable,
                           PYTHONPATH=os.pathsep.join(filter(None, sys.path)),
                           HFMODEM_FILTERED_TEST='1', HFMODEM_STATE_DIR=str(basetemp / 'state'),
                           TEMP=str(directory), TMP=str(directory), PYTHONIOENCODING='utf-8')
        block = C.create_unicode_buffer('\0'.join(
            f'{key}={value}' for key, value in sorted(environment.items(), key=lambda row: row[0].upper())) + '\0')
        code = '''
import sys, pytest
sys.exit(pytest.main(['-q', '-rP', '-p', 'no:cacheprovider', '--basetemp', *sys.argv[1:]]))
'''
        target = str(request.node.path) + '::' + request.node.nodeid.split('::', 1)[1]
        config = ['-c', str(request.config.inipath)] if request.config.inipath else []
        command = C.create_unicode_buffer(subprocess.list2cmdline(
            [sys._base_executable, '-c', code, str(basetemp), *config, target]))
        read, write = os.pipe()
        fds.extend((read, write))
        stdin = os.open(os.devnull, os.O_RDONLY)
        fds.append(stdin)
        handles = (H * 2)(winclaim.handle(stdin), winclaim.handle(write))
        for handle in handles:
            os.set_handle_inheritable(handle, True)
        size = C.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 1, 0, C.byref(size))
        storage = C.create_string_buffer(size.value)
        check(kernel.InitializeProcThreadAttributeList(storage, 1, 0, C.byref(size)))
        attributes = storage
        check(kernel.UpdateProcThreadAttribute(attributes, 0, 0x20002, handles,
                                               C.sizeof(handles), None, None))
        startup = StartupEx(Startup(cb=C.sizeof(StartupEx), flags=0x100,
                                    stdin=handles[0], stdout=handles[1], stderr=handles[1]),
                            C.cast(attributes, P))
        check(advapi.CreateProcessAsUserW(restricted, None, command, None, None, True,
                                         0x80000 | 0x400, block, str(request.config.rootpath),
                                         C.byref(startup), C.byref(process)))
        os.close(write)
        fds.remove(write)
        report = bytearray()
        deadline = time.monotonic() + 120
        while True:
            available = winproc.pipe_available(read)
            if available:
                report.extend(os.read(read, min(available, 65536)))
            elif winproc._wait(process.hProcess, .01):
                if not winproc.pipe_available(read):
                    break
            if time.monotonic() >= deadline:
                raise AssertionError('filtered test exceeded 120 seconds:\n' +
                                     report.decode('utf-8', errors='replace'))
        result = D()
        check(kernel.GetExitCodeProcess(process.hProcess, C.byref(result)))
        report = report.decode('utf-8', errors='replace')
        assert result.value == 0, report
        print(report)
    finally:
        if process.hProcess and not winproc._wait(process.hProcess, 0):
            check(api.TerminateProcess(process.hProcess, 1))
            assert winproc._wait(process.hProcess, 5), 'filtered test did not exit'
        if attributes is not None:
            kernel.DeleteProcThreadAttributeList(attributes)
        for fd in fds:
            os.close(fd)
        for handle in (process.hThread, process.hProcess, restricted, token):
            if handle:
                winclaim.close_handle(handle)
        for memory in (sid, security, medium, C.cast(text, P)):
            if memory:
                api.LocalFree(memory)
