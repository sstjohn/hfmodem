# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Locate hamlib and isolate a daemon from the caller's console signals."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess


def _runnable(path):
    return os.path.isfile(path) and (os.name == 'nt' or os.access(path, os.X_OK))


def executable_path(name, environ=None, *, required=True, prefer_local=False) -> str | None:
    """Find a native Hamlib tool, optionally preferring the POSIX local build."""
    environ = os.environ if environ is None else environ
    names = [name]
    if os.name == 'nt':
        extensions = environ.get('PATHEXT', '.COM;.EXE;.BAT;.CMD').split(';')
        names = [name + ext.lower() for ext in extensions if ext.lower() in {'.exe', '.com'}]
    looked = []

    def in_directory(directory):
        for executable in names:
            candidate = os.path.join(os.path.expanduser(directory), executable)
            looked.append(candidate)
            if _runnable(candidate):
                return candidate
        return None

    configured = environ.get('HAMLIB_BIN')
    if configured:
        if found := in_directory(configured):
            return found
    else:
        looked.append('HAMLIB_BIN (unset)')
    local = os.path.join('~', 'src', 'radio', 'hamlib', '_install', 'bin')
    local_first = prefer_local and os.name != 'nt'
    if local_first and not configured:
        if found := in_directory(local):
            return found
    path = environ.get('PATH', os.defpath)
    if os.name == 'nt':
        for directory in path.split(os.pathsep):
            if directory and (found := in_directory(directory.strip('"'))):
                return found
    elif found := shutil.which(name, path=path):
        return found
    looked.append(f'PATH ({name})')
    if not local_first:
        if found := in_directory(local):
            return found
    if not required:
        return None
    raise FileNotFoundError(name + ' not found; looked in ' + ', '.join(looked)
                            + '; install hamlib or export HAMLIB_BIN=<bin dir>')


def rigctld_path(environ=None) -> str:
    return executable_path('rigctld', environ)


def spawn_rigctld(args, **popen_kwargs):
    if os.name == 'nt':
        popen_kwargs['creationflags'] = (popen_kwargs.get('creationflags', 0)
                                         | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        popen_kwargs['start_new_session'] = True
    return subprocess.Popen([rigctld_path(), *args], **popen_kwargs)


def stop_rigctld(proc, timeout):
    if proc.poll() is not None:
        return proc.returncode
    try:
        if os.name == 'nt':
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except ProcessLookupError:
        return proc.wait()
    except OSError:
        proc.kill()
        return proc.wait()
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()
