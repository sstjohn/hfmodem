# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The POSIX-only standard modules, bound so that Windows can import their users.

The claim protocol, serial PTT and CAT, and terminal handling are built on
`fcntl`, `termios`, `tty`, `pty` and `pwd`, none of which Windows has. A module
binds them with `fcntl = posix.module("fcntl")`. On POSIX that is the real
module, so nothing there changes. On Windows it is a stand-in that imports on
first use, so the module imports and only a call that actually reaches one of
these raises the `ModuleNotFoundError` the import would have.

`SIGHUP` maps to Ctrl-Break on Windows. Console close uses the stop channel
and SIGTERM. Windows lacks the interval timer the unwind deadman runs on;
a prestarted Windows worker
provides the same remaining-time, reschedule and cancel seam.
"""
from __future__ import annotations

import importlib
import os
import signal
import threading
from types import ModuleType

from .windeadman import Watchdog


_windows_deadman = Watchdog() if os.name == "nt" else None

SIGHUP = signal.SIGHUP if os.name == "posix" else signal.SIGBREAK

_stop_callbacks = ()
_stop_callbacks_lock = threading.Lock()


def on_stop(callback):
    # POSIX SIGTERM already interrupts blocking calls; Windows needs an explicit wakeup.
    if os.name != "nt":
        return
    global _stop_callbacks
    with _stop_callbacks_lock:
        _stop_callbacks = (*_stop_callbacks, callback)


def _run_on_stop():
    with _stop_callbacks_lock:
        callbacks = _stop_callbacks
    for callback in callbacks:
        try:
            callback()
        except Exception:
            # One failed wakeup must not prevent the remaining wakeups or SIGTERM.
            pass


def itimer_left() -> float:
    return (signal.getitimer(signal.ITIMER_REAL)[0] if os.name == "posix"
            else _windows_deadman.left())


def cancel_itimer() -> None:
    if os.name == "posix":
        signal.setitimer(signal.ITIMER_REAL, 0)
    else:
        _windows_deadman.cancel()


def set_itimer(seconds: float) -> None:
    """Replace the unwind deadline without changing its termination message."""
    if os.name == "posix":
        signal.setitimer(signal.ITIMER_REAL, seconds)
    else:
        _windows_deadman.reschedule(seconds)


class _Deferred:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr):
        return getattr(importlib.import_module(self._name), attr)

    def __setattr__(self, attr, value) -> None:
        setattr(importlib.import_module(self._name), attr, value)

    def __delattr__(self, attr) -> None:
        delattr(importlib.import_module(self._name), attr)

    def __repr__(self) -> str:
        return f"<{self._name}, not on this platform>"


def module(name: str) -> ModuleType:
    if os.name == "posix":
        return importlib.import_module(name)
    return _Deferred(name)  # type: ignore[return-value]
