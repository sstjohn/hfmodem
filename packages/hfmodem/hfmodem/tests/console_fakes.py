# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Synthetic console process lifecycles for offline tests on either platform."""
import os
import signal
import time

from consolelib import claim, station
from hfmodem.core.native import NativeChild


def install_windows_fixture_cleanup(monkeypatch, children):
    """Model recorded-tree teardown using only remembered synthetic children.

    Synthetic Popen objects have no Win32 handles or descendant processes. Keep
    root identities and claim-wide cleanup selection, then use the same portable
    child lifecycle as ordinary cleanup instead of querying the host process tree.
    """
    def end_recorded(roots, *, allow_cleanup=False, own=False):
        owned = {(child.pid, child.started)
                 for child in getattr(claim._local, 'children', ())} if own else set()
        for child in children:
            if (child.pid, child.started) in roots | owned:
                if claim.end_child(child) is False and not allow_cleanup:
                    return False
    monkeypatch.setattr(claim, '_end_recorded', end_recorded)


def windows_fixture_launch(descriptor, argv, *, role, handles=None, **options):
    """Keep the role/descriptor contract while starting only synthetic processes."""
    if role != 'recorder':
        claim.hardware.inherited(descriptor)
    for fd in (handles or {}).values():
        os.fstat(fd)
    return claim.hardware.Launch(station.subprocess.Popen(argv, **options))


class OfflineChild(NativeChild):
    """A remembered synthetic Popen with the same portable lifecycle methods."""

    def __init__(self, process):
        self.pid, self.started = process.pid, f'fake-{process.pid}'
        self._process, self._win_entry = process, None
        self.stdin, self.stdout, self.stderr = (getattr(process, name, None)
                                               for name in ('stdin', 'stdout', 'stderr'))

    def _alive(self, **kwargs):
        return self._process.poll() is None

    def _signal(self, sig):
        if not self.alive():
            return False
        if sig == getattr(signal, 'SIGKILL', None):
            self._process.kill()
        elif sig == signal.SIGINT and hasattr(self._process, 'interrupt'):
            self._process.interrupt()
        else:
            self._process.terminate()
        return True

    def interrupt(self):
        return self._signal(signal.SIGINT)

    def terminate(self):
        return self._signal(signal.SIGTERM)

    def kill(self):
        if self.alive():
            self._process.kill()

    def wait_gone(self, timeout):
        deadline = time.monotonic() + timeout
        while self.alive():
            if time.monotonic() >= deadline:
                return False
            time.sleep(.001)
        return True
