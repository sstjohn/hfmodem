# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The launch claim, faked for modem tests that run in this process.

Admission and the native supervisor are proved in `tests/core` and
`tests/coexist`; these stand-ins keep the order a modem must follow observable
without a claim descriptor on fd 9.
"""
from __future__ import annotations

import subprocess

from hfmodem.core import claim, native


class Child:
    """A `NativeChild`-shaped handle on a real subprocess, without ps."""

    def __init__(self, process):
        self._process = process
        self.pid = process.pid
        self.stdin, self.stdout, self.stderr = process.stdin, process.stdout, process.stderr

    def alive(self):
        return self._process.poll() is None

    def terminate(self):
        self._process.terminate()

    def kill(self):
        self._process.kill()

    def wait_gone(self, timeout):
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True


def spawn(argv, *, claim_fd, cleanup_fd=None, long_lived=False, device=None,
          own_session=False):
    assert claim_fd == 9 and cleanup_fd is None
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=own_session)
    except OSError as exc:
        raise native.NativeSpawnError(str(exc)) from exc
    return Child(process)


class Admission:
    def __init__(self, events):
        self.events, self.held = events, True

    def opened(self):
        assert self.held, "acknowledged twice, or after the hold was dropped"
        self.held = False
        self.events.append("opened")

    def failed(self):
        self.held = False
        self.events.append("failed")


def install(monkeypatch, events=None, *, spawner=spawn):
    """Admit every launch and run native children as plain subprocesses; returns
    the admission events, in order, appended to `events` when it is given."""
    events = [] if events is None else events

    def inherited(fd, *, launch=False):
        assert (fd, launch) == (9, True)
        events.append("inherited")
        return Admission(events)

    monkeypatch.setattr(claim, "inherited", inherited)
    monkeypatch.setattr(native, "spawn_native", spawner)
    return events
