# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Wall-clock steps as a CPython built without sem_clockwait sees them.

Such a build parks timed waits on the wall clock: a forward step ends one early,
as if it had timed out, and a backward step stretches it.
"""
import threading

import pytest

PARK, JOIN = threading.Event.wait, threading.Thread.join
STEP_S = 0.005


def _capped(wait):
    return lambda self, timeout=None: wait(self, None if timeout is None else min(timeout, STEP_S))


def step_forward(patcher, *, joins=False):
    """Every timed Event wait, and with `joins` every timed join, ends within 5 ms."""
    patcher.setattr(threading.Event, 'wait', _capped(PARK))
    if joins:
        patcher.setattr(threading.Thread, 'join', _capped(JOIN))


def step_backward(patcher):
    """Every timed Event wait outlasts its timeout until set."""
    patcher.setattr(threading.Event, 'wait', lambda self, timeout=None: PARK(self))


@pytest.fixture
def forward_step(monkeypatch):
    step_forward(monkeypatch)


@pytest.fixture
def backward_step(monkeypatch):
    step_backward(monkeypatch)


class SteppedLock:
    """A lock whose timed acquire gives up within 5 ms, as one parked on a
    wall clock that steps forward does."""

    def __init__(self, lock):
        self._lock = lock

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking, timeout if timeout < 0 else min(timeout, STEP_S))

    def release(self):
        self._lock.release()

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()
