# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Resource leases: close revokes admission, then drains admitted operations."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import threading
import time


class ResourceClosed(RuntimeError):
    pass


class LeaseTimeout(RuntimeError):
    pass


class OperationGate:
    def __init__(self):
        self._condition = threading.Condition()
        self._local = threading.local()
        self._closing = False
        self._active = 0

    @property
    def active(self):
        with self._condition:
            return self._active

    @contextmanager
    def lease(self):
        depth = getattr(self._local, "depth", 0)
        with self._condition:
            if self._closing and not depth:
                raise ResourceClosed("resource is closing or closed")
            self._active += 1
            self._local.depth = depth + 1
        try:
            yield
        finally:
            with self._condition:
                self._local.depth -= 1
                self._active -= 1
                self._condition.notify_all()

    def close(self, timeout):
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closing = True
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeaseTimeout("admitted resource operation did not drain")
                self._condition.wait(remaining)


def leased(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.operations.lease():
            return method(self, *args, **kwargs)
    return call
