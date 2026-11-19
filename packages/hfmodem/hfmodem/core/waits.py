# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Timed waits that end on the monotonic clock.

CPython built without sem_clockwait (the python-build-standalone builds uv
installs on Linux) parks every timed Event, Condition, Lock, join and Future
wait on the wall clock. A forward step ends such a wait early, as if it had
timed out; a backward step stretches it. `wait_until` keeps waiting through a
forward step. `sleep_until` and `Timer` are timed by `time.sleep` alone, so a
step either way neither ends nor stretches them: they bound keyed time.
"""
import threading
import time

POLL_S = 0.01


def wait_until(attempt, deadline):
    """True once `attempt(remaining)` succeeds; False only at the monotonic `deadline`.

    `attempt` is one timed wait that returns True on success, such as
    `event.wait` or `lambda r: lock.acquire(True, r)`.
    """
    while True:
        if attempt(max(0.0, deadline - time.monotonic())):
            return True
        if time.monotonic() >= deadline:
            return False


def join(thread, deadline):
    """True once `thread` has ended; False at the monotonic `deadline`."""
    return wait_until(lambda remaining: (thread.join(remaining), not thread.is_alive())[1], deadline)


def result(future, deadline):
    """`future.result()`, raising TimeoutError only at the monotonic `deadline`."""
    while True:
        try:
            return future.result(max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            if future.done():
                return future.result(0)
            if time.monotonic() >= deadline:
                raise


def sleep_until(done, deadline):
    """True once `done()`; False at the monotonic `deadline`, however the wall clock steps."""
    while not done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(remaining, POLL_S))
    return True


class Timer(threading.Timer):
    """threading.Timer whose interval is timed by `sleep_until`."""

    def run(self):
        if not sleep_until(self.finished.is_set, time.monotonic() + self.interval):
            self.function(*self.args, **self.kwargs)
        self.finished.set()
