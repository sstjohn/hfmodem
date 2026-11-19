# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Transmit eligibility keeps its lock wait through a forward wall-clock step."""
import threading
import time

from hfmodem.brubru.access import Access
from hfmodem.tests.clock_steps import SteppedLock


def test_a_forward_step_does_not_deny_eligibility_to_a_briefly_held_lock():
    access = Access()
    lock = threading.RLock()
    access._lock = SteppedLock(lock)
    held = threading.Event()

    def hold():
        with lock:
            held.set()
            time.sleep(0.01)
    threading.Thread(target=hold).start()
    held.wait()
    access.eligible(fresh_after=5.0)
    assert access._fresh_after == 5.0, "the 20 ms lock wait gave up at once"
