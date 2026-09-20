# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The audio callback has to get in, and the load it has to get in past is nasty.

`core.gil` exists because of a CPython handover rule that is easy to state and
very easy to disbelieve: a thread is forced off the GIL only once a waiter has sat
out a whole switch interval *with the GIL never changing hands*, so a thread that
drops it now and then -- which is every numpy call over a few hundred elements --
rearms nothing and keeps the waiter out indefinitely. The load below is the
smallest thing that shows it, and it is the decode's own shape: mostly bytecode,
one small numpy call per thousand instructions.

The margins here are wide on purpose, and both timing arms are `realtime`. What
is being held is the gap between a starved waiter and a fed one, which measures
two orders of magnitude on this machine; a test that pinned either arm to a tight
number would be measuring the scheduler's mood instead.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hfmodem.core import gil

BLOCK_S = 128 / 48000
BURST_S = 1.5


def _waiter(stop: threading.Event, lags: list[float]) -> None:
    """The callback, without the audio: wake on the block grid, note the wait."""
    due = time.monotonic() + BLOCK_S
    while not stop.is_set():
        rest = due - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        now = time.monotonic()
        lags.append(now - due)
        due = max(now, due) + BLOCK_S


def _release(a: np.ndarray) -> np.ndarray:
    """A python frame around the numpy call, because the breather watches for
    python frames. A kernel of pure bytecode and C calls raises none and cannot
    be breathed -- see `core.gil`. A decode raises about seven hundred thousand
    a second."""
    return np.abs(a)


def _starve(seconds: float) -> None:
    """Bytecode with a GIL-releasing numpy call every thousand instructions."""
    a = np.zeros(1000, np.float32)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for _ in range(50):
            x = 0
            for _ in range(1000):
                x += 1
            _release(a)


def _worst_wait(breathing: bool) -> float:
    lags: list[float] = []
    stop = threading.Event()
    th = threading.Thread(target=_waiter, args=(stop, lags), daemon=True)
    th.start()
    time.sleep(0.2)
    lags.clear()
    if breathing:
        with gil.breathing():
            _starve(BURST_S)
    else:
        _starve(BURST_S)
    stop.set()
    th.join(timeout=2.0)
    assert lags, "the waiter never ran at all"
    return max(lags)


@pytest.mark.realtime
def test_a_waiting_thread_is_starved_by_a_decode_shaped_load():
    worst = _worst_wait(breathing=False)
    assert worst > 0.025, (
        f"the load meant to starve a waiter let it in within {worst * 1e3:.1f} ms; "
        "either CPython's handover changed or the numpy call stopped releasing")


@pytest.mark.realtime
def test_breathing_lets_the_waiting_thread_in_on_a_clock():
    worst = _worst_wait(breathing=True)
    assert worst < 0.015, (
        f"breathing left a waiter out for {worst * 1e3:.1f} ms against a "
        f"{gil.PERIOD * 1e3:.0f} ms period")


def test_breathing_gives_its_monitoring_slot_back():
    import sys

    with gil.breathing():
        assert gil._tool in gil._TOOL_IDS
        taken = gil._tool
    assert gil._tool is None
    sys.monitoring.use_tool_id(taken, "check")       # free, or this raises
    sys.monitoring.free_tool_id(taken)


def test_breathing_nests_without_claiming_two_slots():
    with gil.breathing():
        outer = gil._tool
        with gil.breathing():
            assert gil._tool == outer
        assert gil._tool == outer            # the inner exit takes nothing down
    assert gil._tool is None
