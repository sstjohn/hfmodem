# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The audio callback hands a block over without waiting on the thread reading it.

The session's reader sits in `_LiveInput.read`, woken once a block, and the first
Python call after a wake is due a `gil.breathing` nap. With `queue.Queue` that call
was `get`'s own `_qsize`, made under the queue's lock, so the reader napped holding
it and the next callback's `put` waited behind the nap -- and behind whatever kept
the reader from the interpreter after it. On the codec under `--recording
everything` that was a 3.5-7.4 ms callback, the input overflowed, and the session
lost its grid.
"""
import sys
import threading
import time

import numpy as np
import pytest

from hfmodem.core import gil
from hfmodem.shrike import onair

BLOCK = 128
NAP = 0.05


class _Times:
    def __init__(self, t):
        self.currentTime = self.inputBufferAdcTime = self.outputBufferDacTime = t


class _Status:
    input_overflow = input_underflow = output_underflow = priming_output = False


class _Card:
    blocksize, latency = BLOCK, 0.0

    def __init__(self, callback):
        self.callback, self.n = callback, 0

    def start(self):
        self.deliver()

    def deliver(self):
        self.callback(np.zeros((BLOCK, 1), np.float32), BLOCK, _Times(self.n / onair.FS), _Status())
        self.n += BLOCK

    def stop(self):
        pass

    close = stop


def _longest_put(monkeypatch, swap=None):
    """The longest delivery, each made as the reader starts a nap or after 20 ms of none.

    A reader blocked in the queue raises no Python calls and so takes no nap; the
    fallback is what wakes it.
    """
    class _FakeSd:
        @staticmethod
        def InputStream(callback=None, **_):
            return _Card(callback)

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd)
    live = onair._LiveInput("fake", None)       # breathes on this, the reading thread
    if swap is not None:
        swap.put(live._q.get_nowait())          # the block `start` delivered
        live._q = swap
    reader, napping, sleep = threading.get_ident(), threading.Event(), time.sleep

    def nap(seconds):
        if threading.get_ident() == reader and seconds == NAP:
            napping.set()
            sleep(seconds)
            napping.clear()
        else:
            sleep(seconds)

    monkeypatch.setattr(gil, "NAP", NAP)
    monkeypatch.setattr(gil, "PERIOD", 0.0)
    monkeypatch.setattr(time, "sleep", nap)
    held = []

    def callback():
        for _ in range(8):
            napping.wait(0.02)
            t = time.perf_counter()
            live._stream.deliver()
            held.append(time.perf_counter() - t)
            napping.clear()

    card = threading.Thread(target=callback, daemon=True)
    try:
        card.start()
        assert live.read(9 * BLOCK).size == 9 * BLOCK
        card.join(5)
    finally:
        monkeypatch.setattr(gil, "NAP", 0.0)
        live.close()
    assert len(held) == 8
    return max(held)


@pytest.mark.realtime
def test_a_reader_napping_in_read_never_holds_the_callback(monkeypatch):
    longest = _longest_put(monkeypatch)
    assert longest < NAP / 2, f"a callback waited {longest * 1e3:.1f} ms on the reader's nap"


@pytest.mark.realtime
def test_the_reader_used_to_hold_the_callback_through_queue_queue(monkeypatch):
    """The control: the same reader on `queue.Queue` keeps a put out for a nap."""
    import queue
    assert _longest_put(monkeypatch, swap=queue.Queue()) > NAP / 2
