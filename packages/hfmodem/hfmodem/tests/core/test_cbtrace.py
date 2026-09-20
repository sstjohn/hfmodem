# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The trace has to be the counter's own account, not a second opinion on it.

`lost_samples` in the sidecars is `rates.lost_step` summed over a session, and
the whole use of a trace is to say what the callback thread was doing when that
counter moved. A trace that computes loss even slightly differently sends a
reader looking at the wrong callbacks -- so it is held to the same arithmetic
here rather than to a plausible-looking total of its own.

The collector's alibi is the other assertion. `0 of 23 loss intervals had one
running inside them` is what took the garbage collector off the list on
2026-08-26, and it only means that if "inside" means overlapping the interval
the capture went missing in, rather than happening somewhere nearby.
"""
from __future__ import annotations

import numpy as np

from hfmodem.core import cbtrace, rates


class _T:
    def __init__(self, adc: float) -> None:
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = adc + 0.024
        self.currentTime = adc


def _stream(trace: cbtrace.CallbackTrace, ahead: np.ndarray,
            blocksize: int = 128, fs: int = rates.CARD_RATE_HZ) -> None:
    """Feed a series whose converter timestamps run `ahead` of the delivered count."""
    for i, a in enumerate(ahead):
        n0 = i * blocksize
        trace.stamp(i * 1e-3, i * 1e-3 + 5e-6, _T(n0 / fs + a), blocksize, n0, 0)


def test_the_trace_counts_loss_the_way_the_sidecars_do():
    fs, blocksize = rates.CARD_RATE_HZ, 128
    # Flat, then two jumps a reader would have to find, then flat again.
    ahead = np.zeros(400)
    ahead[120:] += 640 / fs
    ahead[300:] += 1024 / fs

    trace = cbtrace.CallbackTrace(blocksize=blocksize, fs=fs)
    _stream(trace, ahead)

    want, prev = 0, None
    for a in ahead:
        want += rates.lost_step(a, prev, blocksize, fs)
        prev = a
    assert trace.lost_samples() == want == 1664
    assert list(trace.losses()) == [120, 300]


def test_a_crystal_never_reaches_the_counter_or_the_trace():
    """A ramp of parts per million cannot accumulate into a half-block step, and
    a trace that reported one would libel every clean stream on the record."""
    fs = rates.CARD_RATE_HZ
    trace = cbtrace.CallbackTrace(fs=fs)
    _stream(trace, np.arange(2000) * 50e-6 * 128 / fs)     # 50 ppm, for a minute
    assert trace.lost_samples() == 0
    assert trace.losses().size == 0


def test_a_full_ring_stops_recording_and_says_how_much_it_missed():
    """Rather than wrapping: the beginning of a session is where the phase is."""
    trace = cbtrace.CallbackTrace(capacity=64)
    _stream(trace, np.zeros(100))
    assert trace.n == 64 and trace.dropped == 36


def test_the_collector_is_only_charged_for_the_intervals_it_ran_in():
    """A collection beside a loss is not a collection inside it."""
    fs = rates.CARD_RATE_HZ
    ahead = np.zeros(10)
    ahead[4:] += 640 / fs
    ahead[8:] += 640 / fs
    trace = cbtrace.CallbackTrace(fs=fs)
    _stream(trace, ahead)

    trace.gc_t[:2] = [3.5e-3, 6.5e-3]      # inside interval 3->4; between 6 and 7
    trace.gc_gen[:2] = [2, 0]
    trace.gc_n = 2

    said = trace._gc_verdict(trace.losses())
    assert "1 of 2 loss intervals had one running inside them" in said
    assert "gen2:1" in said
