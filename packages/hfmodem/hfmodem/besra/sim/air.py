# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`StepAir`: a virtual half-duplex air joining two (or more) `BesraModem`s.

The medium is a synchronous scheduler, not a thread: when a modem transmits a
frame, the burst is queued and, on the next pump step, delivered (optionally
through a channel model) to every other modem's `receive_audio`. Between
deliveries the shared clock advances and every modem's `tick` fires, so ARQ
timers run in virtual time. `run` pumps until the medium goes quiet or a wall of
virtual time passes — deterministic, radioless, and fast.

This is the end-to-end integration harness: two modems, real modulate → channel
→ demodulate on every frame, ARQ on top.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np


class StepAir:
    """Synchronous, N-endpoint, single-stepped virtual air — a deterministic test
    substrate, driven by explicit pump steps rather than a clock thread. (Named to
    stay distinct from sabir's threaded, exactly-two-endpoint `VirtualAir`, which
    is a live modem backend, not a test harness.)"""

    def __init__(self, channel: Callable[[np.ndarray], np.ndarray] | None = None,
                 tick_step: float = 0.05) -> None:
        """`channel` maps a transmitted burst to what the receiver hears (AWGN,
        Watterson, or None for a clean wire). `tick_step` is the virtual-time
        granularity at which ARQ timers advance between bursts."""
        self._channel = channel
        self._tick_step = tick_step
        self._modems: list = []
        self._queue: deque = deque()
        self._clock = 0.0

    def join(self, modem) -> None:
        modem.audio_out = lambda samples, m=modem: self._queue.append((m, samples))
        self._modems.append(modem)

    def run(self, max_time: float = 120.0, quiet_ticks: int = 4) -> None:
        """Pump until the air has been quiet for `quiet_ticks` consecutive steps
        or `max_time` virtual seconds elapse. Delivering a burst may enqueue the
        reply, so a full handshake drains in one run."""
        idle = 0
        while self._clock < max_time:
            if self._queue:
                src, samples = self._queue.popleft()
                heard = self._channel(samples) if self._channel else samples
                for m in self._modems:
                    if m is not src:
                        m.receive_audio(heard)
                idle = 0
                continue
            # Nothing on the air: advance virtual time and run the timers.
            self._clock += self._tick_step
            for m in self._modems:
                m.tick(self._clock)
            idle = idle + 1 if not self._queue else 0
            if idle >= quiet_ticks:
                return

    @property
    def clock(self) -> float:
        return self._clock


class ThreadedAir:
    """A live half-duplex medium for threaded modems: a transmitted burst is
    relayed (optionally through a channel) into every other modem's `deliver_rx`
    queue, where that modem's own pump decodes it. No scheduler — real threads,
    real (wall-clock) ARQ timing. Used to drive `BesraModem`s behind the host
    server with no radio."""

    def __init__(self, channel: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self._channel = channel
        self._modems: list = []

    def join(self, modem) -> None:
        modem.audio_out = lambda samples, m=modem: self._relay(m, samples)
        self._modems.append(modem)

    def _relay(self, src, samples: np.ndarray) -> None:
        heard = self._channel(samples) if self._channel else samples
        for m in self._modems:
            if m is not src:
                m.deliver_rx(heard)
