# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Card-clock channel assessment and byte-valued p-persistence.

Broadband occupancy present at startup is indistinguishable from a noisy quiet
channel when it raises the band and guards equally; it qualifies as the floor.
The operator controls the station and its channel-access parameters.
"""
from __future__ import annotations

from collections import deque
import random
import threading

import numpy as np

from hfmodem.core.rates import CARD_RATE_HZ as RATE


class Access:
    WINDOW = RATE // 10
    HANG = RATE // 5

    def __init__(self, persistence=63, slot_ms=100, *, draw=None):
        self.persistence = persistence
        self.slot_ms = slot_ms
        self.draw = draw or (lambda: random.randrange(256))
        self._lock = threading.RLock()
        self._history = deque(maxlen=300)
        self._qualifying = []
        self.floor = None
        self.stale = False
        self.held = False
        self.transmitting = False
        self._masked_until = 0
        self._busy_until = 0
        self._valid_after = 0
        self._channel_after = 0
        self._pending = np.empty(0)
        self._next = None
        self.evaluated = 0
        self.fault_serial = 0
        self._slot = None
        self._fresh_after = -self.WINDOW
        frequencies = np.fft.rfftfreq(self.WINDOW, 1 / RATE)
        self._band = (frequencies >= 1260) & (frequencies < 1740)
        self._guards = (((frequencies >= 300) & (frequencies < 1200)) |
                        ((frequencies >= 1800) & (frequencies < 2700)))
        self._window = np.hanning(self.WINDOW)

    def fault(self, index, serial):
        with self._lock:
            self.fault_serial = serial
            self._pending = np.empty(0)
            self._next = None
            self._valid_after = index + RATE
            self._slot = None
            self._qualifying.clear()

    def retune(self, index, serial):
        """Discard assessment learned on the previous dial."""
        with self._lock:
            self.fault(index, serial)
            self._history.clear()
            self.floor = None
            self.stale = False
            self._fresh_after = max(self._fresh_after, index)
            self._channel_after = max(self._channel_after, index)

    def require_fresh(self, index):
        with self._lock:
            self._fresh_after = max(self._fresh_after, index)
            self._slot = None

    def tx(self, active, index):
        with self._lock:
            self.transmitting = active
            if not active:
                self._masked_until = index + self.HANG
                self.require_fresh(self._masked_until)

    def hold(self, active):
        with self._lock:
            self.held = active
            if active:
                self._slot = None

    def preamble(self, index):
        with self._lock:
            self._busy_until = max(self._busy_until, index + self.HANG)

    def feed(self, at, samples):
        with self._lock:
            if at < self._channel_after:
                skip = min(len(samples), self._channel_after - at)
                at += skip
                samples = samples[skip:]
                if not len(samples):
                    return
            if self._next is not None and at != self._next:
                self.fault(at, self.fault_serial)
            self._next = at + len(samples)
            origin = at - len(self._pending)
            data = np.concatenate((self._pending, samples))
            used = 0
            while used + self.WINDOW <= len(data):
                end = origin + used + self.WINDOW
                self._evaluate(data[used:used + self.WINDOW], end)
                used += self.WINDOW
            self._pending = data[used:].copy()

    def _evaluate(self, samples, end):
        self.evaluated = end
        while self._history and self._history[0][0] < end - 30 * RATE:
            self._history.popleft()
        self._qualifying = [(at, power) for at, power in self._qualifying
                            if at >= end - 30 * RATE]
        if not self._history and self.floor is not None:
            self.stale = True
        if self.held or self.transmitting or end - self.WINDOW < self._masked_until:
            return
        power = np.abs(np.fft.rfft(samples * self._window)) ** 2
        band = float(power[self._band].mean())
        guard = float(power[self._guards].mean())
        shaped = band > 10 ** 0.6 * max(guard, 1e-30)
        elevated = self.floor is not None and band > 10 ** 0.6 * max(self.floor, 1e-30)
        if shaped or elevated:
            self._busy_until = end + self.HANG
            self._slot = None
            self._qualifying.clear()
            return
        if self.floor is None or self.stale:
            self._qualifying.append((end, band))
            if len(self._qualifying) < 20:
                return
            self._history.extend(self._qualifying)
            self._qualifying.clear()
            self.stale = False
        else:
            self._history.append((end, band))
        self.floor = float(np.percentile([p for _, p in self._history], 10))

    def clear(self, fault_serial=None):
        with self._lock:
            return (self.floor is not None and not self.held and not self.transmitting
                    and self.evaluated >= max(self._busy_until, self._valid_after,
                                              self._masked_until,
                                              self._fresh_after + self.WINDOW)
                    and (fault_serial is None or fault_serial == self.fault_serial))

    def eligible(self, fault_serial=None, *, fresh_after=None, transmitting=False):
        """A short, nonblocking callback, also used after arbiter contention."""
        if not self._lock.acquire(timeout=0.02):
            return False
        try:
            if fresh_after is not None:
                self._fresh_after = max(self._fresh_after, fresh_after)
            clear = self.clear(fault_serial)
            if clear and transmitting:
                self.transmitting = True
            return clear
        finally:
            self._lock.release()

    def contend(self, now, fault_serial=None):
        with self._lock:
            if not self.clear(fault_serial):
                self._slot = None
                return False
            if self._slot is None:
                self._slot = now + self.slot_ms / 1000
                return False
            if now < self._slot:
                return False
            self._slot = now + self.slot_ms / 1000
            return self.draw() <= self.persistence
