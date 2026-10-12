# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Stations on shared channels, in simulation: each one a ``Controller`` and a
``DualReceiver``, each transmission passed through ``sim``'s channel model on
its way to every other station tuned to that channel.

Time moves in steps of ``STEP`` seconds. Half duplex: a station hears
nothing but its own noise while it transmits.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from hfmodem.bateleur import sim
from hfmodem.bateleur.spec import FS as SIM_FS
from hfmodem.bateleur.ale import fsk
from hfmodem.bateleur.ale.controller import Config, Controller

STEP = 0.05


class Station:
    def __init__(self, ether: "Ether", config: Config, **kw) -> None:
        self.ether = ether
        self.rx = fsk.DualReceiver()
        self.channel = 0
        self.tx_until = 0.0
        self.expected: list[int] = []
        self.ctl = None
        self.ctl = Controller(config, tune=self._tune, send=self._send,
                              expect=self._expect, utc=lambda now: ether.utc0 + now, **kw)
        self.rx.set_lp(self.ctl.lp_view())

    def _expect(self, ws: list[int]) -> None:
        self.expected = list(ws)
        self.rx.expect(self.expected)

    def _tune(self, channel: int) -> bool:
        # A new channel is new audio: nothing heard before carries over.
        if channel != self.channel:
            self.rx = fsk.DualReceiver()
            self.rx.expect(self.expected)
            if self.ctl is not None:
                self.rx.set_lp(self.ctl.lp_view())
        self.channel = channel
        return True

    def _send(self, words: list[int]) -> float:
        audio = fsk.modulate(words)
        start = max(self.ether.now, self.tx_until)
        self.tx_until = start + audio.size / fsk.FS
        self.ether.transmit(self, self.channel, start, audio)
        return audio.size / fsk.FS


class Ether:
    def __init__(self, *, snr_db: float = 20.0, preset: str = "awgn", seed: int = 0,
                 reachable=None) -> None:
        self.snr_db, self.preset = snr_db, preset
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.now = 0.0
        self.stations: list[Station] = []
        #: (start sample, audio, channel, sender) as each receiver will hear it.
        self._arrivals: dict[Station, list] = {}
        #: reachable(sender, receiver, channel) -> bool: which paths are open.
        self.reachable = reachable or (lambda a, b, c: True)
        self._noise_sigma = None
        #: The time of day when the simulation starts (for linking
        #: protection's seeds): an afternoon in May 2026.
        self.utc0 = 1_778_255_854.0

    def add(self, config: Config, **kw) -> Station:
        s = Station(self, config, **kw)
        self.stations.append(s)
        self._arrivals[s] = []
        return s

    def transmit(self, sender: Station, channel: int, start: float, audio: np.ndarray) -> None:
        p = float(np.mean(audio ** 2))
        # Noise density set by the first transmission's power: the SNR is the
        # transmission's, in 3 kHz (table A-II).
        if self._noise_sigma is None:
            n0 = p / 10 ** (self.snr_db / 10) / 3000.0
            self._noise_sigma = np.sqrt(n0 * fsk.FS / 2)
        for s in self.stations:
            if s is sender or not self.reachable(sender, s, channel):
                continue
            up = resample_poly(audio, SIM_FS // 1600, fsk.FS // 1600)
            faded = sim.channel(up, snr_db=300, preset=self.preset,
                                seed=int(self.rng.integers(1 << 30)))
            y = resample_poly(faded, fsk.FS // 1600, SIM_FS // 1600)
            self._arrivals[s].append((int(round(start * fsk.FS)), y, channel))

    def run(self, seconds: float, until=None) -> None:
        n = int(round(STEP * fsk.FS))
        end = self.now + seconds
        if self._noise_sigma is None:
            # No transmission yet: noise as a unit-power one would see it.
            n0 = 0.5 / 10 ** (self.snr_db / 10) / 3000.0
            self._noise_sigma = np.sqrt(n0 * fsk.FS / 2)
        while self.now < end:
            t0 = int(round(self.now * fsk.FS))
            self.now += STEP
            for s in self.stations:
                s.ctl.tick(self.now)
            for s in self.stations:
                x = self.rng.normal(0, self._noise_sigma, n)
                if self.now > s.tx_until:
                    for (a0, y, ch) in self._arrivals[s]:
                        if ch != s.channel:
                            continue
                        lo, hi = max(t0, a0), min(t0 + n, a0 + y.size)
                        if hi > lo:
                            x[lo - t0:hi - t0] += y[lo - a0:hi - a0]
                for ev in s.rx.push(x):
                    s.ctl.heard(ev, self.now, age=(s.rx.samples - ev.end) / fsk.FS)
                if s.rx.signal:
                    s.ctl.signal(self.now)
                self._arrivals[s] = [a for a in self._arrivals[s] if a[0] + a[1].size > t0]
            if until is not None and until():
                return
