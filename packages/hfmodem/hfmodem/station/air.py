# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""sabir's air, when the air is the air.

`sim.air.SimulatedAir` runs two endpoints over a simulated channel on one thread, with
virtual time: a transmission moves from one outbox to the other's `on_air`, the
clock advances by the burst's true duration plus a turnaround, and when nothing is
in flight the earliest timer fires. Everything above it — the ARQ state machine,
the host servers, creance driving them over TCP — was written against that port and
does not change here.

This is the same port with the simulation taken out, which is the whole of what M5
means:

  * **One endpoint.** The other station is on the air, and nothing here knows
    anything about it beyond what arrives.
  * **Real time.** `clock()` is the monotonic clock, so a deadline that has already
    passed fires now rather than time teleporting forward to meet it. Under
    simulation a slow decode cost nothing; here it costs the turnaround, and the
    FSM finds out the same way it would find out about a slow peer.
  * **Receive is detection, not handoff.** There is no `dst.link.on_air(wav)` — a
    burst has to be found in a stream that is mostly noise. The station's stream
    lane hands over every sample once and the segmenter brackets what is keyed.
  * **The air does not key.** It asks the arbiter, which owns the radio, and a
    refusal is not distinguishable from a burst the peer never heard, so the FSM's
    retry handles both.

Everything that talks to the endpoint still runs on the one air thread, for the
reason it always did: the ARQ state machine is not thread-safe, and the station's
receive loop and the host server's command handler are two more threads that would
otherwise reach it. Audio crosses in on a queue and commands cross in as posted
closures; nothing else crosses at all.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from hfmodem.core.audio import StreamLane
from hfmodem.sabir.monitor import Segmenter
from hfmodem.sabir.offair import to_analytic, to_real
from hfmodem.sabir.phy.modem import FS


class StationAir:
    """One sabir endpoint on the station's radio.

    A station has one radio, so it has one sabir endpoint. `register` replaces
    rather than appends: the host server serves one connection at a time and gives
    each a fresh core, so a reconnecting application must not find the previous
    session's state machine still mounted.
    """

    def __init__(self, link, lane: StreamLane, *, log=None) -> None:
        self.link = link                    # station.link.SabirLink — lane and arbiter
        self.lane = lane
        if lane.rate != FS:
            raise ValueError(f"sabir runs at {FS} Hz, not {lane.rate}")
        self.seg = Segmenter()
        self._log = log or (lambda *_: None)
        self._end = None
        self._cmds: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        #: Counted rather than logged, so a test can assert on them.
        self.bursts = 0
        self.sent = 0
        self.refused = 0

    # -- the port sabir was written against --------------------------------

    def clock(self) -> float:
        return time.monotonic()

    def register(self, end) -> None:
        self._end = end

    def post(self, fn) -> None:
        self._cmds.put(fn)

    # -- the station side --------------------------------------------------

    def feed(self, samples: np.ndarray) -> None:
        """Hand received audio in from whatever thread polls the lane.

        Segmenting happens on the air thread rather than here: it carries the
        adaptive floor that decides what counts as a burst, and a floor learned
        from two threads' interleaving is not a floor.
        """
        self._audio.put(np.asarray(samples, float))

    def pump(self) -> None:
        """Move one lane poll into the air. The station's receive loop calls this."""
        for _, chunk in self.lane.poll():
            self.feed(chunk)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="station-air",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._cmds.put(lambda: None)
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

    # -- the air thread ----------------------------------------------------

    def _receive(self) -> bool:
        end, found = self._end, False
        while True:
            try:
                chunk = self._audio.get_nowait()
            except queue.Empty:
                break
            for _, burst in self.seg.push(chunk):
                self.bursts += 1
                if end is None:
                    continue
                try:
                    # The card hears a real passband; every sabir demodulator
                    # works on the analytic signal, and under simulation the
                    # question never arose because the transmitter's own analytic
                    # waveform was handed straight across. This is the seam that
                    # substitution exposes, and it is the receiver's to close.
                    end.link.on_air(to_analytic(burst))
                except Exception as exc:            # noqa: BLE001
                    # A burst that will not decode is the ordinary case on HF, and
                    # a decoder that raises on one must not take the station's air
                    # thread down with it.
                    self._log(f"sabir: burst not decoded ({exc})")
                found = True
        return found

    def _transmit(self) -> bool:
        end = self._end
        if end is None:
            return False
        moved = False
        while end.link.outbox:
            wav = end.link.outbox.pop(0)
            end.ptt(True)
            try:
                # The mirror of the receive seam: a transmitter emits the real
                # part, and handing the arbiter a complex array would put the
                # imaginary half somewhere no card can follow.
                sent = self.link.transmit(to_real(wav), responding=True)
            finally:
                end.ptt(False)
            if sent is None:
                self.refused += 1
            else:
                self.sent += 1
            moved = True
        return moved

    def _timers(self) -> None:
        end = self._end
        if end is None:
            return
        d = end.link.fsm.next_deadline()
        if d is not None and d <= self.clock():
            end.link.fsm.on_timer()

    def _loop(self) -> None:
        while self._running:
            busy = False
            try:
                while True:
                    self._cmds.get_nowait()()
                    busy = True
            except queue.Empty:
                pass
            if self._end is not None:
                self._end.after_step()
            busy = self._receive() or busy
            busy = self._transmit() or busy
            self._timers()
            if busy:
                continue
            # Idle. Waiting on the command queue rather than sleeping means a host
            # command is acted on at once; the timeout is the granularity at which
            # a deadline or a lane poll is noticed, and 20 ms is far inside sabir's
            # shortest turnaround.
            try:
                self._cmds.get(timeout=0.02)()
            except queue.Empty:
                pass
