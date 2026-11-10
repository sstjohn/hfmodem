# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Virtual air for hosted modems: real host threads, virtual RF time.

Two sample-level endpoints (`host.SabirModem`) share one simulated
channel. A single air thread owns all link state: host commands arrive as
posted closures, transmissions move between the endpoints' outboxes through
the channel function (advancing the virtual clock by their true duration
plus a turnaround, with PTT reported around each over), and when the air
is quiet the earliest FSM timer fires -- the M3 ``run_pair`` discipline,
made continuous so TCP clients can drive simulated sessions.

StationAir implements the same ``clock``/``post``/``register`` port for
station audio. Simulator results do not establish live-radio behavior.
"""

from __future__ import annotations

import queue
import threading
import time

from hfmodem.sabir.phy.modem import FS


class SimulatedAir:
    """Threaded two-endpoint audio simulation with optional wall-clock pacing.

    PTT callbacks describe simulated activity; no radio or audio device is opened.
    """

    def __init__(self, channel=None, turnaround_s: float = 0.25,
                 realtime: bool = False):
        self.channel = channel or (lambda wav, t: wav)
        self.turnaround_s = turnaround_s
        self.realtime = realtime            # pace virtual time to the wall clock
        self.now = 0.0
        self._cmds: queue.Queue = queue.Queue()
        self._ends: list = []
        self._thread = None
        self._running = False
        self._wall0 = 0.0

    def clock(self) -> float:
        return self.now

    def register(self, end) -> None:
        if len(self._ends) >= 2:
            raise ValueError("a SimulatedAir carries exactly two endpoints")
        self._ends.append(end)

    def post(self, fn) -> None:
        self._cmds.put(fn)

    def start(self) -> None:
        self._running = True
        self._wall0 = time.monotonic() - self.now      # anchor wall to virtual
        self._thread = threading.Thread(target=self._loop, name="virtual-air",
                                        daemon=True)
        self._thread.start()

    # -- pacing (no-ops unless realtime) -----------------------------------
    def _sleep_until(self, virt_target: float) -> None:
        """Hold virtual time at ``virt_target`` until the wall clock catches
        up, in capped increments so ``stop()`` stays responsive."""
        if not self.realtime:
            return
        while self._running:
            left = (self._wall0 + virt_target) - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 0.05))

    def _wait_or_command(self, virt_target: float, *, passive: bool = False) -> bool:
        """Wait for a host command when real-time paced or only passively idle.

        Retry/turn timers advance instantly in unpaced simulations. A passive
        peer-expiry timer instead waits real idle time, allowing the TCP client
        to react to CONNECTED before the virtual session expires.
        """
        if not self.realtime and not passive:
            return False
        wall_start, virtual_start = time.monotonic(), self.now
        target = (wall_start + max(0, virt_target - self.now) if not self.realtime
                  else self._wall0 + virt_target)
        while self._running:
            left = target - time.monotonic()
            if left <= 0:
                return False
            try:
                command = self._cmds.get(timeout=min(left, 0.05))
                if not self.realtime:
                    self.now = min(virt_target, virtual_start + time.monotonic() - wall_start)
                command()
                return True
            except queue.Empty:
                continue
        return False

    def stop(self) -> None:
        self._running = False
        self._cmds.put(lambda: None)
        if self._thread:
            self._thread.join(timeout=10.0)

    # -- the air thread ----------------------------------------------------
    def _report(self) -> None:
        for e in self._ends:
            e.after_step()

    def _move(self) -> bool:
        moved = False
        for i, src in enumerate(self._ends):
            dst = self._ends[1 - i]
            while src.link.outbox:
                wav = src.link.outbox.pop(0)
                src.ptt(True)
                t0 = self.now
                self.now = t0 + wav.size / FS
                self._sleep_until(self.now)         # PTT ON..OFF wall = airtime
                dst.link.on_air(self.channel(wav, t0))
                src.ptt(False)
                self.now += self.turnaround_s
                self._sleep_until(self.now)
                self._report()
                moved = True
        return moved

    def _loop(self) -> None:
        while self._running:
            busy = False
            try:
                while True:
                    self._cmds.get_nowait()()
                    busy = True
            except queue.Empty:
                pass
            self._report()
            busy = self._move() or busy
            if busy:
                continue
            deadlines = [d for e in self._ends
                         if (d := e.link.fsm.next_deadline()) is not None]
            if deadlines:
                d = min(deadlines)
                passive = all(e.link.fsm._deadline is None for e in self._ends)
                if self._wait_or_command(d, passive=passive):
                    continue
                if not self._running:
                    break
                self.now = max(self.now, d) + 1e-3   # always advance (old form)
                for e in self._ends:
                    e.link.fsm.on_timer()
                continue
            try:
                self._cmds.get(timeout=0.05)()
            except queue.Empty:
                pass
