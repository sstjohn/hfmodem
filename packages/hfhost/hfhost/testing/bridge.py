# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""In-process test bridge: two FakeModems wired back-to-back so a real CWP
initiator half talks to a real responder half.

A pump thread moves data-port bytes A->B and B->A, emitting BUFFER
notifications (a rise then the drained value) so the drain metric and the
high-water pacing are exercised for real. rate_* throttles a direction to a
few bytes per tick to force backpressure; corrupt_at flips a byte in the A->B
stream to manufacture a mid-stream desync.
"""

from __future__ import annotations

import threading
import time

from ..client import ModemClient
from ..link import Link, VaraLink
from ..config import ModemConfig, Quirks
from ..transcript import Transcript
from .fakemodem import FakeModem, ThreadFaults


class Bridge:
    def __init__(self, tmp_path, *, rate_ab: int | None = None,
                 rate_ba: int | None = None, corrupt_at: int | None = None,
                 poll_s: float = 0.005) -> None:
        self.tmp_path = tmp_path
        self.faults = ThreadFaults()
        self.a = FakeModem(version="A-1.0", faults=self.faults).start()
        self.b = FakeModem(version="B-1.0", faults=self.faults).start()
        self.rate = {"ab": rate_ab, "ba": rate_ba}
        self.corrupt_at = corrupt_at
        self.poll_s = poll_s

        self._off = {"ab": 0, "ba": 0}
        self._pending = {"ab": bytearray(), "ba": bytearray()}
        self._fwd = {"ab": 0, "ba": 0}         # bytes handed to the far side
        self.peak = {"ab": 0, "ba": 0}
        self._running = False
        self._thread: threading.Thread | None = None

    # -- clients -----------------------------------------------------------

    def link(self, side: str, tr_name: str, **kw) -> Link:
        """A configured Link over this side's fake modem — what consumers use."""
        fm = self.a if side == "a" else self.b
        cfg = ModemConfig(name=side, cmd_port=fm.cmd_port, data_port=fm.data_port,
                          quirks=kw.pop("quirks", Quirks()))
        tr = Transcript(str(self.tmp_path / f"{tr_name}.jsonl"), tr_name)
        kw.setdefault("attach_timeout_s", 2.0)
        return VaraLink(ModemClient(cfg, "127.0.0.1", tr, **kw), cfg)

    def client(self, side: str, tr_name: str, **kw) -> ModemClient:
        return self.link(side, tr_name, **kw).client

    # -- pump --------------------------------------------------------------

    def start(self) -> "Bridge":
        self._running = True
        self._thread = self.faults.watch("bridge", self._pump)
        return self

    def _pump(self) -> None:
        while self._running:
            self._step("ab", self.a, self.b)
            self._step("ba", self.b, self.a)
            time.sleep(self.poll_s)

    def _step(self, key: str, src: FakeModem, dst: FakeModem) -> None:
        with src._cv:
            buf = bytes(src.received_data[self._off[key]:])
        pending = self._pending[key]
        if buf:
            self._off[key] += len(buf)
            pending += buf
            self.peak[key] = max(self.peak[key], len(pending))
            try:
                src.notify(f"BUFFER {len(pending)}")        # the rise
            except RuntimeError:
                return
        if not pending:
            return
        rate = self.rate[key]
        take = len(pending) if rate is None else min(rate, len(pending))
        chunk = bytes(pending[:take])
        del pending[:take]
        if chunk:
            chunk = self._maybe_corrupt(key, chunk)
            try:
                dst.send_data(chunk)
            except RuntimeError:
                # The far end detached while this pump was mid-step. Both
                # `notify` calls around this one already allowed for it; this
                # one did not, so the escaping RuntimeError killed the pump.
                #
                # That was worth fixing but was never the cause of a red gate:
                # a dead thread here is filed by pytest as a warning against
                # whichever test is at a boundary when it lands, and warnings
                # are not errors in this project, so it could not fail anything.
                # It was read as the cause of test_kestrel_pair_matches_golden's
                # one-in-three failure; that failure was a golden-table
                # deviation with its own, unrelated cause.
                #
                # Dropping the chunk is right rather than merciful: the session
                # it was going to no longer exists, which is what a real link
                # does when the far end goes away mid-transfer.
                return
        try:
            src.notify(f"BUFFER {len(pending)}")            # drained value
        except RuntimeError:
            pass

    def _maybe_corrupt(self, key: str, chunk: bytes) -> bytes:
        if key != "ab" or self.corrupt_at is None:
            self._fwd[key] += len(chunk)
            return chunk
        start = self._fwd[key]
        end = start + len(chunk)
        if start <= self.corrupt_at < end:
            i = self.corrupt_at - start
            b = bytearray(chunk)
            b[i] ^= 0xFF
            chunk = bytes(b)
        self._fwd[key] += len(chunk)
        return chunk

    def connect(self, a_call="N0AAA", b_call="N0BBB", bw="2300") -> None:
        """Emit CONNECTED to both clients (the modems' post-handshake signal)."""
        self.a.wait_attached()
        self.b.wait_attached()
        self.a.notify(f"CONNECTED {a_call} {b_call} {bw}")
        self.b.notify(f"CONNECTED {b_call} {a_call} {bw}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.a.stop()           # either may raise on a dead harness thread;
        finally:                    # both modems still get closed
            self.b.stop()


def disconnect_responder(fm):
    """on_command hook: reply OK to DISCONNECT, then emit DISCONNECTED a beat
    later — as a real modem does once the link is torn down, and after the
    initiator has subscribed for it."""
    def handler(cmd):
        if cmd.split()[0].upper() == "DISCONNECT":
            def later():
                try:
                    fm.notify("DISCONNECTED")
                except RuntimeError:
                    pass
            threading.Timer(0.05, later).start()
            return "OK"
        return None
    return handler


class EchoBridge:
    """One FakeModem whose data-port bytes loop straight back to their sender:
    the kestrel-loopback / echo-peer path."""

    def __init__(self, tmp_path, poll_s: float = 0.005) -> None:
        self.tmp_path = tmp_path
        self.faults = ThreadFaults()
        self.m = FakeModem(version="echo-1.0", faults=self.faults).start()
        self.poll_s = poll_s
        self._off = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def link(self, tr_name: str, **kw) -> Link:
        cfg = ModemConfig(name="echo", cmd_port=self.m.cmd_port,
                          data_port=self.m.data_port,
                          quirks=kw.pop("quirks", Quirks()))
        tr = Transcript(str(self.tmp_path / f"{tr_name}.jsonl"), tr_name)
        kw.setdefault("attach_timeout_s", 2.0)
        return VaraLink(ModemClient(cfg, "127.0.0.1", tr, **kw), cfg)

    def client(self, tr_name: str, **kw) -> ModemClient:
        return self.link(tr_name, **kw).client

    def start(self) -> "EchoBridge":
        self._running = True
        self._thread = self.faults.watch("echo-bridge", self._pump)
        return self

    def _pump(self) -> None:
        while self._running:
            with self.m._cv:
                buf = bytes(self.m.received_data[self._off:])
            if buf:
                self._off += len(buf)
                try:
                    self.m.send_data(buf)        # the client may have hung up
                    self.m.notify("BUFFER 0")    # between the read and here
                except RuntimeError:
                    pass
            time.sleep(self.poll_s)

    def connect(self, call="N0CAL") -> None:
        self.m.wait_attached()
        self.m.notify(f"CONNECTED {call} {call} 2300")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.m.stop()
