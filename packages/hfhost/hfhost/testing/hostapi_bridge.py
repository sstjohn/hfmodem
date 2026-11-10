# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two structured-dialect fake modems wired back to back.

The point is narrow: the Link seam claims a scenario
runs unchanged over either host dialect, and that claim is only worth anything
if the same CWP session actually completes over both. The VARA `Bridge` proves
it for one dialect; this proves it for the other, against the same scenarios.

Simpler than the VARA bridge because the structured dialect has no separate data
socket and no BUFFER cadence to reproduce: payload submitted with Send is handed
to the far modem as DataReceived, and queue depth is reported as LinkStats.
"""

from __future__ import annotations

import threading
import time

from .. import hostapi
from ..config import ModemConfig, Quirks
from ..hostapi import HostApiClient
from ..link import HostApiLink, Link
from ..transcript import Transcript
from .fakehostapi import FakeHostApiModem


class HostApiBridge:
    def __init__(self, tmp_path, *, rate_ab: int | None = None,
                 rate_ba: int | None = None, poll_s: float = 0.005) -> None:
        self.tmp_path = tmp_path
        # A Connect on either side links both, the way a real pair does. The
        # alternative -- a test-side timer that calls connect() -- races any
        # choreography that resets state first, because the timer fires on the
        # wall clock and the reset does not.
        self.a = FakeHostApiModem(modem_version="A/1.0",
                                  on_command=self._on_command)
        self.b = FakeHostApiModem(modem_version="B/1.0",
                                  on_command=self._on_command)
        self.rate = {"ab": rate_ab, "ba": rate_ba}
        self.poll_s = poll_s

        self._off = {"ab": 0, "ba": 0}
        self._pending = {"ab": bytearray(), "ba": bytearray()}
        self.peak = {"ab": 0, "ba": 0}
        self._running = False
        self._thread: threading.Thread | None = None

    def _on_command(self, msg, modem) -> None:
        if msg.get("m") != hostapi.CONNECT:
            return
        peer = self.b if modem is self.a else self.a
        caller = (modem.identity or "N0AAA")
        peer.connected = True
        peer.state(hostapi.ST_CONNECTED, peer_id=caller)

    def link(self, side: str, tr_name: str, **kw) -> Link:
        fm = self.a if side == "a" else self.b
        cfg = ModemConfig(name=side, dialect="hostapi", cmd_port=fm.port,
                          data_port=0, quirks=kw.pop("quirks", Quirks()))
        tr = Transcript(str(self.tmp_path / f"{tr_name}.jsonl"), tr_name)
        return HostApiLink(HostApiClient(cfg, transcript=tr), cfg)

    def start(self) -> "HostApiBridge":
        self._running = True
        self._thread = threading.Thread(target=self._pump, name="hostapi-bridge",
                                        daemon=True)
        self._thread.start()
        return self

    def connect(self, a_call="N0AAA", b_call="N0BBB") -> None:
        self.a.state(hostapi.ST_CONNECTED, peer_id=b_call)
        self.b.state(hostapi.ST_CONNECTED, peer_id=a_call)

    def disconnect(self) -> None:
        for fm in (self.a, self.b):
            fm.state(hostapi.ST_DISCONNECTED, reason=hostapi.RS_REMOTE)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.a.close()
        self.b.close()

    def _pump(self) -> None:
        while self._running:
            self._step("ab", self.a, self.b)
            self._step("ba", self.b, self.a)
            time.sleep(self.poll_s)

    def _step(self, key: str, src: FakeHostApiModem, dst: FakeHostApiModem) -> None:
        with src._lock:
            buf = bytes(src.sent[self._off[key]:])
        pending = self._pending[key]
        if buf:
            self._off[key] += len(buf)
            pending += buf
            self.peak[key] = max(self.peak[key], len(pending))
            src.emit(hostapi.LINK_STATS, gear="floor", rung=1,
                     queue_bytes=len(pending), throughput_bps=300.0)
        if not pending:
            return
        rate = self.rate[key]
        take = len(pending) if rate is None else min(rate, len(pending))
        chunk = bytes(pending[:take])
        del pending[:take]
        if chunk:
            dst.send_data(chunk)
        src.emit(hostapi.LINK_STATS, gear="floor", rung=1,
                 queue_bytes=len(pending), throughput_bps=300.0)
