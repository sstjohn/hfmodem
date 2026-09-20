# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One test body, both dialects.

Every test here is parameterized over the two host dialects and asserts on the
Link surface only. If a behaviour cannot be stated identically for both, it does
not belong in this file — it belongs in the dialect's own conformance catalog.
"""

import queue
import tempfile
import time
from pathlib import Path

import pytest

from creance.link import (CONNECTED, DISCONNECTED, PTT, STATS,
                          HostApiLink, VaraLink, open_link)
from hfhost import hostapi
from hfhost.client import ModemClient
from hfhost.config import ModemConfig
from hfhost.hostapi import HostApiClient, ST_CONNECTED, ST_DISCONNECTED
from hfhost.testing.fakehostapi import FakeHostApiModem
from hfhost.testing.fakemodem import FakeModem
from hfhost.transcript import Transcript

MYCALL = "K7CRE"


def _transcript() -> Transcript:
    """ModemClient keeps a daemon-level transcript that per-session ones swap
    against, so there is always a base one; tests need a real file, not None."""
    d = tempfile.mkdtemp(prefix="creance-link-")
    return Transcript(str(Path(d) / "t.jsonl"), "test")


class VaraHarness:
    dialect = "vara"

    def __init__(self):
        self.modem = FakeModem(echo=True, buffer_notifications=True).start()
        cfg = ModemConfig(name="m", dialect="vara",
                          cmd_port=self.modem.cmd_port,
                          data_port=self.modem.data_port)
        self.t = _transcript()
        self.link = VaraLink(ModemClient(cfg, "127.0.0.1", self.t), cfg)

    def announce_connected(self, peer="K7ABC"):
        self.modem.notify(f"CONNECTED {peer} {MYCALL} 2300")

    def announce_disconnected(self):
        self.modem.notify("DISCONNECTED")

    def announce_queue(self, n):
        self.modem.notify(f"BUFFER {n}")

    def announce_ptt(self, on):
        self.modem.notify("PTT ON" if on else "PTT OFF")

    def close(self):
        self.link.close()
        self.modem.stop()
        self.t.close()


class HostApiHarness:
    dialect = "hostapi"

    def __init__(self):
        self.modem = FakeHostApiModem(echo=True)
        cfg = ModemConfig(name="m", dialect="hostapi",
                          cmd_port=self.modem.port, data_port=0)
        self.link = HostApiLink(HostApiClient(cfg), cfg)

    def announce_connected(self, peer="K7ABC"):
        self.modem.state(ST_CONNECTED, peer_id=peer)

    def announce_disconnected(self):
        self.modem.state(ST_DISCONNECTED, reason=hostapi.RS_REMOTE)

    def announce_queue(self, n):
        self.modem.emit(hostapi.LINK_STATS, gear="floor", rung=1,
                        queue_bytes=n, throughput_bps=300.0)

    def announce_ptt(self, on):
        self.modem.emit(hostapi.PHYSICAL_STATE, ptt=on)

    def close(self):
        self.link.close()
        self.modem.close()


@pytest.fixture(params=[VaraHarness, HostApiHarness],
                ids=["vara", "hostapi"])
def harness(request):
    h = request.param()
    h.link.configure(MYCALL)
    h.link.attach()
    yield h
    h.close()


def _wait(q, kind, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            ev = q.get(timeout=max(0.01, end - time.monotonic()))
        except queue.Empty:
            break
        if ev.kind == kind:
            return ev
    raise AssertionError(f"no {kind} event within {timeout}s")


def _until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached")


def test_opens_and_reports_a_version(harness):
    assert harness.link.name == "m"
    assert isinstance(harness.link.version, str)


def test_connected_event_names_the_peer(harness):
    q = queue.Queue()
    harness.link.subscribe(q)
    harness.announce_connected("K7ABC")
    ev = _wait(q, CONNECTED)
    assert ev.peer == "K7ABC"
    assert ev.modem == "m"
    _until(lambda: harness.link.connected)


def test_disconnected_event_and_state(harness):
    q = queue.Queue()
    harness.link.subscribe(q)
    harness.announce_connected()
    _until(lambda: harness.link.connected)
    harness.announce_disconnected()
    _wait(q, DISCONNECTED)
    _until(lambda: not harness.link.connected)


def test_queue_depth_is_reported(harness):
    q = queue.Queue()
    harness.link.subscribe(q)
    harness.announce_queue(4096)
    _until(lambda: harness.link.queue_bytes == 4096)


def test_ptt_is_reported(harness):
    q = queue.Queue()
    harness.link.subscribe(q)
    harness.announce_ptt(True)
    ev = _wait(q, PTT)
    assert ev.fields["on"] is True


def test_data_round_trip(harness):
    harness.announce_connected()
    _until(lambda: harness.link.connected)
    harness.link.send(b"the same bytes either way")
    assert harness.link.recv(timeout=3.0) == b"the same bytes either way"


def test_peek_does_not_consume(harness):
    harness.announce_connected()
    _until(lambda: harness.link.connected)
    harness.link.send(b"CRN1payload")
    assert harness.link.peek(4, timeout=3.0) == b"CRN1"
    assert harness.link.recv(11, timeout=1.0) == b"CRN1payload"


def test_epoch_fences_a_previous_session(harness):
    harness.announce_connected()
    _until(lambda: harness.link.connected)
    harness.link.send(b"stale")
    _until(lambda: harness.link.peek(5, timeout=0.5) == b"stale")
    assert harness.link.bump_epoch() == b"stale"
    assert harness.link.recv(16, timeout=0.05) == b""


def test_recv_refuses_a_short_read(harness):
    """Both dialects return nothing rather than a partial count: a caller that
    asked for n is framing something, and a short read loses that silently."""
    assert harness.link.recv(16, timeout=0.05) == b""


def test_partial_data_is_not_handed_back_early(harness):
    harness.announce_connected()
    _until(lambda: harness.link.connected)
    harness.link.send(b"1234")
    _until(lambda: harness.link.peek(4, timeout=0.5) == b"1234")
    assert harness.link.recv(8, timeout=0.1) == b""      # still buffered
    assert harness.link.recv(4, timeout=0.5) == b"1234"


# -- dialect-specific truths, stated once and honestly -------------------


def test_stats_only_exist_on_the_structured_dialect():
    """Absence is the honest answer for VARA: it has no telemetry to report,
    and a synthesized STATS event would be a fabricated measurement."""
    h = HostApiHarness()
    try:
        h.link.configure(MYCALL)
        h.link.attach()
        q = queue.Queue()
        h.link.subscribe(q)
        h.announce_queue(128)
        ev = _wait(q, STATS)
        assert ev.fields["gear"] == "floor"
    finally:
        h.close()


def test_open_link_picks_the_dialect_from_config():
    m = FakeHostApiModem()
    try:
        cfg = ModemConfig(name="h", dialect="hostapi", cmd_port=m.port, data_port=0)
        lk = open_link(cfg, "127.0.0.1", _transcript())
        assert isinstance(lk, HostApiLink)
        lk.close()
    finally:
        m.close()

    fm = FakeModem().start()
    try:
        cfg = ModemConfig(name="v", cmd_port=fm.cmd_port, data_port=fm.data_port)
        assert cfg.dialect == "vara"          # the default, so old configs work
        lk = open_link(cfg, "127.0.0.1", _transcript())
        assert isinstance(lk, VaraLink)
        lk.close()
    finally:
        fm.stop()


def test_malformed_connected_is_reported_not_guessed():
    """Pat panics on a CONNECTED whose fields it cannot read. We must never
    invent a peer for one — an unparsed line is reported as OTHER."""
    h = VaraHarness()
    try:
        h.link.configure(MYCALL)
        h.link.attach()
        q = queue.Queue()
        h.link.subscribe(q)
        h.modem.notify("CONNECTED")
        ev = q.get(timeout=3.0)
        assert ev.kind != CONNECTED
    finally:
        h.close()
