# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CWP scenario halves talking end-to-end over the in-process bridge."""

import threading
import time
from dataclasses import replace

import pytest

from creance import hproto, scenarios
from creance.hproto import Desync, End
from creance.scenarios import (CwpSession, SCENARIOS, SessionCancelled,
                               fallback_sink, initiate, respond)
from hfhost.transcript import Transcript, read

from hfhost.testing.bridge import Bridge, EchoBridge


class StubClient:
    """A Link that swallows writes instantly and replays a scripted rx
    stream: queue_bytes never rises, so the high-water wait never runs."""

    def __init__(self, transcript, *, rx=(), cancel_after=None,
                 cancel=None, name="stub"):
        self.name = name
        self.epoch = 0
        self.queue_bytes = 0
        self.transcript = transcript
        self.writes = []
        self._rx = list(rx)
        self._cancel_after = cancel_after
        self._cancel = cancel

    def send(self, blob, label=""):
        self.writes.append((label, blob))
        self.transcript.data(self.name, "tx", blob, label=label)
        if self._cancel_after is not None and len(self.writes) >= self._cancel_after:
            self._cancel.set()

    def recv(self, n=None, timeout=0.0, epoch=None, cancel=None):
        return self._rx.pop(0) if self._rx else b""

    def wait_for(self, kind, timeout, cancel=None):
        return None


def stub_session(tmp_path, name="stub", **kw):
    tr = Transcript(str(tmp_path / f"{name}.jsonl"), name)
    cancel = threading.Event()
    client = StubClient(tr, cancel=cancel, name=name, **kw)
    return CwpSession(client, tr, sid="SID", mycall="N0AAA", cancel=cancel,
                      epoch=0, recv_timeout_s=1.0), client, cancel


def wait_connected(*clients, timeout=2.0):
    deadline = time.monotonic() + timeout
    for c in clients:
        while not c.connected and time.monotonic() < deadline:
            time.sleep(0.005)
        assert c.connected, f"{c.name} never saw CONNECTED"


@pytest.fixture
def bridge(tmp_path):
    b = Bridge(tmp_path).start()
    yield b
    b.stop()


def make_sessions(bridge, tmp_path, *, i_high=None, r_high=None,
                  i_recv=30.0, r_recv=30.0):
    ci = bridge.link("a", "init")
    cr = bridge.link("b", "resp")
    ci.attach()
    cr.attach()
    bridge.connect()
    wait_connected(ci, cr)
    kw_i = {"recv_timeout_s": i_recv}
    kw_r = {"recv_timeout_s": r_recv}
    if i_high is not None:
        kw_i["high_water"] = i_high
    if r_high is not None:
        kw_r["high_water"] = r_high
    si = CwpSession(ci, ci.transcript, sid="SID-I", mycall="N0AAA", **kw_i)
    sr = CwpSession(cr, cr.transcript, sid="SID-R", mycall="N0BBB", **kw_r)
    return ci, cr, si, sr


def run_responder(sr, out, hello_timeout=5.0, rewrite=None):
    """Run the responder half in a thread. rewrite doctors the parsed HELLO —
    the only way, in one process with one scenario registry, to play a far end
    whose build does not have the scenario the initiator asked for."""
    def go():
        try:
            h = sr.await_hello(timeout=hello_timeout)
            if h is None:
                out["r"] = scenarios.ScenarioResult("no_hello", {})
                return
            out["r"] = respond(sr, rewrite(h) if rewrite else h)
        except Exception as exc:                       # surface in the test
            out["r_exc"] = exc
    th = threading.Thread(target=go, name="responder", daemon=True)
    th.start()
    return th


# -- happy paths -------------------------------------------------------------

def test_unidir_happy_path(tmp_path):
    # Throttled so the transfer takes real time: an untimed pipe measures
    # nothing, and the rates below have to be checkable.
    rate = 4096                      # bytes per 5 ms poll tick
    b = Bridge(tmp_path, rate_ab=rate, poll_s=0.005).start()
    try:
        ci, cr, si, sr = make_sessions(b, tmp_path)
        out = {}
        th = run_responder(sr, out)
        res_i = initiate(si, "unidir", {"size": "24k", "payload": "prbs9"})
        th.join(20)
        res_r = out["r"]

        assert res_i.outcome == "ok"
        assert res_r.outcome == "ok"
        assert si.tx_sha256 == sr.rx_sha256
        assert si.payload_tx == sr.payload_rx == 24576
        assert res_i.stats["far_bytes"] == 24576
        # metrics see the far-end REPORT and the drain terminal condition
        from creance.metrics import compute
        m = compute(read(str(ci.transcript.path))[0],
                    {"sid": "SID-I", "modem": "a", "scenario": "unidir",
                     "outcome": res_i.outcome})
        assert m.report_sha == si.tx_sha256
        assert m.report_sha_match is True
        assert m.drain_bps_local is not None and m.drain_window_s is not None
        assert m.drain_bytes == si.payload_tx + 6 * hproto._HDR_LEN + 6 * 4

        # The bridge moves `rate` bytes per tick; both rates must land in that
        # neighbourhood rather than at pipe speed or at handshake speed.
        wire_bps = rate * 8 / 0.005
        assert 0.05 * wire_bps < m.goodput_bps_far < wire_bps
        assert 0.05 * wire_bps < m.drain_bps_local < wire_bps
    finally:
        b.stop()


def test_single_chunk_transfer_is_not_absurd(tmp_path):
    """The whole payload in one DATA frame used to time the far-end goodput
    from the frame's arrival to the END — a window that excludes the transfer
    itself and reported >100 Mbps over a 1 kbps link."""
    b = Bridge(tmp_path, rate_ab=512, poll_s=0.005).start()
    try:
        ci, cr, si, sr = make_sessions(b, tmp_path)
        out = {}
        th = run_responder(sr, out)
        res_i = initiate(si, "unidir", {"size": "1k", "payload": "prbs9"})
        th.join(20)
        assert res_i.outcome == "ok" and out["r"].outcome == "ok"
        far_dur = res_i.stats["far_dur_s"]
        assert far_dur >= 0.005                       # a real window, not noise
        assert res_i.stats["far_bytes"] * 8 / far_dur < 512 * 8 / 0.005
    finally:
        b.stop()


def test_connect_scenario(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    th = run_responder(sr, out)
    res_i = initiate(si, "connect", {})
    th.join(10)
    assert res_i.outcome == "ok" and out["r"].outcome == "ok"


def test_reverse_scenario(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    th = run_responder(sr, out)
    res_i = initiate(si, "reverse", {"size": "3k"})
    th.join(10)
    assert res_i.outcome == "ok" and out["r"].outcome == "ok"
    assert sr.tx_sha256 == si.rx_sha256
    assert si.payload_rx == 3072


def test_bidir_scenario(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    th = run_responder(sr, out)
    res_i = initiate(si, "bidir", {"size": "2k"})
    th.join(10)
    assert res_i.outcome == "ok" and out["r"].outcome == "ok"
    assert si.tx_sha256 == sr.rx_sha256
    assert sr.tx_sha256 == si.rx_sha256


def test_echo_scenario_integrity(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    th = run_responder(sr, out)
    res_i = initiate(si, "echo", {"size": "2k"})
    th.join(10)
    assert res_i.outcome == "ok" and out["r"].outcome == "ok"
    assert res_i.stats["mismatched_chunks"] == 0
    assert "turnaround_s" in res_i.stats
    assert si.tx_sha256 == sr.rx_sha256


# -- echo peer (self-loopback) ----------------------------------------------

def test_echo_peer_detection(tmp_path):
    eb = EchoBridge(tmp_path).start()
    try:
        c = eb.link("selftest")
        c.attach()
        eb.connect()
        wait_connected(c)
        s = CwpSession(c, c.transcript, sid="SELF", mycall="N0CAL",
                       recv_timeout_s=10.0)
        res = initiate(s, "unidir", {"size": "3k"})
        assert res.outcome == "echo_peer"
        assert res.stats["match"] is True
        events = [r.fields.get("text") for r in read(str(c.transcript.path))[0]
                  if r.kind == "hproto"]
        assert "echo_peer" in events
        confs = [r for r in read(str(c.transcript.path))[0] if r.kind == "conf"]
        assert any(f.fields.get("check") == "echo_integrity"
                   and f.fields.get("verdict") == "PASS" for f in confs)
    finally:
        eb.stop()


# -- refused -----------------------------------------------------------------

def test_refused_unknown_scenario(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    # the far end runs a build without this scenario
    th = run_responder(sr, out,
                       rewrite=lambda h: replace(h, scenario="nosuch"))
    res_i = initiate(si, "unidir", {"size": "1k"})
    th.join(10)
    assert res_i.outcome == "refused"
    assert out["r"].outcome == "refused"
    i_events = [r.fields for r in read(str(ci.transcript.path))[0]
                if r.kind == "hproto" and r.fields.get("text") == "hello_ack_rx"]
    assert i_events and i_events[0]["accept"] is False


# -- plain peer --------------------------------------------------------------

def test_plain_peer_initiator_disconnect(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    drained = {}

    def drain():
        drained["n"] = sr.drain_sink(idle_s=0.3, max_s=5.0)
    th = threading.Thread(target=drain, daemon=True)
    th.start()
    res = initiate(si, "unidir", {"size": "1k"}, on_plain_peer="disconnect",
                   ack_timeout_s=0.5)
    th.join(6)
    assert res.outcome == "plain_peer"
    assert res.stats["policy"] == "disconnect"
    assert "bytes" not in res.stats             # disconnect sends nothing extra


def test_plain_peer_initiator_blind(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    drained = {}

    def drain():
        drained["n"] = sr.drain_sink(idle_s=0.4, max_s=6.0)
    th = threading.Thread(target=drain, daemon=True)
    th.start()
    res = initiate(si, "unidir", {"size": "1k"}, on_plain_peer="blind",
                   ack_timeout_s=0.5)
    th.join(8)
    assert res.outcome == "plain_peer"
    assert res.stats["policy"] == "blind"
    assert res.stats["bytes"] == 1024           # raw payload pushed blind


def test_responder_not_cwp_sink(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    ci.send(b"just some plain winlink chatter, no CWP magic here at all")
    out = {}

    def go():
        try:
            sr.await_hello(timeout=1.0)
            out["r"] = "unexpected"
        except Exception as exc:
            out["exc"] = exc
            out["r"] = fallback_sink(sr, buffered=exc.buffered, idle_s=0.4)
    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(8)
    from creance.hproto import NotCwp
    assert isinstance(out["exc"], NotCwp)
    assert out["r"].outcome == "plain_peer"
    assert out["r"].stats["bytes"] >= len(b"just some plain")


# -- report missing ----------------------------------------------------------

def silence_the_report(monkeypatch):
    """Patch the shared registry so the responder's unidir half never sends
    its REPORT; the initiator half is untouched."""
    from creance.scenarios import Scenario, ScenarioResult

    def r_no_report(s, hello):
        end = s.await_end()
        return ScenarioResult("ok" if end is not None else "failed:end_missing",
                              {"bytes": s.payload_rx})
    reg = dict(SCENARIOS)
    reg["unidir"] = Scenario("unidir", SCENARIOS["unidir"].initiate, r_no_report)
    monkeypatch.setattr(scenarios, "SCENARIOS", reg)


def test_report_missing(bridge, tmp_path, monkeypatch):
    silence_the_report(monkeypatch)
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    th = run_responder(sr, out)
    res_i = initiate(si, "unidir", {"size": "1k", "report_timeout_s": 0.8})
    th.join(10)
    assert res_i.outcome == "report_missing"
    assert out["r"].outcome == "ok"            # far side still got the bytes
    assert si.tx_sha256 == sr.rx_sha256


# -- desync ------------------------------------------------------------------

def test_desync_midstream(tmp_path):
    b = Bridge(tmp_path, corrupt_at=3000).start()
    try:
        ci, cr, si, sr = make_sessions(b, tmp_path, i_recv=8.0, r_recv=8.0)
        out = {}
        th = run_responder(sr, out)
        res_i = initiate(si, "unidir", {"size": "10k", "report_timeout_s": 1.0})
        th.join(12)
        assert out["r"].outcome == "failed:cwp_desync"
        confs = [r for r in read(str(cr.transcript.path))[0]
                 if r.kind == "conf" and r.fields.get("check") == "cwp_desync"]
        assert confs and confs[0].fields["verdict"] == "FAIL"
        desyncs = [r for r in read(str(cr.transcript.path))[0]
                   if r.kind == "hproto" and r.fields.get("text") == "desync"]
        assert desyncs
        # initiator saw no REPORT, so it reports the missing report
        assert res_i.outcome in ("report_missing", "failed:tx_stall")
    finally:
        b.stop()


# -- cancellation ------------------------------------------------------------

def test_cancel_stops_the_fast_send_path(tmp_path):
    """When the modem drains faster than we fill, the high-water wait never
    runs — and an unchecked send path pushes the rest of the payload out after
    the watchdog gave up, straight into the next session's stream."""
    s, client, cancel = stub_session(tmp_path, cancel_after=2)
    with pytest.raises(SessionCancelled):
        s.send_frames_paced(b"\x00" * (16 * hproto.MAX_PAYLOAD), label="prbs9")
    assert len(client.writes) <= 3
    assert s.payload_tx <= 3 * hproto.MAX_PAYLOAD


def test_cancel_stops_the_raw_send_path(tmp_path):
    s, client, cancel = stub_session(tmp_path, cancel_after=1)
    with pytest.raises(SessionCancelled):
        s.send_raw_paced(b"\x00" * (16 * hproto.MAX_PAYLOAD), label="prbs9")
    assert len(client.writes) <= 2


def test_cancel_before_first_frame_sends_nothing(tmp_path):
    s, client, cancel = stub_session(tmp_path)
    cancel.set()
    with pytest.raises(SessionCancelled):
        s.send_data_frame(b"payload", label="prbs9")
    assert client.writes == []


def test_bidir_tx_thread_does_not_outlive_the_session(bridge, tmp_path):
    ci, cr, si, sr = make_sessions(bridge, tmp_path)
    out = {}
    th = run_responder(sr, out)
    initiate(si, "bidir", {"size": "8k"})
    th.join(10)
    live = [t for t in threading.enumerate() if t.name.endswith("-bidir-tx")]
    assert live == []


# -- desync salvage ----------------------------------------------------------

def test_end_in_a_damaged_feed_is_still_delivered(tmp_path):
    end = End(sha256="ab" * 32, bytes=8, dur_s=2.0)
    stream = hproto.pack(hproto.DATA, b"12345678") + end.pack() + b"garbage!!"
    s, client, _ = stub_session(tmp_path, rx=[stream])
    assert s.await_end() == end          # not swallowed with the damage
    assert s.desync is not None
    events = [r.fields.get("text") for r in read(str(client.transcript.path))[0]
              if r.kind == "hproto"]
    assert "desync" in events            # and the defect is still on record
    with pytest.raises(Desync):          # next read re-raises; no resync
        s.recv_frame(timeout=0.1)


def test_salvaged_end_still_fails_the_session(tmp_path):
    from creance.hproto import Hello
    hello = Hello(sid="SID-R", call="N0BBB", scenario="unidir")
    end = End(sha256="ab" * 32, bytes=8, dur_s=2.0)
    stream = hproto.pack(hproto.DATA, b"12345678") + end.pack() + b"garbage!!"
    s, client, _ = stub_session(tmp_path, rx=[stream])
    res = respond(s, hello)
    assert res.outcome == "failed:cwp_desync"


# -- duration granularity ----------------------------------------------------

@pytest.mark.realtime
def test_duration_is_honoured_at_frame_granularity():
    """A slow link must stop one frame past the deadline, not one whole block:
    at block granularity a --duration run overruns by minutes."""
    block = b"\x00" * (32 * hproto.MAX_PAYLOAD)      # 128 KiB: 32 frames
    per_chunk = 0.01
    t0 = time.monotonic()
    sent = 0
    for _ in scenarios._stream(block, 0.05):
        time.sleep(per_chunk)
        sent += 1
    elapsed = time.monotonic() - t0
    assert 0 < sent < 32                             # stopped mid-block
    assert elapsed < 0.05 + 4 * per_chunk            # not 32 * per_chunk


# -- backpressure ------------------------------------------------------------

def test_backpressure_high_water_respected(tmp_path):
    high = 8192
    b = Bridge(tmp_path, rate_ab=1024, poll_s=0.003).start()
    try:
        ci, cr, si, sr = make_sessions(b, tmp_path, i_high=high,
                                       i_recv=20.0, r_recv=20.0)
        out = {}
        th = run_responder(sr, out)
        res_i = initiate(si, "unidir", {"size": "24k",
                                        "report_timeout_s": 10.0})
        th.join(25)
        assert res_i.outcome == "ok"
        assert si.tx_sha256 == sr.rx_sha256
        one_frame = 4096 + 11
        assert b.peak["ab"] <= high + 2 * one_frame, b.peak
    finally:
        b.stop()
