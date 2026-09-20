# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""run_session end-to-end: a real initiator half against a real responder half
driven over the bridge, plus the failure exits (connect timeout, report
missing) — all producing a session.json."""

import json
import threading
import time

import pytest

from creance import scenarios
from creance.config import (Config, InitiatorConfig, ModemConfig, Quirks,
                            ResponderConfig, SiteConfig)
from creance.initiator import run_session
from creance.metrics import SessionMetrics
from creance.scenarios import CwpSession, SCENARIOS, Scenario, ScenarioResult, respond

from hfhost.testing.bridge import Bridge, disconnect_responder


def make_config(bridge, *, sim_time=False, on_plain_peer="disconnect",
                connect_timeout_s=5.0, max_session_s=20.0):
    modem = ModemConfig(name="a", cmd_port=bridge.a.cmd_port,
                        data_port=bridge.a.data_port, sim_time=sim_time,
                        quirks=Quirks(version_reply=True, iamalive_s=0.0))
    return Config(
        site=SiteConfig(name="home", mycall="N0AAA", results_dir="results"),
        responder=ResponderConfig(max_session_s=20.0),
        initiator=InitiatorConfig(connect_timeout_s=connect_timeout_s,
                                  max_session_s=max_session_s,
                                  on_plain_peer=on_plain_peer),
        modems=(modem,))


def start_responder(bridge, tmp_path, *, sid="SID-R"):
    """Attach a responder client to modem B and run its CWP half once the
    initiator connects. Returns (thread, result-holder)."""
    out: dict = {}

    def go():
        cr = bridge.link("b", "resp")
        cr.attach()
        try:
            deadline = time.monotonic() + 10.0
            while not cr.connected and time.monotonic() < deadline:
                time.sleep(0.01)
            sr = CwpSession(cr, cr.transcript, sid=sid, mycall="N0BBB",
                            recv_timeout_s=15.0)
            h = sr.await_hello(timeout=10.0)
            if h is None:
                out["r"] = ScenarioResult("no_hello", {})
                return
            out["r"] = respond(sr, h)
        except Exception as exc:
            out["exc"] = exc
        finally:
            cr.close()

    th = threading.Thread(target=go, name="responder", daemon=True)
    th.start()
    return th, out


def drive_connect(bridge, delay=0.3):
    """Emit CONNECTED to both ends shortly after the initiator's CONNECT.

    The helper deliberately outlives tests that never issue a CONNECT — it waits
    up to 8 s. If the fixture tears the bridge down first, ``bridge.connect()``
    raises in this daemon thread, and pytest attributes the unraisable exception
    to whichever test happens to be running when it lands. That produced a roving
    one-test-per-run failure across unrelated modules for weeks. The thread has no
    one to report to once its bridge is gone, so it stops quietly."""
    def go():
        try:
            bridge.a.wait_command("CONNECT", timeout=8.0)
            time.sleep(delay)
            bridge.connect()
        except Exception:
            return
    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


@pytest.fixture
def bridge(tmp_path):
    b = Bridge(tmp_path).start()
    b.a.on_command = disconnect_responder(b.a)   # initiator's teardown wait ends
    b.b.on_command = disconnect_responder(b.b)
    yield b
    b.stop()


def test_run_session_unidir_ok(bridge, tmp_path):
    cfg = make_config(bridge)
    th_r, out = start_responder(bridge, tmp_path)
    drive_connect(bridge)
    m = run_session(cfg, "a", "K7XYZ", "unidir",
                    {"size": "4k", "payload": "prbs9"},
                    label="run1", results_root=tmp_path / "results")
    th_r.join(15)

    assert m.outcome == "ok"
    assert out["r"].outcome == "ok"
    assert m.scenario == "unidir" and m.label == "run1"
    assert m.version == "A-1.0"
    assert m.connect_s is not None
    assert m.handshake_s is not None
    assert m.goodput_bps_far is not None

    sid_dir = tmp_path / "results" / "home" / time.strftime("%Y-%m-%d") / m.sid
    assert (sid_dir / "session.json").is_file()
    obj = json.loads((sid_dir / "session.json").read_text())
    assert obj["meta"]["outcome"] == "ok"
    assert obj["meta"]["sim_time"] is False
    # DISCONNECT was issued
    assert bridge.a.wait_command("DISCONNECT", timeout=1.0) is not None


def test_peer_identity_joins_the_two_sites_records(bridge, tmp_path):
    """Each end records the other's sid and call. Nothing else joins the two
    session.json files: the sites share no clock, so wall times cannot."""
    cfg = make_config(bridge)
    th_r, out = start_responder(bridge, tmp_path, sid="SID-R")
    drive_connect(bridge)
    m = run_session(cfg, "a", "K7XYZ", "unidir", {"size": "1k"},
                    results_root=tmp_path / "results")
    th_r.join(15)
    assert m.outcome == "ok"

    sid_dir = tmp_path / "results" / "home" / time.strftime("%Y-%m-%d") / m.sid
    meta = json.loads((sid_dir / "session.json").read_text())["meta"]
    assert meta["peer_sid"] == "SID-R" and meta["peer_call"] == "N0BBB"

    # the far site's record of the same exchange, as its responder wrote it
    far = SessionMetrics(sid="SID-R", site="remote", peer_sid=m.sid,
                         peer_call=cfg.site.mycall)
    assert far.peer_sid == meta["sid"] and meta["peer_sid"] == far.sid


def test_run_session_sim_time_tag(bridge, tmp_path):
    cfg = make_config(bridge, sim_time=True)
    th_r, out = start_responder(bridge, tmp_path)
    drive_connect(bridge)
    m = run_session(cfg, "a", "K7XYZ", "unidir", {"size": "2k"},
                    results_root=tmp_path / "results")
    th_r.join(15)
    assert m.outcome == "ok"
    assert m.sim_time is True
    assert "sim_time" in m.suppressed
    assert m.goodput_bps_far is None            # suppressed on simulated air


def test_run_session_connect_timeout(bridge, tmp_path):
    cfg = make_config(bridge, connect_timeout_s=0.6)
    # no drive_connect: CONNECTED never arrives
    m = run_session(cfg, "a", "K7XYZ", "unidir", {"size": "1k"},
                    results_root=tmp_path / "results")
    assert m.outcome == "failed:connect_timeout"
    sid_dir = tmp_path / "results" / "home" / time.strftime("%Y-%m-%d") / m.sid
    assert (sid_dir / "session.json").is_file()
    assert bridge.a.wait_command("ABORT", timeout=2.0) is not None


@pytest.mark.realtime
def test_run_session_watchdog_caps_a_silent_peer(bridge, tmp_path):
    """A peer that connects and then says nothing: without a session cap the
    initiator waits out one recv timeout per frame, forever."""
    cfg = make_config(bridge, max_session_s=3.0)
    drive_connect(bridge)
    t0 = time.monotonic()
    m = run_session(cfg, "a", "K7XYZ", "unidir",
                    {"size": "1k", "report_timeout_s": 300.0},
                    results_root=tmp_path / "results")
    elapsed = time.monotonic() - t0
    assert m.outcome == "aborted:watchdog"
    assert elapsed < 20.0                  # capped, not 300 s of report wait
    sid_dir = tmp_path / "results" / "home" / time.strftime("%Y-%m-%d") / m.sid
    assert (sid_dir / "session.json").is_file()


@pytest.mark.realtime
def test_run_session_caller_budget_overrides_config(bridge, tmp_path):
    cfg = make_config(bridge, max_session_s=600.0)
    drive_connect(bridge)
    t0 = time.monotonic()
    m = run_session(cfg, "a", "K7XYZ", "unidir",
                    {"size": "1k", "report_timeout_s": 300.0},
                    results_root=tmp_path / "results", max_session_s=3.0)
    assert m.outcome == "aborted:watchdog"
    assert time.monotonic() - t0 < 20.0


def silence_the_report(monkeypatch):
    """Patch the shared registry so the responder's unidir half never sends
    its REPORT; the initiator half is untouched."""
    def r_no_report(s, hello):
        end = s.await_end()
        return ScenarioResult("ok" if end is not None else "failed:end_missing",
                              {"bytes": s.payload_rx})
    reg = dict(SCENARIOS)
    reg["unidir"] = Scenario("unidir", SCENARIOS["unidir"].initiate, r_no_report)
    monkeypatch.setattr(scenarios, "SCENARIOS", reg)


def test_run_session_report_missing_still_disconnects(bridge, tmp_path,
                                                     monkeypatch):
    silence_the_report(monkeypatch)
    cfg = make_config(bridge)
    th_r, out = start_responder(bridge, tmp_path)
    drive_connect(bridge)
    m = run_session(cfg, "a", "K7XYZ", "unidir",
                    {"size": "1k", "report_timeout_s": 0.8},
                    results_root=tmp_path / "results")
    th_r.join(15)
    assert m.outcome == "report_missing"
    assert bridge.a.wait_command("DISCONNECT", timeout=1.0) is not None
    sid_dir = tmp_path / "results" / "home" / time.strftime("%Y-%m-%d") / m.sid
    assert (sid_dir / "session.json").is_file()
