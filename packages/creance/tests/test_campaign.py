# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Campaign runner: plan validation, --all derivation, continue-past-failure
over real bridged sessions, and the deadline early-stop."""

import os
import socket
import threading
import time
from dataclasses import replace

import pytest

from creance.campaign import Run, derive_all, load_plan, run_campaign
from creance.config import (Config, ConfigError, InitiatorConfig, ModemConfig,
                            Quirks, ResponderConfig, SiteConfig)
from creance.scenarios import CwpSession, respond

from hfhost.testing.bridge import Bridge, disconnect_responder


def _dead_ports() -> tuple[int, int]:
    ports = []
    for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        s.close()
    return ports[0], ports[1]


def make_config(bridge, dead, *, connect_timeout_s=5.0, max_session_s=15.0):
    live = ModemConfig(name="a", cmd_port=bridge.a.cmd_port,
                       data_port=bridge.a.data_port,
                       quirks=Quirks(iamalive_s=0.0))
    deadm = ModemConfig(name="dead", cmd_port=dead[0], data_port=dead[1],
                        quirks=Quirks(iamalive_s=0.0))
    return Config(
        site=SiteConfig(name="home", mycall="N0AAA"),
        responder=ResponderConfig(max_session_s=15.0),
        initiator=InitiatorConfig(connect_timeout_s=connect_timeout_s,
                                  max_session_s=max_session_s),
        modems=(live, deadm))


class PeerService:
    """Services `n` sequential bridged CWP sessions on modem B. Per session it
    attaches a fresh B client, waits for the next CONNECT the initiator issues
    on A, then emits CONNECTED to both ends — so CONNECTED never lands on a
    stale B session — and runs the responder half."""

    def __init__(self, bridge, tmp_path, n):
        self.bridge = bridge
        self.tmp_path = tmp_path
        self.n = n
        self.results = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _connects(self):
        return sum(1 for c in list(self.bridge.a.commands)
                   if c.startswith("CONNECT"))

    def _run(self):
        seen = 0
        for i in range(self.n):
            if self._stop.is_set():
                return
            cr = self.bridge.link("b", f"resp{i}")
            try:
                cr.attach()
            except Exception:
                continue
            deadline = time.monotonic() + 15.0
            while self._connects() <= seen and time.monotonic() < deadline \
                    and not self._stop.is_set():
                time.sleep(0.02)
            seen += 1
            self.bridge.a.notify("CONNECTED N0AAA K7XYZ 2300")
            self.bridge.b.notify("CONNECTED N0BBB N0AAA 2300")
            while not cr.connected and time.monotonic() < deadline \
                    and not self._stop.is_set():
                time.sleep(0.01)
            sr = CwpSession(cr, cr.transcript, sid=f"R{i}", mycall="N0BBB",
                            recv_timeout_s=15.0)
            try:
                h = sr.await_hello(timeout=10.0)
                if h is not None:
                    self.results.append(respond(sr, h).outcome)
            except Exception:
                pass
            time.sleep(0.25)             # let the bridge forward the REPORT
            cr.close()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# -- plan validation ---------------------------------------------------------

def test_load_plan_ok(tmp_path):
    (tmp_path / "plan.toml").write_text(
        '[[run]]\nmodem="a"\ndst="K7XYZ"\nscenario="unidir"\n'
        '[run.params]\nsize="10k"\n'
        '[[run]]\nmodem="a"\ndst="K7XYZ"\nscenario="connect"\nrepeat=2\n')
    runs = load_plan(tmp_path / "plan.toml")
    assert len(runs) == 2
    assert runs[0].params == {"size": "10k"}
    assert runs[1].repeat == 2


@pytest.mark.parametrize("body,match", [
    ('[[run]]\ndst="X"\nscenario="unidir"\n', "missing modem"),
    ('[[run]]\nmodem="a"\nscenario="unidir"\n', "missing dst"),
    ('[[run]]\nmodem="a"\ndst="X"\n', "missing scenario"),
    ('[[run]]\nmodem="a"\ndst="X"\nscenario="u"\nbogus=1\n', "unknown key"),
    ('[[run]]\nmodem="a"\ndst="X"\nscenario="u"\nparams=5\n', "params must be a table"),
    ('[other]\nx=1\n', "no \\[\\[run\\]\\] entries"),
])
def test_load_plan_errors(tmp_path, body, match):
    (tmp_path / "plan.toml").write_text(body)
    with pytest.raises(ConfigError, match=match):
        load_plan(tmp_path / "plan.toml")


def test_derive_all(tmp_path):
    bridge = Bridge(tmp_path)
    try:
        cfg = make_config(bridge, _dead_ports())
        runs = derive_all(cfg, "K7XYZ")
        assert len(runs) == 2 * 4              # two modems x four sweep points
        scns = {(r.scenario, tuple(sorted(r.params.items()))) for r in runs}
        assert ("connect", ()) in scns
        assert ("unidir", (("size", "1k"),)) in scns
        assert ("echo", (("size", "10k"),)) in scns
        # a cool-down, not back-to-back CONNECTs at a far end still draining
        assert all(r.interval_s > 0 for r in runs)
        with pytest.raises(ConfigError, match="requires a dst"):
            derive_all(cfg, "")
    finally:
        bridge.stop()


# -- execution ---------------------------------------------------------------

def test_campaign_continues_past_failure(tmp_path):
    bridge = Bridge(tmp_path).start()
    bridge.a.on_command = disconnect_responder(bridge.a)
    bridge.b.on_command = disconnect_responder(bridge.b)
    try:
        cfg = make_config(bridge, _dead_ports())
        peer = PeerService(bridge, tmp_path, n=2).start()
        plan = [
            Run(modem="a", dst="K7XYZ", scenario="unidir", params={"size": "2k"}),
            Run(modem="dead", dst="K7XYZ", scenario="unidir", params={"size": "2k"}),
            Run(modem="a", dst="K7XYZ", scenario="unidir", params={"size": "2k"}),
        ]
        metrics, text = run_campaign(cfg, plan, label="camp",
                                     results_root=tmp_path / "results")
        peer.stop()

        assert len(metrics) == 3
        assert metrics[0].outcome == "ok"
        assert metrics[1].outcome.startswith("failed")     # dead modem
        assert metrics[2].outcome == "ok"

        day = time.strftime("%Y-%m-%d")
        sid_dirs = list((tmp_path / "results" / "home" / day).glob("*/session.json"))
        assert len(sid_dirs) == 3
        campaign_txt = list((tmp_path / "results" / "home" / day).glob("campaign-*.txt"))
        assert len(campaign_txt) == 1
        assert campaign_txt[0].read_text() == text
        assert "unidir" in text
    finally:
        bridge.stop()


def test_campaign_enforces_retention_at_an_initiator_only_site(tmp_path):
    """No responder daemon runs here, so run_campaign is the only thing that
    ever sweeps this tree."""
    bridge = Bridge(tmp_path).start()
    bridge.a.on_command = disconnect_responder(bridge.a)
    bridge.b.on_command = disconnect_responder(bridge.b)
    try:
        cfg = make_config(bridge, _dead_ports())
        cfg = replace(cfg, responder=replace(cfg.responder,
                                             retention_max_age_s=1.0))
        results = tmp_path / "results"
        stale = results / "home" / "2020-01-01" / "20200101T000000-a-dead"
        stale.mkdir(parents=True)
        (stale / "transcript.jsonl").write_text("", encoding="utf-8")
        os.utime(stale / "transcript.jsonl", (0, 0))

        peer = PeerService(bridge, tmp_path, n=1).start()
        plan = [Run(modem="a", dst="K7XYZ", scenario="unidir",
                    params={"size": "1k"})]
        metrics, _ = run_campaign(cfg, plan, results_root=results)
        peer.stop()

        assert metrics[0].outcome == "ok"
        assert not stale.exists()                    # swept by age
        day = time.strftime("%Y-%m-%d")
        assert list((results / "home" / day).glob("*/session.json"))
    finally:
        bridge.stop()


def test_campaign_deadline_stops_early(tmp_path):
    bridge = Bridge(tmp_path).start()
    try:
        cfg = make_config(bridge, _dead_ports())
        plan = [Run(modem="dead", dst="K7XYZ", scenario="connect")
                for _ in range(3)]
        # each dead run fails fast (refused ports); a big deadline runs all 3
        metrics_all, _ = run_campaign(cfg, plan,
                                      results_root=tmp_path / "results")
        assert len(metrics_all) == 3
        assert all(m.outcome.startswith("failed") for m in metrics_all)

        # a spent deadline stops after the first
        metrics_short, _ = run_campaign(cfg, plan, deadline_s=-1.0,
                                        results_root=tmp_path / "results2")
        assert len(metrics_short) == 0
    finally:
        bridge.stop()


@pytest.mark.realtime
def test_campaign_deadline_bounds_a_hanging_session(tmp_path):
    """The deadline used to be checked only between runs, so one session that
    never returned held an unattended overnight matrix open forever."""
    bridge = Bridge(tmp_path).start()
    try:
        # a live modem that connects but never answers: the session would
        # otherwise sit out the full connect timeout, twice over
        cfg = make_config(bridge, _dead_ports(), connect_timeout_s=120.0,
                          max_session_s=120.0)
        plan = [Run(modem="a", dst="K7XYZ", scenario="unidir",
                    params={"size": "1k"}) for _ in range(2)]
        t0 = time.monotonic()
        metrics, _ = run_campaign(cfg, plan, deadline_s=2.0,
                                  results_root=tmp_path / "results")
        elapsed = time.monotonic() - t0
        assert elapsed < 60.0                  # not 2 x 120 s
        assert metrics and metrics[0].outcome.startswith("aborted")
    finally:
        bridge.stop()
