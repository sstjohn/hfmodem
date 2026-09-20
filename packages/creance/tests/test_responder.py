# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Responder daemon state machine, driven against real FakeModems.

Every case here is a state-machine test: arbitration outcomes, worker
lifecycle, epoch fencing between sessions, and the recovery paths that keep an
unattended site alive.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from creance import hproto
from creance import responder as R
from creance.config import (Config, InitiatorConfig, ModemConfig, Quirks,
                            ResponderConfig, SiteConfig)
from creance.responder import Responder, sweep_results
from creance.scenarios import CwpSession, initiate
from hfhost.transcript import Transcript
from hfhost.transcript import read as read_transcript

from hfhost.testing.bridge import Bridge
from hfhost.testing.fakemodem import FakeModem


def wait(pred, timeout=8.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


def _reopened_small(path):
    try:
        return path.stat().st_size < R.TRANSCRIPT_CAP
    except FileNotFoundError:
        return False            # mid-rotation: replaced, not yet reopened


class FakeProc:
    def __init__(self, alive=True):
        self.alive = alive


class FakeSupervisor:
    """Records policy calls; process_for() answers with FakeProcs so the
    attach-failure escalation path can be exercised without spawning."""

    def __init__(self):
        self.procs = {}
        self.ensured = []
        self.acquired = []
        self.released = []
        self.respawned = []
        self.activity = []
        self.recycle_due = set()
        self.stopped = False

    def ensure(self, name, now=None):
        self.ensured.append(name)
        return True

    def acquire(self, name):
        self.acquired.append(name)
        return self.procs.get(name)

    def release(self, name):
        self.released.append(name)

    def process_for(self, name):
        return self.procs.get(name)

    def kill_and_respawn(self, name):
        self.respawned.append(name)

    def note_activity(self, name):
        self.activity.append(name)

    def idle_recycle_due(self, name, now=None):
        if name in self.recycle_due:      # one-shot: the recycle resets the timer
            self.recycle_due.discard(name)
            return True
        return False

    def stop_all(self):
        self.stopped = True


class Daemon:
    """A Responder wired to FakeModems and running its loop in a thread."""

    def __init__(self, tmp_path, modems, *, pending_timeout_s=0.4,
                 hello_timeout_s=0.4, max_session_s=5.0, on_plain_peer="sink",
                 quirks=None, retention_max_age_s=R.RETENTION_MAX_AGE_S,
                 retention_max_bytes=R.RETENTION_MAX_BYTES, **kw):
        quirks = quirks or Quirks(version_reply=True, iamalive_s=0.0)
        cfg = Config(
            site=SiteConfig(name="home", mycall="N0RSP",
                            results_dir=str(tmp_path / "results")),
            responder=ResponderConfig(pending_timeout_s=pending_timeout_s,
                                      hello_timeout_s=hello_timeout_s,
                                      max_session_s=max_session_s,
                                      on_plain_peer=on_plain_peer,
                                      retention_max_age_s=retention_max_age_s,
                                      retention_max_bytes=retention_max_bytes,
                                      retention_interval_s=0.0),
            initiator=InitiatorConfig(),
            modems=tuple(
                ModemConfig(name=name, cmd_port=fm.cmd_port,
                            data_port=fm.data_port, quirks=quirks)
                for name, fm in modems))
        self.config = cfg
        self.results = tmp_path / "results"
        self.sup = FakeSupervisor()
        kw.setdefault("tick_s", 0.05)
        kw.setdefault("retry_s", 0.05)
        kw.setdefault("attach_timeout_s", 2.0)
        kw.setdefault("join_timeout_s", 2.0)
        kw.setdefault("settle_s", 0.2)
        kw.setdefault("sink_idle_s", 0.25)
        kw.setdefault("attach_fail_limit", 2)
        self.r = Responder(cfg, results_root=self.results, supervisor=self.sup, **kw)
        self._thread = None

    def start(self):
        self.r.start()
        self._thread = threading.Thread(target=self.r.run_forever, daemon=True)
        self._thread.start()
        return self.r

    def stop(self):
        self.r.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def session_dir(self, sid):
        return self.results / "home" / time.strftime("%Y-%m-%d") / sid

    def session_json(self, sid):
        return json.loads((self.session_dir(sid) / "session.json").read_text())

    def daemon_records(self):
        return read_transcript(str(self.results / "home" / "responder.jsonl"))[0]


@pytest.fixture
def modem():
    fm = FakeModem(version="fake-1.2").start()
    yield fm
    fm.stop()


@pytest.fixture
def modem2():
    fm = FakeModem(version="fake-2.0").start()
    yield fm
    fm.stop()


@pytest.fixture
def daemon(tmp_path):
    made = []

    def make(modems, **kw):
        d = Daemon(tmp_path, modems, **kw)
        made.append(d)
        d.start()
        return d

    yield make
    for d in made:
        d.stop()


def connect(fm, dst="K7XYZ"):
    fm.notify(f"CONNECTED N0RSP {dst} 2300")


def hello(sid="S1", scenario="unidir", params=None):
    return hproto.Hello(sid=sid, call="K7XYZ", scenario=scenario,
                        params=params or {}).pack()


# -- INIT ---------------------------------------------------------------------

def test_init_attaches_with_desired_state(daemon, modem):
    d = daemon([("a", modem)])
    assert d.r.state == R.LISTEN_ALL
    assert d.r.modem_states == {"a": "up"}
    assert modem.session_log[0][:4] == ["MYCALL N0RSP", "BW2300",
                                        "COMPRESSION OFF", "LISTEN ON"]
    assert d.sup.acquired == ["a"]
    assert d.r._links["a"].version == "fake-1.2"


def test_init_runs_with_surviving_subset(tmp_path, modem):
    dead = FakeModem(data_listener=False).start()
    try:
        d = Daemon(tmp_path, [("a", modem), ("dead", dead)])
        d.start()
        try:
            assert d.r.modem_states == {"a": "up", "dead": "down"}
            connect(modem)
            assert wait(lambda: d.r.state == R.SESSION)
        finally:
            d.stop()
    finally:
        dead.stop()


# -- ARBITRATING --------------------------------------------------------------

def test_spurious_pending_then_cancelpending(daemon, modem, modem2):
    d = daemon([("a", modem), ("b", modem2)])
    modem.notify("PENDING")
    assert wait(lambda: d.r.state == R.ARBITRATING)
    assert set(d.r.pending) == {"a"}
    modem.notify("CANCELPENDING")
    assert wait(lambda: d.r.state == R.LISTEN_ALL)
    assert d.r.pending == {}
    assert "LISTEN OFF" not in modem2.commands      # PENDING disturbs nobody
    assert d.r.sessions_completed == 0


def test_pending_times_out_back_to_listen_all(daemon, modem):
    d = daemon([("a", modem)], pending_timeout_s=0.15)
    modem.notify("PENDING")
    assert wait(lambda: d.r.state == R.ARBITRATING)
    assert wait(lambda: d.r.state == R.LISTEN_ALL, timeout=3.0)
    assert d.r.pending == {}


def test_dual_pending_one_connected_wins(daemon, modem, modem2):
    d = daemon([("a", modem), ("b", modem2)])
    modem.notify("PENDING")
    modem2.notify("PENDING")
    assert wait(lambda: set(d.r.pending) == {"a", "b"})
    assert d.r.state == R.ARBITRATING

    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    assert d.r.winner == "a"
    assert d.r.pending == {}

    # the loser is muted; the winner is never touched
    assert wait(lambda: "LISTEN OFF" in modem2.commands)
    assert "LISTEN OFF" not in modem.commands
    assert modem2.commands == ["MYCALL N0RSP", "BW2300", "COMPRESSION OFF",
                               "LISTEN ON", "VERSION", "LISTEN OFF"]

    # session dies of its own hello timeout; LISTEN comes back on everywhere
    assert wait(lambda: d.r.state == R.LISTEN_ALL, timeout=5.0)
    assert modem2.commands[-1] == "LISTEN ON"


def test_connected_without_pending_starts_session(daemon, modem):
    d = daemon([("a", modem)])
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    assert d.r.pending == {}
    assert wait(lambda: d.r.sessions_completed == 1, timeout=5.0)


def test_second_connected_gets_abort_and_warning(daemon, modem, modem2):
    d = daemon([("a", modem), ("b", modem2)], max_session_s=3.0)
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    sid = d.r._sid
    connect(modem2)
    assert wait(lambda: "ABORT" in modem2.commands)
    assert d.r.winner == "a"                       # first in the queue wins
    assert "ABORT" not in modem.commands

    assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
    findings = [r.fields for r in read_transcript(
        str(d.session_dir(sid) / "transcript.jsonl"))[0] if r.kind == "conf"]
    warn = [f for f in findings if f["check"] == "double_connect"]
    assert warn and warn[0]["verdict"] == "WARN"


# -- SESSION ------------------------------------------------------------------

def test_cwp_unidir_end_to_end(tmp_path):
    bridge = Bridge(tmp_path).start()
    try:
        d = Daemon(tmp_path, [("b", bridge.b)], hello_timeout_s=5.0,
                   max_session_s=20.0)
        d.start()
        try:
            init = bridge.link("a", "init")
            init.configure("N0AAA")
            init.attach()
            bridge.connect(a_call="N0AAA", b_call="N0RSP")
            assert wait(lambda: d.r.state == R.SESSION)

            sess = CwpSession(init, init.transcript, sid="SID-I",
                              mycall="N0AAA", recv_timeout_s=20.0)
            result = initiate(sess, "unidir", {"size": "4k", "payload": "prbs9"})
            assert result.outcome == "ok"
            bridge.b.notify("DISCONNECTED")

            assert wait(lambda: d.r.sessions_completed == 1, timeout=15.0)
            m = d.r.last_metrics
            assert m.outcome == "ok"
            assert m.scenario == "unidir"
            assert m.params == {"size": "4k", "payload": "prbs9"}
            assert m.bytes_rx >= 4096
            assert m.handshake_s is not None
            assert m.end_sha_match is True

            obj = d.session_json(m.sid)
            assert obj["meta"]["outcome"] == "ok"
            assert obj["meta"]["modem"] == "b"
            assert obj["meta"]["site"] == "home"
            assert obj["metrics"]["bytes_rx"] == m.bytes_rx
            init.close()
        finally:
            d.stop()
    finally:
        bridge.stop()


def test_plain_peer_falls_back_to_sink(daemon, modem):
    d = daemon([("a", modem)])
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    modem.send_data(b"HELO pat, this is not CWP\r\n")

    assert wait(lambda: d.r.sessions_completed == 1, timeout=5.0)
    m = d.r.last_metrics
    assert m.outcome == "plain_peer"
    assert m.bytes_rx == 27
    assert d.session_json(m.sid)["meta"]["outcome"] == "plain_peer"
    assert d.r.state == R.LISTEN_ALL


def test_short_read_then_disconnect_records_cleanly(daemon, modem):
    d = daemon([("a", modem)], hello_timeout_s=5.0, max_session_s=20.0)
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    modem.send_data(b"CR")                  # <4 bytes: never resolves to CWP
    modem.notify("DISCONNECTED")

    assert wait(lambda: d.r.sessions_completed == 1, timeout=5.0)
    m = d.r.last_metrics
    assert m.outcome == "failed:link_lost"
    assert m.bytes_rx == 2
    assert (d.session_dir(m.sid) / "session.json").is_file()
    assert d.r.state == R.LISTEN_ALL
    assert not d.r.worker_alive


def test_per_modem_hello_timeout_overrides_the_default(daemon, modem):
    # a PACTOR-class modem gets a generous hello budget, a fast one a short
    # one; the per-modem quirk wins over [responder]
    d = daemon([("a", modem)], hello_timeout_s=30.0, max_session_s=60.0,
               quirks=Quirks(version_reply=True, iamalive_s=0.0,
                             hello_timeout_s=0.2))
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    assert wait(lambda: d.r.sessions_completed == 1, timeout=5.0)
    assert d.r.last_metrics.outcome == "plain_peer"


def test_plain_peer_disconnect_policy_hangs_up(daemon, modem):
    d = daemon([("a", modem)], on_plain_peer="disconnect")
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    modem.send_data(b"not cwp at all")
    assert wait(lambda: "DISCONNECT" in modem.commands, timeout=5.0)
    assert wait(lambda: d.r.sessions_completed == 1, timeout=5.0)
    assert d.r.last_metrics.outcome == "plain_peer"


def test_disconnected_outside_session_is_ignored(daemon, modem):
    d = daemon([("a", modem)])
    modem.notify("DISCONNECTED")             # kestrel emits these when idle
    modem.notify("DISCONNECTED")
    time.sleep(0.2)
    assert d.r.state == R.LISTEN_ALL
    assert d.r.sessions_completed == 0
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)


# -- worker lifecycle ---------------------------------------------------------

def _unidir_exchange(fm, payload, sid="S2"):
    fm.send_data(hello(sid=sid))
    fm.send_data(hproto.pack(hproto.DATA, payload))
    fm.send_data(hproto.End(sha256=hashlib.sha256(payload).hexdigest(),
                            bytes=len(payload), dur_s=1.0).pack())
    return (len(hello(sid=sid)) + len(hproto.pack(hproto.DATA, payload))
            + len(hproto.End(sha256=hashlib.sha256(payload).hexdigest(),
                             bytes=len(payload), dur_s=1.0).pack()))


def test_watchdog_kills_blocked_worker_and_next_session_is_clean(daemon, modem):
    d = daemon([("a", modem)], hello_timeout_s=2.0, max_session_s=1.0)
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    # HELLO late in the session budget, then silence: when the watchdog fires
    # the worker is deep inside a blocking read with plenty of time left
    time.sleep(0.4)
    modem.send_data(hello(sid="S1"))
    assert wait(lambda: d.r.worker_alive and modem.received_data)
    assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)

    first = d.r.last_metrics
    assert first.outcome == "aborted:watchdog"
    assert "ABORT" in modem.commands
    assert not d.r.worker_alive               # no zombie holding the rx buffer
    assert d.r.state == R.LISTEN_ALL

    # the dead session's peer dribbles in late; it must never reach session 2
    modem.send_data(b"ZOMBIE-LATE-BYTES")
    assert wait(lambda: any(
        r.fields.get("check") == "data_outside_session"
        for r in d.daemon_records()), timeout=3.0)

    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    payload = b"payload!" * 16
    sent = _unidir_exchange(modem, payload)
    assert wait(lambda: d.r.sessions_completed == 2, timeout=6.0)

    second = d.r.last_metrics
    assert second.sid != first.sid
    assert second.outcome == "ok"             # contaminated bytes would desync
    assert second.bytes_rx == sent
    assert second.end_sha_match is True


def test_late_data_after_disconnect_is_fenced_out_of_session(daemon, modem):
    d = daemon([("a", modem)])
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    modem.send_data(b"zzzz")
    modem.notify("DISCONNECTED")
    assert wait(lambda: d.r.sessions_completed == 1, timeout=5.0)
    first = d.r.last_metrics
    assert first.bytes_rx == 4

    modem.send_data(b"LATELATE")              # after the epoch fence
    assert wait(lambda: any(
        r.fields.get("check") == "data_outside_session"
        and r.fields.get("bytes") == 8 for r in d.daemon_records()), timeout=3.0)

    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    modem.send_data(b"abcd")
    modem.notify("DISCONNECTED")
    assert wait(lambda: d.r.sessions_completed == 2, timeout=5.0)
    second = d.r.last_metrics
    assert second.bytes_rx == 4               # the late 8 bytes never bled in


def _hand_pumped(tmp_path, modem, **kw):
    """A daemon whose loop is not running: events pile up in the queue so a
    tick can be dispatched by hand, exactly in the window the sweep used to
    lose a HELLO in."""
    kw.setdefault("hello_timeout_s", 5.0)
    kw.setdefault("max_session_s", 20.0)
    d = Daemon(tmp_path, [("a", modem)], **kw)
    d.r.start()
    return d


def _no_data_outside_session(d):
    return not [r for r in d.daemon_records()
                if r.fields.get("check") == "data_outside_session"]


def test_tick_does_not_eat_a_hello_that_beat_the_connected_event(tmp_path, modem):
    d = _hand_pumped(tmp_path, modem)
    try:
        client = d.r._links["a"]
        connect(modem)
        assert wait(lambda: client.connected)
        sent = _unidir_exchange(modem, b"payload!" * 16, sid="S3")
        assert wait(lambda: len(client.client._rx) >= sent)

        d.r._on_tick()                            # CONNECTED still queued
        assert d.r.state == R.LISTEN_ALL
        assert len(client.client._rx) >= sent            # the HELLO survived the sweep
        assert _no_data_outside_session(d)

        d.start()                                 # let the loop catch up
        assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
        assert d.r.last_metrics.outcome == "ok"
        assert d.r.last_metrics.scenario == "unidir"
    finally:
        d.stop()


def test_burst_of_stale_ticks_does_not_eat_a_queued_sessions_hello(tmp_path, modem):
    # a drain outlasting a tick leaves several ticks queued ahead of the next
    # CONNECTED; they are dispatched back to back, idle, over a live HELLO
    d = _hand_pumped(tmp_path, modem)
    try:
        client = d.r._links["a"]
        connect(modem)
        assert wait(lambda: client.connected)
        sent = _unidir_exchange(modem, b"payload!" * 16, sid="S4")
        assert wait(lambda: len(client.client._rx) >= sent)

        for _ in range(4):
            d.r._on_tick()
        assert len(client.client._rx) >= sent
        assert _no_data_outside_session(d)

        d.start()
        assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
        assert d.r.last_metrics.outcome == "ok"
    finally:
        d.stop()


def test_sweep_still_discards_bytes_no_session_claims(tmp_path, modem):
    d = _hand_pumped(tmp_path, modem)
    try:
        client = d.r._links["a"]
        modem.send_data(b"ZOMBIE-LATE-BYTES")
        assert wait(lambda: len(client.client._rx) == 17)
        d.r._on_tick()                            # grace: nothing discarded yet
        assert len(client.client._rx) == 17
        time.sleep(d.r.tick_s)
        d.r._on_tick()                            # grace expired, no session came
        assert client.client._rx == b""
        assert any(r.fields.get("check") == "data_outside_session"
                   and r.fields.get("bytes") == 17 for r in d.daemon_records())
    finally:
        d.stop()


def test_session_meta_records_the_peers_identity(daemon, modem):
    # the only key that joins this record to the initiator's, two sites apart
    d = daemon([("a", modem)], hello_timeout_s=5.0, max_session_s=20.0)
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    _unidir_exchange(modem, b"payload!" * 16, sid="PEER-SID-9")
    assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
    meta = d.session_json(d.r.last_metrics.sid)["meta"]
    assert meta["peer_sid"] == "PEER-SID-9" and meta["peer_call"] == "K7XYZ"


# -- hangup -------------------------------------------------------------------

def test_hangup_waits_for_the_modem_to_drain(tmp_path, modem):
    """DISCONNECT rides the cmd socket while the REPORT rides the data socket;
    hanging up before the modem's own queue drains loses the REPORT."""
    d = _hand_pumped(tmp_path, modem, settle_s=3.0)
    try:
        client = d.r._links["a"]
        connect(modem)
        assert wait(lambda: client.connected)
        client.client.buffer_bytes = 512                 # a REPORT still queued
        drained = threading.Event()

        def drain():
            time.sleep(0.3)
            modem.notify("BUFFER 0")
            drained.set()

        threading.Thread(target=drain, daemon=True).start()
        d.r._cancel_reason = "watchdog"           # this path is ours to hang up
        d.r._hangup("a", client, None)
        assert drained.is_set()
        assert "DISCONNECT" in modem.commands
    finally:
        d.stop()


def test_a_completed_session_lets_the_initiator_hang_up(tmp_path, modem):
    d = _hand_pumped(tmp_path, modem, settle_s=3.0)
    try:
        client = d.r._links["a"]
        connect(modem)
        assert wait(lambda: client.connected)
        threading.Timer(0.2, lambda: modem.notify("DISCONNECTED")).start()
        d.r._cancel_reason = "complete"
        d.r._hangup("a", client, None)
        assert "DISCONNECT" not in modem.commands
    finally:
        d.stop()


def test_a_silent_peer_still_gets_hung_up_on(tmp_path, modem):
    d = _hand_pumped(tmp_path, modem, settle_s=0.2)
    try:
        client = d.r._links["a"]
        connect(modem)
        assert wait(lambda: client.connected)
        d.r._cancel_reason = "complete"
        d.r._hangup("a", client, None)            # nobody hangs up: we do
        assert "DISCONNECT" in modem.commands
    finally:
        d.stop()


def test_reap_closes_a_parked_transcript_once_its_writer_is_gone(daemon, modem,
                                                                 tmp_path):
    # a stuck scenario thread still holds the session transcript; closing it
    # under the thread turns its next write into a crash, and dropping the
    # handle leaks the fd
    d = daemon([("a", modem)])
    t = Transcript(str(tmp_path / "parked.jsonl"), "parked")
    gone = threading.Event()
    d.r._closing.append((gone.is_set, t))
    d.r._reap()
    assert not t._fh.closed
    gone.set()
    d.r._reap()
    assert t._fh.closed and not d.r._closing


def test_detach_mid_session_is_modem_lost_then_recovers(daemon, modem):
    d = daemon([("a", modem)], hello_timeout_s=5.0, max_session_s=20.0)
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    modem.drop()                              # the modem process vanishes

    assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
    assert d.r.last_metrics.outcome == "failed:modem_lost"
    assert not d.r.worker_alive

    assert wait(lambda: d.r.modem_states["a"] == "up", timeout=6.0)
    assert modem.attach_count >= 2
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION, timeout=6.0)


# -- supervision --------------------------------------------------------------

def test_attach_failures_escalate_to_respawn(tmp_path, modem):
    dead = FakeModem(data_listener=False).start()
    try:
        d = Daemon(tmp_path, [("dead", dead)])
        d.sup.procs["dead"] = FakeProc(alive=True)
        d.start()
        try:
            assert wait(lambda: "dead" in d.sup.respawned, timeout=5.0)
            assert d.r.modem_states["dead"] == "down"
        finally:
            d.stop()
    finally:
        dead.stop()


def test_attach_failures_do_not_respawn_external_modems(tmp_path):
    dead = FakeModem(data_listener=False).start()
    try:
        d = Daemon(tmp_path, [("dead", dead)])      # no process handle
        d.start()
        try:
            time.sleep(0.4)
            assert d.sup.respawned == []            # can't un-wedge what we didn't spawn
        finally:
            d.stop()
    finally:
        dead.stop()


def test_idle_recycle_restarts_deaf_modem(daemon, modem):
    d = daemon([("a", modem)])
    assert modem.attach_count == 1
    d.sup.recycle_due.add("a")
    assert wait(lambda: "a" in d.sup.respawned, timeout=5.0)
    assert wait(lambda: modem.attach_count >= 2, timeout=5.0)
    assert wait(lambda: d.r.modem_states["a"] == "up", timeout=5.0)
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION, timeout=5.0)


def test_stale_modem_is_recycled_though_still_attached(daemon, modem):
    # attached, LISTEN ON, and utterly deaf: the heartbeat that flows from
    # attach on every server has stopped, and nothing else would ever notice
    d = daemon([("a", modem)],
               quirks=Quirks(version_reply=True, iamalive_s=0.05))
    assert d.r.modem_states == {"a": "up"}
    assert d.r._links["a"].attached
    assert wait(lambda: "a" in d.sup.respawned, timeout=5.0)
    assert any(r.fields.get("text") == "stale_recycle" or
               r.fields.get("note") == "stale_recycle" for r in d.daemon_records())
    assert wait(lambda: modem.attach_count >= 2, timeout=5.0)


def test_iamalive_keeps_a_modem_off_the_recycle_path(daemon, modem):
    d = daemon([("a", modem)],
               quirks=Quirks(version_reply=True, iamalive_s=0.5))
    for _ in range(6):
        modem.notify("IAMALIVE")
        time.sleep(0.1)
    assert d.sup.respawned == []


def test_ensure_is_skipped_while_the_attachment_is_held(daemon, modem):
    # ensure() on an external modem is a paired TCP probe; run once per session
    # while we hold the server's only slot it lands in the accept backlog as a
    # bogus zero-length session
    d = daemon([("a", modem)])
    before = len(d.sup.ensured)
    connect(modem)
    assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
    assert d.sup.ensured[before:] == []
    assert modem.attach_count == 1


def test_bring_up_retries_do_not_leak_supervisor_holders(tmp_path):
    dead = FakeModem(data_listener=False).start()
    try:
        d = Daemon(tmp_path, [("dead", dead)])
        d.start()
        try:
            assert wait(lambda: d.r._links["dead"].attach_failures >= 3,
                        timeout=5.0)
            assert d.sup.acquired == ["dead"]      # one holder, however many tries
        finally:
            d.stop()
        assert d.sup.released == ["dead"]          # and it is given back exactly once
    finally:
        dead.stop()


def test_status_json_reports_daemon_health(daemon, modem, tmp_path):
    d = daemon([("a", modem)])
    status = tmp_path / "results" / "home" / "status.json"
    assert wait(lambda: status.is_file())
    obj = json.loads(status.read_text())
    assert obj["state"] == R.LISTEN_ALL
    assert obj["modems"] == {"a": "up"}
    assert obj["sessions_completed"] == 0 and obj["last_session"] is None
    assert obj["uptime_s"] >= 0 and obj["mycall"] == "N0RSP"

    connect(modem)
    assert wait(lambda: d.r.sessions_completed == 1, timeout=6.0)
    assert wait(lambda: json.loads(status.read_text())["sessions_completed"] == 1)
    last = json.loads(status.read_text())["last_session"]
    assert last["modem"] == "a" and last["outcome"] == d.r.last_metrics.outcome


def test_daemon_transcript_rotates_instead_of_growing(daemon, modem, tmp_path):
    d = daemon([("a", modem)])
    path = tmp_path / "results" / "home" / "responder.jsonl"
    assert wait(lambda: path.is_file())
    with open(path, "a") as fh:
        fh.write("x" * (R.TRANSCRIPT_CAP + 1))
    assert wait(lambda: (path.with_name("responder.jsonl.1")).is_file(),
                timeout=5.0)
    # The rotated sibling appears the instant os.replace returns, and for the
    # few microseconds until the new Transcript opens its file there is no
    # responder.jsonl at all. Sampling the size there raised FileNotFoundError
    # -- rarely on a quiet machine, and a thread switch inside that window
    # widens it to the whole 5 ms switch interval. Rotation is done when the
    # path is back AND small, so wait for both.
    assert wait(lambda: _reopened_small(path), timeout=5.0), \
        "responder.jsonl was never reopened below the cap after rotation"
    modem.notify("PENDING")                        # the client still writes
    assert wait(lambda: set(d.r.pending) == {"a"})
    assert any(r.fields.get("text", "").startswith("PENDING")
               for r in d.daemon_records())


def test_fanin_drops_are_bounded_and_counted(daemon, modem):
    d = daemon([("a", modem)])
    fan = d.r._fanin["a"]
    fan._depth = 2
    for _ in range(20):
        fan.put(object())
    assert fan.dropped >= 10
    assert wait(lambda: any(r.fields.get("dropped") for r in d.daemon_records()))


def test_activity_noted_on_pending_and_connected(daemon, modem):
    d = daemon([("a", modem)])
    modem.notify("PENDING")
    assert wait(lambda: d.sup.activity.count("a") >= 1)
    connect(modem)
    assert wait(lambda: d.sup.activity.count("a") >= 2)


# -- retention ----------------------------------------------------------------

def _make_session(root, day, sid, size=100, age_s=0.0):
    sid_dir = Path(root) / "home" / day / sid
    sid_dir.mkdir(parents=True, exist_ok=True)
    (sid_dir / "transcript.jsonl").write_bytes(b"x" * size)
    if age_s:
        when = time.time() - age_s
        os.utime(sid_dir / "transcript.jsonl", (when, when))
    return sid_dir


def test_sweep_results_drops_aged_sessions(tmp_path):
    old = _make_session(tmp_path, "2020-01-01", "old", age_s=86400)
    older = _make_session(tmp_path, "2020-01-01", "older", age_s=172800)
    fresh = _make_session(tmp_path, "2030-01-01", "fresh")
    removed = sweep_results(tmp_path, max_age_s=3600)
    assert set(removed) == {old, older}
    assert not old.exists() and not older.exists()
    assert fresh.exists()


def test_sweep_results_enforces_size_cap(tmp_path):
    a = _make_session(tmp_path, "2030-01-01", "a", size=1000, age_s=300)
    b = _make_session(tmp_path, "2030-01-01", "b", size=1000, age_s=200)
    c = _make_session(tmp_path, "2030-01-01", "c", size=1000, age_s=100)
    removed = sweep_results(tmp_path, max_bytes=2500)
    assert removed == [a]
    assert b.exists() and c.exists()


def test_sweep_results_keeps_the_newest_whatever_the_cap(tmp_path):
    a = _make_session(tmp_path, "2030-01-01", "a", size=1000, age_s=100)
    b = _make_session(tmp_path, "2030-01-01", "b", size=1000)
    sweep_results(tmp_path, max_bytes=1)
    assert not a.exists() and b.exists()


def test_sweep_results_collects_stray_files(tmp_path):
    site = tmp_path / "home"
    (site / "2030-01-01").mkdir(parents=True)
    fresh = _make_session(tmp_path, "2030-01-01", "fresh")
    old_report = site / "2020-01-01" / "campaign-20200101T000000.txt"
    old_report.parent.mkdir(parents=True, exist_ok=True)
    old_report.write_text("x" * 100)
    rotated = site / "responder.jsonl.1"
    rotated.write_text("x" * 100)
    for p in (old_report, rotated):
        os.utime(p, (time.time() - 86400,) * 2)
    live = site / "responder.jsonl"
    live.write_text("x" * 100)
    os.utime(live, (time.time() - 86400,) * 2)
    modem_log = site / "modem-a.log"
    modem_log.write_text("x" * 100)
    os.utime(modem_log, (time.time() - 86400,) * 2)

    removed = sweep_results(tmp_path, max_age_s=3600)
    assert set(removed) == {old_report, rotated}
    assert fresh.exists()
    assert live.exists() and modem_log.exists()   # the daemon holds these open


def test_maintenance_sweeps_the_results_tree_between_sessions(daemon, modem,
                                                              tmp_path):
    root = tmp_path / "results"
    stale = _make_session(root, "2020-01-01", "stale", age_s=86400)
    _make_session(root, "2030-01-01", "fresh")     # so stale is not the newest
    daemon([("a", modem)], retention_max_age_s=60.0)
    assert wait(lambda: not stale.exists(), timeout=6.0)


# -- shutdown -----------------------------------------------------------------

def test_stop_is_clean_mid_session(tmp_path, modem):
    d = Daemon(tmp_path, [("a", modem)], hello_timeout_s=5.0, max_session_s=20.0)
    d.start()
    connect(modem)
    assert wait(lambda: d.r.state == R.SESSION)
    d.stop()
    assert modem.wait_detached(2.0)
    assert d.sup.released == ["a"]
    assert d.sup.stopped is False             # supervisor was injected, not owned
