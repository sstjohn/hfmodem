# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ModemProcess and Supervisor: spawn, restart policy, circuit breaker,
spawn_group refcounting, kill_and_respawn, probes, idle recycle."""

import contextlib
import functools
import inspect
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import pytest

from hfhost.config import ModemConfig, Quirks, SpawnGroupConfig
from hfhost.supervisor import (
    _MODEM_UNKEY_BUDGET_S,
    _UNKEY_GRACE_S,
    ModemProcess,
    SpawnError,
    Supervisor,
    probe,
)

SLEEPER = (sys.executable, "-c", "import time; time.sleep(60)")
FAST_EXIT = (sys.executable, "-c", "import sys; sys.exit(1)")

FAST = dict(fast_fail_s=5.0, breaker_threshold=3, down_retry_s=0.3,
            backoff_initial_s=0.01, backoff_max_s=0.03)


def wait_for(pred, timeout=3.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


# -- ModemProcess ------------------------------------------------------------

def test_spawn_alive_stop(tmp_path):
    p = ModemProcess("s", SLEEPER, log_path=str(tmp_path / "s.log"))
    p.start()
    try:
        assert p.alive
        assert p.state == "running"
        assert os.path.exists(tmp_path / "s.log")
    finally:
        p.stop()
    assert not p.alive
    assert p.state == "stopped"


def test_spawn_failure_raises(tmp_path):
    p = ModemProcess("bad", ("/nonexistent/definitely-not-a-binary",),
                     log_path=str(tmp_path / "bad.log"))
    with pytest.raises(SpawnError):
        p.start()


def test_log_output_and_rotation(tmp_path):
    log = tmp_path / "echo.log"
    with open(log, "wb") as fh:                  # oversized pre-existing log
        fh.seek(6 * 1024 * 1024 - 1)
        fh.write(b"\0")
    p = ModemProcess("e", (sys.executable, "-c", "print('hello-from-modem')"),
                     log_path=str(log))
    p.start()
    assert wait_for(lambda: not p.alive)
    p.stop()
    assert os.path.exists(str(log) + ".1")       # rotated
    assert wait_for(lambda: b"hello-from-modem" in open(log, "rb").read())


def test_restart_backoff_and_circuit_breaker(tmp_path):
    p = ModemProcess("f", FAST_EXIT, log_path=str(tmp_path / "f.log"), **FAST)
    assert wait_for(lambda: (p.ensure_running(), p.state == "down")[1])
    assert not p.ensure_running()                # down: slow-retry gate holds
    assert p.state == "down"
    time.sleep(0.35)                             # past down_retry_s: half-open
    p.ensure_running()
    assert wait_for(lambda: (p.ensure_running(), p.state == "down")[1], 2.0)
    p.stop()


def test_breaker_resets_on_healthy_uptime(tmp_path):
    p = ModemProcess("h", SLEEPER, log_path=str(tmp_path / "h.log"),
                     fast_fail_s=0.05, breaker_threshold=3,
                     down_retry_s=60.0, backoff_initial_s=0.01,
                     backoff_max_s=0.03)
    p._fast_fails = 2                            # one fast failure from DOWN
    p.start()
    try:
        time.sleep(0.1)                          # uptime beyond fast_fail_s
        assert p.ensure_running()
        assert p._fast_fails == 0                # healthy uptime resets breaker
    finally:
        p.stop()


def test_kill_and_respawn(tmp_path):
    p = ModemProcess("k", SLEEPER, log_path=str(tmp_path / "k.log"), **FAST)
    p.start()
    try:
        pid1 = p.pid
        p.kill_and_respawn()
        assert p.alive
        assert p.pid != pid1
        assert p.state == "running"
    finally:
        p.stop()


_KEYED_MODEM_SRC = (
    "import os, signal, sys, time\n"
    "def unkey(*_):\n"
    "    time.sleep(float(sys.argv[3]))\n"           # the ladder, against a mute rig
    "    open(sys.argv[1], 'w').write('down')\n"     # stands in for the ioctl
    "    raise SystemExit(0)\n"
    "signal.signal(signal.SIGTERM,\n"
    "              signal.SIG_IGN if sys.argv[2] == 'deaf' else unkey)\n"
    "open(sys.argv[1] + '.up', 'w').write(str(os.getpid()))\n"
    "while True: time.sleep(0.02)\n"
)


def keyed_modem(marker, *, deaf: bool = False, unkey_s: float = 0.0):
    """A modem that drops its key on SIGTERM, leaving `marker` behind — or one
    that has stopped listening for signals at all. Signalled before its handler
    is installed, either one dies on the default disposition and proves nothing,
    so it writes `<marker>.up` once it is listening. `unkey_s` is how long it
    spends with the key still down before the marker appears."""
    return (sys.executable, "-c", _KEYED_MODEM_SRC, str(marker),
            "deaf" if deaf else "unkeys", str(unkey_s))


def gone(pid: int) -> bool:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, 0)
        return False
    return True


@pytest.mark.realtime
def test_kill_and_respawn_lets_a_keyed_modem_unkey_first(tmp_path):
    """A recycle fires when the modem is presumed unresponsive, which is when
    it is likeliest to be stuck mid-transmit rather than idle — and SIGKILL is
    the one signal it cannot unkey through. So the recycle signals first, and
    a modem that answers is not made to wait out the grace to prove it."""
    marker, up = tmp_path / "unkeyed", tmp_path / "unkeyed.up"
    p = ModemProcess("k", keyed_modem(marker), log_path=str(tmp_path / "k.log"),
                     **FAST)
    p.start()
    try:
        assert wait_for(up.exists)
        pid = p.pid
        started = time.monotonic()
        p.kill_and_respawn()
        assert marker.exists(), "recycled the modem without letting it unkey"
        assert time.monotonic() - started < 1.0
        assert p.alive and p.pid != pid
    finally:
        p.stop()


@pytest.mark.realtime
def test_kill_and_respawn_insists_when_the_modem_ignores_sigterm(tmp_path):
    """The grace is an opportunity, not a promise to wait: a modem that will
    not take the signal is killed and replaced anyway, and does not survive the
    recycle that was meant to end it."""
    marker, up = tmp_path / "unkeyed", tmp_path / "unkeyed.up"
    p = ModemProcess("d", keyed_modem(marker, deaf=True),
                     log_path=str(tmp_path / "d.log"), **FAST)
    p.start()
    try:
        assert wait_for(up.exists)
        pid = p.pid
        started = time.monotonic()
        p.kill_and_respawn(grace=0.3)
        assert time.monotonic() - started >= 0.3
        assert not marker.exists()
        assert wait_for(lambda: gone(pid))
        assert p.alive and p.pid != pid
    finally:
        p.stop(term_timeout_s=0.3)


@pytest.mark.realtime
def test_the_grace_covers_a_modem_spending_its_unkey_budget(tmp_path):
    """A modem asking a slow or silent rigctld spends seconds with the key still
    down; SIGKILL landing inside that window leaves the carrier up with nothing
    running to drop it. How fast a healthy child exits is a different quantity
    and cannot size this."""
    marker, up = tmp_path / "unkeyed", tmp_path / "unkeyed.up"
    p = ModemProcess("slow", keyed_modem(marker, unkey_s=3.0),
                     log_path=str(tmp_path / "slow.log"), **FAST)
    p.start()
    try:
        assert wait_for(up.exists)
        pid = p.pid
        p.kill_and_respawn()
        assert marker.exists(), "SIGKILLed mid-unkey"
        assert p.alive and p.pid != pid
    finally:
        p.stop()


def test_neither_stop_nor_recycle_undercuts_the_unkey_budget():
    """The two waits are the modem's unkey ladder plus room to exit, not two
    numbers that happen to be ordered."""
    assert _UNKEY_GRACE_S > _MODEM_UNKEY_BUDGET_S
    for fn, arg in ((ModemProcess.stop, "term_timeout_s"),
                    (ModemProcess.kill_and_respawn, "grace")):
        assert inspect.signature(fn).parameters[arg].default >= _UNKEY_GRACE_S, fn


_GRANDCHILD_SRC = (
    "import subprocess, sys, time\n"
    "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
    "open(sys.argv[1], 'w').write(str(gc.pid))\n"
    "time.sleep(30)\n"
)
_GRANDCHILD_SRC_IGNORE_TERM = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n" + _GRANDCHILD_SRC


def _leader_with_pipe_holding_grandchild(tmp_path, name, ignore_term=False):
    """A process whose stdout is a pipe, that forks a grandchild inheriting
    that same fd and outlives it. Killing the leader alone never makes the
    pipe's read end see EOF."""
    pidfile = tmp_path / f"{name}.pid"
    src = _GRANDCHILD_SRC_IGNORE_TERM if ignore_term else _GRANDCHILD_SRC
    proc = subprocess.Popen([sys.executable, "-c", src, str(pidfile)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    assert wait_for(pidfile.exists, timeout=5.0)
    return proc, int(pidfile.read_text())


def _reap_leader(proc: subprocess.Popen) -> None:
    """Collect the leader these two tests deliberately wedge.

    `_reap` gives up on the held-open pipe and raises, which is what they are
    proving -- and leaves the leader killed but never waited on. Both then left a
    defunct child of the run behind them, which is the exact shape of the leak
    these tests are about, one level up.
    """
    with contextlib.suppress(OSError):
        proc.stdout.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_REAP_WAIT_S)


#: What the killed leader gets to be collected in, once the grandchild holding its
#: pipe is gone. It is already SIGKILLed by the time this is reached.
_REAP_WAIT_S = 5.0


def _run_bounded(fn, timeout=15.0):
    """Run fn() on a thread and report whether it returned within timeout,
    and any exception it raised. A thread, not the call itself, is what keeps
    a genuine wedge from hanging the suite."""
    result = {}
    def target():
        try:
            fn()
        except BaseException as exc:                # noqa: BLE001
            result["error"] = exc
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive(), result.get("error")


def test_kill_and_respawn_reports_a_grandchild_still_holding_stdout(tmp_path):
    """`_reap` on the recycle's SIGKILL branch, reached by a leader that ignores
    SIGTERM: communicate(timeout=...) bounds a genuinely held-open pipe and
    reports it, which is what stands between today's accidental safety
    (log-file stdout) and a real wedge the day someone switches that redirect
    to PIPE."""
    proc, gc_pid = _leader_with_pipe_holding_grandchild(tmp_path, "kr",
                                                       ignore_term=True)
    p = ModemProcess("kr", SLEEPER, log_path=str(tmp_path / "kr.log"))
    p._proc = proc
    assert proc.poll() is None, "leader exited before kill_and_respawn ran"
    try:
        returned, error = _run_bounded(lambda: p.kill_and_respawn(grace=0.3))
        assert returned, "kill_and_respawn wedged past its own bound"
        assert isinstance(error, RuntimeError) and "held open" in str(error)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(gc_pid, signal.SIGKILL)
        _reap_leader(proc)
        p.stop()


def test_kill_and_respawn_replaces_the_modem_even_when_the_stop_reports(tmp_path):
    """The recycle exists to put a fresh modem there. `_reap` reporting a
    held-open pipe has still killed the process, so skipping the respawn leaves
    the entry with no handle and no spawn: a modem that never comes back and
    never says so."""
    proc, gc_pid = _leader_with_pipe_holding_grandchild(tmp_path, "kb",
                                                       ignore_term=True)
    p = ModemProcess("kb", SLEEPER, log_path=str(tmp_path / "kb.log"))
    p._proc = proc
    try:
        returned, error = _run_bounded(lambda: p.kill_and_respawn(grace=0.3))
        assert returned and isinstance(error, RuntimeError)
        assert p.alive and p.pid != proc.pid
        assert p.state == "running"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(gc_pid, signal.SIGKILL)
        _reap_leader(proc)
        p.stop()


def test_stop_reports_a_grandchild_still_holding_stdout_after_kill(tmp_path):
    """Same proof against stop()'s SIGKILL fallback: SIGTERM is ignored so the
    grace expires and the kill()/_reap() branch runs."""
    proc, gc_pid = _leader_with_pipe_holding_grandchild(tmp_path, "st", ignore_term=True)
    p = ModemProcess("st", SLEEPER, log_path=str(tmp_path / "st.log"))
    p._proc = proc
    assert proc.poll() is None, "leader exited before stop() ran"
    try:
        returned, error = _run_bounded(lambda: p.stop(term_timeout_s=0.3))
        assert returned, "stop() wedged past its own bound"
        assert isinstance(error, RuntimeError) and "held open" in str(error)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(gc_pid, signal.SIGKILL)
        _reap_leader(proc)


def test_idle_recycle_due(tmp_path):
    p = ModemProcess("i", SLEEPER, log_path=str(tmp_path / "i.log"),
                     max_idle_recycle_s=0.1)
    p.start()
    try:
        now = time.monotonic()
        p.note_activity(now)
        assert not p.idle_recycle_due(now + 0.05)
        assert p.idle_recycle_due(now + 0.15)
        p.note_activity(now + 0.2)
        assert not p.idle_recycle_due(now + 0.25)
    finally:
        p.stop()
    assert not p.idle_recycle_due()              # dead process: never due


# -- probe -------------------------------------------------------------------

def _listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(4)
    return s


def test_probe_paired():
    cmd_srv, data_srv = _listener(), _listener()
    cmd_port = cmd_srv.getsockname()[1]
    data_port = data_srv.getsockname()[1]
    assert probe("127.0.0.1", cmd_port, data_port, timeout=0.5)
    data_srv.close()
    assert not probe("127.0.0.1", cmd_port, data_port, timeout=0.5)
    cmd_srv.close()
    assert not probe("127.0.0.1", cmd_port, data_port, timeout=0.5)


# -- Supervisor --------------------------------------------------------------

@dataclass
class Inventory:
    """The whole of ModemSource — Supervisor is given no more than this."""
    modems: tuple[ModemConfig, ...]
    spawn_groups: tuple[SpawnGroupConfig, ...] = field(default_factory=tuple)

    def modem(self, name: str) -> ModemConfig:
        for m in self.modems:
            if m.name == name:
                return m
        raise KeyError(name)


def make_config(modems, groups=()):
    return Inventory(tuple(modems), tuple(groups))


def test_spawn_group_refcounting(tmp_path):
    group = SpawnGroupConfig(name="pair", spawn=SLEEPER)
    cfg = make_config(
        [ModemConfig(name="A", cmd_port=1, data_port=2, spawn_group="pair"),
         ModemConfig(name="B", cmd_port=3, data_port=4, spawn_group="pair")],
        [group])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        pa = sup.acquire("A")
        pb = sup.acquire("B")
        assert pa is pb                          # one process fronts both
        assert pa.alive
        pid = pa.pid
        sup.release("A")
        assert pa.alive                          # B still holds it
        assert pa.pid == pid
        sup.release("B")
        assert not pa.alive                      # last holder released
    finally:
        sup.stop_all()


def test_nested_acquire_release_of_one_modem(tmp_path):
    cfg = make_config(
        [ModemConfig(name="m", cmd_port=1, data_port=2, spawn=SLEEPER)])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        proc = sup.acquire("m")
        assert sup.acquire("m") is proc
        pid = proc.pid
        sup.release("m")
        assert proc.alive                        # the second holder still wants it
        assert proc.pid == pid
        sup.release("m")
        assert not proc.alive
    finally:
        sup.stop_all()


def test_nested_acquire_within_a_spawn_group(tmp_path):
    group = SpawnGroupConfig(name="pair", spawn=SLEEPER)
    cfg = make_config(
        [ModemConfig(name="A", cmd_port=1, data_port=2, spawn_group="pair"),
         ModemConfig(name="B", cmd_port=3, data_port=4, spawn_group="pair")],
        [group])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        proc = sup.acquire("A")
        sup.acquire("A")
        sup.acquire("B")
        pid = proc.pid
        sup.release("A")
        sup.release("B")
        assert proc.alive                        # A's second acquire outlives B
        assert proc.pid == pid
        sup.release("A")
        assert not proc.alive
    finally:
        sup.stop_all()


def test_unbalanced_release_does_not_underflow(tmp_path):
    cfg = make_config(
        [ModemConfig(name="m", cmd_port=1, data_port=2, spawn=SLEEPER)])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        proc = sup.acquire("m")
        sup.release("m")
        sup.release("m")                         # stray release: no credit banked
        proc = sup.acquire("m")
        assert proc.alive
        sup.release("m")
        assert not proc.alive
    finally:
        sup.stop_all()


def test_supervisor_start_on_demand_and_respawn(tmp_path):
    cfg = make_config(
        [ModemConfig(name="m", cmd_port=1, data_port=2, spawn=SLEEPER)])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        assert sup.ensure("m")
        proc = sup.process_for("m")
        assert proc.alive
        pid = proc.pid
        sup.kill_and_respawn("m")
        assert proc.alive
        assert proc.pid != pid
        sup.note_activity("m")
        assert not sup.idle_recycle_due("m")
    finally:
        sup.stop_all()


def test_supervisor_external_modem_uses_probe(tmp_path):
    cmd_srv, data_srv = _listener(), _listener()
    cfg = make_config(
        [ModemConfig(name="ext", cmd_port=cmd_srv.getsockname()[1],
                     data_port=data_srv.getsockname()[1])])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2)
    assert sup.process_for("ext") is None
    assert sup.acquire("ext") is None
    assert sup.ensure("ext")                     # ports open: attachable
    cmd_srv.close()
    data_srv.close()
    assert not sup.ensure("ext")


def test_supervisor_never_double_spawns_over_external(tmp_path):
    cmd_srv, data_srv = _listener(), _listener()
    cfg = make_config(
        [ModemConfig(name="m", cmd_port=cmd_srv.getsockname()[1],
                     data_port=data_srv.getsockname()[1], spawn=SLEEPER)])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        assert sup.ensure("m")                   # external instance owns ports
        assert not sup.process_for("m").alive    # so nothing was spawned
    finally:
        sup.stop_all()
        cmd_srv.close()
        data_srv.close()


def test_a_failed_acquire_leaves_no_holder_behind(tmp_path):
    """A SpawnError out of `ensure()` must not bank a holder nobody will
    release: the refcount would never reach zero again and the process would
    never be stopped by a release."""
    binary = tmp_path / "modem"
    cfg = make_config(
        [ModemConfig(name="m", cmd_port=1, data_port=2, spawn=(str(binary),))])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    try:
        with pytest.raises(SpawnError):
            sup.acquire("m")
        binary.write_text("#!/bin/sh\nexec sleep 60\n")   # the operator fixes it
        os.chmod(binary, 0o755)
        proc = sup.acquire("m")
        assert proc.alive
        sup.release("m")
        assert not proc.alive
    finally:
        sup.stop_all()


@pytest.mark.realtime
def test_stop_all_stops_every_modem_and_reports_the_ones_it_could_not(tmp_path):
    """A station stop that gives up on the first raise strands every modem after
    it — including one with the transmitter up — running and un-SIGTERMed. And
    with two wedged, a report naming only the first hides the other."""
    leaders = [_leader_with_pipe_holding_grandchild(tmp_path, name, ignore_term=True)
               for name in ("w1", "w2")]
    cfg = make_config(
        [ModemConfig(name="w1", cmd_port=1, data_port=2, spawn=SLEEPER),
         ModemConfig(name="w2", cmd_port=3, data_port=4, spawn=SLEEPER),
         ModemConfig(name="keyed", cmd_port=5, data_port=6, spawn=SLEEPER)])
    sup = Supervisor(cfg, str(tmp_path), probe_timeout_s=0.2, **FAST)
    for name, (leader, _) in zip(("w1", "w2"), leaders):
        stuck = sup.process_for(name)
        stuck._proc = leader
        # the wedge is the grandchild, not the grace: don't wait the whole one out
        stuck.stop = functools.partial(stuck.stop, term_timeout_s=0.3)
    keyed = sup.acquire("keyed")
    assert keyed.alive
    try:
        returned, error = _run_bounded(sup.stop_all)
        assert returned, "stop_all wedged past its own bound"
        assert not keyed.alive, "a leaked grandchild stranded a running modem"
        assert isinstance(error, ExceptionGroup)
        assert len(error.exceptions) == 2, "a second failure was lost"
        assert all("held open" in str(e) for e in error.exceptions)
        assert {"stopping w1", "stopping w2"} == {
            note for e in error.exceptions for note in e.__notes__}
    finally:
        for leader, gc_pid in leaders:
            with contextlib.suppress(ProcessLookupError):
                os.kill(gc_pid, signal.SIGKILL)
            _reap_leader(leader)
        keyed.stop()


def test_supervisor_group_idle_recycle_from_quirks(tmp_path):
    group = SpawnGroupConfig(name="g", spawn=SLEEPER)
    cfg = make_config(
        [ModemConfig(name="A", cmd_port=1, data_port=2, spawn_group="g",
                     quirks=Quirks(max_idle_recycle_s=100.0)),
         ModemConfig(name="B", cmd_port=3, data_port=4, spawn_group="g",
                     quirks=Quirks(max_idle_recycle_s=50.0))],
        [group])
    sup = Supervisor(cfg, str(tmp_path), **FAST)
    proc = sup.process_for("A")
    assert proc.max_idle_recycle_s == 50.0       # strictest member wins
    sup.stop_all()
