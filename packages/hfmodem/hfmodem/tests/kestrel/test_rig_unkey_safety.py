# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Regression: Ctrl-C must not be able to deadlock the process with the rig keyed.

`Rig.key` used to hold the PTT lock across a rigctld round trip — three attempts at a
three-second socket timeout, so nine seconds when the daemon is slow. A SIGINT handler
runs on the thread it interrupts, and it unkeyed under that same non-reentrant lock, so
a Ctrl-C landing in that window hung forever *with the transmitter keyed*; the watchdog
could not save it either, because it waited on the same lock. That is the worst state
this program has, and the window is widest exactly when rigctld is misbehaving, which is
when an operator reaches for Ctrl-C.

Everything here runs against a fake rigctld on an ephemeral loopback port. Nothing in
this file may touch 4533: there is a real transmitter on the end of it.

The signal tests run in a child process, deliberately. `Rig.panic` arms a deadman that
calls `os._exit`, which would take the test session with it.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hfmodem.tests.kestrel import corpora, fake_rigctld
from hfmodem.tests.kestrel.fake_rigctld import FakeRigctld

bridge = corpora.harness("vara_rig_bridge")

_TOOLS = str(corpora.TOOLS)


@pytest.fixture
def rigctld():
    yield from fake_rigctld.serving()


@pytest.fixture
def rig():
    """An armed `Rig` against a fake daemon, keeping its log and standing it down after."""
    made = []

    def make(server, **kw):
        logged: list[str] = []
        r = bridge.Rig(f"127.0.0.1:{server.port}", armed=True, log=logged.append)
        r.logged = logged
        for k, v in kw.items():
            setattr(r, k, v)
        made.append(r)
        return r

    yield make
    for r in made:
        r._stop.set()


# -- the deadlock, in a child process ------------------------------------------------

_DRIVER = """
import signal, sys, time
sys.path.insert(0, {tools!r})
from vara_rig_bridge import Rig

rig = Rig("127.0.0.1:" + sys.argv[1], armed=True, log=lambda m: print(m, flush=True))

def on_signal(signum, _frame):
    rig.panic("signal %d" % signum)
    sys.exit(1)

for _s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_s, on_signal)
print("READY", flush=True)
{body}
"""

# Keys up on the main thread, so the signal lands on the thread holding the PTT lock.
_KEY_AND_WAIT = """
rig.key(True)
time.sleep(60)
"""

# The same, but nothing may escape: SystemExit from the handler is swallowed, which is
# what a stray `except BaseException` or a wedged interpreter shutdown looks like. Only
# the deadman can end this process.
_KEY_AND_SWALLOW = """
try:
    rig.key(True)
except BaseException:
    pass
while True:
    try:
        time.sleep(0.05)
    except BaseException:
        pass
"""


def _run_driver(tmp_path: Path, port: int, body: str) -> subprocess.Popen:
    src = tmp_path / "driver.py"
    src.write_text(_DRIVER.format(tools=_TOOLS, body=body))
    with (tmp_path / "driver.log").open("w") as out:
        return subprocess.Popen([sys.executable, str(src), str(port)],
                                env=corpora.child_env(),
                                stdout=out, stderr=subprocess.STDOUT, cwd=_TOOLS)


def _interrupt_mid_key(server: FakeRigctld, proc: subprocess.Popen) -> None:
    """Signal the child at the worst moment: its ``T 1`` is on the wire, lock held."""
    assert server.wait_for("T 1", timeout=15), "child never keyed"
    proc.send_signal(signal.SIGINT)


@pytest.mark.parametrize("server_kw", [
    pytest.param({"delay": 4.0}, id="slow-daemon"),
    pytest.param({"delay": 30.0, "answer": False}, id="wedged-daemon"),
])
def test_signal_mid_key_unkeys_and_exits(tmp_path, rigctld, server_kw):
    """Ctrl-C inside a stalled key-up: transmitter down, process dead, seconds not never.

    Against the pre-fix code this hangs for good — the handler blocks acquiring the
    lock its own thread is holding — so `wait` times out and the child is left keyed.
    """
    server = rigctld(**server_kw)
    proc = _run_driver(tmp_path, server.port, _KEY_AND_WAIT)
    try:
        _interrupt_mid_key(server, proc)
        assert proc.wait(timeout=4) != 0, "child exited cleanly; it was interrupted"
        assert server.seen("T 0"), "no unkey reached the wire"
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def test_deadman_kills_a_process_that_swallows_the_exit(tmp_path, rigctld):
    """The unkey is worth nothing if the process then refuses to die."""
    server = rigctld(delay=30.0, answer=False)
    proc = _run_driver(tmp_path, server.port, _KEY_AND_SWALLOW)
    try:
        _interrupt_mid_key(server, proc)
        assert proc.wait(timeout=5) != 0
        assert server.seen("T 0"), "no unkey reached the wire"
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def test_a_signal_is_answered_even_with_rigctld_gone(tmp_path, rigctld):
    """Nothing to talk to at all is still an exit, not a hang."""
    server = rigctld(delay=4.0)
    proc = _run_driver(tmp_path, server.port, _KEY_AND_WAIT)
    try:
        assert server.wait_for("T 1", timeout=15)
        server.close()                      # the daemon dies under us, mid-over
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=4) != 0
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


# -- the watchdog, in process --------------------------------------------------------

def test_watchdog_unkeys_while_the_key_path_holds_the_lock(monkeypatch, rigctld, rig):
    """The last line of defence must not queue behind the fault it exists to fix."""
    monkeypatch.setattr(bridge, "MAX_KEY_S", 0.2)
    server = rigctld(delay=3.0)
    r = rig(server, keyed=True, key_since=time.time() - 10)
    threading.Thread(target=r.key, args=(True,), daemon=True).start()
    assert server.wait_for("T 1")
    assert server.wait_for("T 0", timeout=2), "watchdog never got its unkey out"
    assert any("WATCHDOG" in m for m in r.logged)


def test_shutdown_unkeys_before_standing_the_watchdog_down(rigctld, rig):
    """Ordering: the stop flag used to be set first, disarming the watchdog early."""
    server = rigctld()
    r = rig(server, keyed=True, key_since=time.time())
    watchdog_alive = []
    unkey = r.unkey_hard
    r.unkey_hard = lambda why: (watchdog_alive.append(not r._stop.is_set()), unkey(why))[1]
    r.shutdown()
    assert watchdog_alive == [True], "the watchdog was stood down before the unkey ran"
    assert r._stop.is_set() and server.ptt == 0


# -- confirmation, in process --------------------------------------------------------

def test_unkey_is_not_believed_until_the_rig_confirms_it(rigctld, rig):
    """``RPRT 0`` means rigctld took the command, not that the transmitter dropped."""
    server = rigctld(ignore_unkeys=2)
    server.ptt = 1
    r = rig(server, keyed=True, key_since=time.time())
    r.key(False)
    assert server.seen("T 0") >= 3, "gave up on the first accepted write"
    assert server.ptt == 0 and r.keyed is False


def test_an_unkey_that_never_confirms_is_reported_as_stuck(rigctld, rig):
    """A rig that stays keyed however politely it is asked has to be shouted about."""
    server = rigctld(ignore_unkeys=10**6)
    server.ptt = 1
    r = rig(server, keyed=True, key_since=time.time())
    r.unkey_hard("test")
    assert server.seen("T 0") > 1
    assert any("MAY BE STUCK" in m for m in r.logged), r.logged


@pytest.mark.realtime
def test_a_silent_daemon_is_not_mistaken_for_a_confirmed_unkey(rigctld, rig):
    """The empty reply from a wedged daemon used to read as 'no RPRT, nothing wrong'."""
    server = rigctld(delay=30.0, answer=False)
    r = rig(server, keyed=True, key_since=time.time())
    t0 = time.monotonic()
    r.unkey_hard("test")
    assert time.monotonic() - t0 < 2, "forced unkey outran its budget"
    assert server.seen("T 0"), "nothing reached the wire"
    assert any("MAY BE STUCK" in m for m in r.logged), r.logged


# -- ordering against an in-flight key-up --------------------------------------------

def test_a_forced_unkey_wins_a_race_with_an_in_flight_key_up(rigctld, rig):
    """The unkey takes no lock, so it can overtake a ``T 1`` rigctld has not applied.

    Whoever holds the lock has to notice and put the rig back down on the way out,
    or the transmitter comes up *after* the operator asked for it to go down.
    """
    server = rigctld(delay=0.5)
    r = rig(server)
    keyer = threading.Thread(target=r.key, args=(True,), daemon=True)
    keyer.start()
    assert server.wait_for("T 1")
    r._no_tx.set()
    r.unkey_hard("test")
    keyer.join(timeout=10)
    assert not keyer.is_alive()
    assert server.ptt == 0, "transmitter left keyed by the losing side of the race"
    assert r.keyed is False
    assert [c for c in server.commands if c.startswith("T ")][-1] == "T 0"


def test_a_retired_rig_refuses_to_key_again(rigctld, rig):
    server = rigctld()
    r = rig(server)
    r._no_tx.set()
    r.key(True)
    assert not server.seen("T 1")
    assert r.keyed is False


def test_disarmed_rigs_never_reach_the_wire(rigctld):
    """The dry run is the default, and it must stay incapable of keying."""
    server = rigctld()
    r = bridge.Rig(f"127.0.0.1:{server.port}", armed=False, log=lambda m: None)
    try:
        r.key(True)
        r.unkey_hard("test")
        r.shutdown()
        assert server.commands == []
    finally:
        r._stop.set()


# -- the tool's own handlers ---------------------------------------------------------

# The driver above installs handlers of its own. This one asks the bridge for the
# dispositions the bridge ships, which is what an operator's run gets.
_TOOL_DRIVER = """
import sys, time
sys.path.insert(0, {tools!r})
import vara_rig_bridge as vrb

rig = vrb.Rig("127.0.0.1:" + sys.argv[1], armed=True, log=lambda m: print(m, flush=True))
vrb._install_unkey_handlers(rig)
print("READY", flush=True)
{body}
"""


def _run_tool_driver(tmp_path: Path, port: int, body: str) -> subprocess.Popen:
    src = tmp_path / "tool_driver.py"
    src.write_text(_TOOL_DRIVER.format(tools=_TOOLS, body=body))
    with (tmp_path / "tool_driver.log").open("w") as out:
        return subprocess.Popen([sys.executable, str(src), str(port)],
                                env=corpora.child_env(),
                                stdout=out, stderr=subprocess.STDOUT, cwd=_TOOLS)


@pytest.mark.parametrize("sig", [
    pytest.param(signal.SIGTERM, id="SIGTERM"),
    pytest.param(signal.SIGHUP, id="SIGHUP"),
])
def test_the_bridge_answers_every_signal_that_ends_a_session(tmp_path, rigctld, sig):
    """SIGTERM is `timeout(1)` and a supervisor recycle; SIGHUP is a dropped ssh.
    The bridge took over SIGINT and SIGTERM only, so a closed terminal mid-over left
    the transmitter up with neither a `finally:` nor `atexit` to reach it."""
    server = rigctld(delay=4.0)
    proc = _run_tool_driver(tmp_path, server.port, _KEY_AND_WAIT)
    try:
        assert server.wait_for("T 1", timeout=15), "the child never keyed"
        proc.send_signal(sig)
        assert proc.wait(timeout=6) != 0, "the child exited cleanly; it was signalled"
        assert server.seen("T 0"), f"{sig.name} left the transmitter up"
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def test_the_connect_tool_takes_the_bridges_handlers_and_not_a_copy():
    """One `Rig`, and therefore one set of dispositions to keep it down.

    `kestrel_connect` imports this `Rig` and had a second `_install_unkey_handlers`
    over it, identical but for the docstring and a hardcoded signal tuple. Two
    copies is four signals to keep in step and nothing that notices when they
    drift — and the drift is not hypothetical: the reason `FATAL_SIGNALS` exists is
    that SIGHUP was added to some of these sites and not others, which is a dropped
    ssh session leaving the transmitter up.
    """
    kc = corpora.harness("kestrel_connect")

    assert kc.Rig is bridge.Rig, "the tools no longer share a rig; this test is moot"
    assert kc._install_unkey_handlers is bridge._install_unkey_handlers, (
        "kestrel_connect installs handlers of its own again")
    assert bridge.FATAL_SIGNALS == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP), (
        "the shared tuple no longer covers what the deleted copy hardcoded")


def test_the_deadman_is_the_kernels_and_needs_no_new_thread(rigctld, rig):
    """`core.rig` rejects a thread here in prose and the bridge started one anyway:
    starting a thread takes interpreter locks the interrupted thread may hold, so the
    one mechanism whose whole purpose is to be unblockable by that thread could be
    blocked by it. A SIGALRM needs nothing but the kernel."""
    r = rig(rigctld())
    prior = signal.getsignal(signal.SIGALRM)
    threads = threading.active_count()
    try:
        r._arm_deadman()
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0, "no timer was set"
        assert threading.active_count() == threads, "the deadman started a thread"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior)
