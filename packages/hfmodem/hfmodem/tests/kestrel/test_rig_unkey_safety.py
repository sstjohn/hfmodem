# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Signals, watchdogs and forced unkeys must never queue behind a stalled key-up.

The fake LineKeyer shows bridge decisions, not kernel ioctl or RF behaviour. Signal
cases run in children: the kernel deadman deliberately calls os._exit.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hfmodem.core import ptt as core_ptt
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.fake_line_keyer import FakeLineKeyer

bridge = corpora.harness("vara_rig_bridge")
_TOOLS = str(corpora.TOOLS)


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(bridge, "LineKeyer", FakeLineKeyer)

    def no_cat(*args, **kwargs):
        raise AssertionError("PTT reached CAT")

    monkeypatch.setattr(bridge.Rig, "_cmd", no_cat)
    monkeypatch.setattr(core_ptt, "drop_rts", no_cat)
    made = []

    def make(**kw):
        logged = []
        r = bridge.Rig(None, armed=True, log=logged.append,
                       line_ptt=True, ptt_device="/dev/fake")
        r.logged = logged
        for k, v in kw.items():
            setattr(r, k, v)
        made.append(r)
        return r

    yield make
    for r in made:
        if r._keyer is not None:
            r.shutdown()
        else:
            r.retire()


_DRIVER = """
import signal, sys, time
from pathlib import Path
sys.path.insert(0, {tools!r})
import vara_rig_bridge as vrb
from hfmodem.tests.kestrel.fake_line_keyer import FakeLineKeyer

def fake_line(device, log):
    return FakeLineKeyer(device, log, delay={delay!r}, events=Path(sys.argv[1]))

vrb.LineKeyer = fake_line
rig = vrb.Rig(None, armed=True, log=lambda m: print(m, flush=True),
              line_ptt=True, ptt_device="/dev/fake")
def no_cat(*args, **kwargs):
    raise AssertionError("PTT reached CAT")
rig._cmd = no_cat
{handlers}
print("READY", flush=True)
{body}
"""
_HANDLERS = """
def on_signal(signum, _frame):
    rig.panic("signal %d" % signum)
    sys.exit(1)
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, on_signal)
"""
_KEY_AND_WAIT = """
rig.key(True)
time.sleep(60)
"""
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


def _run_driver(tmp_path: Path, body: str, *, delay=30.0, tool_handlers=False):
    events = tmp_path / "line-events"
    src = tmp_path / "driver.py"
    src.write_text(_DRIVER.format(
        tools=_TOOLS, delay=delay, body=body,
        handlers="vrb._install_unkey_handlers(rig)" if tool_handlers else _HANDLERS))
    with (tmp_path / "driver.log").open("w") as out:
        proc = subprocess.Popen([sys.executable, str(src), str(events)],
                                env=corpora.child_env(), stdout=out,
                                stderr=subprocess.STDOUT, cwd=_TOOLS)
    return proc, events


def _events(path):
    return path.read_text().splitlines() if path.exists() else []


def _wait_for_key(proc, events):
    end = time.monotonic() + 15
    while time.monotonic() < end and proc.poll() is None:
        if "True" in _events(events):
            return
        time.sleep(0.01)
    pytest.fail("child never entered key-up")


def _stop(proc):
    if proc.poll() is None:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("delay", [4.0, 30.0], ids=["slow-line", "wedged-line"])
def test_signal_mid_key_unkeys_and_exits(tmp_path, delay):
    proc, events = _run_driver(tmp_path, _KEY_AND_WAIT, delay=delay)
    try:
        _wait_for_key(proc, events)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=4) != 0
        assert "False" in _events(events), "signal never deasserted the line"
    finally:
        _stop(proc)


def test_deadman_kills_a_process_that_swallows_the_exit(tmp_path):
    proc, events = _run_driver(tmp_path, _KEY_AND_SWALLOW)
    try:
        _wait_for_key(proc, events)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=5) != 0
        assert "False" in _events(events)
    finally:
        _stop(proc)


def test_a_signal_is_answered_even_with_the_line_gone(tmp_path):
    proc, events = _run_driver(tmp_path, _KEY_AND_WAIT.replace(
        "rig.key(True)", "rig._keyer.gone = True\nrig.key(True)"))
    try:
        _wait_for_key(proc, events)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=4) != 0
        assert "False" in _events(events), "failed line was never asked to unkey"
    finally:
        _stop(proc)


def test_watchdog_unkeys_while_the_key_path_holds_the_lock(monkeypatch, rig):
    monkeypatch.setattr(bridge, "MAX_KEY_S", 0.2)
    r = rig(keyed=True, key_since=time.time() - 10)
    line = r._keyer
    line.key_release = threading.Event()
    keyer = threading.Thread(target=r.key, args=(True,), daemon=True)
    keyer.start()
    try:
        assert line.key_started.wait(1)
        assert line.unkeyed.wait(2), "watchdog queued behind key-up"
        assert any("WATCHDOG" in m for m in r.logged)
    finally:
        r._no_tx.set()
        line.key_release.set()
        keyer.join(2)
    assert not keyer.is_alive()
    assert not line.high


def test_shutdown_unkeys_before_standing_the_watchdog_down(rig):
    r = rig(keyed=True, key_since=time.time())
    line = r._keyer
    line.high = True
    watchdog_alive = []
    unkey = r.unkey_hard
    r.unkey_hard = lambda why: (watchdog_alive.append(not r._stop.is_set()), unkey(why))[1]
    r.shutdown()
    assert watchdog_alive == [True]
    assert r._stop.is_set() and not line.high
    assert line.calls[-1] == "hand_back"
    # The fixture can shut it down again without sampling the first-shutdown order.
    r.unkey_hard = unkey


def test_unkey_is_not_believed_until_the_rig_confirms_it(rig):
    """A failed line readback must not clear keyed bookkeeping or permit another key."""
    r = rig(keyed=True, key_since=time.time())
    r._keyer.high = True
    r._keyer.result = False
    assert r.key(False) is False
    assert r.keyed and r.retired
    r._keyer.result = True
    assert r.key(False)
    assert not r.keyed and not r._keyer.high
    assert r.key_since == 0.0
    assert r.retired


def test_an_unkey_that_never_confirms_is_reported_as_stuck(rig):
    r = rig(keyed=True, key_since=time.time())
    r._keyer.result = False
    assert r.unkey_hard("test") is False
    assert r.retired
    assert not r.key(True), "an unconfirmed unkey allowed another transmission"


def test_a_forced_unkey_wins_a_race_with_an_in_flight_key_up(rig):
    r = rig()
    line = r._keyer
    line.key_release = threading.Event()
    result = []
    keyer = threading.Thread(target=lambda: result.append(r.key(True)), daemon=True)
    keyer.start()
    try:
        assert line.key_started.wait(1)
        r._no_tx.set()
        assert r.unkey_hard("test")
        assert not line.high
    finally:
        line.key_release.set()
        keyer.join(2)
    assert not keyer.is_alive()
    assert result == [False]
    assert not line.high and not r.keyed
    assert line.calls == ["arm", True, False, False]


def test_a_retired_rig_refuses_to_key_again(rig):
    r = rig()
    r.retire()
    assert not r.key(True)
    assert r._keyer.calls == ["arm"]
    assert not r.keyed


def test_disarmed_rigs_never_reach_the_wire(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disarmed rig touched a device or CAT")

    monkeypatch.setattr(bridge, "LineKeyer", forbidden)
    monkeypatch.setattr(bridge.Rig, "_cmd", forbidden)
    r = bridge.Rig(None, armed=False, log=lambda m: None)
    try:
        assert not r.key(True)
        assert r.unkey_hard("test")
        r.shutdown()
    finally:
        r.retire()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_the_bridge_answers_every_signal_that_ends_a_session(tmp_path, sig):
    proc, events = _run_driver(tmp_path, _KEY_AND_WAIT, tool_handlers=True)
    try:
        _wait_for_key(proc, events)
        proc.send_signal(sig)
        assert proc.wait(timeout=6) != 0
        assert "False" in _events(events), f"{sig.name} left the transmitter up"
    finally:
        _stop(proc)


def test_the_connect_tool_takes_the_bridges_handlers_and_not_a_copy():
    kc = corpora.harness("kestrel_connect")
    assert kc.Rig is bridge.Rig
    assert kc._install_unkey_handlers is bridge._install_unkey_handlers
    assert bridge.FATAL_SIGNALS == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def test_the_deadman_is_the_kernels_and_needs_no_new_thread(rig):
    r = rig()
    prior = signal.getsignal(signal.SIGALRM)
    threads = threading.active_count()
    try:
        r._arm_deadman()
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        assert threading.active_count() == threads
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior)
