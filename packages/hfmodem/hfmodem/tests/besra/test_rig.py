# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The persistent-`rigctl` Rig, exercised against a fake rigctl — no radio.

The real-rig PTT path (alignment item 4) fires only when a Rig is built with a
serial device, so nothing in the suite touched it: set/PTT are pipe writes to one
long-lived process, and the emergency unkey deliberately does NOT use that pipe —
it kills the process and keys down through a fresh one-shot, so a wedged pipe can
never block the unkey. This drives that logic against a stand-in rigctl that just
records what it was told, so the orchestration is a test rather than a claim.
"""

from __future__ import annotations

import atexit
import stat
import sys
import time
import types
from pathlib import Path

import pytest

from hfmodem.besra.radio import Rig, tune_atu
from hfmodem.core import audio
from hfmodem.core.occupied import FILTER_HZ

_FAKE = r"""#!/usr/bin/env python3
import sys
a = sys.argv[1:]
serial = a[a.index("-r") + 1]                      # we pass the log path as the serial
log = serial + ".cmds"
pos, i = [], 0
while i < len(a):                                  # strip -m/-r/-s <val>, keep positionals
    if a[i] in ("-m", "-r", "-s"): i += 2; continue
    pos.append(a[i]); i += 1
if pos[:1] == ["\\dump_caps"]:
    print("Model name: Fake FT-891"); sys.exit(0)
if pos[:1] == ["F"] and pos[-1:] == ["f"]:         # qsy: set then read back, one shot
    open(log, "a").write("ONESHOT " + " ".join(pos) + "\n")
    print(pos[1]); sys.exit(0)
if pos:                                            # one-shot command (the independent unkey)
    open(log, "a").write("ONESHOT " + " ".join(pos) + "\n"); sys.exit(0)
with open(log, "a") as f:                          # persistent: record piped commands
    for line in sys.stdin:
        f.write("PIPE " + line); f.flush()
"""


@pytest.fixture
def fake_rig(tmp_path):
    exe = tmp_path / "fake_rigctl"
    exe.write_text(_FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    serial = str(tmp_path / "rig")                 # the fake writes <serial>.cmds
    rig = Rig(1036, serial, 38400, rigctl=str(exe))
    yield rig, Path(serial + ".cmds")
    # Rig registers an atexit unkey; drop it so it does not spawn against a
    # torn-down tmp_path at interpreter exit.
    atexit.unregister(rig._atexit_unkey)
    rig._kill()


def _cmds(log: Path, tries: int = 50) -> str:
    for _ in range(tries):                         # the pipe flushes async; give it a moment
        if log.exists():
            return log.read_text()
        time.sleep(0.02)
    return ""


def test_base_routes_direct_or_through_rigctld(tmp_path):
    """Direct CAT opens the serial port; a rigctld address routes through the shared
    daemon as netrigctl (model 2) — the station rule so nothing contends for the port."""
    exe = tmp_path / "rigctl"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    direct = Rig(1036, "/dev/ttyUSB0", 38400, rigctl=str(exe))
    assert direct._base() == [str(exe), "-m", "1036", "-r", "/dev/ttyUSB0", "-s", "38400"]
    atexit.unregister(direct._atexit_unkey)

    daemon = Rig(1036, "/dev/ttyUSB0", 38400, rigctl=str(exe), rigctld="localhost:4532")
    assert daemon._base() == [str(exe), "-m", "2", "-r", "localhost:4532"]
    atexit.unregister(daemon._atexit_unkey)


def test_a_rigctl_nobody_can_find_refuses_at_construction(tmp_path, monkeypatch):
    """The bare default used to ride along until a `Popen` deep in a keying path
    raised FileNotFoundError with the audio device already open. The refusal now
    happens where `shrike.ota`'s does — at construction, naming what is missing
    and how to supply it."""
    monkeypatch.setenv("PATH", str(tmp_path))       # a PATH with no hamlib on it
    with pytest.raises(SystemExit, match="NOT KEYING.*rigctl is not on PATH"):
        Rig(1036, "/dev/ttyUSB0", 38400)
    with pytest.raises(SystemExit, match="NOT KEYING: no executable rigctl"):
        Rig(1036, "/dev/ttyUSB0", 38400, rigctl=str(tmp_path / "nowhere/rigctl"))


def test_set_and_ptt_are_persistent_pipe_writes(fake_rig):
    rig, log = fake_rig
    rig.set_freq(7103500)
    rig.set_mode("PKTUSB")
    rig.ptt(True)
    rig.ptt(False)
    for _ in range(50):
        c = _cmds(log)
        if "T 0" in c and "F 7103500" in c:
            break
        time.sleep(0.02)
    assert "PIPE F 7103500" in c
    assert f"PIPE M PKTUSB {FILTER_HZ}" in c
    assert "PIPE T 1" in c and "PIPE T 0" in c
    # one long-lived process served them all — not a spawn per command
    assert rig._proc is not None and rig._proc.poll() is None


def test_set_mode_never_asks_for_the_rigs_default_width(fake_rig):
    """``M <mode> 0`` is not "leave the filter alone" — hamlib's third argument is
    the passband and 0 means "use the rig's default for this mode", which on this
    FT-891 in PKTUSB is 1700 Hz. So every retune reset the operator's filter to
    1700 without saying so, and the front panel read 1700 on 2026-08-15 while two
    callers believed they had asked for 3000 and 2700. An explicit width, and
    never that one.
    """
    rig, log = fake_rig
    rig.set_mode("PKTUSB")
    rig.set_mode("PKTUSB", 2400)
    for _ in range(50):
        c = _cmds(log)
        if c.count("M PKTUSB") >= 2:
            break
        time.sleep(0.02)
    assert f"PIPE M PKTUSB {FILTER_HZ}" in c
    assert "PIPE M PKTUSB 2400" in c
    assert "M PKTUSB 0" not in c


def test_qsy_sets_and_reads_back_in_one_shot(fake_rig):
    """The set and its readback travel in one rigctl invocation: the persistent
    pipe's writes are asynchronous, so a set through it followed by a
    kill-and-read could overtake the set and verify nothing."""
    rig, log = fake_rig
    rig.set_freq(7000000)                          # opens the persistent process
    assert rig.qsy(7103500) == 7103500
    assert "ONESHOT F 7103500 f" in _cmds(log)
    assert rig._proc is None                       # reads need exclusive access


def test_qsy_answers_none_when_the_rig_will_not_say(tmp_path):
    exe = tmp_path / "rigctl"
    exe.write_text("#!/bin/sh\nexit 0\n")          # takes the command, reports nothing
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    rig = Rig(1036, str(tmp_path / "rig"), 38400, rigctl=str(exe))
    try:
        assert rig.qsy(7100000) is None
    finally:
        atexit.unregister(rig._atexit_unkey)


def test_identify_reads_the_model(fake_rig):
    rig, _ = fake_rig
    rig.set_freq(7000000)                          # opens the persistent process
    assert rig.identify() == "Fake FT-891"
    assert rig._proc is None                        # identify kills it for exclusive access


def test_unkey_is_independent_of_the_pipe(fake_rig):
    rig, log = fake_rig
    rig.ptt(True)
    _cmds(log)
    proc = rig._proc
    rig.unkey("test")
    # the persistent process is killed (freeing the CAT port)...
    assert rig._proc is None
    assert proc.poll() is not None
    # ...and the unkey went out as a fresh one-shot, not down the (now-dead) pipe
    assert "ONESHOT T 0" in _cmds(log)


def test_a_clean_stop_is_not_announced_as_an_emergency(fake_rig, caplog):
    """One `emergency unkey: stop` at WARNING closed every clean ARDOP run on
    2026-08-20. The ladder is right to run at teardown; the word is not. What
    the shutdown did is already in the log at INFO, from the ladder itself."""
    import logging

    rig, log = fake_rig
    rig.ptt(True)
    _cmds(log)
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")

    rig.stop()

    assert "ONESHOT T 0" in _cmds(log), "a quieter teardown that stopped unkeying"
    assert "emergency" not in caplog.text, caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text


def test_the_atexit_backstop_does_not_run_the_ladder_twice(fake_rig, caplog):
    """`stop` and the atexit handler both reach the unkey, 7 ms apart, and on
    the daemon path the second ran the whole ladder again — a second one-shot
    at a CAT port already given up, and two more lines between the operator and
    the reason the session ended. After `stop` nothing can key through this rig
    again, so the backstop has nothing left to put down."""
    import logging

    rig, log = fake_rig
    rig.ptt(True)
    _cmds(log)
    rig.stop()
    assert _cmds(log).count("ONESHOT T 0") == 1
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")

    rig._atexit_unkey()

    assert _cmds(log).count("ONESHOT T 0") == 1, "the backstop keyed down a second time"
    assert caplog.records == [], caplog.text


def test_a_real_emergency_still_says_so(fake_rig, caplog):
    """The other half: the watchdog, the refused key-up and the signal handlers
    all arrive at `unkey`, and each is a reason to shout."""
    import logging

    rig, _ = fake_rig
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")

    rig.unkey("watchdog: keyed past 30.0s")

    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == \
        ["emergency unkey: watchdog: keyed past 30.0s"], caplog.text


def test_an_unkey_the_oneshot_could_not_confirm_still_escalates(tmp_path, monkeypatch,
                                                                caplog):
    """rigctl reports a dead rigctld by EXITING NONZERO in milliseconds --
    connection refused, or an in-band refusal -- and `subprocess.run` does not
    raise for that. The escalation read it as success: no `drop_rts`, no
    alarm, no retire, nothing in the log. The returncode check lives in
    `core.ptt.OneShotRigctl` now, for every modem at once, and the ladder --
    with the drop this intercepts -- in `core.ptt.Keyer`."""
    import logging

    from hfmodem.core import ptt as core_ptt

    exe = tmp_path / "fake_rigctl"
    exe.write_text("#!/bin/sh\nexit 2\n")
    exe.chmod(0o755)
    rig = Rig(1036, "/dev/null", 38400, rigctl=str(exe), ptt_device="/dev/null")
    atexit.unregister(rig._atexit_unkey)
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or False)
    monkeypatch.setattr(core_ptt, "UNKEY_BUDGET_S", 0.2)   # real seconds, hurried
    caplog.set_level(logging.ERROR, logger="hfmodem.besra.radio")

    rig.unkey("daemon dead")

    assert dropped == ["/dev/null"], "the last resort never ran on a nonzero exit"
    assert "MAY BE STUCK" in caplog.text and "manually" in caplog.text, caplog.text
    assert rig.retired, ("a transmitter nobody has confirmed down is not one "
                         "this program may key again")


def test_an_unkey_the_oneshot_confirmed_does_not_escalate(fake_rig, monkeypatch):
    """...and only the unconfirmed one. `drop_rts` opens the keying port, which
    is what makes it work and what makes it wrong as a routine unkey -- and an
    alarm that fires on every clean shutdown is one an operator learns past."""
    from hfmodem.core import ptt as core_ptt

    rig, _ = fake_rig
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or False)
    rig.unkey("test")
    assert dropped == []
    assert not rig.retired


def test_pipe_reopens_after_the_process_dies(fake_rig):
    rig, log = fake_rig
    rig.set_freq(7100000)
    _cmds(log)
    rig._proc.kill(); rig._proc.wait()             # simulate a dropped process
    rig.set_freq(7200000)                          # must transparently reopen
    for _ in range(50):
        if "F 7200000" in _cmds(log):
            break
        time.sleep(0.02)
    assert "PIPE F 7200000" in _cmds(log)
    assert rig._proc is not None and rig._proc.poll() is None


def test_tune_atu_warms_audio_then_keys_then_unkeys(monkeypatch):
    """The safe keyed-carrier discipline the stuck-PTT incident violated: warm the
    audio device BEFORE PTT (no keyed-into-dead-air), then key, then the tone, then
    unkey. Order is the assertion."""
    ev = []

    class _FR:
        settle = 0.0                               # the wait is the radio's, not ours
        def ptt(self, on): ev.append(f"ptt{int(on)}")
        def unkey(self, why=""): ev.append("unkey")

    class _Out:
        def __init__(self, **kw): pass
        def start(self): pass
        def stop(self): pass
        def close(self): ev.append("close")
        def write(self, a): ev.append("play")

    fake = types.ModuleType("sounddevice")
    fake.OutputStream = _Out
    fake.stop = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(audio, "_WARMED", set())   # a fresh card is a cold one

    tune_atu(_FR(), seconds=0.02)
    # The close matters as much as the order: a stream left open after the tune
    # keeps a data-audio-keyed rig transmitting with CAT reporting PTT down.
    assert ev == ["play", "close", "ptt1", "play", "close", "ptt0"], ev
