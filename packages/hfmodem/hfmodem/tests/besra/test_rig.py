# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

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
from hfmodem.tests import fakeclaim

_FAKE = r"""#!/usr/bin/env python3
import os, sys
a = sys.argv[1:]
serial = a[a.index("-r") + 1]                      # we pass the log path as the serial
log = serial + ".cmds"
pos, i = [], 0
while i < len(a):
    if a[i] in ("-m", "-r", "-s", "-P"): i += 2; continue
    pos.append(a[i]); i += 1
if pos[:1] == ["\\dump_caps"]:
    print("Model name: Fake FT-891"); sys.exit(0)
if pos[:1] == ["F"] and pos[-1:] == ["f"]:         # qsy: set then read back, one shot
    open(log, "a").write("ONESHOT " + " ".join(pos) + "\n")
    print(pos[1]); sys.exit(0)
if pos:
    open(log, "a").write("ONESHOT " + " ".join(pos) + "\n"); sys.exit(0)
if os.environ.get("BESRA_FAKE_DIAGNOSTICS"):
    for stream in (sys.stdout.buffer, sys.stderr.buffer):
        stream.write(b"diagnostic\n" * 20000); stream.flush()
with open(log, "a") as f:                          # persistent: record piped commands
    for line in sys.stdin:
        f.write("PIPE " + line); f.flush()
"""


@pytest.fixture
def native_cat(monkeypatch):
    """Direct CAT children, admitted and run as plain subprocesses: the
    persistent one names its device, a one-shot carries only the claim."""
    def spawn(argv, **kwargs):
        serial = argv[argv.index("-r") + 1]
        assert argv[argv.index("-P") + 1] == "NONE"
        assert kwargs in (dict(claim_fd=9, long_lived=True, device=serial), dict(claim_fd=9))
        return fakeclaim.spawn(argv, **kwargs)

    return fakeclaim.install(monkeypatch, spawner=spawn)


@pytest.fixture
def fake_rig(tmp_path, native_cat):
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
    assert direct._base() == [str(exe), "-m", "1036", "-r", "/dev/ttyUSB0", "-s", "38400", "-P", "NONE"]
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


def test_only_cat_commands_are_persistent_pipe_writes(fake_rig):
    rig, log = fake_rig
    rig.set_freq(7103500)
    rig.set_mode("PKTUSB")
    rig.ptt(True)
    rig.ptt(False)
    for _ in range(50):
        c = _cmds(log)
        if f"PIPE M PKTUSB {FILTER_HZ}" in c and "F 7103500" in c:
            break
        time.sleep(0.02)
    assert "PIPE F 7103500" in c
    assert f"PIPE M PKTUSB {FILTER_HZ}" in c
    assert "PIPE T" not in c
    # one long-lived process served them all — not a spawn per command
    assert rig._proc is not None and rig._proc.alive()


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


def test_both_native_diagnostic_pipes_are_drained(fake_rig, monkeypatch):
    # More than either pipe can hold: the child cannot consume a CAT command
    # until both streams have readers. No hardware or supervisor is involved.
    monkeypatch.setenv("BESRA_FAKE_DIAGNOSTICS", "1")
    rig, log = fake_rig
    assert rig._write("F", "7103500")
    assert "PIPE F 7103500" in _cmds(log)
    assert rig._proc.alive()


def test_qsy_sets_and_reads_back_in_one_shot(fake_rig):
    """The set and its readback travel in one rigctl invocation: the persistent
    pipe's writes are asynchronous, so a set through it followed by a
    kill-and-read could overtake the set and verify nothing."""
    rig, log = fake_rig
    rig.set_freq(7000000)                          # opens the persistent process
    assert rig.qsy(7103500) == 7103500
    assert "ONESHOT F 7103500 f" in _cmds(log)
    assert rig._proc is None                       # reads need exclusive access


def test_qsy_answers_none_when_the_rig_will_not_say(tmp_path, native_cat):
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


def test_unkey_closes_cat_without_sending_ptt(fake_rig):
    rig, log = fake_rig
    rig.set_freq(7103500)
    _cmds(log)
    proc = rig._proc
    rig.unkey("test")
    assert rig._proc is None
    assert not proc.alive()
    assert "T 0" not in _cmds(log)
    assert rig._write("F", "7200000") is False


def test_a_clean_stop_is_not_announced_as_an_emergency(fake_rig, caplog):
    """One `emergency unkey: stop` at WARNING closed every clean ARDOP run on
    2026-08-20. The ladder is right to run at teardown; the word is not. What
    the shutdown did is already in the log at INFO, from the ladder itself."""
    import logging

    rig, log = fake_rig
    rig.set_freq(7103500)
    _cmds(log)
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")

    rig.stop()

    assert rig._proc is None
    assert "T 0" not in _cmds(log)
    assert "emergency" not in caplog.text, caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text


def test_the_atexit_backstop_does_not_run_the_ladder_twice(fake_rig, caplog):
    import logging

    rig, log = fake_rig
    rig.set_freq(7103500)
    _cmds(log)
    rig.stop()
    assert "T 0" not in _cmds(log)
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")

    rig._atexit_unkey()

    assert "T 0" not in _cmds(log)
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


def test_pipe_reopens_after_the_process_dies(fake_rig):
    rig, log = fake_rig
    rig.set_freq(7100000)
    _cmds(log)
    rig._proc.kill()                              # simulate a dropped process
    assert rig._proc.wait_gone(2)
    rig.set_freq(7200000)                          # must transparently reopen
    for _ in range(50):
        if "F 7200000" in _cmds(log):
            break
        time.sleep(0.02)
    assert "PIPE F 7200000" in _cmds(log)
    assert rig._proc is not None and rig._proc.alive()


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
