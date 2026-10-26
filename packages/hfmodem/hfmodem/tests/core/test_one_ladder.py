# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One failure, one escalation: every modem keys through the same `core.ptt.LineKeyer`.

The stuck-key defect was found and fixed three times — shrike, besra, the VARA
bridge — on different days, because each modem carried private answers to the
same questions: whether to believe a write, what an unconfirmed unkey means,
when to stop trusting the rig. The answers live in `core.ptt.Keyer` now, and
this file is the pin: the same failure, injected once, must produce the same
escalation whichever modem is holding the key.

The record behind it: on 2026-08-13 every armed VARA attempt held ~8.4 s of
carrier past its last sample while rigctld accepted ``T 0`` and went silent,
and the keying line read LOW at once, eight of eight
(`working/rig-session-20260813-*`). So no daemon is on the transmit path at
all: the line is keyed and unkeyed by ioctl, its readback is the verdict, and a
line that will not read low alarms and retires the rig.

Everything runs against fakes: a rigctl that exits nonzero, a socket that
accepts and never answers, and a keying line that only pretends. No serial
port, no daemon, no transmitter.
"""
from __future__ import annotations

import atexit
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from hfmodem.core import ptt as core_ptt
from hfmodem.shrike import ota

_TOOLS = Path(__file__).resolve().parents[5] / "tools"
sys.path.insert(0, str(_TOOLS))

requires_bridge = pytest.mark.skipif(
    not (_TOOLS / "vara_rig_bridge.py").exists(),
    reason=f"{_TOOLS}/vara_rig_bridge.py is not present (installed-wheel run)")

#: A keying line the arm gates accept (a character device); the tests replace
#: `core.ptt.RtsPtt` or the ioctls, so nothing opens it.
LINE = "/dev/null"


def _dead_rigctl(tmp_path: Path) -> Path:
    """A rigctl whose daemon is gone: logs its argv, exits 2 — in milliseconds,
    the way the real one reports a dead rigctld, which `subprocess.run` does
    not raise for."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    exe = d / "rigctl"
    exe.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$0.calls"\nexit 2\n')
    exe.chmod(0o755)
    return exe


# -- the rule itself, at the core --------------------------------------------------


@pytest.mark.parametrize("readback", [True, None])
def test_unconfirmed_line_unkey_alarms_and_retires_without_retry(monkeypatch, readback):
    """No daemon retry budget remains (§4.5); high and unknown both retire."""
    from hfmodem.tests.shrike.test_p1_line_keyer import FakeLine
    port = FakeLine(LINE)
    monkeypatch.setattr(core_ptt, "RtsPtt", lambda _: port)
    said = []
    keyer = core_ptt.LineKeyer(LINE, said.append)
    keyer.arm()
    assert keyer.key(True)
    monkeypatch.setattr(port, "sense", lambda: readback)
    before = port.calls.count(False)
    started = time.monotonic()
    assert keyer.unkey() is False
    assert time.monotonic() - started < 1.0
    assert port.calls.count(False) == before + 1
    assert keyer.must_retire
    assert any("MAY BE STUCK" in m for m in said)
    assert keyer.key(True) is False
    keyer.hand_back()


def test_shrike_meets_the_one_ladder(tmp_path, capsys, monkeypatch):
    from hfmodem.tests.shrike.test_p1_line_keyer import FakeLine
    port = FakeLine(LINE)
    monkeypatch.setattr(core_ptt, "RtsPtt", lambda _: port)
    exe = _dead_rigctl(tmp_path)
    rig = ota.Rig(1036, LINE, 38400, ptt_port=LINE,
                  rigctl_dir=str(exe.parent))
    assert rig.ptt(True)
    t0 = time.monotonic()
    rig.stop()
    took = time.monotonic() - t0
    out = capsys.readouterr().out
    assert port.calls[-3:] == [False, "sense", "release"]
    assert port.released_low is True and port.state is False
    assert "PTT OFF -> line (reads low)" in out
    assert not Path(str(exe) + ".calls").exists(), "PTT reached CAT"
    assert took < 2.0, f"the line ladder took {took:.1f}s"
    assert rig.ptt(True) is False, "the rig keyed again after stop"


def test_besra_meets_the_one_ladder(tmp_path, caplog, monkeypatch):
    from hfmodem.besra.radio import Rig
    from hfmodem.tests.shrike.test_p1_line_keyer import FakeLine
    port = FakeLine(LINE)
    monkeypatch.setattr(core_ptt, "RtsPtt", lambda _: port)
    exe = _dead_rigctl(tmp_path)
    rig = Rig(1036, LINE, 38400, rigctl=str(exe), ptt_device=LINE, line_ptt=True)
    atexit.unregister(rig._atexit_unkey)
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")
    assert rig.ptt(True)
    t0 = time.monotonic()
    rig.unkey("watchdog")
    took = time.monotonic() - t0
    assert port.calls[-2:] == [False, "sense"]
    assert port.state is False
    assert "PTT OFF -> line (reads low)" in caplog.text
    assert not Path(str(exe) + ".calls").exists(), "PTT reached CAT"
    assert took < 2.0
    # An unconfirmed line unkey, unlike a dead CAT daemon, must retire.
    assert rig.ptt(True)
    port.readable = False
    rig.unkey("watchdog")
    assert rig.retired and "MAY BE STUCK" in caplog.text
    assert rig.ptt(True) is False
    port.readable = True
    rig.stop()
    assert port.released_low is True


class _SilentDaemon(threading.Thread):
    """2026-08-13's exact shape: accepts the connection, reads the command,
    answers nothing, holds the socket open."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self.got: list[bytes] = []
        self._open: list[socket.socket] = []
        self._closed = False

    def run(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                self.got.append(conn.recv(64))
            except OSError:
                pass
            self._open.append(conn)

    def close(self) -> None:
        self._closed = True
        self._sock.close()
        for c in self._open:
            c.close()


@requires_bridge
def test_a_line_keyed_rig_does_not_notice_a_wedged_daemon(monkeypatch):
    """The configuration the 2026-08-13 evidence argues for, proven whole: with
    ``line_ptt`` the daemon that dies on the first key-down is simply not on the
    transmit path. Key and unkey complete against a daemon that answers nothing,
    in well under a second, and not one ``T`` byte crosses the socket."""
    import fcntl
    import os
    import termios
    import struct
    import vara_rig_bridge

    class _Port:
        bits = termios.TIOCM_RTS

        def ioctl(self, fd, req, arg):
            if req == termios.TIOCMBIC:
                self.bits &= ~struct.unpack("I", arg)[0]
            elif req == termios.TIOCMBIS:
                self.bits |= struct.unpack("I", arg)[0]
            elif req == termios.TIOCMGET:
                return struct.pack("I", self.bits)
            return arg

    port = _Port()
    monkeypatch.setattr(os, "open", lambda *a, **k: 7)
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(fcntl, "ioctl", port.ioctl)
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: [0, 0, 0, 0, 0, 0, []])
    monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: None)

    srv = _SilentDaemon()
    srv.start()
    said: list[str] = []
    rig = vara_rig_bridge.Rig(f"127.0.0.1:{srv.port}", armed=True,
                              log=said.append, ptt_device=LINE, line_ptt=True)
    try:
        t0 = time.monotonic()
        assert rig.key(True, "burst") is True
        assert port.bits & termios.TIOCM_RTS
        assert rig.key(False) is True
        assert not (port.bits & termios.TIOCM_RTS)
        took = time.monotonic() - t0
    finally:
        rig._stop.set()
        srv.close()
    assert took < 1.0, f"a line key round trip took {took:.2f}s"
    assert not any(b"T" in g for g in srv.got), (
        f"the transmit path reached for the daemon: {srv.got}")
    assert not rig.retired, "a clean line unkey must not retire the rig"


# -- and no modem keeps a private copy ----------------------------------------------


def test_no_modem_spells_the_escalation_itself():
    """Load-bearing safety text and the drop rung, kept in `core.ptt`.

    A copy that contains the right words is still a copy — three of them is how
    the same defect came to be fixed three times. So the pin is on absence: no
    modem calls `drop_rts` or spells the keyer's verdicts on its own; a line it
    does not hold goes down through `core.ptt.drop_unheld_line`.
    """
    from hfmodem.besra import radio
    sources = {
        "shrike/ota.py": Path(ota.__file__).read_text(),
        "besra/radio.py": Path(radio.__file__).read_text(),
    }
    if (_TOOLS / "vara_rig_bridge.py").exists():
        sources["tools/vara_rig_bridge.py"] = (
            _TOOLS / "vara_rig_bridge.py").read_text()
    for name, text in sources.items():
        assert "drop_rts(" not in text, (
            f"{name} reaches for the drop rung itself; use drop_unheld_line")
        for phrase in ("CONFIRMED DOWN", "MAY BE STUCK"):
            assert phrase not in text, (
                f"{name} spells {phrase!r} itself: the escalation's words "
                f"belong to core.ptt.Keyer and nowhere else")
