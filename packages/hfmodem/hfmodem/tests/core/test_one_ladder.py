# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One failure, one escalation: the unkey ladder is the same object in all three.

The stuck-key defect was found and fixed three times — shrike, besra, the VARA
bridge — on different days, because each modem carried private answers to the
same four questions: how many times to retry, whether to believe a write, what
an unconfirmed unkey means, when to stop trusting the rig. The answers live in
`core.ptt.Keyer` now, and this file is the pin: the same failure, injected once,
must produce the same escalation whichever modem is holding the key — which is
what makes a fix in one a fix in all.

The failure injected is the one the record keeps producing. On 2026-08-13 every
armed VARA attempt held ~8.4 s of carrier past its last sample: four ``T 0``s in
a row were accepted by rigctld and answered with silence, ~1.5 s each, before
the ladder consulted the keying line — which read LOW at once, first try, eight
of eight across the night (`working/rig-session-20260813-*`; the `_Wire` mode
was "accepted the command and went silent" every time). The rule that ends that
class, pinned here at the core and at each modem: **a transport that answers
keeps the unkey budget; a transport that goes quiet forfeits it after one ask
when a keying line is on hand** — and the line's readback, not the daemon's
silence, is the verdict the operator is given.

Everything runs against fakes: a rigctl that exits nonzero, a socket that
accepts and never answers, and `drop_rts` intercepted at the one place it is
called from now. No serial port, no daemon, no transmitter.
"""
from __future__ import annotations

import atexit
import logging
import re
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

#: The one sentence every modem's operator reads when the daemon forfeits and
#: the line answers. Matching it in all three logs is the sameness being pinned;
#: only the transport's name may differ.
_CONFIRMED_DOWN = re.compile(
    r"\*\*\* \S+ stopped answering \(.+\) — PTT CONFIRMED DOWN on the line "
    r"itself\. Nothing further will be transmitted through this rig\. \*\*\*")

#: A keying line the arm gates accept (a character device) whose drop is then
#: intercepted at `core.ptt.drop_rts` — nothing opens it.
LINE = "/dev/null"


@pytest.fixture
def line(monkeypatch):
    """`drop_rts`, intercepted where the ladder calls it, confirming LOW."""
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or True)
    return dropped


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


def _asks(exe: Path) -> list[str]:
    calls = Path(str(exe) + ".calls")
    if not calls.exists():
        return []
    return [ln for ln in calls.read_text().splitlines() if "T 0" in ln]


# -- the rule itself, at the core --------------------------------------------------


def test_an_answering_transport_keeps_the_budget_a_quiet_one_forfeits_it(line):
    """The split the 2026-08-13 record settles. A daemon that answers ``T 0``
    and still reports the rig up is alive with the delay at the radio, and
    repeating the write is the only medicine — the budget bounds how long. A
    daemon that goes silent gives a retry nothing to use, and the line is one
    ioctl away, so its first silence spends its turn."""

    class _T:
        place, peer = "rig", "fakectl"
        paced, reads_back = True, True
        attempt_timeout = 0.05
        taken_note = "unverified"

        def __init__(self, quiet: bool) -> None:
            self.quiet = quiet
            self.asks = 0

        def drive(self, on, timeout, *, confirm_silence=True):
            self.asks += 1
            return core_ptt.Attempt(
                "failed", why="rig still reports PTT on", detail="d",
                state=None if self.quiet else True,
                state_read=not self.quiet, quiet=self.quiet)

        def release(self):
            return None

    said: list[str] = []
    alive = _T(quiet=False)
    assert core_ptt.Keyer(alive, said.append, ptt_device=LINE).unkey(
        budget=0.4, attempt_timeout=0.05) is True
    assert alive.asks > 1, "an answering transport was denied its budget"
    assert line == [LINE], "the budget ended without the line being consulted"

    line.clear()
    said.clear()
    quiet = _T(quiet=True)
    assert core_ptt.Keyer(quiet, said.append, ptt_device=LINE).unkey(
        budget=0.4, attempt_timeout=0.05) is True
    assert quiet.asks == 1, (
        f"{quiet.asks} asks went to a transport that answered the first with "
        f"silence — the shape that held 4.7 s of unintended carrier on "
        f"2026-08-13, four asks at a time")
    assert line == [LINE]
    assert _CONFIRMED_DOWN.search(" ".join(said)), said


# -- the same failure, one modem at a time ------------------------------------------


def test_shrike_meets_the_one_ladder(tmp_path, capsys, line):
    exe = _dead_rigctl(tmp_path)
    rig = ota.Rig(1036, LINE, 38400, ptt_type="RTS", ptt_port=LINE,
                  rigctl_dir=str(exe.parent))
    t0 = time.monotonic()
    rig.stop()
    took = time.monotonic() - t0
    out = capsys.readouterr().out
    assert line == [LINE]
    assert _CONFIRMED_DOWN.search(out), out
    assert len(_asks(exe)) == 1, "a quiet one-shot was asked again"
    assert took < 2.0, f"the ladder spent {took:.1f}s on a daemon that was gone"
    assert rig.ptt(True) is False, "the rig keyed again after the ladder ended it"


def test_besra_meets_the_one_ladder(tmp_path, caplog, line):
    from hfmodem.besra.radio import Rig
    exe = _dead_rigctl(tmp_path)
    rig = Rig(1036, LINE, 38400, rigctl=str(exe), ptt_device=LINE)
    atexit.unregister(rig._atexit_unkey)
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")
    t0 = time.monotonic()
    rig.unkey("watchdog")
    took = time.monotonic() - t0
    assert line == [LINE]
    assert _CONFIRMED_DOWN.search(caplog.text), caplog.text
    assert len(_asks(exe)) == 1, "a quiet one-shot was asked again"
    assert took < 2.0, f"the ladder spent {took:.1f}s on a daemon that was gone"
    assert rig.retired
    assert rig.ptt(True) is False, "the rig keyed again after the ladder ended it"


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
def test_the_bridge_meets_the_one_ladder_and_84s_of_carrier_cannot_recur(line):
    """THE REGRESSION. Keyed 3.7 s ago, the daemon accepts ``T 0`` and goes
    silent: last night this bought four asks and ~4.7 s of unintended carrier
    before the line was consulted. Now silence forfeits the budget — one ask,
    the line, the verdict — and the wire record still reaches the log."""
    import vara_rig_bridge
    said: list[str] = []
    srv = _SilentDaemon()
    srv.start()
    rig = vara_rig_bridge.Rig(f"127.0.0.1:{srv.port}", armed=True,
                              log=said.append, ptt_device=LINE)
    try:
        rig.keyed = True
        rig.key_since = time.time() - 3.7
        t0 = time.monotonic()
        assert rig.key(False) is True
        took = time.monotonic() - t0
    finally:
        rig._stop.set()
        srv.close()
    unkeys = [g for g in srv.got if b"T 0" in g]
    assert len(unkeys) == 1, (
        f"{len(unkeys)} T 0s went to a daemon that answered the first with "
        f"silence — the ladder is spending its budget on the component that "
        f"already went quiet")
    assert took < 2.0, f"the unkey held the carrier {took:.1f}s asking a silent daemon"
    assert line == [LINE]
    joined = " ".join(said)
    assert _CONFIRMED_DOWN.search(joined), said
    # The forensics survive the shortcut: which way the daemon failed, the
    # bytes, and how long the carrier had been standing.
    assert any("PTT OFF attempt 1" in m and "went silent" in m
               and r"sent b'T 0\n'" in m and "PTT held" in m for m in said), said
    # And never the go-to-the-radio alarm over a line that read low.
    assert "MAY BE STUCK" not in joined and "manually" not in joined, said
    assert rig.retired
    assert rig.key(True) is False, "the rig keyed again after the ladder ended it"
    assert rig.keyed is False


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
    """Load-bearing safety text and the drop rung, kept in `core.ptt.Keyer`.

    A copy that contains the right words is still a copy — three of them is how
    the same defect came to be fixed three times. So the pin is on absence: no
    modem calls `drop_rts` or spells the ladder's verdicts on its own.
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
            f"{name} reaches for the drop rung itself; the ladder owns it")
        for phrase in ("CONFIRMED DOWN", "MAY BE STUCK"):
            assert phrase not in text, (
                f"{name} spells {phrase!r} itself: the escalation's words "
                f"belong to core.ptt.Keyer and nowhere else")
