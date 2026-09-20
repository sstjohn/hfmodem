# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra keying the line itself, with no daemon in the transmit path.

On 2026-08-09 rigctld stopped answering on the FIRST key-down of a session, every
session, at 50 W into a real antenna — and none of it reproduced off the air: CAT
alone answered 40 of 40, key/unkey without audio 15 of 15, CAT under a streaming
codec 30 of 30. RF into the USB serial adapter is what those three lack. besra's
emergency unkey caught it twice in one session and that is the wrong place to be
catching it, so with ``line_ptt`` the ordinary key and unkey are ioctls on a
descriptor `Rig` holds for the session and hamlib is left to set frequency and
mode, before there is any RF about.

WHAT IS TESTED HERE, AND WHAT CANNOT BE. Whether the kernel moves a pin is the
kernel's business, and a pty will not stand in for a real port — on macOS
``TIOCMBIS`` against a pty returns ENOTTY, so a fixture built on one tests nothing
but its own fiction. So the decisions are asserted against a fake descriptor, the
same way `tests/core/test_ptt_last_resort.py` does. What no test here can show is
behaviour under RF; only the rig can.
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import struct
import subprocess
import termios

import pytest

from hfmodem.besra.radio import Rig


class _FakePort:
    """Answers TIOCMGET from its own bit state, and records what was asked.

    `stuck` is a bitmask TIOCMBIC cannot clear and `dead` one TIOCMBIS cannot
    set — lines the driver accepts the command for and does not move."""

    def __init__(self, stuck: int = 0, dead: int = 0) -> None:
        self.bits = termios.TIOCM_RTS       # as an open() leaves it
        self.stuck = stuck
        self.dead = dead
        self.calls: list[int] = []
        self.closed = False

    def ioctl(self, fd, request, arg):
        self.calls.append(request)
        if request == termios.TIOCMBIC:
            self.bits &= ~(struct.unpack("I", arg)[0] & ~self.stuck)
            return arg
        if request == termios.TIOCMBIS:
            self.bits |= struct.unpack("I", arg)[0] & ~self.dead
            return arg
        if request == termios.TIOCMGET:
            return struct.pack("I", self.bits)
        raise AssertionError(f"unexpected ioctl {request:#x}")

    @property
    def high(self) -> bool:
        return bool(self.bits & termios.TIOCM_RTS)


class _BlindPort(_FakePort):
    """Takes TIOCMBIS and TIOCMBIC, fails TIOCMGET — a driver that accepts the
    command and cannot report the line, so the pin can be high with nothing
    able to confirm it."""

    def ioctl(self, fd, request, arg):
        if request == termios.TIOCMGET:
            self.calls.append(request)
            raise OSError(6, "Device not configured")
        return super().ioctl(fd, request, arg)


def _wire(monkeypatch, p: _FakePort) -> _FakePort:
    monkeypatch.setattr(os, "open", lambda *a, **k: 7)
    monkeypatch.setattr(os, "close", lambda fd: setattr(p, "closed", True))
    monkeypatch.setattr(fcntl, "ioctl", p.ioctl)
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: [0, 0, 0, 0, 0, 0, []])
    monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: None)
    return p


def _no_daemon(*a, **k):
    raise AssertionError("this rig reached for hamlib")


@pytest.fixture
def line_rig(caplog, monkeypatch):
    """Build a line-keyed rig on a given fake port, every route to hamlib closed.

    The guard belongs to the fixture rather than to one test because it is a
    property of the whole configuration: a `line_ptt` rig is given a rigctld
    address it can reach nothing at, and is expected never to try. Anything that
    still asked rigctl to key or unkey fails here rather than on the air.
    """
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")
    monkeypatch.setattr(Rig, "_write", _no_daemon)
    monkeypatch.setattr(subprocess, "Popen", _no_daemon)
    monkeypatch.setattr(subprocess, "run", _no_daemon)
    rigs = []

    def build(p: _FakePort) -> Rig:
        _wire(monkeypatch, p)
        # rigctl="/bin/sh": an executable that always exists, so construction
        # passes the missing-hamlib refusal; the daemon guard proves nothing
        # spawns it.
        r = Rig(1036, "/dev/no-such-cat", 38400, rigctl="/bin/sh",
                rigctld="127.0.0.1:1", ptt_device="/dev/null", line_ptt=True)
        rigs.append(r)
        return r

    yield build
    # The atexit unkey would run against a torn-down fake, on a descriptor number
    # that by then belongs to somebody else.
    for r in rigs:
        atexit.unregister(r._atexit_unkey)
        r._keyer = None


@pytest.fixture
def port():
    return _FakePort()


@pytest.fixture
def rig(line_rig, port):
    return line_rig(port)


# -- the transmit path -------------------------------------------------------


def test_the_transmit_path_keys_the_line_and_not_the_daemon(rig, port):
    rig.ptt(True)
    assert port.high, "the rig never keyed the line"
    rig.ptt(False)
    assert not port.high
    assert not rig.retired, "a clean unkey must not retire the rig"


def test_the_emergency_unkey_takes_the_line_down_itself(rig, port, caplog):
    """The watchdog, `stop` and the signal handlers all arrive here. It was the
    hamlib one-shot in this path that timed out twice in one session.

    A line that ends low is not enough to assert: the last-resort `drop_rts`
    OPENS the port to do it, which is exactly the sequence that must not be
    reached mid-session — an open raises RTS, so a rig that is genuinely down
    gets keyed for the length of an `open()`. So the evidence is that the port
    was never reopened and nothing was said about a failed one-shot.
    """
    rig.ptt(True)
    caplog.clear()
    rig.unkey("watchdog")
    assert not port.high
    assert not port.closed, "the unkey reopened the port it already holds"
    assert "one-shot" not in caplog.text, caplog.text


def test_arming_the_line_leaves_it_low(rig, port):
    """Opening a serial port asserts RTS, which here IS key-down. Holding the
    descriptor for a session is only safe because that assertion is undone on the
    way in — before any caller can key, and before there is audio to modulate it."""
    assert port.calls, "the line was never touched at arm time"
    assert port.calls[0] == termios.TIOCMBIC, "the line must be cleared first"
    assert not port.high


def test_a_key_that_does_not_reach_the_line_is_not_reported_as_transmitting(
        line_rig, caplog):
    """Reporting a key that did not happen is worse than failing it: the modem
    plays a whole burst into a dead line and waits out an answer nobody was ever
    asked for."""
    p = _FakePort(dead=termios.TIOCM_RTS)
    rig = line_rig(p)
    rig.ptt(True)
    assert "NOT TRANSMITTING" in caplog.text, caplog.text
    assert not p.high
    assert not rig.retired, "a line confirmed back low is still a line to trust"


def test_an_unconfirmed_key_up_still_lowers_the_line(line_rig, caplog):
    """TIOCMBIS accepted, TIOCMGET broken: the pin may be high while the log
    says NOT TRANSMITTING and nothing here believes the rig is keyed — which
    blinds the watchdog, the one guard sized for a line left up. The only safe
    exit is the drop-to-be-sure the bridge's line branch already takes."""
    p = _BlindPort()
    rig = line_rig(p)
    rig.ptt(True)
    assert not p.high, (
        "the ioctl may have raised the pin; an unconfirmed key-up must deassert")
    assert "NOT TRANSMITTING" in caplog.text, caplog.text
    assert rig.retired, "a line that cannot be read cannot be trusted to key again"


def test_a_line_that_will_not_unkey_raises_the_alarm_and_retires(line_rig, caplog):
    """The one case that must keep the loud message: the line itself has failed,
    so nothing is left to try and the operator does need to go to the rig."""
    rig = line_rig(_FakePort(stuck=termios.TIOCM_RTS))
    rig.ptt(False)
    assert "MAY BE STUCK" in caplog.text and "manually" in caplog.text, caplog.text
    assert rig.retired, "a transmitter that will not go down must not be keyed again"


def test_a_retired_rig_still_will_not_key_the_line(rig, port):
    rig.retire()
    rig.ptt(True)
    assert not port.high, "a retired rig keyed the transmitter"


def test_the_keyer_s_verdict_reaches_the_caller(line_rig, port):
    """`LineKeyer.key` already knew whether the line moved; `ptt` swallowed the
    answer, so `RadioLink._transmit` played whole frames into a key-up the
    keyer had refused and dropped. What the keyer read back is what `ptt`
    returns."""
    rig = line_rig(port)
    assert rig.ptt(True) is True
    assert rig.ptt(False) is True
    dead = line_rig(_FakePort(dead=termios.TIOCM_RTS))
    assert dead.ptt(True) is False, "an unconfirmed key-up must not read as keyed"


# -- handing the line back ---------------------------------------------------


def test_stop_takes_the_line_down_and_hands_the_port_back(rig, port, caplog):
    rig.ptt(True)
    rig.stop()
    assert not port.high, "shutdown left the transmitter up"
    assert port.closed, "shutdown kept the port against the next session"
    assert "reads LOW" in caplog.text, caplog.text


def test_a_clean_shutdown_says_nothing_loud_on_its_way_out(rig, caplog):
    """`stop` releases the port and the atexit unkey runs after it. Reopening
    would raise RTS, so nothing is reopened — and a shutdown that went perfectly
    must not print the loudest message in the program on its way out.

    The word `emergency` is worth something only while it is rare: this station
    has one genuine emergency unkey, and an operator who has read it at the end
    of every clean run is an operator who reads past it on the one that meant
    it."""
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")
    rig.stop()
    rig._atexit_unkey()
    assert "emergency" not in caplog.text, caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
    assert "reads LOW" in caplog.text, "a clean shutdown left no evidence either"


def test_the_backstop_still_fires_for_an_exit_that_never_reached_stop(rig, port, caplog):
    """The atexit handler earns its warning on the exits `stop` never reaches —
    an uncaught exception out of a keyed session — and only there."""
    caplog.set_level(logging.INFO, logger="hfmodem.besra.radio")
    rig.ptt(True)
    rig._atexit_unkey()
    assert not port.high, "the backstop left the transmitter up"
    assert "emergency unkey: atexit" in caplog.text, caplog.text


# -- the configuration itself ------------------------------------------------


def test_the_line_must_be_named(port):
    """Falling back to the daemon would be the defect wearing the fix's name: the
    operator asked for the line and would be told nothing while every key went
    back over the link RF disrupts."""
    with pytest.raises(ValueError):
        Rig(1036, "/dev/no-such-cat", 38400, rigctl="/bin/sh", line_ptt=True)


def test_without_the_flag_the_daemon_still_keys(monkeypatch):
    """The other three callers of this class are unchanged."""
    r = Rig(1036, "/dev/no-such-cat", 38400, rigctl="/bin/sh", rigctld="127.0.0.1:1")
    atexit.unregister(r._atexit_unkey)
    sent = []
    monkeypatch.setattr(Rig, "_write", lambda self, *a: sent.append(a))
    r.ptt(True)
    r.ptt(False)
    assert sent == [("T", "1"), ("T", "0")]
