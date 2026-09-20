# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The keying line: the order of operations, which is where the defects were.

What is and is not provable here, stated up front because it bounds every claim
below. On Darwin a pty returns ENOTTY for `TIOCMGET`, `TIOCMBIS` and `TIOCMBIC` on
**both** ends — measured, not assumed — so no pseudo-terminal can carry a modem
control line. `tcgetattr`/`tcsetattr` do work on one.

That splits cleanly:

  provable here      the sequence of ioctls, the termios flags, the refusals, and
                     that nothing reopens a port it just released
  needs hardware     that a pin moves, that HUPCL drops the line on process death,
                     the open-time transient on a real CP2105, unplug, actuation
                     latency

The first is where every review finding lived: opening a tty raises RTS, so a
deassert has to come first; and a "close and reopen to force a deassert" step
re-raises it. Both are order bugs, and order is exactly what a recording fake can
see.
"""
from __future__ import annotations

import fcntl
import os
import pty
import struct
import termios

import pytest

from hfmodem.core.ptt import PttError, RtsPtt, derive_ptt_port, require_char_device


@pytest.fixture
def recorder(monkeypatch):
    """Record every ioctl in order, and let TIOCM* succeed on a pty."""
    calls: list[tuple[str, int]] = []
    names = {termios.TIOCMBIS: "BIS", termios.TIOCMBIC: "BIC", termios.TIOCMGET: "GET"}
    state = {"lines": 0}
    real = fcntl.ioctl

    def fake(fd, op, arg=0, *a, **kw):
        if op in names:
            bits = struct.unpack("I", arg)[0] if isinstance(arg, bytes) else 0
            calls.append((names[op], bits))
            if op == termios.TIOCMBIS:
                state["lines"] |= bits
            elif op == termios.TIOCMBIC:
                state["lines"] &= ~bits
            else:
                return struct.pack("I", state["lines"])
            return b""
        return real(fd, op, arg, *a, **kw)

    monkeypatch.setattr(fcntl, "ioctl", fake)
    return calls, state


@pytest.fixture
def port():
    controller, device = pty.openpty()
    yield os.ttyname(device)
    for fd in (controller, device):
        try:
            os.close(fd)
        except OSError:
            pass


# --- the order findings -------------------------------------------------------

def test_open_deasserts_before_anything_else(port, recorder):
    """Opening a tty raises RTS and DTR. If the first thing we do is not put them
    down, arm() keys the transmitter."""
    calls, _ = recorder
    p = RtsPtt(port)
    p.open()
    assert calls, "no ioctl at all — the open did not touch the lines"
    op, bits = calls[0]
    assert op == "BIC"
    assert bits & termios.TIOCM_RTS and bits & termios.TIOCM_DTR, (
        "the first ioctl must clear both lines, not just the one we key with")
    p.release()


def test_the_line_is_down_after_open(port, recorder):
    _, state = recorder
    p = RtsPtt(port)
    p.open()
    assert state["lines"] == 0
    assert p.sense() is False
    p.release()


def test_release_deasserts_then_closes_and_never_reopens(port, recorder):
    """The review's sharpest finding: a panic path that closed and reopened the
    fd to 'force' a deassert would re-raise RTS — keying the transmitter it was
    called to bring down."""
    calls, state = recorder
    p = RtsPtt(port)
    p.open()
    p.assert_(True)
    assert state["lines"] & termios.TIOCM_RTS
    calls.clear()
    p.release()
    assert state["lines"] == 0
    # One deassert, then a readback, then the close. The invariant this guards is
    # that nothing SETS a line on the way down — a close-and-reopen to "force" a
    # deassert would raise RTS and key the transmitter it was called to bring
    # down. A read cannot do that, and it is what lets the panic path confirm the
    # unkey on a station whose daemon owns no PTT to be asked about.
    assert [c[0] for c in calls] == ["BIC", "GET"], (
        f"release is a deassert, a readback and a close, got {calls}")
    assert p.released_low is True, "release did not confirm the line went low"
    assert p.sense() is None      # the port is gone, not asserted
    with pytest.raises(PttError, match="not open"):
        p.assert_(True)


def test_release_is_idempotent(port, recorder):
    p = RtsPtt(port)
    p.open()
    p.release()
    p.release()


# --- termios ------------------------------------------------------------------

def test_hupcl_and_clocal_are_set_and_crtscts_cleared(port, recorder):
    """HUPCL governs deassert-on-last-close and hamlib explicitly clears it, so
    it cannot be inherited. CRTSCTS would give the line to the driver."""
    p = RtsPtt(port)
    p.open()
    cflag = termios.tcgetattr(p._fd)[2]
    assert cflag & termios.HUPCL
    assert cflag & termios.CLOCAL
    assert not cflag & termios.CRTSCTS
    p.release()


def test_the_fd_is_close_on_exec(port, recorder):
    """HUPCL fires on *last* close. A child inheriting the port defeats it."""
    p = RtsPtt(port)
    p.open()
    assert fcntl.fcntl(p._fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    p.release()


# --- refusals -----------------------------------------------------------------

def test_a_missing_port_is_refused_by_name():
    """The path must be in the message, at open and from the bare check alike.

    The bare check is what the transmit tools run at their arm gates: on
    2026-08-10 a placeholder default — a path that exists on no machine —
    armed a 27-cycle session that logged 22.1 s of carrier while never
    touching the radio, because nothing between the flag and the ioctl had
    ever asked whether the line was there. Construction stays permissive so
    preflight can report an absent adapter as a finding instead of a crash;
    everything past construction must refuse."""
    with pytest.raises(PttError, match="cu.nonexistent-adapter"):
        RtsPtt("/dev/cu.nonexistent-adapter").open()
    with pytest.raises(PttError, match="cu.usbserial-XXXXB1"):
        require_char_device("/dev/cu.usbserial-XXXXB1")


def test_a_path_that_is_not_a_character_device_is_refused(tmp_path):
    """It used to be refused by accident — TIOCMBIC returning ENOTTY, a message
    with no path in it — which is a refusal only code can love."""
    f = tmp_path / "notatty"
    f.write_bytes(b"")
    with pytest.raises(PttError, match="not a character device"):
        RtsPtt(str(f)).open()


def test_a_character_device_with_no_lines_behind_it_is_still_refused():
    """The stat cannot see past the device node — /dev/null is a character
    device with nothing to key — so the ioctl refusals behind it keep their
    job."""
    with pytest.raises(PttError):
        RtsPtt("/dev/null").open()


def test_an_unknown_line_is_refused():
    with pytest.raises(ValueError, match="rts or dtr"):
        RtsPtt("/dev/null", line="cts")


def test_a_pty_cannot_carry_a_keying_line(port):
    """Without the recorder, a real pty refuses TIOCMBIC with ENOTTY. This is the
    measurement that bounds every claim in this file — and it is the right
    answer: a pseudo-terminal has no modem control lines to offer."""
    with pytest.raises(PttError, match="TIOCMBIC"):
        RtsPtt(port).open()


# --- port derivation ----------------------------------------------------------

def test_the_ptt_interface_is_derived_from_the_cat_one():
    assert derive_ptt_port("/dev/cu.usbserial-XXXXB0") == "/dev/cu.usbserial-XXXXB1"


@pytest.mark.parametrize("bad", ["", "/dev/cu.usbmodem1234", "/dev/ttyS0", "/dev/cu.Bluetooth"])
def test_an_underivable_port_is_refused_rather_than_guessed(bad):
    """Guessing which line keys a transmitter is not a default."""
    with pytest.raises(ValueError):
        derive_ptt_port(bad)


def test_a_trailing_digit_is_not_enough_to_derive_from():
    """On Linux /dev/ttyUSB0 and /dev/ttyUSB1 are two different adapters, not two
    interfaces of one. A rule that advanced any trailing digit would point the
    keying line at hardware nobody asked about — possibly another radio's CAT
    port. Caught by this test, which is why it exists."""
    for name in ("/dev/ttyUSB0", "/dev/ttyACM0", "/dev/cu.usbserial0"):
        with pytest.raises(ValueError):
            derive_ptt_port(name)
