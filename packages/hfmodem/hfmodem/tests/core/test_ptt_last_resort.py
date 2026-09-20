# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keying without the daemon, and the unkey that runs when the daemon is gone.

On 2026-08-09 four unkeys timed out and one session hung after its first burst,
all at 50 W into a real antenna, and none of it reproduced off the air -- CAT
alone answered 40 of 40, key/unkey without audio 15 of 15, CAT under a streaming
codec 30 of 30. RF into the serial adapter is what those three lack. Two things
follow, and both are tested here: the emergency unkey must not travel the link it
is backing up, and the ordinary transmit path should not travel it either, since
on this station PTT *is* a line `RtsPtt` already knows how to drive.

WHAT IS TESTED HERE, AND WHAT CANNOT BE. Whether the kernel drops a pin is the
kernel's business, and a pty will not stand in for a real port -- on macOS
`TIOCMBIS` against a pty returns ENOTTY, so a fixture built on one tests nothing
but its own fiction. So the decisions are asserted against a fake descriptor and
the station's own adapter is exercised when it is present. What no test here can
show is behaviour under RF; only the rig can.
"""
from __future__ import annotations

import fcntl
import os
import re
import struct
import subprocess
import warnings
import sys
import termios
from pathlib import Path

import pytest

from hfmodem.core import busy
from hfmodem.core.ptt import PttError, RtsPtt, arming_refusal, drop_rts
from hfmodem.tests.kestrel import corpora

_TOOLS = Path(__file__).resolve().parents[5] / "tools"
sys.path.insert(0, str(_TOOLS))

#: `vara_rig_bridge` is a station tool rather than a package module, and `tools/`
#: does not cross the publication boundary. The `drop_rts` tests above it need
#: nothing but a fake port and run anywhere; the line-keying tests below drive the
#: bridge, and in a tree without one they say so rather than erroring on import.
requires_bridge = pytest.mark.skipif(
    not (_TOOLS / "vara_rig_bridge.py").exists(),
    reason=f"{_TOOLS}/vara_rig_bridge.py is not present (installed-wheel run)")

_ONAIR = _TOOLS / "onair.sh"
#: What a script started here is run with. `<tree>/.venv/bin/python` once, which no
#: worktree has, so this skipped in every worktree run and said so only in a count.
#: The interpreter running the suite is the one the operator invoked.
_VENV_PY = Path(sys.executable)

#: `tools/lib/ports.sh` sets `VENV="$REPO/.venv/bin"` outright, so the two tests
#: that drive `onair.sh` can only run where the tree it is in has a virtualenv --
#: and a worktree never does. That is a limitation of the launcher rather than of
#: the tests, so it is named rather than worked around, and it warns: a skip
#: reason is invisible without `-rs`, and 36 passing beside it says nothing.
_LAUNCHER_PY = _TOOLS.parent / ".venv" / "bin" / "python"
_NO_LAUNCHER_PY = (
    f"no interpreter at {_LAUNCHER_PY}, which tools/lib/ports.sh names outright -- "
    f"so the two tests that run tools/onair.sh cannot run in this tree at all, and "
    f"neither can the launcher. Every worktree of this checkout is in that state.")
if not _LAUNCHER_PY.exists():
    warnings.warn(_NO_LAUNCHER_PY, stacklevel=1)

requires_launcher_python = pytest.mark.skipif(not _LAUNCHER_PY.exists(),
                                              reason=_NO_LAUNCHER_PY)

requires_launcher = pytest.mark.skipif(
    not _ONAIR.exists(), reason=f"{_ONAIR} is not present (installed-wheel run)")

#: The station's PTT line, taken from the same PTT_PORT the tools/ scripts
#: honour, so one export configures both. Unset means "not at the radio" and the
#: hardware test below skips saying exactly what would turn it on; set, the test
#: FAILS rather than skipping if the line does not drop -- including when the
#: named port is absent. This was a hardcoded path once, and redacting it to a
#: placeholder that exists on no machine turned the guard off everywhere, the
#: station included, while this comment went on promising it had teeth.
STATION_PTT = os.environ.get("PTT_PORT")


#: The port every faked test points at. It must be a real character device —
#: `RtsPtt` now stats the path at construction, before any call a fixture can
#: fake — and /dev/null is the one such device every machine has. Nothing here
#: ever actually opens it: `os.open` is monkeypatched away.
FAKE_DEV = "/dev/null"


class _FakePort:
    """Answers TIOCMGET from its own bit state, and records what was asked.

    `stuck` is a bitmask TIOCMBIC cannot clear and `dead` one TIOCMBIS cannot
    set — lines the driver accepts the command for and does not move."""

    def __init__(self, stuck: int = 0, dead: int = 0):
        self.bits = termios.TIOCM_RTS | termios.TIOCM_DTR   # as an open() leaves it
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


class _BlindPort(_FakePort):
    """Takes TIOCMBIS and TIOCMBIC, fails TIOCMGET — a driver that accepts the
    command and cannot report the line, so the pin can be high with nothing
    able to confirm it."""

    def ioctl(self, fd, request, arg):
        if request == termios.TIOCMGET:
            self.calls.append(request)
            raise OSError(6, "Device not configured")
        return super().ioctl(fd, request, arg)


class _VanishedPort(_FakePort):
    """Accepts the open-time ioctls, then the adapter is gone: everything after
    the second open-time deassert is refused the way a pulled cable refuses it."""

    def ioctl(self, fd, request, arg):
        if self.calls.count(termios.TIOCMBIC) >= 2:
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


@pytest.fixture
def port(monkeypatch):
    return _wire(monkeypatch, _FakePort())


# -- the emergency unkey -----------------------------------------------------------


def test_it_takes_the_line_down_and_reads_it_back(port):
    said = []
    assert drop_rts(FAKE_DEV, said.append) is True
    assert termios.TIOCMBIC in port.calls, "the line was never cleared"
    assert termios.TIOCMGET in port.calls, "the line was never read back"
    assert not (port.bits & termios.TIOCM_RTS)
    assert port.closed, "a leaked descriptor holds the port against the next attempt"
    assert any("LOW (PTT down)" in m for m in said), said


def test_a_line_that_will_not_go_low_is_reported_as_such(monkeypatch):
    """The dangerous case: the ioctls are accepted and the line does not move.
    Saying True here tells a caller the transmitter is down when it is not."""
    p = _wire(monkeypatch, _FakePort(stuck=termios.TIOCM_RTS | termios.TIOCM_DTR))
    said = []
    assert drop_rts(FAKE_DEV, said.append) is False
    assert any("NOT CONFIRMED LOW" in m for m in said), said
    # The two verdicts must part at the first words: this one is read at the
    # moment a transmitter may be stuck, and it must not open by asserting the
    # outcome it then retracts.
    assert not any("taken down" in m for m in said), said
    assert any(m.startswith("NOT CONFIRMED LOW") for m in said), said
    assert p.closed


def test_a_dtr_keyed_rig_is_not_proved_down_off_rts(monkeypatch):
    """The deassert clears both output lines, so the readback must confirm both:
    a rig keyed via DTR, with DTR stuck high, was being reported LOW off the one
    pin that never keyed it."""
    _wire(monkeypatch, _FakePort(stuck=termios.TIOCM_DTR))
    said = []
    assert drop_rts(FAKE_DEV, said.append) is False
    assert any("NOT CONFIRMED LOW" in m for m in said), said


def test_an_adapter_that_dies_under_tcsetattr_is_reported_not_raised(monkeypatch):
    """`termios.error` subclasses Exception, not OSError, so a bare `tcsetattr`
    escaped both `open()`'s re-raise and the `except PttError` in every caller
    -- `drop_rts` raised on the one path whose whole contract is that it never
    does. The neighbouring `tcgetattr` was already wrapped; this is its twin."""
    p = _wire(monkeypatch, _FakePort())

    def refused(fd, when, attrs):
        raise termios.error(5, "Input/output error")
    monkeypatch.setattr(termios, "tcsetattr", refused)
    said = []
    assert drop_rts(FAKE_DEV, said.append) is False
    assert any("keying line" in m for m in said), said
    assert p.closed, "the descriptor must be handed back when configure fails"


def test_a_close_that_fails_during_a_failed_open_is_still_a_ptt_error(monkeypatch):
    """The cleanup close in `open()` runs only when an ioctl has already failed
    -- an adapter dying mid-open -- which is precisely the moment a close
    plausibly raises OSError too. Bare, that OSError replaced the PttError and
    sailed past `except PttError` in every caller of the never-raises
    contract, `drop_rts` first among them. `release`'s close was already
    wrapped; this is its twin."""
    _wire(monkeypatch, _FakePort())

    def refused(fd, when, attrs):
        raise termios.error(5, "Input/output error")
    monkeypatch.setattr(termios, "tcsetattr", refused)
    closed: list[int] = []

    def dying_close(fd):
        closed.append(fd)
        raise OSError(6, "Device not configured")
    monkeypatch.setattr(os, "close", dying_close)

    with pytest.raises(PttError):
        RtsPtt(FAKE_DEV).open()
    assert closed, "the failed open never tried to hand the descriptor back"

    closed.clear()
    said: list[str] = []
    assert drop_rts(FAKE_DEV, said.append) is False
    assert closed and any("keying line" in m for m in said), said


def test_an_adapter_that_vanishes_after_open_is_reported_not_raised(monkeypatch):
    """The panic path's own panic: the open-time ioctls are the last ones the
    adapter answers, and the deassert inside `release` is refused. `drop_rts`
    promises never to raise -- an exception here replaces the caller's own
    account of the failure that got it called."""
    p = _wire(monkeypatch, _VanishedPort())
    said = []
    assert drop_rts(FAKE_DEV, said.append) is False
    assert any("NOT CONFIRMED LOW" in m for m in said), said
    assert p.closed, "the descriptor must be handed back even when the deassert fails"


def test_it_reports_rather_than_raises_when_it_cannot_get_at_the_line():
    """It runs after everything else has failed; raising would lose the caller's own
    account of why."""
    said = []
    assert drop_rts(None, said.append) is False
    assert any("no PTT device known" in m for m in said), said

    said.clear()
    assert drop_rts("/dev/cu.no-such-adapter", said.append) is False
    assert any("/dev/cu.no-such-adapter" in m for m in said), said


@pytest.mark.skipif(not STATION_PTT,
                    reason="set PTT_PORT to the station's PTT adapter to run "
                           "against the real line")
def test_the_stations_own_adapter_goes_low():
    """No fake in the path. With PTT_PORT set this is the real thing, and it
    fails rather than skips when it does not work -- a port that is named but
    absent is one of the ways it does not work."""
    said = []
    assert drop_rts(STATION_PTT, said.append) is True, said


# -- keying the line instead of asking a daemon to ---------------------------------
#
# The emergency path above worked four times on 2026-08-09, and each time it was
# cleaning up after the same failure: rigctld stopped answering on the FIRST
# key-down of the session, every session, with the antenna radiating. One burst is
# not a session, so the transmit path stops going through the daemon at all.


def _rig(log, **kw):
    import vara_rig_bridge
    return vara_rig_bridge.Rig("127.0.0.1:1", armed=True, log=log,
                               ptt_device=FAKE_DEV, **kw)


@requires_bridge
def test_the_transmit_path_never_touches_the_daemon(port, monkeypatch):
    """`Rig` is pointed at a port nothing listens on and told to key the line, so
    every `_cmd` would raise. An implementation that still asked rigctld to key or
    unkey fails here rather than on the air."""
    import vara_rig_bridge

    def no_daemon(*a, **k):
        raise AssertionError("the transmit path reached for rigctld")

    monkeypatch.setattr(vara_rig_bridge.Rig, "_cmd", no_daemon)
    said = []
    rig = _rig(said.append, line_ptt=True)
    assert rig.key(True, "burst") is True
    assert port.bits & termios.TIOCM_RTS, "the rig never keyed the line"
    assert rig.keyed is True
    assert rig.key(False) is True
    assert not (port.bits & termios.TIOCM_RTS)
    assert rig.keyed is False
    assert not rig.retired, "a clean unkey must not retire the rig"
    assert any("PTT ON -> line" in m for m in said), said


@requires_bridge
def test_arming_the_line_leaves_it_low(port):
    """Opening a serial port asserts RTS, which here IS key-down. Holding the
    descriptor for a session is only safe because that assertion is undone on the
    way in -- before any caller can key, and before there is audio to modulate it."""
    said = []
    _rig(said.append, line_ptt=True)
    assert port.calls, "the line was never touched at arm time"
    assert port.calls[0] == termios.TIOCMBIC, "the line must be cleared first"
    assert not (port.bits & termios.TIOCM_RTS)


@requires_bridge
def test_a_key_that_does_not_reach_the_line_is_not_reported_as_transmitting(
        monkeypatch):
    """Reporting a key that did not happen is worse than failing it: the modem plays
    a whole burst into a receiver and hears nothing back."""
    _wire(monkeypatch, _FakePort(dead=termios.TIOCM_RTS))
    said = []
    rig = _rig(said.append, line_ptt=True)
    assert rig.key(True, "burst") is False
    assert rig.keyed is False
    assert any("NOT CONFIRMED" in m for m in said), said
    assert not rig.retired, "a line confirmed back low is still a line to trust"


@requires_bridge
def test_an_unconfirmed_key_up_still_lowers_the_line(monkeypatch):
    """TIOCMBIS accepted, TIOCMGET broken: the pin may be high while `keyed`
    stays False -- which blinds the watchdog, the one guard sized for a line
    left up. The only safe exit is the drop-to-be-sure the daemon branch
    already takes."""
    p = _wire(monkeypatch, _BlindPort())
    said = []
    rig = _rig(said.append, line_ptt=True)
    assert rig.key(True, "burst") is False
    assert not (p.bits & termios.TIOCM_RTS), (
        "the ioctl may have raised the pin; an unconfirmed key-up must deassert")
    assert rig.retired, "a line that cannot be read cannot be trusted to key again"


@requires_bridge
def test_a_line_that_will_not_unkey_still_raises_the_alarm(monkeypatch):
    """The one case that must keep the loud message: the ioctl path itself has
    failed, so nothing is left to try and the operator does need to go to the rig."""
    _wire(monkeypatch, _FakePort(stuck=termios.TIOCM_RTS))
    said = []
    rig = _rig(said.append, line_ptt=True)
    assert rig.key(False) is False
    joined = " ".join(said)
    assert "MAY BE STUCK" in joined and "manually" in joined, said
    assert rig.retired, "a transmitter that will not go down must not be keyed again"


@requires_bridge
def test_shutdown_hands_the_line_back(port):
    said = []
    rig = _rig(said.append, line_ptt=True)
    rig.key(True, "burst")
    rig.shutdown()
    assert not (port.bits & termios.TIOCM_RTS), "shutdown left the transmitter up"
    assert port.closed, "shutdown kept the port against the next session"


# -- and the daemon path, for the tools that still use it --------------------------


@requires_bridge
def test_a_line_confirmed_low_is_not_reported_as_maybe_stuck(port):
    """The first version of this path cleared RTS, read the line LOW, and then said
    "TRANSMITTER MAY BE STUCK. Unkey it manually now." -- on air, twice, at the
    radio. Reading the line low is BETTER evidence than the reply rigctld failed to
    give: the ioctl went to the driver rather than over the link that just failed.

    The rig is still retired, because with the daemon keying it a daemon that has
    stopped answering ends the session whatever the line reads. What must not happen
    is sending the operator to a transmitter that is demonstrably down.
    """
    said = []
    rig = _rig(said.append)                     # daemon-keyed: no line keyer
    rig.keyed = True
    assert rig._unkey(0.2, 0.05) is True, said
    joined = " ".join(said)
    assert "CONFIRMED DOWN" in joined, said
    assert "MAY BE STUCK" not in joined and "manually" not in joined, said
    assert rig.retired, "a daemon that stopped answering must still end the session"
    assert rig.keyed is False


# -- arming without the last resort is refused --------------------------------------
#
# The fallback above only runs if it knows the line, and on 2026-08-10 it did not:
# the vara launch armed kestrel without --ptt-device, rigctld took the unkey and
# never answered, and drop_rts had nothing to pull down — the transmitter stayed
# keyed with no modulation until the operator was told to unkey by hand. The line
# ran LOW in under a second once it was named. So --arm now refuses to run without
# it, the same treatment --expect-model gets and for the same reason.


def _kc_main(monkeypatch, capsys, *extra):
    kc = pytest.importorskip("kestrel_connect")
    monkeypatch.setattr(sys, "argv",
                        ["kestrel_connect.py", "--gateway", "W1AW",
                         "--mycall", "W9SSJ", "--arm", "--expect-model", "FT-891",
                         *extra])
    rc = kc.main()
    return rc, capsys.readouterr().out


def test_arming_without_the_keying_line_is_refused(monkeypatch, capsys):
    """What reaches the operator is the gate's own sentence, whole: a tool that
    imports `arming_refusal` and then says something else of its own is the drift
    this is here to prevent."""
    rc, out = _kc_main(monkeypatch, capsys)
    assert rc == 2
    assert arming_refusal(None) in out, out


def test_arming_on_a_keying_line_that_is_not_there_is_refused(monkeypatch, capsys):
    """The placeholder default is the other half of the same defect: a path that
    exists on no machine armed a session that could never unkey itself."""
    rc, out = _kc_main(monkeypatch, capsys,
                       "--ptt-device", "/dev/cu.usbserial-XXXXB1")
    assert rc == 2
    assert arming_refusal("/dev/cu.usbserial-XXXXB1") in out, out


def test_arming_with_a_real_line_named_gets_past_the_gate(monkeypatch, capsys):
    """A refusal gate that also refuses the armed-correctly case would just be
    the transmitter switched off. The next hardware-free check in main is the
    mail-file load, so reaching MAIL REFUSED proves the gate opened."""
    rc, out = _kc_main(monkeypatch, capsys, "--ptt-device", FAKE_DEV,
                       "--mail-send", "/no/such/mail.b2f")
    assert rc == 2
    assert "MAIL REFUSED" in out and "--ptt-device" not in out, out


def _bridge_main(monkeypatch, tmp_path, *extra):
    vrb = pytest.importorskip("vara_rig_bridge")
    monkeypatch.setattr(sys, "argv",
                        ["vara_rig_bridge.py", "--mycall", "W9SSJ",
                         "--record", str(tmp_path / "cap"),
                         "--rigctld", "127.0.0.1:1",
                         "--expect-model", "FT-891", "--arm", *extra])
    return vrb.main()


@requires_bridge
def test_bridge_arming_without_the_keying_line_is_refused(monkeypatch, capsys, tmp_path):
    """The bridge is the tool that OWNS `Rig`, and it was the fourth armed tool
    with the hole the other three were closed for: run standalone with --arm,
    its last-resort unkey had nothing to pull."""
    rc = _bridge_main(monkeypatch, tmp_path)
    assert rc == 2
    assert "--ptt-device" in capsys.readouterr().out


@requires_bridge
def test_bridge_arming_on_a_keying_line_that_is_not_there_is_refused(
        monkeypatch, capsys, tmp_path):
    rc = _bridge_main(monkeypatch, tmp_path,
                      "--ptt-device", "/dev/cu.usbserial-XXXXB1")
    assert rc == 2
    out = capsys.readouterr().out
    assert "REFUSING TO ARM" in out and "/dev/cu.usbserial-XXXXB1" in out, out


@requires_bridge
def test_bridge_arming_through_the_daemon_is_refused(monkeypatch, capsys, tmp_path):
    """2026-08-13, eight of eight: rigctld answered every command until the
    first key-down, then answered nothing — socket healthy, rig side gone. An
    armed bridge without --line-ptt transmits one over and then cannot unkey
    through the thing that keyed it, so --arm refuses to run that shape."""
    rc = _bridge_main(monkeypatch, tmp_path, "--ptt-device", FAKE_DEV)
    assert rc == 2
    out = capsys.readouterr().out
    assert "--line-ptt" in out and "may not hold the key" in out, out


@requires_bridge
def test_bridge_arming_with_the_line_gets_past_the_gate(monkeypatch, tmp_path):
    """Past the gate, the next thing main touches is the keying line itself --
    /dev/null stats as a character device but drives no modem lines, so
    `LineKeyer.arm` refuses it as a `PttError`, which proves every gate before
    it opened."""
    with pytest.raises(PttError):
        _bridge_main(monkeypatch, tmp_path, "--ptt-device", FAKE_DEV, "--line-ptt")


# -- the gate itself lives in one place ---------------------------------------------


def test_the_arm_gate_is_one_function():
    refusal = arming_refusal(None)
    assert refusal and "--arm requires --ptt-device" in refusal, refusal
    refusal = arming_refusal("/dev/cu.no-such-adapter")
    assert refusal and refusal.startswith("REFUSING TO ARM"), refusal
    assert "/dev/cu.no-such-adapter" in refusal
    assert arming_refusal(FAKE_DEV) is None


@requires_bridge
def test_every_transmit_tool_speaks_the_one_gate():
    """Load-bearing safety text, kept in `core.ptt.arming_refusal`.

    This used to let a tool keep its own copy so long as the copy still contained
    the right words, and a copy that contains the right words is still a copy:
    `ptt_tail_check` sat there saying the last-resort unkey "would have nothing to
    pull" where the gate says "no line to pull down", and the pin passed. So the
    pin is on the call — every armed tool asks the one function — and on the
    absence of a second spelling anywhere in tools/, which is what a copy is.
    """
    for name in ("vara_rig_bridge.py", "onair_session.py", "kestrel_connect.py",
                 "ptt_tail_check.py"):
        text = (_TOOLS / name).read_text()
        assert "arming_refusal" in text, f"{name} does not call the shared arm gate"
    # The opening of each refusal, which any copy of it carries however its tail
    # has drifted. `REFUSING TO ARM` itself is not one of these: the model check
    # beside this gate opens on the same words and is entitled to.
    openings = ("--arm requires --ptt-device", "last-resort unkey would have")
    for tool in sorted(_TOOLS.glob("*.py")):
        text = tool.read_text()
        for opening in openings:
            assert opening not in text, (
                f"{tool.name} spells the arm gate's refusal itself: {opening!r} "
                "belongs to core.ptt.arming_refusal and nowhere else")


# Every launcher in tools/, not just onair.sh: a keying rule pinned against one
# of them is a rule the others go on breaking. onair.sh moved its armed launches
# onto the line, and campaign.sh went on starting rigctld with the keying line
# and arming kestrel without --line-ptt — unattended — for another day.
def _launchers() -> list[Path]:
    return sorted(_TOOLS.glob("*.sh"))


def _armed_lines(text: str) -> list[str]:
    """Armed launches, continuations joined and comments dropped."""
    return [ln for ln in re.sub(r"\\\s*\n\s*", " ", text).splitlines()
            if "--arm" in ln and not ln.lstrip().startswith("#")]


@requires_launcher
def test_every_armed_launch_in_the_launchers_names_the_keying_line():
    """The launchers, checked as text: every command that passes --arm must pass
    --ptt-device in the same breath. This is the exact gap the vara verb had
    while vara-mail beside it did not."""
    armed = {s.name: _armed_lines(s.read_text()) for s in _launchers()}
    assert armed.get("onair.sh"), "no armed launch in onair.sh — the launcher has changed shape"
    for name, lines in armed.items():
        for ln in lines:
            assert "--ptt-device" in ln, f"{name}: an armed launch with no keying line: {ln}"


@requires_launcher
def test_no_launchers_daemon_ever_holds_the_key():
    """onair.sh's vara verb was the last to key through the daemon by hand, and
    its 2026-08-13 wire records ended the argument: rigctld answered every
    command until the first key-down, then answered nothing — the socket
    connecting in 1 ms the whole time — while the line read LOW in microseconds,
    eight of eight. So no launcher starts rigctld holding the keying line, and
    every armed launch drives that line itself, the shape vara-mail proved.

    Swept over all of them at once. campaign.sh kept the daemon-keyed shape
    through two rounds of this fix landing next door, and unattended it is the
    launcher with nobody there to hear the carrier."""
    for script in _launchers():
        text = script.read_text()
        assert "rigctld_start rts" not in text, (
            f"{script.name} starts the daemon holding the keying line again")
        assert "ptt_type=RTS" not in text and "ptt_pathname" not in text, (
            f"{script.name} configures rigctld to key again")
        for ln in _armed_lines(text):
            assert "--line-ptt" in ln, (
                f"{script.name}: an armed launch that keys through the daemon: {ln}")


#: The launcher's own sense, lifted out of the heredoc it lives in and pointed at
#: the station codec by name. It is looked for in the file that sources as well as
#: the one that launches, so consolidating the two senses into a shared file is a
#: move this pin follows rather than one it blocks -- an earlier version of it
#: asserted the string `is_busy` appeared in campaign.sh, which made the guard
#: written to keep the unattended path listening the thing preventing it from
#: sharing the listener every other verb goes through.
def _sense_source(name: str, repo: str = ".", dial: int = 7101500) -> str:
    for text in _launcher_texts(name):
        if "<<PY" not in text:
            continue
        body = text.split("<<PY", 1)[1].split("\n", 1)[1]
        # The three the shell interpolates. `$REPO` and `$dial` name where a
        # refused window is kept and what to call it, and neither is knowable to
        # the heredoc on its own -- see `keep` in tools/lib/rig.sh.
        return (body.split("\nPY\n", 1)[0].replace("$AUDIO", _CODEC)
                .replace("$REPO", repo).replace("$dial", str(dial)))
    raise AssertionError(f"no channel sense in {name} or anything it sources")


def _launcher_texts(name: str) -> list[str]:
    text = (_TOOLS / name).read_text()
    sourced = [(_TOOLS / m).read_text()
               for m in re.findall(r'source\s+"\$\(dirname[^)]*\)/(\S+)"', text)
               if (_TOOLS / m).exists()]
    return [text, *sourced]


#: What the station calls its codec, and what the stub sound card answers to.
_CODEC = "USB Audio Device"
_STUB = _TOOLS / "tests" / "stub_sounddevice.py"


def _sense(tmp_path, name, channel, *, deaf=False):
    """Run a launcher's sense against a scripted channel. Returns the exit
    status and the windows it actually listened in, `5c` a clear five seconds."""
    (tmp_path / "sounddevice.py").write_text(_STUB.read_text())
    src = tmp_path / "sense.py"
    src.write_text(_sense_source(name, repo=str(tmp_path)))
    log = tmp_path / "windows"
    log.unlink(missing_ok=True)                 # the stub appends; one run per file
    env = corpora.child_env(CHANNEL=channel, WINDOW_LOG=str(log))
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env["PYTHONPATH"]])
    if deaf:
        env["DEAF"] = "1"
    r = subprocess.run([str(_VENV_PY), str(src), "pactor", ""], env=env,
                       capture_output=True, text=True)
    windows = log.read_text().split() if log.exists() else []
    return r.returncode, "".join(windows), r.stdout + r.stderr


@requires_launcher
@pytest.mark.skipif(not _STUB.exists(), reason=f"{_STUB} is not present")
def test_the_unattended_sense_refuses_a_busy_channel(tmp_path):
    """The unattended sense, run against a scripted channel through the stub
    sound card `tools/tests/test_onair_gate.sh` drives the attended gate with.

    STRICTER THAN THE ATTENDED GATE, AND DELIBERATELY. With no wait budget that
    gate is one window; this takes two and refuses if either of them is
    occupied, because a real over has gaps in it and the operator who would hear
    one is by definition not here. Both halves are measured below: a channel
    that goes busy only in the second window is still refused, and the second
    window is not taken at all once the first has refused.

    Both are `core.busy.WINDOW_S` long. They were 5 s until the thresholds were
    measured at that length: the verified-clear 6950 kHz control reads busy on
    one 5 s window in six, which unattended is an attempt skipped for nothing."""
    want = f"{busy.WINDOW_S:.0f}"
    rc, windows, out = _sense(tmp_path, "campaign.sh", "cc")
    assert rc == 0, f"a clear channel was refused: {out}"
    assert windows == f"{want}c{want}c", (
        f"the unattended sense took '{windows}', not two {want} s windows")

    rc, windows, out = _sense(tmp_path, "campaign.sh", "b")
    assert rc == 1, f"an occupied channel was not refused: {out}"
    assert windows == f"{want}b", f"it went on listening after refusing: '{windows}'"

    rc, windows, out = _sense(tmp_path, "campaign.sh", "cb")
    assert rc == 1, (
        "a channel taken again in the second window passed the unattended "
        f"sense: {out}")

    rc, _, out = _sense(tmp_path, "campaign.sh", "c", deaf=True)
    assert rc == 2, (
        "a station that cannot hear the channel did not come back as 2 -- and "
        f"1 is busy to the caller, which would skip rather than stop: {out}")


@requires_launcher
def test_the_unattended_pactor_attempt_listens_before_it_calls():
    """`shrike.onair` has no occupancy sense of its own — `onair.sh pactor`
    senses the channel outside the modem for exactly that reason — and
    campaign.sh calls the same modem unattended. Until now every PACTOR attempt
    in a campaign transmitted without listening, on a shared band, with nobody
    watching, while the file's own header claimed the occupancy refusal stood
    on every attempt.

    What the sense decides is measured above; this is the ordering, which no
    stub can show: the sense stands between the attempt and the modem rather
    than beside it.

    The operator's ears outrank the sense, and --force is theirs to give. An
    unattended pass has no ears, so it is held to more caution, not less: no
    force path at all, a busy channel skips the attempt, and a receiver that
    cannot hear the channel ends the run — every later attempt would be
    exactly as blind."""
    texts = _launcher_texts("campaign.sh")
    text = next((t for t in texts if "pactor_try()" in t), "")
    assert text, "no unattended PACTOR attempt in campaign.sh or what it sources"
    body = text.split("pactor_try()", 1)[1].split("\n}", 1)[0]
    # Where the invocation RUNS, not where it is spelled: the attempt builds the
    # array first so a dry run can print the very words it would have keyed with.
    launch = max(body.rfind('"${cmd[@]}"'), body.rfind("hfmodem.shrike.onair"))
    assert launch > 0, "the unattended PACTOR attempt no longer launches shrike"
    # The sense is reached through `gate_unattended` since campaign.sh and the
    # autopilot started making one attempt between them.
    sensed = [i for i in (body.find("gate_unattended"), body.find("sense_channel"))
              if 0 <= i < launch]
    assert sensed, (
        "the PACTOR attempt reaches shrike without sensing the channel first")
    for ln in "\n".join(texts).splitlines():
        code = ln.split("#", 1)[0]
        if code.lstrip().startswith(("printf", "echo")):
            continue                    # saying the word cannot arm anything
        assert "--force" not in code, (
            f"a campaign passes --force with nobody there to have decided it: {ln}")
    deaf = [ln for ln in text.splitlines() if "cannot hear" in ln]
    assert deaf and any("die" in ln for ln in deaf), (
        "a deaf receiver does not end the run — unattended, a station that "
        "cannot hear the channel may not go on calling on it")


# -- and nothing that ships names a station of its own -------------------------------
#
# `CAT_PORT="${CAT_PORT:-/dev/cu.usbserial-XXXXB0}"` — a redaction placeholder, a
# device path with the adapter's serial Xed out, left reachable as a live default.
# It exists on no machine, so every real run needed both ports exported in front of
# it and nothing said so: on 2026-08-10 a station that was on and correctly cabled
# was told "CAT port … missing — is the rig on and cabled?" and went after its
# cable. The same shape had already let a 27-cycle session report 22.1 s of carrier
# into a radio it never touched.
#
# onair.sh was fixed that day and the three unattended launchers beside it kept the
# same line for another, each sitting next to a copy nobody read at the same time.
# So this scans all four and the station template together, and the port contract
# they share now lives in one file rather than four.

#: Everything these files may say about /dev. `/dev/null` is a redirection, not a
#: station; a glob is an offer to go and look. A path that is neither is a device
#: node someone hardcoded, which is the defect whatever serial number it carries.
_DEV = re.compile(r"/dev/[\w.*/-]+")
_NOT_A_STATION = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")

_PORTS_LIB = "lib/ports.sh"
_SIBLINGS = ("campaign.sh", "daywatch.sh", "nightwatch.sh")
_TEMPLATE = _TOOLS.parent / "examples" / "station.toml"

requires_template = pytest.mark.skipif(
    not _TEMPLATE.exists(), reason=f"{_TEMPLATE} is not present")


@requires_launcher
@pytest.mark.parametrize("name", ("onair.sh", *_SIBLINGS, _PORTS_LIB))
def test_no_launcher_names_a_device_node(name):
    named = [d for d in _DEV.findall((_TOOLS / name).read_text())
             if "*" not in d and d not in _NOT_A_STATION]
    assert not named, f"{name} names a device node of its own: {named}"


@requires_launcher
def test_the_port_contract_is_one_file_the_launchers_all_go_through():
    """Four scripts, one environment contract. Kept in four copies it was fixed in
    one of them, and the placeholder outlived the fix in the other three."""
    defaults = re.findall(r'^(CAT_PORT|PTT_PORT)="\$\{\1:?-([^}]*)\}"',
                          (_TOOLS / _PORTS_LIB).read_text(), re.M)
    assert len(defaults) == 2, f"{_PORTS_LIB} no longer takes both ports from the environment"
    for var, value in defaults:
        assert not value, f"{var} has a built-in default again: {value!r}"

    for name in ("onair.sh", *_SIBLINGS):
        text = (_TOOLS / name).read_text()
        assert _PORTS_LIB in text, f"{name} does not source the shared port contract"
        assert not re.search(r"^\s*(CAT_PORT|PTT_PORT)=", text, re.M), (
            f"{name} sets a serial port of its own instead of taking the shared one")
        assert re.search(r"\brequire_(cat|ports)\b", text), (
            f"{name} sources the contract and never asks it for the ports")


@requires_template
def test_the_station_template_ships_no_keying_line():
    """The template is edited field by field, and a plausible-looking port is the
    one field an operator has no reason to touch: it looks configured already. So
    the key is absent, and the station that results receives rather than pretending
    to key."""
    import tomllib

    named = [d for d in _DEV.findall(_TEMPLATE.read_text())
             if "*" not in d and d not in _NOT_A_STATION]
    assert not named, f"the station template names a device node: {named}"

    raw = tomllib.loads(_TEMPLATE.read_text())
    assert "port" not in raw["rig"].get("ptt", {}), (
        "examples/station.toml ships a keying line again")


@requires_template
def test_a_template_edited_but_for_the_keying_line_is_refused_by_name(tmp_path, capsys):
    """And the absence has to reach the operator as an answer rather than a
    traceback: `transmit = true` with no line named is the case this is for."""
    from hfmodem import cli

    path = tmp_path / "station.toml"
    path.write_text(_TEMPLATE.read_text().replace("transmit   = false",
                                                  "transmit   = true"))
    assert cli.main(["rig", str(path)]) == 2
    err = capsys.readouterr().err
    assert "[rig.ptt] port" in err and "unset" in err, err


def _launch(*argv, **env):
    """Run the launcher with the station's ports out of the environment. Every
    case below is refused before `require_hardware` stats anything, so nothing
    here opens a port, a codec or a transmitter."""
    e = {k: v for k, v in os.environ.items() if k not in ("CAT_PORT", "PTT_PORT")}
    return subprocess.run(["bash", str(_ONAIR), *argv], env={**e, **env},
                          capture_output=True, text=True, timeout=120)


@requires_launcher
def test_an_unconfigured_station_is_told_that_and_not_that_its_cabling_is_bad():
    """The operator meets this before anything is stated about the radio: which
    variables to set, and how to find the device names on this machine."""
    r = _launch("check")
    assert r.returncode == 1
    assert "CAT_PORT" in r.stderr and "PTT_PORT" in r.stderr, r.stderr
    assert re.search(r"ls /dev/\S+", r.stderr), r.stderr
    assert "cabled" not in r.stderr, (
        "an unconfigured station is being sent after its cabling", r.stderr)


@requires_launcher
@requires_launcher_python
def test_one_adapter_needs_one_variable():
    """CAT_PORT alone is enough on the two-interface adapter this station keys
    through: PTT is derived, and the run gets as far as the hardware check, which
    is the first thing entitled to talk about cables — the port it names is one
    the operator named first."""
    r = _launch("check", CAT_PORT="/dev/cu.usbserial-FAKE0B0")
    assert r.returncode == 1
    assert "CAT port /dev/cu.usbserial-FAKE0B0 missing" in r.stderr, r.stderr
    assert "PTT_PORT" not in r.stderr, ("the keying line was not derived", r.stderr)


@requires_launcher
@requires_launcher_python
def test_a_keying_line_that_cannot_be_derived_is_asked_for_rather_than_guessed():
    """`derive_ptt_port`'s narrow rule, reaching the operator: a name it does not
    recognise is refused here rather than advanced into whatever adapter happens
    to answer to the next index."""
    r = _launch("check", CAT_PORT="/dev/cu.Bluetooth-Incoming-Port")
    assert r.returncode == 1
    assert "cannot derive a PTT port" in r.stderr, r.stderr
    assert "set PTT_PORT" in r.stderr, r.stderr


# -- reaping a modem that may have the transmitter up -------------------------
#
# Read as text and not by running it. `tools/lib/ports.sh` names `$REPO/.venv`
# outright, so the launcher does not start in a worktree at all (see
# `_NO_LAUNCHER_PY` above) -- and this is the one property of the cleanup path
# that has to hold in every tree, including the ones where it cannot be driven.


def _function_body(text: str, name: str) -> str:
    return text.split(f"\n{name}() {{", 1)[1].split("\n}", 1)[0]


@requires_launcher
def test_the_reaper_asks_before_it_insists():
    """SIGKILL is the one signal a modem cannot unkey through: the handler, the
    `finally` chain and the keying watchdog all die with it, and whatever the line
    was doing it goes on doing -- process death is not known to lower it on this
    hardware (`core.ptt`, and `docs/STATION.md` for the measurement nobody has
    made). An orphan is by definition the process most likely to be holding a
    transmitter up with nothing left watching it, so it gets what
    `hfhost.supervisor._terminate` gives a modem it recycles: SIGTERM, a bounded
    moment to unkey in, and only then the kill.

    This opened with SIGKILL and nothing else, which is `kill_and_respawn`'s own
    closed bug living on one directory over.
    """
    body = _function_body(_ONAIR.read_text(), "reap_orphans")
    assert "SIGTERM" in body and "-TERM" in body, (
        "the reaper no longer asks the orphan to put its own key down first")
    assert body.index("-TERM") < body.index("-9"), (
        "the reaper kills before it asks, which is a modem that cannot unkey")
    assert "REAP_GRACE_S" in body, (
        "no bounded wait between the ask and the kill -- either it does not wait, "
        "or it waits without a bound, and one of those hangs the cleanup verb")


@requires_launcher
def test_the_cleanup_verb_reads_the_line_back_after_it_kills():
    """`down` ends by proving the transmitter is down off the receiver; the keying
    line itself has to be proved too, and after the reaping rather than before it.
    Both `reap_orphans` and `free_port` end in SIGKILL, and a signal is not
    evidence -- `prove_down` drops both output lines on the hardware and reads them
    back, which is the only statement in this file about the line rather than about
    the room.

    Before `prove_unkeyed` and not after: that one exits 1 on a quiet band, and
    under `set -e` it would take the line verdict down with it.
    """
    text = _ONAIR.read_text()
    down = text.split("\ndown)", 1)[1]
    assert "prove_down" in down, (
        "onair.sh down kills three ways and never reads the keying line back")
    assert down.index("reap_orphans") < down.index("prove_down"), (
        "the line is proved before the reaping that could raise it")
    assert down.index("prove_down") < down.index("prove_unkeyed"), (
        "a quiet band would exit 1 before the line verdict was printed")
