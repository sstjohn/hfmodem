# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A pseudo-terminal that answers Yaesu CAT, so PTT can be watched without a radio.

The transmit path's PTT is a line written to a long-lived `rigctl`'s stdin, and
that write returns in microseconds whatever the rig is doing. Every figure shrike
has ever printed about its own keying -- `keyed_s` above all -- is the interval
between two of those writes, so it reports the schedule that was intended and not
the keying that happened. An operator watching the rig on 2026-07-28 saw the two
diverge completely: the key held down across five bursts, then a run of rapid
toggles, while the log read 0.998 s keyed for 0.960 s of audio, every cycle.

This closes that gap on a bench. `rigctl` is given a pty instead of
/dev/cu.usbserial-*, with the SAME model number, baud and arguments the on-air
path uses, so the whole real stack runs: `ota.Rig`, the FT-891 backend, its
50 ms post_write_delay, its `TX;` read-back after every PTT change. What is
different is only what is on the far end of the wire -- this, rather than a
radio. Every `TX1;`/`TX0;` is timestamped as its terminator arrives, on
`time.monotonic()`, which is the clock the audio stream is on: those timestamps
are the keying, and they are what the assertions are made against.

Keying itself has since come OFF the CAT wire -- it is RTS on the rig's second
port, one ioctl in hamlib's frontend -- and a pty cannot carry a modem-control
line on Darwin, which would have left the shipping path with no bench at all.
`ptyrts.c` supplies the missing lines to that one fd and announces every
transition down the pty, so an RTS edge is timestamped here exactly as a `TX1;`
is, on the same clock, at the instant hamlib acts.

`answer_delay` is the one knob that matters. A rig is not obliged to answer `TX;`
promptly -- it has just been told to transmit -- and hamlib's validation of a PTT
change re-sends `TX1;` and waits again for as long as that takes (newcat.c,
`newcat_set_cmd_validate`: the `goto repeat` on an empty read sits ABOVE the
retry counter, so the loop has no bound). Set it and the stand-in reproduces a
rig that is slow while transmitting.
"""
from __future__ import annotations

import os
import pty
import select
import shutil
import subprocess
import termios
import threading
import time
from pathlib import Path

# PATH first, matching `ota.Rig`'s own resolution; the second entry is one
# station's private Hamlib build, kept so its bench keeps running off PATH.
RIGCTL = (shutil.which("rigctl")
          or os.path.expanduser("~/src/radio/hamlib/_install/bin/rigctl"))
FT891_MODEL, FT891_BAUD = 1036, 38400
_SHIM_SRC = Path(__file__).with_name("ptyrts.c")
_SHIM_LIB = Path(__file__).parent / "__pycache__" / "ptyrts.dylib"

# Enough of an FT-891's CAT state for `rig_open` and the poll thread to be
# satisfied. ID0135 is what the backend matches on (newcat.c, NC_RIGID_FT891);
# the rest only have to be well-formed, since nothing here is testing frequency
# or mode. Anything not listed is answered "?;", which is a real rig's reply to a
# command it does not implement.
REGISTERS = {
    "ID": "0135", "PS": "1", "AI": "0", "TX": "0", "KS": "020",
    "FA": "007100000", "FB": "007100000", "MD": "0C", "VS": "0", "PC": "050",
    "FT": "0", "ST": "0", "SY": "0", "NA": "00", "NB": "00", "RA": "00",
    "SH": "000", "SM": "0000", "SQ": "0000", "AG": "0100", "PA": "00",
    "EX": "0000", "MC": "001", "RG": "0100", "VX": "0", "BS": "00",
    "IF": "001007100000+000000C10000000",
}


class PtyRig:
    """The far end of the CAT wire: a pty, answering as an FT-891 would.

    `edges` is the measurement -- (monotonic, keyed) for every PTT change, taken
    when the command's terminator lands, which is the instant a rig acts on it.
    `commands` is everything else that crossed the wire, which is how a stall in
    the backend is told apart from a stall in the caller.

    `ptt_line` says which of them is the keying: "TX" for a CAT command, "RTS"
    for the modem-control line the interposer reports. An instance stood up as a
    rig's PTT port sees no CAT at all, and that emptiness is itself a result.
    """

    def __init__(self, *, answer_delay: float = 0.0, ptt_line: str = "TX"):
        self._controller, self._device = pty.openpty()
        # The line discipline echoes by default, and an echo would come back as
        # if the rig had sent it. rigctl turns echo off when it configures the
        # port; this covers the window before that.
        attrs = termios.tcgetattr(self._device)
        attrs[3] &= ~(termios.ECHO | termios.ECHOE | termios.ICANON | termios.ISIG)
        termios.tcsetattr(self._device, termios.TCSANOW, attrs)
        self.path = os.ttyname(self._device)
        self.answer_delay = answer_delay
        self.ptt_line = ptt_line
        self.regs = dict(REGISTERS)
        self.edges: list[tuple[float, bool]] = []
        self.commands: list[tuple[float, str]] = []
        self._keyed_at = 0.0
        self._run = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        buf = b""
        while self._run:
            # Never a bare read: the far end is a subprocess that may be wedged,
            # and this thread has to be able to notice `close()`.
            if not select.select([self._controller], [], [], 0.1)[0]:
                continue
            try:
                data = os.read(self._controller, 4096)
            except OSError:
                break
            if not data:
                break
            now = time.monotonic()
            buf += data
            while b";" in buf:
                cmd, buf = buf.split(b";", 1)
                self._handle(cmd.decode("latin1", "replace") + ";", now)
            # A pty's buffer is small and rigctl blocks on a full one. Anything
            # that arrives without a terminator is a truncated command, not ours
            # to keep.
            if len(buf) > 256:
                buf = b""

    def _handle(self, cmd: str, now: float) -> None:
        if cmd.startswith("!"):
            # A modem-control line moving, reported by the interposer (ptyrts.c)
            # because the kernel will not carry one on a pty. Kept out of
            # `commands`, which is the CAT wire: the point of keying on RTS is
            # that nothing crosses that wire at all.
            if cmd[1:4] == self.ptt_line:
                self.edges.append((now, cmd[4] == "1"))
            return
        self.commands.append((now, cmd))
        head, arg = cmd[:2], cmd[2:-1]
        if head == "TX" and arg:
            self.regs["TX"] = arg
            self.edges.append((now, arg == "1"))
            self._keyed_at = now
            return
        if arg:
            self.regs[head] = arg
            return
        if head == "TX" and self.answer_delay:
            # A rig with something better to do than answer a status query while
            # it is coming up on transmit.
            if now - self._keyed_at < self.answer_delay:
                return
        value = self.regs.get(head)
        self._write((head + value + ";") if value is not None else "?;")

    def _write(self, s: str) -> None:
        try:
            os.write(self._controller, s.encode())
        except OSError:
            pass

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=1.0)
        for fd in (self._controller, self._device):
            try:
                os.close(fd)
            except OSError:
                pass


def have_rigctl(path: str = RIGCTL) -> bool:
    return os.path.exists(path)


def rts_shim() -> str | None:
    """Build ptyrts.c and return the dylib to insert, or None if it cannot be.

    Without it there is no bench for RTS keying at all: Darwin answers TIOCMBIS
    on a pty with ENOTTY, so `rig_open` fails on a pty PTT port before a single
    command is read. With it, the whole shipping stack runs -- rigctl, the
    frontend's PTT dispatch, the ioctl -- against a pty that can hold a line.
    """
    if not _SHIM_SRC.exists():
        return None
    if not _SHIM_LIB.exists() or _SHIM_LIB.stat().st_mtime < _SHIM_SRC.stat().st_mtime:
        _SHIM_LIB.parent.mkdir(exist_ok=True)
        cc = subprocess.run(["clang", "-dynamiclib", "-O1", "-o", str(_SHIM_LIB),
                             str(_SHIM_SRC)], capture_output=True, text=True)
        if cc.returncode:
            print(f"  (ptyrts.c did not build: {cc.stderr.strip().splitlines()[-1:]})")
            return None
    return str(_SHIM_LIB)
