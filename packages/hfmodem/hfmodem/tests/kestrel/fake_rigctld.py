# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A rigctld that answers like the FT-891 and misbehaves on demand, on a private port.

Two clients in this tree talk to rigctld and they talk to it differently:
:class:`vara_rig_bridge.Rig` opens a connection per command, and
:class:`onair_session.Cat` keeps one open for the whole session. This serves both —
every line on a connection is a command, for as long as the connection lives.

What it models is what the tree has actually been caught by:

``RPRT 0`` on success and ``RPRT <negative>`` on refusal, in band, without raising;
a ``t`` that reports the PTT line rather than the last thing written to it
(``ignore_unkeys`` acknowledges an unkey and leaves the transmitter up);
a filter ladder, so a passband asked for comes back as the nearest one the rig has
rather than the number requested; and a daemon that stops answering partway through
a session, which is how the last rehearsal ended.

Nothing here may touch 4533: there is a real transmitter on the end of it.
"""
from __future__ import annotations

import socket
import threading
import time

# Enough of `\dump_caps` to be found by label. `Rig.model` searches for the line
# rather than taking a fixed one, so the shape matters and the length does not.
_CAPS = """Caps dump for model: 1036
Model name:\t{model}
Mfg name:\tYaesu
Backend version:\t20240101.0
Rig type:\tTransceiver
PTT type:\tRig capable
"""

# The FT-891's PKTUSB filters. The widest is 2.4 kHz, which is why `Cat.set_mode`
# asks for a width and then only requires the answer to be wide enough.
FT891_PASSBANDS = (500, 1800, 2400)


class FakeRigctld(threading.Thread):
    """Serves loopback on an ephemeral port; ``port`` is where.

    ``delay`` stalls every reply, ``answer=False`` takes the command and says nothing
    at all (a daemon that has stopped talking to the radio still completes the
    connection and still swallows the bytes), ``ignore_unkeys`` acknowledges ``T 0``
    with ``RPRT 0`` while leaving the PTT line up, ``refuse`` names command letters
    that come back ``RPRT -1``, and ``deaf_after`` goes silent once that many commands
    have been taken.
    """

    def __init__(self, delay: float = 0.0, answer: bool = True, ignore_unkeys: int = 0,
                 model: str = "FT-891", freq: int = 7099500, mode: str = "PKTUSB",
                 passbands: tuple[int, ...] = FT891_PASSBANDS,
                 refuse: tuple[str, ...] = (), deaf_after: int | None = None):
        super().__init__(daemon=True)
        self.delay, self.answer, self.ignore_unkeys = delay, answer, ignore_unkeys
        self.model, self.freq, self.mode = model, freq, mode
        self.passbands = tuple(sorted(passbands))
        self.passband = self.passbands[-1]
        self.refuse = set(refuse)
        self.deaf_after = deaf_after
        self.commands: list[str] = []
        # Wall clock, matching the timestamps the fake audio device writes, so a
        # test can ask how long after the last sample of a burst the key came up.
        self.times: list[float] = []
        self.ptt = 0
        # Every transition of the PTT *line*, which is not the same as every ``T``
        # taken: an unkey this daemon acknowledges and does not apply leaves the
        # transmitter up, and how long it stayed up is the question.
        self.ptt_edges: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(32)
        self.port = self._sock.getsockname()[1]
        self._closed = False

    # -- the wire ---------------------------------------------------------------
    def run(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(60)
            buf = b""
            while not self._closed:
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode(errors="replace").strip()
                    if not cmd:
                        continue
                    with self._lock:
                        self.commands.append(cmd)
                        self.times.append(time.time())
                        n = len(self.commands)
                    if self.delay:
                        time.sleep(self.delay)
                    if not self.answer or (self.deaf_after is not None
                                           and n > self.deaf_after):
                        continue          # taken off the wire, never applied to a rig
                    try:
                        conn.sendall(self._reply(cmd))
                    except OSError:
                        return

    def _reply(self, cmd: str) -> bytes:
        head = cmd.split()[0] if cmd.split() else ""
        if head in self.refuse:
            return b"RPRT -1\n"
        if cmd.startswith(";"):
            return self._extended(cmd[1:].strip())
        if cmd.startswith("T "):
            with self._lock:
                was = self.ptt
                if cmd.split()[1] == "1":
                    self.ptt = 1
                elif self.ignore_unkeys > 0:
                    self.ignore_unkeys -= 1        # accepted; the rig does not drop
                else:
                    self.ptt = 0
                if self.ptt != was:
                    self.ptt_edges.append((time.time(), self.ptt))
            return b"RPRT 0\n"
        if cmd == "t":
            return f"{self.ptt}\n".encode()
        if cmd.startswith("F "):
            with self._lock:
                self.freq = int(float(cmd.split()[1]))
            return b"RPRT 0\n"
        if cmd == "f":
            return f"{self.freq}\n".encode()
        if cmd.startswith("M "):
            parts = cmd.split()
            with self._lock:
                self.mode = parts[1]
                self.passband = self._nearest(int(parts[2]) if len(parts) > 2 else 0)
            return b"RPRT 0\n"
        if cmd == "m":
            return f"{self.mode}\n{self.passband}\n".encode()
        if cmd.startswith("\\dump_caps"):
            return _CAPS.format(model=self.model).encode()
        return b"RPRT 0\n"

    def _extended(self, cmd: str) -> bytes:
        """rigctld's one-line form: the command echoed, then ``field: value`` pairs.

        `Rig.state` asks in this form because a bare ``m`` answers on two lines, and a
        reader taking one line per command has the stray one turn up as the next
        command's reply. It is the form the real daemon answers a leading separator
        with, and until it was served here nothing exercised that path at all.
        """
        with self._lock:
            fields = {"f": [("Frequency", self.freq)],
                      "m": [("Mode", self.mode), ("Passband", self.passband)]}.get(cmd)
        if fields is None:
            return self._reply(cmd)
        return (f"{cmd}:;" + "".join(f"{k}: {v};" for k, v in fields)
                + "RPRT 0\n").encode()

    def _nearest(self, asked: int) -> int:
        """The widest filter no wider than what was asked for — a rig picks off a
        ladder, so asking for 3 kHz on an FT-891 gets 2.4 back and that is correct."""
        below = [p for p in self.passbands if p <= asked]
        return below[-1] if below else self.passbands[0]

    # -- what the tests read ----------------------------------------------------
    def seen(self, cmd: str) -> int:
        with self._lock:
            return self.commands.count(cmd)

    def stamped(self, cmd: str) -> list[float]:
        """When each ``cmd`` arrived, in ``time.time()``."""
        with self._lock:
            return [t for c, t in zip(self.commands, self.times) if c == cmd]

    def dropped_after(self, t: float) -> float | None:
        """When the PTT line next went low after ``t``, or None if it never did."""
        with self._lock:
            return next((at for at, state in self.ptt_edges if at >= t and not state),
                        None)

    def raised_before(self, t: float) -> float | None:
        """When the PTT line last went high at or before ``t``, or None.

        The near edge of a keyed transmission, as ``dropped_after`` is the far one.
        Between it and the first sample there is a transmitter on the air carrying
        nothing, so how far apart they are is a question about the emission.
        """
        with self._lock:
            return next((at for at, state in reversed(self.ptt_edges)
                         if at <= t and state), None)

    def wait_for(self, cmd: str, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.seen(cmd):
                return True
            time.sleep(0.01)
        return False

    def close(self) -> None:
        self._closed = True
        self._sock.close()


def serving():
    """Body of a ``rigctld`` fixture: yields a factory, closes what it made.

    Used as ``yield from serving()`` rather than exported as a fixture, so each test
    module's fixture is declared where it is used and the linter can still see that
    the name is defined once.
    """
    made: list[FakeRigctld] = []

    def start(**kw) -> FakeRigctld:
        s = FakeRigctld(**kw)
        s.start()
        made.append(s)
        return s

    yield start
    for s in made:
        s.close()
