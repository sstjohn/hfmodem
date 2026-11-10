# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""rigctld client: CAT and PTT for the M3 two-site OTA milestone.

Speaks hamlib's rigctld network protocol — one text command per line, one
``RPRT n`` line back (n=0 success) — over a persistent socket, following the
pattern shrike uses in ``hfmodem.shrike.ota``.

M1 ships the transport only, and nothing wires it in: the modems in this
dialect *notify* PTT and never key hardware, so creance is the only thing that
could actuate a radio, and doing so without the rest of the M3 machinery would
key a transmitter with no watchdog. What M3 adds:

- PTT actuation driven from the modems' PTT notifications, on a dedicated fast
  path (cmd-reader thread -> Rig, event loop informed afterwards) so transcript
  writes never add keying jitter;
- a first-claim mutex across modems at a multi-modem site, plus a lockout
  watchdog that unkeys if the claim holder goes quiet;
- frequency/mode setup per scheduled cross-site beacon session.

Until then this class is exercised only by its own tests.
"""

from __future__ import annotations

import socket
import threading

from .config import RigConfig


class RigError(RuntimeError):
    pass


class Rig:
    """One rigctld connection. Thread-safe: every exchange holds the lock, so
    a reply can never be mistaken for another caller's."""

    def __init__(self, cfg: RigConfig | None = None, *, host: str | None = None,
                 port: int | None = None, timeout_s: float = 2.0) -> None:
        cfg = cfg or RigConfig()
        self.host = host if host is not None else cfg.host
        self.port = port if port is not None else cfg.port
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._buf = b""

    # -- transport ---------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port),
                                            timeout=self.timeout_s)
        except OSError as exc:
            raise RigError(f"rigctld {self.host}:{self.port}: {exc}") from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock, self._buf = sock, b""

    def close(self) -> None:
        with self._lock:
            sock, self._sock = self._sock, None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def __enter__(self) -> "Rig":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def command(self, text: str) -> str:
        """Send one rigctl command and return its reply line. RigError on a
        transport failure or a non-zero RPRT; the socket is dropped on error so
        the next call reconnects rather than reading a stale reply."""
        with self._lock:
            self._connect()
            try:
                self._sock.sendall(text.encode("ascii") + b"\n")
                reply = self._readline()
            except OSError as exc:
                self._drop()
                raise RigError(f"rigctld: {text!r}: {exc}") from exc
            if reply.startswith("RPRT "):
                code = reply[5:].strip()
                if code != "0":
                    raise RigError(f"rigctld: {text!r} -> RPRT {code}")
            return reply

    def _readline(self) -> str:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise OSError("rigctld closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", "replace").strip()

    def _drop(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # -- CAT ---------------------------------------------------------------

    def set_freq(self, hz: float) -> None:
        self.command(f"F {int(hz)}")

    def set_mode(self, mode: str, passband_hz: int = 0) -> None:
        """passband 0 means the rig's default for that mode."""
        self.command(f"M {mode.upper()} {int(passband_hz)}")

    def ptt(self, on: bool) -> None:
        self.command(f"T {1 if on else 0}")
