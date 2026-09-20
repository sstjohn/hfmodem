# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Host-side TCP server for VARA's documented protocol -- the modem side.

Where a client library is the *application* side (what VarAC / Pat / Winlink
Express do when they talk to a VARA modem), this is the *modem* side:
it opens the two TCP server sockets a real VARA process opens (command 8300,
data 8301), speaks the exact same ``\\r``-terminated ASCII protocol back, and
delegates the actual link behaviour to a pluggable :class:`ModemCore`
(``modem_core.py``). With the bundled :class:`LoopbackModem` a real client can
connect, establish a session, transfer data (echoed back) and disconnect --
with no radio present.

Because this server and any conforming client derive from the same documented
protocol, wiring a client to this server is a loopback that exercises both
against each other.

Scope / seams:
  * Transport + protocol syntax + config state live here.
  * Link/waveform behaviour lives behind :class:`ModemCore`.
  * Anything the 2022 EA5HVK doc does not pin down is flagged ``# UNVERIFIED``,
    rather than guessed at.
"""

from __future__ import annotations

import socket
import threading
import time

from .modem_core import LoopbackModem, ModemCore, ModemObserver
from .protocol import (
    BANDWIDTHS,
    COMPRESSION_MODES,
    CR,
    DEFAULT_CMD_PORT,
    DEFAULT_DATA_PORT,
    OK,
    VERSION_STRING,
    WRONG,
)


class SessionState:
    DISCONNECTED = "disconnected"
    LISTENING = "listening"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"


class HostSession(ModemObserver):
    """One application attached to the modem: its cmd socket + data socket.

    Implements :class:`ModemObserver`; every modem callback is turned into the
    documented wire message and written back to the application.
    """

    def __init__(self, cmd_conn: socket.socket, data_conn: socket.socket,
                 modem: ModemCore, *, iamalive_interval: float = 60.0,
                 log=None) -> None:
        self._cmd = cmd_conn
        self._data = data_conn
        self._modem = modem
        self._iamalive_interval = iamalive_interval
        self._log = log or (lambda *a: None)

        self._send_lock = threading.Lock()
        self._closed = False

        # configuration state (set by host commands)
        self.mycall: list[str] = []
        self.bandwidth = "500"
        self.compression = "TEXT"
        self.listening = False
        self.chat = False          # CHAT ON gates SN reporting (spec §7.3)
        self.state = SessionState.DISCONNECTED

        self._threads: list[threading.Thread] = []

    # -- run/teardown ------------------------------------------------------

    def run(self) -> None:
        self._modem.start(self)
        self._threads = [
            threading.Thread(target=self._cmd_reader, name="cmd-rx", daemon=True),
            threading.Thread(target=self._data_reader, name="data-rx", daemon=True),
            threading.Thread(target=self._heartbeat, name="iamalive", daemon=True),
        ]
        for th in self._threads:
            th.start()
        # Block until the command reader ends (client hung up / socket error).
        self._threads[0].join()
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._modem.abort()
        except Exception:
            pass
        try:
            self._modem.stop()
        except Exception:
            pass
        for s in (self._data, self._cmd):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

    # -- outbound helpers --------------------------------------------------

    def _send_cmd(self, msg: str) -> None:
        """Write one ``\\r``-terminated message to the command socket."""
        if self._closed:
            return
        with self._send_lock:
            try:
                self._cmd.sendall(msg.encode("ascii", errors="replace") + CR)
                self._log("modem->app", msg)
            except OSError:
                pass

    def _send_data(self, blob: bytes) -> None:
        if self._closed:
            return
        try:
            self._data.sendall(blob)
        except OSError:
            pass

    # -- ModemObserver (modem -> host wire messages) -----------------------

    def modem_connected(self, src: str, dst: str, bw: str | None) -> None:
        self.state = SessionState.CONNECTED
        msg = f"CONNECTED {src} {dst}" + (f" {bw}" if bw else "")
        self._send_cmd(msg)

    def modem_disconnected(self) -> None:
        self.state = (SessionState.LISTENING if self.listening
                      else SessionState.DISCONNECTED)
        self._send_cmd("DISCONNECTED")

    def modem_ptt(self, on: bool) -> None:
        self._send_cmd("PTT ON" if on else "PTT OFF")

    def modem_buffer(self, nbytes: int) -> None:
        self._send_cmd(f"BUFFER {nbytes}")

    def modem_data_received(self, blob: bytes) -> None:
        self._send_data(blob)

    def modem_busy(self, on: bool) -> None:
        self._send_cmd("BUSY ON" if on else "BUSY OFF")

    def modem_pending(self, cancel: bool = False) -> None:
        self._send_cmd("CANCELPENDING" if cancel else "PENDING")

    def modem_registered(self, text: str = "LINK REGISTERED") -> None:
        self._send_cmd(text)

    def modem_bitrate(self, level: int, bps: int, tx: bool) -> None:
        self._send_cmd(f"BITRATE ({level}) {bps} bps {'TX' if tx else 'RX'}")

    def modem_snr(self, sn: int) -> None:
        # SN is emitted only while chat mode is on. Winlink/Pat never send
        # CHAT ON, so this is invisible to them (spec §7.3); the gating lives
        # here, in one place, so a modem can call modem_snr on every decode.
        if self.chat:
            self._send_cmd(f"SN {sn}")

    # -- inbound: command channel -----------------------------------------

    def _cmd_reader(self) -> None:
        buf = b""
        try:
            while not self._closed:
                chunk = self._cmd.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while CR in buf:
                    line, buf = buf.split(CR, 1)
                    if line:
                        self._dispatch(line.decode("ascii", errors="replace"))
        except OSError:
            pass

    def _dispatch(self, cmd: str) -> None:
        self._log("app->modem", cmd)
        cmd = cmd.strip()
        if not cmd:
            return
        verb = cmd.split()[0].upper()
        up = cmd.upper()

        # -- configuration -------------------------------------------------
        if verb == "MYCALL":
            calls = cmd.split()[1:]
            if not calls or len(calls) > 5:
                self._send_cmd(WRONG)
                return
            self.mycall = calls
            self._modem.set_mycall(calls)
            self._send_cmd(OK)

        elif up == "LISTEN ON":
            self.listening = True
            self._modem.set_listen(True)
            if self.state == SessionState.DISCONNECTED:
                self.state = SessionState.LISTENING
            self._send_cmd(OK)

        elif up == "LISTEN OFF":
            self.listening = False
            self._modem.set_listen(False)
            if self.state == SessionState.LISTENING:
                self.state = SessionState.DISCONNECTED
            self._send_cmd(OK)

        elif verb == "BW" or (verb.startswith("BW") and verb[2:].isdigit()):
            # Accept both "BW500" (the form Pat sends) and "BW 500".
            arg = verb[2:] if len(verb) > 2 else (cmd.split()[1] if len(cmd.split()) > 1 else "")
            if arg in BANDWIDTHS:
                self.bandwidth = arg
                self._modem.set_bandwidth(arg)
                self._send_cmd(OK)
            else:
                self._send_cmd(WRONG)

        elif verb == "COMPRESSION":
            arg = up.split()[1] if len(up.split()) > 1 else ""
            if arg in COMPRESSION_MODES:
                self.compression = arg
                self._modem.set_compression(arg)
                self._send_cmd(OK)
            else:
                self._send_cmd(WRONG)

        # -- session verbs -------------------------------------------------
        elif verb == "CONNECT":
            parts = cmd.split()
            if len(parts) < 3:
                self._send_cmd(WRONG)
                return
            src, dst = parts[1], parts[2]
            vias = parts[4:] if len(parts) > 3 and parts[3].upper() == "VIA" else None
            self._send_cmd(OK)
            self.state = SessionState.CONNECTING
            self._modem.connect(src, dst, vias)

        elif up == "DISCONNECT":
            self._send_cmd(OK)
            self.state = SessionState.DISCONNECTING
            self._modem.disconnect()

        elif up == "ABORT":
            self._send_cmd(OK)
            self._modem.abort()

        # -- mode/timing toggles (accepted; no link effect in loopback) ----
        elif up in ("WINLINK SESSION", "P2P SESSION"):
            self._send_cmd(OK)

        elif up in ("CHAT ON", "CHAT OFF"):
            if up == "CHAT ON":
                # doc: CHAT ON implies LISTEN ON and enables SN reports.
                self.chat = True
                self.listening = True
                self._modem.set_listen(True)
                if self.state == SessionState.DISCONNECTED:
                    self.state = SessionState.LISTENING
            else:
                self.chat = False   # CHAT OFF stops SN reporting
            self._send_cmd(OK)

        elif verb == "CQFRAME":
            self._send_cmd(OK)

        # -- newer / Pat-Vara commands (UNVERIFIED vs 2022 doc) ------------
        elif up in ("PUBLIC ON", "PUBLIC OFF", "CWID ON", "CWID OFF"):
            self._send_cmd(OK)  # UNVERIFIED: not in EA5HVK 2022 list

        elif verb == "VERSION":
            self._send_cmd(VERSION_STRING)  # UNVERIFIED: Pat-Vara only

        else:
            self._send_cmd(WRONG)

    # -- inbound: data channel --------------------------------------------

    def _data_reader(self) -> None:
        try:
            while not self._closed:
                chunk = self._data.recv(65536)
                if not chunk:
                    break
                self._modem.transmit(chunk)
        except OSError:
            pass

    # -- heartbeat ---------------------------------------------------------

    def _heartbeat(self) -> None:
        while not self._closed:
            time.sleep(self._iamalive_interval)
            if not self._closed:
                self._send_cmd("IAMALIVE")


class VaraServer:
    """Listens on the command + data ports and serves attached applications.

    Serves one application session at a time (as a real VARA process does).
    When the application drops, it loops back to accept the next one. A fresh
    :class:`ModemCore` is created per session via ``modem_factory``.
    """

    def __init__(self, host: str = "127.0.0.1",
                 cmd_port: int = DEFAULT_CMD_PORT,
                 data_port: int = DEFAULT_DATA_PORT,
                 modem_factory=LoopbackModem,
                 iamalive_interval: float = 60.0,
                 log=None) -> None:
        self.host = host
        self.modem_factory = modem_factory
        self.iamalive_interval = iamalive_interval
        self._log = log or (lambda *a: None)

        self._cmd_srv = self._listen_socket(host, cmd_port)
        self._data_srv = self._listen_socket(host, data_port)
        self.cmd_port = self._cmd_srv.getsockname()[1]
        self.data_port = self._data_srv.getsockname()[1]

        self._running = False
        self._accept_thread: threading.Thread | None = None
        self._session: HostSession | None = None

    @staticmethod
    def _listen_socket(host: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        return s

    def serve_forever(self) -> None:
        """Blocking accept loop. Each iteration serves one full app session."""
        self._running = True
        while self._running:
            try:
                cmd_conn, _ = self._cmd_srv.accept()
            except OSError:
                break
            cmd_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                # The application opens the data socket right after the command
                # socket (Pat does). UNVERIFIED: real
                # VARA's exact accept ordering/pairing when multiple apps race.
                data_conn, _ = self._data_srv.accept()
            except OSError:
                cmd_conn.close()
                break
            data_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self._log("server", "application attached")
            session = HostSession(
                cmd_conn, data_conn, self.modem_factory(),
                iamalive_interval=self.iamalive_interval, log=self._log)
            self._session = session
            try:
                session.run()  # blocks until the app disconnects
            finally:
                self._session = None
                self._log("server", "application detached")

    def start_background(self) -> None:
        """Run :meth:`serve_forever` in a daemon thread (for tests/embedding)."""
        self._accept_thread = threading.Thread(
            target=self.serve_forever, name="vara-accept", daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._session:
            self._session.close()
        for s in (self._cmd_srv, self._data_srv):
            try:
                s.close()
            except OSError:
                pass
