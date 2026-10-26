# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Host-side TCP server for VARA's documented protocol -- the modem side.

Where a client library is the *application* side (what Pat / Winlink
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
import re
import threading
from collections import deque
from hfmodem.core.socketpair import SocketPairAcceptor
from hfmodem.core.hostio import Connection, MAX_OUTPUT, MAX_PENDING

from .kiss import KissPort
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
                 log=None, kiss=None) -> None:
        self._cmd = Connection(cmd_conn, line=True) if isinstance(cmd_conn, socket.socket) else cmd_conn
        self._data = Connection(data_conn) if isinstance(data_conn, socket.socket) else data_conn
        self._modem = modem
        self._kiss = kiss
        self._iamalive_interval = iamalive_interval
        self._log = log or (lambda *a: None)

        self._send_lock = threading.Lock()
        self._operation_lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._command_depth = 0
        self._broadcast_tokens = deque(maxlen=256)
        self._draining_events = False
        self._events = deque()
        self._event_bytes = 0
        self._event_failure = False
        self._closed = False
        self._done = threading.Event()
        self._cq_reply_ready = threading.Event()
        self._cq_reply_ready.set()

        # configuration state (set by host commands)
        self.mycall: list[str] = []
        self.bandwidth = "500"
        self.compression = getattr(modem, 'default_compression', 'TEXT')
        self.session_mode = None
        self.listening = False
        self.chat = False
        self.listen_cq = False
        self.state = SessionState.DISCONNECTED
        self._connect_epoch = 0

        self._threads: list[threading.Thread] = []
        self.retirement_error = None

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
        for th in self._threads[1:]:
            th.join(timeout=1)

    def close(self) -> None:
        with self._operation_lock:
            self._close()

    def ensure_retired(self):
        """Verify a managed attachment can release its exclusive station lease."""
        for worker in self._threads:
            if worker is not threading.current_thread():
                worker.join(2.)
                if worker.is_alive():
                    raise RuntimeError('VARA host reader did not retire')
        for conn in (self._cmd, self._data):
            if isinstance(conn, Connection):
                conn.join_workers()
        if self.retirement_error is not None:
            raise RuntimeError('VARA adapter stop did not retire') from self.retirement_error

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._done.set()
        with self._event_lock:
            self._events.clear()
            self._event_bytes = 0
        try:
            self._modem.abort()
        except Exception:
            pass
        try:
            self._modem.stop()
        except Exception as exc:
            self.retirement_error = exc
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
                writer = self._cmd.send_bounded if isinstance(self._cmd, Connection) else self._cmd.sendall
                writer(msg.encode("ascii", errors="replace") + CR)
                self._log("modem->app", msg)
            except OSError:
                pass

    def _send_data(self, blob: bytes) -> None:
        if self._closed:
            return
        try:
            writer = self._data.send_bounded if isinstance(self._data, Connection) else self._data.sendall
            writer(blob)
        except OSError:
            pass

    def _event(self, kind, value):
        """Stage callbacks until command admission is serialized, without waiting
        for the host command's lock on an RF owner thread."""
        with self._event_lock:
            if self._closed or self._event_failure:
                return
            if self._command_depth or self._draining_events:
                size = len(value) + (kind == 'command')
                if len(self._events) >= MAX_PENDING or self._event_bytes + size > MAX_OUTPUT:
                    self._event_failure = True
                    return  # The command owner retires the failed attachment.
                self._events.append((kind, value))
                self._event_bytes += size
                return
        self._write_event(kind, value)

    def _write_event(self, kind, value):
        if kind == 'command':
            self._send_cmd(value)
        else:
            self._send_data(value)

    def _finish_command(self):
        with self._event_lock:
            self._command_depth -= 1
            if self._command_depth:
                return
            self._draining_events = True
        try:
            # Bound both storage and work if callbacks keep producing output
            # while the command owner drains it.
            for _ in range(MAX_PENDING):
                with self._event_lock:
                    if self._closed or self._event_failure:
                        break
                    if not self._events:
                        self._draining_events = False
                        return
                    kind, value = self._events.popleft()
                    self._event_bytes -= len(value) + (kind == 'command')
                self._write_event(kind, value)
            with self._event_lock:
                failed = self._event_failure or bool(self._events)
                self._event_failure = failed
                self._draining_events = False
            if failed:
                self.close()
        except BaseException:
            self.close()
            raise

    # -- ModemObserver (modem -> host wire messages) -----------------------

    def modem_connected(self, src: str, dst: str, bw: str | None) -> None:
        self.state = SessionState.CONNECTED
        msg = f"CONNECTED {src} {dst}" + (f" {bw}" if bw else "")
        self._event('command', msg)

    def modem_disconnected(self) -> None:
        self.state = (SessionState.LISTENING if self.listening
                      else SessionState.DISCONNECTED)
        self._event('command', "DISCONNECTED")

    def modem_ptt(self, on: bool) -> None:
        self._event('command', "PTT ON" if on else "PTT OFF")

    def modem_buffer(self, nbytes: int) -> None:
        # Stock reports its buffer only while a link is up: bytes written before
        # CONNECTED, and clears answered then, draw no BUFFER line (bench 2 4b
        # gate3-stock-ref), and the first after CONNECTED is the drain's BUFFER 0.
        if self.state in (SessionState.CONNECTED, SessionState.DISCONNECTING):
            self._event('command', f"BUFFER {nbytes}")

    def modem_data_received(self, blob: bytes) -> None:
        self._event('data', bytes(blob))

    def modem_warning(self, message: str) -> None:
        self._event('command', message)

    def modem_disconnect_refused(self, message: str) -> None:
        self.state = SessionState.CONNECTED
        self.modem_warning(message)

    def modem_busy(self, on: bool) -> None:
        self._event('command', "BUSY ON" if on else "BUSY OFF")

    def modem_pending(self, cancel: bool = False) -> None:
        self._event('command', "CANCELPENDING" if cancel else "PENDING")

    def modem_registered(self, text: str = "LINK REGISTERED") -> None:
        self._event('command', text)

    def modem_bitrate(self, level: int, bps: int, tx: bool) -> None:
        self._event('command', f"BITRATE ({level})  {bps} bps {'TX' if tx else 'RX'}")

    def modem_snr(self, sn: float | None) -> None:
        self._event('command', 'SN --' if sn is None else f"SN {sn:.1f}")

    def modem_cqframe(self, call: str, bw: str, sn: float | None) -> None:
        if self.listen_cq:
            self.modem_bitrate(1, 18, False)
            self.modem_snr(sn)
            self._event('command', f"CQFRAME {call} {bw}")

    def modem_forced_frame_refused(self) -> None:
        """One-shot extension failed after admission; never a successful beacon."""
        self._event('command', WRONG)

    def modem_beacon(self, result) -> None:
        """A heard KISS broadcast: SN on the command port as every chat-mode frame
        earns one, the twenty characters on the KISS port as stock hands them over."""
        if self.chat:
            self.modem_snr(result.sn)
        if self._kiss is not None:
            self._kiss.deliver(result.text.ljust(20))

    def modem_broadcast(self, result) -> None:
        """Decoded unlinked UI data remains on KISS, never the contact stream."""
        if self._kiss is not None:
            self._kiss.deliver_frame(result.command, result.encode())

    def modem_clean_tx_buffer(self, result: str) -> None:
        self._event('command', f"CLEANTXBUFFER {result}")

    # -- inbound: command channel -----------------------------------------

    def _cmd_reader(self) -> None:
        buf = b""
        try:
            while not self._closed:
                chunk = self._cmd.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 65536:
                    self.close()
                    return
                while CR in buf:
                    line, buf = buf.split(CR, 1)
                    if line:
                        self._dispatch(line.decode("ascii", errors="replace"))
                        while not self._cq_reply_ready.wait(.1) and not self._closed:
                            pass
        except OSError:
            pass

    def _dispatch(self, cmd: str) -> None:
        with self._operation_lock:
            if not self._closed:
                with self._event_lock:
                    self._command_depth += 1
                try:
                    try:
                        self._dispatch_live(cmd)
                    except (ValueError, NotImplementedError) as error:
                        # Keep stock wire grammar; the local log explains refusal.
                        verb = cmd.split()[0] if cmd.split() else '<empty>'
                        self._log('rejected', f'{verb}: {error}')
                        self._send_cmd(WRONG)
                finally:
                    self._finish_command()

    def _dispatch_live(self, cmd: str) -> None:
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
            self._modem.set_mycall(calls)
            self.mycall = calls
            self._send_cmd(OK)

        elif up in ("LISTEN ON", "LISTEN CQ", "LISTEN OFF"):
            previous = self.listening, self.listen_cq
            self.listening = up != "LISTEN OFF"
            self.listen_cq = up == "LISTEN CQ"
            try:
                # A live LISTEN command can disconnect synchronously. Its
                # callback must see the newly selected idle state.
                self._modem.set_listen(self.listening)
            except Exception:
                self.listening, self.listen_cq = previous
                raise
            if self.state in (SessionState.DISCONNECTED, SessionState.LISTENING):
                self.state = (SessionState.LISTENING if self.listening
                              else SessionState.DISCONNECTED)
            self._send_cmd(OK)

        elif verb == "BW" or (verb.startswith("BW") and verb[2:].isdigit()):
            # Accept both "BW500" (the form Pat sends) and "BW 500".
            arg = verb[2:] if len(verb) > 2 else (cmd.split()[1] if len(cmd.split()) > 1 else "")
            if arg in BANDWIDTHS:
                self._modem.set_bandwidth(arg)
                self.bandwidth = arg
                self._send_cmd(OK)
            else:
                self._send_cmd(WRONG)

        elif verb == "COMPRESSION":
            arg = up.split()[1] if len(up.split()) > 1 else ""
            if len(up.split()) == 2 and arg in COMPRESSION_MODES:
                from ..vara.compression import validate_mode
                arg = validate_mode(arg)
                self._modem.set_compression(arg)
                self.compression = arg
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
            if self.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
                self._send_cmd(WRONG)
                return
            with self._event_lock:
                previous = self.state
                self._connect_epoch += 1
                self.state = SessionState.CONNECTING
            try:
                self._modem.connect(src, dst, vias)
            except Exception:
                if self.state == SessionState.CONNECTING:
                    self.state = previous
                raise
            self._send_cmd(OK)

        elif up == "DISCONNECT":
            previous = self.state
            self.state = SessionState.DISCONNECTING
            try:
                self._modem.disconnect()
            except Exception:
                if self.state == SessionState.DISCONNECTING:
                    self.state = previous
                raise
            self._send_cmd(OK)

        elif up == "ABORT":
            completed_abort = getattr(self._modem, 'abort_with_completion', None)
            if completed_abort is None:
                self._modem.abort()
            else:
                with self._event_lock:
                    epoch = self._connect_epoch
                def completed():
                    # The RF owner must not take _operation_lock: a host command
                    # can hold it while waiting for the modem's owner lock.
                    with self._event_lock:
                        if self._connect_epoch == epoch:
                            self.state = (SessionState.LISTENING if self.listening
                                          else SessionState.DISCONNECTED)
                completed_abort(completed)
            self._send_cmd(OK)

        # -- mode/timing selection ---------------------------------------
        elif up in ("WINLINK SESSION", "P2P SESSION"):
            mode = up.split()[0]
            self._modem.set_session_mode(mode)
            self.session_mode = mode
            self._send_cmd(OK)

        elif up in ("CHAT ON", "CHAT OFF"):
            self._modem.set_chat(up == 'CHAT ON')
            if up == "CHAT ON":
                self.chat = True
                self.listen_cq = True
                self.listening = True
                self._modem.set_listen(True)
                if self.state == SessionState.DISCONNECTED:
                    self.state = SessionState.LISTENING
            else:
                self.chat = False
                self.listen_cq = False
            self._send_cmd(OK)

        elif up == "CLEANTXBUFFER":
            self._modem.clean_tx_buffer()

        elif verb == "BROADCAST":
            if not getattr(self._modem, 'supports_broadcast', False):
                raise ValueError('BROADCAST is unsupported')
            if up == 'BROADCAST CAPS':
                self._send_cmd('BROADCAST CAPS 0 1')
                return
            parts = cmd.split()
            if (len(parts) != 4 or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', parts[1])
                    or parts[2] not in ('0', '1')):
                raise ValueError('BROADCAST requires token, command 0/1 and UI hex')
            token = parts[1]
            if token in self._broadcast_tokens:
                raise ValueError('BROADCAST token was already used')
            if self.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
                raise ValueError('VARA contact is already active')
            if len(parts[3]) > 512:
                raise ValueError('BROADCAST envelope is too large')
            raw = bytes.fromhex(parts[3])
            def completed(sent, reason):
                status = 'SENT' if sent else 'FAILED ' + reason
                self._event('command', f'BROADCAST {token} {status}')
            self._modem.send_broadcast(int(parts[2]), raw, completed=completed)
            self._broadcast_tokens.append(token)
            self._send_cmd(f'BROADCAST {token} QUEUED')

        elif verb == "BEACONFORCE":
            # Kestrel extension: admission ACK, not proof of RF transmission.
            if not getattr(self._modem, "supports_frame_force", False):
                raise ValueError("BEACONFORCE is unsupported")
            if self.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
                raise ValueError("VARA contact is already active")
            text = cmd.partition(" ")[2]
            self._modem.send_beacon(text, force=True)
            self._send_cmd(OK)

        elif verb in ("CQFRAME", "CQFRAMEFORCE"):
            if verb == "CQFRAMEFORCE" and not getattr(self._modem, "supports_frame_force", False):
                raise ValueError("CQFRAMEFORCE is unsupported")
            parts = cmd.split(" ")
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"{verb} requires a call and bandwidth")
            if self.state not in (SessionState.DISCONNECTED, SessionState.LISTENING):
                raise ValueError("VARA contact is already active")
            from ..rx.cq import build_frame
            build_frame(parts[1], 500)
            try:
                if verb == "CQFRAMEFORCE":
                    self._modem.send_cq(parts[1], int(parts[2]) if parts[2] in BANDWIDTHS else 0,
                                        force=True)
                else:
                    self._modem.send_cq(parts[1], int(parts[2]) if parts[2] in BANDWIDTHS else 0)
            except ValueError:
                self._cq_reply_ready.clear()
                def reject():
                    try:
                        self._event('command', WRONG)
                    finally:
                        self._cq_reply_ready.set()
                if not self._modem.defer_cq_rejection(reject):
                    self._cq_reply_ready.set()
                    raise
                return
            self._send_cmd(OK)

        # -- compatibility commands --------------------------------------
        elif up in ("PUBLIC ON", "PUBLIC OFF", "CWID ON", "CWID OFF"):
            # Stock 4.9.0 acknowledges these exact forms without changing a
            # setting or invoking an RF control.
            self._send_cmd(OK)

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
                with self._operation_lock:
                    if self._closed:
                        break
                    self._modem.transmit(chunk)
        except OSError:
            pass
        finally:
            self.close()

    # -- heartbeat ---------------------------------------------------------

    def _heartbeat(self) -> None:
        while not self._done.wait(self._iamalive_interval):
            with self._operation_lock:
                if not self._closed:
                    try:
                        self._modem.request_heartbeat(lambda: self._event('command', 'IAMALIVE'))
                    except OSError:
                        self._close()


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
                 log=None, half_open_timeout: float = 5.0,
                 kiss_port: int | None = None) -> None:
        self.host = host
        self.modem_factory = modem_factory
        self.iamalive_interval = iamalive_interval
        self._log = log or (lambda *a: None)

        self._cmd_srv = self._listen_socket(host, cmd_port)
        try:
            self._data_srv = self._listen_socket(host, data_port)
            self._kiss = (KissPort(self._kiss_text, on_frame=self._kiss_frame, host=host, port=kiss_port, log=self._log)
                          if kiss_port is not None else None)
        except OSError:
            self._cmd_srv.close()
            raise
        self.cmd_port = self._cmd_srv.getsockname()[1]
        self.data_port = self._data_srv.getsockname()[1]
        self.kiss_port = self._kiss.port if self._kiss is not None else None

        self._running = False
        self._accept_thread: threading.Thread | None = None
        self._session: HostSession | None = None
        self.pairing_guarantee = "requires exclusive supervised endpoint isolation"
        self._pairs = SocketPairAcceptor((self._cmd_srv, self._data_srv),
                                         self._serve_pair, half_open_timeout=half_open_timeout)

    @staticmethod
    def _listen_socket(host: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        return s

    def serve_forever(self) -> None:
        self.start_background()
        while self._running:
            threading.Event().wait(.1)

    session_class = HostSession

    def _kiss_text(self, text):
        session = self._session
        if session is None:
            raise ValueError('no application attached')
        session._modem.send_beacon(text)

    def _kiss_frame(self, command, payload):
        session = self._session
        if session is None:
            raise ValueError('no application attached')
        session._modem.send_broadcast(command, payload)

    def _serve_pair(self, cmd_conn, data_conn):
        self._log("server", "application attached")
        session = self.session_class(cmd_conn, data_conn, self.modem_factory(),
                                     iamalive_interval=self.iamalive_interval, log=self._log,
                                     kiss=self._kiss)
        self._session = session
        try:
            session.run()
        finally:
            session.close()
            self._session = None
            self._log("server", "application detached")

    def start_background(self) -> None:
        """Run :meth:`serve_forever` in a daemon thread (for tests/embedding)."""
        self._running = True
        if self._kiss is not None:
            self._kiss.start()
        self._pairs.start()

    def stop(self, timeout=5.0) -> bool:
        self._running = False
        if self._session:
            self._session.close()
        if self._kiss is not None:
            self._kiss.stop(timeout)
        self._pairs.stop(timeout=timeout)
        return True
