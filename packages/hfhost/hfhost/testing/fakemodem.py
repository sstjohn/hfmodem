# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A scriptable fake VARA-dialect modem server for tests.

Mimics the real servers' attachment contract exactly: one app at a time, cmd
socket accepted first, then the data socket — so a half-attach wedges the
accept loop just like kestrel/sabir. Known verbs draw OK, unknown WRONG,
VERSION its version string; on_command overrides any of that (return a str,
a list of lines, or "" for silence; None falls through to the default).

Test controls: notify()/send_data() inject traffic, drop() kills the current
session (EOF at the client), session_log records the per-attach command
sequence, and data_listener=False simulates a dead data port for the
atomic-attach failure path.

Fidelity to the real servers is opt-in, because most tests want a silent
modem: buffer_notifications gives kestrel's BUFFER n / BUFFER 0 cadence,
iamalive_s the heartbeat, disconnected_on_disconnect the DISCONNECTED both
servers send, discard_before_connect kestrel's LoopbackModem.transmit drop of
pre-CONNECT writes, and echo (with echo_delay_s) its data loopback — a delay
longer than teardown reproduces the late-echo window epoch fencing exists for.
"""

from __future__ import annotations

import queue
import socket
import threading
import time

CR = b"\r"

_KNOWN_VERBS = {"MYCALL", "LISTEN", "CONNECT", "DISCONNECT", "ABORT",
                "COMPRESSION", "CHAT", "BW500", "BW2300", "BW2750",
                "WINLINK", "P2P", "PUBLIC", "CWID", "CQFRAME"}


def _listen_socket(host: str, port: int = 0) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(4)
    return s


def _verb(line: str) -> str:
    head = line.split()
    return head[0].upper() if head else ""


class ThreadFaults(list):
    """Faults escaping harness threads, held until the owning test tears down.

    A thread that dies hands its exception to pytest's threadexception hook,
    which files it as a warning against whichever test is at a setup/call/
    teardown boundary when it lands -- so the report names a test that had
    nothing to do with it, and, warnings not being errors here, the guilty test
    stays green either way. Collecting the fault and raising it from stop()
    puts it back on the test that owns the thread, named and with its cause.
    """

    def watch(self, name: str, fn, *args) -> threading.Thread:
        def run() -> None:
            try:
                fn(*args)
            except BaseException as exc:
                self.append(f"{name}: {exc!r}")
        t = threading.Thread(target=run, name=name, daemon=True)
        t.start()
        return t

    def raise_any(self) -> None:
        if self:
            raise RuntimeError("harness thread died -- " + "; ".join(self))


def _refused_port(host: str) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeModem:
    def __init__(self, *, host: str = "127.0.0.1",
                 ports: tuple[int, int] | None = None,
                 on_command=None, version: str = "1.0-fake",
                 data_listener: bool = True,
                 buffer_notifications: bool = False,
                 iamalive_s: float | None = None,
                 disconnected_on_disconnect: bool = False,
                 discard_before_connect: bool = False,
                 echo: bool = False, echo_delay_s: float = 0.0,
                 faults: ThreadFaults | None = None) -> None:
        self.host = host
        self.faults = ThreadFaults() if faults is None else faults
        self.on_command = on_command
        self.version = version
        self.buffer_notifications = buffer_notifications
        self.iamalive_s = iamalive_s
        self.disconnected_on_disconnect = disconnected_on_disconnect
        self.discard_before_connect = discard_before_connect
        self.echo = echo
        self.echo_delay_s = echo_delay_s

        self._cmd_srv = _listen_socket(host, ports[0] if ports else 0)
        self._data_srv = _listen_socket(host, ports[1] if ports else 0) \
            if data_listener else None
        self.cmd_port = self._cmd_srv.getsockname()[1]
        self.data_port = (self._data_srv.getsockname()[1]
                          if self._data_srv else _refused_port(host))

        self.commands: list[str] = []              # all sessions, in order
        self.session_log: list[list[str]] = []     # per-attach command lists
        self.received_data = bytearray()
        self.discarded_data = bytearray()   # dropped by discard_before_connect
        self.attach_count = 0
        self.connected = False
        self.buffer_bytes = 0

        self._cv = threading.Condition()
        self._session: tuple[socket.socket, socket.socket] | None = None
        self._send_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._txq: queue.Queue | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FakeModem":
        self._running = True
        self._thread = self.faults.watch("fakemodem", self._serve)
        return self

    def stop(self) -> None:
        self._running = False
        for srv in (self._cmd_srv, self._data_srv):
            if srv is not None:
                try:
                    srv.close()
                except OSError:
                    pass
        self.drop()
        self.faults.raise_any()

    def _serve(self) -> None:
        while self._running:
            try:
                cmd_conn, _ = self._cmd_srv.accept()
            except OSError:
                break
            if self._data_srv is None:
                cmd_conn.close()
                continue
            try:
                # cmd first, then block on data — the real servers' contract
                data_conn, _ = self._data_srv.accept()
            except OSError:
                cmd_conn.close()
                break
            for c in (cmd_conn, data_conn):
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            txq: queue.Queue = queue.Queue()
            session = (cmd_conn, data_conn)
            with self._cv:
                self._session = session
                self._txq = txq
                self.attach_count += 1
                self.connected = False
                self.buffer_bytes = 0
                self.session_log.append([])
                self._cv.notify_all()
            for name, target, args in (("fakemodem-data", self._data_loop, (data_conn,)),
                                       ("fakemodem-tx", self._tx_worker, (data_conn, txq)),
                                       ("fakemodem-heartbeat", self._heartbeat, (session,))):
                self.faults.watch(name, target, *args)
            self._cmd_loop(cmd_conn)      # session ends when cmd channel EOFs
            txq.put(None)
            self.drop()

    # -- session I/O -------------------------------------------------------

    def _cmd_loop(self, conn: socket.socket) -> None:
        buf = b""
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while CR in buf:
                    raw, buf = buf.split(CR, 1)
                    if raw:
                        self._dispatch(raw.decode("ascii", "replace"))
        except OSError:
            pass

    def _dispatch(self, cmd: str) -> None:
        with self._cv:
            self.commands.append(cmd)
            self.session_log[-1].append(cmd)
            self._cv.notify_all()
        verb = _verb(cmd)
        reply = self.on_command(cmd) if self.on_command else None
        if reply is None:
            if verb == "VERSION":
                reply = "VERSION " + self.version
            elif verb in _KNOWN_VERBS:
                reply = "OK"
            else:
                reply = "WRONG"
        for line in ([reply] if isinstance(reply, str) else reply):
            if line:
                self._emit(line)
        if verb in ("DISCONNECT", "ABORT") and self.disconnected_on_disconnect:
            self._emit("DISCONNECTED")

    def _data_loop(self, conn: socket.socket) -> None:
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                with self._cv:
                    if self.discard_before_connect and not self.connected:
                        self.discarded_data += chunk
                        self._cv.notify_all()
                        continue
                    self.received_data += chunk
                    self.buffer_bytes += len(chunk)
                    depth = self.buffer_bytes
                    txq = self._txq
                    self._cv.notify_all()
                if self.buffer_notifications:
                    self._emit(f"BUFFER {depth}")
                if txq is not None:
                    txq.put(chunk)
        except OSError:
            pass

    def _tx_worker(self, conn: socket.socket, txq: queue.Queue) -> None:
        """Drains the modem's TX queue: echo (after echo_delay_s) then report
        the drain. Echoes even once disconnected — kestrel's worker snapshots
        the link state before its burst, which is what opens the late echo."""
        while True:
            chunk = txq.get()
            if chunk is None:
                return
            if self.echo_delay_s:
                time.sleep(self.echo_delay_s)
            if self.echo:
                try:
                    conn.sendall(chunk)
                except OSError:
                    pass
            with self._cv:
                self.buffer_bytes = max(0, self.buffer_bytes - len(chunk))
                depth = self.buffer_bytes
                self._cv.notify_all()
            if self.buffer_notifications:
                self._emit(f"BUFFER {depth}")

    def _heartbeat(self, session: tuple) -> None:
        while self.iamalive_s:
            with self._cv:
                if self._cv.wait_for(lambda: self._session is not session,
                                     self.iamalive_s):
                    return
            self._emit("IAMALIVE")      # sleeps first, like both real servers

    # -- test controls -----------------------------------------------------

    def notify(self, line: str) -> None:
        """Send one CR-terminated line on the current session's cmd socket.
        Link state is tracked from the wire, so a test that fakes a connection
        with notify("CONNECTED ...") also unblocks discard_before_connect."""
        with self._cv:
            session = self._session
            head = _verb(line)
            if head == "CONNECTED":
                self.connected = True
            elif head == "DISCONNECTED":
                self.connected = False
                self.buffer_bytes = 0
            self._cv.notify_all()
        if session is None:
            raise RuntimeError("no attached session")
        with self._send_lock:
            try:
                session[0].sendall(line.encode("ascii") + CR)
            except OSError:
                pass

    def _emit(self, line: str) -> None:
        """notify() for the modem's own threads. A test that calls notify()
        with nothing attached has made a mistake and should hear about it; a
        server thread that reaches the same point has merely been outrun by a
        client hanging up, which is the ordinary end of every session."""
        try:
            self.notify(line)
        except RuntimeError:
            pass

    def send_data(self, blob: bytes) -> None:
        with self._cv:
            session = self._session
        if session is None:
            raise RuntimeError("no attached session")
        try:
            session[1].sendall(blob)
        except OSError:
            pass

    def drop(self) -> None:
        """Close the current session's sockets: EOF injection at the client."""
        with self._cv:
            session, self._session = self._session, None
            self.connected = False
            self._cv.notify_all()
        if session is not None:
            for conn in session:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass

    # -- waiting helpers ---------------------------------------------------

    def wait_attached(self, timeout: float = 2.0) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._session is not None, timeout)

    def wait_detached(self, timeout: float = 2.0) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._session is None, timeout)

    def wait_command(self, prefix: str, timeout: float = 2.0) -> str | None:
        def find():
            for c in self.commands:
                if c.startswith(prefix):
                    return c
            return None
        with self._cv:
            self._cv.wait_for(lambda: find() is not None, timeout)
            return find()

    def wait_data(self, n: int, timeout: float = 2.0) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: len(self.received_data) >= n, timeout)
