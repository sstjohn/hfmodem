# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""bateleur's host dialect: a command port and a data port.

Neither standard defines how an application drives a modem, and the dialects
the commercial and military modems speak are their own. This one is small and
written down in ``docs/protocols/bateleur/HOST-API.md``; it is not a copy of any
other modem's, and an application written for one of those will not attach
here unchanged.

**Command port.** ASCII lines, one command each, answered with one line:
``OK ...`` or ``ERR <reason>``. Between answers the modem also writes
unsolicited lines — ``RX START``, ``RX END``, ``TX START``, ``TX END`` — so a
client reads the port as a stream of lines and matches answers to the
commands it sent, in order.

**Data port.** Bytes written are queued for transmission, and go out on
``SEND`` (or on their own after ``AUTOSEND`` milliseconds of quiet). Bytes
received off the air are written back as they are decoded.

One application at a time: a new connection on either port replaces the old
one, which is closed. A reconnecting application must not find the previous
one still attached and reading its traffic.
"""
from __future__ import annotations

import socket
import threading
import time

from hfmodem.core import hostio

from .modem import BateleurModem, Mode, ModemError, Observer

DEFAULT_CMD_PORT = 8310
DEFAULT_DATA_PORT = 8311

VERSION = "bateleur 1"


class BateleurHost(Observer):
    def __init__(self, modem: BateleurModem, *, cmd_port: int = DEFAULT_CMD_PORT,
                 data_port: int = DEFAULT_DATA_PORT, host: str = "127.0.0.1") -> None:
        self.modem = modem
        modem.observer = self
        self._bind_host = host
        self._want = (cmd_port, data_port)
        self.cmd_port = self.data_port = 0
        self._srv: list[socket.socket] = []
        self._cmd: socket.socket | None = None
        self._data: socket.socket | None = None
        self._lock = threading.Lock()
        self._txbuf = bytearray()
        self._last_data = 0.0
        self.autosend_ms: int | None = None
        self._running = False
        self._threads: list[threading.Thread] = []

    # -- lifecycle ---------------------------------------------------------

    def start_background(self) -> None:
        """Bind both ports before returning, then accept on threads."""
        cmd = self._listen(self._want[0])
        try:
            data = self._listen(self._want[1])
        except OSError:
            cmd.close()
            raise
        self._srv = [cmd, data]
        self.cmd_port = cmd.getsockname()[1]
        self.data_port = data.getsockname()[1]
        self._running = True
        for fn, s in ((self._accept_cmd, cmd), (self._accept_data, data)):
            t = threading.Thread(target=fn, args=(s,), daemon=True, name="bateleur-host")
            t.start()
            self._threads.append(t)
        t = threading.Thread(target=self._autosender, daemon=True, name="bateleur-autosend")
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        self.modem.abort()
        for s in self._srv + [self._cmd, self._data]:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        for t in self._threads:
            t.join(timeout=2.0)

    def _listen(self, port: int) -> socket.socket:
        return hostio.listen(self._bind_host, port)

    def _take(self, srv):
        while self._running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if not self._running:
                conn.close()
                continue
            yield hostio.Connection(conn, line=srv is self._srv[0])

    # -- the command port --------------------------------------------------

    def _accept_cmd(self, srv) -> None:
        for conn in self._take(srv):
            with self._lock:
                old, self._cmd = self._cmd, conn
            if old is not None:
                old.close()
            threading.Thread(target=self._serve_cmd, args=(conn,), daemon=True).start()

    def _serve_cmd(self, conn: socket.socket) -> None:
        hostio.serve_lines(conn, lambda: self._running, self.command, self._line)

    def command(self, line: str) -> str:
        """One command, answered. Public so the dialect is testable without
        a socket."""
        word, _, rest = line.partition(" ")
        word = word.upper()
        args = rest.split()
        try:
            if word == "VERSION":
                return f"OK {VERSION}"
            if word == "MODE":
                if not args:
                    m = self.modem.mode
                    return f"OK MODE {m.standard} {m.name}"
                if len(args) != 2:
                    raise ModemError("MODE <110|4285|4529|4539|WBHF|39|16|FSK> <mode>")
                mode = Mode.parse(args[0], args[1])
                self._on_modem(lambda: self.modem.set_mode(mode))
                return f"OK MODE {mode.standard} {mode.name}"
            if word == "BITORDER":
                if len(args) != 1:
                    raise ModemError("BITORDER <LSB|MSB>")
                self._on_modem(lambda: self.modem.set_bit_order(args[0]))
                return f"OK BITORDER {args[0].upper()}"
            if word == "FRAMING":
                if len(args) == 1 and args[0].upper() == "SYNC":
                    fmt = "SYNC"
                elif len(args) == 2 and args[0].upper() == "ASYNC":
                    fmt = args[1].upper()
                else:
                    raise ModemError("FRAMING SYNC | FRAMING ASYNC <format, e.g. 8N1>")
                from .charframe import CharFormat
                if fmt != "SYNC":
                    CharFormat.parse(fmt)               # refuse a bad one here, not later
                self._on_modem(lambda: self.modem.set_framing(fmt))
                return "OK FRAMING SYNC" if fmt == "SYNC" else f"OK FRAMING ASYNC {fmt}"
            if word == "ZEROINTERLEAVE":
                if len(args) != 1 or args[0].upper() not in ("ON", "OFF"):
                    raise ModemError("ZEROINTERLEAVE <ON|OFF>")
                on = args[0].upper() == "ON"
                self._on_modem(lambda: self.modem.set_zero_interleave(on))
                return f"OK ZEROINTERLEAVE {args[0].upper()}"
            if word == "LISTEN":
                if len(args) != 1 or args[0].upper() not in ("ON", "OFF"):
                    raise ModemError("LISTEN <ON|OFF>")
                on = args[0].upper() == "ON"
                self._on_modem(lambda: setattr(self.modem, "listen", on))
                return f"OK LISTEN {args[0].upper()}"
            if word == "SEND":
                n = self._flush()
                return f"OK SEND {n}"
            if word == "AUTOSEND":
                if len(args) != 1:
                    raise ModemError("AUTOSEND <milliseconds|OFF>")
                if args[0].upper() == "OFF":
                    self.autosend_ms = None
                else:
                    ms = int(args[0])
                    if ms <= 0:
                        raise ModemError("AUTOSEND takes a positive number of milliseconds")
                    self.autosend_ms = ms
                return f"OK AUTOSEND {args[0].upper()}"
            if word == "ABORT":
                with self._lock:
                    n = len(self._txbuf)
                    self._txbuf.clear()
                self.modem.abort()
                return f"OK ABORT {n}"
            if word == "BUFFER":
                with self._lock:
                    return f"OK BUFFER {len(self._txbuf)}"
            if word == "STATUS":
                m = self.modem
                rx = self._on_modem(lambda: "receiving" if m.receiving else "idle")
                tx = "sending" if m.transmitting else "idle"
                with self._lock:
                    n = len(self._txbuf)
                return (f"OK STATUS MODE {m.mode.standard} {m.mode.name} RX {rx} "
                        f"TX {tx} BUFFER {n}")
            raise ModemError(f"unknown command {word!r}")
        except Exception as exc:
            return f"ERR {exc}"

    def _on_modem(self, fn, timeout: float | None = None) -> None:
        """Run ``fn`` on the modem's thread and wait for it, so a command's
        answer reports what the modem did rather than what it was asked."""
        return hostio.posted(self.modem.post, fn, timeout)

    def _line(self, conn, text: str) -> None:
        try:
            conn.sendall(text.encode("ascii", "replace") + b"\r\n")
        except OSError:
            pass

    def _event(self, text: str) -> None:
        with self._lock:
            conn = self._cmd
        if conn is not None:
            conn.send_later(text.encode("ascii", "replace") + b"\r\n")

    # -- the data port -----------------------------------------------------

    def _accept_data(self, srv) -> None:
        for conn in self._take(srv):
            with self._lock:
                old, self._data = self._data, conn
            if old is not None:
                old.close()
            threading.Thread(target=self._serve_data, args=(conn,), daemon=True).start()

    def _serve_data(self, conn: socket.socket) -> None:
        try:
            self._read_data(conn)
        finally:
            conn.close()

    def _read_data(self, conn):
        while self._running and not conn.closed:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            with self._lock:
                if len(self._txbuf) + len(chunk) > 65536:
                    conn.close()
                    return
                if conn is not self._data:
                    return
                self._txbuf += chunk
                self._last_data = time.monotonic()

    def _flush(self) -> int:
        with self._lock:
            data = bytes(self._txbuf)
            if data:
                self.modem.queue_transmit(data)
                self._txbuf.clear()
        return len(data)

    def _autosender(self) -> None:
        while self._running:
            time.sleep(0.05)
            ms = self.autosend_ms
            if ms is None:
                continue
            with self._lock:
                due = self._txbuf and time.monotonic() - self._last_data >= ms / 1000
            if due:
                try:
                    self._flush()
                except (ModemError, RuntimeError) as exc:
                    self.autosend_ms = None
                    self._event(f"ERR AUTOSEND {exc}")

    # -- what the modem reports (its thread) --------------------------------

    def rx_started(self, standard, mode, freq_hz) -> None:
        self._event(f"RX START {standard} {mode} OFFSET {freq_hz:+.1f}")

    def rx_data(self, data: bytes) -> None:
        with self._lock:
            conn = self._data
        if conn is not None:
            try:
                conn.send_later(data)
            except OSError:
                pass

    def rx_ended(self, complete, snr_db, why) -> None:
        state = "EOM" if complete else "LOST"
        self._event(f"RX END {state} SNR {snr_db:.0f} {why}")

    def tx_started(self, mode, n_bytes, seconds) -> None:
        self._event(f"TX START {mode.standard} {mode.name} {n_bytes} {seconds:.1f}")

    def tx_ended(self, sent, why) -> None:
        self._event("TX END" if sent else f"TX REFUSED {why}")
