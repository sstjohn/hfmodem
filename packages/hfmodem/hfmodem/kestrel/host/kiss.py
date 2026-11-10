# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA's KISS port: VarAC's advanced beacon is a command-2 frame carrying twenty
characters, the same frame in both directions (2026-10-23: a VARA HF station keys
one record-1 burst for `C0 02 <text> C0` and the far station hands the frame back as
sent). Commands 0 and 1 carry separately validated direct AX.25 UI envelopes;
command 1 uses the measured seven-character address form.

A VARA HF station opens the port at start-up whatever the host does ("Enable KISS"), so the
port belongs to the server and outlives attachments.
"""
from __future__ import annotations

import re
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from hfmodem.ax25 import kiss

BEACON = 2
ADVANCED_BEACON = re.compile(r'2[01][A-Z][A-R]{2}[0-9]{2}[A-Xa-x]{2}[A-Z0-9]{3,7}')
OUTPUT_FRAMES = 256
FULL_TIMEOUT = 30


@dataclass
class _Client:
    parser: kiss.Parser = field(default_factory=kiss.Parser)
    output: deque[bytes] = field(default_factory=deque)
    pending: bytes = b''
    full_since: float | None = None
    dropped: int = 0
    reported: int = 0


class KissPort:
    def __init__(self, on_text, *, on_frame=None, host='127.0.0.1', port=8100, log=None,
                 clock=time.monotonic):
        self.on_text, self.on_frame = on_text, on_frame
        self._log = log or (lambda *a: None)
        self._clock = clock
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._listener.bind((host, port))
            self._listener.listen(4)
            self._listener.setblocking(False)
        except OSError:
            self._listener.close()
            raise
        self.host, self.port = self._listener.getsockname()[:2]
        self._clients: dict[socket.socket, _Client] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='vara-kiss', daemon=True)

    def start(self):
        self._thread.start()

    def deliver(self, text: str) -> None:
        self.deliver_frame(BEACON, text.encode('ascii'))

    def deliver_frame(self, command: int, payload: bytes) -> None:
        wire = kiss.encode(payload, command)
        with self._lock:
            for client in self._clients.values():
                if len(client.output) == OUTPUT_FRAMES:
                    client.output.popleft()
                    client.dropped += 1
                client.output.append(wire)
                if len(client.output) == OUTPUT_FRAMES and client.full_since is None:
                    client.full_since = self._clock()

    def _drop(self, client):
        with self._lock:
            self._clients.pop(client, None)
        client.close()

    def _run(self):
        try:
            self._serve()
        finally:
            with self._lock:
                clients = list(self._clients)
            for client in clients:
                self._drop(client)

    def _serve(self):
        while not self._stop.is_set():
            logs = []
            with self._lock:
                expired = []
                for s, client in self._clients.items():
                    if client.dropped != client.reported:
                        logs.append(f'slow client {s!r}: dropped {client.dropped} oldest frames')
                        client.reported = client.dropped
                    if (client.full_since is not None
                            and self._clock() - client.full_since >= FULL_TIMEOUT):
                        expired.append(s)
                writable = [s for s, c in self._clients.items() if c.pending or c.output]
            for message in logs:
                self._log('kiss', message)
            for s in expired:
                self._log('kiss', f'slow client {s!r} disconnected: queue full for 30 s')
                self._drop(s)
            with self._lock:
                clients = list(self._clients)
                writable = [s for s in writable if s in self._clients]
            try:
                readable, writable, _ = select.select(
                    [self._listener, *clients], writable, [], .2)
            except (OSError, ValueError):
                continue
            for s in readable:
                if s is self._listener:
                    try:
                        client, _ = self._listener.accept()
                    except OSError:
                        continue
                    client.setblocking(False)
                    with self._lock:
                        self._clients[client] = _Client()
                    continue
                try:
                    data = s.recv(8192)
                except BlockingIOError:
                    continue
                except OSError:
                    data = b''
                if not data:
                    self._drop(s)
                    continue
                with self._lock:
                    client = self._clients.get(s)
                    if client is None:
                        continue
                    frames = client.parser.feed(data)
                for port, command, payload in frames:
                    self._frame(port, command, payload)
            for s in writable:
                self._write(s)

    def _write(self, s):
        try:
            with self._lock:
                client = self._clients.get(s)
                if client is None or not (client.pending or client.output):
                    return
                wire = client.pending or client.output[0]
                sent = s.send(wire)
                if sent:
                    if not client.pending:
                        client.output.popleft()
                        client.full_since = None
                    client.pending = wire[sent:]
            if not sent:
                self._drop(s)
        except BlockingIOError:
            pass
        except OSError:
            self._drop(s)

    def _frame(self, port, command, payload):
        if port:
            return
        try:
            if command == BEACON:
                text = payload.decode('ascii')
                if len(text) > 20 or not ADVANCED_BEACON.fullmatch(text.rstrip(' ')):
                    raise ValueError('not advanced beacon text')
                self.on_text(text)
            elif command in (0, 1) and self.on_frame is not None:
                self.on_frame(command, payload)
        except Exception as error:
            self._log('kiss', f'frame ignored: {error!r}')

    def stop(self, timeout=5.0) -> bool:
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout)
        return not self._thread.is_alive()
