# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock's KISS port: VarAC's advanced beacon is a command-2 frame carrying twenty
characters, the same frame in both directions (bench 3, 10-23: stock keys one
record-1 burst for `C0 02 <text> C0` and the far stock hands the frame back as
sent). Commands 0 and 1 carry separately validated direct AX.25 UI envelopes;
command 1 uses the measured seven-character address form.

Stock opens the port at start-up whatever the host does ("Enable KISS"), so the
port belongs to the server and outlives attachments; a frame received with no
application attached has no modem to key and is dropped.
"""
from __future__ import annotations

import select
import socket
import threading

from hfmodem.ax25 import kiss

BEACON = 2


class KissPort:
    def __init__(self, on_text, *, on_frame=None, host='127.0.0.1', port=8100, log=None):
        self.on_text, self.on_frame = on_text, on_frame
        self._log = log or (lambda *a: None)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._listener.bind((host, port))
            self._listener.listen(4)
        except OSError:
            self._listener.close()
            raise
        self.host, self.port = self._listener.getsockname()[:2]
        self._clients: dict[socket.socket, kiss.Parser] = {}
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
            clients = list(self._clients)
        for client in clients:
            try:
                client.settimeout(2)
                client.sendall(wire)
            except OSError:
                self._drop(client)

    def _drop(self, client):
        with self._lock:
            self._clients.pop(client, None)
        client.close()

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                clients = list(self._clients)
            try:
                readable, _, _ = select.select([self._listener, *clients], [], [], .2)
            except (OSError, ValueError):
                continue
            for s in readable:
                if s is self._listener:
                    client, _ = self._listener.accept()
                    with self._lock:
                        self._clients[client] = kiss.Parser()
                    continue
                try:
                    data = s.recv(8192)
                except OSError:
                    data = b''
                if not data:
                    self._drop(s)
                    continue
                for port, command, payload in self._clients[s].feed(data):
                    if port:
                        continue
                    try:
                        if command == BEACON:
                            self.on_text(payload.decode('ascii'))
                        elif command in (0, 1) and self.on_frame is not None:
                            self.on_frame(command, payload)
                    except Exception as error:
                        self._log('kiss', f'frame dropped: {error!r}')

    def stop(self, timeout=5.0) -> bool:
        self._stop.set()
        self._listener.close()
        with self._lock:
            clients, self._clients = list(self._clients), {}
        for client in clients:
            client.close()
        self._thread.join(timeout)
        return not self._thread.is_alive()
