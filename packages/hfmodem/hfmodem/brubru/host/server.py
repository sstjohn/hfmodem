# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One KISS client, incremental framing, and bounded nonblocking output."""
from __future__ import annotations

from collections import deque
import logging
import select
import socket
import threading
import time

from hfmodem.ax25.kiss import Parser, encode

_HTTP = (b'GET ', b'HEAD ', b'POST ', b'PUT ', b'DELETE ', b'CONNECT ',
         b'OPTIONS ', b'TRACE ', b'PATCH ', b'PRI ')


class KissServer:
    def __init__(self, link, *, host='127.0.0.1', port=0, log=None):
        self.link = link
        self.host, self.port = host, port
        self.log = log or logging.getLogger(__name__).warning
        self._listener = None
        self._client = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._out = deque()
        self._offset = 0
        self._send_started = None
        self._opening = bytearray()
        self._sniffing = True
        self._parser = Parser()
        self.events = deque(maxlen=64)

    def start(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(4)
            listener.setblocking(False)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self.port = listener.getsockname()[1]
        self.link.host = self
        self._thread = threading.Thread(target=self._run, name='rpr-kiss', daemon=True)
        self._thread.start()

    def _event(self, message):
        self.events.append(message)
        self.log(message)

    def _accept(self, client):
        with self._lock:
            if self._stop.is_set() or self._client is not None:
                client.close()
                self._event('rpr KISS second client refused')
                return False
            client.setblocking(False)
            self._client = client
            self._opening.clear()
            self._sniffing = True
            self._parser = Parser()
            self._offset = 0
            self._send_started = None
            return True

    def send_frame(self, raw):
        wire = encode(raw)
        with self._lock:
            if self._client is None or self._stop.is_set():
                return
            if len(self._out) == 32:
                # Never remove the prefix of a frame already partly on the socket.
                del self._out[1 if self._offset else 0]
                self._event('rpr KISS outbound queue full: dropped oldest frame')
            self._out.append(wire)

    def _receive(self, data):
        if self._sniffing:
            self._opening.extend(data)
            opening = bytes(self._opening)
            if any(opening.startswith(method) for method in _HTTP):
                self._event('rpr KISS HTTP opening rejected')
                self._disconnect()
                return
            if any(method.startswith(opening) for method in _HTTP):
                return
            data = opening
            self._opening.clear()
            self._sniffing = False
        for port, command, payload in self._parser.feed(data):
            if port:
                continue
            if command == 0:
                self.link.enqueue(payload)
            elif command in (1, 2, 3) and len(payload) == 1:
                self.link.set_parameter(command, payload[0])
            # FullDuplex and SetHardware are accepted without changing the station.

    def _send(self, now):
        with self._lock:
            if self._client is None or not self._out:
                return
            if self._send_started is None:
                self._send_started = now
            if now - self._send_started >= 2:
                self._event('rpr KISS send timed out')
                self._disconnect()
                return
            try:
                count = self._client.send(self._out[0][self._offset:])
            except BlockingIOError:
                return
            if not count:
                raise ConnectionError('KISS socket closed during send')
            self._offset += count
            if self._offset == len(self._out[0]):
                self._out.popleft()
                self._offset = 0
                self._send_started = None

    def _disconnect(self):
        with self._lock:
            client, self._client = self._client, None
            if client is not None:
                client.close()
            self._out.clear()
            self._offset = 0
            self._send_started = None
        if client is not None:
            self.link.drop_pending()

    def _run(self):
        try:
            while not self._stop.is_set():
                client = self._client
                readers = [self._listener] + ([client] if client is not None else [])
                try:
                    readable, _, _ = select.select(readers, [], [], 0.02)
                    if self._listener in readable:
                        incoming, _ = self._listener.accept()
                        self._accept(incoming)
                    if client is not None and client in readable:
                        data = client.recv(8192)
                        if data:
                            self._receive(data)
                        else:
                            self._disconnect()
                    self._send(time.monotonic())
                except (OSError, ValueError):
                    if not self._stop.is_set():
                        self._disconnect()
        finally:
            self._disconnect()

    def stop(self):
        self._stop.set()
        self._disconnect()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(2.2)
        self.link.drop_pending()
        if self.link.host is self:
            self.link.host = None
