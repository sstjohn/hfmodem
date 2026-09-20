# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Threaded application client for Sabir's framed CBOR host interface."""
from __future__ import annotations

import socket
import threading
import time

from . import messages as M
from .hostlink import HostClient as Frames


class HostClient:
    def __init__(self, port: int, mycall: str,
                 host: str = "127.0.0.1", timeout: float = 10.0):
        self.mycall = mycall
        conn = socket.create_connection((host, port), timeout)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._frames = Frames(conn)
        hello = self._frames.hello()
        if hello is None or hello.get("m") != M.HELLO or hello.get("proto") != M.PROTO:
            self._frames.close()
            raise ValueError("invalid modem greeting")
        conn.settimeout(None)                 # one persistent, blocking reader
        self.messages: list[dict] = []
        self.state = M.ST_DISCONNECTED
        self._received = bytearray()
        self._pending: set[int] = set()
        self._next_id = 0
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._closed = False
        self._deflate = False
        self.command(M.SET_IDENTITY, station_id=mycall)
        self.command(M.SUBSCRIBE, events=[M.LINK_STATS, M.PHYSICAL_STATE])
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def command(self, m: int, **fields) -> None:
        with self._send_lock:
            self._frames.send({"m": m, **fields})

    def _reader(self) -> None:
        try:
            while (msg := self._frames.recv()) is not None:
                with self._cond:
                    self.messages.append(msg)
                    if msg["m"] == M.STATE_CHANGED:
                        self.state = msg["state"]
                    elif msg["m"] == M.DATA_RECEIVED:
                        self._received.extend(msg["data"])
                    elif msg["m"] == M.SEND_PROGRESS and msg["delivered"]:
                        self._pending.discard(msg["id"])
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _wait(self, pred, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not pred():
                left = deadline - time.monotonic()
                if self._closed or left <= 0:
                    return False
                self._cond.wait(left)
            return True

    def listen(self, on: bool = True) -> None:
        self.command(M.LISTEN, on=on)

    def set_compression(self, enabled: bool) -> None:
        self._deflate = bool(enabled)

    def connect(self, dst: str, timeout: float = 60.0) -> bool:
        self.command(M.CONNECT, peer_id=dst)
        return self.wait_connected(timeout)

    def wait_connected(self, timeout: float = 60.0) -> bool:
        return self._wait(lambda: self.state == M.ST_CONNECTED, timeout)

    def send(self, blob: bytes) -> None:
        with self._cond:
            self._next_id += 1
            msg_id = self._next_id
            self._pending.add(msg_id)
        self.command(M.SEND, data=blob, id=msg_id, deflate=self._deflate)

    def recv(self, n: int, timeout: float = 60.0) -> bytes:
        self._wait(lambda: len(self._received) >= n, timeout)
        with self._cond:
            out = bytes(self._received[:n])
            del self._received[:n]
            return out

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait until every submitted message has been peer-acknowledged."""
        return self._wait(lambda: not self._pending, timeout)

    def disconnect(self, timeout: float = 60.0) -> bool:
        self.command(M.DISCONNECT)
        return self.wait_disconnected(timeout)

    def wait_disconnected(self, timeout: float = 60.0) -> bool:
        return self._wait(lambda: self.state in (M.ST_DISCONNECTED, M.ST_LISTENING), timeout)

    def close(self) -> None:
        self._frames.close()
        self._thread.join(timeout=2)
