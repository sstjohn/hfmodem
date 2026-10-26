# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Lifecycle of legacy command/data attachments, without identity claims.

The caller must isolate both endpoints to one supervised application. Same-IP
matching and these internal generations cannot distinguish racing applications.
Readmission requires its supervisor to quiesce old connection attempts.
"""
import select
import socket
import threading
import time


def hang_up(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class SocketPairAcceptor:
    def __init__(self, listeners, serve, *, half_open_timeout=5.0):
        if half_open_timeout <= 0:
            raise ValueError("half-open lifetime must be positive")
        self.listeners = tuple(listeners)
        self.serve = serve
        self.timeout = half_open_timeout
        self.generation = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = {}
        self._deadline = None
        self._active = None
        self._worker = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="host-pair-accept", daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        self._stop.set()
        with self._lock:
            self._clear_pending()
            if self._active:
                for conn in self._active:
                    hang_up(conn)
        for listener in self.listeners:
            hang_up(listener)
        for th in (self._thread, self._worker):
            if th is not None and th is not threading.current_thread():
                th.join(timeout=max(0., deadline-time.monotonic()))
                if th.is_alive():
                    raise RuntimeError('legacy socket pair worker did not retire')
        return True

    def _clear_pending(self):
        for conn, _ in self._pending.values():
            hang_up(conn)
        self._pending.clear()
        self._deadline = None

    def _run(self):
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select(self.listeners, [], [], .05)
            except (OSError, ValueError):
                break
            with self._lock:
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    self._clear_pending()
                for listener in ready:
                    try:
                        conn, peer = listener.accept()
                    except OSError:
                        continue
                    index = self.listeners.index(listener)
                    if self._active is not None or self._stop.is_set():
                        hang_up(conn)
                        continue
                    if index in self._pending:
                        # Detectable overlap: neither pending half is adopted.
                        hang_up(conn)
                        self._clear_pending()
                        continue
                    if self._pending and peer[0] != next(iter(self._pending.values()))[1][0]:
                        hang_up(conn)
                        self._clear_pending()
                        continue
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._pending[index] = (conn, peer)
                    if self._deadline is None:
                        self._deadline = time.monotonic() + self.timeout
                    if len(self._pending) == 2:
                        pair = tuple(self._pending[i][0] for i in (0, 1))
                        self._pending.clear()
                        self._deadline = None
                        self._active = pair
                        self.generation += 1
                        self._worker = threading.Thread(target=self._serve,
                            args=(pair,), name=f"host-attachment-{self.generation}", daemon=True)
                        self._worker.start()

    def _serve(self, pair):
        try:
            self.serve(*pair)
        finally:
            for conn in pair:
                hang_up(conn)
            with self._lock:
                self._active = None
