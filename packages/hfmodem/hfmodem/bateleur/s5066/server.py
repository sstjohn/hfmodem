# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The subnetwork interface on TCP (Annex A.2.2): port 5066, S_primitives.

Each connection is one client: it binds a SAP and trades S_primitives in
the framing of ``primitives`` (preamble, version, size, body). Many clients
at once, one SAP each, as the annex has them. Everything reaches the node
on the thread that owns it, through ``post``.
"""
from __future__ import annotations

import socket
import threading

from hfmodem.core import hostio

from . import primitives as P

DEFAULT_PORT = 5066


class SisServer:
    def __init__(self, node, post, *, port: int = DEFAULT_PORT,
                 host: str = "127.0.0.1", log=None) -> None:
        self.node = node
        self.post = post
        self._cleanup = getattr(getattr(post, "__self__", None), "post_cleanup", post)
        self._want = port
        self._bind_host = host
        self._log = log or (lambda *_: None)
        self.port = 0
        self._srv: socket.socket | None = None
        self._conns: set[socket.socket] = set()
        self._workers: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        s = hostio.listen(self._bind_host, self._want, backlog=8)
        self._srv, self.port = s, s.getsockname()[1]
        self._running = True
        self._thread = threading.Thread(target=self._accept, daemon=True, name="s5066-sis")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._srv is not None:
            self._srv.close()
        with self._lock:
            for c in list(self._conns):
                c.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join()

    def _accept(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if len(self._conns) >= 16 or not self._running:
                conn.close()
                continue
            conn = hostio.Connection(conn)
            worker = threading.Thread(target=self._client, args=(conn,), daemon=True)
            with self._lock:
                if not self._running:
                    conn.close()
                    continue
                self._conns.add(conn)
                self._workers.add(worker)
                worker.start()

    def _client(self, conn):
        try:
            self._serve(conn)
        finally:
            with self._lock:
                self._workers.discard(threading.current_thread())

    def _serve(self, conn: socket.socket) -> None:
        def send(p: P.Primitive) -> None:
            conn.send_later(P.encode(p))

        try:
            att = hostio.posted(self.post, lambda: self.node.attach(send), timeout=None,
                                request=(conn, lambda: self._running))
        except Exception as exc:
            self._log(f"s5066: attach failed ({type(exc).__name__}: {exc})")
            with self._lock:
                self._conns.discard(conn)
            conn.close()
            return
        pending = threading.BoundedSemaphore(64)

        def handle(p):
            try:
                if self._running and not conn.closed:
                    att.handle(p)
            finally:
                pending.release()

        framer = P.Framer()
        try:
            while self._running:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                for p in framer.push(chunk):
                    if not self._running or conn.closed or not pending.acquire(blocking=False):
                        conn.close()
                        break
                    try:
                        self.post(lambda p=p: handle(p))
                    except RuntimeError:
                        pending.release()
                        conn.close()
                        break
        finally:
            conn.close()
            self._cleanup(att.close)
            with self._lock:
                self._conns.discard(conn)
            conn.close()


class NodeService:
    """A node on a modem and its client port, started and stopped as one:
    what the station serves for ``[protocols.bateleur] host = "s5066"``."""

    def __init__(self, modem, address, mode, *, port: int = DEFAULT_PORT,
                 host: str = "127.0.0.1", log=None, **node_kw) -> None:
        from .onair import OnAir, parse_mode
        if isinstance(mode, str):
            mode = parse_mode(mode)
        self.onair = OnAir(modem, address, mode, log=log, **node_kw)
        self.sis = SisServer(self.onair.node, modem.post, port=port, host=host, log=log)

    @property
    def port(self) -> int:
        return self.sis.port

    def start_background(self) -> None:
        self.sis.start_background()
        self.onair.start()

    def stop(self) -> None:
        self.onair.stop()
        self.sis.stop()
