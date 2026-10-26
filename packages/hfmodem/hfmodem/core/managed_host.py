# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Authenticated attachment broker for a process-supervised legacy endpoint.

The broker owns no radio. Its injected handler owns one command/data attachment
or one serial stream, selected explicitly by the deployment.
A supervisor holds an authenticated heartbeat lease and tags both forwarded
streams with an increasing attachment counter. Losing the lease retires all its streams
and joins the handler before another supervisor can acquire ownership.

This is a private protocol between the station and its trusted supervisor, not
an extension to VARA or ARDOP and not a substitute for supervisor-side application
identity checks. It deliberately exposes no unauthenticated legacy backend pair.
"""
from dataclasses import dataclass, field
import hmac
import ipaddress
import secrets
import socket
import struct
import threading
import time

from .hostio import listen
from .socketpair import hang_up

# Version, deployment secret, role, supervisor lease, attachment. Role zero
# holds the lease and sends one zero heartbeat byte; one/two carry command/data,
# and three carries a serial byte stream. A broker admits exactly one profile.
HEADER = struct.Struct('!4s32sB16s16s')
MAGIC = b'HFC1'
ACCEPTED, REFUSED = b'\x00', b'\x01'


@dataclass
class _Lease:
    identity: bytes
    control: socket.socket
    pending: dict = field(default_factory=dict)
    pending_id: bytes | None = None
    pending_until: float | None = None
    active: tuple | None = None
    worker: threading.Thread | None = None
    retiring: bool = False
    # Attachment IDs are increasing unsigned counters within a server-issued
    # lease. Remembering its high-water mark is bounded and rejects old retries.
    retired_sequence: int = 0


class ManagedHost:
    """One supervised application at a time, with bounded authentication work.

    The handler must return only after all of its own readers/effects retire.
    Failure to join closes this broker permanently rather than releasing a lease.
    """
    def __init__(self, serve_pair, *, secret: bytes, host='127.0.0.1', port=0,
                 half_open_timeout=5., heartbeat_timeout=5., join_timeout=30.,
                 start_owner=lambda: None, stop_owner=lambda: None,
                 stream_roles=(1, 2)):
        if not isinstance(secret, bytes) or len(secret) != 32:
            raise ValueError('managed endpoint requires a 32-byte deployment secret')
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError('managed endpoint must bind a loopback address')
        if min(half_open_timeout, heartbeat_timeout, join_timeout) <= 0:
            raise ValueError('managed endpoint deadlines must be positive')
        if stream_roles not in ((1, 2), (3,)):
            raise ValueError('managed endpoint requires command/data or one serial stream')
        self.stream_roles = stream_roles
        self.serve_pair, self._secret = serve_pair, secret
        self.host, self.port = host, port
        self.half_open_timeout = half_open_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.join_timeout = join_timeout
        self.failure = ''
        self.start_owner, self.stop_owner = start_owner, stop_owner
        self._owner_started = False
        self._listener = self._thread = self._lease = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sockets, self._workers = set(), set()

    def start(self):
        if self._listener is not None or self._stop.is_set():
            raise RuntimeError('managed endpoint already started or stopped')
        self._listener = listen(self.host, self.port, backlog=8)
        try:
            self.start_owner()
        except BaseException:
            hang_up(self._listener)
            self._listener = None
            raise
        self._owner_started = True
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept, name='managed-host-accept', daemon=True)
        self._thread.start()

    def _close_pending(self, lease):
        if lease.pending_id is not None:
            lease.retired_sequence = max(lease.retired_sequence,
                                        int.from_bytes(lease.pending_id, 'big'))
        for conn in lease.pending.values():
            hang_up(conn)
        lease.pending.clear()
        lease.pending_id = lease.pending_until = None

    def _accept(self):
        while not self._stop.is_set():
            with self._lock:
                lease = self._lease
                if (lease is not None and lease.pending_until is not None
                        and time.monotonic() >= lease.pending_until):
                    self._close_pending(lease)
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with self._lock:
                self._workers = {worker for worker in self._workers if worker.is_alive()}
                # Three live lease/stream handlers plus a bounded number of
                # incomplete handshakes. Never spawn unbounded socket readers.
                if self._stop.is_set() or self.failure or len(self._workers) >= 8:
                    hang_up(conn)
                    continue
                self._sockets.add(conn)
                worker = threading.Thread(target=self._authenticate, args=(conn,),
                                          name='managed-host-auth', daemon=True)
                self._workers.add(worker)
                worker.start()

    def _authenticate(self, conn):
        adopted = False
        try:
            conn.settimeout(2.)
            data = bytearray()
            deadline = time.monotonic() + 2.
            while len(data) < HEADER.size:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return
                conn.settimeout(remaining)
                chunk = conn.recv(HEADER.size-len(data))
                if not chunk:
                    return
                data.extend(chunk)
            magic, secret, role, identity, attachment = HEADER.unpack(data)
            if magic != MAGIC or not hmac.compare_digest(secret, self._secret):
                return
            if identity == bytes(16) or role not in (0, *self.stream_roles):
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if role == 0:
                self._control(conn, identity, attachment)
            else:
                adopted = self._stream(conn, identity, attachment, role)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                self._sockets.discard(conn)
            if not adopted:
                hang_up(conn)

    def _control(self, conn, identity, attachment):
        # The server issues the live epoch. Replaying a dead supervisor's
        # registration cannot make its old forwarded stream headers valid.
        lease = _Lease(secrets.token_bytes(16), conn)
        with self._lock:
            if (self._stop.is_set() or self.failure or self._lease is not None
                    or attachment != bytes(16)):
                conn.sendall(REFUSED)
                return
            self._lease = lease
        try:
            conn.sendall(ACCEPTED + lease.identity)
            conn.settimeout(self.heartbeat_timeout)
            while not self._stop.is_set():
                if conn.recv(1) != b'\x00':
                    break
        finally:
            with self._lock:
                lease.retiring = True
                self._close_pending(lease)
                if lease.active is not None:
                    for stream in lease.active:
                        hang_up(stream)
                worker = lease.worker
            if worker is not None:
                worker.join(self.join_timeout)
            with self._lock:
                if worker is not None and worker.is_alive():
                    self.failure = 'managed attachment did not retire'
                elif self._lease is lease:
                    self._lease = None

    def _stream(self, conn, identity, attachment, role):
        with self._lock:
            lease = self._lease
            sequence = int.from_bytes(attachment, 'big')
            if (self._stop.is_set() or self.failure or lease is None or lease.retiring
                    or identity != lease.identity or attachment == bytes(16)
                    or sequence <= lease.retired_sequence
                    or lease.active is not None
                    or lease.worker is not None and lease.worker.is_alive()):
                conn.sendall(REFUSED)
                return False
            if lease.pending and (lease.pending_id != attachment or role in lease.pending):
                self._close_pending(lease)
                conn.sendall(REFUSED)
                return False
            # A pending deadline can expire between the accept-loop check and
            # this authenticated stream's arrival. Never complete that old pair.
            if lease.pending_until is not None and time.monotonic() >= lease.pending_until:
                self._close_pending(lease)
                conn.sendall(REFUSED)
                return False
            conn.sendall(ACCEPTED)
            conn.settimeout(None)
            lease.pending[role] = conn
            lease.pending_id = attachment
            if lease.pending_until is None:
                lease.pending_until = time.monotonic() + self.half_open_timeout
            if len(lease.pending) == len(self.stream_roles):
                pair = lease.active = tuple(lease.pending[side] for side in self.stream_roles)
                lease.retired_sequence = sequence
                lease.pending.clear()
                lease.pending_id = lease.pending_until = None
                lease.worker = threading.Thread(target=self._serve, args=(lease, pair),
                                                name='managed-host-attachment', daemon=True)
                lease.worker.start()
            return True

    def _serve(self, lease, pair):
        try:
            self.serve_pair(*pair)
        except OSError:
            # Closing either socket, including our lease retirement, is an
            # ordinary transport end. The handler still owes all reader joins.
            pass
        except Exception:
            # Fail the entire supervisor lease on a backend defect. Do not let
            # a replacement attachment inherit a possibly live failed handler.
            with self._lock:
                self.failure = 'managed attachment backend failed'
            hang_up(lease.control)
        finally:
            for conn in pair:
                hang_up(conn)
            with self._lock:
                if lease.active is pair:
                    lease.active = None

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._listener is not None:
            hang_up(self._listener)
        with self._lock:
            if self._lease is not None:
                hang_up(self._lease.control)
                self._close_pending(self._lease)
                if self._lease.active:
                    for conn in self._lease.active:
                        hang_up(conn)
            for conn in tuple(self._sockets):
                hang_up(conn)
            workers = tuple(self._workers)
        if self._thread is not None:
            self._thread.join(3)
            if self._thread.is_alive():
                raise RuntimeError('managed endpoint acceptor did not retire')
        for worker in workers:
            worker.join(self.join_timeout + 1)
            if worker.is_alive():
                raise RuntimeError('managed endpoint worker did not retire')
        if self._owner_started:
            self.stop_owner()
            self._owner_started = False
        if self.failure:
            raise RuntimeError(self.failure)
        return True
