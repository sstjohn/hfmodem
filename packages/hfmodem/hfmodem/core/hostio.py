# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Host output backpressure and cancellable calls to a modem thread."""
from __future__ import annotations

from collections import deque
import queue
import re
import select
import socket
import threading
import time

MAX_LINE = 4096
MAX_PENDING = 64
MAX_OUTPUT = 256 * 1024
MAX_DELIVERY = 8 * 1024 * 1024
STALL_S = 2.0
_request = threading.local()


def listen(host, port, backlog=1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(backlog)
        sock.settimeout(0.2)
        return sock
    except OSError:
        sock.close()
        raise


_HTTP_METHODS = (b"GET", b"POST", b"HEAD", b"PUT", b"DELETE", b"OPTIONS",
                 b"PATCH", b"CONNECT", b"TRACE")
_HTTP_REQUEST = re.compile(rb"^(?:" + b"|".join(_HTTP_METHODS) + rb") +(?:/|\*|https?://)")
_HTTP_VERSION = re.compile(rb"(?:^|[ \t])HTTP/1\.[0-9](?:[ \t]|$)")


class Opening:
    """Hold only the first command line or a possible HTTP method prefix."""

    def __init__(self, *, line=False, line_limit=MAX_LINE):
        self.line = line
        self.line_limit = line_limit
        self.pending = bytearray()
        self.done = False

    def push(self, chunk):
        if self.done:
            return chunk
        self.pending.extend(chunk)
        data = bytes(self.pending)
        first = re.split(rb"[\r\n]", data, maxsplit=1)[0]
        if _HTTP_REQUEST.match(first) or _HTTP_VERSION.search(first):
            raise ConnectionAbortedError("HTTP request on a modem host port")
        complete = len(first) != len(data)
        possible = any((method + b" " + target).startswith(first)
                       for method in _HTTP_METHODS for target in (b"/", b"*", b"http://", b"https://"))
        if not complete and (self.line or possible):
            if self.line_limit is not None and len(data) > self.line_limit:
                raise ConnectionAbortedError("host opening exceeds line limit")
            return b""
        self.done = True
        self.pending.clear()
        return data


def join_threads(threads, deadline):
    threads = tuple(t for t in threads if t is not None)
    current = threading.current_thread()
    for thread in threads:
        if thread is not current:
            thread.join(max(0.0, deadline - time.monotonic()))
    return all(not thread.is_alive() for thread in threads)


class Connection:
    """One writer owns the socket; producers wait for output capacity."""

    def __init__(self, sock, *, line=False, eager=False, line_limit=MAX_LINE, nodelay=True):
        self.sock = sock
        self._opening = Opening(line=line, line_limit=line_limit)
        self._input = bytearray()
        sock.settimeout(0.5)
        if nodelay and getattr(sock, 'family', socket.AF_INET) in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.closed = False
        self._bytes = 0
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._send_lock = threading.Lock()
        self._delivery = deque()
        self._delivery_bytes = 0
        self._delivering = False
        self._close_lock = threading.Lock()
        self._closing = False
        self._q = queue.Queue(maxsize=MAX_PENDING)
        self._ready = threading.Event()
        self._opening_thread = None
        self._writer = threading.Thread(target=self._write, daemon=True, name="host-writer")
        self._writer.start()
        self._delivery_thread = threading.Thread(target=self._deliver, daemon=True,
                                                 name="host-delivery")
        self._delivery_thread.start()
        if eager:
            self._opening_thread = threading.Thread(target=self._read_opening, daemon=True,
                                                     name="host-opening")
            self._opening_thread.start()
        else:
            self._ready.set()

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def settimeout(self, value):
        self.sock.settimeout(0.5 if value is None else min(0.5, max(0.01, value)))

    def join_workers(self, timeout=2.):
        """Verify retirement before a supervisor releases this attachment."""
        deadline = time.monotonic()+timeout
        for worker in (self._writer, self._delivery_thread, self._opening_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(max(0., deadline-time.monotonic()))
                if worker.is_alive():
                    raise RuntimeError('host socket worker did not retire')

    def _read_opening(self):
        try:
            while not self.closed and not self._opening.done:
                try:
                    chunk = self.sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    self.close()
                    break
                self._input.extend(self._opening.push(chunk))
        except OSError:
            self.close()
        finally:
            self._ready.set()

    def recv(self, size):
        self._ready.wait()
        while not self.closed:
            try:
                if self._input:
                    data = bytes(self._input[:size])
                    del self._input[:size]
                    return data
                chunk = self.sock.recv(size)
                if not chunk:
                    return b""
                try:
                    data = self._opening.push(chunk)
                except ConnectionAbortedError:
                    self.close()
                    return b""
                if data:
                    self._input.extend(data[size:])
                    return data[:size]
            except socket.timeout:
                continue
        return b""

    def is_open(self):
        if self.closed:
            return False
        try:
            ready, _, _ = select.select([self.sock], [], [], 0)
            if ready and not self.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                self.close()
        except BlockingIOError:
            pass
        except (OSError, ValueError):
            self.close()
        return not self.closed

    def sendall(self, data):
        self._sendall(data, draining=False)

    def send_later(self, data):
        """Retain received bursts through short host stalls without blocking the modem."""
        data = bytes(data)
        with self._drained:
            if self.closed or self._closing:
                return
            stalled = self._delivery_bytes + len(data) > MAX_DELIVERY
            if not stalled:
                self._delivery.append(data)
                self._delivery_bytes += len(data)
                self._drained.notify_all()
        if stalled:
            self.cancel()

    def send_bounded(self, data):
        """Nonblocking modem output with explicit attachment failure on overflow.

        Existing send_later callers retain their own policy. Legacy dual-socket
        modem hosts use this bounded policy so stopped consumers cannot grow
        memory indefinitely or block a receive/ACK clock.
        """
        data = bytes(data)
        with self._drained:
            if self.closed or self._closing:
                return False
            overflow = (self._delivery_bytes + self._bytes + len(data) > MAX_OUTPUT
                        or len(self._delivery) >= MAX_PENDING)
            if not overflow:
                self._delivery.append(data)
                self._delivery_bytes += len(data)
                self._drained.notify_all()
                return True
        self.close(discard=True)
        return False

    def _deliver(self):
        while True:
            with self._drained:
                self._drained.wait_for(lambda: self.closed or self._delivery)
                if self.closed:
                    return
                data = self._delivery.popleft()
                self._delivery_bytes -= len(data)
                self._delivering = True
            try:
                self._sendall(data, draining=True)
            finally:
                with self._drained:
                    self._delivery_bytes -= len(data)
                    self._delivering = False
                    self._drained.notify_all()

    def _sendall(self, data, *, draining):
        data = memoryview(bytes(data))
        deadline = time.monotonic() + STALL_S
        if not self._send_lock.acquire(timeout=STALL_S):
            self.cancel()
            return
        stalled = False
        try:
            for start in range(0, len(data), MAX_OUTPUT):
                chunk = bytes(data[start:start + MAX_OUTPUT])
                with self._drained:
                    ready = self._drained.wait_for(
                        lambda: self.closed or (self._closing and not draining)
                        or (not self._q.full() and self._bytes + len(chunk) <= MAX_OUTPUT),
                        timeout=max(0, deadline - time.monotonic()))
                    if not ready:
                        stalled = True
                        break
                    if self.closed or (self._closing and not draining):
                        return
                    self._q.put_nowait(chunk)
                    self._bytes += len(chunk)
        finally:
            self._send_lock.release()
            if stalled:
                self.cancel()

    def _write(self):
        while not self.closed:
            try:
                data = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._drained:
                self._drained.notify_all()
            view = memoryview(data)
            deadline = time.monotonic() + STALL_S
            try:
                while view and not self.closed:
                    try:
                        sent = self.sock.send(view)
                    except (socket.timeout, BlockingIOError):
                        if time.monotonic() >= deadline:
                            self.cancel()
                            return
                        continue
                    if not sent:
                        raise ConnectionResetError("host socket stopped writing")
                    view = view[sent:]
                    deadline = time.monotonic() + STALL_S
            except OSError:
                self.close()
                return
            with self._drained:
                self._bytes -= len(data)
                self._drained.notify_all()

    def _drain_output(self):
        with self._drained:
            self._closing = True
            self._drained.notify_all()
            if threading.current_thread() is not self._writer:
                self._drained.wait_for(lambda: self.closed or (
                    not self._bytes and not self._delivery and not self._delivering), timeout=0.25)

    def shutdown(self, how):
        if how == socket.SHUT_RDWR:
            self.close()
        elif how == socket.SHUT_WR:
            with self._close_lock:
                self._drain_output()
                self.sock.shutdown(how)
        else:
            self.sock.shutdown(how)

    def close(self, *, discard=False):
        if not discard:
            self._drain_output()
        self.cancel()

    def cancel(self):
        with self._drained:
            if self.closed:
                return
            self.closed = True
            self._delivery.clear()
            self._delivery_bytes = 0
            self._drained.notify_all()
        self._ready.set()
        try:
            self._q.put_nowait(b"")
        except queue.Full:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def join(self, timeout: float) -> bool:
        return join_threads((self._writer, self._delivery_thread, self._opening_thread),
                            time.monotonic() + timeout)


def posted(post, fn, timeout=5.0, *, request=None):
    if request is None:
        request = getattr(_request, "current", None)
    done = threading.Event()
    lock = threading.Lock()
    state = "pending"
    result = []
    error = []

    def run():
        nonlocal state
        with lock:
            if state == "cancelled":
                return
            state = "running"
        try:
            if request is not None:
                conn, running = request
                connected = conn.is_open() if hasattr(conn, "is_open") else not conn.closed
                if not connected or not running():
                    raise ConnectionAbortedError("host connection closed; command cancelled")
            result.append(fn())
        except Exception as exc:
            error.append(exc)
        finally:
            done.set()

    post(run)
    if timeout is None:
        while not done.wait(0.1):
            if request is not None:
                conn, running = request
                if not running() or not conn.is_open():
                    with lock:
                        if state == "pending":
                            state = "cancelled"
                            raise ConnectionAbortedError("host connection closed; command cancelled")
    elif not done.wait(timeout):
        with lock:
            if state == "pending":
                state = "cancelled"
                raise TimeoutError(f"the modem did not answer in {timeout:g} s; command cancelled")
        done.wait()
    if error:
        raise error[0]
    return result[0]


def lines(conn, running):
    buf = bytearray()
    opening = Opening(line=True)
    while running() and not getattr(conn, "closed", False):
        try:
            chunk = conn.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            return
        if not chunk:
            return
        try:
            chunk = opening.push(chunk)
        except ConnectionAbortedError:
            return
        for c in chunk:
            if not running() or getattr(conn, "closed", False):
                return
            if c in (10, 13):
                line = bytes(buf).decode("ascii", "replace").strip()
                buf.clear()
                if not line:
                    continue
                yield line
            else:
                buf.append(c)
                if len(buf) > MAX_LINE:
                    return


def serve_lines(conn, running, command, reply):
    previous = getattr(_request, "current", None)
    _request.current = conn, running
    try:
        for line in lines(conn, running):
            reply(conn, command(line))
    finally:
        _request.current = previous
        conn.close()
