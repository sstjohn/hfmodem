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

MAX_LINE = 4096
MAX_PENDING = 64
MAX_OUTPUT = 256 * 1024
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


class Connection:
    """One writer owns the socket; producers wait for output capacity."""

    def __init__(self, sock, *, line=False, eager=False, line_limit=MAX_LINE, nodelay=True):
        self.sock = sock
        self._opening = Opening(line=line, line_limit=line_limit)
        self._input = bytearray()
        sock.settimeout(0.5)
        if nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.closed = False
        self._bytes = 0
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._send_lock = threading.Lock()
        self._delivery = deque()
        self._delivering = False
        self._close_lock = threading.Lock()
        self._closing = False
        self._q = queue.Queue(maxsize=MAX_PENDING)
        self._writer = threading.Thread(target=self._write, daemon=True, name="host-writer")
        self._writer.start()
        self._delivery_thread = threading.Thread(target=self._deliver, daemon=True,
                                                 name="host-delivery")
        self._delivery_thread.start()
        self._ready = threading.Event()
        if eager:
            threading.Thread(target=self._read_opening, daemon=True, name="host-opening").start()
        else:
            self._ready.set()

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def settimeout(self, value):
        self.sock.settimeout(0.5 if value is None else min(0.5, max(0.01, value)))

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
        """Hand off modem output without waiting for a client's socket.

        The backlog must retain received data while the client is stalled;
        applying its backpressure to reception would lose incoming samples.
        """
        data = bytes(data)
        with self._drained:
            if not self.closed and not self._closing:
                self._delivery.append(data)
                self._drained.notify_all()

    def _deliver(self):
        while True:
            with self._drained:
                self._drained.wait_for(lambda: self.closed or self._delivery)
                if self.closed:
                    return
                data = self._delivery.popleft()
                self._delivering = True
            try:
                self._sendall(data, draining=True)
            finally:
                with self._drained:
                    self._delivering = False
                    self._drained.notify_all()

    def _sendall(self, data, *, draining):
        data = memoryview(bytes(data))
        with self._send_lock:
            for start in range(0, len(data), MAX_OUTPUT):
                chunk = bytes(data[start:start + MAX_OUTPUT])
                with self._drained:
                    self._drained.wait_for(
                        lambda: self.closed or (self._closing and not draining)
                        or (not self._q.full() and self._bytes + len(chunk) <= MAX_OUTPUT))
                    if self.closed or (self._closing and not draining):
                        return
                    self._q.put_nowait(chunk)
                    self._bytes += len(chunk)

    def _write(self):
        while not self.closed:
            try:
                data = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._drained:
                self._drained.notify_all()
            view = memoryview(data)
            try:
                while view and not self.closed:
                    try:
                        sent = self.sock.send(view)
                    except (socket.timeout, BlockingIOError):
                        continue
                    if not sent:
                        raise ConnectionResetError("host socket stopped writing")
                    view = view[sent:]
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

    def close(self):
        if not self._close_lock.acquire(blocking=False):
            return
        try:
            if self.closed:
                return
            self._drain_output()
            with self._drained:
                self.closed = True
                self._delivery.clear()
                self._drained.notify_all()
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
        finally:
            self._close_lock.release()


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
