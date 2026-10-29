# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded single writer for a framed stream; completions belong to its owner.

An unsuccessful sendall has unknown delivery extent. It poisons the stream:
neither a retry nor another application frame is safe on that socket.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import socket
import threading


class WriterUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    write_id: int
    token: object
    disposition: str
    error: str = ''


class DataWriter:
    def __init__(self, stream, *, max_bytes=262144, max_frames=64):
        if max_bytes < 1 or max_frames < 1:
            raise ValueError('writer bounds must be positive')
        self.stream = stream
        self.max_bytes, self.max_frames = max_bytes, max_frames
        self._cv = threading.Condition()
        self._queue, self._done = deque(), deque()
        self._bytes = self._frames = self._serial = 0
        self._closed = self._poisoned = False
        self._thread = threading.Thread(target=self._work, name='varac-data-writer', daemon=True)
        self._thread.start()

    @property
    def pending(self):
        with self._cv:
            return self._frames > 0

    @property
    def poisoned(self):
        with self._cv:
            return self._poisoned

    def submit(self, payload, token=None):
        payload = bytes(payload)
        with self._cv:
            if self._closed or self._poisoned:
                raise WriterUnavailable('data stream is closed or poisoned')
            # Completed but unconsumed records also count: owner must poll.
            if (self._frames + len(self._done) >= self.max_frames
                    or self._bytes + len(payload) > self.max_bytes):
                raise WriterUnavailable('data writer queue is full')
            self._serial += 1
            self._queue.append((self._serial, payload, token))
            self._frames += 1
            self._bytes += len(payload)
            self._cv.notify()
            return self._serial

    def poll(self):
        with self._cv:
            result = list(self._done)
            self._done.clear()
            return result

    def _cancel_queued(self):
        while self._queue:
            number, payload, token = self._queue.popleft()
            self._frames -= 1
            self._bytes -= len(payload)
            self._done.append(Completion(number, token, 'cancelled'))

    def _work(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    return
                number, payload, token = self._queue.popleft()
            try:
                self.stream.sendall(payload)
            except Exception as exc:
                with self._cv:
                    self._poisoned = True
                    self._frames -= 1
                    self._bytes -= len(payload)
                    self._done.append(Completion(number, token, 'unknown', str(exc)))
                    self._cancel_queued()
                return
            with self._cv:
                self._frames -= 1
                self._bytes -= len(payload)
                self._done.append(Completion(number, token, 'complete'))

    def close(self, *, timeout=0.1):
        """Interrupt real socket I/O; never wait indefinitely on a foreign stream."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cancel_queued()
            self._cv.notify_all()
        try:
            self.stream.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        self._thread.join(timeout=max(0, timeout))
