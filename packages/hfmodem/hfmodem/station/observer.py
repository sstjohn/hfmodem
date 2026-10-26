# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import socket
import threading
import time

from hfmodem.core.operations import LeaseTimeout


@dataclass(frozen=True)
class _Item:
    generation: object
    socket: object
    payload: bytes


@dataclass(frozen=True)
class _Flight:
    item: _Item
    deadline: float


class BoundedObserver:
    """Capacity is shared across generations; overflow refuses the delivering generation."""

    def __init__(self, shutdown_client, *, capacity=256, write_timeout=2.0):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not math.isfinite(write_timeout) or write_timeout <= 0:
            raise ValueError("write_timeout must be finite and positive")
        self._shutdown_client = shutdown_client
        self._capacity = capacity
        self._write_timeout = write_timeout
        self._condition = threading.Condition()
        self._queue = deque()
        self._members = {}
        self._closed = deque(maxlen=256)
        self._failed = set()
        self._notifications = deque()
        self._inflight = None
        self._stopping = False
        self._writer = threading.Thread(
            target=self._write, name="observer-writer", daemon=True)
        self._watchdog = threading.Thread(
            target=self._watch, name="observer-watchdog", daemon=True)
        self._writer.start()
        self._watchdog.start()

    def install(self, generation, sock, *, replace=None):
        with self._condition:
            if self._stopping or generation in self._closed or generation in self._failed:
                raise ValueError("generation is closed")
            if any(sock is member for members in self._members.values() for member in members):
                raise ValueError("socket is already installed")
            members = self._members.get(generation, [])
            if replace is not None:
                if not any(replace is member for member in members):
                    raise ValueError("displaced socket is not in this generation")
                self._retire_socket(generation, replace)
                members = self._members[generation]
            members.append(sock)
            self._members[generation] = members
            self._condition.notify_all()

    def deliver(self, generation, sock, payload):
        payload = bytes(payload)
        with self._condition:
            if (self._stopping or generation in self._failed
                    or not any(sock is member for member in self._members.get(generation, ()))):
                return False
            if len(self._queue) >= self._capacity:
                self._failed.add(generation)
                self._condition.notify_all()
                return False
            self._queue.append(_Item(generation, sock, payload))
            self._condition.notify_all()
            return True

    def close_generation(self, generation):
        with self._condition:
            self._retire_generation(generation)

    def stop(self, timeout=2.0):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._stopping = True
            for generation in list(self._members):
                self._retire_generation(generation)
            self._condition.notify_all()
        current = threading.current_thread()
        for thread in (self._writer, self._watchdog):
            if thread is not current:
                thread.join(max(0, deadline - time.monotonic()))
        alive = [thread.name for thread in (self._writer, self._watchdog)
                 if thread is not current and thread.is_alive()]
        if alive:
            raise LeaseTimeout("observer threads did not stop: " + ", ".join(alive))

    @staticmethod
    def _shutdown(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _retire_socket(self, generation, sock):
        self._queue = deque(item for item in self._queue if item.socket is not sock)
        self._shutdown(sock)
        if self._inflight is not None and self._inflight.item.socket is sock:
            self._inflight = None
        self._members[generation] = [member for member in self._members[generation]
                                     if member is not sock]

    def _retire_generation(self, generation, *, notify=False):
        members = self._members.pop(generation, None)
        self._failed.discard(generation)
        if members is None:
            return
        self._closed.append(generation)
        self._queue = deque(item for item in self._queue if item.generation != generation)
        for sock in members:
            self._shutdown(sock)
        if self._inflight is not None and self._inflight.item.generation == generation:
            self._inflight = None
        if notify:
            self._notifications.append(generation)
        self._condition.notify_all()

    def _write(self):
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                item = self._queue.popleft()
                if item.generation in self._failed:
                    continue
                flight = _Flight(item, time.monotonic() + self._write_timeout)
                self._inflight = flight
                self._condition.notify_all()
            failed = False
            try:
                item.socket.sendall(item.payload)
            except OSError:
                failed = True
            except Exception:
                failed = True
                logging.exception("observer sendall failed for %r", item.generation)
            finally:
                with self._condition:
                    if self._inflight is flight:
                        self._inflight = None
                        if failed or time.monotonic() >= flight.deadline:
                            self._failed.add(item.generation)
                    self._condition.notify_all()

    def _watch(self):
        while True:
            with self._condition:
                for generation in list(self._failed):
                    self._retire_generation(generation, notify=True)
                flight = self._inflight
                if flight is not None and time.monotonic() >= flight.deadline:
                    self._retire_generation(flight.item.generation, notify=True)
                if self._notifications:
                    generation = self._notifications.popleft()
                elif self._stopping:
                    return
                else:
                    delay = (max(0, self._inflight.deadline - time.monotonic())
                             if self._inflight is not None else None)
                    self._condition.wait(delay)
                    continue
            try:
                self._shutdown_client(generation)
            except Exception:
                logging.exception("observer shutdown_client failed for %r", generation)
