# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in, receive-only PSKReporter exporter. No replay or text-name ingestion.

Wire reference: https://beta.pskreporter.info/pskdev (2026-10-24).
Frequency is the received RF centre, not a CQ's advertised destination.
SNR is deliberately omitted: our current decoder estimates have not been
validated against PSKReporter's required 2500 Hz reference bandwidth.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import re
import math
import secrets
import socket
import struct
import threading
import time


SOFTWARE = 'hfmodem 0.3.0-rc0'
ENDPOINT = ('report.pskreporter.info', 4739)
INTERVAL = 300.0
ENTERPRISE = 30351
RECEIVER = 0x9992
SENDER = 0x9993
_CALL = re.compile(r'(?=.{3,16}\Z)(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]+(?:/[A-Z0-9]+)*\Z')
_GRID = re.compile(r'[A-R]{2}[0-9]{2}(?:[A-X]{2}(?:[0-9]{2})?)?\Z', re.I)


def _string(value):
    raw = value.encode('utf-8')
    if len(raw) > 254 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError('invalid PSKReporter string')
    return bytes([len(raw)]) + raw


def _set(kind, data):
    data += b'\0' * (-len(data) % 4)
    return struct.pack('!HH', kind, 4 + len(data)) + data


def _field(number, size):
    return struct.pack('!HHI', 0x8000 | number, size, ENTERPRISE)


# Exact layouts of the public three-field receiver and five-field sender examples.
RECEIVER_TEMPLATE = _set(3, struct.pack('!HHH', RECEIVER, 3, 1)
                         + b''.join(_field(n, 65535) for n in (2, 4, 8)))
SENDER_TEMPLATE = _set(2, struct.pack('!HH', SENDER, 5)
                       + _field(1, 65535) + _field(5, 4)
                       + _field(10, 65535) + _field(11, 1)
                       + struct.pack('!HH', 150, 4))


@dataclass(frozen=True)
class Spot:
    call: str
    frequency_hz: int
    received_at: int
    source: str

    def __post_init__(self):
        if not _CALL.fullmatch(self.call):
            raise ValueError('invalid transmitting callsign')
        if (type(self.frequency_hz) is not int or not 1 <= self.frequency_hz <= 0xffffffff
                or type(self.received_at) is not int or not 0 <= self.received_at <= 0xffffffff):
            raise ValueError('invalid received frequency/time')
        if self.source not in ('cq', 'beacon'):
            raise ValueError('only validated direct CQ/beacon decodes may be reported')


@dataclass(frozen=True)
class Config:
    call: str
    locator: str
    enabled: bool = False
    capacity: int = 256

    def __post_init__(self):
        if not _CALL.fullmatch(self.call) or not _GRID.fullmatch(self.locator):
            raise ValueError('PSKReporter requires a callsign and Maidenhead locator')
        if type(self.capacity) is not int or not 1 <= self.capacity <= 4096:
            raise ValueError('invalid PSKReporter queue capacity')


class Encoder:
    def __init__(self, call, locator, *, software=None, domain=None):
        self.receiver = _set(RECEIVER, _string(call) + _string(locator)
                             + _string(software or SOFTWARE))
        self.domain = secrets.randbits(32) if domain is None else domain
        self.sequence = 0

    def packet(self, spots, *, now, templates=True, mode='VARAC'):
        body = (RECEIVER_TEMPLATE + SENDER_TEMPLATE if templates else b'') + self.receiver
        records = b''.join(_string(s.call) + struct.pack('!I', s.frequency_hz)
                           + _string(mode) + b'\x01' + struct.pack('!I', s.received_at)
                           for s in spots)
        body += _set(SENDER, records)
        return struct.pack('!HHIII', 10, len(body) + 16, int(now),
                           self.sequence, self.domain) + body


class Reporter:
    """Bounded daemon worker; enqueue never performs DNS or network I/O.

    Disabled construction creates no socket/thread. Explicit stop discards queued
    reports; normal close asks the same worker for a bounded final UDP drain.
    UDP submission is not server delivery.
    """
    def __init__(self, config, *, clock=time.monotonic, wall=time.time,
                 socket_factory=socket.socket, resolver=socket.getaddrinfo,
                 start_worker=True):
        self.config = config
        self.clock, self.wall = clock, wall
        self._socket_factory, self._resolver = socket_factory, resolver
        self._condition = threading.Condition()
        self._queue = deque()
        self._recent = OrderedDict()
        self._stop = False
        self._send_lock = threading.Lock()
        self._inflight = False
        self._attempting = False
        self._active_count = 0
        self._closing = False
        self.unsent_at_shutdown = self.inflight_at_shutdown = 0
        self._encoder = Encoder(config.call, config.locator)
        self._socket = None
        self._address = None
        self._next = clock() + INTERVAL
        self._last_template = None
        self._packets = 0
        self.submitted = self.dropped = self.duplicates = 0
        self.error = None
        self._worker = None
        if config.enabled and start_worker:
            self._worker = threading.Thread(target=self._run, name='pskreporter', daemon=True)
            self._worker.start()

    def accept_heard(self, record, *, live=False):
        """Explicit live adapter only; historical events are never auto-replayed."""
        if not live or record.get('rf_source') not in ('cq', 'beacon'):
            return False
        try:
            spot = Spot(record['call'], record['rx_frequency_hz'],
                        record['received_unix'], record['rf_source'])
        except (KeyError, ValueError, TypeError):
            return False
        return self.offer(spot)

    def offer(self, spot):
        with self._condition:
            if not self.config.enabled or self._stop or self._closing or spot.call == self.config.call:
                return False
            now = self.clock()
            while self._recent and next(iter(self._recent.values())) <= now - INTERVAL:
                self._recent.popitem(last=False)
            if spot.call in self._recent or any(item.call == spot.call for item in self._queue):
                self.duplicates += 1
                return False
            if len(self._queue) >= self.config.capacity:
                self.dropped += 1
                return False
            self._recent[spot.call] = now
            while len(self._recent) > 4096:
                self._recent.popitem(last=False)
            self._queue.append(spot)
            self._condition.notify_all()
            return True

    def status(self):
        with self._condition:
            enabled = self.config.enabled and not self._stop
            if not enabled:
                worker_state = 'stopping' if self._attempting else 'off'
            elif self._attempting:
                worker_state = 'flushing' if self._closing else 'uploading'
            elif self._closing:
                worker_state = 'flushing'
            elif self._worker is None:
                worker_state = 'not started'
            else:
                worker_state = 'waiting' if self._worker.is_alive() else 'stopped'
            countdown = (max(0, math.ceil(self._next - self.clock()))
                         if enabled and self._queue and not self._attempting
                         and not self._closing and worker_state != 'stopped' else None)
            return dict(enabled=enabled,
                        next_upload_seconds=countdown, worker_state=worker_state,
                        stopping=self._stop and self._attempting,
                        unsent_at_shutdown=self.unsent_at_shutdown,
                        inflight_at_shutdown=self.inflight_at_shutdown,
                        queued=len(self._queue), submitted=self.submitted,
                        dropped=self.dropped, duplicates=self.duplicates,
                        error=self.error, delivery='UDP submission; server receipt unconfirmed')

    def _batch(self, *, flush=False):
        """Worker-owned send, exposed for deterministic fake-UDP tests."""
        with self._condition:
            now = self.clock()
            if self._stop or not self.config.enabled or (not flush and now < self._next):
                return
            self._next = now + INTERVAL
            if not self._queue:
                return
            templates = (self._packets < 3 or self._last_template is None
                         or now - self._last_template >= 3600)
            spots = []
            for spot in self._queue:
                trial = self._encoder.packet(spots + [spot], now=self.wall(), templates=templates)
                if len(trial) > 1200:
                    break
                spots.append(spot)
            packet = self._encoder.packet(spots, now=self.wall(), templates=templates)
            self._attempting = True
            self._active_count = len(spots)
        try:
            if self._socket is None:
                resolved = self._resolver(*ENDPOINT, type=socket.SOCK_DGRAM)
                family, kind, proto, _, address = resolved[0]
                sock = self._socket_factory(family, kind, proto)
                sock.settimeout(2.0)
                self._socket, self._address = sock, address
            with self._send_lock:
                with self._condition:
                    if self._stop:
                        return
                    self._inflight = True
                sent = self._socket.sendto(packet, self._address)
                if sent != len(packet):
                    raise OSError('short UDP submission')
                with self._condition:
                    if not self._stop:
                        for _ in spots:
                            self._queue.popleft()
                    self.submitted += len(spots)
                    self._encoder.sequence = (self._encoder.sequence + len(spots)) & 0xffffffff
                    self._packets += 1
                    if templates:
                        self._last_template = now
                    self.error = None
                    self._inflight = self._attempting = False
                    return True
        except OSError as exc:
            with self._condition:
                self.error = str(exc)
            return
        finally:
            with self._condition:
                self._inflight = self._attempting = False

    def _run(self):
        try:
            while True:
                with self._condition:
                    if self._stop:
                        return
                    flush = self._closing
                    if flush and not self._queue:
                        return
                    delay = self._next - self.clock()
                    if not flush and delay > 0:
                        self._condition.wait(delay)
                        continue
                if not self._batch(flush=flush) and (flush or self._closing):
                    return  # Fire-and-forget: no shutdown retry or ACK wait.
        finally:
            if self._socket is not None:
                self._socket.close()

    def close(self, *, timeout=2.5):
        """Normal shutdown: bounded worker-only final UDP drain, no server ACK."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('shutdown timeout must be finite and nonnegative')
        with self._condition:
            if self._stop or not self.config.enabled:
                return self.status()
            self._closing = True
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name='pskreporter', daemon=True)
                self._worker.start()
            self._condition.notify_all()
        self._worker.join(timeout)
        with self._condition:
            if not self._stop:
                self.inflight_at_shutdown = self._active_count if self._inflight else 0
                self.unsent_at_shutdown = max(0, len(self._queue) - self.inflight_at_shutdown)
                self._stop = True
                self._queue.clear()
                self._condition.notify_all()
        # Prevent DNS completing after the deadline from admitting another send.
        self.stop(wait=0)
        return self.status()

    def stop(self, *, wait=0.2):
        with self._condition:
            self._stop = True
            self._queue.clear()
            self._condition.notify_all()
        # DNS can finish later, but no new send can be admitted after _stop.
        # A previously admitted send is visible as stopping until it returns.
        if self._send_lock.acquire(timeout=wait):
            self._send_lock.release()
        if self._worker is not None:
            self._worker.join(wait)
        elif self._socket is not None and not self._inflight:
            self._socket.close()
