# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA station transport: indexed capture, one owner, no device or PTT owner."""
from collections import deque
import logging
import queue
import threading
import time

import numpy as np

from .audio_runtime import AudioHostRuntime
from ..monitor import Segmenter
from ..vara.vara_arq import VaraIO, VaraState

log = logging.getLogger(__name__)
RATE = 48000
CHUNK = 4800
# The standalone transport discards 0.1 s past transmission, bounded by the
# earliest measured peer answer. This station seam still needs physical timing
# qualification; it does not infer the hardware's mute recovery from software.
ECHO_GUARD = 4800


class StationTransport(VaraIO):
    station_ptt = True
    def __init__(self, link):
        self.link = link
        from hfmodem.station.contact import ContactOwner
        self.contact = ContactOwner(link.station.arbiter, 'vara')
        self._channel_boundary = self.contact.receive_boundary
        self._claimant = None
        self._claimant_lock = threading.Lock()
        self.frames = queue.Queue(maxsize=128)
        self.fault = threading.Event()
        self.stop = threading.Event()
        self._hs = None
        self._rx_generation = 0
        self._skip_before = 0
        self._segmenter = Segmenter()
        self._bursts = deque()
        self._metadata = None
        self._sent = False
        self._freshness = (0, 0)
        self._seen_flush = 0
        self.messages = deque(maxlen=128)

    @property
    def hs(self):
        return self._hs

    @hs.setter
    def hs(self, value):
        if value is not self._hs:
            self._hs = value
            self._reset_rx(int(self.link.station.audio.samples))

    def bind_station(self, value):
        self.hs = value

    def _reset_rx(self, floor):
        self._skip_before = max(self._skip_before, floor)
        self._rx_generation += 1
        self._segmenter = Segmenter()
        self._bursts.clear()

    @property
    def receive_epoch(self):
        return self._rx_generation, self._freshness[0], self.contact.receive_boundary[0]

    def flushed(self, at):
        serial, floor = self._freshness
        self._freshness = serial+1, max(floor, at)

    def before_step(self):
        boundary = self.contact.receive_boundary
        if boundary != self._channel_boundary:
            self._reset_rx(max(boundary[1], int(self.link.station.audio.samples)))
            if self.hs is not None:
                self.hs._reset_stream()
            self._channel_boundary = boundary
        serial, floor = self._freshness
        if serial != self._seen_flush:
            self._skip_before = max(self._skip_before, floor)
            self._segmenter = Segmenter()
            self._bursts.clear()
            if self.hs is not None:
                self.hs._reset_stream()
            self._seen_flush = serial

    @property
    def receiving(self):
        return self._segmenter is not None and bool(self._segmenter.inb)

    def feed(self, at, samples):
        if self.stop.is_set():
            return
        generation = self._rx_generation
        for start in range(0, len(samples), CHUNK):
            try:
                self.frames.put_nowait((generation, at+start,
                    np.array(samples[start:start+CHUNK], np.float32, copy=True)))
            except queue.Full:
                self.fault.set()
                return

    def key(self, on):
        # The physical key belongs exclusively to ProtocolLink's arbiter.
        if on:
            self._sent = False

    def tx_metadata(self, **details):
        self._metadata = details

    def tx(self, samples):
        self._sent = False
        if (self.stop.is_set() or self.fault.is_set()
                or self._seen_flush != self._freshness[0]):
            return
        metadata, self._metadata = self._metadata, None
        if metadata is None or str(metadata.get('bandwidth_hz')) not in ('500', '2300', '2750'):
            raise ValueError('VARA burst lacks a qualified bandwidth declaration')
        self._reset_rx(int(self.link.station.audio.samples))
        self.link._transmitting = True
        try:
            self._sent = self.link.emit(samples, str(metadata['bandwidth_hz']),
                                       on_ptt=self.hs.io.report_ptt if self.hs is not None else None,
                                       responding=self.hs is not None and
                                       (self.hs.role == 'responder' or
                                        self.hs.state in (VaraState.CONNECTED, VaraState.DISCONNECTING)))
        finally:
            self._skip_before = max(self._skip_before,
                int(self.link.station.audio.samples)+ECHO_GUARD)
            self.link._transmitting = False

    def tx_went_out(self):
        return self._sent

    def pending(self):
        pass

    def admit_contact(self, modem):
        with self._claimant_lock:
            if self.stop.is_set() or self._claimant not in (None, modem):
                return False
            if not self.contact.admit(receive_epoch=self._channel_boundary[0]):
                return False
            self._claimant = modem
            return True

    def retire_contact(self, modem, *, idle):
        with self._claimant_lock:
            if self._claimant is modem and idle:
                self.contact.close()
                self._claimant = None

    def connected(self, caller, called, bw):
        pass

    def log(self, message):
        self.messages.append(message)
        log.debug('%s', message)

    def next_rx_burst(self, timeout, hs=None):
        deadline = time.monotonic()+timeout
        while not self.stop.is_set():
            self.before_step()
            if self.fault.is_set():
                self.fault.clear()
                self._reset_rx(int(self.link.station.audio.samples))
                raise ConnectionAbortedError('VARA station capture discontinuity/overflow')
            if self._bursts:
                return self._bursts.popleft()
            left = deadline-time.monotonic()
            if left <= 0:
                return None
            try:
                generation, at, samples = self.frames.get(timeout=min(left, .05))
            except queue.Empty:
                return None
            if hs is None or generation != self._rx_generation:
                continue
            skip = max(0, self._skip_before-at)
            samples = samples[skip:]
            if not len(samples):
                continue
            hs.on_rx_stream(samples)
            if generation != self._rx_generation:
                return None  # Stream recognition transmitted; retire its old bracket.
            for _, burst in self._segmenter.push(samples):
                if len(self._bursts) >= 8:
                    self.fault.set()
                    raise ConnectionAbortedError('VARA station burst queue full')
                self._bursts.append(burst)
            activity = getattr(hs.io, 'channel_activity', None) if hasattr(hs, 'io') else None
            if activity is not None:
                activity(self.receiving)
        return None


class StationVaraRuntime:
    def __init__(self, link):
        self.transport = StationTransport(link)
        self.owner = AudioHostRuntime(self.transport)
        self._thread = None
        self.failure = None

    @property
    def active(self):
        modem = self.owner.modem
        return bool(modem is not None and not modem._stopping and
                    (modem._intent_active or modem.station.state in
                     (VaraState.CONNECTING, VaraState.CONNECTED, VaraState.DISCONNECTING)))

    def modem_factory(self):
        if self.transport.stop.is_set():
            raise ConnectionAbortedError('VARA station owner stopped')
        return self.owner.modem_factory()

    def start(self):
        if self._thread is not None:
            raise RuntimeError('VARA station owner already started')
        self._thread = threading.Thread(target=self._run, name='station-vara-owner', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self.transport.stop.is_set():
                try:
                    self.owner.step(timeout=.05)
                except Exception as exc:
                    self.failure = str(exc)
                    modem = self.owner.modem
                    if modem is not None:
                        observer, was_connected = modem.observer, modem.connected
                        if observer is not None:
                            observer.modem_warning(f'VARA station owner: {exc}')
                        if not modem._stopping:
                            try:
                                modem.abort()
                            except OSError:
                                modem.stop()  # Even a full command queue must retire.
                        modem.poll()
                        if observer is not None and not was_connected:
                            observer.modem_disconnected()
                    self.transport.hs = None
                finally:
                    modem = self.owner.modem
                    if modem is not None:
                        with modem._lock:
                            self.transport.retire_contact(modem, idle=not self.active)
        finally:
            self._retire()

    def _retire(self):
        modem = self.owner.modem
        if modem is not None:
            modem.stop()
            modem.poll()
        while not self.owner._new.empty():
            pending = self.owner._new.get_nowait()
            pending.stop()
            pending.poll()
        while not self.transport.frames.empty():
            self.transport.frames.get_nowait()
        self.transport._bursts.clear()
        self.transport._segmenter = None
        self.transport.contact.close()

    def stop(self):
        self.transport.stop.set()
        if self._thread is not None:
            self._thread.join(30)
            if self._thread.is_alive():
                raise RuntimeError('VARA station owner still emitting/retiring')
        else:
            self._retire()
