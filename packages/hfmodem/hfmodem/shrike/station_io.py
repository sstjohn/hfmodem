# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PACTOR's indexed I/O port over the station's existing capture and arbiter.

There is no device, keyer, decoder, ARQ timer or worker in this adapter. The
station feeds StreamChunks; the contact owner reads them using the existing
onair driver's sample indices. Closing the port retires only this contact.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import math
import queue
import threading
import time

import numpy as np

from hfmodem.core.audio import StreamChunk
from hfmodem.core import cwid
from hfmodem.core.regulatory import centred
from hfmodem.station.arbiter import LIVE, TxRequest

FS = 48000
CHUNK = 4800


class _CaptureFrames(queue.Queue):
    """Bound queued PCM without charging every short card callback a full slot.

    Merge only consecutive, already queued samples. A waiting reader still gets
    each feed immediately; there is no batching delay in a receive aperture.
    """
    def put_nowait(self, frame):
        with self.not_full:
            if self.queue:
                previous = self.queue[-1]
                if (previous.samples.size and frame.samples.size
                        and previous.at + len(previous.samples) == frame.at
                        and len(previous.samples) + len(frame.samples) <= CHUNK):
                    self.queue[-1] = StreamChunk(previous.at,
                        np.concatenate((previous.samples, frame.samples)),
                        previous.faults + frame.faults, ())
                    return
            if self._qsize() >= self.maxsize:
                raise queue.Full
            self._put(frame)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class StationInput:
    """One contact's passive sample-clock I/O; feed may run on the station thread.

    The first read/transmit/flush binds the contact owner. All subsequent
    consumer operations, including close, require that same thread object.
    Capture faults and overflow permanently retire the port. Only an exact
    flush from this port's own completed transmission is authorized.
    """

    fs = FS

    def __init__(self, link, *, depth=512, stall_timeout=2.0, responding=lambda: False,
                 owner_poll=None, scannable=lambda: False):
        if (not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0
                or not math.isfinite(stall_timeout) or stall_timeout <= 0
                or not callable(responding)
                or not callable(scannable)
                or owner_poll is not None and not callable(owner_poll)):
            raise ValueError('positive capture depth and finite stall timeout required')
        if link.waveform.rate != FS:
            raise ValueError('PACTOR station input requires native 48000 Hz')
        self.link = link
        self.audio = link.station.audio
        arbiter = link.station.arbiter
        history = arbiter.transmit_history() if arbiter is not None else None
        self.pos = int(self.audio.samples)
        if history is not None and history.capture_floor > self.pos:
            raise ValueError('PACTOR capture epoch must wait past retired station output')
        self._epoch_floor = self.pos
        self.channel_epoch = arbiter.receive_boundary[0] if arbiter is not None else 0
        self._birth_tx = history
        self._birth_fault_serial = getattr(getattr(link, 'lane', None), 'fault_serial', 0)
        self._next = self.pos
        self._skip_before = self.pos
        self._completed_tx_end = None
        self._frames = _CaptureFrames(maxsize=depth)
        self._stop = threading.Event()
        self._overflow = threading.Event()
        self._feed_lock = threading.Lock()
        self._owner = None
        self._rest = np.zeros(0, np.float32)
        self._flushes = deque(maxlen=128)
        self._fault = None
        self.stall_timeout = stall_timeout
        self.responding = responding
        self.scannable = scannable
        self._scan_token = None
        self.scan_epoch = 0
        self.capture_retired = False
        self.owner_poll = owner_poll
        self._polling = False
        self._identifying = None
        self.keyed_at = None
        self.ever_keyed = False
        self.keyed_s = 0.0
        self.corrected_emission_start = self.corrected_emission_end = None
        self.output_enqueue_sample = self.driver_offset_samples = None
        self.lost = self.xruns = self.underruns = 0
        self.lost_at = []
        self.xrun_at = None

    @property
    def closed(self):
        return self._stop.is_set()

    @property
    def samples(self):
        return int(self.audio.samples)

    @property
    def holdback(self):
        return self.audio.holdback

    @property
    def _blk(self):
        return self.audio.blocksize

    @property
    def _lat(self):
        return self.audio.lat

    @property
    def tx_latency_n(self):
        return self.audio.tx_latency_n

    @property
    def key_notice(self):
        return self.audio.loop_lat + 3 * self.audio.blocksize

    def sample_now(self):
        return self.audio.sample_now()

    def begin_scan_hold(self):
        # This can run inside serial owner polling. It neither consumes capture
        # nor submits a key, and must not recursively service the endpoint.
        self._check_owner()
        if self._scan_token is not None:
            return True
        if self.closed or self._fault is not None or self._overflow.is_set():
            return False
        arbiter = self.link.station.arbiter
        if arbiter is None:
            return False
        token = arbiter.acquire_scan_hold('pactor', eligible=self.scannable)
        if token is None:
            return False
        self._scan_token = token
        self.scan_epoch += 1
        self.capture_retired = True
        return True

    def end_scan_hold(self):
        self._check_owner()
        # Release is also required while unwinding a closed/faulted port.
        if self._scan_token is not None:
            self.link.station.arbiter.release_scan_hold(self._scan_token)
            self._scan_token = None
            self.scan_epoch += 1

    def clamp_late(self, at):
        return max(0, self.samples + self.key_notice - at)

    def _check_owner(self):
        if self._owner is None:
            self._owner = threading.current_thread()
        elif self._owner is not threading.current_thread():
            raise RuntimeError('PACTOR station I/O must run on its contact owner')

    def _check(self, *, service=True):
        self._check_owner()
        if self._polling:
            raise RuntimeError('PACTOR owner I/O poll cannot consume capture or transmit')
        self._check_state()
        if service and self.owner_poll is not None:
            self._polling = True
            try:
                arbiter = self.link.station.arbiter
                if arbiter is None:
                    self.owner_poll()
                else:
                    arbiter.run_owner_poll(self.owner_poll)
            except BaseException:
                self._fault = 'PACTOR owner I/O failed'
                self._stop.set()
                raise
            finally:
                self._polling = False
            self._check_state()

    def _check_state(self):
        arbiter = self.link.station.arbiter
        if arbiter is not None and arbiter.receive_boundary[0] != self.channel_epoch:
            self.capture_retired = True
        if self._overflow.is_set():
            self._fail('station capture queue overflow')
        if self._fault is not None:
            raise ConnectionAbortedError(self._fault)
        if self.closed:
            raise ConnectionAbortedError('PACTOR station input closed')
        if self.capture_retired:
            raise ConnectionAbortedError('PACTOR capture retired by scanner hold')

    def _fail(self, reason):
        self._fault = reason
        self.close()
        raise ConnectionAbortedError(reason)

    def feed(self, chunk: StreamChunk):
        """Copy bounded fresh capture from the station; never touch the host."""
        with self._feed_lock:
            if self.closed or self._overflow.is_set() or self.capture_retired:
                return
            # A freshly published contact port may be handed older blocks
            # already waiting in the station's lane. They are outside this
            # port's epoch, including faults strictly before its birth floor.
            # Current-boundary faults remain fatal; no old waveform or decoder
            # evidence is spliced into the new contact.
            faults = tuple(f for f in chunk.faults if f.index >= self._epoch_floor
                and not (f.cause == 'flush' and self._birth_tx is not None
                         and self._birth_tx.sequence > 0
                         and f.index == self._birth_tx.end == self._epoch_floor
                         and f.serial <= self._birth_fault_serial))
            if chunk.at + len(chunk.samples) <= self._epoch_floor and not faults:
                return
            first = min(len(chunk.samples), max(0, self._epoch_floor - chunk.at))
            for offset in range(first, max(first + 1, len(chunk.samples)), CHUNK):
                samples = np.array(chunk.samples[offset:offset + CHUNK], np.float32, copy=True)
                part = StreamChunk(chunk.at + offset, samples,
                                   faults if offset == first else (), ())
                try:
                    self._frames.put_nowait(part)
                except queue.Full:
                    self._overflow.set()
                    return

    def _capture(self, timeout):
        self._check()
        try:
            chunk = self._frames.get(timeout=timeout)
        except queue.Empty:
            return False
        for fault in chunk.faults:
            if fault.cause == 'flush' and fault.index in self._flushes:
                self._flushes.remove(fault.index)
            else:
                self._fail(f'station capture {fault.cause} at sample {fault.index}')
        end = chunk.at + len(chunk.samples)
        if end <= self._skip_before:
            # Queued pre-transmit capture is older than our authorized floor.
            return True
        if chunk.at > self._next:
            self._fail(f'station capture index gap: {self._next}..{chunk.at}')
        if chunk.at < self._next and self._next != self._skip_before:
            self._fail(f'station capture overlap at sample {chunk.at}')
        skip = self._next - chunk.at
        self._rest = chunk.samples[skip:]
        self._next = end
        return True

    def read(self, count):
        """Read consecutive samples without filling a missing interval with zeros."""
        self._check()
        count = max(0, int(count))
        parts = []
        deadline = time.monotonic() + self.stall_timeout
        while count:
            self._check()
            if not self._rest.size:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._fail('station capture stopped')
                if self._capture(min(left, .05)):
                    deadline = time.monotonic() + self.stall_timeout
                continue
            take = min(count, len(self._rest))
            parts.append(self._rest[:take])
            self._rest = self._rest[take:]
            self.pos += take
            count -= take
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def take(self, seconds):
        return self.read(round(seconds * FS))

    def take_until(self, index):
        return self.read(max(0, int(index) - self.pos))

    def read_ready(self):
        self._check()
        parts = []
        queued = self._frames.qsize()
        for _ in range(queued + 1):
            if self._rest.size:
                parts.append(self._rest)
                self.pos += len(self._rest)
                self._rest = np.zeros(0, np.float32)
            if not self._capture(0):
                break
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def wait_until(self, index):
        self._check()
        began = time.monotonic()
        self.audio.wait_until(index,
            deadline=began + max(0, index - self.sample_now()) / FS + self.stall_timeout,
            cancelled=self._wait_cancelled if self.owner_poll is not None else self._stop.is_set)
        self._check()
        return (time.monotonic() - began) * 1000

    def _wait_cancelled(self):
        self._check()
        return self.closed

    def flush_to(self, index):
        self._check()
        index = int(index)
        if index < self.pos:
            raise ValueError('PACTOR capture cannot move backwards')
        if index > self.pos and index != self._completed_tx_end:
            raise ValueError('PACTOR capture flush requires its completed transmission')
        # Preserve a delivered suffix if the floor falls inside that chunk.
        rest_at = self._next - len(self._rest)
        self._rest = self._rest[max(0, index - rest_at):]
        self.pos = index
        self._next = max(self._next, index)
        self._skip_before = max(self._skip_before, index)

    @contextmanager
    def identification_scope(self, call, *, centre_hz):
        """Credit only the native producer's configured CW identity after output.

        The producer still owns the rendered CW, gain and guarded raster slot.
        This scope gives that emission the station's identification request
        type, so an overdue ID can discharge its own interlock.
        """
        self._check()
        arbiter = self.link.station.arbiter
        identity = arbiter.identity if arbiter is not None else None
        if (identity is None or identity.mode != 'cw' or call != identity.mycall
                or not math.isfinite(centre_hz)):
            raise ValueError('PACTOR CW must identify the configured station')
        if self._identifying is not None:
            raise RuntimeError('PACTOR identification scope is already active')
        self._identifying = (call, centre_hz, identity)
        try:
            yield
        finally:
            self._identifying = None

    def transmit(self, audio, *, at=None, settle=0.0, key=None,
                 max_key=40.0, admit=None):
        """Use only the station arbiter, including its actual PTT notifications.

        The driver's key callback is deliberately never called: it does not own
        the station key. Producer admission is rechecked by the arbiter just
        before physical key-up. Output gain survives the arbiter's normalization.
        """
        self._check()
        self.corrected_emission_start = self.corrected_emission_end = None
        samples = np.asarray(audio, np.float32)
        if (samples.ndim != 1 or not samples.size or not np.isfinite(samples).all()
                or not math.isfinite(settle) or settle < 0
                or not math.isfinite(max_key) or max_key <= 0
                or len(samples) / FS > max_key):
            raise ValueError('invalid or overlong PACTOR station transmission')
        station = self.link.station
        if station.arbiter is None or station.rig is None:
            raise ConnectionAbortedError('PACTOR station has no transmit owner')
        peak = float(np.max(np.abs(samples)))
        ceiling = float(station.arbiter.drive)
        if not math.isfinite(ceiling) or ceiling <= 0 or peak > ceiling + 1e-7:
            raise ValueError('PACTOR waveform exceeds the station output ceiling')
        gain = min(1.0, peak / ceiling)
        self.keyed_at = None
        self.keyed_s = 0.0

        def ptt(on):
            now = int(self.sample_now())
            if on:
                self.keyed_at = now
                self.ever_keyed = True
            elif self.keyed_at is not None:
                self.keyed_s = (now - self.keyed_at) / FS

        def current():
            # Final key admission must not run a parser or other owner work.
            self._check(service=False)
            if self._identifying is not None:
                call, _, identity = self._identifying
                if (station.arbiter.identity is not identity or identity.mode != 'cw'
                        or identity.mycall != call):
                    raise ValueError('PACTOR station CW identity changed before keying')
            if admit is not None:
                admit()

        current()
        emission = (self.link.waveform.emission(station.rig.dial(),
                        power_w=station.cfg.declared_power_w)
                    if self._identifying is None else
                    centred(station.rig.dial(), cwid.BANDWIDTH_HZ,
                            centre_hz=self._identifying[1], power_w=station.cfg.declared_power_w))
        request = TxRequest(priority=LIVE, seq=0,
            protocol='pactor' if self._identifying is None else 'ident', audio=samples,
            rate=FS, at=at, settle_s=settle, responding=self.responding(), gain=gain,
            emission=emission,
            on_ptt=ptt, admit=current,
            owner_poll=self._check if self.owner_poll is not None else None)
        end = station.arbiter.submit(request)
        first = end - len(samples)
        self.corrected_emission_start, self.corrected_emission_end = first, end
        self._completed_tx_end = end
        self._flushes.append(end)
        return first, end

    def cancel(self):
        """Wake an owner from another thread without touching its host state."""
        self._stop.set()

    def close(self):
        self._check_owner()
        if self._polling:
            raise RuntimeError('PACTOR owner I/O poll must cancel instead of closing capture')
        self.end_scan_hold()
        with self._feed_lock:
            self._stop.set()
            self._rest = np.zeros(0, np.float32)
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break

    def clock_report(self):
        return f'PACTOR station clock: sample {self.samples}; shared station capture'

    def stream_report(self):
        return f'PACTOR station input: {"closed" if self.closed else "open"}; {self._fault or "no capture fault"}'
