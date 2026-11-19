# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Chat-owned recording and monitor fanout from a borrowed station capture.

Only the capture callback queues copies. The controller owner calls ``drain``
to schedule fanout on the sole recording worker. Disk drains, contact cutouts
and final storage must retire before chat admission can be released.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
import threading
import time

import numpy as np

from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.waits import join, result


@dataclass(frozen=True)
class CaptureGap:
    at: int
    end: int | None
    cause: str


@dataclass(frozen=True)
class _Chunk:
    at: int
    samples: np.ndarray
    keyed: bool


class CaptureStartupCleanup:
    """Retain partially constructed consumers until their closes have retired."""

    def __init__(self, *resources):
        self.resources = tuple(resource for resource in resources if resource is not None)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chat-capture-startup')
        self._future = None

    def _finish(self):
        failures = []
        for resource in self.resources:
            try:
                resource.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError('; '.join(map(str, failures))) from failures[0]

    def close(self, *, timeout=5, deadline=None):
        end = time.monotonic() + max(0, timeout)
        if deadline is not None:
            end = min(end, deadline)
        if self._future is None:
            self._future = self._worker.submit(self._finish)
            self._worker.shutdown(wait=False)
        for worker in tuple(self._worker._threads):
            if not join(worker, end):
                raise RuntimeError('chat capture startup cleanup still pending')
        self._future.result()


class ChatCapture:
    """Own an injected CaptureRecorder and optional SlotWatch, never the audio.

    ``keyed`` is a nonblocking observation of this station's physical key state;
    unknown key state should return true to mask passive monitoring conservatively.
    The recorder receives original samples, including keyed intervals. ``origin``
    maps station card indices to sample offsets in this chat's recording.
    Construction takes ownership of recorder/monitor after successful subscribe.
    """

    def __init__(self, audio, recorder, monitor=None, *, keyed=lambda: False,
                 capacity_samples=10 * CARD_RATE_HZ, max_chunks=256,
                 contact_log=None):
        if capacity_samples <= 0 or max_chunks <= 0:
            raise ValueError('capture queue bounds must be positive')
        self.audio, self.recorder, self.monitor = audio, recorder, monitor
        self.contact_log = contact_log
        self.origin = int(audio.samples)
        self.received_through = self.flushed_through = self.origin
        self.rate = CARD_RATE_HZ
        self._keyed = keyed
        self._capacity, self._max_chunks = capacity_samples, max_chunks
        self._queued_samples = 0
        self._buffered_chunks = 0
        self._queue = deque()
        self._lock = threading.Lock()
        self._closing = self._detached = self._closed = False
        self._failure = None
        self.gaps = []
        self._saves = []
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chat-capture')
        self._finalizer = None
        self._drain_future = None
        self.recorder.facts.update(station_origin_sample=self.origin)
        try:
            audio.subscribe(self, removable=True)
        except BaseException:
            self._worker.shutdown(wait=False, cancel_futures=True)
            raise

    @property
    def position(self):
        """File offset corresponding to the last accepted capture callback."""
        return self.received_through - self.origin

    def file_offset(self, card_index):
        if card_index < self.origin:
            raise ValueError('capture index predates chat recording')
        return int(card_index) - self.origin

    def _fail(self, message):
        if self._failure is None:
            self._failure = RuntimeError(message)

    def push(self, block):
        """Audio callback: bounded copied samples, no file writes or consumers."""
        with self._lock:
            if self._detached or self._failure is not None:
                return
            skip = max(0, self.origin - block.n0)
            samples = block.samples[skip:]
            if not len(samples):
                return
            if (self._queued_samples + len(samples) > self._capacity
                    or self._buffered_chunks >= self._max_chunks):
                self._fail('chat capture queue overflow; recording is incomplete')
                return
            try:
                chunk = _Chunk(block.n0 + skip,
                               np.array(samples, dtype=np.float32, copy=True), bool(self._keyed()))
            except Exception as error:
                self._fail(f'chat capture callback failed: {error}')
                return
            self._queue.append(chunk)
            self._buffered_chunks += 1
            self._queued_samples += len(samples)
            self.received_through = chunk.at + len(samples)

    def fault(self, card_index, cause):
        with self._lock:
            if self._detached or self._failure is not None:
                return
            if self._buffered_chunks >= self._max_chunks:
                self._fail('chat capture gap queue overflow; recording is incomplete')
                return
            self._queue.append(CaptureGap(int(card_index), None, str(cause)))
            self._buffered_chunks += 1

    def resync(self, card_index):
        self.fault(card_index, 'capture resync')

    def flush_to(self, card_index):
        # Audio's receive-lane flush must never cut our keyed recording samples.
        pass

    def _gap(self, gap):
        if len(self.gaps) >= 1024:
            raise RuntimeError('chat capture gap evidence overflow')
        self.gaps.append(gap)
        if self.monitor is not None:
            # SlotWatch's existing queued gap protocol resets decoder continuity.
            self.monitor.put(('gap', 0 if gap.end is None else gap.end - gap.at))

    def drain(self):
        """Owner-only nonblocking scheduling, bounded across queued/active audio."""
        if self._finalizer is not None:
            raise RuntimeError('chat capture finalization already started')
        self._check_saves()
        if self._drain_future is not None:
            if not self._drain_future.done():
                return self._drain_future
            self._drain_future.result()
        with self._lock:
            queued, self._queue = self._queue, deque()
        if queued:
            self._drain_future = self._worker.submit(self._process_chunks, queued)
        elif self._failure is not None:
            raise self._failure
        return self._drain_future

    def _process_chunks(self, queued):
        try:
            for item in queued:
                if isinstance(item, CaptureGap):
                    self._gap(item)
                    continue
                if item.at < self.flushed_through:
                    raise RuntimeError('chat capture indices overlap or regress')
                if item.at > self.flushed_through:
                    gap = CaptureGap(self.flushed_through, item.at, 'capture index gap')
                    self._gap(gap)
                    # Preserve card-to-file offsets. Pad in bounded pieces and
                    # identify every inserted interval in recording provenance.
                    remaining = gap.end - gap.at
                    if remaining > self._capacity:
                        raise RuntimeError('capture index gap exceeds padding bound')
                    while remaining:
                        n = min(remaining, CARD_RATE_HZ)
                        self.recorder.push(np.zeros(n, np.float32))
                        self.recorder.drain()
                        remaining -= n
                self.recorder.push(item.samples)
                self.recorder.drain()
                self.flushed_through = item.at + len(item.samples)
                if self.monitor is not None:
                    self.monitor.put(np.zeros_like(item.samples) if item.keyed else item.samples.copy())
        except Exception as error:
            self._fail(f'chat capture drain failed: {error}')
        finally:
            with self._lock:
                self._queued_samples -= sum(len(item.samples) for item in queued if isinstance(item, _Chunk))
                self._buffered_chunks -= len(queued)
        if self._failure is not None:
            raise self._failure
        return self.flushed_through

    def flush_through(self, card_index, *, timeout=5, deadline=None):
        """Flush accepted chunks through a requested station sample boundary."""
        end = time.monotonic() + max(0, timeout)
        if deadline is not None:
            end = min(end, deadline)
        # At most one active batch and one callback queue. Do not queue arbitrary
        # disk work while a previous batch is blocked in a filesystem call.
        while True:
            future = self.drain()
            if future is not None:
                try:
                    result(future, end)
                except FutureTimeout as error:
                    raise RuntimeError('chat capture flush still pending') from error
            with self._lock:
                queued = bool(self._queue)
            if self.flushed_through >= card_index or not queued:
                break
        if self.flushed_through < card_index:
            raise RuntimeError('chat capture ending samples still pending')
        return self.file_offset(card_index)

    def contact(self, record, *, card_index=None, offset_hz=0, timeout=5, deadline=None):
        """Update contact provenance; finish only after its ending samples flush."""
        if self._closing:
            raise RuntimeError('chat capture is closing')
        if self.contact_log is None:
            raise ValueError('no ContactLog associated with chat capture')
        at = self.received_through if card_index is None else int(card_index)
        native = self.contact_log.update(record, sample=self.file_offset(at),
                                         sample_rate=self.rate, offset_hz=offset_hz)
        if record.get('ended_utc'):
            self._check_saves()
            if len(self._saves) >= 128:
                self._fail('chat capture contact-save queue overflow')
                raise self._failure
            self.flush_through(at, timeout=timeout, deadline=deadline)
            self._saves.append(self._worker.submit(self.contact_log.finish, native))
        return native

    def _check_saves(self):
        pending = []
        for future in self._saves:
            if future.done():
                try:
                    future.result()
                except Exception as error:
                    self._fail(f'chat capture save failed: {error}')
                    raise self._failure from error
            else:
                pending.append(future)
        self._saves = pending

    def _finalize(self):
        self.recorder.facts.update(
            station_origin_sample=self.origin,
            station_end_sample=self.flushed_through,
            capture_gaps=[dict(at=gap.at, end=gap.end, cause=gap.cause) for gap in self.gaps],
            chat_capture_incomplete=self._failure is not None)
        try:
            if self.monitor is not None:
                self.monitor.close()
        finally:
            self.recorder.close()

    def close(self, *, timeout=5, deadline=None):
        """Detach, flush, finish saves and join storage within one bounded wait.

        Timeout retains this object and its tracked worker for a later retry.
        A failed recording/save remains a teardown failure even after retirement.
        """
        end = time.monotonic() + max(0, timeout)
        if deadline is not None:
            end = min(end, deadline)
        self._closing = True
        if not self._detached:
            self.audio.unsubscribe(self, timeout=max(0, end - time.monotonic()))
            self._detached = True
        if self._finalizer is None:
            try:
                self.flush_through(self.received_through, deadline=end)
            except RuntimeError:
                # A pending disk write is still live: do not schedule closure
                # over it or release admission. A real fault is sticky and may
                # proceed to finalization once that worker call has returned.
                if self._drain_future is not None and not self._drain_future.done():
                    raise
                if self._failure is None:
                    raise
            self._finalizer = self._worker.submit(self._finalize)
            self._worker.shutdown(wait=False)
        for worker in tuple(self._worker._threads):
            if not join(worker, end):
                raise RuntimeError('chat capture worker retirement still pending')
        try:
            self._check_saves()
            self._finalizer.result()
        except Exception as error:
            self._fail(f'chat capture save failed: {error}')
        self._closed = True
        if self._failure is not None:
            raise self._failure
