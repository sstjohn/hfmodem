# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA station transport: indexed capture, one owner, no device or PTT owner."""
from collections import deque
import logging
import queue
import threading
import time

import numpy as np

from hfmodem.core import gil
from hfmodem.core.recording import inherit_boundaries
from hfmodem.core.waits import join, wait_until

from .audio_runtime import AudioHostRuntime
from .ui_channel import UIChannel
from .kiss_tx import KissTxDisposition as TxDisposition
from ..monitor import Segmenter
from ..vara.vara_arq import VaraIO, VaraState

log = logging.getLogger(__name__)
RATE = 48000
CHUNK = 4800
# The standalone transport discards 0.1 s past transmission, bounded by the
# earliest measured peer answer. This station seam still needs physical timing
# qualification; it does not infer the hardware's mute recovery from software.
ECHO_GUARD = 4800
# A receive backlog past this is logged, and again when it clears. Only the lag
# past it ages VARA's frames: the freshness limits were proven with a healthy
# host's lag inside them.
LAG_NOTICE = RATE // 10


class StationTransport(VaraIO):
    station_ptt = True
    def __init__(self, link):
        self.link = link
        from hfmodem.station.contact import ContactOwner
        self.contact = ContactOwner(link.station.arbiter, 'vara')
        self._init_receive_state()

    def _init_receive_state(self):
        self._channel_boundary = self.contact.receive_boundary
        self._claimant = None
        self._claimant_lock = threading.Lock()
        self.frames = queue.Queue(maxsize=128)
        self.fault = threading.Event()
        self.stop = threading.Event()
        self._hs = None
        self.tune_epoch = self._decoder_epoch = 0
        self._rx_generation = 0
        self._skip_before = 0
        self.input_clock = 0
        self._rx_end = None
        self._segment_start = None
        self.burst_start = None
        self._segmenter = Segmenter()
        self._ui_channel = UIChannel(self._segmenter, getattr(self, "receive_context", None))
        self._bursts = deque()
        self._metadata = None
        self._sent = False
        self._kiss_keyed = self._physical_keyed = False
        self._freshness = (0, 0)
        self._seen_flush = 0
        self._lagging = False
        self.burst_backlog = 0
        self.messages = deque(maxlen=128)

    def diagnostic(self, kind, *, authority=None, **fields):
        router = getattr(self.link.station, 'diagnostics', None)
        if router is not None:
            router.put(kind, authority=authority, **fields)

    @property
    def hs(self):
        return self._hs

    @hs.setter
    def hs(self, value):
        if value is not self._hs:
            self._decoder_epoch = max(self._decoder_epoch,
                                      getattr(self._hs, '_decoder_epoch', 0)) + 1
            self._hs = value
            if value is not None:
                bind = getattr(value, 'bind_capture', None)
                if bind is not None:
                    bind(self.tune_epoch, self._decoder_epoch)
                value.receive_context = getattr(self, "receive_context", None)
                if value.receive_context is not None:
                    value.cq_detector = value.receive_context.detector

    def bind_receive_context(self, context):
        self.receive_context = context
        self._ui_channel = UIChannel(self._segmenter, context)

    def bind_station(self, value):
        self.hs = value

    def _reset_rx(self, floor, *, advance_generation=True):
        self._skip_before = max(self._skip_before, floor)
        if advance_generation:
            self._rx_generation += 1
        self._segmenter = Segmenter()
        self._ui_channel = UIChannel(self._segmenter, getattr(self, "receive_context", None))
        self._bursts.clear()
        self._segment_start = None
        self._rx_end = floor
        self.burst_start = None
        self.burst_backlog = 0
        self.input_clock = max(self.input_clock, floor)
        context = getattr(self, "receive_context", None)
        if context is not None:
            context.gap(floor)
        gap = getattr(self.hs, 'rx_gap', None)
        if gap is not None:
            gap(floor)

    def retuned(self, *, floor=None):
        boundary = self.contact.receive_boundary
        self.tune_epoch += 1
        self._reset_rx(int(self.link.station.audio.samples) if floor is None else floor)
        context = getattr(self, "receive_context", None)
        if context is not None:
            context.retune(self.tune_epoch, self.input_clock)
        retune = getattr(self.hs, 'retuned', None)
        if retune is not None:
            retune(tune_epoch=self.tune_epoch)
        elif self.hs is not None:
            self.hs._reset_stream()
        self._channel_boundary = boundary

    def _receive_frontier(self, hs):
        report = getattr(hs, 'set_receive_frontier', None)
        if report is not None:
            from ..arq.phy import RECEIVE_BACKWARD_REACH
            oldest = (self._segment_start + self._segmenter.oldest_retained_sample
                      if self._segment_start is not None else self.input_clock)
            starts = [oldest, *(at for at, _ in self._bursts)]
            if self.burst_start is not None:
                starts.append(self.burst_start)
            with self.frames.mutex:
                starts.extend(max(at, self._skip_before) for generation, at, _ in self.frames.queue
                              if generation == self._rx_generation)
            report('segmented', min(starts) - RECEIVE_BACKWARD_REACH)

    @property
    def receive_epoch(self):
        return self._rx_generation, self._freshness[0], self.contact.receive_boundary[0]

    def flushed(self, at):
        serial, floor = self._freshness
        self._freshness = serial+1, max(floor, at)

    def before_step(self):
        boundary = self.contact.receive_boundary
        if boundary != self._channel_boundary:
            self.retuned(floor=max(boundary[1], int(self.link.station.audio.samples)))
        serial, floor = self._freshness
        if serial != self._seen_flush:
            self._reset_rx(floor, advance_generation=False)
            if self.hs is not None:
                self.hs._reset_stream()
            self._seen_flush = serial

    @property
    def receiving(self):
        return self._segmenter is not None and bool(self._segmenter.inb)

    def channel_active(self, bw):
        return self._ui_channel.active

    @property
    def kiss_receiving(self):
        return self._ui_channel.active

    def feed(self, at, samples):
        if self.stop.is_set():
            return
        self.input_clock = max(self.input_clock, at + len(samples))
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

    def tx(self, samples, *, marks=None):
        self._sent = False
        self._kiss_keyed = False
        if (self.stop.is_set() or self.fault.is_set()
                or self._seen_flush != self._freshness[0]):
            return
        metadata, self._metadata = self._metadata, None
        if metadata is None or str(metadata.get('bandwidth_hz')) not in ('500', '2300', '2750'):
            raise ValueError('VARA burst lacks a qualified bandwidth declaration')
        self._reset_rx(int(self.link.station.audio.samples))
        # Only a host-owned handshake carries chat authority; a bare VaraIO has none.
        modem = getattr(self.hs.io, 'modem', None) if self.hs is not None else None
        self.link._transmitting = True
        try:
            playback = {'marks': marks} if marks is not None else {}
            self._sent = self.link.emit(samples, str(metadata['bandwidth_hz']),
                                       **playback,
                                       on_ptt=self._report_ptt,
                                       authority=getattr(modem, 'authority', None),
                                       fence=getattr(modem, 'control_fence', None),
                                       responding=self.hs is not None and
                                       (self.hs.role == 'responder' or
                                        self.hs.state in (VaraState.CONNECTED, VaraState.DISCONNECTING)))
        finally:
            self._skip_before = max(self._skip_before,
                int(self.link.station.audio.samples)+ECHO_GUARD)
            self.link._transmitting = False

    def _report_ptt(self, on):
        self._physical_keyed = bool(on)
        self._kiss_keyed |= bool(on)
        if self.hs is not None:
            self.hs.io.report_ptt(on)

    def unkey_confirmed(self):
        rig = getattr(self.link.station, 'rig', None)
        return (not self._physical_keyed and not getattr(rig, 'retired', False)
                and getattr(rig, 'line_state', 'low') == 'low')

    def admit_kiss(self, modem):
        arbiter = self.link.station.arbiter
        if (self.stop.is_set() or self.fault.is_set() or arbiter is None
                or 'vara' in getattr(arbiter, 'disabled', ())
                or getattr(getattr(self.link.station, 'rig', None), 'retired', False)):
            return TxDisposition.REFUSED_BEFORE_KEY
        if (getattr(getattr(self.link.station, 'rig', None), 'keyed', False)
                or not self.admit_contact(modem)):
            return TxDisposition.CONTENDED
        return None

    def kiss_tx_result(self):
        if self._sent:
            return TxDisposition.SENT
        return (TxDisposition.PARTIAL if self._kiss_keyed
                else TxDisposition.REFUSED_BEFORE_KEY)

    def tx_went_out(self):
        return self._sent

    def tx_playback_end(self):
        """Monotonic last-sample time for the successful burst just emitted."""
        return self.link.playback_ended_at if self._sent else None

    def pending(self):
        pass

    def admit_contact(self, modem):
        with self._claimant_lock:
            if self.stop.is_set() or self._claimant not in (None, modem):
                return False
            if not self.contact.admit(receive_epoch=self._channel_boundary[0],
                                      authority=getattr(modem, 'authority', None),
                                      fence=getattr(modem, 'control_fence', None)):
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

    def delivery_lag(self, end):
        """How many samples capture has run past ``end``: the queueing between
        the card and the handshake, including the converter's input latency."""
        now = end if self._rx_end is None else max(end, self._rx_end)
        audio = self.link.station.audio
        # A replay counts what it pumped rather than time, and a WASAPI anchor
        # only estimates capture time; neither may age a frame.
        limitation = getattr(audio, 'timestamp_limitation', 'replay')
        if limitation is None:
            now = max(now, audio.sample_now())
        return max(0, int(now) - end)

    def _backlog(self, end):
        lag = self.delivery_lag(end)
        return lag, max(0, lag - LAG_NOTICE)

    def _note_lag(self, lag):
        lagging = lag > LAG_NOTICE
        if lagging != self._lagging:
            self._lagging = lagging
            self.log(f"receive backlog {lag / RATE:.3f} s" if lagging
                     else f"receive backlog cleared, {lag / RATE:.3f} s")

    def next_rx_burst(self, timeout, hs=None):
        self.burst_start = None
        self.burst_backlog = 0
        self._receive_frontier(hs)
        fed = 0
        deadline = time.monotonic()+timeout
        while not self.stop.is_set():
            self.before_step()
            if self.fault.is_set():
                self.fault.clear()
                self._reset_rx(int(self.link.station.audio.samples))
                raise ConnectionAbortedError('VARA station capture discontinuity/overflow')
            if self._bursts:
                self.burst_start, burst = self._bursts.popleft()
                # The stream's backlog at handover, not the bracket's: the
                # hangover behind every bracket is in the limits as proven.
                self.burst_backlog = self._backlog(self._rx_end)[1]
                self._receive_frontier(hs)
                return burst
            left = deadline-time.monotonic()
            # Indexed consumers can resume admission on the next owner step;
            # legacy stream callbacks still drain the queued capture.
            if left <= 0 or (fed >= CHUNK and hasattr(hs, 'rx_gap')
                            and getattr(self, 'receive_context', None) is not None):
                return None
            try:
                generation, at, samples = self.frames.get(timeout=min(left, .05))
            except queue.Empty:
                return None
            if hs is None or generation != self._rx_generation:
                continue
            skip = max(0, self._skip_before-at,
                       (self._rx_end - at) if self._rx_end is not None else 0)
            samples = samples[skip:]
            if not len(samples):
                continue
            at += skip
            fed += len(samples)
            if self._rx_end is not None and at > self._rx_end:
                self._segmenter = Segmenter()
                self._ui_channel = UIChannel(self._segmenter, getattr(self, "receive_context", None))
                self._bursts.clear()
                self._segment_start = None
                gap = getattr(hs, 'rx_gap', None)
                if gap is not None:
                    gap(at)
            if self._segment_start is None:
                self._segment_start = at
            self._rx_end = at + len(samples)
            self._receive_frontier(hs)
            context = getattr(self, "receive_context", None)
            if context is not None:
                self._segmenter.capture_start = self._segment_start
                context.feed(samples, at, sink=self.cq_sink)
            if hasattr(hs, 'rx_gap'):
                lag, backlog = self._backlog(at + len(samples))
                self._note_lag(lag)
                hs.on_rx_stream(samples, sample_start=at, backlog=backlog)
            else:
                hs.on_rx_stream(samples)
            if generation != self._rx_generation:
                return None  # Stream recognition transmitted; retire its old bracket.
            for start, burst in self._ui_channel.push(samples, sink=self.cq_sink,
                    ui=True):
                if len(self._bursts) >= 8:
                    self.fault.set()
                    raise ConnectionAbortedError('VARA station burst queue full')
                self._bursts.append((self._segment_start + start, burst))
            self._receive_frontier(hs)
            activity = getattr(hs.io, 'channel_activity', None) if hasattr(hs, 'io') else None
            if activity is not None:
                activity(self.channel_active(None))
            if getattr(hs, 'final_request_due_in', lambda: None)() is not None:
                return None
        return None


class StationVaraRuntime:
    def __init__(self, link):
        from hfmodem.station.chat import AttachmentGate
        self.attachments = AttachmentGate()
        inherit_boundaries(self, link.station)
        self.transport = inherit_boundaries(StationTransport(link), self)
        self.owner = AudioHostRuntime(self.transport)
        self._thread = None
        self.failure = None
        self._calls = queue.Queue(maxsize=1)
        self._paused = False

    @property
    def paused(self):
        return self._paused

    def retune(self, callback, valid, *, timeout=10):
        """Run bounded CAT work at an audio-owner boundary, never concurrently."""
        if self.transport.stop.is_set() or self._thread is None or not self._thread.is_alive():
            raise ConnectionAbortedError('VARA station owner is not running')
        call = dict(callback=callback, valid=valid, lock=threading.Lock(),
                    done=threading.Event(), cancelled=False, result=None, error=None)
        try:
            self._calls.put_nowait(call)
        except queue.Full as error:
            raise TimeoutError('VARA owner retune queue is full') from error
        if not wait_until(call['done'].wait, time.monotonic() + timeout):
            # The callback owns this lock through completion. A running CAT
            # operation cannot be reported as cancelled while it still moves.
            if call['lock'].acquire(blocking=False):
                try:
                    if not call['done'].is_set():
                        call['cancelled'] = True
                        raise TimeoutError('VARA owner retune timed out before execution')
                finally:
                    call['lock'].release()
            else:
                raise TimeoutError('VARA owner retune remains in progress')
        if call['error'] is not None:
            raise call['error']
        return call['result']

    def _service_calls(self):
        try:
            call = self._calls.get_nowait()
        except queue.Empty:
            return
        with call['lock']:
            if call['cancelled']:
                call['done'].set()
                return
            try:
                if self.transport.stop.is_set() or not call['valid']():
                    raise ConnectionAbortedError('VARA retune authority changed')
                self._paused = True
                call['result'] = call['callback']()
            except BaseException as error:
                call['error'] = error
            finally:
                self._paused = False
                call['done'].set()

    @property
    def clock(self):
        return self.owner.clock

    def bind_kiss(self, kiss, tx_queue):
        self.owner.bind_kiss(kiss, tx_queue)

    @property
    def active(self):
        modem = self.owner.modem
        return bool(modem is not None and not modem._stopping and
                    (modem._intent_active or modem.station.state in
                     (VaraState.CONNECTING, VaraState.CONNECTED, VaraState.DISCONNECTING)))

    def modem_factory(self, *, authority=None, fence=None):
        if self.transport.stop.is_set():
            raise ConnectionAbortedError('VARA station owner stopped')
        return self.owner.modem_factory(authority=authority, fence=fence)

    def start(self):
        if self._thread is not None:
            raise RuntimeError('VARA station owner already started')
        self._thread = threading.Thread(target=self._run, name='station-vara-owner', daemon=True)
        self._thread.start()

    @gil.joined()
    def _run(self):
        try:
            while not self.transport.stop.is_set():
                try:
                    self._service_calls()
                    self.owner.step(timeout=.05)
                except Exception as exc:
                    self.failure = str(exc)
                    modem = self.owner.modem
                    if modem is not None:
                        self.transport.diagnostic('runtime_failure',
                            authority=getattr(modem, 'authority', None), text=str(exc))
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
        while not self._calls.empty():
            call = self._calls.get_nowait()
            with call['lock']:
                call['error'] = ConnectionAbortedError('VARA station owner stopped')
                call['done'].set()
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
            if not join(self._thread, time.monotonic() + 30):
                raise RuntimeError('VARA station owner still emitting/retiring')
        else:
            self._retire()
