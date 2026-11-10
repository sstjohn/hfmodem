# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host API binding for the VARA HF driver.

The audio owner supplies a VaraIO transport and calls ``poll()`` before/after
feeding ``station.on_rx_stream`` (and its normal idle/retry cadence). Host
threads only enqueue commands: radio and driver state stay on the owner thread.
Use this as VaraServer's modem_factory when embedding that driver. The default
host CLI remains the radio-less loopback server.
"""
from __future__ import annotations

import inspect
import queue
import re
import threading
import time
from collections import deque

from .modem_core import ModemCore
from .kiss_tx import KissTxDisposition as TxDisposition
from ..rx import beacon as B
from ..rx.cq import build_frame
from ..rx.cq_stream import CQDetector
from ..rx.receive_context import ReceiveContext
from ..tx.beacon import synth_beacon
from ..tx.cq import synth_frame
from ..vara.vara_arq import CALLER_UNKNOWN, VaraStationHandshake, VaraState

#: VARA's BUSY (an idle VARA HF station hearing 14 CQFRAMEs): ON
#: 0.02-0.21 s after the far end keys, OFF 1.48-1.75 s after it unkeys, and never for
#: continuous band noise at -5 dB SNR. Our gate closes
#: 0.13 s after a signal, so a 1.45 s hold lands OFF inside that span, and a minimum
#: on-time no later than that ON keeps static crashes out of it.
BUSY_ON_S = .2
BUSY_HOLD_S = 1.45


from hfmodem.core.recording import contact_ended, inherit_boundaries


class _HostIO:
    def __init__(self, modem, transport):
        self.modem = modem
        self.transport = transport
        self._refused = False
        self.cq_sink = self.cq

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def key(self, on):
        if on and self.modem._stopping:
            self._refused = True
            return
        self.modem._keyed_clean = self.modem.station.clean_verdict if on else None
        try:
            self.transport.key(on)
        except BaseException:
            if (not on and self.modem._broadcast_bracket
                    and not getattr(self.transport, 'station_ptt', False)):
                self.report_ptt(False)
            raise
        if not getattr(self.transport, 'station_ptt', False):
            self.report_ptt(on)

    def report_ptt(self, on):
        keyed = on and not getattr(self.transport, '_refused', False)
        if keyed:
            self.modem.key_sequence += 1
        was_keyed = self.modem._ptt_active
        self.modem._ptt_active = bool(keyed)
        if not on and was_keyed:
            self.modem.last_unkey_at = self.modem._clock()
            self.modem.unkey_sequence += 1
            if self.modem.kiss_tx_queue is not None:
                self.modem.kiss_tx_queue.unkey()
        if self.modem._broadcast_bracket:
            if keyed and not was_keyed:
                self.modem._broadcast_keyed = True
                if self.modem.observer:
                    self.modem.observer.modem_pending()
            elif not self.modem._broadcast_keyed:
                return
        if self.modem.observer:
            self.modem.observer.modem_ptt(on)
            if self.modem._broadcast_bracket:
                if not on:
                    self.modem.observer.modem_pending(cancel=True)
                self.modem.observer.modem_busy(on)
        if self.modem._broadcast_bracket and not on:
            self.modem._broadcast_keyed = False

    def tx(self, samples, *, marks=None):
        self._refused = self.modem._stopping
        if not self._refused:
            if marks is None:
                self.transport.tx(samples)
            elif 'marks' in inspect.signature(self.transport.tx).parameters:
                self.transport.tx(samples, marks=marks)
            else:
                if not getattr(self.transport, '_refused', False):
                    for k in range(len(marks[0])):
                        marks[1](k)
                self.transport.tx(samples)

    def tx_went_out(self):
        if self._refused or self.modem._stopping:
            return False
        verdict = getattr(self.transport, 'tx_went_out', None)
        return verdict() if verdict is not None else True

    def connected(self, caller, called, bw):
        if self.modem._stopping:
            return
        self.modem._incoming_pending_at = None
        self.transport.connected(caller, called, bw)
        self.modem._connected = True
        self.modem._intent_active = True
        if self.modem.observer:
            self.modem.observer.modem_connected(caller, called, bw)
            # VARA reports the received caller's capability after CONNECTED.
            # Our local high-speed enable is not evidence about that caller.
            report = getattr(self.modem.observer, 'modem_registered', None)
            marked = self.modem.station.peer_high_speed
            if report is not None and marked is not None and not self.modem._stopping:
                report('LINK REGISTERED' if marked else 'LINK UNREGISTERED')

    def pending(self):
        if self.modem._stopping:
            return
        # A caller retained after a failed acquisition belongs only to that
        # bracket. Never expose it as evidence for a subsequent attempt.
        self.modem.attempt_caller = None
        self.modem._incoming_pending_at = self.modem._clock()
        self.transport.pending()
        if self.modem.observer:
            self.modem.observer.modem_pending()

    def channel_activity(self, busy):
        self.modem.report_channel_activity(busy, kiss_active=self.transport.kiss_receiving)

    def admit_contact(self):
        with self.modem._lock:
            admit = getattr(self.transport, 'admit_contact', None)
            return not self.modem._stopping and (admit is None or admit(self.modem))

    def received_over_acknowledged(self):
        self.modem.received_ack_cursor = self.modem.received_cursor

    def data(self, payload):
        self.modem.received_cursor += len(payload)
        if self.modem.observer:
            self.modem.observer.modem_data_received(payload)
        else:
            self.transport.data(payload)

    def snr(self, sn):
        if self.modem.observer:
            self.modem.observer.modem_snr(sn)

    def bitrate(self, level, bps, tx):
        if self.modem.observer:
            self.modem.observer.modem_bitrate(level, bps, tx)

    def cq(self, result):
        if not self.modem.observer:
            return
        from ..broadcast import Broadcast
        from ..ui import UIFrame
        if isinstance(result, (Broadcast, UIFrame)):
            self.modem.observer.modem_broadcast(result)
        elif isinstance(result, B.BeaconResult):
            self.modem.observer.modem_beacon(result)
        else:
            self.modem.observer.modem_cqframe(result.call, str(result.bandwidth), result.sn)

    def compression_error(self, message):
        if self.modem.observer:
            self.modem.observer.modem_warning(message)
        else:
            self.transport.compression_error(message)


class VaraHostModem(ModemCore):
    """Queue-based binding; ``poll`` belongs to the audio driver's owner thread."""
    default_compression = 'TEXT'

    def __init__(self, transport, *, bandwidth='2300', high_speed=True, clock=time.monotonic,
                 kiss_tx_queue=None, receive_context=None):
        inherit_boundaries(self, transport)
        self.observer = None
        self._clock = clock
        self.last_unkey_at = None
        self.unkey_sequence = self.key_sequence = 0
        self.received_cursor = self.received_ack_cursor = 0
        self._transport = transport
        self.receive_context = receive_context or getattr(transport, 'receive_context', None)
        # Admission needs an indexed input clock that the transport advances;
        # a legacy burst transport cannot release a scanner watermark hold.
        if self.receive_context is None and hasattr(transport, 'input_clock'):
            self.receive_context = ReceiveContext(sample_start=transport.input_clock,
                tune_epoch=getattr(transport, 'tune_epoch', 0))
        self._cq_detector = (self.receive_context.detector if self.receive_context is not None
                             else CQDetector())
        if getattr(transport, 'receive_context', None) is not self.receive_context:
            transport.receive_context = self.receive_context
            bind = getattr(transport, 'bind_receive_context', None)
            if bind is not None:
                bind(self.receive_context)
        self._events = queue.Queue(maxsize=64)
        self._lock = threading.RLock()
        self._queued_original = self._accepted = self._confirmed_before = self._discarded_before = 0
        self._admission_closing = False
        self._intent_active = False
        self._cq_pending = False
        self.kiss_tx_queue = kiss_tx_queue
        self._cq_forced = False
        self._cq_started = False
        self._cq_rejections = []
        self._cq_reply_lock = threading.Lock()
        self.generation = 0
        self.retired = deque(maxlen=16)
        self._stopping = False
        self._retired_attachment = False
        self._heartbeat_reply = None
        self._registration_pending = True
        self._channel_busy = None
        self._activity = self._activity_since = None
        self.attempt_caller = None
        self._bandwidth = bandwidth
        self._high_speed = high_speed
        self._compression = self.default_compression
        self._mycall = []
        self._wanted_calls = []
        self._wanted_bandwidth = bandwidth
        self._wanted_compression = self._compression
        self._session_mode = self._wanted_session_mode = None
        self._listen = False
        self._incoming_pending_at = None
        self._chat = False
        self._connected = False
        self._closing = False
        self._close_discards = False
        self._close_reason = 'link closed'
        self._buffer = None
        self._keyed_clean = None
        self._ptt_active = False
        self._broadcast_bracket = self._broadcast_keyed = False
        self.station = self._new_station()

    def _new_station(self):
        station = VaraStationHandshake(self._mycall, _HostIO(self, self._transport),
                                       bw=self._bandwidth, high_speed=self._high_speed,
                                       compression=self._compression, chat=self._chat,
                                       cr_retry_form='rekey32' if self._session_mode else 'full')
        inherit_boundaries(station, self)
        station.receive_context = self.receive_context
        station.cq_detector = self._cq_detector
        return station

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def connect_retry_quiet_seconds(self):
        # Explicit host profile: conservative quiet intervals after actual
        # unkey, beyond the observed VARA HF timing. These are not
        # claims about exact native timer constants or physical RF timing.
        return {'WINLINK': 1.30, 'P2P': 1.85}.get(self._session_mode)

    @property
    def accepted_total(self): return self._accepted

    @property
    def confirmed_total(self): return self._confirmed_before+self.station.payload_ledger.confirmed

    @property
    def discarded_total(self): return self._discarded_before+self.station.payload_ledger.discarded

    def _replace_station(self, *, preserve_data=False, reason='replacement'):
        old = self.station
        contact_ended(old, reason)
        if self.incoming_pending:
            self.attempt_caller = old.caller if old.caller not in ('', CALLER_UNKNOWN) else None
        pending = self.station.preconnect_data if preserve_data else ()
        if preserve_data:
            ledger = old.payload_ledger
            ledger.reframe()
        else:
            original = old.retire_payload()
            ledger = old.payload_ledger
            self._confirmed_before += ledger.confirmed
            self._discarded_before += ledger.discarded
            if ledger.accepted:
                self.retired.append(dict(generation=self.generation, reason=reason,
                    accepted=ledger.accepted, confirmed=ledger.confirmed,
                    discarded=ledger.discarded, unconfirmed=original))
        self.station = self._new_station()
        self.received_cursor = self.received_ack_cursor = 0
        self.last_unkey_at = None
        self.generation += 1
        bind = getattr(self._transport, 'bind_station', None)
        if bind is not None:
            bind(self.station)
        self._closing = False
        self._close_discards = False
        self._close_reason = 'link closed'
        if not preserve_data:
            self._intent_active = False
        self._cancel_pending()
        if preserve_data:
            self.station.payload_ledger = ledger
            for payload in pending:
                self.station._preconnect_writes.append(payload)
                self.station._queue_host_payload(payload, accept=False)

    def start(self, observer):
        if self._stopping:
            raise RuntimeError("stopped VARA adapter cannot be reused; create a fresh attachment")
        self.observer = observer

    def stop(self):
        with self._lock:
            self._stopping = True
            self._heartbeat_reply = None

    def request_heartbeat(self, reply):
        with self._lock:
            if self._stopping:
                raise ConnectionAbortedError('VARA attachment retired')
            # One pending liveness request, independent of the host-command
            # queue capacity. A stalled owner must not accumulate timer work.
            if self._heartbeat_reply is None:
                self._heartbeat_reply = reply

    def report_channel_activity(self, active, *, kiss_active=None):
        """Publish owner-observed receive activity as VARA's BUSY, independently of link state."""
        now, active = self._clock(), bool(active)
        if active != self._activity:
            self._activity, self._activity_since = active, now
        held = now - self._activity_since
        busy = self._channel_busy
        if active and not busy and held >= BUSY_ON_S:
            busy = True
        elif not active and busy is not False and (busy is None or held >= BUSY_HOLD_S):
            busy = False
        if self.kiss_tx_queue is not None:
            self.kiss_tx_queue.activity(active if kiss_active is None else kiss_active)
        if self._stopping or busy == self._channel_busy:
            return
        self._channel_busy = busy
        detail = getattr(self._transport, 'busy_detail', None)
        if detail is not None:
            self._transport.log(f'BUSY {"ON" if busy else "OFF"}: {detail()}')
        report = getattr(self.observer, 'modem_busy', None)
        if report is not None:
            report(busy)

    def _enqueue(self, event):
        if self._stopping:
            raise ConnectionAbortedError('VARA attachment retired')
        try:
            self._events.put_nowait(event)
        except queue.Full:
            raise ConnectionAbortedError('VARA owner command queue full') from None

    @property
    def connected(self):
        return self._connected

    @property
    def incoming_pending(self):
        return self._incoming_pending_at is not None

    @property
    def incoming_pending_at(self):
        return self._incoming_pending_at

    def _cancel_pending(self):
        if self.incoming_pending:
            self._incoming_pending_at = None
            if self.observer is not None and not self._stopping:
                self.observer.modem_pending(cancel=True)

    def set_mycall(self, calls):
        with self._lock:
            if (self._intent_active or self.station.state in
                    (VaraState.CONNECTING, VaraState.CONNECTED, VaraState.DISCONNECTING)) \
                    and list(calls) != self._wanted_calls:
                raise ValueError('VARA identity changes require an idle contact')
            self._enqueue(('calls', list(calls)))
            self._wanted_calls = list(calls)

    def set_bandwidth(self, bw):
        with self._lock:
            if (self._intent_active or self.station.state in
                    (VaraState.CONNECTING, VaraState.CONNECTED, VaraState.DISCONNECTING)) \
                    and bw != self._wanted_bandwidth:
                raise ValueError('VARA bandwidth changes require an idle contact')
            self._enqueue(('bandwidth', bw))
            self._wanted_bandwidth = bw

    def set_compression(self, mode):
        from ..vara.compression import validate_mode
        mode = validate_mode(mode)
        with self._lock:
            if (self._intent_active or self.station.state in
                    (VaraState.CONNECTING, VaraState.CONNECTED, VaraState.DISCONNECTING)) \
                    and mode != self._wanted_compression:
                raise ValueError('VARA compression changes require an idle contact')
            self._enqueue(('compression', mode))
            self._wanted_compression = mode

    def set_listen(self, on):
        with self._lock:
            self._enqueue(('listen', on))
            if self.station.state in (VaraState.CONNECTED, VaraState.DISCONNECTING):
                self._admission_closing = True

    def set_session_mode(self, mode):
        if mode not in ('WINLINK', 'P2P'):
            raise ValueError('unknown VARA session mode')
        with self._lock:
            if (self._intent_active or self.station.state in
                    (VaraState.CONNECTING, VaraState.CONNECTED, VaraState.DISCONNECTING)) \
                    and mode != self._wanted_session_mode:
                raise ValueError('VARA session timing changes require an idle contact')
            self._enqueue(('session', mode))
            self._wanted_session_mode = mode

    def set_chat(self, on):
        with self._lock:
            self._enqueue(('chat', on))

    def clean_tx_buffer(self):
        with self._lock:
            # VARA answers at once. The owner may be keying, or about to: the
            # verdict it published at key-up, or the station's own now, answers
            # a refusal here; only a clear that succeeds waits for the owner.
            verdict = self._keyed_clean or self.station.clean_verdict
            if self.observer and (verdict == 'FAILED ' or verdict == 'BUFFEREMPTY'
                                  and not self._queued_original):
                self.observer.modem_clean_tx_buffer(verdict)
                return
            self._enqueue(('clean',))

    def connect(self, src, dst, vias=None):
        if vias:
            raise ValueError('VARA repeater connections are not qualified')
        with self._lock:
            if self._intent_active or self.station.state in (VaraState.CONNECTING, VaraState.CONNECTED,
                                      VaraState.DISCONNECTING):
                raise ValueError('VARA contact is already active')
            admit = getattr(self._transport, 'admit_contact', None)
            if admit is not None and not admit(self):
                raise ValueError('STATION CHANNEL OWNER BUSY')
            self._enqueue(('connect', src, dst))
            self._admission_closing = False
            self._intent_active = True

    def transmit(self, blob):
        with self._lock:
            if self._admission_closing:
                raise ConnectionAbortedError('VARA contact is closing')
            if (self._queued_original+self.station.payload_ledger.pending+len(blob)
                    > self.station.payload_ledger.limit):
                raise ConnectionAbortedError('VARA original payload queue full')
            self._enqueue(('data', bytes(blob)))
            self._queued_original += len(blob)
            self._accepted += len(blob)

    supports_frame_force = True  # Kestrel-only host extension, not part of VARA.

    def _frame_admission_refusal(self):
        """Explain the existing idle-frame guards; force changes none of them."""
        for blocked, reason in (
                (self._cq_pending, 'frame pending'), (self._intent_active, 'contact intent pending'),
                (self.incoming_pending, 'incoming acquisition'), (self._stopping, 'attachment stopping'),
                (self._admission_closing, 'contact close admission pending'),
                (self._ptt_active, 'PTT active'), (self._keyed_clean is not None, 'keyed frame'),
                (self.station.state not in (VaraState.DISCONNECTED, VaraState.LISTENING),
                 f'protocol state {self.station.state}')):
            if blocked:
                return reason
        return None

    def _send_broadcast(self, message):
        station = self.station
        if self._frame_admission_refusal() or self.observer is None or self._channel_busy:
            return TxDisposition.CONTENDED
        from ..tx.broadcast import synth_broadcast
        from ..tx.ui import packetize
        level, frames = packetize(message)
        audio = synth_broadcast(message)
        from ..rx.broadcast import PERIOD
        offsets = [0] + [67072 + k * PERIOD[level] for k in range(len(frames))]
        def started(k):
            if k and self._broadcast_keyed and self.observer:
                self.observer.modem_bitrate(level, {1: 18, 2: 41, 4: 88}[level], True)
        self.attempt_caller = None
        self._broadcast_bracket = True
        try:
            station._key(True)
            note = getattr(station.io, 'tx_metadata', None)
            if note is not None:
                note(protocol='VARA', packet_kind='BROADCAST', label='BROADCAST',
                     speed_level=level, bandwidth_hz=500,
                     modulation='DBPSK' if level == 4 else 'index modulation',
                     host_bitrate_level=level, frame_count=len(frames),
                     source=message.source, destination=message.destination,
                     link_state=station.state)
            station.io.tx(audio, marks=(offsets, started))
            sent = station._tx_went_out()
            result = self._kiss_tx_result(sent)
        finally:
            try:
                station._key(False)
            finally:
                if self._broadcast_keyed:
                    station.io.report_ptt(False)
                self._broadcast_bracket = self._broadcast_keyed = False
        self._check_unkey()
        return result

    def _check_unkey(self):
        unkeyed = getattr(self._transport, 'unkey_confirmed', None)
        if self._ptt_active or (unkeyed is not None and not unkeyed()):
            raise RuntimeError('Broadcast physical unkey was not confirmed')

    def _kiss_tx_result(self, sent):
        result = getattr(self._transport, 'kiss_tx_result', None)
        if result is not None:
            return result()
        if sent:
            return TxDisposition.SENT
        if getattr(self._transport, '_refused', False):
            return TxDisposition.REFUSED_BEFORE_KEY
        return TxDisposition.PARTIAL

    def service_kiss_queue(self):
        tx_queue = self.kiss_tx_queue
        if tx_queue is None:
            return None
        with self._lock:
            if (self._frame_admission_refusal() or self.observer is None
                    or self._channel_busy or self._activity):
                return None
            head = tx_queue.claim()
            if head is None:
                return None
            result = TxDisposition.REFUSED_BEFORE_KEY
            keys = self.key_sequence
            try:
                admit = getattr(self._transport, 'admit_kiss', None)
                result = admit(self) if admit is not None else None
                if result is None:
                    result = self._send_broadcast(head.value)
                    if result is TxDisposition.PARTIAL:
                        raise OSError('KISS transmit failed after keying')
                    if result is TxDisposition.UNKEY_UNCONFIRMED:
                        raise RuntimeError('KISS physical unkey was not confirmed')
            except BaseException:
                unkeyed = getattr(self._transport, 'unkey_confirmed', None)
                result = (TxDisposition.UNKEY_UNCONFIRMED
                          if result is TxDisposition.UNKEY_UNCONFIRMED or self._ptt_active
                          or (unkeyed is not None and not unkeyed())
                          else TxDisposition.PARTIAL
                          if result is TxDisposition.PARTIAL or self.key_sequence != keys
                          else TxDisposition.REFUSED_BEFORE_KEY)
                tx_queue.finish(head, result)
                raise
            tx_queue.finish(head, result)
            return result

    def send_cq(self, call, bw, *, force=False):
        if self._cq_pending:
            raise ValueError('CQ already pending')
        with self._lock:
            if reason := self._frame_admission_refusal():
                raise ValueError(f'VARA contact is already active: {reason}')
            build_frame(call, 500)
            if str(bw) not in ('500', '2300', '2750'):
                if force:
                    raise ValueError('unsupported forced CQ bandwidth')
                return
            frame = build_frame(call, int(bw))
            self._enqueue(('cq', frame, force, self.station if force else None))
            with self._cq_reply_lock:
                self._cq_pending = True
                self._cq_forced = force
                self._cq_started = False

    def send_beacon(self, text, *, force=False):
        """One KISS broadcast frame, keyed as a CQ is: idle station, clear channel."""
        B.build_payload(text)
        if force and not re.fullmatch(r'2[01][A-Z][A-R]{2}[0-9]{2}[A-X]{2}[A-Z0-9]{3,7} *', text):
            raise ValueError('BEACONFORCE requires advanced beacon text')
        with self._lock:
            if reason := self._frame_admission_refusal():
                raise ValueError(f'VARA contact is already active: {reason}')
            self._enqueue(('beacon', text, force, self.station if force else None))
            with self._cq_reply_lock:
                self._cq_pending = True
                self._cq_forced = force
                self._cq_started = False

    def defer_cq_rejection(self, reply):
        with self._cq_reply_lock:
            if not self._cq_pending or self._cq_started or len(self._cq_rejections) >= 64:
                return False
            self._cq_rejections.append(reply)
            return True

    def _release_cq_rejections(self):
        with self._cq_reply_lock:
            self._cq_started = True
            replies, self._cq_rejections = self._cq_rejections, []
        for reply in replies:
            reply()

    def _send_cq(self, kind, value, force=False, admitted_station=None):
        # Kestrel-only one-shot override: bypass detected channel occupancy,
        # never connection ownership, a prior key-up, or transport/hardware guards.
        if (self.observer is None or (self._channel_busy and not force)
                or self._stopping or self._admission_closing or self._intent_active
                or self.incoming_pending or self._ptt_active or self._keyed_clean is not None
                or (admitted_station is not None and admitted_station is not self.station)
                or self.station.state not in (VaraState.DISCONNECTED, VaraState.LISTENING)):
            return
        beacon = kind == 'beacon'
        audio = synth_beacon(value) if beacon else synth_frame(value)
        self.attempt_caller = None
        self.observer.modem_pending()
        try:
            self.station._key(True)
            self._release_cq_rejections()
            self.observer.modem_bitrate(*((2, 41) if beacon else (1, 18)), True)
            self.station._transmit_audio(audio, packet_kind='BEACON' if beacon else 'CQ', modulation='MFSK')
            result = self._kiss_tx_result(self.station._tx_went_out())
            if force and not self.station._tx_went_out():
                self.observer.modem_forced_frame_refused()
        finally:
            try:
                self.station._key(False)
            finally:
                self.observer.modem_pending(cancel=True)
        return result

    def disconnect(self):
        with self._lock:
            self._enqueue(('disconnect',))
            self._admission_closing = True

    def abort(self):
        self.abort_with_completion()

    def abort_with_completion(self, completed=None):
        """Notify internal admission after owner reset; no extra host wire event."""
        if self._cq_pending and not self._cq_forced:
            return
        with self._lock:
            if self._cq_pending and not self._cq_forced:
                return
            self._enqueue(('abort', completed))
            self._admission_closing = False
            self._intent_active = False

    def poll(self):
        """Apply queued host commands and publish driver state on its owner thread."""
        with self._lock:
            self._poll()
            reply, self._heartbeat_reply = self._heartbeat_reply, None
            if reply is not None and not self._stopping:
                reply()

    def _poll(self):
        if self._stopping:
            if self._retired_attachment:
                return
            self._retired_attachment = True
            cancelled = bytearray()
            while not self._events.empty():
                ev = self._events.get_nowait()
                if ev[0] == 'data':
                    cancelled.extend(ev[1])
            if cancelled:
                self._discarded_before += len(cancelled)
                self.retired.append(dict(generation=self.generation,
                    reason='unapplied host queue cancelled', accepted=len(cancelled),
                    confirmed=0, discarded=len(cancelled), unconfirmed=bytes(cancelled)))
            self._queued_original = 0
            self._cq_pending = False
            self._cq_rejections.clear()
            self._replace_station(reason='attachment retired')
            self._connected = False
            self.observer = None
            return
        if self._registration_pending and self.observer is not None:
            # Map the high speed setting onto the host grammar's full/limited
            # report. HFMODEM names this implementation.
            report = getattr(self.observer, 'modem_registered', None)
            if report is not None:
                report('REGISTERED HFMODEM' if self._high_speed else 'REGISTERED')
            self._registration_pending = False
        pending_cq = None
        for _ in range(64):
            if self._stopping:
                self._poll()
                return
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind in ('abort', 'stop'):
                if kind == 'stop':
                    pending_cq = None
                    self._cq_pending = False
                    self._cq_rejections.clear()
                self._replace_station(reason='host abort')
                if self._connected and self.observer:
                    self.observer.modem_disconnected()
                self._connected = False
                if kind == 'stop':
                    self.observer = None
                elif self._listen:
                    self.station.listen()
                if kind == 'abort' and len(ev) > 1 and ev[1] is not None:
                    ev[1]()
            elif kind in ('cq', 'beacon'):
                if pending_cq is None:
                    pending_cq = ev
            elif kind == 'calls':
                self._mycall = ev[1]
                self.station.mycalls = list(self._mycall)
            elif kind == 'compression':
                self.station.set_compression(ev[1])
                self._compression = ev[1]
            elif kind == 'bandwidth':
                self._bandwidth = ev[1]  # next connection, never mutate live PHY
                if (self.station.bw != ev[1] and not self._intent_active
                        and self.station.state in (VaraState.DISCONNECTED, VaraState.LISTENING)):
                    listening = self.station.state == VaraState.LISTENING
                    self._replace_station(preserve_data=True)
                    if listening:
                        self.station.listen()
            elif kind == 'session':
                self._session_mode = ev[1]
            elif kind == 'listen':
                self._listen = ev[1]
                if self.station.state in (VaraState.CONNECTED, VaraState.DISCONNECTING):
                    # LISTEN is a contact boundary even when its value does not
                    # change. Unlike DISCONNECT, VARA does not drain pending
                    # payload. Keep the ledger until actual close retirement:
                    # unsent bytes are discarded, never credited as delivered.
                    self._admission_closing = True
                    self._closing = self._close_discards = True
                    self._close_reason = 'LISTEN ON' if ev[1] else 'LISTEN OFF'
                else:
                    if self.station.state == VaraState.DISCONNECTED:
                        self._replace_station(preserve_data=True)
                    self.station.listen(ev[1])
            elif kind == 'chat':
                self._chat = self.station.chat = ev[1]
            elif kind == 'clean':
                result = self.station.clean_tx_buffer()
                if result == 'OK':
                    self._buffer = self.station.pending_bytes
                if self.observer:
                    self.observer.modem_clean_tx_buffer(result)
                    if result == 'OK':
                        self.observer.modem_buffer(0)
            elif kind == 'connect':
                if pending_cq is not None:
                    pending_cq = None
                    self._cq_pending = False
                # Host writes before CONNECT belong to this connection. Preserve
                # their original bytes, not provisional framing from another BW.
                self._replace_station(preserve_data=True)
                self._intent_active = True
                self.station.originate(ev[2], ev[1])
            elif kind == 'data':
                self._queued_original -= len(ev[1])
                self.station.send(ev[1])
            elif kind == 'disconnect':
                if pending_cq is not None:
                    pending_cq = None
                    self._cq_pending = False
                self._closing = False
                if self.station.state in (VaraState.CONNECTED, VaraState.DISCONNECTING):
                    if self.station.role in ('initiator', 'responder'):
                        self._closing = True
                    elif self.observer:
                        # Do not promise a close the protocol driver cannot key.
                        self.observer.modem_disconnect_refused(
                            'WRONG DISCONNECT: session role is unavailable')
                        self._admission_closing = False
                else:
                    # Defensive DISCONNECT cancels an attempt, never a future link.
                    self._replace_station()
                    if self._listen:
                        self.station.listen()
                    if self.observer:
                        self.observer.modem_disconnected()
                    self._admission_closing = False
        if self._stopping:
            self._poll()
            return
        if self.incoming_pending and self.station.state in (VaraState.DISCONNECTED, VaraState.LISTENING):
            self._replace_station(reason='incoming acquisition failed')
            self._admission_closing = False
            if self._listen:
                self.station.listen()
        if pending_cq is not None:
            try:
                self._send_cq(*pending_cq)
            finally:
                self._release_cq_rejections()
                with self._cq_reply_lock:
                    self._cq_pending = False
                    self._cq_started = False
        if not self._cq_pending and self._cq_rejections:
            self._release_cq_rejections()
            self._cq_started = False
        if self._closing and (self._close_discards or not self.station.data_pending):
            self.station.disconnect(acquire=True)
        live = self.station.state in (VaraState.CONNECTED, VaraState.DISCONNECTING)
        if self._connected and not live:
            contact_ended(self.station, self._close_reason)
            self._connected = False
            self._closing = False
            if self.observer:
                self.observer.modem_disconnected()
            self._replace_station(reason=self._close_reason)
            self._admission_closing = False
            if self._listen:
                self.station.listen()
        # LISTEN accepted behind ABORT can observe the old live station and
        # latch closing admission. Reconcile only after the owner has consumed
        # every queued intent and has actually settled at an idle boundary.
        if (not self._closing and not self._intent_active and self._events.empty()
                and self.station.state in (VaraState.DISCONNECTED, VaraState.LISTENING)):
            self._admission_closing = False
        depth = self.station.pending_bytes
        if depth != self._buffer:
            self._buffer = depth
            if self.observer:
                self.observer.modem_buffer(depth)
