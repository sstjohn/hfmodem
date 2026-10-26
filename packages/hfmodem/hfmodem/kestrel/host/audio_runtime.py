# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Single VARA owner cadence shared by station and standalone audio launchers."""
import queue
import time
import threading

from .vara_modem import VaraHostModem
from ..vara.vara_arq import VaraState


class RetuneRequest:
    """One local owner transaction; cancellation cannot race an executing tune."""
    def __init__(self, callback, modem, station, receive_cursor, unkeyed, valid):
        self.callback, self.modem, self.station = callback, modem, station
        self.receive_cursor, self.unkeyed, self.valid = receive_cursor, unkeyed, valid
        self.done, self.error, self.cancelled = False, None, False
        self._lock = threading.Lock()

    def cancel(self):
        with self._lock:
            if self.done:
                return False
            self.cancelled = self.done = True
            return True


class AudioHostRuntime:
    """Single audio owner; the TCP accept thread only creates/queues adapters."""
    def __init__(self, transport, *, high_speed=True, clock=time.monotonic):
        self.transport = transport
        self.high_speed = high_speed
        self.clock = clock
        self._new = queue.Queue(maxsize=1)
        self.modem = None
        self._station = None
        self._attempt_end = 0.0
        self._setup_window_started = False
        self._retry_at = 0.0
        self._idle_at = 0.0
        self._progress = None
        self._state = None
        self._unkey_sequence = None
        self._diagnostic_modem = None
        self._payload_credit = (0, 0)
        self._retunes = queue.Queue(maxsize=1)
        self._retune = None

    def request_retune(self, callback, *, modem, station, receive_cursor, unkeyed, valid):
        """Queue a receiver move, preserving the live modem and original-byte debt.

        The callback runs on the audio owner only after the received final over
        was acknowledged, the peer completed its handback, and the physical key
        is verified down. Leaving with the handback still owed strands that
        receive state on the new frequency before the peer can finish on CF.
        """
        if receive_cursor <= 0:
            raise ValueError('Receiver retune needs a received payload cursor')
        request = RetuneRequest(callback, modem, station, receive_cursor, unkeyed, valid)
        self._retunes.put_nowait(request)
        return request

    def _service_retune(self):
        if self._retune is None:
            try:
                self._retune = self._retunes.get_nowait()
            except queue.Empty:
                return False
        request = self._retune
        with request._lock, request.modem._lock:
            if request.done:
                self._retune = None
                return False
            try:
                modem = self.modem
                if (modem is not request.modem or modem.station is not request.station
                        or modem.observer is None or modem._stopping or not modem._intent_active
                        or modem._admission_closing
                        or modem.station.state != VaraState.CONNECTED or not request.valid()):
                    request.error = RuntimeError('Link changed before receiver retune')
                    request.done = True
                    self._retune = None
                    return False
                if modem.received_ack_cursor < max(request.receive_cursor, modem.received_cursor):
                    return False
                # A local DATA ACK only advances received_ack_cursor. The
                # sender still owes its handback on the original frequency.
                # KQ4UGU 2026-10-25: tuning 109 ms after unkey discarded that
                # opportunity and left the receiver awaiting release for 81 s.
                if getattr(modem.station, '_answer_owed', None) is not None:
                    return False
                if modem._ptt_active or not request.unkeyed():
                    return False
                reset = getattr(self.transport, 'retuned', None)
                if reset is None:
                    raise RuntimeError('Transport cannot retire old-frequency receive samples')
                try:
                    request.callback()
                finally:
                    # Even a failed CAT operation may have moved the receiver.
                    reset()
                    modem.station.retuned()
                request.done = True
                self._retune = None
                return True
            except BaseException as error:
                request.error, request.done = error, True
                self._retune = None
                # A tune failure leaves frequency uncertain: stop the owner.
                raise

    def modem_factory(self):
        modem = VaraHostModem(self.transport, high_speed=self.high_speed, clock=self.clock)
        try:
            self._new.put_nowait(modem)
        except queue.Full:
            raise ConnectionAbortedError('prior VARA attachment awaits its owner') from None
        return modem

    def _report_payload_credit(self):
        """Observe the original-byte ledger on its owner thread, never BUFFER guesses."""
        if self.modem is None:
            return
        if self.modem is not self._diagnostic_modem:
            self._diagnostic_modem = self.modem
            self._payload_credit = (0, 0)
        credit = (getattr(self.modem, 'confirmed_total', 0),
                  getattr(self.modem, 'discarded_total', 0))
        previous, self._payload_credit = self._payload_credit, credit
        notify = getattr(self.transport, 'diagnostic', None)
        if notify is not None:
            for kind, total, before in zip(('acknowledged', 'discarded'), credit, previous):
                if total > before:
                    try:
                        notify(kind, bytes=total - before)
                    except Exception:
                        pass  # An observer must not interfere with the modem owner.

    def step(self, timeout=0.05):
        before = getattr(self.transport, 'before_step', None)
        if before is not None:
            before()
        if self.modem:
            self.modem.poll()  # detach/abort old clients before serving the next
            self._report_payload_credit()
        while not self._new.empty():
            self.modem = self._new.get()
        if self.modem:
            self.modem.poll()
        if self._service_retune():
            return
        hs = self.modem.station if self.modem and self.modem.observer else None
        self.transport.hs = hs
        epoch = getattr(self.transport, 'receive_epoch',
                        getattr(self.transport, '_rx_generation', None))
        burst = self.transport.next_rx_burst(timeout=timeout, hs=hs)
        if hs is None:
            return
        self.modem.poll()
        if self._service_retune():
            return
        if (hs is not self.modem.station or self.modem.observer is None
                or epoch != getattr(self.transport, 'receive_epoch',
                                    getattr(self.transport, '_rx_generation', None))):
            return  # The attachment was retired while the receiver waited.
        if burst is not None:
            hs.on_rx_audio(burst)
        if self._service_retune():
            return
        active = getattr(self.transport, 'channel_active', None)
        self.modem.report_channel_activity(
            self.transport.receiving if active is None else active(self.modem.bandwidth))
        now = self.clock()
        if hs is not self._station:
            self._station = hs
            self._idle_at = now + 10
            self._progress = None
            self._state = None
        if hs.state == VaraState.CONNECTING and self._state != VaraState.CONNECTING:
            pending_at = self.modem.incoming_pending_at
            # Stock abandons an incomplete inbound handshake 30 seconds
            # after PENDING. Anchor before our response TX, not after unkey.
            self._attempt_end = pending_at + 30 if pending_at is not None else now + 60
            self._setup_window_started = False
            self._retry_at = now + 8
            self._unkey_sequence = None
        self._state = hs.state
        if hs.state == VaraState.CONNECTING:
            if self.modem.incoming_pending:
                self._attempt_end = self.modem.incoming_pending_at + 30
            elif hs.connect_setup_pending and not self._setup_window_started:
                # A valid response advances acquisition to setup. KQ4QWH
                # 2026-10-25 answered the last request; our setup crossed the
                # request deadline and we aborted immediately after unkey.
                # Give this stage one bounded window, anchored to actual TX.
                # Neither repeated responses/setups nor a restart replenish it.
                anchor = self.modem.last_unkey_at
                self._attempt_end = max(self._attempt_end,
                                        (now if anchor is None else anchor) + 30)
                self._setup_window_started = True
            quiet = self.modem.connect_retry_quiet_seconds
            # A setup sent inside RX processing must get its own receive window,
            # even when CHAT keeps the ordinary connect-request cadence.
            anchor_unkey = quiet is not None or getattr(hs, 'connect_setup_pending', False)
            if anchor_unkey and self._unkey_sequence != self.modem.unkey_sequence:
                self._unkey_sequence = self.modem.unkey_sequence
                gap = quiet if quiet is not None and hs.connect_request_pending else 8
                anchor = self.modem.last_unkey_at
                self._retry_at = (now if anchor is None else anchor) + gap
            if now >= self._attempt_end:
                self.modem.abort()
                if not self.modem.connected and not self.modem.incoming_pending:
                    self.modem.observer.modem_disconnected()
            elif now >= self._retry_at and not self.transport.receiving:
                hs.retry_connect()
                gap = quiet if quiet is not None and hs.connect_request_pending else 8
                anchor = self.modem.last_unkey_at
                keyed = anchor_unkey and self._unkey_sequence != self.modem.unkey_sequence
                self._retry_at = ((anchor if keyed and anchor is not None else self.clock()) + gap)
                self._unkey_sequence = self.modem.unkey_sequence
        elif hs.state == VaraState.CONNECTED:
            progress = (hs.progress, hs.idle_keyed)
            if progress != self._progress:
                self._progress, self._idle_at = progress, now + 10
            if not self.transport.receiving:
                query = hs.intermediate_query_due_in()
                retry = hs.data_retry_due_in()
                poll = hs.idle_poll_due_in()
                if query is not None and query <= 0:
                    hs.query_intermediate_answer()
                elif retry is not None and retry <= 0:
                    hs.retry_data_over()
                elif poll is not None and poll <= 0:
                    hs.idle_keepalive()
                    self._idle_at = self.clock() + 10
                elif now >= self._idle_at:
                    hs.idle_keepalive()
                    self._idle_at = self.clock() + 10
        self.modem.poll()
        self._report_payload_credit()
