# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One connected-packet owner over the station RPR lane.

Tracker and AGWPE are mutually exclusive presentations of this engine. Endpoint
threads submit bounded requests; RF frames and timers are consumed on this
owner independently of host polls. The existing RPR worker retains channel
access, waveform generation, the physical arbiter and actual completion.
"""
from dataclasses import dataclass
from concurrent.futures import Future
import queue
import threading
import time

from hfmodem.ax25.connected import Connection


@dataclass
class Attachment:
    host: object
    alive: bool = True
    failure: str = ''
    failure_callback: object = None


class PacketRuntime:
    def __init__(self, link, *, mycall='NOCALL', full_scale_mv=3000, contact=None,
                 clock=time.monotonic):
        if not full_scale_mv > 0:
            raise ValueError('Tracker drive reference must be positive')
        self.link, self.clock = link, clock
        self.full_scale_mv = full_scale_mv
        self.engine = Connection(mycall, self._transmit, clock=clock)
        self.contact = contact
        self._incoming_epoch = None
        if self.contact is not None:
            self.engine.admit_contact = lambda: self.contact.admit(receive_epoch=self._incoming_epoch)
        self.engine.listen = False
        self._attachment = None
        self._transmission = None
        self._generation = None
        self._commands = queue.Queue(maxsize=64)
        self._frames = queue.Queue(maxsize=128)
        self._rf_overflow = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    @property
    def active(self):
        return (self.engine.active or bool(self.engine.controls) or bool(self.engine.repeats)
                or self._transmission is not None)

    def start(self):
        if self.link.host is not None or self.link.packet_engine is not None:
            raise ValueError('RPR lane already has a host owner')
        self.link.host = self
        self.link.packet_engine = self
        self._thread = threading.Thread(target=self._run, name='rpr-ax25-owner', daemon=True)
        self._thread.start()

    def _call(self, fn):
        if self._stop.is_set() or self._thread is None:
            raise ConnectionAbortedError('packet runtime stopped')
        future = Future()
        try:
            self._commands.put_nowait((future, fn))
        except queue.Full:
            raise ConnectionAbortedError('packet owner command queue full') from None
        try:
            return future.result(timeout=2)
        except TimeoutError:
            if future.cancel():
                raise
            return future.result()

    def attach(self, personality, send=None, *, failed=None):
        def install():
            if self._attachment is not None or self._transmission is not None:
                raise ValueError('packet attachment or terminal transmission still active')
            if personality == 'tracker':
                from .tracker import TrackerHost
                host = TrackerHost(self.engine, configure=self._configure, clock=self.clock)
                # A reopened facade reports fresh defaults. Apply those to the
                # retained engine and RF worker too, instead of inheriting an
                # earlier attachment's window, drive or channel-access settings.
                for key in ('F', 'N', 'O', '@I', '@T2', '@T3',
                            '%XR', 'T', 'P', 'W', '%BR'):
                    host.command(f'{key} {host.values[key]}')
                host.values['Y'] = 0
                self.engine.listen = False
            elif personality == 'agwpe':
                from .agwpe import AgwHost
                if send is None:
                    raise ValueError('AGWPE needs a bounded output sink')
                host = AgwHost(self.engine, send, txdelay=min(255, self.link.txdelay // 10),
                               persistence=self.link.access.persistence,
                               slot=min(255, (self.link.access.slot_ms + 9) // 10))
            else:
                raise ValueError('unknown connected-packet personality')
            self._attachment = Attachment(host, failure_callback=failed)
            return self._attachment
        return self._call(install)

    def feed(self, attachment, data):
        data = bytes(data)
        if len(data) > 8192:
            raise ValueError('packet endpoint read exceeds limit')
        def apply():
            if attachment is not self._attachment or not attachment.alive:
                raise ConnectionAbortedError('packet attachment retired')
            try:
                return attachment.host.feed(data) or b''
            except (ValueError, BufferError, ConnectionError) as exc:
                self._fail(str(exc))
                raise
        return self._call(apply)

    def detach(self, attachment):
        # Invalidate old queued requests immediately, before their owner turn.
        attachment.alive = False
        if not self._stop.is_set():
            self._call(lambda: self._retire(attachment))

    def detach_and_wait(self, attachment, *, timeout=30.):
        """Retire the facade and wait for the lane's cancelled RF work to finish.

        A supervisor must keep its lease until the actual packet worker has
        reported completion. Accepting a cancel request alone is insufficient.
        Owner progress and RF completion remain independent of this waiter.
        """
        self.detach(attachment)
        def retired():
            if self._attachment is not None:
                raise RuntimeError('another packet attachment entered during retirement')
            if self.active:
                return False
            if self.contact is not None:
                self.contact.release_if(lambda: not self.active)
            return True
        deadline = time.monotonic() + timeout
        while not self._call(retired):
            if time.monotonic() >= deadline:
                raise RuntimeError('packet RF work did not retire')
            time.sleep(.01)

    def _retire(self, attachment):
        if attachment is self._attachment:
            attachment.host.attachment_lost()
            self._attachment = None
            if self._transmission is not None:
                self._transmission.state.cancel()
            self.link.drop_pending()

    def _fail(self, why):
        if self._attachment is not None:
            self._attachment.failure = why
            self._attachment.alive = False
            attachment = self._attachment
            try:
                if attachment.failure_callback is not None:
                    attachment.failure_callback(why)
            finally:
                self._retire(attachment)

    def send_frame(self, raw, *, channel_epoch=None):
        """The lane's RF receive callback only enqueues; it never runs AX.25."""
        if channel_epoch is None:
            channel_epoch = self.contact.receive_boundary[0] if self.contact is not None else 0
        try:
            self._frames.put_nowait((channel_epoch, bytes(raw)))
        except queue.Full:
            self._rf_overflow.set()

    def _transmit(self, raw):
        if self._transmission is not None or self._stop.is_set():
            return False
        self._transmission = self.link.enqueue(raw)
        self._generation = self.engine.generation
        self.engine.transmission_started()
        return True

    def _configure(self, key, value):
        # This virtual TNC's mV reference is supplied by its deployment. No
        # claim that an arbitrary sound card has an SCS controller's voltage.
        if self._transmission is not None:
            raise ValueError('RF parameter change during a packet transmission')
        with self.link._condition:
            if key == '%XR':
                scale = value / self.full_scale_mv
                if scale > 1:
                    raise ValueError('drive exceeds the configured card voltage reference')
                self.link.drive_scale = scale
            elif key == '%BR':
                if value != 300:
                    raise ValueError('only R300 is supported')
                self.link.level = 0
            elif key == 'T':
                self.link.txdelay = value * 10
            elif key == 'P':
                with self.link.access._lock:
                    self.link.access.persistence = value
            elif key == 'W':
                with self.link.access._lock:
                    self.link.access.slot_ms = value
            else:
                raise ValueError('unbound RF parameter')
            self.link._condition.notify_all()

    def _step(self):
        if self._transmission is not None:
            completion = self._transmission.completion
            if completion is not None:
                self._transmission = None
                if self._generation == self.engine.generation:
                    self.engine.transmission_finished(completion[0] == 'sent')
                    if completion[0] != 'sent':
                        self._fail(f'packet transmission {completion[0]}: {completion[1]}')
                else:
                    self.engine.transmission_finished(True)
        if self._rf_overflow.is_set():
            self._rf_overflow.clear()
            self._fail('packet RF input queue overflow')
        for _ in range(64):
            try:
                future, fn = self._commands.get_nowait()
            except queue.Empty:
                break
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except Exception as exc:
                    future.set_exception(exc)
        for _ in range(128):
            try:
                channel_epoch, raw = self._frames.get_nowait()
            except queue.Empty:
                break
            if self.contact is not None and channel_epoch != self.contact.receive_boundary[0]:
                continue
            self._incoming_epoch = channel_epoch
            try:
                self.engine.on_frame(raw)
            finally:
                self._incoming_epoch = None
        if self._attachment is not None:
            service = getattr(self._attachment.host, 'service', None)
            if service:
                service()
            failure = getattr(self._attachment.host, 'failed', None)
            if failure:
                self._fail(failure)
        self.engine.tick()
        if self.contact is not None:
            self.contact.release_if(lambda: not self.active)

    def _run(self):
        try:
            while not self._stop.wait(.01):
                try:
                    self._step()
                except Exception as exc:
                    self._fail(str(exc))
        finally:
            if self._attachment is not None:
                self._retire(self._attachment)
            self.engine.listen = False
            self.engine.abort('packet owner stopped')
            if self.contact is not None:
                self.contact.close()
            while not self._commands.empty():
                future, _ = self._commands.get_nowait()
                if not future.cancelled():
                    future.set_exception(ConnectionAbortedError('packet owner stopped'))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
            if self._thread.is_alive():
                raise RuntimeError('packet owner did not retire')
        self.link.drop_pending()
        if self._transmission is not None:
            self._transmission.state.cancel()
        if self.link.host is self:
            self.link.host = None
        if self.link.packet_engine is self:
            self.link.packet_engine = None
