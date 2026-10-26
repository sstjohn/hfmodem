# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Station-specific ARDOP owner: one passive BesraModem, bounded input queues.

Host threads wait for their command's actual effect. The station's decoder only
posts frames; this thread alone applies frames, commands and the ARDOP clock.
Transmission stays in the station arbiter through the injected sample sink.
This does not open an audio device or a radio-control connection.
"""
from concurrent.futures import Future
import queue
import threading
import time

from .arq.modem import BesraModem
from .host.modem_core import ModemCore
from .host.protocol import ArdopState


class _Observer:
    def __init__(self, runtime):
        self.runtime = runtime

    def __getattr__(self, name):
        def publish(*args):
            if name == 'modem_ptt' and self.runtime.ptt_from_station:
                return
            self.runtime._refresh()
            observer = self.runtime._observer
            if observer is not None:
                fn = getattr(observer, name, None)
                if fn:
                    fn(*args)
        return publish


class StationArdopModem(ModemCore):
    def __init__(self, transmit, *, clock=time.monotonic, bandwidth=500,
                 capture_position=lambda: 0, ptt_from_station=False, contact=None):
        self.core = BesraModem(bandwidth=bandwidth, threaded=False)
        self.core.audio_out = transmit
        self.clock = clock
        self.ptt_from_station = ptt_from_station
        self.contact = contact
        self._channel_boundary = contact.receive_boundary if contact is not None else (0, 0)
        self.capture_position = capture_position
        self._observer = None
        self._commands = queue.Queue(maxsize=64)
        self._aux_lock = threading.Lock()
        self._aux_pending = 0
        self._frames = queue.Queue(maxsize=128)
        self._fault = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.receive_boundary = (0, 0)
        self._snapshot = {'state': ArdopState.OFFLINE, 'connected': False,
                          'queued': 0, 'bandwidth': bandwidth,
                          'expected_session': None, 'rx_epoch': 0,
                          'identification_pending': False}
        self.core._demod._expect_session = self.expected_session
        self.core._demod._rx_epoch = self.rx_epoch

    @property
    def generation(self):
        return self.receive_boundary[0]

    @property
    def state(self):
        return self._snapshot['state']

    @property
    def connected(self):
        return self._snapshot['connected']

    @property
    def queued(self):
        return self._snapshot['queued']

    @property
    def bandwidth(self):
        return self._snapshot['bandwidth']

    @property
    def emission_bandwidth(self):
        # Read on this owner's synchronous sample-sink callback, after the
        # producer has declared its actual emission and before it clears it.
        return self.core.emission_bandwidth

    @property
    def drive_scale(self):
        return self.core._drivelevel / 100

    @property
    def active(self):
        with self._aux_lock:
            auxiliary = self._aux_pending > 0
        return (auxiliary or self._snapshot['identification_pending']
                or self.state not in (ArdopState.DISC, ArdopState.OFFLINE))

    def _aux_done(self, name):
        if name in ('send_id', 'two_tone_test'):
            with self._aux_lock:
                self._aux_pending -= 1

    def expected_session(self):
        return self._snapshot['expected_session']

    def rx_epoch(self):
        return self._snapshot['rx_epoch']

    def _refresh(self):
        previous = self.state
        self._snapshot = {'state': self.core.state, 'connected': self.core.connected,
                          'queued': self.core.queued, 'bandwidth': self.core.bandwidth,
                          'expected_session': self.core.expected_session(),
                          'rx_epoch': self.core.rx_epoch(),
                          'identification_pending': self.core.identification_pending}
        if previous != ArdopState.DISC and self.state == ArdopState.DISC:
            self._advance_generation()
        if self.contact is not None:
            self.contact.release_if(lambda: not self.active)

    def _advance_generation(self, floor=None):
        self.receive_boundary = (self.generation+1,
                                 int(self.capture_position()) if floor is None else floor)

    @property
    def channel_epoch(self):
        return self._channel_boundary[0]

    def _sync_channel(self):
        boundary = self.contact.receive_boundary if self.contact is not None else (0, 0)
        if boundary != self._channel_boundary:
            self._advance_generation(max(boundary[1], self.receive_boundary[1]))
            self._channel_boundary = boundary

    def start(self, observer):
        if self._thread is not None:
            raise RuntimeError('ARDOP station owner already started')
        self._observer = observer
        self._t0 = self.clock()
        self._thread = threading.Thread(target=self._run, name='station-ardop-owner', daemon=True)
        self._thread.start()
        self._call('start', _Observer(self))

    def _call(self, name, *args):
        if self._stop.is_set():
            raise ConnectionAbortedError('ARDOP station owner stopped')
        if self._thread is None:
            # Initial configuration before the owner begins is not concurrent.
            result = getattr(self.core, name)(*args)
            self._refresh()
            return result
        future = Future()
        if name in ('send_id', 'two_tone_test'):
            with self._aux_lock:
                self._aux_pending += 1
        try:
            self._commands.put_nowait((future, name, args))
        except queue.Full:
            self._aux_done(name)
            raise ConnectionAbortedError('ARDOP owner command queue full') from None
        try:
            return future.result(timeout=30)
        except TimeoutError:
            if future.done() or future.cancel():
                raise
            # A begun submission belongs to the station's bounded transmit
            # operation. Do not report a timeout and let it affect a later host.
            return future.result()

    def deliver_frames(self, frames, *, generation=None, channel_epoch=None):
        try:
            self._frames.put_nowait((self.generation if generation is None else generation,
                                    self.channel_epoch if channel_epoch is None else channel_epoch,
                                    tuple(frames)))
        except queue.Full:
            self._fault.set()

    def receive_fault(self):
        self._fault.set()

    def notify_ptt(self, on):
        observer = self._observer
        if observer is not None:
            observer.modem_ptt(on)

    def _time(self):
        now = self.clock() - self._t0
        self.core._clock = now
        if self.core._session is not None:
            self.core._session._now = now
        return now

    def _run(self):
        try:
            while not self._stop.wait(.01):
                self._sync_channel()
                if self._fault.is_set():
                    self._fault.clear()
                    self.core.abort()
                    self._advance_generation()
                    self._refresh()
                    if self._observer:
                        self._observer.modem_fault('ARDOP station receive queue/clock failed')
                for _ in range(64):
                    try:
                        future, name, args = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if future.set_running_or_notify_cancel():
                        try:
                            self._time()
                            if name == 'connect' and self.contact is not None and not self.contact.admit():
                                if self.contact.arbiter is None:
                                    raise OSError('station cannot transmit without an arbiter')
                                raise ValueError('STATION CHANNEL OWNER BUSY')
                            if name in ('connect', 'abort'):
                                self._advance_generation()
                            result = getattr(self.core, name)(*args)
                            if name in ('set_codec', 'set_mycall'):
                                self._advance_generation()
                            self._refresh()
                            future.set_result(result)
                        except Exception as exc:
                            # A rejected setting must leave a healthy contact alone.
                            # Failed physical submission instead retires the contact.
                            if (isinstance(exc, OSError)
                                    and name not in ('send_id', 'two_tone_test')):
                                self.core.abort()
                            self._refresh()
                            future.set_exception(exc)
                        finally:
                            self._aux_done(name)
                    else:
                        self._aux_done(name)
                for _ in range(128):
                    try:
                        generation, channel_epoch, frames = self._frames.get_nowait()
                    except queue.Empty:
                        break
                    if generation == self.generation and channel_epoch == self.channel_epoch:
                        try:
                            self._time()
                            if (not self.active and self.contact is not None
                                    and not self.contact.admit(receive_epoch=channel_epoch)):
                                self._advance_generation()
                                continue
                            self.core.receive_frames(frames)
                            self._refresh()
                        except Exception as exc:
                            self.core.abort()
                            self._refresh()
                            if self._observer:
                                self._observer.modem_fault(f'ARDOP receive failed: {exc}')
                try:
                    self.core.tick(self._time())
                    self._refresh()
                except Exception as exc:
                    self.core.abort()
                    self._refresh()
                    if self._observer:
                        self._observer.modem_fault(f'ARDOP timing/transmit failed: {exc}')
        finally:
            self.core.abort()
            self.core.stop()
            self._refresh()
            if self.contact is not None:
                self.contact.close()
            while not self._commands.empty():
                future, name, _ = self._commands.get_nowait()
                if not future.cancelled():
                    future.set_exception(ConnectionAbortedError('ARDOP owner stopped'))
                self._aux_done(name)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(30)
            if self._thread.is_alive():
                raise RuntimeError('ARDOP station owner still emitting/retiring')
        self._observer = None

    def set_mycall(self, value): return self._call('set_mycall', value)
    def set_gridsquare(self, value): return self._call('set_gridsquare', value)
    def set_listen(self, value): return self._call('set_listen', value)
    def set_protocolmode(self, value): return self._call('set_protocolmode', value)
    def set_bandwidth(self, hz, forced): return self._call('set_bandwidth', hz, forced)
    def set_codec(self, value): return self._call('set_codec', value)
    def set_drivelevel(self, value): return self._call('set_drivelevel', value)
    def set_autobreak(self, value): return self._call('set_autobreak', value)
    def set_arqtimeout(self, value): return self._call('set_arqtimeout', value)
    def set_leader(self, value): return self._call('set_leader', value)
    def set_trailer(self, value): return self._call('set_trailer', value)
    def set_cwid(self, value): return self._call('set_cwid', value)
    def set_radioctrl(self, value): return self._call('set_radioctrl', value)
    def connect(self, target, repeats=5): return self._call('connect', target, repeats)
    def transmit(self, data): return self._call('transmit', bytes(data))
    def disconnect(self): return self._call('disconnect')
    def abort(self): return self._call('abort')
    def purge_buffer(self): return self._call('purge_buffer')
    def break_link(self): return self._call('break_link')
    def send_id(self): return self._call('send_id')
    def two_tone_test(self): return self._call('two_tone_test')
