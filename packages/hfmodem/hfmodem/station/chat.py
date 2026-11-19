# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import json
import os
import queue
import select
import socket
import threading
import time

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_controller import Controller
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.contact.varac import Client, Flow, HostError, Mailbox, Session, CALLING_HZ
from hfmodem.core.waits import join, result, wait_until
from hfmodem.kestrel.host.server import ReportingSession
from .sessions import Refused


class ControlFence:
    def __init__(self, command, *, clock=time.monotonic):
        self.identity = tuple(command[k] for k in ('client', 'control_id', 'generation'))
        self.clock = clock
        self._state = (True, 0.0, None)
        self._lock = threading.Lock()

    def update(self, control, expires):
        with self._lock:
            live = self._state[0] and control is not None and self.identity == tuple(
                control[k] for k in ('client', 'control_id', 'generation'))
            self._state = (live, expires, control.get('expires_utc') if control else None)

    def revoke(self):
        with self._lock:
            self._state = (False, self._state[1], self._state[2])

    def __call__(self):
        live, expires, _ = self._state
        return live and self.clock() < expires

    @property
    def expires_utc(self):
        return self._state[2]

    def check(self):
        if not self():
            raise Refused('authority-stale', 'chat control lease was revoked or expired')


class AttachmentGate:
    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None
        self._retirement = None

    def reserve(self, owner):
        with self._lock:
            if self._retirement is not None and self._retirement():
                self._owner = self._retirement = None
            if self._owner is not None:
                raise Refused('chat-attached', 'VARA attachment exists or awaits retirement')
            self._owner = owner

    def release(self, owner):
        with self._lock:
            if self._owner is not owner:
                raise RuntimeError('attachment reservation mismatch')
            self._owner = None
            self._retirement = None

    def await_retirement(self, owner, check):
        with self._lock:
            if self._owner is not owner:
                raise RuntimeError('attachment reservation mismatch')
            self._retirement = check


class ReceiveSink:
    def __init__(self):
        self._queue = queue.Queue(128)
        self.failed = False
        self.closed = False

    def deliver(self, payload):
        self.deliver_frame(2, payload.encode('ascii') if isinstance(payload, str) else payload)

    def deliver_frame(self, command, payload):
        if self.closed:
            return
        try:
            self._queue.put_nowait((0, command, bytes(payload)))
        except queue.Full:
            self.failed = True

    def drain(self, client):
        if self.failed:
            raise HostError('private receive bridge overflow')
        while not self._queue.empty():
            client.receive_frame(*self._queue.get_nowait())

    def close(self):
        self.closed = True
        while not self._queue.empty():
            self._queue.get_nowait()


class PrivateSession(ReportingSession):
    def __init__(self, *args, fence, **kwargs):
        self.fence = fence
        super().__init__(*args, data_admission=self._admit_data, **kwargs)
        self.submitter_live = True

    def _admit_data(self):
        if not self.fence():
            raise ConnectionAbortedError('chat authority revoked')

    def _dispatch_live(self, command):
        if command not in ('ABORT', 'LISTEN OFF', 'DISCONNECT') and not self.fence():
            raise ValueError('chat authority revoked')
        super()._dispatch_live(command)

    def beacon_submit(self, text):
        if not wait_until(lambda r: self._operation_lock.acquire(True, r), time.monotonic() + 1):
            raise HostError('beacon admission deadline expired before submission')
        try:
            self.fence.check()
            if not self.submitter_live or self._closed:
                raise HostError('chat attachment retired')
            self._modem.send_beacon(text)
        finally:
            self._operation_lock.release()


class Chat:
    def __init__(self, service, station, *, attachment_factory=None, controller_factory=Controller):
        self.service, self.station = service, station
        self.attachment_factory = attachment_factory or Attachment
        self.controller_factory = controller_factory
        self.generation = 0
        self.state = 'idle'
        self.controller = self.attachment = self.session = self.fence = None
        self._reservation = None
        self._start_command = None
        self._thread = None
        self._subscription = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._failure = None
        self._stopped = False
        self._last_controller = None
        from .chat_transport import ResultPages
        self._pages = ResultPages()

    def lifecycle(self):
        with self._lock:
            return dict(incarnation=self.service.incarnation, generation=self.generation,
                        state=self.state, controller_id=self.controller.controller_id if self.controller else None)

    def live(self):
        controller = self.controller
        if controller is None:
            return None
        projection = controller.projection()
        live = s.to_wire(projection.live)
        if len(s.encode(projection.live)) > 7 * 1024:
            live = None
        return dict(controller_id=projection.controller_id, revision=projection.revision,
                    through_seq=projection.through_seq, state=live)

    def _lease(self, command):
        registry = self.station.sessions
        registry.authorize(command)
        control = registry.snapshot()['control']
        if control is None or any(command.get(k) != control[k]
                                  for k in ('client', 'control_id', 'generation')):
            raise Refused('not-controller', 'chat requires an explicit live control lease')

    def validate(self, command):
        args = command.get('args', {})
        if command['kind'] == 'chat.start':
            if (set(args) != {'expected_incarnation', 'expected_lifecycle_generation', 'expected_controller_id'}
                    or not isinstance(args['expected_incarnation'], str)
                    or type(args['expected_lifecycle_generation']) is not int
                    or args['expected_controller_id'] is not None and not isinstance(args['expected_controller_id'], str)):
                raise ValueError('invalid chat.start schema')
            self._lease(command)
            return None
        request = s.from_wire(args)
        if (not isinstance(request, s.Request) or request.id != command['id']
                or request.action.kind != command['kind'] or not request.actor):
            raise ValueError('expected matching typed chat Request')
        controller = self.controller
        if controller is None or controller.controller_id != request.controller_id:
            raise Refused('controller-stale')
        if s.effect(request.action) is not s.Effect.R:
            self._lease(command)
            if (request.control_id, request.control_generation) != (command['control_id'], command['generation']):
                raise Refused('authority-stale')
            self.fence.check()
            if request.link_generation != controller.snapshot().link.generation:
                raise Refused('link-stale')
        return request

    def reserve(self, command):
        if command['kind'] != 'chat.start':
            return
        args = command['args']
        with self._lock:
            if (args['expected_incarnation'] != self.service.incarnation
                    or args['expected_lifecycle_generation'] != self.generation):
                raise Refused('chat-lifecycle-stale')
            expected = args['expected_controller_id']
            if expected is None:
                if self.state != 'idle' or self._reservation is not None or self.controller is not None:
                    raise Refused('chat-attached')
            elif self.state != 'ready' or self.controller is None or expected != self.controller.controller_id:
                raise Refused('controller-stale')
            if self._reservation is not None:
                raise Refused('chat-start-pending')
            self.generation += 1
            self._reservation = command['id']
            if expected is None:
                self.state = 'starting'
                self._start_command = command
                self.fence = ControlFence(command, clock=self.station.sessions.clock)
                self.station.sessions.watch_control(self.fence.update)
                self._stop.clear()
                self._stopped = False
                self._failure = None
                self.session = None
                self._thread = None

    def dispatch(self, command, complete):
        if command['kind'] != 'chat.start':
            request = self.validate(command)
            receipt = self.controller.submit(request)
            if not receipt.queued:
                raise Refused(receipt.reason)
            complete('done', result={'request_id': receipt.id, 'handoff': True})
            return
        with self._lock:
            if self._reservation != command['id']:
                raise Refused('chat-lifecycle-stale')
            attaching = command['args']['expected_controller_id'] is not None
            if not attaching:
                self._thread = threading.Thread(target=self._run, args=(command, complete), name='station-chat-lifecycle')
                self._thread.start()
                return
        try:
            self._lease(command)
            self.fence.check()
            complete('done', result={'controller_id': self.controller.controller_id})
        finally:
            with self._lock:
                self._reservation = None

    def _run(self, command, complete):
        reported = False
        try:
            self.fence.check()
            self.attachment = self.attachment_factory(self)
            self.controller = self.controller_factory(self.attachment.open)
            result(self.controller.ready, time.monotonic() + 30)
            self.fence.check()
            self._subscription = self.controller.subscribe()
            self.state = 'ready'
            self._reservation = None
            complete('done', result={'controller_id': self.controller.controller_id})
            reported = True
            while not self._stop.wait(.02):
                if not self.fence() or self.controller._shutdown is not None:
                    break
        except BaseException as error:
            self._failure = error
        finally:
            cancelled = self._stop.is_set() or not self.fence()
            if cancelled or self._failure is not None or not reported:
                self.fence.revoke()
            self.state = 'closing'
            deadline = time.monotonic() + 30
            cleanup_errors = []
            if self.attachment is not None:
                self.attachment.deadline = deadline
            if self.controller is None and self.attachment is not None:
                try:
                    self.attachment.close()
                except BaseException as error:
                    cleanup_errors.append(error)
            if self.controller is not None:
                try:
                    result(self.controller.shutdown(), deadline)
                except BaseException as error:
                    self._failure = self._failure or error
                if not wait_until(self.controller.join, deadline):
                    cleanup_errors.append(RuntimeError('chat controller did not retire'))
            self.fence.revoke()
            if self.attachment is not None:
                if not getattr(self.attachment, '_closed', False):
                    cleanup_errors.append(RuntimeError('chat owner did not complete resource retirement'))
                if not cleanup_errors:
                    try:
                        self.attachment.release()
                    except BaseException as error:
                        cleanup_errors.append(error)
            if cleanup_errors:
                error = cleanup_errors[0]
                if self._failure is not None:
                    error.__cause__ = self._failure
                self._failure = error
                self.station.stop_transmit()
            else:
                self.station.sessions.unwatch_control(self.fence.update)
            reason = ('chat-cleanup-incomplete' if cleanup_errors else
                      'chat-start-cancelled' if cancelled else
                      getattr(self._failure, 'reason', 'chat-start-failed'))
            status = 'refused' if isinstance(self._failure, Refused) and reason not in (
                'chat-cleanup-incomplete', 'chat-start-cancelled') and self.session is None else 'failed'
            detail = str(self._failure or reason)
            with self._lock:
                self._reservation = None
                if not cleanup_errors:
                    self._last_controller = self.controller
                    self.controller = self.attachment = None
                self.state = 'failed' if cleanup_errors else 'idle'
            if not reported:
                complete(status, reason=reason, detail=detail)

    def revoke(self):
        if self.fence is not None:
            self.fence.revoke()
        self._stop.set()

    def pump(self):
        if self.fence is not None and not self.fence() and not self._stopped:
            self._stopped = True
            if self.session is not None and self.station.arbiter is not None:
                self.station.arbiter.stop(self.session.id)
            self._stop.set()
        if self._subscription is not None:
            for event in self._subscription.drain():
                payload = event.payload
                if isinstance(payload, s.ReadResult):
                    event = replace(event, kind='ResultAvailable', payload=s.ResultAvailable(payload.id))
                if isinstance(payload, s.StateChanged) and len(s.encode(event)) > 7 * 1024:
                    payload = replace(payload, state=None, changed_sections=tuple(set(
                        payload.changed_sections) | {s.ProjectionSection.LIVE}))
                    event = replace(event, payload=payload)
                kind = ('chat.state' if isinstance(payload, s.StateChanged) else
                        'chat.outcome' if isinstance(payload, s.RequestChanged) else 'chat.event')
                self.service.events.publish(kind, s.to_wire(event))

    def read(self, path, query):
        if path == '/chat/result' and 'read_id' in query:
            return self._pages.page(query['read_id'][0], query.get('cursor', [None])[0])
        controller = self.controller or self._last_controller
        if controller is None:
            raise Refused('chat-unavailable')
        if path == '/chat/snapshot':
            value = controller.snapshot()
        elif path == '/chat/projection':
            if query.get('controller_id', [None])[0] != controller.controller_id:
                raise Refused('controller-stale')
            value = controller.projection()
        elif path == '/chat/result':
            value = controller.outcome(query.get('request_id', [''])[0])
        else:
            raise ValueError('unknown chat read')
        if len(s.encode(value)) > 4 * 1024 * 1024:
            return self._pages.capture(s.to_wire(value))
        return s.to_wire(value)

    def close(self, deadline=None):
        self.revoke()
        deadline = deadline if deadline is not None else time.monotonic() + 30
        with self._lock:
            thread = self._thread
            pending = self._start_command if self.state == 'starting' and thread is None else None
            if pending is not None:
                self.state = 'closing'
                self._reservation = None
        if pending is not None:
            self.station.sessions.unwatch_control(self.fence.update)
            self.service._result(pending, 'failed', reason='chat-start-cancelled')
            with self._lock:
                self.state = 'idle'
        if thread is not None:
            if not join(thread, deadline):
                raise RuntimeError('chat teardown deadline; attachment retained')
        if self.state == 'failed':
            raise RuntimeError('chat cleanup incomplete') from self._failure


class Attachment:
    def __init__(self, chat):
        self.chat, self.station, self.fence = chat, chat.station, chat.fence
        self.runtime = None
        self.session = self.client = self.host = self.worker = None
        self.capture = self.bbs = None
        self.deadline = None
        self._startup_deadline = time.monotonic() + 30
        self._initializing = True
        self._held = ExitStack()
        self._reserved = self._closed = self._cleaning = False
        self._armed = False
        self._authority = None
        self._diagnostics = queue.Queue(256)
        self._diagnostic_lost = 0
        self._diagnostic_lock = threading.Lock()
        self._stream = None
        self._host_error = None
        self.sink = ReceiveSink()

    def line_state(self):
        return {'low': False, 'high': True}.get(self.station.rig.line_state)

    def allowed(self):
        return self._armed and self.fence()

    def arm(self, on):
        if on:
            self.fence.check()
            if not self.station.is_ready or self.line_state() is not False:
                raise HostError('station is not ready with a verified low line')
        self._armed = bool(on)

    def _ready(self, streams, timeout):
        if not self._cleaning:
            self.fence.check()
            if self._initializing and time.monotonic() >= self._startup_deadline:
                raise HostError('chat startup deadline expired')
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise HostError('chat cleanup deadline expired')
        if self.runtime.failure is not None or self._host_error is not None:
            raise HostError(f'chat host failed: {self.runtime.failure or self._host_error}')
        if self.capture is not None:
            self.capture.drain()
        if self.client is not None:
            self.sink.drain(self.client)
        if self.session is not None:
            while not self._diagnostics.empty():
                kind, fields = self._diagnostics.get_nowait()
                self.session.diagnostic(kind, **fields)
            with self._diagnostic_lock:
                lost = self._diagnostic_lost
                self._diagnostic_lost = 0
            if lost:
                self.session.diagnostic('dropped', count=lost)
        return select.select(streams, [], [], min(timeout, .05))[0]

    def _diagnostic(self, kind, **fields):
        try:
            self._diagnostics.put_nowait((kind, dict(fields)))
        except queue.Full:
            with self._diagnostic_lock:
                self._diagnostic_lost += 1

    def _emit(self, kind, **fields):
        if self._stream is not None:
            self._stream.write(json.dumps(dict(kind=kind, **fields)) + '\n')
            self._stream.flush()
        if kind == 'contact' and self.capture is not None:
            self.capture.contact(fields, deadline=self.deadline)

    def _say(self, text):
        self._emit('notice', text=text)

    def _host_run(self):
        try:
            self.host.run()
        except BaseException as error:
            self._host_error = error

    def open(self):
        from hfmodem.bbs.service import open_service
        from hfmodem.bbs.config import load as load_bbs_config
        from hfmodem.bbs.store import OwnershipError
        from hfmodem.contact.slots import SlotWatch
        from hfmodem.core.band import AMATEUR_HF_BANDS
        from hfmodem.core.wav import CaptureRecorder
        from .chat_capture import ChatCapture, CaptureStartupCleanup
        from .contactlog import ContactLog

        try:
            self.fence.check()
            link = self.station.lanes.get('vara')
            if link is None or self.station.arbiter is None or self.station.rig is None:
                raise Refused('chat-unavailable', 'owned station VARA lane required')
            self.runtime = link.runtime
            self.runtime.attachments.reserve(self)
            self._reserved = True
            config = self.station.cfg
            bbs_path = config.bbs.config
            if bbs_path:
                bbs_path = Path(bbs_path).expanduser()
                if not bbs_path.is_absolute():
                    if not self.station.station_file:
                        raise Refused('bbs-config', 'injected BBS configuration path must be absolute')
                    bbs_path = Path(self.station.station_file).parent / bbs_path
                try:
                    load_bbs_config(bbs_path)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    raise Refused('bbs-config', str(error)) from error
                try:
                    self.bbs = open_service(bbs_path)
                except OwnershipError as error:
                    raise Refused('bbs-owned', str(error)) from error
            root = Path(os.environ.get('HFMODEM_DATA_DIR', 'operator-data'))
            directory = root / 'chat' / self.chat.service.incarnation / str(self.chat.generation)
            directory.mkdir(parents=True, exist_ok=False)
            self.chat.session = self.station.sessions.start(
                kind='chat', lane='vara', attended=True, holds_rig=True,
                directory=directory, client=self.fence.identity[0],
                recording_origin_index=int(self.station.audio.samples))
            self._authority = self.chat.session.authority
            self._stream = self._held.enter_context((directory / 'events.jsonl').open('w'))
            dial = self.station.rig.dial()
            band = next((m for lo, hi, m in AMATEUR_HF_BANDS if lo * 1e6 <= dial <= hi * 1e6), None)
            if band not in CALLING_HZ:
                raise Refused('chat-band', 'current station band has no VarAC calling frequency')
            slots = SlotWatch(dial, CALLING_HZ[band], mycall=config.station.mycall,
                              passband=config.rig.filter_hz)
            recorder = None
            try:
                recorder = CaptureRecorder(directory / 'receive.wav', rate=48000, source='station chat')
                self.capture = ChatCapture(self.station.audio, recorder, slots,
                    keyed=lambda: self.line_state() is not False, contact_log=ContactLog(directory))
            except BaseException:
                # Ownership has not transferred to ChatCapture yet. Retain the
                # partial consumers and their bounded cleanup worker just as
                # for a fully constructed capture; never compress on this owner.
                self.capture = CaptureStartupCleanup(slots, recorder)
                raise
            evidence = self._held.enter_context(self.runtime.owner.subscribe_payload_evidence())
            self._held.callback(self.station.diagnostics.subscribe(self._authority, self._diagnostic))
            command, host_command = socket.socketpair()
            data, host_data = socket.socketpair()
            for stream in (command, host_command, data, host_data):
                stream.settimeout(1)
                self._held.callback(stream.close)
            modem = self.runtime.modem_factory(authority=self._authority, fence=self.allowed)
            self.host = PrivateSession(host_command, host_data, modem, kiss=self.sink, fence=self.fence)
            self.worker = threading.Thread(target=self._host_run, name='station-chat-adapter')
            self.worker.start()
            self.client = Client(command, data, ready=self._ready, beacon_submit=self.host.beacon_submit)
            self.client.initialize(config.station.mycall)
            self.client.enable_writer()
            self.client.data_writer.before_write = self.fence.check
            flow = Flow(self.client, self.tune, band=band,
                        profile={'NAME': config.station.name, 'QTH': config.station.qth, 'LOC': config.station.grid})
            flow.dial = dial
            self.session = Session(self.client, flow, say=self._say, emit=self._emit,
                arm=self.arm, line_state=self.line_state, mailbox=Mailbox(root / 'vmail'),
                slots=slots, band_tuner=self.tune, identify=self.identify,
                idcall=config.station.idcall or config.station.mycall,
                power=self.power if self.station.rig.cat is not None else None,
                power_clamp_w=config.rig.power_clamp_w,
                bbs_core=self.bbs.core if self.bbs else None,
                bbs_catalog=self.bbs.catalog if self.bbs else None,
                bbs_reload=self.bbs.reload if self.bbs else None)
            self._held.callback(self.session.parking.store.close)
            from hfmodem.contact.chat_session import received_retune_adapter
            flow.received_retune_steps = received_retune_adapter(
                self.runtime.owner, self.client, lambda hz: self._tune_boundary(hz, received=True),
                lambda: self.line_state() is False, completed=flow.received_tuned,
                quitting=lambda: self._cleaning or not self.fence())
            adapter = AttachedSession(self, delivery_evidence=evidence,
                                      delivery_boundary=self.host.delivery_boundary)
            self._hold_power()
            self.session.monitor()
            self._initializing = False
            self.station.sessions.running(self.chat.session.id)
            return adapter
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise error from cleanup
            raise

    def power(self, watts=None):
        from hfmodem.core.rig import RigError
        from hfmodem.core.rigs import CAT_BACKENDS
        import math
        self.fence.check()
        rig, config = self.station.rig, self.station.cfg
        if rig.cat is None:
            return None
        deadline = rig.cat.clock() + 48000
        with rig.operations.lease(), rig.hold(deadline) as held:
            self.fence.check()
            if not held:
                raise HostError('station CAT lease refused')
            if watts is not None:
                if self.line_state() is not False or self.client.ptt:
                    raise HostError('power change requires a verified low line')
                from hfmodem.core.config import clamp_power
                watts = clamp_power(watts, config.rig.power_clamp_w, '/power')[0]
                try:
                    rig.cat.set_power(watts)
                except RigError as error:
                    raise OSError(str(error)) from error
            self.fence.check()
            try:
                if CAT_BACKENDS[config.rig.cat] == 'serial':
                    reply = rig.cat.read('PC;', 'PC')
                    value = float(reply)
                else:
                    reply = rig.cat.command('l RFPOWER')
                    values = [line.strip() for line in reply.splitlines()
                              if line.strip() and ':' not in line and not line.startswith('RPRT ')]
                    if len(values) != 1:
                        return None
                    value = float(values[0]) * config.full_scale_w
                return value if math.isfinite(value) and value >= 0 else None
            except (RigError, ValueError):
                return None

    def _hold_power(self):
        clamp = self.station.cfg.rig.power_clamp_w
        if clamp is None or self.station.rig.cat is None:
            return
        found = self.power()
        if found is None:
            self._say(f'warning: RFPOWER did not read, so the {clamp:g} W station clamp is unchecked')
        elif found > clamp + .5:
            self._say(f'RFPOWER reads {found:g} W, above the {clamp:g} W station clamp; lowering it to {clamp:g} W')
            self.session._apply_power(clamp, clamp)

    def tune(self, hz):
        self.fence.check()
        timeout = min(10., max(0., (self.deadline or time.monotonic() + 10.) - time.monotonic()))
        return self.runtime.retune(lambda: self._tune_boundary(hz), valid=self.fence, timeout=timeout)

    def _tune_boundary(self, hz, *, received=False):
        from hfmodem.core.band import centre_hz, QSY_TOLERANCE_HZ
        from hfmodem.core.rigs import RIGS
        self.fence.check()
        arbiter, rig = self.station.arbiter, self.station.rig
        deadline = min(self.deadline or float('inf'), time.monotonic() + 10.)
        token = self.runtime.transport.contact.token
        hold = (arbiter.try_contact_hold('vara', token, authority=self._authority,
                fence=self.host._modem.control_fence, paused=lambda: received or self.runtime.paused)
                if token is not None else arbiter.try_hold())
        with hold as acquired:
            if not acquired or self.line_state() is not False:
                raise HostError('station retune admission refused')
            self.fence.check()
            if time.monotonic() >= deadline:
                raise HostError('station retune deadline expired')
            profile = RIGS.get(rig.model, {})
            desired_width = profile.get('rx_width_hz', 3000)
            attempted = False
            try:
                # The rig and CAT leases keep shutdown from closing resources
                # between dial movement, band-stack restoration and readback.
                with rig.operations.lease(), rig.hold(
                        rig.cat.clock() + max(0., deadline-time.monotonic()) * 48000) as cat_held:
                    if not cat_held:
                        raise HostError('station CAT lease refused')
                    self.fence.check()
                    attempted = True
                    rig.tune(centre_hz(hz))
                    if rig.cat.set_mode('PKTUSB', 0 if 'rx_width_hz' in profile else desired_width) is False:
                        raise HostError('station CAT refused packet mode')
                    if rig.model == 'ftdx10':
                        from hfmodem.core.ftdx10 import filter_commands
                        for command in filter_commands(profile):
                            rig.cat.write(command)
                        # The band's filter stack answers setters before it
                        # settles. Recheck authority throughout that interval.
                        settle = time.monotonic() + 1.
                        while time.monotonic() < settle:
                            self.fence.check()
                            if time.monotonic() >= deadline:
                                raise HostError('station band setup deadline expired')
                            time.sleep(max(0., min(.05, settle-time.monotonic())))
                    mode, roof = rig.cat.receiver_filters()
                    if (abs(rig.cat.freq() - hz) > QSY_TOLERANCE_HZ
                            or mode.mode != 'PKTUSB' or (mode.passband_hz or 0) < 500):
                        raise HostError('station tune/mode/filter readback failed')
                    if rig.model == 'ftdx10':
                        shift = rig.cat.read('IS0;', 'IS0')
                        if (mode.passband_hz != desired_width or roof != profile['roofing_hz']
                                or shift != f"0{profile['if_shift_hz']:+05d}"):
                            raise HostError('station band filter/roofing/shift readback failed')
                    self.fence.check()
                    if time.monotonic() >= deadline:
                        raise HostError('station tune readback deadline expired')
                    bandwidth = self.client.bandwidth if self.client is not None else 500
                    if mode.passband_hz < int(bandwidth):
                        self._say(f'warning: rig passband {mode.passband_hz} Hz is narrower than '
                                  f'the {bandwidth} Hz session')
            finally:
                if attempted:
                    self.runtime.transport.retuned()
            return dict(verified=True, atu_match_known=False)

    def identify(self, bandwidth, after_peer):
        from .arbiter import ERRAND, RequestState, TxRequest
        from hfmodem.core import cwid
        from hfmodem.core.regulatory import centred
        cancel, state = threading.Event(), RequestState()
        errors = []
        completed = threading.Event()
        def work():
            try:
                start = time.monotonic()
                while after_peer and (time.monotonic() - start < .5 or
                        self.runtime.transport.receiving and time.monotonic() - start < 8):
                    self.fence.check()
                    if cancel.wait(.05):
                        return
                self.fence.check()
                if cancel.is_set():
                    return
                arbiter = self.station.arbiter
                identity = arbiter.identity
                if identity is None:
                    raise HostError('station identification is not configured')
                request = TxRequest(ERRAND, -1, protocol='ident', audio=identity.burst(),
                    emission=centred(self.station.rig.dial(), cwid.BANDWIDTH_HZ,
                        centre_hz=cwid.TONE_HZ, power_w=self.station.cfg.declared_power_w),
                    authority=self._authority, control_fence=self.host._modem.control_fence,
                    state=state, settle_s=self.station.cfg.rig.ptt.settle_s)
                result = arbiter.submit_result(request)
                if result.outcome != 'sent' or result.end is None or result.end <= 0:
                    raise HostError('station identification has no completion evidence: '
                                    f'{result.outcome}: {result.reason}')
                # TxArbiter records note_sent only after positive completion.
                completed.set()
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=work, name='station-chat-ident')
        worker.start()
        end = min(self.deadline or float('inf'), time.monotonic() + 20.)
        try:
            while worker.is_alive() and time.monotonic() < end:
                self.fence.check()
                yield .05
        finally:
            cancel.set()
            state.cancel()
            if worker.is_alive():
                self.station.arbiter.stop(self.chat.session.id)
            now = time.monotonic()
            if not join(worker, min(now + 3, self.deadline or now + 3)):
                raise HostError('chat identification worker did not retire')
            if errors:
                raise HostError(f'chat identification failed: {errors[0]}') from errors[0]
            if not completed.is_set():
                raise HostError('chat identification did not complete')

    def close(self, *, unwinding=None):
        if self._closed:
            return
        self._cleaning = True
        self.deadline = self.deadline or time.monotonic() + 30
        failures = []
        def close(callback):
            try:
                callback()
            except BaseException as error:
                failures.append(error)
        if self.session is not None and not getattr(self, '_quit_called', False):
            # open() can fail after creating Session and then be cleaned again
            # by Controller's factory-failure path. Quit changes backend state
            # and must never be replayed by that retry.
            self._quit_called = True
            try:
                self.session.quit(unwinding=unwinding)
            except BaseException as error:
                self._quit_error = error
        self._armed = False
        if getattr(self, '_quit_error', None) is not None:
            failures.append(self._quit_error)
        if self.client is not None:
            close(lambda: self.client.close(deadline=self.deadline))
            writer = getattr(self.client, 'data_writer', None)
            if writer is not None:
                # DataWriter.close's short compatibility wait is not proof of
                # retirement. Closing Client's sockets should wake a blocked
                # write; retain admission until its actual worker has exited.
                def retire_writer():
                    if not join(writer._thread, self.deadline):
                        raise RuntimeError('chat data writer did not retire')
                close(retire_writer)
        if self.host is not None:
            self.host.submitter_live = False
            close(lambda: self.host.close(deadline=self.deadline))
        if self.worker is not None:
            if not join(self.worker, self.deadline):
                failures.append(RuntimeError('chat host worker did not retire'))
            else:
                close(lambda: self.host.ensure_retired(deadline=self.deadline))
                while not self.host._modem._retired_attachment and time.monotonic() < self.deadline:
                    time.sleep(.01)
                if not self.host._modem._retired_attachment:
                    failures.append(RuntimeError('runtime did not acknowledge modem retirement'))
        close(self.sink.close)
        if self.capture is not None:
            close(lambda: self.capture.close(deadline=self.deadline))
        if self.bbs is not None:
            close(lambda: self.bbs.close(deadline=self.deadline))
        if not getattr(self, '_held_closed', False):
            self._held_closed = True
            try:
                self._held.close()
            except BaseException as error:
                self._held_error = error
        if getattr(self, '_held_error', None) is not None:
            failures.append(self._held_error)
        # A refusal before reserving an attachment never owned a transmitter.
        # In particular, a station with no rig must be able to refuse startup
        # without turning that expected refusal into incomplete cleanup.
        if self._reserved or self.chat.session is not None or self.host is not None:
            try:
                if self.line_state() is not False:
                    failures.append(RuntimeError('chat teardown lacks low-line proof'))
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError('; '.join(map(str, failures))) from failures[0]
        self._closed = True

    def release(self):
        if not self._closed:
            raise RuntimeError('chat resources have not retired')
        if self.chat.session is not None and not getattr(self, '_authority_revoked', False):
            self.station.arbiter.stop(self.chat.session.id)
            self._authority_revoked = True
        if self.chat.session is not None and not getattr(self, '_session_ended', False):
            self.station.sessions.end(self.chat.session.id, outcome='stopped', verdict={'line': 'low'})
            self._session_ended = True
        if self._reserved:
            self.runtime.attachments.release(self)
            self._reserved = False


class AttachedSession(SessionAdapter):
    def __init__(self, attachment, **kwargs):
        self.attachment = attachment
        super().__init__(attachment.session, **kwargs)

    def _client_semantic(self, kind, **details):
        if kind in ('command_submitting', 'write_submitting', 'operation_started'):
            command = details.get('command')
            if not (self.attachment._cleaning and command in ('LISTEN OFF', 'ABORT', 'DISCONNECT')):
                self.attachment.fence.check()
        super()._client_semantic(kind, **details)

    def _semantic(self, kind, **details):
        if kind == 'operation_resuming' and not self.attachment._cleaning:
            self.attachment.fence.check()
        super()._semantic(kind, **details)

    def execute(self, request):
        if s.effect(request.action) is not s.Effect.R:
            self.attachment.fence.check()
        return super().execute(request)

    def poll(self, timeout_s):
        self.attachment.fence.check()
        return super().poll(timeout_s)

    def project(self):
        result = super().project()
        a = self.attachment
        client, control, generation = a.fence.identity
        return replace(result, ownership=s.Ownership('station', client, control, generation,
                       a.fence.expires_utc, 'ready' if a.fence() else 'revoked', a.fence() and not a._cleaning))

    def quit(self, *, unwinding=None):
        if not self._quit:
            self._quit = True
            try:
                self.attachment.close(unwinding=unwinding)
                self._finish_exit()
            finally:
                self._observe_line()
                self._flush()
