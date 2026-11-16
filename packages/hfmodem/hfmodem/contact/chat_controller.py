# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from threading import Condition, Lock, Thread, get_ident
import sys
import time
from typing import Callable, Protocol
from uuid import uuid4

from . import chat_state as s

TERMINAL = frozenset(('done', 'refused', 'failed', 'cancelled', 'unknown'))
PRIORITY = frozenset(('chat.cancel_wait', 'chat.close', 'chat.exit'))


@dataclass(frozen=True, slots=True)
class Disposition:
    state: str
    result: s.ApplicationResult | s.ReadValue | None = None
    stage: str | None = None
    reason_code: str | None = None
    detail: str = ''
    bound_generation: int | None = None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class _LegacyLine:
    line: str | None
    completion: Future


class SessionPort(Protocol):
    on_event: Callable

    def poll(self, timeout_s: float) -> None: ...
    def project(self) -> s.Snapshot: ...
    def preflight(self, action: s.OperationalAction) -> s.Capability: ...
    def execute(self, request: s.Request) -> Disposition: ...
    def legacy_line(self, line: str) -> bool: ...
    def quit(self, *, unwinding: bool = False) -> None: ...


class Subscription:
    def __init__(self, controller_id, after_seq, *, max_events=128, max_bytes=256 * 1024):
        self.controller_id = controller_id
        self.after_seq = after_seq
        self.max_events, self.max_bytes = max_events, max_bytes
        self._items = deque()
        self._bytes = 0
        self._lock = Lock()
        self.closed = False

    def put(self, event):
        with self._lock:
            if self.closed or event.seq <= self.after_seq:
                return
            size = len(s.encode(event))
            if len(self._items) >= self.max_events or self._bytes + size > self.max_bytes:
                after = self._items[0][0].seq - 1 if self._items else self.after_seq
                self._items.clear()
                self._bytes = 0
                event = replace(event, kind='gap', request_id=None,
                                payload=s.Gap(after, event.seq, 'subscriber overflow'))
                size = len(s.encode(event))
            self._items.append((event, size))
            self._bytes += size

    def drain(self):
        with self._lock:
            values = tuple(event for event, _ in self._items)
            self._items.clear()
            self._bytes = 0
            if values:
                self.after_seq = values[-1].seq
            return values

    def close(self):
        with self._lock:
            self.closed = True
            self._items.clear()
            self._bytes = 0


class Controller:
    _require_low_line = True

    def __init__(self, port: SessionPort | Callable[[], SessionPort], *, clock=time.monotonic,
                 utc=lambda: datetime.now(timezone.utc).isoformat(), controller_id=None,
                 max_records=2000, history_bytes=2 * 1024 * 1024, record_bytes=48 * 1024):
        self.controller_id = controller_id or str(uuid4())
        self.clock, self.utc = clock, utc
        self.max_records, self.history_bytes, self.record_bytes = max_records, history_bytes, record_bytes
        self._factory = port
        self._legacy = deque()
        self._condition = Condition()
        self._queue, self._priority = deque(), deque()
        self._checks = deque()
        self._submitted = {}
        self._journal = {}
        self._text_outcomes = {}
        self._text_requests = {}
        self._exit_results = {}
        self._subscribers = []
        self._events = deque()
        self._event_bytes = 0
        self._event_sizes = deque()
        self._record_sizes = deque()
        self._record_bytes = 0
        self._wire_sizes = {}
        self._snapshot_overhead = (2 + len('"_type":') + len(s.encode('Snapshot'))
                                   + sum(len(s.encode(f.name)) + 2 for f in fields(s.Snapshot)))
        self._seq = 0
        self._revision = 0
        self._dirty = set()
        self._last_publish = float('-inf')
        self._presentation = s.initial_snapshot().presentation
        self._snapshot = replace(s.initial_snapshot(), controller_id=self.controller_id)
        self._projection = s.project(self._snapshot)
        self._accepting = True
        self._shutdown = None
        self._unwinding = False
        self._ready = Future()
        self._ready.set_running_or_notify_cancel()
        self._start()

    def _start(self):
        self._thread = Thread(target=self._run, name='chat-owner')
        self._thread.start()

    @property
    def ready(self):
        return self._ready

    def snapshot(self):
        with self._condition:
            return self._snapshot

    def projection(self):
        with self._condition:
            return self._projection

    def outcome(self, request_id):
        with self._condition:
            return self._journal.get(request_id, s.NotFound(request_id))

    def submit(self, request: s.Request) -> s.Receipt:
        try:
            s.from_wire(s.to_wire(request))
            s.effect(request.action)
        except (TypeError, ValueError, AttributeError) as error:
            return s.Receipt(getattr(request, 'id', ''), False, str(error))
        with self._condition:
            old = self._submitted.get(request.id)
            if old is not None:
                return s.Receipt(request.id, old == request,
                                 None if old == request else 'request-id-conflict')
            if request.controller_id != self.controller_id:
                return s.Receipt(request.id, False, 'controller-stale')
            if not self._accepting:
                return s.Receipt(request.id, False, 'controller-closing')
            queue = self._priority if request.action.kind in PRIORITY else self._queue
            limit = 3 if queue is self._priority else 64
            size = len(queue) + (len(self._legacy) if queue is self._queue else 0)
            if size >= limit or (queue is self._priority and any(
                    item.action.kind == request.action.kind for item in queue)):
                return s.Receipt(request.id, False, 'queue-full')
            self._submitted[request.id] = request
            queue.append(request)
            self._condition.notify()
        return s.Receipt(request.id, True, None)

    def check(self, action):
        future = Future()
        with self._condition:
            if not self._accepting or len(self._checks) >= 64:
                future.set_exception(RuntimeError('controller unavailable'))
            else:
                self._checks.append((action, future))
                self._condition.notify()
        return future

    def _legacy_line(self, line):
        future = Future()
        with self._condition:
            if not self._accepting or len(self._legacy) + len(self._queue) >= 64:
                future.set_exception(RuntimeError('controller unavailable'))
            else:
                self._legacy.append(_LegacyLine(line, future))
                self._condition.notify()
        return future

    def subscribe(self, after_seq=0):
        with self._condition:
            sub = Subscription(self.controller_id, after_seq)
            if self._events and after_seq < self._events[0].seq - 1:
                e = self._events[0]
                sub.put(replace(e, kind='gap', payload=s.Gap(after_seq, e.seq, 'history unavailable')))
            for event in self._events:
                sub.put(event)
            self._subscribers.append(sub)
            return sub

    def shutdown(self, *, unwinding=False):
        with self._condition:
            self._unwinding |= unwinding
            if self._shutdown is None:
                self._shutdown = Future()
                self._shutdown.set_running_or_notify_cancel()
                self._accepting = False
                self._condition.notify_all()
            return self._shutdown

    def join(self, timeout=None):
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _assert_owner(self):
        if get_ident() != self._owner:
            raise RuntimeError('SessionPort access outside controller owner')

    def _publish(self, payload, request_id=None, generation=None):
        self._assert_owner()
        self._seq += 1
        if isinstance(payload, s.StateChanged):
            payload = replace(payload, through_seq=self._seq)
        event = s.EventEnvelope(self._seq, self.controller_id, self.utc(), self.clock(),
                                generation, request_id, type(payload).__name__, payload)
        with self._condition:
            self._events.append(event)
            size = len(s.encode(event))
            self._event_sizes.append(size)
            self._event_bytes += size
            while len(self._events) > 128 or self._event_bytes > 256 * 1024:
                self._events.popleft()
                self._event_bytes -= self._event_sizes.popleft()
            for sub in self._subscribers:
                sub.put(event)
            self._subscribers = [sub for sub in self._subscribers if not sub.closed]
        return event

    def _request_change(self, request, disposition):
        with self._condition:
            old = self._journal.get(request.id)
            if old and old.state.state in TERMINAL:
                return
            previous = old.state if old else None
            state = s.RequestState(
                request.id, request.action.kind, request.link_generation,
                disposition.bound_generation if disposition.bound_generation is not None else
                previous.bound_generation if previous else None,
                disposition.attempt_id or (previous.attempt_id if previous else None),
                disposition.state, disposition.stage, disposition.reason_code, disposition.detail,
                self.clock())
            self._journal[request.id] = s.RequestResult(
                state, disposition.result if disposition.result is not None else old.result if old else None,
                old.text_outcomes if old else ())
        self._publish(s.RequestChanged(state), request.id, state.bound_generation)
        if disposition.result is not None:
            payload = (s.ReadResult(request.id, disposition.result) if s.effect(request.action) is s.Effect.R
                       else s.ApplicationOutcome(disposition.result))
            if len(s.encode(payload)) > 63 * 1024:
                payload = s.ResultAvailable(request.id)
            self._publish(payload, request.id, state.bound_generation)

    def _event(self, payload, request_id=None, generation=None):
        self._assert_owner()
        if isinstance(payload, Disposition):
            request = self._submitted.get(request_id)
            if request is not None:
                if isinstance(request.action, s.ChatExit) and payload.state in TERMINAL:
                    self._exit_results[request.id] = payload
                else:
                    self._request_change(request, payload)
        elif isinstance(payload, (s.TextRecord, s.Notice, s.SlotRecord, s.BBSPage)):
            self._display(payload, request_id, generation)
        elif isinstance(payload, s.TxTextOutcome):
            self._text_outcomes[payload.text_id] = payload
            records = []
            for index, record in enumerate(self._presentation.records):
                if record.id == payload.text_id:
                    record = replace(record, updated_seq=self._seq + 1, tx=payload)
                    size = len(s.encode(record))
                    self._record_bytes += size - self._record_sizes[index]
                    self._record_sizes[index] = size
                    self._publish(s.DisplayChanged(record), request_id, generation)
                records.append(record)
            self._presentation = self._retain_records(self._presentation, records)
            rid = self._text_requests.get(payload.text_id) or request_id
            if rid is not None:
                self._text_requests[payload.text_id] = rid
            with self._condition:
                if rid in self._journal:
                    result = self._journal[rid]
                    outcomes = {o.text_id: o for o in result.text_outcomes}
                    outcomes[payload.text_id] = payload
                    self._journal[rid] = replace(result, text_outcomes=tuple(outcomes.values()))
            self._publish(payload, request_id, generation)
        else:
            self._publish(payload, request_id, generation)
        self._refresh()

    def _display(self, payload, request_id, generation):
        p = self._presentation
        is_page = isinstance(payload, s.BBSPage)
        truncated = False
        if len(s.encode(payload)) > self.record_bytes - 1024:
            marker = '\n[content omitted]'
            if is_page:
                text = ''.join(span.text for span in payload.spans)
            else:
                text = payload.text
            low, high = 0, len(text)
            while low < high:
                mid = (low + high + 1) // 2
                short = text[:mid] + marker
                candidate = (replace(payload, spans=(s.StyledSpan(short, None, None, False),))
                             if is_page else replace(payload, text=short))
                if len(s.encode(candidate)) <= self.record_bytes - 1024:
                    low = mid
                else:
                    high = mid - 1
            short = text[:low] + marker
            payload = (replace(payload, spans=(s.StyledSpan(short, None, None, False),))
                       if is_page else replace(payload, text=short))
            truncated = True
        ident = (payload.page_id if is_page else payload.id if isinstance(payload, s.TextRecord)
                 else str(uuid4()))
        revision = p.bbs_revision + 1 if is_page else None
        record = s.DisplayRecord(ident, self._seq + 1, self._seq + 1, generation,
                                 p.bbs_context if is_page else None, revision, payload,
                                 self._text_outcomes.get(ident))
        self._text_requests[ident] = request_id
        size = len(s.encode(record))
        self._record_sizes.append(size)
        self._record_bytes += size
        records = p.records + (record,)
        p = replace(p, bbs_revision=revision if is_page else p.bbs_revision,
                    bbs_complete=p.bbs_complete and not (is_page and truncated),
                    bbs_reconnect_required=p.bbs_reconnect_required or (is_page and truncated))
        self._presentation = self._retain_records(p, records)
        self._publish(s.DisplayChanged(record), request_id, generation)

    def _retain_records(self, presentation, records):
        evicted = presentation.evicted_through
        lost = False
        count = 0
        while count < len(records) and (len(records) - count > self.max_records
                                       or self._record_bytes > self.history_bytes):
            removed = records[count]
            count += 1
            self._record_bytes -= self._record_sizes.popleft()
            evicted = max(evicted or 0, removed.created_seq)
            lost |= removed.bbs_context is not None and removed.bbs_context == presentation.bbs_context
        records = records[count:]
        return replace(presentation, records=tuple(records), evicted_through=evicted,
                       bbs_complete=presentation.bbs_complete and not lost,
                       bbs_reconnect_required=presentation.bbs_reconnect_required or lost)

    def _refresh(self):
        self._assert_owner()
        raw = self._port.project()
        old = self._snapshot
        p = self._presentation
        if raw.link.generation != old.link.generation and not raw.link.connected:
            p = replace(p, bbs_context=None, bbs_revision=0, bbs_complete=True,
                        bbs_reconnect_required=False)
        remote = raw.application.bbs.remote_view
        if remote and p.bbs_context is None:
            p = replace(p, bbs_context=str(uuid4()), bbs_revision=0)
        elif not remote and p.bbs_context is not None:
            p = replace(p, bbs_context=None, bbs_revision=0)
        if (raw.application.bbs.remote_ansi, raw.application.bbs.dimensions) != (
                old.application.bbs.remote_ansi, old.application.bbs.dimensions):
            p = replace(p, bbs_revision=p.bbs_revision + 1)
        self._presentation = p
        with self._condition:
            ops = tuple(v.state for v in self._journal.values()
                        if s.ACTIONS[v.state.kind][1] is not s.Effect.R)
        pending = tuple(r for r in ops if r.state not in TERMINAL)
        terminal = tuple(r for r in ops if r.state in TERMINAL)[-128:]
        raw = replace(raw, controller_id=self.controller_id, presentation=p,
                      requests=pending + terminal, revision=old.revision, through_seq=self._seq,
                      sampled_at=old.sampled_at, utc=old.utc,
                      ownership=replace(raw.ownership, accepting=self._accepting))
        raw = self._bound_collections(raw)
        fields = ('identity', 'link', 'operation', 'tx', 'policy', 'beacon', 'broadcast',
                  'application', 'receive', 'ownership', 'capabilities', 'requests', 'presentation')
        changed = {f for f in fields if getattr(old, f) != getattr(raw, f)}
        if changed:
            self._revision += 1
            mapping = {'application': 'application', 'receive': 'receive',
                       'capabilities': 'capabilities', 'requests': 'requests'}
            self._dirty.update(s.ProjectionSection(mapping.get(f, 'live')) for f in changed)
        sampled = self.clock() if changed else old.sampled_at
        stamp = self.utc() if changed else old.utc
        raw = replace(raw, revision=self._revision, through_seq=self._seq,
                      sampled_at=sampled, utc=stamp)
        raw = self._bound_collections(raw)
        with self._condition:
            self._snapshot, self._projection = raw, s.project(raw)
        if self._dirty and self.clock() - self._last_publish >= .2:
            state = s.live(raw)
            event = s.StateChanged(self._revision, self._seq + 1, state,
                                   tuple(sorted(self._dirty, key=lambda x: x.value)))
            if len(s.encode(event)) > 7 * 1024:
                event = replace(event, state=None)
            self._publish(event)
            self._dirty.clear()
            self._last_publish = self.clock()
            with self._condition:
                self._snapshot = replace(raw, through_seq=self._seq)
                self._projection = s.project(self._snapshot)

    def _snapshot_bytes(self, raw):
        total = self._snapshot_overhead
        for field in fields(raw):
            value = getattr(raw, field.name)
            if field.name == 'presentation':
                value = replace(value, records=())
                total += self._record_bytes + max(0, len(raw.presentation.records) - 1)
            cached = self._wire_sizes.get(field.name)
            if cached is None or cached[0] != value:
                cached = self._wire_sizes[field.name] = (value, len(s.encode(value)))
            total += cached[1]
        return total

    def _bound_collections(self, raw):
        if self._snapshot_bytes(raw) <= 4 * 1024 * 1024:
            return raw
        omitted = list(raw.omitted_sections)
        for section, field in (('receive', 'heard'), ('receive', 'slots'),
                               ('application', 'mailbox'), ('application', 'notices')):
            obj = getattr(raw, section)
            if getattr(obj, field):
                omitted.append(section + '.' + field)
                raw = replace(raw, **{section: replace(obj, **{field: ()})},
                              omitted_sections=tuple(omitted))
            if self._snapshot_bytes(raw) <= 4 * 1024 * 1024:
                return raw
        if raw.application.parking and raw.application.parking.mail:
            omitted.append('application.parking.mail')
            raw = replace(raw, application=replace(raw.application,
                          parking=replace(raw.application.parking, mail=())),
                          omitted_sections=tuple(omitted))
            if self._snapshot_bytes(raw) <= 4 * 1024 * 1024:
                return raw
        if raw.application.files.entries:
            raw = replace(raw, application=replace(raw.application,
                          files=replace(raw.application.files, entries=())),
                          omitted_sections=tuple(omitted + ['application.files.entries']))
        if self._snapshot_bytes(raw) > 4 * 1024 * 1024:
            raise ValueError('required state exceeds snapshot budget')
        return raw

    def _read(self, action):
        snap = self._snapshot
        values = {
            'chat.snapshot': snap, 'chat.projection': self._projection,
            'broadcast.status': snap.broadcast, 'beacon.hop.status': snap.beacon.hop,
            'email.status': snap.application.email,
            'pskreporter.status': s.ReporterStatus(snap.policy.reporter_status)}
        if isinstance(action, s.ChatOutcome):
            return self.outcome(action.request_id)
        if isinstance(action, s.ChatCheck):
            return self._check(action.action)
        return values.get(action.kind)

    def _check(self, action):
        if s.effect(action) is s.Effect.R:
            raise ValueError('check requires an operational action')
        return s.CapabilityCheck(self.controller_id, self._revision, self._seq,
                                 action, self._port.preflight(action))

    def _execute(self, request):
        mutation = s.effect(request.action) is not s.Effect.R
        snap = self._snapshot
        reason = None
        if mutation and not self._accepting:
            reason = 'controller-closing'
        if mutation:
            if request.link_generation != snap.link.generation:
                reason = 'link-stale'
            own = snap.ownership
            if own.mode == 'station' and (not own.accepting or own.health != 'ready' or
                    (request.control_id, request.control_generation) !=
                    (own.control_id, own.control_generation)):
                reason = 'authority-stale'
            if reason is None and isinstance(request.action, (s.BBSEnter, s.BBSInput)):
                p = self._presentation
                if p.bbs_reconnect_required:
                    reason = 'bbs-reconnect-required'
                elif isinstance(request.action, s.BBSInput) and (
                        not p.bbs_complete or not snap.application.bbs.remote_view or
                        (request.action.context_id, request.action.expected_revision) !=
                        (p.bbs_context, p.bbs_revision) or not any(
                            r.bbs_context == p.bbs_context for r in p.records)):
                    reason = 'bbs-stale-view'
            cap = self._port.preflight(request.action)
            if reason is None and cap.availability == 'blocked':
                reason = cap.reason_code or 'guard-refused'
        if reason:
            self._request_change(request, Disposition('refused', reason_code=reason))
            return
        self._request_change(request, Disposition('accepted', stage='queued'))
        creating = isinstance(request.action, (s.ChatCall, s.ChatAnswer, s.ChatCQ, s.VMailCollect, s.BeaconOnce))
        self._request_change(request, Disposition('running',
            bound_generation=request.link_generation if mutation and not creating else None))
        try:
            if isinstance(request.action, s.BBSEnter):
                self._presentation = replace(self._presentation, bbs_context=str(uuid4()),
                                             bbs_revision=0, bbs_complete=True)
            value = self._read(request.action) if not mutation else None
            disposition = Disposition('done', value) if value is not None else self._port.execute(request)
            if isinstance(request.action, s.ChatExit) and disposition.state in TERMINAL:
                self._exit_results[request.id] = disposition
                disposition = Disposition('running', stage='waiting')
            self._request_change(request, disposition)
            if isinstance(request.action, s.BBSInput) and disposition.state not in ('refused', 'failed'):
                self._presentation = replace(self._presentation,
                                             bbs_revision=self._presentation.bbs_revision + 1)
        except Exception as error:
            self._request_change(request, Disposition('failed', reason_code='execution-error', detail=str(error)))
        if isinstance(request.action, s.ChatExit):
            self.shutdown()

    def _run(self):
        for _ in self._run_steps():
            pass

    def _idle_poll(self):
        self._port.poll(.01)
        self._refresh()

    def _wait_for_work(self):
        with self._condition:
            if not (self._queue or self._priority or self._checks or self._legacy or self._shutdown):
                self._condition.wait(.01)

    def _run_steps(self):
        self._owner = get_ident()
        failure = None
        unkey = 'unknown'
        self._port = None
        check = None
        legacy = None
        try:
            self._port = self._factory() if callable(self._factory) else self._factory
            self._port.on_event = self._event
            self._refresh()
            self._ready.set_result(self._snapshot)
            while True:
                yield
                with self._condition:
                    if self._shutdown is not None:
                        break
                    request = self._priority.popleft() if self._priority else None
                    legacy = self._legacy.popleft() if self._legacy and request is None else None
                    check = self._checks.popleft() if self._checks else None
                if legacy is not None:
                    if legacy.completion.set_running_or_notify_cancel():
                        try:
                            result = (self._port.poll(.1) if legacy.line is None else
                                      self._port.legacy_line(legacy.line))
                            self._refresh()
                        except KeyboardInterrupt as error:
                            legacy.completion.set_exception(error)
                            self.shutdown()
                        else:
                            legacy.completion.set_result(result)
                    legacy = None
                if request is not None:
                    self._execute(request)
                    self._refresh()
                with self._condition:
                    if self._shutdown is not None:
                        break
                self._idle_poll()
                if check is not None:
                    action, future = check
                    if future.set_running_or_notify_cancel():
                        try:
                            future.set_result(self._check(action))
                        except Exception as error:
                            future.set_exception(error)
                    check = None
                with self._condition:
                    if self._shutdown is not None:
                        break
                    request = self._queue.popleft() if self._queue and not self._priority else None
                if request is not None:
                    self._execute(request)
                    self._refresh()
                self._wait_for_work()
        except BaseException as error:
            failure = error
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            completion = self.shutdown()
            if self._port is not None:
                try:
                    self._port.quit(unwinding=failure is not None or self._unwinding)
                    final_state = self._port.project()
                    line = final_state.tx.physical_line
                    unkey = 'low' if line is False else 'high' if line is True else 'unknown'
                    with self._condition:
                        self._snapshot = replace(final_state, controller_id=self.controller_id,
                            revision=self._revision, through_seq=self._seq, presentation=self._presentation)
                    if line is not False and (self._require_low_line or any(
                            isinstance(r.action, s.ChatExit) for r in self._submitted.values())):
                        raise RuntimeError('shutdown lacks low-line proof')
                except BaseException as error:
                    if failure is None:
                        failure = error
                    else:
                        failure.__cause__ = error
            for request in tuple(self._submitted.values()):
                result = self._journal.get(request.id)
                if result is None or result.state.state not in TERMINAL:
                    if isinstance(request.action, s.ChatExit):
                        disposition = self._exit_results.get(request.id)
                        if failure is not None:
                            disposition = Disposition('failed', reason_code='shutdown-failed', detail=str(failure))
                        elif disposition is None:
                            disposition = Disposition('failed', reason_code='unkey-unknown')
                        self._request_change(request, disposition)
                        continue
                    state = 'unknown' if result and result.state.stage in ('sent', 'waiting') else 'cancelled'
                    self._request_change(request, Disposition(state, reason_code='shutdown'))
            if check is not None and not check[1].done():
                check[1].set_exception(RuntimeError('controller closed'))
            for _, future in self._checks:
                if not future.done():
                    future.set_exception(RuntimeError('controller closed'))
            self._publish(s.Lifecycle('failed' if failure else 'closed',
                                     str(failure) if failure else None, unkey))
            with self._condition:
                requests = tuple(v.state for v in self._journal.values() if s.ACTIONS[v.state.kind][1] is not s.Effect.R)
                self._snapshot = replace(self._snapshot, through_seq=self._seq, requests=requests,
                                         ownership=replace(self._snapshot.ownership, accepting=False))
                self._projection = s.project(self._snapshot)
            if failure:
                completion.set_exception(failure)
            else:
                completion.set_result(None)
            for pending in (*([legacy] if legacy is not None else []), *self._legacy):
                if not pending.completion.done():
                    pending.completion.set_exception(failure or RuntimeError('controller closed'))


class _LineController(Controller):
    _require_low_line = False

    def _start(self):
        self._steps = self._run_steps()
        self._advance()

    def _advance(self):
        if hasattr(self, '_owner'):
            self._assert_owner()
        next(self._steps, None)

    def _idle_poll(self):
        pass

    def _wait_for_work(self):
        pass

    def join(self, timeout=None):
        return self._shutdown is not None and self._shutdown.done()


class LineAdapter:
    def __init__(self, port):
        self.controller = _LineController(port)
        try:
            self.controller.ready.result()
        except BaseException:
            try:
                self.quit()
            finally:
                self.controller.join()
            raise

    def tick(self):
        return self.command(None)

    def command(self, line):
        completion = self.controller._legacy_line(line)
        self.controller._advance()
        return completion.result()

    def quit(self):
        # terminal() calls this from its finally, so a reader fault is still propagating.
        unwinding = sys.exc_info()[0] is not None
        try:
            completion = self.controller.shutdown(unwinding=unwinding)
            self.controller._advance()
            completion.result()
        finally:
            self.controller.join()
