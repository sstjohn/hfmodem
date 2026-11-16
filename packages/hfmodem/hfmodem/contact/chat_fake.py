# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from threading import Lock, get_ident
from uuid import NAMESPACE_URL, uuid5

from . import chat_state as s
from .chat_controller import Disposition


class FakeClock:
    def __init__(self, now=0.):
        self._now = float(now)
        self._lock = Lock()

    def __call__(self):
        with self._lock:
            return self._now

    def advance(self, seconds):
        if seconds < 0:
            raise ValueError('fake clock cannot run backwards')
        with self._lock:
            self._now += seconds
            return self._now


@dataclass(frozen=True, slots=True)
class ActionExpectation:
    action: s.Action
    submitted_generation: int
    bound_generation: int | None
    actor: str
    control_id: str | None = None
    control_generation: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    payload: s.EventPayload | Disposition
    request_id: str | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    at: float
    trigger: str
    expected_action: s.Action | s.Request | ActionExpectation | None
    patch: tuple[tuple[str, object], ...]
    events: tuple[s.EventPayload | Disposition | ScenarioEvent, ...]


@dataclass(frozen=True, slots=True)
class Pending:
    request: s.Request
    due: float | None
    result: s.ApplicationResult
    stage: str
    family: str | None = None
    attempt_id: str | None = None
    bound_generation: int | None = None


FAMILIES = {
    'bbs.list': 'directory', 'bbs.get': 'file', 'bbs.request': 'file',
    'vmail.send': 'mail', 'email.send': 'mail', 'bbs.term': 'term',
    'beacon.once': 'beacon.once',
}
_CURRENT_GENERATION = object()


def _immutable(value):
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return
    if isinstance(value, tuple):
        for item in value:
            _immutable(item)
        return
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        for field in fields(value):
            _immutable(getattr(value, field.name))
        return
    raise TypeError('scenario values must be immutable')


def patched(record, changes):
    for path, value in changes:
        first, dot, rest = path.partition('.')
        child = patched(getattr(record, first), ((rest, value),)) if dot else value
        record = replace(record, **{first: child})
    return record


class FakeSession:
    def __init__(self, clock=None, scenario=()):
        self.clock = clock or FakeClock()
        self.scenario_name = scenario if isinstance(scenario, str) else None
        self.scenario = tuple(SCENARIOS[scenario] if isinstance(scenario, str) else scenario)
        _immutable(self.scenario)
        self._steps = list(self.scenario)
        self.state = patched(s.initial_snapshot(), (
            ('identity.mycall', 'W9SSJ'), ('identity.idcall', 'W9SSJ'),
            ('identity.channel_hz', 14105000), ('identity.bandwidth_hz', 500),
            ('tx.physical_line', False), ('policy.auto_requested', True),
            ('policy.auto_effective', True), ('policy.auto_state', 'on'),
        ))
        self.on_event = lambda payload, request_id=None, generation=None: None
        self.trace = []
        self.effects = []
        self.event_trace = []
        self.shutdown_trace = []
        self.legacy_trace = []
        self._owner = None
        self._counter = 0
        self._pending = {}
        self._reservations = {}
        self._generation = 0
        self._later = []
        self._text_outcomes = {}
        self._barriers = deque()
        self._barrier_lock = Lock()
        self._released = set()
        self._closed = False
        self._broadcast_retune = None
        self.quit_error = None
        self.poll_error = None
        self.poll_count = 0

    def _own(self):
        ident = get_ident()
        if self._owner is None:
            self._owner = ident
        elif self._owner != ident:
            raise RuntimeError('fake SessionPort accessed outside owner')

    def _id(self, prefix):
        self._counter += 1
        return str(uuid5(NAMESPACE_URL, f'chat-fake:{prefix}:{self._counter}'))

    def advance(self, seconds):
        advance = getattr(self.clock, 'advance', None)
        if advance is None:
            raise TypeError('injected clock must supply advance(seconds)')
        advance(seconds)
        return self.barrier()

    def barrier(self):
        future = Future()
        with self._barrier_lock:
            self._barriers.append(future)
        return future

    def release(self, name):
        with self._barrier_lock:
            self._released.add(name)
        return self.barrier()

    def _emit(self, payload, request_id=None, generation=_CURRENT_GENERATION):
        if generation is _CURRENT_GENERATION:
            generation = self.state.link.generation
        self.event_trace.append(ScenarioEvent(payload, request_id, generation))
        if isinstance(payload, s.TxTextOutcome):
            self._text_outcomes[payload.text_id] = payload
        self.on_event(payload, request_id, generation)

    def _apply(self, step, request=None):
        self.state = patched(self.state, step.patch)
        self._sync_generation()
        disposition = None
        for event in step.events:
            if isinstance(event, ScenarioEvent):
                self._emit(event.payload, event.request_id, event.generation)
            elif isinstance(event, Disposition) and request is not None:
                disposition = event
            else:
                self._emit(event, request.id if request is not None else None)
        return disposition

    def _sync_generation(self):
        generation = self.state.link.generation
        if generation == self._generation:
            return
        self._generation = generation
        self._retire_writes('unknown', before_generation=generation)
        self._reservations = {family: value for family, value in self._reservations.items()
                              if family == 'beacon.once'}
        for pending in tuple(self._pending.values()):
            if pending.bound_generation is not None and pending.bound_generation != generation:
                self._finish(pending, 'unknown' if pending.stage in ('sent', 'waiting') else 'cancelled',
                             reason='link-replaced')
        draft = self.state.application.draft
        if draft is not None and draft.generation != generation:
            self.state = patched(self.state, (('application.draft', None),))

    def poll(self, timeout_s):
        self._own()
        if self._closed:
            return
        self.poll_count += 1
        if self.poll_error is not None:
            raise self.poll_error
        now = self.clock()
        for step in tuple(self._steps):
            if step.expected_action is not None or step.at > now:
                continue
            if step.trigger.startswith('barrier:'):
                with self._barrier_lock:
                    if step.trigger[8:] not in self._released:
                        continue
            elif step.trigger != 'poll':
                continue
            self._steps.remove(step)
            self._apply(step)
        for pending in tuple(self._pending.values()):
            if pending.request.action.kind == 'bbs.list' and pending.stage == 'queued':
                if self.state.link.connected and self.state.identity.channel_hz != self.state.identity.calling_hz:
                    pending = replace(pending, stage='waiting', due=now + .5)
                    self._pending[pending.request.id] = pending
                    self._reservations['directory'] = s.ServiceReservation('directory', 'pending', pending.request.id)
                    self.state = patched(self.state, (('application.bbs_list_generation', None),
                                                     ('application.bbs_waiting', True)))
                    self._emit(Disposition('running', stage='waiting', bound_generation=pending.bound_generation),
                               pending.request.id)
            if pending.due is not None and pending.due <= now:
                self._complete(pending)
        for at, event in tuple(self._later):
            if at <= now:
                self._later.remove((at, event))
                self._emit(event.payload, event.request_id, event.generation)
        with self._barrier_lock:
            barriers = tuple(self._barriers)
            self._barriers.clear()
        for future in barriers:
            if not future.cancelled():
                future.set_result(self.poll_count)

    def project(self):
        self._own()
        capabilities = tuple(self._capability(kind) for kind in s.ACTIONS)
        return replace(self.state, capabilities=capabilities)

    def _reservation(self, kind):
        family = FAMILIES.get(kind)
        if family is None:
            return None
        return self._reservations.get(family, s.ServiceReservation(family, 'available', None))

    def _capability(self, kind, reason=None, detail=None):
        reason = reason or self._kind_reason(kind)
        reservation = self._reservation(kind)
        if reason is None and reservation is not None and reservation.state != 'available':
            reason = ('beacon-pending' if reservation.family == 'beacon.once' else
                      'service-pending' if reservation.state == 'pending' else 'reconnect-required')
        return s.Capability(kind, s.ACTIONS[kind][1], 'blocked' if reason else 'allowed',
                            reservation, reason, detail or reason)

    def preflight(self, action):
        self._own()
        kind = action.kind
        s.effect(action)
        reason = self._kind_reason(kind)
        if isinstance(action, s.BroadcastSend) and action.text.strip() in ('clear', 'yes', 'no'):
            reason = 'broadcast-acknowledgement-action-required'
        if isinstance(action, s.ChatSend) and any(tag in action.text for tag in ('<BLR>', '<BG:', '<SM>', 'BBS TERM')):
            reason = 'reserved-service'
        if isinstance(action, s.ChatTag) and action.name in ('BLR', 'BG', 'SM'):
            reason = 'reserved-service'
        if isinstance(action, s.BBSInput) and any(tag in action.text for tag in ('<BLR>', '<BG:', '<SM>', 'BBS TERM')):
            reason = 'reserved-service'
        return self._capability(kind, reason)

    def _kind_reason(self, kind):
        reason = None
        state = self.state
        if self._closed:
            reason = 'controller-closed'
        elif kind in ('chat.send', 'chat.tag', 'vmail.compose', 'email.compose', 'vmail.send', 'email.send',
                      'bbs.list', 'bbs.get', 'bbs.request', 'bbs.capture', 'bbs.enter', 'bbs.term', 'bbs.input',
                      'chat.qsy', 'inquire.heard') and not state.link.connected:
            reason = 'link-required'
        elif kind == 'chat.send' and state.application.draft is not None:
            reason = 'draft-active'
        elif kind == 'chat.send' and state.application.bbs.owned:
            reason = 'bbs-owned'
        elif kind in ('chat.call', 'chat.answer', 'chat.cq', 'vmail.collect', 'beacon.once', 'band.set') and (
                state.link.connected or state.link.pending):
            reason = 'link-active'
        elif kind in ('bbs.enter', 'bbs.get', 'bbs.request', 'email.compose') and (
                state.identity.channel_hz == state.identity.calling_hz):
            reason = 'off-cf-required'
        elif kind == 'bbs.term' and not state.application.bbs.remote_view:
            reason = 'bbs-entry-required'
        elif kind in ('chat.call', 'chat.answer', 'chat.cq', 'vmail.collect', 'band.set', 'follow.set',
                      'auto.set', 'away.set', 'away.back') and state.broadcast.possibly_outstanding:
            reason = 'broadcast-acknowledgement-required'
        elif kind == 'broadcast.send' and state.broadcast.selected_hz is None:
            reason = 'broadcast-selection-required'
        elif kind in ('broadcast.select', 'broadcast.send', 'beacon.once') and (
                state.tx.physical_line is not False or state.tx.host_ptt):
            reason = 'line-not-low'
        elif kind == 'broadcast.send' and state.receive.host_busy:
            reason = 'channel-busy'
        elif kind in ('bbs.get', 'bbs.request') and state.application.files.list_generation != state.link.generation:
            reason = 'directory-stale'
        elif kind == 'bbs.capture' and (state.application.files.offer is None or
                                      state.application.files.generation != state.link.generation):
            reason = 'offer-required'
        return reason

    def _expected(self, request):
        scripted = [step for step in self._steps if step.expected_action is not None]
        if not scripted:
            return None
        step = scripted[0]
        if step.at > self.clock():
            raise AssertionError(f'action before scenario time {step.at}: {request.action}')
        expected = step.expected_action
        if isinstance(expected, s.Request):
            assert request == expected, (request, expected)
        elif isinstance(expected, ActionExpectation):
            assert request.action == expected.action
            assert request.link_generation == expected.submitted_generation
            assert request.actor == expected.actor
            assert request.control_id == expected.control_id
            assert request.control_generation == expected.control_generation
            pending = self._pending.get(request.id)
            actual_bound = (pending.bound_generation if pending else None
                            if request.action.kind in ('chat.call', 'chat.answer', 'chat.cq',
                                                       'vmail.collect', 'beacon.once')
                            else self.state.link.generation)
            assert actual_bound == expected.bound_generation
        else:
            assert request.action == expected, (request.action, expected)
        assert step.trigger in ('action', request.action.kind)
        self._steps.remove(step)
        return step

    def execute(self, request):
        self._own()
        step = self._expected(request)
        self.trace.append(request)
        cap = self.preflight(request.action)
        if cap.availability == 'blocked':
            return Disposition('refused', reason_code=cap.reason_code, detail=cap.reason or '')
        if request.link_generation != self.state.link.generation and s.effect(request.action) is not s.Effect.R:
            return Disposition('refused', reason_code='link-stale')
        own = self.state.ownership
        if s.effect(request.action) is not s.Effect.R and own.mode == 'station' and (
                not own.accepting or own.health != 'ready' or
                (request.control_id, request.control_generation) != (own.control_id, own.control_generation)):
            return Disposition('refused', reason_code='authority-stale')
        if not s.preserves_draft(request.action):
            self.state = patched(self.state, (('application.draft', None),))
        if step is not None and any(isinstance(e, Disposition) for e in step.events):
            self.effects.append((request.id, s.effect(request.action)))
            return self._apply(step, request)
        result = self._execute(request)
        if result.state != 'refused':
            self.effects.append((request.id, s.effect(request.action)))
        if step is not None:
            self._apply(step, request)
        return result

    def _write(self, request, text, source='chat'):
        text_id = self._id('text')
        framed = len(text.encode('utf-8')) + 4
        self._emit(s.TextRecord(text_id, 'tx', text, self.state.link.peer, source), request.id)
        queued = s.TxTextOutcome(text_id, None, 'queued', framed, 0, 0, framed, 'verified')
        self._emit(queued, request.id)
        self._later += [
            (self.clock() + .25, ScenarioEvent(replace(queued, status='partial',
                acknowledged_bytes=framed // 2, unknown_bytes=framed - framed // 2), request.id,
                self.state.link.generation)),
            (self.clock() + .5, ScenarioEvent(replace(queued, status='acknowledged',
                acknowledged_bytes=framed, unknown_bytes=0), request.id, self.state.link.generation)),
        ]
        return s.WriteAccepted(None, text_id)

    def _retire_writes(self, disposition, *, before_generation=None):
        pending = {}
        for at, event in tuple(self._later):
            if isinstance(event.payload, s.TxTextOutcome) and (
                    before_generation is None or event.generation != before_generation):
                self._later.remove((at, event))
                pending[event.payload.text_id] = event
        for text_id, event in pending.items():
            current = self._text_outcomes[text_id]
            remaining = current.framed_bytes - current.acknowledged_bytes - current.discarded_bytes
            outcome = replace(current, status=disposition,
                discarded_bytes=current.discarded_bytes + (remaining if disposition == 'discarded' else 0),
                unknown_bytes=remaining if disposition == 'unknown' else 0)
            self._emit(outcome, event.request_id, event.generation)

    def _wait(self, request, result, *, seconds=.5, stage='waiting', attempt_id=None, bound=None):
        family = FAMILIES.get(request.action.kind)
        if family:
            self._reservations[family] = s.ServiceReservation(family, 'pending', request.id)
        if bound is None and request.action.kind not in ('chat.cq', 'beacon.once'):
            bound = self.state.link.generation
        pending = Pending(request, None if seconds is None else self.clock() + seconds,
                          result, stage, family, attempt_id, bound)
        self._pending[request.id] = pending
        return Disposition('running', stage=stage, bound_generation=bound, attempt_id=attempt_id)

    def _finish(self, pending, state='done', *, reason=None):
        if self._pending.pop(pending.request.id, None) is None:
            return
        if pending.family:
            available = pending.family == 'beacon.once' or pending.stage == 'queued' and state != 'done'
            self._reservations[pending.family] = s.ServiceReservation(
                pending.family, 'available' if available else 'spent', None if available else pending.request.id)
        self._emit(Disposition(state, pending.result, reason_code=reason,
                               bound_generation=pending.bound_generation, attempt_id=pending.attempt_id),
                   pending.request.id, pending.bound_generation)

    def _complete(self, pending):
        kind = pending.request.action.kind
        if kind in ('chat.call', 'chat.answer', 'vmail.collect'):
            self.state = patched(self.state, (('link.state', 'established'), ('link.connected', True),
                ('link.pending', False), ('link.flow_state', s.State.CONNECTED),
                ('operation.phase', s.Phase.READY_SLOT), ('operation.active', False),
                ('operation.description', 'link established; no exchange in progress')))
        elif kind == 'chat.qsy':
            slot = pending.request.action.slot
            self.state = patched(self.state, (('identity.slot', slot),
                ('identity.channel_hz', self.state.identity.calling_hz + slot * 250),
                ('operation.phase', s.Phase.READY_SLOT), ('operation.active', False),
                ('operation.subphase', None), ('operation.description', 'link established; QSY complete')))
        elif kind == 'chat.cq':
            self.state = patched(self.state, (('operation.phase', s.Phase.MONITORING),
                                             ('operation.active', False)))
        elif kind == 'bbs.list':
            self.state = patched(self.state, (('application.bbs_waiting', False),
                ('application.files.entries', pending.result.entries),
                ('application.files.list_generation', self.state.link.generation)))
        elif kind in ('vmail.send', 'email.send'):
            self.state = patched(self.state, (('application.vmail_ack_pending', False),
                ('application.email.waiting', False), ('operation.phase', s.Phase.READY_SLOT)))
        elif kind == 'bbs.term':
            a = pending.request.action
            self.state = patched(self.state, (('application.bbs.remote_ansi', a.mode == 'ansi16'),
                ('application.bbs.dimensions', (a.width, a.height)),
                ('application.bbs.requested_term', None)))
        elif kind == 'beacon.once':
            self.state = patched(self.state, (('beacon.once', None), ('beacon.once_request_id', None),
                ('beacon.last_tx', self.clock()), ('operation.phase', s.Phase.MONITORING)))
        elif kind == 'broadcast.send':
            self.state = patched(self.state, (('broadcast.active', False), ('broadcast.phase', None),
                ('broadcast.result', 'sent')))
        elif kind in ('bbs.get', 'bbs.request'):
            self.state = patched(self.state, (('application.files.active', False),
                ('application.files.receiving', False), ('application.files.downloading', False)))
        self._finish(pending)

    def _draft_guard(self, request):
        draft = self.state.application.draft
        action = request.action
        if draft is None or draft.id != action.draft_id or draft.revision != action.expected_revision:
            return 'draft-stale'
        if draft.owner != request.actor:
            return 'draft-owned'
        if draft.generation != self.state.link.generation or draft.peer != self.state.link.peer:
            return 'draft-link-stale'
        return None

    def _execute(self, request):
        a, kind = request.action, request.action.kind
        applied = lambda: Disposition('done', s.Applied(self.state.revision))
        if kind == 'heard.list':
            return Disposition('done', s.HeardRows(self.state.receive.heard))
        if kind == 'slots.list':
            return Disposition('done', s.SlotRows(self.state.receive.slots))
        if kind == 'files.list':
            files = self.state.application.files
            return Disposition('done', s.FileEntries(files.list_generation, files.entries))
        if kind == 'vmail.list':
            return Disposition('done', s.MailHeaders(self.state.application.mailbox))
        if kind == 'vmail.pending':
            return Disposition('done', s.MailNotices(self.state.application.notices))
        if kind == 'chat.help':
            return Disposition('done', s.HelpText(a.topic, 'Fake Kestrel chat: choose an explicit action.'))
        if kind == 'vmail.read':
            header = next((h for h in self.state.application.mailbox if h.number == a.number), None)
            return Disposition('done', s.MailBody(header, 'Synthetic mail body') if header else s.NotFound(str(a.number)))
        if kind in ('vmail.compose', 'email.compose'):
            old = self.state.application.draft
            if old is not None and old.owner != request.actor:
                return Disposition('refused', reason_code='draft-owned')
            draft = s.Draft(self._id('draft'), 0, request.actor, kind.split('.')[0],
                self.state.link.generation, self.state.link.peer, a.subject,
                getattr(a, 'recipient', None), getattr(a, 'reply_to', None), ())
            self.state = patched(self.state, (('application.draft', draft),
                ('operation.phase', s.Phase.EMAIL_COMPOSE if kind == 'email.compose' else s.Phase.COMPOSING)))
            return Disposition('done', s.DraftAccepted(draft))
        if kind in ('draft.append', 'draft.drop', 'vmail.send', 'email.send'):
            if reason := self._draft_guard(request):
                return Disposition('refused', reason_code=reason)
            draft = self.state.application.draft
            if kind == 'draft.append':
                lines = draft.lines + (a.line,)
                size = len('\r\n'.join(lines).encode('utf-8'))
                if size > 64 * 1024 or draft.kind == 'email' and size + len(draft.subject.encode()) + 100 > 500:
                    return Disposition('refused', reason_code='draft-too-large')
                draft = replace(draft, lines=lines, revision=draft.revision + 1)
                self.state = patched(self.state, (('application.draft', draft),))
                return Disposition('done', s.DraftAccepted(draft))
            if kind != 'draft.drop' and draft.kind != kind.split('.')[0]:
                return Disposition('refused', reason_code='draft-kind')
            self.state = patched(self.state, (('application.draft', None),))
            if kind == 'draft.drop':
                return Disposition('done', s.DraftAccepted(None))
            self._write(request, '\r\n'.join(draft.lines), draft.kind)
            self.state = patched(self.state, (('application.vmail_ack_pending', draft.kind == 'vmail'),
                ('application.email.waiting', draft.kind == 'email'),
                ('operation.phase', s.Phase.VMAIL_ACK if draft.kind == 'vmail' else s.Phase.EMAIL)))
            return self._wait(request, s.MailResult(draft.kind, 'smr' if draft.kind == 'vmail' else 'er'))
        if kind in ('chat.send', 'chat.tag', 'inquire.heard', 'bbs.input'):
            text = (a.text if hasattr(a, 'text') else '<LHR>' if kind == 'inquire.heard'
                    else f'<{a.name}' + (f':{a.value}' if a.value is not None else '') + '>')
            return Disposition('done', self._write(request, text))
        if kind in ('chat.call', 'chat.answer', 'vmail.collect'):
            peer = getattr(a, 'peer', None)
            if peer is None:
                row = next((row for row in self.state.receive.heard if row.n == a.heard_number), None)
                if row is None:
                    return Disposition('refused', reason_code='heard-not-found')
                peer = row.call
            generation = self.state.link.generation + 1
            attempt = self._id('attempt')
            self.state = patched(self.state, (('link.generation', generation), ('link.peer', peer),
                ('link.peer_source', 'operator-target'), ('link.pending', True), ('link.state', 'connecting'),
                ('link.flow_state', s.State.CONNECTING), ('operation.phase', s.Phase.CONNECTING),
                ('operation.active', True), ('operation.description', 'waiting for modem link establishment')))
            self._sync_generation()
            return self._wait(request, s.ConnectionResult(attempt, generation, 'connected'),
                              seconds=1., attempt_id=attempt, bound=generation)
        if kind == 'chat.cq':
            self.state = patched(self.state, (('operation.phase', s.Phase.CQ), ('operation.active', True)))
            result = s.ConnectionResult(None, None, 'busy' if self.state.receive.host_busy else 'no-answer')
            return self._wait(request, result, seconds=None if a.repeat else 1.)
        if kind == 'chat.qsy':
            self.state = patched(self.state, (('operation.phase', s.Phase.QSY),
                ('operation.subphase', 'readiness'), ('operation.qsy_slot', a.slot),
                ('operation.description', 'waiting for peer auto-QSY readiness'), ('operation.active', True)))
            return self._wait(request, s.Applied(self.state.revision), seconds=1., stage='queued')
        if kind == 'chat.cancel_wait':
            target = next(iter(self._pending.values()), None)
            for pending in tuple(self._pending.values()):
                self._finish(pending, 'unknown' if pending.stage in ('sent', 'waiting') else 'cancelled',
                             reason='operator-stop')
            self.state = patched(self.state, (('operation.active', False), ('application.bbs_waiting', False),
                ('application.bbs_list_generation', None), ('beacon.once', None),
                ('beacon.once_request_id', None), ('application.email.waiting', False),
                ('application.vmail_ack_pending', False)))
            return Disposition('done', s.WaitStopped(target.request.id if target else None,
                                                      'wait-only' if target else 'no-op'))
        if kind in ('chat.close', 'chat.exit'):
            if self.state.tx.physical_line is not False:
                return Disposition('failed', s.LinkClosed('high' if self.state.tx.physical_line else 'unknown'),
                                   reason_code='unkey-unverified')
            self.state = patched(self.state, (('link.connected', False), ('link.pending', False),
                ('link.generation', self.state.link.generation + 1), ('link.state', 'idle'),
                ('link.flow_state', s.State.MONITORING), ('application.bbs.remote_view', False),
                ('application.bbs.owned', False), ('beacon.minutes', None), ('beacon.once', None),
                ('beacon.once_request_id', None), ('operation.phase', s.Phase.MONITORING)))
            self._sync_generation()
            return Disposition('done', s.LinkClosed('low'))
        if kind == 'bbs.enter':
            self.state = patched(self.state, (('application.bbs.remote_view', True),
                ('application.bbs.owned', True), ('application.bbs.generation', self.state.link.generation)))
            return Disposition('done', self._write(request, 'BBS', 'bbs'))
        if kind == 'bbs.term':
            if a.mode == 'plain':
                self.state = patched(self.state, (('application.bbs.remote_ansi', False),))
            self.state = patched(self.state, (('application.bbs.requested_term', (a.mode, a.width, a.height)),))
            self._write(request, f'BBS TERM {a.mode.upper()} {a.width} {a.height}', 'bbs')
            return self._wait(request, s.TermResult(a.mode, a.width, a.height))
        if kind == 'bbs.list':
            queued = self.state.identity.channel_hz == self.state.identity.calling_hz
            self.state = patched(self.state, (('application.bbs_list_generation',
                self.state.link.generation if queued else None), ('application.bbs_waiting', not queued)))
            return self._wait(request, s.DirectoryResult((s.FileEntry('hello.txt', '2026-10-28', 5),), 'complete'),
                              seconds=None if queued else .5, stage='queued' if queued else 'waiting')
        if kind in ('bbs.get', 'bbs.request'):
            entries = self.state.application.files.entries
            entry = (entries[int(a.selection) - 1] if a.selection.isdecimal() and 1 <= int(a.selection) <= len(entries)
                     else next((e for e in entries if e.name == a.selection), None))
            if entry is None:
                return Disposition('refused', reason_code='file-not-found')
            self.state = patched(self.state, (('application.files.active', True),
                ('application.files.requested', entry.name), ('application.files.received_bytes', 0),
                ('application.files.downloading', kind == 'bbs.get'),
                ('application.files.generation', self.state.link.generation)))
            return self._wait(request, s.FileResult('validated-saved' if kind == 'bbs.get' else 'offer-recorded',
                entry.name, self._id('evidence'), entry.size if kind == 'bbs.get' else 0))
        if kind == 'bbs.capture':
            self.state = patched(self.state, (('application.files.opaque', True), ('application.files.active', True)))
            return Disposition('done', s.FileResult('capture-started', self.state.application.files.requested,
                                                    self._id('evidence'), 0))
        if kind == 'bbs.reload':
            return self._wait(request, s.ReloadResult(self.state.revision + 1))
        if kind == 'beacon.once':
            self.state = patched(self.state, (('beacon.once', 'force' if a.force else 'now'),
                ('beacon.once_request_id', request.id)))
            return self._wait(request, s.BeaconResult('sent'), seconds=1., stage='queued')
        if kind in ('beacon.configure', 'beacon.hop'):
            generation = self.state.beacon.generation + 1
            if kind == 'beacon.hop' or a.minutes is None:
                for pending in tuple(self._pending.values()):
                    if pending.family == 'beacon.once':
                        self._finish(pending, 'cancelled', reason='schedule-replaced')
                self.state = patched(self.state, (('beacon.once', None), ('beacon.once_request_id', None)))
            self.state = patched(self.state, (('beacon.minutes', a.minutes), ('beacon.generation', generation),
                ('beacon.due', self.clock() if a.minutes is not None else None)))
            if kind == 'beacon.hop':
                self.state = patched(self.state, (('beacon.hop.active', True), ('beacon.hop.auto', a.auto),
                    ('beacon.hop.bands', a.bands), ('beacon.hop.minutes', a.minutes)))
            return applied()
        if kind in ('broadcast.select', 'broadcast.off') and self.state.broadcast.possibly_outstanding:
            self._broadcast_retune = request
            return Disposition('done', s.Applied(0))
        if kind == 'broadcast.clear':
            self._broadcast_retune = None
            self.state = patched(self.state, (('broadcast.possibly_outstanding', 0),))
            return Disposition('done', s.Applied(0))
        if kind == 'broadcast.confirm':
            pending, self._broadcast_retune = self._broadcast_retune, None
            if pending is not None and a.proceed:
                self.state = patched(self.state, (('broadcast.possibly_outstanding', 0),))
                return self._execute(pending)
            return Disposition('done', s.Applied(0))
        if kind == 'broadcast.select':
            hz = self.state.identity.calling_hz + (a.slot or 0) * 250
            self.state = patched(self.state, (('broadcast.selected_hz', hz), ('identity.channel_hz', hz)))
            return applied()
        if kind == 'broadcast.off':
            self.state = patched(self.state, (('broadcast.selected_hz', None),
                ('identity.channel_hz', self.state.identity.calling_hz)))
            return applied()
        if kind == 'broadcast.send':
            outstanding = self.state.broadcast.possibly_outstanding + 1
            self.state = patched(self.state, (('broadcast.active', True), ('broadcast.possibly_outstanding', outstanding),
                ('broadcast.phase', 'queued; waiting for RF completion and unkey')))
            return self._wait(request, s.BroadcastResult(outstanding, 'sent'))
        if kind == 'auto.set':
            self.state = patched(self.state, (('policy.auto_requested', a.on), ('policy.auto_effective', a.on),
                ('policy.auto_state', 'on' if a.on else 'off')))
        elif kind == 'away.set':
            self.state = patched(self.state, (('policy.away_text', a.text), ('policy.auto_requested', True),
                ('policy.auto_effective', True), ('policy.auto_state', 'on')))
        elif kind == 'away.back':
            self.state = patched(self.state, (('policy.away_text', None),))
        elif kind == 'verbose.set':
            self.state = patched(self.state, (('policy.verbose', a.on),))
        elif kind == 'follow.set':
            self.state = patched(self.state, (('policy.following', a.target),))
        elif kind == 'band.set':
            frequencies = {10: 28105000, 12: 24925000, 15: 21105000, 17: 18105000,
                           20: 14105000, 30: 10133000, 40: 7105000, 80: 3595000}
            if a.band_m not in frequencies:
                return Disposition('refused', reason_code='band-invalid')
            self.state = patched(self.state, (('identity.band_m', a.band_m),
                ('identity.calling_hz', frequencies[a.band_m]), ('identity.channel_hz', frequencies[a.band_m])))
        elif kind == 'band.ready':
            self.state = patched(self.state, (('policy.auto_effective', a.ready and self.state.policy.auto_requested),
                ('policy.auto_state', 'on' if a.ready else 'suspended'),
                ('policy.suspension_reason', None if a.ready else 'unready band')))
        elif kind == 'buffer.clear':
            self._retire_writes('discarded')
            self.state = patched(self.state, (('tx.buffer_bytes', 0),))
        elif kind == 'pskreporter.off':
            self.state = patched(self.state, (('policy.reporter_status', (('enabled', 'false'),)),))
        elif kind == 'vmail.dismiss':
            notices = tuple(n for n in self.state.application.notices if n.source != a.peer)
            self.state = patched(self.state, (('application.notices', notices),))
        else:
            raise AssertionError(f'unimplemented fake action: {kind}')
        return applied()

    def legacy_line(self, line):
        self._own()
        self.legacy_trace.append(line)
        if line.startswith('/') and line.split()[0] not in (
                '/status', '/help', '/heard', '/slots', '/follow', '/auto', '/away', '/back', '/verbose',
                '/pskreporter', '/bbs'):
            self.state = patched(self.state, (('application.draft', None),))

        return line.strip() != '/quit'

    def quit(self, *, unwinding=False):
        self._own()
        if self._closed:
            return
        self._closed = True
        for pending in tuple(self._pending.values()):
            self._finish(pending, 'unknown' if pending.stage in ('sent', 'waiting') else 'cancelled',
                         reason='shutdown')
        self.state = patched(self.state, (('beacon.minutes', None), ('beacon.once', None),
            ('beacon.once_request_id', None), ('beacon.hop.active', False),
            ('link.connected', False), ('link.pending', False), ('link.host_closed', True),
            ('tx.host_ptt', False)))
        self.shutdown_trace.extend(('session.quit', 'client.close', 'runtime.close', 'slots.close',
            'reporter.close', 'saves.wait', 'streams.close', 'terminal.close', 'catalog.close', 'rig.shutdown'))
        if self.quit_error is not None:
            raise self.quit_error

    def assert_complete(self):
        assert not self._steps, f'unconsumed scenario steps: {self._steps}'


CONNECTED = (
    ('link.generation', 1), ('link.state', 'established'), ('link.flow_state', s.State.CONNECTED),
    ('link.connected', True), ('link.pending', False), ('link.peer', 'N0XYZ'),
    ('link.peer_source', 'host-connected'), ('identity.channel_hz', 14106500),
    ('identity.slot', 12), ('operation.phase', s.Phase.READY_SLOT),
    ('operation.description', 'link established; no exchange in progress'),
)
HEARD = (
    s.HeardRow(1, 'N0XYZ', 12, 500, -5., 0., False, '', 'B', '', 20, 14106500),
    s.HeardRow(2, 'K7QB', None, 500, None, 0., True, 'CN87', 'V', 'CF', 20, 14105000),
)
SLOTS = (
    s.SlotRow(12, 1500, 'busy', 0., ('?',), None, (), False, '?', '?', 'signal',
              0., 0., None, True),
    s.SlotRow(13, 1750, 'quiet', None, (), None, (), False, '', '', '', None, 0., None, False),
)
BBS_ATTACK = s.BBSPage('bbs-attack', 'ansi16', 80, 24, (
    s.StyledSpan('Menu [bold]literal[/bold]\n', 32, None, True),
    s.StyledSpan('\x1b]52;c;ZXhmaWw=\x07\x1b[2J\x85Choose 1 > ', 37, 40, False),
))

SCENARIOS = {
    'idle-monitoring': (
        ScenarioStep(0., 'poll', None, (
            ('receive.heard', HEARD), ('receive.slots', SLOTS), ('receive.watched', (12, 13)),
            ('receive.passband_edges', (('wide', (300., 2700.)),)), ('receive.wide_offset', 125.),
            ('receive.table_audio_time', 10.), ('receive.observed_at', 0.),
            ('receive.coverage', (s.Coverage(12, 'audio gap', 0., 10., 3., .05, 'observed-events'),)),
        ), (s.SlotRecord(12, 10., 'text', 'passive decoded text; identity uncertain', '?'),)),
        ScenarioStep(1., 'poll', None, (('policy.auto_requested', False),
            ('policy.auto_effective', False), ('policy.auto_state', 'off')), ()),
        ScenarioStep(2., 'poll', None, (('policy.auto_requested', True),
            ('policy.auto_effective', False), ('policy.auto_state', 'suspended'),
            ('policy.suspension_reason', 'unready band')), ()),
        ScenarioStep(61., 'poll', None, (('receive.coverage', ()),), ()),
    ),
    'incoming-call': (
        ScenarioStep(0., 'poll', None, (('link.generation', 1), ('link.pending', True),
            ('link.state', 'connecting'), ('operation.phase', s.Phase.CONNECTING)), ()),
        ScenarioStep(1., 'poll', None, CONNECTED + (('identity.channel_hz', 14105000),
            ('identity.slot', None), ('policy.away_text', 'Back later'),
            ('policy.away_owed_generation', 1), ('application.bbs_list_generation', 1)),
            (s.TextRecord('incoming-awq', 'tx', '<AWQ>', 'N0XYZ', 'away'),)),
        ScenarioStep(2., 'poll', None, (('identity.channel_hz', 14106500), ('identity.slot', 12),
            ('policy.away_owed_generation', None)),
            (s.TextRecord('incoming-away', 'tx', '<AWAY>\r\nBack later', 'N0XYZ', 'away'),)),
    ),
    'two-way-chat': (
        ScenarioStep(0., 'poll', None, CONNECTED, ()),
        ScenarioStep(0., 'chat.send', s.ChatSend('hello'), (), (
            s.TextRecord('two-way-tx', 'tx', 'hello', 'N0XYZ', 'chat'),
            s.TxTextOutcome('two-way-tx', 1, 'queued', 9, 0, 0, 9, 'verified'),
            Disposition('done', s.WriteAccepted(1, 'two-way-tx')),
        )),
        ScenarioStep(.25, 'poll', None, (('tx.host_ptt', True), ('tx.buffer_bytes', 0),
            ('tx.unreported', True)), (
            s.TextRecord('two-way-rx', 'rx', 'independent reply', 'N0XYZ', 'chat'),
            s.TxTextOutcome('two-way-tx', 1, 'partial', 9, 4, 0, 5, 'verified'),
        )),
        ScenarioStep(.5, 'poll', None, (('tx.host_ptt', False),), (
            s.TxTextOutcome('two-way-tx', 1, 'acknowledged', 9, 9, 0, 0, 'verified'),)),
        ScenarioStep(.75, 'poll', None, (('tx.host_ptt', True),),
            (s.Notice('Modem frame; no payload delivery implied', 'info'),)),
        ScenarioStep(1., 'poll', None, (('tx.host_ptt', False),), (
            s.TextRecord('two-way-unknown', 'tx', 'unresolved write', 'N0XYZ', 'chat'),
            s.TxTextOutcome('two-way-unknown', None, 'unknown', 20, 0, 0, 20, 'gap'),
            s.TxCredit(4, 'discarded', None),
        )),
    ),
    'qsy': (
        ScenarioStep(0., 'poll', None, CONNECTED + (('identity.channel_hz', 14105000),), ()),
        ScenarioStep(0., 'chat.qsy', s.ChatQSY(12), (), ()),
        ScenarioStep(.2, 'poll', None, (('operation.subphase', 'invitation delivery'),), ()),
        ScenarioStep(.4, 'poll', None, (('operation.subphase', 'peer acceptance'),), ()),
        ScenarioStep(.6, 'poll', None, (('operation.subphase', 'retune'),), ()),
        ScenarioStep(.8, 'poll', None, (('operation.subphase', 'opening'),), ()),
    ),
    'vmail-email': (
        ScenarioStep(0., 'poll', None, CONNECTED, ()),
        ScenarioStep(0., 'vmail.compose', s.VMailCompose('Greetings'), (), ()),
        ScenarioStep(.1, 'poll', None, (), (s.Notice('Modem ACK, SMR and email ER are separate evidence', 'info'),)),
    ),
    'bbs': (
        ScenarioStep(0., 'poll', None, CONNECTED + (('identity.channel_hz', 14105000),), ()),
        ScenarioStep(0., 'bbs.list', s.BBSList(), (), ()),
        ScenarioStep(.25, 'poll', None, (('identity.channel_hz', 14106500),), ()),
        ScenarioStep(1., 'bbs.enter', s.BBSEnter(), (), ()),
        ScenarioStep(1.1, 'poll', None, (), (BBS_ATTACK,)),
        ScenarioStep(2., 'poll', None, (('application.bbs.remote_view', False),
            ('application.bbs.owned', False)), (s.Notice('Returned to ordinary chat.', 'info'),)),
    ),
    'broadcast': (
        ScenarioStep(0., 'poll', None, (('receive.heard', HEARD), ('receive.host_busy', False)), ()),
        ScenarioStep(0., 'broadcast.select', s.BroadcastSelect(12), (), ()),
        ScenarioStep(0., 'broadcast.send', s.BroadcastSend('hello all'), (), ()),
        ScenarioStep(1., 'poll', None, (), (s.Notice('Broadcast sent; reception unconfirmed', 'info'),)),
    ),
    'failure-close': (
        ScenarioStep(0., 'poll', None, CONNECTED + (('beacon.minutes', 10),
            ('beacon.due', 600.), ('beacon.expires', 86400.)), ()),
        ScenarioStep(1., 'poll', None, (('link.connected', False), ('link.generation', 2),
            ('link.state', 'idle'), ('tx.physical_line', True), ('operation.phase', s.Phase.SETTLING)),
            (s.Notice('Peer disconnected; waiting for physical unkey', 'warning'),)),
        ScenarioStep(2., 'poll', None, (('tx.physical_line', False), ('operation.phase', s.Phase.MONITORING)), ()),
        ScenarioStep(3., 'poll', None, (('link.host_closed', True), ('link.state', 'host-lost'),
            ('ownership.health', 'failed'), ('operation.phase', s.Phase.HOST_CLOSED)),
            (s.Lifecycle('failed', 'synthetic host EOF', 'unknown'),)),
    ),
}
