# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

import queue

from bisect import bisect_right
from collections import deque
from dataclasses import replace
from threading import get_ident
from uuid import uuid4

from . import chat_state as s
from .chat_controller import Disposition, TERMINAL
from . import varac as v

SERVICE_FAMILIES = {
    'bbs.list': 'directory', 'bbs.get': 'file', 'bbs.request': 'file',
    'vmail.send': 'mail', 'email.send': 'mail', 'bbs.term': 'term',
    'beacon.once': 'beacon',
}


def _covered(ranges, start, end):
    index = max(0, bisect_right(ranges, (start, end)) - 1)
    total = 0
    while index < len(ranges) and ranges[index][0] < end:
        begin, stop = ranges[index]
        total += max(0, min(stop, end) - max(begin, start))
        index += 1
    return total


class SessionAdapter:
    def __init__(self, session: v.Session, *, ownership=None, delivery_evidence=None, delivery_boundary=None,
                 legacy=False):
        self.session = session
        self.on_event = lambda *args: None
        self._owner = None
        self._ownership = ownership or s.Ownership('local', None, None, None, None, 'ready', True)
        self._active = None
        self._legacy = legacy
        self._requests = {}
        self._results = {}
        self._operations = {}
        self._reservations = {}
        self._reservation_generation = session.client.link_generation
        self._sent = set()
        self._deadlines = {}
        self._attempts = {}
        self._attempt = None
        self._frame = None
        self._listening = session.client.listening
        self._generation = session.client.link_generation
        self._link_transition = None
        self._link_facts = None
        self._credits = s.Credit(None, 0, 0, 'unattributed')
        self._delivery_evidence = delivery_evidence or session.delivery_evidence
        self._delivery_boundary = delivery_boundary or session.delivery_boundary
        self._delivery_binding = None
        self._delivery_sequence = -1
        self._delivery_writes = deque()
        self._delivery_bytes = 0
        self._email_acknowledged = set()
        self._monitor_epoch = None
        self._draft = None
        self._legacy_draft = None
        self._retired_draft = None
        self._beacon_id = None
        self._beacon_running = False
        self._expiring = False
        self._line = None
        self._line_at = None
        self._mailbox = ()
        self._mail_loaded = False
        self._notices = ()
        self._receive = s.initial_snapshot().receive
        self._coverage = deque()
        self._queued = deque()
        self._quit = False
        self._last_write = None
        self._error = None
        self._old_semantic = session.semantic
        self._old_client_semantic = session.client.semantic
        self._old_emit, self._old_say = session.emit, session.say
        self._old_event = session.client.event
        self._old_line = session.line_state
        session.semantic = self._semantic
        session.client.semantic = self._client_semantic
        session.emit, session.say = self._emit, self._say
        session.client.event = self._client_event
        session.line_state = self._observe_line

    def _assert_owner(self):
        ident = get_ident()
        if self._owner is None:
            self._owner = ident
        if self._owner != ident:
            raise RuntimeError('real Session accessed outside owner')

    def _observe_line(self):
        value = self._old_line()
        self._line, self._line_at = value, self.session.client.clock()
        return value

    def _queue(self, payload, rid=None, generation=None):
        self._queued.append((payload, rid, generation))

    def _flush(self):
        while self._queued:
            self.on_event(*self._queued.popleft())

    def _say(self, text):
        self._old_say(text)
        self._queue(s.Notice(str(text), 'info'))

    def _emit(self, kind, **details):
        self._old_emit(kind, **details)
        c = self.session.client
        if kind in ('rx', 'tx') and 'text' in details:
            ident = str(uuid4())
            if kind == 'tx':
                self._last_write = ident
            self._queue(s.TextRecord(ident, kind, details['text'], self.session.peer, 'contact'),
                        self._active if kind == 'tx' else None, c.link_generation)
        elif kind == 'tx' and details.get('status') in ('acknowledged', 'discarded'):
            field = details['status'] + '_bytes'
            self._credits = replace(self._credits, **{field: getattr(self._credits, field) + details['bytes']})
            self._queue(s.TxCredit(details['bytes'], details['status'], None), None, None)
        elif kind == 'band':
            self._coverage.clear()
            self._receive = replace(self._receive, coverage=(), slots=(), watched=(),
                                    passband_edges=(), wide_offset=None, table_audio_time=None)
        elif kind == 'passband':
            self._receive = replace(self._receive,
                passband_edges=tuple((k, tuple(val)) for k, val in details['edges'].items()),
                watched=tuple(details['slots']), wide_offset=details['wide_offset'], observed_at=c.clock())
        elif kind == 'bbs_download':
            self._settle_family('file', 'done', s.FileResult('validated-saved',
                                self.session.files.requested, details['path'], details['size']))
        elif kind in ('vmail', 'vmail_notice'):
            self._refresh_mail()

    def _client_event(self, event):
        if event.kind == 'host' and event.raw.startswith('CONNECTED ') and self._delivery_boundary is not None:
            self._delivery_binding = None
            try:
                boundary = self._delivery_boundary(event.raw)
                if boundary is not None and boundary.offset == 0 and boundary.queued_original == 0:
                    self.bind_delivery(boundary.epoch, self.session.client.link_generation)
            except (RuntimeError, ValueError):
                self._delivery_binding = None
        generation = self.session.client.link_generation
        if generation != self._reservation_generation:
            self._reservations = {k: value for k, value in self._reservations.items() if k == 'beacon'}
            self._reservation_generation = generation
        if self._frame is not None and (event.kind == 'kiss-sent' or
                event.kind == 'command' and event.raw.startswith(('CQFRAME', 'BEACONFORCE'))):
            self._frame['submitted'] = True
        self._old_event(event)

    def _new_attempt(self, rid, peer=None):
        ident = str(uuid4())
        self._attempt = ident
        self._attempts[ident] = dict(request_id=rid, peer=peer, generation=None, ended=False)
        return ident

    def _client_semantic(self, kind, **details):
        self._old_client_semantic(kind, **details)
        c, se = self.session.client, self.session
        if kind == 'write_failed':
            text_id = str(uuid4())
            self._queue(s.TextRecord(text_id, 'tx', details['text'], se.peer, 'contact'),
                        self._active, c.link_generation)
            self._queue(s.TxTextOutcome(text_id, None, 'unknown', len(details['payload']),
                                       0, 0, len(details['payload']), 'unattributed'),
                        self._active, c.link_generation)
            self._retire_delivery()
        elif kind == 'write_completed':
            for write in self._delivery_writes:
                if (write['write_id'], write['generation']) == (details['write_id'], details['generation']):
                    write['socket'] = details['disposition']
                    if details['disposition'] != 'complete':
                        self._read_delivery()
                        self._delivery_outcome(write, cancelled=details['disposition'] == 'cancelled')
                        self._retire_delivery()
            self._read_delivery()
        elif kind == 'frame_started':
            self._frame = dict(id=str(uuid4()), request_id=self._active or self._beacon_id or self._active_operation('chat.cq'),
                               submitted=False, completed=False, running=True)
        elif kind == 'frame_finished':
            if self._frame:
                self._frame['running'] = False
            if self._frame and (details['completed'] or details.get('not_submitted') or not self._frame['submitted']):
                self._frame = None
        elif kind == 'outbound_bound':
            if self._attempt is not None:
                attempt = self._attempts[self._attempt]
                if not attempt['ended'] and attempt['generation'] is None:
                    attempt['generation'] = details['new']
                    self._progress(attempt['request_id'], bound_generation=details['new'],
                                   attempt_id=self._attempt)
        elif kind == 'host_before' and not self._legacy:
            raw = details['raw']
            if raw == 'PENDING' and self._frame is not None:
                self._frame['submitted'] = True
            eligible = (c.listening and self._frame is None and not c.unresolved_kiss
                        and not c._outbound_connect_pending
                        and not c.connected)
            if raw.startswith('CONNECTED '):
                words = raw.split()
                peer = next((call for call in words[1:3] if call != c.mycall), se.peer)
                attempt = self._attempts.get(self._attempt)
                expected = (self._frame is None and attempt is not None and not attempt['ended']
                            and (c.pending or c._outbound_connect_pending)
                            and attempt['peer'] in (None, peer))
                if not eligible and not expected:
                    c._retire_link_messages()
                    for rid, request in self._requests.items():
                        if request.action.kind not in ('beacon.once', 'chat.close', 'chat.exit', 'chat.cancel_wait'):
                            self._finish(rid, 'unknown' if rid in self._sent else 'cancelled', code='link-replaced')
                    self._reservations = {k: value for k, value in self._reservations.items() if k == 'beacon'}
                    if attempt is not None:
                        attempt['ended'] = True
                    self._attempt = None
                    self._retire_draft()
            if raw == 'PENDING' and eligible and not c.pending:
                rid = self._active_operation('chat.cq')
                self._new_attempt(rid)
            if raw.startswith('CONNECTED ') and eligible and not c.pending:
                self._new_attempt(self._active_operation('chat.cq'))
                c._retire_link_messages()
                self._attempts[self._attempt]['generation'] = c.link_generation
        elif kind == 'host_after':
            raw = details['raw']
            if raw.startswith('CONNECTED ') or raw == 'CANCELPENDING':
                attempt = self._attempts.get(self._attempt)
                if (attempt is not None and not attempt['ended']
                        and attempt['generation'] != c.link_generation):
                    attempt['generation'] = c.link_generation
                    self._progress(attempt['request_id'], bound_generation=c.link_generation,
                                   attempt_id=self._attempt)
            if raw == 'PENDING' and self._frame is None and self._attempt is not None:
                attempt = self._attempts[self._attempt]
                if not attempt['ended'] and attempt['generation'] is None:
                    attempt['generation'] = c.link_generation
                    self._progress(attempt['request_id'], bound_generation=c.link_generation,
                                   attempt_id=self._attempt)
            if raw == 'CANCELPENDING' and self._frame is None and not c._outbound_connect_pending:
                if self._attempt is not None:
                    self._attempts[self._attempt]['ended'] = True
            if raw == 'CANCELPENDING' and self._frame is not None:
                self._frame['completed'] = True
            if self._frame is not None and self._frame['completed'] and not c.ptt and not self._frame['running']:
                self._frame = None
            if raw.startswith('CONNECTED ') and self._attempt is not None:
                attempt = self._attempts[self._attempt]
                if attempt['peer'] is not None and se.peer != attempt['peer']:
                    self._finish(attempt['request_id'], 'unknown', code='peer-replaced')
                    attempt['ended'] = True
                elif attempt['peer'] is None:
                    attempt['peer'] = se.peer
            if raw == 'DISCONNECTED':
                if self._attempt is not None:
                    self._attempts[self._attempt]['ended'] = True
                self._retire_generation()
        elif kind == 'command_submitting':
            rid = self._active
            command = details['command']
            if command.startswith('CONNECT ') and self._attempt is not None:
                rid = self._attempts[self._attempt]['request_id']
            if rid and (command == 'CLEANTXBUFFER' or command.startswith('CONNECT ')):
                self._sent.add(rid)
                self._progress(rid, stage='sent')
        elif kind == 'write_submitting':
            if self._active:
                self._sent.add(self._active)
        elif kind == 'write_admitted':
            rid = self._active
            text = details['text'].strip()
            family = ('directory' if text == '<BLR>' else 'file' if text.startswith('<BG:') else
                      'mail' if text.startswith('<SM>') else 'term' if text.startswith('BBS TERM ')
                      and not text.startswith('BBS TERM OK ') else None)
            if rid is None and not self._legacy:
                rid = self._active_operation('bbs.list') if text == '<BLR>' else None
            if rid is None and family is not None:
                previous = self._family_id(family)
                if previous:
                    self._finish(previous, 'unknown' if previous in self._sent else 'cancelled',
                                 code='legacy-overlap')
                self._reservations[family] = s.ServiceReservation(family, 'spent', None)
            if rid:
                self._sent.add(rid)
                family = SERVICE_FAMILIES.get(self._requests[rid].action.kind)
                if family:
                    self._reservations[family] = s.ServiceReservation(family, 'pending', rid)
                self._progress(rid, stage='sent')
            text_id = self._last_write or str(uuid4())
            binding = self._delivery_binding
            start = binding[2] if binding and binding[1] == c.link_generation else None
            write = dict(text_id=text_id, write_id=details['write_id'], payload=bytes(details['payload']),
                         request_id=rid, generation=c.link_generation, start=start,
                         epoch=binding[0] if start is not None else None,
                         socket='complete' if details['write_id'] is None else 'pending',
                         acknowledged=0, discarded=0, outcome=None, retired=False)
            self._delivery_writes.append(write)
            self._delivery_bytes += len(write['payload'])
            while len(self._delivery_writes) > 2000 or self._delivery_bytes > 2 * 1024 * 1024:
                evicted = self._delivery_writes.popleft()
                self._delivery_bytes -= len(evicted['payload'])
                evicted['retired'] = True
                self._delivery_outcome(evicted, verified=bool(evicted['acknowledged'] or evicted['discarded']))
            if start is not None:
                self._delivery_binding = (binding[0], binding[1], start + len(write['payload']))
            self._delivery_outcome(write)

    def bind_delivery(self, epoch, generation, *, offset=0):
        self._assert_owner()
        if (generation != self.session.client.link_generation or offset < 0 or not epoch or
                any(w['generation'] == generation for w in self._delivery_writes)):
            raise ValueError('delivery binding requires a fresh current stream boundary')
        self._delivery_binding = (epoch, generation, offset)
        self._delivery_sequence = -1

    def _delivery_outcome(self, write, *, verified=False, cancelled=False):
        size = len(write['payload'])
        ack, discard = write['acknowledged'], write['discarded']
        if cancelled:
            discard = write['discarded'] = size - ack
        unknown = size - ack - discard
        status = ('acknowledged' if ack == size else 'discarded' if discard and not unknown else
                  'unknown' if write['retired'] or write['socket'] == 'unknown' else
                  'partial' if ack or discard else 'queued')
        outcome = s.TxTextOutcome(write['text_id'], write['write_id'], status, size, ack, discard, unknown,
                                  'verified' if verified or cancelled else 'unattributed')
        if outcome != write['outcome']:
            write['outcome'] = outcome
            self._queue(outcome, write['request_id'], write['generation'])
        rid = write['request_id']
        if (verified and ack == size and rid is not None and rid not in self._email_acknowledged
                and isinstance(self._requests[rid].action, s.EmailSend)):
            self._email_acknowledged.add(rid)
            self._queue(s.ApplicationOutcome(s.MailResult('email', 'modem-ack')), rid, write['generation'])

    def observe_delivery(self, evidence):
        self._assert_owner()
        binding = self._delivery_binding
        if binding is None or evidence.epoch != binding[0] or evidence.sequence < self._delivery_sequence:
            return
        self._delivery_sequence = evidence.sequence
        for write in self._delivery_writes:
            if write['epoch'] != evidence.epoch or write['retired']:
                continue
            start, size = write['start'], len(write['payload'])
            if write['socket'] != 'complete' or start + size > evidence.accepted:
                if evidence.retired:
                    write['retired'] = True
                    self._delivery_outcome(write)
                continue
            if not evidence.complete:
                write['retired'] = True
                self._delivery_outcome(write, verified=bool(write['acknowledged'] or write['discarded']))
                continue
            write['acknowledged'] = _covered(evidence.acknowledged, start, start + size)
            write['discarded'] = _covered(evidence.discarded, start, start + size)
            write['retired'] = evidence.retired
            self._delivery_outcome(write, verified=True)

    def _read_delivery(self):
        if self._delivery_evidence is not None:
            captured = self._delivery_evidence()
            if captured and self._delivery_binding and not any(
                    evidence.epoch == self._delivery_binding[0] for evidence in captured):
                epoch = self._delivery_binding[0]
                self._delivery_binding = None
                for write in self._delivery_writes:
                    if write['epoch'] == epoch and not write['retired']:
                        write['retired'] = True
                        self._delivery_outcome(write, verified=bool(write['acknowledged'] or write['discarded']))
            for evidence in captured:
                self.observe_delivery(evidence)

    def _semantic(self, kind, **details):
        self._old_semantic(kind, **details)
        if kind == 'attempt_started':
            self._new_attempt(self._active, details['peer'])
        elif kind == 'operation_started':
            rid = self._active or (self._beacon_id if self._beacon_running else None)
            if rid is None and details.get('operation') == 'bbs.list':
                rid = self._active_operation('bbs.list')
            if rid:
                self._operations[id(details['steps'])] = rid
        elif kind in ('operation_finished', 'operation_failed'):
            rid = self._operations.pop(id(details['steps']), None)
            if rid:
                if kind == 'operation_failed':
                    error = details['error']
                    closing = self._requests[rid].action.kind in ('chat.close', 'chat.exit', 'chat.cancel_wait')
                    state = 'unknown' if rid in self._sent and not closing and isinstance(error, (TimeoutError, v.HostError)) else 'failed'
                    self._finish(rid, state, code='operation-error', detail=str(error))
                else:
                    self._operation_done(rid, details['value'])
        elif kind == 'error':
            self._error = details['error']
        elif kind == 'directory_rejected':
            self._settle_family('directory', 'failed', s.DirectoryResult((), 'rejected'))
        elif kind == 'directory_received':
            self._settle_family('directory', 'done', s.DirectoryResult(tuple(
                s.FileEntry(e.name, e.created, e.size) for e in details['entries']), 'complete'))
        elif kind == 'mail_notices_changed':
            self._refresh_mail()
        elif kind == 'mail_ack_outcome':
            rid = self._family_id('mail')
            mail_kind, tag = details['mail_kind'], details['tag']
            if rid and ((mail_kind == 'direct' and isinstance(self._requests[rid].action, s.VMailSend))
                        or (mail_kind == 'email' and isinstance(self._requests[rid].action, s.EmailSend))):
                if tag is None:
                    self._finish(rid, 'unknown', code='shutdown' if self.session.client.closed else 'mail-unresolved',
                                 detail=details['reason'])
                elif mail_kind == 'direct':
                    self._finish(rid, 'done' if tag == 'SMR' else 'failed', s.MailResult('vmail', tag.lower()))
                elif tag == 'ER':
                    self._finish(rid, 'done', s.MailResult('email', 'er'))
                elif tag in ('EJ', 'SMF', 'SMFP'):
                    self._finish(rid, 'failed', s.MailResult('email', 'rejected'))
        elif kind == 'beacon_scheduled':
            if self._beacon_id and self._beacon_id != self._active:
                self._finish(self._beacon_id, 'cancelled', code='legacy-replacement')
            self._beacon_id = self._active
        elif kind == 'beacon_expiry_started':
            self._expiring = True
        elif kind == 'beacon_expiry_finished':
            self._expiring = False
            if not details['restored']:
                self._finish(self._beacon_id, 'failed', code='expiry-failed')
        elif kind == 'beacon_removed' and not self._expiring and not self._beacon_running:
            self._finish(self._beacon_id, 'cancelled', code='beacon-removed')
        elif kind == 'beacon_started':
            self._beacon_running = bool(details['once'])
        elif kind == 'beacon_failed':
            self._finish(self._beacon_id, 'unknown' if self._frame else 'failed',
                         s.BeaconResult('not-sent'), code='beacon-failed', detail=str(details['error']))
        elif kind == 'beacon_sent':
            self._finish(self._beacon_id, 'done', s.BeaconResult('sent'))
        elif kind == 'broadcast_submitting':
            rid = self._active_operation('broadcast.send')
            if rid:
                self._sent.add(rid)
                self._progress(rid, stage='sent')
        elif kind == 'broadcast_sent':
            self._finish(self._active_operation('broadcast.send'), 'done',
                         s.BroadcastResult(self.session.broadcast.possibly_outstanding, 'sent'))
        elif kind == 'broadcast_cancelled':
            rid = self._active_operation('broadcast.send')
            self._finish(rid, 'unknown' if rid in self._sent else 'cancelled', code='broadcast-cancelled')
        elif kind == 'email_rejected':
            self._settle_family('mail', 'failed', s.MailResult('email', 'rejected'))
        elif kind == 'file_outcome':
            result = s.FileResult(details['result'], self.session.files.requested,
                                  details.get('evidence_id'), self.session.files.received_bytes)
            self._settle_family('file', details['state'], result)
        elif kind == 'reload_finished':
            rid = self._active_operation('bbs.reload')
            self._finish(rid, 'done', s.ReloadResult(details['revision']))
        elif kind == 'reload_failed':
            self._finish(self._active_operation('bbs.reload'), 'failed',
                         code='reload-failed', detail=str(details['error']))
        elif kind == 'monitor_failed':
            self._receive = replace(self._receive, monitor_error=str(details['error']))
        elif kind == 'slot_observed':
            said = details['said']
            if said.kind == 'coverage':
                self._coverage.append((self.session.client.clock(), said))
            else:
                self._queue(s.SlotRecord(said.slot, said.at, said.kind, said.text, said.station or None))
        elif kind == 'bbs_document':
            b = self.session.bbs
            spans = tuple(s.StyledSpan(span.text, span.style.foreground if b.remote_ansi else None,
                          span.style.background if b.remote_ansi else None,
                          span.style.bold if b.remote_ansi else False) for span in details['document'].spans)
            self._queue(s.BBSPage(str(uuid4()), 'ansi16' if b.remote_ansi else 'plain',
                                 *b.remote_dimensions, spans), None, self.session.client.link_generation)

    def _progress(self, rid, **kwargs):
        if rid and rid not in self._results:
            self._queue(Disposition('running', **kwargs), rid, self.session.client.link_generation)

    def _active_operation(self, kind):
        return next((rid for rid, r in self._requests.items()
                     if r.action.kind == kind and rid not in self._results), None)

    def _family_id(self, family):
        reservation = self._reservations.get(family)
        return reservation.request_id if reservation and reservation.state == 'pending' else None

    def _settle_family(self, family, state, result):
        self._finish(self._family_id(family), state, result)

    def _finish(self, rid, state, result=None, *, code=None, detail=''):
        if rid is None or rid in self._results:
            return
        attempt_id = next((ident for ident, a in reversed(tuple(self._attempts.items()))
                           if a['request_id'] == rid), None)
        generation = self._attempts[attempt_id]['generation'] if attempt_id else None
        disposition = Disposition(state, result, reason_code=code, detail=detail,
                                  bound_generation=generation, attempt_id=attempt_id)
        self._results[rid] = disposition
        family = SERVICE_FAMILIES.get(self._requests[rid].action.kind)
        reservation = self._reservations.get(family)
        if reservation is not None and reservation.request_id == rid:
            self._reservations[family] = s.ServiceReservation(
                family, 'spent' if rid in self._sent and family != 'beacon' else 'available',
                rid if rid in self._sent and family != 'beacon' else None)
        if rid == self._beacon_id:
            self._beacon_id, self._beacon_running = None, False
        self._deadlines.pop(rid, None)
        self._queue(disposition, rid, generation)

    def _operation_done(self, rid, value):
        action = self._requests[rid].action
        if isinstance(action, (s.ChatCall, s.ChatAnswer, s.VMailCollect, s.ChatCQ)):
            attempt_id = next((i for i, a in reversed(tuple(self._attempts.items())) if a['request_id'] == rid), None)
            generation = self._attempts[attempt_id]['generation'] if attempt_id else None
            if isinstance(action, s.ChatCQ):
                state = 'cancelled' if self.session.flow.cancelled else 'done'
                outcome = 'connected' if value else 'no-answer' if self.session._called else 'busy'
            else:
                state = 'done' if value else 'cancelled'
                outcome = 'connected' if value else 'no-answer'
            self._finish(rid, state, s.ConnectionResult(attempt_id, generation, outcome))
        elif isinstance(action, (s.ChatClose, s.ChatExit, s.ChatCancelWait)):
            line = self._observe_line()
            good = line is False and not self.session.client.connected
            outcome = (s.WaitStopped(None, 'link-closed') if isinstance(action, s.ChatCancelWait) else
                       s.LinkClosed('low' if line is False else 'high' if line else 'unknown'))
            self._finish(rid, 'done' if good else 'failed', outcome)
        elif isinstance(action, s.ChatQSY):
            self._finish(rid, 'done', s.Applied(0))

    def _retire_delivery(self):
        self._read_delivery()
        self._delivery_binding = None
        for write in self._delivery_writes:
            if not write['retired']:
                write['retired'] = True
                self._delivery_outcome(write, verified=bool(write['acknowledged'] or write['discarded']))

    def _retire_generation(self):
        self._retire_delivery()
        for rid, request in self._requests.items():
            if rid in self._results or request.action.kind in (
                    'beacon.once', 'chat.close', 'chat.exit', 'chat.cancel_wait',
                    'chat.call', 'chat.answer', 'vmail.collect') or (
                    isinstance(request.action, s.ChatCQ) and request.action.repeat):
                continue
            self._finish(rid, 'unknown' if rid in self._sent else 'cancelled', code='link-retired')
        self._reservations = {k: val for k, val in self._reservations.items() if k == 'beacon'}
        self._retire_draft()

    def _retire_draft(self):
        self._retired_draft = self.session.email.draft or self.session.compose
        self._draft = None

    def _refresh_mail(self):
        self._mail_loaded = True
        self._mailbox = tuple(s.MailHeader(n, str(row['id']), row['direction'], row['from'],
                             row['to'], row['subject'], row['utc'])
                              for n, row in enumerate(self.session.mailbox.entries(), 1))
        self._notices = tuple(s.MailNotice(row['source'], row['destination'], row['utc'],
                              row['dial_hz'], row.get('received', 0), row.get('urgent', False))
                               for row in self.session.mail_collection._load())

    def _sync_draft(self):
        se = self.session
        if se.email.draft is not None:
            generation, email, lines = se.email.draft
            raw = se.email.draft
            kind, subject, recipient, reply_to = 'email', email.subject, email.recipient, email.reply_to
        elif se.compose is not None:
            subject, lines = se.compose
            raw = se.compose
            generation = se.client.link_generation
            kind, recipient, reply_to = 'vmail', None, None
        else:
            self._draft = self._legacy_draft = None
            return
        if raw is self._retired_draft:
            self._draft = None
            return
        current = self._draft
        content = (kind, generation, se.peer, subject, recipient, reply_to, tuple(lines))
        if current is not None and (current.kind, current.generation, current.peer, current.subject,
                                   current.recipient, current.reply_to, current.lines) == content:
            self._legacy_draft = raw
            return
        retained = current is not None and current.owner == 'legacy' and raw is self._legacy_draft
        self._draft = s.Draft(current.id if retained else str(uuid4()), current.revision + 1 if retained else 0,
                              'legacy', *content)
        self._legacy_draft = raw

    def project(self):
        self._assert_owner()
        se, c, f = self.session, self.session.client, self.session.flow
        now = c.clock()
        self._sync_draft()
        if not self._mail_loaded:
            self._refresh_mail()
        facts = (c.link_generation, c.connected, c.pending, c.closed, se._closing, f.state, se.peer)
        if self._link_facts is not None and (c.closed or self._link_facts[0] != c.link_generation):
            self._retire_delivery()
        if self._link_facts != facts:
            self._link_facts = facts
            self._generation = c.link_generation
            self._link_transition = now
        epoch = (id(se.slots), getattr(se.slots, '_generation', None), f.band)
        if self._monitor_epoch is not None and self._monitor_epoch != epoch:
            self._coverage.clear()
        self._monitor_epoch = epoch
        slot = next((n for n in f.codec.slots if v.slot_frequency(f.band, n, f.codec) == f.dial), None)
        key, description, next_action = se._operator_phase()
        phase, subphase = key if isinstance(key, tuple) else (key, None)
        pending, data_text = se._data_status()
        contact = se.contact or {}
        setup = tuple((str(k), str(val)) for k, val in (se.band_setup or {}).items())
        p = s.initial_snapshot()
        identity = s.Identity(c.mycall, se.idcall, f.band, f.dial, se.calling_hz, slot,
                              c.bandwidth, se._band_fault, setup)
        link = s.Link(c.link_generation,
            'host-lost' if c.closed else 'closing' if se._closing else 'established' if c.connected else
            'connecting' if c.pending or f.state is v.State.CONNECTING else 'idle',
            s.State(f.state.value), se.peer, 'host-connected' if c.connected else
            'operator-target' if se.peer and se.intent else None, contact.get('n'), contact.get('how'),
            contact.get('started_utc'), c.connected, c.pending, c.closed, self._link_transition)
        operation = s.Operation(s.Phase(phase), subphase, description, next_action, se.intent,
            se.steps is not None, f.qsy_slot, f.qsy_receiving, None, se._closing, se._quitting,
            str(se._close_failed) if se._close_failed else None)
        unreported = c._sent_after is not None and c.buffer_sequence <= c._sent_after
        tx = s.Tx(c.ptt, self._line, self._line_at, c.buffer, c.buffer_sequence, c.sequence,
                  c._sent_after, bool(c.data_writer and c.data_writer.pending), unreported,
                  pending, data_text, c._stream_poisoned,
                  (self._credits,) if self._credits.acknowledged_bytes or self._credits.discarded_bytes else ())
        reason = ('closing' if se._closing or se._quitting else 'passive-follow' if se._follow_tuned else
                  'broadcast' if se.broadcast.selected is not None else
                  'unready-band' if se.auto and not se._auto_answer else None)
        reporter = tuple((str(k), str(val)) for k, val in se.reporter.status().items()) if se.reporter else (('enabled', 'false'),)
        policy = s.Policy(se.auto, se._auto_answer and reason is None,
                          'off' if not se.auto else 'suspended' if reason else 'on', reason,
                          se.away, se._away_owed, se.new_vmails, se.following, se._follow_tuned,
                          se.verbose, reporter)
        h = se.hopping
        hop = s.Hop(h.active, h.auto, tuple(h.bands), getattr(h, 'minutes', None), h.due,
                    h.pending, h.paused, tuple(h.readiness.items()), tuple(h.last_tx.items()))
        beacon = s.Beacon(se.beacon, se._beacon_due, se._beacon_expires, se._beacon_last,
                         se._beacon_now or None, self._beacon_id, c.advanced_beacon_available,
                         se._beacon_generation, se.beacon_pause_reason(line_state=lambda: self._line), hop)
        b = se.broadcast
        broadcast = s.Broadcast(b.selected, b.active, b.phase, b.cancelled, b.possibly_outstanding, b.result)
        b, files = se.bbs, se.files
        bbs = s.BBSState(b.active or se.bbs_host.active, b.remote_view, b.owned, b.sid, b.generation, b.remote_ansi,
                        tuple(b.remote_dimensions), b.requested_term, b.reload_future is not None,
                        se.bbs_host.name if se.bbs_host.active else None)
        file_state = s.FileState(files.active, files.downloading, files.receiving, files.opaque,
            files.generation, files.list_generation, files.requested,
            tuple(s.FileEntry(e.name, e.created, e.size) for e in files.entries),
            str(files.offer.wire_name) if files.offer else None, files.received_bytes,
            files.offer_deadline, files.failure)
        parking, ack = se.parking, se.mail_ack
        settings = parking.store.settings
        parked = tuple(s.ParkedMail(row['id'], row['from'], row['to'], row['subject'],
                         row['state'], row['attempts'], row['expires_unix'], row['last_error'],
                         tuple(row['notification'].items()))
                       for row in parking.store.records if row['owner'] == v.canonical_call(c.mycall)) if parking.store.available else ()
        parking_state = s.ParkingState(parking.store.available, parking.store.error,
            settings['parking'], settings['relay'], settings['expiry_days'],
            settings['per_destination'], settings['total'], parking.active, parking.started, parking.stopped,
            parking.notifying, tuple(parking.candidates),
            (parking.draft['recipient'], parking.draft['subject'], tuple(parking.draft['lines'])) if parking.draft else None,
            parked)
        application = s.Application(self._draft, se._vmail_sent, se._bbs_list_queued, se._bbs_waiting,
            s.EmailState(se.email.waiting, se.email.phase, se.email.sent_generation), bbs,
            file_state, self._mailbox, self._notices,
            s.MailAckState(ack.generation, ack.state, ack.kind, ack.result, ack.reason), parking_state)
        heard = tuple(s.HeardRow(e.n, e.call, e.slot, e.bw, e.sn, e.at, e.beacon, e.grid,
                                e.status, e.via, e.band, se.where(e)) for e in se.heard)
        if self._quit:
            slots, table_at = self._receive.slots, self._receive.table_audio_time
        else:
            rows, table_at = se.slots.table() if se.slots is not None else ((), None)
            slots = tuple(s.SlotRow(e.slot, e.audio_hz, e.state, e.heard, tuple(e.calls), e.level,
                                  tuple(e.tags), e.home, e.caller, e.called, e.stage, e.since,
                                  e.at, e.sn, e.one_side) for e in rows)
        while self._coverage and self._coverage[0][0] <= now - 60:
            self._coverage.popleft()
        coverage = []
        for channel in dict.fromkeys(said.slot for _, said in self._coverage):
            observations = [(at, said) for at, said in self._coverage if said.slot == channel]
            at, said = observations[-1]
            durations = [item.duration_s for _, item in observations]
            lost = sum(durations) if all(d is not None for d in durations) else None
            coverage.append(s.Coverage(channel, said.text, at, said.at, lost,
                                      min(1., lost / 60) if lost is not None else None, 'observed-events'))
        old = self._receive
        observed = now if (heard, slots, table_at, c.busy, c.snr, c.bitrate) != (
            old.heard, old.slots, old.table_audio_time, old.host_busy, old.snr, old.bitrate) else old.observed_at
        self._receive = replace(old, host_busy=c.busy, snr=c.snr, bitrate=c.bitrate,
                                heard=heard, slots=slots, watched=tuple(row.slot for row in slots) if se.slots else old.watched,
                                table_audio_time=table_at, observed_at=observed, coverage=tuple(coverage))
        capabilities = tuple(self._capability(kind) for kind in s.ACTIONS)
        return replace(p, identity=identity, link=link, operation=operation, tx=tx, policy=policy,
                       beacon=beacon, broadcast=broadcast, application=application,
                       receive=self._receive, ownership=self._ownership, capabilities=capabilities)

    def _capability(self, kind, *, code=None, reason=None):
        family = SERVICE_FAMILIES.get(kind)
        reservation = self._reservations.get(family, s.ServiceReservation(family, 'available', None)) if family else None
        if reservation and reservation.state != 'available':
            code = 'beacon-pending' if family == 'beacon' else (
                'service-pending' if reservation.state == 'pending' else 'reconnect-required')
            reason = 'Wait for the pending operation' if reservation.state == 'pending' else 'Reconnect before another request'
        return s.Capability(kind, s.ACTIONS[kind][1], 'blocked' if code else
                            'check-on-execution' if s.ACTIONS[kind][1] is s.Effect.X else 'allowed',
                            reservation, code, reason)

    def preflight(self, action):
        self._assert_owner()
        se, c = self.session, self.session.client
        if isinstance(action, s.ChatTag) and action.name in ('BLR', 'LHR') and action.value is None:
            delegated = s.BBSList() if action.name == 'BLR' else s.InquireHeard()
            return replace(self.preflight(delegated), kind=action.kind)
        cap = self._capability(action.kind)
        if cap.availability == 'blocked':
            return cap
        try:
            if se.broadcast.possibly_outstanding and isinstance(action, (
                    s.ChatCall, s.ChatAnswer, s.ChatCQ, s.VMailCollect, s.BandSet, s.FollowSet,
                    s.AutoSet, s.AwaySet, s.AwayBack)):
                raise ValueError('Acknowledge outstanding frames with /all clear before retuning')
            if isinstance(action, (s.ChatCall, s.ChatAnswer, s.ChatCQ, s.BeaconOnce, s.VMailCollect)) and self._frame:
                raise ValueError('an interrupted submitted frame remains unresolved')
            if isinstance(action, (s.ChatSend, s.ChatTag, s.InquireHeard, s.VMailCompose, s.EmailCompose,
                                   s.VMailSend, s.EmailSend, s.BBSList, s.BBSGet, s.BBSRequest,
                                   s.BBSCapture, s.BBSEnter, s.BBSTerm, s.BBSInput, s.ChatQSY)):
                if not c.connected or c.closed:
                    raise ValueError('an established link is required')
            if isinstance(action, (s.ChatSend, s.ChatTag, s.VMailCompose, s.EmailCompose, s.VMailSend, s.EmailSend)):
                se.check_application_input()
            if isinstance(action, (s.ChatSend, s.ChatTag)):
                if se.bbs.owned or self._draft or se.compose or se.email.draft:
                    raise ValueError('use the current draft or BBS input action')
                if se._closing or se.broadcast.active or se.files.active or se.bbs_host.active:
                    raise ValueError('another operation owns the link')
            if isinstance(action, (s.ChatSend, s.ChatTag, s.BBSInput)):
                text = (v.emit_tag(action.name, action.value) if isinstance(action, s.ChatTag) else action.text)
                if any(marker in text.upper() for marker in ('<BLR>', '<LHR>', '<BG:', '<SM>', 'BBS TERM')):
                    raise ValueError('reserved service request requires its typed action')
                if len(text.encode('utf-8')) > 65536:
                    raise ValueError('text exceeds 64 KiB')
            if isinstance(action, (s.ChatCall, s.ChatAnswer, s.ChatCQ, s.VMailCollect)):
                if se._band_fault:
                    raise ValueError('band setup unverified')
                if se._closing_dial is not None:
                    raise ValueError('closing identification pending')
                if not (se.repeat or se.intent == 'beacon'):
                    se.check_outgoing()
            if isinstance(action, s.ChatCall):
                v.callsign(action.peer)
                if action.slot is not None:
                    v.slot_frequency(se.flow.band, action.slot, se.flow.codec)
            if isinstance(action, s.ChatAnswer):
                if not 1 <= action.heard_number <= len(se.heard):
                    raise ValueError('heard number outside current list')
                if se.heard[action.heard_number - 1].band not in (None, se.flow.band):
                    raise ValueError('heard station is on another band')
            if isinstance(action, (s.ChatQSY, s.ChatCQ)) and action.slot is not None:
                v.slot_frequency(se.flow.band, action.slot, se.flow.codec)
            if isinstance(action, s.ChatQSY) and se.steps is not None:
                raise ValueError('another exchange is in progress')
            if isinstance(action, s.BBSList):
                se.check_list_bbs()
            if isinstance(action, s.InquireHeard):
                se.check_inquire_heard()
            if isinstance(action, s.VMailCompose):
                se.check_compose_vmail(action.subject)
            if isinstance(action, s.EmailCompose):
                se.email.check_compose(action.recipient, action.reply_to, action.subject)
            if isinstance(action, (s.DraftAppend, s.DraftDrop, s.VMailSend, s.EmailSend)):
                draft = self._draft
                if draft is None or (draft.id, draft.revision) != (action.draft_id, action.expected_revision):
                    return self._capability(action.kind, code='draft-stale', reason='Draft ID or revision changed')
                if (draft.generation, draft.peer) != (c.link_generation, se.peer):
                    return self._capability(action.kind, code='draft-link-stale', reason='Draft link changed')
                if isinstance(action, (s.VMailSend, s.EmailSend)) and draft.kind != ('vmail' if isinstance(action, s.VMailSend) else 'email'):
                    raise ValueError('draft kind differs')
                if isinstance(action, s.VMailSend):
                    se.check_send_vmail()
                    self._validate_draft(draft, draft.lines)
                if isinstance(action, s.EmailSend):
                    se.email._guard(draft.generation)
                    self._validate_draft(draft, draft.lines)
                    if se.steps is not None:
                        raise ValueError('another exchange is active; draft retained')
                if isinstance(action, s.DraftAppend):
                    self._validate_draft(draft, draft.lines + (action.line,))
            if isinstance(action, (s.BroadcastSelect, s.BroadcastSend, s.BroadcastOff)):
                se.broadcast.check_idle()
                if isinstance(action, s.BroadcastSend) and action.text.strip() in ('clear', 'yes', 'no'):
                    raise ValueError('Use the typed broadcast acknowledgement action')
                if self._line is True:
                    raise ValueError('keying line HIGH')
                if isinstance(action, s.BroadcastSend):
                    if self._line is not False or c.busy or se.broadcast.selected != se.flow.dial or se.broadcast.selected is None:
                        raise ValueError('broadcast channel or unkey not ready')
                    from ..kestrel.broadcast import Broadcast
                    from ..kestrel.rx.broadcast import packetize
                    packetize(Broadcast(1, c.mycall, 'ALL', action.text.encode('utf-8')))
            if isinstance(action, s.BeaconOnce):
                if se._beacon_now or self._beacon_running:
                    return self._capability(action.kind, code='beacon-pending', reason='A one-shot beacon is pending')
                se.check_beacon_now(line_state=lambda: self._line)
            if isinstance(action, s.BeaconConfigure) and action.minutes is not None and action.minutes <= 0:
                raise ValueError('positive beacon minutes required')
            if isinstance(action, s.BeaconHop):
                if (len(set(action.bands)) != len(action.bands) or not action.bands or
                        any(b not in v.CALLING_HZ for b in action.bands) or action.minutes < 5):
                    raise ValueError('invalid hopping configuration')
            if isinstance(action, s.BandSet):
                se.check_change_band(action.band_m, line_state=lambda: self._line)
            if isinstance(action, s.AwaySet) and any(ch in action.text for ch in '<>\r\n'):
                raise ValueError('away text must be one plain line')
            if isinstance(action, (s.BBSGet, s.BBSRequest, s.BBSCapture)):
                se.files.guard()
            if isinstance(action, s.BBSEnter) and not se.bbs.settled():
                raise ValueError('BBS entry needs a settled off-CF link')
            if isinstance(action, s.BBSTerm):
                se.bbs.check_term(action.mode, action.width, action.height)
            if isinstance(action, s.BBSReload) and (se.bbs.core is None or se.bbs.reload is None or se.bbs.reload_future is not None):
                raise ValueError('BBS reload unavailable or pending')
        except (ValueError, v.HostError, TypeError) as error:
            return self._capability(action.kind, code='guard-refused', reason=str(error))
        return cap

    def _validate_draft(self, draft, lines):
        if len('\n'.join(lines).encode('utf-8')) > 65536:
            raise ValueError('draft exceeds 64 KiB')
        if draft.kind == 'vmail':
            from .mail_parking import validate_content
            validate_content(draft.subject, '\r\n'.join(lines), local=True)
            v.VMail(draft.subject, '\r\n'.join(lines)).wire()
        if draft.kind == 'email':
            from .email_gateway import Email
            Email(draft.recipient, draft.reply_to, draft.subject, '\r\n'.join(lines)).wire()

    def _compose(self, request):
        a, se = request.action, self.session
        if self._draft is not None and self._draft.owner != request.actor:
            return Disposition('refused', reason_code='draft-owned', detail='Another view owns the draft')
        if isinstance(a, s.VMailCompose):
            se.compose_vmail(a.subject)
            se.email.draft = None
            kind, recipient, reply_to = 'vmail', None, None
        else:
            se.email.command(['send', a.recipient, a.reply_to, a.subject])
            se.compose = None
            kind, recipient, reply_to = 'email', a.recipient, a.reply_to
        self._draft = s.Draft(str(uuid4()), 0, request.actor, kind, se.client.link_generation,
                              se.peer, a.subject, recipient, reply_to, ())
        return Disposition('done', s.DraftAccepted(self._draft))

    def execute(self, request):
        self._assert_owner()
        se, c, a = self.session, self.session.client, request.action
        if isinstance(a, s.ChatTag) and a.name in ('BLR', 'LHR') and a.value is None:
            a = s.BBSList() if a.name == 'BLR' else s.InquireHeard()
            request = replace(request, action=a)
        self._requests[request.id] = request
        self._active, self._error, self._last_write = request.id, None, None
        try:
            capability = self.preflight(a)
            if capability.availability == 'blocked':
                self._finish(request.id, 'refused', code=capability.reason_code, detail=capability.reason or '')
                return self._results[request.id]
            if isinstance(a, (s.DraftAppend, s.DraftDrop, s.VMailSend, s.EmailSend)):
                if self._draft.owner != request.actor:
                    self._finish(request.id, 'refused', code='draft-owned')
                    return self._results[request.id]
            if self._draft and not s.preserves_draft(a):
                self._draft, se.compose, se.email.draft = None, None, None
            family = SERVICE_FAMILIES.get(a.kind)
            if family:
                self._reservations[family] = s.ServiceReservation(family, 'pending', request.id)
            if isinstance(a, (s.VMailCompose, s.EmailCompose)):
                disposition = self._compose(request)
            elif isinstance(a, s.DraftAppend):
                draft = self._draft
                self._validate_draft(draft, draft.lines + (a.line,))
                if draft.kind == 'vmail':
                    se.compose[1].append(a.line)
                else:
                    se.email.draft[2].append(a.line)
                self._draft = replace(draft, revision=draft.revision + 1, lines=draft.lines + (a.line,))
                disposition = Disposition('done', s.DraftAccepted(self._draft))
            elif isinstance(a, s.DraftDrop):
                self._draft, se.compose, se.email.draft = None, None, None
                disposition = Disposition('done', s.DraftAccepted(None))
            elif isinstance(a, (s.VMailSend, s.EmailSend)):
                self._draft = None
                if isinstance(a, s.VMailSend):
                    se.send_vmail()
                else:
                    se.email.line('.')
                disposition = Disposition('running', stage='waiting', bound_generation=c.link_generation)
            elif isinstance(a, s.ChatSend):
                wid = c.send_text(a.text + '\n')
                disposition = Disposition('done', s.WriteAccepted(wid, self._last_write))
            elif isinstance(a, s.ChatTag):
                c.send_tag(a.name, a.value)
                disposition = Disposition('done', s.WriteAccepted(None, self._last_write))
            elif isinstance(a, s.InquireHeard):
                se.inquire_heard()
                disposition = Disposition('done', s.WriteAccepted(None, self._last_write))
            elif isinstance(a, (s.ChatCall, s.ChatAnswer, s.ChatCQ, s.VMailCollect)):
                if isinstance(a, s.ChatCall):
                    se.call(a.peer, a.slot, auto_qsy=a.auto_qsy)
                elif isinstance(a, s.ChatAnswer):
                    se.answer(a.heard_number)
                elif isinstance(a, s.ChatCQ):
                    se.cq(a.slot, a.repeat)
                else:
                    se.mail_collection.command(['collect', a.peer])
                disposition = Disposition('running', stage='waiting')
            elif isinstance(a, s.ChatQSY):
                se._run(se.flow.request_qsy_steps(a.slot))
                disposition = Disposition('running', stage='waiting', bound_generation=c.link_generation)
            elif isinstance(a, s.ChatCancelWait):
                targets = [rid for rid in self._requests if rid != request.id and rid not in self._results]
                before, steps, beacon = c.connected, se.steps, se._beacon_now
                se.stop()
                for rid in targets:
                    if self._requests[rid].action.kind == 'beacon.once' and (se._beacon_now or self._beacon_running):
                        continue
                    self._finish(rid, 'unknown' if rid in self._sent else 'cancelled', code='operator-cancel')
                effect = 'link-closed' if before and not c.connected else 'wait-only' if steps or beacon or targets else 'no-op'
                disposition = (Disposition('running', stage='waiting') if se.steps is not None and se._closing else
                               Disposition('done', s.WaitStopped(targets[0] if targets else None, effect)))
            elif isinstance(a, s.ChatClose):
                if se.steps is not None:
                    se.steps.close()
                se._run(se._farewell(a.text))
                disposition = Disposition('running', stage='waiting')
            elif isinstance(a, s.ChatExit):
                disposition = Disposition('running', stage='waiting')
            elif isinstance(a, s.BBSList):
                se.list_bbs()
                disposition = Disposition('running', stage='queued' if se._bbs_list_queued is not None else 'waiting',
                                          bound_generation=c.link_generation)
            elif isinstance(a, (s.BBSGet, s.BBSRequest, s.BBSCapture)):
                if isinstance(a, s.BBSCapture):
                    se.files.capture()
                    disposition = Disposition('done', s.FileResult('capture-started', se.files.requested,
                                              str(se.files.offer_path) if se.files.offer_path else None, se.files.received_bytes))
                else:
                    se.files.command('get' if isinstance(a, s.BBSGet) else 'request', a.selection)
                    disposition = Disposition('running', stage='waiting', bound_generation=c.link_generation)
            elif isinstance(a, (s.BBSEnter, s.BBSTerm, s.BBSInput, s.BBSReload)):
                if isinstance(a, s.BBSEnter):
                    se.bbs.command('/bbs enter')
                elif isinstance(a, s.BBSTerm):
                    se.bbs.command(f'/bbs term {a.mode} {a.width} {a.height}')
                    self._deadlines[request.id] = c.clock() + 60
                elif isinstance(a, s.BBSInput):
                    # Typed menu text is never reinterpreted as a local slash command.
                    c.send_text(a.text if a.text else 'BBS MORE')
                else:
                    se.bbs.command('/bbs reload')
                disposition = (Disposition('running', stage='waiting') if isinstance(a, (s.BBSTerm, s.BBSReload))
                               else Disposition('done', s.WriteAccepted(None, self._last_write)))
            elif isinstance(a, (s.HeardList, s.SlotsList, s.FilesList, s.VMailList, s.VMailPending)):
                if isinstance(a, (s.VMailList, s.VMailPending)):
                    self._refresh_mail()
                snap = self.project()
                values = {'heard.list': s.HeardRows(snap.receive.heard),
                          'slots.list': s.SlotRows(snap.receive.slots),
                          'files.list': s.FileEntries(snap.application.files.list_generation, snap.application.files.entries),
                          'vmail.list': s.MailHeaders(self._mailbox),
                          'vmail.pending': s.MailNotices(self._notices)}
                disposition = Disposition('done', values[a.kind])
            elif isinstance(a, s.VMailRead):
                row = se.mailbox.read(a.number)
                header = s.MailHeader(a.number, row['id'], row['direction'], row['from'], row['to'], row['subject'], row['utc'])
                disposition = Disposition('done', s.MailBody(header, row['body']))
            elif isinstance(a, s.ChatHelp):
                disposition = Disposition('done', s.HelpText(a.topic, '\n'.join(se.COMMANDS)))
            else:
                self._apply(a)
                disposition = (Disposition('running', stage='queued') if isinstance(a, (s.BeaconOnce, s.BroadcastSend))
                               else Disposition('done', s.Applied(0)))
            if self._error is not None:
                raise self._error
            if request.id in self._results:
                disposition = self._results[request.id]
            if disposition.state in TERMINAL:
                self._finish(request.id, disposition.state, disposition.result,
                             code=disposition.reason_code, detail=disposition.detail)
            return disposition
        except (ValueError, v.HostError, OSError, TimeoutError) as error:
            self._finish(request.id, 'unknown' if request.id in self._sent else 'failed',
                         code='execution-error', detail=str(error))
            return self._results[request.id]
        finally:
            self._active = None
            self._flush()

    def _apply(self, a):
        se, c = self.session, self.session.client
        if isinstance(a, s.BufferClear):
            c.clean_buffer()
        elif isinstance(a, s.FollowSet):
            se.follow(a.target)
        elif isinstance(a, s.BandSet):
            se.change_band(a.band_m)
        elif isinstance(a, s.BandReady):
            se.band_ready(a.ready)
        elif isinstance(a, s.BroadcastSelect):
            se.broadcast.command('cf' if a.slot is None else f'slot {a.slot}')
        elif isinstance(a, s.BroadcastOff):
            se.broadcast.command('off')
        elif isinstance(a, s.BroadcastSend):
            se.broadcast.command(a.text)
        elif isinstance(a, s.BroadcastClear):
            se.broadcast.command('clear')
        elif isinstance(a, s.BroadcastConfirm):
            se.broadcast.command('yes' if a.proceed else 'no')
        elif isinstance(a, s.BeaconConfigure):
            se.set_beacon(a.minutes)
        elif isinstance(a, s.BeaconOnce):
            self._beacon_id = self._active
            se.beacon_now(force=a.force)
        elif isinstance(a, s.BeaconHop):
            se.set_hopping((['auto'] if a.auto else []) + [str(b) for b in a.bands] + ['every', str(a.minutes)])
        elif isinstance(a, s.AwaySet):
            se.set_away(a.text)
        elif isinstance(a, s.AutoSet):
            se.set_auto(a.on)
        elif isinstance(a, s.AwayBack):
            se.set_back()
        elif isinstance(a, s.VerboseSet):
            se.verbose = a.on
        elif isinstance(a, s.PSKReporterOff):
            if se.reporter:
                se.reporter.stop()
        elif isinstance(a, s.VMailDismiss):
            se.mail_collection.command(['dismiss', a.peer])
            self._refresh_mail()
        else:
            raise ValueError('unsupported typed action')

    def poll(self, timeout_s):
        self._assert_owner()
        se, c = self.session, self.session.client
        self._observe_line()
        se.tick(timeout_s)
        self._read_delivery()
        self._observe_line()
        for rid, deadline in tuple(self._deadlines.items()):
            if c.clock() >= deadline:
                self._finish(rid, 'unknown', code='response-timeout')
        rid = self._family_id('term')
        if rid and se.bbs.requested_term is None:
            a = self._requests[rid].action
            if se.bbs.remote_view and se.bbs.remote_dimensions == (a.width, a.height) and se.bbs.remote_ansi == (a.mode == 'ansi16'):
                self._finish(rid, 'done', s.TermResult(a.mode, a.width, a.height))
        rid = self._family_id('file')
        if rid:
            if se.files.failure:
                self._finish(rid, 'failed', s.FileResult('failed', se.files.requested, None, se.files.received_bytes),
                             detail=se.files.failure)
            elif se.files.offer is not None and isinstance(self._requests[rid].action, s.BBSRequest):
                self._finish(rid, 'done', s.FileResult('offer-recorded', se.files.requested,
                             str(se.files.offer_path), se.files.received_bytes))
        if self._draft and (self._draft.generation != c.link_generation or self._draft.peer != se.peer):
            self._retire_draft()
        self._flush()

    def legacy_line(self, line):
        self._assert_owner()
        before = self.session.compose, self.session.email.draft
        legacy, self._legacy = self._legacy, True
        try:
            running = self.session.command(line)
        finally:
            self._legacy = legacy
        after = self.session.compose, self.session.email.draft
        if any(a is not b for a, b in zip(before, after)) or not line.startswith('/'):
            self._draft = None
        self._sync_draft()
        self._flush()
        return running

    def quit(self, *, unwinding=None):
        self._assert_owner()
        if self._quit:
            return
        self._quit = True
        try:
            self.session.quit(unwinding=unwinding)
            self._finish_exit()
        finally:
            self._flush()

    def _finish_exit(self):
        line = self._observe_line()
        rid = self._active_operation('chat.exit')
        self._finish(rid, 'done' if line is False else 'failed',
                     s.LinkClosed('low' if line is False else 'high' if line else 'unknown'))
        if rid is not None and line is not False:
            raise RuntimeError('shutdown lacks low-line proof')


class OwnedSessionAdapter(SessionAdapter):
    """Inject a local runtime's already-ordered ExitStack into Session ownership.

    The factory registers its bounded finalizers in the same order as the line
    launcher. Session quits on its controller owner before the stack closes;
    this adapter never constructs audio, a Host, a rig or a terminal itself.
    """

    def __init__(self, session, *, cleanup, failure=lambda: None, **kwargs):
        super().__init__(session, **kwargs)
        self._owned_cleanup = cleanup
        self._owned_cleanup_called = False
        self._owned_cleanup_error = None
        self._runtime_failure = None
        self._external_failure = failure

    def poll(self, timeout_s):
        try:
            return super().poll(timeout_s)
        except BaseException as error:
            self._runtime_failure = error
            raise

    def quit(self, *, unwinding=None):
        self._assert_owner()
        if self._owned_cleanup_called:
            if self._owned_cleanup_error is not None:
                raise self._owned_cleanup_error
            return
        self._owned_cleanup_called = True
        failure = None
        try:
            super().quit(unwinding=unwinding)
        except BaseException as error:
            failure = error
        try:
            incoming = self._runtime_failure or failure or self._external_failure()
            self._owned_cleanup.__exit__(type(incoming) if incoming is not None else None,
                                         incoming, incoming.__traceback__ if incoming is not None else None)
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.__cause__ = error
        self._flush()
        try:
            if self._observe_line() is not False:
                raise RuntimeError('local chat teardown lacks low-line proof')
        except BaseException as error:
            if failure is None:
                failure = error
        self._owned_cleanup_error = failure
        if failure is not None:
            raise failure


def received_retune_adapter(runtime, client, tuner, unkeyed, *, completed=None, quitting=lambda: False):
    """Bridge cooperative application work to the sole modem owner's boundary."""
    def retune_steps(target, generation):
        modem = runtime.modem
        if modem is None:
            raise v.HostError('Receiver retune has no modem owner')
        valid = lambda: (client.link_generation == generation
                         and client.connected and not client.closed and not quitting())
        try:
            request = runtime.request_retune(
                lambda: tuner(target), modem=modem, station=modem.station,
                receive_cursor=modem.received_cursor, unkeyed=unkeyed, valid=valid)
        except (ValueError, queue.Full) as error:
            raise v.HostError(f'Receiver retune cannot be queued: {error}') from error
        try:
            while not request.done:
                if not valid():
                    raise v.HostError('Link changed during receiver retune')
                yield .05
            if request.error is not None:
                raise v.HostError(f'Receiver retune failed: {request.error}') from request.error
            if request.cancelled:
                raise v.HostError('Receiver retune cancelled')
        finally:
            request.cancel(timeout=.1 if quitting() else -1)
            if request.done and not request.cancelled and request.error is None:
                if completed is not None:
                    completed(target)
    return retune_steps
