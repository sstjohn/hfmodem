# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 transparent source stream and ownership of bytes lent by the ARQ."""

from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy

from . import compress, spec
from .p4credit import SourceReceipt, SourceResult
from .p4observation import Kind


class P4StreamFailure(ValueError):
    """The application stream can no longer be reconstructed on this link."""


class P4LateFrame(Exception):
    pass


def observed_extension(frame):
    status = frame.status
    return (status.extended and not status.source_mode
            and frame.data.startswith(b"\x01\x01\x01")
            and (status.protocol in (0, 3)
                 # WS8EOC 1027: repeated CRC-valid extended, empty changeover.
                 # This admits the recorded control, not unknown source modes
                 # or an unobserved extended changeover carrying user bytes.
                 or status.protocol == 2 and not spec.field_payload(frame.data[3:])))


@dataclass
class _Credit:
    prefix: bytes
    provenance: str
    state: str = 'candidate'
    cursor: int = 0
    held: bytes = b''
    expected_counter: int | None = None

    @property
    def live(self):
        return self.state in ('candidate', 'armed', 'matching')


class P4Host:
    def __init__(self, host=None, *, terminal=True):
        self.host = host
        self.terminal = terminal
        self.supervisor = compress.Supervisor()
        self.last = None
        self.last_credit = 0
        self.result = SourceResult()
        self._receipts = {}
        self._active = None
        self._changeover_receipt = None
        self._legacy_serial = 0
        self.failure = None
        self._admit = None
        self._delivery_journal = None
        self._transaction = False
        self._effect_cursor = 0
        self._source_kind = None
        self._source_counter = None
        self._source_width = 0
        self._covered = b''
        self._resegmented = False

    def candidate(self, receipt, prefix, provenance):
        if receipt in self._receipts:
            return
        if not 1 <= len(prefix) <= 31 or provenance not in ('opening', 'changeover'):
            raise ValueError('invalid BODY credit candidate')
        if self.failure:
            raise P4StreamFailure(self.failure)
        if self._active is not None:
            raise ValueError('cancel the previous BODY receipt before registering another')
        self._receipts[receipt] = _Credit(bytes(prefix), provenance)
        self._active = receipt
        if provenance == 'changeover':
            self._changeover_receipt = receipt

    def arm(self, receipt):
        credit = self._receipts.get(receipt)
        if credit is None:
            return 'ineligible'
        if not credit.live:
            return 'spent'
        if credit.state == 'candidate':
            credit.state = 'armed'
        return 'armed'

    def _result(self, receipt, before=0, *, delivery=b'', duplicate=False,
                committed=0, released=0, reason=None, replayed=0, new=0,
                resegmented=False):
        credit = self._receipts.get(receipt)
        held = len(credit.held) if credit else 0
        self.last_credit = committed
        self.result = SourceResult(
            delivery=delivery, duplicate=duplicate, held_delta=held-before,
            held_total=held, committed_credit=committed, released_source=released,
            state=credit.state if credit else 'idle', cursor=credit.cursor if credit else 0,
            expected_counter=credit.expected_counter if credit else None,
            receipt=receipt, reason=reason, incomplete=self.supervisor._state != compress._TEXT,
            replayed_source_bytes=replayed, new_source_bytes=new, resegmented=resegmented)
        return self.result

    @contextmanager
    def admission(self, admit):
        previous = self._admit
        if admit is not None:
            self._admit = (lambda: previous() and admit()) if previous else admit
        try:
            yield
        finally:
            self._admit = previous

    def _guarded(self, token, call, admit=None):
        """Retry source parsing without repeating completed external effects.

        A callback may succeed before the next callback exhausts admission.
        Parsing state can be restored, but that successful delivery cannot be
        undone. A bounded journal retains its exact callback identity until
        this same operation completes; another operation cannot bypass it.
        """
        admit = admit or self._admit
        journal = self._delivery_journal
        if self._transaction:
            with self.admission(admit):
                return call()
        if journal is not None:
            if journal['token'] != token:
                raise P4LateFrame
            admit = admit or journal['admit']
        if admit is None:
            return call()
        if not admit():
            if journal is None and token[0] != 'accept':
                self._delivery_journal = dict(token=token, done=[], admit=admit)
            raise P4LateFrame
        fields = ('last', 'last_credit', 'result', 'supervisor', '_receipts', '_active',
                  '_changeover_receipt', '_legacy_serial', 'failure', '_source_kind',
                  '_source_counter', '_source_width', '_covered', '_resegmented')
        saved = {name: deepcopy(getattr(self, name)) for name in fields}
        if journal is None:
            journal = dict(token=token, done=[], admit=admit)
            self._delivery_journal = journal
        self._effect_cursor = 0
        self._transaction = True
        try:
            with self.admission(admit):
                result = call()
        except P4LateFrame:
            for name, value in saved.items():
                setattr(self, name, value)
            # A refused release must retain admission across a later timeout or
            # shutdown, even when no callback has completed yet.
            if not journal['done'] and token[0] == 'accept' and not journal.get('release'):
                self._delivery_journal = None
            raise
        except Exception:
            # Ordinary stream failures retain their existing committed state.
            self._delivery_journal = None
            raise
        else:
            self._delivery_journal = None
            return result
        finally:
            self._transaction = False

    def _callback(self, name, *args):
        journal = self._delivery_journal if self._transaction else None
        identity = (name, args)
        if journal is not None and self._effect_cursor < len(journal['done']):
            if journal['done'][self._effect_cursor] != identity:
                raise P4StreamFailure('inconsistent retry of partially delivered P4 source')
            self._effect_cursor += 1
            return
        if self._admit is not None and not self._admit():
            raise P4LateFrame
        getattr(self.host, name)(*args)
        if journal is not None:
            journal['done'].append(identity)
            self._effect_cursor += 1

    def _feed(self, field):
        payload = self.supervisor.feed(field)
        if self._admit is not None and not self._admit():
            raise P4LateFrame
        if self.host is not None:
            guarded = self._admit is not None and hasattr(self.host, '_p4_sf8_admit')
            if guarded:
                previous = self.host._p4_sf8_admit
                self.host._p4_sf8_admit = self._admit
            try:
                if payload:
                    self._callback('deliver', payload)
                for block in self.supervisor.blocks:
                    self._callback('supervisor', *block)
                self.supervisor.blocks.clear()
            finally:
                if guarded:
                    self.host._p4_sf8_admit = previous
        return payload

    def _retire(self, receipt, state):
        credit = self._receipts[receipt]
        credit.state = state
        if self._active == receipt:
            self._active = None
        if self._changeover_receipt == receipt:
            self._changeover_receipt = None

    def _cancel(self, receipt):
        credit = self._receipts.get(receipt)
        if credit is None or not credit.live:
            return b'', 0
        held, credit.held = credit.held, b''
        if held and self._transaction and self._delivery_journal is not None:
            # An accept may cancel held source internally before its first
            # callback. Rollback must retain that release's admission too.
            self._delivery_journal['release'] = True
        self._retire(receipt, 'cancelled')
        return self._feed(held) if held else b'', len(held)

    def cancel(self, receipt, reason, *, admit=None):
        return self._guarded(("cancel", receipt),
                             lambda: self._cancel_result(receipt, reason), admit)

    def _cancel_result(self, receipt, reason):
        credit = self._receipts.get(receipt)
        before = len(credit.held) if credit else 0
        self._result(receipt, before)
        payload, released = self._cancel(receipt)
        return self._result(receipt, before, delivery=payload, released=released, reason=reason)

    def end_epoch(self, epoch, reason, *, terminal=False, admit=None):
        return self._guarded(("end-epoch", epoch),
            lambda: self._end_epoch(epoch, reason, terminal=terminal), admit)

    def _end_epoch(self, epoch, reason, *, terminal=False):
        receipt = self._active
        if receipt is not None and receipt.epoch != epoch:
            raise ValueError('cannot end a different active receive epoch')
        result = self.cancel(receipt, reason)
        self.last = None
        self._reset_source()
        if terminal and result.incomplete and self.failure is None:
            self.failure = reason + '; incomplete supervisor source'
        return result

    def _reset_source(self):
        self._source_kind = self._source_counter = None
        self._source_width = 0
        self._covered = b''
        self._resegmented = False

    def _counted_source(self, kind, counter, field):
        covered = self._covered
        same = counter == self._source_counter
        advance = (self._source_counter is not None
                   and counter == (self._source_counter+1) % 4)
        unresolved = len(covered) > self._source_width
        take = min(len(field), len(covered))
        # A format change can restart the source at a different counter after
        # intervening speed controls. Require the entire retained prefix, not
        # a suffix/text overlap, before crediting a counter-gap replay.
        restart = (kind != self._source_kind and self._source_width > 0
                   and (advance and len(field) >= self._source_width
                        and field[:take] == covered[:take]
                        or not advance and len(field) >= len(covered)
                        and field.startswith(covered)))
        if advance and not restart:
            covered = covered[self._source_width:]
        take = min(len(field), len(covered))
        compatible = field[:take] == covered[:take]
        reason = None
        if (same or advance or restart) and compatible:
            novel = field[take:]
            covered += novel
            self._resegmented = same or unresolved or restart
            continuation = same or unresolved or restart
        else:
            if unresolved:
                self.failure = ('ambiguous P4 source replay: overlap mismatch'
                                if same or advance else 'ambiguous P4 source replay: counter gap')
                raise P4StreamFailure(self.failure)
            if self._source_counter is not None:
                reason = 'unclassified same-counter source' if same else 'unclassified counter gap'
            novel, covered, take = field, field, 0
            self._resegmented = False
            continuation = False
        self._source_kind = kind
        self._source_counter = counter
        self._source_width = len(field)
        self._covered = covered
        return novel, take, continuation, reason

    def accept(self, frame, *, opening_prefix=None, changeover=False, counted=False,
               receipt_context=None, admit=None):
        return self._guarded(('accept', frame.status.byte, frame.data),
            lambda: self._accept(frame, opening_prefix=opening_prefix, changeover=changeover,
                                 counted=counted, receipt_context=receipt_context), admit)

    def _accept(self, frame, *, opening_prefix=None, changeover=False, counted=False,
                receipt_context=None):
        context_receipt = receipt_context.receipt if receipt_context else None
        receipt = self._active if self._active is not None else context_receipt
        credit = self._receipts.get(receipt)
        before = len(credit.held) if credit else 0
        self._result(receipt, before)
        if self.failure:
            raise P4StreamFailure(self.failure)
        field = frame.data
        status = frame.status
        observed = observed_extension(frame)
        if status.source_mode or status.extended and not observed:
            prefix = field[:2 + field[1]] if len(field) >= 2 else field
            reason = (f"unsupported P4 source status 0x{status.byte:02x}"
                      if status.source_mode else
                      f"unsupported P4 extended status 0x{status.byte:02x} "
                      f"prefix {prefix.hex()}")
            payload, released = self._cancel(receipt)
            self._result(receipt, before, delivery=payload, released=released, reason=reason)
            if self.terminal:
                self.failure = reason
            raise P4StreamFailure(reason)
        identity = (status.byte, frame.data)
        kind = receipt_context.kind if receipt_context else Kind.NORMAL if counted else None
        # Chirps carry the same counted source, even when a long field is
        # retransmitted as several 22-byte fields. Opening BODY credit has its
        # own receipt/counter rules below and must resolve before this history.
        counted_source = (kind in (Kind.NORMAL, Kind.LONG)
                          or kind == Kind.CHIRP and not (credit and credit.live))
        if not counted_source:
            self._reset_source()
        if identity == self.last:
            if counted_source:
                self._source_kind = kind
            self._result(receipt, before, delivery=None, duplicate=True,
                         replayed=len(spec.field_payload(field[3:] if observed else field)),
                         resegmented=self._resegmented)
            return None
        if self._admit is not None and not self._admit():
            raise P4LateFrame
        self.last = identity
        field = spec.field_payload(field[3:] if observed else field)
        replayed, continuation, source_reason = 0, False, None
        if counted_source:
            field, replayed, continuation, source_reason = self._counted_source(kind, status.counter, field)
        new = len(field)
        payload, released, committed = b'', 0, 0
        interrupted, reason = False, None
        if credit is not None and credit.live and not continuation:
            matching = credit.state == 'matching'
            chirp = (kind == Kind.CHIRP and context_receipt == receipt
                     and status.protocol == 3)
            if chirp and (matching or credit.state == 'armed' and status.byte == 0x61):
                expected = credit.expected_counter if matching else 1
                remaining = credit.prefix[credit.cursor:]
                take = min(len(field), len(remaining))
                if status.counter == expected and field[:take] == remaining[:take]:
                    credit.state = 'matching'
                    self._changeover_receipt = None
                    credit.expected_counter = (status.counter+1) % 4
                    credit.cursor += take
                    credit.held += field[:take]
                    if field and credit.cursor == len(credit.prefix):
                        committed = len(credit.prefix)
                        credit.held = b''
                        self._retire(receipt, 'committed')
                        payload = self._feed(field[take:])
                    self._result(receipt, before, delivery=payload, committed=committed)
                    return payload
                interrupted = True
                reason = 'counter gap' if status.counter != expected else 'prefix mismatch'
            elif matching or kind == Kind.CHIRP:
                interrupted = True
                reason = 'incompatible source' if matching else 'ineligible chirp'
            elif receipt_context is not None and context_receipt != receipt:
                interrupted = True
                reason = 'lost receipt'
            elif kind == Kind.BODY and identity != receipt.identity:
                interrupted = True
                reason = 'incompatible BODY'
            elif status.protocol == 1:
                interrupted = True
                reason = 'source close'
            if interrupted:
                payload, released = self._cancel(receipt)
            elif counted or kind in (Kind.NORMAL, Kind.LONG) or opening_prefix is not None:
                legacy_changeover = self._changeover_receipt == receipt and counted
                self._retire(receipt, 'spent')
                if (opening_prefix is not None and len(opening_prefix) == 31
                        and not status.extended and status.counter == 1
                        and status.protocol != 1 and len(frame.data) > 31
                        and field.startswith(credit.prefix) and credit.prefix == opening_prefix):
                    committed = len(credit.prefix)
                    field = field[committed:]
                elif legacy_changeover and field.startswith(credit.prefix):
                    committed = len(credit.prefix)
                    field = field[committed:]
        elif (not continuation and receipt_context is None and opening_prefix is not None
                and len(opening_prefix) == 31 and not status.extended
                and status.counter == 1 and status.protocol != 1
                and len(frame.data) > 31 and field.startswith(opening_prefix)):
            committed = 31
            field = field[31:]
        if changeover and field and self._active is not None:
            flushed, extra = self._cancel(self._active)
            payload += flushed
            released += extra
        payload += self._feed(field)
        if changeover and field:
            new_receipt = context_receipt
            if new_receipt is None:
                self._legacy_serial += 1
                new_receipt = SourceReceipt(-1, (Kind.BODY, self._legacy_serial), identity)
            self.candidate(new_receipt, field, 'changeover')
        self._result(receipt, before, delivery=payload, committed=committed,
                     released=released, reason=reason or source_reason,
                     replayed=replayed, new=new, resegmented=self._resegmented)
        if kind == Kind.BODY or kind == Kind.CHIRP and not counted_source:
            self._reset_source()
        return payload


class P4Outbox:
    def __init__(self, arq):
        self.arq = arq
        self.inflight = b""

    @property
    def queued(self):
        return bool(self.arq._outbuf)

    def transfer(self, n=31):
        if not self.inflight:
            self.inflight = self.arq.lend_bytes(n)
        return self.inflight

    def commit(self):
        if self.inflight:
            self.arq.settle_bytes(self.inflight)
            self.inflight = b""

    def requeue(self):
        if self.inflight:
            self.arq.return_bytes(self.inflight)
            self.inflight = b""
