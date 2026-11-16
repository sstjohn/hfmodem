# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Original payload credit alongside VARA's distinct wire-unit BUFFER.

Plain wire bytes settle individually. A FILES container credits its original
block only when the complete header/body is peer-confirmed. Reframing before
negotiation changes representations, never admission or delivery totals.
"""
from collections import deque
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class PayloadEvidence:
    epoch: str
    sequence: int
    accepted: int
    acknowledged: tuple[tuple[int, int], ...]
    discarded: tuple[tuple[int, int], ...]
    retired: bool
    complete: bool


@dataclass
class _Span:
    remaining: int
    original: bytes
    atomic: bool
    settled: int = 0
    start: int = 0


class PayloadLedger:
    def __init__(self, limit=65536):
        if type(limit) is not int or limit <= 0:
            raise ValueError('payload limit must be a positive byte count')
        self.limit = limit
        self.spans = deque()
        self.accepted = self.confirmed = self.discarded = 0
        self.wire_pending = self.wire_confirmed = 0
        self.last_unconfirmed = b''
        self.retired = False
        self.epoch = str(uuid4())
        self.sequence = 0
        self._acknowledged, self._discarded = [], []
        self._complete = True
        self._represented = 0

    def evidence(self):
        return PayloadEvidence(self.epoch, self.sequence, self.accepted,
                               tuple(self._acknowledged), tuple(self._discarded),
                               self.retired, self._complete)

    def _record(self, ranges, start, end):
        if start == end:
            return
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        elif len(ranges) < 4096:
            ranges.append((start, end))
        else:
            self._complete = False

    @property
    def pending(self):
        return self.accepted-self.confirmed-self.discarded

    @property
    def unconfirmed(self):
        return b''.join(span.original for span in self.spans)

    def queue(self, wire, original, *, atomic=False, accept=True):
        if accept and self.pending+len(original) > self.limit:
            raise BufferError('VARA original payload queue full')
        if not wire:
            if original:
                raise ValueError('missing original payload representation')
            return
        if not atomic and len(wire) != len(original):
            raise ValueError('plain payload representation has different length')
        start = self.accepted if accept else self._represented
        self.spans.append(_Span(len(wire), bytes(original), atomic, start=start))
        self._represented = start + len(original)
        self.wire_pending += len(wire)
        if accept:
            self.accepted += len(original)
            self.sequence += 1

    def settle(self, n):
        if not 0 <= n <= self.wire_pending:
            raise ValueError('settlement exceeds admitted VARA wire bytes')
        self.wire_pending -= n
        self.wire_confirmed += n
        out = bytearray()
        while n:
            span = self.spans[0]
            take = min(n, span.remaining)
            if not span.atomic:
                out += span.original[:take]
                self._record(self._acknowledged, span.start, span.start + take)
                span.start += take
                span.original = span.original[take:]
            span.remaining -= take
            span.settled += take
            n -= take
            if not span.remaining:
                self._record(self._acknowledged, span.start, span.start + len(span.original))
                out += self.spans.popleft().original
        self.confirmed += len(out)
        self.sequence += 1
        return bytes(out)

    def reframe(self):
        if self.wire_confirmed or self.confirmed or self.retired:
            raise ValueError('cannot reframe after settlement or retirement')
        self._represented = self.spans[0].start if self.spans else self.accepted
        self.spans.clear()
        self.wire_pending = 0

    def bind_prefix(self, wire, original):
        """Bind a TEXT over to still-unconfirmed original bytes, without credit.

        Unsent TEXT is provisionally plain so it can follow PHY rate changes.
        Once an over reaches the transport, its complete encoded representation
        replaces that prefix atomically. A NAK may bind a smaller re-encoded
        prefix; all remaining originals stay queued, with admission and prior
        confirmation unchanged. FILES continues to use its container mapping.
        """
        wire, original = bytes(wire), bytes(original)
        pending = self.unconfirmed
        if self.spans and self.spans[0].atomic and self.spans[0].settled:
            raise ValueError('cannot rebind a partially settled TEXT span')
        if not original:
            if wire:
                raise ValueError('wire bytes without an original TEXT prefix')
            return
        if not wire or not pending.startswith(original):
            raise ValueError('TEXT over does not represent the unconfirmed prefix')
        remainder = pending[len(original):]
        start = self.spans[0].start
        self.spans.clear()
        self.spans.append(_Span(len(wire), original, True, start=start))
        if remainder:
            self.spans.append(_Span(len(remainder), remainder, False, start=start + len(original)))
        self.wire_pending = len(wire) + len(remainder)

    def clear(self):
        """Discard unsent originals at the host's request; the contact goes on."""
        self.discarded += self.pending
        for span in self.spans:
            self._record(self._discarded, span.start, span.start + len(span.original))
        self.spans.clear()
        self.wire_pending = 0
        self.sequence += 1

    def retire(self):
        original = self.unconfirmed
        if original:
            self.last_unconfirmed = original
        self.clear()
        self.retired = True
        return original
