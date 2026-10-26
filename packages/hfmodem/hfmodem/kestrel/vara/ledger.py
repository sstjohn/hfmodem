# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Original payload credit alongside VARA's distinct wire-unit BUFFER.

Plain wire bytes settle individually. A FILES container credits its original
block only when the complete header/body is peer-confirmed. Reframing before
negotiation changes representations, never admission or delivery totals.
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class _Span:
    remaining: int
    original: bytes
    atomic: bool
    settled: int = 0


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
        self.spans.append(_Span(len(wire), bytes(original), atomic))
        self.wire_pending += len(wire)
        if accept:
            self.accepted += len(original)

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
                span.original = span.original[take:]
            span.remaining -= take
            span.settled += take
            n -= take
            if not span.remaining:
                out += self.spans.popleft().original
        self.confirmed += len(out)
        return bytes(out)

    def reframe(self):
        if self.wire_confirmed or self.confirmed or self.retired:
            raise ValueError('cannot reframe after settlement or retirement')
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
        self.spans.clear()
        self.spans.append(_Span(len(wire), original, True))
        if remainder:
            self.spans.append(_Span(len(remainder), remainder, False))
        self.wire_pending = len(wire) + len(remainder)

    def clear(self):
        """Discard unsent originals at the host's request; the contact goes on."""
        self.discarded += self.pending
        self.spans.clear()
        self.wire_pending = 0

    def retire(self):
        original = self.unconfirmed
        if original:
            self.last_unconfirmed = original
        self.clear()
        self.retired = True
        return original
