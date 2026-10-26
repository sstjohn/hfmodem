# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Original-byte provenance for the PACTOR character stream.

The ARQ still schedules in wire bytes. Moving a field between its queue and
inflight packet does not move this ledger: only peer settlement or explicit
retirement does. A partially settled escape retains its original byte until
the complete representation has been confirmed.
"""
from collections import deque
from dataclasses import dataclass

from .compress import transparent, SB, RUN_PREFIX
from .spec import IDLE


@dataclass
class _Span:
    remaining: int
    original: bytes
    atomic: bool = True


class PayloadLedger:
    def __init__(self):
        self.spans = deque()
        self.accepted = self.confirmed = self.discarded = 0
        self.wire_pending = 0

    @property
    def pending(self):
        return self.accepted - self.confirmed - self.discarded

    @property
    def unconfirmed(self):
        return b"".join(span.original for span in self.spans)

    def queue(self, wire: bytes, original: bytes | None = None):
        if original is None:
            if wire:
                self.spans.append(_Span(len(wire), b""))
        else:
            if transparent(original) != wire:
                raise ValueError("host provenance does not match transparent stream")
            self.accepted += len(original)
            start = 0
            for index, byte in enumerate(original):
                if byte not in (SB, RUN_PREFIX, IDLE):
                    continue
                if start < index:
                    blob = original[start:index]
                    self.spans.append(_Span(len(blob), blob, False))
                self.spans.append(_Span(2, bytes((byte,))))
                start = index + 1
            if start < len(original):
                blob = original[start:]
                self.spans.append(_Span(len(blob), blob, False))
        self.wire_pending += len(wire)

    def settle(self, n: int) -> bytes:
        if not 0 <= n <= self.wire_pending:
            raise ValueError("settlement exceeds pending wire bytes")
        self.wire_pending -= n
        out = bytearray()
        while n:
            span = self.spans[0]
            take = min(n, span.remaining)
            if not span.atomic:
                out += span.original[:take]
                span.original = span.original[take:]
            span.remaining -= take
            n -= take
            if not span.remaining:
                out += self.spans.popleft().original
        self.confirmed += len(out)
        return bytes(out)

    def boundary_after(self, offset: int) -> int:
        """Do not purge half of a character already offered to the peer."""
        if not 0 <= offset <= self.wire_pending:
            raise ValueError("invalid retirement offset")
        end = 0
        for span in self.spans:
            if end >= offset:
                break
            if not span.atomic and end + span.remaining >= offset:
                return offset
            end += span.remaining
        return end

    def discard(self, start: int, stop: int):
        if not 0 <= start <= stop <= self.wire_pending:
            raise ValueError("invalid retirement range")
        kept, offset, count = deque(), 0, 0
        for span in self.spans:
            end = offset + span.remaining
            if start <= offset and end <= stop:
                count += len(span.original)
            else:
                if offset < stop and end > start:
                    if span.atomic:
                        raise ValueError("retirement splits a character")
                    prefix = max(0, start - offset)
                    suffix = max(0, end - stop)
                    if prefix:
                        kept.append(_Span(prefix, span.original[:prefix], False))
                    if suffix:
                        kept.append(_Span(suffix, span.original[-suffix:], False))
                    count += span.remaining - prefix - suffix
                else:
                    kept.append(span)
            offset = end
        self.spans = kept
        self.wire_pending -= stop - start
        self.discarded += count

    def retire(self):
        self.discard(0, self.wire_pending)
