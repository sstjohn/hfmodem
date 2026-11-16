# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bounded, in-memory station transport retention and output budgets."""
from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import json
import threading
import time

from . import wire

KiB = 1024
MiB = 1024 * KiB


def _utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class EncodedEvent(dict):
    """An ordinary event mapping with its immutable wire representation."""

    def __init__(self, **values):
        super().__init__(values)
        self.encoded = (f"id: {self['seq']}\nevent: {self['type']}\n"
                        f"data: {json.dumps(self, separators=(',', ':'), ensure_ascii=False)}\n\n").encode('utf-8')


def encode_event(event):
    """Return the bytes already encoded at publication (or encode a legacy event)."""
    if isinstance(event, EncodedEvent):
        return event.encoded
    return EncodedEvent(**event).encoded


class Events:
    def __init__(self, *, capacity=10000, age=3600, clock=time.monotonic,
                 command_bytes=2 * MiB, other_bytes=6 * MiB,
                 batch_bytes=256 * KiB, batch_count=128, max_event_bytes=64 * KiB):
        if min(capacity, age, command_bytes, other_bytes, batch_bytes,
               batch_count, max_event_bytes) <= 0:
            raise ValueError('event limits must be positive')
        self.capacity, self.age, self.clock = capacity, age, clock
        self.batch_bytes, self.batch_count = batch_bytes, batch_count
        self.max_event_bytes = min(max_event_bytes, batch_bytes)
        self.condition = threading.Condition(threading.RLock())
        self.ring = deque()
        self.seq = 0
        self.byte_limits = {'command': command_bytes, 'other': other_bytes}
        self.byte_counts = {'command': 0, 'other': 0}
        self._partitions = {'command': OrderedDict(), 'other': OrderedDict()}

    @staticmethod
    def _partition(event):
        return 'command' if event['type'] == 'command' else 'other'

    @property
    def encoded_bytes(self):
        return sum(self.byte_counts.values())

    def publish(self, type, payload):
        if type not in wire.EVENT_TYPES:
            raise ValueError(f'unknown event type: {type}')
        with self.condition:
            # JSON round-trip detaches mutable caller-owned payloads. Encoding is
            # retained once; readers never serialize the ring again.
            event = EncodedEvent(seq=self.seq + 1, utc=_utc(), type=type,
                                 data=json.loads(json.dumps(payload)))
            partition = self._partition(event)
            size = len(event.encoded)
            event_limit = min(self.max_event_bytes, 8 * KiB) if type == 'chat.state' else self.max_event_bytes
            if size > min(event_limit, self.byte_limits[partition]):
                raise ValueError('event exceeds encoded transport limit; use a result reference')
            self.seq += 1
            entry = (self.clock(), event)
            self.ring.append(entry)
            self._partitions[partition][event['seq']] = entry
            self.byte_counts[partition] += size
            self._trim()
            while len(self.ring) > self.capacity:
                self._remove(self.ring[0][1])
            while self.byte_counts[partition] > self.byte_limits[partition]:
                _, oldest = next(iter(self._partitions[partition].values()))
                self._remove(oldest)
            self.condition.notify_all()
            return event

    def _remove(self, event):
        partition = self._partition(event)
        entry = self._partitions[partition].pop(event['seq'])
        self.ring.remove(entry)
        self.byte_counts[partition] -= len(event.encoded)

    def _trim(self):
        cutoff = self.clock() - self.age
        while self.ring and self.ring[0][0] <= cutoff:
            self._remove(self.ring[0][1])

    @staticmethod
    def _gap(after, following):
        return EncodedEvent(seq=following - 1, utc=_utc(), type='gap',
                            data={'from': after, 'to': following})

    def after(self, seq):
        """Return one bounded batch, inserting gaps for every missing interval.

        A gap's sequence is the final absent sequence; ``to`` is the following
        available sequence, preserving the existing station gap convention.
        """
        with self.condition:
            self._trim()
            batch, size = [], 0

            def append(event):
                nonlocal size
                if len(batch) >= self.batch_count or size + len(event.encoded) > self.batch_bytes:
                    return False
                batch.append(event)
                size += len(event.encoded)
                return True

            if seq > self.seq:
                oldest = self.ring[0][1]['seq'] if self.ring else self.seq + 1
                if not append(self._gap(seq, oldest)):
                    return batch
                seq = oldest - 1
            for _, event in self.ring:
                if event['seq'] <= seq:
                    continue
                if event['seq'] > seq + 1:
                    gap = self._gap(seq, event['seq'])
                    if not append(gap):
                        return batch
                    seq = gap['seq']
                if not append(event):
                    return batch
                seq = event['seq']
            if seq < self.seq:
                append(self._gap(seq, self.seq + 1))
            return batch


class TokenBucket:
    """Nonblocking byte budget. Rejected writes do not reserve future tokens."""

    def __init__(self, rate, burst, *, clock=time.monotonic):
        if rate <= 0 or burst <= 0:
            raise ValueError('rate and burst must be positive')
        self.rate, self.burst, self.clock = rate, burst, clock
        self._tokens, self._updated = float(burst), clock()
        self._lock = threading.Lock()

    @classmethod
    def sse(cls, **kwargs):
        return cls(256 * KiB, 256 * KiB, **kwargs)

    @classmethod
    def bulk(cls, **kwargs):
        return cls(4 * MiB, 4 * MiB, **kwargs)

    def _refill(self):
        now = self.clock()
        self._tokens = min(self.burst, self._tokens + max(0, now - self._updated) * self.rate)
        self._updated = now

    def consume(self, size):
        if size < 0:
            raise ValueError('size must be nonnegative')
        with self._lock:
            self._refill()
            if size > self._tokens:
                return False
            self._tokens -= size
            return True

    def refund(self, size):
        if size < 0:
            raise ValueError('size must be nonnegative')
        with self._lock:
            self._refill()
            self._tokens = min(self.burst, self._tokens + size)

    def retry_after(self, size):
        if size < 0:
            raise ValueError('size must be nonnegative')
        with self._lock:
            self._refill()
            return float('inf') if size > self.burst else max(0, (size - self._tokens) / self.rate)


class TerminalResults:
    """Independent bounded terminal result cache; cache misses are expired.

    No tombstone collection grows behind eviction. A lookup never retries an
    operation, and a missing result is explicitly unknown/expired.
    """

    def __init__(self, *, capacity=1000, byte_limit=2 * MiB):
        if capacity <= 0 or byte_limit <= 0:
            raise ValueError('result limits must be positive')
        self.capacity, self.byte_limit = capacity, byte_limit
        self.encoded_bytes = 0
        self._results = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, incarnation, client, id, result):
        key = (incarnation, client, id)
        # Include identity and JSON escaping in the retained byte accounting.
        encoded = json.dumps({'incarnation': incarnation, 'client': client,
                              'id': id, 'result': result}).encode('utf-8')
        with self._lock:
            if key in self._results:
                return
            if len(encoded) <= self.byte_limit:
                self._results[key] = encoded
                self.encoded_bytes += len(encoded)
            while (len(self._results) > self.capacity
                   or self.encoded_bytes > self.byte_limit):
                _, removed = self._results.popitem(last=False)
                self.encoded_bytes -= len(removed)

    def lookup(self, incarnation, client, id):
        with self._lock:
            encoded = self._results.get((incarnation, client, id))
        if encoded is None:
            return {'status': 'expired'}
        return {'status': 'found', 'result': json.loads(encoded)['result']}


class ResultPages:
    """Bounded immutable captures with stable, read-only JSON result cursors.

    Collection pages retain their ReadValue type and metadata, splitting only
    the largest collection or text field. Individual rows remain atomic.
    """

    def __init__(self, *, page_bytes=4 * MiB, capacity=16, byte_limit=32 * MiB,
                 age=3600, clock=time.monotonic):
        if min(page_bytes, capacity, byte_limit, age) <= 0:
            raise ValueError('page limits must be positive')
        self.page_bytes, self.capacity, self.byte_limit = page_bytes, capacity, byte_limit
        self.age, self.clock = age, clock
        self.encoded_bytes = 0
        self._captures = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

    def _trim(self):
        while self._captures:
            _, (at, pages) = next(iter(self._captures.items()))
            if (len(self._captures) <= self.capacity and self.encoded_bytes <= self.byte_limit
                    and at > self.clock() - self.age):
                return
            self._captures.popitem(last=False)
            self.encoded_bytes -= sum(map(len, pages))

    def capture(self, value, *, read_id=None):
        import uuid
        read_id = read_id or uuid.uuid4().hex
        if not isinstance(read_id, str) or not read_id:
            raise ValueError('expected read ID')
        value = json.loads(self._encode(value))
        candidates = []
        def visit(item, path):
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, (*path, key))
            elif isinstance(item, (list, str)) and item:
                candidates.append((len(self._encode(item)), path, item))
        visit(value, ())
        def envelope(part, index, more):
            return {'_type': 'ReadPage', 'read_id': read_id, 'cursor': str(index),
                    'next_cursor': str(index + 1) if more else None, 'value': part}
        direct = self._encode(envelope(value, 0, False))
        if len(direct) <= self.page_bytes:
            pages = [direct]
        else:
            if not candidates:
                raise ValueError('result has no pageable collection or text')
            _, path, collection = max(candidates, key=lambda item: item[0])
            # Reuse a detached metadata tree while searching, then retain only
            # encoded pages. No caller-owned object survives this method.
            def part(begin, end):
                replacement = collection[begin:end]
                if not path:
                    return replacement
                parent = value
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = replacement
                return value
            pages, offset = [], 0
            while offset < len(collection):
                index = len(pages)
                low, high = offset, len(collection)
                while low < high:
                    mid = (low + high + 1) // 2
                    body = self._encode(envelope(part(offset, mid), index, mid < len(collection)))
                    if len(body) <= self.page_bytes:
                        low = mid
                    else:
                        high = mid - 1
                if low == offset:
                    raise ValueError('result metadata or single item exceeds page budget')
                pages.append(self._encode(envelope(part(offset, low), index, low < len(collection))))
                offset = low
                if sum(map(len, pages)) > self.byte_limit:
                    raise ValueError('result capture exceeds retention budget')
        retained = sum(map(len, pages))
        if retained > self.byte_limit:
            raise ValueError('result capture exceeds retention budget')
        with self._lock:
            self._trim()
            if read_id in self._captures:
                raise ValueError('read ID already captured')
            self._captures[read_id] = (self.clock(), tuple(pages))
            self.encoded_bytes += retained
            self._trim()
            return json.loads(pages[0])

    def page(self, read_id, cursor=None):
        with self._lock:
            self._trim()
            capture = self._captures.get(read_id)
            if capture is None:
                raise ValueError('result capture expired')
            pages = capture[1]
            cursor = '0' if cursor is None else cursor
            if (not isinstance(cursor, str) or not cursor.isascii() or not cursor.isdigit()
                    or len(cursor) > 12 or str(int(cursor)) != cursor or int(cursor) >= len(pages)):
                raise ValueError('invalid result cursor')
            return json.loads(pages[int(cursor)])
