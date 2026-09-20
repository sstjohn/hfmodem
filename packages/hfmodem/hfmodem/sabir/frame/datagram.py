# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Connectionless object records: versioned, self-identifying and bounded.

No acknowledgement or prior session is required. SHA-256 detects corruption;
it is not authentication. Optional XOR parity repairs one erased fragment.
Future outer codes and envelope versions use explicit identifiers, never an
implicit reinterpretation of existing bytes.
"""
from dataclasses import dataclass
from hashlib import sha256
from collections import OrderedDict
import struct
import time
import uuid

VERSION = 1
PARITY = 1
FIXED = struct.Struct(">BBHH16sIIQ32sBB")
MAX_OBJECT = 8 * 1024 * 1024
MAX_FRAGMENTS = 65535


def identity(value):
    raw = value.upper().encode("ascii")
    if not 1 <= len(raw) <= 63 or any(c < 33 or c > 126 for c in raw):
        raise ValueError("identity must contain 1..63 printable ASCII characters")
    return raw


@dataclass(frozen=True)
class Fragment:
    source: str
    destination: str
    message_id: bytes
    service: int
    index: int
    count: int
    size: int
    digest: bytes
    data: bytes
    parity: bool = False

    def pack(self):
        src, dst = identity(self.source), identity(self.destination)
        if (len(self.message_id) != 16 or len(self.digest) != 32
                or not 1 <= self.count <= MAX_FRAGMENTS
                or not 0 <= self.size <= MAX_OBJECT
                or self.count > max(1, self.size)
                or len(self.data) > 8192
                or (self.size > 0 and not self.data)
                or not 0 <= self.service <= 65535
                or (self.index != self.count if self.parity else
                    not 0 <= self.index < self.count)):
            raise ValueError("invalid fragment metadata")
        header_len = FIXED.size + len(src) + len(dst)
        return FIXED.pack(VERSION, PARITY if self.parity else 0, header_len,
                          self.service, self.message_id, self.index, self.count,
                          self.size, self.digest, len(src), len(dst)) + src + dst + self.data

    @classmethod
    def unpack(cls, raw):
        if len(raw) < FIXED.size:
            raise ValueError("truncated fragment")
        v, flags, hlen, service, mid, idx, count, size, digest, slen, dlen = FIXED.unpack_from(raw)
        if v != VERSION or flags & ~PARITY or hlen != FIXED.size + slen + dlen or hlen > len(raw):
            raise ValueError("unsupported or malformed fragment header")
        src = raw[FIXED.size:FIXED.size + slen].decode("ascii")
        dst = raw[FIXED.size + slen:hlen].decode("ascii")
        obj = cls(src, dst, mid, service, idx, count, size, digest, raw[hlen:], bool(flags))
        if obj.pack() != raw:
            raise ValueError("noncanonical fragment")
        return obj


def fragment_object(data, source, destination="*", *, service=0, fragment_bytes=512,
                    parity=True, message_id=None):
    data = bytes(data)
    if len(data) > MAX_OBJECT or not 1 <= fragment_bytes <= 8192:
        raise ValueError("object or fragment exceeds limit")
    mid = uuid.uuid4().bytes if message_id is None else bytes(message_id)
    chunks = [data[i:i + fragment_bytes] for i in range(0, len(data), fragment_bytes)] or [b""]
    digest = sha256(data).digest()
    records = [Fragment(source, destination, mid, service, i, len(chunks), len(data), digest, ch)
               for i, ch in enumerate(chunks)]
    if parity and len(chunks) > 1:
        repair = bytearray(fragment_bytes)
        for ch in chunks:
            for i, b in enumerate(ch):
                repair[i] ^= b
        records.append(Fragment(source, destination, mid, service, len(chunks), len(chunks),
                                len(data), digest, bytes(repair), True))
    for record in records:
        record.pack()  # fail before handing a partial object to the caller
    return records


class Reassembler:
    def __init__(self, *, max_bytes=MAX_OBJECT, max_objects=32, ttl=600, clock=time.monotonic):
        if max_bytes <= 0 or max_objects <= 0 or ttl <= 0:
            raise ValueError("receiver limits must be positive")
        self.max_bytes, self.max_objects, self.ttl, self.clock = max_bytes, max_objects, ttl, clock
        self.pending = OrderedDict()
        self.done = OrderedDict()

    def accept(self, record):
        f = Fragment.unpack(record)
        now = self.clock()
        for table in (self.pending, self.done):
            for key in list(table):
                if now - table[key][0] > self.ttl:
                    del table[key]
        key = (f.source, f.destination, f.message_id, f.service)
        if key in self.done:
            return None
        if f.size > self.max_bytes or len(f.data) > self.max_bytes:
            raise ValueError("object exceeds receiver memory budget")
        meta = (f.count, f.size, f.digest)
        if key not in self.pending:
            while len(self.pending) >= self.max_objects:
                self.pending.popitem(last=False)
            self.pending[key] = (now, meta, {})
        _, old_meta, pieces = self.pending[key]
        if old_meta != meta or (f.index in pieces and pieces[f.index] != f.data):
            raise ValueError("conflicting fragment identity")
        pieces[f.index] = f.data
        while sum(sum(len(p) + 128 for p in entry[2].values()) for entry in self.pending.values()) > self.max_bytes:
            self.pending.popitem(last=False)
        if key not in self.pending:
            return None
        missing = [i for i in range(f.count) if i not in pieces]
        if len(missing) == 1 and f.count in pieces:
            repair = bytearray(pieces[f.count])
            width = len(repair)
            if not width or not (f.count - 1) * width <= f.size <= f.count * width:
                raise ValueError("invalid parity geometry")
            for i, ch in pieces.items():
                if i == f.count:
                    continue
                want = width if i < f.count - 1 else f.size - (f.count - 1) * width
                if len(ch) != want:
                    raise ValueError("invalid fragment length")
                for j, b in enumerate(ch):
                    repair[j] ^= b
            idx = missing[0]
            pieces[idx] = bytes(repair[:width if idx < f.count - 1 else f.size - idx * width])
            missing = []
        if missing:
            return None
        data = b"".join(pieces[i] for i in range(f.count))
        del self.pending[key]
        if len(data) != f.size or sha256(data).digest() != f.digest:
            raise ValueError("object integrity check failed")
        self.done[key] = (now,)
        while len(self.done) > self.max_objects * 4:
            self.done.popitem(last=False)
        return f, data
