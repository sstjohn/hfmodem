# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A focused CBOR codec (RFC 8949) for the host interface.

Only the subset HOST-API.md uses: unsigned and negative integers, byte and
text strings, arrays, integer-keyed maps, the booleans, null, and float64.
No tags, bignums, indefinite-length items, or half/single floats -- a full
library would carry machinery this interface never emits. Encoding is deterministic for this profile: shortest integer arguments,
float64, and map keys sorted by their encoded bytes.
"""

from __future__ import annotations

import struct
from typing import Any


def _head(major: int, arg: int) -> bytes:
    mt = major << 5
    if arg < 24:
        return bytes([mt | arg])
    if arg < 0x100:
        return bytes([mt | 24, arg])
    if arg < 0x10000:
        return bytes([mt | 25]) + arg.to_bytes(2, "big")
    if arg < 0x100000000:
        return bytes([mt | 26]) + arg.to_bytes(4, "big")
    return bytes([mt | 27]) + arg.to_bytes(8, "big")


def encode(obj: Any) -> bytes:
    if obj is None:
        return b"\xf6"
    if obj is True:
        return b"\xf5"
    if obj is False:
        return b"\xf4"
    if isinstance(obj, int):
        return _head(0, obj) if obj >= 0 else _head(1, -1 - obj)
    if isinstance(obj, float):
        return b"\xfb" + struct.pack(">d", obj)
    if isinstance(obj, (bytes, bytearray)):
        return _head(2, len(obj)) + bytes(obj)
    if isinstance(obj, str):
        u = obj.encode("utf-8")
        return _head(3, len(u)) + u
    if isinstance(obj, (list, tuple)):
        return _head(4, len(obj)) + b"".join(encode(x) for x in obj)
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda item: encode(item[0]))
        return _head(5, len(items)) + b"".join(
            encode(k) + encode(v) for k, v in items)
    raise TypeError(f"cannot CBOR-encode {type(obj).__name__}")


class _Reader:
    __slots__ = ("buf", "i")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.i = 0

    def _take(self, n: int) -> bytes:
        j = self.i + n
        if j > len(self.buf):
            raise ValueError("truncated CBOR")
        chunk = self.buf[self.i:j]
        self.i = j
        return chunk

    def _arg(self, info: int) -> int:
        if info < 24:
            return info
        if info == 24:
            return self._take(1)[0]
        if info == 25:
            return int.from_bytes(self._take(2), "big")
        if info == 26:
            return int.from_bytes(self._take(4), "big")
        if info == 27:
            return int.from_bytes(self._take(8), "big")
        raise ValueError(f"bad CBOR argument {info}")

    def item(self) -> Any:
        b = self._take(1)[0]
        major, info = b >> 5, b & 0x1F
        if major == 0:
            return self._arg(info)
        if major == 1:
            return -1 - self._arg(info)
        if major == 2:
            return self._take(self._arg(info))
        if major == 3:
            return self._take(self._arg(info)).decode("utf-8")
        if major == 4:
            return [self.item() for _ in range(self._arg(info))]
        if major == 5:
            out = {}
            for _ in range(self._arg(info)):
                key = self.item()
                if type(key) not in (int, str) or key in out:
                    raise ValueError("invalid or duplicate CBOR map key")
                out[key] = self.item()
            return out
        if major == 6:
            raise ValueError("CBOR tags are not supported")
        if major == 7:
            if info == 20:
                return False
            if info == 21:
                return True
            if info == 22:
                return None
            if info == 27:
                return struct.unpack(">d", self._take(8))[0]
            raise ValueError(f"unsupported CBOR simple/float {info}")
        raise ValueError(f"bad CBOR major type {major}")


def decode(buf: bytes) -> Any:
    r = _Reader(buf)
    obj = r.item()
    if r.i != len(buf):
        raise ValueError("trailing bytes after CBOR item")
    return obj
