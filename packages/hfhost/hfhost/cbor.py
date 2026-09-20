# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A CBOR codec (RFC 8949) covering the subset a structured host interface uses.

Encoding is deterministic (§4.2.1): shortest-form argument encoding and map keys
sorted ascending. Two implementations therefore produce identical bytes for the
same message, which is what lets a transcript recorded at one site be compared
byte-for-byte against one recorded at the other.

Indefinite-length items, tags, and half/single floats decode but are never
emitted; there is no reason for a host interface to produce them.
"""

from __future__ import annotations

import struct
from typing import Any

__all__ = ["encode", "decode", "CborError"]


class CborError(ValueError):
    pass


def _head(major: int, arg: int) -> bytes:
    mt = major << 5
    if arg < 24:
        return bytes([mt | arg])
    for n, (limit, code) in enumerate(((1 << 8, 24), (1 << 16, 25),
                                       (1 << 32, 26), (1 << 64, 27))):
        if arg < limit:
            return bytes([mt | code]) + arg.to_bytes(1 << n, "big")
    raise CborError(f"integer too large to encode: {arg}")


def encode(obj: Any) -> bytes:
    if obj is None:
        return b"\xf6"
    if obj is True:
        return b"\xf5"
    if obj is False:
        return b"\xf4"
    if isinstance(obj, int):
        return _head(0, obj) if obj >= 0 else _head(1, -obj - 1)
    if isinstance(obj, float):
        return b"\xfb" + struct.pack(">d", obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        return _head(2, len(b)) + b
    if isinstance(obj, str):
        b = obj.encode("utf-8")
        return _head(3, len(b)) + b
    if isinstance(obj, (list, tuple)):
        return _head(4, len(obj)) + b"".join(encode(v) for v in obj)
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: encode(kv[0]))
        return _head(5, len(items)) + b"".join(encode(k) + encode(v)
                                               for k, v in items)
    raise CborError(f"cannot encode {type(obj).__name__}")


def _arg(buf: memoryview, i: int) -> tuple[int | None, int]:
    """Return (argument, next index). None means indefinite length."""
    minor = buf[i] & 0x1F
    i += 1
    if minor < 24:
        return minor, i
    if minor in (24, 25, 26, 27):
        n = 1 << (minor - 24)
        if i + n > len(buf):
            raise CborError("truncated argument")
        return int.from_bytes(buf[i:i + n], "big"), i + n
    if minor == 31:
        return None, i
    raise CborError(f"reserved additional-information value {minor}")


def _item(buf: memoryview, i: int) -> tuple[Any, int]:
    if i >= len(buf):
        raise CborError("truncated item")
    major, ai = buf[i] >> 5, buf[i] & 0x1F
    arg, i = _arg(buf, i)

    if major == 0:
        return arg, i
    if major == 1:
        return -arg - 1, i
    if major in (2, 3):
        if arg is None:
            out = bytearray()
            while True:
                if i >= len(buf):
                    raise CborError("unterminated indefinite string")
                if buf[i] == 0xFF:
                    i += 1
                    break
                chunk, i = _item(buf, i)
                out += chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            return (out.decode("utf-8") if major == 3 else bytes(out)), i
        if i + arg > len(buf):
            raise CborError("truncated string")
        raw = bytes(buf[i:i + arg])
        i += arg
        return (raw.decode("utf-8") if major == 3 else raw), i
    if major == 4:
        out = []
        if arg is None:
            while True:
                if i >= len(buf):
                    raise CborError("unterminated indefinite array")
                if buf[i] == 0xFF:
                    return out, i + 1
                v, i = _item(buf, i)
                out.append(v)
        for _ in range(arg):
            v, i = _item(buf, i)
            out.append(v)
        return out, i
    if major == 5:
        out = {}
        if arg is None:
            while True:
                if i >= len(buf):
                    raise CborError("unterminated indefinite map")
                if buf[i] == 0xFF:
                    return out, i + 1
                k, i = _item(buf, i)
                v, i = _item(buf, i)
                out[k] = v
        for _ in range(arg):
            k, i = _item(buf, i)
            v, i = _item(buf, i)
            out[k] = v
        return out, i
    if major == 6:                       # tag: transparent, value passes through
        v, i = _item(buf, i)
        return v, i

    if ai == 25:
        return _half(arg), i
    if ai == 26:
        return struct.unpack(">f", buf[i - 4:i])[0], i
    if ai == 27:
        return struct.unpack(">d", buf[i - 8:i])[0], i
    if arg == 20:
        return False, i
    if arg == 21:
        return True, i
    if arg in (22, 23):                  # null, undefined
        return None, i
    raise CborError(f"unsupported simple value {arg}")


def _half(bits: int) -> float:
    exp, frac = (bits >> 10) & 0x1F, bits & 0x3FF
    if exp == 0:
        val = frac * 2.0 ** -24
    elif exp == 31:
        val = float("inf") if frac == 0 else float("nan")
    else:
        val = (frac + 1024) * 2.0 ** (exp - 25)
    return -val if bits & 0x8000 else val


def decode(data: bytes) -> Any:
    """Decode one CBOR item. Trailing bytes are an error — a host frame carries
    exactly one message, and silent tolerance would hide a framing bug."""
    buf = memoryview(data)
    obj, i = _item(buf, 0)
    if i != len(buf):
        raise CborError(f"{len(buf) - i} trailing byte(s) after item")
    return obj
