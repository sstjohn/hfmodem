# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""DEFLATE (RFC 1951) payload records -- sabir's only compression, ever.

§97.113(a)(4) forbids obscuring meaning; §97.309(a)(4) allows any publicly
documented technique. DEFLATE is exactly that, so it is the one codec this
modem will apply -- off by default, and explicitly signaled per record so a
receiver never has to guess.

The host data stream crosses the link as self-delimiting records:
``[flags:1][length:3][body]``, flag 0 = raw, 1 = raw-DEFLATE body. A record
is only sent compressed when that actually made it smaller, so compression
can never cost on-air bytes beyond the 4-byte header every record carries.

Any other flag value fails loud (NEGOTIATION.md §5.1): the flag names the
body's encoding, so a record under an unknown flag can be delimited -- the
length field owes nothing to the flag -- but never delivered.
"""

from __future__ import annotations

import zlib
from hashlib import sha256
from hmac import compare_digest

RAW, DEFLATE = 0, 1
HEADER_BYTES = 4
HASHED = 0x10
MAX_RECORD = 8 * 1024 * 1024


class UnknownRecordFlag(ValueError):
    """A record flag this build does not know. The body is under an encoding
    we cannot invert, so delivering it would hand the host garbage; the
    session drops instead (SPEC.md §6.3.1)."""


def pack(blob: bytes, compress: bool = False, *, integrity: bool = True) -> bytes:
    if not integrity:
        raise ValueError("record integrity is mandatory")
    if len(blob) > MAX_RECORD:
        raise ValueError("record exceeds limit")
    body, flag = blob, RAW
    if compress:
        z = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        d = z.compress(blob) + z.flush()
        if len(d) < len(blob):
            body, flag = d, DEFLATE
    header = bytes([flag | HASHED]) + (len(body) + 32).to_bytes(3, "big")
    return header + body + sha256(header + body).digest()



class Unpacker:
    """Feed the delivered link stream in arbitrary chunks; yields payloads."""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, chunk: bytes):
        self._buf += chunk
        while len(self._buf) >= HEADER_BYTES:
            flag = self._buf[0]
            if flag not in (HASHED | RAW, HASHED | DEFLATE):
                raise UnknownRecordFlag(f"record flag {flag:#04x}")
            n = int.from_bytes(self._buf[1:4], "big")
            if n > MAX_RECORD + 32:
                raise ValueError("record exceeds receiver limit")
            if len(self._buf) < HEADER_BYTES + n:
                return
            body = bytes(self._buf[HEADER_BYTES : HEADER_BYTES + n])
            if flag & HASHED:
                if n < 32 or not compare_digest(body[-32:], sha256(bytes(self._buf[:4]) + body[:-32]).digest()):
                    raise ValueError("record integrity check failed")
                body = body[:-32]
            del self._buf[: HEADER_BYTES + n]
            if flag & DEFLATE:
                z = zlib.decompressobj(-zlib.MAX_WBITS)
                body = z.decompress(body, MAX_RECORD + 1)
                if len(body) > MAX_RECORD or not z.eof or z.unused_data:
                    raise ValueError("invalid or oversized DEFLATE record")
            yield body
