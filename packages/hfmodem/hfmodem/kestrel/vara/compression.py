# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA 4.9 link compression, independently implemented from bench findings.

FILES is block-framed raw deflate. TEXT uses a partially known static code;
its encoder is available for generated tests, not enabled on live links.
No stock captures or binaries are required by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import zlib

MODES = ('OFF', 'TEXT', 'FILES')
TEXT_FLAG = 0x20
BLOCK_LIMIT = 32_500
DELIVERY_LIMIT = 50_000
MODE_BITS = {'OFF': 0, 'TEXT': 2, 'FILES': 4}


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f'unknown compression mode: {mode!r}')
    return mode


def caller_mode(mode: str, high_speed: bool) -> str:
    """TEXT advertisement remains disabled until all receive tables are known."""
    mode = validate_mode(mode)
    return 'FILES' if high_speed and mode == 'FILES' else 'OFF'


def setup_mode(marker: int) -> str:
    bits = marker & 6
    if bits == 6:
        raise ValueError('conflicting TEXT and FILES link-setup flags')
    return {0: 'OFF', 2: 'TEXT', 4: 'FILES'}[bits]


def encode_files(data: bytes) -> bytes:
    out = bytearray()
    for start in range(0, len(data), BLOCK_LIMIT):
        block = data[start:start + BLOCK_LIMIT]
        d = zlib.compressobj(wbits=-15)
        packed = d.compress(block) + d.flush()
        compressed = len(packed) < len(block)
        body = packed if compressed else block
        out += (len(body) | (0x8000 if compressed else 0)).to_bytes(2, 'big')
        out += body
    return bytes(out)


# Prose table, line header AD. Integer byte keys avoid text encoding ambiguity.
_BYTE_ROWS = (
    (b' ', '011'),
    (b'irae', '0001 0010 0100 1001'),
    (b'uclsnto', '00000 01011 10001 10100 11100 11110 11111'),
    (b'fb\r,@.gmhpd', '000010 001100 001101 001110 001111 010101 100000 101011 101100 110101 111011'),
    (b'v-kwy', '0000110 0000111 1000010 1010100 1110101'),
    (b'IRBZOJ256GD78MLEP3YTWQ4S1FU9NCKAH0VX',
     '10101010 10101011 10110101 10111000 10111001 10111010 10111101 10111110 10111111 '
     '11000000 11000001 11000010 11000011 11000100 11000101 11000110 11000111 11001000 '
     '11001001 11001010 11001011 11001100 11001101 11001110 11001111 11010000 11010001 '
     '11010010 11010011 11011000 11011001 11011010 11011011 11011101 11011110 11011111'),
    (b':*', '010100001 110111001'),
    (b'<=#$[\\]_!";>?()/\'jqxz',
     '1000011100 1000011101 1000011110 1000011111 1011010000 1011010001 1011011000 '
     '1011011001 1011011100 1011011101 1011101101 1011101110 1011101111 1011110000 '
     '1011110001 1011110011 1110100001 1110100010 1110100011 1110100100 1110100101'),
)
BYTE_CODES = {b: c for chars, row in _BYTE_ROWS
              for b, c in zip(chars, row.split(), strict=True)}
TOKENS = {b' VARA': '101101001', b' has ': '01010011'}
ESC = '1011101100'
LF = '10000'
PROSE = '10101101'
SYMBOLS = {code: bytes([b]) for b, code in BYTE_CODES.items()}
SYMBOLS.update({code: word for word, code in TOKENS.items()})
PREFIXES = {code[:n] for code in (*SYMBOLS, ESC) for n in range(1, len(code))}


@dataclass(frozen=True)
class Decoded:
    data: bytes
    error: str | None = None
    bit: int = 0


def decode_text(payload: bytes) -> Decoded:
    """Read one independent over. On failure return only its verified prefix."""
    bits = ''.join(f'{b:08b}' for b in payload)
    if not bits:
        return Decoded(b'', 'empty TEXT over')
    end = len(bits.rstrip(bits[-1]))
    if not 1 <= len(bits) - end <= 8:
        return Decoded(b'', 'invalid TEXT padding')
    out = bytearray()
    p = 0

    def read(n: int, what: str) -> str:
        nonlocal p
        if p + n > end:
            raise ValueError(f'unfinished {what}')
        s = bits[p:p + n]
        p += n
        return s

    def line_header() -> None:
        header = read(8, 'line header')
        if header != PROSE:
            raise ValueError(f'unsupported line header 0x{int(header, 2):02x}')

    def after_cr() -> None:
        nonlocal p
        if p + len(LF) <= end and bits.startswith(LF, p):
            out.append(10)
            p += len(LF)
        if p < end:
            line_header()

    try:
        header = read(3, 'over header')
        raw = header == '010'
        if header == '000':
            line_header()
        elif header not in ('001', '010'):
            raise ValueError(f'unsupported over header {header}')
        while p < end:
            if raw:
                b = int(read(8, 'raw byte'), 2)
                out.append(b)
                raw = b not in BYTE_CODES
                if b == 13:
                    after_cr()
                continue
            code = ''
            while True:
                code += read(1, 'symbol')
                if code == ESC:
                    raw = True
                    # A complete ESC may end an over; the next one opens 010.
                    break
                if code in SYMBOLS:
                    symbol = SYMBOLS[code]
                    out.extend(symbol)
                    if symbol == b'\r':
                        after_cr()
                    break
                if code not in PREFIXES:
                    raise ValueError(f'unknown code {code}')
    except ValueError as exc:
        return Decoded(bytes(out), str(exc), p)
    return Decoded(bytes(out))


def encode_text(data: bytes, *, first: bool = True, raw: bool = False) -> bytes:
    """Generated prose-table over, for tests; live TEXT transmission is gated.

    The caller splits input only between symbols, raw bytes or complete CRLF
    framing. ``raw`` describes an escape run continuing from a previous over.
    """
    if first and raw:
        raise ValueError('a first over starts coded')
    bits = ('000' + PROSE) if first else ('010' if raw else '001')
    i = 0
    while i < len(data):
        b = data[i]
        if raw:
            bits += f'{b:08b}'
            raw = b not in BYTE_CODES
            i += 1
        else:
            token = next((word for word in TOKENS if data.startswith(word, i)), None)
            if token is not None:
                bits += TOKENS[token]
                i += len(token)
                continue
            if b not in BYTE_CODES:
                bits += ESC
                raw = True
                continue
            bits += BYTE_CODES[b]
            i += 1
        if b == 13:
            if data[i:i + 1] == b'\n':
                bits += LF
                i += 1
            if i < len(data):
                bits += PROSE
    bits += ('1' if bits[-1] == '0' else '0') * (8 - len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, 'big')


class ReceiveCompression:
    """One direction, after ARQ duplicate suppression, for one session.

    A fault permanently latches this stream until a new instance is created.
    ``feed`` reports it once; later input cannot splice data across the gap.
    Offsets are wire-byte offsets (including FILES headers).
    """
    def __init__(self, mode: str = 'OFF'):
        self.mode = validate_mode(mode)
        self.buffer = bytearray()
        self.offset = 0
        self.damaged = False

    def _fail(self, data: bytes, message: str) -> tuple[bytes, str]:
        self.damaged = True
        self.buffer.clear()
        return data, f'WRONG COMPRESSED DATA offset={self.offset}: {message}'

    def feed(self, payload: bytes, field: int, last: bool) -> tuple[bytes, str | None]:
        if self.damaged:
            return b'', None
        if self.mode != 'FILES':
            if self.mode == 'TEXT' and field & TEXT_FLAG:
                decoded = decode_text(payload)
                if decoded.error:
                    return self._fail(decoded.data, f'{decoded.error} bit={decoded.bit}')
                data = decoded.data
            else:
                data = payload
            self.offset += len(payload)
            return data, None
        self.buffer.extend(payload)
        out = bytearray()
        while len(self.buffer) >= 2:
            header = int.from_bytes(self.buffer[:2], 'big')
            n = header & 0x7fff
            if not header & 0x8000 and n > BLOCK_LIMIT:
                return self._fail(bytes(out), 'stored block exceeds 32500 bytes')
            if len(self.buffer) < 2 + n:
                break
            body = bytes(self.buffer[2:2 + n])
            if header & 0x8000:
                try:
                    d = zlib.decompressobj(-15)
                    block = d.decompress(body, BLOCK_LIMIT + 1)
                except zlib.error:
                    return self._fail(bytes(out), 'invalid deflate stream')
                if (not d.eof or d.unused_data or d.unconsumed_tail
                        or len(block) > BLOCK_LIMIT):
                    return self._fail(bytes(out), 'incomplete, trailing or oversized deflate stream')
            else:
                block = body
            out.extend(block)
            del self.buffer[:2 + n]
            self.offset += 2 + n
        if last and self.buffer:
            return self._fail(bytes(out), 'delivery ends inside a block')
        return bytes(out), None
