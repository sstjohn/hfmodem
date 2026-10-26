# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA 4.9 link compression, independently implemented from bench findings.

FILES is block-framed raw deflate. TEXT uses the 4.9.0 dictionaries and state
transitions, transmitting through the prose code with plain fallback.
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
    if mode == 'ON':
        mode = 'TEXT'  # Stock's legacy host command selects TEXT on the wire.
    if mode not in MODES:
        raise ValueError(f'unknown compression mode: {mode!r}')
    return mode


def caller_mode(mode: str, high_speed: bool) -> str:
    """Advertise the selected encoding when the link supports compression."""
    mode = validate_mode(mode)
    return mode if high_speed else 'OFF'


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
TOKENS = {b' VARA': '101101001', b' has ': '01010011', b'& ': '1110100000'}
ESC = '1011101100'
LF = '10000'
PROSE = '10101101'
SYMBOLS = {code: bytes([b]) for b, code in BYTE_CODES.items()}
SYMBOLS.update({code: word for word, code in TOKENS.items()})
PROSE_PHRASES = {
    '10000110': b'Welcome to the ', '01010010': b' RMS Trimode',
    '01010001': b' minutes remaining with ',
    '101101101': b' Connected to CMS\r', '101101111': b' RMS Relay',
    '111010011': b'forwarded to the recipient via HF radio',
    '110111000': b' through the Internet ', '010100000': b' >\r',
    '1011110010': b' //WL2K ',
}
SYMBOLS.update(PROSE_PHRASES)
PREFIXES = {code[:n] for code in (*SYMBOLS, ESC) for n in range(1, len(code))}

# FW line form, initially measured by synthetic stock-interface substitutions.
FW_PREFIX = '000'
COMMENT_PREFIX = '011'
PQ_PREFIX = '1011'
PR_PREFIX = '1100'
PM_PREFIX = '10100'
CMS_PREFIX = '1101'
FW_ESC = '01100101'
FW_SPACE = '0110011'
FW_CODES = dict(zip(b'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-\r', (
    '10101 10000 111001 00111 01011 111101 111000 100010 100101 '
    '110001 1101 101110 00000 11101 100011 011000 1011001 111111 '
    '010000 111110 1001001 110000 10100 001100 1011000 010001 '
    '001101 111100 00001 00011 10011 00101 00010 01101 101101 00100 '
    '0111 10010001').split(), strict=True))
FW_SYMBOLS = {code: bytes([b]) for b, code in FW_CODES.items()}
FW_SYMBOLS.update({'01001': b' DE ', '110011': b' (', '101111': b')\r', '110010': b' >\r'})
FW_SYMBOLS.update({'10010000': b'WL2K', '01100100': b'IS0GRB', '01010': b'\r'})
FW_PREFIXES = {code[:n] for code in (*FW_SYMBOLS, FW_ESC, FW_SPACE)
               for n in range(1, len(code))}

WORD_CODES = dict(zip(b'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \r', (
    '01011 01110 00100 01000 00000 00110 00101 01101 01100 '
    '01001 10001 00001 10010 00111 01010 01111 00010 10000 '
    '00011 110011 111111 110001 110110 110100 110000 110010 '
    '111110 111001 110101 111100 111000 111010 111011 111101 110111 100111 '
    '1010 1011').split(), strict=True))
WORD_ESC = '100110'
WORD_SYMBOLS = {code: bytes([b]) for b, code in WORD_CODES.items()}
WORD_PREFIXES = {code[:n] for code in (*WORD_SYMBOLS, WORD_ESC) for n in range(1, len(code))}

NUMBER_CODES = dict(zip(b'0123456789 \r', (
    '111110 00 010110 111111 01000 010111 01010 111100 111101 01001 10 011'
).split(), strict=True))
NUMBER_ESC = '1110'
NUMBER_SYMBOLS = {code: bytes([b]) for b, code in NUMBER_CODES.items()}
NUMBER_SYMBOLS['110'] = b' 0\r'
NUMBER_PREFIXES = {code[:n] for code in (*NUMBER_SYMBOLS, NUMBER_ESC)
                   for n in range(1, len(code))}

# Stock-interface SID digit/capability substitutions, extended with the
# remaining version, whitespace and table-transition forms.
SID_PREFIX = '1110'
SID_ESC = '1011001'                       # returns to the FW table
SID_COMMENT = '1000'                      # emits '; ' and selects FW
SID_CODES = dict(zip(b'0123456789.\n', (
    '1001100 011 1001101 1001111 1001010 101111 1011010 1001001 1011000 1001011 11 1010'
).split(), strict=True))
SID_SYMBOLS = {code: bytes([byte]) for byte, code in SID_CODES.items()}
SID_SYMBOLS.update({'000': b'-B2FWIHJM$]\r', '010': b'-B2FHM$]\r'})
SID_SYMBOLS.update({
    '0010': b';PQ: ', '0011': b';PR: ', '101110': b'-B2FIHJM$]\r',
    '1011011': b'-B2FWIHM$]\r', '1001110': b' ', '1001000': b'\r',
})
SID_PREFIXES = {code[:n] for code in (*SID_SYMBOLS, SID_ESC, SID_COMMENT)
                for n in range(1, len(code))}

# The remaining dictionary and transitions are protocol facts from the pinned
# distributed 4.9.0 binary. No vendor
# executable code or captured frames are embedded here.
SFI_ESC = '01000'
SFI_SYMBOLS = dict(zip((
    '1001 0111 00 101 1100 1000 0110 0101 11110 11111 01001 11100 '
    '11010 11011 11101').split(), (
    b' On 20', b'UTC}\r\n', b'0', b'-', b' ', b':', b'1', b'2', b'3',
    b'4', b'5', b'6', b'7', b'8', b'9'), strict=True))
SFI_PREFIXES = {code[:n] for code in (*SFI_SYMBOLS, SFI_ESC)
               for n in range(1, len(code))}

# A line prefix emits its literal and selects the following symbol table.
# Table zero continues reading prefixes, including complete lines and LF.
LINE_SYMBOLS = {
    '000': (b';FW: ', 7), '011': (b'; ', 7),
    '010': (b'[RMS Express-', 4), '1110': (b'[WL2K-', 4),
    '1111': (b'FF\r', 0), '1011': (b';PQ: ', 6),
    '1100': (b';PR: ', 6), '1101': (b'CMS via ', 7),
    '0010': (b'RMS Trimode ', 4), '0011': (b'{SFI = ', 3),
    '10100': (b';PM: ', 7), '10010': (b'FC EM ', 5), '10011': (b'F> ', 5),
    '101010': (b';WL2K-Radio/Internet_Network\r', 0),
    '100010': (b'[RMS Relay-', 4), '1010111': (b'[BPQ-', 4),
    '1000111': (b';FR:\r', 0), '1000110': (b'*** ', 7),
    '10101100': (b'de ', 7), PROSE: (b'', 1), LF: (b'\n', 0),
}
LINE_PREFIXES = {code[:n] for code in LINE_SYMBOLS for n in range(1, len(code))}


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
    table = 1

    def read(n: int, what: str) -> str:
        nonlocal p
        if p + n > end:
            raise ValueError(f'unfinished {what}')
        s = bits[p:p + n]
        p += n
        return s

    def optional_lf() -> None:
        nonlocal p
        if p + len(LF) <= end and bits.startswith(LF, p):
            out.append(10)
            p += len(LF)

    def line_header() -> None:
        nonlocal table
        table = 0
        while table == 0:
            code = ''
            while code not in LINE_SYMBOLS:
                code += read(1, 'line prefix')
                if code not in LINE_SYMBOLS and code not in LINE_PREFIXES:
                    raise ValueError(f'unknown line prefix {code}')
            literal, table = LINE_SYMBOLS[code]
            out.extend(literal)
            if p == end:
                return

    def after_cr() -> None:
        optional_lf()
        if p < end:
            line_header()

    try:
        header = read(3, 'over header')
        raw = header == '010'
        if header == '000':
            line_header()
        elif header in ('011', '100', '101', '110', '111'):
            table = int(header, 2)
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
            symbols, prefixes = {
                1: (SYMBOLS, PREFIXES), 3: (SFI_SYMBOLS, SFI_PREFIXES),
                4: (SID_SYMBOLS, SID_PREFIXES),
                5: (WORD_SYMBOLS, WORD_PREFIXES),
                6: (NUMBER_SYMBOLS, NUMBER_PREFIXES), 7: (FW_SYMBOLS, FW_PREFIXES),
            }[table]
            while True:
                code += read(1, 'symbol')
                if table == 4 and code in (SID_ESC, SID_COMMENT):
                    if code == SID_COMMENT:
                        out.extend(b'; ')
                    table = 7
                    break
                if ((table == 3 and code == SFI_ESC)
                        or (table == 7 and code == FW_ESC)
                        or (table == 5 and code == WORD_ESC)
                        or (table == 6 and code == NUMBER_ESC)):
                    table = 1
                    break
                if table == 7 and code == FW_SPACE:
                    out.append(32)
                    table = 5
                    break
                if table == 1 and code == ESC:
                    raw = True
                    # A complete ESC may end an over; the next one opens 010.
                    break
                if code in symbols:
                    symbol = symbols[code]
                    out.extend(symbol)
                    if table == 5 and symbol == b' ':
                        table = 6
                    elif table == 1 and code == '01010001':
                        table = 7
                    elif table == 4 and symbol in (b';PQ: ', b';PR: '):
                        table = 6
                    elif table == 4 and symbol == b' ':
                        table = 1
                    elif table == 3 and symbol == b'UTC}\r\n':
                        if p < end:
                            line_header()
                    # SID suffix tokens include CR but stay in their table;
                    # stock uses explicit FW/prose escapes for following text.
                    if symbol.endswith(b'\r') and (table != 4 or symbol == b'\r'):
                        after_cr()
                    break
                if code not in prefixes:
                    raise ValueError(f'unknown code {code}')
    except ValueError as exc:
        return Decoded(bytes(out), str(exc), p)
    return Decoded(bytes(out))


def encode_text(data: bytes, *, first: bool = True, raw: bool = False) -> bytes:
    """Encode one independent over using the measured prose table and escapes.

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
