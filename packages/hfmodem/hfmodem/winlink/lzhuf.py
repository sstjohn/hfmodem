# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The FBB compressed-forwarding codec: LZHUF over a 2048-byte window, CRC-wrapped.

This is the compression Winlink's B2 forwarding applies to every message body
[FBB doc, compressed forward; B2F spec]. The algorithm is Okumura/Yoshizaki
LZHUF — LZSS string matching under an adaptive Huffman code — with one
parameter the Winlink implementation changed and documents in its published
source: the sliding window is **2048** bytes, not the 4096 of the classic
program. Back-references are relative, so the narrower stream happens to read
under a wider decoder — the direction that bites is encode, where a
classic-window encoder emits positions a 2048-window gateway wraps to the
wrong bytes while every CRC passes (the CRC covers the compressed bytes, not
the output). The window size is pinned by test against real material.

Stream layout, first byte first  [FBB doc, compressed forward]:

  * CRC-16 over everything that follows, little-endian.  The polynomial is
    CCITT 0x1021, MSB-first, initial value 0 (CRC-16/XMODEM).
  * uncompressed size, 4 bytes little-endian.
  * the LZHUF bit stream: adaptive-Huffman symbols 0-255 (literals) and
    256-313 (match lengths 3-60), each match followed by its window position —
    upper bits through the fixed canonical table below, lower 6 bits raw.

The window starts as 0x20 bytes with writing at N-F, both coders update one
adaptive tree per symbol, and the tree halves its counts when the root reaches
0x8000: every detail here is observable in the output bytes, and the pair
encoder/decoder is anchored to an independently produced compressed message,
byte-exact in both directions (tests/winlink/test_lzhuf.py).
"""
from __future__ import annotations

N = 2048                    # sliding window (Winlink value; classic LZHUF used 4096)
F = 60                      # lookahead: longest match, and so the highest symbol
THRESHOLD = 2               # matches this short are cheaper as literals
NCHAR = 256 - THRESHOLD + F         # 314 symbols: 256 literals + 58 lengths
T = NCHAR * 2 - 1                   # tree size
ROOT = T - 1
MAX_FREQ = 0x8000

# Canonical code for the upper bits of a window position: 1 code of 3 bits,
# 3 of 4, 8 of 5, 12 of 6, 24 of 7 and 16 of 8, left-aligned in a byte.
P_LEN = (3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8)
P_CODE = (0x00, 0x20, 0x30, 0x40, 0x50, 0x58, 0x60, 0x68,
          0x70, 0x78, 0x80, 0x88, 0x90, 0x94, 0x98, 0x9C,
          0xA0, 0xA4, 0xA8, 0xAC, 0xB0, 0xB4, 0xB8, 0xBC,
          0xC0, 0xC2, 0xC4, 0xC6, 0xC8, 0xCA, 0xCC, 0xCE,
          0xD0, 0xD2, 0xD4, 0xD6, 0xD8, 0xDA, 0xDC, 0xDE,
          0xE0, 0xE2, 0xE4, 0xE6, 0xE8, 0xEA, 0xEC, 0xEE,
          0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
          0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF)

# The decode side of the same table, derived rather than transcribed: for each
# byte value, which code prefixes it and how long that code is.
_D_CODE = [0] * 256
_D_LEN = [0] * 256
for _v in range(64):
    for _b in range(P_CODE[_v], P_CODE[_v] + (1 << (8 - P_LEN[_v]))):
        _D_CODE[_b] = _v
        _D_LEN[_b] = P_LEN[_v]

_CRC_TABLE = []
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021 if _c & 0x8000 else _c << 1) & 0xFFFF
    _CRC_TABLE.append(_c)


def crc16(data: bytes) -> int:
    """CRC-16/XMODEM: poly 0x1021 MSB-first, initial 0. Check value of
    b'123456789' is 0x31C3."""
    crc = 0
    for b in data:
        crc = ((crc << 8) ^ _CRC_TABLE[(crc >> 8) ^ b]) & 0xFFFF
    return crc


class LzhufError(ValueError):
    """The stream is not a well-formed FBB compressed body."""


class _Tree:
    """The adaptive Huffman tree both coders keep in lock-step.

    Leaves start at count 1 so every symbol is encodable from the first bit;
    `update` bubbles a symbol up past siblings of lower count, and `reconst`
    halves everything when the root saturates — the decoder replays the same
    walk, so the trees never diverge.
    """

    def __init__(self):
        self.freq = [0] * (T + 1)
        self.son = [0] * T
        self.prnt = [0] * (T + NCHAR)
        for i in range(NCHAR):
            self.freq[i] = 1
            self.son[i] = i + T
            self.prnt[i + T] = i
        i, j = 0, NCHAR
        while j <= ROOT:
            self.freq[j] = self.freq[i] + self.freq[i + 1]
            self.son[j] = i
            self.prnt[i] = self.prnt[i + 1] = j
            i += 2
            j += 1
        self.freq[T] = 0xFFFF
        self.prnt[ROOT] = 0

    def _reconst(self):
        freq, son, prnt = self.freq, self.son, self.prnt
        j = 0
        for i in range(T):
            if son[i] >= T:
                freq[j] = (freq[i] + 1) >> 1
                son[j] = son[i]
                j += 1
        i, j = 0, NCHAR
        while j < T:
            f = freq[j] = (freq[i] + freq[i + 1]) & 0xFFFF
            k = j - 1
            while f < freq[k]:
                k -= 1
            k += 1
            for n in range(j, k, -1):
                freq[n] = freq[n - 1]
                son[n] = son[n - 1]
            freq[k] = f
            son[k] = i
            i += 2
            j += 1
        for i in range(T):
            k = son[i]
            prnt[k] = i
            if k < T:
                prnt[k + 1] = i

    def update(self, c: int):
        freq, son, prnt = self.freq, self.son, self.prnt
        if freq[ROOT] == MAX_FREQ:
            self._reconst()
        c = prnt[c + T]
        while True:
            freq[c] += 1
            k = freq[c]
            n = c + 1
            if k > freq[n]:
                while k > freq[n + 1]:
                    n += 1
                freq[c] = freq[n]
                freq[n] = k
                i = son[c]
                prnt[i] = n
                if i < T:
                    prnt[i + 1] = n
                j = son[n]
                son[n] = i
                prnt[j] = c
                if j < T:
                    prnt[j + 1] = c
                son[c] = j
                c = n
            c = prnt[c]
            if c == 0:
                return

    def encode_symbol(self, c: int) -> tuple[int, int]:
        """(nbits, code left-aligned in 16) for symbol c, then update."""
        code = nbits = 0
        k = self.prnt[c + T]
        while k != ROOT:
            code >>= 1
            if k & 1:
                code += 0x8000
            nbits += 1
            k = self.prnt[k]
        self.update(c)
        return nbits, code


class _BitReader:
    """MSB-first reader over the compressed bytes. Reads past the end yield
    zeros, exactly as the published decoder's byte fetch does — a truncated
    stream is caught by the CRC, which covers every byte actually present."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.buf = 0
        self.nbits = 0

    def _fill(self):
        while self.nbits <= 8:
            c = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.buf = (self.buf | (c << (8 - self.nbits))) & 0xFFFF
            self.nbits += 8

    def bit(self) -> int:
        self._fill()
        r = (self.buf >> 15) & 1
        self.buf = (self.buf << 1) & 0xFFFF
        self.nbits -= 1
        return r

    def byte(self) -> int:
        self._fill()
        r = (self.buf >> 8) & 0xFF
        self.buf = (self.buf << 8) & 0xFFFF
        self.nbits -= 8
        return r


class _BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.buf = 0
        self.nbits = 0

    def put(self, n: int, c: int):
        self.buf = (self.buf | (c >> self.nbits)) & 0xFFFF
        self.nbits += n
        if self.nbits >= 8:
            self.out.append((self.buf >> 8) & 0xFF)
            self.nbits -= 8
            if self.nbits >= 8:
                self.out.append(self.buf & 0xFF)
                self.nbits -= 8
                self.buf = (c << (n - self.nbits)) & 0xFFFF
            else:
                self.buf = ((self.buf & 0xFF) << 8) & 0xFFFF

    def flush(self):
        if self.nbits > 0:
            self.out.append((self.buf >> 8) & 0xFF)
            self.nbits = 0


def decompress(blob: bytes, expected_size: int | None = None) -> bytes:
    """One compressed body back to its bytes, or LzhufError.

    `expected_size` is the proposal's uncompressed size when the caller has
    one; the declared size must then match before any decoding happens. When
    it does not have one, the declared size is capped — the field is 32 bits
    and arrives from the air, and decoding is the expensive step.
    """
    if len(blob) < 6:
        raise LzhufError("stream shorter than its CRC and length fields")
    supplied = blob[0] | (blob[1] << 8)
    if crc16(blob[2:]) != supplied:
        raise LzhufError("CRC mismatch")
    size = blob[2] | (blob[3] << 8) | (blob[4] << 16) | (blob[5] << 24)
    if expected_size is not None and size != expected_size:
        raise LzhufError(f"declared size {size}, proposal said {expected_size}")
    if expected_size is None and size > (1 << 26):
        raise LzhufError(f"declared size {size} is not a message")
    if size == 0:
        return b""
    tree = _Tree()
    rd = _BitReader(blob[6:])
    win = bytearray(N)
    for i in range(N - F):
        win[i] = 0x20
    r = N - F
    out = bytearray()
    while len(out) < size:
        c = tree.son[ROOT]
        while c < T:
            c = tree.son[c + rd.bit()]
        c -= T
        tree.update(c)
        if c < 256:
            out.append(c)
            win[r] = c
            r = (r + 1) & (N - 1)
        else:
            head = rd.byte()
            low = head
            for _ in range(_D_LEN[head] - 2):
                low = ((low << 1) | rd.bit()) & 0xFFFF
            pos = (_D_CODE[head] << 6) | (low & 0x3F)
            i = (r - pos - 1) & (N - 1)
            for k in range(c - 255 + THRESHOLD):
                ch = win[(i + k) & (N - 1)]
                out.append(ch)
                win[r] = ch
                r = (r + 1) & (N - 1)
    del out[size:]                     # a match may overshoot the declared end
    return bytes(out)


class _MatchTree:
    """The encoder's string index: a binary search tree over window positions,
    keyed on the F bytes at each, kept exactly as the published encoder keeps
    it. Exactly, because the choice among equal-length matches shows in the
    output bytes, and matching the reference encoder byte for byte is what the
    exactness test buys us."""

    def __init__(self, win: bytearray):
        self.win = win
        self.lson = [0] * (N + 1)
        self.dad = [N] * (N + 1)
        self.rson = [N] * (N + 257)
        self.match_position = 0
        self.match_length = 0

    def insert(self, r: int):
        win = self.win
        lson, dad, rson = self.lson, self.dad, self.rson
        geq = True
        p = N + 1 + win[r]
        rson[r] = lson[r] = N
        self.match_length = 0
        while True:
            if geq:
                if rson[p] == N:
                    rson[p] = r
                    dad[r] = p
                    return
                p = rson[p]
            else:
                if lson[p] == N:
                    lson[p] = r
                    dad[r] = p
                    return
                p = lson[p]
            i = 1
            while i < F and win[r + i] == win[p + i]:
                i += 1
            geq = win[r + i] >= win[p + i] or i == F
            if i > THRESHOLD:
                if i > self.match_length:
                    self.match_position = ((r - p) & (N - 1)) - 1
                    self.match_length = i
                    if i >= F:
                        break
                if i == self.match_length:
                    c = ((r - p) & (N - 1)) - 1
                    if c < self.match_position:
                        self.match_position = c
        # A full-length match: take over p's place in the tree.
        dad[r] = dad[p]
        lson[r] = lson[p]
        rson[r] = rson[p]
        dad[lson[p]] = r
        dad[rson[p]] = r
        if rson[dad[p]] == p:
            rson[dad[p]] = r
        else:
            lson[dad[p]] = r
        dad[p] = N

    def delete(self, p: int):
        lson, dad, rson = self.lson, self.dad, self.rson
        if dad[p] == N:
            return
        if rson[p] == N:
            q = lson[p]
        elif lson[p] == N:
            q = rson[p]
        else:
            q = lson[p]
            if rson[q] != N:
                while rson[q] != N:
                    q = rson[q]
                rson[dad[q]] = lson[q]
                dad[lson[q]] = dad[q]
                lson[q] = lson[p]
                dad[lson[p]] = q
            rson[q] = rson[p]
            dad[rson[p]] = q
        dad[q] = dad[p]
        if rson[dad[p]] == p:
            rson[dad[p]] = q
        else:
            lson[dad[p]] = q
        dad[p] = N


def compress(data: bytes) -> bytes:
    """Bytes to one compressed body: CRC, length, LZHUF stream."""
    wr = _BitWriter()
    size = len(data)
    wr.out += bytes((size & 0xFF, (size >> 8) & 0xFF,
                     (size >> 16) & 0xFF, (size >> 24) & 0xFF))
    if size:
        tree = _Tree()
        win = bytearray(N + F)
        for i in range(N - F):
            win[i] = 0x20
        mt = _MatchTree(win)
        inptr = 0
        s = 0
        r = N - F
        length = 0
        while length < F and inptr < size:
            win[r + length] = data[inptr]
            inptr += 1
            length += 1
        for i in range(1, F + 1):
            mt.insert(r - i)
        mt.insert(r)
        while length > 0:
            if mt.match_length > length:
                mt.match_length = length
            if mt.match_length <= THRESHOLD:
                mt.match_length = 1
                wr.put(*tree.encode_symbol(win[r]))
            else:
                wr.put(*tree.encode_symbol(255 - THRESHOLD + mt.match_length))
                pos = mt.match_position
                wr.put(P_LEN[pos >> 6], P_CODE[pos >> 6] << 8)
                wr.put(6, (pos & 0x3F) << 10)
            last = mt.match_length
            i = 0
            while i < last and inptr < size:
                i += 1
                mt.delete(s)
                ch = data[inptr]
                inptr += 1
                win[s] = ch
                if s < F - 1:
                    win[s + N] = ch    # the wrapped copy InsertNode reads past N
                s = (s + 1) & (N - 1)
                r = (r + 1) & (N - 1)
                mt.insert(r)
            while i < last:
                i += 1
                mt.delete(s)
                s = (s + 1) & (N - 1)
                r = (r + 1) & (N - 1)
                length -= 1
                if length > 0:
                    mt.insert(r)
        wr.flush()
    stream = bytes(wr.out)
    crc = crc16(stream)
    return bytes((crc & 0xFF, (crc >> 8) & 0xFF)) + stream
