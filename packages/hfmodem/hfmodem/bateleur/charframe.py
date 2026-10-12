# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Start-stop characters over a bit-synchronous waveform.

MIL-STD-188-110B C.5.4.1.1, the conventional asynchronous interface. The DTE
gives words of a start bit, an N-bit character (5 to 8 bits, any parity bit
included) and 1 or 2 stop bits. The whole word is sent contiguously, and
further stop bits fill any gap between words. There is no modem-defined null
character. The start bit is 0 (SPACE), stop bits and fill are 1 (MARK), and
the character goes least significant bit first, as a serial port sends it.

A station set this way (as many hardware modems and MARS stations are) puts
these words on the air. The 188-110A recordings published as test messages
are such a stream, 8N1. Without this module, bateleur sends what C.5.4.1.2
calls the high-speed asynchronous interface: the bytes themselves, no start
or stop bits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CharFormat:
    data_bits: int = 8                  # the character, parity not included
    parity: str = "N"                   # N, E, O, M (mark) or S (space)
    stop_bits: int = 1

    @classmethod
    def parse(cls, text: str) -> "CharFormat":
        """"8N1", "7E1", "5N2" and the like."""
        t = text.strip().upper()
        if len(t) != 3 or not t[0].isdigit() or not t[2].isdigit() or t[1] not in "NEOMS":
            raise ValueError(f"a character format is like 8N1 or 7E1, not {text!r}")
        f = cls(int(t[0]), t[1], int(t[2]))
        n = f.data_bits + (f.parity != "N")
        if not 5 <= n <= 8:
            raise ValueError("a character is 5 to 8 bits, any parity bit included "
                             "(110B C.5.4.1.1)")
        if f.stop_bits not in (1, 2):
            raise ValueError("1 or 2 stop bits (110B C.5.4.1.1)")
        return f

    @property
    def name(self) -> str:
        return f"{self.data_bits}{self.parity}{self.stop_bits}"

    @property
    def word_bits(self) -> int:
        return 1 + self.data_bits + (self.parity != "N") + self.stop_bits

    def parity_bit(self, value: int) -> int | None:
        ones = bin(value & ((1 << self.data_bits) - 1)).count("1")
        return {"N": None, "E": ones & 1, "O": (ones & 1) ^ 1, "M": 1, "S": 0}[self.parity]


def frame(data: bytes, fmt: CharFormat) -> np.ndarray:
    """Bytes -> the words' bits as they go on the air, first bit first. Each
    byte's low ``data_bits`` are the character."""
    out = []
    for v in data:
        w = [0] + [(v >> i) & 1 for i in range(fmt.data_bits)]
        p = fmt.parity_bit(v)
        if p is not None:
            w.append(p)
        w += [1] * fmt.stop_bits
        out += w
    return np.array(out, dtype=np.uint8)


def pack(bits: np.ndarray, order: str = "lsb") -> bytes:
    """Framed bits -> bytes that a waveform taking a byte stream in ``order``
    sends as those bits, filled out to a whole byte with stop bits."""
    b = np.asarray(bits, dtype=np.uint8)
    b = np.concatenate([b, np.ones((-b.size) % 8, dtype=np.uint8)])
    return np.packbits(b.reshape(-1, 8), axis=1,
                       bitorder="little" if order == "lsb" else "big").tobytes()


def unpack(data: bytes, order: str = "lsb") -> np.ndarray:
    """The inverse of ``pack``: a received byte stream back to its bits."""
    a = np.frombuffer(bytes(data), dtype=np.uint8)
    return np.unpackbits(a[:, None], axis=1,
                         bitorder="little" if order == "lsb" else "big").ravel()


class Deframer:
    """Received bits -> characters, a piece at a time. It waits through stop
    bits (1s) for a start bit (0), takes the word, and checks its stop bits.
    A word whose stop bit is a 0 is a framing error: it is dropped, and the
    hunt for a start bit begins again just after the false one, as a serial
    port's receiver does. Parity is checked and counted, not enforced."""

    def __init__(self, fmt: CharFormat) -> None:
        self.fmt = fmt
        self._bits = np.zeros(0, dtype=np.uint8)
        self.framing_errors = 0
        self.parity_errors = 0

    def push(self, bits) -> bytes:
        b = np.concatenate([self._bits, np.asarray(bits, dtype=np.uint8)])
        f = self.fmt
        par = f.parity != "N"
        need = 1 + f.data_bits + par + 1          # the first stop bit decides
        out = bytearray()
        i = 0
        while True:
            z = np.flatnonzero(b[i:] == 0)
            if z.size == 0:
                i = b.size
                break
            i += int(z[0])
            if i + need > b.size:
                break                             # the word is not all here yet
            w = b[i: i + need]
            if w[-1] != 1:
                self.framing_errors += 1
                i += 1
                continue
            v = int(sum(int(w[1 + k]) << k for k in range(f.data_bits)))
            if par and int(w[1 + f.data_bits]) != f.parity_bit(v):
                self.parity_errors += 1
            out.append(v)
            i += need                             # a second stop bit is idle as well
        self._bits = b[i:]
        return bytes(out)
