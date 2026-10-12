# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Robust Packet framing and coding, with FCS-free AX.25 boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

import numpy as np

from hfmodem.ax25 import frame as ax25
from hfmodem.core.crc import crc16
from hfmodem.core.fec import ConvCode, Puncture, viterbi_decode

CLASS_BYTES = (15, 21, 45, 75, 105, 126, 147, 168, 189, 210, 231, 252, 273, 294, 312)
MAX_EXT = 16
REFERENCE = 0x3A
ALPHABET = ax25.CALLSIGN_CHARS + '!"#'
CODE = ConvCode(9, (0o657, 0o435))
PUNCTURE = Puncture(((1, 1, 0), (0, 1, 1)))


def pn9(count: int) -> bytes:
    state = 1
    result = bytearray(count)
    for i in range(count * 8):
        state = (state >> 1) | (((state ^ (state >> 4)) & 1) << 8)
        if i % 8 == 7:
            result[i // 8] = state & 255
    return bytes(result)


PREAMBLE = pn9(8)


def scramble(packet: bytes) -> bytes:
    if not packet:
        raise ValueError("missing packet tail")
    return bytes(a ^ b for a, b in zip(packet[:-1], pn9(len(packet) - 1))) + packet[-1:]


@lru_cache(maxsize=64)
def interleaver(count: int) -> np.ndarray:
    if not 64 <= count <= 16 * 825:
        raise ValueError("unsupported interleaver length")
    for limit, width, taps in ((1024, 10, (9, 2)), (4096, 12, (11, 5, 3, 0)),
                               (8192, 13, (12, 3, 2, 0)), (32768, 15, (14, 0))):
        if count < limit:
            break
    remap = [0]
    state = 30
    for _ in range(63):
        state = ((state << 1) | ((state >> 5) ^ state) & 1) & 63
        remap.append(state)
    positions = [0]
    state = 1
    while len(positions) < count:
        feedback = sum((state >> bit) & 1 for bit in taps) & 1
        state = ((state << 1) | feedback) & ((1 << width) - 1)
        if state < count:
            positions.append(remap[state] if state < 64 else state)
    result = np.asarray(positions)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=1)
def header_codebook() -> np.ndarray:
    seed = np.unpackbits(np.array([0x3A, 0xC5, 0x6A, 0x5C, 0xA3, 0x5C, 0xC6, 0xAC],
                                  dtype=np.uint8), bitorder="little").reshape(8, 8)
    order = (0, 6, 2, 5, 1, 3, 7, 4)
    book = np.empty((64, 64), dtype=np.int16)
    for index in range(64):
        row, col = divmod(index, 8)
        r = np.array([(order[row] & s).bit_count() & 1 for s in range(8)])
        c = np.array([(order[col] & s).bit_count() & 1 for s in range(8)])
        bits = seed ^ r[:, None] ^ c[None, :] ^ (row >= 5) ^ (col in (2, 4, 5))
        book[index] = 1 - 2 * bits.ravel()
    book.flags.writeable = False
    return book


def header_bits(index: int) -> np.ndarray:
    if not 0 <= index < 64:
        raise ValueError("invalid header index")
    return ((1 - header_codebook()[index]) // 2).reshape(8, 8)


def decode_header(soft: np.ndarray) -> tuple[int, float]:
    soft = np.asarray(soft, float).ravel()
    if soft.size != 64 or not np.isfinite(soft).all() or not np.any(soft):
        raise ValueError("invalid header soft values")
    scores = header_codebook() @ soft / np.abs(soft).sum()
    index = int(scores.argmax())
    if scores[index] < 0.5:
        raise ValueError("header correlation below threshold")
    return index, float(scores[index])


def packet_bytes(cls: int, ext: int = 0) -> int:
    if not 0 <= cls <= 15 or not 0 <= ext <= MAX_EXT or (cls != 15 and ext != 0):
        raise ValueError("invalid class or extension")
    return CLASS_BYTES[cls] if cls < 15 else 345 + 30 * ext


def length_class(size: int) -> tuple[int, int]:
    if not 1 <= size <= packet_bytes(15, MAX_EXT):
        raise ValueError("packet size outside class table")
    for cls, capacity in enumerate(CLASS_BYTES):
        if size <= capacity:
            return cls, 0
    return 15, max(0, (size - 345 + 29) // 30)


def coded_bits(n: int, level: int) -> int:
    if level not in (0, 1) or n not in (*CLASS_BYTES, *range(345, 826, 30)):
        raise ValueError("invalid packet capacity or level")
    return n * 16 if level == 0 else n * 32 // 3


def symbol_count(cls: int, ext: int, level: int) -> int:
    return 17 + 8 * (cls == 15) + coded_bits(packet_bytes(cls, ext), level) // (8 << level)


def encode_packet(packet: bytes, level: int) -> np.ndarray:
    count = coded_bits(len(packet), level)
    if packet[-1] != 0:
        raise ValueError("encoder tail must be zero")
    bits = np.unpackbits(np.frombuffer(scramble(packet), np.uint8), bitorder="little")
    coded = CODE.encode(bits, terminate=False)
    if level:
        coded = PUNCTURE.apply(coded)
    result = np.empty(count, np.uint8)
    result[interleaver(count)] = coded
    return result


def decode_packet(soft: np.ndarray, n: int, level: int) -> tuple[bytes, float]:
    soft = np.asarray(soft, float).ravel()
    if len(soft) != coded_bits(n, level) or not np.isfinite(soft).all():
        raise ValueError("invalid or truncated coded packet")
    ordered = soft[interleaver(len(soft))]
    mother = PUNCTURE.depuncture(ordered, 8 * n) if level else ordered
    bits = viterbi_decode(mother, CODE, pinned=True)
    recoded = 1 - 2 * CODE.encode(bits).astype(float)
    heard = mother != 0
    agreement = float(np.mean(recoded[heard] * mother[heard] > 0)) if heard.any() else 0.0
    data = np.packbits(bits, bitorder="little").tobytes()
    return scramble(data + b'\0')[:-1], agreement


def compress_address(raw: bytes) -> bytes:
    ax25.validate_address(raw)
    number = sum(ALPHABET.index(chr(b >> 1)) * 40 ** i for i, b in enumerate(raw[:6]))
    return number.to_bytes(4, "little") + raw[6:]


def expand_address(raw: bytes) -> bytes:
    if len(raw) != 5:
        raise ValueError("truncated compressed address")
    number = int.from_bytes(raw[:4], "little")
    chars = bytearray()
    for _ in range(6):
        number, digit = divmod(number, 40)
        chars.append(ord(ALPHABET[digit]) << 1)
    if number:
        raise ValueError("compressed callsign overflow")
    result = bytes(chars) + raw[4:]
    ax25.validate_address(result)
    return result


def expand(data: bytes) -> tuple[bytes, bytes]:
    addresses = bytearray()
    for count in range(1, 11):
        raw = expand_address(data[5 * (count - 1):5 * count])
        addresses.extend(raw)
        if raw[6] & 1:
            if count < 2:
                raise ValueError("too few compressed addresses")
            return bytes(addresses), data[5 * count:]
    raise ValueError("missing address terminator")


def _parts(raw: bytes) -> tuple[bytes, bytes]:
    parsed = ax25.parse(raw)
    if len(raw) > 399:
        raise ValueError("AX.25 frame exceeds 399 bytes")
    addresses = b"".join(parsed.addresses)
    return addresses, raw[len(addresses):]


def _compress(addresses: bytes) -> bytes:
    return b"".join(compress_address(addresses[i:i + 7]) for i in range(0, len(addresses), 7))


def _without_c(addresses: bytes) -> bytes:
    out = bytearray(addresses)
    out[6] &= 0x7F
    out[13] &= 0x7F
    return bytes(out)


def _complementary(addresses: bytes) -> bool:
    return bool((addresses[6] ^ addresses[13]) & 0x80)


@dataclass(frozen=True)
class Packet:
    data: bytes
    cls: int
    ext: int

    def __post_init__(self) -> None:
        if len(self.data) != packet_bytes(self.cls, self.ext) or self.data[-1] != 0:
            raise ValueError("packet capacity or encoder tail does not match header")


def pack(content: bytes, *, bare: bool) -> Packet:
    if not content:
        raise ValueError("empty content")
    cls, ext = length_class(len(content) + 1)
    padding = packet_bytes(cls, ext) - len(content) - 1
    if bare:
        if padding > 31:
            raise ValueError("bare padding exceeds 31")
        content = bytes([content[0] | padding]) + content[1:]
    return Packet(content + bytes(padding + 1), cls, ext)


def assemble(frames: list[bytes], suggestion: int = 0) -> Packet:
    if not 1 <= len(frames) <= 8 or not 0 <= suggestion <= 3:
        raise ValueError("invalid frame count or speed suggestion")
    parts = [_parts(f) for f in frames]
    addresses, body = parts[0]
    compressed = _compress(addresses) + body
    head = suggestion << 5
    if len(frames) == 1:
        return pack(bytes([head]) + compressed + crc16(frames[0]).to_bytes(2, "little"), bare=True)
    if any(not _complementary(a) or _without_c(a) != _without_c(addresses) for a, _ in parts):
        raise ValueError("frames cannot share a container without alteration")
    content = bytearray([head | 0x80 | (len(frames) - 1)])
    content.extend(len(compressed).to_bytes(2, "little"))
    content.extend(compressed)
    for a, body in parts[1:]:
        content.extend((len(body) | (a[6] & 0x80) << 8).to_bytes(2, "little"))
        content.extend(body)
    content.extend(crc16(content).to_bytes(2, "little"))
    return pack(bytes(content), bare=False)


def pack_frames(frames: list[bytes], suggestion: int = 0, *, max_capacity: int = 825) -> list[Packet]:
    """Pack in order; log and skip frames that fit no permitted bare class."""
    if not frames:
        raise ValueError("empty frame list")
    if not 0 <= suggestion <= 3:
        raise ValueError("invalid speed suggestion")
    for raw in frames:
        ax25.parse(raw)
    def checked(group):
        packet = assemble(group, suggestion)
        if len(packet.data) > max_capacity:
            raise ValueError("packet exceeds permitted class capacity")
        return packet

    result = []
    group: list[bytes] = []
    for index, raw in enumerate(frames):
        # A singleton must be sendable when an address/count boundary splits it.
        try:
            checked([raw])
        except ValueError as exc:
            logging.getLogger(__name__).warning("cannot pack frame %d: %s", index, exc)
            continue
        if group:
            try:
                checked([*group, raw])
            except ValueError:
                result.append(checked(group))
                group = []
        group.append(raw)
    if group:
        result.append(checked(group))
    return result


def parse(packet: bytes) -> tuple[int, list[bytes]]:
    """Validate a decoded packet without its tail before returning any frames."""
    if len(packet) + 1 not in (*CLASS_BYTES, *range(345, 826, 30)):
        raise ValueError("invalid decoded packet length")
    head = packet[0]
    suggestion = (head >> 5) & 3
    if not head & 0x80:
        padding = head & 31
        end = len(packet) - padding
        if end <= 1:
            raise ValueError("invalid bare padding")
        addresses, body = expand(packet[1:end])
        frame = ax25.check_fcs(addresses + body)
        return suggestion, [frame]
    cursor = 1
    records = []
    for i in range((head & 7) + 1):
        if cursor + 2 > len(packet):
            raise ValueError("truncated container length")
        word = int.from_bytes(packet[cursor:cursor + 2], "little")
        size = word & 511
        cursor += 2
        if not size or cursor + size > len(packet) - 2:
            raise ValueError("container length exceeds packet")
        records.append((word >> 15, packet[cursor:cursor + size]))
        cursor += size
    if crc16(packet[:cursor]) != int.from_bytes(packet[cursor:cursor + 2], "little"):
        raise ValueError("container CRC failure")
    addresses, body = expand(records[0][1])
    frames = [addresses + body]
    for c, body in records[1:]:
        a = bytearray(addresses)
        a[6] = (a[6] & 0x7F) | c << 7
        a[13] = (a[13] & 0x7F) | (c ^ 1) << 7
        frames.append(bytes(a) + body)
    for raw in frames:
        ax25.parse(raw)
    return suggestion, frames
