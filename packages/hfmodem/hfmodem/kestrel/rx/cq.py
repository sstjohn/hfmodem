# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The measured narrow CQ record, independent of the ARQ level ladder."""
from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np

from . import tablegen
from ..coding import turbo
from ..coding.crc import crc16_genibus
from ..vara.vara_frames import _pack_callsign6, unpack_callsign6


@dataclass(frozen=True)
class CQRecord:
    record: int = 14
    information_bits: int = 64
    coded_bits: int = 204
    columns: int = 92
    symbol_samples: int = 2048
    sample_rate: int = 48000


CQ = CQRecord()
HEADER = np.array([74, 52, 74, 58, 64, 76, 58, 74, 52, 52, 70, 72, 72, 54, 50, 58, 68, 52, 74, 70, 50, 64, 54, 60, 54, 60, 70, 62, 56, 56, 66, 60], dtype=int)
TRAINING = np.array([74, 67], dtype=int)
REFERENCES = np.array([2, 6, 10, 14, 17, 21, 25, 29, 33, 36, 40, 44, 48, 52, 55, 59, 63, 67, 71, 74, 78, 82, 86, 89], dtype=int)
ALLOCATION = np.array([59, 62, 55, 54, 59, 72, 61, 72, 67, 54, 55, 64, 57, 54, 73, 60, 67, 72, 69, 60, 61, 58, 67, 62, 54, 60, 71, 58, 73, 62, 61, 60, 67, 62, 55, 64, 63, 54, 71, 56, 55, 72, 65, 62, 69, 74, 63, 54, 55, 54, 59, 72, 65, 60, 71, 56, 57, 72, 59, 66, 57, 68, 54, 64, 73, 72, 71, 54, 73, 56, 54, 62, 71, 62, 55, 62, 57, 74, 69, 54, 59, 62, 61, 70, 73, 56, 73, 62, 71, 60, 59, 72], dtype=int)
REFERENCE_CLASSES = np.array([6, 4, 6, 0, 4, 1, 7, 1, 3, 7, 1, 6, 1, 1, 2, 1, 5, 1, 6, 2, 5, 2, 7, 5], dtype=int)
DATA = np.setdiff1d(np.arange(CQ.columns), REFERENCES)
GRAY = tablegen.clsparm_gray()[16:24]
REFERENCE_BINS = 54 + (ALLOCATION[REFERENCES] - 54 + 2 * REFERENCE_CLASSES) % 21
DATA_BINS = 54 + (ALLOCATION[DATA, None] - 54 + 2 * GRAY[None, :]) % 21
LABELS = (np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1


@lru_cache(maxsize=1)
def tables():
    place, state = tablegen.permutation(CQ.coded_bits, tablegen.BW500[CQ.record][0])
    perm, _ = tablegen.permutation(CQ.information_bits, state)
    return place, perm, tablegen.whitener(CQ.information_bits)


def build_frame(call: str, bandwidth: int) -> bytes:
    match = re.fullmatch(r'([A-Z0-9]{2,7})(?:-([1-9]|1[0-5]))?', call.upper())
    if match is None or bandwidth not in (500, 2300, 2750):
        raise ValueError('unsupported CQ identity or bandwidth')
    body = bytearray(_pack_callsign6(match[1]).ljust(6, b'\0'))
    body[5] |= ({500: 1, 2300: 2, 2750: 3}[bandwidth] << 4) | int(match[2] or 0)
    body = bytes(body)
    return body + crc16_genibus(body).to_bytes(2, 'big')


@dataclass(frozen=True)
class CQResult:
    frame: bytes
    call: str
    bandwidth: int
    sample: int = 0
    sn: float | None = 0.0


def parse_frame(frame: bytes, *, sample=0, sn=0.0) -> CQResult | None:
    if len(frame) != 8 or crc16_genibus(frame[:6]) != int.from_bytes(frame[6:], 'big'):
        return None
    bandwidth = {1: 500, 2: 2300, 3: 2750}.get((frame[5] >> 4) & 3)
    base = unpack_callsign6(frame[:5] + bytes([frame[5] & 0xc0]))
    suffix = frame[5] & 15
    call = base + (f'-{suffix}' if suffix else '')
    try:
        canonical = build_frame(call, bandwidth)
    except ValueError:
        return None
    mask = 0xff if len(base) == 7 else 0xf0
    if (canonical[:4] != frame[:4] or canonical[4] & mask != frame[4] & mask
            or canonical[5] != frame[5]):
        return None
    return CQResult(frame, call, bandwidth, sample, None if sn is None else float(sn))


def decode_power(power: np.ndarray) -> bytes | None:
    selected = power[DATA[:, None], DATA_BINS - 54]
    noise = np.maximum(np.median(power[DATA], axis=1), 1e-20)
    ll = np.stack([(np.max(np.where(LABELS[:, b] == 0, selected, -np.inf), axis=1)
                    - np.max(np.where(LABELS[:, b] == 1, selected, -np.inf), axis=1))
                   / noise for b in range(3)], axis=1)
    place, perm, pn = tables()
    coded = np.empty(CQ.coded_bits)
    coded[place] = np.clip(ll.ravel(), -8, 8)
    def check(bits):
        frame = np.packbits(bits ^ pn).tobytes()
        return crc16_genibus(frame[:6]) == int.from_bytes(frame[6:], 'big')
    bits = turbo.decode_r13(coded, CQ.information_bits, perm, iters=12, check=check)
    return np.packbits(bits ^ pn).tobytes() if check(bits) else None
