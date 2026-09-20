# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Explicit DATA profiles, independent of the baseline speed ladder.

IDs describe complete immutable waveform/coding/layout tuples. Receiver-local
algorithms and selection policies do not affect their identity.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.fec.tbcc import CATBCC
from hfmodem.sabir.frame.codec import FrameCodec, crc16, crc_ok
from hfmodem.sabir.phy.modem import GEARS


@dataclass(frozen=True)
class DataProfile:
    id: int
    name: str
    frame_cws: int
    grouped: bool = False
    floor: str | None = None
    bandwidth: int = 2750


PROFILES = {p.name: p for p in (
    DataProfile(3, "robust", 4, bandwidth=1500),
    DataProfile(4, "workhorse", 12, bandwidth=1500),
    DataProfile(5, "workhorse34", 16, bandwidth=1500),
    DataProfile(6, "fast", 48, True), DataProfile(7, "max", 60, True),
    DataProfile(16, "doppler", 8),
    DataProfile(17, "sparse34", 16, bandwidth=1500),
    DataProfile(18, "narrow", 1, floor="floor", bandwidth=500),
    DataProfile(19, "narrow2", 1, floor="floor2", bandwidth=500),
    DataProfile(20, "narrow4", 1, floor="floor4", bandwidth=500),
    DataProfile(21, "wide256", 64, True),
)}
BY_ID = {p.id: p for p in PROFILES.values()}
EXTENDED = tuple(p.name for p in PROFILES.values() if p.id >= 16)


class FloorDataCodec:
    """22-byte CRC-terminated block: length, <=19 data bytes, zero pad, CRC16.

    The ARQ store holds 352 coded LLRs, so retransmissions retain soft evidence.
    Whitening/interleaving belongs to FloorModem, not this codec.
    """
    data_bytes = 19
    code = SimpleNamespace(n=352, k=176)

    def __init__(self):
        self.tb = CATBCC()

    def chunk(self, payload):
        return [payload[i:i + 19] for i in range(0, len(payload), 19)] or [b""]

    def encode_cw(self, chunk):
        if len(chunk) > 19:
            raise ValueError("floor DATA chunk exceeds 19 bytes")
        block = bytes([len(chunk)]) + chunk.ljust(19, b"\0")
        return self.tb.encode(block + crc16(block).to_bytes(2, "big"))

    def decode_cws(self, llrs):
        out = []
        for row in np.atleast_2d(llrs):
            block, _ = self.tb.decode(row, 22, crc_ok)
            valid = (block is not None and block[0] <= 19
                     and not any(block[1 + block[0]:20]))
            out.append(block[1:1 + block[0]] if valid else None)
        return out, {}


def codec_for(name):
    return (FloorDataCodec() if PROFILES[name].floor else
            FrameCodec(QCLDPC(GEARS[name].code)))
