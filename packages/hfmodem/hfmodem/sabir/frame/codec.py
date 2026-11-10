# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Payload framing: bytes <-> coded, whitened, interleaved bits, per gear.

Per codeword (k info bits = k/8 bytes): ``[len:1][data:k/8-3][zero pad][CRC-16:2]``.
CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no final XOR) over
the first k/8-2 bytes. The n coded bits are whitened with PN9 (LFSR
x^9 + x^5 + 1, seed 0x1FF, restarted per codeword) and block-interleaved
32 x n/32 (written row-wise, read column-wise). Multi-codeword frames are
*striped* -- bit i of codeword j lands at stream position i*n_cw + j -- so
every codeword spans the whole frame's fade cycles and coherence bands
instead of its own 1/n_cw slice. A gear with ``repeat`` R > 1 transmits the
whole stream R times back to back
(~0.7 s apart -- a full fade cycle on the Poor profile) and the receiver
soft-combines the copies by summing LLRs. Tail padding to a whole number of
OFDM symbols is the PHY's job.

The receiver inverts everything from LLRs: de-whitening is a sign flip, so it
costs nothing in soft-ness.

"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.fec import QCLDPC

def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1) & 0xFFFF
    return crc


def crc_ok(block: bytes) -> bool:
    """True iff the block ends in CRC-16 over the rest -- the shape of every
    sabir control/floor block, and the CA-TBCC list screen."""
    return len(block) > 2 and crc16(block[:-2]) == int.from_bytes(
        block[-2:], "big")


def pn9(n: int) -> np.ndarray:
    """PN9 whitening sequence: LFSR x^9 + x^5 + 1, seed 0x1FF, LSB out."""
    state = 0x1FF
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = state & 1
        state = (state >> 1) | (((state ^ (state >> 5)) & 1) << 8)
    return out


class FrameCodec:
    def __init__(self, code: QCLDPC | None = None, repeat: int = 1):
        self.code = code or QCLDPC()
        self.repeat = repeat
        n, k = self.code.n, self.code.k
        self.info_bytes = k // 8
        self.data_bytes = self.info_bytes - 3
        self.perm = np.arange(n).reshape(32, n // 32).T.ravel()
        self.inv_perm = np.argsort(self.perm)
        self.pn = pn9(n)

    def chunk(self, payload: bytes) -> list[bytes]:
        """Split a payload into per-codeword chunks (last one may be short)."""
        return [payload[i : i + self.data_bytes]
                for i in range(0, len(payload), self.data_bytes)] or [b""]

    # -- transmit -------------------------------------------------------------
    def encode_cw(self, chunk: bytes) -> np.ndarray:
        """One chunk (<= data_bytes) -> one whitened, interleaved codeword."""
        if len(chunk) > self.data_bytes:
            raise ValueError("DATA chunk exceeds codeword capacity")
        block = bytes([len(chunk)]) + chunk.ljust(self.data_bytes, b"\0")
        block += crc16(block).to_bytes(2, "big")
        info = np.unpackbits(np.frombuffer(block, dtype=np.uint8)).astype(np.int64)
        return (self.code.encode(info).astype(np.int64) ^ self.pn)[self.perm]

    def encode(self, payload: bytes) -> np.ndarray:
        """Payload to striped, repeated codewords; PHY adds symbol padding."""
        out = [self.encode_cw(chunk) for chunk in self.chunk(payload)]
        return np.tile(np.stack(out).T.ravel(), self.repeat)

    # -- receive --------------------------------------------------------------
    def decode_cws(self, llr: np.ndarray) -> tuple[list[bytes | None], dict]:
        """Per-codeword LLR rows (n_cw, n) -> per-codeword chunks or None."""
        deint = np.atleast_2d(llr)[:, self.inv_perm] * (1 - 2 * self.pn)
        hard, dec_ok, iters = self.code.decode(deint)
        chunks: list[bytes | None] = []
        for w in range(hard.shape[0]):
            block = np.packbits(hard[w, : self.code.k].astype(np.uint8)).tobytes()
            good = (crc16(block[: self.info_bytes - 2]) == int.from_bytes(
                block[self.info_bytes - 2 : self.info_bytes], "big")
                and block[0] <= self.data_bytes
                and not any(block[1 + block[0]:self.info_bytes - 2]))
            chunks.append(block[1 : 1 + block[0]] if good else None)
        return chunks, {"ldpc_ok": dec_ok.tolist(), "iters": iters.tolist()}

    def decode(self, llr: np.ndarray) -> tuple[bytes | None, dict]:
        """LLRs -> (payload bytes or None if any codeword failed, stats)."""
        L = self.code.n
        n_cw = llr.size // (self.repeat * L)
        if n_cw == 0:
            return None, {"n_cw": 0, "crc_ok": []}
        stream = n_cw * L
        cw_llr = llr[: self.repeat * stream].reshape(
            self.repeat, stream).sum(axis=0).reshape(L, n_cw).T
        chunks, st = self.decode_cws(cw_llr)
        good_cw = [c is not None for c in chunks]
        payload = b"".join(c for c in chunks if c is not None)
        stats = {"n_cw": n_cw, "crc_ok": good_cw, **st}
        return (payload if all(good_cw) else None), stats
