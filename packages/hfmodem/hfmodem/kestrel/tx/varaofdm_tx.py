# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Transmitter for VARA's coherent OFDM levels, the inverse of ``rx.varaofdm``.

The symbol shape is the tape's own, rendered back to the 16-bit floor
(docs/protocols/vara/evidence/TAPER-WINDOW.md): each 1024-sample IFFT is taken
cyclically from 128 samples before its origin to 32 after its end, both 32-sample
ends shaped by ``sin²(π(i + 0.5)/64)``, and overlap-added at 1152. The burst
opens on the first symbol's own rise and closes 1024 samples after the last
origin, falling over its last 32 samples; it is exactly ``nsym × 1152`` long.

The lead rows are ``LUT[Int(Rnd·2^bpc)]``, bin-major, from the session's
``VB6Rnd``, which the caller owns and this advances by ``lead × carriers`` per
frame; a two-frame over draws two lead blocks and keys the second
(LEAD-GENERATORS.md). No receiver reads them.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..coding.crc import crc16_genibus
from ..rx import varaofdm as _rx
from ..rx.varaofdm import CP, DATA, SYM, TU, OfdmRecord
from .varahf2300_tx import VB6Rnd

#: RMS of a rendered burst [ours]; the receiver is scale-invariant.
OUT_RMS = 0.25

_RAMP = np.sin(np.pi * (np.arange(32) + 0.5) / 64) ** 2
_SUFFIX = len(_RAMP)

#: The byte ahead of a short frame's ``82`` control byte at every high speed capable
#: level above 4, tone rows included.
SHORT_TAIL = {2300: 0x06, 2750: 0x06, 500: 0x07}


def build_frame(payload: bytes, control: int, bw: int, level: int, *, caller: str) -> bytes:
    """payload + control byte + CRC-16/GENIBUS, ``frame_bytes`` long.

    A payload short of the row's capacity carries the tape's fill ahead of the
    control byte: ``14``, the high byte of the caller's callsign CRC (``09`` for
    N0CALL, ``f6`` for W9SSJ), zeros, then ``06`` (``07`` at BW500); with two
    bytes spare it is ``14 06``/``14 07``. The short frames on tape all carry control
    ``82``; which byte to send is the caller's."""
    r = _rx.ROWS[bw, level]
    spare = r.frame_bytes - 3 - len(payload)
    if spare < 0:
        raise ValueError(f"{len(payload)} bytes exceed BW{bw} L{level}'s {r.frame_bytes - 3}")
    tail = bytes([SHORT_TAIL[bw]])
    key = crc16_genibus(caller.upper().encode()) >> 8
    fill = (b"" if not spare else tail if spare == 1 else b"\x14" + tail if spare == 2
            else bytes([0x14, key]) + bytes(spare - 3) + tail)
    body = bytes(payload) + fill + bytes([control])
    return body + crc16_genibus(body).to_bytes(2, "big")


def _frame_cells(frame: bytes, r: OfdmRecord) -> np.ndarray:
    """One frame's cells after its lead rows, scrambled."""
    kind, fill = r.layout
    v = _rx.onair(frame, r).reshape(-1, r.bpc) @ (1 << np.arange(r.bpc - 1, -1, -1))
    x = np.ones(kind.shape, complex)
    x[kind == DATA] = r.lut[v[fill[kind == DATA]]]
    if not r.qam:
        x[r.lead:] = np.cumprod(x[r.lead:], axis=0)
    return (x * r.scramble)[r.lead:]


def cells(frames: Sequence[bytes], bw: int, level: int, rnd: VB6Rnd) -> np.ndarray:
    """(symbols, carriers): the lead from ``rnd``, then each frame's cells."""
    r = _rx.ROWS[bw, level]
    if len(frames) != 1 and not (r.qam and len(frames) == 2):
        raise ValueError(f"BW{bw} L{level} carries {'one or two frames' if r.qam else 'one frame'}")
    for _ in frames:
        lead = [rnd.draw(r.bpc) for _ in range(r.lead * r.carriers)]
    head = r.lut[np.array(lead)].reshape(r.carriers, r.lead).T
    return np.concatenate([head, *(_frame_cells(f, r) for f in frames)])


def render(frames: bytes | Sequence[bytes], bw: int, level: int, rnd: VB6Rnd) -> np.ndarray:
    """One OFDM over at ``bw``/``level``: one frame, or two at a QAM row."""
    frames = [frames] if isinstance(frames, (bytes, bytearray)) else list(frames)
    r = _rx.ROWS[bw, level]
    x = cells(frames, bw, level, rnd)
    spec = np.zeros((len(x), TU), complex)
    spec[:, r.bins] = x
    u = np.fft.ifft(spec, axis=1).real
    n = len(x) * SYM
    span = np.arange(-CP, TU + _SUFFIX)
    shape = np.ones(len(span))
    shape[:_SUFFIX] = _RAMP
    shape[-_SUFFIX:] = _RAMP[::-1]
    out = np.zeros(n + _SUFFIX)
    for s, sym in enumerate(u):
        out[s * SYM + CP + span] += shape * sym[span % TU]
    out = out[:n]
    out[-_SUFFIX:] *= _RAMP[::-1]
    return out * (OUT_RMS / np.sqrt(np.mean(out ** 2)))


def synth_burst(payload: bytes, bw: int, level: int, control: int, rnd: VB6Rnd, *,
                caller: str) -> np.ndarray:
    return render(build_frame(payload, control, bw, level, caller=caller), bw, level, rnd)
