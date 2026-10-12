# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tone family's transmitter, the forward law of ``rx.varatone``.

Per band: the lead at t = 0..7, the reference at t = 8, the frame's differential
increments with pilot columns repeating, all times conj(grid row), one pulse per
symbol at the band's centre and stagger, summed real. The burst's onset and tail
are the pulses' own: the first and last symbols ramp in and out over the pulse.

The lead and the reference's phase come from the session's Rnd stream
(docs/protocols/vara/evidence/LEAD-GENERATORS.md): 9 x bands draws of
``Int(Rnd*4)``, time-major, Gray to quarter turns; row 0 is the reference, rows
1..8 the lead, each times conj(grid row) at column t, the columns the data at
t + 8 reads. Read off BW2300 L5-9, and confirmed cell by cell at BW2750 L5-10
and BW500 L5-6 (LEAD-CHECK-tone.md).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ..rx import varatone as rx
from ..rx.varatone import C0, FS, H, ROWS, ToneRecord
from ..coding.crc import crc16_genibus
from .varahf2300_tx import VB6Rnd
from .varaofdm_tx import SHORT_TAIL

#: Burst r.m.s. [ours]: the BW500 base level's, so a gear change does not move the drive.
RMS = 106.5


def build_frame(payload: bytes, control: int, rec: ToneRecord, *, caller: str) -> bytes:
    """payload | control | CRC-16/GENIBUS. A short payload carries the tape's fill,
    keyed to the session's caller as every short VARA body is  [arq.phy.vara_body]."""
    n = rec.payload_bytes
    if len(payload) < n:
        if len(payload) > n - 3:
            raise ValueError(f"a short payload holds at most {n - 3} bytes at {rec.bw} L{rec.level}")
        key = crc16_genibus(caller.upper().encode()) >> 8
        payload = bytes(payload) + bytes([0x14, key]) + bytes(n - 3 - len(payload)) + bytes([SHORT_TAIL[rec.bw]])
    elif len(payload) > n:
        raise ValueError(f"{rec.bw} L{rec.level} carries {n} payload bytes")
    body = bytes(payload) + bytes([control & 0xFF])
    c = crc16_genibus(body)
    return body + bytes((c >> 8, c & 0xFF))


def quadrants(frame: bytes, rec: ToneRecord) -> np.ndarray:
    """(bands, nsym - C0) cumulative quadrant from the reference: increments with
    pilot columns holding."""
    b = rx.onair_bits(frame, rec)
    inc = rx.GRAY[b[0::2] * 2 + b[1::2]].reshape(rec.nb, rec.ndata)
    step = np.zeros((rec.nb, rec.nsym - C0), int)
    np.put_along_axis(step, rx.data_symbols(rec) - C0, inc, 1)
    return np.cumsum(step, 1) % 4


def body(frame: bytes, rec: ToneRecord, ref: np.ndarray) -> np.ndarray:
    """(bands, nsym - C0) cells from the reference on: i^(ref + quadrant) * conj(grid
    row). An unmeasured grid column keys the cell unrotated."""
    g = np.maximum(rx.grid(rec.nb)[:, :rec.nsym - C0], 0)
    return 1j ** ((ref[:, None] + quadrants(frame, rec)) % 4) * np.exp(-2j * np.pi * g / 128)


def cells(frame: bytes, rec: ToneRecord, rnd: VB6Rnd) -> np.ndarray:
    """(bands, nsym) cells as keyed, drawing the over's lead from ``rnd``."""
    q = rx.GRAY[np.array([rnd.draw(2) for _ in range(9 * rec.nb)]).reshape(9, rec.nb).T]
    lead = 1j ** q[:, 1:] * np.exp(-2j * np.pi * rx.grid(rec.nb)[:, :C0] / 128)
    return np.concatenate([lead, body(frame, rec, q[:, 0])], 1)


def render(c: np.ndarray, rec: ToneRecord) -> np.ndarray:
    """Cells to real audio, band 0's t = 0 pulse centred on sample ``len(pulse)//2``."""
    p = rec.taps
    hp = len(p) // 2
    n = 2 * hp + max(rec.stagger) + H * (rec.nsym - 1) + 1
    out = np.zeros(n)
    k = np.arange(n)
    for b, fc in enumerate(rec.centres):
        z = np.zeros(n, complex)
        z[rec.stagger[b] + H * np.arange(rec.nsym) + hp] = c[b]
        bb = fftconvolve(z, p)[hp:hp + n]
        out += np.real(bb * np.exp(2j * np.pi * fc * k / FS))
    return out


def synth_frame(frame: bytes, bw: int, level: int, rnd: VB6Rnd) -> np.ndarray:
    """One built frame keyed at ``bw``/``level``, its lead drawn from ``rnd``."""
    rec = ROWS[bw, level]
    x = render(cells(frame, rec, rnd), rec)
    return x * (RMS / np.sqrt(np.mean(x ** 2)))


def synth_burst(payload: bytes, bw: int, level: int, control: int, rnd: VB6Rnd, *,
                caller: str) -> np.ndarray:
    return synth_frame(build_frame(payload, control, ROWS[bw, level], caller=caller),
                       bw, level, rnd)
