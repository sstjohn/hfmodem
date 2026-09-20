# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW500 transmitter — records 3 and 2, the forward law of spec/03 §3.6.5.

Exact inverse of the validated receiver ``kestrel/rx/varahf500.py``. Renders a
payload to transmit audio (section refs are spec/03 §3.6):

    payload(43B) + marker + CRC-16/GENIBUS  =  frame bytes (46B, 368 bits)  §3.6.4
      -> XOR pre-FEC PN whitener                                            §3.6.4
      -> turbo (13,15)-octal encode -> 748 coded bits                       §3.6.4
      -> Stage-1 raster place:  coded[P[n]] -> emit[n]  (P = IDX1[3::17][:748])  §3.6.3
      -> differential-BPSK accumulate per sub-band across columns, frame data at
         emission columns c0..c0+392 (c0 = 9)                               §3.6.1/§3.6.2
      -> per-cell reference cells (+-1)
      -> ⊗ grid480 reference:  ck = A * cell * conj(grid480[idx])           §3.6.2/§3.6.5
      -> up-convert 1350/1650 Hz, H=512, upper subband delayed H/2
         using the measured shared real pulse                              §3.6.1
      -> real 48 kHz audio.

Record 2 (host ``BITRATE (3)``, ``level=ROBUST_LEVEL``) is the gear below it and a
different waveform family — 34 payload bytes over 226 one-hot columns of 1024
samples, four training columns ahead of them — and records 1 and 0 (host levels
2 and 1) are the same family below it. Same coding chain on each record's own
interleaver columns; see "the index ladder" below.

Reuses the RX's fixed tables and turbo codec. ``bw500_pulse`` holds the measured
base-level envelope, independently checked against native stock payloads.
The historical filtered correlation estimate in ``assets/prototype_pulse.npz``
is retained for its other readers, but is not the emitted synthesis pulse.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..coding import turbo as _turbo
from ..coding.crc import crc16_genibus
from ..rx import tablegen
from ..rx import varahf500 as _rx
from .bw500_pulse import PULSE

FS = _rx.FS
H = _rx.H
NSYM = _rx.NSYM
CODED = _rx.CODED
N_SRC = _rx.N_SRC
OFF = _rx.OFF
BASE_LEVEL = _rx.BASE_LEVEL
ROBUST_LEVEL = _rx.ROBUST_LEVEL
C0 = 9                      # frame-start column offset (constant, = RX lock; spec §3.6.2)
OUT_SCALE = 178.0          # output-amplitude scalar [ours]; RX is scale-invariant

# shared fixed tables from the RX
_GRID480 = _rx._GRID480
_CELLS = _rx._CELLS
_P = _rx._P
_PN = _rx._PN
grid480_index = _rx.grid480_index

_CELL_INDEX = {(s, co): n for n, (s, co) in enumerate(_CELLS)}   # (sub,col) -> emit index

# The two subbands share one real envelope and are staggered by half a hop.
# The old low-passed correlation estimate blurred that envelope and placing
# both carriers on the same centers changed the native waveform substantially.
_DSPAN = np.arange(-H // 2, H // 2 + 1)
_PULSE = {s: np.asarray(PULSE, dtype=float) for s in (0, 1)}
_F0 = {0: 1350.0, 1: 1650.0}


# --------------------------------------------------------------------------- #
def build_frame(payload: bytes, marker: int) -> bytes:
    """43 payload bytes + 1 marker byte + 2-byte CRC-16/GENIBUS = 46-byte frame."""
    if len(payload) != _rx.FRAME_PAYLOAD:
        raise ValueError(f"payload must be {_rx.FRAME_PAYLOAD} bytes")
    body = bytes(payload) + bytes([marker & 0xFF])          # 44 bytes protected by CRC
    c = crc16_genibus(body)
    return body + bytes([(c >> 8) & 0xFF, c & 0xFF])         # 46 bytes


def encode_coded(frame_bytes: bytes) -> np.ndarray:
    """46-byte frame -> 748 coded bits (post-whiten, post-turbo)."""
    if len(frame_bytes) != 46:
        raise ValueError("frame must be 46 bytes (368 bits)")
    src_bits = np.unpackbits(np.frombuffer(frame_bytes, np.uint8)).astype(int)  # 368
    inp = src_bits ^ _PN[:N_SRC]                             # apply pre-FEC whitener
    pf = _turbo.encode(inp)                                  # 748 coded bits
    assert pf.size == CODED
    return pf


def build_cells(pf: np.ndarray, ncol: int, ref: int = +1) -> dict:
    """Accumulate the +-1 differential-BPSK reference cells per sub-band.

    Frame column fc (0..392) sits at emission column c0+fc. Data cells flip the
    running product by their coded bit; non-data (pilot) columns and the fc=0
    reference do not flip. Columns before/after the frame are held at the
    reference (clean, de-rotatable filler)."""
    grid = {0: np.full(ncol, ref, dtype=float), 1: np.full(ncol, ref, dtype=float)}
    for s in (0, 1):
        val = float(ref)
        for fc in range(NSYM):          # 0..393
            ec = C0 + fc
            if ec >= ncol:
                break
            if fc >= 1 and (s, fc) in _CELL_INDEX:
                bit = int(pf[_P[_CELL_INDEX[(s, fc)]]])
                val = val * (1.0 if bit == 0 else -1.0)
            grid[s][ec] = val
    return grid


def symbols(pf: np.ndarray, ncol: int | None = None, ref: int = +1) -> dict:
    """Air symbols ck[sub][emission_col] = OUT_SCALE * cell * conj(grid480[idx])."""
    if ncol is None:
        ncol = C0 + NSYM + 8
    grid = build_cells(pf, ncol, ref)
    ck = {}
    for s in (0, 1):
        idx = grid480_index(s, ncol)
        ck[s] = OUT_SCALE * grid[s] * np.conj(_GRID480[idx[:ncol]])
    return ck


def synth_burst(frame_bytes: bytes, onset: int = 0, ncol: int | None = None,
                length: int | None = None, ref: int = +1,
                level: int = BASE_LEVEL) -> np.ndarray:
    """Render one data frame to real 48 kHz audio.

    ``onset`` is the sample index of emission column 0's centre; ``length`` the
    total output length (defaults to just past the last symbol). ``ncol`` and
    ``ref`` belong to the base level's differential grid: an index record's
    emission length is fixed by the record."""
    if level != BASE_LEVEL:
        return synth_l3_burst(frame_bytes, onset, length, level)
    pf = encode_coded(frame_bytes)
    if ncol is None:
        ncol = C0 + NSYM + 8
    ck = symbols(pf, ncol, ref)
    last_center = onset + (ncol - 1) * H
    if length is None:
        length = last_center + H // 2 + _DSPAN[-1] + 1
    air = np.zeros(length)
    for s in (0, 1):
        bb = np.zeros(length, dtype=complex)
        p = _PULSE[s]
        for ec in range(ncol):
            c = onset + ec * H + s * (H // 2)
            lo, hi = c + _DSPAN[0], c + _DSPAN[-1] + 1
            if lo < 0 or hi > length:
                continue
            bb[lo:hi] += ck[s][ec] * p
        nn = np.arange(length)
        air += np.real(bb * np.exp(2j * np.pi * _F0[s] * nn / FS))
    return air


# --------------------------------------------------------------------------- #
# the index ladder (host BITRATE(3), (2), (1)): one-hot index modulation, the exact
# inverse of ``rx.varahf500._l3_align``/``_l3_decode``; ``level`` picks the record
# (``rx.varahf500.INDEX_RECORDS``) and defaults to record 2, the one a stock pair
# keys the over that closes a delivery at.
#
# Scale: a column is a single tone, so this puts the burst at the base level's
# r.m.s. — a gear change may not move the station's drive.
L3_OUT_SCALE = 29.5


def _l3_training_bins(level: int = ROBUST_LEVEL) -> list:
    """The training columns ahead of the frame.

    A STAND-IN, and the one thing in this module that is not the recording's own
    value. VARA draws these from its VB6 Rnd stream — at BW2300 the law is
    ``alloc[1:1+n]`` offset by ``Int(Rnd*16)`` per column — but BW500's stream is
    unlocatable: its base level carries no one-hot preamble to invert, and the
    whole corpus holds two record-2 overs, whose eight bins cannot fix a 24-bit
    stream position and a gap. So the allocation stands with the draw at zero.

    Nothing reads these for correctness. A receiver cannot know the transmitter's
    stream position — the two sides of a link sit at different states — so the
    columns are training: energy in the comb for gain, timing and frequency.
    """
    r = _rx.INDEX_RECORDS[level]
    return [int(b) for b in tablegen.bw500_alloc(r.rec)[1:1 + r.lead]]


def build_l3_frame(payload: bytes, marker: int, level: int = ROBUST_LEVEL) -> bytes:
    """Payload bytes + 1 marker byte + 2-byte CRC-16/GENIBUS = the record's frame."""
    r = _rx.INDEX_RECORDS[level]
    if len(payload) != r.payload:
        raise ValueError(f"payload must be {r.payload} bytes at record {r.rec}")
    body = bytes(payload) + bytes([marker & 0xFF])
    c = crc16_genibus(body)
    return body + bytes([(c >> 8) & 0xFF, c & 0xFF])


def encode_l3_coded(frame_bytes: bytes, level: int = ROBUST_LEVEL) -> np.ndarray:
    """Frame -> on-air bits (post-whiten, post-turbo, post-interleave)."""
    r = _rx.INDEX_RECORDS[level]
    if len(frame_bytes) != r.frame:
        raise ValueError(f"frame must be {r.frame} bytes at record {r.rec}")
    bits = np.unpackbits(np.frombuffer(frame_bytes, np.uint8)).astype(int)
    info = np.zeros(r.n_src, int)
    info[:len(bits)] = bits[:r.n_src]
    info ^= _rx._PN[:r.n_src]
    _, _, _, _, _, place, perm = _rx._l3_tables(level)
    coded = (_turbo.encode_r13(info, perm) if r.rate13
             else _turbo.encode(info, perm=perm))
    return coded[place]


def l3_column_bins(frame_bytes: bytes, level: int = ROBUST_LEVEL) -> np.ndarray:
    """The lit bin of each of the frame's emission columns.

    Reference columns take their class outright; a data column takes the Gray
    image of its three on-air bits, both as a stride on the record's comb."""
    r = _rx.INDEX_RECORDS[level]
    ref, ref_bin, data, _, _, _, _ = _rx._l3_tables(level)
    alloc = tablegen.bw500_alloc(r.rec)
    row = tablegen.clsparm_gray()[(_rx.L3_BPC - 2) * 16:][:2 ** _rx.L3_BPC]
    onair = encode_l3_coded(frame_bytes, level)
    out = np.empty(r.ncols, int)
    out[ref] = ref_bin
    for i, col in enumerate(data):
        v = 0
        for b in onair[i * _rx.L3_BPC:(i + 1) * _rx.L3_BPC]:
            v = (v << 1) | int(b)
        out[col] = ((int(alloc[col]) + r.stride * int(row[v]) - r.first_bin)
                    % r.span) + r.first_bin
    return out


def synth_l3_burst(frame_bytes: bytes | Sequence[bytes], onset: int = 0,
                   length: int | None = None, level: int = ROBUST_LEVEL) -> np.ndarray:
    """Render an index-record burst — one frame, or several back to back — to
    real 48 kHz audio.

    ``onset`` is the first sample of training column 0 — the columns butt up
    against each other with no pulse shaping, so it is a boundary here where the
    base level's ``onset`` is a symbol centre. Later frames follow the first on
    the same grid with no training of their own, as the stock sender keys them
    [rx.varahf500._l3_frames]."""
    dw = _rx.INDEX_RECORDS[level].dw
    frames = [frame_bytes] if isinstance(frame_bytes, bytes) else list(frame_bytes)
    bins = _l3_training_bins(level) + [b for f in frames
                                       for b in l3_column_bins(f, level)]
    if length is None:
        length = onset + len(bins) * dw
    air = np.zeros(length)
    for i, binpos in enumerate(bins):
        sym = np.zeros(dw, complex)
        sym[binpos] = -1j
        lo = onset + i * dw
        if lo < 0 or lo + dw > length:
            continue
        air[lo:lo + dw] = np.fft.ifft(sym).real * dw
    return air * L3_OUT_SCALE
