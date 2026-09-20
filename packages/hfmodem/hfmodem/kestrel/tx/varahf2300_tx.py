# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2300 wideband OFDM DATA transmitter — base + high-throughput speed levels.

Exact forward (inverse) of ``kestrel/rx/varahf2300.py``; built from the same spec
sections (spec/01 §2300, spec/03 §3.5.3-3.5.4, spec/06 §6.1c-6.1d). Mirrors the
BW500 TX (``kestrel/tx/varahf500_tx.py``): it reuses the RX's fixed tables, ladder
and turbo codec and adds only the forward synthesis.

Chain (section refs in the RX module):

    payload (+marker for base) + CRC-16/GENIBUS   =  frame bytes
      -> serialize MSB-first, zero-pad to N info bits
      -> XOR PN whitener (BW2300 PN table; spec/03 §3.5.3)
      -> turbo (13,15) encode: rate-1/2 (base, il2 col) | rate-1/2/punctured (high)
      -> channel interleave (BW2300 il1 col):  onair[k] = coded[pi[k]]
      -> value law:
           index         : bpc on-air bits -> Gray -> one lit bin (−1j) in the
                           record's FFT, one OFDM symbol per column, no CP.
           high  (const) : bpc on-air bits -> Gray -> LUT[pos+6*coded] cell, packed
                           span/symbol onto a 1024-IFFT + 128-sample CP.
      -> real 48 kHz audio.

Every index-law over is prefixed by the training preamble (spec/01 §2300 "gDC
REFERENCE PREAMBLE"): 2 silent PTT lead-in blocks, then one-hot ``−1j`` tones whose
bins come from VARA's VB6 ``Rnd`` stream, then data column 1. It carries no payload;
emitting it makes the burst time-domain identical to VARA's. Its length is the
record's own symbol count — 12 / 6 / 4 / 2 at records 3 / 2 / 1 / 0 — which is a
fixed 6144 samples only at the two records where 6144 happens to divide evenly;
below them it is 4096. It has no table of its own: symbol ``s``, counted from 1,
takes the record's own column-allocation table at index ``s`` and offsets it by a
draw exactly as a data column offsets it by a Gray value.

No twiddle on the DATA path (spec/01 §2300 2026-07-20). The exact VARA carrier
raster (alloc/map) is a spec gap for the high levels; kestrel uses a self-consistent
sequential carrier assignment (identical TX/RX), which round-trips byte-exact.
"""
from __future__ import annotations

import numpy as np

from ..coding import turbo as _turbo
from ..coding.crc import crc16_genibus
from ..rx import varahf2300 as _rx

FS = _rx.FS
RECORDS = _rx.RECORDS
LEVELS = _rx.LEVELS
BASE_LEVEL = _rx.BASE_LEVEL
payload_bytes = _rx.payload_bytes

OUT_SCALE = 1.0            # RX is scale-invariant (base) / self-consistent (high)

# Training preamble (spec/01 §2300): its fixed length in samples, and the VB6 Rnd
# state VARA reaches by the first DATA over of an ARQ session.
PREAMBLE_SAMPLES = 6144
PREAMBLE_SEED = 4107106


# --------------------------------------------------------------------------- #
class VB6Rnd:
    """VB6 ``Rnd``: 24-bit LCG, advance-then-read (spec/01 §2300).

    VARA never calls ``Randomize``, so the preamble stream is the same on every run;
    the seed is a session state, not a per-payload one."""

    def __init__(self, state: int = PREAMBLE_SEED):
        self.state = state

    def draw(self, bits: int = 4) -> int:
        """``Int(Rnd()*2**bits)`` from the advanced 24-bit state."""
        self.state = (self.state * 0x43FD43FD + 0xC39EC3) % (1 << 24)
        return self.state >> (24 - bits)

    def draw16(self) -> int:
        return self.draw(4)


def preamble_alloc(level: int = BASE_LEVEL) -> np.ndarray:
    """The record's own column allocations, one per training symbol.

    The preamble has no allocation table of its own: it walks the record's column
    table from index 1, the way VB6 walks a zero-based array from its first
    element. Record 3's 12 entries are the base level's published preamble table,
    and record 2's 6 fall out of the same slice of its own column table.
    """
    r = RECORDS[level]
    return _rx._BASE_TABLES[level][0][1:1 + r.lead - _rx._BASE_LEADIN]


def preamble_bins(rnd: VB6Rnd, level: int = BASE_LEVEL) -> list:
    """The preamble bins for one over, consuming one draw of ``rnd`` per symbol:
    ``bin = ((alloc + stride·Int(Rnd()*16) − first_bin) % span) + first_bin`` — a
    data column's own law with the draw standing in for the Gray value."""
    r = RECORDS[level]
    # Native BW2750 L2 training uses Int(Rnd()*8); L3/L4 use *16.
    bits = 3 if level in (100, 101) else 4
    return [((int(a) + r.stride * rnd.draw(bits) - r.first_bin) % r.span) + r.first_bin
            for a in preamble_alloc(level)]


#: Draws the stream has taken before a session's first over, by the base record
#: it is keyed at. Five at BW2750 and none at BW2300, off the 2026-07-21
#: loopback: the same modem keyed a BW2750 link-setup at offset 5 and, in a
#: second session 50 s later, a BW2300 one at offset 0, so the lead follows the
#: bandwidth and not the process's history.
STREAM_LEAD = {100: 5, 101: 5, 102: 5, _rx.BASE_LEVELS["2750"]: 5}


def over_preamble_bins(over: int, seed: int = PREAMBLE_SEED,
                       level: int = BASE_LEVEL) -> list:
    """Preamble bins for DATA over ``over``, drawn from one continuous Rnd stream:
    one draw per symbol, overs back to back, plus one extra draw between overs 0
    and 1, behind the bandwidth's own lead  [see STREAM_LEAD]."""
    rnd = VB6Rnd(seed)
    for _ in range(STREAM_LEAD.get(level, 0)):
        rnd.draw16()
    for o in range(over + 1):
        if o == 1:
            rnd.draw16()
        # Fixed lower-level DATA follows a base-record link setup. Consume
        # its twelve training draws before the selected DATA stream begins.
        prior_level = 103 if o == 0 and over > 0 and level in (100, 101, 102) else level
        bins = preamble_bins(rnd, prior_level)
    return bins


def synth_preamble(bins, level: int = BASE_LEVEL) -> np.ndarray:
    """Silent lead-in + the one-hot ``−1j`` training symbols -> audio."""
    r = RECORDS[level]
    out = np.zeros((_rx._BASE_LEADIN + len(bins)) * r.dw50)
    for i, binpos in enumerate(bins):
        sym = np.zeros(r.dw50, complex)
        sym[binpos] = -1j
        blk = _rx._BASE_LEADIN + i
        out[blk * r.dw50:(blk + 1) * r.dw50] = np.fft.ifft(sym).real * r.dw50
    return out * OUT_SCALE


# --------------------------------------------------------------------------- #
def build_frame(payload: bytes, level: int = BASE_LEVEL) -> bytes:
    """payload + CRC-16/GENIBUS -> frame bytes (exactly frame_bytes long)."""
    r = RECORDS[level]
    plen = payload_bytes(level)
    if len(payload) != plen:
        raise ValueError(f"payload must be {plen} bytes at level {level}")
    c = crc16_genibus(payload)
    frame = bytes(payload) + bytes([(c >> 8) & 0xFF, c & 0xFF])
    assert len(frame) == r.frame_bytes
    return frame


def frame_to_onair(frame_bytes: bytes, level: int) -> np.ndarray:
    """frame bytes -> on-air coded-bit stream (post whiten/turbo/channel-interleave)."""
    r = RECORDS[level]
    if len(frame_bytes) != r.frame_bytes:
        raise ValueError(f"frame must be {r.frame_bytes} bytes at level {level}")
    bits = np.unpackbits(np.frombuffer(frame_bytes, np.uint8)).astype(int)
    info = np.zeros(r.n_info, int)
    info[:len(bits)] = bits[:r.n_info]          # MSB-first, zero-pad tail
    info = info ^ _rx._PN[:r.n_info]
    perm = _rx.turbo_perm(level)
    if r.coded == 2 * r.n_info + 12:
        coded = _turbo.encode(info, perm=perm)
    elif r.coded == 3 * r.n_info + 12:
        coded = _turbo.encode_r13(info, perm)
    else:
        coded = _turbo.encode_punctured(info, perm, r.coded)
    pi = _rx.chan_perm(level)
    return coded[pi]


# --------------------------------------------------------------------------- #
def _synth_index(onair: np.ndarray, level: int) -> np.ndarray:
    """One-hot index modulation (spec/01 §2300 promoted law): walk the record's
    emission columns; each column = one lit bin (−1j) in a dw50-FFT. Data columns
    (map1==0) place `bin = ((alloc + stride·gray − first_bin) % span) + first_bin`
    from bpc on-air bits; reference columns (map1!=0) place their class (map2) bin and
    consume no bits. Byte-matches VARA's on-air layout (interop-validated)."""
    r = RECORDS[level]
    row = _rx._CLSPARM[r.bpc]
    alloc, map1, map2 = _rx._BASE_TABLES[level]
    out = np.empty(r.ncols * r.dw50)
    bi = 0
    for col in range(r.ncols):
        if map1[col] != 0:                     # reference column
            gray = int(map2[col])
        else:
            v = 0
            for b in onair[bi:bi + r.bpc]:
                v = (v << 1) | int(b)
            bi += r.bpc
            gray = row[v]
        binpos = ((int(alloc[col]) + r.stride * gray - r.first_bin) % r.span) + r.first_bin
        sym = np.zeros(r.dw50, complex)
        sym[binpos] = -1j
        out[col * r.dw50:(col + 1) * r.dw50] = np.fft.ifft(sym).real * r.dw50
    return out * OUT_SCALE


def _synth_const(onair: np.ndarray, level: int) -> np.ndarray:
    """High-throughput dense constellation: span cells / (1024-IFFT + 128 CP) symbol."""
    r = _rx.RECORDS[level]
    pts = _rx.const_points(level)          # 2**bpc cells, already Gray-ordered
    fft = r.dw50 - r.cp
    ncell = (r.coded + r.bpc - 1) // r.bpc
    # pack on-air bits (zero-padded to a whole cell) -> constellation cells
    padded = np.zeros(ncell * r.bpc, int)
    padded[:r.coded] = onair[:r.coded]
    cells = np.empty(ncell, complex)
    for i in range(ncell):
        v = 0
        for b in padded[i * r.bpc:i * r.bpc + r.bpc]:
            v = (v << 1) | int(b)
        cells[i] = pts[v]
    nsym = (ncell + r.span - 1) // r.span
    out = np.empty(nsym * r.dw50)
    for s in range(nsym):
        sp = np.zeros(fft, complex)
        chunk = cells[s * r.span:(s + 1) * r.span]
        sp[r.first_bin:r.first_bin + len(chunk)] = chunk
        t = np.fft.ifft(sp).real                # real folding (undone by ×2 in RX)
        out[s * r.dw50:s * r.dw50 + r.cp] = t[-r.cp:]     # cyclic prefix
        out[s * r.dw50 + r.cp:(s + 1) * r.dw50] = t
    return out * OUT_SCALE


def synth_frame(frame_bytes: bytes, level: int = BASE_LEVEL,
                over: int | None = 0) -> np.ndarray:
    """Render an already-built frame (payload+marker+CRC) to audio.

    ``over`` is the frame's position in the session's preamble stream; ``None``
    emits the data columns alone (the preamble is training only), which is what a
    record carrying no preamble (``lead`` 0) emits either way."""
    onair = frame_to_onair(frame_bytes, level)
    r = RECORDS[level]
    if r.law != "index":
        return _synth_const(onair, level)
    data = _synth_index(onair, level)
    if over is None or not r.lead:
        return data
    return np.concatenate([synth_preamble(over_preamble_bins(over, level=level), level), data])


def synth_burst(payload: bytes, level: int = BASE_LEVEL,
                over: int | None = 0) -> np.ndarray:
    """Render one BW2300 data frame at ``level`` to real 48 kHz transmit audio."""
    return synth_frame(build_frame(payload, level), level, over)
