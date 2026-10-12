# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA's coherent OFDM levels: BW2300 L10-16, BW2750 L11-17, BW500 L7-13.

One law on three carrier sets, closed byte-exact from known plaintext
(docs/protocols/vara/evidence/KPA-bw2300-const.md, KPA-bw500-const.md,
KPA-bw2750-const.md). A symbol is a 1024-point IFFT behind a 128-sample prefix,
1152 samples; cells sit on FFT bins, 46.875 Hz apart.

  PSK rows   4 lead symbols, a +1 reference at symbol 4, 148 data symbols; each
             carrier is time-differential from the reference; gap cells are
             repeats (a +1 step)
  QAM rows   5 lead symbols, then one or two 154-symbol frames; coherent cells
             on +1 pilots at frame rows 8j (even bins) / 8j+1 (odd bins); a gap
             cell is an extra +1 right after one of its carrier's pilot rows
  every row  cells after the lead go on air as ``value * exp(-2j*pi*S/128)``
             (``tablegen.scramble``), column 0 at the reference (PSK) or at each
             frame's first row (QAM); coded cells fill bin-major, symbol order,
             skipping the reference, pilots and gap cells

A frame is payload + control byte + CRC-16/GENIBUS, whitened, turbo coded on
the record's stage-2 permutation, group punctured, and interleaved by its
stage-1 permutation, exactly as the index levels are.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import hilbert

from . import tablegen
from .symbols import SymbolFrame
from ..coding import turbo as _turbo
from ..coding.crc import crc16_genibus

FS = 48000
TU, CP = 1024, 128
SYM = TU + CP
PSK_LEAD, QAM_LEAD, FRAME = 4, 5, 154
PSK_SYMS = PSK_LEAD + 1 + 148
PILOT_ROWS = 20

#: Where the receiver opens its FFT window: 48 samples ahead of the phase-exact
#: origin, the middle of the prefix's 96 clean samples, so a grid estimate off
#: by a few dozen samples either way still reads no neighbouring symbol.
_BACKOFF = 48

_PN = (tablegen.whitener() & 1).astype(int)
_SPECS = {2300: tablegen.BW2300, 2750: tablegen.BW2750, 500: tablegen.BW500}
# per bandwidth: first OFDM level, first bin, carriers
_FAMILIES = {2300: (10, 8, 49), 2750: (11, 3, 59), 500: (7, 27, 11)}
# LUT column by row within a family: QPSK x3, 8PSK x2, 16QAM, 32-cross
_POS = (1, 1, 1, 2, 2, 3, 4)

LEAD, MARK, DATA = 0, 1, 2


@dataclass(frozen=True)
class OfdmRecord:
    bw: int
    level: int
    pos: int
    first_bin: int
    carriers: int

    @property
    def rec(self) -> int:
        """The record in the bandwidth's interleaver chain: host level L is record L - 1."""
        return self.level - 1

    @property
    def n_info(self) -> int:
        return _SPECS[self.bw][self.rec][2]

    @property
    def coded(self) -> int:
        return _SPECS[self.bw][self.rec][1]

    @property
    def frame_bytes(self) -> int:
        return self.n_info // 8

    @property
    def bpc(self) -> int:
        return self.pos + 1

    @property
    def qam(self) -> bool:
        return self.pos >= 3

    @property
    def lead(self) -> int:
        return QAM_LEAD if self.qam else PSK_LEAD

    @property
    def bins(self) -> slice:
        return slice(self.first_bin, self.first_bin + self.carriers)

    @cached_property
    def lut(self) -> np.ndarray:
        return np.array([tablegen._column(self.pos, v) for v in range(2 ** self.bpc)])

    @property
    def gap_cells(self) -> np.ndarray:
        """(carriers, cells per carrier): PSK repeat symbols, or QAM pilot rows j
        whose next row is the extra +1."""
        return tablegen.gap_cells(f"bw{self.bw}", self.rec)

    @cached_property
    def perms(self) -> tuple[np.ndarray, np.ndarray]:
        """(channel interleaver, turbo interleaver): the record's two permutations."""
        seed, coded, n = _SPECS[self.bw][self.rec]
        chan, s = tablegen.permutation(coded, seed)
        return chan, tablegen.permutation(n, s)[0]

    @cached_property
    def layout(self) -> tuple[np.ndarray, np.ndarray]:
        """(kind, fill) over one frame's rows, lead included: kind LEAD/MARK/DATA
        per (symbol, carrier), fill the data cell's coded index, bin-major."""
        b = np.arange(self.carriers)
        odd = (self.first_bin + b) % 2
        if self.qam:
            kind = np.full((QAM_LEAD + FRAME, self.carriers), DATA)
            for j in range(PILOT_ROWS):
                kind[QAM_LEAD + 8 * j + odd, b] = MARK
            for j in self.gap_cells.T:
                kind[QAM_LEAD + 8 * j + odd + 1, b] = MARK
        else:
            kind = np.full((PSK_SYMS, self.carriers), DATA)
            kind[PSK_LEAD] = MARK
            for s in self.gap_cells.T:
                kind[s, b] = MARK
        kind[:self.lead] = LEAD
        fill = np.full(kind.shape, -1)
        data = kind.T == DATA
        fill.T[data] = np.arange(data.sum())
        return kind, fill

    @cached_property
    def scramble(self) -> np.ndarray:
        """One frame's per-cell factor ``exp(-2j*pi*S/128)``; 1 on the lead rows."""
        nsym = len(self.layout[0])
        S = tablegen.scramble()[self.bins, :nsym - self.lead].T
        out = np.ones((nsym, self.carriers), complex)
        out[self.lead:] = np.exp(-2j * np.pi * S / 128)
        return out

    def nsym(self, frames: int = 1) -> int:
        return PSK_SYMS if not self.qam else QAM_LEAD + FRAME * frames


ROWS = {(bw, lv0 + i): OfdmRecord(bw, lv0 + i, pos, fb, n)
        for bw, (lv0, fb, n) in _FAMILIES.items()
        for i, pos in enumerate(_POS)}


def frame_rows(r: OfdmRecord, f: int) -> slice:
    """Burst rows of frame ``f``'s grid, its lead rows included (a second QAM
    frame's "lead" is the tail of the first, which the layout never reads)."""
    return slice(FRAME * f, FRAME * f + len(r.layout[0]))


# --------------------------------------------------------------------------- #
# coding
def check(frame: bytes) -> bool:
    return len(frame) > 2 and crc16_genibus(frame[:-2]).to_bytes(2, "big") == frame[-2:]


def onair(frame: bytes, r: OfdmRecord) -> np.ndarray:
    """Frame bytes -> on-air coded bits: whiten, turbo, group puncture, interleave."""
    if len(frame) != r.frame_bytes:
        raise ValueError(f"frame must be {r.frame_bytes} bytes at BW{r.bw} L{r.level}")
    info = np.zeros(r.n_info, int)
    info[:8 * len(frame)] = np.unpackbits(np.frombuffer(frame, np.uint8))
    chan, turbo = r.perms
    return _turbo.encode_punctured(info ^ _PN[:r.n_info], turbo, r.coded)[chan]


def decode_frame(llr: np.ndarray, r: OfdmRecord, iters: int = 8) -> bytes:
    """On-air bit LLRs (positive for zero) -> frame bytes, CRC unchecked."""
    chan, turbo = r.perms
    Lc = np.empty(r.coded)
    Lc[chan] = np.clip(llr, -_LLR_CAP, _LLR_CAP)
    pn = _PN[:r.n_info]

    def crc_ok(bits) -> bool:
        return check(np.packbits((bits ^ pn).astype(np.uint8)).tobytes()[:r.frame_bytes])

    bits = _turbo.decode_punctured(Lc, r.n_info, turbo, r.coded, iters=iters, check=crc_ok)
    return np.packbits((bits ^ pn).astype(np.uint8)).tobytes()[:r.frame_bytes]


_LLR_CAP = 20.0


# --------------------------------------------------------------------------- #
# the cell grid
_NBINS = 64


def _cell_grid(x: np.ndarray) -> tuple[np.ndarray, float, int]:
    """``x`` on its own symbol grid, mixed down by its carrier offset: rows of FFT
    bins 0..63, one per 1152-sample symbol, phase-exact at the transmitter's
    window; and that offset in Hz.

    Both come from the prefix: every symbol repeats its last 1024 samples' head
    1024 samples later, so the lag-1024 correlation over a prefix-long window,
    folded at the symbol period, peaks 112 samples ahead of the origin (the middle
    of the prefix's 32-sample blend), and its phase there is the offset's turn
    over 1024 samples: unambiguous to half a bin, 23.4 Hz."""
    n = len(x) + 2 * SYM
    z = hilbert(np.concatenate([np.zeros(SYM), np.asarray(x, float), np.zeros(SYM)]),
                next_fast_len(n))[:n]
    c = np.concatenate([[0.0], np.cumsum(z[TU:] * np.conj(z[:-TU]))])
    e = np.concatenate([[0.0], np.cumsum(np.abs(z[:-TU]) ** 2 + np.abs(z[TU:]) ** 2)])
    m = (len(c) - 1 - CP) // SYM * SYM
    corr = (c[CP:CP + m] - c[:m]).reshape(-1, SYM).sum(0)
    energy = (e[CP:CP + m] - e[:m]).reshape(-1, SYM).sum(0)
    peak = int(np.argmax(np.abs(corr) / (energy / 2 + 1e-30)))
    cfo = float(np.angle(corr[peak])) * FS / (2 * np.pi * TU)
    z *= np.exp(-2j * np.pi * cfo * np.arange(n) / FS)
    origin = (peak + CP - 16) % SYM
    at = origin - _BACKOFF + SYM * np.arange((n - origin + _BACKOFF - TU) // SYM + 1)
    at = at[at >= 0]
    spec = np.fft.fft(z[at[:, None] + np.arange(TU)], axis=1)[:, :_NBINS] / 2
    return (spec * np.exp(2j * np.pi * np.arange(_NBINS) * _BACKOFF / TU),
            cfo, int(at[0] + _BACKOFF - SYM))


def cell_grid(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Corrected FFT cells and carrier offset; see :func:`_cell_grid`."""
    cells, cfo, _ = _cell_grid(x)
    return cells, cfo


# --------------------------------------------------------------------------- #
# payload-blind scores
def _psk_steps(cells: np.ndarray, r: OfdmRecord, start: int) -> np.ndarray:
    """Descrambled symbol-to-symbol steps from the reference on: row k is the
    step into symbol ``PSK_LEAD + 1 + k``."""
    d = cells[start + PSK_LEAD:start + PSK_SYMS, r.bins] * np.conj(r.scramble[PSK_LEAD:])
    return d[1:] * np.conj(d[:-1])


def _lattice(z: np.ndarray, m: int) -> tuple[float, complex]:
    """How closely ``z`` sits on an m-PSK lattice, and the lattice's rotation."""
    a = np.abs(z)
    num = np.sum(z ** m / np.maximum(a, 1e-30) ** (m - 1))
    return float(np.abs(num) / (a.sum() + 1e-30)), np.exp(1j * np.angle(num) / m)


def _psk_score(cells: np.ndarray, rows: list[OfdmRecord], start: int) -> float:
    """How nearly the steps into a PSK row's repeat cells read +1, best row.

    The lattice the data steps sit on would do as well on a clean burst, but its
    8th power buries 8PSK in noise some 3 dB before the turbo code gives up."""
    z = _psk_steps(cells, rows[0], start)
    best = 0.0
    for r in rows:
        g = z[r.gap_cells - PSK_LEAD - 1, np.arange(r.carriers)[:, None]]
        best = max(best, float(g.real.sum() / (np.abs(g).sum() + 1e-30)))
    return best


def _pilots(cells: np.ndarray, r: OfdmRecord, start: int, f: int = 0) -> np.ndarray:
    """(PILOT_ROWS, carriers) descrambled pilot cells of frame ``f``."""
    rows = cells[start:][frame_rows(r, f)][:, r.bins] * np.conj(r.scramble)
    j = QAM_LEAD + 8 * np.arange(PILOT_ROWS)[:, None] + (r.first_bin + np.arange(r.carriers)) % 2
    return np.take_along_axis(rows, j, 0)


def _qam_score(cells: np.ndarray, r: OfdmRecord, start: int, f: int = 0) -> float:
    """Pilot coherence from one pilot row to the next on each carrier, which a
    slow drift leaves near 1 and noise near 0."""
    p = _pilots(cells, r, start, f)
    q = p[1:] * np.conj(p[:-1])
    return float(np.abs(q.sum()) / (np.abs(q).sum() + 1e-30))


def _floor(terms: int) -> float:
    # five standard deviations of the score on noise, and never under 0.1
    return max(0.1, 5 / np.sqrt(terms))


@dataclass(frozen=True)
class _Lock:
    qam: bool
    start: int
    frames: int
    score: float


def _pilot_drift(cells: np.ndarray, r: OfdmRecord, lock: _Lock) -> float:
    """The carrier offset left on a QAM burst, Hz, from each carrier's turn between
    its pilots eight symbols apart: unambiguous to 2.6 Hz, which the prefix's
    estimate is well inside."""
    q = sum(np.sum(p[1:] * np.conj(p[:-1]))
            for p in (_pilots(cells, r, lock.start, f) for f in range(lock.frames)))
    return float(np.angle(q)) * FS / (2 * np.pi * 8 * SYM)


def locate(cells: np.ndarray, bw: int) -> _Lock | None:
    """The burst's class and first symbol, payload-blind: at PSK the descrambled
    steps into some PSK row's repeat cells are +1, at QAM the pilot rows are +1
    on every carrier."""
    lv0 = _FAMILIES[bw][0]
    psk = [ROWS[bw, lv] for lv in range(lv0, lv0 + 5)]
    best = None
    for r in (psk[0], ROWS[bw, lv0 + 5]):
        if r.qam:
            terms = (PILOT_ROWS - 1) * r.carriers
            scores = [_qam_score(cells, r, s) for s in range(len(cells) - r.nsym() + 1)]
        else:
            # Re of a random unit step has variance 1/2
            terms = 2 * r.gap_cells.size
            scores = [_psk_score(cells, psk, s) for s in range(len(cells) - r.nsym() + 1)]
        scores = np.array(scores)
        if not len(scores) or scores.max() < _floor(terms) or (best and scores.max() <= best.score):
            continue
        start = int(scores.argmax())
        frames = 1
        if r.qam:
            # a two-frame over reads as a one-frame burst at either frame
            ok = scores >= _floor(terms)
            if start >= FRAME and ok[start - FRAME]:
                start -= FRAME
            frames = 2 if start + FRAME < len(ok) and ok[start + FRAME] else 1
        best = _Lock(r.qam, start, frames, float(scores.max()))
    return best


# --------------------------------------------------------------------------- #
# demodulation
def _gains(d: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Per-carrier complex gain at every row, interpolated in time between the +1
    cells and held beyond the first and last."""
    t = np.arange(len(d))
    out = np.empty(d.shape, complex)
    for b in range(d.shape[1]):
        k = np.flatnonzero(kind[:, b] == MARK)
        out[:, b] = np.interp(t, k, d[k, b].real) + 1j * np.interp(t, k, d[k, b].imag)
    return out


def _points(cells: np.ndarray, r: OfdmRecord, start: int, f: int) -> tuple[np.ndarray, float]:
    """(the frame's data cells as constellation points in fill order, how well
    the record's own gap cells read +1)."""
    kind, fill = r.layout
    d = cells[start:][frame_rows(r, f)][:, r.bins] * np.conj(r.scramble)
    if r.qam:
        # Silence has no gain: its cells read nan and never confirm.
        with np.errstate(divide="ignore", invalid="ignore"):
            z = d / _gains(d, kind)
        gap = kind == MARK
        gap[QAM_LEAD + 8 * np.arange(PILOT_ROWS)[:, None]
            + (r.first_bin + np.arange(r.carriers)) % 2, np.arange(r.carriers)] = False
        ref = z[gap]
        confirm = float(np.sum(ref.real) / (np.abs(ref).sum() + 1e-30))
    else:
        step = np.zeros(d.shape, complex)
        step[PSK_LEAD + 1:] = d[PSK_LEAD + 1:] * np.conj(d[PSK_LEAD:-1])
        body = step[PSK_LEAD + 1:]
        _, rot = _lattice(body, 2 ** r.bpc)
        power = np.mean(np.abs(d[PSK_LEAD:]) ** 2, axis=0)
        z = step * np.conj(rot) / (power + 1e-30)
        ref = z[kind == MARK][r.carriers:]
        confirm = float(np.sum(ref.real) / (np.abs(ref).sum() + 1e-30))
    out = np.empty(r.coded // r.bpc, complex)
    out[fill[kind == DATA]] = z[kind == DATA]
    return out, confirm


def _llr(z: np.ndarray, r: OfdmRecord) -> tuple[np.ndarray, np.ndarray, float]:
    """Max-log per-bit LLRs (positive for zero), MSB first; the hard labels; and
    the points' SNR in dB."""
    dist = np.abs(z[:, None] - r.lut[None]) ** 2
    near = dist.min(1)
    sig2 = max(float(near.mean()), 1e-6)
    labels = np.arange(2 ** r.bpc)
    llr = np.stack([(np.where((labels >> (r.bpc - 1 - i)) & 1, dist, np.inf).min(1)
                     - np.where((labels >> (r.bpc - 1 - i)) & 1, np.inf, dist).min(1))
                    for i in range(r.bpc)], 1) / sig2
    hard = dist.argmin(1)
    snr = 10 * np.log10(np.mean(np.abs(r.lut[hard]) ** 2) / sig2)
    # a differential step carries the noise of two cells
    return llr.ravel(), hard, snr + (0.0 if r.qam else 3.01)


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quality:
    snr_db: float              # the demodulated points' SNR, per cell
    raw_errors: int | None     # channel bit errors before FEC, None without a clean frame
    raw_bits: int
    offset_hz: float           # the carrier's offset from our grid


@dataclass(frozen=True)
class OfdmResult:
    level: int | None
    frames: tuple[bytes, ...]  # CRC-clean frames in burst order: payload | control | CRC
    quality: Quality

    @property
    def ok(self) -> bool:
        return bool(self.frames)

    @property
    def payload(self) -> bytes:
        return b"".join(f[:-3] for f in self.frames)

    @property
    def control(self) -> int | None:
        return self.frames[-1][-3] if self.frames else None



def _observe(pts, r, result, verdicts, lock, origin, emit):
    kind, fill = r.layout
    rows, carriers = np.nonzero(kind == DATA)
    order = np.argsort(fill[kind == DATA])
    rows, carriers = rows[order], carriers[order]
    for f, (z, _) in enumerate(pts):
        ok, snr = verdicts[f]
        base = origin + (lock.start + FRAME * f) * SYM
        emit(SymbolFrame(r.level, f, ok, base + r.lead * SYM,
                         base + len(kind) * SYM,
                         base + rows * SYM, (carriers + r.first_bin) * FS / TU,
                         z.copy(), r.lut.copy(),
                         'equalized QAM' if r.qam else 'differential PSK',
                         snr, result.quality.offset_hz))


def _read(pts: list, r: OfdmRecord, cfo: float, verdicts=None) -> OfdmResult:
    frames, snrs, errors, bits = [], [], 0, 0
    for z, _ in pts:
        llr, hard, snr = _llr(z, r)
        frame = decode_frame(llr, r)
        snrs.append(snr)
        bits += r.coded
        valid = check(frame)
        if verdicts is not None:
            verdicts.append((valid, snr))
        if valid:
            frames.append(frame)
            got = ((hard[:, None] >> np.arange(r.bpc - 1, -1, -1)) & 1).ravel()
            errors += int((got != onair(frame, r)).sum())
    return OfdmResult(r.level, tuple(frames),
                      Quality(float(np.mean(snrs)), errors if frames else None, bits, cfo))


def decode_burst(audio: np.ndarray, bw: int, level: int | None = None, *,
                 on_symbols: Callable[[SymbolFrame], None] | None = None) -> OfdmResult:
    """The OFDM burst in ``audio`` at ``bw``.

    The class (PSK or QAM, one or two frames) and the burst's first symbol are
    found payload-blind; each row of the class is then confirmed on its own gap
    cells before it costs a turbo pass, ``level`` first when it names one, and
    the first row whose frames come out CRC-clean is the read."""
    cells, cfo, origin = _cell_grid(audio)
    lock = locate(cells, bw)
    if lock is None:
        return OfdmResult(level, (), Quality(float("nan"), None, 0, cfo))
    lv0 = _FAMILIES[bw][0]
    if lock.qam:
        # The pilot interpolation reads a residual turn as fading; a common phase
        # per symbol takes it out exactly.
        df = _pilot_drift(cells, ROWS[bw, lv0 + 5], lock)
        cells = cells * np.exp(-2j * np.pi * df * SYM / FS * np.arange(len(cells)))[:, None]
        cfo += df
    trials = []
    for lv in range(lv0, lv0 + len(_POS)):
        r = ROWS[bw, lv]
        if r.qam != lock.qam:
            continue
        pts = [_points(cells, r, lock.start, f) for f in range(lock.frames)]
        confirm = min(c for _, c in pts)
        if confirm >= 0.5:
            trials.append((lv != level, -confirm, lv, r, pts))
    first = None
    first_points = None
    for *_, r, pts in sorted(trials, key=lambda t: t[:3]):
        verdicts = [] if on_symbols is not None else None
        res = _read(pts, r, cfo, verdicts)
        if res.ok:
            if on_symbols is not None:
                _observe(pts, r, res, verdicts, lock, origin, on_symbols)
            return res
        if first is None:
            first, first_points = res, (pts, r, verdicts)
    if on_symbols is not None and first is not None:
        _observe(first_points[0], first_points[1], first, first_points[2],
                 lock, origin, on_symbols)
    return first or OfdmResult(level, (), Quality(float("nan"), None, 0, cfo))
