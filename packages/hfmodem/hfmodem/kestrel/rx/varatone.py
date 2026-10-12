# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tone family: VARA HF levels 5-9 at BW2300, 5-10 at BW2750, 5-6 at BW500.

Closed from known plaintext on high speed capable overs (docs/protocols/vara/evidence/
KPA-bw2300-subband.md, KPA-tone-bw2750-bw500.md). Per band b of a row:

    t 0..7    lead (session training; no receiver reads it)
    t 8       reference, cell 1
    t 9..     data and pilot columns; cell(t) = cell(t-1) * i^Gray(2 bits), a pilot
              column repeats cell(t-1)
    on air    cell(t) * conj(grid_row(b)[t-8]), one pulse per 512 samples at the
              band's centre, offset by its stagger

The coded bits fill band 0's data columns in time order, then band 1's, and so on.
The coding chain is the index family's at rate 1/2: frame -> whiten -> turbo with
the record's il2 column -> il1 gather. A frame is payload | control | CRC-16/GENIBUS.

Everything in a row is the record's own: pilots (``tablegen.tone_pilots``), the
data column count (``coded / 2 / bands``) and so the symbol count. What is written
here per row is the band spacing, the stagger and the pulse.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache, cached_property, lru_cache

import numpy as np
from scipy import fft as _fft

from . import tablegen
from .symbols import SymbolFrame
from ..coding import turbo
from ..coding.crc import crc16_genibus
from ..tx.bw500_pulse import PULSE as _P500
from ..tx.tone_pulse import P8, P9

FS = 48000
H = 512
C0 = 8                        # the reference symbol; 0..7 are the lead
PULSES = {"P500": np.asarray(_P500), "P8": np.asarray(P8), "P9": np.asarray(P9)}
GRAY = np.array([0, 1, 3, 2])
#: Increment per on-air bit pair, MSB first: LUT column 1, the QPSK Gray column.
POINTS = tablegen.constellation_lut()[1::6][:4]
_SPEC = {2300: "bw2300", 2750: "bw2750", 500: "bw500"}
_PN = (tablegen.whitener() & 1).astype(int)
_LLR_CAP = 20.0


@dataclass(frozen=True)
class ToneRecord:
    bw: int
    level: int
    step: float                # band spacing, Hz, centred on 1500
    stagger: tuple[int, ...]   # per band, samples
    pulse: str

    @property
    def rec(self) -> int:
        return self.level - 1

    @property
    def nb(self) -> int:
        return len(self.stagger)

    @property
    def centres(self) -> tuple[float, ...]:
        return tuple(1500 + self.step * (b - (self.nb - 1) / 2) for b in range(self.nb))

    @property
    def n_info(self) -> int:
        return getattr(tablegen, _SPEC[self.bw].upper())[self.rec][2]

    @property
    def coded(self) -> int:
        return getattr(tablegen, _SPEC[self.bw].upper())[self.rec][1]

    @property
    def frame_bytes(self) -> int:
        return self.n_info // 8

    @property
    def payload_bytes(self) -> int:
        return self.frame_bytes - 3

    @property
    def ndata(self) -> int:
        return self.coded // 2 // self.nb

    @property
    def pilots(self) -> tuple[tuple[int, ...], ...]:
        """Per band, the pilot symbols t."""
        return tuple(tuple(C0 + co for co in band)
                     for band in tablegen.tone_pilots(_SPEC[self.bw], self.rec))

    @property
    def nsym(self) -> int:
        return C0 + 1 + self.ndata + len(self.pilots[0])

    @property
    def taps(self) -> np.ndarray:
        return PULSES[self.pulse]


_S3 = (0, 176, 336)


def _alt(nb: int) -> tuple[int, ...]:
    return tuple(256 * (b % 2) for b in range(nb))


ROWS = {(r.bw, r.level): r for r in (
    ToneRecord(2300, 5, 600, _S3, "P500"),
    ToneRecord(2300, 6, 500, _alt(4), "P500"),
    ToneRecord(2300, 7, 320, _alt(6), "P8"),
    ToneRecord(2300, 8, 228, _alt(8), "P8"),
    ToneRecord(2300, 9, 210, _alt(10), "P9"),
    ToneRecord(2750, 5, 900, _S3, "P500"),
    ToneRecord(2750, 6, 660, _alt(4), "P500"),
    ToneRecord(2750, 7, 420, _alt(6), "P8"),
    ToneRecord(2750, 8, 320, _alt(8), "P8"),
    ToneRecord(2750, 9, 250, _alt(10), "P9"),
    ToneRecord(2750, 10, 210, _alt(13), "P9"),
    ToneRecord(500, 5, 300, (0, 256), "P500"),
    ToneRecord(500, 6, 180, _S3, "P8"),
)}


@cache
def grid(nb: int) -> np.ndarray:
    """Grid rows 0..nb-1 as lattice integers by column from the reference, -1 where
    unmeasured."""
    rows = [tablegen.grid_row(b) for b in range(nb)]
    out = np.full((nb, max(map(len, rows))), -1, np.int64)
    for b, r in enumerate(rows):
        out[b, :len(r)] = r
    out.flags.writeable = False
    return out


@cache
def _perms(rec: ToneRecord) -> tuple[np.ndarray, np.ndarray]:
    seed, n1, n2 = getattr(tablegen, _SPEC[rec.bw].upper())[rec.rec]
    p1, s = tablegen.permutation(n1, seed)
    p2, _ = tablegen.permutation(n2, s)
    return p1, p2


@cache
def data_symbols(rec: ToneRecord) -> np.ndarray:
    """(bands, ndata) symbol index of each data cell, in fill order."""
    return np.array([[t for t in range(C0 + 1, rec.nsym) if t not in ps] for ps in rec.pilots])


def onair_bits(frame: bytes, rec: ToneRecord) -> np.ndarray:
    p1, p2 = _perms(rec)
    bits = np.unpackbits(np.frombuffer(frame, np.uint8)).astype(int)
    info = np.zeros(rec.n_info, int)
    info[:len(bits)] = bits[:rec.n_info]
    return turbo.encode(info ^ _PN[:rec.n_info], perm=p2)[p1]


def crc_ok(frame: bytes) -> bool:
    c = crc16_genibus(frame[:-2])
    return frame[-2:] == bytes((c >> 8, c & 0xFF))


# --------------------------------------------------------------------------- #
_TAPS = max(len(p) for p in PULSES.values())


@lru_cache(maxsize=8)
def _pulse_spectrum(pulse: str, size: int) -> np.ndarray:
    return _fft.fft(PULSES[pulse][::-1], size)


class Audio:
    """Audio held as its transform, at a length on which every whole hertz is a
    bin: a band is mixed down by rolling the spectrum, so every band and trial of
    a burst shares one forward transform, and each fractional carrier offset
    costs one more."""

    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, float)
        self.size = FS * _fft.next_fast_len(-(-(len(self.x) + _TAPS - 1) // FS))
        self._spectra: dict[float, np.ndarray] = {}

    def _spectrum(self, frac: float) -> np.ndarray:
        if frac not in self._spectra:
            x = self.x
            if frac:
                x = x * np.exp(-2j * np.pi * frac * np.arange(len(x)) / FS)
            self._spectra[frac] = _fft.fft(x, self.size)
        return self._spectra[frac]

    def matched(self, centre: float, df: float, pulse: str) -> np.ndarray:
        """The audio mixed down at ``centre + df`` and correlated with the pulse:
        sample ``n`` reads a pulse centred on ``n``. The adjoint of the render."""
        whole = round(df)
        spec = self._spectrum(df - whole)
        shift = (round(centre) + whole) * (self.size // FS)
        y = _fft.ifft(np.roll(spec, -shift) * _pulse_spectrum(pulse, self.size))
        start = (len(PULSES[pulse]) - 1) // 2
        return y[start:start + len(self.x)]


def _collapse(v: np.ndarray, known: np.ndarray) -> np.ndarray:
    """|E (d/|d|)^4| over the last axis: ~1 where the increments sit on the QPSK lattice."""
    d = v[..., 1:] * np.conj(v[..., :-1])
    w = (d / np.maximum(np.abs(d), 1e-30)) ** 4
    ok = known[1:] & known[:-1]
    return np.abs(w[..., ok].mean(-1))


def _derotate(cells: np.ndarray, rec: ToneRecord) -> tuple[np.ndarray, np.ndarray]:
    """Cells from t = C0 on, de-rotated by each band's grid row, and where the row is known."""
    g = grid(rec.nb)[:, :rec.nsym - C0]
    return cells[:, C0:] * np.exp(2j * np.pi * g / 128), g >= 0


def lock(y0: np.ndarray, rec: ToneRecord) -> tuple[int, float]:
    """Band 0's t = 0 in the matched-filter output, payload-blind: the energy phase
    within a symbol, then the symbol start whose grid-row-0 de-rotated increments
    collapse hardest onto the QPSK lattice, then the energy peak within +-48."""
    e = np.abs(y0) ** 2
    n = len(e) // H * H
    p0 = int(e[:n].reshape(-1, H).mean(0).argmax())
    v = y0[p0::H]
    if len(v) < rec.nsym:
        return p0, 0.0
    g = grid(1)[0, :rec.nsym - C0]
    win = np.lib.stride_tricks.sliding_window_view(v, rec.nsym)[:, C0:] * np.exp(2j * np.pi * g / 128)
    q = _collapse(win, g >= 0)
    j = int(q.argmax())
    c = p0 + H * j
    t = H * np.arange(rec.nsym)
    lo = max(-48, -c)
    hi = min(48, len(e) - 1 - c - t[-1])
    c += lo + int(np.argmax([e[c + d + t].sum() for d in range(lo, hi + 1)]))
    return c, float(q[j])


def sample(y: np.ndarray, rec: ToneRecord, n0: int) -> np.ndarray:
    """Cells (bands, nsym) off the matched-filter output, band 0's t = 0 at ``n0``."""
    k = n0 + np.array(rec.stagger)[:, None] + H * np.arange(rec.nsym)[None, :]
    ok = (k >= 0) & (k < y.shape[1])
    out = np.zeros(k.shape, complex)
    out[ok] = np.take_along_axis(y, np.clip(k, 0, y.shape[1] - 1), 1)[ok]
    return out


def turn(cells: np.ndarray, rec: ToneRecord) -> float:
    """The carrier offset's turn per symbol, radians, payload-blind: coarse off the
    pilot columns, whose increment is 1 (unambiguous to 46.9 Hz), and fine off every
    increment's QPSK lattice, whose quarter turn (11.7 Hz) the coarse one resolves."""
    v, known = _derotate(cells, rec)
    d = v[:, 1:] * np.conj(v[:, :-1])
    ok = known[:, 1:] & known[:, :-1]
    pilot = np.zeros_like(ok)
    for b, ts in enumerate(rec.pilots):
        pilot[b, np.array(ts) - C0 - 1] = True
    coarse = np.angle(d[ok & pilot].sum())
    fine = np.angle(np.sum((d[ok] / np.maximum(np.abs(d[ok]), 1e-30)) ** 4)) / 4
    return float(fine + np.pi / 2 * np.round((coarse - fine) / (np.pi / 2)))


def band_scores(cells: np.ndarray, rec: ToneRecord) -> np.ndarray:
    v, known = _derotate(cells, rec)
    return np.array([_collapse(v[b], known[b]) for b in range(rec.nb)])


def soft_bits(cells: np.ndarray, rec: ToneRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(llr, d, read)``: max-log LLRs in fill order, the unit-scaled increments, and
    which increments were readable (both cells on a known grid column)."""
    v, known = _derotate(cells, rec)
    ts = data_symbols(rec) - C0
    rows = np.arange(rec.nb)[:, None]
    read = known[rows, ts] & known[rows, ts - 1]
    scale = np.array([np.median(np.abs(v[b][known[b]])) for b in range(rec.nb)])[:, None]
    v = v / np.maximum(scale, 1e-30)
    d = (v[rows, ts] * np.conj(v[rows, ts - 1])).ravel()
    read = read.ravel()
    m = -np.abs(d[:, None] - POINTS[None, :]) ** 2
    m /= np.mean(-m[read].max(1)) + 1e-3
    llr = np.empty((len(d), 2))
    for j in range(2):
        bit = (np.arange(4) >> (1 - j)) & 1
        llr[:, j] = m[:, bit == 0].max(1) - m[:, bit == 1].max(1)
    llr[~read] = 0
    return np.clip(llr.ravel(), -_LLR_CAP, _LLR_CAP), d, read


def decode_frame(llr: np.ndarray, rec: ToneRecord) -> bytes:
    p1, p2 = _perms(rec)
    lc = np.zeros(rec.coded)
    lc[p1] = llr

    def frame(bits):
        return np.packbits((bits ^ _PN[:rec.n_info]).astype(np.uint8)).tobytes()[:rec.frame_bytes]

    return frame(turbo.decode_from_coded_llr(lc, rec.n_info, iters=12, perm=p2,
                                             check=lambda b: crc_ok(frame(b))))


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quality:
    snr_db: float              # cell SNR off the increments' distance to the lattice
    raw_errors: int | None     # channel bit errors before FEC, None without a clean frame
    raw_bits: int
    offset_hz: float           # the carrier's offset from our grid


@dataclass(frozen=True)
class ToneResult:
    level: int
    frames: tuple[bytes, ...]  # CRC-clean frames: payload | control | CRC
    quality: Quality
    lock: float                # band 0's lattice collapse, ~1 on a tone burst of this row
    onset: int                 # sample of band 0's t = 0 pulse centre

    @property
    def ok(self) -> bool:
        return bool(self.frames)

    @property
    def payload(self) -> bytes:
        return b"".join(f[:-3] for f in self.frames)

    @property
    def control(self) -> int | None:
        return self.frames[-1][-3] if self.frames else None


@dataclass
class _Read:
    x: Audio
    rec: ToneRecord
    n0: int
    lock: float
    cells: np.ndarray
    df: float

    @cached_property
    def score(self) -> float:
        return float(band_scores(self.cells, self.rec).min())


#: Band 0 locks at whichever of these offsets reads best: within 6 Hz of the
#: carrier a 2049-tap pulse loses under 0.1 dB, where 20 Hz off costs it 0.45 dB
#: and the neighbours' leakage, enough to lose the lock at the code's edge.
_TRIALS = (-12.0, 0.0, 12.0)


def _read(x: Audio, rec: ToneRecord, df: float | None = None) -> _Read:
    if df is None:
        tried = [(x.matched(rec.centres[0], d, rec.pulse), d) for d in _TRIALS]
        locks = [lock(y, rec) for y, _ in tried]
        best = int(np.argmax([q for _, q in locks]))
        (y0, df), (n0, q) = tried[best], locks[best]
    else:
        y0 = x.matched(rec.centres[0], df, rec.pulse)
        n0, q = lock(y0, rec)
    y = np.stack([y0] + [x.matched(c, df, rec.pulse) for c in rec.centres[1:]])
    return _Read(x, rec, n0, q, sample(y, rec, n0), df)


def _tuned(r: _Read) -> _Read:
    """``r`` with its carrier offset taken out: mixed down again when the offset
    would cost the matched filter (a 2049-tap pulse turns 0.27 rad end to end at
    1 Hz), then with what turn is left taken off each symbol."""
    rec = r.rec
    df = turn(r.cells, rec) * FS / (2 * np.pi * H)
    if abs(df) > 1:
        r = _read(r.x, rec, r.df + df)
    left = turn(r.cells, rec)
    return replace(r, cells=r.cells * np.exp(-1j * left * np.arange(rec.nsym)),
                   df=r.df + left * FS / (2 * np.pi * H))


def _decode(r: _Read, on_symbols=None) -> ToneResult:
    r = _tuned(r)
    rec = r.rec
    llr, d, read = soft_bits(r.cells, rec)
    hard = np.abs(d[:, None] - POINTS[None, :]).argmin(1)
    evm = np.mean(np.abs(d - POINTS[hard])[read] ** 2)
    frame = decode_frame(llr, rec)
    good = crc_ok(frame)
    errors = None
    if good:
        sent = onair_bits(frame, rec).reshape(-1, 2)
        got = np.stack([(hard >> 1) & 1, hard & 1], 1)
        errors = int((got != sent)[read].sum())
    q = Quality(float(-10 * np.log10(max(evm, 1e-12))), errors, 2 * int(read.sum()), r.df)
    if on_symbols is not None:
        times = r.n0 + np.asarray(rec.stagger)[:, None] + data_symbols(rec) * H
        carriers = np.broadcast_to(np.asarray(rec.centres)[:, None], (rec.nb, rec.ndata))
        on_symbols(SymbolFrame(rec.level, 0, good, r.n0 + C0 * H,
                               r.n0 + max(rec.stagger) + rec.nsym * H,
                               times.ravel()[read].copy(), carriers.ravel()[read].copy(),
                               d[read].copy(), POINTS.copy(), 'differential tone QPSK',
                               q.snr_db, q.offset_hz))
    return ToneResult(rec.level, (frame,) if good else (), q, r.lock, r.n0)


def decode_burst(audio: np.ndarray, bw: int, level: int | None = None, *,
                 on_symbols: Callable[[SymbolFrame], None] | None = None) -> ToneResult:
    """One tone burst in ``audio``. ``None`` classifies it within the bandwidth's
    tone rows by the weakest band's lattice collapse on each row's geometry
    (centres, stagger, grid rows) and reads it there. ``level`` reads it at that
    row first and, when that fails, at the best-scoring other row, since the
    peer may have keyed another; the failed read is returned when neither is clean."""
    x = Audio(audio)
    reads = {lv: _read(x, rec) for (b, lv), rec in ROWS.items()
             if b == bw and (level is None or lv == level)}
    if level is None:
        return _decode(max(reads.values(), key=lambda r: r.score), on_symbols)
    first_symbols = []
    first = _decode(reads[level], first_symbols.append if on_symbols else None)
    if first.ok:
        if on_symbols:
            on_symbols(first_symbols[0])
        return first
    other = max((_read(x, rec) for (b, lv), rec in ROWS.items() if b == bw and lv != level),
                key=lambda r: r.score)
    if other.score <= reads[level].score:
        if on_symbols:
            on_symbols(first_symbols[0])
        return first
    other_symbols = []
    res = _decode(other, other_symbols.append if on_symbols else None)
    if on_symbols:
        on_symbols((other_symbols if res.ok else first_symbols)[0])
    return res if res.ok else first
