# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every emission in a window of audio, found and measured whether or not a
modem will claim it.

The grid is the passband plan's (4096-point Hann, 20 ms hop, 11.7 Hz bins) so
a display can draw :meth:`Tracker.frames` without a second STFT. A burst is a
connected region of the spectrogram over a per-bin floor; the bursts of one
over (the sub-bands of a multi-tone over, the hops of a single-tone MFSK one)
are gathered into one; and a track is the overs that share a band signature,
so two links whose overs alternate stay two tracks however closely their
skirts sit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from hfmodem.core.busy import MUTE_DROP_DB
from hfmodem.monitor import FS

NFFT = 4096
HOP = 960
BIN_HZ = FS / NFFT
ROW_S = HOP / FS

#: 160 ms: long enough to put a burst's parts in one row, short enough to see
#: an ARQ turnaround.
SMOOTH_ROWS = 8
MASK_DB = 6.0
BRIDGE_BINS = 2
#: A burst must reach this far over the floor somewhere, and last this many
#: rows. Over a minute of white noise that leaves no track at all.
SEED_DB = 10.0
MIN_BURST_ROWS = 4

#: A run continues a burst only when this much of the smaller of the two is
#: shared; plain adjacency welds one side of a link to the other.
LINK_FRAC = 0.5
#: A burst survives this many empty rows, so a one-row fade does not end it.
HANDOFF_ROWS = 2
#: Once a burst is this old its band is established: a run spilling more than
#: SPLIT_BINS past its core is the head of the next over, keying while this
#: one's tail is still in the smoothing, and is split off.
YOUNG_ROWS = 15
SPLIT_BINS = 8
CORE_FRAC = 0.25
#: A burst holding less than this share of its core in a contested run is a
#: tail, and gives the run up.
KEEP_FRAC = 0.25
#: Two live bursts whose cores overlap this much are one emission; two links
#: that coincide in time and band are one burst here.
MERGE_FRAC = 0.2
#: What a burst's spectrum is summed over beyond its own runs.
PAD_BINS = 2

#: The bursts of one over. Sub-bands keyed together share most of their time;
#: single-tone MFSK (VARA 500 holds each tone 0.1-0.4 s and hops across
#: 600 Hz) hands from one short narrow burst to the next. A CW ID keyed in a
#: PACTOR link's band as the link ends shares neither.
GROUP_BINS = 48
GROUP_FRAC = 0.5
HOP_BINS = 8
HOP_S = 0.6
#: VARA 2300's handshake tones hop across 797-2297 Hz. An over of at least
#: HOP_BAND_MIN hops, hops being most of it, is placed by where its tones
#: fell, less the outer tenth either side; not where that band holds a live
#: track's and is over HOP_OVER times as wide, when the hops are that
#: emission's own narrow tones: DL6MAA's PACTOR-3 at 41-48 s placed
#: 1066-2895 Hz round its 1125-1910 Hz link, and the link's track ended at
#: 50.6 s instead of running to 70. A BW500 over reaching the rest of its
#: band (1102-1711 Hz round a 1289-1711 Hz track) is 1.4 times as wide.
HOP_GAP_BINS = 136
HOP_BAND_MIN = 8
HOP_BAND_SHARE = 0.75
HOP_BAND_PCT = 10
HOP_OVER = 1.8
HANDOFF_S = 0.15
#: An over still going after this long joins its track now rather than at
#: its end.
LONG_S = 2.0

EDGE_DB = 26.0
#: Over the floor at all, where a short burst's mean is too noisy to hold a
#: -26 dB edge.
EDGE_GUARD_DB = 3.0
#: The edge walk steps over the dips of a comb (VARA's carriers sit 47 Hz
#: apart).
EDGE_BRIDGE_BINS = 4
EDGE_PAD_BINS = 4

#: A burst is a live track's when both its edges lie within a bin of the
#: track's, or when the two bands share this much of their union: over a
#: fading PACTOR or VARA link the -26 dB edges of one over and the next wander
#: by several bins, and within one bin the link came out as a track per over.
#: Two links whose skirts touch share a bin or two of it.
MATCH_BINS = 1
MATCH_SHARED = 0.7
#: A burst and a live track whose bands nest, the narrower covering at least
#: this share of the wider, are one emission: on pos_pactor1_local one tone of a PACTOR-1 link
#: covers a quarter of its band. WS8EOC's CW ID inside its link's band covers
#: an eighth and stays apart.
PART_FRAC = 0.2
TRACK_GAP_S = 2.0
UPDATE_S = 0.5
REPORT_S = 0.3

FLOOR_S = 30.0
FLOOR_EVERY_S = 2.0
FLOOR_DECIM = 5
FLOOR_MIN_S = 2.0
#: Under this the two medians are splitting noise from noise, and the floor is
#: the plain median.
SPLIT_DB = 6.0
SPLIT_ITERS = 8

#: The receiver's own mute, measured off the stream as tools/vara_monitor's
#: MuteFloor measures it: row level is bimodal wherever a local transmitter
#: keys, and on this station the mute can be 12-26 dB deep, short of
#: MUTE_DROP_DB, which alone catches only the PTT transitions. MuteFloor lets
#: a mute hold at most 45 % of the window; a PACTOR arm keys 0.96 s of every
#: 1.25 s and its tap is muted for 80 % of the session, so here the guard
#: against a strong station's gaps is the PTT transition alone.
MUTE_Q = 20
MUTE_WINDOW_S = 60.0
MUTE_ARM_S = 5.0
MUTE_EVERY_S = 1.0
MUTE_SEED_DB = 8.0
MUTE_MIN_DEPTH_DB = 10.0
MUTE_FRAC = (0.02, 0.9)
MUTE_DEEP_DB = 40.0
MUTE_DEEP_ROWS = 4

CADENCE_S = 10.0
CADENCE_MIN_S = 5.0
CADENCE_LAGS_S = (0.3, 3.0)
#: A period the window holds fewer times than this folds into noise that can
#: pass the null test: FT8 read 2.53 s off five seconds of envelope.
CADENCE_CYCLES = 3
FOLD_BIN_S = 0.06
#: A lag half the candidate's holding this share of its correlation is the
#: period: DL6MAA's PACTOR-3 alternates its packets over two cycles and the
#: link reads 2.5 s a quarter of the time at 0.8.
FUNDAMENTAL = 0.6
#: The raster-fold null: the same envelope folded at periods beside the
#: candidate, which no cycle has.
NULL_RATIOS = (0.72, 0.80, 0.89, 1.12, 1.28)
NULL_DB = 6.0
#: Two tracks that both key on a cycle, their band envelopes over the last
#: CADENCE_S correlating this well, are one emission's sub-bands.
STEP_CORR = 0.7
ARQ_CYCLES = (1.25, 1.4)
ARQ_TOL_S = 0.05

CREST_S = 0.5
RING_S = 3.0
#: Within this of DC a window holds the input's DC offset and its Hann skirt,
#: which on dl6maa in a 0-12 kHz window outweighed the PACTOR link beside it;
#: within this of Nyquist, whatever the converter folds there.
DC_HZ = 60.0


@dataclass
class TrackReport:
    t: float
    kind: str                    # "track" | "track-end"
    track: int
    t_start: float
    centre_hz: float
    lo_hz: float
    hi_hz: float
    level_dbfs: float            # RMS in the band, as core.busy: a full-scale sine is -3.0
    over_db: float               # its strongest bin over the floor there, as SEED_DB reads
    crest_db: float | None
    cadence_s: float | None
    label: str
    bursts: int
    active_s: float

    @property
    def bw_hz(self) -> float:
        return round(self.hi_hz - self.lo_hz + BIN_HZ, 1)


_SERIAL = count()


@dataclass(eq=False)
class _Burst:
    start: float
    last: float
    spec: np.ndarray
    cnt: np.ndarray
    lo: int = 1 << 30
    hi: int = -1
    rows: int = 0
    peak_db: float = 0.0
    runs: list[tuple[int, int]] = field(default_factory=list)
    hits: np.ndarray | None = None
    idle: int = 0
    hops: int = 0
    hop_cores: list[tuple[int, int]] = field(default_factory=list)
    parts: int = 0
    track: "_Track | None" = None
    group: "_Burst | None" = None
    first: tuple[int, int] = (0, 0)
    serial: int = field(default_factory=lambda: next(_SERIAL))

    def core(self) -> tuple[int, int]:
        """Where the burst has been most of the time, rather than everywhere
        its onset splattered."""
        idx = np.flatnonzero(self.hits >= CORE_FRAC * self.rows) if self.hits is not None else []
        return (int(idx[0]), int(idx[-1])) if len(idx) else (self.lo, self.hi)

    def mean(self) -> np.ndarray:
        return _mean(self.spec, self.cnt)

    def add(self, other: "_Burst") -> None:
        self.spec = self.spec + other.spec
        self.cnt = self.cnt + other.cnt
        self.start, self.last = min(self.start, other.start), max(self.last, other.last)
        self.lo, self.hi = min(self.lo, other.lo), max(self.hi, other.hi)
        self.peak_db = max(self.peak_db, other.peak_db)


@dataclass(eq=False)
class _Track:
    id: int
    start: float
    last: float
    spec: np.ndarray
    cnt: np.ndarray
    centre: float
    lo: int
    hi: int
    active: float = 0.0
    until: float = -1.0
    bursts: int = 0
    crest_db: float | None = None
    edges: dict[int, tuple[int, int]] = field(default_factory=dict)
    stepped: bool = False

    def held(self, t0: float, t1: float) -> None:
        """Count ``t0..t1`` as time the track was keyed, once however many of
        its bursts overlap there."""
        self.active += max(0.0, t1 - max(t0, self.until))
        self.until = max(self.until, t1)


def _mean(spec: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Each bin's power while the emission held it, so an MFSK tone keyed a
    tenth of the time counts at its own level."""
    return spec / np.maximum(cnt, 1)


def signature(mean: np.ndarray, floor: np.ndarray, lo: int, hi: int,
              bridge: int = EDGE_BRIDGE_BINS) -> tuple[float, int, int] | None:
    """``(centroid, lo, hi)`` in bins: the -26 dB edges of a mean spectrum's
    excess over the floor, looked for within ``lo..hi`` and a few bins either
    side, and the power-weighted centroid between them, which is sub-bin."""
    a, b = max(0, lo - EDGE_PAD_BINS), min(len(mean), hi + EDGE_PAD_BINS + 1)
    ex = np.clip(mean[a:b] - floor[a:b], 0.0, None)
    if not ex.size or ex.max() <= 0:
        return None
    k = int(np.argmax(ex))
    keep = (ex >= ex[k] * 10 ** (-EDGE_DB / 10)) & (
        mean[a:b] >= floor[a:b] * 10 ** (EDGE_GUARD_DB / 10))
    keep[k] = True
    idx = np.flatnonzero(keep)
    gaps = np.flatnonzero(np.diff(idx) > bridge + 1)
    starts, ends = np.r_[idx[0], idx[gaps + 1]], np.r_[idx[gaps], idx[-1]]
    i = int(np.searchsorted(ends, k))
    l, r = int(starts[i]), int(ends[i])
    w = ex[l:r + 1]
    return a + float((np.arange(l, r + 1) * w).sum() / w.sum()), a + l, a + r


def two_medians_floor(hist: np.ndarray) -> np.ndarray:
    """Per column of dB rows: the lower cluster's median where the column
    splits in two, else its median. Sorted once, so each pass is index
    arithmetic."""
    s = np.sort(hist, axis=0)
    t, k = s.shape
    cols = np.arange(k)
    lo, hi = s[t // 10], s[(9 * t) // 10]
    for _ in range(SPLIT_ITERS):
        c = np.clip((s < (0.5 * (lo + hi))[None]).sum(axis=0), 1, t - 1)
        lo = s[(c - 1) // 2, cols]
        hi = s[c + (t - c - 1) // 2, cols]
    return np.where(hi - lo >= SPLIT_DB, lo, s[t // 2])


def fold_depth(env_db: np.ndarray, period_s: float) -> float:
    ph = (np.arange(len(env_db)) * ROW_S / period_s) % 1.0
    nb = max(4, int(round(period_s / FOLD_BIN_S)))
    b = np.minimum((ph * nb).astype(int), nb - 1)
    cnt = np.bincount(b, minlength=nb)
    if (cnt == 0).any():
        return 0.0
    m = np.bincount(b, env_db, minlength=nb) / cnt
    return float(m.max() - m.min())


def cadence(env_db: np.ndarray) -> float | None:
    """The envelope's dominant period from its autocorrelation, reported only
    where folding at it stands NULL_DB deeper than folding beside it."""
    x = env_db - env_db.mean()
    n = len(x)
    lo, hi = int(CADENCE_LAGS_S[0] / ROW_S), min(int(CADENCE_LAGS_S[1] / ROW_S), n // CADENCE_CYCLES)
    if hi <= lo + 2:
        return None
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0] * n / (n - np.arange(n))
    k = lo + int(np.argmax(acf[lo:hi + 1]))
    while k // 2 >= lo and acf[k // 2 - 1:k // 2 + 2].max() >= FUNDAMENTAL * acf[k]:
        k = k // 2 - 1 + int(np.argmax(acf[k // 2 - 1:k // 2 + 2]))
    k = min(max(k, lo), hi)
    y0, y1, y2 = acf[k - 1:k + 2]
    den = y0 - 2 * y1 + y2
    p = (k + (float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0)) * ROW_S
    null = np.mean([fold_depth(env_db, p * r) for r in NULL_RATIOS])
    return round(p, 3) if fold_depth(env_db, p) - null >= NULL_DB else None


def shape_label(bw_hz: float, cadence_s: float | None) -> str:
    """A description of the emission's shape, never an identification."""
    arq = cadence_s is not None and any(abs(cadence_s - c) <= ARQ_TOL_S
                                        for c in ARQ_CYCLES)
    if bw_hz < 60:
        return "CW-shaped"
    if 1500 <= bw_hz <= 2800:
        return "2k-multicarrier"
    if arq and bw_hz <= 1000 or 150 <= bw_hz < 350:
        return "FSK-shaped"
    if 350 <= bw_hz <= 700:
        return "500-wide"
    return "UNKNOWN"


def _runs(mask: np.ndarray, bridge: int = BRIDGE_BINS) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    cut = np.flatnonzero(np.diff(idx) > bridge + 1)
    return list(zip(np.r_[idx[0], idx[cut + 1]].tolist(), np.r_[idx[cut], idx[-1]].tolist()))


def _linked(runs: list[tuple[int, int]], run: tuple[int, int]) -> bool:
    s, e = run
    for p, q in runs:
        ov = min(e, q) - max(s, p) + 1
        if ov > 0 and ov >= LINK_FRAC * min(e - s + 1, q - p + 1):
            return True
    return False


def _divide(run: tuple[int, int], hit: list[_Burst]
            ) -> list[tuple[tuple[int, int], _Burst | None, bool]]:
    """Share a run among the bursts it continues, bin by bin to the nearest
    core: ``(piece, burst, counts)``. None starts a burst of its own; a piece
    that does not count is an established burst's skirt reaching toward a
    neighbour, which continues it but never enters its spectrum. Nor does any
    piece of a run that had to be shared: that row holds two emissions."""
    s, e = run
    bins = np.arange(s, e + 1)
    cores = [b.core() for b in hit]
    dist = np.array([np.maximum(lo - bins, 0) + np.maximum(bins - hi, 0) for lo, hi in cores])
    owner = dist.argmin(axis=0)
    if len(hit) > 1:
        share = np.bincount(owner, minlength=len(hit))
        weak = [i for i, (lo, hi) in enumerate(cores) if share[i] < KEEP_FRAC * (hi - lo + 1)]
        if len(weak) < len(hit):
            dist[weak] = 1 << 30
            owner = dist.argmin(axis=0)
    far = dist[owner, np.arange(len(bins))]
    out: list[tuple[tuple[int, int], _Burst | None, bool]] = []
    edges = np.flatnonzero(np.diff(owner)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(bins)]):
        b, seg = hit[owner[lo]], far[lo:hi]
        if b.rows >= YOUNG_ROWS and (seg > 0).sum() > SPLIT_BINS:
            out += [((s + lo + p, s + lo + q), b, True) for p, q in _runs(seg == 0, 0)]
            out += [((s + lo + p, s + lo + q), b, False) if q - p < SPLIT_BINS
                    else ((s + lo + p, s + lo + q), None, True) for p, q in _runs(seg > 0, 0)]
        else:
            out.append(((s + lo, s + hi - 1), b, True))
    return out


def _hop(h: _Burst) -> bool:
    lo, hi = h.core()
    return hi - lo < HOP_BINS and h.last - h.start <= HOP_S


def _together(h: _Burst, g: _Burst) -> bool:
    lo, hi = h.core()
    gap = max(lo - g.hi, g.lo - hi, 0)
    if (gap <= HOP_GAP_BINS and g.hops and _hop(h)
            and h.start <= g.last + HANDOFF_S and h.last >= g.start):
        return True
    ov = min(h.last, g.last) - max(h.start, g.start)
    return gap <= GROUP_BINS and ov >= GROUP_FRAC * min(h.last - h.start, g.last - g.start)


def _in_step(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    return (a is not None and b is not None and len(a) == len(b)
            and float(np.mean(a * b)) >= STEP_CORR)


def _nested(a: int, b: int, c: int, d: int) -> bool:
    """Band a..b lies within c..d and covers at least PART_FRAC of it."""
    return a >= c - MATCH_BINS and b <= d + MATCH_BINS and b - a + 1 >= PART_FRAC * (d - c + 1)


class MuteSplit:
    """Where the band ends and the receiver's own mute begins, in row level."""

    def __init__(self) -> None:
        self.levels: deque[float] = deque(maxlen=int(MUTE_WINDOW_S / ROW_S))
        self.threshold: float | None = None
        self._due = 0

    def push(self, level_db: float) -> bool:
        """Feed one row's level; True when the threshold was re-measured."""
        self.levels.append(level_db)
        self._due -= 1
        if self._due > 0 or len(self.levels) * ROW_S < MUTE_ARM_S:
            return False
        self._due = int(MUTE_EVERY_S / ROW_S)
        self.threshold = self._split()
        return True

    def _split(self) -> float | None:
        d = np.fromiter(self.levels, float, len(self.levels))
        band = float(np.median(d))
        low = d[d < band - MUTE_SEED_DB]
        if low.size < MUTE_FRAC[0] * d.size:
            return None
        mute = float(np.median(low))
        for _ in range(SPLIT_ITERS):
            lo, hi = d[d < (cut := 0.5 * (band + mute))], d[d >= cut]
            if not lo.size or not hi.size:
                return None
            band, mute = float(np.median(hi)), float(np.median(lo))
        cut = 0.5 * (band + mute)
        if not MUTE_FRAC[0] * d.size <= (d < cut).sum() <= MUTE_FRAC[1] * d.size:
            return None
        if band - mute < MUTE_MIN_DEPTH_DB or (d < band - MUTE_DEEP_DB).sum() < MUTE_DEEP_ROWS:
            return None
        return cut


class Tracker:
    """Push ``(t0, pcm)`` at 48 kHz, ``t0`` the sample index of ``pcm[0]``;
    get back a :class:`TrackReport` for every live track each half second
    and one as each ends. ``pcm`` is float full scale, or int16."""

    def __init__(self, lo_hz: float = 300.0, hi_hz: float = 2900.0,
                 keep_s: float = CADENCE_S) -> None:
        self.k0 = int(np.ceil(max(lo_hz, DC_HZ) / BIN_HZ))
        self.k1 = int(np.floor(min(hi_hz, FS / 2 - DC_HZ) / BIN_HZ)) + 1
        self.freqs = np.arange(self.k0, self.k1) * BIN_HZ
        self._win = np.hanning(NFFT + 1)[:-1]
        self._scale = 2.0 / (NFFT * float((self._win ** 2).sum()))
        self._buf = np.zeros(0)
        self._base: int | None = None
        self._prev = np.zeros((SMOOTH_ROWS - 1, self.k1 - self.k0))
        self._n = 0
        self._heard = 0
        keep = int(max(keep_s, CADENCE_S) / ROW_S)
        self._keep = int(keep_s / ROW_S)
        self._rows: deque[np.ndarray] = deque(maxlen=keep)
        self._times: deque[float] = deque(maxlen=keep)
        self._muted: deque[bool] = deque(maxlen=keep)
        self._hist: deque[tuple[np.ndarray, float]] = deque(
            maxlen=int(FLOOR_S / ROW_S / FLOOR_DECIM))
        self._mute = MuteSplit()
        self._band: deque[float] = deque(maxlen=int(FLOOR_S / ROW_S))
        self._floor_db: np.ndarray | None = None
        self._floor: np.ndarray | None = None
        self._floor_due = 0.0
        self._audio = np.zeros(0)
        self._audio_t = 0.0
        self._open: list[_Burst] = []
        self._groups: list[_Burst] = []
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._tick = 0.0
        self.now = 0.0

    def frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t, freqs, power_dbfs)``: the retained STFT rows on the grid, with
        rows the receiver was muted through as NaN."""
        n = min(self._keep, len(self._rows))
        if not n:
            return np.zeros(0), self.freqs, np.zeros((0, len(self.freqs)))
        p = 10 * np.log10(np.vstack(list(self._rows)[-n:]) + 1e-30)
        p[np.array(list(self._muted)[-n:])] = np.nan
        return np.array(list(self._times)[-n:]), self.freqs, p

    def push(self, t0: int, pcm: np.ndarray) -> list[TrackReport]:
        x = np.asarray(pcm)
        x = x / 32768.0 if x.dtype == np.int16 else x.astype(float)
        if self._base is None or t0 != self._base + len(self._buf):
            self._buf, self._base = np.zeros(0), t0
            self._audio, self._audio_t = np.zeros(0), t0 / FS
        self._buf = np.concatenate([self._buf, x])
        self._audio = np.concatenate([self._audio, x])
        if (cut := len(self._audio) - int(RING_S * FS)) > 0:
            self._audio = self._audio[cut:]
            self._audio_t += cut / FS
        n = (len(self._buf) - NFFT) // HOP + 1
        if n <= 0:
            return []
        fr = sliding_window_view(self._buf, NFFT)[::HOP][:n]
        p = np.abs(np.fft.rfft(fr * self._win, axis=1)[:, self.k0:self.k1]) ** 2 * self._scale
        t_first = (self._base + NFFT / 2) / FS
        self._buf = self._buf[n * HOP:]
        self._base += n * HOP
        full = np.vstack([self._prev, p])
        self._prev = full[-(SMOOTH_ROWS - 1):]
        c = np.cumsum(np.vstack([np.zeros((1, full.shape[1])), full]), axis=0)
        sm = (c[SMOOTH_ROWS:] - c[:-SMOOTH_ROWS]) / SMOOTH_ROWS
        sm[:, 1:-1] = (sm[:, :-2] + sm[:, 1:-1] + sm[:, 2:]) / 3.0
        sdb = 10 * np.log10(sm + 1e-30)
        out: list[TrackReport] = []
        for i in range(n):
            out += self._step(t_first + i * ROW_S, p[i], sdb[i])
        return out

    def flush(self) -> list[TrackReport]:
        """End every track at the end of the stream."""
        for b in list(self._open):
            self._close(b)
        self._settle(float("inf"))
        out = [self._report(tr, "track-end") for tr in self._tracks if self._shown(tr)]
        self._tracks = []
        return out

    def _step(self, t: float, p: np.ndarray, sdb: np.ndarray) -> list[TrackReport]:
        self.now = t
        self._n += 1
        level = float(np.percentile(10 * np.log10(p + 1e-30), MUTE_Q))
        armed = self._mute.threshold is not None
        if self._mute.push(level) and not armed and self._mute.threshold is not None:
            self._disown(self._mute.threshold)
        muted = ((self._mute.threshold is not None and level <= self._mute.threshold)
                 or (len(self._band) * ROW_S >= 1.0
                     and level < float(np.median(self._band)) - MUTE_DROP_DB))
        if not muted:
            self._band.append(level)
        self._heard = 0 if muted else self._heard + 1
        self._rows.append(p.astype(np.float32))
        self._times.append(t)
        self._muted.append(bool(muted))
        if self._heard > SMOOTH_ROWS and self._n % FLOOR_DECIM == 0:
            self._hist.append((sdb.astype(np.float32), level))
        held = len(self._hist) * FLOOR_DECIM * ROW_S
        if held >= FLOOR_MIN_S and t >= self._floor_due:
            self._floor_db = two_medians_floor(np.vstack([h for h, _ in self._hist]))
            self._floor = 10 ** (self._floor_db / 10)
            self._floor_due = t + (FLOOR_EVERY_S if held >= FLOOR_S else UPDATE_S)
        if self._floor_db is not None:
            runs = [] if muted else _runs(sdb >= self._floor_db + MASK_DB)
            self._grow(t, p, runs, sdb - self._floor_db)
        if t < self._tick:
            return []
        self._tick = t + UPDATE_S
        return self._update(t)

    def _disown(self, threshold: float) -> None:
        """The mute has just been measured; the rows the receiver was deaf
        through before then come out of the floor's history. Rows within the
        smoothing of a mute never go in: they hold part of it, and a floor
        learned from them sits under the band until it ages out."""
        keep = [(h, lv) for h, lv in self._hist if lv > threshold]
        self._hist.clear()
        self._hist.extend(keep)
        self._floor_due = 0.0

    def _grow(self, t: float, p: np.ndarray, runs: list[tuple[int, int]],
              over: np.ndarray) -> None:
        got: dict[int, list[tuple[int, int]]] = {}
        counted: dict[int, list[tuple[int, int]]] = {}
        for run in runs:
            hit = [b for b in self._open if _linked(b.runs, run)]
            if len(hit) > 1 and all(b.rows < YOUNG_ROWS for b in hit):
                for o in hit[1:]:
                    self._absorb(hit[0], o)
                    got.setdefault(id(hit[0]), []).extend(got.pop(id(o), []))
                    counted.setdefault(id(hit[0]), []).extend(counted.pop(id(o), []))
                hit = hit[:1]
            pieces = _divide(run, hit) if hit else [(run, None, True)]
            shared = len(pieces) > 1
            for piece, b, counts in pieces:
                if b is None:
                    b = _Burst(start=t, last=t, spec=np.zeros_like(p), cnt=np.zeros(len(p)),
                               hits=np.zeros(len(p), dtype=np.int32))
                    self._open.append(b)
                got.setdefault(id(b), []).append(piece)
                if counts and not shared:
                    counted.setdefault(id(b), []).append(piece)
        for b in list(self._open):
            mine = got.get(id(b))
            if not mine:
                b.idle += 1
                if b.idle > HANDOFF_ROWS:
                    self._close(b)
                continue
            b.idle, b.runs, b.last = 0, mine, t
            b.rows += 1
            for s, e in mine:
                b.hits[s:e + 1] += 1
                b.peak_db = max(b.peak_db, float(over[s:e + 1].max()))
            b.lo, b.hi = min(b.lo, mine[0][0]), max(b.hi, max(e for _, e in mine))
            region = np.zeros(len(p), dtype=bool)
            for s, e in counted.get(id(b), []):
                region[max(0, s - PAD_BINS):e + PAD_BINS + 1] = True
            b.spec[region] += p[region]
            b.cnt[region] += 1
            if b.track is not None:
                b.track.spec[region] += p[region]
                b.track.cnt[region] += 1
                b.group.spec[region] += p[region]
                b.group.cnt[region] += 1
                b.group.lo, b.group.hi = min(b.group.lo, b.lo), max(b.group.hi, b.hi)
                b.track.held(t - ROW_S, t)
                b.track.last = t
        self._merge()

    def _merge(self) -> None:
        live = [b for b in self._open if b.idle == 0]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if a not in self._open or b not in self._open:
                    continue
                (alo, ahi), (blo, bhi) = a.core(), b.core()
                ov = min(ahi, bhi) - max(alo, blo) + 1
                if ov >= MERGE_FRAC * min(ahi - alo + 1, bhi - blo + 1):
                    self._absorb(a, b)

    def _absorb(self, a: _Burst, b: _Burst) -> None:
        a.add(b)
        a.rows = max(a.rows, b.rows)
        a.runs = a.runs + b.runs
        a.hits = a.hits + b.hits
        a.track, a.group = a.track or b.track, a.group or b.group
        self._open.remove(b)

    def _close(self, b: _Burst) -> None:
        self._open.remove(b)
        if b.track is None and b.rows >= MIN_BURST_ROWS:
            self._group(b)

    def _group(self, h: _Burst) -> None:
        for g in self._groups:
            if _together(h, g):
                break
        else:
            g = _Burst(start=h.start, last=h.last, spec=np.zeros_like(h.spec),
                       cnt=np.zeros_like(h.cnt))
            self._groups.append(g)
        g.add(h)
        g.parts += 1
        if _hop(h):
            g.hops += 1
            g.hop_cores.append(h.core())
        g.rows = int(round((g.last - g.start) / ROW_S)) + 1
        if g.track is not None:
            g.track.spec += h.spec
            g.track.cnt += h.cnt
            g.track.held(h.start, h.last)
            g.track.last = max(g.track.last, h.last)

    def _settle(self, t: float) -> None:
        for b in self._open:
            if b.track is None and b.last - b.start >= LONG_S and b.peak_db >= SEED_DB:
                lo, hi = b.core()
                g = next((g for g in self._groups if g.track is not None and _together(b, g)
                          and max(lo - g.hi, g.lo - hi) < 0), None)
                if g is None:
                    g = _Burst(start=b.start, last=b.last, spec=np.zeros_like(b.spec),
                               cnt=np.zeros_like(b.cnt))
                    self._groups.append(g)
                g.add(b)
                g.rows = int(round((g.last - g.start) / ROW_S)) + 1
                if g.track is None:
                    self._assign(g)
                else:
                    g.track.spec += b.spec
                    g.track.cnt += b.cnt
                    g.track.held(b.start, b.last)
                b.track, b.group = g.track, g
        for g in list(self._groups):
            if g.track is not None:
                # An over still going refines its edges as it goes, looked for
                # within its bursts' cores: their hulls only ever grow, and the
                # peak of all of one can be a line far off the over.
                mine = [b for b in self._open if b.group is g]
                g.last = max([g.last] + [b.last for b in mine])
                cores = [b.core() for b in mine] or [g.first]
                span = min(c[0] for c in cores), max(c[1] for c in cores)
                if g.serial in g.track.edges and (
                        sig := signature(g.mean(), self._floor, *span)) is not None:
                    g.track.edges[g.serial] = sig[1:]
            if t - g.last > HANDOFF_S + SMOOTH_ROWS * ROW_S and not any(
                    b.start <= g.last + HANDOFF_S for b in self._open if b.track is None):
                self._groups.remove(g)
                if g.track is None and g.peak_db >= SEED_DB:
                    self._assign(g)

    def _assign(self, g: _Burst) -> None:
        sig = None
        if g.hops >= HOP_BAND_MIN and g.hops >= HOP_BAND_SHARE * g.parts:
            sig = self._hop_signature(g)
            if sig is not None and any(
                    _nested(tr.lo, tr.hi, sig[1], sig[2])
                    and sig[2] - sig[1] > HOP_OVER * (tr.hi - tr.lo)
                    for tr in self._tracks if g.start - tr.last <= TRACK_GAP_S):
                sig = None
        sig = sig or signature(g.mean(), self._floor, g.lo, g.hi)
        if sig is None:
            return
        c, lo, hi = sig
        tr = self._match(g.start, lo, hi)
        if tr is None:
            tr = _Track(self._next_id, g.start, g.last, np.zeros_like(g.spec),
                        np.zeros_like(g.cnt), c, lo, hi)
            self._next_id += 1
            self._tracks.append(tr)
        tr.spec += g.spec
        tr.cnt += g.cnt
        g.first = (lo, hi)
        if hi - lo + 1 >= MATCH_SHARED * (tr.hi - tr.lo + 1) or not tr.edges:
            tr.edges[g.serial] = g.first
        tr.held(g.start, g.last)
        tr.bursts += 1
        tr.start, tr.last = min(tr.start, g.start), max(tr.last, g.last)
        g.track = tr
        self._measure(tr)
        self._crest(tr, g.start, g.last)

    def _hop_signature(self, g: _Burst) -> tuple[float, int, int] | None:
        """An over of hops is a sparse comb, and where it sits is where most of
        its tones fell: a peer answering as our mute lifts is keyed with the
        receiver's own unmute transient, a few strong tones at the edges."""
        lo = int(np.percentile([a for a, _ in g.hop_cores], HOP_BAND_PCT))
        hi = int(np.percentile([b for _, b in g.hop_cores], 100 - HOP_BAND_PCT))
        w = np.clip(g.mean()[lo:hi + 1] - self._floor[lo:hi + 1], 0.0, None)
        if w.sum() <= 0:
            return None
        return lo + float((np.arange(len(w)) * w).sum() / w.sum()), lo, hi

    def _match(self, start: float, lo: int, hi: int) -> "_Track | None":
        """The live track whose band this burst's edges share; failing that,
        one nested with it, the narrower filling a good part of the wider:
        fading leaves one over reaching only part of its link's band, and the
        link's first over can have been that part. A track made of sub-bands
        keyed in step takes any burst inside its band."""
        live = [tr for tr in self._tracks if start - tr.last <= TRACK_GAP_S]

        def shared(tr: _Track) -> float:
            return (min(hi, tr.hi) - max(lo, tr.lo) + 1) / (max(hi, tr.hi) - min(lo, tr.lo) + 1)

        same = [tr for tr in live if max(abs(lo - tr.lo), abs(hi - tr.hi)) <= MATCH_BINS
                or shared(tr) >= MATCH_SHARED]
        if same:
            return max(same, key=shared)
        part = [tr for tr in live if _nested(lo, hi, tr.lo, tr.hi) or _nested(tr.lo, tr.hi, lo, hi)
                or tr.stepped and lo >= tr.lo - MATCH_BINS and hi <= tr.hi + MATCH_BINS]
        return max(part, key=lambda tr: tr.last) if part else None

    def _measure(self, tr: _Track) -> None:
        """The track's edges are the median of its whole overs' -26 dB edges,
        so one over that catches a neighbour's skirt does not move them, and
        one that fades to part of the band is not counted in them; its centre
        is the centroid of all it has held between them."""
        lo, hi = (int(np.median(e)) for e in zip(*tr.edges.values()))
        w = np.clip(_mean(tr.spec, tr.cnt)[lo:hi + 1] - self._floor[lo:hi + 1], 0.0, None)
        if w.sum() > 0:
            tr.centre, tr.lo, tr.hi = lo + float((np.arange(len(w)) * w).sum() / w.sum()), lo, hi

    def _update(self, t: float) -> list[TrackReport]:
        self._settle(t)
        out = self._fold(t)
        pending = self._open + self._groups
        for tr in list(self._tracks):
            if t - tr.last > TRACK_GAP_S and not any(
                    b.track is tr or b.track is None and b.start - tr.last <= TRACK_GAP_S
                    for b in pending):
                self._tracks.remove(tr)
                if self._shown(tr):
                    out.append(self._report(tr, "track-end"))
                continue
            self._measure(tr)
            if any(b.track is tr for b in self._open):
                self._crest(tr, t - CREST_S, tr.last)
            if self._shown(tr):
                out.append(self._report(tr, "track"))
        return out

    def _fold(self, t: float) -> list[TrackReport]:
        """Two live tracks are one emission when their bands have come to nest,
        a track that widens as its link's overs arrive otherwise running beside
        it; or when both key on a cycle and their envelopes move together, the
        sub-bands of one emission whose middle the floor has learned. DL6MAA never leaves a gap,
        so its floor is its own band, and when it widened to PACTOR-3 at 47 s
        only its new edges, 410-633 and 2367-2590 Hz, stood over it."""
        out: list[TrackReport] = []
        live = list(self._tracks)
        keyed = {tr.id: self._keying(tr, t) for tr in live}
        for old in sorted(live, key=lambda tr: tr.start):
            for young in live:
                if (young is old or young not in self._tracks or old not in self._tracks
                        or young.start < old.start):
                    continue
                nested = t - min(old.last, young.last) <= TRACK_GAP_S and (
                    _nested(young.lo, young.hi, old.lo, old.hi)
                    or _nested(old.lo, old.hi, young.lo, young.hi))
                if not nested and not _in_step(keyed[old.id], keyed[young.id]):
                    continue
                if self._shown(young):
                    out.append(self._report(young, "track-end"))
                if not nested:
                    old.edges = {-young.id: (min(old.lo, young.lo), max(old.hi, young.hi))}
                    old.stepped = True
                else:
                    old.edges.update(young.edges)
                old.spec += young.spec
                old.cnt += young.cnt
                old.bursts += young.bursts
                old.held(young.start, young.last)
                old.last = max(old.last, young.last)
                for b in self._open + self._groups:
                    if b.track is young:
                        b.track = old
                self._tracks.remove(young)
                self._measure(old)
        return out

    def _keying(self, tr: _Track, t: float) -> np.ndarray | None:
        """The track's band envelope over the last CADENCE_S, standardised, if
        it keys on a cycle at all."""
        env = self._envelope(tr, t)
        if env is None or cadence(env) is None or env.std() == 0:
            return None
        return (env - env.mean()) / env.std()

    @staticmethod
    def _shown(tr: _Track) -> bool:
        return tr.bursts >= 2 or tr.active >= REPORT_S

    def _crest(self, tr: _Track, t0: float, t1: float) -> None:
        """Peak to RMS of the track's band, isolated from the audio ring, over
        its latest half second of activity."""
        t0 = max(t0, t1 - CREST_S)
        i = max(0, int((t0 - NFFT / 2 / FS - self._audio_t) * FS))
        j = int((t1 + NFFT / 2 / FS - self._audio_t) * FS)
        seg = self._audio[i:j]
        if len(seg) < NFFT:
            return
        f = np.fft.rfft(seg)
        k = np.fft.rfftfreq(len(seg), 1 / FS)
        f[(k < (self.k0 + tr.lo - 0.5) * BIN_HZ) | (k > (self.k0 + tr.hi + 0.5) * BIN_HZ)] = 0
        y = np.fft.irfft(f, len(seg))
        rms = float(np.sqrt(np.mean(y * y)))
        if rms > 0:
            tr.crest_db = round(float(20 * np.log10(np.abs(y).max() / rms)), 2)

    def _envelope(self, tr: _Track, until: float) -> np.ndarray | None:
        """The track's band level, row by row, over the CADENCE_S up to
        ``until``; rows the receiver was muted through are the median."""
        times = np.fromiter(self._times, float, len(self._times))
        keep = np.flatnonzero((times <= until + ROW_S / 2) & (times > until - CADENCE_S))
        if len(keep) * ROW_S < CADENCE_MIN_S:
            return None
        rows = list(self._rows)
        env = 10 * np.log10(np.vstack([rows[i] for i in keep])[:, tr.lo:tr.hi + 1].sum(axis=1)
                            + 1e-30)
        mute = np.array(list(self._muted))[keep]
        if mute.all():
            return None
        env[mute] = np.median(env[~mute])
        return env

    def _report(self, tr: _Track, kind: str) -> TrackReport:
        """A track one bin wide reports its stronger neighbour too: a Hann
        mainlobe is never narrower than two bins."""
        mean = _mean(tr.spec, tr.cnt)
        level = float(10 * np.log10(mean[tr.lo:tr.hi + 1].sum() + 1e-30))
        over = float(10 * np.log10((mean[tr.lo:tr.hi + 1] + 1e-30)
                                   / (self._floor[tr.lo:tr.hi + 1] + 1e-30)).max())
        env = self._envelope(tr, self.now if kind == "track" else tr.last)
        cad = cadence(env) if env is not None else None
        lo, hi = tr.lo, tr.hi
        if lo == hi:
            side = mean[lo - 1] if lo > 0 else -1.0
            lo, hi = (lo - 1, hi) if hi == len(mean) - 1 or side >= mean[hi + 1] else (lo, hi + 1)
        return TrackReport(
            t=round(self.now, 3), kind=kind, track=tr.id, t_start=round(tr.start, 3),
            centre_hz=round(float((self.k0 + tr.centre) * BIN_HZ), 2),
            lo_hz=round(float((self.k0 + lo) * BIN_HZ), 1),
            hi_hz=round(float((self.k0 + hi) * BIN_HZ), 1),
            level_dbfs=round(level, 1), over_db=round(over, 1),
            crest_db=tr.crest_db, cadence_s=None if cad is None else float(cad),
            label=shape_label((tr.hi - tr.lo + 1) * BIN_HZ, cad),
            bursts=tr.bursts, active_s=round(tr.active, 2))
