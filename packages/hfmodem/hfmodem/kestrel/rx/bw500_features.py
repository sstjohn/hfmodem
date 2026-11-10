# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Block multirate BW500 features on the shared causal 6 kHz stream."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows
from scipy.signal import firwin

from . import varahf500 as rx

FS = 48000
BLOCK = 1024
AA = firwin(129, 2800, fs=FS)
LP = firwin(101, 120, fs=6000)
DELAY = 64 + 8 * 50
REF = np.array([rx._GRID480[rx.grid480_index(s, rx.NSYM)] for s, _ in rx.SUBBANDS])
D = REF[:, 1:] * REF[:, :-1].conj()
Q = np.zeros_like(D)
np.divide(D * D, np.abs(D) ** 2, out=Q, where=np.abs(D) > 0)


class Decimator:
    def __init__(self):
        self.raw = np.zeros(1152)
        self.windows = windows(self.raw, 129)[::8]
        self.filter = AA[::-1].copy()

    def push(self, x):
        self.raw[128:] = x
        y = np.einsum('ij,j->i', self.windows, self.filter)
        self.raw[:128] = self.raw[-128:]
        return y


class Lattice:
    def __init__(self, phases=4, pairs=384):
        self.phases = phases
        self.step = 64 // phases
        self.pairs = pairs
        self.q = Q[:, :9 + pairs]
        assert np.all((np.abs(self.q) > 0).sum(axis=1) == pairs)
        self.span = self.q.shape[1] * phases
        n = 128 // self.step
        frequency = -2j * np.pi * np.array([1350, 1650]) / 6000
        self.mix = np.exp(frequency[:, None] * np.arange(640)).reshape(2, 5, 128)[:, :, ::self.step]
        self.raw = np.zeros(228)
        self.windows = windows(self.raw, 101)[::self.step]
        self.lp = LP[::-1] * np.exp(frequency[:, None] * np.arange(-100, 1))
        self.score_buffer = np.zeros((2, self.span + n), complex)
        self.score_windows = windows(self.score_buffer[:, self.phases:], self.span - self.phases + 1, axis=1)[:, :, ::self.phases]
        self.z = np.zeros((2, phases + n), complex)
        self.offsets = np.arange(n) * self.step * 8 - DELAY
        self.empty = np.empty(0, int), np.empty((2, 0))
        self.ticks = 0
        self.blocks = 0
        self.retained = 0

    def push(self, x, score=True):
        self.raw[100:] = x
        self.z[:, self.phases:] = np.einsum('jn,sn->sj', self.windows, self.lp) * self.mix[:, self.blocks % 5]
        self.raw[:100] = self.raw[-100:]
        n = 128 // self.step
        d = self.z[:, self.phases:] * self.z[:, :-self.phases].conj()
        self.z[:, :self.phases] = self.z[:, -self.phases:]
        p = np.abs(d) ** 2
        u = self.score_buffer[:, self.retained:self.retained + n]
        u.fill(0)
        np.divide(d * d, p, out=u, where=p > 1e-100)
        first = self.ticks - self.retained
        total = self.retained + n
        count = max(0, total - self.span)
        if count and score:
            v = np.ascontiguousarray(self.score_windows[:, :count])
            scores = np.abs((v @ self.q[:, :, None])[:, :, 0]) / self.pairs
            starts = first * self.step * 8 + self.offsets[:count]
        else:
            starts, scores = self.empty
        self.retained = min(total, self.span)
        if count:
            self.score_buffer[:, :self.retained] = self.score_buffer[:, count:total]
        self.ticks += n
        self.blocks += 1
        return starts, scores


class Index:
    def __init__(self, dw, phases=4):
        self.dw = dw // 8
        self.hop = self.dw // phases
        self.phases = phases
        self.levels = [lv for lv, r in rx.INDEX_RECORDS.items() if r.dw == dw]
        r = rx.INDEX_RECORDS[self.levels[0]]
        self.band = slice(r.first_bin, r.first_bin + r.span)
        self.raw = np.zeros(self.dw + 128)
        self.raw_count = 0
        self.windows = windows(self.raw, self.dw)[::self.hop]
        self.transforms = 0
        refs = [rx._l3_tables(lv)[:2] for lv in self.levels]
        self.last = np.array([int(ref.max()) * phases for ref, _ in refs])
        self.keep = max(rx.INDEX_RECORDS[lv].ncols for lv in self.levels) * phases
        n = 128 // self.hop
        width = max(len(ref) for ref, _ in refs)
        self.offsets = np.zeros((len(refs), n, width), int)
        self.bins = np.full((len(refs), 1, width), -2, int)
        self.starts = np.arange(n) * self.hop * 8
        self.leads = [64 + rx.INDEX_RECORDS[lv].lead * dw for lv in self.levels]
        for i, (ref, bins) in enumerate(refs):
            self.offsets[i, :, :len(ref)] = np.arange(n)[:, None] + ref * phases - self.last[i]
            self.bins[i, 0, :len(ref)] = bins - r.first_bin
        self.lit = np.full(self.keep + n, -1, int)
        self.scored = dict.fromkeys(self.levels, 0)
        self.retained = 0

    def push(self, x):
        self.raw[self.raw_count:self.raw_count + len(x)] = x
        self.raw_count += len(x)
        count = max(0, (self.raw_count - self.dw) // self.hop + 1)
        if not count:
            return []
        power = np.abs(np.fft.rfft(self.windows[:count], axis=1)[:, self.band]) ** 2
        lit = power.argmax(axis=1)
        lit[power.max(axis=1) < 1e-100] = -1
        self.lit[self.keep:self.keep + count] = lit
        hits = (self.lit[self.offsets[:, :count] + self.keep] == self.bins).sum(axis=2)
        previous = self.transforms
        self.transforms += count
        used = count * self.hop
        self.raw[:self.raw_count - used] = self.raw[used:self.raw_count]
        self.raw_count -= used
        out = []
        for i, lv in enumerate(self.levels):
            begin = max(0, int(self.last[i]) - previous)
            if begin < count:
                starts = self.starts[begin:count] + (previous - int(self.last[i])) * self.hop * 8 - self.leads[i]
                out.append((lv, starts, hits[i, begin:count]))
                self.scored[lv] = self.transforms - int(self.last[i])
        self.lit[:self.keep] = self.lit[count:self.keep + count]
        return out


class FeatureTrack:
    def __init__(self, phases=8, stage='all'):
        if phases not in (2, 4, 8) or stage not in ('front', 'subbands', 'lattice', 'all'):
            raise ValueError('Invalid BW500 feature configuration')
        self.low_start = 0
        self.last_low = np.empty(0)
        self.front = Decimator()
        self.lattice = Lattice(phases)
        self.index = [Index(1024, phases), Index(2048, phases)]
        self.stage = stage
        self.pending = np.empty(0)
        self.samples = 0

    def feed(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError('BW500 input must be one-dimensional')
        raw = np.concatenate((self.pending, x)) if len(self.pending) else x
        out = []
        used = len(raw) // BLOCK * BLOCK
        self.low_start = self.samples // 8
        self.last_low = np.empty(used // 8) if used != BLOCK else None
        for j in range(0, used, BLOCK):
            low = self.front.push(raw[j:j + BLOCK])
            if used == BLOCK:
                self.last_low = low
            else:
                self.last_low[j // 8:(j + BLOCK) // 8] = low
            self.samples += BLOCK
            if self.stage == 'front':
                continue
            starts, scores = self.lattice.push(low, score=self.stage != 'subbands')
            if len(starts):
                out.append((4, starts, scores, self.samples))
            if self.stage == 'all':
                for idx in self.index:
                    out.extend((lv, st, sc, self.samples) for lv, st, sc in idx.push(low))
        self.pending = raw[used:].copy()
        return out


def exact_control_candidates(bins, clear, templates, shifts, hop=64, clearance=2):
    """Necessary clear-symbol sequences, shared across a callback's templates."""
    bins, templates = np.asarray(bins), np.asarray(templates)
    if bins.ndim == 1:
        bins = bins[:, None]
        templates = templates[:, :, None]
    count, width, carriers = templates.shape
    n = len(bins) - (width - 1) * hop
    if n <= 0:
        return np.zeros(count, bool)
    indices = np.arange(n)[:, None] + np.arange(width) * hop
    valid = (clear[indices] >= clearance).all(axis=1)
    cells = bins[indices[valid]].reshape(-1, width * carriers)
    weights = 128 ** np.arange(width * carriers, dtype=np.int64)
    signatures = cells @ weights
    wanted = (templates.reshape(count, -1) @ weights)[:, None] + np.asarray(shifts) * weights.sum()
    return np.isin(wanted, signatures).any(axis=1)


def payload_fit(bins, clear, tones, shifts, npre, xlen, n_sym, hop, woff,
                stride, grid, clearance, min_tones=13, min_heard=8):
    bins, clear = np.asarray(bins), np.asarray(clear)
    tones, shifts = np.asarray(tones), np.asarray(shifts)
    if (bins.ndim != 1 or clear.ndim != 1 or tones.ndim != 1 or shifts.ndim != 1
            or len(bins) != len(clear) or len(tones) != n_sym or not len(shifts)
            or not 0 <= npre <= n_sym or n_sym < 1 or grid <= 0 or hop <= 0):
        raise ValueError('Invalid tone-fit dimensions or sample grid')
    lo, hi = -(n_sym - 1) * hop, xlen - stride
    if not len(bins) or hi < lo:
        return np.full(n_sym, -1, np.int32), 0
    starts = np.arange(lo, hi + 1, grid)
    offsets = np.arange(n_sym) * hop + woff
    n = len(starts)
    half = 2 * (woff % grid) == grid
    if hop % grid == 0 and (not half or (hop // grid) % 2 == 0):
        origins = np.rint((starts + woff) / grid).astype(np.int64)
        row_lookup = origins - origins[0]
        n = int(row_lookup[-1]) + 1
        delta = origins[0] + np.arange(n_sym) * (hop // grid)
        valid = clear >= clearance
        jump = hop // grid
        pad = n_sym * jump + abs(int(delta[0])) + n + len(bins)
        padded = np.zeros(((pad + jump - 1) // jump) * jump, dtype=np.int32)
        base = max(0, -int(delta[npre] if npre < n_sym else delta[0]))
        padded[base:base + len(bins)] = valid
        sums = padded.reshape(-1, jump).cumsum(axis=0).ravel()
        first = base + np.arange(n) + (delta[npre] if npre < n_sym else delta[0])
        last = first + (n_sym - npre - 1) * jump
        counts = sums[last] - np.where(first >= jump, sums[np.maximum(0, first - jump)], 0)
        if npre == n_sym:
            counts.fill(0)
        occurrences = np.flatnonzero(valid)
        difference = bins[occurrences, None] - tones[None, npre:]
        low_shift, high_shift = int(shifts.min()), int(shifts.max())
        if high_shift - low_shift <= 1024:
            lookup = np.full(high_shift - low_shift + 1, -1, dtype=np.int64)
            for i in range(len(shifts) - 1, -1, -1):
                lookup[shifts[i] - low_shift] = i
            event, column = np.nonzero((difference >= low_shift) & (difference <= high_shift))
            shift_ids = lookup[difference[event, column] - low_shift]
            rows = occurrences[event] - delta[npre:][column]
            inside = (rows >= 0) & (rows < n) & (shift_ids >= 0)
            matches = np.bincount(shift_ids[inside] * n + rows[inside],
                                  minlength=len(shifts) * n).reshape(len(shifts), n)
        else:
            rows = occurrences[:, None] - delta[npre:]
            inside = (rows >= 0) & (rows < n)
            matches = np.array([np.bincount(rows[inside & (difference == shift)], minlength=n)
                                for shift in shifts])

        matches = matches[:, row_lookup]
        counts = counts[row_lookup]
    else:
        indices = np.rint((starts[:, None] + offsets) / grid).astype(np.int64)
        clipped = indices.clip(0, len(bins) - 1)
        valid = (indices >= 0) & (indices < len(bins)) & (clear[clipped] >= clearance)
        counts = valid[:, npre:].sum(axis=1)
        difference = bins[clipped[:, npre:]] - tones[npre:]
        matches = np.array([((difference == shift) & valid[:, npre:]).sum(axis=1)
                            for shift in shifts])
    accepted = (matches >= min_tones) | ((counts >= min_heard) & (matches == counts))
    rank = ((accepted * (n_sym + 1) + matches) * (n_sym + 1) - counts)
    shift_index, row = np.unravel_index(rank.argmax(), rank.shape)
    at = int(starts[row])
    indices = np.rint((at + offsets) / grid).astype(np.int64)
    clipped = indices.clip(0, len(bins) - 1)
    valid = ((indices >= 0) & (indices < len(bins)) & (clear[clipped] >= clearance)
             & (bins[clipped] >= 0))
    return np.where(valid, bins[clipped] - shifts[shift_index], -1).astype(np.int32), at
