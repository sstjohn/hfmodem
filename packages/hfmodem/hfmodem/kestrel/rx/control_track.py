# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bounded, absolute-time control spectra from the shared 6 kHz stream."""

from collections import OrderedDict

import numpy as np


class _Rows:
    __slots__ = ("starts", "bins", "clear", "resid", "power", "live")

    def __init__(self, capacity, live=False, band_width=0):
        self.starts = np.full(capacity, -1, dtype=np.int64)
        self.bins = None if live else np.empty((capacity, 3), dtype=np.int32)
        self.clear = None if live else np.empty((capacity, 2))
        self.resid = None if live else np.empty(capacity)
        self.power = None if live else np.empty((capacity, band_width))
        self.live = np.empty(capacity, dtype=bool) if live else None

    def copy_from(self, old, grid):
        keep = np.flatnonzero(old.starts >= 0)
        if len(keep) > len(self.starts):
            keep = keep[np.argsort(old.starts[keep])[-len(self.starts):]]
        dst = (old.starts[keep] // grid) % len(self.starts)
        self.starts[dst] = old.starts[keep]
        if self.live is not None:
            self.live[dst] = old.live[keep]
        else:
            self.bins[dst] = old.bins[keep]
            self.clear[dst] = old.clear[keep]
            self.resid[dst] = old.resid[keep]
            self.power[dst] = old.power[keep]


class ControlTrack:
    NFFT = 2048
    DECIMATION = 8
    FIR_DELAY = 64

    def __init__(self, *, retain_seconds: float = 6.0,
                 max_cache: int = 8192, deaf_dbfs: float = -50.0):
        self._raw_capacity = max(self.NFFT + self.FIR_DELAY,
                                 int(retain_seconds * 48000))
        self._low_capacity = self._raw_capacity // self.DECIMATION + 256
        self._max_cache = int(max_cache)
        if self._max_cache < 1:
            raise ValueError("max_cache must be positive")
        self.deaf_dbfs = float(deaf_dbfs)
        self._low = np.empty(self._low_capacity, dtype=np.float64)
        self._power = np.empty(self._raw_capacity + 1, dtype=np.float64)
        self._window = np.hanning(self.NFFT)[::self.DECIMATION]
        self.reset()

    def reset(self, raw_origin: int = 0) -> None:
        self.raw_origin = int(raw_origin)
        self.raw_available = self.raw_origin
        self._low_count = 0
        self._raw_count = 0
        self._power_sum = 0.0
        self._power[0] = 0.0
        self._have_raw = True
        self._cache = OrderedDict()
        self._live_cache = OrderedDict()
        self.transforms = 0
        self.cache_hits = 0

    @property
    def effective_end(self) -> int:
        return max(self.raw_origin, min(
            self.raw_available, self.raw_origin + self._low_count * self.DECIMATION
            - self.FIR_DELAY))

    @property
    def oldest_start(self) -> int:
        low = (self.raw_origin
               + max(0, self._low_count - self._low_capacity) * self.DECIMATION
               - self.FIR_DELAY)
        raw = self.raw_origin + max(0, self._raw_count - self._raw_capacity)
        return max(self.raw_origin, low, raw if self._have_raw else low)

    @property
    def cache_size(self) -> int:
        return sum(np.count_nonzero(c.starts >= 0) for c in self._cache.values())

    def _rows(self, caches, key, size, grid, *, live=False):
        old = caches.pop(key, None)
        capacity = min(self._max_cache, max(64, size,
                       len(old.starts) if old is not None else 0))
        allocated = sum(len(c.starts) for c in caches.values())
        while allocated + capacity > self._max_cache:
            _, dropped = caches.popitem(last=False)
            allocated -= len(dropped.starts)
        if old is None or len(old.starts) != capacity:
            rows = _Rows(capacity, live=live,
                         band_width=0 if live else key[2] - key[1] + 1)
            if old is not None:
                rows.copy_from(old, grid)
        else:
            rows = old
        caches[key] = rows
        return rows

    def feed(self, low: np.ndarray, raw_available: int, *,
             raw: np.ndarray | None = None) -> None:
        """Append decimator output and, optionally, the corresponding raw input."""
        raw_available = int(raw_available)
        if raw_available < self.raw_available:
            raise ValueError("raw time moved backwards; reset on discontinuity")
        low = np.asarray(low, dtype=np.float64).reshape(-1)
        count = len(low)
        kept = min(count, self._low_capacity)
        idx = np.arange(self._low_count + count - kept,
                        self._low_count + count) % self._low_capacity
        self._low[idx] = low[count - kept:]
        self._low_count += count
        if raw is None:
            self._have_raw = False
        else:
            raw = np.asarray(raw, dtype=np.float64).reshape(-1)
            if len(raw) != raw_available - self.raw_available:
                raise ValueError("raw length does not match raw_available")
            if not self._have_raw:
                raise ValueError("raw input must be supplied on every feed or none")
            power = np.cumsum(raw * raw) + self._power_sum
            kept = min(len(power), len(self._power))
            if kept:
                idx = np.arange(self._raw_count + len(raw) - kept + 1,
                                self._raw_count + len(raw) + 1) % len(self._power)
                self._power[idx] = power[len(power) - kept:]
                self._power_sum = float(power[-1])
            self._raw_count += len(raw)
        self.raw_available = raw_available

    def _starts(self, start: int, stop: int, grid: int) -> np.ndarray:
        start, stop, grid = int(start), int(stop), int(grid)
        if grid <= 0 or grid % self.DECIMATION:
            raise ValueError("grid must be a positive multiple of eight raw samples")
        if (start - self.raw_origin) % self.DECIMATION:
            raise ValueError("start must lie on the decimator sample lattice")
        if start < self.oldest_start:
            raise ValueError("requested samples are outside retained history")
        end = min(stop, self.effective_end) - self.NFFT + 1
        return np.arange(start, max(start, end), grid, dtype=np.int64)

    def track(self, start: int, stop: int, band: tuple[int, int] = (29, 98), *,
              bin_offset: float = 0.0, grid: int = 32
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows begin at ``start + i*grid`` in absolute raw sample coordinates."""
        return self._analyze(start, stop, band, bin_offset, grid)[:3]

    def power(self, start: int, stop: int, band: tuple[int, int],
              bin_offset: float = 0.0, grid: int = 32) -> np.ndarray:
        """Band powers on the track's windows and original full-rate FFT scale."""
        return self._analyze(start, stop, band, bin_offset, grid, power=True)[3]

    def _analyze(self, start, stop, band, bin_offset, grid, *, power=False):
        lo, hi = map(int, band)
        if not 1 <= lo <= hi < 128:
            raise ValueError("band must leave room for adjacent 6 kHz DFT bins")
        bin_offset = float(bin_offset)
        if not np.isfinite(bin_offset):
            raise ValueError("bin_offset must be finite")
        starts = self._starts(start, stop, grid)
        bins = np.empty((len(starts), 3), dtype=np.int32)
        clear = np.empty((len(starts), 2))
        resid = np.empty(len(starts))
        powers = np.empty((len(starts), hi - lo + 1)) if power else None
        if not len(starts):
            return bins, clear, resid, powers
        key = (int(grid), lo, hi, bin_offset, int(start) % int(grid))
        cache = self._rows(self._cache, key, len(starts), grid)
        slots = (starts // grid) % len(cache.starts)
        hit = cache.starts[slots] == starts
        retained = len(starts) <= len(cache.starts)
        if not retained:
            bins[hit] = cache.bins[slots[hit]]
            clear[hit] = cache.clear[slots[hit]]
            resid[hit] = cache.resid[slots[hit]]
            if power:
                powers[hit] = cache.power[slots[hit]]
        self.cache_hits += int(np.count_nonzero(hit))
        missing = np.flatnonzero(~hit)
        win = self._window
        if bin_offset:
            win = win * np.exp(-2j * np.pi * bin_offset * np.arange(256) / 256)
        for at in range(0, len(missing), 64):
            out = np.asarray(missing[at:at + 64])
            low_start = ((starts[out] - self.raw_origin + self.FIR_DELAY)
                         // self.DECIMATION)
            idx = (low_start[:, None] + np.arange(256)) % self._low_capacity
            blk = self._low[idx] * win
            spectrum = (np.fft.fft(blk, axis=1) if bin_offset
                        else np.fft.rfft(blk, axis=1))
            full = np.abs(spectrum) * self.DECIMATION
            mag = full[:, lo:hi + 1]
            band_power = mag ** 2
            if power:
                powers[out] = band_power
            saved = out >= len(starts) - len(cache.starts)
            cache.power[slots[out[saved]]] = band_power[saved]
            cols = np.arange(mag.shape[1])
            rows = np.arange(len(out))
            peaks = []
            for k in range(3):
                j = np.argmax(mag, axis=1)
                b = lo + j
                bins[out, k] = b
                if k == 0:
                    left, mid, right = (full[rows, b - 1], full[rows, b],
                                        full[rows, b + 1])
                    resid[out] = 2 * (right - left) / (left + 2 * mid + right + 1e-20)
                peaks.append(mag[rows, j])
                mag[np.abs(cols[None, :] - j[:, None]) <= 1] = 0
            for k in range(2):
                clear[out, k] = 20 * np.log10((peaks[k] + 1e-20)
                                             / (peaks[k + 1] + 1e-20))
            self.transforms += len(out)
        keep = missing[missing >= len(starts) - len(cache.starts)]
        dst = slots[keep]
        cache.starts[dst] = starts[keep]
        cache.bins[dst] = bins[keep]
        cache.clear[dst] = clear[keep]
        cache.resid[dst] = resid[keep]
        if retained:
            first = int(slots[0])
            end = first + len(starts)
            def read(values):
                if end <= len(values):
                    return values[first:end].copy()
                return np.concatenate((values[first:], values[:end - len(values)]))
            return read(cache.bins), read(cache.clear), read(cache.resid), read(cache.power) if power else None
        return bins, clear, resid, powers

    def live(self, start: int, stop: int, *, grid: int = 32) -> np.ndarray:
        starts = self._starts(start, stop, grid)
        result = np.empty(len(starts), dtype=bool)
        if not len(starts):
            return result
        key = (int(grid), int(start) % int(grid))
        cache = self._rows(self._live_cache, key, len(starts), grid, live=True)
        slots = (starts // grid) % len(cache.starts)
        hit = cache.starts[slots] == starts
        result[hit] = cache.live[slots[hit]]
        out = np.flatnonzero(~hit)
        missing_starts = starts[out]
        if self._have_raw:
            first = missing_starts - self.raw_origin
            power = (self._power[(first + self.NFFT) % len(self._power)]
                     - self._power[first % len(self._power)]) / self.NFFT
        else:
            first = ((missing_starts - self.raw_origin + self.FIR_DELAY)
                     // self.DECIMATION)
            power = np.empty(len(missing_starts))
            for at in range(0, len(missing_starts), 64):
                idx = (first[at:at + 64, None] + np.arange(256)) % self._low_capacity
                power[at:at + 64] = np.mean(self._low[idx] ** 2, axis=1)
        rms = np.sqrt(np.maximum(power, 0.0))
        result[out] = 20 * np.log10(rms + 1e-15) > self.deaf_dbfs
        keep = np.arange(max(0, len(starts) - len(cache.starts)), len(starts))
        dst = slots[keep]
        cache.starts[dst] = starts[keep]
        cache.live[dst] = result[keep]
        return result
