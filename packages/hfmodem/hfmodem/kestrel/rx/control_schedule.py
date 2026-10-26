# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Absolute control rows from the shared causal decimator."""

import bisect

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _accumulate(out, index, weights):
    """``out.flat[index] += weights``, repeats summed first as bincount sums them,
    over the span of ``out`` a scan touches rather than the whole ring."""
    if not len(index):
        return
    low = int(index.min())
    span = np.bincount(index - low, weights)
    flat = out.reshape(-1)
    flat[low:low + len(span)] += span.astype(out.dtype)


class HardBatch:
    capacity = 8192

    def __init__(self, templates, matches_only=False):
        self.matches_only = matches_only
        self.minimum = np.array([(4 * min(len(t) - pre, 8) + 4) // 5 for t, pre in templates])
        self.candidates = [set() for _ in templates]
        self.templates = templates
        self.lengths = np.array([len(t) for t, _ in templates])
        self.sizes = (self.lengths + 1).astype(np.int16)
        self.offsets = np.concatenate([np.arange(pre, len(t)) * 64
                                       for t, pre in templates])
        self.owner = np.concatenate([np.full(len(t) - pre, j, np.int64)
                                    for j, (t, pre) in enumerate(templates)])
        groups = list(dict.fromkeys((len(t), pre) for t, pre in templates))
        self.count_owner = np.array([groups.index((len(t), pre)) for t, pre in templates])
        self.count_offsets = np.concatenate([np.arange(pre, length) * 64 for length, pre in groups])
        self.count_template = np.concatenate([np.full(length - pre, j, np.int64)
                                              for j, (length, pre) in enumerate(groups)])
        tones = np.concatenate([t[pre:] for t, pre in templates])
        difference = np.arange(29, 99)[:, None] - tones
        hits = [np.flatnonzero(abs(row) <= 2) for row in difference]
        width = max(map(len, hits))
        self.hit_offsets = np.zeros((70, width), np.int64)
        self.hit_owner = np.zeros((70, width), np.int64)
        self.hit_shift = np.full((70, width), -1, np.int64)
        for tone, indices in enumerate(hits):
            self.hit_offsets[tone, :len(indices)] = self.offsets[indices]
            self.hit_owner[tone, :len(indices)] = self.owner[indices]
            self.hit_shift[tone, :len(indices)] = difference[tone, indices] + 2
        self.tags = np.full(self.capacity, np.iinfo(np.int64).min, np.int64)
        self.counts = np.zeros((len(groups), self.capacity), np.int16)
        self.matches = np.zeros((len(templates), self.capacity, 5), np.int16)
        self.steps = np.arange(self.capacity)
        self.aliases = np.array([0, 1])
        self.first = self.stop = self.allocated = None
        self.entries = np.zeros(len(templates), np.int64)
        self.expiries = self.entries.copy()
        self.ranked = None

    def extend(self, tones, npre, source, first, stop):
        self.advance(source, first, stop)
        addition = HardBatch([(np.asarray(tones), npre)], self.matches_only)
        addition.advance(source, first, stop)
        batch = HardBatch([*self.templates, (np.asarray(tones), npre)], self.matches_only)
        batch.tags[:] = self.tags
        batch.counts[:len(self.counts)] = self.counts
        batch.counts[batch.count_owner[-1]] = addition.counts[0]
        batch.matches[:-1], batch.matches[-1] = self.matches, addition.matches[0]
        batch.candidates = [*self.candidates, addition.candidates[0]]
        batch.entries[:-1], batch.entries[-1] = self.entries, addition.entries[0]
        batch.expiries[:-1], batch.expiries[-1] = self.expiries, addition.expiries[0]
        batch.first, batch.stop, batch.allocated = self.first, self.stop, self.allocated
        batch.source = source
        return batch

    def advance(self, source, first, stop):
        if (first, stop) == (self.first, self.stop):
            return
        if self.first is None:
            self.first = self.stop = first
        if first < self.first or stop < self.stop:
            raise RuntimeError('control view moved backwards')
        begin = stop + 64 - self.capacity if self.allocated is None else self.allocated
        if stop + 64 - begin > self.capacity:
            raise RuntimeError('undrained hard-score consumer')
        tags = self.steps[:stop + 64 - begin] + begin
        slots = tags % self.capacity
        self.tags[slots] = tags
        self.counts[:, slots] = 0
        self.matches[:, slots] = 0
        self.allocated = stop + 64
        leaving = np.arange(self.first + (self.first & 1), min(first, self.stop), 2)
        entering = np.arange(max(first, self.stop) + (max(first, self.stop) & 1), stop, 2)
        rows = np.concatenate((leaving, entering))
        signs = np.ones(len(rows), np.int8)
        signs[:len(leaving)] = -1
        if len(rows):
            if rows[0] < max(0, source.next_row - source.capacity) or rows[-1] >= source.next_row:
                raise RuntimeError('lost control ring history')
            row_slots = rows % source.capacity
            valid = source.clear[row_slots, 0] >= 2
            rows, row_slots, signs = rows[valid], row_slots[valid], signs[valid]
            if not self.matches_only:
                starts = rows[:, None, None] - self.count_offsets[None, :, None] - self.aliases
                target = starts % self.capacity
                present = self.tags[target] == starts
                targets = (target + self.count_template[None, :, None] * self.capacity)[present]
                weights = np.broadcast_to(signs[:, None, None], starts.shape)[present]
                _accumulate(self.counts, targets, weights)
                self.entries += np.bincount(targets[weights > 0] // self.capacity,
                                             minlength=len(self.counts))[self.count_owner]
                self.expiries += np.bincount(targets[weights < 0] // self.capacity,
                                              minlength=len(self.counts))[self.count_owner]
            tone = source.bins[row_slots, 0] - 29
            starts = rows[:, None, None] - self.hit_offsets[tone, :, None] - self.aliases
            target = starts % self.capacity
            shifts = self.hit_shift[tone, :, None]
            valid = (shifts >= 0) & (self.tags[target] == starts)
            targets = ((target + self.hit_owner[tone, :, None] * self.capacity) * 5 + shifts)[valid]
            weights = np.broadcast_to(signs[:, None, None], starts.shape)[valid]
            np.add.at(self.matches.reshape(-1), targets, weights)
            if self.matches_only:
                owners = targets // (self.capacity * 5)
                passing = self.matches.reshape(-1)[targets] >= self.minimum[owners]
                for target in np.unique(targets[passing]):
                    owner, item = divmod(int(target), self.capacity * 5)
                    self.candidates[owner].add(item)
        self.source = source
        self.first, self.stop = first, stop
        self.ranked = None

    def match_best(self, low, high, index):
        tones, pre = self.templates[index]
        candidates = self.candidates[index]
        if not candidates:
            return None
        items = np.fromiter(candidates, np.int64)
        matches = self.matches[index]
        valid = matches.reshape(-1)[items] >= self.minimum[index]
        candidates.difference_update(items[~valid])
        slots, shifts = np.divmod(items[valid], 5)
        starts = self.allocated - self.capacity + (slots - self.allocated) % self.capacity
        keep = (starts >= low) & (starts <= high)
        starts, slots, shifts = starts[keep], slots[keep], shifts[keep]
        if not len(starts):
            return None
        rows = starts[:, None] + (starts[:, None] & 1) + np.arange(pre, len(tones)) * 64
        c = ((rows >= self.first) & (rows < self.stop)
             & (self.source.clear[rows % self.source.capacity, 0] >= 2)).sum(1)
        m = matches[slots, shifts]
        answer = (m >= 13) | ((c >= 8) & (m == c))
        rank = np.lexsort((starts, shifts, c, -m, ~answer))
        best = rank[0]
        return int(starts[best]), int(shifts[best]) - 2

    def best(self, low, high, index=0):
        lows = low + (self.lengths[index] - self.lengths) * 64
        begin = int(lows.min())
        if high < begin or high - begin >= self.capacity:
            raise RuntimeError('hard-score start bound exceeded')
        if begin < self.allocated - self.capacity or high >= self.allocated:
            raise RuntimeError('lost hard-score start history')
        key = (begin, high)
        if self.ranked != key:
            starts = np.arange(begin, high + 1)
            starts = starts[((starts & 1) != 0) | np.isin(starts, lows)]
            slots = starts % self.capacity
            m = self.matches[:, slots]
            c = self.counts[:, slots][self.count_owner, :, None]
            n = self.sizes[:, None, None]
            answer = (m >= 13) | ((c >= 8) & (m == c))
            rank = ((answer * n + m) * n - c).transpose(0, 2, 1)
            rank = np.where(starts[None, None, :] >= lows[:, None, None], rank, -32768)
            best = rank.reshape(len(self.templates), -1).argmax(1)
            self.winners = starts[best % len(starts)]
            self.shifts = best // len(starts) - 2
            self.ranked = key
        return int(self.winners[index]), int(self.shifts[index])


class HardScores:
    capacity = 8192

    def __init__(self, tones, npre, batch=None, index=0):
        self.tones = np.asarray(tones, np.int32)
        self.npre = npre
        self.batch = batch or HardBatch([(self.tones, npre)])
        self.index = index

    def advance(self, source, first, stop):
        self.batch.advance(source, first, stop)

    def best(self, low, high):
        return self.batch.best(low, high, self.index)

    @property
    def counts(self):
        return self.batch.counts[self.batch.count_owner[self.index]]

    @property
    def matches(self):
        return self.batch.matches[self.index]

    @property
    def entries(self):
        return int(self.batch.entries[self.index])

    @property
    def expiries(self):
        return int(self.batch.expiries[self.index])

    @property
    def first(self):
        return self.batch.first

    @property
    def stop(self):
        return self.batch.stop


class SoftScores:
    capacity = 8192

    def __init__(self, tones, shift):
        self.tones = np.asarray(tones, np.int64)
        self.cols = np.clip(self.tones + shift - 29, 0, 69)
        self.offsets = np.arange(len(tones)) * 64
        self.total = np.zeros(self.capacity)
        self.tags = np.full(self.capacity, np.iinfo(np.int64).min, np.int64)
        self.first = self.stop = self.allocated = None
        self.entries = self.expiries = 0

    def _events(self, source, first, stop, sign):
        if stop <= first:
            return
        rows = np.arange(first, stop)
        slots = source.interval(first, stop)
        pair = source.power[slots[:, None, None], self.cols].sum(2)
        values = np.log(np.maximum(70 * pair / (source.denominator[slots, None]
                                               + 1e-30), 1e-9) / 2)
        starts = rows[:, None] - self.offsets
        target = starts % self.capacity
        valid = self.tags[target] == starts
        _accumulate(self.total, target[valid], sign * values[valid])
        if sign > 0:
            self.entries += int(valid.sum())
        else:
            self.expiries += int(valid.sum())

    def advance(self, source, first, stop):
        if self.first is None:
            self.first = self.stop = first
        if first < self.first or stop < self.stop:
            raise RuntimeError('soft control view moved backwards')
        begin = stop - self.capacity if self.allocated is None else self.allocated
        if stop - begin > self.capacity:
            raise RuntimeError('undrained soft-score consumer')
        tags = np.arange(begin, stop)
        self.tags[tags % self.capacity] = tags
        self.total[tags % self.capacity] = 0
        self.allocated = stop
        self._events(source, self.first, min(first, self.stop), -1)
        self._events(source, max(first, self.stop), stop, 1)
        self.first, self.stop = first, stop

    def best(self, first, stop):
        starts = np.arange(first, stop - int(self.offsets[-1]))
        if not len(starts):
            return 0., first
        slots = starts % self.capacity
        if not np.array_equal(self.tags[slots], starts):
            raise RuntimeError('lost soft-score history')
        best = int(self.total[slots].argmax())
        return float(self.total[slots[best]] / len(self.tones)), int(starts[best])


class PairScores:
    capacity = 8192

    def __init__(self, rule, tones, offsets):
        self.rule = rule
        self.tones = np.asarray(tones)
        self.offsets = np.asarray(offsets)
        self.tags = np.full(self.capacity, np.iinfo(np.int64).min, np.int64)
        self.counts = np.zeros((self.capacity, 3), np.int16)
        self.lead = np.zeros(self.capacity, bool)
        self.stop = None
        self.entries = 0

    def advance(self, source, first, stop):
        begin = first if self.stop is None else self.stop
        if stop < begin or stop - begin > self.capacity:
            raise RuntimeError('undrained pair-score consumer')
        tags = np.arange(stop - self.capacity if self.stop is None else begin, stop)
        self.tags[tags % self.capacity] = tags
        self.counts[tags % self.capacity] = 0
        self.lead[tags % self.capacity] = False
        rows = np.arange(begin, stop)
        slots = source.interval(begin, stop)
        bins = source.bins[slots, :2]
        clear = source.clear[slots]
        on = source.live[slots]
        both = on & (clear[:, 1] >= 2)
        starts = rows[:, None] - self.offsets
        target = starts % self.capacity
        valid = self.tags[target] == starts
        found = (bins[:, None, :, None] == self.tones[None, :, None, :]).any(3).sum(2)
        if self.rule == 'ack':
            lone = on & ~both & (clear.sum(1) >= 2)
            crowded = both[:, None] & (found == 1)
            count = np.where(both[:, None], np.where(crowded, 1, 2), lone[:, None])
            matched = np.where(both[:, None], found,
                               lone[:, None] & (bins[:, 0, None, None] == self.tones).any(2))
            values = np.stack((count, matched, crowded), axis=2)
            self.lead[(rows - self.offsets[0]) % self.capacity] = count[:, 0] > 0
        else:
            exact = found == 2
            values = np.stack((np.broadcast_to(both[:, None], exact.shape), exact,
                               np.zeros_like(exact)), axis=2)
            self.lead[(rows - self.offsets[0]) % self.capacity] = exact[:, 0]
        _accumulate(self.counts, (target[valid, None] * 3 + np.arange(3)).ravel(),
                    values[valid].ravel())
        self.entries += int(valid.sum())
        self.stop = stop

    def held(self, first, stop, minimum=0, crowded=0):
        starts = np.arange(first, max(first, stop - int(self.offsets[-1])))
        slots = starts % self.capacity
        if not np.array_equal(self.tags[slots], starts):
            raise RuntimeError('lost pair-score history')
        c, m, noise = self.counts[slots].T
        if self.rule == 'ack':
            return self.lead[slots] & (c >= minimum) & (m == c) & (noise <= crowded)
        return self.lead[slots] & (c == len(self.offsets)) & (m >= minimum)


class ControlSchedule:
    capacity = 8192
    hard_limit = 19

    def __init__(self, origin=0, generation=0, offset=0.0, retained=None):
        if not np.isfinite(offset):
            raise ValueError('nonfinite control grid')
        self.retained = retained
        self.origin = int(origin)
        self.generation = int(generation)
        self.offset = float(offset)
        self.window = np.hanning(2048)[::8]
        if offset:
            self.window = self.window * np.exp(-2j * np.pi * offset * np.arange(256) / 256)
        self.low = np.empty(512)
        self.windows = sliding_window_view(self.low, 256)[::4]
        self.low_n = 0
        self.low_first = 0
        self.raw_n = 0
        self.prefix = np.zeros(4096)
        self.power_sum = 0.0
        self.next_row = 0
        self.tags = np.full(self.capacity, -1, np.int64)
        self.bins = np.empty((self.capacity, 3), np.int32)
        self.clear = np.empty((self.capacity, 2))
        self.residual = np.empty(self.capacity)
        self.power = np.empty((self.capacity, 70))
        self.denominator = np.empty(self.capacity)
        self.live = np.empty(self.capacity, bool)
        self.transforms = 0
        self.delivered = 0
        # Logical cache rows above exclude the scratch FFT rows counted here.
        self.fft_rows = 0
        self.rows = np.arange(32)
        self.prefix_steps = np.arange(1, 1025)
        self.skirts = np.clip(np.arange(70)[:, None] + [-1, 0, 1], 0, 69)
        self.live_floor = (10 ** (-50 / 20) - 1e-15) ** 2
        self.contexts = {}
        self.hard = {}
        self.soft = {}
        self.pairs = {}
        self.exact = {}
        self.rebuilds = 0

    def set_context(self, reader, context):
        if self.contexts.get(reader) == context:
            return
        self.contexts[reader] = context
        self.hard = {k: v for k, v in self.hard.items() if k[0] != reader}
        self.soft = {k: v for k, v in self.soft.items() if k[0] != reader}
        self.pairs = {k: v for k, v in self.pairs.items() if k[0] != reader}

    def view(self, start, end, reader='answer'):
        view = ControlView(self, start, end, reader)
        if self.retained is not None:
            self._ensure(view.first, view.stop)
        return view

    def rebuild(self, offset, retained):
        if self.retained is not None:
            replacement = ControlSchedule(self.origin, self.generation, offset, self.retained)
            replacement.raw_n, replacement.next_row = self.raw_n, self.next_row
            replacement.transforms, replacement.delivered = self.transforms, self.delivered
            replacement.fft_rows = self.fft_rows
            replacement.rebuilds = self.rebuilds + 1
            self.__dict__.update(replacement.__dict__)
            return
        old_count = self.transforms
        old_fft_rows = self.fft_rows
        old_rebuilds = self.rebuilds
        first = max(0, self.next_row - self.capacity)
        end = self.next_row
        replacement = ControlSchedule(self.origin, self.generation, offset)
        for row in range(first, end, 32):
            stop = min(end, row + 32)
            a = row * 4 + 8
            n = (stop - row - 1) * 4 + 256
            low = retained._low[np.arange(a, a + n) % retained._low_capacity]
            replacement.low[:n] = low
            replacement.low_n = n
            replacement.low_first = a
            replacement.next_row = row
            replacement.raw_n = (stop - 1) * 32 + 2112
            replacement.feed(np.empty(0), np.empty(0))
            slots = np.arange(row, stop) % self.capacity
            replacement.live[slots] = self.live[slots]
        replacement.low_n = self.low_n
        replacement.low[:self.low_n] = self.low[:self.low_n]
        replacement.low_first = self.low_first
        replacement.raw_n = self.raw_n
        replacement.prefix[:] = self.prefix
        replacement.power_sum = self.power_sum
        replacement.transforms += old_count
        replacement.fft_rows += old_fft_rows
        replacement.rebuilds = old_rebuilds + 1
        self.__dict__.update(replacement.__dict__)

    def feed(self, low, raw):
        if len(raw) > 1024 or len(low) > 128:
            raise ValueError('feed shared-front blocks of at most 1024 samples')
        if self.retained is not None:
            first = self.next_row
            self.raw_n += len(raw)
            self.next_row = max(0, (min(self.raw_n, self.retained._low_count * 8) - 2112) // 32 + 1)
            return first, self.next_row
        prefix = np.cumsum(raw * raw) + self.power_sum
        if len(prefix):
            self.prefix[(self.raw_n + self.prefix_steps[:len(raw)]) % 4096] = prefix
            self.power_sum = prefix[-1]
        self.raw_n += len(raw)
        self.low[self.low_n:self.low_n + len(low)] = low
        self.low_n += len(low)
        first = self.next_row
        start = first * 4 + 8 - self.low_first
        count = max(0, (self.low_n - start - 256) // 4 + 1)
        count = min(count, max(0, (self.raw_n - 2112) // 32 + 1 - first))
        if not count:
            return first, first
        if count > 32:
            raise RuntimeError('undrained control rows')
        rows = self.rows[:count]
        slots = (first + rows) % self.capacity
        windows = self.windows[start // 4:start // 4 + count]
        self._spectra(windows, slots)
        starts = (first + rows) * 32
        rms2 = (self.prefix[(starts + 2048) % 4096] - self.prefix[starts % 4096]) / 2048
        self.live[slots] = rms2 > self.live_floor
        self.tags[slots] = first + rows
        self.next_row += count
        self.transforms += count
        self.delivered += count
        used = start + count * 4
        self.low[:self.low_n - used] = self.low[used:self.low_n]
        self.low_n -= used
        self.low_first += used
        return first, self.next_row

    def _spectra(self, windows, slots):
        rows = np.arange(len(slots))
        # PocketFFT's vector and scalar tail paths can round differently. Pad
        # batches to eight rows, covering the current float64 SIMD widths
        # (1, 2, 4, 8), so callback/view boundaries cannot select a scalar tail.
        # At most seven scratch rows are transformed and never enter the cache.
        count = len(slots)
        block = np.empty(((count + 7) // 8 * 8, 256), dtype=self.window.dtype)
        np.multiply(windows, self.window, out=block[:count])
        block[count:] = 0
        spectrum = (np.fft.fft(block, axis=1) if self.offset
                    else np.fft.rfft(block, axis=1))[:count]
        self.fft_rows += len(block)
        full = np.abs(spectrum[:, 28:100]) * 8
        mag = full[:, 1:71].copy()
        power = mag * mag
        self.power[slots] = power
        self.denominator[slots] = power.sum(axis=1)
        peaks = np.empty((len(slots), 3))
        for k in range(3):
            j = mag.argmax(axis=1)
            self.bins[slots, k] = j + 29
            peaks[:, k] = mag[rows, j]
            if k == 0:
                left, mid, right = full[rows, j], full[rows, j + 1], full[rows, j + 2]
                self.residual[slots] = 2 * (right - left) / (left + 2 * mid + right + 1e-20)
            if k < 2:
                mag[rows[:, None], self.skirts[j]] = 0
        self.clear[slots] = 20 * np.log10((peaks[:, :2] + 1e-20) / (peaks[:, 1:] + 1e-20))

    def _ensure(self, first, stop):
        if first < max(0, self.next_row - self.capacity):
            raise RuntimeError('lost control ring history')
        source = self.retained
        rows = np.arange(first, stop)
        rows = rows[self.tags[rows % self.capacity] != rows]
        boundaries = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            for at in range(begin, end, 256):
                block = rows[at:min(end, at + 256)]
                if not len(block):
                    continue
                slots = block % self.capacity
                first = int(block[0]) * 4 + 8
                length = (len(block) - 1) * 4 + 256
                if first < max(0, source._low_count - source._low_capacity):
                    raise RuntimeError('lost retained control samples')
                offset = first % source._low_capacity
                low = source._low[offset:offset + length]
                if len(low) < length:
                    low = np.concatenate((low, source._low[:length - len(low)]))
                self._spectra(sliding_window_view(low, 256)[::4], slots)
                starts = block * 32
                rms2 = (source._power[(starts + 2048) % len(source._power)]
                        - source._power[starts % len(source._power)]) / 2048
                self.live[slots] = rms2 > self.live_floor
                self.tags[slots] = block
        self.transforms += len(rows)
        self.delivered += len(rows)

    def interval(self, first, stop):
        if first < max(0, self.next_row - self.capacity) or stop > self.next_row:
            raise RuntimeError('lost control ring history')
        slots = np.arange(first, stop) % self.capacity
        if not np.array_equal(self.tags[slots], np.arange(first, stop)):
            raise RuntimeError('invalid control row tags')
        return slots


class ControlView:
    def __init__(self, source, start, end, reader):
        self.source = source
        self.start, self.end, self.reader = int(start), int(end), reader
        self.first = max(0, (self.start - source.origin + 31) // 32)
        self.stop = min(source.next_row, max(self.first, (end - source.origin - 2112) // 32 + 1))
        self._arrays = None

    def payload_fit(self, tones, npre, require_match=False):
        key = (self.reader, tuple(tones), npre, require_match)
        source = self.source
        score = source.hard.get(key)
        if score is None:
            active = sum(k[0] == self.reader for k in source.hard)
            if active >= (1 if self.reader == 'request' else source.hard_limit):
                raise RuntimeError('hard control template registry overflow')
            keys = [k for k in source.hard if k[0] == self.reader and k[3] == require_match] + [key]
            batch = (source.hard[keys[0]].batch.extend(tones, npre, source, self.first, self.stop)
                     if len(keys) > 1 else HardBatch([(np.asarray(tones), npre)], require_match))
            source.hard.update({k: HardScores(k[1], k[2], batch, j) for j, k in enumerate(keys)})
            score = source.hard[key]
        score.advance(source, self.first, self.stop)
        low = (self.start - source.origin + 31) // 32 - (len(tones) - 1) * 64
        high = (self.end - source.origin - 2080) // 32
        if high < low or self.stop <= self.first:
            return np.full(len(tones), -1, np.int32), 0
        if require_match:
            best = score.batch.match_best(low, high, score.index)
            if best is None:
                return np.full(len(tones), -1, np.int32), 0
            at, shift = best
        else:
            at, shift = score.best(low, high)
        indices = np.rint((at * 32 + np.arange(len(tones)) * 2048 + 16) / 32).astype(np.int64)
        slots = indices % source.capacity
        valid = ((indices >= self.first) & (indices < self.stop)
                 & (source.clear[slots, 0] >= 2))
        heard = np.where(valid, source.bins[slots, 0] - shift, -1).astype(np.int32)
        return heard, source.origin + at * 32 - self.start

    def frame_fit(self, tones, shift):
        key = (self.reader, tuple(map(tuple, tones)), shift)
        source = self.source
        score = source.soft.get(key)
        if score is None:
            if sum(k[0] == self.reader for k in source.soft) >= 4:
                raise RuntimeError('soft control template registry overflow')
            score = source.soft[key] = SoftScores(tones, shift)
        score.advance(source, self.first, self.stop)
        value, at = score.best(self.first, self.stop)
        return value, source.origin + at * 32 - self.start

    def pair_held(self, rule, tones, offsets, minimum=0, crowded=0):
        key = (self.reader, rule, tuple(map(tuple, tones)), tuple(offsets))
        score = self.source.pairs.get(key)
        if score is None:
            if sum(k[0] == self.reader for k in self.source.pairs) >= 4:
                raise RuntimeError('pair control template registry overflow')
            score = self.source.pairs[key] = PairScores(rule, tones, offsets)
        score.advance(self.source, self.first, self.stop)
        return score.held(self.first, self.stop, minimum, crowded)

    def exact_candidates(self, templates, shifts, pair=False):
        """exact_control_candidates over this view, from window starts kept per
        template set: each absolute row's window is scored once. ``pair`` reads
        the sorted top two bins and the second's clearance, else the top bin and
        its own. None where the view's rows are not contiguous."""
        source = self.source
        offset = self.start - source.origin
        if offset % 32 == 16:
            return None
        templates = np.asarray(templates)
        count, width = templates.shape[:2]
        carriers = 2 if pair else 1
        n = max(0, (self.end - self.start - 2048) // 32 + 1) - (width - 1) * 64
        if n <= 0:
            return np.zeros(count, bool)
        first = int(np.rint(offset / 32))
        low, high = max(first, self.first), min(first + n, self.stop - (width - 1) * 64)
        key = (pair, templates.tobytes(), templates.shape, tuple(np.asarray(shifts).tolist()))
        entry = source.exact.get(key)
        if entry is None or not entry[0] <= low <= entry[1]:
            entry = source.exact[key] = [low, low, [[] for _ in range(count)]]
        _, scored, hits = entry
        if scored < high:
            starts = np.arange(scored, high)
            rows = starts[:, None] + np.arange(width) * 64
            slots = rows % source.capacity
            valid = (source.clear[slots, carriers - 1] >= 2).all(axis=1)
            bins = source.bins[slots[valid], :carriers]
            if pair:
                bins = np.sort(bins, axis=2)
            weights = 128 ** np.arange(width * carriers, dtype=np.int64)
            signatures = bins.reshape(-1, width * carriers) @ weights
            wanted = ((templates.reshape(count, -1) @ weights)[:, None]
                      + np.asarray(shifts) * weights.sum())
            for j in range(count):
                hits[j].extend(starts[valid][np.isin(signatures, wanted[j])].tolist())
            entry[1] = high
        return np.array([bisect.bisect_left(h, high) > bisect.bisect_left(h, low) for h in hits])

    def _materialize(self):
        if self._arrays is None:
            n = max(0, (self.end - self.start - 2048) // 32 + 1)
            rows = np.rint((self.start - self.source.origin + np.arange(n) * 32) / 32).astype(np.int64)
            valid = (rows >= self.first) & (rows < self.stop)
            slots = rows % self.source.capacity
            self._arrays = (np.where(valid[:, None], self.source.bins[slots], -1),
                            np.where(valid[:, None], self.source.clear[slots], -np.inf),
                            np.where(valid, self.source.residual[slots], 0),
                            valid & self.source.live[slots])
        return self._arrays

    def __getitem__(self, index):
        return self._materialize()[index]

    def __iter__(self):
        return iter(self._materialize()[:3])

    @property
    def live(self):
        return self._materialize()[3]
