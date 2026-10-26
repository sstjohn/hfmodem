# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bound native station diagnostics while preserving lifetime event indexes.

The native driver uses len(history) as an append counter and saves indexes for
later slices. A plain deque would stop that counter at capacity. These histories
keep absolute indexes and counts, retaining only the latest diagnostic objects.
They contain no pending payload, ARQ decisions or original recovery bytes.
"""
from collections import deque
import operator


class History:
    def __init__(self, capacity=4096):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('history capacity must be a positive integer')
        self._items = deque(maxlen=capacity)
        self.count = 0

    @property
    def first(self):
        return self.count - len(self._items)

    @property
    def retained(self):
        return len(self._items)

    def __len__(self):
        """Lifetime append count, including objects no longer retained."""
        return self.count

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def append(self, value):
        self._items.append(value)
        self.count += 1

    def clear(self):
        self._items.clear()
        self.count = 0

    def tail(self):
        # Avoid list(self): its length hint is the lifetime count.
        return list(self._items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            if step > 0:
                if start < self.first:
                    start += ((self.first - start + step - 1) // step) * step
            else:
                stop = max(stop, self.first - 1)
            return [self._items[n - self.first] for n in range(start, stop, step)]
        index = operator.index(key)
        if index < 0:
            index += self.count
        if not self.first <= index < self.count:
            raise IndexError('diagnostic history index was retired or is out of range')
        return self._items[index - self.first]


class WordHistory(History):
    def __init__(self, capacity, *, p1_word):
        super().__init__(capacity)
        self._p1_word = p1_word
        self.p1_words = 0

    def append(self, value):
        self.p1_words += bool(self._p1_word(value))
        super().append(value)

    def clear(self):
        super().clear()
        self.p1_words = 0


class Marks(set):
    """Sparse absolute event indexes, bounded by their associated history."""
    def __init__(self, history):
        super().__init__()
        self.history = history

    def add(self, index):
        self.difference_update(n for n in tuple(self) if n < self.history.first)
        super().add(index)


class RecentMap(dict):
    """Bound display-only values indexed by monotonically offered cycle."""
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        while len(self) > self.capacity:
            del self[next(iter(self))]


def bind_grid(grid, *, capacity):
    """Entry diagnostics never drive timing, acceptance or byte settlement."""
    if grid.entry_answers or grid.entry_tracked or grid.p2_keyed_cycles:
        raise ValueError('bounded grid diagnostics require a fresh grid')
    grid.entry_answers = History(capacity)
    grid.p2_keyed_cycles = History(capacity)
    grid.entry_tracked = RecentMap(capacity)


def bind(tx, rx, *, capacity=4096):
    """Only a fresh native station transport may replace diagnostic storage."""
    names = ('p1_seq', 'seq_sent', 'slots_used', 'keyed', 'leads')
    if any(getattr(tx, name) for name in names) or rx.cs_log or rx._anchored_at:
        raise ValueError('bounded diagnostics require a fresh native transport')
    tx._bounded_diagnostics = True
    tx._diagnostic_capacity = capacity
    for name in names:
        setattr(tx, name, History(capacity))
    tx.p1_breakin_at = Marks(tx.p1_seq)
    tx.breakin_at = Marks(tx.seq_sent)
    from .onair import _cs_errors
    from .spec import Protocol
    rx.cs_log = WordHistory(capacity, p1_word=lambda event:
        event.protocol == Protocol.PACTOR1 and (event.cs is not None or event.spare is not None)
        and _cs_errors(event) == 0)
    # Duplicate suppression needs the rolling decoder's recent anchors rather
    # than a session transcript. 512 native cycle anchors exceed its RX window.
    rx._anchored_at = History(max(512, capacity))
