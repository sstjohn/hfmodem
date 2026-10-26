# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Causal BW500 candidate grouping and bounded decoded-over carry."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
from scipy.ndimage import maximum_filter1d

from . import varahf500 as rx
from .bw500_features import FeatureTrack
from .bw500_refine import demod as demod_local

BLOCK = 1024
CARRY = max((2 * r.ncols + r.lead) * r.dw for r in rx.INDEX_RECORDS.values()) + rx.FS
ANSWER_GAP = {1: .230, 2: .180, 3: .179, 4: .089}


@dataclass
class Epoch:
    start: int
    score: float
    available: int
    bands: tuple = ()


@dataclass
class Candidate:
    level: int
    start: int
    score: float
    decision: int
    last: int
    epochs: tuple = ()


@dataclass
class Over:
    candidate: Candidate
    generation: int = 0
    frames: list = field(default_factory=list)
    status: str = 'open'
    end: int = 0
    first_end: int = 0  # Aligned first-frame boundary, never the speculative pair end.
    decision: int = 0
    refined_score: float = 0.
    onsets: tuple = ()
    next_check: int = 0
    pair: bool = False
    alignment: object = None
    epoch_try: int = 0
    low_rate: bool = False
    full_onsets: tuple = ()
    work: object = None
    steps: int = 0

    @property
    def result(self):
        return rx.DecodeResult(self.frames, [f for f in self.frames if not f.is_connect],
                               observation_complete=self.status == 'closed', refined_lock=self.refined_score)



class Grouper:
    def __init__(self):
        self.groups = {}
        self.tail = (np.empty(0, np.int64), np.empty((2, 0)))
        self.next_epoch = None
        self.last_available = 0
        self.count = 0

    def feed(self, batches, eof=False, available=None):
        ready = []
        for level, starts, scores, feature_available in batches:
            if level == 4:
                self.last_available = feature_available
                old, values = self.tail
                st = np.concatenate((old, starts))
                sc = np.concatenate((values, scores), axis=1)
                if not len(st):
                    continue
                stop = max(0, len(st) - 4)
                begin = 0 if self.next_epoch is None else int(np.searchsorted(st, self.next_epoch))
                starts = st[begin:stop]
                if len(starts):
                    self.next_epoch = int(starts[-1]) + 64
                self.tail = st[-8:].copy(), sc[:, -8:].copy()
                if sc.max(axis=1).min() < .15:
                    self._group(level, starts, None, (), feature_available, ready)
                    continue
                bands = maximum_filter1d(sc, 9, axis=1, mode='constant')
                scores = bands[:, begin:stop].min(axis=0)
                values = bands[:, begin:stop].T
            else:
                values = [()] * len(starts)
            self._group(level, starts, scores, values, feature_available, ready)
        if eof:
            st, sc = self.tail
            if len(st):
                begin = 0 if self.next_epoch is None else int(np.searchsorted(st, self.next_epoch))
                bands = maximum_filter1d(sc, 9, axis=1, mode='constant')
                feature_available = available if available is not None else self.last_available
                self._group(4, st[begin:], bands[:, begin:].min(axis=0),
                            bands[:, begin:].T, feature_available, ready)
            self.tail = (np.empty(0, np.int64), np.empty((2, 0)))
            self.next_epoch = None
            if available is not None:
                for group in self.groups.values():
                    group.decision = available
            ready.extend(self.groups.values())
            self.groups.clear()
        self.count += len(ready)
        if available is not None:
            for group in ready:
                group.decision = available
        return ready

    def _group(self, level, starts, scores, bands, available, ready):
        threshold = .15 if level == 4 else 12
        passing = () if scores is None else np.flatnonzero((starts >= -8192) & (scores >= threshold))
        if not len(passing):
            group = self.groups.get(level)
            if group is not None and len(starts) and starts[-1] - group.last > 1024:
                group.decision = available
                ready.append(self.groups.pop(level))
            return
        indices = np.unique(np.append(passing, len(starts) - 1))
        for i in indices:
            at, score, band = starts[i], scores[i], bands[i]
            at, score = int(at), float(score)
            group = self.groups.get(level)
            if group is not None and at - group.last > 1024:
                group.decision = available
                ready.append(group)
                del self.groups[level]
                group = None
            if at < -8192 or score < threshold:
                continue
            epoch = Epoch(at, score, available, tuple(float(v) for v in band))
            if group is None:
                self.groups[level] = Candidate(level, at, score, available, at, (epoch,))
                continue
            group.last = at
            epochs = [e for e in group.epochs if e.start != at] + [epoch]
            group.epochs = tuple(sorted(epochs, key=lambda e: (-e.score, e.start))[:2])
            group.start, group.score = group.epochs[0].start, group.epochs[0].score


class RawCarry:
    def __init__(self, capacity=CARRY + 8192):
        self.capacity = capacity
        self.buffer = np.empty(2 * capacity)
        self.lo = self.hi = self.origin = 0

    @property
    def samples(self):
        return self.origin + self.hi - self.lo

    @property
    def raw(self):
        return self.buffer[self.lo:self.hi]

    def origin_after(self, n):
        return self.origin + max(0, self.hi - self.lo + n - self.capacity)

    def feed(self, x):
        if len(x) > self.capacity:
            raise ValueError('raw carry must be fed in bounded blocks')
        drop = max(0, self.hi - self.lo + len(x) - self.capacity)
        self.lo += drop
        self.origin += drop
        if self.hi + len(x) > len(self.buffer):
            n = self.hi - self.lo
            self.buffer[:n] = self.buffer[self.lo:self.hi]
            self.lo, self.hi = 0, n
        self.buffer[self.hi:self.hi + len(x)] = x
        self.hi += len(x)


class Receiver:
    """Streaming base receiver.

    ``observe_pairs`` is for passive listeners without the link's pairing history.
    It tries aligned partners, publishes only validated frames, and leaves an
    unread partner incomplete. Default callers retain their ``may_pair`` contract.
    """
    def __init__(self, *, front=None, generation=0, observe_pairs=False):
        self.front = front if front is not None else FeatureTrack()
        self.front.feed(np.zeros(8192))
        self.generation = generation
        self.raw = RawCarry()
        self.low = RawCarry(capacity=(CARRY + 8192) // 8)
        self.low.feed(getattr(self.front, 'last_low', np.empty(0)))
        self.low_samples = getattr(self.front, 'samples', 0)
        self.grouper = Grouper()
        self.pending = []
        self.history = []
        self.overflows = self.refinements = self.turbo_attempts = self.expired = 0
        self.consumed = -1
        # Passive listeners have no ARQ history predicting a partner. Only CRC
        # validated intervals own audio; a failed partner cannot consume a later over.
        self.observe_pairs = observe_pairs
        self.confirmed = []

    def feed(self, samples, *, batches=None, may_pair=False,
             eof=False, decision_sample=None, _budget=None):
        samples = np.asarray(samples, dtype=float)
        if _budget is None:
            _budget = [max(1, -(-len(samples) // BLOCK))]
        if decision_sample is None:
            decision_sample = self.raw.samples + len(samples)
        if batches is None and len(samples) > BLOCK:
            out = []
            for j in range(0, len(samples), BLOCK):
                out.extend(self.feed(samples[j:j + BLOCK], may_pair=may_pair,
                                     decision_sample=decision_sample, _budget=None if eof else _budget))
            if eof:
                out.extend(self.feed(np.empty(0), may_pair=may_pair, eof=True,
                                     decision_sample=decision_sample, _budget=_budget))
            return out
        # An over the carry is about to drop has starved past its window: it
        # goes unread rather than taking the receiver down with it.
        out = []
        origin = self.raw.origin_after(len(samples))
        if self.observe_pairs:
            self.confirmed = [(a, b) for a, b in self.confirmed if b > origin]
        for over in sorted((p for p in self.pending if p.candidate.start < origin),
                           key=lambda p: p.candidate.start):
            if self.observe_pairs:
                if over.status == 'open':
                    over.status = 'truncated'
                self._publish_observed(over, out, decision_sample)
            self.retire(over)
            self.expired += 1
        self.raw.feed(samples)
        if batches is None:
            batches = self.front.feed(samples)
        if getattr(self.front, 'samples', 0) > self.low_samples:
            self.low.feed(self.front.last_low)
            self.low_samples = self.front.samples
        batches = [(lv, starts - 8192, scores, available - 8192)
                   for lv, starts, scores, available in batches]
        for candidate in self.grouper.feed(batches, eof=eof, available=decision_sample):
            candidate.start = max(0, candidate.start)
            if self._owned(candidate.start):
                continue
            if any(candidate.level == p.candidate.level
                   and abs(candidate.start - p.candidate.start) <= 1024 for p in self.pending):
                continue
            if len(self.pending) == 4:
                self.overflows += 1
                if not self.observe_pairs:
                    raise RuntimeError('BW500 pending candidate overflow')
                # Keep the four-candidate bound without resetting the whole
                # passive reader and losing already validated text.
                oldest = min(self.pending, key=lambda p: p.candidate.start)
                if oldest.status == 'open':
                    oldest.status = 'truncated'
                self._publish_observed(oldest, out, decision_sample)
                self.retire(oldest)
            rec = rx.INDEX_RECORDS.get(candidate.level)
            one = (rec.lead + rec.ncols) * rec.dw if rec else (rx.NSYM + 9) * rx.H
            self.pending.append(Over(candidate, generation=self.generation,
                                     next_check=candidate.start + one + rx._PAD))
        # Least-served first, newest on a tie: a failing candidate's resumable
        # decode holds the budget neither from an over behind it, which retires
        # it once it decodes, nor from the over it sits inside, whose partner
        # frame would otherwise decode alone and retire it.
        for over in sorted(self.pending, key=lambda o: (o.steps, -o.candidate.start)):
            if self.observe_pairs and over.status != 'open':
                continue
            if self._owned(over.candidate.start):
                self.retire(over)
                continue
            if any(p is not over and p.frames and any(f.crc_ok for f in p.frames)
                   and p.candidate.start < over.candidate.start
                   < (p.first_end if self.observe_pairs else p.end + rx.NSYM * rx.H)
                   for p in self.pending):
                continue
            if self.raw.samples < over.next_check and not eof:
                continue
            if not _budget[0] and not eof:
                continue
            _budget[0] -= 1
            over.steps += 1
            pairing = (True if self.observe_pairs else
                       may_pair(over.candidate.level, over.candidate.start)
                       if callable(may_pair) else may_pair)
            self._advance(over, pairing, eof)
            if over.status != 'open' and not self.observe_pairs:
                over.decision = decision_sample
                out.append(over)
                self.pending.remove(over)
                self.history.append(over)
                self.history = self.history[-4:]
                if any(f.crc_ok for f in over.frames):
                    self.consumed = max(self.consumed, over.end)
        if self.observe_pairs:
            # Decode later candidates while a possible partner is pending, but
            # publish in time order. This lets the aligned pair claim its partner
            # before an overlapping independent candidate can duplicate it.
            for over in sorted(self.pending, key=lambda o: o.candidate.start):
                if over.status == 'open':
                    break
                self._publish_observed(over, out, decision_sample)
                self.retire(over)
        return out

    def _owned(self, start):
        if self.observe_pairs:
            return any(a <= start < b for a, b in self.confirmed)
        return start < self.consumed

    def _publish_observed(self, over, out, decision_sample):
        fresh = []
        for i, frame in enumerate(over.frames):
            if not frame.crc_ok:
                continue
            a = over.candidate.start if i == 0 else over.first_end
            b = over.first_end if i == 0 else over.end
            if b <= a:
                continue
            # Candidate alignment can differ slightly. Ownership follows the
            # same frame interval, not payload equality (retries remain visible).
            if any(min(b, hi) - max(a, lo) > (b - a) / 2 for lo, hi in self.confirmed):
                continue
            fresh.append(frame)
            self.confirmed.append((a, b))
        if fresh:
            # A failed speculative partner remains incomplete, even though its
            # validated first frame is useful passive evidence.
            published = replace(over, frames=fresh, decision=decision_sample, work=None)
            out.append(published)
            self.history.append(published)
            self.history = self.history[-4:]

    def retire(self, over):
        if over.work is not None:
            over.work.close()
            over.work = None
        self.pending.remove(over)

    def _advance(self, over, may_pair, eof):
        self._draining = eof
        if over.work is None:
            over.work = self._advance_work(over, may_pair, eof)
        while True:
            try:
                next(over.work)
            except StopIteration:
                over.work = None
                return
            if not eof:
                return

    def _advance_work(self, over, may_pair, eof):
        origin, x = self.raw.origin, self.raw.raw.copy()
        low_origin, low = self.low.origin * 8 - 8256, self.low.raw.copy()
        start = max(0, over.candidate.start - origin)
        level = over.candidate.level
        rec = rx.INDEX_RECORDS.get(level)
        slot = rec.ncols * rec.dw if rec else rx.NSYM * rx.H
        if not over.frames:
            self.refinements += 1
            if rec:
                for trial in (start, max(0, start - rec.dw // 2)):
                    alignment = rx._l3_align(x, trial, trial + (rec.lead + rec.ncols) * rec.dw, level)
                    frames = rx._l3_frames(trial, trial + (rec.lead + rec.ncols) * rec.dw,
                                          *alignment, level, pooled=False)
                    self.turbo_attempts += 1
                    if frames and frames[0].crc_ok:
                        break
                if not frames:
                    frames = [rx.FrameResult(start=trial, onset=(0,), c0=rec.lead,
                                            self_consistency=0., crc_ok=False, frame_bytes=b'', level=level)]
                over.frames = [replace(f, start=f.start + origin) for f in frames[:1]]
                over.refined_score = frames[0].self_consistency
                over.end = origin + alignment[0] + (alignment[1] + rec.ncols) * rec.dw
                over.alignment = (alignment[0] + origin, *alignment[1:3])
            else:
                cols, onsets, score = demod_local(low, low_origin, over.candidate.start)
                over.low_rate = bool(onsets)
                if not onsets:
                    fixed = (tuple(o - over.candidate.start for o in over.full_onsets)
                             if over.full_onsets else None)
                    cols, offsets, score = yield from rx._demod_aligned_steps(x, start, fixed)
                    onsets = tuple(offsets.values())
                    over.full_onsets = tuple(over.candidate.start + o for o in onsets)
                over.onsets = onsets
                over.refined_score = score
                gh = rx._frame_gh(cols, 0)
                if gh is None:
                    over.status = 'truncated'
                    return
                for c0 in (9, 8, 10, 7, 11):
                    sc, c0, crc, body, setup = (yield from rx._decode_cells_steps(gh, 0, c0))
                    self.turbo_attempts += 1
                    if crc:
                        break
                    yield
                if not crc and over.low_rate and not over.full_onsets:
                    cols, offsets, score = yield from rx._demod_aligned_steps(x, start)
                    over.onsets = tuple(offsets.values())
                    over.full_onsets = tuple(over.candidate.start + o for o in over.onsets)
                    over.low_rate = False
                    over.refined_score = score
                    gh = rx._frame_gh(cols, 0)
                    if gh is not None:
                        for c0 in (9, 8, 10, 7, 11):
                            sc, c0, crc, body, setup = (yield from rx._decode_cells_steps(gh, 0, c0))
                            self.turbo_attempts += 1
                            if crc:
                                break
                            yield
                over.frames = [rx.FrameResult(start=origin + start, onset=over.onsets, c0=c0,
                    self_consistency=sc, crc_ok=crc, frame_bytes=body, is_connect=setup,
                    burst_columns=rx.NSYM + c0)]
                over.end = origin + start + max(over.onsets) + (rx.NSYM + c0) * rx.H - rx.H // 2
            if not any(f.crc_ok for f in over.frames):
                if over.epoch_try + 1 < len(over.candidate.epochs):
                    over.epoch_try += 1
                    epoch = over.candidate.epochs[over.epoch_try]
                    alternate = max(0, epoch.start)
                    if alternate != over.candidate.start and alternate >= self.raw.origin:
                        over.candidate.start, over.candidate.score = alternate, epoch.score
                        over.frames, over.onsets, over.alignment = [], (), None
                        over.first_end = 0
                        over.refined_score, over.end, over.decision = 0., 0, 0
                        over.pair = over.low_rate = False
                        if not all(-rx._ALIGN_EARLY * rx.H <= o - alternate < rx._ALIGN_LATE * rx.H
                                   for o in over.full_onsets):
                            over.full_onsets = ()
                        one = (rec.lead + rec.ncols) * rec.dw if rec else (rx.NSYM + 9) * rx.H
                        over.next_check = alternate + one + rx._PAD
                        if self.raw.samples >= over.next_check or self._draining:
                            yield from self._advance_work(over, may_pair, eof)
                        return
                if rec and not may_pair:
                    frames = rx._l3_frames(trial, trial + (rec.lead + rec.ncols) * rec.dw,
                                          *alignment, level, pooled=True)
                    self.turbo_attempts += 1
                    if frames and frames[0].crc_ok:
                        over.frames = [replace(frames[0], start=frames[0].start + origin)]
            over.first_end = over.end
            frame = over.frames[0]
            over.pair = bool(may_pair and (not frame.crc_ok or (
                not frame.is_connect and not (frame.marker & 0x02))))
            if over.pair:
                over.frames = [replace(frame, frames_in_burst=2)]
        origin, x = self.raw.origin, self.raw.raw.copy()
        start = max(0, over.candidate.start - origin)
        end = over.end - origin
        finish = over.end + (slot if over.pair else 0)
        over.next_check = finish + round(ANSWER_GAP[level] * rx.FS)
        if self.raw.samples < over.next_check and not self._draining:
            return
        if self.raw.samples < finish:
            over.end = finish
            over.status = 'truncated'
            return
        if not over.pair:
            over.status = 'closed'
            return
        if level == 4 and not np.any(x[end + rx._PAD:end + slot]):
            over.end = finish
            over.status = 'truncated'
            return
        if rec:
            off, lead, matched = over.alignment
            off -= origin
            band = rx._l3_band(x, off, lead + 2 * rec.ncols, level, raw=True)
            frames = rx._l3_frames(start, end + slot, off, lead, matched, band, level, pooled=False)
            second = frames[1:] if len(frames) > 1 else []
            if second:
                first_body, first_crc = over.frames[0].frame_bytes, over.frames[0].crc_ok
                if not first_crc:
                    first_body, first_crc = rx._l3_decode(band, lead, level, pooled=True)
                    if first_crc:
                        recovered = replace(over.frames[0], frame_bytes=first_body, crc_ok=True)
                        first_crc = not recovered.is_connect and not (recovered.marker & 0x02)
                if first_crc:
                    partner = second[0]
                    if not partner.crc_ok:
                        body, crc = rx._l3_decode(band, partner.c0, level, pooled=True)
                        if crc:
                            partner = replace(partner, frame_bytes=body, crc_ok=True)
                    if partner.crc_ok:
                        if not over.frames[0].crc_ok:
                            over.frames[0] = replace(over.frames[0], frame_bytes=first_body, crc_ok=True)
                        second = [partner]
        else:
            second_onsets = over.onsets
            if over.low_rate:
                cols, _, _ = demod_local(self.low.raw, self.low.origin * 8 - 8256,
                                         over.candidate.start, over.onsets)
            else:
                cols, _, _ = rx._demod_aligned(x, start, over.onsets)
            gh = rx._frame_gh(cols, 1)
            second = []
            if gh is not None:
                sc, c0, crc, body, setup = (yield from rx._decode_cells_steps(gh, 1, over.frames[0].c0))
                self.turbo_attempts += 1
                if not crc and over.low_rate:
                    fixed = (tuple(o - over.candidate.start for o in over.full_onsets)
                             if over.full_onsets else None)
                    cols, offsets, _ = yield from rx._demod_aligned_steps(x, start, fixed)
                    second_onsets = tuple(offsets.values())
                    over.full_onsets = tuple(over.candidate.start + o for o in offsets.values())
                    gh = rx._frame_gh(cols, 1)
                    if gh is not None:
                        for trial in dict.fromkeys((over.frames[0].c0, 9, 8, 10, 7, 11)):
                            sc, c0, crc, body, setup = (yield from rx._decode_cells_steps(gh, 1, trial))
                            self.turbo_attempts += 1
                            if crc:
                                break
                            yield
                second = [rx.FrameResult(start=start, onset=second_onsets, c0=c0,
                    self_consistency=sc, crc_ok=crc, frame_bytes=body, is_connect=setup)]
        over.frames.extend(replace(f, start=over.frames[0].start, frames_in_burst=2) for f in second)
        over.end = finish
        over.status = 'closed' if any(f.crc_ok for f in second) else 'truncated'


def _batch_base_ended(samples, over):
    frame = over.frames[0]
    starts = over.candidate.start + np.array(over.onsets) + (rx.NSYM + frame.c0) * rx.H
    half = len(rx._LP) // 2
    if len(samples) <= max(starts) + 6 * rx.H + half:
        return False
    cols, _, _ = rx._demod_aligned(samples, over.candidate.start, over.onsets)
    before = np.array([np.mean(abs(cols[sub][frame.c0:rx.NSYM + frame.c0]) ** 2)
                       for sub, _ in rx.SUBBANDS])
    offsets = np.arange(-half, half + 1)
    indices = starts[:, None, None] + np.arange(1, 7)[None, :, None] * rx.H + offsets
    kernel = rx._LP * np.exp(-2j * np.pi * np.array([f for _, f in rx.SUBBANDS])[:, None]
                             * offsets / rx.FS)
    values = np.einsum('skn,sn->sk', samples[indices], kernel)
    return bool(np.all(np.mean(abs(values) ** 2, axis=1) < .25 * before))


def decode(samples, *, expect_pair=None):
    samples = np.asarray(samples)
    receiver = Receiver()
    spans = rx.burst_spans(samples) if len(samples) > rx.BURST_MIN else []

    def paired(level, start):
        if expect_pair is not None:
            return expect_pair(level) if callable(expect_pair) else expect_pair
        rec = rx.INDEX_RECORDS.get(level)
        lead, slot = (rec.lead * rec.dw, rec.ncols * rec.dw) if rec else (9 * rx.H, rx.NSYM * rx.H)
        span = next(((a, b) for a, b in spans if abs(a - start) <= 8192), None)
        return span is None or span[1] - span[0] > lead + 1.5 * slot

    overs = receiver.feed(samples, may_pair=paired, eof=True)
    for over in overs:
        if (expect_pair is None and over.candidate.level == 4 and over.pair and over.status == 'truncated'
                and over.end > len(samples) and over.frames[0].crc_ok
                and not any(abs(a - over.candidate.start) <= 8192 for a, _ in spans)
                and _batch_base_ended(samples, over)):
            over.frames = [replace(over.frames[0], frames_in_burst=1)]
            over.end -= rx.NSYM * rx.H
            over.status = 'closed'
    frames = [f for over in overs for f in over.frames]
    for i, frame in enumerate(frames):
        span = next(((a, b) for a, b in spans if abs(a - frame.start) <= 8192), None)
        if span is not None:
            start, stop = span
            frames[i] = replace(frame, start=start,
                onset=tuple(o + frame.start - start for o in frame.onset),
                burst_columns=(stop - start) // rx.H)
        if frame.crc_ok:
            start, stop = span or (frame.start, frame.start + frame.burst_columns * rx.H)
            frames[i] = replace(frames[i], quality=rx.quality(samples[start:stop]))
    unread_shape = False
    if not frames:
        for start, stop in spans:
            columns = (stop - start) // rx.H
            frames.append(rx.FrameResult(start=start, onset=(), c0=0, self_consistency=0.,
                crc_ok=False, frame_bytes=b'', burst_columns=columns,
                frames_in_burst=rx.frames_carried(columns)))
            unread_shape = True
    return rx.DecodeResult(frames, [f for f in frames if not f.is_connect],
                           observation_complete=all(o.status == 'closed' for o in overs),
                           refined_lock=max((o.refined_score for o in overs), default=0.),
                           unread_shape=unread_shape)
