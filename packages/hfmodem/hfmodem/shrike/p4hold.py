# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import dataclass
from time import monotonic

import numpy as np

from . import p4breakin, p4breakintx, p4frame, p4host, p4rx, p4short, p4shorttx


@dataclass
class Copy:
    onset: int
    offset: int
    cfo: float
    q: float
    llr: np.ndarray
    inline_q: float | None = None
    coherence: float = 0.


class BreakinHold:
    def __init__(self, fs):
        self.fs = fs
        self.events = []
        self.epoch = 0
        self.unread = 0
        self.kind = "body"
        self.acked_counter = None
        self.now = monotonic
        self.decode_times = []
        self.quick_decode_times = []
        self.reset()

    def reset(self, reason=None):
        if reason and self.members:
            self.events.append(dict(event='epoch closed', reason=reason, epoch=self.epoch))
        self._normal_results = {}
        self._normal_deferred = set()
        self._normal_refined = False
        self.members = []
        self.frame = self.codeword = self.work = None
        self.offset = 0
        self.last = None
        self.unread = 0
        self.epoch += 1

    def peer(self, obs):
        kind = 'normal' if obs.kind.value == 'normal' else 'body'
        if (kind != self.kind or (obs.kind.value not in ('body', 'normal') and obs.word != 'Y27')):
            self.reset('peer kind')

    def _select(self, kind):
        if self.kind != kind:
            self.reset('peer kind')
            self.kind = kind

    def status_ok(self, frame, *, combined):
        status = frame.status
        c = self.acked_counter
        counters = {0, 1} if c is None else {0, c, (c + 1) % 4}
        valid = status.counter in counters
        if combined:
            valid &= ((not status.extended or p4host.observed_extension(frame))
                      and not status.source_mode
                      and (self.kind == 'normal' or status.protocol in (0, 1, 2)))
        return valid

    def _decode(self, scores):
        if self.kind == 'body':
            return p4breakin.decode_scores(scores)
        return self._decode_normal(scores, iterations=8)

    def _decode_normal(self, scores, *, iterations):
        frame, polarity = p4short.decode_scores(scores, iterations=iterations)
        self._normal_crc_valid = frame is not None
        if frame is not None and polarity != 1:
            self.events.append(dict(event='polarity-inverted', at=self.last))
            return None
        return frame

    def _codeword(self, frame):
        renderer = p4shorttx if self.kind == 'normal' else p4breakintx
        return renderer.coded_bits(p4frame.encode_field(frame.data, frame.status))

    def _accept(self, frame=None):
        if not self.members:
            return None
        if frame is None:
            frame = self._decode(sum(m.llr for m in self.members))
        if frame is None:
            return None
        if not self.status_ok(frame, combined=len(self.members) > 1):
            self.events.append(dict(event='status-rejected', status=frame.status.byte))
            return None
        codeword = self._codeword(frame)
        keep = [m for m in self.members if p4breakin.distance(m.llr, codeword) <= .40]
        if sum(m.q for m in keep) < .5 * sum(m.q for m in self.members):
            return None
        for member in self.members:
            if not any(member is m for m in keep):
                self.events.append(dict(event='member dropped', at=member.onset))
        self.members = keep
        self.frame, self.codeword = frame, codeword
        self.events.append(dict(event='held', at=self.last, copies=len(keep)))
        return frame

    def inline(self, audio, origin, head):
        self._select("body")
        at = head.stream_sample
        if self.last == at:
            return self.frame
        if self.last is not None:
            cycles = round((at - self.last) / (1.25 * self.fs))
            if cycles < 1 or abs(at - self.last - cycles * 1.25 * self.fs) > .005 * self.fs:
                self.reset('off raster')
        if at < origin:
            return None
        lo = max(0, at-origin-round(.050*self.fs))
        hi = min(len(audio), at-origin+round(.820*self.fs))
        z = p4rx.audio_to_baseband(np.asarray(audio)[lo:hi], self.fs)
        nominal = round((at-origin-lo)*28800/self.fs) + 432*16
        offset = self.offset
        llr = p4breakin.body_scores(z, first_training_start=int(nominal + offset),
                                    frequency_offset=head.cfo_hz)
        self.last = at
        if llr is None:
            return None
        q = float(np.mean(abs(llr)))
        if self.frame is not None:
            if q >= np.median([m.q if m.inline_q is None else m.inline_q for m in self.members]) and p4breakin.distance(llr, self.codeword) > .45:
                self.reset('different body')
                self.last = at
                self.members.append(Copy(at, offset, head.cfo_hz, q, llr, q))
                self.work = (z, nominal, head.cfo_hz, at)
                return None
            self.events.append(dict(event='held-same', at=at))
            return self.frame
        self.members.append(Copy(at, offset, head.cfo_hz, q, llr, q))
        self.work = (z, nominal, head.cfo_hz, at)
        frame = self._accept()
        if frame is None and len(self.members) > 1:
            single = (p4breakin.body_scores(z, first_training_start=int(nominal),
                                           frequency_offset=head.cfo_hz) if offset else llr)
            frame = None if single is None else p4breakin.decode_scores(single)
            if frame is not None and not self.status_ok(frame, combined=False):
                self.events.append(dict(event='fallback status-rejected', status=frame.status.byte))
                return None
            if frame is not None:
                q = float(np.mean(abs(single)))
                self.members[-1] = Copy(at, 0, head.cfo_hz, q, single, q)
            if frame is not None:
                member, work = self.members[-1], self.work
                self.reset('valid single copy')
                self.last, self.work = at, work
                self.members = [member]
                frame = self._accept(frame)
        return frame

    @property
    def decode_budget(self):
        if self.decode_times:
            return float(np.percentile(self.decode_times, 95))
        # Calibrate the cold full pass from the same trellis's one-pass cost.
        return min(.055, 8*self.quick_decode_budget) if self.quick_decode_times else .055

    @property
    def quick_decode_budget(self):
        return (float(np.percentile(self.quick_decode_times, 95))
                if self.quick_decode_times else .012)

    def normal(self, at, offset, cfo, coherence, scores, work, *, deadline=None):
        self._select('normal')
        if self.last != at:
            if self.last is not None:
                cycles = round((at-self.last)/(1.25*self.fs))
                if cycles < 1 or abs(at-self.last-cycles*1.25*self.fs) > .005*self.fs:
                    self.reset('off raster')
            q = float(np.mean(abs(scores)))
            if self.frame is not None:
                if q >= np.median([m.q for m in self.members]) and p4breakin.distance(scores, self.codeword) > .45:
                    self.reset('different body')
                else:
                    self.last = at
                    self.events.append(dict(event='held-same', at=at))
                    return self.frame
            self.last = at
            self.members.append(Copy(at, offset, cfo, q, scores, q, coherence))
            self._normal_results.clear()
            self._normal_deferred.clear()
            self.work = work
            self._normal_refined = False
        return self._normal_decode(deadline)

    def _normal_decode(self, deadline=None):
        if self.frame is not None or not self.members:
            return self.frame
        for single in (False, True) if len(self.members) > 1 else (False,):
            members = self.members[-1:] if single else self.members
            order = (1, 8)
            if deadline is not None:
                allowance = deadline-self.now()
                full = self.decode_budget
                if full <= allowance < full+self.quick_decode_budget:
                    order = (8, 1)
            for iterations in order:
                event = None
                key = (iterations, tuple((m.onset, m.offset, m.cfo, m.llr.tobytes()) for m in members))
                if key not in self._normal_results:
                    start = self.now()
                    budget = self.quick_decode_budget if iterations == 1 else self.decode_budget
                    allowance = None if deadline is None else deadline-start
                    event = dict(event='normal-decode', at=self.last,
                        stage='single' if single else 'sum', copies=len(members), iterations=iterations,
                        allowance_ms=None if allowance is None else 1000*allowance,
                        budget_ms=1000*budget, elapsed_ms=0., crc_valid=None)
                    if allowance is not None and budget > allowance:
                        if key not in self._normal_deferred:
                            self.events.append(dict(event, reason='budget'))
                            self._normal_deferred.add(key)
                        continue
                    scores = sum(m.llr for m in members)
                    frame = (self._decode_normal(scores, iterations=1) if iterations == 1
                             else self._decode(scores))
                    elapsed = max(0., self.now()-start)
                    times = self.quick_decode_times if iterations == 1 else self.decode_times
                    times.append(elapsed)
                    del times[:-32]
                    self._normal_results[key] = frame
                    event.update(elapsed_ms=1000*elapsed, crc_valid=self._normal_crc_valid,
                                 reason='CRC' if not self._normal_crc_valid else 'polarity')
                    self.events.append(event)
                frame = self._normal_results.get(key)
                if frame is None:
                    continue
                if not self.status_ok(frame, combined=len(members) > 1):
                    if event is not None:
                        event['reason'] = 'status'
                    self.events.append(dict(event='status-rejected', status=frame.status.byte))
                    continue
                if single:
                    self.members = members
                    self.events.append(dict(event='valid single copy', at=self.last))
                accepted = self._accept(frame)
                if event is not None:
                    event['reason'] = 'accepted' if accepted is not None else 'distance'
                if accepted is not None:
                    return accepted
        return None

    def refine(self, *, deadline=None):
        if self.kind == 'normal':
            deadline = self.now()+.250 if deadline is None else deadline
            frame = self._normal_decode(deadline)
            if frame is not None or self.work is None or self._normal_refined:
                return frame
            if self.now()+self.decode_budget+.010 > deadline:
                return None
            self._normal_refined = True
            z, nominal, cfo, at = self.work
            placed = p4short.place_short(z, nominal, cfo, offsets=np.arange(-48, 49, 2))
            if placed is None:
                return None
            coherence, start, refined = placed
            member = next((m for m in self.members if m.onset == at), None)
            if member is None or coherence <= member.coherence:
                return None
            result = p4short.short_scores(z, first_training_start=start,
                                          frequency_offset=refined, refine=0)
            if result is not None:
                member.offset, member.cfo, member.llr = start-nominal, refined, result[2]
                member.q, member.coherence = float(np.mean(abs(member.llr))), coherence
                return self._normal_decode(deadline)
            return None
        if self.work is None or self.frame is not None:
            return self.frame
        deadline = monotonic() + .250 if deadline is None else deadline
        z, nominal, cfo, at = self.work
        self.work = None
        result = p4breakin.search_scores(z, nominal, cfo, deadline=deadline)
        if result is None or monotonic() >= deadline:
            self.events.append(dict(event='search overrun', at=at))
            return None
        offset, q, llr = result
        member = next((m for m in self.members if m.onset == at), None)
        if member is None:
            return None
        old = member.offset, member.q, member.llr
        members, events = self.members.copy(), len(self.events)
        member.offset, member.q, member.llr = offset, q, llr
        frame = self._accept()
        if monotonic() >= deadline:
            member.offset, member.q, member.llr = old
            self.frame = self.codeword = None
            self.members = members
            del self.events[events:]
            self.events.append(dict(event='search overrun', at=at))
            return None
        self.offset = int(round(np.median([m.offset for m in self.members])))
        return frame
