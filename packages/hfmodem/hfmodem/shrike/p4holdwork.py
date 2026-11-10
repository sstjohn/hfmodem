# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owned, finite P1 answer windows awaiting P4 takeover detection."""

import numpy as np


class HoldAudio:
    def __init__(self, fs):
        self.fs = fs
        self.origin = 0
        self.audio = np.empty(0, np.float32)
        self.gaps = []
        self.read_to = {}
        self.pending = False
        self.expires = None
        self.wall_expires = None

    def retain(self, audio, origin, gaps=()):
        """Copy capture storage and keep holes at their absolute positions."""
        if not len(audio):
            return
        end = origin+len(audio)
        old_end = self.origin+len(self.audio)
        lo = max(min(self.origin, origin) if len(self.audio) else origin,
                 max(end, old_end)-round(8*self.fs))
        hi = max(end, old_end)
        owned = np.zeros(hi-lo, np.float32)
        if end > lo:
            a = max(lo, origin)
            owned[a-lo:end-lo] = np.asarray(audio)[a-origin:]
        # Samples already owned cannot be replaced by a reused capture view.
        if len(self.audio) and old_end > lo:
            a = max(lo, self.origin)
            owned[a-lo:old_end-lo] = self.audio[a-self.origin:]
        if len(self.audio) and origin > old_end:
            self.gaps.append((old_end, origin))
        self.gaps.extend(gaps)
        self.gaps = sorted(set((max(lo, a), min(hi, b)) for a, b in self.gaps
                               if a < hi and b > lo))
        for a, b in self.gaps:
            owned[a-lo:b-lo] = 0
        self.origin, self.audio = lo, owned
        self.read_to = {at: stop for at, stop in self.read_to.items() if stop > lo}

    def opportunities(self, raster, carrier_end):
        lo, end, fs = self.origin, self.origin+len(self.audio), self.fs
        if raster.sending:
            first = (lo-raster.rx_due(0)-round(.275*fs))//raster.cycle_n
            last = (end-raster.rx_due(0)+round(.275*fs))//raster.cycle_n
            for k in range(first, last+1):
                due = raster.rx_due(k*raster.ticks)
                carrier = due-raster.d
                begin = max(lo, carrier-round(.275*fs))
                stop = min(end, carrier+raster.d_max_n+round(.275*fs))
                if stop > max(begin, carrier_end) and stop > self.read_to.get(due, -1):
                    yield due, begin, stop
        else:
            # Preserve the IRS's latest post-CS position. ISS recovery above
            # has different carrier geometry and must visit every missed CS.
            k = max(0, (end-carrier_end-raster.d_max_n)//raster.cycle_n)
            carrier = carrier_end+k*raster.cycle_n
            due = carrier+raster.d
            begin = max(lo, carrier-round(.240*fs))
            stop = min(end, carrier+raster.d_max_n+round(.275*fs))
            if stop > begin and stop > self.read_to.get(due, -1):
                yield due, begin, stop

    def read(self, receiver, raster, carrier_end, work, wall_now):
        if self.expires is not None and (work.sample_now() >= self.expires
                                         or wall_now() >= self.wall_expires):
            heads = {o.stream_sample for o in receiver.observations
                     if o.provisional and o.word in ('X27', 'Y27')}
            receiver.expire_hold_words(heads)
            self.pending = False
            positions = set(self.read_to) | {
                due for due, _, _ in self.opportunities(raster, carrier_end)}
            for due in positions:
                begin = due-raster.d-round(.275*self.fs)
                stop = due-raster.d+raster.d_max_n+round(.275*self.fs)
                if any(begin <= at < stop for at in heads):
                    self.read_to[due] = stop
            self.expires = self.wall_expires = None
        self.pending = False
        for due, begin, stop in self.opportunities(raster, carrier_end):
            if stop-max(begin, carrier_end) < round(.12*self.fs):
                continue
            before = len(work.deferred)
            records = receiver.read_hold_word(self.audio[begin-self.origin:stop-self.origin],
                begin, due=due, carrier_end=carrier_end, work=work)
            if len(work.deferred) > before:
                # Only detected takeover evidence reserves a P1 key. Unread
                # work survives, but noise must not suppress P1 CS/ACK.
                break
            self.read_to[due] = stop
            if records:
                break
        self.pending |= any(o.provisional and o.word in ('X27', 'Y27')
                            and o.stream_sample >= carrier_end-round(.240*self.fs)
                            and o.stream_sample+round(.240*self.fs) > self.origin
                            for o in receiver.observations)
        if self.pending and self.expires is None:
            self.expires = work.sample_now()+round(2.5*self.fs)
            self.wall_expires = wall_now()+2.5
        if not self.pending:
            self.expires = self.wall_expires = None
