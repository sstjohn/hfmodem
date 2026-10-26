# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Session-local beacon opportunities and evidence; no radio or network access."""
from collections import deque


MIN_DWELL_MINUTES = 5
DEFAULT_DWELL_MINUTES = 15
BEACON_COOLDOWN_SECONDS = 15 * 60
HISTORY_S = 7 * 86400


class BeaconHop:
    def __init__(self, channels, clock, wall_clock):
        self.channels, self.clock, self.wall_clock = tuple(channels), clock, wall_clock
        self.readiness = {}
        self.last_tx = {}
        self.heard = deque(maxlen=8192)
        self.listening = {}
        self.last_observed = {}
        self._sample = None
        self.disable()

    @property
    def active(self):
        return bool(self.bands)

    def disable(self):
        # Readiness, evidence and actual transmissions survive mode changes.
        self.bands = ()
        self.auto = False
        self.pending = None
        self.paused = None
        self.due = None

    def configure(self, words):
        words = list(words)
        auto = words[:1] == ['auto']
        if auto:
            words.pop(0)
        minutes = DEFAULT_DWELL_MINUTES
        if 'every' in words:
            at = words.index('every')
            if at != len(words) - 2:
                raise ValueError('use /beacon hop [auto] BANDS every MINUTES')
            minutes = int(words[-1])
            words = words[:at]
        elif words and int(words[-1]) not in self.channels:
            minutes = int(words.pop())
        bands = tuple(map(int, words)) if words else self.channels if auto else ()
        if (not bands or len(set(bands)) != len(bands)
                or any(b not in self.channels for b in bands) or minutes < MIN_DWELL_MINUTES):
            raise ValueError('hop needs distinct supported bands and a dwell of at least 5 minutes')
        self.bands, self.auto, self.minutes = bands, auto, minutes
        self.index = self.choices = 0
        self.pending = self.paused = None
        self.due = self.clock()

    def ready(self, band):
        return self.readiness.get(band, 'unknown') == 'ready'

    def cooldown(self, band):
        return max(0, self.last_tx.get(band, float('-inf')) + BEACON_COOLDOWN_SECONDS - self.clock())

    def defer(self):
        self.pending = None
        self.due = self.clock() + self.minutes * 60

    def arrived(self, band):
        self.pending = band
        self.due = self.clock() + self.minutes * 60
        if band in self.bands:
            self.index = (self.bands.index(band) + 1) % len(self.bands)

    def observe(self, band, call, source):
        self.heard.append((self.wall_clock(), band, call, source))

    def sample(self, band, monitoring):
        now, utc = self.clock(), self.wall_clock()
        if monitoring:
            self.last_observed[band] = now
            if self._sample is not None:
                before, previous = self._sample
                if previous == band:
                    key = (int(utc // 3600), band)
                    # Do not credit suspended processes or gaps in polling as listening.
                    self.listening[key] = self.listening.get(key, 0) + min(1, max(0, now - before))
        self._sample = (now, band) if monitoring else None
        self.listening = {k: v for k, v in self.listening.items() if k[0] >= (utc - HISTORY_S) // 3600}

    def evidence(self, band):
        now = self.wall_clock()
        hour = int(now // 3600) % 24
        nearby = lambda h: (h - hour) % 24 in (0, 1, 23)
        calls = {call for at, b, call, _ in self.heard
                 if b == band and 0 <= now - at <= HISTORY_S and nearby(int(at // 3600) % 24)}
        seconds = sum(s for (h, b), s in self.listening.items() if b == band and nearby(h % 24))
        return len(calls), seconds

    def choose(self):
        """Consume at most one opportunity, never sweep bands on a busy refusal."""
        self.pending = None
        self.due = self.clock() + self.minutes * 60
        if not self.auto:
            band = self.bands[self.index]
            if not self.ready(band):
                self.paused = 'readiness'
                return None, (f'hopping paused: {band} m is {self.readiness.get(band, "unknown")}; '
                              f'use /band {band}, verify the match, then /band ready')
            self.index = (self.index + 1) % len(self.bands)
            if self.cooldown(band):
                return None, f'{band} m skipped: last beacon less than 15 min ago'
            return band, 'next band in configured rotation'
        eligible = [b for b in self.bands if self.ready(b) and not self.cooldown(b)]
        excluded = ', '.join(f'{b} m ({self.readiness.get(b, "unknown") if not self.ready(b) else "15 min cooldown"})'
                             for b in self.bands if b not in eligible)
        if not eligible:
            return None, f'no eligible hop band; next opportunity in {self.minutes} min; excluded: {excluded}'
        self.choices += 1
        evidence = {b: self.evidence(b) for b in eligible}
        least = lambda b: (self.last_observed.get(b, float('-inf')), self.bands.index(b))
        if self.choices % 3 == 0 or not any(n for n, _ in evidence.values()):
            band = min(eligible, key=least)
            reason = 'exploration; least recently monitored eligible band'
        else:
            # Deduplicated activity per monitored minute, with a one-minute floor.
            band = max(eligible, key=lambda b: (evidence[b][0] / max(1, evidence[b][1] / 60),
                                               -self.last_observed.get(b, float('-inf'))))
            n, seconds = evidence[band]
            reason = (f'{n} unique stations heard near {int(self.wall_clock() // 3600) % 24:02d} UTC; '
                      f'{seconds / 60:.1f} monitored min in recent history')
        if excluded:
            reason += f'; excluded: {excluded}'
        return band, reason
