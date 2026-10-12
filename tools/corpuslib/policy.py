"""Evidence, not energy, controls how long a channel owns the receiver."""
from dataclasses import dataclass
import math


READABLE = {'P1-CONNECT', 'P1-CALLB', 'P1-CS', 'P1-PACKET', 'P3-HEADER', 'P3-CS', 'P4-CHIRP-CRC'}


@dataclass
class Hold:
    dwell: float = 60
    quiet: float = 120
    candidate: float = 15
    maximum: float = 1800
    last_readable: float = -math.inf
    last_useful: float = -math.inf
    first_candidate: float | None = None
    p4_count: int = 0
    last_p4: float = -math.inf
    useful: int = 0

    def observe(self, t, kind):
        # Event time, not decoder delivery time. Late results cannot keep a dead
        # channel alive; repeated overlapping windows cannot count twice.
        if kind == 'CENTERED-FSK-CANDIDATE' and self.first_candidate is None:
            self.first_candidate = t
            return 'bounded-fsk-candidate'
        if kind in READABLE and t > self.last_readable + .1:
            self.last_readable = t
            self.last_useful = max(t, self.last_useful)
            self.useful += 1
            return 'readable-pactor'
        if kind in ('P4-HEADER', 'P4-CHIRP-HEADER') and t > self.last_p4 + .5:
            self.last_p4 = t
            self.p4_count += 1
            if self.p4_count == 1:
                # A real header near the end of a visit must not lose its
                # confirmation window to an unrelated early FSK shape.
                self.first_candidate = t
            # A header correlation is not a packet/CRC. Only extend a setup
            # already witnessed, and only after two distinct header candidates.
            if self.useful and self.p4_count >= 2 and t - self.last_useful <= self.quiet:
                self.last_useful = max(t, self.last_useful)
                return 'p4-after-readable-setup'
            return 'bounded-p4-candidate'
        return 'no-hold-evidence'

    def decision(self, now):
        if now >= self.maximum:
            return False, 'maximum-dwell; exchange-may-be-incomplete'
        if now - self.last_useful < self.quiet:
            return True, 'following-pactor'
        if self.first_candidate is not None and now < self.first_candidate + self.candidate:
            return True, 'checking-p4-candidate'
        if now < self.dwell:
            return True, 'initial-dwell'
        return False, 'no-recent-copyable-pactor'
