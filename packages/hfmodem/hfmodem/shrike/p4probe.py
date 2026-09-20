# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded P4 entry experiment state, independent of connected-mode ARQ."""
from dataclasses import dataclass, field
import hashlib
import time


@dataclass
class EntryProbe:
    attempts: int = 2
    timeout: float = 60.0
    listen_seconds: float = 4.0
    requested: bool = False
    emitting: bool = False
    grant_sample: int = 0
    grant_wall: float = 0.0
    payload: bytes = b''
    status: int = 0
    trigger_word: str = '0x59A'
    reason: str = 'no grant received'
    emissions: list = field(default_factory=list)
    windows: list = field(default_factory=list)

    def request(self, sample: int, payload: bytes, status: int) -> None:
        if self.requested:
            return
        self.requested = True
        self.grant_sample, self.grant_wall = sample, time.monotonic()
        self.payload, self.status = bytes(payload), status
        self.reason = 'grant received; probe pending'

    def deadline(self, fs: int) -> int:
        return self.grant_sample + round(self.timeout * fs)

    def room(self, sample: int, seconds: float, fs: int) -> bool:
        return (sample + round(seconds * fs) <= self.deadline(fs)
                and time.monotonic() + seconds <= self.grant_wall + self.timeout)

    def result(self, fs: int = 48000) -> dict:
        return dict(scope='P4 entry probe; no connected P4 acceptance claim',
                    sample_rate=fs, requested=self.requested, reason=self.reason,
                    trigger_word=self.trigger_word,
                    burst_limit=self.attempts, timeout_seconds=self.timeout,
                    receive_seconds=self.listen_seconds,
                    grant_sample=self.grant_sample if self.requested else None,
                    payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                    status=self.status, emissions=self.emissions, windows=self.windows)
