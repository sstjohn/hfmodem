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
    counter_experiment: bool = False
    schedule: tuple = ()
    chirp_field: bytes | None = None
    packet: str = 'chirp'
    cadence: str = 'spaced'
    values: tuple = ()
    headers: tuple = ()
    replies: tuple = ()
    path: str | None = None
    heard: list = field(default_factory=list)
    cfo_hz: float | None = None
    floor_db: float | None = None
    received: list = field(default_factory=list, repr=False)
    guard_exceptions: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    link_events: list = field(default_factory=list)
    stopped: str | None = None

    def statuses(self) -> list[int | None]:
        # A CLONE CARRIES THE TAPE'S OWN FIELD. There is no status byte of ours
        # in it, so the schedule is a column of `None` and the query says so.
        if self.packet == 'robust':
            return [None] * self.attempts
        if self.schedule:
            return list(self.schedule)
        return [0x61, 0x61, 0x62] if self.counter_experiment else [self.status] * self.attempts

    def request(self, sample: int, payload: bytes, status: int) -> None:
        if self.requested:
            return
        self.requested = True
        self.grant_sample, self.grant_wall = sample, time.monotonic()
        # The operator's field is the query, so it outranks whatever the ARQ
        # seam was carrying when the trigger arrived.
        self.payload = bytes(payload if self.chirp_field is None else self.chirp_field)
        self.status = status
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
                    status=self.status, statuses=self.statuses(),
                    packet=self.packet, cadence=self.cadence, values=list(self.values),
                    headers=list(self.headers), replies=list(self.replies),
                    path=self.path, heard=self.heard,
                    payload_len=len(self.payload),
                    peer_offset_hz=self.cfo_hz, quiet_floor_db=self.floor_db,
                    emissions=self.emissions, windows=self.windows,
                    guard_exceptions=self.guard_exceptions, queries=self.queries,
                    link_events=self.link_events, stopped=self.stopped)
