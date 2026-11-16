# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded robust opening and connected P4 receive records."""
from dataclasses import asdict, dataclass, field
import hashlib
import time
import subprocess
from pathlib import Path

from .timing import timed_record


def code_revision(directory=None):
    try:
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=directory,
                stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        return dict(sha=git('rev-parse', '--short', 'HEAD'),
                    dirty=bool(git('status', '--porcelain', '--untracked-files=normal')))
    except (OSError, subprocess.SubprocessError):
        return None


def probe_config(args):
    knobs = {key: value for key, value in vars(args).items() if key.startswith('p4_')}
    for key in ('after_connect', 'after_robust', 'p1_robust_stall', 'p1_status_bits45',
                'hold', 'tx_latency_ms', 'qrm_guard', 'long_cycle', 'tx_drive', 'p1_drive'):
        knobs[key] = getattr(args, key, None)
    from .p4turn import PEER_ACK
    knobs['p4_peer_ack'] = dict(PEER_ACK) | dict(knobs.get('p4_peer_ack') or {})
    plan = getattr(args, 'tx_timing_plan', None)
    timing = asdict(plan) if plan is not None else {}
    offset = knobs.get('p4_answer_offset', .920)
    latency = timing.get('tx_latency_ms', knobs['tx_latency_ms'] or 0.)
    residual = timing.get('p4_answer_residual_ms', 0.)
    correction = (timing.get('latency_samples', round(latency*48))
                  + timing.get('p4_answer_residual_samples', round(residual*48))
                  - timing.get('geometry_samples', 0)) / 48000
    knobs.update(tx_latency_ms=latency, p4_answer_residual_ms=residual)
    timing.update(answer_offset_s=offset, tx_latency_ms=latency,
                  p4_answer_residual_ms=residual,
                  scheduled_offset_s=offset-correction,
                  long_answer_offset_s=3.400,
                  long_scheduled_offset_s=3.400-correction)
    return dict(knobs, timing_plan=timing,
                revision=code_revision(Path(__file__).resolve().parent))


@dataclass
class EntryProbe:
    attempts: int = 2
    timeout: float = 20.0
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
    values: tuple = (7, 38)

    headers: tuple = field(init=False)

    def __post_init__(self):
        self.attempts = len(self.values)
        self.headers = tuple('B' if value in self.values[:i] else 'A'
                             for i, value in enumerate(self.values))

    packet = 'robust'
    cadence = 'consecutive'
    path: str | None = None
    heard: list = field(default_factory=list)
    cfo_hz: float | None = None
    floor_db: float | None = None
    received: list = field(default_factory=list, repr=False)
    queries: list = field(default_factory=list)
    link_events: list = field(default_factory=list)
    stopped: str | None = None
    config: dict = field(default_factory=dict)
    id: list = field(default_factory=list)
    closing_id_completed: bool | None = None

    def note_id(self, payload, kind, start, end):
        call = getattr(self, 'mycall', '')
        if call and call.upper().encode('ascii') in payload.upper():
            self.id.append(dict(kind=str(kind), call=call.upper(),
                                start_sample=start, end_sample=end))

    def statuses(self) -> list[int | None]:
        return [None] * self.attempts

    def request(self, sample: int, payload: bytes, status: int) -> None:
        if self.requested:
            return
        self.requested = True
        self.grant_sample, self.grant_wall = sample, time.monotonic()
        self.payload = bytes(payload)
        self.status = status
        self.reason = 'grant received; probe pending'

    def deadline(self, fs: int) -> int:
        return self.grant_sample + round(self.timeout * fs)

    def room(self, sample: int, seconds: float, fs: int) -> bool:
        return (sample + round(seconds * fs) <= self.deadline(fs)
                and time.monotonic() + seconds <= self.grant_wall + self.timeout)

    def result(self, fs: int = 48000) -> dict:
        chains = {}
        for event in self.link_events:
            if (event.get('event') == 'Keyed' and event.get('answers_counter') is not None
                    and event.get('accepted_target') is not None):
                chains.setdefault(event['accepted_target'], event['chain_epoch'])
        copies = {}
        seen = {}
        previous = None
        chain = None
        for row in self.heard:
            if 'counter' not in row:
                continue
            epoch = row.get('receive_epoch', row.get('chain_epoch'))
            chain = chains.get(row.get('onset_sample'), chain)
            identity = epoch, chain, row['counter']
            onset = epoch, row.get('onset_sample'), row['counter']
            if onset not in seen:
                copies[identity] = copies.get(identity, 0) + 1 if identity == previous else 1
                seen[onset] = copies[identity]
                previous = identity
            row['copy_index'] = seen[onset]
        last_tx = max((e.get('audio_end', 0) for e in self.emissions), default=0)
        last_id = max((e.get('end_sample') or 0 for e in self.id if e['kind'] != 'cw'),
                      default=0)
        closing_id = any(e['kind'] == 'cw' and e.get('start_sample') is not None
                         and e['start_sample'] >= max(last_tx, last_id)
                         and (e.get('end_sample') or 0) > e['start_sample'] for e in self.id)
        result = dict(schema='p4-mail/1', config=self.config, id=self.id,
                    missing_closing_id=not closing_id or self.closing_id_completed is False,
                    scope='P4 entry probe; no connected P4 acceptance claim',
                    sample_rate=fs, requested=self.requested, reason=self.reason,
                    trigger_word=self.trigger_word,
                    burst_limit=self.attempts, timeout_seconds=self.timeout,
                    receive_seconds=self.listen_seconds,
                    grant_sample=self.grant_sample if self.requested else None,
                    payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                    status=self.status, statuses=self.statuses(),
                    packet=self.packet, cadence=self.cadence, values=list(self.values),
                    headers=list(self.headers),
                    path=self.path, heard=self.heard,
                    payload_len=len(self.payload),
                    peer_offset_hz=self.cfo_hz, quiet_floor_db=self.floor_db,
                    emissions=self.emissions, windows=self.windows,
                    queries=self.queries,
                    link_events=self.link_events, stopped=self.stopped)
        return timed_record(result, getattr(self, 'timing', {}),
                            ('emissions', 'windows', 'queries',
                             'heard', 'link_events', 'id'))
