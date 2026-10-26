# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re

from .observer import IN_Q_OVERFLOW, MACOS_TRACE_GAP, Trace


class OracleFailure(AssertionError):
    pass


@dataclass
class Evidence:
    run: str
    paths: set
    platform: str
    events: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    trace: Trace | None = None
    inotify: list = field(default_factory=list)
    tracer_returncode: int | None = None
    tree_finished: bool = False
    errors: list = field(default_factory=list)
    lost: bool = False


def read_events(path):
    raw = Path(path).read_text()
    if raw and not raw.endswith('\n'):
        raise OracleFailure('truncated participant event log')
    try:
        return [json.loads(line) for line in raw.splitlines()]
    except ValueError as exc:
        raise OracleFailure('malformed participant event log') from exc


def _path(path):
    return os.path.realpath(str(path).removesuffix(' (deleted)'))


def validate(evidence):
    if evidence.errors:
        raise OracleFailure(f'observer failed: {evidence.errors}')
    if evidence.lost:
        raise OracleFailure('a participant event write was lost')
    if any(event['mask'] & IN_Q_OVERFLOW for event in evidence.inotify):
        raise OracleFailure('inotify queue overflow')
    registered = {}
    for event in evidence.events:
        if event.get('run') != evidence.run:
            raise OracleFailure('event from another run')
        name = event['event']
        if name == 'claim_acquired' or (name == 'inherited' and event.get('verdict') is True):
            registered.setdefault(event['pid'], []).append(event)
        elif name in ('native_spawn', 'native_admitted'):
            if 'child_pid' not in event or 'child_started' not in event:
                raise OracleFailure('spawn event lacks child identity')
            registered.setdefault(event['child_pid'], []).append(
                dict(event, start=event['child_started']))
    paths = {_path(path) for path in evidence.paths}
    for event in evidence.events:
        if (event['event'] == 'device_open' and _path(event['path']) in paths
                and event['pid'] not in registered):
            raise OracleFailure(f"unregistered device opener: {event['pid']}")
    for snapshot in evidence.snapshots:
        for pid, descriptors in snapshot.descriptors.items():
            if not any(_path(fd.get('path', '')) in paths for fd in descriptors):
                continue
            events = registered.get(pid, [])
            if not events:
                raise OracleFailure(f'unlogged sampled holder: {pid}')
            if pid not in snapshot.participants:
                raise OracleFailure(f'holder missing from process inventory: {pid}')
            start = snapshot.participants[pid].get('event_start')
            if start and not any(e.get('start') == start for e in events):
                raise OracleFailure(f'holder start identity mismatch: {pid}')
    if evidence.platform != 'linux':
        return [MACOS_TRACE_GAP]
    trace = evidence.trace
    if trace is None or not trace.pids:
        raise OracleFailure('Linux requires a nonempty syscall trace')
    if (not evidence.tree_finished or evidence.tracer_returncode is None
            or evidence.tracer_returncode < 0 or trace.pids != trace.exited):
        raise OracleFailure('tracer exited or trace ends before its tree')
    groups = {}
    for record in trace.records:
        if record['syscall'] in ('clone', 'clone3') and 'CLONE_THREAD' in record['args']:
            if record['value'] is not None and record['value'] > 0:
                groups[record['value']] = record['pid']

    def process(pid):
        seen = set()
        while pid in groups:
            if pid in seen:
                raise OracleFailure('cyclic thread identities')
            seen.add(pid)
            pid = groups[pid]
        return pid

    for record in trace.records:
        if 'path' in record and _path(record['path']) in paths:
            pid = process(record['pid'])
            if pid not in registered:
                raise OracleFailure(f'unlogged traced opener: {pid}')
    return []


def assert_released(snapshot, *, paths, claim_path, line=None, require_line=False):
    watched = {_path(path) for path in paths}
    for pid, descriptors in snapshot.descriptors.items():
        if any(_path(fd.get('path', '')) in watched for fd in descriptors):
            raise OracleFailure(f'surviving hardware authority: {pid}')
    if snapshot.locks is not None:
        info = Path(claim_path).stat()
        for line_record in snapshot.locks.splitlines():
            match = re.search(r'\b([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\b', line_record)
            if not match:
                raise OracleFailure('unparseable kernel lock record')
            identity = (int(match[1], 16), int(match[2], 16), int(match[3]))
            if identity == (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino):
                raise OracleFailure('claim still locked in kernel')
    elif snapshot.claim_free is not True:
        raise OracleFailure('claim freedom is unobserved')
    if require_line and line is not False:
        raise OracleFailure('release line is not confirmed low')


def assert_claim_possession(snapshot, *, pid, claim_path, nonce):
    matches = [fd for fd in snapshot.descriptors.get(pid, [])
               if _path(fd.get('path', '')) == _path(claim_path) and fd.get('pos') == nonce]
    if not matches:
        raise OracleFailure('claim descriptor or nonce not observed')
    if snapshot.locks is not None:
        if not any(any('FLOCK' in lock and 'WRITE' in lock and '->' not in lock
                       for lock in fd.get('locks', [])) for fd in matches):
            raise OracleFailure('offset matches but descriptor does not hold the claim lock')
    elif snapshot.claim_free is not False:
        raise OracleFailure('claim lock not observed at barrier')


def pulse_upper_bound(samples, *, limit_ns=20_000_000, asymmetry_ns=None,
                      blindness_ns=None, reporting_delay_ns=None, shortest_pulse_ns=None):
    """Bound a pulse using a witness with known shortest_pulse_ns on both branches.

    Even a detected high requires shortest_pulse_ns <= limit_ns; detection alone
    does not establish sufficient witness resolution. Detected highs also need
    asymmetry_ns, while all-low samples need blindness_ns and reporting_delay_ns.
    """
    if len(samples) < 2 or samples[0][2] is not False or samples[-1][2] is not False:
        raise OracleFailure('unresolved pulse: missing low baseline or final low')
    for index, (before, after, level) in enumerate(samples):
        if type(level) is not bool or after < before:
            raise OracleFailure('unresolved line sample')
        if after - before > 2_000_000:
            raise OracleFailure('sampler gap exceeds 2 ms')
        if index:
            previous = samples[index - 1]
            if before < previous[1] or after - previous[0] > 2_000_000:
                raise OracleFailure('sampler gap exceeds 2 ms or clock reversed')
    if shortest_pulse_ns is None or shortest_pulse_ns > limit_ns:
        raise OracleFailure('unresolved pulse: witness resolution is insufficient')
    highs = [index for index, sample in enumerate(samples) if sample[2]]
    if highs:
        if asymmetry_ns is None or asymmetry_ns < 0:
            raise OracleFailure('unresolved pulse: reporting asymmetry is unknown')
        bound = samples[highs[-1] + 1][1] - samples[highs[0] - 1][0] + asymmetry_ns
    else:
        if (blindness_ns is None or reporting_delay_ns is None
                or blindness_ns < 0 or reporting_delay_ns < 0):
            raise OracleFailure('unresolved pulse: blindness or reporting delay is unknown')
        bound = max(b[1] - a[0] for a, b in zip(samples, samples[1:]))
        bound += blindness_ns + reporting_delay_ns
    if bound > limit_ns:
        raise OracleFailure(f'pulse upper bound {bound} exceeds {limit_ns} ns')
    return bound


def assert_untraced_bound(*, traced, bound_ns):
    if traced and bound_ns < 100_000_000:
        raise OracleFailure('bounds below 100 ms require untraced measurements')
