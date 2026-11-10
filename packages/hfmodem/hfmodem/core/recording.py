# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recording policy, sample clocks, contact selection and serial storage."""
from __future__ import annotations

import argparse
import bisect
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from operator import itemgetter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
import uuid
import wave

import numpy as np

if os.name == 'nt':
    import msvcrt
    fcntl = None
else:
    import fcntl

from hfmodem.core import gil, wav
from hfmodem.core.capture_identity import _absolute

_CLOSE_TIMEOUT = 30
_WINDOWS = os.name == 'nt'
#: Windows byte-range locks are mandatory against every other handle, so the
#: candidate lock sits past the identity that readers open the file for.
_LOCK_OFFSET = 1 << 30
# Between boundaries, per-block progress reaches the index at most this often.
_INDEX_INTERVAL = 0.25
# How much PCM waits for the drainer, well inside the two-second queue bound.
# A block that woke the drainer on arrival put a GIL contender on every audio
# callback; batched, the drainer runs ten times a second whatever the block size.
_DRAIN_INTERVAL = 0.1
_CLOCK_KEYS = ('monotonic_origin', 'clock_uncertainty_s', 'measured_rate', 'adc_origin', 'uncertain')
POLICIES = ('off', 'contacts', 'everything')
OFF_WARNING = 'Recording off: session and contact audio will not be saved.'


def publish_policy(session, *, write=None):
    facts = {key: session.index[key] for key in ('policy', 'source')}
    def publish():
        session._root.mkdir(parents=True, exist_ok=True)
        (write or wav._write_json)(session._root / 'recording-policy.json', facts)
    session.after_pending(publish)


def add_recording_argument(parser):
    parser.add_argument('--recording', choices=POLICIES, default=None)


@dataclass(frozen=True)
class ResolvedRecording:
    policy: str
    source: str


def resolve_recording(explicit=None, *, config=None, no_record=False, record=None,
                      environ=None, report=print):
    environ = os.environ if environ is None else environ
    inherited = environ.get('HFMODEM_RECORDING')
    if inherited is not None and inherited not in POLICIES:
        raise ValueError('Invalid HFMODEM_RECORDING policy')
    choices = [v for v in (explicit, 'off' if no_record else None,
                           'everything' if record is not None else None) if v is not None]
    if any(v not in POLICIES for v in choices):
        raise ValueError('Invalid recording policy')
    if len(set(choices)) > 1:
        raise ValueError('Contradictory explicit recording selections')
    if config is None and environ.get('HFMODEM_STATION'):
        from hfmodem.core.config import load
        config = load(environ['HFMODEM_STATION'], environ=environ)
    if choices:
        result = ResolvedRecording(choices[0], 'explicit')
    elif inherited is not None:
        result = ResolvedRecording(inherited, 'inherited')
    elif config is not None:
        result = ResolvedRecording(config.recording.policy, 'station')
    else:
        result = ResolvedRecording('everything', 'default')
    environ['HFMODEM_RECORDING'] = result.policy
    if result.policy == 'off' and inherited is None:
        report(OFF_WARNING)
    return result


def _id():
    return uuid.uuid4().hex


def _gap(gaps, first, last):
    """Merge [first, last) into sorted, disjoint gaps; new loss arrives at the tail."""
    if last <= first:
        return
    if not gaps or first > gaps[-1][1]:
        gaps.append([first, last])
        return
    i = bisect.bisect_right(gaps, first, key=itemgetter(0))
    if i and gaps[i - 1][1] >= first:
        i -= 1
    j = i
    while j < len(gaps) and gaps[j][0] <= last:
        j += 1
    if j > i:
        first, last = min(first, gaps[i][0]), max(last, gaps[j - 1][1])
    gaps[i:j] = [[first, last]]


def _holes(descriptor):
    """Fill inside one segment: the stretches between its spans."""
    spans = sorted(span['absolute'] for span in descriptor.get('spans', ()))
    return [(a[1], b[0]) for a, b in zip(spans, spans[1:]) if b[0] > a[1]]


def _relative(path, root):
    # Either side may be in Windows extended-length form; recorded names are
    # portable either way.
    return _absolute(path).relative_to(_absolute(root)).as_posix()


def _local(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        return None
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents) if p == root or root in p.parents):
        return None
    return path


def _durable_json(path, facts):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.partial', delete=False) as out:
            temporary = Path(out.name)
            # One encoded write: json.dump streams each token through the file
            # object, which dominated a long contact's per-segment index rewrites.
            # Encoded through iterencode rather than json.dumps, whose one-shot
            # C encoder holds the GIL for the whole index and starves the audio
            # callback; the generator is preemptible at every switch interval.
            out.write(''.join(json.JSONEncoder(indent=2).iterencode(facts)) + '\n')
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _sync_directory(path.parent)


def _sync_directory(path):
    # Windows cannot open a directory to fsync it; NTFS journals the rename.
    if os.name == 'nt':
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lane_stem(name, stem='receive'):
    return stem + '-l' + name.encode('utf-8').hex()


def _lane_name(stem):
    match = re.fullmatch(r'.+-l([0-9a-f]+)', stem)
    if match:
        try:
            return bytes.fromhex(match[1]).decode('utf-8')
        except (ValueError, UnicodeError):
            pass
    return None


def _retained_wav(path):
    with path.open('rb') as stream, wave.open(stream, 'rb') as audio:
        rate = audio.getframerate()
        width = audio.getnchannels() * audio.getsampwidth()
        stream.seek(12)
        while header := stream.read(8):
            tag, size = struct.unpack('<4sI', header)
            if tag == b'data':
                start = stream.tell()
                length = stream.seek(0, os.SEEK_END) - start
                return rate, length // width, length != size or bool(length % width)
            stream.seek(size + (size & 1), os.SEEK_CUR)
    raise ValueError('Recording has no PCM payload')


def load_recording_index(directory, *, report=print):
    """Read the session cache and adopt unindexed recording artifacts as pending.

    Missing clock mappings stay unknown. Candidate lifetimes are not resumed.
    """
    root = _absolute(directory)
    def local(name):
        return _local(root, name)
    def read(path):
        if path is None or not path.is_file():
            return {}
        try:
            facts = json.loads(path.read_text())
            return facts if isinstance(facts, dict) else {}
        except (OSError, ValueError) as exc:
            report(f'Recording recovery {path}: {exc}')
            return {}
    index = read(local('recording.json'))
    for key, value in (('lanes', {}), ('contacts', {}), ('segments', [])):
        index.setdefault(key, value)
    families = {}
    def family(source, lane=None, contact=None):
        path = local(source)
        if path is not None:
            families.setdefault(path.with_suffix(''), (lane or _lane_name(path.stem), contact))
    for name, lane in index['lanes'].items():
        if lane.get('source'):
            family(lane['source'], name)
    for cid, contact in index['contacts'].items():
        for name, ref in contact.get('lanes', {}).items():
            if ref.get('source') and ref.get('kind') == 'cutout':
                family(ref['source'], name, cid)
    if root.is_dir() and not root.is_symlink():
        for directory in root.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            if re.fullmatch('[0-9a-f]{32}', directory.name):
                family(_relative(directory / 'receive.wav', root))
            elif re.fullmatch('contact-[0-9a-f]{32}', directory.name):
                for child in directory.iterdir():
                    if child.is_dir() and re.fullmatch('[0-9a-f]{32}', child.name):
                        source = _relative(child / 'receive.wav', root)
                        path = local(source)
                        if path is not None:
                            families.setdefault(path.with_suffix(''), (None, None))
    origins = {}
    if root.is_dir() and not root.is_symlink():
        for marker in sorted(root.rglob('*.lane')):
            if local(_relative(marker, root)) is None:
                continue
            match = re.fullmatch(r'(.+)-o(unknown|[0-9]{20})\.lane', marker.name)
            if not match:
                continue
            name = _lane_name(match[1])
            if name is None:
                continue
            base = marker.with_name(match[1])
            facts = read(marker)
            row = index['lanes'].setdefault(name, {})
            for key in ('rate', 'clock', 'source_id', 'index_id'):
                if key in facts:
                    row[key] = facts[key]
            if match[2] != 'unknown':
                origins[name] = int(match[2])
            source = facts.get('source', _relative(base.with_suffix('.wav'), root))
            if local(source) is None:
                continue
            row.setdefault('source', source)
            families[base] = (name, None)
        for directory in root.rglob('*'):
            if not directory.is_dir() or local(_relative(directory, root)) is None:
                continue
            if not re.fullmatch('[0-9a-f]{32}', directory.name):
                continue
            for path in directory.iterdir():
                match = re.fullmatch(r'\.?(.+-l[0-9a-f]+)-\d{8}T\d{4}Z-s\d{20}(?:\.pending)?\.(wav|flac|json)', path.name)
                if match:
                    family(_relative(path.with_name(match[1] + '.wav'), root))
    index['segments'] = [d for d in index['segments']
                         if local(d['source']) is not None and
                         ('pending_source' not in d or local(d['pending_source']) is not None)]
    descriptors = {str(Path(d['source']).with_suffix('')): d for d in index['segments']}
    for descriptor in descriptors.values():
        descriptor['recovered'] = True
        if descriptor['state'] == 'storing':
            descriptor['state'] = 'pending'
    for base, (name, cid) in families.items():
        if not base.parent.is_dir():
            continue
        candidate_dir = base.parent.parent
        if name is not None and cid is None and not candidate_dir.name.startswith('contact-'):
            index['lanes'].setdefault(name, {'source': _relative(base.with_suffix('.wav'), root)})
        candidate = read(local(_relative(candidate_dir / 'candidate.json', root)))
        if candidate:
            identity = read(local(_relative(candidate_dir / 'candidate.identity', root)))
            lock = read(local(_relative(candidate_dir / 'candidate.lock', root)))
            valid = identity == lock and all(candidate.get(k) == identity.get(k)
                      for k in ('candidate_id', 'contact_id', 'directory'))
            valid = (valid and isinstance(candidate.get('directory'), str)
                     and _absolute(candidate['directory']) == _absolute(candidate_dir)
                     and identity.get('candidate_id'))
            candidate_state = candidate.get('state') if valid else 'unknown'
            cid = candidate.get('contact_id') if valid else cid
        else:
            candidate_state = None
        paths = set()
        aggregate = read(local(_relative(base.with_suffix('.json'), root)))
        if not aggregate and name is not None:
            source = local(index['lanes'].get(name, {}).get('source', ''))
            if source is not None and source.parent == base.parent:
                aggregate = read(local(_relative(source.with_suffix('.json'), root)))
        for segment in (*aggregate.get('segments', []), *aggregate.get('segment_artifacts', {})):
            if Path(segment).name == segment:
                paths.add(base.parent / segment)
        for path in base.parent.iterdir():
            if re.fullmatch(r'\.?' + re.escape(base.name) + r'-\d{8}T\d{4}Z(?:-s\d{20})?(?:\.pending)?\.(wav|flac|json)', path.name):
                paths.add(path)
        for physical in sorted(paths):
            if local(_relative(physical, root)) is None:
                continue
            staged = physical.name.startswith('.') and physical.name.endswith('.pending.wav')
            path = physical.with_name(physical.name[1:].replace('.pending.wav', '.wav')) if staged else physical
            side = (read(local(_relative(path.with_suffix('.json'), root))) or
                    aggregate.get('segment_artifacts', {}).get(path.with_suffix('.wav').name, {}))
            if path.suffix == '.json':
                audio_name = side.get('wav', path.with_suffix('.wav').name)
                if Path(audio_name).name != audio_name or Path(audio_name).suffix not in ('.wav', '.flac'):
                    continue
                path = path.with_name(audio_name)
            if local(_relative(path, root)) is None:
                continue
            source = _relative(path, root)
            key = str(Path(source).with_suffix(''))
            descriptor = descriptors.get(key)
            if descriptor is None:
                lane = index['lanes'].get(name, {})
                descriptor = {'source': source, 'lane': name, 'rate': side.get('samplerate', lane.get('rate')),
                              'samples': side.get('samples'), 'actual': None, 'spans': [],
                              'source_id': lane.get('source_id'), 'index_id': lane.get('index_id'),
                              'state': 'pending', 'recovered': True}
                expected = next((c for c in lane.get('expected', [])
                                 if str(Path(c['source']).with_suffix('')) == key), {})
                descriptor.update({k: deepcopy(expected[k]) for k in ('actual', 'spans') if k in expected})
                descriptors[key] = descriptor
                index['segments'].append(descriptor)
            if staged:
                descriptor['pending_source'] = _relative(physical, root)
            if side.get('capture_state') == 'finalized':
                descriptor['stored_state'] = 'finalized'
                stored = Path(side.get('wav', path.name))
                if (stored.name == str(stored) and stored.stem == path.stem and
                        stored.suffix in ('.wav', '.flac') and
                        local(_relative(path.with_name(stored.name), root)) is not None):
                    descriptor['source'] = _relative(path.with_name(stored.name), root)
            if aggregate.get('artifact_id') or side.get('artifact_id'):
                descriptor['owned'] = True
            if cid is not None:
                descriptor['contact_id'] = cid
            if candidate_state is not None:
                descriptor['candidate_state'] = candidate_state
            if candidate_dir.name.startswith('contact-'):
                descriptor['contact_directory'] = _relative(candidate_dir, root)
            if physical.suffix == '.wav' and physical.is_file():
                try:
                    rate, samples, damaged = _retained_wav(physical)
                    descriptor.update(rate=rate, samples=samples)
                    if damaged:
                        descriptor['state'] = 'partial'
                except (OSError, EOFError, ValueError, struct.error, wave.Error) as exc:
                    report(f'Recording recovery {physical}: {exc}')
            match = re.search(r'-s(\d{20})\.', path.name)
            if match and descriptor.get('samples') is not None:
                first = int(match[1])
                last = first + descriptor['samples']
                if descriptor.get('actual') != [first, last] or not descriptor.get('spans'):
                    descriptor.update(actual=[first, last], spans=[{'absolute': [first, last], 'offset': 0}])
        if name is not None and cid is None and not candidate_dir.name.startswith('contact-'):
            lane = index['lanes'][name]
            coverage = sorted((d for d in descriptors.values() if d.get('lane') == name and
                               not d.get('contact_id') and d.get('actual') is not None),
                              key=lambda d: d['actual'][0])
            origin = origins.get(name, coverage[0]['actual'][0] if coverage else lane.get('origin'))
            if origin is not None:
                gaps, end = [], origin
                for d in coverage:
                    _gap(gaps, end, d['actual'][0])
                    for hole in _holes(d):
                        _gap(gaps, *hole)
                    end = max(end, d['actual'][1])
                if lane.get('closed'):
                    for first, last in lane.get('gaps', []):
                        _gap(gaps, max(first, end), last)
                    end = max(end, lane.get('position', end))
                else:
                    lane['unknown_from'] = end
                lane.update(origin=origin, position=end, gaps=gaps,
                            expected=[{k: deepcopy(d[k]) for k in ('source', 'actual', 'spans')} for d in coverage])
                if coverage:
                    lane['rate'] = coverage[0]['rate']
    for contact in index['contacts'].values():
        if contact.get('state') == 'discarded':
            continue
        for name, ref in contact.get('lanes', {}).items():
            lane = index['lanes'].get(name, {})
            if lane.get('closed') or not ref.get('source'):
                continue
            base = local(ref['source'])
            if base is None:
                continue
            stems = (base.stem, _lane_stem(name, base.stem))
            member = re.compile('(?:' + '|'.join(re.escape(stem) for stem in stems) +
                                r')(?:-\d{8}T\d{4}Z(?:-s\d{20})?)?')
            coverage = sorted((d for d in descriptors.values()
                               if d.get('lane') == name and d.get('actual') is not None and
                               local(d['source']).parent == base.parent and
                               member.fullmatch(Path(d['source']).stem)),
                              key=lambda d: d['actual'][0])
            end = max((d['actual'][1] for d in coverage), default=lane.get('origin'))
            if end is None:
                continue
            start, stop = ref['requested']
            limit = end if stop is None else min(stop, end)
            first = start
            if 'selection' in ref:
                first = max(first, ref['selection'][0])
                limit = min(limit, ref['selection'][1])
            gaps, previous = [], lane.get('origin', coverage[0]['actual'][0] if coverage else end)
            selected = []
            for d in coverage:
                _gap(gaps, max(first, previous), min(limit, d['actual'][0]))
                for a, b in _holes(d):
                    _gap(gaps, max(first, a), min(limit, b))
                previous = max(previous, d['actual'][1])
                spans = [{'absolute': [max(first, span['absolute'][0]), min(limit, span['absolute'][1])],
                          'offset': span['offset'] + max(0, first - span['absolute'][0])}
                         for span in d['spans'] if span['absolute'][0] < limit and span['absolute'][1] > first]
                if spans:
                    selected.append({'source': d['source'], 'actual': [spans[0]['absolute'][0], spans[-1]['absolute'][1]],
                                     'spans': spans})
            anchor = min(max(start, lane.get('origin', end)), stop) if stop is not None else max(start, end)
            ref['actual'] = ([selected[0]['actual'][0], selected[-1]['actual'][1]] if selected else [anchor, anchor])
            ref['missing_pre_roll'] = ref['actual'][0] - start
            if stop is not None:
                contact_end = contact.get('end_positions', {}).get(ref['clock'], stop - 5 * ref['rate'])
                ref['missing_post_roll'] = max(0, stop - max(contact_end, ref['actual'][1] if selected else contact_end))
            ref.update(gaps=gaps, unknown_from=end, state='partial',
                       segments=[d['source'] for d in selected], coverage=selected)
            if 'selection' in ref:
                ref['selection'] = list(ref['actual'])
            contact['state'] = 'partial'
    return index


def _background(report):
    # Windows only. On macOS a background-QoS worker holding the GIL kept the
    # callback out 8-28 ms, so there and on Linux the workers stay at normal
    # priority and breathe instead.
    if sys.platform != 'win32':
        return
    from hfmodem.core import winproc
    try:
        winproc.lower_thread_priority()
    except OSError as exc:
        report(f'Recording storage priority unavailable: {exc}')


def record_window(lane, samples):
    """Admit a whole finite window against the separate acquired-window budget."""
    if len(samples):
        lane._push(lane.position, samples, acquired=True)


def _lock_candidate(file):
    """Exclusive and nonblocking; BlockingIOError while another owner holds it."""
    if fcntl is not None:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    position = file.tell()
    file.seek(_LOCK_OFFSET)
    try:
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        raise BlockingIOError(exc.errno, 'candidate lock is held') from None
    finally:
        file.seek(position)


def _control_end(value):
    """A descriptor number, or on Windows the name of a launch extra holding it."""
    if value.isdigit():
        return int(value)
    from hfmodem.core import claim
    return claim.extra(value)


def control_fds(environ=None):
    values = (os.environ if environ is None else environ).get('HFMODEM_RECORDING_CONTROL_OUT_FDS', '')
    return tuple(_control_end(value) for value in values.split(',') if value)


def control_fd(environ=None):
    return _control_end((os.environ if environ is None else environ)['HFMODEM_RECORDING_CONTROL_FD'])


def control_launch(env, *, read=None):
    """`(env, spawn options)` handing `read` and env's out ends to exactly one child."""
    env = dict(env)
    out = control_fds(env)
    env.pop('HFMODEM_RECORDING_CONTROL_OUT_FDS', None)
    if read is None:
        env.pop('HFMODEM_RECORDING_CONTROL_FD', None)
    if not _WINDOWS:
        if out:
            env['HFMODEM_RECORDING_CONTROL_OUT_FDS'] = ','.join(map(str, out))
        if read is not None:
            env['HFMODEM_RECORDING_CONTROL_FD'] = str(read)
        passed = (*out, *(() if read is None else (read,)))
        return env, (dict(pass_fds=passed) if passed else {})
    handles = {f'recording-out-{i}': fd for i, fd in enumerate(out)}
    if handles:
        env['HFMODEM_RECORDING_CONTROL_OUT_FDS'] = ','.join(handles)
    if read is not None:
        handles['recording-control'] = read
        env['HFMODEM_RECORDING_CONTROL_FD'] = 'recording-control'
    return env, (dict(handles=handles) if handles else {})


class ChildControlSender:
    """Bounded delivery to one attached recorder; end and close have reserved slots."""

    def __init__(self, fd, capacity=64, report=print):
        self.capacity, self.report = capacity, report
        self._fd = fd
        os.set_blocking(fd, False)
        self._condition = threading.Condition()
        self._queue, self._active = deque(), {}
        self._pending_ids = set()
        self._retiring = set()
        self._closed = False
        self._loss = False
        self._session, self._anchors = None, {}
        self._thread = threading.Thread(target=self._run, args=(fd,), name='recording-control',
                                        daemon=True)
        self._thread.start()

    def bind(self, session, anchors):
        with self._condition:
            self._session, self._anchors = session, deepcopy(anchors)
        session._child_controls.append(self)

    def _source_time(self, lane, position):
        source = lane.row.get('source_anchor')
        if source is not None:
            parent = next((item for item in self._session.lanes.values()
                           if item.clock == source['clock']), None)
            timing = parent.timing() if parent is not None else {}
            if 'monotonic_origin' in timing:
                sample = source['sample'] + (position - source['position']) * parent.rate / lane.rate
                return timing['monotonic_origin'] + sample / timing.get('measured_rate', parent.rate)
        timing = lane.timing()
        if 'monotonic_origin' in timing:
            return timing['monotonic_origin'] + position / timing.get('measured_rate', lane.rate)
        return None

    def send(self, record):
        record = deepcopy(record)
        if self._session is not None:
            positions = record.get('positions', {})
            record.setdefault('source_anchors', {
                lane.name: dict(clock=lane.clock, position=positions[lane.clock],
                                rate=timing.get('measured_rate', lane.rate),
                                monotonic=self._source_time(lane, positions[lane.clock]),
                                uncertainty_s=timing.get('clock_uncertainty_s'))
                for lane in self._session.lanes.values() if lane.clock in positions
                for timing in (lane.timing(),)})
        with self._condition:
            if self._closed:
                return
            op, cid = record['op'], record.get('contact_id')
            record.setdefault('anchors', deepcopy(self._anchors))
            record.setdefault('monotonic', time.monotonic())
            if op == 'begin':
                if cid in self._active:
                    return
                if len(self._pending_ids) >= self.capacity:
                    self._loss = True
                    self._condition.notify()
                    return
                self._active[cid] = record
                self._pending_ids.add(cid)
            elif op == 'end':
                if cid not in self._active:
                    return
                del self._active[cid]
            if self._fd is None:
                # No recorder: keep the lifecycle for the next one, write nothing.
                if op == 'end':
                    self._pending_ids.discard(cid)
                return
            if op not in ('end', 'close') and len(self._queue) >= self.capacity:
                self._loss = True
                # An undelivered begin has no matching child lifetime.
                if op == 'begin':
                    self._active.pop(cid, None)
                    self._pending_ids.discard(cid)
                self._condition.notify()
                return
            if op == 'end':
                # Pending anchors are expendable; lifecycle decisions are not.
                self._queue = deque(item for item in self._queue if item[1]['op'] != 'anchor')
            self._queue.append((self._fd, record))
            self._condition.notify()

    def replace(self, fd, anchors):
        os.set_blocking(fd, False)
        with self._condition:
            if self._closed or self._retiring:
                os.close(fd)
                raise RuntimeError('Recording child replacement is pending or closed')
            old, self._fd = self._fd, fd
            self._anchors = deepcopy(anchors)
            if old is not None:
                self._retiring.add(old)
                self._queue.append((old, dict(op='close')))
            for record in self._active.values():
                record['anchors'] = deepcopy(anchors)
                if anchors:
                    record['positions'] = {a['parent_clock']: a['sample'] for a in anchors.values()}
                record['monotonic'] = time.monotonic()
                self._queue.append((fd, deepcopy(record)))
            self._condition.notify()

    def idle(self):
        """No recorder reads the pipe any more: write nothing until `replace`. A
        write the worker has already taken may still go out and fail."""
        with self._condition:
            if self._closed or self._fd is None:
                return
            old, self._fd = self._fd, None
            for fd, record in self._queue:
                if fd == old and record['op'] == 'end':
                    self._pending_ids.discard(record.get('contact_id'))
            self._queue = deque(item for item in self._queue if item[0] != old)
            self._retiring.add(old)
            self._queue.append((old, dict(op='close')))
            self._condition.notify()

    def _run(self, first):
        import select
        descriptors = {first}
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._queue or self._loss or self._closed, timeout=1)
                    loss, self._loss = self._loss, False
                    item = self._queue.popleft() if self._queue else None
                    done = self._closed and not self._queue
                if loss:
                    try:
                        self.report('Recording child control overflow: coverage lost')
                    except Exception:
                        pass
                if item is not None:
                    fd, record = item
                    descriptors.add(fd)
                    data = (json.dumps(record) + '\n').encode()
                    deadline = time.monotonic() + _CLOSE_TIMEOUT
                    try:
                        while data:
                            if time.monotonic() >= deadline:
                                raise TimeoutError('child pipe stalled')
                            # Windows select() takes only sockets; there the write itself polls.
                            if os.name == 'nt' or select.select([], [fd], [], .1)[1]:
                                try:
                                    data = data[os.write(fd, data):]
                                except BlockingIOError:
                                    if os.name == 'nt':
                                        time.sleep(.01)
                    except (OSError, TimeoutError) as exc:
                        # A retired pipe's close may find its recorder already gone.
                        if not (record['op'] == 'close' and fd != self._fd):
                            try:
                                self.report(f'Recording child control unavailable: {exc}')
                            except Exception:
                                pass
                    if record['op'] == 'end':
                        with self._condition:
                            self._pending_ids.discard(record['contact_id'])
                    if record['op'] == 'close' and fd != self._fd:
                        os.close(fd)
                        descriptors.discard(fd)
                        with self._condition:
                            self._retiring.discard(fd)
                if done:
                    return
        finally:
            for fd in descriptors:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def close(self, timeout=_CLOSE_TIMEOUT):
        with self._condition:
            if not self._closed:
                if self._fd is not None:
                    self._queue.append((self._fd, dict(op='close')))
                self._closed = True
                self._condition.notify()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self.report('Recording child control close timed out: coverage lost')


class ChildControlReceiver:
    """Nonblocking JSON pipe reader with independent child source clocks."""

    def __init__(self, session, fd):
        self.session, self.fd = session, fd
        os.set_blocking(fd, False)
        self._buffer = b''
        self._active = {}
        self._closed = False

    def _anchors(self, record):
        anchors = record.get('anchors', {})
        if anchors:
            return anchors
        source = next((a for a in record.get('source_anchors', {}).values()
                       if a.get('monotonic') is not None), None)
        if source is None:
            return {}
        return {lane.name: dict(clock=lane.clock, parent_clock=source['clock'],
                    sample=source['position'], parent_rate=source['rate'],
                    rate=timing.get('measured_rate', lane.rate),
                    position=round((source['monotonic'] - timing['monotonic_origin']) *
                                   timing.get('measured_rate', lane.rate)),
                    monotonic=source['monotonic'],
                    uncertainty_s=(source['uncertainty_s'] + timing['clock_uncertainty_s']
                                   if source.get('uncertainty_s') is not None and
                                   timing.get('clock_uncertainty_s') is not None else None))
                for lane in self.session.lanes.values() for timing in (lane.timing(),)
                if 'monotonic_origin' in timing}

    def _positions(self, record):
        positions = record.get('positions', {})
        anchors = self._anchors(record)
        if anchors:
            return {a['clock']: round(a['position'] +
                    (positions[a['parent_clock']] - a['sample']) * a['rate'] / a['parent_rate'])
                    for a in anchors.values() if a['parent_clock'] in positions}
        if any(lane.clock in positions for lane in self.session.lanes.values()):
            return dict(positions)
        return {lane.clock: round((record.get('monotonic', time.monotonic()) -
                timing.get('monotonic_origin', time.monotonic())) *
                timing.get('measured_rate', lane.rate))
                for lane in self.session.lanes.values() for timing in (lane.timing(),)}

    def _finish(self):
        if self._closed:
            return
        self._closed = True
        if isinstance(self.session, ChildControlSender):
            self.session.close()
            return
        positions = {lane.clock: lane.position for lane in self.session.lanes.values()}
        for cid, provisional in self._active.items():
            self.session.end(cid, positions, keep=not provisional, reason='child control interrupted')
        self._active.clear()

    def _receive(self, record):
        if self._closed:
            return
        if isinstance(self.session, ChildControlSender):
            if record['op'] == 'close':
                self._finish()
            else:
                self.session.send(record)
            return
        op, cid = record['op'], record.get('contact_id')
        if op == 'close':
            self._finish()
        elif op == 'begin' and cid not in self._active:
            provisional = bool(record.get('provisional', False))
            self._active[cid] = provisional
            self.session.begin(cid, self._positions(record), provisional=provisional,
                               anchors=self._anchors(record),
                               parent_source_anchors=record.get('source_anchors', {}),
                               boundary_monotonic=record.get('monotonic'))
        elif op == 'end' and cid in self._active:
            self._active.pop(cid)
            self.session.end(cid, self._positions(record), keep=record.get('keep', True),
                             reason=record.get('reason', 'disconnect'),
                             end_parent_source_anchors=record.get('source_anchors', {}),
                             end_boundary_monotonic=record.get('monotonic'))

    def drain(self):
        if self._closed:
            return
        try:
            for _ in range(64):
                try:
                    data = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                if not data:
                    self._finish()
                    break
                self._buffer += data
                if len(self._buffer) > 262144:
                    raise ValueError('oversized child control record')
                while b'\n' in self._buffer:
                    line, self._buffer = self._buffer.split(b'\n', 1)
                    self._receive(json.loads(line))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reporter = getattr(self.session, 'report_later', self.session.report)
            reporter(f'Recording child control: {exc}')
            self._finish()


class _Inbox:
    """The drainer's queue: a producer may leave an item for the next tick."""

    def __init__(self):
        self.queue = deque()
        self._wake = threading.Event()

    def put(self, item, *, wake=True):
        self.queue.append(item)
        if wake:
            self._wake.set()

    def take(self, timeout):
        self._wake.wait(timeout)
        self._wake.clear()
        items = []
        while self.queue:
            items.append(self.queue.popleft())
        return items


def _runs(blocks):
    """Contiguous (first, pcm) blocks joined, in order; a hole starts a new run."""
    runs = []
    for first, pcm in blocks:
        if runs and runs[-1][1] == first:
            runs[-1][1] += len(pcm)
            runs[-1][2].append(pcm)
        else:
            runs.append([first, first + len(pcm), [pcm]])
    return [(first, np.concatenate(parts)) for first, _, parts in runs]


class Lane:
    """A bounded input queue and a monotonically advancing source clock."""

    def __init__(self, session, name, rate, clock, ownership):
        self.session, self.name, self.rate, self.clock = session, name, rate, clock
        self.ownership = ownership
        self.position = 0
        self.origin = None
        self.consumed = 0
        self.delivered = 0
        self.dropped = 0
        self.queued = 0
        self.window_queued = 0
        self.gaps = []
        self.disabled = False
        self.ring = deque() if session.policy == 'contacts' else None
        self.writer = None
        self.family = None
        self.gap_boundary = None
        self._drop_notice = None
        self._warnings = set()
        self._reported = None
        self._adc_previous = None
        self._adc_fit = None
        self._adc_anchors = []
        self._adc_stride = 1
        self._adc_seen = 0
        self._adc_residuals = deque(maxlen=64)
        self._adc_monotonic = None
        self._adc_facts = {}
        # Published by the producer under the input lock and copied into the
        # row by the drainer, which is the only thread that iterates rows.
        self._timing = None
        self._clock_default = {}
        self.row = {'rate': rate, 'clock': clock, 'source_id': _id(), 'index_id': _id(),
                    'gaps': [], 'state': 'ongoing', 'position': 0}

    def _mark_origin(self):
        if self.family is None or self.disabled:
            return
        origin = 'unknown' if self.origin is None else f'{self.origin:020d}'
        path = self.family.with_name(f'{self.family.stem}-o{origin}.lane')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            facts = {k: self.row[k] for k in ('rate', 'clock', 'source_id', 'index_id')}
            facts['source'] = self.row.get('source', _relative(self.family, self.session._root))
            with path.open('xb') as marker:
                marker.write(json.dumps(facts).encode('utf-8'))
                marker.flush()
                os.fsync(marker.fileno())
            for directory in (path.parent, path.parent.parent):
                _sync_directory(directory)
        except Exception as exc:
            self.disabled = True
            self.session._enqueue(lambda error=exc: self.session.report(f'Recording {self.name} origin: {error}'))

    def _clock(self, facts):
        # The caller holds the input lock. Producer facts outrank an owner's.
        for key, value in self._clock_default.items():
            facts.setdefault(key, value)
        if self._timing is not None:
            if 'measured_rate' not in self._timing:
                facts.pop('measured_rate', None)
            facts.update(self._timing)
        return facts

    def _settle(self):
        self.row.update(position=self.position, origin=self.origin, gaps=[list(g) for g in self.gaps])
        self._clock(self.row)
        if self.disabled and self.row['state'] == 'ongoing':
            self.row['state'] = 'disabled'

    def unrecorded(self):
        """Delivered samples that no finalized segment of this lane holds."""
        with self.session._state_lock:
            kept = sum(b - a for row in self.session.index['segments']
                       if row.get('lane') == self.name and 'contact_id' not in row
                       and row.get('state') == 'finalized'
                       for a, b in (span['absolute'] for span in row.get('spans', ())))
            with self.session._input_lock:
                return self.delivered - kept

    def timing(self):
        """Clock facts as of the latest block, ahead of the next index snapshot."""
        with self.session._input_lock:
            return self._clock({key: self.row[key] for key in _CLOCK_KEYS if key in self.row})

    def push_adc(self, pcm, times, *, overflow=False, captured_at=None):
        """Place ADC samples on the source clock; unknown loss has a lower bound of one.

        `times` is None where the host's stamps carry no converter time: blocks
        are then contiguous and only a reported overflow is loss.
        """
        adc = getattr(times, 'inputBufferAdcTime', None)
        current = getattr(times, 'currentTime', None)
        facts = self._adc_facts
        first = self.position
        rate = facts.get('measured_rate', self.rate)
        unknown = False
        # CoreAudio raises the overflow flag a few blocks after a stall the
        # converter's stamps have already placed, and on stalls that lost
        # nothing (two per session on the FTDX10's codec). Where the stamps
        # judge the block, the flag adds no loss of its own.
        judged = adc is not None and self._adc_previous is not None
        if judged:
            _, count = self._adc_previous
            fit_adc, fit_sample = self._adc_fit
            error = fit_sample + (adc - fit_adc) * rate - first
            residuals = np.asarray(self._adc_residuals)
            spread = (np.quantile(np.abs(residuals - np.median(residuals)), .9)
                      if len(residuals) >= 8 else 0.)
            # A flagged block has lost whatever the stamps say beyond rounding.
            tolerance = 2. if overflow else max(2., count / 8, 3 * spread)
            if abs(error) > tolerance:
                if error > 0:
                    first += round(error)
                else:
                    unknown = True
            elif not overflow:
                self._adc_residuals.append(error)
        if unknown or overflow and not judged and first <= self.position and self.origin is not None:
            facts.setdefault('uncertain', dict(from_sample=self.position,
                                              reason='input loss length unknown'))
            first = self.position + 1
        if adc is None:
            self._adc_previous = self._adc_fit = None
            self._adc_anchors.clear()
            self._adc_seen, self._adc_stride = 0, 1
            self._adc_residuals.clear()
            facts.pop('measured_rate', None)
            rate = self.rate
        else:
            new_epoch = self._adc_fit is None or first > self.position
            if new_epoch:
                self._adc_fit = (adc, first)
                self._adc_anchors.clear()
                self._adc_seen, self._adc_stride = 0, 1
                facts.pop('measured_rate', None)
                rate = self.rate
            if self._adc_seen % self._adc_stride == 0:
                self._adc_anchors.append((adc, first))
                if len(self._adc_anchors) == 65:
                    self._adc_anchors = self._adc_anchors[::2]
                    self._adc_stride *= 2
                # Separated anchors measure the crystal; individual residuals
                # never correct its rate. Thinning keeps the full epoch bounded.
                if len(self._adc_anchors) >= 33 and self._adc_seen % (16 * self._adc_stride) == 0:
                    anchors = self._adc_anchors
                    separation = len(anchors) // 2
                    rate = float(np.median([(b[1] - a[1]) / (b[0] - a[0])
                        for i, a in enumerate(anchors) for b in anchors[i + separation:]
                        if b[0] > a[0]]))
                    fit_adc, fit_sample = anchors[0]
                    phase = float(np.median([n - fit_sample - (t - fit_adc) * rate
                                             for t, n in anchors]))
                    self._adc_fit = (fit_adc - phase / rate, fit_sample)
                    facts['measured_rate'] = rate
                    self._adc_residuals.clear()
                    self._adc_residuals.extend((t - self._adc_fit[0]) * rate + fit_sample - n
                                              for t, n in anchors)
            self._adc_seen += 1
            self._adc_previous = (adc, len(pcm))
            facts['adc_origin'] = self._adc_fit[0] - self._adc_fit[1] / rate
        if self.origin is None or first > self.position or adc is not None and new_epoch:
            now = time.monotonic() if captured_at is None else captured_at
            origin = now + (adc - current if adc is not None and current is not None else 0)
            self._adc_monotonic = (origin, first, adc)
            # PortAudio and monotonic are sampled separately; no error bound is supplied.
            facts['clock_uncertainty_s'] = None
        if self._adc_monotonic is not None:
            origin, sample, anchor_adc = self._adc_monotonic
            facts['monotonic_origin'] = (
                origin + self._adc_fit[0] - anchor_adc - self._adc_fit[1] / rate
                if anchor_adc is not None and self._adc_fit is not None else origin - sample / rate)
        self.push(first, pcm, timing=dict(facts))

    def push(self, first_sample, pcm, *, timing=None):
        """Admit PCM at a source sample; `timing` replaces the lane's published clock facts.

        An empty `pcm` publishes `timing` alone.
        """
        self._push(first_sample, pcm, acquired=False, timing=timing)

    def _push(self, first_sample, pcm, *, acquired, timing=None):
        rec = self.session
        count = len(pcm)
        block = None if rec.policy == 'off' else np.array(pcm, copy=True)
        with rec._input_lock:
            if rec._closing:
                return
            if timing is not None:
                self._timing = timing
            if not count:
                return
            first, last = int(first_sample), int(first_sample) + count
            if (not self._clock_default and 'monotonic_origin' not in self.row
                    and 'monotonic_origin' not in (self._timing or ())):
                self._clock_default = dict(monotonic_origin=time.monotonic() - first / self.rate,
                                           clock_uncertainty_s=count / self.rate)
            if self.origin is None:
                self.origin = first
                self.consumed = first
                self._mark_origin()
            if first < self.position:
                rec._notice(self, 'overlap', f'Recording {self.name}: overlapping input dropped')
                return
            if first > self.position and self.position != 0:
                _gap(self.gaps, self.position, first)
            self.position = last
            self.delivered += count
            if rec.policy == 'off':
                self.consumed = last
                return
            queued = self.window_queued if acquired else self.queued
            capacity = max(5 * self.rate, count) if acquired else 2 * self.rate
            if count + queued > capacity:
                _gap(self.gaps, first, last)
                self.dropped += count
                rec._notice(self, 'overflow', f'Recording {self.name}: queue overflow; dropped blocks')
                if self._drop_notice is None:
                    self._drop_notice = [last]
                    rec._pcm.put(('advance', self, self._drop_notice))
                else:
                    self._drop_notice[0] = last
                return
            if acquired:
                self.window_queued += count
            else:
                self.queued += count
            self._drop_notice = None
            rec._pcm.put(('window' if acquired else 'pcm', self, first, block),
                         wake=acquired or self.queued >= _DRAIN_INTERVAL * self.rate)


class RecordingSession:
    """One PCM drainer and one storage worker; producers persist lane origins."""

    # State -> input or job queue; neither takes state in return. Publication
    # is storage-only and never nests with state, input or the job queue.
    def __init__(self, policy, directory, *, source='default', report=print,
                 publish=None, clock=time.monotonic, disk_usage=shutil.disk_usage, attached_children=True):
        if policy not in POLICIES:
            raise ValueError('Invalid recording policy')
        self.policy, self.directory = policy, Path(directory).absolute()
        # Contact recordings nest deep enough under a station data directory to
        # pass MAX_PATH, so every file operation goes through the extended form.
        self._root = _absolute(self.directory)
        self.report, self.publish = report, publish
        self.clock, self.disk_usage = clock, disk_usage
        self.lanes = {}
        self.contacts = {}
        self._child_controls = []
        self.index = {'policy': policy, 'source': source, 'lanes': {}, 'contacts': {}, 'segments': []}
        self._input_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._publication_lock = threading.Lock()
        self._snapshot = None
        self._stale = False
        self._snapshot_due = float('-inf')
        self._closing = self._closed = False
        self._check_at = self._warn_at = float('-inf')
        self._space_pending = False
        self._jobs = deque()
        self._storage_condition = threading.Condition()
        self._queued_descriptors = 0
        self._storage_stop = False
        self._faults = set()
        if policy != 'off':
            self._pcm = _Inbox()
            self._start_storage()
            try:
                self._storage_call(self._reserve_index)
            except Exception as exc:
                self._enqueue(lambda error=exc: self.report(f'Recording index: {error}'))
            self._pcm_thread = threading.Thread(target=self._drain, name='recording-pcm', daemon=True)
            self._pcm_thread.start()
        elif self._root.is_dir():
            self.index.update(load_recording_index(self._root, report=self.report))
        for fd in control_fds() if attached_children else ():
            ChildControlSender(os.dup(fd), report=report).bind(self, anchors={})

    def _reserve_index(self):
        recovered = load_recording_index(self._root, report=self.report)
        with self._state_lock:
            self.index.update(recovered)
        self._save()
        self._flush_index()

    def _storage_call(self, fn):
        finished, errors = threading.Event(), []
        def run():
            try:
                fn()
                self._flush_index()
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        self._enqueue(run)
        if not finished.wait(30):
            raise TimeoutError('Recording storage operation timed out')
        if errors:
            raise errors[0]

    def _publish_io(self, fn):
        with self._publication_lock:
            return fn()

    def _queue_publication(self, fn):
        self._enqueue(lambda: self._publish_io(fn))

    def _save(self):
        """Snapshot the index for the storage worker: segment, contact, control and storage boundaries."""
        with self._state_lock:
            with self._input_lock:
                for lane in self.lanes.values():
                    lane._settle()
            if self.policy == 'off':
                if not self.contacts:
                    return
            self._stale = False
            self._snapshot_due = time.monotonic() + _INDEX_INTERVAL
            self._audio_states()
            wake = self._snapshot is None
            self._snapshot = deepcopy(self.index)
            if wake and threading.current_thread() is not self._storage_thread:
                self._enqueue(lambda: None)

    def _touch(self):
        """Per-block progress: snapshot at most once per `_INDEX_INTERVAL`."""
        if time.monotonic() >= self._snapshot_due:
            self._save()
        else:
            self._stale = True

    def _flush_index(self, *, wait=False):
        with self._state_lock:
            snapshot, self._snapshot = self._snapshot, None
        if snapshot is None:
            return
        def publish():
            self._root.mkdir(parents=True, exist_ok=True)
            _durable_json(self._root / 'recording.json', snapshot)
        try:
            self._publish_io(publish)
        except Exception:
            with self._state_lock:
                if self._snapshot is None:
                    self._snapshot = snapshot
            raise

    def _notice(self, lane, kind, message):
        if kind in lane._warnings:
            return
        lane._warnings.add(kind)
        self._start_storage()
        def deliver():
            try:
                self.report(message)
            finally:
                with self._input_lock:
                    lane._warnings.discard(kind)
        self._enqueue(deliver)

    def _start_storage(self):
        if not hasattr(self, '_storage_thread'):
            self._storage_thread = threading.Thread(target=self._storage, name='recording-store', daemon=True)
            self._storage_thread.start()

    def stream(self, name, *, rate, clock, ownership=None):
        if name in self.index['lanes']:
            raise ValueError(f'Duplicate recording lane {name}')
        if rate <= 0 or int(rate) != rate:
            raise ValueError('Recording rate must be a positive integer')
        lane = Lane(self, name, rate, clock, ownership)
        # The drainer may be copying the index for another lane's blocks.
        with self._state_lock:
            self.lanes[name] = lane
            self.index['lanes'][name] = lane.row
            if self.policy != 'off':
                path = ownership.base if ownership is not None else self._root / _id() / 'receive.wav'
                lane.family = path.with_name(_lane_stem(name, path.stem) + '.wav')
                if ownership is None:
                    path = lane.family
                lane.row['source'] = _relative(path, self._root)
                lane._mark_origin()
            if self.policy == 'everything':
                try:
                    lane.writer = self._writer(lane, path)
                except Exception as exc:
                    self._failure(lane, exc, lane.consumed, lane.consumed)
                    if not lane.disabled:
                        lane.disabled = True
                        lane.row['state'] = 'failed'
        if self.policy != 'off':
            self._save()
            try:
                self._storage_call(lambda: None)
            except Exception as exc:
                self._enqueue(lambda error=exc: self.report(f'Recording index: {error}'))
        return lane

    def begin(self, contact_id, positions, *, provisional=False, **metadata):
        self._control('begin', contact_id, dict(positions), provisional, deepcopy(metadata))
        for child in self._child_controls:
            child.send(dict(op='begin', contact_id=contact_id, positions=positions, provisional=provisional))

    def end(self, contact_id, positions, *, keep=True, reason='disconnect', decided=None, **metadata):
        self._control('end', contact_id, dict(positions), keep, reason, deepcopy(metadata))
        if decided is not None:
            self._control('decided', decided)
        for child in self._child_controls:
            child.send(dict(op='end', contact_id=contact_id, positions=positions, keep=keep, reason=reason))

    def enrich(self, contact_id, **metadata):
        self._control('enrich', contact_id, deepcopy(metadata))

    def state(self, lane, sample, *, clock_generation=0, **changes):
        """Order decoder state changes with PCM on one lane's source clock."""
        self._control('state', lane, dict(sample=sample, clock_generation=clock_generation,
                                        **deepcopy(changes)))

    def _state_change(self, name, change):
        lane = self.lanes[name]
        if getattr(lane, 'context_generation', change['clock_generation']) != change['clock_generation']:
            lane.context_initial, lane.context_history = {}, []
        lane.context_generation = change['clock_generation']
        history = getattr(lane, 'context_history', [])
        history.append(change)
        cutoff = change['sample'] - 60 * lane.rate
        initial = getattr(lane, 'context_initial', {})
        while history and history[0]['sample'] < cutoff:
            old = history.pop(0)
            if old['clock_generation'] != initial.get('clock_generation'):
                initial = {}
            initial.update(old)
        lane.context_initial, lane.context_history = initial, history
        for contact in self.contacts.values():
            if not contact['done'] and name in contact['refs']:
                ctx = contact['row'].setdefault('context', {})
                ctx.setdefault('lane_states', {}).setdefault(name, {}).setdefault('changes', []).append(deepcopy(change))
                primary = 'rx' if 'rx' in contact['refs'] else next(iter(contact['refs']))
                if name == primary:
                    ctx.setdefault('changes', []).append(deepcopy(change))

    def after_pending(self, fn):
        self._control('barrier', fn)

    def _control(self, *args):
        with self._input_lock:
            if self._closing:
                return
            if self.policy == 'off':
                self._start_storage()
                self._enqueue(lambda: self._metadata_control(*args))
            else:
                for lane in self.lanes.values():
                    lane._drop_notice = None
                self._pcm.put(args)

    def _metadata_control(self, action, *args):
        publish = None
        with self._state_lock:
            if action == 'begin':
                self._begin(*args)
            elif action == 'end':
                publish = self._end(*args)
            elif action == 'enrich':
                contact_id, metadata = args
                if contact_id in self.contacts:
                    self.contacts[contact_id]['row'].update(metadata)
            elif action == 'state':
                self._state_change(*args)
            self._save()
        self._flush_index(wait=True)
        if publish is not None:
            self._publish(publish)
        if action == 'barrier':
            args[0]()
        elif action == 'decided':
            args[0]()

    def _enqueue(self, fn, *, descriptor=False):
        with self._storage_condition:
            self._jobs.append((fn, descriptor))
            self._storage_condition.notify()

    def _report(self, message):
        try:
            self.report(message)
        except Exception:
            pass

    def _storage_fault(self, exc, message):
        # A full disk fails every index write after it; the first one says so.
        code = getattr(exc, 'errno', None)
        if code is None or code not in self._faults:
            self._faults.add(code)
            self._report(message)

    def report_later(self, message):
        self._start_storage()
        self._enqueue(lambda: self._report(message))

    def _storage(self):
        _background(self._report)
        with gil.joined():
            self._store_jobs()

    def _store_jobs(self):
        while True:
            with self._storage_condition:
                self._storage_condition.wait_for(lambda: self._jobs or self._storage_stop, timeout=1)
                if not self._jobs:
                    if self._storage_stop:
                        return
                    continue
                fn, descriptor = self._jobs.popleft()
                if descriptor:
                    self._queued_descriptors -= 1
            try:
                fn()
                self._flush_index(wait=True)
            except Exception as exc:
                self._storage_fault(exc, f'Recording storage: {exc}')

    def _drain(self):
        with gil.joined():
            self._drain_blocks()

    def _drain_blocks(self):
        while True:
            pending = {}
            for item in self._pcm.take(_DRAIN_INTERVAL) + [None]:
                if item is not None and item[0] == 'pcm':
                    pending.setdefault(item[1], []).append(item[2:])
                    continue
                # Lanes are independent, so each lane's PCM is joined up to the
                # next control item and the control item still sees all of it.
                for lane, blocks in pending.items():
                    try:
                        runs = _runs(blocks)
                    except Exception:
                        runs = blocks  # joining is an economy; each block still stands alone
                    for first, block in runs:
                        self._handle('pcm', lane, first, block)
                pending.clear()
                if item is None:
                    break
                if self._handle(*item) == 'close':
                    return
            if self._stale and time.monotonic() >= self._snapshot_due:
                try:
                    self._save()
                except Exception as exc:
                    self._enqueue(lambda error=exc: self.report(f'Recording: {error}'))

    def _handle(self, action, *args):
        try:
            with self._state_lock:
                if action == 'seal':
                    for lane in self.lanes.values():
                        writers = [lane.writer, *(c['writers'].get(lane.name) for c in self.contacts.values())]
                        for writer in writers:
                            if writer is not None:
                                try:
                                    writer.seal()
                                except Exception as exc:
                                    spans = list(writer.coverage.values())
                                    if spans:
                                        self._failure(lane, exc, *spans[-1]['actual'])
                elif action == 'close':
                    self._finish_session()
                elif action == 'ready':
                    self._activate(*args)
                elif action == 'candidate-ready':
                    self._resume_candidate(*args)
                if action in ('pcm', 'window'):
                    lane, first, block = args
                    if action == 'pcm':
                        with self._input_lock:
                            lane.queued -= len(block)
                    try:
                        self._consume(lane, first, block)
                    finally:
                        if action == 'window':
                            with self._input_lock:
                                lane.window_queued -= len(block)
                elif action == 'advance':
                    lane, notice = args
                    with self._input_lock:
                        last = notice[0]
                        if lane._drop_notice is notice:
                            lane._drop_notice = None
                    self._break_lane(lane, lane.consumed, last)
                    lane.consumed = max(lane.consumed, last)
                    self._update_contacts(lane)
                elif action == 'begin':
                    self._begin(*args)
                elif action == 'end':
                    self._end(*args)
                elif action == 'enrich':
                    contact_id, metadata = args
                    if contact_id in self.contacts:
                        self.contacts[contact_id]['row'].update(metadata)
                elif action == 'state':
                    self._state_change(*args)
                elif action == 'barrier':
                    self._enqueue(lambda fn=args[0]: self._pcm.put(('barrier-ready', fn)))
                elif action == 'barrier-ready':
                    self._save()
                    self._enqueue(args[0])
                elif action == 'decided':
                    args[0]()
                if action in ('pcm', 'window', 'advance'):
                    self._touch()
                else:
                    self._save()
        except Exception as exc:
            self._enqueue(lambda error=exc: self.report(f'Recording: {error}'))
            if action in ('pcm', 'window'):
                self._lost(args[0], args[1] + len(args[2]))
        return action

    def _lost(self, lane, last):
        """A block the drainer could not consume still advanced its source clock."""
        try:
            with self._state_lock:
                self._break_lane(lane, lane.consumed, last)
                lane.consumed = max(lane.consumed, last)
                self._update_contacts(lane)
                self._save()
        except Exception as exc:
            self._enqueue(lambda error=exc: self.report(f'Recording: {error}'))

    def _check_space(self):
        try:
            self._space()
        finally:
            with self._state_lock:
                self._space_pending = False

    def _space(self):
        now = self.clock()
        if now - self._check_at < 60:
            return
        self._check_at = now
        try:
            free = self.disk_usage(self._root).free
        except OSError as exc:
            self.report(f'Recording space check: {exc}')
            return
        with self._state_lock:
            threshold = max(2 << 30, sum(l.rate * 2 * 7200 for l in self.lanes.values() if not l.disabled))
        if free < threshold and now - self._warn_at >= 600:
            self._warn_at = now
            self.report(f'Recording space low: {free} bytes available; threshold {threshold}')

    def _writer(self, lane, path, contact=None):
        writer = wav.SegmentedWriter(path, rate=lane.rate, ownership=lane.ownership if contact is None else None,
                                     defer_finalize=True, publication=self._queue_publication,
                                     sample_clock=True,
                                     sample_stem=lane.family.stem if contact is None else None,
                                     # Every writer failure is raised into _failure, which reports it.
                                     report=lambda message: None,
                                     started=contact['started'] if contact else None,
                                     closed=lambda p: self._segment(lane, writer, p, contact))
        writer.coverage = {}
        if contact is None:
            lane.row['expected'] = []
        return writer

    def _segment(self, lane, writer, path, contact):
        descriptor = writer.coverage.get(path, {'source': _relative(path, self._root),
                                               'actual': None, 'spans': []})
        descriptor.update(lane=lane.name, samples=writer._written, rate=lane.rate,
                          source_id=lane.row['source_id'], index_id=lane.row['index_id'],
                          capture_state=lane.row.get('capture_state', 'finalized'),
                          state='partial' if writer._segment_failed else 'pending')
        if contact is not None:
            descriptor['contact_id'] = contact['row']['id']
        if path in writer._staged:
            descriptor['pending_source'] = _relative(writer._staged[path], self._root)
        job = (descriptor, path, writer)
        with self._state_lock:
            self.index['segments'].append(descriptor)
            self._save()
        if descriptor['state'] == 'partial':
            if descriptor['actual'] is not None:
                with self._input_lock:
                    _gap(lane.gaps, *descriptor['actual'])
            if contact is not None and contact['provisional']:
                contact['retention_error'] = True
            return
        if contact is not None and contact['provisional']:
            if contact['decision'] is False:
                return
            if not contact.get('kept'):
                return
            self._admit(job)
        else:
            self._admit(job)

    def _admit(self, job):
        descriptor, path, writer = job
        with self._state_lock:
            with self._storage_condition:
                if self._queued_descriptors >= 8:
                    return
                self._queued_descriptors += 1
            descriptor['state'] = 'storing'
            self._save()
        self._enqueue(lambda: self._store(job), descriptor=True)

    def _store(self, job):
        descriptor, path, writer = job
        owner = writer.owners.get(path) if writer is not None else None
        try:
            local = _local(self._root, descriptor['source'])
            if (local is None or local != _absolute(path) or
                    ('pending_source' in descriptor and
                     _local(self._root, descriptor['pending_source']) is None)):
                raise ValueError('Recording segment path is outside session ownership')
            self._flush_index(wait=True)
            if writer is not None and writer._ownership is not None and owner is None:
                raise ValueError('Recording segment ownership is unavailable')
            with wave.open(str(path), 'rb') as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (
                        1, 2, descriptor['rate'], descriptor['samples']):
                    raise ValueError('Incomplete or mismatched recording PCM')
                remaining = descriptor['samples'] * 2
                while remaining:
                    data = audio.readframes(min(65536, remaining // 2))
                    if not data:
                        raise ValueError('Truncated recording PCM')
                    remaining -= len(data)
            facts = self._publish_io(lambda: wav.store(path, ownership=owner))
            if descriptor['actual'] is not None:
                facts.update(start_stream_sample=descriptor['actual'][0],
                             end_stream_sample=descriptor['actual'][1])
            state = ('finalized' if facts.get('sha256') and
                     descriptor.get('capture_state', 'finalized') == 'finalized' else 'partial')
            if owner is None:
                self._publish_io(lambda: wav._write_json(path.with_suffix('.json'), facts | {'capture_state': state,
                                    'samples': descriptor['samples'], 'samplerate': descriptor['rate']}))
            else:
                from hfmodem.core.capture_identity import audio_metadata
                prior = audio_metadata(path)
                self._publish_io(lambda: owner.sidecar.update({k: v for k, v in prior.items()
                                      if k not in owner.audio.record and k not in ('metadata_version', 'previous_sha256')}
                                     | facts | {'capture_state': state}))
            with self._state_lock:
                contact = self.contacts.get(descriptor.get('contact_id'))
                if state != 'finalized' and contact is not None and contact['provisional']:
                    contact['retention_error'] = True
                descriptor.pop('pending_source', None)
                descriptor.update(facts, state=state,
                                  source=_relative(path.with_name(facts['wav']), self._root))
                self._save()
            self._flush_index(wait=True)
            if contact is not None:
                self._release_contact(contact)
        except Exception as exc:
            self._storage_fault(exc, f'Recording storage {path}: {exc}')
            with self._state_lock:
                descriptor['state'] = 'partial'
                contact = self.contacts.get(descriptor.get('contact_id'))
                if contact is not None and contact['provisional']:
                    contact['retention_error'] = True
                self._save()

    def _failure(self, lane, exc, first, last):
        import errno
        with self._input_lock:
            _gap(lane.gaps, first, last)
        previous = lane._reported
        lane._reported = exc
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            if lane.disabled:
                return
            lane.disabled = True
            lane.row['state'] = 'disabled'
            self._enqueue(lambda: self.report(
                f'RECORDING STOPPED -- {lane.name}: the disk is full ({exc}); the session carries on unrecorded'))
            return
        if type(exc) is type(previous) and exc.args == previous.args:
            return
        self._enqueue(lambda: self.report(f'Recording {lane.name}: {exc}'))

    def _close_writer(self, lane, writer):
        try:
            writer.close()
        except Exception as exc:
            spans = list(writer.coverage.values())
            first, last = spans[-1]['actual'] if spans else (lane.consumed, lane.consumed)
            self._failure(lane, exc, first, last)

    def _write(self, lane, writer, pcm, first, *, fill=False):
        try:
            writer.set_position(first)
        except Exception as exc:
            spans = list(writer.coverage.values())
            if spans:
                self._failure(lane, exc, *spans[-1]['actual'])
            self._failure(lane, exc, first, first + len(pcm))
            return
        offset = 0
        for segment_index, path, within, count in list(writer.layout(len(pcm))):
            start, stop = first + offset, first + offset + count
            if lane.disabled:
                with self._input_lock:
                    _gap(lane.gaps, start, stop)
                offset += count
                continue
            opened = path not in writer.coverage
            if opened:
                writer.coverage[path] = {'source': _relative(path, self._root),
                                        'actual': [start, start], 'spans': []}
                if writer is lane.writer:
                    lane.row['expected'].append(writer.coverage[path])
            coverage = writer.coverage[path]
            coverage['actual'][1] = stop
            spans = coverage['spans']
            if not fill:
                if spans and spans[-1]['absolute'][1] == start:
                    spans[-1]['absolute'][1] = stop
                else:
                    spans.append({'absolute': [start, stop], 'offset': within})
            # A segment's expected path reaches the index before its PCM does.
            if opened:
                self._save()
            else:
                self._touch()
            try:
                writer.queue_through(writer.samples + count)
                writer.writeframes(pcm[offset:offset + count].tobytes())
                if writer._failed_index == segment_index:
                    self._failure(lane, writer.error, start, stop)
            except Exception as exc:
                bounds = coverage['actual'] if writer._failed_index == segment_index else (start, stop)
                self._failure(lane, exc, *bounds)
            offset += count

    def _break_lane(self, lane, first, last):
        if last <= first:
            return
        with self._input_lock:
            _gap(lane.gaps, first, last)
        # A hole ends its segment, so the files alone place every sample. One
        # within a second of the last segment's start stays in it as silence
        # the lane's gaps mark: churn stays at a segment a second and a lost
        # block costs the recording that block, never the second around it.
        close = (lane.gap_boundary is not None and first - lane.gap_boundary < lane.rate
                 and last - first <= lane.rate)
        writers = [(lane.writer, None), *((c['writers'].get(lane.name), c) for c in self.contacts.values())]
        for writer, contact in writers:
            if writer is None or writer._wav is None:
                continue
            if close and writer.position == first and self._fillable(contact, lane, writer, last - first):
                self._write(lane, writer, np.zeros(last - first, '<i2'), first, fill=True)
                continue
            try:
                writer.seal()
            except Exception as exc:
                coverage = writer.coverage.get(writer.paths[-1])
                self._failure(lane, exc, *(coverage['actual'] if coverage else (first, last)))
        if not close:
            lane.gap_boundary = last

    def _fillable(self, contact, lane, writer, count):
        if contact is None:
            return True
        stop = contact['refs'][lane.name]['requested'][1]
        return (not contact['done'] and stop is None and lane.name not in contact['waiting']
                and (not contact['provisional'] or self._candidate_ready(contact, writer, count)))

    def _consume(self, lane, first, block):
        if not self._space_pending:
            self._space_pending = True
            self._enqueue(self._check_space)
        if first > lane.consumed:
            self._break_lane(lane, lane.consumed, first)
            lane.consumed = first
        finite = np.isfinite(block)
        if not finite.all():
            self._notice(lane, 'nonfinite', f'Recording {lane.name}: nonfinite PCM is unavailable')
            edges = np.flatnonzero(np.r_[True, finite[1:] != finite[:-1], True])
            for a, b in zip(edges[:-1], edges[1:]):
                if finite[a]:
                    self._consume(lane, first + int(a), block[a:b])
                else:
                    self._break_lane(lane, first + int(a), first + int(b))
                    lane.consumed = first + int(b)
                    self._update_contacts(lane)
            return
        pcm = np.round(np.clip(block, -1.0, 32767 / 32768) * 32768.0).astype('<i2')
        last = first + len(pcm)
        if lane.disabled:
            with self._input_lock:
                _gap(lane.gaps, first, last)
        if lane.writer is not None:
            self._write(lane, lane.writer, pcm, first)
        if self.policy == 'contacts':
            for contact in self.contacts.values():
                if contact['done'] or lane.name not in contact['refs']:
                    continue
                self._contact_pcm(contact, lane, first, pcm)
            lane.ring.append((first, pcm))
            cutoff = last - 60 * lane.rate
            while lane.ring and lane.ring[0][0] + len(lane.ring[0][1]) <= cutoff:
                lane.ring.popleft()
            if lane.ring and lane.ring[0][0] < cutoff:
                start, data = lane.ring.popleft()
                lane.ring.appendleft((cutoff, data[cutoff - start:].copy()))
        lane.consumed = last
        self._update_contacts(lane)

    def _begin(self, contact_id, positions, provisional, metadata):
        if contact_id in self.contacts:
            return
        directory = self._root / ('contact-' + _id())
        row = {'id': contact_id, 'contact_id': contact_id, 'positions': positions, 'state': 'ongoing',
               'directory': _relative(directory, self._root), 'lanes': {},
               'provisional': provisional, **metadata}
        if 'context' not in row:
            primary = self.lanes.get('rx', next(iter(self.lanes.values()), None))
            initial = {k: v for k, v in getattr(primary, 'context_initial', {}).items()
                       if k not in ('sample', 'clock_generation')}
            row['context'] = {'initial': deepcopy(initial),
                              'changes': deepcopy(getattr(primary, 'context_history', []))}
        primary = self.lanes.get('rx', next(iter(self.lanes.values()), None))
        state = deepcopy(row['context'].get('initial', {}))
        start = positions.get(primary.clock) if primary is not None else None
        generation = getattr(primary, 'context_generation', 0)
        for change in sorted(row['context'].get('changes', []), key=lambda c: c['sample']):
            if (start is not None and change['sample'] <= start
                    and change.get('clock_generation') == generation):
                state.update(change)
        key = row.get('key')
        calls = state.get('calls', {})
        row.setdefault('protocol', key[1] if key else state.get('protocol'))
        row.setdefault('calls', {'local': key[3] if key else calls.get('local'),
                                 'peer': row.get('peer', calls.get('peer'))})
        row.setdefault('local_role', state.get('local_role'))
        contact = {'row': row, 'refs': row['lanes'], 'writers': {}, 'provisional': provisional,
                   'decision': None, 'done': False, 'directory': directory, 'lock': None,
                   'interval_states': {}, 'started': int(time.time()), 'preparing': False,
                   'buffers': {}, 'waiting': {}, 'authorized': set(), 'losses': {}}
        self.contacts[contact_id] = contact
        self.index['contacts'][contact_id] = row
        if self.policy == 'off':
            return
        if provisional:
            contact['preparing'] = True
            contact['candidate'] = {'candidate_id': _id(), 'contact_id': contact_id,
                                    'directory': str(self.directory / directory.name),
                                    'state': 'undecided', 'owned_files': []}
        for lane in self.lanes.values():
            if lane.clock not in positions:
                continue
            start = positions[lane.clock] - 60 * lane.rate
            available = lane.ring[0][0] if lane.ring else lane.origin
            actual = max(start, available if available is not None else positions[lane.clock])
            ref = {'kind': 'interval' if self.policy == 'everything' else 'cutout',
                   'lane': lane.name, 'rate': lane.rate, 'clock': lane.clock,
                   'source_id': lane.row['source_id'], 'index_id': lane.row['index_id'],
                   'requested': [start, None], 'actual': [actual, actual],
                   'missing_pre_roll': actual - start, 'missing_post_roll': 0,
                   'gaps': [], 'state': 'ongoing', 'segments': []}
            contact['refs'][lane.name] = ref
            ctx = row['context']
            ctx.setdefault('lanes', {}).setdefault(lane.name, {}).setdefault(
                'clock_generation', getattr(lane, 'context_generation', 0))
            ctx.setdefault('lane_states', {}).setdefault(lane.name, {
                'initial': deepcopy({k: v for k, v in getattr(lane, 'context_initial', {}).items()
                                     if k not in ('sample', 'clock_generation')}),
                'changes': deepcopy(getattr(lane, 'context_history', []))})
            if self.policy == 'contacts':
                path = directory / _id() / (_lane_stem(lane.name) + '.wav')
                ref['source'] = _relative(path, self._root)
                snapshot = [(max(first, actual), data[max(0, actual - first):].copy())
                            for first, data in lane.ring if first + len(data) > actual]
                if provisional:
                    contact['buffers'][lane.name] = {'pre': snapshot, 'live': [], 'samples': 0}
                    capacity = max(1, wav.SEGMENT_SAMPLES * lane.rate // 48000)
                    spans = []
                    for first, data in snapshot:
                        _gap(spans, first, first + len(data))
                    for first, last in spans or [(lane.consumed, lane.consumed)]:
                        self._candidate_paths(contact, path, lane.rate, capacity, 0,
                                              last - first + 2 * capacity, origin=first)
                else:
                    self._open_contact_lane(contact, lane, snapshot)
            else:
                ref['source'] = lane.row['source']
        self._reconcile(contact)
        self._save()
        if provisional:
            self._enqueue(lambda: self._prepare_candidate(contact))

    def _open_contact_lane(self, contact, lane, blocks):
        ref = contact['refs'][lane.name]
        if lane.disabled:
            ref['state'] = 'partial'
            return
        try:
            writer = self._writer(lane, self._root / ref['source'], contact)
        except Exception as exc:
            ref['state'] = 'partial'
            self._failure(lane, exc, ref['actual'][0], lane.consumed)
            return
        contact['writers'][lane.name] = writer
        for first, data in blocks:
            self._contact_pcm(contact, lane, first, data)

    def _prepare_candidate(self, contact):
        directory = contact['directory']
        identity = {k: contact['candidate'][k] for k in ('candidate_id', 'contact_id', 'directory')}
        ready = False
        def prepare():
            directory.mkdir()
            lock = (directory / 'candidate.lock').open('x+b')
            _lock_candidate(lock)
            contact['lock'] = lock
            lock.write(json.dumps(identity).encode())
            lock.flush()
            os.fsync(lock.fileno())
            _durable_json(directory / 'candidate.identity', identity)
        try:
            self._publish_io(prepare)
            self._candidate_save(contact)
            self._flush_index()
            ready = True
        finally:
            self._pcm.put(('ready', contact, ready))

    def _activate(self, contact, ready):
        contact['preparing'] = False
        if not ready:
            contact['retention_error'] = True
        for name, buffer in contact['buffers'].items():
            lane = self.lanes[name]
            blocks = buffer['pre'] + buffer['live']
            if ready:
                self._open_contact_lane(contact, lane, blocks)
            else:
                for first, data in blocks:
                    _gap(contact['losses'].setdefault(name, []), first, first + len(data))
        contact['buffers'].clear()
        if contact['decision'] is not None:
            self._resolve_decision(contact)
        self._update_contacts()

    def _resume_candidate(self, contact, lane):
        blocks = contact['waiting'].pop(lane.name)
        if contact['decision'] is False:
            return
        for first, pcm in blocks:
            self._contact_pcm(contact, lane, first, pcm, retry=False)
        self._update_contacts(lane)

    def _contact_pcm(self, contact, lane, first, pcm, *, retry=True):
        ref = contact['refs'][lane.name]
        stop = ref['requested'][1]
        end = first + len(pcm) if stop is None else min(stop, first + len(pcm))
        if end <= first:
            return
        pcm = pcm[:end - first]
        if lane.name in contact['waiting']:
            blocks = contact['waiting'][lane.name]
            if sum(len(data) for _, data in blocks) + len(pcm) <= 2 * lane.rate:
                blocks.append((first, pcm.copy()))
                return
        elif contact['preparing']:
            buffer = contact['buffers'][lane.name]
            if buffer['samples'] + len(pcm) <= 2 * lane.rate:
                buffer['live'].append((first, pcm.copy()))
                buffer['samples'] += len(pcm)
                return
        else:
            writer = contact['writers'].get(lane.name)
            if writer is not None and not lane.disabled:
                try:
                    writer.set_position(first)
                except Exception as exc:
                    coverage = list(writer.coverage.values())
                    if coverage:
                        self._failure(lane, exc, *coverage[-1]['actual'])
                    self._failure(lane, exc, first, end)
                    _gap(contact['losses'].setdefault(lane.name, []), first, end)
                    return
                self._candidate_files(contact, writer, len(pcm))
                if not contact['provisional'] or self._candidate_ready(contact, writer, len(pcm)):
                    self._write(lane, writer, pcm, first)
                    return
                if retry and len(pcm) <= 2 * lane.rate:
                    contact['waiting'][lane.name] = [(first, pcm.copy())]
                    self._enqueue(lambda: self._pcm.put(('candidate-ready', contact, lane)))
                    return
        _gap(contact['losses'].setdefault(lane.name, []), first, end)
        with self._input_lock:
            lane.dropped += end - first
            self._notice(lane, 'candidate', f'Recording {lane.name}: candidate storage pending; dropped blocks')

    def _reconcile(self, contact):
        for name, ref in contact['refs'].items():
            lane = self.lanes[name]
            start, stop = ref['requested']
            limit = lane.consumed if stop is None else stop
            edge = contact['row']['positions'][lane.clock]
            with self._input_lock:
                if lane.origin is not None:
                    _gap(lane.gaps, edge, lane.origin)
                elif stop is not None and self._closing:
                    _gap(lane.gaps, edge, stop)
                gaps = [list(g) for g in lane.gaps]
            for first, last in contact['losses'].get(name, ()):
                _gap(gaps, first, last)
            ref['gaps'] = [[max(a, start), min(b, limit)] for a, b in gaps if a < limit and b > start]
            writer = contact['writers'].get(name, lane.writer)
            spans = [span['absolute'] for coverage in (writer.coverage.values() if writer else ())
                     for span in coverage['spans']
                     if span['absolute'][0] < limit and span['absolute'][1] > start]
            if 'selection' in ref:
                first, last = ref['selection']
                spans = [[max(a, first), min(b, last)] for a, b in spans if a < last and b > first]
            available = []
            for first, last in spans:
                for a, b in gaps:
                    if b <= first or a >= last:
                        continue
                    if a > first:
                        available.append([first, a])
                    first = max(first, b)
                if first < last:
                    available.append([first, last])
            spans = available
            if spans:
                ref['actual'] = [max(start, min(a for a, _ in spans)),
                                 min(limit, max(b for _, b in spans))]
            else:
                origin = max(start, lane.origin if lane.origin is not None else edge)
                if 'selection' in ref:
                    origin = ref['selection'][0]
                anchor = min(origin, limit) if stop is not None else origin
                ref['actual'] = [anchor, anchor]
            ref['missing_pre_roll'] = ref['actual'][0] - start
            if stop is not None:
                end = contact['row']['end_positions'].get(lane.clock, limit - 5 * lane.rate)
                last = ref['actual'][1] if spans else end
                ref['missing_post_roll'] = max(0, stop - max(end, last))

    def _end(self, contact_id, positions, keep, reason, metadata):
        contact = self.contacts.get(contact_id)
        if contact is None or contact['decision'] is not None:
            return
        contact['decision'] = bool(keep)
        row = contact['row']
        row.update(metadata, end_positions=positions, reason=reason)
        if reason != 'disconnect':
            row['state'] = 'partial'
        if self.policy == 'off':
            contact['done'] = True
            if row['state'] != 'partial':
                row['state'] = 'finalized' if keep else 'discarded'
            if keep:
                return contact
            return
        for name, ref in contact['refs'].items():
            lane = self.lanes[name]
            end = positions.get(lane.clock, lane.consumed)
            ref['requested'][1] = end + 5 * lane.rate
        if not contact['preparing']:
            self._resolve_decision(contact)

    def _resolve_decision(self, contact):
        if contact['provisional'] and contact['decision']:
            for name, writer in contact['writers'].items():
                lane = self.lanes[name]
                stop = contact['refs'][name]['requested'][1]
                self._candidate_files(contact, writer, max(0, stop - lane.consumed))
        if contact['provisional']:
            self._enqueue(lambda: self._decide(contact))
        if not contact['decision']:
            self._reconcile(contact)
            for name, writer in contact['writers'].items():
                self._close_writer(self.lanes[name], writer)
            self._reconcile(contact)
            contact['done'] = True
            contact['row']['state'] = 'discarded'
            self._enqueue(lambda: self._discard(contact))
        else:
            self._update_contacts()

    def _decide(self, contact):
        with self._state_lock:
            previous = contact['candidate']['state']
            contact['candidate']['state'] = 'keep' if contact['decision'] else 'discard'
            facts = deepcopy(contact['candidate'])
        try:
            self._publish_io(lambda: _durable_json(contact['directory'] / 'candidate.json', facts))
        except Exception:
            with self._state_lock:
                contact['candidate']['state'] = previous
            raise
        with self._state_lock:
            contact['kept'] = contact['decision']
            held = [d for d in self.index['segments'] if d.get('contact_id') == contact['row']['id']
                    and d['state'] == 'pending']
        if contact['decision']:
            for descriptor in held:
                with self._state_lock:
                    descriptor['state'] = 'storing'
                    self._save()
                self._store((descriptor, self._root / descriptor['source'],
                             contact['writers'].get(descriptor['lane'])))

    def _update_contacts(self, lane=None, closing=False):
        for contact in self.contacts.values():
            # A finished contact is reconciled by every snapshot (_audio_states).
            if contact['done']:
                continue
            self._reconcile(contact)
            if contact['preparing'] or contact['waiting']:
                continue
            if contact['decision'] is not True:
                continue
            if not closing and any(self.lanes[n].consumed < ref['requested'][1]
                                   for n, ref in contact['refs'].items()):
                continue
            for name, writer in contact['writers'].items():
                self._close_writer(self.lanes[name], writer)
            self._reconcile(contact)
            contact['done'] = True
            for name, ref in contact['refs'].items():
                ref['selection'] = list(ref['actual'])
                contact['interval_states'][name] = ('partial' if ref['missing_post_roll'] or ref['gaps']
                                                     else 'finalized')
                ref['state'] = 'pending'
                writer = contact['writers'].get(name, self.lanes[name].writer)
                if writer is None:
                    contact['interval_states'][name] = 'partial'
                ref['coverage'] = deepcopy([coverage for coverage in (writer.coverage.values() if writer else ())
                                            if coverage['actual'][0] < ref['actual'][1]
                                            and coverage['actual'][1] > ref['actual'][0]])
            self._enqueue(lambda c=contact: self._publish(c))

    def _audio_states(self):
        for contact in self.contacts.values():
            if not contact['done'] or contact['decision'] is not True or self.policy == 'off':
                continue
            self._reconcile(contact)
            for name, ref in contact['refs'].items():
                writer = contact['writers'].get(name, self.lanes[name].writer)
                first, last = ref['selection']
                coverage = [c for c in (writer.coverage.values() if writer else ())
                            if c['actual'][0] < last and c['actual'][1] > first]
                states = [c.get('state', 'pending') for c in coverage]
                ref['segments'] = [c['source'] for c in coverage]
                ref['state'] = ('partial' if writer is None or 'partial' in states or ref['gaps'] or
                                ref['missing_post_roll'] or contact['interval_states'].get(name) == 'partial'
                                else 'pending' if not states or any(s != 'finalized' for s in states)
                                else 'finalized')
            if contact['row'].get('reason') != 'disconnect':
                contact['row']['state'] = 'partial'
            else:
                states = [r['state'] for r in contact['refs'].values()]
                contact['row']['state'] = ('partial' if 'partial' in states else
                                          'pending' if 'pending' in states else 'finalized')
            if contact['provisional'] and (not contact.get('kept') or contact.get('retention_error')):
                contact['row']['state'] = 'pending'

    def _publish(self, contact):
        with self._state_lock:
            if self.policy != 'off' and contact['provisional'] and (
                    not contact.get('kept') or contact.get('retention_error')):
                contact['row']['state'] = 'pending'
                self._save()
                return
            row = deepcopy(contact['row'])
            for name in row['lanes']:
                row.setdefault('context', {}).setdefault('lanes', {}).setdefault(name, {}).setdefault(
                    'clock_generation', getattr(self.lanes[name], 'context_generation', 0))
        self._context_record(row)
        with self._state_lock:
            contact['row']['context'] = row['context']
            for name, ref in row['lanes'].items():
                contact['refs'][name].update(clock_generation=ref['clock_generation'],
                                            participation=ref['participation'])
            self._save()
            row = deepcopy(contact['row'])
        try:
            self._flush_index(wait=True)
            if self.publish is not None:
                self._publish_io(lambda: self.publish(row))
        except Exception:
            with self._state_lock:
                contact['retention_error'] = True
                self._save()
            raise
        with self._state_lock:
            contact['publication_done'] = True
        self._release_contact(contact)

    def _context_record(self, row):
        ctx = deepcopy(row.get('context', {}))
        ctx.update(version=1, contact_id=row['id'])
        ctx.setdefault('id', _id())
        ctx['endpoint_generation'] = row.get('endpoint_generation',
            row.get('key', [None])[-1] if row.get('key') else ctx.get('endpoint_generation'))
        ctx.setdefault('initial', {'expected_session': None})
        ctx['initial'].setdefault('expected_session', None)
        ctx.setdefault('protocol', row.get('protocol'))
        ctx.setdefault('calls', dict(row['calls'], peer=row.get('peer', row['calls'].get('peer'))))
        ctx.setdefault('local_role', row.get('local_role'))
        for name in ('transcript', 'probe'):
            link = ctx.get(name)
            if not link or link.get('sha256'):
                continue
            try:
                path = _local(self._root, link['path'])
                first, last = link['range']
                if first < 0 or last < first:
                    raise ValueError('Invalid context range')
                digest, remaining = hashlib.sha256(), last - first
                with path.open('rb') as stream:
                    stream.seek(first)
                    while remaining:
                        block = stream.read(min(65536, remaining))
                        if not block:
                            raise ValueError('Context range unavailable')
                        digest.update(block)
                        remaining -= len(block)
                link['sha256'] = digest.hexdigest()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                ctx.update(state='incomplete', reason=str(exc))
        ctx.setdefault('lanes', {})
        for name, ref in row['lanes'].items():
            mapping = ctx['lanes'].setdefault(name, {})
            old_id = mapping.get('source_id')
            mapping.update({k: ref[k] for k in ('source_id', 'index_id', 'clock', 'rate')})
            mapping.setdefault('clock_generation', ref.get('clock_generation', 0))
            ref['clock_generation'] = mapping['clock_generation']
            mapping['retained'] = list(ref['actual'])
            for tx in ctx.get('tx', []):
                if old_id is not None and tx.get('source_id') == old_id:
                    tx['source_id'] = ref['source_id']
            ref['participation'] = [row['positions'].get(ref['clock']),
                                    row.get('end_positions', {}).get(ref['clock'])]
            schedule = ctx.get('lane_states', {}).get(name)
            if schedule is not None:
                changes = sorted((c for c in schedule.get('changes', [])
                                  if c['clock_generation'] == mapping['clock_generation']), key=lambda c: c['sample'])
                first, last = ref['actual']
                for change in changes:
                    if change['sample'] <= first:
                        schedule.setdefault('initial', {}).update({k: v for k, v in change.items()
                            if k not in ('sample', 'clock_generation')})
                schedule['changes'] = [c for c in changes if first < c['sample'] < last]
        primary = row['lanes'].get('rx', next(iter(row['lanes'].values()), None))
        if primary is not None:
            first, last = ctx['retained'] = list(primary['actual'])
            generation = primary['clock_generation']
            changes = sorted((c for c in ctx.get('changes', [])
                              if c.get('clock_generation') == generation), key=lambda c: c['sample'])
            for change in changes:
                if change['sample'] <= first:
                    ctx['initial'].update({k: v for k, v in change.items()
                                           if k not in ('sample', 'clock_generation')})
            ctx['changes'] = [c for c in changes if first < c['sample'] < last]
            grid = [g for g in ctx.get('grid', []) if g.get('clock_generation') == generation]
            prior = [g for g in grid if g['sample'] <= first]
            ctx['grid'] = prior[-1:] + [g for g in grid if first < g['sample'] < last]
        def retained_tx(tx):
            first = tx.get('ptt_start_sample', tx.get('audio_start_sample'))
            last = tx.get('ptt_end_sample', tx.get('audio_end_sample'))
            return first is not None and last is not None and any(
                tx.get('source_id') == ref['source_id'] and tx.get('clock_generation') == ref['clock_generation']
                and first < ref['actual'][1] and last > ref['actual'][0] for ref in row['lanes'].values())
        if 'tx' in ctx:
            ctx['tx'] = [tx for tx in ctx['tx'] if retained_tx(tx)]
        row['context'] = ctx

    def _release_contact(self, contact):
        with self._state_lock:
            if not contact.get('publication_done') or contact.get('retention_error'):
                return
            if any(d['state'] != 'finalized' for d in self.index['segments']
                   if d.get('contact_id') == contact['row']['id']):
                return
            lock, contact['lock'] = contact['lock'], None
        if lock is not None:
            lock.close()

    def _candidate_save(self, contact):
        with self._state_lock:
            facts = deepcopy(contact['candidate'])
        try:
            self._publish_io(lambda: _durable_json(contact['directory'] / 'candidate.json', facts))
        except Exception:
            with self._state_lock:
                contact['retention_error'] = True
                self._save()
            raise
        with self._state_lock:
            contact['authorized'].update(facts['owned_files'])

    def _candidate_paths(self, contact, path, rate, capacity, first, stop, *, origin=0):
        owned = contact['candidate']['owned_files']
        paths = [path.with_suffix('.json')]
        for i in range(first // capacity, (max(first + 1, stop) - 1) // capacity + 1):
            stamp = time.strftime('%Y%m%dT%H%MZ', time.gmtime(contact['started']))
            paths.append(path.with_name(f'{path.stem}-{stamp}-s{origin + i * capacity:020d}.wav'))
        for item in paths:
            for p in (item, item.with_suffix('.json')):
                name = _relative(p, contact['directory'])
                if name not in owned:
                    owned.append(name)

    def _candidate_files(self, contact, writer, additional=0):
        if not contact['provisional']:
            return
        before = len(contact['candidate']['owned_files'])
        if writer.position is None:
            return
        for _, path, _, _ in writer.layout(additional + 2 * writer.segment_samples):
            for item in (path, path.with_suffix('.json')):
                name = _relative(item, contact['directory'])
                if name not in contact['candidate']['owned_files']:
                    contact['candidate']['owned_files'].append(name)
        if len(contact['candidate']['owned_files']) != before:
            self._enqueue(lambda: self._candidate_save(contact))

    def _candidate_ready(self, contact, writer, additional):
        for _, path, _, _ in writer.layout(additional):
            if _relative(path, contact['directory']) not in contact['authorized']:
                return False
        return True

    def _discard(self, contact):
        if not contact['provisional']:
            return
        try:
            self._publish_io(lambda: _remove_candidate(contact['directory']))
            with self._state_lock:
                prefix = _relative(contact['directory'], self._root) + '/'
                self.index['segments'][:] = [d for d in self.index['segments'] if not d['source'].startswith(prefix)]
                self._save()
        finally:
            if contact['lock'] is not None:
                contact['lock'].close()
                contact['lock'] = None

    def _finish_session(self):
        for cid, contact in self.contacts.items():
            if contact['decision'] is None:
                publish = self._end(cid, {l.clock: l.consumed for l in self.lanes.values()},
                                    not contact['provisional'], 'owner teardown', {})
                if publish is not None:
                    self._enqueue(lambda c=publish: self._publish(c))
        for lane in self.lanes.values():
            if lane.writer is not None:
                self._close_writer(lane, lane.writer)
        self._update_contacts(closing=True)
        for lane in self.lanes.values():
            lane.row['closed'] = True
        self._save()

    def _pending(self):
        for descriptor in self.index['segments']:
            if descriptor['state'] != 'pending':
                continue
            if descriptor.get('recovered'):
                if descriptor.get('stored_state') == 'finalized':
                    descriptor['state'] = 'finalized'
                    self._save()
                    continue
                if (descriptor.get('contact_directory') or descriptor.get('pending_source') or descriptor.get('owned') or
                        descriptor.get('samples') is None or descriptor.get('rate') is None):
                    continue
            contact = self.contacts.get(descriptor.get('contact_id'))
            if contact is not None and contact['provisional'] and not contact.get('kept'):
                continue
            path = self._root / descriptor['source']
            writers = [l.writer for l in self.lanes.values()]
            writers.extend(w for c in self.contacts.values() for w in c['writers'].values())
            writer = next((w for w in writers if w is not None and path in w.owners), None)
            with self._state_lock:
                descriptor['state'] = 'storing'
                self._save()
            self._store((descriptor, path, writer))

    def close(self):
        for child in self._child_controls:
            child.close()
        deadline = time.monotonic() + _CLOSE_TIMEOUT
        def remaining():
            return max(0, deadline - time.monotonic())
        if not self._input_lock.acquire(timeout=remaining()):
            raise TimeoutError('Recording input close timed out')
        try:
            if self._closed:
                return
            if not self._closing:
                self._closing = True
                if self.policy != 'off':
                    self._pcm.put(('seal',))
                    self._pcm.put(('barrier', lambda: self._pcm.put(('close',))))
                elif hasattr(self, '_storage_thread'):
                    self._enqueue(self._finish_session)
        finally:
            self._input_lock.release()
        if self.policy != 'off':
            self._pcm_thread.join(timeout=remaining())
            if self._pcm_thread.is_alive():
                raise TimeoutError('Recording PCM close timed out')
            self._enqueue(self._pending)
        if hasattr(self, '_storage_thread'):
            finished = threading.Event()
            def finish():
                try:
                    self._save()
                    self._flush_index()
                finally:
                    finished.set()
            self._enqueue(finish)
            if not finished.wait(remaining()):
                raise TimeoutError('Recording pending storage timed out')
            with self._storage_condition:
                self._storage_stop = True
                self._storage_condition.notify()
            self._storage_thread.join(timeout=remaining())
            if self._storage_thread.is_alive():
                raise TimeoutError('Recording worker close timed out')
        self._closed = True


def lane_positions(session, positions=None):
    """Map a source snapshot through submitted-lane sample anchors."""
    snapshot = ({lane.clock: lane.position for lane in session.lanes.values()}
                if positions is None else dict(positions))
    result = dict(snapshot)
    by_clock = {lane.clock: lane for lane in session.lanes.values()}
    for lane in session.lanes.values():
        anchor = lane.row.get('source_anchor')
        if anchor is not None and anchor['clock'] in snapshot:
            source = by_clock[anchor['clock']]
            result[lane.clock] = anchor['position'] + round(
                (snapshot[source.clock] - anchor['sample']) * lane.rate / source.rate)
    return result


def inherit_boundaries(target, owner):
    """Inject an owner's optional source hooks without replacing observers."""
    for name in ('recording_boundaries', 'recording_positions'):
        value = getattr(owner, name, None)
        if value is not None:
            setattr(target, name, value)
    return target


def contact_established(source, protocol, local, peer, *, slot="link", generation=None):
    boundaries = getattr(source, 'recording_boundaries', None)
    if boundaries is None:
        return
    try:
        key = getattr(source, '_recording_key', None)
        if key is not None and generation is not None and generation < key[-1]:
            return
        if key is None or generation is not None and generation > key[-1]:
            generation = (getattr(source, '_recording_generation', 0) + 1
                          if generation is None else generation)
            source._recording_generation = generation
            endpoint = getattr(source, '_recording_endpoint', None) or _id()
            source._recording_endpoint = endpoint
            key = (boundaries.run, protocol, endpoint, local, slot, generation)
            source._recording_key = key
            positions = source.recording_positions()
        else:
            positions = {}
        source._recording_id = boundaries.established(key, peer, positions)
    except Exception as exc:
        boundaries.report(f'Recording boundary: {exc}')


def contact_ended(source, reason='disconnect'):
    key = getattr(source, '_recording_key', None)
    if key is None:
        return
    source._recording_key = None
    boundaries = source.recording_boundaries
    try:
        boundaries.ended(key, source.recording_positions(), reason)
    except Exception as exc:
        boundaries.report(f'Recording boundary: {exc}')


class ContactBoundaries:
    """Idempotent endpoint generations; callbacks isolate recording failures."""

    def __init__(self, session, *, run):
        self.session, self.run = session, run
        self._keys = {}
        self._lock = threading.Lock()

    def report(self, message):
        try:
            reporter = getattr(self.session, 'report_later', self.session.report)
            reporter(message)
        except Exception:
            pass

    def _call(self, method, *args, **kwargs):
        try:
            getattr(self.session, method)(*args, **kwargs)
        except Exception as exc:
            self.report(f'Recording boundary: {exc}')

    def established(self, key, peer, positions, *, provisional=False):
        with self._lock:
            if len(key) != 6 or key[0] != self.run:
                return
            slot, generation = tuple(key[:-1]), key[-1]
            old = self._keys.get(slot)
            if old is not None:
                gen, contact_id, active = old
                if generation < gen or generation == gen and not active:
                    return
                if generation == gen:
                    if peer:
                        self._call('enrich', contact_id, peer=peer)
                    return contact_id
                if active:
                    self._call('end', contact_id, dict(positions), reason='generation retired')
            contact_id = _id()
            self._keys[slot] = (generation, contact_id, True)
            for name, lane in getattr(self.session, 'lanes', {}).items():
                if lane.clock in positions:
                    self._call('state', name, positions[lane.clock],
                               clock_generation=getattr(lane, 'context_generation', 0),
                               protocol=key[1], calls=dict(local=key[3], peer=peer))
            self._call('begin', contact_id, dict(positions), provisional=provisional,
                       peer=peer, key=list(key))
            return contact_id

    def ended(self, key, positions, reason, *, keep=True, decided=None):
        with self._lock:
            old = self._keys.get(tuple(key[:-1]))
            if old is None or old[0] != key[-1] or not old[2]:
                return
            self._keys[tuple(key[:-1])] = (old[0], old[1], False)
            self._call('end', old[1], dict(positions), reason=reason, keep=keep, decided=decided)

    def close_owner(self, positions, reason):
        with self._lock:
            for slot, (generation, contact_id, active) in list(self._keys.items()):
                if active:
                    self._keys[slot] = generation, contact_id, False
                    self._call('end', contact_id, dict(positions), reason=reason)


def _remove_candidate(directory):
    for name in ('candidate.json', 'candidate.identity', 'candidate.lock'):
        if (directory / name).is_symlink():
            raise ValueError('Candidate identity/state is a symlink')
    facts = json.loads((directory / 'candidate.json').read_text())
    identity = json.loads((directory / 'candidate.lock').read_text())
    identity_path = directory / 'candidate.identity'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Candidate identity mismatch')
    if any(facts.get(k) != identity[k] for k in ('candidate_id', 'contact_id', 'directory')):
        raise ValueError('Candidate identity mismatch')
    if _absolute(identity['directory']) != _absolute(directory) or not identity['candidate_id']:
        raise ValueError('Candidate directory identity mismatch')
    if facts['state'] not in ('undecided', 'discard'):
        raise ValueError(f"Candidate state {facts['state']!r} cannot be discarded")
    paths = []
    for name in facts['owned_files']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Candidate owned path escapes directory')
        path = directory / relative
        if any(p.is_symlink() for p in [path, *path.parents] if p != directory.parent):
            raise ValueError('Candidate owned path is a symlink')
        if path.suffix not in ('.wav', '.json') or path.name == 'candidate.json':
            raise ValueError('Unknown candidate temporary file')
        side = path.with_suffix('.json')
        if side.exists() and json.loads(side.read_text()).get('capture_state') == 'finalized':
            raise ValueError('Finalized audio cannot be discarded')
        paths.append(path)
    for path in paths:
        path.unlink(missing_ok=True)
    identity_path.unlink(missing_ok=True)
    (directory / 'candidate.json').unlink()


def discard_candidate(directory):
    directory = _absolute(directory)
    if any(p.is_symlink() for p in [directory, *directory.parents]):
        raise ValueError('Candidate directory is a symlink')
    if not (directory / 'candidate.json').is_file():
        raise ValueError('Missing candidate identity/state')
    lock_path = directory / 'candidate.lock'
    if lock_path.is_symlink():
        raise ValueError('Candidate lock is a symlink')
    with lock_path.open('r+b') as lock:
        try:
            _lock_candidate(lock)
        except BlockingIOError:
            raise ValueError('Candidate has an active owner lock') from None
        _remove_candidate(directory)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('discard-candidate').add_argument('directory')
    args = parser.parse_args(argv)
    try:
        discard_candidate(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Cannot discard candidate: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
