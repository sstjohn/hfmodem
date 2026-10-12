"""Exclusive PACTOR evidence ownership, with no device or modem imports.

Identity describes filesystem provenance; it grants no radio admission. All
allocation and reservations belong outside audio callbacks and reply scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from types import MappingProxyType
import uuid

SCHEMA = 1
MANIFEST = 'capture-identity.json'
DIRECTORY_OWNER = '.capture-directory-owner'
KINDS = frozenset(('arm', 'launch-group', 'monitor-analysis', 'offline-replay'))
_SCAFFOLD = ('.capture-writers', '.capture-artifacts', '.capture-writer-status')


class IdentityError(ValueError):
    """Refuse unknown, conflicting, reused or changed evidence ownership."""


def _uuid(value):
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:
        raise IdentityError('Invalid capture identity') from exc
    if str(parsed) != str(value):
        raise IdentityError('Capture identity must be a canonical full UUID')
    return str(parsed)


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def relative(value):
    """Validate a reference before creating any producer or output root."""
    raw = os.fspath(value)
    path = Path(raw)
    if (not raw or path.is_absolute() or raw.endswith('/') or '\x00' in raw
            or any(x in ('', '.', '..') for x in raw.split('/'))):
        raise IdentityError('Artifact references must be confined relative names')
    if any(len(x.encode()) > 255 for x in path.parts):
        raise IdentityError('Artifact reference component is too long')
    return path


def _directory(path):
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise IdentityError('Registered root is not a real directory')


def _path(parent, name):
    path = parent / relative(name)
    # Inspect every existing component without following a symlink. Parent
    # roots were independently pinned; intentionally resolved worktree parents
    # are permitted, newly managed roots and their children are not symlinks.
    current = parent
    for component in path.relative_to(parent).parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise IdentityError('Artifact references cannot traverse symlinks')
        if stat.S_ISDIR(mode):
            # The exact owning root is the starting parent, never a traversed
            # component. Descendant roots have their own writer namespace,
            # including mirrors sharing an identity and partial allocations.
            markers = (MANIFEST, '.capture-writers', '.capture-artifacts')
            if any((current / marker).exists() or (current / marker).is_symlink()
                   for marker in markers):
                raise IdentityError('Artifact references cannot cross capture roots')
    return path


def _directory_context(capture, root_role, root=None):
    root = capture.root(root_role) if root is None else root
    return dict(capture_id=capture.capture_id, registry_sha256=capture.digest,
                root_role=root_role, root_path=str(root))


def _directory_record(root, path, context):
    return dict(directory_ownership_schema=1, **context,
                relative_directory=str(path.relative_to(root)))


# Ownership is pinned by registered path. Older records also carry inode
# fields, which any copy or restore changes; they are read and ignored.
_LEGACY_INODE_FIELDS = ('root_identity', 'directory_identity')


def _check_directory(root, path, context):
    """Observe complete exact-root authority; never adopt or repair a directory."""
    marker = path / DIRECTORY_OWNER
    try:
        mode = marker.lstat().st_mode
    except FileNotFoundError as exc:
        raise IdentityError('Directory has no complete ownership record') from exc
    if not stat.S_ISREG(mode):
        raise IdentityError('Directory ownership requires a regular record')
    try:
        row = json.loads(marker.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IdentityError('Malformed directory ownership record') from exc
    if not isinstance(row, dict):
        raise IdentityError('Malformed directory ownership record')
    for name in _LEGACY_INODE_FIELDS:
        row.pop(name, None)
    if row != _directory_record(root, path, context):
        raise IdentityError('Directory ownership differs from its registered root')


def _bind_directory(root, path, context):
    """Only an exclusive mkdir winner may bind its directory."""
    _directory(path)
    _publish(path / DIRECTORY_OWNER, _json(_directory_record(root, path, context)))
    _check_directory(root, path, context)


def _owned_directory(root, name, context, *, create=False, exclusive_leaf=False):
    """Componentwise creation, with no existing unbound-directory takeover."""
    parts = relative(name).parts
    current = root
    for index, component in enumerate(parts):
        if component == DIRECTORY_OWNER:
            raise IdentityError('Directory ownership record name is reserved')
        current /= component
        created = False
        if create:
            try:
                current.mkdir()
                created = True
            except FileExistsError:
                if exclusive_leaf and index == len(parts) - 1:
                    raise
        if created:
            _bind_directory(root, current, context)
        else:
            _directory(current)  # Reject symlinks and non-directories.
            _check_directory(root, current, context)
    return current


def _owned_path(capture, root_role, name, *, create_parents=False):
    name = relative(name)
    if DIRECTORY_OWNER in name.parts:
        raise IdentityError('Directory ownership record name is reserved')
    root = capture.root(root_role)
    context = _directory_context(capture, root_role, root)
    if name.parent != Path('.'):
        _owned_directory(root, name.parent, context, create=create_parents)
    return _path(root, name)


def _exclusive(path, raw):
    with path.open('xb') as sink:
        sink.write(raw)
        sink.flush()
        os.fsync(sink.fileno())


def _publish(path, raw, *, published=None):
    """Same-filesystem no-replace publication; failed temporary files remain."""
    temporary = path.with_name('.' + path.name + '.' + str(uuid.uuid4()) + '.partial')
    _exclusive(temporary, raw)
    os.link(temporary, path, follow_symlinks=False)
    if published is not None:
        published()  # Internal pure-memory notification, before cleanup can fail.
    temporary.unlink()


def effective_outdir(arguments, expected):
    """Check the final repeated option before witness or device startup."""
    selected = None
    for at, word in enumerate(arguments):
        if word == '--outdir':
            if at + 1 == len(arguments) or arguments[at + 1].startswith('--'):
                raise IdentityError('Missing output directory argument')
            selected = arguments[at + 1]
        elif word.startswith('--outdir='):
            selected = word.split('=', 1)[1]
    expected = Path(expected).absolute()
    if selected is not None and Path(selected).absolute() != expected:
        raise IdentityError('Effective output directory conflicts with registered modem root')
    return expected


def prepare_output(outdir=None, *, capture_id=None, root=None, role=None, offline=False,
                   default_parent='captures'):
    """Direct/delegated output claim, before a producer or device query."""
    if bool(capture_id) != bool(root) or (role is not None and not capture_id):
        raise IdentityError('Output delegation requires identity, root and writer role')
    if capture_id:
        capture = load(Path(root).expanduser())
        if capture.capture_id != capture_id or role != 'modem':
            raise IdentityError('Modem delegation differs from the registered identity')
        path = capture.root('capture')
        if outdir is not None and Path(outdir).expanduser().absolute() != path:
            raise IdentityError('Modem output differs from its delegated root')
    elif outdir is None:
        capture = allocate_default(default_parent, kind='offline-replay' if offline else 'arm')
        path = capture.root('capture')
    else:
        path = Path(outdir).expanduser().absolute()
        capture = allocate({'capture': path}, {'modem': 'capture'},
                           kind='offline-replay' if offline else 'arm')
    return capture, capture.claim('modem', capture_id=capture.capture_id, root=path)


@dataclass(frozen=True)
class Capture:
    """Immutable root registry, loaded identically by all delegated writers."""
    record: dict = field(repr=False)
    digest: str
    _end_requested: object = field(default_factory=threading.Event, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'record', _freeze(self.record))

    @property
    def capture_id(self):
        return self.record['capture_id']

    @property
    def kind(self):
        return self.record['kind']

    def root(self, role):
        try:
            row = self.record['roots'][role]
        except KeyError as exc:
            raise IdentityError('Unregistered root role') from exc
        path = Path(row['path'])
        if path.parent.resolve() != Path(row['resolved_parent']):
            raise IdentityError('Registered parent location changed')
        _directory(path)
        raw = (path / MANIFEST).read_bytes()
        if _hash(raw) != self.digest:
            raise IdentityError('Registered root manifests disagree')
        return path

    def ended(self):
        root = self.root(self.record['primary_root'])
        return (root / '.capture-ended.json').exists()

    def end(self):
        """End role/reservation admission; old reservations may still finish."""
        root = self.root(self.record['primary_root'])
        self._end_requested.set()
        existing = root / '.capture-ended.json'
        if existing.exists():
            if json.loads(existing.read_text()).get('capture_id') != self.capture_id:
                raise IdentityError('Foreign capture end marker')
            return
        _publish(root / '.capture-ended.json', _json({
            'capture_id': self.capture_id, 'arm_ended': True,
            'ended_utc': datetime.now(timezone.utc).isoformat(),
            'ended_monotonic_ns': time.monotonic_ns(), 'clock_host': os.uname().nodename}))

    def lifecycle(self):
        """Read bound writer journals; ending a session never certifies audio."""
        rows, unclaimed = {}, []
        for role, binding in self.record['writers'].items():
            root = self.root(binding['root_role'])
            claim = root / '.capture-writers' / (role + '.json')
            if not claim.exists():
                unclaimed.append(role)
                continue
            data = json.loads(claim.read_text())
            if data.get('capture_id') != self.capture_id or data.get('registry_sha256') != self.digest or data.get('writer_id') != binding['writer_id']:
                raise IdentityError('Writer lifecycle claim differs from registry')
            status = root / '.capture-writer-status' / (role + '.json')
            if not status.exists():
                rows[role] = {'state': 'unknown'}
                continue
            row = metadata_record(status)
            if row['writer_id'] != binding['writer_id'] or row['writer_role'] != role:
                raise IdentityError('Writer status belongs to another role')
            rows[role] = {k: row[k] for k in ('state', 'pending', 'written', 'lost', 'partial', 'components')}
        artifact_states = {}
        for root_role in self.record['roots']:
            root = self.root(root_role)
            for reservation in (root / '.capture-artifacts').glob('*.json'):
                item = json.loads(reservation.read_text())
                if item.get('capture_id') != self.capture_id or item.get('registry_sha256') != self.digest:
                    raise IdentityError('Artifact lifecycle reservation differs from registry')
                path = _path(root, item['artifact_path'])
                state = 'missing'
                if path.suffix in ('.wav', '.flac'):
                    candidates = [path] if path.exists() else [path.with_suffix('.wav' if path.suffix == '.flac' else '.flac')]
                    for candidate in candidates:
                        if not candidate.exists():
                            continue
                        validate_artifact(candidate)
                        side = json.loads(candidate.with_suffix('.json').read_text())
                        # A counterpart can satisfy only this exact reserved audio pair.
                        for key in ('capture_id', 'registry_sha256', 'writer_id',
                                    'writer_role', 'clock_id'):
                            if side.get(key) != item.get(key):
                                raise IdentityError('audio counterpart ownership mismatch')
                        paired_id = ('artifact_id' if path.suffix == '.wav'
                                     else 'container_artifact_id')
                        # Before conversion, the unused FLAC slot has no container
                        # hash/id in the WAV journal; writer, clock and stem bind it.
                        if (paired_id in side or path.suffix == '.wav') and (
                                side.get(paired_id) != item.get('artifact_id')):
                            raise IdentityError('audio counterpart reservation mismatch')
                        if side.get('capture_state') == 'finalized':
                            state = 'finalized' if candidate == path else 'alternative-not-used'
                        else:
                            state = 'partial'
                elif path.exists():
                    artifact_binding(path)
                    state = 'present'
                artifact_states[item['artifact_id']] = state
        artifacts_ready = all(state in ('finalized', 'alternative-not-used', 'present') for state in artifact_states.values())
        ended = self.ended()
        ready = bool(rows) and all(row.get('state') == 'finalized' and not any(row.get(k, 0) for k in ('pending', 'lost', 'partial')) for row in rows.values())
        return dict(capture_id=self.capture_id, registry_sha256=self.digest,
                    arm_ended=ended, writers=rows, unclaimed_writers=unclaimed,
                    claimed_writers_complete=ready,
                    artifact_states=artifact_states, artifacts_complete=artifacts_ready,
                    capture_complete=ended and ready and artifacts_ready and not unclaimed)

    def claim(self, role, *, capture_id=None, root=None):
        if capture_id is not None and capture_id != self.capture_id:
            raise IdentityError('Delegated capture identity mismatch')
        try:
            binding = self.record['writers'][role]
        except KeyError as exc:
            raise IdentityError('Unregistered writer role') from exc
        registered = self.root(binding['root_role'])
        if root is not None and Path(root).absolute() != registered:
            raise IdentityError('Delegated writer root mismatch')
        if self.ended():
            raise IdentityError('Completed capture cannot acquire a writer')
        claim = registered / '.capture-writers' / (role + '.json')
        _owned_directory(registered, '.capture-writers',
                         _directory_context(self, binding['root_role'], registered))
        _publish(claim, _json({'schema': SCHEMA, 'capture_id': self.capture_id,
                              'writer_id': binding['writer_id'], 'role': role,
                              'registry_sha256': self.digest}))
        writer = Writer(self, role, binding["writer_id"], registered, _hash(claim.read_bytes()))
        writer._initialize_status()
        return writer


def allocate(roots, writers, *, kind='arm', capture_id=None, now=None,
             primary_root=None, parent_capture_id=None, parent_kind=None,
             transcript=None, source=None):
    """Exclusively claim exact roots. Partial allocation is never cleaned up."""
    if kind not in KINDS or not roots or not writers:
        raise IdentityError('Invalid capture registry')
    identity = _uuid(capture_id or uuid.uuid4())
    for name in (*roots, *writers):
        if relative(name).parts != (name,):
            raise IdentityError('Roles must be simple names')
    if any(role not in roots for role in writers.values()):
        raise IdentityError('Writer references an unregistered root')
    if parent_capture_id is not None:
        parent_capture_id = _uuid(parent_capture_id)
        if parent_kind not in KINDS:
            raise IdentityError('Missing parent kind')
    if transcript is not None:
        if transcript['root_role'] not in roots:
            raise IdentityError('Transcript references an unregistered root')
        relative(transcript['path'])
    paths = {role: Path(path).absolute() for role, path in roots.items()}
    if len(set(paths.values())) != len(paths):
        raise IdentityError('Root roles require distinct registered paths')
    primary_root = primary_root or next(iter(paths))
    if primary_root not in paths:
        raise IdentityError('Unregistered primary root')
    # Do not follow a preexisting final-root symlink, including dangling ones.
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError('Capture output already exists; choose a new path')
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise IdentityError('Allocation time requires an explicit timezone')
    registered = {}
    for role, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent.resolve()
        try:
            path.mkdir()  # mkdir is the exclusive allocation operation.
        except FileExistsError as exc:
            if registered:
                raise IdentityError('Partial root allocation preserved; choose new paths') from exc
            raise
        # Known shared scaffolding is bound before any role may claim. In
        # particular concurrent role initialization must not race to create
        # the common status parent after publishing its irreversible claim.
        for name in _SCAFFOLD:
            (path / name).mkdir()
        registered[role] = {'path': str(path), 'resolved_parent': str(parent)}
    record = {'schema': SCHEMA, 'kind': kind, 'capture_id': identity,
              'parent_capture_id': parent_capture_id, 'parent_kind': parent_kind,
              'allocated_utc': stamp.astimezone(timezone.utc).isoformat(),
              'legacy_minute': stamp.astimezone().strftime('onair-%m%d-%H%M'),
              'timezone': str(stamp.astimezone().tzinfo),
              'primary_root': primary_root, 'roots': registered,
              'writers': {role: {'root_role': root,
                                 'writer_id': str(uuid.uuid5(uuid.UUID(identity), role))}
                          for role, root in writers.items()},
              'transcript': transcript, 'source': source or {}}
    raw = _json(record)
    capture = Capture(record, _hash(raw))
    for role, path in paths.items():
        context = _directory_context(capture, role, path)
        for name in _SCAFFOLD:
            _bind_directory(path, path / name, context)
    for path in paths.values():
        _publish(path / MANIFEST, raw)
    return capture


def allocate_default(parent, *, kind='arm', prefix='onair', now=None, uuid_factory=uuid.uuid4,
                     writers=None):
    """Bounded full-UUID allocation; only collisions are retried."""
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise IdentityError('Allocation time requires an explicit timezone')
    for _ in range(32):
        identity = _uuid(uuid_factory())
        name = f'{prefix}-{stamp.astimezone(timezone.utc):%Y%m%dT%H%M%S%fZ}-{identity}'
        try:
            return allocate({'capture': Path(parent) / name}, writers or {'modem': 'capture'},
                            kind=kind, capture_id=identity, now=stamp)
        except FileExistsError:
            continue
    raise FileExistsError('Capture UUID allocation exhausted 32 attempts')


def allocate_arm(repo, *, transcripts=None, log_label=None, native_root=None, kind='arm',
                 now=None, parent_capture_id=None, parent_kind=None, arguments=(), source=None,
                 uuid_factory=uuid.uuid4):
    """Plan and claim all PACTOR arm roots before any producer/device context.

    Console names remain readable; physical directories carry the same full
    identity as the launcher and transcript registry. No old minute alias exists.
    """
    repo = Path(repo).absolute()
    transcripts = Path(transcripts or repo / 'transcripts').absolute()
    label = relative(log_label or 'arm')
    # Validate the eventual leaf before any root is created, without truncation.
    relative(str(label.with_name(label.name + '--' + '0' * 36 + '.log')))
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise IdentityError('Allocation time requires an explicit timezone')
    for _ in range(32):
        identity = _uuid(uuid_factory())
        basename = f'onair-{stamp.astimezone(timezone.utc):%Y%m%dT%H%M%S%fZ}-{identity}'
        roots = {'capture': repo / 'captures' / basename,
                 'transcript': transcripts / basename}
        writers = {'launcher-transcript': 'transcript', 'launcher-metadata': 'transcript'}
        if kind == 'arm':
            writers.update({'modem': 'capture', 'launcher-witness': 'capture'})
        if native_root is not None:
            requested = Path(native_root).absolute()
            physical = requested.with_name(requested.name + '--' + identity)
            relative(physical.name)
            roots['native'] = physical
            writers.update({'native-transcript': 'native', 'native-metadata': 'native',
                            'tap': 'native', 'console-witness': 'native'})
        effective_outdir(arguments, roots['capture'])
        transcript = str(label.with_name(label.name + '--' + identity + '.log'))
        try:
            result = allocate(roots, writers, kind=kind, capture_id=identity, now=stamp,
                              parent_capture_id=parent_capture_id, parent_kind=parent_kind,
                              primary_root='capture',
                              transcript={'root_role': 'transcript', 'path': transcript},
                              source=source)
        except FileExistsError:
            continue
        return result
    raise FileExistsError('PACTOR arm allocation exhausted 32 attempts')


def load(root):
    """Validate registered mirrors; legacy explicit paths remain unbound."""
    root = Path(root).absolute()
    _directory(root)
    raw = (root / MANIFEST).read_bytes()
    try:
        record = json.loads(raw)
        if record['schema'] != SCHEMA or record['kind'] not in KINDS:
            raise IdentityError('Unsupported capture identity schema')
        _uuid(record['capture_id'])
        if not any(Path(row['path']) == root for row in record['roots'].values()):
            raise IdentityError('Extra unregistered root claims a capture identity')
        capture = Capture(record, _hash(raw))
        for role in record['roots']:
            if relative(role).parts != (role,):
                raise IdentityError('Invalid registered root role')
            capture.root(role)
        for role, row in record['writers'].items():
            if relative(role).parts != (role,) or row['root_role'] not in record['roots']:
                raise IdentityError('Invalid writer binding')
            expected = str(uuid.uuid5(uuid.UUID(capture.capture_id), role))
            if row['writer_id'] != expected:
                raise IdentityError('Conflicting writer binding')
        for root_role in record['roots']:
            path = capture.root(root_role)
            for claim_path in (path / '.capture-writers').glob('*.json'):
                if claim_path.is_symlink():
                    raise IdentityError('Writer claim cannot be a symlink')
                claimed = json.loads(claim_path.read_text())
                role = claimed['role']
                row = record['writers'].get(role)
                if (row is None or claim_path.name != role + '.json'
                        or row['root_role'] != root_role
                        or claimed['capture_id'] != capture.capture_id
                        or claimed['writer_id'] != row['writer_id']
                        or claimed['registry_sha256'] != capture.digest):
                    raise IdentityError('Duplicate or conflicting writer claim')
        return capture
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise IdentityError('Malformed capture registry') from exc


NEW_NAME = re.compile(r'^onair-\d{8}T\d{12}Z-[0-9a-f-]{36}$')


def provenance(root):
    """Closed reader status; a new name cannot fall back to legacy inference."""
    root = Path(root)
    try:
        capture = load(root)
    except FileNotFoundError:
        return {'status': 'incomplete' if _requires_manifest(root) else 'legacy-unbound'}
    except (IdentityError, OSError, ValueError):
        return {'status': 'ambiguous'}
    return {'status': 'bound', 'capture_id': capture.capture_id, 'kind': capture.kind, 'allocated_utc': capture.record['allocated_utc']}


def validate_artifact(path, *, clock_id=None):
    """Validate new sidecar ownership against its original reservation.

    Exact legacy paths remain usable and unbound. No inferred association is
    returned when an identity is absent, conflicted, or incomplete.
    """
    path = Path(path).absolute()
    root = next((p for p in (path.parent, *path.parents) if (p / MANIFEST).exists()), None)
    if root is None:
        if any((_requires_manifest(p) for p in path.parents)):
            raise IdentityError('New capture has no mandatory identity')
        side_path = path.with_suffix('.json')
        if side_path.exists():
            side = json.loads(side_path.read_text())
            if any((name in side for name in ('capture_id', 'artifact_id', 'capture_schema'))):
                raise IdentityError('Artifact identity has no root registry')
        return {'status': 'legacy-unbound'}
    artifact_binding(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise IdentityError('Recording requires a regular declared artifact')
    capture = load(root)
    relative_path = str(path.relative_to(root))
    key = hashlib.sha256(relative_path.encode()).hexdigest()
    reservation = json.loads((root / '.capture-artifacts' / (key + '.json')).read_text())
    side = json.loads(path.with_suffix('.json').read_text())
    for name in ('capture_id', 'writer_id', 'writer_role', 'clock_id', 'registry_sha256'):
        if side.get(name) != reservation.get(name):
            raise IdentityError('Sidecar does not belong to declared artifact')
    artifact_id = side.get('container_artifact_id') if path.suffix == '.flac' else side.get('artifact_id')
    if artifact_id != reservation.get('artifact_id'):
        raise IdentityError('Sidecar artifact identity differs from recording')
    if reservation['capture_id'] != capture.capture_id or reservation['artifact_path'] != relative_path:
        raise IdentityError('Artifact reservation does not belong to root')
    binding = capture.record['writers'].get(reservation['writer_role'])
    if binding is None or binding['writer_id'] != reservation['writer_id'] or capture.root(binding['root_role']) != root:
        raise IdentityError('Artifact writer does not belong to root')
    if clock_id is not None and reservation['clock_id'] != clock_id:
        raise IdentityError('Artifact clock differs from selected clock')
    recorded_hash = side.get('container_sha256') if path.suffix == '.flac' else side.get('sha256')
    if recorded_hash is not None:
        with path.open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != recorded_hash:
                raise IdentityError('Recording bytes differ from finalized sidecar')
    return {'status': 'bound', 'capture_id': capture.capture_id, 'artifact_id': reservation['artifact_id'], 'clock_id': reservation['clock_id']}


@dataclass(frozen=True)
class Writer:
    capture: Capture = field(repr=False)
    role: str
    writer_id: str
    root: Path
    claim_sha256: str
    _tickets: dict = field(default_factory=dict, repr=False, compare=False)
    _ticket_lock: object = field(default_factory=threading.Lock, repr=False, compare=False)
    _status: dict = field(default_factory=dict, repr=False, compare=False)

    def _initialize_status(self):
        slot = self.reserve('.capture-writer-status/' + self.role + '.json').versions()
        self._status['slot'] = slot
        self._status['components'] = {}
        self._status['lock'] = threading.RLock()
        slot.update(dict(state='ongoing', pending=0, written=0, lost=0, partial=0, components={}))

    def update_status(self, *, state, pending=0, written=0, lost=0, partial=0, component='producer'):
        """Worker/close journal; independent producer components share a role.

        A complete continuous tap cannot erase unfinished deferred windows (or
        the reverse). This method performs I/O and is never a callback/queue API.
        """
        if state not in ('ongoing', 'finalized', 'partial'):
            raise IdentityError('Unknown writer lifecycle state')
        if not isinstance(component, str) or not component or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in component):
            raise IdentityError('Invalid lifecycle component')
        counts = dict(pending=pending, written=written, lost=lost, partial=partial)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values()):
            raise IdentityError('Writer counts must be nonnegative integers')
        if state == 'finalized' and any(counts[k] for k in ('pending', 'lost', 'partial')):
            raise IdentityError('Incomplete writer cannot be finalized')
        with self._status['lock']:
            self._status['components'][component] = dict(state=state, **counts)
            components = dict(self._status['components'])
            totals = {k: sum(row[k] for row in components.values()) for k in counts}
            combined = 'partial' if any(row['state'] == 'partial' for row in components.values()) else 'finalized' if all(row['state'] == 'finalized' for row in components.values()) else 'ongoing'
            if self._status.get('failed'):
                combined, totals['partial'] = 'partial', max(totals['partial'], 1)
            return self._status['slot'].update(dict(state=combined, **totals, components=components))

    def check(self):
        binding = self.capture.record['writers'][self.role]
        root = self.capture.root(binding['root_role'])
        _owned_directory(root, '.capture-writers',
                         _directory_context(self.capture, binding['root_role'], root))
        claim = self.root / '.capture-writers' / (self.role + '.json')
        if _hash(claim.read_bytes()) != self.claim_sha256:
            raise IdentityError('Writer claim changed')

    def reserve(self, name, *, clock_id=None):
        return self._reserve(name, clock_id=clock_id)

    def ticket(self, name, *, clock_id=None):
        """Bind a queued name in memory; no filesystem I/O on scheduling paths.

        The persistent sole-writer role already owns the artifact family. The
        recording worker performs the exclusive file reservation/publication;
        this one-use fixed-name ticket admits only work queued before local end.
        """
        name = str(relative(name))
        if DIRECTORY_OWNER in Path(name).parts:
            raise IdentityError('Directory ownership record name is reserved')
        with self._ticket_lock:
            if self.capture._end_requested.is_set():
                raise IdentityError('Ended capture cannot queue new artifacts')
            if name in self._tickets:
                raise IdentityError('Artifact name already queued')
            token = object()
            self._tickets[name] = token
        return Ticket(self, name, clock_id, token, time.monotonic_ns(), os.uname().nodename)

    def _reserve(self, name, *, clock_id=None, queued=False):
        self.check()
        if not queued and self.capture.ended():
            raise IdentityError('Ended capture cannot reserve new artifacts')
        root_role = self.capture.record['writers'][self.role]['root_role']
        path = _owned_path(self.capture, root_role, name, create_parents=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError('Artifact already exists')
        # Stable artifact path identity makes duplicate reservation exclusive.
        identity = str(uuid.uuid5(uuid.UUID(self.writer_id), str(relative(name))))
        # Different roles cannot reserve the same destination either.
        key = hashlib.sha256(str(relative(name)).encode()).hexdigest()
        record = {'capture_schema': SCHEMA, 'capture_id': self.capture.capture_id,
                  'artifact_id': identity, 'writer_id': self.writer_id,
                  'writer_role': self.role, 'artifact_path': str(relative(name)),
                  'clock_id': clock_id, 'registry_sha256': self.capture.digest}
        reservation = self.root / '.capture-artifacts' / (key + '.json')
        _owned_directory(self.root, '.capture-artifacts',
                         _directory_context(self.capture, root_role, self.root))
        _publish(reservation, _json(record))
        return Artifact(self, path, MappingProxyType(record), reservation,
                        _hash(reservation.read_bytes()))


def retained_writer(capture, role, name, descriptor):
    """Continue an already claimed owner through its actual retained artifact FD.

    This never claims/reclaims a role. Producers still check arm-ending before
    starting; completion helpers can finish an originally reserved artifact later.
    """
    import fcntl
    binding = capture.record['writers'].get(role)
    if binding is None:
        raise IdentityError('unregistered retained writer')
    root = capture.root(binding['root_role'])
    path = _owned_path(capture, binding['root_role'], name)
    actual, expected = os.fstat(int(descriptor)), path.lstat()
    writable = fcntl.fcntl(int(descriptor), fcntl.F_GETFL) & os.O_ACCMODE
    if (writable == os.O_RDONLY or not stat.S_ISREG(actual.st_mode)
            or not stat.S_ISREG(expected.st_mode)
            or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)):
        raise IdentityError('retained artifact descriptor mismatch')
    claim = root / '.capture-writers' / (role + '.json')
    raw = claim.read_bytes()
    row = json.loads(raw)
    if (row.get('capture_id') != capture.capture_id
            or row.get('writer_id') != binding['writer_id']
            or row.get('role') != role
            or row.get('registry_sha256') != capture.digest):
        raise IdentityError('retained writer claim mismatch')
    owned = artifact_binding(path)
    if owned['writer_role'] != role or owned['writer_id'] != binding['writer_id']:
        raise IdentityError('retained artifact owner mismatch')
    writer = Writer(capture, role, binding['writer_id'], root, _hash(raw))
    writer.check()
    status_path = _owned_path(capture, binding['root_role'],
                              '.capture-writer-status/' + role + '.json')
    status = metadata_record(status_path)
    status_raw = status_path.read_bytes()
    version = status.get('metadata_version')
    if (not isinstance(version, int) or isinstance(version, bool) or version < 0
            or json.loads(status_raw) != status):
        raise IdentityError('retained status version mismatch')
    journal = status_path.parent / ('.' + status_path.name + '.versions')
    latest = _owned_path(capture, binding['root_role'],
                         str((journal / f'{version:08d}.json').relative_to(root)))
    try:
        if latest.read_bytes() != status_raw:
            raise IdentityError('retained status journal mismatch')
        if version == 0:
            if status.get('previous_sha256') is not None:
                raise IdentityError('retained initial status chain mismatch')
        else:
            previous = _owned_path(capture, binding['root_role'],
                                   str((journal / f'{version - 1:08d}.json').relative_to(root)))
            if _hash(previous.read_bytes()) != status.get('previous_sha256'):
                raise IdentityError('retained status chain mismatch')
    except FileNotFoundError as error:
        raise IdentityError('retained status journal missing') from error
    key = hashlib.sha256(str(status_path.relative_to(root)).encode()).hexdigest()
    reservation = root / '.capture-artifacts' / (key + '.json')
    reserved_raw = reservation.read_bytes()
    reserved = json.loads(reserved_raw)
    expected_id = str(uuid.uuid5(uuid.UUID(writer.writer_id), str(status_path.relative_to(root))))
    if (reserved.get('artifact_id') != expected_id
            or status.get('writer_id') != writer.writer_id
            or status.get('writer_role') != role):
        raise IdentityError('retained writer status mismatch')
    artifact = Artifact(writer, status_path, MappingProxyType(reserved), reservation,
                        _hash(reserved_raw))
    slot = MetadataSlot(artifact, subject=artifact)
    slot.version = version + 1
    slot.previous = _hash(status_raw)
    writer._status.update(slot=slot, components=dict(status['components']),
                          lock=threading.RLock())
    if status['state'] == 'partial':
        writer._status['failed'] = True
    return writer


@dataclass(frozen=True)
class Ticket:
    writer: Writer = field(repr=False)
    name: str
    clock_id: str | None
    _token: object = field(repr=False)
    _queued_ns: int = field(repr=False)
    _clock_host: str = field(repr=False)

    def reserve(self):
        with self.writer._ticket_lock:
            if self.writer._tickets.get(self.name) is not self._token:
                raise IdentityError('Queued reservation was already consumed or differs')
            # Retain the consumed name as a tombstone: no later job may reuse it.
            self.writer._tickets[self.name] = None
        if self.writer.capture.ended():
            root = self.writer.capture.root(self.writer.capture.record['primary_root'])
            ended = json.loads((root / '.capture-ended.json').read_text())
            if (ended.get('capture_id') != self.writer.capture.capture_id
                    or ended.get('clock_host') != self._clock_host
                    or self._queued_ns >= ended.get('ended_monotonic_ns', 0)):
                raise IdentityError('Queued artifact was not bound before arm end')
        return self.writer._reserve(self.name, clock_id=self.clock_id, queued=True)


@dataclass(frozen=True)
class Artifact:
    writer: Writer = field(repr=False)
    path: Path
    record: object = field(repr=False)
    reservation: Path = field(repr=False)
    reservation_sha256: str

    def check(self):
        self.writer.check()
        root_role = self.writer.capture.record['writers'][self.writer.role]['root_role']
        if _owned_path(self.writer.capture, root_role, self.record['artifact_path']) != self.path:
            raise IdentityError('Artifact destination changed')
        _owned_path(self.writer.capture, root_role,
                    self.reservation.relative_to(self.writer.root))
        if _hash(self.reservation.read_bytes()) != self.reservation_sha256:
            raise IdentityError('Artifact reservation changed')

    def open(self):
        """Exclusive continuous file; callers retain this one handle."""
        self.check()
        return self.path.open('xb')

    def publish(self, raw, *, published=None):
        self.check()
        if published is None:
            _publish(self.path, raw)
        else:
            _publish(self.path, raw, published=published)

    def temporary(self):
        """Retained same-filesystem temporary; never an old recording path."""
        self.check()
        path = self.path.with_name('.' + self.path.name + '.' + str(uuid.uuid4()) + '.partial')
        return path, path.open('xb')

    def publish_file(self, temporary, *, published=None):
        self.check()
        temporary = Path(temporary)
        if (temporary.parent != self.path.parent
                or not stat.S_ISREG(temporary.lstat().st_mode)):
            raise IdentityError('Publication requires a regular same-root temporary')
        os.link(temporary, self.path, follow_symlinks=False)
        if published is not None:
            published()  # No journaling or filesystem work in this notification.
        temporary.unlink()

    def metadata(self, facts):
        """Bind measurements additively; identities cannot be overwritten."""
        forbidden = {name for name in set(facts) & set(self.record)
                     if facts[name] != self.record[name]}
        if forbidden:
            raise IdentityError('Measurement fields conflict with artifact identity')
        return dict(facts) | dict(self.record)

    def versions(self, name=None):
        """Reserve a companion snapshot before opening a continuous producer."""
        slot = self if name is None else self.writer.reserve(name, clock_id=self.record['clock_id'])
        return MetadataSlot(slot, subject=self)


class MetadataSlot:
    """Immutable versions and one sole-writer compatibility JSON snapshot."""
    def __init__(self, artifact, *, subject=None):
        self.artifact = artifact
        self.subject = subject or artifact
        self.version = 0
        self.previous = None
        self.lock = threading.Lock()

    def update(self, facts):
        with self.lock:
            self.artifact.check()
            path = self.artifact.path
            self.subject.check()
            row = self.subject.metadata(facts)
            row.update(metadata_version=self.version, previous_sha256=self.previous)
            raw = _json(row)
            journal = path.parent / ('.' + path.name + '.versions')
            writer = self.artifact.writer
            root_role = writer.capture.record['writers'][writer.role]['root_role']
            context = _directory_context(writer.capture, root_role, writer.root)
            if self.version == 0:
                _owned_directory(writer.root, journal.relative_to(writer.root), context,
                                 create=True, exclusive_leaf=True)
            else:
                _owned_directory(writer.root, journal.relative_to(writer.root), context)
                current = path.read_bytes()
                if _hash(current) != self.previous:
                    raise IdentityError('Metadata slot changed or is foreign')
            _publish(journal / f'{self.version:08d}.json', raw)
            if self.version == 0:
                _publish(path, raw)
            else:
                temporary = path.with_name('.' + path.name + '.' + str(uuid.uuid4()) + '.partial')
                _exclusive(temporary, raw)
                # This is the sole narrow same-writer replacement permission.
                if _hash(path.read_bytes()) != self.previous:
                    raise IdentityError('Metadata slot changed before publication')
                os.replace(temporary, path)
            self.previous = _hash(raw)
            self.version += 1
            return row


@dataclass(frozen=True)
class Recording:
    """All recording outputs reserved before a producer is opened."""
    audio: Artifact
    container: Artifact
    sidecar: MetadataSlot = field(repr=False)

    @classmethod
    def reserve(cls, writer, name, *, clock_id):
        name = relative(name)
        if name.suffix != '.wav':
            raise IdentityError('Recording reservation requires a WAV name')
        audio = writer.reserve(name, clock_id=clock_id)
        container = writer.reserve(name.with_suffix('.flac'), clock_id=clock_id)
        return cls(audio, container, audio.versions(name.with_suffix('.json')))

    @classmethod
    def queued(cls, writer, name, *, clock_id):
        name = relative(name)
        if name.suffix != '.wav':
            raise IdentityError('Queued recording requires a WAV name')
        return QueuedRecording(writer.ticket(name, clock_id=clock_id),
                               writer.ticket(name.with_suffix('.flac'), clock_id=clock_id),
                               writer.ticket(name.with_suffix('.json'), clock_id=clock_id))


@dataclass(frozen=True)
class QueuedRecording:
    """A fixed destination for each old queued job, independent of later runs."""
    audio: Ticket
    container: Ticket
    sidecar: Ticket

    def reserve(self):
        audio, container, sidecar = self.audio.reserve(), self.container.reserve(), self.sidecar.reserve()
        return Recording(audio, container, MetadataSlot(sidecar, subject=audio))


def select_roots(candidates, *, preferred_role=None):
    """Resolve registered mirrors; distinct legacy or conflicting roots refuse.

    A single legacy root retains its previous unbound meaning. New names require
    a registry. Multiple roots are permitted only when every root is a member of
    the same exact frozen registry, never merely because a UUID happens to match.
    """
    paths = list(dict.fromkeys(Path(p).absolute() for p in candidates))
    if not paths:
        raise IdentityError('No capture root candidates')
    states = [provenance(p) for p in paths]
    if len(paths) == 1 and states[0]['status'] == 'legacy-unbound':
        return paths[0]
    if any(row['status'] != 'bound' for row in states):
        raise IdentityError('Capture root selection is incomplete or ambiguous')
    captures = [load(path) for path in paths]
    first = captures[0]
    if any(c.capture_id != first.capture_id or c.digest != first.digest for c in captures[1:]):
        raise IdentityError('Capture roots have conflicting registries')
    registry = {role: first.root(role) for role in first.record['roots']}
    if any(p not in registry.values() for p in paths):
        raise IdentityError('Capture identity appears at an unregistered root')
    if len(paths) == 1:
        return paths[0]
    role = preferred_role or first.record['primary_root']
    if role not in registry or registry[role] not in paths:
        raise IdentityError('Multiple capture roots require a present selected role')
    return registry[role]


def allocate_analysis(source_root, *, now=None, uuid_factory=uuid.uuid4):
    """New attached monitor child; never reuse the old arm/monitor output."""
    source_root = Path(source_root).absolute()
    _directory(source_root)
    state = provenance(source_root)
    if state['status'] not in ('bound', 'legacy-unbound'):
        raise IdentityError('Attached monitor source is incomplete or ambiguous')
    source = {'root': str(source_root), 'status': state['status'],
              'capture_id': state.get('capture_id'), 'registry_sha256': None}
    if state['status'] == 'bound':
        source['registry_sha256'] = load(source_root).digest
    parent = source_root / 'monitor-runs'
    parent.mkdir(exist_ok=True)
    _directory(parent)
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise IdentityError('Monitor time requires an explicit timezone')
    base = parent / f'monitor-{stamp.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}'
    for n in range(1, 100):
        root = base if n == 1 else base.with_name(f'{base.name}-{n}')
        try:
            capture = allocate({'analysis': root}, {'analysis': 'analysis'},
                kind='monitor-analysis', capture_id=_uuid(uuid_factory()), now=stamp,
                primary_root='analysis', source=source)
        except FileExistsError:
            continue
        return capture
    raise FileExistsError('Attached monitor allocation exhausted 99 attempts')


def _requires_manifest(root):
    root = Path(root)
    return bool(NEW_NAME.fullmatch(root.name) or
        re.search(r'--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', root.name) or
        (root.parent.name == 'monitor-runs' and root.name.startswith('monitor-')) or
        (root / '.capture-writers').exists() or (root / '.capture-artifacts').exists())


def artifact_binding(path):
    """Metadata binding for any declared artifact, including text transcripts.

    A shared capture identity is not an audio clock conversion. Callers receive
    the artifact's own clock identifier and must retain existing explicit clock
    alignment rules. Audio readers additionally use validate_artifact.
    """
    path = Path(path).absolute()
    root = next((p for p in path.parents if (p / MANIFEST).exists()), None)
    if root is None:
        if any(_requires_manifest(p) for p in path.parents):
            raise IdentityError('Declared artifact has no mandatory registry')
        return {'status': 'legacy-unbound'}
    capture = load(root)
    name = str(relative(path.relative_to(root)))
    key = hashlib.sha256(name.encode()).hexdigest()
    row = json.loads((root / '.capture-artifacts' / (key + '.json')).read_text())
    binding = capture.record['writers'].get(row.get('writer_role'))
    if (row.get('capture_id') != capture.capture_id or row.get('artifact_path') != name or
        row.get('registry_sha256') != capture.digest or binding is None or
        binding['writer_id'] != row.get('writer_id') or capture.root(binding['root_role']) != root):
        raise IdentityError('Artifact does not belong to its registered writer')
    if path.is_symlink():
        raise IdentityError('Declared artifact cannot be a symlink')
    return {'status': 'bound', 'capture_id': capture.capture_id, 'kind': capture.kind,
            'artifact_id': row['artifact_id'], 'writer_id': row['writer_id'],
            'writer_role': row['writer_role'], 'clock_id': row['clock_id'],
            'registry_sha256': capture.digest, 'root_role': binding['root_role'],
            'artifact_path': name}


def evidence_reference(path):
    """Bind a reader's exact file/root without using a name, mtime or alias.

    Capture allocation time is provenance, not an audio clock conversion.
    Legacy paths remain explicitly unbound. New malformed/missing identity
    never reaches minute-name fallback.
    """
    path = Path(path).absolute()
    candidates = (path, *path.parents) if path.is_dir() else path.parents
    root = next((p for p in candidates if (p / MANIFEST).exists()), None)
    if root is None:
        if any(_requires_manifest(p) for p in candidates):
            raise IdentityError('New evidence has no mandatory identity')
        return {'status': 'legacy-unbound'}
    artifact = {} if path == root else artifact_binding(path)
    capture = load(root)
    stamp = datetime.fromisoformat(capture.record['allocated_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise IdentityError('Capture allocation time has no timezone')
    result = dict(artifact)
    result.update(status='bound', capture_id=capture.capture_id, kind=capture.kind,
                  allocated_utc=stamp.astimezone(timezone.utc).isoformat(),
                  registry_sha256=capture.digest,
                  primary_root=str(capture.root(capture.record['primary_root'])))
    ended_path = capture.root(capture.record['primary_root']) / '.capture-ended.json'
    if ended_path.exists():
        ended = json.loads(ended_path.read_text())
        if ended.get('capture_id') != capture.capture_id or ended.get('arm_ended') is not True:
            raise IdentityError('Foreign capture end provenance')
        ended_stamp = datetime.fromisoformat(ended['ended_utc'].replace('Z', '+00:00'))
        if ended_stamp.tzinfo is None:
            raise IdentityError('Capture end time has no timezone')
        result['arm_ended_utc'] = ended_stamp.astimezone(timezone.utc).isoformat()
    declared = capture.record.get('transcript')
    if declared:
        result['transcript'] = str(capture.root(declared['root_role']) / declared['path'])
    return result


def metadata_record(path):
    """Read compatible metadata, validating a new record's declared artifact."""
    path = Path(path)
    binding = artifact_binding(path)
    row = json.loads(path.read_text())
    if binding['status'] == 'bound':
        for key in ('capture_id', 'artifact_id', 'writer_id', 'writer_role',
                    'clock_id', 'registry_sha256'):
            if row.get(key) != binding[key]:
                raise IdentityError('Metadata does not belong to its reserved artifact')
    elif any(key in row for key in ('capture_id', 'artifact_id', 'capture_schema')):
        raise IdentityError('Metadata identity has no registry')
    return row


def recording_windows(paths, *, capture_id, clock_id=None):
    """Select new arm windows by identity/clock/sample placement, never mtime.

    This is only arm association and stable ordering. It does not convert the
    local audio clock into UTC or align an independent witness clock.
    """
    capture_id = _uuid(capture_id)
    rows, clocks = [], set()
    for path in paths:
        path = Path(path)
        binding = validate_artifact(path, clock_id=clock_id)
        if binding['status'] != 'bound' or binding['capture_id'] != capture_id:
            raise IdentityError('Window does not belong to the selected capture')
        facts = json.loads(path.with_suffix('.json').read_text())
        end, count = facts.get('end_stream_sample'), facts.get('samples')
        if (type(end) is not int or type(count) is not int or count < 0
                or end < count):
            raise IdentityError('Window has no valid local sample placement')
        clocks.add(binding['clock_id'])
        rows.append((end - count, end, str(path), path))
    if len(clocks) > 1:
        raise IdentityError('Windows name different local stream clocks')
    return [row[3] for row in sorted(rows)]


def associated_transcripts(repo, provenance_path, record, legacy_candidates=()):
    """Resolve delivery/evidence logs without a basename first/last winner."""
    repo, provenance_path = Path(repo), Path(provenance_path)
    if provenance_path.exists():
        metadata_record(provenance_path)
    candidates = [provenance_path.parent / 'arm.log']
    source = record.get('source') or {}
    converted = source.get('converted_from')
    if converted:
        converted = Path(converted)
        candidates.append(converted if converted.is_absolute() else repo / converted)
    reference = evidence_reference(provenance_path.parent)
    if reference['status'] != 'bound' and converted:
        reference = evidence_reference(candidates[-1])
        if reference['status'] == 'bound':
            candidates = [candidates[-1]]  # Derived copy has no writer authority.
    if reference['status'] == 'bound':
        expected = record.get('capture_id') or (source.get('capture_reference') or {}).get('capture_id')
        expected_registry = record.get('registry_sha256') or (source.get('capture_reference') or {}).get('registry_sha256')
        if expected not in (None, reference['capture_id']) or expected_registry not in (None, reference['registry_sha256']):
            raise IdentityError('Evidence record and transcript registry disagree')
        declared = reference.get('transcript')
        if declared:
            candidates.append(Path(declared))
        candidates = list(dict.fromkeys(path for path in candidates if path.is_file()))
        for path in candidates:
            actual = evidence_reference(path)
            if (actual.get('capture_id') != reference['capture_id'] or
                    actual.get('registry_sha256') != reference['registry_sha256']):
                raise IdentityError('Transcript does not belong to selected capture')
        return candidates
    if record.get('capture_id') or (source.get('capture_reference') or {}).get('capture_id'):
        raise IdentityError('Claimed source identity has no validated transcript/root')
    legacy_candidates = list(dict.fromkeys(Path(path) for path in legacy_candidates))
    if len(legacy_candidates) > 1:
        raise IdentityError('Legacy transcript basename has multiple candidates')
    return list(dict.fromkeys(path for path in candidates + legacy_candidates if path.is_file()))
