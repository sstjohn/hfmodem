# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Explicit local BBS exports. No protocol, network, or radio operations.

An allowed callsign is a policy on the current link's claimed identity, not
cryptographic authentication. Binary transfer framing is deliberately separate.
"""
from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
import re
import stat
import threading


class ExportError(ValueError):
    """The requested export is disabled, inaccessible, unsafe or has changed."""


@dataclass(frozen=True)
class Limits:
    files: int = 500
    file_bytes: int = 1024 * 1024
    total_bytes: int = 16 * 1024 * 1024
    directory_entries: int = 2000

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.files, self.file_bytes, self.total_bytes, self.directory_entries)):
            raise ValueError('Export limits must be positive integers')


@dataclass(frozen=True)
class Entry:
    name: str
    size: int
    mtime_ns: int
    identity: tuple = field(repr=False)
    sha256: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Snapshot:
    entry: Entry
    data: bytes = field(repr=False)
    sha256: str
    generation: int


@dataclass(frozen=True)
class Status:
    enabled: bool
    usable: bool
    root: str | None
    files: int
    total_bytes: int
    allowed_calls: tuple[str, ...]
    generation: int
    reason: str = ''


def safe_name(name):
    return (isinstance(name, str) and bool(name) and not name.startswith('.')
            and len(name.encode('utf-8', errors='surrogatepass')) <= 255
            and name.isprintable() and not any(c in name for c in '/\\<>|:')
            and name == name.strip())


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _call(value):
    value = str(value).strip().upper()
    if not re.fullmatch(r'[A-Z0-9]{3,10}(?:-[0-9]{1,2})?', value):
        raise ExportError('Invalid allowed callsign')
    return value


class HostCatalog:
    """Off by default; shares only safe regular files directly in an opted-in root.

    Holds a directory descriptor, checks its path identity before each request,
    and opens entries relative to that descriptor without following symlinks.
    Hard links, hidden files and unsafe names are excluded from both operations.
    Any count/size limit failure refuses the whole catalog, never a partial list.
    Listings bind content; readiness checks inspect metadata without reading it.
    """
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self._lock = threading.RLock()
        self._fd = None
        self._root = None
        self._root_identity = None
        self._allowed = ()
        self.generation = 0

    def disable(self):
        with self._lock:
            self.generation += 1
            if self._fd is not None:
                os.close(self._fd)
            self._fd = self._root = self._root_identity = None
            self._allowed = ()

    close = disable

    def enable(self, root, *, allowed_calls=()):
        """Replace the export. A failed replacement leaves hosting disabled."""
        with self._lock:
            self.disable()
            try:
                allowed = tuple(sorted({_call(call) for call in allowed_calls}))
                # Do not resolve a root symlink into a different directory.
                path = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                self._fd, self._root = fd, path
                info = os.fstat(fd)
                self._root_identity = (info.st_dev, info.st_ino)
                self._allowed = allowed
                self._catalog()
            except (OSError, ValueError, TypeError) as error:
                self.disable()
                raise ExportError(f'Cannot enable BBS export: {error}') from error
            return self.status()

    def _check_root(self):
        if self._fd is None:
            raise ExportError('BBS hosting is off')
        info = os.stat(self._root, follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode) or not info.st_nlink
                or (info.st_dev, info.st_ino) != self._root_identity):
            raise ExportError('BBS export root changed; enable it again explicitly')

    def _authorize(self, peer):
        self._check_root()
        if self._allowed and _call(peer) not in self._allowed:
            raise ExportError('This link identity is not allowed to use the BBS export')

    def _entry(self, name, info):
        if not safe_name(name) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return None
        if info.st_size > self.limits.file_bytes:
            raise ExportError('BBS export exceeds the per-file byte limit')
        return Entry(name, info.st_size, info.st_mtime_ns, _identity(info))

    def _catalog(self):
        self._check_root()
        entries, total = [], 0
        with os.scandir(self._fd) as scan:
            for count, item in enumerate(scan, 1):
                if count > self.limits.directory_entries:
                    raise ExportError('BBS export exceeds the directory scan limit')
                if not safe_name(item.name):
                    continue
                entry = self._entry(item.name, item.stat(follow_symlinks=False))
                if entry is None:
                    continue
                entries.append(entry)
                total += entry.size
                if len(entries) > self.limits.files or total > self.limits.total_bytes:
                    raise ExportError('BBS export exceeds the catalog count/byte limit')
        return tuple(sorted(entries, key=lambda entry: entry.name))

    def list_files(self, peer=''):
        with self._lock:
            try:
                self._authorize(peer)
                return tuple(self._snapshot(entry).entry for entry in self._catalog())
            except OSError as error:
                raise ExportError('BBS export is unavailable') from error

    def status(self):
        with self._lock:
            entries, reason = (), ''
            try:
                entries = self._catalog()
            except (ExportError, OSError) as error:
                reason = str(error)
            return Status(self._fd is not None, not reason,
                          str(self._root) if self._root is not None else None,
                          len(entries), sum(e.size for e in entries), self._allowed,
                          self.generation, reason)

    @property
    def usable(self):
        """Local export readiness only, not eligibility to advertise service B.

        A complete, validated wire implementation is additionally required
        before any frontend may advertise file hosting to a peer.
        """
        return self.status().usable

    def read_file(self, name, peer='', *, expected=None):
        """Bounded immutable bytes, optionally tied to a previously listed Entry.

        An expected entry must include the listing's content fingerprint: file
        timestamps and size need not change on every write. Neither operation
        locks local writers or promises a filesystem-atomic read.
        The caller must separately check generation before continuing a transfer
        after hosting is turned off or its configuration changes.
        """
        with self._lock:
            if not safe_name(name):
                raise ExportError('Unsafe BBS filename')
            try:
                self._authorize(peer)
                entry = next((e for e in self._catalog() if e.name == name), None)
                if entry is None or expected is not None and (
                        not isinstance(expected, Entry) or expected.sha256 is None
                        or entry != replace(expected, sha256=None)):
                    raise ExportError('BBS file is unavailable or changed since listing')
                snapshot = self._snapshot(entry)
                if expected is not None and snapshot.sha256 != expected.sha256:
                    raise ExportError('BBS file changed since listing')
                return snapshot
            except OSError as error:
                raise ExportError('BBS file is unavailable') from error

    def _snapshot(self, entry):
        # NONBLOCK prevents a concurrently substituted FIFO blocking open.
        fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=self._fd)
        try:
            before = os.fstat(fd)
            if self._entry(entry.name, before) != entry:
                raise ExportError('BBS file changed before snapshot')
            chunks, remaining = [], entry.size + 1
            while remaining:
                part = os.read(fd, min(remaining, 65536))
                if not part:
                    break
                chunks.append(part)
                remaining -= len(part)
            data = b''.join(chunks)
            if len(data) != entry.size or _identity(os.fstat(fd)) != entry.identity:
                raise ExportError('BBS file changed during snapshot')
            self._check_root()
            digest = hashlib.sha256(data).hexdigest()
            return Snapshot(replace(entry, sha256=digest), data, digest, self.generation)
        finally:
            os.close(fd)
