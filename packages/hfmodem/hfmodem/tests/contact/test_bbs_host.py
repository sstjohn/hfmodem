# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import os
from dataclasses import replace

import pytest

from hfmodem.contact import bbs
from hfmodem.contact.bbs import ExportError, HostCatalog, Limits


@pytest.fixture
def host():
    value = HostCatalog()
    yield value
    value.close()


def test_off_by_default_and_explicit_snapshot(tmp_path, host):
    assert not host.status().enabled
    with pytest.raises(ExportError, match='off'):
        host.list_files()
    (tmp_path / 'hello.txt').write_bytes(b'hello\x00\xff')
    status = host.enable(tmp_path)
    assert status.enabled and status.usable and status.files == 1
    entry, = host.list_files()
    snapshot = host.read_file(entry.name, expected=entry)
    assert snapshot.data == b'hello\x00\xff'
    assert snapshot.sha256 == hashlib.sha256(snapshot.data).hexdigest()
    (tmp_path / 'hello.txt').write_bytes(b'changed')
    assert snapshot.data == b'hello\x00\xff'
    with pytest.raises(ExportError, match='changed'):
        host.read_file(entry.name, expected=entry)
    host.disable()
    assert host.generation != snapshot.generation
    assert not host.usable


@pytest.fixture
def timestamp_collision(monkeypatch):
    """Model writes that share both filesystem timestamps, on every platform."""
    identity, entry = bbs._identity, HostCatalog._entry

    def same_times(info):
        return (*identity(info)[:3], 0, 0)

    def same_display_time(self, name, info):
        value = entry(self, name, info)
        return replace(value, mtime_ns=0) if value is not None else None

    monkeypatch.setattr(bbs, '_identity', same_times)
    monkeypatch.setattr(HostCatalog, '_entry', same_display_time)


def test_same_metadata_changed_content_is_refused_before_first_read(tmp_path, host, timestamp_collision):
    target = tmp_path / 'hello.txt'
    target.write_bytes(b'listing')
    host.enable(tmp_path)
    expected, = host.list_files()
    target.write_bytes(b'changed')
    current, = host._catalog()
    assert current.identity == expected.identity and current.mtime_ns == expected.mtime_ns
    with pytest.raises(ExportError, match='changed'):
        host.read_file(expected.name, expected=expected)


def test_same_metadata_unchanged_content_is_accepted(tmp_path, host, timestamp_collision):
    target = tmp_path / 'hello.txt'
    target.write_bytes(b'listing')
    host.enable(tmp_path)
    expected, = host.list_files()
    target.write_bytes(b'listing')
    snapshot = host.read_file(expected.name, expected=expected)
    assert snapshot.entry == expected
    assert snapshot.data == b'listing'
    assert snapshot.sha256 == hashlib.sha256(b'listing').hexdigest()


@pytest.mark.parametrize('kind', ['metadata_only', 'wrong_type', 'other_file'])
def test_expected_entry_must_bind_the_requested_content(tmp_path, host, kind):
    (tmp_path / 'a').write_bytes(b'first')
    (tmp_path / 'b').write_bytes(b'other')
    host.enable(tmp_path)
    first, other = host.list_files()
    expected = (replace(first, sha256=None) if kind == 'metadata_only' else
                object() if kind == 'wrong_type' else other)
    with pytest.raises(ExportError, match='changed'):
        host.read_file(first.name, expected=expected)


def test_status_reads_no_payload_and_listing_and_target_reads_are_bounded(tmp_path, host, monkeypatch):
    payloads = {'a': b'first', 'b': b'other file'}
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    read, catalog = os.read, host._catalog
    reads, scans = [], []

    def tracked_read(fd, size):
        data = read(fd, size)
        reads.append((os.fstat(fd).st_ino, size, len(data)))
        return data

    def tracked_catalog():
        scans.append(1)
        return catalog()

    monkeypatch.setattr(os, 'read', tracked_read)
    monkeypatch.setattr(host, '_catalog', tracked_catalog)
    host.enable(tmp_path)
    assert host.status().usable and host.usable
    assert reads == []
    scans.clear()
    entries = host.list_files()
    assert len(scans) == 1
    assert sum(row[2] for row in reads) == sum(map(len, payloads.values()))
    assert all(row[1] <= len(payloads[entry.name]) + 1
               for entry in entries for row in reads if row[0] == entry.identity[1])
    reads.clear()
    scans.clear()
    snapshot = host.read_file('a', expected=entries[0])
    assert len(scans) == 1
    assert {row[0] for row in reads} == {entries[0].identity[1]}
    assert sum(row[2] for row in reads) == len(payloads['a'])
    assert snapshot.data == payloads['a']


@pytest.mark.parametrize('name', ['../secret', '/secret', '.secret', 'a/b', 'a\\b',
                                 'a|b', '<BG>', 'a:b', 'a\nb', ' leading', 'trail '])
def test_unsafe_requests_rejected(tmp_path, host, name):
    host.enable(tmp_path)
    with pytest.raises(ExportError, match='Unsafe'):
        host.read_file(name)


def test_only_flat_safe_single_link_regular_files(tmp_path, host):
    root = tmp_path / 'share'
    root.mkdir()
    secret = tmp_path / 'secret'
    secret.write_bytes(b'private')
    (root / 'symlink').symlink_to(secret)
    os.link(secret, root / 'hardlink')
    (root / '.hidden').write_bytes(b'hidden')
    (root / 'subdir').mkdir()
    (root / 'subdir' / 'nested').write_bytes(b'nested')
    os.mkfifo(root / 'fifo')
    (root / 'z.txt').write_bytes(b'z')
    (root / 'a.txt').write_bytes(b'a')
    host.enable(root)
    assert [e.name for e in host.list_files()] == ['a.txt', 'z.txt']
    for name in ['symlink', 'hardlink', 'subdir', 'fifo']:
        with pytest.raises(ExportError):
            host.read_file(name)


def test_symlink_root_and_failed_reconfiguration_disable(tmp_path, host):
    root = tmp_path / 'root'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    host.enable(root)
    with pytest.raises(ExportError):
        host.enable(alias)
    assert not host.status().enabled


def test_replaced_root_never_serves_old_or_new_files(tmp_path, host):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'x').write_bytes(b'old')
    host.enable(root)
    root.rename(tmp_path / 'old')
    root.mkdir()
    (root / 'x').write_bytes(b'new')
    assert not host.usable
    with pytest.raises(ExportError, match='root changed'):
        host.read_file('x')


@pytest.mark.parametrize('limits,files', [
    (Limits(files=1), {'a': b'a', 'b': b'b'}),
    (Limits(file_bytes=1), {'a': b'ab'}),
    (Limits(total_bytes=1), {'a': b'a', 'b': b'b'}),
    (Limits(directory_entries=1), {'.a': b'', '.b': b''}),
])
def test_bounds_fail_whole_catalog(tmp_path, limits, files):
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    host = HostCatalog(limits)
    with pytest.raises(ExportError, match='limit'):
        host.enable(tmp_path)
    assert not host.status().enabled


def test_limits_rechecked_after_enable(tmp_path, host):
    host.limits = Limits(file_bytes=1)
    host.enable(tmp_path)
    (tmp_path / 'large').write_bytes(b'ab')
    assert not host.usable
    with pytest.raises(ExportError, match='limit'):
        host.list_files()


def test_allowlist_is_current_claimed_identity_exact_not_authentication(tmp_path, host):
    (tmp_path / 'x').write_bytes(b'x')
    host.enable(tmp_path, allowed_calls=['w9ssj-1', 'K0EPI'])
    assert host.status().allowed_calls == ('K0EPI', 'W9SSJ-1')
    assert host.read_file('x', 'w9ssj-1').data == b'x'
    for peer in ['W9SSJ', 'W9SSJ-2', 'KL7QR', '']:
        with pytest.raises(ExportError):
            host.list_files(peer)
        with pytest.raises(ExportError):
            host.read_file('x', peer)


def test_symlink_swap_between_catalog_and_open_is_refused(tmp_path, host, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    target = root / 'x'
    target.write_bytes(b'public')
    secret = tmp_path / 'secret'
    secret.write_bytes(b'private')
    host.enable(root)
    original = os.open

    def swapped(path, flags, *args, **kwargs):
        if path == 'x':
            target.unlink()
            target.symlink_to(secret)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', swapped)
    with pytest.raises(ExportError):
        host.read_file('x')


def test_change_during_snapshot_is_refused(tmp_path, host, monkeypatch):
    target = tmp_path / 'x'
    target.write_bytes(b'old')
    host.enable(tmp_path)
    original = os.read
    reads = []

    def changed(fd, size):
        reads.append(size)
        data = original(fd, size)
        if len(reads) == 1:
            target.write_bytes(b'new and larger')
        return data

    monkeypatch.setattr(os, 'read', changed)
    with pytest.raises(ExportError, match='changed during'):
        host.read_file('x')
    assert sum(reads) <= 5  # bounded by original size plus one, never new size
