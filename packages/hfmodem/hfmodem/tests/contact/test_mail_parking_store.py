# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json

import pytest

from hfmodem.contact import mail_parking as p
from hfmodem.contact.varac import VMail


class Clock:
    def __init__(self):
        self.wall = 1793203200.0
        self.mono = 1000.0

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds

    def store(self, root):
        return p.ParkedStore(root, clock=lambda: self.wall, monotonic=lambda: self.mono)


def deposit(store, **changes):
    fields = dict(owner='W9SSJ', sender='K1ABC', recipient='W9SSJ-1', via='K2ABC',
                  subject='hello', body='café\r\n73', composed='2026-10-28 12:00:00')
    fields.update(changes)
    return store.accept(**fields)


@pytest.mark.parametrize(('raw', 'canonical'), [
    ('w9ssj', 'W9SSJ'), ('W9SSJ-0', 'W9SSJ'), ('w9ssj-00', 'W9SSJ'),
    ('W9SSJ-01', 'W9SSJ-1'), ('W9SSJ-15', 'W9SSJ-15'),
])
def test_parking_address_canonical_full_identity(raw, canonical):
    assert p.parking_address(raw) == canonical


@pytest.mark.parametrize('raw', ['ALL', 'CQ', 'all-0', 'CQ-1', 'W9SSJ-16', 'W9SSJ-001',
                                'X-W9SSJ', 'K1ABC/P', 'K1234567', 'alice@example.com',
                                '', '1234', 'HELLO', ' W9SSJ', None])
def test_parking_address_rejects_group_and_unrepresentable_destinations(raw):
    with pytest.raises(ValueError):
        p.parking_address(raw)


def test_store_defaults_canonical_identity_custody_and_snapshot(tmp_path):
    store = p.ParkedStore(tmp_path)
    assert store.available
    assert store.settings == dict(parking=True, relay=True, expiry_days=30,
                                  per_destination=100, total=1000)
    row, duplicate = deposit(store, owner='w9ssj-0', recipient='w9ssj-01', sender='k1abc')
    assert not duplicate
    assert (row['id'], row['owner'], row['to'], row['from'], row['via']) == (
        1, 'W9SSJ', 'W9SSJ-1', 'K1ABC', 'K2ABC')
    assert row['state'] == 'queued' and row['attempts'] == 0
    assert row['expires_unix'] - row['received_unix'] == 30 * 86400
    assert store.pending('W9SSJ-00', 'W9SSJ-01') == [row]
    assert not store.pending('W9SSJ-1')
    row['body'] = 'attempted mutation'
    assert store.get(1)['body'] == 'café\r\n73'
    data = json.loads(store.path.read_text())
    assert data['version'] == 1 and data['next_id'] == 2
    assert data['dedup'] == {store.get(1)['dedup_key']: 1}
    store.close()
    restarted = p.ParkedStore(tmp_path)
    assert restarted.available and restarted.get(1)['body'] == 'café\r\n73'
    restarted.close()


def test_lifetime_lock_fails_closed_and_releases_on_close(tmp_path):
    first = p.ParkedStore(tmp_path)
    second = p.ParkedStore(tmp_path)
    assert first.available and not second.available
    with pytest.raises(OSError, match='unavailable'):
        deposit(second)
    first.close()
    third = p.ParkedStore(tmp_path)
    assert third.available
    third.close()


@pytest.mark.parametrize('raw', ['{', '[]', '{"version":2}', '{"version":1}'])
def test_corrupt_store_is_preserved_and_unavailable(tmp_path, raw):
    path = tmp_path / 'parked-mail.json'
    path.write_text(raw)
    store = p.ParkedStore(tmp_path)
    assert not store.available and store.error
    assert path.read_text() == raw
    with pytest.raises(OSError, match='unavailable'):
        store.configure(parking=True)


@pytest.mark.parametrize('change', [
    lambda data: data.update(version=True),
    lambda data: data.update(version=1.0),
    lambda data: data['records'][0].update(state='lost'),
    lambda data: data['records'][0].update(to='W9SSJ-01'),
    lambda data: data['records'][0].update(composed_utc='not a timestamp'),
    lambda data: data['records'][0].update(attempts=-1),
    lambda data: data['records'][0].update(id=2),
    lambda data: data['dedup'].clear(),
    lambda data: data['settings'].update(parking='yes'),
    lambda data: data['settings'].update(total=0),
    lambda data: data['records'][0].update(notification={'last_attempt': 1}),
    lambda data: data['records'][0].update(notification=dict(
        last_attempt=True, next_eligible=1801, last_sent=None)),
    lambda data: data['records'][0].update(notification=dict(
        last_attempt=1, next_eligible=1800, last_sent=None)),
    lambda data: data['records'][0].update(notification=dict(
        last_attempt=1, next_eligible=1801, last_sent='sent')),
])
def test_schema_validation_preserves_invalid_existing_snapshot(tmp_path, change):
    store = p.ParkedStore(tmp_path)
    deposit(store)
    path = store.path
    store.close()
    data = json.loads(path.read_text())
    change(data)
    raw = json.dumps(data)
    path.write_text(raw)
    restarted = p.ParkedStore(tmp_path)
    assert not restarted.available
    assert path.read_text() == raw


def test_hash_collision_cannot_acknowledge_different_canonical_fields(tmp_path, monkeypatch):
    store = p.ParkedStore(tmp_path)
    original, _ = deposit(store)
    monkeypatch.setattr(p, '_key', lambda *args: original['dedup_key'])
    with pytest.raises(ValueError, match='hash collision'):
        deposit(store, body='different body')
    assert store.records == [original]
    same, duplicate = deposit(store, via='K3ABC')
    assert same == original and duplicate
    store.close()


def test_stable_dedup_tombstone_survives_delivery_restart_and_parking_off(tmp_path):
    store = p.ParkedStore(tmp_path)
    original, _ = deposit(store)
    store.begin(original['id'])
    store.resolve(original['id'], 'SMR')
    store.configure(parking=False)
    store.close()
    restarted = p.ParkedStore(tmp_path)
    repeated, duplicate = deposit(restarted, via='K3ABC')
    assert duplicate and repeated['id'] == original['id']
    assert repeated['state'] == 'delivered'
    assert len(restarted.records) == 1
    with pytest.raises(p.ParkingDisabled):
        deposit(restarted, body='another body')
    assert not restarted.pending('W9SSJ')
    restarted.close()


def test_no_tme_uses_contact_only_dedup_and_keeps_estimate_local(tmp_path):
    clock = Clock()
    store = clock.store(tmp_path)
    first, duplicate = deposit(store, composed=None, contact=1)
    assert not duplicate and first['composed_utc'] is None and first['dedup_key'] is None
    assert first['estimated_composed_utc'] == first['received_utc']
    repeated, duplicate = deposit(store, composed=None, contact=1)
    assert duplicate and repeated['id'] == first['id']
    new_contact, duplicate = deposit(store, composed=None, contact=2)
    assert not duplicate and new_contact['id'] != first['id']
    assert json.loads(store.path.read_text())['dedup'] == {}
    store.close()
    clock.advance(86400)
    restarted = clock.store(tmp_path)
    saved = restarted.get(first['id'])
    onward = VMail(saved['subject'], saved['body'], sender=saved['from'],
                   composed=saved['composed_utc']).wire()
    assert '<TME:' not in onward and '<FRM:K1ABC>' in onward
    third, duplicate = deposit(restarted, composed=None, contact=1)
    assert not duplicate and third['id'] == 3
    restarted.close()


@pytest.mark.parametrize('subject', ['', 'a'])
def test_subjectless_and_missing_metadata_remain_forwardable_after_restart(tmp_path, subject):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store, subject=subject, composed=None)
    store.close()
    restarted = p.ParkedStore(tmp_path)
    saved = restarted.get(row['id'])
    wire = VMail(saved['subject'], saved['body'], sender=saved['from'],
                 composed=saved['composed_utc']).wire()
    parsed = VMail.parse(wire)
    assert parsed.subject == subject and parsed.composed is None
    assert parsed.sender == 'K1ABC' and parsed.body == row['body']
    restarted.close()


def test_local_composition_requires_subject_and_rejects_canonical_self(tmp_path):
    store = p.ParkedStore(tmp_path)
    with pytest.raises(ValueError, match='subject'):
        deposit(store, subject='', origin='local')
    with pytest.raises(ValueError, match='inbox'):
        deposit(store, owner='W9SSJ-0', recipient='w9ssj-00', origin='local')
    store.configure(parking=False)
    row, duplicate = deposit(store, sender='W9SSJ', via='W9SSJ', origin='local')
    assert not duplicate and row['origin'] == 'local'
    store.close()


def test_content_limit_counts_crlf_and_unicode_without_truncation(tmp_path):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store, subject='s', body='é' * 497 + '\n')
    assert row['body'] == 'é' * 497 + '\n'
    with pytest.raises(ValueError, match='500-character'):
        deposit(store, subject='s', body='é' * 498 + '\n')
    with pytest.raises(ValueError, match='escaping'):
        deposit(store, body='<SMR>')
    with pytest.raises(ValueError, match='escaping'):
        deposit(store, subject='line\nline')
    store.close()


def test_capture_over_500_encoded_bytes_fits_content_admission(tmp_path):
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'forwarded-vmail-1024.json').read_text())
    wire = bytes.fromhex(fixture['wire_hex'])
    mail = VMail.parse(wire.partition(b' ')[2].decode())
    assert len(wire.partition(b' ')[2]) > 500
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store, sender=mail.sender, subject=mail.subject,
                     body=mail.body, composed=mail.composed)
    assert row['subject'] == mail.subject and row['body'] == mail.body
    store.close()


def test_quota_counts_inflight_uncertain_and_exact_destination(tmp_path):
    store = p.ParkedStore(tmp_path)
    store.configure(per_destination=1, total=2)
    first, _ = deposit(store)
    store.begin(first['id'])
    with pytest.raises(ValueError, match='destination quota'):
        deposit(store, subject='another')
    second, _ = deposit(store, recipient='W9SSJ-2')
    store.begin(second['id'])
    store.fail(second['id'], 'lost acknowledgement')
    with pytest.raises(ValueError, match='total quota'):
        deposit(store, recipient='W9SSJ-3')
    store.resolve(first['id'], 'SMR')
    third, _ = deposit(store, subject='third')
    assert third['id'] == 3
    store.close()


def test_expiry_retains_content_and_dedup_evidence(tmp_path):
    clock = Clock()
    store = clock.store(tmp_path)
    store.configure(expiry_days=1)
    original, _ = deposit(store)
    clock.advance(86400)
    assert not store.pending('W9SSJ')
    expired = store.get(original['id'])
    assert expired['state'] == 'expired' and expired['body'] == original['body']
    assert expired['expired_utc']
    same, duplicate = deposit(store)
    assert duplicate and same['state'] == 'expired'
    with pytest.raises(ValueError, match='eligible'):
        store.begin(original['id'])
    store.close()


@pytest.mark.parametrize('terminal', ['SMR', 'SMF', 'SMFP'])
def test_attempt_is_durable_and_only_smr_delivers(tmp_path, terminal):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store)
    started = store.begin(row['id'])
    assert started['attempts'] == 1 and started['state'] == 'in_flight'
    assert json.loads(store.path.read_text())['records'][0]['state'] == 'in_flight'
    result = store.resolve(row['id'], terminal)
    assert result['state'] == ('delivered' if terminal == 'SMR' else 'queued')
    assert bool(result['delivered_utc']) == (terminal == 'SMR')
    assert result['last_error'] == (None if terminal == 'SMR' else terminal)
    store.close()


def test_restart_recovers_unfinished_attempt_as_uncertain(tmp_path):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store)
    store.begin(row['id'])
    store.close()
    restarted = p.ParkedStore(tmp_path)
    result = restarted.get(row['id'])
    assert result['state'] == 'uncertain' and result['attempts'] == 1
    assert restarted.pending('W9SSJ', 'W9SSJ-1') == [result]
    assert json.loads(restarted.path.read_text())['records'][0]['state'] == 'uncertain'
    restarted.close()


@pytest.mark.parametrize('state', ['queued', 'uncertain'])
def test_cancel_retains_content_and_stable_tombstone(tmp_path, state):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store)
    if state == 'uncertain':
        store.begin(row['id'])
        store.fail(row['id'], 'timeout')
    cancelled = store.cancel(row['id'])
    assert cancelled['state'] == 'cancelled' and cancelled['body'] == row['body']
    assert cancelled['cancelled_utc'] and cancelled['cancel_requested']
    assert not store.pending('W9SSJ')
    same, duplicate = deposit(store)
    assert duplicate and same['state'] == 'cancelled'
    store.close()


def test_cancel_inflight_honors_late_smr_and_suppresses_restart_retry(tmp_path):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store)
    store.begin(row['id'])
    cancelled = store.cancel(row['id'])
    assert cancelled['state'] == 'in_flight' and cancelled['cancel_requested']
    store.fail(row['id'], 'operator stopped waiting')
    delivered = store.resolve(row['id'], 'SMR')
    assert delivered['state'] == 'delivered' and delivered['delivered_utc']
    other, _ = deposit(store, subject='next')
    store.begin(other['id'])
    store.cancel(other['id'])
    store.close()
    restarted = p.ParkedStore(tmp_path)
    assert restarted.get(other['id'])['state'] == 'cancelled'
    assert not restarted.pending('W9SSJ')
    restarted.close()


@pytest.mark.parametrize('operation', ['file_fsync_before', 'file_fsync_after',
                                       'replace_before', 'replace_after',
                                       'directory_fsync_before', 'directory_fsync_after'])
def test_commit_failures_never_acknowledge_and_preserve_prior_custody(tmp_path, monkeypatch, operation):
    store = p.ParkedStore(tmp_path)
    prior, _ = deposit(store)
    fsync, replace = p.os.fsync, p.os.replace
    calls = 0

    def fail_fsync(fd):
        nonlocal calls
        calls += 1
        target = (calls == 1 and operation.startswith('file_fsync') or
                  calls == 2 and operation.startswith('directory_fsync'))
        if target and operation.endswith('before'):
            raise OSError('injected disk failure')
        fsync(fd)
        if target:
            raise OSError('injected disk failure')

    def fail_replace(source, destination):
        if operation == 'replace_before':
            raise OSError('injected disk failure')
        replace(source, destination)
        if operation == 'replace_after':
            raise OSError('injected disk failure')

    with monkeypatch.context() as patch:
        patch.setattr(p.os, 'fsync', fail_fsync)
        patch.setattr(p.os, 'replace', fail_replace)
        with pytest.raises(OSError, match='injected disk failure'):
            deposit(store, subject='unacknowledged')
    assert not store.available
    store.close()
    restarted = p.ParkedStore(tmp_path)
    assert restarted.available and restarted.get(prior['id']) == prior
    rows = restarted.records
    assert len(rows) == (2 if operation in ('replace_after', 'directory_fsync_before',
                                          'directory_fsync_after') else 1)
    retry, duplicate = deposit(restarted, subject='unacknowledged')
    assert retry['id'] == 2 and duplicate == (len(rows) == 2)
    assert not list(tmp_path.glob('.parked-mail-*'))
    restarted.close()


def test_settings_persist_and_failed_write_does_not_claim_change(tmp_path, monkeypatch):
    store = p.ParkedStore(tmp_path)
    store.configure(parking=False, relay=False, expiry_days=12, total=23, per_destination=4)
    expected = store.settings
    store.close()
    restarted = p.ParkedStore(tmp_path)
    assert restarted.settings == expected
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError('disk full')
        patch.setattr(p.os, 'replace', fail)
        with pytest.raises(OSError, match='disk full'):
            restarted.configure(parking=True)
    assert restarted.settings == expected and not restarted.available
    restarted.close()
    again = p.ParkedStore(tmp_path)
    assert again.settings == expected
    again.close()


def test_whole_snapshot_bound_refuses_new_deposits_without_deleting(tmp_path, monkeypatch):
    store = p.ParkedStore(tmp_path)
    row, _ = deposit(store)
    before = store.path.read_bytes()
    monkeypatch.setattr(p, 'STORE_MAX_BYTES', len(before) + 10)
    with pytest.raises(ValueError, match='full'):
        deposit(store, subject='another')
    assert store.available and store.get(row['id']) == row
    assert store.path.read_bytes() == before
    store.close()


def test_admission_reserves_space_for_full_custody_lifecycle(tmp_path, monkeypatch):
    clock = Clock()
    reference = clock.store(tmp_path / 'reference')
    deposit(reference)
    size = len(reference.path.read_bytes())
    reference.close()
    reserve = p.LIFECYCLE_RESERVE_BYTES + p.COOLDOWN_RESERVE_BYTES
    monkeypatch.setattr(p, 'STORE_MAX_BYTES', size + reserve - 1)
    store = clock.store(tmp_path / 'bounded')
    with pytest.raises(ValueError, match='metadata reserve'):
        deposit(store)
    assert not store.records
    monkeypatch.setattr(p, 'STORE_MAX_BYTES', size + reserve)
    row, _ = deposit(store)
    store.begin(row['id'])
    store.reserve_notice('W9SSJ', 'W9SSJ-1')
    store.notice_sent('W9SSJ', 'W9SSJ-1')
    failed = store.fail(row['id'], 'é' * 10000)
    assert len(failed['last_error'].encode()) <= p.ERROR_MAX_BYTES
    store.resolve(row['id'], 'SMR')
    assert store.get(row['id'])['state'] == 'delivered'
    assert len(store.path.read_bytes()) <= p.STORE_MAX_BYTES
    store.close()


def test_notification_cooldown_reservation_sent_and_restart_rollback(tmp_path):
    clock = Clock()
    store = clock.store(tmp_path)
    row, _ = deposit(store)
    assert store.eligible('W9SSJ-0', 'W9SSJ-01')
    store.reserve_notice('W9SSJ', 'W9SSJ-1')
    attempt = store.notice('W9SSJ', 'W9SSJ-1')
    assert attempt == dict(last_attempt=clock.wall, next_eligible=clock.wall + 1800,
                           last_sent=None)
    assert store.get(row['id'])['notification'] == attempt
    assert not store.eligible('W9SSJ', 'W9SSJ-1')
    assert store.eligible('W9SSJ', 'W9SSJ-2')
    clock.advance(20)
    store.notice_sent('W9SSJ', 'W9SSJ-1')
    assert store.notice('W9SSJ', 'W9SSJ-1')['last_sent'] == clock.wall
    assert store.get(row['id'])['state'] == 'queued'
    clock.wall += 86400
    assert not store.eligible('W9SSJ', 'W9SSJ-1')
    clock.wall -= 90000
    store.close()
    restarted = clock.store(tmp_path)
    assert not restarted.eligible('W9SSJ', 'W9SSJ-1')
    clock.advance(1800)
    assert not restarted.eligible('W9SSJ', 'W9SSJ-1')
    clock.advance(3600)
    assert restarted.eligible('W9SSJ', 'W9SSJ-1')
    restarted.close()


def test_notification_failed_admission_preserves_full_cooldown(tmp_path):
    clock = Clock()
    store = clock.store(tmp_path)
    deposit(store)
    store.reserve_notice('W9SSJ', 'W9SSJ-1')
    store.close()
    restarted = clock.store(tmp_path)
    assert restarted.notice('W9SSJ', 'W9SSJ-1')['last_sent'] is None
    with pytest.raises(ValueError, match='cooldown'):
        restarted.reserve_notice('W9SSJ', 'W9SSJ-1')
    clock.advance(1799)
    assert not restarted.eligible('W9SSJ', 'W9SSJ-1')
    clock.advance(1)
    assert restarted.eligible('W9SSJ', 'W9SSJ-1')
    restarted.close()
