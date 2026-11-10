# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from hfmodem.bbs.messages import MessageStore, OperatorMessages, MessageError, MessageNotFound, MessageLimit, DuplicateRequest
from hfmodem.bbs.store import Store, OwnershipError


@pytest.fixture
def board(tmp_path):
    with Store(tmp_path) as store:
        a, b = store.player('K1ABC-1'), store.player('W9SSJ')
        now = [datetime(2026, 10, 26, 12, tzinfo=timezone.utc)]
        messages = MessageStore(store, {'general': 'General', 'radio': 'Radio'}, utcnow=lambda: now[0])
        yield store, a, b, messages, now


def test_public_post_reply_headers_reads_and_areas(board):
    store, a, b, messages, now = board
    original = messages.post_public(a.id, 'general', 'Hello', 'First\nSecond', 'first')
    reply = messages.post_public(b.id, 'general', 'Re hello', 'Reply', 'reply', parent_id=original.id)
    assert original.id == original.thread_id
    assert reply.parent_id == original.id and reply.thread_id == original.id
    assert messages.get(b.id, original.id).body == 'First\nSecond'
    assert messages.get(b.id, original.id).author == 'K1ABC'
    page = messages.list_public(b.id, 'general')
    assert [m.id for m in page.messages] == [original.id, reply.id]
    assert all(m.body == '' for m in page.messages)
    assert page.next_cursor is None
    assert messages.unread_counts(b.id) == {'general': 1, 'radio': 0, 'sysop': 0}
    messages.mark_read(b.id, original.id)
    messages.mark_read(b.id, original.id)
    assert messages.get(b.id, original.id).read
    assert messages.unread_counts(b.id)['general'] == 0
    assert messages.unread_counts(a.id)['general'] == 1


def test_sysop_private_thread_operator_reply_and_no_callsign_privilege(board):
    store, a, sysop_claim, messages, now = board
    private = messages.post_sysop(a.id, 'Question', 'Private body', 'private')
    operator = OperatorMessages(store, messages.areas, utcnow=lambda: now[0])
    response = operator.reply_sysop(private.id, 'Operator answer', 'operator-answer')
    caller_reply = messages.post_sysop(a.id, 'More', 'Thanks', 'thanks', parent_id=response.id)
    assert response.author == 'Sysop' and response.author_id is None
    assert caller_reply.thread_id == private.id
    assert len(operator.list_sysop().messages) == 3
    assert len(messages.list_sysop(a.id).messages) == 3
    assert messages.list_sysop(sysop_claim.id).messages == ()
    assert messages.unread_counts(sysop_claim.id)['sysop'] == 0
    for operation in (lambda: messages.get(sysop_claim.id, private.id), lambda: messages.mark_read(sysop_claim.id, private.id),
                      lambda: messages.post_sysop(sysop_claim.id, 'Intrude', 'No', 'intrude', parent_id=private.id)):
        with pytest.raises(MessageNotFound):
            operation()


def test_parent_scope_area_visibility_validation(board):
    store, a, b, messages, now = board
    public = messages.post_public(a.id, 'general', 'Public', 'Body', 'public')
    private = messages.post_sysop(a.id, 'Private', 'Body', 'private')
    for operation in (lambda: messages.post_public(a.id, 'radio', 'Cross', 'Body', 'cross1', parent_id=public.id),
                      lambda: messages.post_public(a.id, 'general', 'Cross', 'Body', 'cross2', parent_id=private.id),
                      lambda: messages.post_sysop(a.id, 'Cross', 'Body', 'cross3', parent_id=public.id),
                      lambda: messages.post_public(a.id, 'general', 'Missing', 'Body', 'missing', parent_id=999),
                      lambda: OperatorMessages(store).reply_sysop(public.id, 'Invalid', 'operator-cross')):
        with pytest.raises(MessageNotFound):
            operation()


def test_idempotence_before_cap_same_key_content_and_actor_bound(board):
    store, a, b, messages, now = board
    limited = MessageStore(store, posts_per_day=1, utcnow=lambda: now[0])
    original = limited.post_public(a.id, 'general', 'Once', 'Body', 'request')
    assert limited.post_public(a.id, 'general', 'Once', 'Body', 'request') == original
    with pytest.raises(MessageLimit):
        limited.post_sysop(a.id, 'Twice', 'Body', 'request2')
    for player, subject, body in ((a.id, 'Changed', 'Body'), (a.id, 'Once', 'Changed'), (b.id, 'Once', 'Body')):
        with pytest.raises(DuplicateRequest):
            limited.post_public(player, 'general', subject, body, 'request')
    disabled = MessageStore(store, posts_per_day=0)
    assert disabled.post_public(a.id, 'general', 'Once', 'Body', 'request').id == original.id
    with pytest.raises(MessageLimit):
        disabled.post_sysop(b.id, 'No', 'Body', 'zero')


def test_public_and_sysop_share_daily_cap_and_backward_calendar_no_reset(board):
    store, a, b, messages, now = board
    limited = MessageStore(store, posts_per_day=2, utcnow=lambda: now[0])
    limited.post_public(a.id, 'general', 'One', 'Body', '1')
    limited.post_sysop(a.id, 'Two', 'Body', '2')
    now[0] -= timedelta(days=1)
    with pytest.raises(MessageLimit):
        limited.post_public(a.id, 'general', 'Three', 'Body', '3')
    now[0] += timedelta(days=2)
    assert limited.post_public(a.id, 'general', 'Next day', 'Body', '4')


def test_header_cursor_is_bounded_stable_and_scope_filtered(board):
    store, a, b, messages, now = board
    for i in range(15):
        messages.post_public(a.id, 'general', f'Subject {i}', 'Body', f'm{i}')
    first = messages.list_public(b.id, 'general')
    assert len(first.messages) == 10 and first.next_cursor == first.messages[-1].id
    messages.post_sysop(a.id, 'Private', 'No leak', 'private')
    messages.post_public(a.id, 'general', 'Later', 'Body', 'later')
    second = messages.list_public(b.id, 'general', after=first.next_cursor)
    assert len(second.messages) == 6 and second.next_cursor is None
    assert set(m.id for m in first.messages).isdisjoint(m.id for m in second.messages)
    for arguments in ({'limit': 11}, {'limit': 0}, {'after': -1}, {'after': True}, {'after': 2**80}):
        with pytest.raises(MessageError):
            messages.list_public(a.id, 'general', **arguments)


def test_moderation_hides_content_counts_and_parent_but_keeps_reply_tree(board):
    store, a, b, messages, now = board
    original = messages.post_public(a.id, 'general', 'Secret subject', 'Secret body', 'one')
    child = messages.post_public(b.id, 'general', 'Child', 'Visible', 'two', parent_id=original.id)
    operator = OperatorMessages(store)
    operator.hide(original.id)
    hidden = messages.get(b.id, original.id)
    assert hidden.hidden and hidden.subject == '[removed]' and hidden.body == '' and hidden.author == ''
    assert [m.id for m in messages.list_public(b.id, 'general').messages] == [child.id]
    assert messages.unread_counts(b.id)['general'] == 0
    assert operator.get(original.id).body == 'Secret body'
    assert messages.get(b.id, child.id).parent_id == original.id
    with pytest.raises(MessageNotFound):
        messages.mark_read(b.id, original.id)
    with pytest.raises(MessageNotFound):
        messages.post_public(b.id, 'general', 'Reply', 'Body', 'three', parent_id=original.id)
    assert messages.post_public(a.id, 'general', 'Secret subject', 'Secret body', 'one').id == original.id
    operator.unhide(original.id)
    assert messages.unread_counts(b.id)['general'] == 1
    assert messages.get(b.id, original.id).body == 'Secret body'


def test_removed_area_not_visible_or_counted(board):
    store, a, b, messages, now = board
    posted = messages.post_public(a.id, 'radio', 'Title', 'Body', 'radio')
    changed = MessageStore(store, {'general': 'General'})
    assert changed.unread_counts(a.id) == {'general': 0, 'sysop': 0}
    with pytest.raises(MessageNotFound):
        changed.get(a.id, posted.id)
    with pytest.raises(MessageNotFound):
        changed.mark_read(a.id, posted.id)


@pytest.mark.parametrize('subject,body', [('', 'Body'), ('Title', ''), ('x'*73, 'Body'), ('é'*37, 'Body'), ('Title', 'é'*1025), ('Title', 'Bad\x1b[31m'), ('Title\nTwo', 'Body'), ('Title', 'Bad\rBody'), ('Title', '\ud800')])
def test_text_bounds_and_controls(board, subject, body):
    store, a, b, messages, now = board
    with pytest.raises(MessageError):
        messages.post_public(a.id, 'general', subject, body, 'invalid')
    assert messages.list_public(a.id, 'general').messages == ()


def test_utf8_exact_boundary_and_timestamp_injection(board):
    store, a, b, messages, now = board
    posted = messages.post_public(a.id, 'general', 'é'*36, 'é'*1024, 'boundary')
    assert posted.created_at == now[0].isoformat()
    assert len(posted.body.encode()) == 2048
    naive = MessageStore(store, utcnow=lambda: datetime(2026, 1, 1))
    with pytest.raises(MessageError):
        naive.post_sysop(a.id, 'Invalid date', 'Body', 'naive')


def test_global_record_cap_applies_even_after_hide_and_to_operator(board, monkeypatch):
    import hfmodem.bbs.messages as module
    store, a, b, messages, now = board
    private = messages.post_sysop(a.id, 'Question', 'Body', 'one')
    monkeypatch.setattr(module, 'MAX_MESSAGES', 1)
    operator = OperatorMessages(store)
    operator.hide(private.id)
    with pytest.raises(MessageLimit):
        messages.post_public(b.id, 'general', 'No capacity', 'Body', 'two')
    operator.unhide(private.id)
    with pytest.raises(MessageLimit):
        operator.reply_sysop(private.id, 'Answer', 'three')


def test_merge_retains_authors_ownership_read_union_and_daily_charge(board):
    store, a, b, messages, now = board
    public = messages.post_public(a.id, 'general', 'One', 'Body', 'public')
    private = messages.post_sysop(b.id, 'Two', 'Body', 'private')
    response = OperatorMessages(store, utcnow=lambda: now[0]).reply_sysop(private.id, 'Answer', 'answer')
    messages.mark_read(a.id, public.id)
    messages.mark_read(b.id, public.id)
    messages.mark_read(b.id, response.id)
    store.merge_players(a.id, b.id)
    assert store.player('W9SSJ').id == a.id
    assert messages.get(a.id, private.id).owner_id == a.id
    assert messages.get(a.id, private.id).author_id == a.id
    assert messages.get(a.id, response.id).read
    assert messages.get(a.id, public.id).read
    assert messages.post_sysop(a.id, 'Two', 'Body', 'private').id == private.id
    assert list(store.db.execute('PRAGMA foreign_key_check')) == []
    limited = MessageStore(store, posts_per_day=2, utcnow=lambda: now[0])
    with pytest.raises(MessageLimit):
        limited.post_public(a.id, 'general', 'Three', 'Body', 'three')


def test_restart_and_ssid_share_messages_read_and_deduplication(tmp_path):
    with Store(tmp_path) as store:
        player = store.player('K1ABC-1')
        messages = MessageStore(store)
        posted = messages.post_sysop(player.id, 'Hello', 'Persisted', 'stable')
        operator = OperatorMessages(store)
        reply = operator.reply_sysop(posted.id, 'Persistent answer', 'reply-stable')
        messages.mark_read(player.id, reply.id)
        operator.mark_read(posted.id)
        with pytest.raises(OwnershipError):
            Store(tmp_path)
    with Store(tmp_path) as store:
        player = store.player('K1ABC-2')
        messages = MessageStore(store)
        assert messages.get(player.id, posted.id).read
        assert messages.get(player.id, reply.id).read
        assert OperatorMessages(store).get(posted.id).read
        assert messages.post_sysop(player.id, 'Hello', 'Persisted', 'stable').id == posted.id
        assert messages.unread_counts(player.id)['sysop'] == 0


def _v1_database(root):
    # Derive a real v1 fixture from the released schema, with no message tables.
    with Store(root) as store:
        player = store.player('K1ABC')
        store.set_preferences(player.id, expert=False)
        with store.transaction() as db:
            db.execute('INSERT INTO usage VALUES(?,?,?)', (player.id, 'UTC:2026-10-26', 1234567))
            db.execute("INSERT INTO reservations VALUES('reservation','lease',?,'UTC:2026-10-26',0,2,1234567,0)", (player.id,))
            db.execute('DROP TABLE operator_message_reads')
            db.execute('DROP TABLE message_reads')
            db.execute('DROP TABLE message_requests')
            db.execute('DROP TABLE messages')
            db.execute('PRAGMA user_version=1')
    checkpoint = root / 'games/zork/players' / player.id / 'checkpoints/manifest.json'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{"current":"existing"}')
    return player, checkpoint


def test_v1_migration_preserves_players_quota_checkpoints_and_backup(tmp_path):
    player, checkpoint = _v1_database(tmp_path)
    before = checkpoint.read_bytes()
    with Store(tmp_path) as store:
        assert store.db.execute('PRAGMA user_version').fetchone()[0] == 2
        assert store.player('K1ABC-2').id == player.id
        assert not store.preferences(player.id)['expert']
        assert store.db.execute('SELECT charged_us FROM usage').fetchone()[0] == 1234567
        assert store.db.execute('SELECT id FROM reservations').fetchone()[0] == 'reservation'
        assert list(store.db.execute('PRAGMA foreign_key_check')) == []
        assert MessageStore(store).post_sysop(player.id, 'After migration', 'Body', 'migrated')
    assert checkpoint.read_bytes() == before
    backup = sqlite3.connect(tmp_path / 'players.sqlite.pre-v2.backup')
    assert backup.execute('PRAGMA user_version').fetchone()[0] == 1
    assert backup.execute('SELECT charged_us FROM usage').fetchone()[0] == 1234567
    backup.close()


def test_migration_failure_rolls_back_ddl_version_and_releases_owner(tmp_path, monkeypatch):
    player, checkpoint = _v1_database(tmp_path)
    def fail(db):
        db.execute('CREATE TABLE incomplete(value TEXT)')
        raise OSError('simulated migration interruption')
    with monkeypatch.context() as patch:
        patch.setattr(Store, '_migrate_v2', staticmethod(fail))
        with pytest.raises(OSError):
            Store(tmp_path)
    db = sqlite3.connect(tmp_path / 'players.sqlite')
    assert db.execute('PRAGMA user_version').fetchone()[0] == 1
    assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='incomplete'").fetchone()
    assert db.execute('SELECT charged_us FROM usage').fetchone()[0] == 1234567
    db.close()
    with Store(tmp_path) as store:
        assert store.player('K1ABC').id == player.id
        assert store.db.execute('PRAGMA user_version').fetchone()[0] == 2



def test_operator_public_listing_and_independent_read_receipts(board):
    store, a, b, messages, now = board
    public = messages.post_public(a.id, 'general', 'Title', 'Body', 'public')
    private = messages.post_sysop(a.id, 'Private', 'Body', 'private')
    operator = OperatorMessages(store)
    assert not operator.get(private.id).read
    assert not operator.list_sysop().messages[0].read
    assert operator.list_public('general').messages[0].id == public.id
    operator.mark_read(private.id)
    operator.mark_read(private.id)
    assert operator.list_sysop().messages[0].read
    assert not messages.get(b.id, public.id).read
    messages.mark_read(a.id, public.id)
    assert not operator.get(public.id).read
    operator.hide(public.id)
    assert operator.list_public('general').messages[0].hidden
    assert not operator.get(public.id).read
    assert messages.list_public(a.id, 'general').messages == ()


def test_retry_after_committed_but_unknown_completion_does_not_duplicate(board, monkeypatch):
    from contextlib import contextmanager
    store, a, b, messages, now = board
    normal_transaction = store.transaction
    @contextmanager
    def lost_completion():
        with normal_transaction() as db:
            yield db
        raise sqlite3.OperationalError('commit succeeded but completion was lost')
    with monkeypatch.context() as patch:
        patch.setattr(store, 'transaction', lost_completion)
        with pytest.raises(sqlite3.OperationalError):
            messages.post_sysop(a.id, 'Once', 'Body', 'uncertain')
    posted = messages.post_sysop(a.id, 'Once', 'Body', 'uncertain')
    assert [m.id for m in messages.list_sysop(a.id).messages] == [posted.id]


def test_concurrent_same_request_commits_one_message(board):
    from concurrent.futures import ThreadPoolExecutor
    store, a, b, messages, now = board
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(messages.post_sysop, a.id, 'Once', 'Body', 'concurrent') for _ in range(2)]
        results = [future.result() for future in futures]
    assert results[0].id == results[1].id
    assert len(messages.list_sysop(a.id).messages) == 1



def test_author_read_receipt_does_not_mark_other_readers_or_recipients(board):
    store, a, b, messages, now = board
    public = messages.post_public(a.id, 'general', 'Mine', 'Body', 'mine')
    assert public.read
    assert messages.unread_counts(a.id)['general'] == 0
    assert messages.unread_counts(b.id)['general'] == 1
    private = messages.post_sysop(a.id, 'Question', 'Body', 'question')
    assert private.read
    operator = OperatorMessages(store)
    assert not operator.get(private.id).read
    reply = operator.reply_sysop(private.id, 'Answer', 'answer')
    assert reply.read
    assert not messages.get(a.id, reply.id).read
    assert messages.unread_counts(a.id)['sysop'] == 1
    assert not operator.get(private.id).read
