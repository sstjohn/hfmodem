# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import io
import threading
import time
import uuid
from dataclasses import replace

import pytest

from hfmodem.bbs.config import BBSConfig
from hfmodem.bbs.core import BBSCore
from hfmodem.bbs.frontends.local import run
from hfmodem.bbs.model import Presentation
from hfmodem.bbs.presentation import menu
from hfmodem.bbs.store import Store
from hfmodem.bbs.messages import OperatorMessages
from hfmodem.bbs.config import Limits
from hfmodem.tests.bbs.test_core import Clock, Game, drain, settle


@pytest.fixture
def board(tmp_path):
    clock = Clock()
    store = Store(tmp_path / 'state')
    config = BBSConfig(state_root=store.root, message_areas={'general': 'General', 'radio': 'Radio'})
    core = BBSCore(config, store, games={'game': Game}, clock=clock, utcnow=clock.utc)
    sid = core.open('K1ABC-1')
    initial = drain(core, sid)
    assert initial == menu('main', title=config.title)
    sender = store.player('K2XYZ').id
    yield core, store, sender, sid
    core.shutdown()
    settle(core)
    core.shutdown()
    assert core.quiescent
    store.close()


def command(core, sid, text):
    accepted = core.input(sid, uuid.uuid4().hex, text)
    return accepted, drain(core, sid)


def seed(board, body='Selected body only', area='general', subject='Subject'):
    core, _, sender, _ = board
    return core.message_ui.store.post_public(sender, area, subject, body, uuid.uuid4().hex)


def test_requested_counts_headers_and_selected_body_only(board):
    core, _, _, sid = board
    selected = seed(board)
    other = seed(board, body='Do not dump this other body', subject='Other subject')
    accepted, text = command(core, sid, 'M')
    assert accepted and 'general General: 2 unread' in text and 'sysop: 0 unread' in text
    _, text = command(core, sid, 'AREA general')
    assert selected.subject in text and other.subject in text
    assert selected.body not in text and other.body not in text
    _, text = command(core, sid, f'READ {selected.id}')
    assert selected.body in text and other.body not in text
    assert core.message_ui.store.get(core.session.player.id, selected.id).read
    _, text = command(core, sid, 'M')
    assert 'general General: 1 unread' in text


def test_headers_are_bounded_and_next_is_explicit(board):
    core, _, _, sid = board
    for number in range(12):
        seed(board, subject=f'Header {number}')
    _, text = command(core, sid, 'AREA general')
    assert sum(line.startswith('*') for line in text.splitlines()) == 10
    assert 'NEXT lists more' in text and 'Selected body' not in text
    _, text = command(core, sid, 'NEXT')
    assert sum(line.startswith('*') for line in text.splitlines()) == 2


def test_explicit_post_reply_and_author_receipt(board):
    core, _, _, sid = board
    parent = seed(board)
    command(core, sid, 'AREA general')
    assert command(core, sid, 'POST Hello')[0]
    assert command(core, sid, 'first line')[0]
    assert command(core, sid, '.blank')[0]
    assert command(core, sid, 'last line')[0]
    _, text = command(core, sid, '.send')
    assert text.startswith('Posted #')
    posts = core.message_ui.store.list_public(core.session.player.id, 'general').messages
    mine = next(post for post in posts if post.author_id == core.session.player.id)
    saved = core.message_ui.store.get(core.session.player.id, mine.id)
    assert saved.body == 'first line\n\nlast line' and saved.read
    _, text = command(core, sid, f'REPLY {parent.id}')
    assert 'Draft:' in text
    command(core, sid, 'A reply')
    command(core, sid, '.send')
    reply = core.message_ui.store.list_public(core.session.player.id, 'general').messages[-1]
    assert reply.parent_id == parent.id and reply.thread_id == parent.thread_id


def test_sysop_threads_are_caller_owned_not_operator_privileges(board):
    core, store, _, sid = board
    _, text = command(core, sid, 'SYSOP Question')
    assert 'Draft:' in text
    command(core, sid, 'Please ask the operator')
    command(core, sid, '.send')
    mine = core.message_ui.store.list_sysop(core.session.player.id).messages[0]
    assert mine.scope == 'sysop' and mine.owner_id == core.session.player.id
    command(core, sid, '.send')
    assert len(core.message_ui.store.list_sysop(core.session.player.id).messages) == 1
    other = store.player('W9SSJ').id  # A claimed sysop callsign grants no role.
    assert not core.message_ui.store.list_sysop(other).messages
    core.close(sid)
    replacement = core.open('W9SSJ')
    drain(core, replacement)
    command(core, replacement, 'AREA sysop')
    _, text = command(core, replacement, f'READ {mine.id}')
    assert 'Please ask the operator' not in text


@pytest.mark.parametrize('text', ['Q', 'G', 'BBS EXIT', 'BBS MENU', 'PLAY game', 'AREA radio'])
def test_draft_requires_explicit_cancel_before_navigation(board, text):
    core, _, _, sid = board
    command(core, sid, 'AREA general')
    command(core, sid, 'POST Draft')
    command(core, sid, 'Unsent body')
    accepted, output = command(core, sid, text)
    assert not accepted and '.cancel' in output
    assert core.message_ui.drafting and core.session.state == 'lobby'
    command(core, sid, '.cancel')
    assert not core.message_ui.drafting
    assert core.message_ui.store.list_public(core.session.player.id, 'general').messages == ()


def test_subject_body_unicode_limits_and_replayed_ids(board):
    core, _, _, sid = board
    command(core, sid, 'AREA general')
    assert not command(core, sid, 'POST ' + 'é' * 37)[0]
    assert command(core, sid, 'POST ' + 'é' * 36)[0]
    for _ in range(8):
        assert command(core, sid, 'x' * 250)[0]
    before = core.message_ui.view.draft.body
    assert not command(core, sid, 'x' * 250)[0]
    assert core.message_ui.view.draft.body == before and len(before.encode()) <= 2048
    assert not command(core, sid, '\ud800')[0]
    key = 'send-once'
    assert core.input(sid, key, '.send')
    drain(core, sid)
    assert not core.input(sid, key, '.send')
    assert len(core.message_ui.store.list_public(core.session.player.id, 'general').messages) == 1


def test_ambiguous_commit_retries_same_locked_draft_key_once(board):
    core, _, _, sid = board
    messages = core.message_ui.store
    original, calls = messages.post_public, []

    def ambiguous(*args, **kwargs):
        calls.append(args[-1])
        result = original(*args, **kwargs)
        if len(calls) == 1:
            raise OSError('error after durable commit')
        return result

    messages.post_public = ambiguous
    command(core, sid, 'AREA general')
    command(core, sid, 'POST Retry')
    command(core, sid, 'Only one post')
    _, text = command(core, sid, '.send')
    assert 'not confirmed' in text and core.message_ui.drafting
    assert not command(core, sid, 'change the attempted post')[0]
    _, text = command(core, sid, '.send')
    assert 'Posted #' in text and calls[0] == calls[1]
    assert len(messages.list_public(core.session.player.id, 'general').messages) == 1


def test_late_commit_after_cancel_and_disconnect_never_sends_receipt_or_reposts(board):
    core, _, _, sid = board
    messages = core.message_ui.store
    original = messages.post_public
    started, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    messages.post_public = blocked
    command(core, sid, 'AREA general')
    command(core, sid, 'POST Late')
    command(core, sid, 'Explicitly submitted')
    assert core.input(sid, uuid.uuid4().hex, '.send')
    assert started.wait(1)
    assert core.input(sid, uuid.uuid4().hex, '.cancel')
    event = core.poll(sid)[0]
    assert 'may already exist' in event.text
    core.output_done(sid, event.output_id)
    core.close(sid)
    replacement = core.open('K1ABC-2')
    release.set()
    text = drain(core, replacement)
    assert 'Posted #' not in text and not core.message_ui.drafting
    assert len(messages.list_public(core.session.player.id, 'general').messages) == 1


def test_read_receipt_waits_all_original_pages_repeat_does_not_advance(board):
    core, _, _, sid = board
    item = seed(board, 'Long selected text. ' * 90)
    command(core, sid, 'AREA general')
    core.session.presentation = Presentation(page_bytes=128, width=40)
    core.input(sid, uuid.uuid4().hex, f'READ {item.id}')
    settle(core)
    first = core.poll(sid)[0]
    assert first.more
    core.output_done(sid, first.output_id)
    assert not core.message_ui.store.get(core.session.player.id, item.id).read
    core.input(sid, uuid.uuid4().hex, 'R')
    repeat = core.poll(sid)[0]
    assert repeat.text == first.text
    core.output_done(sid, repeat.output_id)
    assert not core.message_ui.store.get(core.session.player.id, item.id).read
    drain(core, sid)
    assert core.message_ui.store.get(core.session.player.id, item.id).read


def test_skip_disconnect_repeat_leave_unread_and_same_player_reconnect(board):
    core, _, _, sid = board
    item = seed(board, 'Long selected text. ' * 90)
    command(core, sid, 'AREA general')
    core.session.presentation = Presentation(page_bytes=128)
    core.input(sid, uuid.uuid4().hex, f'READ {item.id}')
    settle(core)
    event = core.poll(sid)[0]
    core.output_done(sid, event.output_id)
    command(core, sid, 'Q')  # Pager Stop-reading, not area navigation.
    command(core, sid, 'BBS REPEAT')
    assert not core.message_ui.store.get(core.session.player.id, item.id).read
    core.close(sid)
    replacement = core.open('K1ABC-2')
    drain(core, replacement)
    _, text = command(core, replacement, 'M')
    assert 'general General: 1 unread' in text


@pytest.mark.parametrize('disposition', ['retry-safe', 'complete'])
def test_stale_inflight_body_neither_retries_nor_pauses_new_area(board, disposition):
    core, _, _, sid = board
    item = seed(board, 'OLD PRIVATE DISPLAY ' * 80)
    command(core, sid, 'AREA general')
    core.session.presentation = Presentation(page_bytes=128)
    core.input(sid, uuid.uuid4().hex, f'READ {item.id}')
    settle(core)
    old = core.poll(sid)[0]
    assert old.more
    assert core.input(sid, uuid.uuid4().hex, 'AREA radio')
    settle(core)
    core.output_done(sid, old.output_id, disposition)
    assert not core.session.paused
    text = drain(core, sid)
    assert 'OLD PRIVATE DISPLAY' not in text
    assert not core.message_ui.store.get(core.session.player.id, item.id).read


def test_stale_read_worker_cannot_display_into_new_area(board):
    core, _, _, sid = board
    item = seed(board, 'Old area body')
    command(core, sid, 'AREA general')
    original = core.message_ui.store.get
    started, release = threading.Event(), threading.Event()

    def blocked(*args):
        started.set()
        assert release.wait(2)
        return original(*args)

    core.message_ui.store.get = blocked
    core.input(sid, uuid.uuid4().hex, f'READ {item.id}')
    assert started.wait(1)
    core.input(sid, uuid.uuid4().hex, 'AREA radio')
    release.set()
    text = drain(core, sid)
    assert 'Old area body' not in text and 'radio' in text


def test_navigation_from_paused_body_clears_only_old_message_pager(board):
    core, _, _, sid = board
    item = seed(board, 'Old paused text ' * 90)
    command(core, sid, 'AREA general')
    core.session.presentation = Presentation(page_bytes=128)
    core.input(sid, uuid.uuid4().hex, f'READ {item.id}')
    settle(core)
    event = core.poll(sid)[0]
    core.output_done(sid, event.output_id)
    assert core.session.paused
    core.input(sid, uuid.uuid4().hex, 'AREA radio')
    settle(core)
    assert not core.session.paused
    text = drain(core, sid)
    assert 'Old paused text' not in text and 'radio' in text
    assert not core.message_ui.store.get(core.session.player.id, item.id).read


def test_sysop_operator_reply_is_unread_until_full_caller_body_delivery(board):
    core, store, _, sid = board
    command(core, sid, 'SYSOP Question')
    command(core, sid, 'A caller question')
    command(core, sid, '.send')
    question = core.message_ui.store.list_sysop(core.session.player.id).messages[0]
    operator = OperatorMessages(store, core.config.message_areas)
    reply = operator.reply_sysop(question.id, 'Operator answer', uuid.uuid4().hex)
    _, text = command(core, sid, 'M')
    assert 'sysop: 1 unread' in text
    _, text = command(core, sid, 'AREA sysop')
    assert 'Operator answer' not in text
    assert not core.message_ui.store.get(core.session.player.id, reply.id).read
    _, text = command(core, sid, f'READ {reply.id}')
    assert 'Operator answer' in text
    assert core.message_ui.store.get(core.session.player.id, reply.id).read


def test_hidden_tombstone_excludes_unread_and_cannot_be_replied_to(board):
    core, store, sender, sid = board
    parent = seed(board, 'Hidden original body')
    child = core.message_ui.store.post_public(sender, 'general', 'Child', 'Visible child body',
                                               uuid.uuid4().hex, parent_id=parent.id)
    OperatorMessages(store, core.config.message_areas).hide(parent.id)
    _, text = command(core, sid, 'M')
    assert 'general General: 1 unread' in text
    command(core, sid, 'AREA general')
    _, text = command(core, sid, f'READ {parent.id}')
    assert '[removed]' in text and 'Hidden original body' not in text
    command(core, sid, f'REPLY {parent.id}')
    assert not core.message_ui.drafting
    _, text = command(core, sid, f'READ {child.id}')
    assert 'Visible child body' in text


def test_failed_read_ledger_write_keeps_message_unread(board):
    core, _, _, sid = board
    item = seed(board)
    command(core, sid, 'AREA general')

    def fail(*args):
        raise OSError('full disk')

    core.message_ui.store.mark_read = fail
    _, text = command(core, sid, f'READ {item.id}')
    assert item.body in text and not core.message_ui.store.get(core.session.player.id, item.id).read


def test_unrelated_door_policy_reload_preserves_unsent_draft_and_key(board):
    core, _, _, sid = board
    command(core, sid, 'AREA general')
    command(core, sid, 'POST Kept')
    command(core, sid, 'Still not sent')
    draft = core.message_ui.view.draft
    core.reload_policy(replace(core.config, limits=Limits(900, 3600, 0)))
    assert core.message_ui.view.draft is draft and draft.body == 'Still not sent'
    assert not core.message_ui.store.list_public(core.session.player.id, 'general').messages


def test_message_words_in_game_are_literal_game_input(board):
    core, _, _, sid = board
    command(core, sid, 'PLAY game')
    game = core.session.adapter
    for line in ('M', 'AREA general', 'POST Subject', 'REPLY 1', 'SYSOP Question', '.send'):
        command(core, sid, line)
    assert [text for text, _ in game.submissions] == ['M', 'AREA general', 'POST Subject', 'REPLY 1', 'SYSOP Question', '.send']


def test_local_pipeline_posts_and_eof_discards_unsent_draft(tmp_path):
    store = Store(tmp_path / 'state')
    core = BBSCore(BBSConfig(state_root=store.root), store)
    incoming = io.StringIO('M\nAREA general\nPOST Local\nOne local post\n.send\nPOST Unsent\nNot posted\n')
    output = io.StringIO()
    try:
        assert run(core, 'K1ABC', incoming=incoming, outgoing=output) == 0
        player = store.player('K1ABC').id
        posts = core.message_ui.store.list_public(player, 'general').messages
        assert len(posts) == 1 and posts[0].subject == 'Local'
        assert 'Posted #' in output.getvalue() and not core.message_ui.drafting
    finally:
        store.close()


@pytest.fixture
def radio_board(tmp_path):
    from hfmodem.tests.bbs.test_varac_entrance import board as radio_fixture
    yield from radio_fixture.__wrapped__(tmp_path)


def test_actual_varac_message_parser_post_reply_unread_and_game_ownership(radio_board):
    from hfmodem.tests.bbs.test_varac_entrance import drive, enter, sent
    d, core, _, _ = radio_board
    enter(radio_board)
    for line in ('M', 'AREA general', 'POST Radio', 'One complete VarAC message', '.send'):
        d.receive(line)
        drive(d, core)
    posts = core.message_ui.store.list_public(core.session.player.id, 'general').messages
    assert len(posts) == 1
    for line in (f'REPLY {posts[0].id}', 'A reply over framed messages', '.send'):
        d.receive(line)
        drive(d, core)
    assert len(core.message_ui.store.list_public(core.session.player.id, 'general').messages) == 2
    d.receive('M')
    drive(d, core)
    assert any('general General: 0 unread' in text for text in sent(d))
    d.receive('Q')
    drive(d, core)
    d.receive('PLAY adventure')
    drive(d, core)
    game = core.session.adapter
    d.receive('SYSOP is game text')
    drive(d, core)
    assert game.submissions[-1] == ('SYSOP is game text', 'default')


def test_varac_partial_read_skip_full_receipt_and_explicit_draft_cancel(radio_board):
    from hfmodem.tests.bbs.test_varac_entrance import drive, enter
    d, core, _, _ = radio_board
    author = core.store.player('K2XYZ').id
    item = core.message_ui.store.post_public(author, 'general', 'Paged radio message',
                                              'Radio body text. ' * 110, uuid.uuid4().hex)
    enter(radio_board)
    d.receive('AREA general')
    drive(d, core)
    d.receive(f'READ {item.id}')
    for _ in range(200):
        d.tick()
        if core.session.paused:
            break
        time.sleep(.001)
    assert core.session.paused
    assert not core.message_ui.store.get(core.session.player.id, item.id).read
    d.receive('Q')
    drive(d, core)
    assert not core.message_ui.store.get(core.session.player.id, item.id).read
    d.receive(f'READ {item.id}')
    drive(d, core)
    assert core.message_ui.store.get(core.session.player.id, item.id).read
    for line in ('POST Unsent radio draft', 'Do not publish', '.cancel'):
        d.receive(line)
        drive(d, core)
    assert len(core.message_ui.store.list_public(core.session.player.id, 'general').messages) == 1


def test_varac_multiline_draft_submission_cannot_execute_send(radio_board):
    from hfmodem.tests.bbs.test_varac_entrance import drive, enter
    d, core, _, _ = radio_board
    enter(radio_board)
    for line in ('AREA general', 'POST One command'):
        d.receive(line)
        drive(d, core)
    d.receive('body\n.send')
    drive(d, core)
    assert core.message_ui.drafting and core.message_ui.view.draft.body == ''
    assert not core.message_ui.store.list_public(core.session.player.id, 'general').messages
