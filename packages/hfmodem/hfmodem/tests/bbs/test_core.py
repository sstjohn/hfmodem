# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import time

import pytest

from hfmodem.bbs.config import BBSConfig, Limits
from hfmodem.bbs.core import BBSCore
from hfmodem.bbs.games.base import GameEvent
from hfmodem.bbs.model import BusyError
from hfmodem.bbs.policy import ConnectionLease
from hfmodem.bbs.store import Store
from hfmodem.contact.bbs import HostCatalog


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def utc(self):
        return datetime(2026, 10, 26, tzinfo=timezone.utc) + timedelta(seconds=self.now)


class Game:
    capabilities = frozenset({'structured_prompts'})

    def __init__(self):
        self.events = []
        self.submissions = []
        self.closed = False

    def start(self, player, directory):
        self.player, self.directory = player, directory
        self.events += [GameEvent('text', 'A room.\n'), GameEvent('prompt', '> ', 'line')]

    def poll(self):
        result, self.events = self.events, []
        return result

    def submit(self, text, mode='line'):
        self.submissions.append((text, mode))
        self.events += [GameEvent('text', 'You moved.\n'), GameEvent('prompt', '> ', 'line')]
        return True

    def close(self, reason):
        self.closed = True
        self.events.append(GameEvent('exited'))


def settle(core):
    for _ in range(300):
        core.tick()
        if not core._jobs:
            return
        time.sleep(0.001)
    pytest.fail('Persistence worker did not complete')


def drain(core, sid):
    output = []
    for _ in range(300):
        settle(core)
        if core.session.paused:
            core.input(sid, f'more-{len(core.session.input_ids)}', 'BBS MORE')
        events = core.poll(sid)
        if not events:
            return '\n'.join(output)
        event = events[0]
        output.append(event.text)
        core.output_done(sid, event.output_id)
    pytest.fail('Display did not drain')


@pytest.fixture
def board(tmp_path):
    clock = Clock()
    config = BBSConfig(state_root=tmp_path, title='Test Board')
    store = Store(tmp_path)
    # These scenarios exercise the explicitly selected classic experience.
    store.set_preferences(store.player('K1ABC').id, expert=False)
    made = []

    def factory():
        game = Game()
        made.append(game)
        return game

    core = BBSCore(config, store, games={'adventure': factory}, clock=clock, utcnow=clock.utc)
    yield core, clock, store, made
    core.shutdown()
    settle(core)
    core.shutdown()
    store.close()


def enter(board):
    core, *_ = board
    sid = core.open('K1ABC-1')
    assert 'Test Board' in drain(core, sid)
    return sid


def launch(board, sid):
    core, *_ = board
    core.input(sid, 'play', 'PLAY adventure')
    assert 'A room.' in drain(core, sid)
    assert core.session.state == 'playing'


def test_classic_menus_back_and_no_accidental_disconnect(board):
    core, *_ = board
    sid = enter(board)
    core.input(sid, 'q-main', 'Q')
    assert 'G says goodbye' in drain(core, sid)
    core.input(sid, 'df', 'DF')
    assert core.session.view == 'main'
    drain(core, sid)
    core.input(sid, 'doors', 'D')
    assert '[1] adventure' in drain(core, sid)
    core.input(sid, 'back', 'Q')
    assert 'Main:' in drain(core, sid)
    core.input(sid, 'goodbye', 'G')
    assert core.poll(sid)[0].kind == 'close'


def test_one_page_in_flight_retry_and_stale_completion(board):
    core, *_ = board
    sid = core.open('K1ABC')
    settle(core)
    first = core.poll(sid)[0]
    assert core.poll(sid) == []
    assert not core.output_done(sid, 'wrong')
    core.output_done(sid, first.output_id, 'retry-safe')
    retry = core.poll(sid)[0]
    assert retry.text == first.text and retry.output_id != first.output_id
    core.close(sid)
    replacement = core.open('K1ABC')
    assert replacement != sid
    assert not core.output_done(sid, retry.output_id)


def test_replay_ids_not_identical_game_text(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    assert core.input(sid, 'turn1', 'north')
    assert not core.input(sid, 'turn1', 'north')
    drain(core, sid)
    core.input(sid, 'turn2', 'north')
    drain(core, sid)
    assert games[0].submissions == [('north', 'default'), ('north', 'default')]
    assert not core.input(sid, 'script', 'north\nsouth')
    assert len(games[0].submissions) == 2


def test_game_letters_and_reserved_literal_belong_to_game(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    for n, text in enumerate(('Q', 'G', '?', 'BBS SEND BBS MENU', 'BBS ENTER', 'BBS KEY F')):
        core.input(sid, f'cmd{n}', text)
        drain(core, sid)
    assert games[0].submissions == [('Q', 'default'), ('G', 'default'), ('?', 'default'),
                                     ('BBS MENU', 'line'), ('', 'pause'), ('F', 'key')]


def test_paginate_repeat_skip_never_submits_game_input(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    games[0].events = [GameEvent('text', 'long output ' * 300), GameEvent('prompt', '> ', 'line')]
    first = core.poll(sid)[0]
    core.output_done(sid, first.output_id)
    assert core.session.paused
    core.input(sid, 'repeat', 'R')
    repeated = core.poll(sid)[0]
    assert repeated.text == first.text
    core.output_done(sid, repeated.output_id)
    core.input(sid, 'skip', 'Q')
    assert '> ' in drain(core, sid)
    assert games[0].submissions == []


def test_graceful_cancel_preserves_output_then_confirm_cleanup(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    core._display('unread clue ' * 100)
    saved = tuple(core.session.pending)
    core.input(sid, 'menu1', 'BBS MENU')
    assert core.session.state == 'leaving' and not games[0].closed
    drain(core, sid)
    core.input(sid, 'cancel', 'BBS CANCEL')
    assert core.session.state == 'playing'
    assert tuple(core.session.pending) == saved
    drain(core, sid)
    core.input(sid, 'menu2', 'BBS MENU')
    drain(core, sid)
    core.input(sid, 'confirm', 'BBS CONFIRM')
    assert 'Doors' in drain(core, sid)
    assert games[0].closed and core.session.state == 'lobby'


def test_failed_checkpoint_offers_cancel_without_success(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    game = games[0]
    game.capabilities |= {'checkpoint'}
    game.checkpoint = lambda: True
    core.input(sid, 'menu', 'BBS MENU')
    assert core.session.waiting_save
    game.events.append(GameEvent('error', 'secret local path'))
    text = drain(core, sid)
    assert 'Save failed' in text and 'secret local path' not in text
    core.input(sid, 'cancel', 'BBS CANCEL')
    assert core.session.state == 'playing' and not game.closed


def test_forced_close_late_game_output_fenced_and_busy_until_reaped(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    game = games[0]
    game.close = lambda reason: setattr(game, 'closed', True)
    core.close(sid, 'lost link')
    game.events.append(GameEvent('text', 'must never appear'))
    with pytest.raises(BusyError):
        core.open('K2XYZ')
    game.events.append(GameEvent('exited'))
    replacement = core.open('K2XYZ')
    assert 'must never appear' not in drain(core, replacement)


def test_zero_quota_denied_before_factory_and_policy_reload_fences(board):
    core, _, _, games = board
    sid = enter(board)
    core.reload_policy(replace(core.config, limits=Limits(900, 3600, 0)))
    core.input(sid, 'play-zero', 'PLAY adventure')
    assert 'allowance exhausted' in drain(core, sid)
    assert games == []
    core.reload_policy(replace(core.config, limits=Limits()))
    core.input(sid, 'play-now', 'PLAY adventure')
    drain(core, sid)
    core.reload_policy(replace(core.config, limits=Limits(900, 3600, 0)))
    assert games[0].closed
    assert core.session.run_id is None


def test_paid_interval_exhaustion_and_persistent_spent_time(board):
    core, clock, store, games = board
    sid = enter(board)
    core.reload_policy(replace(core.config, limits=Limits(900, 3600, 2)))
    launch(board, sid)
    clock.now = 1
    settle(core)  # cannot prepay beyond daily allowance; paid second remains
    assert not games[0].closed
    clock.now = 2
    core.tick()
    assert games[0].closed
    drain(core, sid)
    core.close(sid)
    settle(core)
    other = core.open('K1ABC-2')
    drain(core, other)
    core.input(other, 'again', 'PLAY adventure')
    assert 'allowance exhausted' in drain(core, other)
    assert len(games) == 1
    assert store.player('K1ABC').id == core.session.player.id


def test_original_connection_deadline_survives_exit_and_reentry(board):
    core, clock, *_ = board
    connection = ConnectionLease.start(Limits(10, 20, 100), 0)
    clock.now = 5
    sid = core.open('K1ABC', connection=connection)
    drain(core, sid)
    core.input(sid, 'exit', 'BBS EXIT')
    assert core.poll(sid)[0].kind == 'exit'
    clock.now = 19
    new = core.open('K1ABC', connection=connection)
    drain(core, new)
    clock.now = 20
    assert core.poll(new)[0].kind == 'close'
    assert core.session.state == 'closed'


def test_ansi_plain_ack_and_preference_cannot_enable_color(board):
    core, _, store, _ = board
    player = store.player('K1ABC')
    store.set_preferences(player.id, color='ansi16')
    sid = enter(board)
    assert core.session.presentation.mode == 'plain'
    core.input(sid, 'term', 'BBS TERM ANSI16 80 24')
    ack = core.poll(sid)[0]
    assert ack.text == 'BBS TERM OK ANSI16 80 24' and '\x1b' not in ack.text
    core.output_done(sid, ack.output_id)
    core._display('\x1b[31mcolor')
    colored = core.poll(sid)[0]
    assert '\x1b[' in colored.text and colored.text.endswith('\x1b[0m')


def test_stale_numbered_door_and_catalog_snapshot(board, tmp_path):
    core, *_ = board
    sid = enter(board)
    core.input(sid, 'doors', 'D')
    drain(core, sid)
    core.games = {'different': Game}
    core.input(sid, 'number', '1')
    assert 'menu changed' in drain(core, sid)
    exports = tmp_path / 'exports'
    exports.mkdir()
    note = exports / 'news.txt'
    note.write_text('Original bulletin')
    catalog = HostCatalog()
    catalog.enable(exports)
    core.catalog = catalog
    try:
        core.input(sid, 'files', 'FILES')
        assert 'news.txt' in drain(core, sid)
        note.write_text('Changed bulletin')
        core.input(sid, 'read', 'READ 1')
        assert 'operation failed' in drain(core, sid)
    finally:
        catalog.close()


def test_supported_binary_get_handoff_and_refusal_returns_same_lobby(board, tmp_path):
    core, *_ = board
    sid = enter(board)
    exports = tmp_path / 'exports'
    exports.mkdir()
    (exports / 'data.bin').write_bytes(b'\0\xff\x01')
    catalog = HostCatalog()
    catalog.enable(exports)
    core.catalog, core.transfer_supported = catalog, True
    try:
        core.input(sid, 'list', 'FILES')
        drain(core, sid)
        core.input(sid, 'get', 'GET 1')
        settle(core)
        transfer = core.poll(sid)[0]
        assert transfer.kind == 'transfer' and transfer.payload.data == b'\0\xff\x01'
        assert core.session.state == 'transfer'
        core.input(sid, 'overlap', 'PLAY adventure')
        assert core.session.adapter is None
        core.output_done(sid, transfer.output_id, 'refused')
        assert core.session.state == 'lobby'
        assert 'Files' in drain(core, sid)
    finally:
        catalog.close()


def test_door_revocation_with_unsettled_output_forces_attachment_close(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    core._display('currently transmitting')
    inflight = core.poll(sid)[0]
    core.reload_policy(replace(core.config, limits=Limits(900, 3600, 0)))
    assert core.session.state == 'closed' and games[0].closed
    assert not core.output_done(sid, inflight.output_id)
    assert core.poll(sid)[0].kind == 'close'


@pytest.mark.parametrize('action', ['DF', 'D extra', 'PLAY missing', 'READ 99',
                                   'BBS FLY', 'BBS STATUS extra', 'T extra', 'TIME extra', 'STATUS extra',
                                   'BBS TERM ANSI16 1 1', 'BBS MORE'])
def test_irrelevant_rejected_input_and_zero_duration_notice_do_not_reset_idle(board, action):
    core, clock, *_ = board
    sid = enter(board)
    core.session.connection.limits = Limits(5, 3600, 3600)
    baseline = core.session.connection.last_activity
    clock.now = 4
    assert not core.input(sid, 'irrelevant', action)
    assert core.session.connection.last_activity == baseline
    drain(core, sid)
    assert core.session.connection.last_activity == baseline
    clock.now = 5
    core.tick()
    assert core.session.state == 'closed'


def test_valid_menu_and_successful_game_input_reset_idle(board):
    core, clock, _, games = board
    sid = enter(board)
    core.session.connection.limits = Limits(5, 3600, 3600)
    clock.now = 4
    assert core.input(sid, 'help', '?')
    drain(core, sid)
    clock.now = 5
    core.tick()
    assert core.session.state == 'lobby'
    launch(board, sid)
    clock.now = 8
    assert core.input(sid, 'north', 'north')
    assert games[0].submissions == [('north', 'default')]
    assert core.session.connection.last_activity == 8


def test_rejected_game_submission_does_not_reset_idle(board):
    core, clock, _, games = board
    sid = enter(board)
    launch(board, sid)
    core.session.connection.limits = Limits(5, 3600, 3600)
    baseline = core.session.connection.last_activity
    games[0].submit = lambda text, mode: False
    clock.now = 4
    assert not core.input(sid, 'rejected', 'north')
    drain(core, sid)
    assert core.session.connection.last_activity == baseline
    clock.now = 5
    core.tick()
    assert core.session.state == 'closed'


def test_managed_restore_has_fresh_response_timer_and_backend_generation(board):
    core, clock, _, games = board
    clock.now = 100
    sid = enter(board)
    launch(board, sid)
    game = games[0]
    game.capabilities |= {'restore'}
    game.restore = lambda: True
    old_run, occupancy = core.session.run_id, core.session.usage.id
    assert core.input(sid, 'restore', 'BBS RESTORE')
    assert core.session.turn_started == 100 and core.session.run_id != old_run
    assert core.session.usage.id == occupancy
    clock.now = 101
    core.tick()
    assert core.session.state == 'playing' and not game.closed
