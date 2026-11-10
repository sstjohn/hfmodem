# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
import threading

from hfmodem.bbs.config import Limits
from hfmodem.bbs.games.base import GameEvent
from hfmodem.bbs.model import Presentation
from hfmodem.bbs.model import BusyError
import pytest
from hfmodem.tests.bbs import test_core
from hfmodem.tests.bbs.test_core import enter, launch, drain, settle

board = test_core.board


def test_waiting_at_more_is_idle_time(board):
    core, clock, *_ = board
    core.config = replace(core.config, limits=Limits(5, 100, 100))
    sid = enter(board)
    core.session.presentation = Presentation(height=8)
    core._display('\n'.join(str(n) for n in range(30)))
    event = core.poll(sid)[0]
    core.output_done(sid, event.output_id)
    assert core.session.paused and not core.session.connection.busy
    clock.now = 6
    core.tick()
    assert core.session.state == 'closed'


def test_native_transfer_owns_lobby_until_receipt(board):
    core, *_ = board
    sid = enter(board)
    output_id = core.claim_transfer(sid)
    assert output_id
    core.input(sid, 'menu-during-file', 'BBS MENU')
    core.input(sid, 'play-during-file', 'PLAY adventure')
    assert core.session.state == 'transfer' and core.session.adapter is None
    assert core.claim_transfer(sid) is None
    core.output_done(sid, output_id, 'complete')
    assert core.session.state == 'lobby'


def test_close_during_transfer_fences_late_receipt(board):
    core, *_ = board
    sid = enter(board)
    output_id = core.claim_transfer(sid)
    core.input(sid, 'exit-file', 'BBS EXIT')
    assert core.session.state == 'closed'
    assert not core.output_done(sid, output_id)


def test_policy_reload_during_admission_returns_to_lobby(board, monkeypatch):
    core, *_ = board
    sid = enter(board)
    entered, release = threading.Event(), threading.Event()
    original = core.quota.admit
    def gated(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(core.quota, 'admit', gated)
    try:
        core.input(sid, 'play-race', 'PLAY adventure')
        assert entered.wait(1)
        core.reload_policy(replace(core.config, limits=Limits(900, 3600, 0)))
    finally:
        release.set()
    settle(core)
    assert core.session.state == 'lobby'
    assert core.session.adapter is None


def test_hung_leave_save_has_bounded_confirmation_wait(board):
    core, clock, _, games = board
    sid = enter(board)
    launch(board, sid)
    game = games[0]
    game.capabilities |= {'checkpoint'}
    game.checkpoint = lambda: True
    core.input(sid, 'menu-save', 'BBS MENU')
    drain(core, sid)
    assert core.session.waiting_save
    clock.now = 5
    core.tick()
    assert not core.session.waiting_save
    assert 'unconfirmed' in drain(core, sid)
    core.input(sid, 'cancel-save', 'BBS CANCEL')
    assert core.session.state == 'playing'
    assert not core.session.ready


def test_saved_and_exited_same_batch_cannot_strand_cleanup(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    game = games[0]
    game.capabilities |= {'checkpoint'}
    game.checkpoint = lambda: True
    core.input(sid, 'menu-save', 'BBS MENU')
    drain(core, sid)
    game.events += [GameEvent('saved'), GameEvent('exited')]
    core.tick()
    settle(core)
    assert not core._cleanup
    assert core.session.state == 'lobby'


def test_cancel_leave_preserves_arriving_output_and_readiness(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    game = games[0]
    core.session.ready = False
    core.input(sid, 'leave-during-turn', 'BBS MENU')
    drain(core, sid)
    game.events += [GameEvent('text', 'An important clue.\n'), GameEvent('prompt', '> ', 'line')]
    core.tick()
    assert not core.session.pending
    core.input(sid, 'cancel-after-turn', 'BBS CANCEL')
    assert core.session.ready
    assert 'An important clue.' in drain(core, sid)


def test_uncertain_backend_cleanup_retains_service_fence(board):
    core, _, _, games = board
    sid = enter(board)
    launch(board, sid)
    games[0].events += [GameEvent('exited', outcome='cleanup_uncertain')]
    core.tick()
    assert core.session.state == 'closed'
    assert not core.quiescent
    with pytest.raises(BusyError):
        core.open('K2ABC')
    # Fake backend provides later confirmation solely to clean the test fixture.
    core._uncertain_cleanup = False
    core.tick()


def test_pending_preference_work_cannot_complete_into_a_game(board, monkeypatch):
    core, _, store, _ = board
    sid = enter(board)
    core.input(sid, 'preferences', 'P')
    drain(core, sid)
    entered, release = threading.Event(), threading.Event()
    original = store.set_preferences
    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(store, 'set_preferences', gated)
    try:
        core.input(sid, 'expert', 'E')
        assert entered.wait(1)
        core.input(sid, 'overlap-preferences', 'PLAY adventure')
        assert core.session.state == 'lobby'
        assert core.session.adapter is None
    finally:
        release.set()
    assert 'Preferences saved.' in drain(core, sid)
    assert core.session.prefs['expert']
