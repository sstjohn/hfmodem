# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Managed selection is a menu transaction, not unmetered backend occupancy."""
from dataclasses import replace
from pathlib import Path
import threading

import pytest

from hfmodem.bbs.config import BBSConfig, Limits
from hfmodem.bbs.core import BBSCore
from hfmodem.bbs.games.base import GameEvent
from hfmodem.bbs.games.registry import ZorkFactory
from hfmodem.bbs.games.zmachine import Checkpoints, PROFILE
from hfmodem.bbs.store import Store
from hfmodem.tests.bbs.test_core import Clock, Game, drain, settle
from hfmodem.tests.bbs.test_games import quetzal


class ResumeGame(Game):
    capabilities = frozenset({'structured_prompts', 'restore'})

    def __init__(self, factory):
        super().__init__()
        self.factory, self.restore_calls = factory, 0

    def start(self, player, directory):
        self.player, self.directory = player, directory
        self.events += [GameEvent('text', 'FRESH SCENE MUST NOT LEAK\n'),
                        GameEvent('prompt', '> ', self.factory.start_mode)]

    def restore(self):
        self.restore_calls += 1
        if self.factory.restore_events is not None:
            self.events.extend(self.factory.restore_events)
        return self.factory.restore_accepted


class Factory:
    available = True

    def __init__(self):
        self.saved = True
        self.probe_started = threading.Event()
        self.probe_gate = None
        self.probe_thread = None
        self.made = []
        self.start_mode = 'line'
        self.restore_accepted = True
        self.restore_events = [GameEvent('text', 'RESTORED ROOM\n'), GameEvent('restored'),
                               GameEvent('prompt', '> ', 'line')]
        self.make_game = lambda: ResumeGame(self)

    def saved_game(self, directory):
        self.probe_directory, self.probe_thread = directory, threading.get_ident()
        self.probe_started.set()
        if self.probe_gate is not None and not self.probe_gate.wait(2):
            raise RuntimeError('test probe was not released')
        if isinstance(self.saved, Exception):
            raise self.saved
        return self.saved

    def __call__(self):
        game = self.make_game()
        self.made.append(game)
        return game


@pytest.fixture
def board(tmp_path):
    clock = Clock()
    config = BBSConfig(state_root=tmp_path / 'bbs')
    store = Store(config.state_root)
    factory = Factory()
    core = BBSCore(config, store, games={'story': factory}, clock=clock, utcnow=clock.utc)
    sid = core.open('K1ABC-1')
    drain(core, sid)
    yield core, clock, store, factory, sid
    if factory.probe_gate:
        factory.probe_gate.set()
    core.shutdown()
    settle(core)
    core.shutdown()
    assert core.quiescent
    store.close()


def choose(board):
    core, _, _, factory, sid = board
    assert core.input(sid, 'select', 'PLAY story')
    text = drain(core, sid)
    assert text == 'Saved game. R resume; N new; Q back.'
    assert core.session.state == 'choosing' and core.session.ready
    assert factory.made == []
    return text


def test_probe_runs_off_owner_and_choice_uses_no_door_allowance(board):
    core, _, _, factory, sid = board
    choose(board)
    assert factory.probe_thread != threading.get_ident()
    assert factory.probe_directory == core.store.root / 'games/story/players' / core.session.player.id
    status = core.quota.status(core.session.player.id, 'K1ABC')
    assert status.used_seconds == status.reserved_seconds == 0
    assert core.input(sid, 'back', 'Q')
    drain(core, sid)
    assert core.session.state == 'lobby' and core.session.view == 'doors'
    assert factory.made == []


def test_no_save_and_generic_factories_launch_without_choice(board):
    core, _, _, factory, sid = board
    factory.saved = False
    assert core.input(sid, 'select', 'PLAY story')
    text = drain(core, sid)
    assert 'FRESH SCENE' in text and 'Saved game.' not in text
    assert core.session.state == 'playing' and factory.made[0].restore_calls == 0


def test_resume_waits_actual_prompt_and_restore_confirmation_once(board):
    core, _, _, factory, sid = board
    choose(board)
    factory.restore_events = None
    assert core.input(sid, 'resume', 'R')
    assert drain(core, sid) == ''
    game = factory.made[0]
    assert game.restore_calls == 1 and core.session.state == 'starting'
    assert core.session.resume_phase == 'confirming' and not core.session.ready
    assert not core.input(sid, 'too-early', 'north')
    drain(core, sid)
    game.events = [GameEvent('text', 'RESTORED ROOM\n'), GameEvent('restored')]
    assert drain(core, sid) == ''
    assert core.session.resume_phase == 'boundary' and not core.session.ready
    assert game.submissions == []
    game.events = [GameEvent('restored'), GameEvent('prompt', '> ', 'line')]
    text = drain(core, sid)
    assert 'Checkpoint restored.' in text and 'RESTORED ROOM' in text
    assert 'FRESH SCENE' not in text
    assert game.restore_calls == 1
    assert core.session.state == 'playing' and core.session.ready
    assert core.session.managed_operation is core.session.resume_phase is None
    assert core.input(sid, 'resume-again', 'R')  # R is ordinary game text now.
    drain(core, sid)
    assert game.restore_calls == 1 and game.submissions == [('R', 'default')]


def test_new_preserves_existing_checkpoint_bytes_and_never_restores(board):
    core, _, _, factory, sid = board
    choose(board)
    checkpoints = factory.probe_directory / 'checkpoints'
    checkpoints.mkdir(parents=True)
    manifest = checkpoints / 'manifest.json'
    payload = checkpoints / 'previous-payload.qzl'
    manifest.write_bytes(b'previous manifest')
    payload.write_bytes(b'confirmed save generation')
    assert core.input(sid, 'new', 'N')
    assert 'FRESH SCENE' in drain(core, sid)
    assert manifest.read_bytes() == b'previous manifest'
    assert payload.read_bytes() == b'confirmed save generation'
    assert factory.made[0].restore_calls == 0


@pytest.mark.parametrize('action', ['Q', 'BBS MENU', 'BBS EXIT'])
def test_cancelled_probe_never_applies_to_current_attachment(board, action):
    core, _, _, factory, sid = board
    factory.probe_gate = threading.Event()
    assert core.input(sid, 'select', 'PLAY story')
    assert factory.probe_started.wait(1)
    assert core.input(sid, 'cancel', action)
    factory.probe_gate.set()
    drain(core, sid)
    assert factory.made == []
    assert core.session.state in ('lobby', 'closed') and core.session.run_id is None


def test_disconnect_and_same_player_reconnect_fence_probe_completion(board):
    core, _, _, factory, sid = board
    factory.probe_gate = threading.Event()
    core.input(sid, 'select', 'PLAY story')
    assert factory.probe_started.wait(1)
    old_player = core.session.player.id
    core.close(sid, 'lost connection')
    replacement = core.open('K1ABC-2')
    factory.probe_gate.set()
    text = drain(core, replacement)
    assert core.session.player.id == old_player and replacement != sid
    assert 'Saved game' not in text and factory.made == []


@pytest.mark.parametrize('during_probe', [False, True])
def test_zero_reload_invalidates_probe_or_choice_and_never_creates_backend(board, during_probe):
    core, _, _, factory, sid = board
    if during_probe:
        factory.probe_gate = threading.Event()
        core.input(sid, 'select', 'PLAY story')
        assert factory.probe_started.wait(1)
    else:
        choose(board)
    core.reload_policy(replace(core.config, limits=Limits(900, 3600, 0)))
    if factory.probe_gate:
        factory.probe_gate.set()
    drain(core, sid)
    assert core.session.state == 'lobby' and core.session.run_id is None
    assert factory.made == []
    core.input(sid, 'select-again', 'PLAY story')
    drain(core, sid)
    assert core.session.state == 'choosing'
    assert core.input(sid, 'resume-zero', 'R')
    assert 'allowance exhausted' in drain(core, sid)
    assert factory.made == []


def test_changed_factory_at_choice_cannot_launch_replacement(board):
    core, _, _, factory, sid = board
    choose(board)
    replacement = Factory()
    core.games['story'] = replacement
    assert not core.input(sid, 'resume', 'R')
    drain(core, sid)
    assert factory.made == replacement.made == []
    assert core.session.state == 'lobby'


@pytest.mark.parametrize('hard', [False, True])
def test_choice_wait_counts_idle_and_connection_deadline_remains_authoritative(board, hard):
    core, clock, _, factory, sid = board
    choose(board)
    core.session.connection.limits = Limits(5, 6 if hard else 100, 3600)
    assert not core.session.connection.busy
    if hard:
        clock.now = 4
        assert core.input(sid, 'help', '?')
        drain(core, sid)
        assert not core.session.connection.idle_expired(6)
    clock.now = 6 if hard else 5
    core.tick()
    assert core.session.state == 'closed' and factory.made == []


def test_short_resume_confirmation_and_room_share_one_page(board):
    core, _, _, _, sid = board
    choose(board)
    core.input(sid, 'resume', 'R')
    settle(core)
    event = core.poll(sid)[0]
    assert event.text == 'Checkpoint restored.\nRESTORED ROOM\n> '
    assert not event.more and not core.session.pending
    core.output_done(sid, event.output_id)
    assert core.poll(sid) == []


@pytest.mark.parametrize('failure', ['reject', 'error', 'unconfirmed', 'exited', 'key'])
def test_resume_failures_suppress_fresh_scene_and_cleanup_before_doors(board, failure):
    core, _, _, factory, sid = board
    choose(board)
    if failure == 'reject':
        factory.restore_accepted = False
    elif failure == 'error':
        factory.restore_events = [GameEvent('error', '/private/secret runtime path'), GameEvent('exited')]
    elif failure == 'unconfirmed':
        factory.restore_events = [GameEvent('prompt', '> ', 'line')]
    elif failure == 'exited':
        factory.restore_events = [GameEvent('exited')]
    elif failure == 'key':
        factory.start_mode = 'key'
    assert core.input(sid, 'resume', 'R')
    text = drain(core, sid)
    assert 'Saved game retained.' in text and 'FRESH SCENE' not in text and '/private/secret' not in text
    assert factory.made[0].closed
    assert core.session.state == 'lobby' and core.session.managed_operation is None


def test_resume_timeout_and_following_new_game_error_do_not_inherit_restore_operation(board):
    core, clock, _, factory, sid = board
    choose(board)
    factory.restore_events = None
    core.input(sid, 'resume', 'R')
    drain(core, sid)
    clock.now = 5
    settle(core)  # Durably renew so the startup bound, not unpaid time, expires.
    clock.now = 10.1
    assert 'Resume timed out. Saved game retained.' in drain(core, sid)
    assert core.session.managed_operation is None and core.session.state == 'lobby'
    core.input(sid, 'again', 'PLAY story')
    drain(core, sid)
    core.input(sid, 'new', 'N')
    drain(core, sid)
    factory.made[-1].events = [GameEvent('error')]
    assert 'Game failed' in drain(core, sid)
    assert factory.made[-1].closed and core.session.state == 'lobby'


def test_factory_without_managed_capabilities_cannot_leave_unstarted_cleanup_ticket(board):
    core, _, _, factory, sid = board
    choose(board)
    factory.make_game = Game
    core.input(sid, 'resume', 'R')
    assert 'Resume failed. Saved game retained.' in drain(core, sid)
    assert core.session.state == 'lobby' and core._cleanup == []


def test_corrupt_probe_explains_new_back_without_auto_launch(board):
    core, _, _, factory, sid = board
    factory.saved = ValueError('incompatible private/story/path')
    core.input(sid, 'select', 'PLAY story')
    assert drain(core, sid) == 'Save unavailable. N new; Q back.'
    assert not core.input(sid, 'resume', 'R')
    drain(core, sid)
    assert factory.made == []
    core.input(sid, 'new', 'N')
    assert 'FRESH SCENE' in drain(core, sid)


def test_zork_probe_distinguishes_missing_valid_corrupt_and_incompatible_without_writes(tmp_path):
    factory = ZorkFactory(Path('/unused/dfrotz'), Path('/unused/zork.z3'), 'r' * 64, 's' * 64)
    directory = tmp_path / 'player'
    assert not factory.saved_game(directory) and not directory.exists()
    directory.mkdir()
    source = directory / 'source.qzl'
    source.write_bytes(quetzal())
    manager = Checkpoints(directory, {'profile': PROFILE, 'runtime_sha256': 'r' * 64, 'story_sha256': 's' * 64})
    generation = manager.publish(source)
    manifest = (manager.root / 'manifest.json').read_bytes()
    assert factory.saved_game(directory)
    assert (manager.root / 'manifest.json').read_bytes() == manifest
    incompatible = replace(factory, story_sha256='changed')
    with pytest.raises(ValueError, match='another runtime or story'):
        incompatible.saved_game(directory)
    payload = manager.root / generation / 'payload.qzl'
    payload.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        factory.saved_game(directory)
    assert payload.read_bytes() == b'corrupt' and (manager.root / 'manifest.json').read_bytes() == manifest
