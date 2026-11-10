# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Single caller, transport-independent BBS with fenced worker completions.

Public operations only schedule blocking store/catalog work. ``tick`` applies
completed work on the owner thread; no future is waited on by a radio caller.
Adapters themselves implement the nonblocking games/base.py contract.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
import time
import uuid

from .model import BusyError, OutputEvent, Presentation
from .policy import ConnectionLease, QuotaManager
from .presentation import StyleParser, duration, menu, pages


def _token():
    return uuid.uuid4().hex


@dataclass
class _Session:
    id: str
    peer: str
    callsign: str
    mode: str
    presentation: Presentation
    connection: object
    player: object = None
    prefs: dict = field(default_factory=dict)
    state: str = 'loading'
    view: str = 'main'
    run_id: str | None = None
    turn_id: str | None = None
    adapter: object = None
    usage: object = None
    ready: bool = False
    prompt: str = ''
    input_mode: str = 'unknown'
    managed_operation: str | None = None
    pending: deque = field(default_factory=deque)
    outstanding: OutputEvent | None = None
    last_page: OutputEvent | None = None
    paused: bool = False
    input_ids: set = field(default_factory=set)
    game_snapshot: tuple = ()
    file_snapshot: tuple = ()
    file_generation: int = -1
    parser: StyleParser = field(default_factory=StyleParser)
    leave_saved: object = None
    leave_target: str = 'doors'
    waiting_save: bool = False
    started: float = 0
    turn_started: float = 0
    warnings: set = field(default_factory=set)
    renewal_pending: bool = False
    renewal_failed: bool = False
    admitted_revision: int = 0
    deadline_prepared: bool = False
    selection_factory: object = None
    resume_available: bool = False
    resume_phase: str | None = None
    resume_text: object = None
    resume_bytes: int = 0
    message_view: object = None


class BBSCore:
    """One application owner. ``games`` maps display IDs to adapter factories.

    A factory may have ``title`` and ``available`` attributes; ``available`` can
    be a boolean or callable. Optional ``saved_game(directory) -> bool`` is a
    read-only worker probe for a managed Resume/New choice before admission.
    Installing a runtime never occurs from this API.
    Call ``shutdown`` and wait for ``quiescent`` before releasing Store's lock.
    """
    def __init__(self, config, store, *, games=None, catalog=None,
                 clock=time.monotonic, utcnow=None, transfer_supported=False):
        self.config, self.store = config, store
        self.games, self.catalog = dict(games or {}), catalog
        self.clock = clock
        self.transfer_supported = transfer_supported
        options = {'monotonic': clock}
        if utcnow is not None:
            options['utcnow'] = utcnow
        self.quota = QuotaManager(store, config, **options)
        self.session = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='bbs-store')
        self._jobs = []
        self._cleanup = []
        self._uncertain_cleanup = False
        self._revoked = {}
        self._shutdown = False
        from .message_ui import MessageUI
        self.message_ui = MessageUI(self, utcnow=utcnow)

    def _job(self, kind, func, *args, run=True, **kwargs):
        if len(self._jobs) >= 32 and kind != 'release':
            self._notice('BBS work queue is busy; try again after the current operation.')
            return False
        s = self.session
        tag = (s.id if s else None, s.run_id if s and run else None, self.quota.revision)
        self._jobs.append((kind, tag, self._executor.submit(func, *args, **kwargs)))
        return True

    def open(self, peer, presentation=None, *, connection=None, mode='local'):
        self.tick()
        if self._shutdown or self._cleanup or self._uncertain_cleanup or self.session and self.session.state != 'closed':
            raise BusyError('BBS busy')
        callsign = self.config.resolve(peer)
        connection = connection or ConnectionLease.start(self.config.effective_limits(peer), self.clock())
        if connection.expired(self.clock()):
            raise ValueError('Connection allowance expired')
        connection.accept(self.clock())  # Explicit entry is meaningful interaction, never a hard-deadline renewal.
        s = _Session(_token(), peer, callsign, mode, presentation or Presentation(), connection)
        self.session = s
        self.message_ui.attach()
        if not self._job('player', self.store.player, peer, mode=mode, aliases=self.config.aliases, run=False):
            self.close(s.id, 'work queue busy')
            raise BusyError('BBS work queue busy')
        self._busy()
        return s.id

    def _current(self, sid):
        return self.session if self.session and self.session.id == sid else None

    def _busy(self):
        s = self.session
        if s and s.state != 'closed':
            busy = bool(s.outstanding or s.state in ('loading', 'probing', 'admitting', 'starting', 'stopping', 'transfer')
                        or s.waiting_save or s.adapter and not s.ready or self.message_ui.busy)
            s.connection.set_busy(busy, self.clock())

    def _display(self, text, *, kind='page', payload=None, hold=False):
        s = self.session
        if not s or s.state == 'closed':
            return
        presentation = replace(s.presentation, height=max(8, s.prefs.get('page_length', s.presentation.height)))
        rendered = pages(text, presentation)
        if len(rendered) > 1 and s.prefs.get('pause', True):
            rendered = pages(text, presentation, continuation='\n[BBS MORE | R | Q]')
        held = s.leave_saved[0] if s.leave_saved else ()
        target = held if hold else s.pending
        if sum(len(e.text.encode()) for e in (*s.pending, *held)) + sum(len(t.encode()) for t in rendered) > 32768:
            self._force_game('Game output exceeded the display limit.')
            return
        for i, chunk in enumerate(rendered):
            page_payload = payload(i, len(rendered)) if callable(payload) else payload
            target.append(OutputEvent(kind, _token(), chunk, s.id, s.run_id, s.turn_id,
                                         self.quota.revision, i < len(rendered) - 1, page_payload))
        self._busy()

    def _notice(self, text):
        self._display(text)

    def _available(self):
        values = []
        for key, factory in self.games.items():
            available = getattr(factory, 'available', True)
            if available() if callable(available) else available:
                values.append((key, getattr(factory, 'title', key)))
        return tuple(values)

    def _menu(self, *, full=False):
        s = self.session
        if s.view == 'doors':
            s.game_snapshot = self._available()
        self._display(menu(s.view, title=self.config.title,
                           games=[title for _, title in s.game_snapshot],
                           expert=s.prefs.get('expert', True) and not full))

    def _lobby(self, view='main'):
        s = self.session
        self.message_ui.navigate(view)
        s.state, s.view, s.ready = 'lobby', view, True
        self._menu()

    def _status(self):
        s = self.session
        left = duration(s.connection.remaining(self.clock()))
        state = f'Game: {s.run_id and s.view or "none"}; state: {s.state}; input: {s.input_mode}; '
        caps = ', '.join(sorted(s.adapter.capabilities)) if s.adapter else 'none'
        self._notice(f'Connection: {left} left; idle allowance: {s.connection.limits.idle_seconds} sec.\n'
                     f'{state}capabilities: {caps}; pending pages: {len(s.pending)}.')
        if s.player:
            self._job('status', self.quota.status, s.player.id, s.callsign, run=False)

    def input(self, session_id, input_id, action):
        self.tick()
        s = self._current(session_id)
        if not s or s.state == 'closed' or not isinstance(input_id, str) or len(input_id) > 128 or input_id in s.input_ids:
            return False
        if not isinstance(action, str) or '\n' in action or '\r' in action:
            self._notice('Send one command at a time.')
            return False
        try:
            input_bytes = len(action.encode('utf-8'))
        except UnicodeError:
            self._notice('Command is not valid UTF-8 text.')
            return False
        if input_bytes > 256 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in action):
            self._notice('Command is too long or contains control characters.')
            return False
        # Bound replay history without forgetting IDs within this attachment.
        if len(s.input_ids) >= 100000:
            self.close(s.id, 'input limit')
            return False
        s.input_ids.add(input_id)
        raw, upper = action.strip(), action.strip().upper()
        explicit = upper == 'BBS' or upper.startswith('BBS ')
        command = raw[4:].strip() if upper.startswith('BBS ') else ('HELP' if upper == 'BBS' else raw)
        word, _, argument = command.partition(' ')
        word = word.upper()
        if explicit and word in ('STATUS', 'MENU', 'GOODBYE', 'EXIT', 'MORE', 'REPEAT', 'SKIP',
                                  'SAVE', 'RESTORE', 'ENTER', 'CANCEL', 'CONFIRM') and argument:
            self._notice('That control takes no argument.')
            return False
        if s.state == 'lobby' and word in ('T', 'TIME', 'STATUS') and argument:
            self._notice('That control takes no argument.')
            return False
        pager_control = (explicit and word in ('MORE', 'REPEAT', 'SKIP')
                         or s.paused and upper in ('', 'R', 'Q', '?'))
        if s.state == 'lobby' and self.message_ui.drafting and not pager_control and not (explicit and word == 'STATUS'):
            accepted = self.message_ui.draft_input(action)
            if accepted:
                s.connection.accept(self.clock())
            self._busy()
            return accepted
        if s.state == 'leaving':
            if explicit and word == 'CANCEL' and not s.waiting_save:
                s.pending, s.last_page, s.paused, s.ready = s.leave_saved
                s.leave_saved, s.state = None, 'playing'
                s.outstanding = None
            elif explicit and word == 'CONFIRM' and not s.waiting_save:
                self._commit_leave()
            else:
                self._notice('Waiting for checkpoint.' if s.waiting_save else 'BBS CANCEL returns; BBS CONFIRM leaves without a new save.')
                return False
            s.connection.accept(self.clock())
            self._busy()
            return True
        accepted = True
        if explicit and word in ('GOODBYE', 'EXIT', 'MENU'):
            accepted = self._leave({'GOODBYE': 'close', 'EXIT': 'exit', 'MENU': 'doors'}[word])
        elif explicit and word == 'STATUS' or s.state == 'lobby' and word in ('T', 'TIME', 'STATUS'):
            self._status()
        elif explicit and word == 'TERM':
            accepted = self._term(argument)
        elif word in ('MORE', 'REPEAT', 'SKIP') and (explicit or s.paused) or s.paused and upper in ('', 'R', 'Q', '?'):
            accepted = self._pager({'': 'MORE', 'R': 'REPEAT', 'Q': 'SKIP', '?': 'HELP'}.get(upper, word))
        elif s.state in ('probing', 'choosing'):
            accepted = self._selection_input(command)
        elif s.state == 'lobby':
            accepted = self._lobby_input(command)
        elif s.state == 'playing':
            if s.pending or s.outstanding or s.paused:
                self._notice('Read the current page first: BBS MORE, BBS REPEAT, or BBS SKIP.')
                accepted = False
            elif not s.ready:
                self._notice('Game is still working.')
                accepted = False
            elif explicit and word in ('SAVE', 'RESTORE'):
                capability = 'checkpoint' if word == 'SAVE' else 'restore'
                method = getattr(s.adapter, capability, None)
                if capability not in s.adapter.capabilities or method is None or not method():
                    self._notice('That managed operation is unavailable at this input boundary.')
                    accepted = False
                else:
                    s.ready = False
                    s.managed_operation = capability
                    s.turn_started = self.clock()
                    if capability == 'restore':
                        # Backend generation changes, while the independent
                        # continuous occupancy lease retains its original ID.
                        s.run_id, s.turn_id, s.parser = _token(), _token(), StyleParser()
            elif explicit and word in ('ENTER', 'KEY', 'LINE', 'SEND'):
                mode = {'ENTER': 'pause', 'KEY': 'key', 'LINE': 'line', 'SEND': 'line'}[word]
                accepted = self._submit_game('' if word == 'ENTER' else argument, mode)
            elif explicit:
                self._notice('Unsupported BBS control. Use BBS MENU, STATUS, MORE, or EXIT.')
                accepted = False
            else:
                accepted = self._submit_game(action, 'default')
        else:
            self._notice('Please wait for the current operation.')
            accepted = False
        if accepted:
            s.connection.accept(self.clock())
        self._busy()
        return accepted

    def _lobby_input(self, command):
        s = self.session
        message_action = self.message_ui.command(command)
        if message_action is not None:
            return message_action
        word, _, arg = command.strip().partition(' ')
        word = word.upper()
        changes_preferences = word == 'MENUS' or s.view == 'preferences' and word in ('E', 'P', 'L')
        if changes_preferences and any(kind == 'preferences' and tag[0] == s.id
                                       for kind, tag, _ in self._jobs):
            self._notice('Wait for preferences to save.')
            return False
        if arg and word not in ('L', 'READ', 'GET', 'PLAY', 'MENUS'):
            self._notice('Choose one menu item without extra text.')
            return False
        accepted = True
        if word in ('G', 'GOODBYE', 'EXIT'):
            accepted = self._leave('close' if word != 'EXIT' else 'exit')
        elif word in ('?', 'HELP', ''):
            self._menu(full=True)
        elif word in ('Q', 'MENU'):
            if s.view == 'main':
                self._notice('You are at Main. G says goodbye.')
            else:
                self._lobby()
        elif word in ('D', 'DOORS'):
            self._lobby('doors')
            if not s.prefs.get('expert', True):
                self._job('door_status', self.quota.status, s.player.id, s.callsign, run=False)
        elif word == 'MENUS':
            if arg.upper() not in ('ON', 'OFF'):
                self._notice('MENUS ON or MENUS OFF.')
                return False
            accepted = self._job('preferences', self.store.set_preferences, s.player.id,
                                 expert=arg.upper() == 'OFF', run=False)
        elif word in ('N', 'NEWS'):
            self._display(self.config.news or 'No news.')
        elif word in ('B', 'BULLETINS', 'F', 'FILES'):
            self.message_ui.navigate('files')
            s.view = 'bulletins' if word in ('B', 'BULLETINS') else 'files'
            if self.catalog is None:
                self._notice('No shared catalog configured.')
            else:
                self._job('files', self._list_files, s.peer, run=False)
        elif word in ('P', 'PREFERENCES') and s.view != 'preferences':
            self._lobby('preferences')
        elif s.view == 'preferences' and word == 'C':
            self._notice('Color: BBS TERM ANSI16 or BBS TERM PLAIN.')
        elif s.view == 'preferences' and word in ('E', 'P', 'L'):
            key = {'E': 'expert', 'P': 'pause', 'L': 'page_length'}[word]
            if word == 'L':
                try:
                    value = int(arg)
                    if not 8 <= value <= 100:
                        raise ValueError()
                except ValueError:
                    self._notice('Page length must be 8 through 100.')
                    return False
            else:
                value = not s.prefs.get(key, key == 'pause')
            accepted = self._job('preferences', self.store.set_preferences, s.player.id, run=False, **{key: value})
        elif word in ('READ', 'GET'):
            accepted = self._read_file(arg, transfer=word == 'GET')
        elif word == 'PLAY':
            accepted = self._launch(arg.lower())
        elif s.view == 'doors' and word.isdecimal():
            if s.game_snapshot != self._available():
                self._notice('The doors menu changed. Please select from a fresh menu.')
                self._menu(full=True)
                accepted = False
            elif 1 <= int(word) <= len(s.game_snapshot):
                accepted = self._launch(s.game_snapshot[int(word)-1][0])
            else:
                self._notice('No such door.')
                accepted = False
        elif s.view in ('files', 'bulletins') and word.isdecimal():
            accepted = self._read_file(word)
        else:
            self._notice('Choose one menu item. ? shows this menu.')
            accepted = False
        return accepted

    def _list_files(self, peer):
        generation = self.catalog.generation
        entries = self.catalog.list_files(peer)
        if generation != self.catalog.generation:
            raise ValueError('Catalog changed')
        return generation, entries

    def _read_file(self, arg, transfer=False):
        s = self.session
        try:
            number = int(arg)
            if not 1 <= number <= len(s.file_snapshot) or self.catalog.generation != s.file_generation:
                raise ValueError()
        except (ValueError, AttributeError):
            self._notice('List FILES first; select a current file number.')
            return False
        if transfer:
            if not self.transfer_supported:
                self._notice('Binary transfer is unavailable on this entrance.')
                return False
            if s.pending or s.outstanding or s.paused or self._jobs:
                self._notice('Finish the lobby output before a transfer.')
                return False
            s.state = 'transfer'
        entry = s.file_snapshot[number-1]
        accepted = self._job('get' if transfer else 'read', self.catalog.read_file, entry.name, s.peer, expected=entry, run=False)
        if not accepted and transfer:
            s.state = 'lobby'
        return accepted

    def _term(self, argument):
        s = self.session
        if s.pending or s.outstanding or s.paused:
            self._notice('Wait for a settled display boundary before changing terminal mode.')
            return False
        try:
            parts = argument.upper().split()
            mode = parts[0].lower()
            if len(parts) not in (1, 3):
                raise ValueError()
            width, height = map(int, parts[1:]) if len(parts) == 3 else (80, 24)
            presentation = Presentation(mode, width, height, s.presentation.page_bytes, mode == 'ansi16')
        except (ValueError, IndexError):
            self._notice('Use BBS TERM PLAIN or BBS TERM ANSI16 80 24 (20-160 by 8-100).')
            return False
        # Ack is explicitly plain and handed off before any colored page.
        text = f'BBS TERM OK {mode.upper()} {width} {height}'
        s.pending.append(OutputEvent('page', _token(), text, s.id, s.run_id, s.turn_id, self.quota.revision))
        s.presentation = presentation
        return True

    def _pager(self, command):
        s = self.session
        if command == 'SKIP':
            self.message_ui.skip()
        if command == 'HELP':
            if s.outstanding:
                return False
            s.pending.appendleft(OutputEvent('pager-help', _token(), 'BBS MORE; R repeat; Q skip',
                                            s.id, s.run_id, s.turn_id, self.quota.revision, True))
            s.paused = False
        elif command == 'MORE':
            if s.outstanding:
                return False
            if not s.pending:
                self._notice('No more pages.')
                return False
            s.paused = False
        elif command == 'REPEAT':
            if not s.outstanding and s.last_page:
                s.pending.appendleft(replace(s.last_page, output_id=_token()))
                s.paused = False
            elif not s.last_page:
                self._notice('No previous page.')
                return False
            else:
                return False
        elif command == 'SKIP':
            if not s.pending:
                return False
            s.pending.clear()
            s.paused = False
            if s.adapter and 'structured_prompts' in s.adapter.capabilities and s.prompt:
                self._display(s.prompt)
        return True

    def _launch(self, key):
        s = self.session
        if key not in dict(self._available()):
            self._notice('That game is not installed and available.')
            return False
        if s.outstanding or s.pending or s.paused or self._jobs:
            self._notice('Finish the menu output before opening a door.')
            return False
        self.message_ui.navigate('doors')
        s.state, s.run_id, s.turn_id = 'probing', _token(), None
        s.renewal_failed, s.renewal_pending = False, False
        s.parser = StyleParser()
        s.view, s.started = key, self.clock()
        s.admitted_revision = self.quota.revision
        s.selection_factory = self.games[key]
        s.resume_available, s.resume_phase, s.resume_text = False, None, None
        s.managed_operation = None
        probe = getattr(s.selection_factory, 'saved_game', None)
        if probe is not None:
            if not self._job('saved_probe', probe, self._game_directory()):
                self._selection_cancel()
                return False
        else:
            return self._admit_selected(False)
        return True

    def _game_directory(self):
        s = self.session
        return Path(self.store.root) / 'games' / s.view / 'players' / s.player.id

    def _choice_prompt(self):
        return ('Saved game. R resume; N new; Q back.' if self.session.resume_available
                else 'Save unavailable. N new; Q back.')

    def _selection_input(self, command):
        s = self.session
        word = command.upper()
        if word in ('Q', 'MENU'):
            self._selection_cancel()
        elif word in ('G', 'GOODBYE', 'EXIT'):
            self._leave('exit' if word == 'EXIT' else 'close')
        elif word in ('?', 'HELP'):
            self._notice(self._choice_prompt() if s.state == 'choosing' else 'Checking saved game. Q back.')
        elif s.state == 'choosing' and word in ('R', 'RESUME', 'N', 'NEW'):
            if s.pending or s.outstanding or s.paused:
                return False
            if word in ('R', 'RESUME') and not s.resume_available:
                self._notice(self._choice_prompt())
                return False
            return self._admit_selected(word in ('R', 'RESUME'))
        else:
            self._notice(self._choice_prompt() if s.state == 'choosing' else 'Checking saved game. Q back.')
            return False
        return True

    def _selection_cancel(self, notice=None):
        self.session.leave_target = 'doors'
        self._commit_leave()
        if notice:
            self._notice(notice)

    def _admit_selected(self, resume):
        s = self.session
        if self.games.get(s.view) is not s.selection_factory or s.view not in dict(self._available()):
            self._selection_cancel('Door changed; select again.')
            return False
        s.state, s.ready = 'admitting', False
        s.resume_phase = 'startup' if resume else None
        s.admitted_revision = self.quota.revision
        if not self._job('admit', self.quota.admit, s.player.id, s.callsign, s.id, s.run_id):
            self._selection_cancel()
            return False
        return True

    def _submit_game(self, text, mode):
        s = self.session
        if not self.quota.authorized(s.usage):
            self._force_game('Door allowance expired. Lobby services remain available.')
            return False
        if s.connection.remaining(self.clock()) <= 10:
            self._notice('Connection is ending; no new game turns.')
            return False
        if mode == 'key' and len(text) != 1:
            self._notice('BBS KEY requires one character.')
            return False
        if s.adapter.submit(text, mode):
            s.turn_id, s.ready, s.turn_started = _token(), False, self.clock()
            return True
        else:
            self._notice('Game cannot accept that input now.')
            return False

    def _leave(self, target):
        s = self.session
        if self.message_ui.drafting:
            self._notice('Draft pending. .cancel before leaving.')
            return False
        if s.state in ('probing', 'choosing') or s.resume_phase is not None:
            s.leave_target = target
            self._commit_leave()
            return True
        if s.state == 'transfer':
            if target == 'doors':
                self._notice('Finish or cancel the file transfer before entering Doors.')
                return False
            else:
                self._finish_leave(target)
            return True
        if s.adapter:
            if s.outstanding:
                self._notice('Wait for the current output handoff before leaving.')
                return False
            s.leave_saved = (s.pending, s.last_page, s.paused, s.ready)
            s.pending, s.paused, s.state = deque(), False, 'leaving'
            s.leave_target = target
            method = getattr(s.adapter, 'checkpoint', None)
            if s.ready and 'checkpoint' in s.adapter.capabilities and method and method():
                s.waiting_save = True
                s.turn_started = self.clock()
                self._notice('Saving before leaving...')
            else:
                s.waiting_save = False
                self._notice('Progress since the last confirmed save may be lost. BBS CANCEL returns; BBS CONFIRM leaves.')
        elif s.state in ('admitting', 'starting'):
            s.leave_target = target
            self._commit_leave()
        else:
            self._finish_leave(target)
        return True

    def _commit_leave(self):
        s = self.session
        target = s.leave_target
        s.pending.clear()
        s.outstanding = None
        s.paused, s.last_page, s.leave_saved, s.waiting_save = False, None, None, False
        self._stop_adapter('menu')
        if self._cleanup:
            s.state, s.leave_target = 'stopping', target
        else:
            self._finish_leave(target)

    def _finish_leave(self, target):
        if target == 'doors':
            self._lobby('doors')
        else:
            self.close(self.session.id, 'goodbye' if target == 'close' else 'exit', force=True)
            # The entrance owns whether EXIT returns to chat.
            event = self.session.pending[-1]
            self.session.pending[-1] = replace(event, kind='exit' if target == 'exit' else 'close')

    def claim_transfer(self, session_id):
        """Hand a settled lobby to its entrance's native file protocol."""
        self.tick()
        s = self._current(session_id)
        if (not s or s.state != 'lobby' or s.pending or s.outstanding or s.paused
                or self._jobs or self.message_ui.drafting):
            return None
        event = OutputEvent('transfer', _token(), '', s.id, policy_revision=self.quota.revision)
        s.state, s.outstanding = 'transfer', event
        self._busy()
        return event.output_id

    def _stop_adapter(self, reason):
        s = self.session
        if s.run_id:
            self._revoked[(s.id, s.run_id)] = self.clock()
        if s.usage:
            self._revoked[(s.id, s.usage.id)] = self.clock()
        s.parser = StyleParser()
        s.selection_factory, s.resume_phase, s.resume_text = None, None, None
        s.resume_available = False
        s.managed_operation = None
        s.renewal_pending, s.renewal_failed = False, False
        if s.usage:
            self._job('release', self.quota.release, s.usage, stopped_at=self.clock())
            s.usage = None
        if s.adapter:
            adapter, s.adapter = s.adapter, None
            adapter.close(reason)
            self._cleanup.append(adapter)
        s.run_id, s.turn_id, s.ready = None, None, False

    def _force_game(self, reason):
        s = self.session
        if not s:
            return
        if s.outstanding:
            # The entrance must tear down/cancel an output whose transmission
            # cannot be settled before access is revoked.
            self.close(s.id, reason)
            return
        s.pending.clear()
        s.outstanding, s.paused, s.last_page = None, False, None
        self._stop_adapter(reason)
        self.message_ui.invalidate(drop_draft=True)
        s.state, s.leave_target = 'stopping', 'doors'
        self._notice(reason)
        if not self._cleanup:
            self._lobby('doors')

    def close(self, session_id, reason='disconnect', force=True):
        s = self._current(session_id)
        if not s or s.state == 'closed':
            return
        if not force:
            self._leave('close')
            return
        self._stop_adapter(reason)
        self.message_ui.invalidate(drop_draft=True)
        s.state = 'closed'
        s.pending.clear()
        s.outstanding, s.last_page, s.paused = None, None, False
        s.pending.append(OutputEvent('close', _token(), reason, s.id, policy_revision=self.quota.revision))

    def poll(self, session_id):
        self.tick()
        s = self._current(session_id)
        if not s or s.outstanding or s.paused or not s.pending:
            return []
        while s.pending:
            event = s.pending.popleft()
            if self.message_ui.current_event(event):
                break
        else:
            return []
        s.outstanding = event
        self._busy()
        return [event]

    def output_done(self, session_id, output_id, disposition='complete'):
        self.tick()
        s = self._current(session_id)
        if not s or not s.outstanding or s.outstanding.output_id != output_id:
            return False
        event, s.outstanding = s.outstanding, None
        if not self.message_ui.current_event(event):
            if disposition in ('fatal', 'connection-fatal'):
                self.close(s.id, 'output failure')
            self._busy()
            return True
        if disposition in ('fatal', 'connection-fatal'):
            self.close(s.id, 'output failure')
        elif disposition in ('retry', 'retry-safe'):
            s.pending.appendleft(replace(event, output_id=_token()))
        elif disposition in ('complete', 'completed'):
            if event.kind != 'pager-help' and self.message_ui.current_event(event):
                s.last_page = event
            self.message_ui.delivered(event)
            s.paused = bool(event.more and s.pending and s.prefs.get('pause', True))
            if event.kind == 'transfer':
                self._lobby('files')
        elif event.kind == 'transfer' and disposition in ('refused', 'aborted', 'failed'):
            self._lobby('files')
        else:
            self.close(s.id, 'unknown output disposition')
        self._busy()
        return True

    def tick(self):
        now = self.clock()
        for adapter in tuple(self._cleanup):
            events = adapter.poll()
            if any(event.kind == 'exited' and event.outcome == 'cleanup_uncertain' for event in events):
                self._uncertain_cleanup = True
            elif any(event.kind == 'exited' for event in events):
                self._cleanup.remove(adapter)
        s = self.session
        if s and s.state != 'closed':
            if s.connection.expired(now) or s.connection.idle_expired(now):
                self.close(s.id, 'connection allowance expired' if s.connection.expired(now) else 'idle allowance expired')
            elif s.usage and not self.quota.authorized(s.usage):
                self._force_game('Door allowance expired. Lobby services remain available.')
        for item in tuple(self._jobs):
            kind, (sid, run, revision), future = item
            if not future.done():
                continue
            self._jobs.remove(item)
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            s = self.session
            valid = s and s.id == sid and s.state != 'closed' and (run is None or s.run_id == run)
            if kind == 'get':
                valid = valid and s.state == 'transfer'
            if kind == 'renew' and s and s.id == sid and s.state != 'closed' and result is not None:
                # Renewal belongs to continuous occupancy, independent of a
                # restore's new backend generation; it cannot restore output.
                valid = bool(s.usage and s.usage.id == result.id)
            if kind in ('admit', 'renew', 'saved_probe'):
                valid = valid and revision == self.quota.revision
            if not valid:
                if result is not None and kind in ('admit', 'renew'):
                    cutoff = result.start if kind == 'admit' else self._revoked.get((sid, result.id), self.clock())
                    self._jobs.append(('release', (None, None, 0), self._executor.submit(self.quota.release, result, stopped_at=cutoff)))
                if kind == 'admit' and s and s.id == sid and s.run_id == run and s.state == 'admitting':
                    self._force_game('Door admission cancelled by a policy change. Select the door again.')
                if kind == 'saved_probe' and s and s.id == sid and s.run_id == run and s.state == 'probing':
                    self._selection_cancel('Door policy changed; select again.')
                continue
            self._complete(kind, result, error)
        s = self.session
        if not s or s.state == 'closed':
            return
        if s.state == 'stopping' and not self._cleanup:
            self._finish_leave(s.leave_target)
        if s.adapter:
            self._game_events(s.adapter.poll())
            if s.state == 'leaving' and s.waiting_save and now - s.turn_started >= 5:
                s.waiting_save = False
                pending, last_page, paused, _ = s.leave_saved
                s.leave_saved = (pending, last_page, paused, False)
                s.turn_started = now
                self._notice('Save is unconfirmed. BBS CANCEL returns; BBS CONFIRM leaves without a new confirmed save.')
            if s.state == 'starting' and now - s.started >= 10:
                self._force_game('Resume timed out. Saved game retained.' if s.resume_phase else 'Game startup timed out.')
            elif s.state == 'playing' and not s.ready and 'structured_prompts' in s.adapter.capabilities and now - s.turn_started >= 30:
                self._force_game('Game response timed out.')
        if s.usage and not s.renewal_pending and not s.renewal_failed and s.usage.deadline - now <= 5:
            s.renewal_pending = True
            if not self._job('renew', self.quota.renew, s.usage):
                s.renewal_pending, s.renewal_failed = False, True
        remaining = s.connection.remaining(now)
        if remaining <= 10 and not s.deadline_prepared:
            s.deadline_prepared = True
            if s.adapter and s.ready and 'checkpoint' in s.adapter.capabilities:
                if s.adapter.checkpoint():
                    s.ready = False
                    s.managed_operation = 'checkpoint'
                    s.turn_started = now
        for threshold in (300, 60):
            if s.connection.limits.connection_seconds > threshold and remaining <= threshold and threshold not in s.warnings:
                s.warnings.add(threshold)
                if not s.pending and not s.outstanding:
                    self._notice(f'Connection: {duration(remaining)} left.')
        self._busy()

    def _complete(self, kind, result, error):
        s = self.session
        if kind == 'messages':
            if result is not None:
                self.message_ui.complete(result)
            return
        if error:
            if kind == 'saved_probe':
                if self.games.get(s.view) is not s.selection_factory or s.view not in dict(self._available()):
                    self._selection_cancel('Door changed; select again.')
                else:
                    s.state, s.ready, s.resume_available = 'choosing', True, False
                    self._notice(self._choice_prompt())
            elif kind in ('admit', 'renew'):
                if kind == 'renew':
                    s.renewal_pending = False
                    s.renewal_failed = True
                    # Paid time stays authorized until its original cutoff.
                    return
                self._force_game('Door access unavailable or daily allowance exhausted. Check Time left for reset details.')
            else:
                self._notice('That operation failed; no success was recorded.')
                if kind == 'get':
                    s.state = 'lobby'
                if kind == 'player':
                    self.close(s.id, 'player store unavailable')
            return
        if kind == 'player':
            s.player, s.prefs = result, dict(result.preferences)
            verbose = not s.prefs.get('expert', True)
            if verbose:
                self._notice(f'Welcome, {s.callsign}. Connection: {duration(s.connection.remaining(self.clock()))} left.')
                if self.config.sysop:
                    self._notice(f'Sysop: {self.config.sysop}')
                if self.config.news:
                    self._display(self.config.news)
            self._lobby()
            if verbose:
                self._job('status', self.quota.status, s.player.id, s.callsign, run=False)
        elif kind in ('status', 'door_status'):
            self._notice(f'Doors today: {duration(result.remaining_seconds)} left; used {duration(result.used_seconds)}.\n'
                         f'Reset: {result.reset_at.isoformat()} {result.timezone}.')
        elif kind == 'preferences':
            s.prefs = dict(result)
            self._notice('Preferences saved.')
            self._menu()
        elif kind == 'files':
            s.file_generation, s.file_snapshot = result
            listing = '\n'.join(f'[{n}] {e.name} ({e.size} bytes)' for n, e in enumerate(s.file_snapshot, 1))
            self._notice(s.view.title() + '\n' + (listing or 'No exported files.') + '\nREAD N reads text; Q returns to Main.')
        elif kind == 'read':
            try:
                if len(result.data) > 32768 or b'\0' in result.data:
                    raise ValueError()
                text = result.data.decode('utf-8')
            except (UnicodeError, ValueError):
                self._notice('This file is not a bounded UTF-8 text bulletin.')
            else:
                self._display(text)
        elif kind == 'get':
            s.pending.append(OutputEvent('transfer', _token(), 'Sending selected file.', s.id,
                                         policy_revision=self.quota.revision, payload=result))
        elif kind == 'saved_probe':
            if self.games.get(s.view) is not s.selection_factory or s.view not in dict(self._available()):
                self._selection_cancel('Door changed; select again.')
            elif type(result) is not bool:
                s.state, s.ready, s.resume_available = 'choosing', True, False
                self._notice(self._choice_prompt())
            elif result:
                s.state, s.ready, s.resume_available = 'choosing', True, True
                self._notice(self._choice_prompt())
            else:
                self._admit_selected(False)
        elif kind == 'admit':
            s.usage = result
            if not self.quota.authorized(result):
                self._force_game('Door admission expired before startup.')
                return
            if self.games.get(s.view) is not s.selection_factory or s.view not in dict(self._available()):
                self._force_game('Door changed; select again.')
                return
            try:
                adapter = s.selection_factory()
                if s.resume_phase and not {'restore', 'structured_prompts'} <= adapter.capabilities:
                    raise ValueError('managed resume unavailable')
                s.adapter = adapter
                s.adapter.start(s.player.id, self._game_directory())
                s.state, s.started = 'starting', self.clock()
            except Exception:
                self._force_game('Resume failed. Saved game retained.' if s.resume_phase else 'Game could not start.')
        elif kind == 'renew':
            s.usage, s.renewal_pending = result, False

    def _game_events(self, events):
        adapter = self.session.adapter
        try:
            self._apply_game_events(events)
        finally:
            # Lifecycle evidence survives an early return for save/error. A
            # worker can finish and exit in one batch before the next tick.
            if adapter in self._cleanup and any(event.kind == 'exited' and event.outcome != 'cleanup_uncertain' for event in events):
                self._cleanup.remove(adapter)

    def _apply_game_events(self, events):
        s = self.session
        if s.resume_phase:
            self._resume_events(events)
            return
        document = []
        for event in events:
            if event.kind in ('text', 'prompt'):
                clean = s.parser.feed(event.text)
                document.extend(clean.spans)
                if event.kind == 'prompt':
                    s.prompt = event.text or ('Game input:' if 'structured_prompts' in s.adapter.capabilities else '')
            if event.kind in ('ready', 'prompt'):
                s.ready = True
                s.input_mode = event.mode
                if s.state == 'leaving' and s.leave_saved:
                    pending, last, paused, _ = s.leave_saved
                    s.leave_saved = (pending, last, paused, True)
                if s.state == 'starting':
                    s.state = 'playing'
            elif event.kind == 'saved':
                if s.state == 'leaving' and s.waiting_save:
                    self._commit_leave()
                    return
                self._notice('Checkpoint saved.')
                s.managed_operation = None
                s.ready = True
            elif event.kind == 'restored':
                document.clear()
                s.turn_id = _token()
                s.pending.clear()
                s.last_page, s.paused = None, False
                s.ready = True
                self._notice('Checkpoint restored.')
                s.managed_operation = None
            elif event.kind == 'error':
                if s.state == 'leaving' and s.waiting_save:
                    s.waiting_save = False
                    self._notice('Save failed. BBS CANCEL returns; BBS CONFIRM leaves without a new save.')
                elif s.managed_operation:
                    s.managed_operation = None
                    s.ready = True
                    self._notice('Checkpoint operation failed; previous confirmed save retained.')
                else:
                    self._force_game('Game failed; the last command may have taken effect.')
                    return
            elif event.kind == 'exited':
                if event.outcome == 'cleanup_uncertain':
                    self._uncertain_cleanup = True
                    self.close(s.id, 'Backend cleanup is uncertain; operator recovery required.')
                    return
                if document:
                    from .presentation import StyledText
                    self._display(StyledText(document))
                s.adapter = None
                self._stop_adapter('game exit')
                self._lobby('doors')
                return
        if document:
            from .presentation import StyledText
            self._display(StyledText(document), hold=s.state == 'leaving' and s.leave_saved is not None)

    def _resume_events(self, events):
        """Suppress fresh startup; publish only a confirmed restored boundary."""
        from .presentation import StyledText
        s = self.session
        for index, event in enumerate(events):
            if event.kind in ('error', 'exited'):
                if event.kind == 'exited' and event.outcome == 'cleanup_uncertain':
                    self._uncertain_cleanup = True
                    self.close(s.id, 'Backend cleanup is uncertain; operator recovery required.')
                else:
                    self._force_game('Resume failed. Saved game retained.')
                return
            if s.resume_phase == 'startup':
                if event.kind == 'prompt':
                    # Checkpoint operations are valid only at the adapter's
                    # explicit line-input boundary, never a quiet-output guess.
                    if event.mode != 'line':
                        self._force_game('Resume unavailable at this prompt. Saved game retained.')
                        return
                    try:
                        submitted = s.adapter.restore()
                    except Exception:
                        submitted = False
                    if not submitted:
                        self._force_game('Resume failed. Saved game retained.')
                        return
                    s.resume_phase, s.managed_operation = 'confirming', 'restore'
                    s.resume_text, s.resume_bytes, s.parser = StyledText(), 0, StyleParser()
                    s.run_id, s.turn_id, s.turn_started = _token(), _token(), self.clock()
                    s.ready = False
                continue
            if event.kind in ('text', 'prompt'):
                # The worker orders stdout before its explicit control hooks.
                # Hold restore-operation text until success and a new prompt;
                # this includes restored text emitted before the success hook.
                clean = s.parser.feed(event.text)
                s.resume_bytes += len(clean.text.encode())
                if s.resume_bytes > 32768:
                    self._force_game('Resume output exceeded the display limit. Saved game retained.')
                    return
                s.resume_text.spans.extend(clean.spans)
            if event.kind == 'restored':
                s.resume_phase = 'boundary'
            elif event.kind == 'prompt':
                if s.resume_phase != 'boundary':
                    self._force_game('Resume was not confirmed. Saved game retained.')
                    return
                document = s.resume_text
                s.state, s.ready, s.input_mode = 'playing', True, event.mode
                s.prompt = event.text or 'Game input:'
                s.resume_phase, s.resume_text, s.managed_operation = None, None, None
                s.turn_id, s.turn_started = _token(), self.clock()
                combined = StyledText()
                combined.append('Checkpoint restored.\n')
                combined.spans.extend(document.spans)
                self._display(combined)
                # Any later events remain ordinary game output; the selection
                # can never issue a second restore at a later game prompt.
                self._apply_game_events(events[index + 1:])
                return

    def reload_policy(self, config):
        self.config = config
        revision = self.quota.reload(config)
        self.message_ui.reload()
        if self.session and self.session.state in ('probing', 'choosing'):
            self._selection_cancel('Door policy changed; select again.')
        self.tick()
        return revision

    @property
    def quiescent(self):
        return not self._cleanup and not self._jobs and not self._uncertain_cleanup

    def shutdown(self):
        """Begin nonblocking teardown; keep ticking until quiescent."""
        if self.session:
            self.close(self.session.id, 'service shutdown')
        self._shutdown = True
        if self.quiescent:
            self._executor.shutdown(wait=False)
