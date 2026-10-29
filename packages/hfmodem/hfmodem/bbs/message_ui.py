# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Small lobby message views, explicit drafts and complete-display receipts."""
from dataclasses import dataclass, field
import uuid


def token():
    return uuid.uuid4().hex


@dataclass
class Draft:
    key: str
    scope: str
    area: str | None
    subject: str
    parent_id: int | None = None
    lines: list = field(default_factory=list)
    attempted: bool = False
    posting: bool = False

    @property
    def body(self):
        return '\n'.join(self.lines)


@dataclass
class MessageView:
    generation: str = field(default_factory=token)
    area: str | None = None
    cursor: int | None = None
    draft: Draft | None = None
    requests: set = field(default_factory=set)
    receipt: dict | None = None


@dataclass(frozen=True)
class Result:
    stamp: tuple
    request: str
    operation: str
    value: object = None
    failed: bool = False
    context: object = None


def _work(stamp, request, operation, context, function, args, kwargs):
    try:
        return Result(stamp, request, operation, function(*args, **kwargs), context=context)
    except Exception:
        # No local paths, raw SQLite errors or operator diagnostics to callers.
        return Result(stamp, request, operation, failed=True, context=context)


class MessageUI:
    def __init__(self, core, *, utcnow=None):
        self.core = core
        self.utcnow = utcnow
        self.reload()

    def reload(self):
        from .messages import MessageStore
        config = self.core.config
        areas = getattr(config, 'message_areas', {'general': 'General'})
        limit = getattr(config, 'message_posts_per_day', 20)
        previous = getattr(self, 'store', None)
        if previous and previous.areas == areas and previous.posts_per_day == limit:
            return  # An unrelated door-policy reload must not discard a draft.
        self.store = MessageStore(self.core.store, areas, utcnow=self.utcnow, posts_per_day=limit)
        if self.core.session:
            self.invalidate()
            if self.view.draft:
                self.view.draft.posting = False
                self._display('Message settings changed. Draft kept; .cancel discards.')

    @property
    def view(self):
        return self.core.session.message_view

    @property
    def busy(self):
        s = self.core.session
        return bool(s and s.state == 'lobby' and s.message_view and s.message_view.requests)

    @property
    def drafting(self):
        return bool(self.core.session and self.view and self.view.draft)

    def attach(self):
        self.core.session.message_view = MessageView()

    def stamp(self):
        return self.core.session.id, self.view.generation

    def invalidate(self, *, drop_draft=False):
        s, view = self.core.session, self.view
        if view is None:
            return
        view.generation, view.requests, view.receipt, view.cursor = token(), set(), None, None
        if drop_draft:
            view.draft = None
        if (any(event.kind.startswith('message-') for event in s.pending)
                or s.outstanding and s.outstanding.kind.startswith('message-')
                or s.last_page and s.last_page.kind.startswith('message-')):
            s.paused = False
        s.pending = type(s.pending)(event for event in s.pending if not event.kind.startswith('message-'))
        if s.last_page and s.last_page.kind.startswith('message-'):
            s.last_page = None

    def navigate(self, destination):
        if destination != 'messages':
            self.invalidate(drop_draft=True)
            self.view.area = None

    def current_event(self, event):
        return not event.kind.startswith('message-') or bool(event.payload and event.payload['stamp'] == self.stamp())

    def skip(self):
        self.view.receipt = None

    def _display(self, text):
        self.core._display(text, kind='message-ui', payload={'stamp': self.stamp()})

    def _request(self, operation, function, *args, context=None, **kwargs):
        request, stamp = token(), self.stamp()
        if not self.core._job('messages', _work, stamp, request, operation, context,
                              function, args, kwargs, run=False):
            return False
        self.view.requests.add(request)
        self.core._busy()
        return True

    def _switch(self, area):
        self.invalidate(drop_draft=True)
        self.view.area = area
        self.core.session.view = 'messages'

    def areas(self):
        self._switch(None)
        return self._request('areas', self.store.unread_counts, self.core.session.player.id)

    def area(self, name, *, after=0):
        name = name.lower()
        if name != 'sysop' and name not in self.core.config.message_areas:
            self._display('No such area. M lists areas.')
            return False
        self._switch(name)
        method = self.store.list_sysop if name == 'sysop' else self.store.list_public
        args = (self.core.session.player.id,) if name == 'sysop' else (self.core.session.player.id, name)
        return self._request('headers', method, *args, after=after, limit=10)

    @staticmethod
    def _subject(subject):
        try:
            return bool(subject.strip()) and len(subject.encode('utf-8')) <= 72
        except UnicodeError:
            return False

    def _draft(self, subject, *, scope, area=None, parent_id=None):
        if not self._subject(subject):
            self._display('Subject: 1-72 UTF-8 bytes.')
            return False
        self.invalidate(drop_draft=True)
        self.view.draft = Draft(token(), scope, area, subject.strip(), parent_id)
        self._display('Draft: text lines; .send posts; .cancel discards; .blank adds an empty line; .preview reads draft.')
        return True

    def command(self, command):
        """Return None for another lobby command; games never call this."""
        word, _, arg = command.strip().partition(' ')
        word = word.upper()
        s = self.core.session
        if word in ('M', 'MESSAGES'):
            if arg:
                return False
            return self.areas()
        if word == 'AREA':
            return self.area(arg) if arg and ' ' not in arg else False
        if word == 'SYSOP':
            if not arg:
                return self.area('sysop')
            self._switch('sysop')
            return self._draft(arg, scope='sysop')
        if s.view != 'messages':
            return None
        if word in ('?', 'HELP'):
            self._display('M areas; AREA name; READ id; POST subject; REPLY id [subject]; SYSOP subject; Q back.')
            return True
        if word == 'Q':
            if self.view.area is not None:
                return self.areas()
            return None  # Core's ordinary Q returns to Main.
        if word == 'NEXT':
            if arg or self.view.cursor is None or self.view.area is None:
                return False
            return self.area(self.view.area, after=self.view.cursor)
        if word == 'POST':
            if self.view.area is None or self.view.area == 'sysop':
                self._display('Select a public AREA first, or SYSOP subject.')
                return False
            return self._draft(arg, scope='public', area=self.view.area)
        if word in ('READ', 'REPLY'):
            first, _, subject = arg.partition(' ')
            try:
                number = int(first)
                if number <= 0 or word == 'READ' and subject or word == 'REPLY' and subject and not self._subject(subject):
                    raise ValueError()
            except ValueError:
                self._display('READ id or REPLY id [subject].')
                return False
            if self.view.area is None:
                self._display('Select an AREA first.')
                return False
            self.invalidate()
            return self._request(word.lower(), self.store.get, s.player.id, number, context=subject)
        return None

    def draft_input(self, action):
        draft, s = self.view.draft, self.core.session
        control = action.strip().lower()
        if control == '.cancel':
            attempted = draft.attempted
            self.invalidate(drop_draft=True)
            self._display('Draft cancelled; submitted post may already exist.' if attempted else 'Draft discarded.')
            return True
        if s.pending or s.outstanding or s.paused:
            return False
        if control == '.preview':
            self._display(draft.subject + '\n' + draft.body)
            return True
        if control == '.send':
            if draft.posting:
                return False
            if not draft.body.strip():
                self._display('Draft is empty; add text or .cancel.')
                return False
            method = self.store.post_sysop if draft.scope == 'sysop' else self.store.post_public
            args = (s.player.id,) if draft.scope == 'sysop' else (s.player.id, draft.area)
            accepted = self._request('posted', method, *args, draft.subject, draft.body, draft.key,
                                     parent_id=draft.parent_id, context=draft.key)
            if accepted:
                draft.attempted = draft.posting = True
            return accepted
        if draft.attempted:
            self._display('Submitted draft is locked. .send retries the same post; .cancel discards the draft.')
            return False
        upper = action.strip().upper()
        if upper.startswith('BBS ') or upper.split()[:1] in ([v] for v in ('Q', 'G', 'EXIT', 'GOODBYE', 'MENU', 'PLAY', 'AREA', 'POST', 'REPLY', 'SYSOP', 'DOORS', 'MESSAGES', 'M')):
            self._display('Draft pending. .cancel before leaving or choosing another service.')
            return False
        line = '' if control == '.blank' else action
        candidate = '\n'.join([*draft.lines, line])
        if len(candidate.encode('utf-8')) > 2048:
            self._display('Draft limit: 2048 UTF-8 bytes. .send or .cancel.')
            return False
        draft.lines.append(line)
        return True

    def complete(self, result):
        s = self.core.session
        if not s or s.state != 'lobby' or result.stamp != self.stamp():
            return
        view = self.view
        view.requests.discard(result.request)
        if result.operation == 'mark':
            return
        if result.failed:
            if result.operation == 'posted' and view.draft and view.draft.key == result.context:
                view.draft.posting = False
                self._display('Post not confirmed. .send retries the same post; .cancel discards the draft.')
            else:
                self._display('Message operation unavailable. No success recorded.')
            return
        value = result.value
        if result.operation == 'areas':
            rows = [f'{slug} {name}: {value.get(slug, 0)} unread' for slug, name in self.core.config.message_areas.items()]
            rows += [f'sysop: {value.get("sysop", 0)} unread', 'AREA name; Q back.']
            self._display('\n'.join(rows))
        elif result.operation == 'headers':
            view.cursor = value.next_cursor
            rows = [f'{" " if message.read else "*"}{message.id} {message.author}: {message.subject}'
                    for message in value.messages]
            footer = 'READ id; POST subject; REPLY id; Q areas.' + (' NEXT lists more.' if view.cursor is not None else '')
            self._display(view.area + '\n' + ('\n'.join(rows) or 'No messages.') + '\n' + footer)
        elif result.operation in ('read', 'reply'):
            correct = value.scope == 'sysop' if view.area == 'sysop' else value.scope == 'public' and value.area == view.area
            if not correct:
                self._display('Message is outside this area.')
                return
            if result.operation == 'reply':
                if value.hidden:
                    self._display('Cannot reply to a removed message.')
                    return
                subject = result.context or 'Re: ' + value.subject
                if not result.context:
                    while len(subject.encode('utf-8')) > 72:
                        subject = subject[:-1]
                self._draft(subject, scope=value.scope, area=value.area, parent_id=value.id)
            else:
                self._body(value)
        elif result.operation == 'posted' and view.draft and view.draft.key == result.context:
            view.draft = None
            self._display(f'Posted #{value.id}.')

    def _body(self, message):
        if message.hidden:
            self._display(f'#{message.id} [removed]')
            return
        receipt = dict(key=token(), message=message.id, done=set(), total=None)
        self.view.receipt = receipt
        stamp = self.stamp()

        def payload(index, total):
            receipt['total'] = total
            return dict(stamp=stamp, receipt=receipt['key'], index=index)

        self.core._display(f'#{message.id} {message.author}: {message.subject}\n{message.body}',
                           kind='message-body', payload=payload)

    def delivered(self, event):
        if event.kind != 'message-body' or not self.current_event(event):
            return
        receipt = self.view.receipt
        if not receipt or event.payload['receipt'] != receipt['key']:
            return
        receipt['done'].add(event.payload['index'])
        if len(receipt['done']) == receipt['total']:
            self.view.receipt = None
            self._request('mark', self.store.mark_read, self.core.session.player.id, receipt['message'])
