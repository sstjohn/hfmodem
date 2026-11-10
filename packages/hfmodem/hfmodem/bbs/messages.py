# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded local message areas and private caller/sysop correspondence.

All methods touching SQLite belong on the BBS persistence worker. Callsigns are
low-trust player identities, never operator credentials. OperatorMessages is an
explicit offline capability: its caller must acquire Store's exclusive owner
lock, and it must never be exposed through the remote command dispatcher.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re

from .config import validate_message_areas

MAX_MESSAGES = 100_000


class MessageError(ValueError):
    pass


class MessageNotFound(MessageError):
    pass


class MessageLimit(MessageError):
    pass


class DuplicateRequest(MessageError):
    pass


@dataclass(frozen=True)
class Message:
    id: int
    scope: str
    area: str | None
    owner_id: str | None
    author_id: str | None
    author: str
    subject: str
    body: str
    parent_id: int | None
    thread_id: int
    created_at: str
    hidden: bool
    read: bool


@dataclass(frozen=True)
class MessagePage:
    messages: tuple[Message, ...]
    next_cursor: int | None


def _text(value, maximum, *, multiline=False):
    if not isinstance(value, str) or not value.strip():
        raise MessageError("message text cannot be empty")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise MessageError("invalid message text") from exc
    if size > maximum or any((ord(c) < 32 and not (multiline and c == '\n')) or 127 <= ord(c) <= 159 for c in value):
        raise MessageError(f"message text exceeds {maximum} UTF-8 bytes or contains controls")
    return value


def _identifier(value):
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise MessageNotFound("message unavailable")
    return value


class MessageStore:
    def __init__(self, store, areas=None, *, utcnow=None, posts_per_day=20):
        self.store = store
        self.areas = validate_message_areas({"general": "General"} if areas is None else areas)
        if type(posts_per_day) is not int or not 0 <= posts_per_day <= 1000:
            raise MessageError("posts_per_day must be an integer in 0..1000")
        self.posts_per_day = posts_per_day
        self.utcnow = utcnow or (lambda: datetime.now(timezone.utc))

    def _now(self):
        result = self.utcnow()
        if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
            raise MessageError("message timestamp requires aware datetime")
        return result.astimezone(timezone.utc)

    @staticmethod
    def _player(db, player_id):
        if not isinstance(player_id, str) or not db.execute("SELECT 1 FROM players WHERE id=?", (player_id,)).fetchone():
            raise MessageNotFound("player unavailable")

    def _area(self, area):
        if not isinstance(area, str) or area not in self.areas:
            raise MessageNotFound("message area unavailable")

    @staticmethod
    def _row(db, message_id):
        _identifier(message_id)
        row = db.execute("SELECT m.*,p.callsign AS author FROM messages m LEFT JOIN players p ON p.id=m.author_id WHERE m.id=?", (message_id,)).fetchone()
        if row is None:
            raise MessageNotFound("message unavailable")
        return row

    def _access(self, row, player_id):
        if row['scope'] == 'public':
            self._area(row['area'])
        elif row['owner_id'] != player_id:
            raise MessageNotFound("message unavailable")

    @staticmethod
    def _message(db, row, player_id, *, header=False, operator=False):
        hidden = bool(row['hidden'])
        redacted = hidden and not operator
        read = bool(player_id and db.execute("SELECT 1 FROM message_reads WHERE player_id=? AND message_id=?", (player_id, row['id'])).fetchone())
        if operator:
            read = bool(db.execute("SELECT 1 FROM operator_message_reads WHERE message_id=?", (row['id'],)).fetchone())
        return Message(row['id'], row['scope'], row['area'], row['owner_id'],
                       None if redacted else row['author_id'],
                       '' if redacted else (row['author'] or 'Sysop'),
                       '[removed]' if redacted else row['subject'],
                       '' if header or redacted else row['body'],
                       row['parent_id'], row['thread_id'], row['created_at'], hidden, read)

    def get(self, player_id, message_id):
        with self.store.transaction() as db:
            self._player(db, player_id)
            row = self._row(db, message_id)
            self._access(row, player_id)
            return self._message(db, row, player_id)

    def mark_read(self, player_id, message_id):
        """Call only after confirmed full output handoff; reads never imply this."""
        with self.store.transaction() as db:
            self._player(db, player_id)
            row = self._row(db, message_id)
            self._access(row, player_id)
            if row['hidden']:
                raise MessageNotFound("message unavailable")
            db.execute("INSERT OR IGNORE INTO message_reads VALUES(?,?,?)", (player_id, message_id, self._now().isoformat()))

    def _list(self, player_id, scope, area, after, limit, *, operator=False):
        if type(after) is not int or not 0 <= after <= 2**63 - 1 or type(limit) is not int or not 1 <= limit <= 10:
            raise MessageError("cursor must be nonnegative and page size must be 1..10")
        with self.store.transaction() as db:
            if not operator:
                self._player(db, player_id)
            conditions, values = ["m.scope=?", "m.id>?"], [scope, after]
            if not operator:
                conditions.append("m.hidden=0")
            if scope == 'public':
                self._area(area)
                conditions.append("m.area=?")
                values.append(area)
            elif not operator:
                conditions.append("m.owner_id=?")
                values.append(player_id)
            # Read at most 11 bounded rows; the extra header indicates continuation.
            rows = db.execute("SELECT m.*,p.callsign AS author FROM messages m LEFT JOIN players p ON p.id=m.author_id WHERE " + " AND ".join(conditions) + " ORDER BY m.id LIMIT ?", (*values, limit + 1)).fetchall()
            page = rows[:limit]
            return MessagePage(tuple(self._message(db, row, player_id, header=True, operator=operator) for row in page), page[-1]['id'] if len(rows) > limit else None)

    def list_public(self, player_id, area, *, after=0, limit=10):
        return self._list(player_id, 'public', area, after, limit)

    def list_sysop(self, player_id, *, after=0, limit=10):
        return self._list(player_id, 'sysop', None, after, limit)

    def unread_counts(self, player_id):
        with self.store.transaction() as db:
            self._player(db, player_id)
            result = {area: 0 for area in self.areas}
            result['sysop'] = 0
            for row in db.execute("""SELECT m.scope,m.area,COUNT(*) AS count FROM messages m
                    WHERE m.hidden=0 AND (m.scope='public' OR m.owner_id=?)
                    AND NOT EXISTS(SELECT 1 FROM message_reads r WHERE r.message_id=m.id AND r.player_id=?)
                    GROUP BY m.scope,m.area""", (player_id, player_id)):
                key = 'sysop' if row['scope'] == 'sysop' else row['area']
                if key in result:
                    result[key] = row['count']
            return result

    def post_public(self, player_id, area, subject, body, request_key, *, parent_id=None):
        self._area(area)
        return self._post(player_id, 'public', area, None, subject, body, request_key, parent_id)

    def post_sysop(self, player_id, subject, body, request_key, *, parent_id=None):
        return self._post(player_id, 'sysop', None, player_id, subject, body, request_key, parent_id)

    def _post(self, player_id, scope, area, owner_id, subject, body, request_key, parent_id, *, operator=False):
        subject, body = _text(subject, 72), _text(body, 2048, multiline=True)
        if not isinstance(request_key, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_key):
            raise MessageError("invalid submission request key")
        if parent_id is not None:
            _identifier(parent_id)
        fingerprint = hashlib.sha256(json.dumps([scope, area, subject, body, parent_id], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        with self.store.transaction() as db:
            if not operator:
                self._player(db, player_id)
            duplicate = db.execute("SELECT message_id,fingerprint FROM message_requests WHERE request_key=?", (request_key,)).fetchone()
            if duplicate:
                row = self._row(db, duplicate['message_id'])
                # Check identity as well as content, including after additive merges.
                if duplicate['fingerprint'] != fingerprint or row['author_id'] != player_id:
                    raise DuplicateRequest("submission request key already used for another message")
                if not operator:
                    self._access(row, player_id)
                return self._message(db, row, player_id, operator=operator)
            parent = None
            if parent_id is not None:
                parent = self._row(db, parent_id)
                if not operator:
                    self._access(parent, player_id)
                if parent['hidden'] or parent['scope'] != scope or parent['area'] != area or parent['owner_id'] != owner_id:
                    raise MessageNotFound("reply parent unavailable")
            if operator and (scope != 'sysop' or parent is None):
                raise MessageError("operator replies require an existing sysop thread")
            now = self._now()
            day = now.date().isoformat()
            prior = db.execute("SELECT value FROM metadata WHERE key='message_day_highwater'").fetchone()
            if prior:
                day = max(day, prior[0])
            if not operator:
                used = db.execute("SELECT COUNT(*) FROM messages WHERE author_id=? AND post_day=?", (player_id, day)).fetchone()[0]
                if used >= self.posts_per_day:
                    raise MessageLimit("daily message posting allowance exhausted (resets at UTC midnight)")
            if db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] >= MAX_MESSAGES:
                raise MessageLimit("message store capacity reached")
            db.execute("INSERT INTO metadata VALUES('message_day_highwater',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (day,))
            cursor = db.execute("""INSERT INTO messages(scope,area,owner_id,author_id,subject,body,parent_id,thread_id,created_at,post_day)
                VALUES(?,?,?,?,?,?,?,?,?,?)""", (scope, area, owner_id, player_id, subject, body, parent_id, parent['thread_id'] if parent else None, now.isoformat(), day))
            message_id = cursor.lastrowid
            if parent is None:
                db.execute("UPDATE messages SET thread_id=? WHERE id=?", (message_id, message_id))
            db.execute("INSERT INTO message_requests VALUES(?,?,?)", (request_key, message_id, fingerprint))
            # Composition itself establishes the author's knowledge of this
            # exact body; other readers still need confirmed full delivery.
            if operator:
                db.execute("INSERT INTO operator_message_reads VALUES(?,?)", (message_id, now.isoformat()))
            else:
                db.execute("INSERT INTO message_reads VALUES(?,?,?)", (player_id, message_id, now.isoformat()))
            return self._message(db, self._row(db, message_id), player_id, operator=operator)


class OperatorMessages:
    """Explicit offline operator capability; never authorize from a callsign."""
    def __init__(self, store, areas=None, *, utcnow=None):
        self.messages = MessageStore(store, areas, utcnow=utcnow)
        self.store = store

    def list_sysop(self, *, after=0, limit=10):
        return self.messages._list(None, 'sysop', None, after, limit, operator=True)

    def list_public(self, area, *, after=0, limit=10):
        return self.messages._list(None, 'public', area, after, limit, operator=True)

    def mark_read(self, message_id):
        """Record the operator's confirmed read, independently of caller receipts."""
        with self.store.transaction() as db:
            self.messages._row(db, message_id)
            db.execute("INSERT OR IGNORE INTO operator_message_reads VALUES(?,?)", (message_id, self.messages._now().isoformat()))

    def get(self, message_id):
        with self.store.transaction() as db:
            return self.messages._message(db, self.messages._row(db, message_id), None, operator=True)

    def reply_sysop(self, parent_id, body, request_key, subject=None):
        # _post revalidates the immutable parent in its mutation transaction.
        parent = self.get(parent_id)
        return self.messages._post(None, 'sysop', None, parent.owner_id,
                                   parent.subject if subject is None else subject,
                                   body, request_key, parent_id, operator=True)

    def hide(self, message_id):
        self._visibility(message_id, True)

    def unhide(self, message_id):
        self._visibility(message_id, False)

    def _visibility(self, message_id, hidden):
        with self.store.transaction() as db:
            self.messages._row(db, message_id)
            db.execute("UPDATE messages SET hidden=? WHERE id=?", (int(hidden), message_id))
