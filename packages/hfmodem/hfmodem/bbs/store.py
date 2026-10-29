# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Exclusive BBS ownership and durable player/usage state, separate from saves."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import fcntl
import json
import os
import sqlite3
import threading
import uuid

from .config import normalize_address


class OwnershipError(RuntimeError):
    pass


class OwnerLock:
    """OS lock; inherited descriptors retain ownership after the parent dies.

    Pass ``fileno()`` in subprocess ``pass_fds`` for every writable backend.
    Never unlink lock files, and never explicitly unlock a shared description.
    """
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            os.close(self._fd)
            self._fd = None
            if isinstance(exc, BlockingIOError):
                raise OwnershipError(f"BBS/world already owned: {self.path}") from exc
            raise

    def fileno(self):
        if self._fd is None:
            raise ValueError("closed ownership lock")
        return self._fd

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


WorldLock = OwnerLock
DEFAULT_PREFERENCES = {"expert": True, "page_length": 20, "pause": True, "color": "plain"}


@dataclass(frozen=True)
class Player:
    id: str
    callsign: str
    preferences: dict


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = OwnerLock(self.root / "owner.lock")
        self._mutex = threading.RLock()
        self._usage_active = lambda: False
        self.db = None
        try:
            self.db = sqlite3.connect(self.root / "players.sqlite", check_same_thread=False, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError("unsupported BBS player schema")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            if version == 1:
                self._backup_before_v2()
            if version < 2:
                with self.transaction() as db:
                    if version == 0:
                        statements = (
                            "CREATE TABLE players(id TEXT PRIMARY KEY, callsign TEXT UNIQUE NOT NULL, preferences TEXT NOT NULL)",
                            "CREATE TABLE identities(address TEXT PRIMARY KEY, player_id TEXT NOT NULL REFERENCES players(id))",
                            "CREATE TABLE observations(player_id TEXT NOT NULL REFERENCES players(id), address TEXT NOT NULL, mode TEXT NOT NULL, PRIMARY KEY(player_id,address,mode))",
                            "CREATE TABLE usage(player_id TEXT NOT NULL REFERENCES players(id), day TEXT NOT NULL, charged_us INTEGER NOT NULL CHECK(charged_us>=0), PRIMARY KEY(player_id,day))",
                            "CREATE TABLE reservations(id TEXT PRIMARY KEY, lease_id TEXT NOT NULL, player_id TEXT NOT NULL REFERENCES players(id), day TEXT NOT NULL, start REAL NOT NULL, deadline REAL NOT NULL, charged_us INTEGER NOT NULL, finalized INTEGER NOT NULL DEFAULT 0)",
                            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                        )
                        for statement in statements:
                            db.execute(statement)
                    self._migrate_v2(db)
                    db.execute("PRAGMA user_version=2")
        except BaseException:
            self.close()
            raise

    def _backup_before_v2(self):
        path = self.root / "players.sqlite.pre-v2.backup"
        if path.exists():
            return
        temporary = self.root / (".migration-backup-" + uuid.uuid4().hex)
        try:
            target = sqlite3.connect(temporary)
            try:
                self.db.backup(target)
            finally:
                target.close()
            os.chmod(temporary, 0o600)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _migrate_v2(db):
        # Individual execute calls stay inside the caller's transaction;
        # executescript would implicitly commit and break migration atomicity.
        statements = (
            """CREATE TABLE messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL CHECK(scope IN ('public','sysop')),
                area TEXT,
                owner_id TEXT REFERENCES players(id),
                author_id TEXT REFERENCES players(id),
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                parent_id INTEGER REFERENCES messages(id),
                thread_id INTEGER REFERENCES messages(id),
                created_at TEXT NOT NULL,
                post_day TEXT NOT NULL,
                hidden INTEGER NOT NULL DEFAULT 0 CHECK(hidden IN (0,1)),
                CHECK((scope='public' AND area IS NOT NULL AND owner_id IS NULL)
                   OR (scope='sysop' AND area IS NULL AND owner_id IS NOT NULL)))""",
            "CREATE TABLE message_requests(request_key TEXT PRIMARY KEY, message_id INTEGER NOT NULL UNIQUE REFERENCES messages(id), fingerprint TEXT NOT NULL)",
            "CREATE TABLE message_reads(player_id TEXT NOT NULL REFERENCES players(id), message_id INTEGER NOT NULL REFERENCES messages(id), read_at TEXT NOT NULL, PRIMARY KEY(player_id,message_id))",
            "CREATE TABLE operator_message_reads(message_id INTEGER PRIMARY KEY REFERENCES messages(id), read_at TEXT NOT NULL)",
            "CREATE INDEX messages_public ON messages(scope,area,id)",
            "CREATE INDEX messages_private ON messages(scope,owner_id,id)",
            "CREATE INDEX messages_author_day ON messages(author_id,post_day)",
            "CREATE INDEX messages_thread ON messages(thread_id,id)",
        )
        for statement in statements:
            db.execute(statement)

    @contextmanager
    def transaction(self):
        with self._mutex:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def player(self, address, mode="", protocol_ssid=True, aliases=None):
        canonical = normalize_address(address, protocol_ssid)
        seen = set()
        normalized_aliases = {normalize_address(k): normalize_address(v) for k, v in (aliases or {}).items()}
        while canonical in normalized_aliases:
            if canonical in seen:
                raise ValueError("cyclic alias")
            seen.add(canonical)
            canonical = normalized_aliases[canonical]
        with self.transaction() as db:
            row = db.execute("SELECT p.* FROM identities i JOIN players p ON p.id=i.player_id WHERE i.address=?", (canonical,)).fetchone()
            if row is None:
                ident = uuid.uuid4().hex
                db.execute("INSERT INTO players VALUES(?,?,?)", (ident, canonical, json.dumps(DEFAULT_PREFERENCES)))
                db.execute("INSERT INTO identities VALUES(?,?)", (canonical, ident))
                row = db.execute("SELECT * FROM players WHERE id=?", (ident,)).fetchone()
            # Do not silently abandon an existing identity with spent use when aliases change.
            original = normalize_address(address, protocol_ssid)
            old = db.execute("SELECT player_id FROM identities WHERE address=?", (original,)).fetchone()
            if old is not None and old[0] != row["id"]:
                raise ValueError("alias requires explicit additive player merge")
            db.execute("INSERT OR IGNORE INTO identities VALUES(?,?)", (original, row["id"]))
            db.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?)", (row["id"], address, mode))
            return Player(row["id"], row["callsign"], json.loads(row["preferences"]))

    def preferences(self, player_id):
        with self._mutex:
            row = self.db.execute("SELECT preferences FROM players WHERE id=?", (player_id,)).fetchone()
            if row is None:
                raise KeyError(player_id)
            return {**DEFAULT_PREFERENCES, **json.loads(row[0])}

    def set_preferences(self, player_id, **updates):
        if set(updates) - set(DEFAULT_PREFERENCES):
            raise ValueError("unknown preference")
        for key in ("expert", "pause"):
            if key in updates and type(updates[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        if "page_length" in updates and (type(updates["page_length"]) is not int or not 1 <= updates["page_length"] <= 100):
            raise ValueError("page_length must be 1..100")
        if "color" in updates and updates["color"] not in ("plain", "ansi16"):
            raise ValueError("invalid color preference")
        with self.transaction() as db:
            result = {**self.preferences(player_id), **updates}
            db.execute("UPDATE players SET preferences=? WHERE id=?", (json.dumps(result), player_id))
        return result

    def merge_players(self, target_id, source_id):
        """Offline additive merge; caller must settle current usage first.

        Unfinalized crash reservations remain charged and IDs are preserved.
        Refuse active usage via QuotaManager.merge_players, or close it first.
        Player game files are deliberately not moved or rolled back here.
        """
        if self._usage_active():
            raise OwnershipError("settle active usage before merging players")
        if target_id == source_id:
            return
        with self.transaction() as db:
            for ident in (target_id, source_id):
                if not db.execute("SELECT 1 FROM players WHERE id=?", (ident,)).fetchone():
                    raise KeyError(ident)
            for row in db.execute("SELECT day,charged_us FROM usage WHERE player_id=?", (source_id,)).fetchall():
                db.execute("INSERT INTO usage VALUES(?,?,?) ON CONFLICT(player_id,day) DO UPDATE SET charged_us=charged_us+excluded.charged_us", (target_id, row[0], row[1]))
            db.execute("DELETE FROM usage WHERE player_id=?", (source_id,))
            db.execute("UPDATE reservations SET player_id=? WHERE player_id=?", (target_id, source_id))
            db.execute("UPDATE identities SET player_id=? WHERE player_id=?", (target_id, source_id))
            db.execute("INSERT OR IGNORE INTO observations SELECT ?,address,mode FROM observations WHERE player_id=?", (target_id, source_id))
            db.execute("DELETE FROM observations WHERE player_id=?", (source_id,))
            db.execute("UPDATE messages SET author_id=? WHERE author_id=?", (target_id, source_id))
            db.execute("UPDATE messages SET owner_id=? WHERE owner_id=?", (target_id, source_id))
            db.execute("INSERT OR IGNORE INTO message_reads SELECT ?,message_id,read_at FROM message_reads WHERE player_id=?", (target_id, source_id))
            db.execute("DELETE FROM message_reads WHERE player_id=?", (source_id,))
            db.execute("DELETE FROM players WHERE id=?", (source_id,))

    def transition_quota_timezone(self, new_timezone, wall):
        """Explicit offline operator transition; carries charges, including prepaid.

        Caller must close active usage first and construct a new QuotaManager
        after this transaction. Existing target-day charges are added, never
        replaced; a repeated transition to the current timezone is a no-op.
        """
        from zoneinfo import ZoneInfo
        if self._usage_active():
            raise OwnershipError("settle active usage before timezone transition")
        if wall.tzinfo is None or wall.utcoffset() is None:
            raise ValueError("timezone transition requires aware calendar time")
        target_zone = ZoneInfo(new_timezone)
        with self.transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='quota_timezone'").fetchone()
            old_timezone = row[0] if row else "UTC"
            if old_timezone == new_timezone:
                return
            old_day = f"{old_timezone}:{wall.astimezone(ZoneInfo(old_timezone)).date().isoformat()}"
            new_day = f"{new_timezone}:{wall.astimezone(target_zone).date().isoformat()}"
            for usage in db.execute("SELECT player_id,charged_us FROM usage WHERE day=?", (old_day,)).fetchall():
                db.execute("INSERT INTO usage VALUES(?,?,?) ON CONFLICT(player_id,day) DO UPDATE SET charged_us=charged_us+excluded.charged_us", (usage[0], new_day, usage[1]))
            db.execute("INSERT INTO metadata VALUES('quota_timezone',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (new_timezone,))

    def close(self):
        with self._mutex:
            if self.db is not None:
                self.db.close()
                self.db = None
            self.owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
