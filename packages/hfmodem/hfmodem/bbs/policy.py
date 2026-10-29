# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Monotonic connection leases and durable, bounded daily door reservations."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import math
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from .config import Limits


class PolicyDenied(RuntimeError):
    pass


class StalePolicy(PolicyDenied):
    pass


@dataclass
class ConnectionLease:
    limits: Limits
    started: float
    token: str
    selected: bool = True
    last_activity: float = 0.0
    busy: bool = False
    busy_started: float | None = None

    @classmethod
    def start(cls, limits, now, token=None, selected=True):
        return cls(limits, now, token or uuid.uuid4().hex, selected, now)

    @property
    def deadline(self):
        return self.started + self.limits.connection_seconds

    def select(self, limits, now):
        if self.selected:
            raise ValueError("connection policy already selected")
        self.limits, self.selected = limits, True
        return not self.expired(now)

    def remaining(self, now):
        return max(0.0, self.deadline - now)

    def expired(self, now):
        return now >= self.deadline

    def accept(self, now):
        if not self.expired(now):
            self.last_activity = now
            if self.busy:
                # Accepted input starts a fresh idle allowance; only the busy
                # interval after that input remains to be excluded.
                self.busy_started = now

    def set_busy(self, busy, now):
        if busy and not self.busy:
            self.busy_started = now
        elif self.busy and not busy:
            if self.busy_started is not None:
                self.last_activity += max(0.0, now - self.busy_started)
            self.busy_started = None
        self.busy = busy

    def idle_expired(self, now):
        return not self.busy and now >= self.last_activity + self.limits.idle_seconds


@dataclass(frozen=True)
class Reservation:
    id: str
    start: float
    deadline: float
    day: str
    seconds: float


@dataclass(frozen=True)
class UsageLease:
    id: str
    player_id: str
    callsign: str
    session_id: str
    run_id: str
    revision: int
    reservations: tuple[Reservation, ...]

    @property
    def start(self):
        return self.reservations[0].start

    @property
    def deadline(self):
        return self.reservations[-1].deadline


@dataclass(frozen=True)
class QuotaStatus:
    used_seconds: float
    remaining_seconds: float
    reset_at: datetime
    timezone: str
    reserved_seconds: float = 0.0


class QuotaManager:
    """Blocking ledger operations belong on the application's persistence worker.

    ``authorized`` is I/O-free. Leases are immutable so an in-flight renewal
    cannot extend the owner's locally known cutoff until its result is applied.
    ``release`` refunds only this manager's known live intervals; after a crash
    old rows stay spent. Game checkpoints have no path to this database.
    """
    def __init__(self, store, config, monotonic=time.monotonic, utcnow=None):
        self.store = store
        self._policy = (1, config)
        self.monotonic = monotonic
        self.utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._active = {}
        with store.transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='quota_timezone'").fetchone()
            if row and row[0] != config.quota_timezone:
                raise ValueError("quota timezone requires explicit offline transition")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('quota_timezone',?)", (config.quota_timezone,))
            db.execute("DELETE FROM metadata WHERE key IN ('clock_suspended','clock_checkpoint','wall_highwater')")
        store._usage_active = lambda: bool(self._active)

    @property
    def config(self):
        return self._policy[1]

    @property
    def revision(self):
        return self._policy[0]

    def _clock(self):
        wall = self.utcnow()
        if wall.tzinfo is None or wall.utcoffset() is None:
            raise ValueError("utcnow must return an aware datetime")
        return self.monotonic(), wall.astimezone(timezone.utc)

    def _day(self, wall):
        zone = ZoneInfo(self.config.quota_timezone)
        local = wall.astimezone(zone)
        reset = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), zone)
        return f"{zone.key}:{local.date().isoformat()}", reset

    def _charged(self, db, player_id, day):
        row = db.execute("SELECT charged_us FROM usage WHERE player_id=? AND day=?", (player_id, day)).fetchone()
        return row[0] if row else 0

    def _unused(self, player_id, day, now):
        return sum(max(0.0, r.deadline - max(now, r.start)) for lease in self._active.values()
                   if lease.player_id == player_id for r in lease.reservations if r.day == day)

    def status(self, player_id, canonical_callsign):
        with self._lock, self.store.transaction() as db:
            now, wall = self._clock()
            day, reset = self._day(wall)
            charged = self._charged(db, player_id, day) / 1_000_000
            reserved = self._unused(player_id, day, now)
            used = max(0.0, charged - reserved)
            limit = self.config.effective_limits(canonical_callsign).door_seconds_per_day
            return QuotaStatus(used, max(0.0, limit - used), reset, self.config.quota_timezone, reserved)

    def _reserve(self, db, lease_id, player_id, callsign, start, now, wall):
        interval_wall = wall + timedelta(seconds=start - now)
        day, reset = self._day(interval_wall)
        limit_us = self.config.effective_limits(callsign).door_seconds_per_day * 1_000_000
        remaining = limit_us - self._charged(db, player_id, day)
        boundary_us = math.floor((reset.astimezone(timezone.utc) - interval_wall).total_seconds() * 1_000_000)
        amount = min(10_000_000, remaining, boundary_us)
        if amount <= 0:
            raise PolicyDenied("daily door allowance exhausted")
        seconds = amount / 1_000_000
        reservation = Reservation(uuid.uuid4().hex, start, start + seconds, day, seconds)
        db.execute("INSERT INTO usage VALUES(?,?,?) ON CONFLICT(player_id,day) DO UPDATE SET charged_us=charged_us+excluded.charged_us", (player_id, day, amount))
        db.execute("INSERT INTO reservations VALUES(?,?,?,?,?,?,?,0)", (reservation.id, lease_id, player_id, day, start, reservation.deadline, amount))
        return reservation

    def admit(self, player_id, canonical_callsign, session_id, run_id):
        with self._lock:
            if any(item.player_id == player_id for item in self._active.values()):
                raise PolicyDenied("player already has active door usage")
            lease_id = uuid.uuid4().hex
            revision = self.revision
            with self.store.transaction() as db:
                now, wall = self._clock()
                reservation = self._reserve(db, lease_id, player_id, canonical_callsign, now, now, wall)
            lease = UsageLease(lease_id, player_id, canonical_callsign, session_id, run_id, revision, (reservation,))
            self._active[lease.id] = lease
            if revision != self.revision:
                self._release_at(lease, lease.start)
                raise StalePolicy("policy changed during admission")
            if self.monotonic() >= lease.deadline:
                self._release_at(lease, lease.start)
                raise PolicyDenied("admission completed after paid interval expired")
            return lease

    def authorized(self, lease):
        # Do not acquire the ledger mutex: this check must never wait on disk.
        current = self._active.get(lease.id)
        now = self.monotonic()
        return bool(current and current.session_id == lease.session_id and current.run_id == lease.run_id
                    and lease.revision == self.revision
                    and self.config.effective_limits(lease.callsign).door_seconds_per_day > 0
                    and lease.start <= now < lease.deadline)

    def renew(self, lease):
        with self._lock:
            if lease.revision != self.revision or lease.id not in self._active:
                raise StalePolicy("obsolete quota lease")
            current = self._active[lease.id]
            now = self.monotonic()
            if now >= current.deadline:
                raise PolicyDenied("paid door interval expired")
            reservations = tuple(r for r in current.reservations if r.deadline > now)
            if len(reservations) >= 2:
                return current
            with self.store.transaction() as db:
                now, wall = self._clock()
                reservation = self._reserve(db, current.id, current.player_id, current.callsign, current.deadline, now, wall)
            result = replace(current, reservations=reservations + (reservation,))
            self._active[lease.id] = result
            if lease.revision != self.revision:
                self._release_at(result, min(self.monotonic(), current.deadline))
                raise StalePolicy("policy changed during renewal")
            if self.monotonic() >= current.deadline:
                self._release_at(result, current.deadline)
                raise PolicyDenied("renewal completed after previous paid deadline")
            return result

    def release(self, lease, stopped_at=None):
        now = self.monotonic()
        # Owner captures revocation before queued persistence/cleanup work.
        cutoff = now if stopped_at is None else max(lease.start, min(float(stopped_at), now))
        return self._release_at(lease, cutoff)

    def _release_at(self, lease, now):
        with self._lock:
            current = self._active.get(lease.id)
            if current is None:
                return
            # Fence gameplay before a potentially slow/failing refund.
            del self._active[lease.id]
            with self.store.transaction() as db:
                for row in db.execute("SELECT * FROM reservations WHERE lease_id=? AND finalized=0", (lease.id,)).fetchall():
                    elapsed_us = math.ceil(max(0.0, min(now, row['deadline']) - row['start']) * 1_000_000)
                    used = min(row['charged_us'], elapsed_us)
                    refund = row['charged_us'] - used
                    db.execute("UPDATE usage SET charged_us=charged_us-? WHERE player_id=? AND day=?", (refund, row['player_id'], row['day']))
                    db.execute("UPDATE reservations SET finalized=1, charged_us=? WHERE id=?", (used, row['id']))

    def reload(self, config):
        if config.quota_timezone != self.config.quota_timezone:
            raise ValueError("quota timezone requires explicit offline transition")
        # Revision is changed before waiting for the disk mutex: stale results
        # can never enable play after the operator has revoked access.
        self._policy = (self.revision + 1, config)
        return self.revision

    def merge_players(self, target_id, source_id):
        with self._lock:
            if any(lease.player_id in (target_id, source_id) for lease in self._active.values()):
                raise PolicyDenied("settle active usage before merging players")
            self.store.merge_players(target_id, source_id)
