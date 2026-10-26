# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import threading
import time
import uuid

from . import wire


def utc():
    return datetime.now(timezone.utc)


def timestamp(value):
    return value.isoformat().replace('+00:00', 'Z')


class Refused(RuntimeError):
    def __init__(self, reason, detail='', holder=None):
        super().__init__(detail or reason)
        self.reason, self.holder = reason, holder


@dataclass(frozen=True)
class Session:
    id: str
    kind: str
    lane: str
    attended: bool
    directory: str
    started_utc: str
    client: str
    authority: tuple
    holds_rig: bool = True
    state: str = 'starting'
    outcome: str | None = None
    ended_utc: str | None = None
    returncode: int | None = None
    verdict: dict | None = None
    hand_stopped: bool | None = None
    holder_pid: int | None = None
    station_url: str | None = None
    incarnation: str | None = None
    recording_origin_index: int | None = None

    def record(self):
        result = asdict(self)
        result['authority'] = list(self.authority)
        return result


class Registry:
    def __init__(self, *, records=None, publish=None, clock=time.monotonic, wall=utc,
                 authority=None, available=None):
        self.lock = threading.RLock()
        self._writes = threading.RLock()
        self.warnings = []
        self.records = records
        self.publish = publish or (lambda type, payload: None)
        self.clock, self.wall = clock, wall
        self.authority = authority or (lambda id: (id, 0))
        self.available = available or (lambda: True)
        self.accepting = False
        self._sessions = {}
        self._control = None
        self._generation = 0
        self._expires = 0
        self._recovered = False
        self.provenance = {}

    def _owner(self):
        return next((s for s in tuple(self._sessions.values())
                     if s.holds_rig and s.state != 'ended'), None)

    def _changed(self):
        owner = self._owner()
        self.publish('owner', {'owner': owner.record() if owner else None,
                               'control': deepcopy(self._control)})

    def _prune(self):
        finished = sorted((s for s in self._sessions.values() if s.state == 'ended'),
                          key=lambda s: (s.ended_utc or '', s.id))
        for session in finished[:-100]:
            del self._sessions[session.id]

    def warning(self, detail):
        self.warnings.append(detail)
        self.publish('warning', {'reason': 'session-record', 'detail': detail})

    def _save(self, session, *, best_effort=False):
        # Callers serialize mutations with _writes; disk never holds the registry lock.
        record = session.record()
        if self.records is not None:
            try:
                self.records.write(record)
            except Exception as exc:
                if not best_effort:
                    raise
                self.warning(f'{session.id}: {exc}')
        with self.lock:
            # Teardown can mark stopping while an admission write is in flight.
            if session.state in ('starting', 'running') and not self.accepting:
                session = replace(session, state='stopping')
                record = session.record()
            self._sessions[session.id] = session
            self._prune()
            self.publish('session', record)
            self._changed()
        return deepcopy(session)

    def recover(self):
        with self._writes:
            if self._recovered:
                return
            for record in self.records.read(warning=self.warning) if self.records is not None else ():
                record['authority'] = tuple(record['authority'])
                session = Session(**record)
                if session.state != 'ended':
                    session = replace(session, state='ended', outcome='interrupted',
                                      ended_utc=timestamp(self.wall()), returncode=1,
                                      hand_stopped=False,
                                      verdict={'line': 'unconfirmed', 'reason': 'station gone'})
                    self._save(session, best_effort=True)
                else:
                    with self.lock:
                        self._sessions[session.id] = session
                        self._prune()
            self._recovered = True

    def admit(self):
        with self.lock:
            self.accepting = True

    def start(self, *, kind, lane, attended, directory, client, holds_rig=True,
              recording_origin_index=None):
        with self._writes:
            if not self.accepting or not self.available():
                raise Refused(wire.REFUSAL_TRANSITIONING, 'station is not accepting sessions')
            if holds_rig:
                self.check_holder()
            id = uuid.uuid4().hex
            session = Session(id=id, kind=kind, lane=lane, attended=attended,
                              directory=str(directory), client=client,
                              started_utc=timestamp(self.wall()), authority=self.authority(id),
                              holds_rig=holds_rig, recording_origin_index=recording_origin_index,
                              **self.provenance)
            return self._save(session)

    def check_holder(self):
        with self.lock:
            owner = self._owner()
            if owner is not None:
                raise Refused(wire.REFUSAL_CLAIMED,
                              f'rig held by {owner.kind} session {owner.id} ({owner.client})',
                              owner.record())

    def running(self, id):
        with self._writes:
            session = self._sessions[id]
            if session.state != 'starting':
                raise ValueError(f'cannot run a {session.state} session')
            return self._save(replace(session, state='running'))

    def suspend(self, *, persist=True):
        with self.lock:
            self.accepting = False
            for id, session in list(self._sessions.items()):
                if session.state in ('starting', 'running'):
                    self._sessions[id] = replace(session, state='stopping')
        if persist:
            self.persist_stopping()

    def persist_stopping(self):
        with self._writes:
            for session in list(self._sessions.values()):
                if session.state == 'stopping':
                    self._save(session, best_effort=True)

    def end(self, id, *, outcome, returncode=0, verdict=None, hand_stopped=False,
            best_effort=False):
        with self._writes:
            session = self._sessions.get(id)
            # A session pruned from history has long since ended.
            if session is None or session.state == 'ended':
                return deepcopy(session)
            return self._save(replace(session, state='ended', outcome=outcome,
                                      ended_utc=timestamp(self.wall()), returncode=returncode,
                                      verdict=deepcopy(verdict), hand_stopped=hand_stopped),
                              best_effort=best_effort)

    def finish_all(self, **outcome):
        with self._writes:
            for session in list(self._sessions.values()):
                if session.state != 'ended':
                    self.end(session.id, best_effort=True, **outcome)

    def attended(self, id):
        # The arbiter calls this under admission; never invert that lock with the registry.
        session = self._owner() if id is None else self._sessions.get(id)
        return bool(session and session.attended and session.state in ('starting', 'running'))

    def expire(self):
        with self.lock:
            if self._control is not None and self.clock() >= self._expires:
                self._control = None
                self._changed()

    def control(self, client):
        with self.lock:
            self._generation += 1
            now = self.wall()
            self._expires = self.clock() + 30
            self._control = dict(client=client, control_id=uuid.uuid4().hex,
                                 generation=self._generation, since_utc=timestamp(now),
                                 expires_utc=timestamp(now + timedelta(seconds=30)))
            self._changed()
            return deepcopy(self._control)

    def authorize(self, command):
        if command.get('kind') in ('stop', 'shutdown'):
            return
        with self.lock:
            self.expire()
            if self._control is None and command.get('kind') not in ('heartbeat', 'ack', 'control'):
                return
            generation = command.get('generation')
            if (type(generation) is int and 0 < generation < self._generation
                    and command.get('control_id')):
                raise Refused(wire.REFUSAL_SUPERSEDED)
            control = self._control
            if (control is None or type(generation) is not int
                    or any(command.get(key) != control[key]
                           for key in ('client', 'control_id', 'generation'))):
                raise Refused(wire.REFUSAL_NOT_CONTROLLER)

    def heartbeat(self, command):
        with self.lock:
            self.authorize(command | {'kind': 'heartbeat'})
            self._expires = self.clock() + 30
            self._control['expires_utc'] = timestamp(self.wall() + timedelta(seconds=30))
            self._changed()
            return deepcopy(self._control)

    def snapshot(self):
        with self.lock:
            self.expire()
            owner = self._owner()
            return dict(owner=owner.record() if owner else None,
                        session=owner.record() if owner else None,
                        sessions=[s.record() for s in self._sessions.values()],
                        control=deepcopy(self._control))
