# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from types import MappingProxyType

from .broadcast import Broadcast, encode_ui
from . import mail_storage
from .mail_storage import durable_mkdir


PENDING = frozenset(('queued', 'in_flight', 'uncertain'))
TERMINAL = frozenset(('delivered', 'cancelled', 'expired'))
STORE_MAX_BYTES = 32 * 1024 * 1024
ENVELOPE_MAX_BYTES = 8192
NOTICE_COOLDOWN_S = 1800
LIFECYCLE_RESERVE_BYTES = 2048
COOLDOWN_RESERVE_BYTES = 512
ERROR_MAX_BYTES = 512
DEFAULT_SETTINGS = dict(parking=True, relay=True, expiry_days=30,
                        per_destination=100, total=1000)


class ParkingDisabled(ValueError):
    pass


def parking_address(value):
    from ..kestrel.rx.broadcast import packetize
    if not isinstance(value, str):
        raise ValueError('Parking requires a radio callsign')
    match = re.fullmatch(r'([A-Z0-9]{1,7})(?:-([0-9]{1,2}))?', value.upper())
    if match is None or match[1] in ('ALL', 'CQ'):
        raise ValueError('Parking requires an individual radio callsign')
    base, ssid = match[1], int(match[2] or 0)
    if not any(c.isalpha() for c in base) or not any(c.isdigit() for c in base):
        raise ValueError('Parking requires an individual radio callsign')
    call = base + (f'-{ssid}' if ssid else '')
    encode_ui(call, call, b'<SMP>', command=1)
    packetize(Broadcast(1, call, call, b'<SMP>'))
    return call


def validate_content(subject, body, *, local=False):
    if not isinstance(subject, str) or not isinstance(body, str):
        raise ValueError('VMail subject and body must be text')
    if (local and not subject) or any(c in subject for c in '<>\r\n') or any(c in body for c in '<>'):
        raise ValueError('A VMail needs a one-line subject; no evidenced escaping for < or >')
    normalized = body.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\r\n')
    from .varac import VMAIL_MAX
    if len(subject) + len(normalized) > VMAIL_MAX:
        raise ValueError(f'VMail subject plus body exceeds the {VMAIL_MAX}-character admission limit')


def _utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _key(owner, sender, recipient, composed, subject, body):
    if composed is None:
        return None
    fields = [owner, sender, recipient, composed, subject, body]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def _notice_key(owner, recipient):
    return parking_address(owner) + '/' + parking_address(recipient)


def _encoded(data):
    return (json.dumps(data, ensure_ascii=False, indent=1) + '\n').encode('utf-8')


def _error_text(error):
    return str(error).encode('utf-8')[:ERROR_MAX_BYTES].decode('utf-8', errors='ignore')


class ParkedStore:
    def __init__(self, root, *, clock=time.time, monotonic=time.monotonic):
        self.path = Path(root) / 'parked-mail.json'
        self.clock, self.monotonic = clock, monotonic
        self.error = None
        self._lock = None
        self._contacts = {}
        self._deadlines = {}
        self._data = dict(version=1, next_id=1, settings=dict(DEFAULT_SETTINGS),
                          records=[], dedup={}, cooldowns={})
        try:
            durable_mkdir(self.path.parent)
            self._lock = (self.path.parent / 'parked-mail.lock').open('a+b')
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                if self.path.stat().st_size > STORE_MAX_BYTES:
                    raise ValueError('parking snapshot exceeds 32 MiB')
                data = json.loads(self.path.read_text())
                self._validate(data)
                self._data = data
            for key, notice in self._data['cooldowns'].items():
                self._deadlines[key] = monotonic() + max(0, notice['next_eligible'] - clock())
            self._reindex()
            recovered = copy.deepcopy(self._data)
            for row in recovered['records']:
                if row['state'] == 'in_flight':
                    row['state'] = 'cancelled' if row['cancel_requested'] else 'uncertain'
                    row['last_error'] = 'Process ended before application acknowledgement'
                    if row['cancel_requested']:
                        row['cancelled_utc'] = _utc(clock())
            if recovered != self._data:
                self._commit(recovered)
            self.expire()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.error = str(error)
            self.close()
            try:
                if self.path.stat().st_size > STORE_MAX_BYTES:
                    raise ValueError('parking snapshot exceeds 32 MiB')
                settings = json.loads(self.path.read_text())['settings']
                self._settings(settings)
                self._data['settings'] = settings
            except (OSError, ValueError, TypeError, KeyError):
                pass

    @property
    def available(self):
        return self.error is None and self._lock is not None

    @property
    def settings(self):
        return dict(self._data['settings'])

    @property
    def records(self):
        self._require()
        return copy.deepcopy(self._data['records'])

    def _require(self):
        if not self.available:
            raise OSError('Parking store unavailable: ' + (self.error or 'closed'))

    @staticmethod
    def _settings(settings):
        if not isinstance(settings, dict) or settings.keys() != DEFAULT_SETTINGS.keys():
            raise ValueError('invalid parking settings')
        if any(type(settings[k]) is not bool for k in ('parking', 'relay')):
            raise ValueError('invalid parking switches')
        if any(type(settings[k]) is not int or settings[k] < 1
               for k in ('expiry_days', 'per_destination', 'total')):
            raise ValueError('parking limits must be positive integers')

    @classmethod
    def _validate(cls, data):
        from .varac import canonical_call
        if (not isinstance(data, dict) or type(data.get('version')) is not int
                or data['version'] != 1):
            raise ValueError('unsupported parking snapshot schema')
        cls._settings(data['settings'])
        if (type(data['next_id']) is not int or data['next_id'] < 1
                or not isinstance(data['records'], list)
                or not isinstance(data['dedup'], dict)
                or not isinstance(data['cooldowns'], dict)):
            raise ValueError('invalid parking snapshot')
        ids, dedup = set(), {}
        for row in data['records']:
            if not isinstance(row, dict) or type(row.get('id')) is not int:
                raise ValueError('invalid parking record')
            if row['id'] in ids or not 0 < row['id'] < data['next_id']:
                raise ValueError('invalid parking record ID')
            ids.add(row['id'])
            for name in ('owner', 'from', 'to', 'via'):
                address = canonical_call if row.get('origin') == 'direct' else parking_address
                if address(row[name]) != row[name]:
                    raise ValueError('noncanonical parking address')
            validate_content(row['subject'], row['body'])
            composed = row['composed_utc']
            if composed is not None:
                datetime.strptime(composed, '%Y-%m-%d %H:%M:%S')
            for name in ('received_unix', 'expires_unix'):
                if type(row[name]) not in (int, float) or not math.isfinite(row[name]):
                    raise ValueError('invalid parking timestamp')
            for name in ('received_utc', 'estimated_composed_utc'):
                datetime.strptime(row[name], '%Y-%m-%dT%H:%M:%SZ')
            for name in ('delivered_utc', 'cancelled_utc', 'expired_utc'):
                if row[name] is not None:
                    datetime.strptime(row[name], '%Y-%m-%dT%H:%M:%SZ')
            if (row['origin'] not in ('local', 'remote', 'direct')
                    or row['state'] not in PENDING | TERMINAL
                    or type(row['attempts']) is not int or row['attempts'] < 0
                    or type(row['cancel_requested']) is not bool
                    or not isinstance(row['last_error'], (str, type(None)))
                    or not isinstance(row['notification'], dict)):
                raise ValueError('invalid parking record state')
            if row['notification']:
                cls._validate_notice(row['notification'])
            key = _key(row['owner'], row['from'], row['to'], composed,
                       row['subject'], row['body'])
            if row['dedup_key'] != key or key is not None and key in dedup:
                raise ValueError('invalid parking deduplication identity')
            if key is not None:
                dedup[key] = row['id']
        if data['dedup'] != dedup:
            raise ValueError('invalid parking deduplication index')
        for key, notice in data['cooldowns'].items():
            owner, recipient = key.split('/')
            if key != _notice_key(owner, recipient) or not isinstance(notice, dict):
                raise ValueError('invalid parking notification identity')
            cls._validate_notice(notice)

    @staticmethod
    def _validate_notice(notice):
        if set(notice) != {'last_attempt', 'next_eligible', 'last_sent'}:
            raise ValueError('invalid parking notification fields')
        for field in ('last_attempt', 'next_eligible', 'last_sent'):
            value = notice[field]
            if value is None and field == 'last_sent':
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('invalid parking notification timestamp')
        if notice['next_eligible'] < notice['last_attempt'] + NOTICE_COOLDOWN_S:
            raise ValueError('invalid parking notification cooldown')

    def _commit(self, data):
        self._require()
        encoded = _encoded(data)
        if len(encoded) > STORE_MAX_BYTES:
            raise ValueError('Parking snapshot is full (32 MiB); accepted mail retained')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.parked-mail-',
                                             delete=False) as out:
                temporary = Path(out.name)
                out.write(encoded)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.path)
            mail_storage.sync_directory(self.path.parent)
        except OSError as error:
            # A failed directory fsync can follow a successful replace. Only a
            # fresh locked load can determine which snapshot is now authoritative.
            self.error = str(error)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._data = data
        self._reindex()

    def _reindex(self):
        self._by_id = {row['id']: row for row in self._data['records']}
        self._pending = {}
        self._expires = float('inf')
        for row in sorted(self._data['records'], key=lambda row: (row['received_unix'], row['id'])):
            if row['state'] in PENDING:
                view = MappingProxyType({**row, 'notification': MappingProxyType(row['notification'])})
                self._pending.setdefault((row['owner'], None), []).append(view)
                self._pending.setdefault((row['owner'], row['to']), []).append(view)
                self._expires = min(self._expires, row['expires_unix'])
        self._pending = {key: tuple(rows) for key, rows in self._pending.items()}

    def configure(self, **settings):
        self._require()
        data = copy.deepcopy(self._data)
        data['settings'].update(settings)
        self._settings(data['settings'])
        self._commit(data)

    def accept(self, *, owner, sender, recipient, via, subject, body, composed=None,
               origin='remote', contact=None):
        from .varac import VMail, canonical_call
        self._require()
        address = canonical_call if origin == 'direct' else parking_address
        owner, sender, recipient, via = map(address, (owner, sender, recipient, via))
        if owner == recipient:
            raise ValueError('Mail for this station belongs in its inbox, not the parking queue')
        if origin not in ('local', 'remote', 'direct'):
            raise ValueError('invalid mail origin')
        validate_content(subject, body, local=origin != 'remote')
        if composed is not None:
            datetime.strptime(composed, '%Y-%m-%d %H:%M:%S')
        onward = VMail(subject, body, sender=sender if origin == 'remote' else None,
                       composed=composed).wire()
        if len(onward.encode('utf-8')) > ENVELOPE_MAX_BYTES:
            raise ValueError('VMail envelope exceeds the 8 KiB safety limit')
        key = _key(owner, sender, recipient, composed, subject, body)
        fields = (owner, sender, recipient, composed, subject, body)
        duplicate = self._data['dedup'].get(key) if key is not None else None
        if duplicate is None and contact is not None:
            duplicate = self._contacts.get((contact, fields))
        if duplicate is not None:
            previous = self.get(duplicate)
            if fields != tuple(previous[name] for name in
                               ('owner', 'from', 'to', 'composed_utc', 'subject', 'body')):
                raise ValueError('Parking deduplication hash collision; custody refused')
            return previous, True
        if origin == 'remote' and not self.settings['parking']:
            raise ParkingDisabled('New third-party parking is disabled')
        self.expire()
        pending = [row for row in self._data['records'] if row['state'] in PENDING]
        if len(pending) >= self.settings['total']:
            raise ValueError('Parking queue total quota reached')
        if sum(row['owner'] == owner and row['to'] == recipient for row in pending) >= self.settings['per_destination']:
            raise ValueError('Parking queue destination quota reached')
        now = self.clock()
        row = dict(id=self._data['next_id'], owner=owner, **{'from': sender, 'to': recipient},
                   via=via, subject=subject, body=body, composed_utc=composed,
                   received_utc=_utc(now), received_unix=now, estimated_composed_utc=_utc(now),
                   expires_unix=now + self.settings['expiry_days'] * 86400,
                   origin=origin, state='queued', attempts=0, last_error=None,
                   delivered_utc=None, cancelled_utc=None, expired_utc=None,
                   cancel_requested=False, dedup_key=key, notification={})
        data = copy.deepcopy(self._data)
        data['next_id'] += 1
        data['records'].append(row)
        if key is not None:
            data['dedup'][key] = row['id']
        pending = [row for row in data['records'] if row['state'] in PENDING]
        destinations = {(row['owner'], row['to']) for row in pending}
        reserved = {tuple(key.split('/')) for key in data['cooldowns']}
        reserve = (len(pending) * LIFECYCLE_RESERVE_BYTES
                   + len(destinations - reserved) * COOLDOWN_RESERVE_BYTES)
        # Accepted content must leave space for its attempts, errors, terminal
        # timestamps and notice reservations without relying on destructive purge.
        if len(_encoded(data)) + reserve > STORE_MAX_BYTES:
            raise ValueError('Parking snapshot is full (32 MiB including custody metadata reserve)')
        self._commit(data)
        if contact is not None:
            self._contacts[contact, fields] = row['id']
        return copy.deepcopy(row), False

    def get(self, n):
        self._require()
        row = self._by_id.get(int(n))
        if row is None:
            raise ValueError(f'No parked VMail {n}')
        return copy.deepcopy(row)

    def pending(self, owner, recipient=None, *, snapshot=True):
        from .varac import canonical_call
        self.expire()
        owner = canonical_call(owner)
        recipient = None if recipient is None else canonical_call(recipient)
        rows = self._pending.get((owner, recipient), ())
        if snapshot:
            return [copy.deepcopy(self._by_id[row['id']]) for row in rows]
        return rows

    def expire(self):
        self._require()
        now = self.clock()
        if now < self._expires:
            return
        data = copy.deepcopy(self._data)
        for row in data['records']:
            if row['state'] in PENDING and row['expires_unix'] <= now:
                row['state'], row['expired_utc'] = 'expired', _utc(now)
        self._commit(data)

    def _change(self, n, change):
        self._require()
        data = copy.deepcopy(self._data)
        row = next((row for row in data['records'] if row['id'] == int(n)), None)
        if row is None:
            raise ValueError(f'No parked VMail {n}')
        change(row)
        self._commit(data)
        return copy.deepcopy(row)

    def cancel(self, n):
        def change(row):
            if row['state'] not in PENDING:
                raise ValueError(f"VMail {n} is already {row['state']}")
            row['cancel_requested'] = True
            if row['state'] != 'in_flight':
                row['state'], row['cancelled_utc'] = 'cancelled', _utc(self.clock())
        return self._change(n, change)

    def begin(self, n):
        self.expire()
        def change(row):
            if row['state'] not in ('queued', 'uncertain') or row['cancel_requested']:
                raise ValueError(f'VMail {n} is not eligible to forward')
            row['state'] = 'in_flight'
            row['attempts'] += 1
            row['last_error'] = None
        return self._change(n, change)

    def resolve(self, n, ack):
        if ack not in ('SMR', 'SMF', 'SMFP'):
            raise ValueError('invalid VMail acknowledgement')
        def change(row):
            if (row['state'] not in ('in_flight', 'uncertain')
                    and not (row['state'] == 'cancelled' and row['cancel_requested']
                             and row['attempts'] > 0)):
                raise ValueError(f'VMail {n} has no unresolved attempt')
            if ack == 'SMR':
                row['state'], row['delivered_utc'] = 'delivered', _utc(self.clock())
                row['last_error'] = None
            else:
                row['state'] = 'cancelled' if row['cancel_requested'] else 'queued'
                row['last_error'] = ack
                if row['cancel_requested']:
                    row['cancelled_utc'] = _utc(self.clock())
        return self._change(n, change)

    def fail(self, n, error, *, ambiguous=True):
        def change(row):
            if row['state'] not in PENDING:
                return
            row['state'] = ('cancelled' if row['cancel_requested'] else
                            'uncertain' if ambiguous else 'queued')
            row['last_error'] = _error_text(error)
            if row['cancel_requested']:
                row['cancelled_utc'] = _utc(self.clock())
        return self._change(n, change)

    def notice(self, owner, recipient):
        self._require()
        return copy.deepcopy(self._data['cooldowns'].get(_notice_key(owner, recipient), {}))

    def eligible(self, owner, recipient):
        self._require()
        return self.monotonic() >= self._deadlines.get(_notice_key(owner, recipient), float('-inf'))

    def reserve_notice(self, owner, recipient):
        self._require()
        key = _notice_key(owner, recipient)
        if not self.eligible(owner, recipient):
            raise ValueError('Notification cooldown has not elapsed')
        now = self.clock()
        data = copy.deepcopy(self._data)
        previous = data['cooldowns'].get(key, {})
        data['cooldowns'][key] = dict(last_attempt=now, next_eligible=now + NOTICE_COOLDOWN_S,
                                      last_sent=previous.get('last_sent'))
        for row in data['records']:
            if (row['owner'], row['to']) == tuple(key.split('/')) and row['state'] in PENDING:
                row['notification'] = dict(data['cooldowns'][key])
        self._commit(data)
        self._deadlines[key] = self.monotonic() + NOTICE_COOLDOWN_S

    def notice_sent(self, owner, recipient):
        self._require()
        key = _notice_key(owner, recipient)
        data = copy.deepcopy(self._data)
        if key not in data['cooldowns']:
            raise ValueError('Notification has no durable admission reservation')
        data['cooldowns'][key]['last_sent'] = self.clock()
        for row in data['records']:
            if (row['owner'], row['to']) == tuple(key.split('/')) and row['state'] in PENDING:
                row['notification'] = dict(data['cooldowns'][key])
        self._commit(data)

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None


class MailParking:
    def __init__(self, session, *, wall_clock=time.time):
        self.session = session
        self.store = ParkedStore(session.mailbox.root, clock=wall_clock,
                                 monotonic=session.client.clock)
        self.draft = None
        self.started = None
        self.stopped = None
        self.candidates = {}
        self.notifying = None
        self.clear_since = None
        self.clear_context = None
        self.busy_events = None
        self._unready_notice = False

    @property
    def parking_enabled(self):
        return self.store.settings['parking']

    @property
    def relay_enabled(self):
        return self.store.available and self.store.settings['relay']

    @property
    def active(self):
        s = self.session
        return s.mail_ack.busy or bool(self.started == s.client.link_generation
                                       and self.stopped != self.started and self.pending())

    def pending(self):
        s = self.session
        if not self.store.available or not s.peer:
            return []
        try:
            return self.store.pending(s.client.mycall, s.peer, snapshot=False)
        except OSError as error:
            s.say(f'VMail queue unavailable: {error}')
            return []
        except ValueError:
            return []

    def summary(self):
        s = self.session
        if not self.store.available:
            return f'VMail parking unavailable: {self.store.error}; no new custody or outgoing notices'
        settings = self.store.settings
        try:
            count = len(self.store.pending(s.client.mycall, snapshot=False))
        except (OSError, ValueError) as error:
            return f'VMail parking unavailable: {error}; no new custody or outgoing notices'
        return (f'VMail parking {"on" if settings["parking"] else "off"}; '
              f'relay notices {"on" if settings["relay"] else "off"}; {count} pending; '
              f'expiry {settings["expiry_days"]} days; limits '
              f'{settings["per_destination"]}/destination, {settings["total"]} total; '
              + ('notice band ready' if s.hopping.ready(s.flow.band) else
                 'notices deferred: /band ready required each session'))

    def deposit(self, mail, *, reply=None):
        s, c = self.session, self.session.client
        reply = reply or c.send_tag
        try:
            if len(mail.wire().encode('utf-8')) > ENVELOPE_MAX_BYTES:
                raise ValueError('VMail envelope exceeds 8 KiB')
            record, duplicate = self.store.accept(
                owner=c.mycall, sender=mail.sender or s.peer, recipient=mail.recipient,
                via=s.peer, subject=mail.subject, body=mail.body, composed=mail.composed,
                contact=c.link_generation)
        except ParkingDisabled as error:
            s.say(str(error))
            reply('SMFP')
            return
        except (OSError, ValueError) as error:
            s.say(f'VMail parking refused: {error}')
            reply('SMF')
            return
        reply('SMR')
        s.say(f'Parked VMail {record["id"]} for {record["to"]}: '
              + ('already in durable custody' if duplicate else 'durable custody accepted; not delivered'))
        s.emit('vmail_parked', id=record['id'], duplicate=duplicate, recipient=record['to'])

    def start_delivery(self):
        from .varac import CALLING_HZ, HostError
        s, c = self.session, self.session.client
        if not c.connected or s.flow.dial in CALLING_HZ.values():
            return
        if self.started != c.link_generation:
            self.started = c.link_generation
            self.stopped = None
            try:
                self._deliver(opening=True)
            except (OSError, ValueError, HostError) as error:
                self.stopped = c.link_generation
                s.say(f'VMail forwarding stopped; custody retained: {error}')

    def _outcome(self, n, ack, reason):
        s = self.session
        if ack is None:
            self.store.fail(n, reason)
            self.stopped = s.client.link_generation
        else:
            self.store.resolve(n, ack)
            if ack != 'SMR':
                self.stopped = s.client.link_generation
            if s.mail_ack.kind != 'direct' or ack != 'SMR':
                s.say(f'VMail {n}: {"delivered (SMR)" if ack == "SMR" else ack + "; custody retained"}')

    def _send(self, row, *, compact=False):
        from .varac import HostError, VMail
        s, c = self.session, self.session.client
        s.mail_ack.reserve('direct' if compact else 'parking', row['id'],
                           lambda ack, reason: self._outcome(row['id'], ack, reason))
        try:
            row = self.store.begin(row['id'])
        except (OSError, ValueError):
            s.mail_ack.release_unsent()
            raise
        composed = row['composed_utc']
        composed_time = (datetime.strptime(composed, '%Y-%m-%d %H:%M:%S')
                         .replace(tzinfo=timezone.utc).timestamp() if composed else None)
        if compact and composed_time is not None and 0 <= self.store.clock() - composed_time < 600:
            composed = None
        mail = VMail(row['subject'], row['body'],
                     sender=row['from'] if row['origin'] == 'remote' else None,
                     composed=composed)
        try:
            c.send_vmail(mail)
        except (OSError, ValueError, HostError):
            s.mail_ack.interrupt('write failed after possible admission')
            raise

    def _deliver(self, *, opening=False):
        from .varac import CALLING_HZ
        s, c = self.session, self.session.client
        if (self.started != c.link_generation or self.stopped == c.link_generation
                or not c.connected or c.closed or s._closing or s._quitting
                or s.flow.dial in CALLING_HZ.values() or s.mail_ack.busy
                or c._messages.pending
                or s.bbs.owned or s.files.active or s.bbs_host.active or s._bbs_waiting
                or s.email.draft is not None or s.compose is not None
                or not opening and (s._data_status()[0] or c.data_writer and c.data_writer.pending)):
            return
        pending = self.pending()
        if pending:
            self._send(pending[0])

    def send_direct(self, mail, *, composed=None):
        from .varac import CALLING_HZ, HostError
        s, c = self.session, self.session.client
        if (not c.connected or c.closed or s.mail_ack.busy or s.steps is not None
                or s.bbs.owned or s.files.active or s.bbs_host.active
                or s.email.draft is not None or s.flow.dial in CALLING_HZ.values()):
            raise HostError('VMail requires an off-CF link with no other mail or service exchange')
        composed = composed or datetime.fromtimestamp(self.store.clock(), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        row, duplicate = self.store.accept(owner=c.mycall, sender=c.mycall, recipient=s.peer,
                                           via=c.mycall, subject=mail.subject, body=mail.body,
                                           composed=composed, origin='direct')
        if not duplicate:
            self._send(row, compact=True)

    def local_command(self, line, words):
        if words[:1] != ['/vmail']:
            return False
        s, c = self.session, self.session.client
        args = words[1:]
        if args[:1] in (['parking'], ['relay']):
            if len(args) > 2 or len(args) == 2 and args[1] not in ('on', 'off'):
                raise ValueError('usage: /vmail parking|relay [on|off]')
            if len(args) == 2:
                self.store.configure(**{args[0]: args[1] == 'on'})
                if args == ['relay', 'off']:
                    self.candidates.clear()
                    if self.notifying:
                        s.broadcast.cancel()
            s.say(self.summary())
        elif args[:1] == ['park'] or args[:1] == ['send'] and not c.connected:
            if len(args) < 3:
                raise ValueError('usage: /vmail park CALL SUBJECT (or send CALL SUBJECT while unlinked)')
            target, subject = parking_address(args[1]), ' '.join(args[2:])
            if target == parking_address(c.mycall):
                raise ValueError('Cannot queue mail to the active local callsign')
            validate_content(subject, '', local=True)
            self.draft = dict(owner=parking_address(c.mycall), recipient=target, subject=subject,
                              composed=datetime.fromtimestamp(self.store.clock(), timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                              lines=[])
            s.say(f'Local VMail for {target}; . saves to the queue without transmitting; /stop drops draft')
        elif args[:1] == ['parked']:
            rows = ([self.store.get(int(args[1]))] if len(args) == 2 else self.store.records)
            if len(args) > 2:
                raise ValueError('usage: /vmail parked [N]')
            for row in rows:
                if row['owner'] != parking_address(c.mycall):
                    continue
                s.say(f'Parked {row["id"]}: {row["from"]} via {row["via"]} to {row["to"]}; '
                      f'{row["state"]}; attempts {row["attempts"]}; received {row["received_utc"]}; '
                      f'expires {_utc(row["expires_unix"])}; notice {row["notification"]}; error {row["last_error"]}')
                if len(args) == 2:
                    s.say('composed: ' + (row['composed_utc'] + ' UTC (sender-declared)' if row['composed_utc']
                                          else row['estimated_composed_utc'] + ' (local receive-time estimate; original unknown)'))
                    s.say('subject: ' + row['subject'])
                    s.say(row['body'])
            if not rows:
                s.say('No locally parked VMails')
        elif args[:1] == ['cancel'] and len(args) == 2:
            row = self.store.get(int(args[1]))
            if row['owner'] != parking_address(c.mycall):
                raise ValueError('Parked VMail belongs to another local identity')
            self.store.cancel(row['id'])
            if not self.store.pending(c.mycall, row['to']):
                self.candidates.pop(row['to'], None)
            s.say(f'VMail {row["id"]} cancellation recorded; RF already sent cannot be recalled')
        else:
            return False
        return True

    def line(self, line):
        s = self.session
        if line == '.':
            d = self.draft
            row, _ = self.store.accept(owner=d['owner'], sender=d['owner'], recipient=d['recipient'],
                                       via=d['owner'], subject=d['subject'], body='\r\n'.join(d['lines']),
                                       composed=d['composed'], origin='local')
            self.draft = None
            s.say(f'VMail {row["id"]} queued locally for {row["to"]}; no RF sent')
        elif line.startswith('/'):
            raise ValueError('Local VMail draft active; . saves or /stop drops it')
        else:
            validate_content(self.draft['subject'], '\r\n'.join(self.draft['lines'] + [line]), local=True)
            self.draft['lines'].append(line)

    def heard(self, call, dial):
        s = self.session
        if (not self.relay_enabled or dial != s.calling_hz or dial != s.flow.dial
                or s.client.connected or s.client.pending):
            return
        try:
            call = parking_address(call)
            owner = parking_address(s.client.mycall)
            if s.client.ptt:
                return
            for row in self.store.pending(owner, snapshot=False):
                recipient = row['to']
                if (recipient == call or '-' not in call and recipient.split('-')[0] == call):
                    if self.store.eligible(owner, recipient):
                        self.candidates.setdefault(recipient, (s.client.clock() + 300, dial,
                                                               s.client.link_generation, owner))
                        if not s.hopping.ready(s.flow.band) and not self._unready_notice:
                            s.say('VMail notice deferred: /band ready required each session')
                            self._unready_notice = True
        except (OSError, ValueError) as error:
            s.say(f'VMail notice deferred: {error}')

    def invalidate_notices(self):
        self.candidates.clear()
        self.clear_since = self.clear_context = None
        if self.notifying:
            self.session.broadcast.cancel()

    def notify(self, call):
        from .varac import HostError
        s = self.session
        call = parking_address(call)
        if not self.relay_enabled or not self.store.pending(s.client.mycall, call):
            raise HostError('VMail notice needs relay on and pending mail for this exact callsign')
        if not self.store.eligible(s.client.mycall, call):
            raise HostError('VMail notice cooldown has not elapsed')
        self.candidates[call] = (s.client.clock() + 300, s.flow.dial, s.client.link_generation,
                                 parking_address(s.client.mycall))
        s.say('VMail notice candidate saved for five minutes; awaiting ready band, idle CF and five seconds known clear')

    def _notice_valid(self, owner, recipient, dial, generation):
        s, c = self.session, self.session.client
        return (self.relay_enabled and parking_address(c.mycall) == owner
                and recipient != owner
                and bool(self.store.pending(owner, recipient, snapshot=False))
                and not c.connected and not c.pending and not c.closed
                and c.link_generation == generation and s.flow.dial == dial == s.calling_hz
                and not (s._closing or s._quitting or s._band_fault or s.contact or s.following or s._follow_tuned
                         or s.files.active or s.bbs.active or s.bbs_host.active)
                and s.hopping.ready(s.flow.band) and not s.hopping.cooldown(s.flow.band))

    def _notice_tick(self):
        from .varac import BEACON_CLEAR_S, HostError, State
        s, c = self.session, self.session.client
        now = c.clock()
        context = (s.flow.dial, c.link_generation, c.mycall)
        clear = (c.busy is False and self.busy_events == c.busy_events
                 and context == self.clear_context)
        self.clear_context = context
        self.busy_events = c.busy_events
        self.clear_since = (now if self.clear_since is None else self.clear_since) if clear else None
        if c.connected or c.pending or c.closed or not self.relay_enabled:
            self.candidates.clear()
        if self.notifying:
            if not s.broadcast.active:
                self.notifying = None
            elif not self._notice_valid(*self.notifying):
                s.broadcast.cancel()
            return
        self.candidates = {call: candidate for call, candidate in self.candidates.items()
                           if candidate[0] > now and candidate[1:3] == context[:2]
                           and candidate[3] == parking_address(c.mycall)
                           and self.store.pending(c.mycall, call, snapshot=False)}
        if (not self.candidates or not self.relay_enabled or s.steps is not None
                or s.broadcast.active or s.broadcast.selected is not None
                or s.flow.state is not State.MONITORING or self.clear_since is None
                or now - self.clear_since < BEACON_CLEAR_S):
            return
        owner = parking_address(c.mycall)
        for recipient in list(self.candidates):
            args = (owner, recipient, s.flow.dial, c.link_generation)
            expires = self.candidates[recipient][0]
            if not self._notice_valid(*args) or not self.store.eligible(owner, recipient):
                continue
            def valid():
                return self._notice_valid(*args)
            def before_admit():
                if (not valid() or c.busy is not False or c.ptt or s.line_state() is not False
                        or c.clock() >= expires
                        or c.busy_events != self.busy_events or self.clear_since is None
                        or c.clock() - self.clear_since < BEACON_CLEAR_S):
                    raise HostError('VMail notice eligibility changed before admission')
                self.store.reserve_notice(owner, recipient)
                self.candidates.pop(recipient, None)
            def sent():
                try:
                    self.store.notice_sent(owner, recipient)
                except OSError as error:
                    s.say(f'VMail notice transmitted, but sent status could not be saved; '
                          f'reserved cooldown retained: {error}')
            try:
                s.broadcast.directed(recipient, valid=valid, before_admit=before_admit, sent=sent)
                self.candidates.pop(recipient, None)
                self.notifying = args
            except (OSError, ValueError, HostError) as error:
                s.say(f'VMail notice deferred: {error}')
            break

    def tick(self):
        from .varac import HostError
        try:
            self._deliver()
            self._notice_tick()
        except (OSError, ValueError, HostError) as error:
            self.session.say(f'VMail parking unavailable: {error}')
