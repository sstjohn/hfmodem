# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The QSO log, for any mode: one ADIF record appended per contact.

Serials are per contest id and resume from the highest STX the log, or the
records set aside beside it, already hold for it, so a restarted session never
repeats a number. The file is created, header first, by the first record; a
session that logs nothing leaves it untouched. Each record is one synced write.
A record that cannot be written is set aside in `held` by its owner, for the
next session to write.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import get_args, get_type_hints
import re
import tempfile

_TAG = re.compile(r'<([^:<>]+)(?::(\d+)(?::[^>]*)?)?>')
HEADER = 'hfmodem QSO log\n<ADIF_VER:5>3.1.4<PROGRAMID:7>hfmodem<EOH>\n'


class SessionRecords:
    def __init__(self, directory):
        self.directory = Path(directory)

    def read(self, *, warning=None):
        from .sessions import Session
        types = get_type_hints(Session)
        records = []
        for path in sorted(self.directory.glob('*.json')):
            try:
                text = path.read_text(encoding='utf-8')
            except OSError as exc:
                if warning is not None:
                    warning(f'{path.name}: unreadable record skipped: {exc}')
                continue
            try:
                record = json.loads(text)
                session = Session(**record)
                for field, annotation in types.items():
                    allowed = ((list, tuple) if field == 'authority' else
                               get_args(annotation) or (annotation,))
                    if type(getattr(session, field)) not in allowed:
                        raise ValueError(f'invalid session field: {field}')
                if (record['id'] != path.stem or record['state'] not in
                        ('starting', 'running', 'stopping', 'ended')
                        or not isinstance(record['authority'], (list, tuple))):
                    raise ValueError('invalid session record')
            except (ValueError, TypeError, KeyError) as exc:
                try:
                    path.replace(path.with_name(path.name + '.corrupt'))
                    detail = f'corrupt record moved aside: {exc}'
                except OSError as stuck:
                    detail = f'corrupt record skipped: {exc}; not moved aside: {stuck}'
                if warning is not None:
                    warning(f'{path.name}: {detail}')
                continue
            records.append(record)
        return records

    def write(self, record):
        name = record['id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('invalid session id')
        self.directory.mkdir(parents=True, exist_ok=True)
        spare = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=self.directory, suffix='.tmp', delete=False) as out:
                spare = Path(out.name)
                json.dump(record, out, ensure_ascii=False)
                out.write('\n')
                out.flush()
                os.fsync(out.fileno())
            os.replace(spare, self.directory / f'{name}.json')
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if spare is not None:
                spare.unlink(missing_ok=True)


def read_adif(text: str) -> list[dict[str, str]]:
    """The records of an ADI file, field names upper-cased. The spec counts a
    field's length in characters; a writer that counted UTF-8 bytes instead is
    recognised by the byte reading, and only it, landing on the next tag."""
    records, fields, pos = [], {}, 0
    while m := _TAG.search(text, pos):
        name, n = m[1].strip().upper(), m[2]
        pos = m.end()
        if name == 'EOH':
            fields = {}
        elif name == 'EOR':
            records.append(fields)
            fields = {}
        elif n is not None:
            value = text[pos:pos + int(n)]
            if not re.match(r'\s*(<|$)', text[pos + len(value):]):
                raw = text[pos:pos + int(n)].encode()[:int(n)].decode(errors='ignore')
                if re.match(r'\s*(<|$)', text[pos + len(raw):]):
                    value = raw
            fields[name] = value
            pos += len(value)
    return records


@dataclass(frozen=True)
class Qso:
    call: str
    serial: int
    started: datetime


class QsoLog:
    def __init__(self, path, mycall: str, *, grid: str = '', contest: str | None = None):
        self.path = Path(path)
        self.mycall = mycall.upper()
        self.grid = grid
        self.contest = contest
        self.held = self.path.with_name(self.path.name + '.unwritten')
        self.current: Qso | None = None
        held = [record for path in (self.path, self.held) if path.is_file()
                for record in read_adif(path.read_text(encoding='utf-8'))]
        self.serial = 1 + max((int(r['STX']) for r in held
                               if r.get('CONTEST_ID') == contest and r.get('STX', '').isdigit()),
                              default=0)

    def start(self, call: str) -> Qso:
        self.current = Qso(call.upper(), self.serial, datetime.now(timezone.utc))
        return self.current

    def substitute(self, text: str) -> str:
        return (text.replace('{CALL}', self.current.call if self.current else '')
                .replace('{SERIAL}', f'{self.serial:03d}'))

    def record(self, *, mode: str, freq_hz: int, rst_sent: str, rst_rcvd: str,
               srx: str | None = None) -> tuple[Qso, str]:
        """Close the QSO in progress and advance the serial; the ADIF record is for `write`."""
        qso = self.current
        if qso is None:
            raise ValueError('no QSO in progress to log')
        fields = dict(CALL=qso.call, QSO_DATE=qso.started.strftime('%Y%m%d'),
                      TIME_ON=qso.started.strftime('%H%M%S'), FREQ=f'{freq_hz / 1e6:.6f}',
                      MODE=mode, RST_SENT=rst_sent, RST_RCVD=rst_rcvd,
                      STATION_CALLSIGN=self.mycall, MY_GRIDSQUARE=self.grid)
        if self.contest:
            fields.update(CONTEST_ID=self.contest, STX=str(qso.serial), SRX=srx or '')
        self.serial += 1
        self.current = None
        return qso, ''.join(f'<{name}:{len(value)}>{value}' for name, value in fields.items()
                            if value) + '<EOR>\n'

    def write(self, record: str) -> None:
        fresh = not self.path.is_file()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as out:
            out.write((HEADER if fresh else '') + record)
            out.flush()
            os.fsync(out.fileno())

    def logged(self, record: str) -> bool:
        """Whether the log already holds this record, as a retried or replayed write
        must ask: an append can land before its sync fails. A log that cannot be
        read raises, so the caller keeps the record rather than guess."""
        try:
            return record.strip() in self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return False

    def sync(self) -> None:
        """Make what the log already holds durable."""
        with self.path.open('a', encoding='utf-8') as out:
            os.fsync(out.fileno())

    def kept(self) -> list[str]:
        """The records set aside by an earlier session, oldest first."""
        try:
            text = self.held.read_text(encoding='utf-8')
        except OSError:
            return []
        return [line + '\n' for line in text.splitlines() if line.endswith('<EOR>')]

    def keep(self, records: list[str]) -> bool:
        """Set exactly these records aside, replacing what was; none removes the file."""
        try:
            if not records:
                self.held.unlink(missing_ok=True)
                return True
            spare = self.held.with_name(self.held.name + '.new')
            with spare.open('w', encoding='utf-8') as out:
                out.write(''.join(records))
                out.flush()
                os.fsync(out.fileno())
            os.replace(spare, self.held)
            return True
        except OSError:
            return False

    def commit(self, **fields) -> Qso:
        qso, record = self.record(**fields)
        self.write(record)
        return qso
