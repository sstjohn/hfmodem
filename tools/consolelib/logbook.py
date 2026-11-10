# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One logbook, and it is a reader.

There is no logbook file and there should not be one. Every arm writes
`provenance.json` when it starts and again at teardown -- station, channel, fly
mode, level, every flag, the rig's own state, the runtime digest, the timings and
the verdict -- and that is the log record. A second store would be a second
source of truth, and the two would drift the first time one of them was not
written.

So this walks each registered mode's arms directory, takes the common core
defined in `modes.CORE`, and hands back rows. Spacing, `status` and `log` all read
through here, so there is one reader as well as one record.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import modes, operator, roster, station


def records(mode):
    """Every arm one mode has flown, newest first, exactly as it was written.

    A mode whose older arms were written by something else keeps a reader of its
    own (`Mode.records`) and maps them onto the common core there. That is a
    reader and nothing more: nothing in this file writes, and no arm is rewritten
    on disk to suit the view.
    """
    from hfmodem.core.capture_identity import metadata_record
    read = (mode.records(mode) if mode.records else
            [metadata_record(path) for path in mode.arms.glob('*/provenance.json')])
    # Attended applications write ordinary arms too. A mode's roster/launcher
    # directory is not an exhaustive list of its contact producers.
    for path in (operator.data_root() / 'working').glob('*/arms/*/provenance.json'):
        if path.parent.parent.resolve() == mode.arms.resolve():
            continue
        record = metadata_record(path)
        if (record.get('protocol') or record['arm'].split('-', 1)[0]) == mode.name:
            read.append(record)
    return sorted(read, key=lambda record: record['started_utc'], reverse=True)


def adapted(names=None):
    """The modes that have a profile, and the names of those still to be adapted."""
    flying, waiting = [], []
    for name in names or modes.KNOWN:
        try:
            flying.append(modes.resolve(name))
        except ValueError:
            waiting.append(name)
    return flying, waiting


def moment(text):
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f'Read {text!r} as neither a date (2026-09-16) nor a '
                         'timestamp (2026-09-16T22:00).') from None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def started(row):
    return moment(row['started_utc'])


def entries(names=None, station=None, band=None, hours=None, since=None, until=None,
            reached=False, limit=0):
    """The logbook, filtered the way the questions it has to answer are asked.

    "Have I ever worked this station on this band" is station and band; "what has
    ever answered me on 30 m at night" is band, hours and reached; "which of
    tonight's attempts reached a milestone" is since and reached.
    """
    after, before = (moment(since) if since else None), (moment(until) if until else None)
    flying, waiting = adapted(names)
    rows = [modes.core(record, mode.name) for mode in flying for record in records(mode)]
    if station:
        rows = [row for row in rows if (row['call'] or '').upper() == station.upper()]
    if band:
        rows = [row for row in rows if row['band_m'] == band]
    if hours:
        rows = [row for row in rows if roster.hours_open(hours, started(row).hour)[0]]
    if after:
        rows = [row for row in rows if started(row) >= after]
    if before:
        rows = [row for row in rows if started(row) <= before]
    if reached:
        rows = [row for row in rows if row['reached']]
    rows.sort(key=lambda row: row['started_utc'], reverse=True)
    return (rows[:limit] if limit else rows), waiting


#: VARA's `cleared` is its own sense and every mode's `keyed` is our carrier going
#: up; neither says the far end heard anything.
OURS = ('cleared', 'keyed')


def answered_at(names=None):
    """Every arm each station has flown, tallied by station, band and mode.

    A milestone past `OURS` is an answer here: an arm that only sensed and keyed
    was not heard. It is the cross-mode question the campaign's own three
    outcomes cannot answer: those are written per arm and per campaign, and
    "has this station ever replied to us in the OTHER mode" spans both. The
    multimode pass decides which mode to call a station in out of this, and a
    campaign report says what to do next out of the same reading.
    """
    rows, _ = entries(names=names)
    tally = {}
    for row in rows:
        if not row['call'] or not row['band_m']:
            continue
        seen = tally.setdefault((row['call'].upper(), row['band_m']), {}).setdefault(
            row['mode_name'], dict(arms=0, milestones=0, latest=''))
        seen['arms'] += 1
        if set(row['reached']) - set(OURS):
            seen['milestones'] += 1
            seen['latest'] = max(seen['latest'], row['started_utc'] or '')
    return tally


def where(row):
    directory = row['directory'] or ''
    try:
        return str(Path(directory).relative_to(station.ROOT))
    except ValueError:
        return directory


def spell(rows, waiting=()):
    for row in rows:
        yield (f'{started(row):%Y-%m-%d %H:%MZ}  {row["mode_name"] or "?":6} '
               f'{row["call"] or "?":8} {row["band_m"] or 0:>3} m  {row["centre_hz"]}  '
               f'{row["mode"] or "?":6} {row["level"] or "?":12} '
               f'{", ".join(row["reached"]) or "no milestone"}')
        yield f'{"":20}{where(row)}  --  {row["line"] or "still flying"}'
    stations = {row['call'] for row in rows}
    yield (f'{len(rows)} arm(s), {sum(1 for row in rows if row["reached"])} with a milestone, '
           f'at {len(stations)} station(s).')
    for name in waiting:
        yield f'{name} is not adapted to this console yet, so its arms are not in this view.'
