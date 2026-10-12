"""CLAIM THE RIG: one arm at a time, and what the last one at this gateway did.

The lock is a file the mode owns, so the new console and anything else flying
that mode take the same lease. Spacing is the other half of the claim: how long
since this station last called this gateway, and whether that call was answered.
It is reported and never refused on -- nothing may stand between the operator's
word and the carrier going up.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl

from . import logbook, roster


@contextmanager
def lease(mode):
    mode.lock.parent.mkdir(parents=True, exist_ok=True)
    with mode.lock.open('a') as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another arm from this console holds the lock. Let it finish, '
                               'or stop it in its own terminal.') from None
        yield held


def spacing_note(mode, call, now):
    """How long since the last arm at this gateway, and whether it answered.

    Reported, never refused on. The spacing figure is the operator's to weigh,
    not the console's to enforce.
    """
    rule = roster.spacing(mode)
    for record in logbook.records(mode):
        if record['call'] != call:
            continue
        ended = datetime.fromisoformat(record.get('ended_utc') or record['started_utc'])
        answered = bool((record.get('verdict') or {}).get('reached'))
        since = (now - ended).total_seconds()
        said = (f'  {record["arm"]} ended {since:.0f}s ago and '
                f'{"answered" if answered else "did not answer"}')
        if rule is None:
            return said + f'; this roster carries no spacing figure for {call}.'
        wait = rule['after_an_answered_arm_s'] if answered else rule['after_an_unanswered_arm_s']
        return said + '; ' + (
            f'{wait - since:.0f}s short of the {wait}s that has worked at {call}.'
            if since < wait else f'past the {wait}s that has worked there.')
    return None
