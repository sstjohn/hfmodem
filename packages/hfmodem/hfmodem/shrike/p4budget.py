# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 work admission and end-of-link accounting on decision deadlines."""

import time


class WorkBudget:
    """Admit receive work while preserving time to render and settle the key."""

    COSTS = dict(head=.010, detect=.050, refine=.250, body=.660, packet=.660, long=.030,
                 window=.660)

    def __init__(self, fs, key_sample, sample_now, wall_now=time.monotonic,
                 reserve_s=.110, costs=None):
        self.fs = fs
        self.sample_now = sample_now
        self.wall_now = wall_now
        self.wall_deadline = (wall_now() + (key_sample-sample_now())/fs
                              - reserve_s)
        self.sample_deadline = key_sample-round(reserve_s*fs)
        self.costs = dict(self.COSTS)
        if costs is not None:
            self.costs.update(costs)
        self.deferred = []

    def remaining(self):
        return min((self.sample_deadline-self.sample_now())/self.fs,
                   self.wall_deadline-self.wall_now())

    def admits(self, stage):
        if self.remaining() >= self.costs[stage]:
            return True
        self.deferred.append(stage)
        return False

    def checkpoint(self, stage, reserve_s=.010):
        """Leave a reserve for the next bounded search chunk."""
        if self.remaining() >= reserve_s:
            return True
        self.deferred.append(stage)
        return False


def decision_summary(events, heard, fs):
    decisions = [e for e in events if e.get('event') == 'Decision']
    unread = [e for e in decisions if e.get('reason') == 'body unread at decision']
    deferred = [e for e in decisions if e.get('reason') == 'answer deferred']
    counts = dict(late_crc_valid=0, other_late_admission=0, never_decoded=0,
                  available_at_decision=0)
    for decision in unread + deferred:
        rows = [r for r in heard if r.get('phase') == 'link'
                and abs(r['onset_sample']-decision['at']) < .05*fs
                and not r.get('provisional')
                and not r.get('unexpected')
                and r.get('outcome') not in ('expired', 'contradicted')]
        crc = [r for r in rows if 'status' in r and 'data' in r]
        body = any(r.get('kind') in ('body', 'normal', 'long', 'chirp')
                   or (r.get('extent_s') or 0) > .300 for r in rows)
        admitted = crc + [r for r in rows
                         if (not body and r.get('word') not in (None, 'BREAKIN')
                             and (r['word'] != 'Y27' or r.get('extent_s') is not None))
                         or r.get('kind') == 'short' and r.get('value') is not None]
        if any(r['received_at_sample'] <= decision['decision_at'] for r in admitted):
            category = ('other_late_admission' if decision['reason'] == 'answer deferred'
                        else 'available_at_decision')
        elif crc:
            category = 'late_crc_valid'
        elif admitted:
            category = 'other_late_admission'
        else:
            category = 'never_decoded'
        decision['decode_resolution'] = category
        counts[category] += 1
    return dict(event='DecodeBudgetSummary', decisions=len(decisions),
                unread_decisions=len(unread), deferred_decisions=len(deferred),
                missed_decisions=len(unread)+len(deferred), **counts)
