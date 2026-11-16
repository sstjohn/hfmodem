# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 work admission and end-of-link accounting on decision deadlines."""

import time


class LongDeadline:
    """Frozen work, publication and enqueue limits for one physical LONG."""

    def __init__(self, fs, onset, sample_now, *, lead_s, sf8=False,
                 notice_s=0., wall_now=time.monotonic):
        key = onset + round(3.400 * fs)
        self.work = WorkBudget(fs, key, sample_now, wall_now,
                               reserve_s=lead_s + (.010 if sf8 else 0.) + .005)
        self.publication = WorkBudget(fs, key, sample_now, wall_now,
                                      reserve_s=notice_s + .005)
        self.enqueue = WorkBudget(fs, key, sample_now, wall_now,
                                  reserve_s=notice_s)

    def work_limit(self):
        return self.work.wall_now() + self.work.tighten()

    def publishable(self):
        return self.publication.tighten() >= 0

    def keyable(self):
        return self.enqueue.tighten() >= 0


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

    def tighten(self):
        self.wall_deadline = min(self.wall_deadline, self.wall_now()
            + (self.sample_deadline-self.sample_now())/self.fs)
        return self.remaining()

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


def watch_read_admission(stage, support, now, decision, fs, costs=()):
    """Reserve synchronous prefix work as well as final reads for the parent.

    Stage 1 can prepare both spread forms. Measured stream charges and host
    elapsed time may raise, but never erase, the conservative initial reserve.
    """
    # A partial read runs classification plus feed, not just the cheap header.
    initial = (WorkBudget.COSTS['detect'] if stage == 0 else
               2*WorkBudget.COSTS['long'] if stage == 1 else WorkBudget.COSTS['long'])
    reserve = max((round(initial*fs), *costs))
    return max(support, now)+reserve <= decision, reserve


def decision_summary(events, heard, fs):
    decisions = [e for e in events if e.get('event') == 'Decision']
    unread = [e for e in decisions if e.get('reason') == 'body unread at decision']
    deferred = [e for e in decisions if e.get('reason') == 'answer deferred']
    # Refinement can defer an answer after a provisional response was already
    # selected. The decision then still says "proposal"; the explicit event is
    # the evidence of deferral. Attach it once to that packet's decision.
    for event in events:
        if event.get('event') != 'answer-deferred':
            continue
        if any(abs(e['at']-event['at']) < .05*fs for e in unread+deferred):
            continue
        decision = next((e for e in decisions if abs(e['at']-event['at']) < .05*fs), None)
        if decision is not None:
            deferred.append(decision)
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
            category = ('other_late_admission' if decision in deferred
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
                missed_decisions=len(unread)+len(deferred),
                overdue_decisions=sum(e.get('now_sample', e['decision_at']) > e['decision_at']
                                      for e in decisions),
                refused_transmissions=sum(e.get('event') == 'Refused' for e in events),
                deferred_frames=len({e['at'] for e in events if e.get('event') == 'answer-deferred'}),
                **counts)
