# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Synthetic recording owners and optional source hooks for live edges.

``recording_boundaries`` is the owner's shared ContactBoundaries instance;
``recording_positions`` snapshots its lane clocks on the source callback thread.
These optional hooks are independent of replaceable application observers.
"""
from types import SimpleNamespace as NS
import threading

import pytest

from hfmodem.core.recording import ContactBoundaries
from hfmodem.tests.station.recording_support import bounded, flush, pcm

LOCAL, PEER, OTHER = 'K9ZZT', 'W0QXR', 'N0ZQT'


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class Owner:
    def __init__(self, session, root):
        self.session, self.root = session, root
        self.rx = session.stream('rx', rate=8, clock='card')
        self.tx = session.stream('submitted-tx', rate=5, clock='dac')
        self.edges = ContactBoundaries(session, run='live-run')
        self.starts, self.ends, self.stamps = [], [], []
        begin, end = session.begin, session.end

        def started(cid, positions, **metadata):
            self.starts.append((cid, dict(positions), metadata))
            return begin(cid, positions, **metadata)

        def ended(cid, positions, **metadata):
            self.ends.append((cid, dict(positions), metadata))
            return end(cid, positions, **metadata)

        session.begin, session.end = started, ended
        self.capture = NS(session=session, rx=self.rx, tx=self.tx,
                          boundaries=self.edges, positions=self.positions)

    def positions(self):
        self.stamps.append(threading.get_ident())
        return {lane.clock: lane.position for lane in self.session.lanes.values()}

    def bind(self, source):
        source.recording_boundaries = self.edges
        source.recording_positions = self.positions
        return source

    def feed(self, lane, first, stop):
        for at in range(first, stop, lane.rate):
            lane.push(at, pcm(at, min(at + lane.rate, stop)))
            flush(self.session)

    def advance(self, seconds):
        for lane in (self.rx, self.tx):
            self.feed(lane, lane.position, seconds * lane.rate)

    def rows(self):
        flush(self.session)
        return self.session.index['contacts']

    def finish(self):
        bounded(self.session.close)
        return self.session.index['contacts']

    def one(self):
        rows = self.rows()
        assert len(rows) == 1, 'local established edge must create exactly one contact'
        return next(iter(rows.values()))


@pytest.fixture
def live(sessions):
    def make(policy='contacts', **kwargs):
        session, root = sessions(policy, **kwargs)
        return Owner(session, root)
    return make


class Observer:
    def __init__(self, owner):
        self.owner, self.events = owner, []

    def __getattr__(self, name):
        if not name.startswith('modem_'):
            raise AttributeError(name)

        def deliver(*args):
            self.events.append((name, args, len(self.owner.starts), len(self.owner.ends)))
        return deliver


def assert_edges(owner, *, start, end=None, peer=PEER):
    row = owner.one()
    assert row['positions'] == start
    assert row['peer'] == peer
    assert row['key'][0] == 'live-run'
    assert len(row['key']) == 6
    assert len(owner.starts) == 1
    if end is not None:
        assert row['end_positions'] == end
        assert len(owner.ends) == 1
    return row
