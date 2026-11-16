# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A LONG family's work cutoff does not consume another reader's slot."""
from types import SimpleNamespace
from time import monotonic
import json

import numpy as np
import pytest

from hfmodem.shrike import p4frame, p4oracle
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.shrike.p4observation import Kind, Observation
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect


FS = 48000
ONSET = FS


def prepared_receiver(monkeypatch):
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.clock = Clock(Expect.PACKET, ONSET, .33, 3.75, 0.)
    rx.expected = True
    monkeypatch.setattr(p4oracle, 'robust_at', lambda *a, **kw: dict(
        p4_walsh_peak=0, p4_hdr='?', p4_value=None, coherence=0., first_chip=ONSET))
    return rx


def decoded(kind):
    return Observation(kind, FS, 0, ONSET, ONSET, 3.28, 0., .9,
        placed=True, frame=p4frame.Frame(b'current source', p4frame.Status(1), 0))


def test_expired_sf8_reserve_preserves_sf16_work(monkeypatch):
    rx = prepared_receiver(monkeypatch)
    rx.long_deadline = lambda at, sf8=False: SimpleNamespace(
        work_limit=lambda: monotonic() + (-1 if sf8 else 1))
    wanted = decoded(Kind.LONG)
    calls = []
    def sf16(*a, **kw):
        calls.append('sf16')
        return wanted, .9
    monkeypatch.setattr(rx, '_long', sf16)
    monkeypatch.setattr(rx, '_sf8_long', lambda *a, **kw: calls.append('sf8'))
    assert rx._packet(np.zeros(round(4.4*FS)), 0) is wanted
    assert calls == ['sf16']


def test_expired_long_work_preserves_complete_chirp_reader(monkeypatch):
    rx = prepared_receiver(monkeypatch)
    monkeypatch.setattr(rx, '_long_work_expired', lambda *a, **kw: True)
    def forbidden(*a, **kw):
        raise AssertionError('expired LONG frontend ran')
    for name in ('_long', '_sf8_long', '_normal_long'):
        monkeypatch.setattr(rx, name, forbidden)
    rx._chirp_headers[(ONSET, 0.)] = (ONSET, .9)
    wanted = decoded(Kind.CHIRP)
    monkeypatch.setattr(rx, '_chirp', lambda *a, **kw: wanted)
    assert rx._packet(np.zeros(round(4.4*FS)), 0) is wanted


def test_unread_sf8_preserves_other_complete_readers(monkeypatch):
    rx = prepared_receiver(monkeypatch)
    rx._sf8_prepared[(ONSET, 0.)] = SimpleNamespace(coherence=.9)
    rx._chirp_headers[(ONSET, 0.)] = (ONSET, .9)
    pending = Observation(Kind.LONG, FS, 0, ONSET, ONSET, 3.28, 0., .9,
                          provisional=True, placed=True, sf8=True)
    calls = []
    def sf8(*a, **kw):
        calls.append('sf8')
        return pending
    def other(name):
        def read(*a, **kw):
            calls.append(name)
            return None, 0.
        return read
    monkeypatch.setattr(rx, '_sf8_long', sf8)
    monkeypatch.setattr(rx, '_long', other('sf16'))
    monkeypatch.setattr(rx, '_normal_long', other('normal'))
    wanted = decoded(Kind.CHIRP)
    monkeypatch.setattr(rx, '_chirp', lambda *a, **kw: wanted)
    assert rx._packet(np.zeros(round(4.4*FS)), 0) is wanted
    assert calls == ['sf8', 'sf16', 'normal']


def test_charged_watch_read_preserves_parent_answers():
    """A .330 horizon cannot spend the independent WORD owner's read margin."""
    from hfmodem.tests.shrike.p4_chirp_deadline_support import replay
    arm = 'onair-1003-1112'
    require_input(CAPTURES/arm/'stream.flac')
    probe = json.loads((CAPTURES/arm/'p4-probe.json').read_text())
    decisions = [row for row in probe['link_events'] if row['event'] == 'Decision']

    def cost(method):
        def work(at, args):
            row = min(decisions, key=lambda event: abs(event['at']-at))
            return max(0., row.get('work_ms', {}).get(method, 4)/1000-.003)
        return work

    # The recording's declared stream-clock charge alone must establish the
    # reserve; an unadvanced host timer cannot make that work free.
    result = replay(arm, stop=round(37.8*FS),
                    work_costs={name: cost(name) for name in ('read_partial', 'read_window')},
                    elapsed_clock=lambda: 0.)
    # Each c2 repeat is acked again (WS8EOC 04:48Z), never broken into: every
    # V27 keys in its copy's own answer slot, none slipped by the watch reads.
    keyed = [row for row in result['events'] if row['event'] == 'Keyed']
    assert not [row for row in keyed if row.get('state') == 'IRS_BREAKIN']
    keys = [row for row in keyed if row['at'] > round(29.4*FS)]
    assert [row['at'] for row in keys] == [1411644, 1591649, 1771644]
    assert all(row['word'] == 'V27' and row['state'] == 'IRS_ANSWER'
               and not row.get('payload') for row in keys)
    assert max(row.get('cycle_end_sample', 0) for row in result['events']) > round(36.8*FS)


def test_alias_frontend_requires_completion_reserve_after_spread_work(monkeypatch):
    from hfmodem.shrike import p4receiver
    rx = prepared_receiver(monkeypatch)
    now = [0.]
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: now[0])
    rx.long_deadline = lambda *a, **kw: SimpleNamespace(work_limit=lambda: .025)
    monkeypatch.setattr(rx, '_sf8_allowed', lambda: False)
    rx._long_prepared[(ONSET, 0.)] = {}
    rx._chirp_headers[(ONSET, 0.)] = None
    rx._chirp_sweeps.add((ONSET, 0.))
    def spread(*a, **kw):
        now[0] += .020
        return None, 0.
    def forbidden(*a, **kw):
        raise AssertionError('alias frontend started with only 5 ms of its grant left')
    monkeypatch.setattr(rx, '_long', spread)
    monkeypatch.setattr(rx, '_normal_long', forbidden)
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [])
    assert rx._packet(np.zeros(round(4.4*FS)), 0) is None
    assert now[0] == .020


@pytest.mark.parametrize('family,prepared', [('sf8', False), ('sf8', True),
                                             ('sf16', False), ('sf16', True)])
def test_long_frontend_requires_completion_reserve(monkeypatch, family, prepared):
    rx = prepared_receiver(monkeypatch)
    rx.chirp = False
    rx.long_deadline = lambda *a, **kw: SimpleNamespace(work_limit=lambda: monotonic()+.001)
    if prepared:
        if family == 'sf8':
            rx._sf8_prepared[(ONSET, 0.)] = SimpleNamespace(coherence=.9)
        else:
            rx._long_prepared[(ONSET, 0.)] = {}
    def forbidden(*a, **kw):
        raise AssertionError('LONG frontend started without a completion reserve')
    for name in ('_long', '_sf8_long', '_normal_long'):
        monkeypatch.setattr(rx, name, forbidden)
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [])
    assert rx._packet(np.zeros(round(4.4*FS)), 0) is None
