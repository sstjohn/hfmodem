# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The v53 answer word is the operator's experiment, and a hold that changes nothing ends.

WS8EOC 40 m, 2026-11-07: the gateway sent v53 41 times, we answered U27 on
grid every time, it never released into chirps, and the operator's interrupt
cost the closing ID.
"""
import pytest

from hfmodem.shrike.p4observation import Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, body, machine
from hfmodem.tests.shrike.test_p4_v53_0929 import control

CYCLE = 60000


def irs(**options):
    m, arq = machine(**options)
    m.state = State.IRS_ANSWER
    return m, arq


def copy(m, k, header=None):
    """The k-th v53 copy (0-based), stock shape: A first, B after."""
    at = FS + k*CYCLE
    m.observe(RegisteredObservation(control(at, header=header or ('A' if k == 0 else 'B')),
                                    Registered.NEW), now=at+round(.81*FS))
    p = m.decide(at+44160-m.lead)
    if p is not None and p.state == State.IRS_ANSWER:
        assert (p.stream_sample, p.peer_control) == (at+44160, 53)
    return p


def words(m, n):
    out = []
    for k in range(n):
        p = copy(m, k)
        out.append(p.word)
        m.keyed()
    return out


def test_default_answers_u27_until_the_hold_bound():
    m, _ = irs()
    assert words(m, 23) == ['U27']*23
    assert m.state == State.IRS_ANSWER


def test_header_b_copies_take_their_own_word():
    m, _ = irs(peer_ack={'53/B': 'V27'})
    assert words(m, 4) == ['U27', 'V27', 'V27', 'V27']


def test_unchanged_hold_ends_the_link_at_its_bound():
    m, _ = irs(v53_hold=6)
    assert words(m, 5) == ['U27']*5
    p = copy(m, 5)
    assert m.state == State.IRS_CLOSE
    assert p.state == State.IRS_BREAKIN
    assert m.reason == 'no progress after 6 peer control 53 copies'
    exit_ = [e for e in m.link_events if e['event'] == 'v53-hold-exit']
    assert [(e['copies'], e['action']) for e in exit_] == [(6, 'close')]


def test_default_bound_ends_a_41_copy_hold():
    m, _ = irs()
    words(m, 23)
    p = copy(m, 23)
    assert m.state == State.IRS_CLOSE and p.state == State.IRS_BREAKIN
    assert m.reason == 'no progress after 24 peer control 53 copies'
    assert [e['action'] for e in m.link_events if e['event'] == 'v53-hold-exit'] == ['close']


def test_queued_mail_breaks_in_at_the_bound():
    m, arq = irs(v53_hold=4)
    m.allow_breakin = True
    arq.on_host_data(b'mail remains queued')
    words(m, 3)
    p = copy(m, 3)
    assert p.state == State.IRS_BREAKIN
    assert m.state != State.DOWN
    assert [e['action'] for e in m.link_events if e['event'] == 'v53-hold-exit'] == ['break-in']


def test_progress_restarts_the_count():
    m, _ = irs(v53_hold=4)
    words(m, 3)
    m.observe(body(FS+3*CYCLE), now=FS+3*CYCLE+round(.81*FS))
    assert not m._unsupported_copies
    for k in range(4, 7):
        assert copy(m, k).word == 'U27'
        m.keyed()
    assert m.state == State.IRS_ANSWER


def test_zero_holds_to_the_link_timeout():
    m, _ = irs(v53_hold=0)
    assert words(m, 60) == ['U27']*60


@pytest.mark.parametrize('bad', [dict(v53_hold=-1)])
def test_negative_bounds_are_refused(bad):
    with pytest.raises(ValueError):
        machine(**bad)
