# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""--p4-turn-entry ladder: after V27 answers our hand-back body, key the robust
short ladder every stock new ISS keyed in k0003 before any normal. WS8EOC 15:07Z,
04:45Z and 15:43Z answered V27 to everything, so the replays pin our emissions."""

import pytest

from hfmodem.shrike.p4receiver import Kind
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike import test_p4_irs_slot_fixes_1114 as tapes
from hfmodem.tests.shrike.test_p4_irs_slot_fixes_1114 import FS, keyed
from hfmodem.tests.shrike.test_p4turn import body, machine, word

TURNS = {
    'ws8eoc-1507': ('onair-20261114T150708518858Z-af37855c-ce8b-43e8-8ab2-137f917eaccc', 10.679),
    'ws8eoc-0445': ('onair-20261114T044530343353Z-25ff7e62-51f2-43cd-a9fa-23fbcbb0467f', 39.435),
    'ws8eoc-1543': ('onair-20261114T154324127039Z-4e74eb4f-fd1b-4a53-b076-8d490bee9435', 16.928),
}


def turn(name, monkeypatch, **options):
    capture, body_at = TURNS[name]
    monkeypatch.setitem(tapes.CAPTURES, name, capture)
    probe = tapes.replay(name, **options)
    after = keyed(probe, body_at - .01, body_at + 20)
    hand_back = after[0]
    assert (hand_back['kind'], hand_back['state'], hand_back['status']) == ('body', 'ISS_HANDBACK', 0x40)
    assert round(hand_back['at'] / FS, 3) == body_at
    assert any(e['event'] == 'handback-seed' for e in probe.link_events)
    slots = [e['at'] for e in after]
    assert all(b - a == round(1.25 * FS) for a, b in zip(slots, slots[1:]))
    return after[1:]


@pytest.mark.parametrize('name', TURNS)
def test_the_ladder_follows_the_hand_back_body_in_every_slot(name, monkeypatch):
    after = turn(name, monkeypatch, p4_turn_entry='ladder')
    ladder = after[:8]
    assert [(e['kind'], e['value'], e['state']) for e in ladder] == [('short', 13, 'ISS_LADDER')] * 8
    assert [e['header'] for e in ladder] == ['A'] + ['B'] * 7
    # The unanswered ladder returns to the hold, never closing the link.
    assert after[8]['kind'] == 'normal' and after[8]['state'] == 'ISS_HOLD'
    assert not any(e['state'] == 'ISS_CLOSE' for e in after)


@pytest.mark.parametrize('name', TURNS)
def test_the_default_keys_hold_idles_after_the_hand_back_body(name, monkeypatch):
    after = turn(name, monkeypatch)
    assert after and all(e['kind'] == 'normal' and e['state'] == 'ISS_HOLD' for e in after[:12])


def seeded(data, queued=b''):
    m, a = machine(role='irs', data=data, managed_application=True, turn_entry='ladder')
    m.observe(word('Y27', 0, extent=0.8))
    m.observe(body(0, b'RMS Trimode SID'))
    assert m.decide(round(.846 * FS)).word == 'X27'
    last = m.keyed()
    last = step(m, 'V27', last)
    assert (last.kind, last.status, last.state) == (Kind.BODY, 0x40, State.ISS_HANDBACK)
    if queued:
        a.on_host_data(queued)
    return m, step(m, 'V27', last)


def step(m, name, last):
    at = last.stream_sample + round(.93 * FS)
    m.observe(word(name, at, responds_to=last.stream_sample), now=at + round(.24 * FS))
    assert m.decide(at + round(.256 * FS)).stream_sample == last.stream_sample + round(1.25 * FS)
    return m.keyed()


def test_an_empty_turn_runs_the_sl5_ladder_and_offers_the_turn_back():
    # k0003 1482.730-1488.060: V27, 13 X27, 12 T27, 11 X27, 10, then SL5 c0 0x40.
    m, last = seeded('long')
    for value, answer in ((13, 'X27'), (12, 'T27'), (11, 'X27'), (10, 'V27')):
        assert (last.kind, last.value, last.header) == (Kind.SHORT, value, 'A')
        last = step(m, answer, last)
    assert (last.kind, last.status, last.state) == (Kind.NORMAL, 0x40, State.ISS_HOLD)
    assert m.ladder == (13, 60)


def test_queued_long_data_runs_13_60_and_keys_the_long_at_c2():
    # k0003 1473.708-1477.461 and 1524.906-1529.906: body c0, 13 X27, 60 U27, long c2.
    m, last = seeded('long', b'A' * 200)
    assert last.value == 13
    last = step(m, 'X27', last)
    assert last.value == 60
    last = step(m, 'U27', last)
    assert last.kind == Kind.LONG and last.status & 3 == 2 and last.payload
