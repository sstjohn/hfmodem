# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC 40 m 15:38Z: the gateway granted our bare Y27 with V27 0.945 s after the
key, inside its chirp span. The live arm took the turn only at 77.858 and keyed
c0 bodies the gateway kept answering with V27."""

from hfmodem.shrike.p4turn import Kind, Proposal, State
from hfmodem.tests.shrike import test_p4_irs_slot_fixes_1114 as tapes
from hfmodem.tests.shrike.test_p4_irs_slot_fixes_1114 import FS, keyed
from hfmodem.tests.shrike.test_p4turn import machine

CAPTURE = 'onair-20261114T153835177049Z-ee99ee39-efd1-4d7e-9633-5c9ffd298e05'
# The mail client answered the 54.764 prompt before the 58.513 invitation.
REPLY = (56., b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;')
SLOT = round(1.25 * FS)


def replay(monkeypatch, **options):
    monkeypatch.setitem(tapes.CAPTURES, 'ws8eoc-1538', CAPTURE)
    return tapes.replay('ws8eoc-1538', queued=REPLY, **options)


def test_v27_grant_inside_the_chirp_span_takes_the_turn_and_keys_every_slot(monkeypatch):
    probe = replay(monkeypatch)
    turn = next(e for e in probe.link_events if e['event'] == 'turn')
    asks = [e for e in keyed(probe, 58.) if e['state'] == 'IRS_BREAKIN']
    assert [round(e['at'] / FS, 3) for e in asks] == [61.913]
    assert round(turn['at'] / FS, 3) == 62.858 and turn['responds_to'] == asks[0]['at']
    after = keyed(probe, turn['at'] / FS)
    assert (after[0]['kind'], after[0]['state']) == ('body', 'ISS_DATA')
    assert after[0]['at'] == asks[0]['at'] + 2 * SLOT
    assert all(b['at'] - a['at'] == SLOT for a, b in zip(after, after[1:]))
    assert not any(e['event'] == 'Refused' for e in probe.link_events)


def test_first_body_after_a_v27_grant_of_our_y27_is_c1_when_asked(monkeypatch):
    probe = replay(monkeypatch, p4_y27_first_body='c1')
    turn = next(e for e in probe.link_events if e['event'] == 'turn')
    bodies = [e for e in keyed(probe, turn['at'] / FS) if e['kind'] == 'body' and e.get('payload')]
    assert bodies and all(e['status'] == 0x01 for e in bodies)
    assert any(e['event'] == 'IssAck' and e['expected'] == 'X27' and e['at'] > turn['at']
               for e in probe.link_events)


def test_a_refusal_found_late_moves_the_body_to_the_first_keyable_slot():
    m, _ = machine()
    due = 100 * FS
    m._next = Proposal(Kind.BODY, due, status=0, state=State.ISS_DATA, payload=b'x')
    m.refused('deadline', now=due + 2 * SLOT + 1)
    assert m._next.stream_sample - m.lead >= due + 2 * SLOT + 1
    assert m._next.stream_sample - SLOT - m.lead < due + 2 * SLOT + 1
    assert (m._next.stream_sample - due) % SLOT == 0
    m._next = Proposal(Kind.BODY, due, status=0, state=State.ISS_DATA, payload=b'x')
    m.refused('deadline')
    assert m._next.stream_sample == due + SLOT
