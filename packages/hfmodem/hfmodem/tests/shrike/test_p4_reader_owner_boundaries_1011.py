# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Physical decision boundaries, real phase refinement and exhausted sink counters."""
from unittest.mock import patch
import pytest
from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike.test_p4_part_a_1011 import setup, packet, AT, KEY, CYCLE
from hfmodem.tests.shrike.test_p4_second_delta_driver_1011 import lifecycle
from hfmodem.tests.shrike.test_p4_reader_owner_continuation_1011 import counters
from hfmodem.tests.shrike.reader_owner_support_1011 import record


@pytest.mark.parametrize('driver_lead', [1200, 2400])
@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_r1_physical_before_on_after_decision_and_distinct_leads(driver_lead, delta):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed(KEY)
    m.silence(KEY+60000)  # Genuine repeat selection, without another actual key.
    due = KEY+CYCLE
    now = due-driver_lead+delta
    physical = m.breakin_due(now, lead=driver_lead)
    expected = due+(CYCLE if delta > 0 else 0)
    assert physical == expected
    m.reschedule_breakin(now, lead=driver_lead)
    assert m._next.stream_sample == expected
    actual = onair._p4_link_due(m, [], AT+CYCLE, 3.4, now, driver_lead, False)
    assert actual == expected
    count = m.peer_watch.attempts
    m.refused('deadline')
    assert m.peer_watch.attempts == count == 1
    assert m.breakin_due(now, lead=driver_lead) >= expected
    assert m.breakin_due(now, lead=driver_lead) > m._peer_consumed
    record('physical_boundary', driver_lead=driver_lead, machine_lead=m.lead,
           now=now, delta=delta, physical=physical, refused_due=m._refused_due,
           consumed=m._peer_consumed, count=count)


@pytest.mark.parametrize('refinement', [-24, 0, 24])
def test_r1_genuine_admitted_phase_refinement_keeps_floor_and_answer_precedence(refinement):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed(KEY)
    at = AT+CYCLE+refinement
    reg = packet(at, Kind.CHIRP, 0x62, b'phase source\r'.ljust(22, b'\x1e'),
                 peer_watch_generation=m.peer_watch.generation)
    m.observe(reg, now=at+158400)
    assert m.peer_phase.onset == at
    assert m.next_offer_at == KEY+960000 and m.peer_watch.attempts == 1
    m.reschedule_breakin(at+163200-m.lead)
    p = m.decide(at+163200-m.lead)
    assert p.state == p4turn.State.IRS_ANSWER and p.accepted_target == at
    assert p.stream_sample == at+163200 and m.can_key(p)
    m.keyed(p.stream_sample)
    assert m.next_offer_at == KEY+960000 and m.peer_watch.attempts == 1
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('managed', [False, True])
def test_c3_exhaustion_all_actual_keys_and_probe_counters(managed):
    cls = p4turn.P4Turn
    machines, admissions, actual = [], [], []
    def observed(*args, **kw):
        m = cls(*args, **kw)
        machines.append(m)
        # P4Outbox owns the actual arq supplied by the driver.
        a = m.outbox.arq
        can_key, keyed = m.can_key, m.keyed
        def admit(p):
            before = counters(m, a)
            current, context = m._next, m._accepted_proposal
            ok = can_key(p)
            after = counters(m, a)
            if not ok:
                assert before == after and current is m._next and context is m._accepted_proposal
            admissions.append((id(p), ok, before, after))
            return ok
        def key(at=None):
            p = m._next
            before = counters(m, a)
            sent = keyed(at)
            actual.append((p, sent, before, counters(m, a)))
            return sent
        m.can_key, m.keyed = admit, key
        return m
    with patch.object(p4turn, 'P4Turn', observed):
        m, rx, tx, live, captured, keys, counts, feeds, replays = lifecycle(managed, 'exhaustion')
    record('exhaustion_counters', managed=managed, admissions=admissions,
           actual=[(id(p), sent.stream_sample, sent.word, before, after) for p,sent,before,after in actual],
           replays=replays)
    assert counts[:5] == [0, 1, 2, 3, 4]
    assert [(p.stream_sample, p.word) for p in keys[:5]] == [
        (8191870, 'Y27'), (8371870, 'Y27'), (8551870, 'Y27'), (8731870, 'Y27'), (8911870, 'X27')]
    assert all(row[2][5:] == (b'TEST\r', b'', 0) and row[3][5:] == (b'TEST\r', b'', 0) for row in actual)
    assert sum(not ok for _,ok,_,_ in admissions) >= 2
    assert replays == ['initial', 'final']
    assert not any(p.word == 'Y27' and 8731870 < p.stream_sample < 9811870 for p in keys)
    if not managed:
        assert all(p.word == 'X27' for p in keys[4:])
