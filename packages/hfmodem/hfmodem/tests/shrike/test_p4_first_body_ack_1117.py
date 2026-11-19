# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Two held exchanges from the 11-16 and 11-17 mail arms, both off by default.

Our first body of a turn drew V27 in every slot (WS8EOC 22:08Z, VE3KPG 22:15Z,
K4MSU 05:18Z), where KY2D's in k0003 drew V27 once and went on 13, 60 and its
long. `--p4-turn-entry ladder` now takes that V27 as the body's ack.

As IRS we answered a repeated packet with the same word until the link ended
(WS8EOC c3 x41 under V27, K4MSU c0 x10 under X27), and the partner word moved
the same gateways. `--p4-ack-flip N` switches to the partner after N such repeats; the chain follows
the word keyed, so N more held repeats switch back."""

from hfmodem.shrike.p4receiver import Kind
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike import test_p4_irs_slot_fixes_1114 as tapes
from hfmodem.tests.shrike.test_p4_irs_slot_fixes_1114 import FS, keyed
from hfmodem.tests.shrike.test_p4_turn_ladder_1114 import step
from hfmodem.tests.shrike.test_p4turn import ack_greeting, machine, propose_ack

REPLY = b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;'


def first_body(**options):
    m, a = machine(role='irs', data='long', managed_application=True, **options)
    a.on_host_data(REPLY + b'B' * 200)
    m._begin_data_turn(10 * FS, word_yield=True)
    assert m.decide(10 * FS - m.lead).status == 0
    last = m.keyed()
    assert (last.kind, last.state, last.payload) == (Kind.BODY, State.ISS_DATA, REPLY[:31])
    return m, step(m, 'V27', last)


def test_ladder_entry_takes_v27_to_the_first_body_and_keys_13_60_and_the_long():
    m, last = first_body(turn_entry='ladder')
    assert (last.kind, last.value, last.header) == (Kind.SHORT, 13, 'A')
    last = step(m, 'X27', last)
    assert last.value == 60
    last = step(m, 'U27', last)
    # Straight on from the body, as KY2D's one decoded turn long (INFERRED).
    assert last.kind == Kind.LONG and last.status & 3 == 1 and last.payload[:1] == b'B'


def test_by_default_v27_to_the_first_body_repeats_it():
    m, last = first_body()
    assert (last.kind, last.state, last.payload) == (Kind.BODY, State.ISS_DATA, REPLY[:31])


def test_ladder_entry_leaves_v27_to_a_later_normal_a_repeat():
    m, a = machine(role='irs', data='body', managed_application=True, turn_entry='ladder')
    a.on_host_data(REPLY + b'B' * 200)
    m._begin_data_turn(10 * FS, word_yield=True)
    m.decide(10 * FS - m.lead)
    last = step(m, 'V27', m.keyed())
    assert last.kind == Kind.BODY and last.payload == REPLY[:31]


def answers(arm, **options):
    m, _ = machine(role='irs', normal_ack='toggle', **options)
    out = []
    for frame in ack_greeting(arm):
        p = propose_ack(m, frame)
        out.append((p.word, p.ack_reason))
        m.keyed()
    return out


def test_ack_flip_switches_to_the_partner_and_back_after_n_held_repeats():
    got = answers('2200', ack_flip=2)
    # Five copies of c0, then c1 A, c1 B, c2 A, c2 B, c2 B.
    assert got[:6] == [('X27', 'chain start'), ('X27', 'repeat keeps X27'),
                       ('X27', 'repeat keeps X27'), ('V27', 'flip after 2 repeats under X27'),
                       ('V27', 'repeat keeps V27'), ('X27', 'new packet after V27')]


def test_ack_flip_follows_the_partner_and_switches_back_after_n_more_held_repeats():
    got = answers('2200', ack_flip=1)
    assert got[:5] == [('X27', 'chain start'), ('X27', 'repeat keeps X27'),
                       ('V27', 'flip after 1 repeats under X27'), ('V27', 'repeat keeps V27'),
                       ('X27', 'flip after 1 repeats under V27')]


def test_ack_flip_off_by_default():
    got = answers('2200')
    assert [w for w, _ in got[:5]] == ['X27'] * 5
    assert not any(reason.startswith('flip') for _, reason in got)


TAPES = {
    'ws8eoc-2208': 'onair-20261116T220806820479Z-13ef68ca-eb46-468b-81c3-bd1c32d02229',
    'k4msu-2211': 'onair-20261116T221102981815Z-7f4574c0-1e24-4f9b-bd51-7a09c45b7646',
    'k4msu-0518': 'onair-20261117T051845647451Z-926b324c-530b-4a5d-96ec-af4d5dc86288',
}
STOCK_TURN = dict(p4_turn_entry='ladder', p4_ack_flip=2)


def replay(name, monkeypatch, **options):
    monkeypatch.setitem(tapes.CAPTURES, name, TAPES[name])
    return tapes.replay(name, **options)


def answers_between(probe, start, end):
    return [(round(e['at'] / FS, 3), e['word'], e['ack_reason'])
            for e in keyed(probe, start, end) if e['kind'] == 'word']


def test_ws8eoc_2208_c3_run_flips_after_two_held_v27_repeats(monkeypatch):
    held = answers_between(replay('ws8eoc-2208', monkeypatch), 31.9, 84.)
    assert [w for _, w, _ in held] == ['V27'] * len(held) and len(held) == 42
    run = answers_between(replay('ws8eoc-2208', monkeypatch, **STOCK_TURN),
                          31.9, 84.)
    assert run[:6] == [
        (31.933, 'V27', 'new packet after X27'), (33.179, 'V27', 'repeat keeps V27'),
        (34.429, 'V27', None), (35.68, 'V27', 'repeat keeps V27'), (36.93, 'V27', None),
        (38.179, 'X27', 'flip after 2 repeats under V27')]
    assert {w for _, w, _ in run} == {'V27', 'X27'}


def test_ws8eoc_2208_first_body_of_the_turn_goes_on_to_13_under_the_ladder_entry(monkeypatch):
    probe = replay('ws8eoc-2208', monkeypatch, **STOCK_TURN)
    after = keyed(probe, 111.9, 114.)
    assert [(e['kind'], e['state'], e['status']) for e in after[:2]] == [
        ('body', 'ISS_DATA', 0), ('short', 'ISS_LADDER', 0)]
    assert after[0]['payload'] == REPLY.hex() and after[1]['value'] == 13
    assert any(e['event'] == 'IssAck' and e['word'] == 'V27' and e['verdict'] == 'ack'
               and 112. < e['at'] / FS < 113. for e in probe.link_events)
    default = keyed(replay('ws8eoc-2208', monkeypatch), 101.9, 111.)
    assert [(e['kind'], e['state']) for e in default] == [('body', 'ISS_DATA')] * 8


def test_k4msu_2211_c0_run_flips_after_two_held_x27_repeats(monkeypatch):
    # Eleven slots: the weak Y27 head at 41.006 now draws its answer at 41.929.
    held = answers_between(replay('k4msu-2211', monkeypatch), 39., 53.)
    assert [w for _, w, _ in held] == ['X27'] * 11
    run = answers_between(replay('k4msu-2211', monkeypatch, **STOCK_TURN), 39., 53.)
    assert [w for _, w, _ in run] == ['X27'] * 5 + ['V27'] * 4 + ['X27'] * 2
    assert run[5] == (45.68, 'V27', 'flip after 2 repeats under X27')


def test_k4msu_0518_first_body_after_the_v27_grant_goes_on_to_13(monkeypatch):
    default = keyed(replay('k4msu-0518', monkeypatch), 64., 72.5)
    assert [(e['kind'], e['state']) for e in default] == [('body', 'ISS_DATA')] * 7
    after = keyed(replay('k4msu-0518', monkeypatch, **STOCK_TURN), 64., 72.5)
    assert [(e['kind'], e['value']) for e in after] == [('body', None)] + [('short', 13)] * 6
