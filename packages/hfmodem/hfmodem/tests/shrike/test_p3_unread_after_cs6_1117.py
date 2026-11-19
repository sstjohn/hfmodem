# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""W6IDS 80 m 2026-11-17 05:06Z: the CS2 behind our CS6 that the gateway took as an ACK.

`operator-data/working/onair-20261117T050648301768Z-8b25e309-…`: short SL1
counter 1 read at log 204, TX[10] CS6 at 211; the gateway's first long packet
was not read and TX[11] answered CS2, par(1), at 230. The next packet read was
counter 3 (244) and log 281 records counter 2's bytes as gone. K4MSU 22:00,
WM4RB 06:48, WD1O 07:09 and KB3PCY 08:58 the same night did the same, at both
counter parities.
"""
import pytest

from hfmodem.shrike import arq
from hfmodem.tests.shrike.p3_reply_support import arm_reply, emit_reply, receiving
from hfmodem.tests.shrike.test_p3_keyed_reply import packet


def toggled(tmp_path, seq, policy):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.repeat_gear = 0
    a.cfg.p3_unread_after_cs6 = policy
    if seq == 2:
        arm_reply(tx, live, grid, 2)
        packet(s, tx, 1)
        assert emit_reply(tx, live) == arq.CS_REQUEST
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1000
    a.cfg.long_cycle = True
    arm_reply(tx, live, grid, 3)
    packet(s, tx, seq, wants_long=True)
    words = [emit_reply(tx, live)]
    accepted = a.rx_progress
    for _ in range(2):                  # the first long packet, unread
        arm_reply(tx, live, grid, grid.next_slot(tx.slot))
        words.append(emit_reply(tx, live))
    assert a.rx_seq == seq and a.rx_progress == accepted
    return words


@pytest.mark.parametrize('seq', [1, 2])
def test_an_unread_copy_after_cs6_is_answered_cs6_again(tmp_path, seq):
    assert toggled(tmp_path, seq, 'repeat') == [arq.CS_CYCLE_TOG] * 3


@pytest.mark.parametrize('seq', [1, 2])
def test_parity_restores_the_flown_answer(tmp_path, seq):
    parity = arq.CS_REQUEST if seq % 2 else arq.CS_ACK
    assert toggled(tmp_path, seq, 'parity') == [arq.CS_CYCLE_TOG, parity, parity]


def test_unread_after_cs6_configuration():
    assert arq.ArqConfig().p3_unread_after_cs6 == 'repeat'
    with pytest.raises(ValueError, match='p3_unread_after_cs6'):
        arq.ArqConfig(p3_unread_after_cs6='ack')


# The transition-window path W6IDS ran, under the default. Where the peer's
# packet settles the request (accept, duplicate, or a CRC decision that
# arrives late) nothing is outstanding and the answer is par(N), as before. A
# new packet in the old length renews the CS6, and the unread copies behind
# that renewed request are W6IDS's: CS6 again where par(N) answered.
from hfmodem.tests.shrike import p3_cycle_support, test_p3_cycle_change_1111 as cycle

OUTCOMES = [mark.args for mark in cycle.test_real_arq_cycle_outcomes.pytestmark
            if mark.name == 'parametrize'][0]
DECISIONS = [mark.args for mark in
             cycle.test_crc_decision_crosses_boundaries_without_retroactive_emissions.pytestmark
             if mark.name == 'parametrize'][0]


@pytest.mark.parametrize(*OUTCOMES)
def test_transition_window_outcomes_under_the_default(
        tmp_path, monkeypatch, old_long, response, expected):
    monkeypatch.setattr(p3_cycle_support, 'UNREAD_AFTER_CS6', 'repeat')
    if response == 'new':
        expected = [(slot, arq.CS_CYCLE_TOG) for slot, _ in expected]
    cycle.test_real_arq_cycle_outcomes(tmp_path, monkeypatch, old_long, response, expected)


@pytest.mark.parametrize(*DECISIONS)
def test_late_decisions_under_the_default(
        tmp_path, monkeypatch, old_long, confirmation, delay, expected):
    """The late duplicate settles the request: CS1, not CS6 three times."""
    monkeypatch.setattr(p3_cycle_support, 'UNREAD_AFTER_CS6', 'repeat')
    cycle.test_crc_decision_crosses_boundaries_without_retroactive_emissions(
        tmp_path, monkeypatch, old_long, confirmation, delay, expected)
