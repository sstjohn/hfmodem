# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""
Unread gear replies must not credit payload the receiver has not accepted.
"""

import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests.shrike.p3_reply_support import (
    arm_reply, disarm_reply, emit_reply, host_stub, receiving,
)


def packet(s, tx, seq, *, slot=None, wants_long=False, sl=1, crc=True, payload=b'abc'):
    slot = tx.slot if slot is None else slot
    phase = (tx.raster.boundary(slot) + tx._p3_control_lead(arq.CS_ACK)
             - tx.p3_answer_n)
    status = spec.status_byte(seq) | (spec.STATUS_LONG_CYCLE if wants_long else 0)
    s.host.on_rx_event(rxfront.Event(phase / onair.FS, 'packet', '',
        protocol=spec.Protocol.PACTOR3, packet=(sl, status, payload, crc),
        cycle_long=False))


@pytest.mark.parametrize('answer', [arq.CS_ACK, arq.CS_REQUEST, arq.CS_BREAKIN,
                                  arq.CS_SPEED_UP, arq.CS_NAK, arq.CS_CYCLE_TOG])
@pytest.mark.parametrize('at_threshold', [False, True])
def test_ordinary_reply_marks_only_threshold_cs5_as_maxdown(tmp_path, answer, at_threshold):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.host = host_stub()
    a = tx.host.arq
    a._p3_rx_errors = a.cfg.p3_max_down if at_threshold else 0
    assert tx._p3_ordinary_reply(answer) == (answer, at_threshold and answer == arq.CS_NAK)


@pytest.mark.parametrize('seq', [1, 2])
@pytest.mark.parametrize('word', [arq.CS_SPEED_UP, arq.CS_CYCLE_TOG])
@pytest.mark.parametrize('policy', ['parity', 'repeat'])
def test_keyed_command_answers_two_unread_copies(tmp_path, seq, word, policy):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.repeat_gear = 0
    a.cfg.p3_unread_after_gear = policy
    if seq == 2:
        arm_reply(tx, live, grid, 2)
        packet(s, tx, 1)
        assert emit_reply(tx, live) == arq.CS_REQUEST
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1 if word == arq.CS_SPEED_UP else 1000
    a.cfg.long_cycle = word == arq.CS_CYCLE_TOG
    arm_reply(tx, live, grid, 3)
    packet(s, tx, seq, wants_long=word == arq.CS_CYCLE_TOG)
    words = [emit_reply(tx, live)]
    accepted = a.rx_progress
    for _ in range(2):
        arm_reply(tx, live, grid, grid.next_slot(tx.slot))
        words.append(emit_reply(tx, live))
    parity = arq.CS_REQUEST if seq % 2 else arq.CS_ACK
    assert words == [word] + [word if policy == 'repeat' else parity] * 2
    assert a.rx_seq == seq and a.rx_progress == accepted


@pytest.mark.parametrize('converted', [False, True])
def test_post_key_duplicate_cannot_license_next_unread_slot(tmp_path, converted):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.repeat_gear = 1
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1000
    arm_reply(tx, live, grid, 2)
    packet(s, tx, 1)
    assert emit_reply(tx, live) == arq.CS_REQUEST
    arm_reply(tx, live, grid, 3)
    if converted:
        packet(s, tx, 1)
        assert tx._pending_p3_cs == arq.CS_SPEED_UP
    assert emit_reply(tx, live, late=True) == arq.CS_REQUEST
    packet(s, tx, 1, slot=3)
    assert tx._pending_p3_cs == arq.CS_SPEED_UP
    accepted = a.rx_progress
    words = []
    for slot in (4, 5):
        arm_reply(tx, live, grid, slot)
        words.append(emit_reply(tx, live))
    assert words == [arq.CS_REQUEST, arq.CS_REQUEST]
    assert a.rx_seq == 1 and a.rx_progress == accepted


def test_keyed_repeat_gear_uses_parity_for_unread_next_counter(tmp_path):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.repeat_gear = 1
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1000
    arm_reply(tx, live, grid, 2)
    packet(s, tx, 1)
    assert emit_reply(tx, live) == arq.CS_REQUEST
    arm_reply(tx, live, grid, 3)
    packet(s, tx, 1)
    words = [emit_reply(tx, live)]
    for slot in (4, 5):
        arm_reply(tx, live, grid, slot)
        words.append(emit_reply(tx, live))
    assert words == [arq.CS_SPEED_UP, arq.CS_REQUEST, arq.CS_REQUEST]
    assert a.rx_seq == 1


@pytest.mark.parametrize('hold', ['held', 'once'])
@pytest.mark.parametrize('levels', [(2, 2, 2, 2), (3, 3, 2, 1), (1, 1, 1, 1)])
def test_maxdown_holds_until_the_counter_is_accepted(tmp_path, hold, levels):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    a.cfg.p3_max_down = 2
    a.cfg.speed_up = 'hold'
    arm_reply(tx, live, grid, 2)
    packet(s, tx, 1, sl=levels[0])
    assert emit_reply(tx, live) == arq.CS_REQUEST
    accepted = a.rx_progress
    words = []
    for slot, sl in enumerate(levels, 3):
        arm_reply(tx, live, grid, slot)
        token = a.begin_receive_opportunity()
        packet(s, tx, 2, sl=sl, crc=False)
        words.append(emit_reply(tx, live))
        a.finish_receive_opportunity(token)
    if hold == 'held':
        assert words == [arq.CS_REQUEST, arq.CS_NAK, arq.CS_NAK, arq.CS_NAK]
    elif levels[0] == 1:
        assert words == [arq.CS_REQUEST] * 4
    else:
        assert words == [arq.CS_REQUEST, arq.CS_NAK, arq.CS_REQUEST,
                         arq.CS_NAK if levels[-1] == 2 else arq.CS_REQUEST]
    assert a.rx_seq == 1 and a.rx_progress == accepted
    arm_reply(tx, live, grid, 7)
    packet(s, tx, 2, sl=levels[-1])
    assert emit_reply(tx, live) == arq.CS_ACK
    assert a.rx_seq == 2 and a.rx_progress == accepted + 1
    arm_reply(tx, live, grid, 8)
    packet(s, tx, 3, sl=levels[-1], crc=False)
    assert emit_reply(tx, live) == arq.CS_ACK


@pytest.mark.parametrize('policy', ['parity', 'repeat'])
def test_late_gear_does_not_acknowledge_the_new_acceptance(tmp_path, policy):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.p3_unread_after_gear = policy
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1
    arm_reply(tx, live, grid, 2)
    packet(s, tx, 1)
    assert emit_reply(tx, live) == arq.CS_SPEED_UP
    held = arq.CS_SPEED_UP if policy == 'repeat' else arq.CS_REQUEST
    assert arm_reply(tx, live, grid, 3) == held
    packet(s, tx, 2)
    assert tx._pending_p3_cs == arq.CS_SPEED_UP
    assert emit_reply(tx, live, late=True) == held
    # A late acceptance needs its own acknowledgement.
    assert arm_reply(tx, live, grid, 4) == arq.CS_ACK
    assert emit_reply(tx, live) == arq.CS_ACK
    assert a.rx_seq == 2


@pytest.mark.parametrize('word', [arq.CS_ACK, arq.CS_REQUEST])
def test_explicit_ack_keeps_its_word_without_packet_metadata(tmp_path, word):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1
    arm_reply(tx, live, grid, 2)
    packet(s, tx, 1)
    assert emit_reply(tx, live) == arq.CS_SPEED_UP
    assert arm_reply(tx, live, grid, 3) == arq.CS_REQUEST
    instant = tx.boundary
    tx.send_cs(word)
    assert tx._pending_p3_cs_packet is None
    assert emit_reply(tx, live) == word
    assert tx.tx_audio_start == instant


def test_new_receive_turn_does_not_inherit_old_keyed_gear(tmp_path):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1
    for seq in (1, 2, 3):
        arm_reply(tx, live, grid, seq + 1)
        packet(s, tx, seq)
        assert emit_reply(tx, live) == arq.CS_SPEED_UP
    a._take_link()
    a._give_link()
    assert a.rx_seq == 3
    assert arm_reply(tx, live, grid, 5) == arq.CS_REQUEST
    assert emit_reply(tx, live) == arq.CS_REQUEST


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_armed_maxdown_after_late_valid_copy(tmp_path, hold):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    a.cfg.p3_max_down = 2
    a.cfg.speed_up = 'hold'
    arm_reply(tx, live, grid, 2)
    packet(s, tx, 1, sl=2)
    assert emit_reply(tx, live) == arq.CS_REQUEST
    for slot in (3, 4):
        arm_reply(tx, live, grid, slot)
        packet(s, tx, 2, sl=2, crc=False)
        if slot == 3:
            assert emit_reply(tx, live) == arq.CS_REQUEST
        else:
            disarm_reply(tx, live)
    assert arm_reply(tx, live, grid, 4) == arq.CS_NAK
    packet(s, tx, 1, sl=2)
    assert emit_reply(tx, live, late=True) == arq.CS_NAK
    expected = arq.CS_NAK if hold == 'held' else arq.CS_REQUEST
    assert arm_reply(tx, live, grid, 5) == expected
    assert emit_reply(tx, live) == arq.CS_REQUEST
    assert arm_reply(tx, live, grid, 6) == arq.CS_REQUEST
    assert emit_reply(tx, live) == arq.CS_REQUEST


@pytest.mark.parametrize('sl, seq, unread_sl, word, expected, copies', [
    pytest.param(4, 2, 5, arq.CS_SPEED_UP, arq.CS_ACK, 1, id='ve3kpg-onair-0920-1246-191'),
    pytest.param(2, 1, 3, arq.CS_SPEED_UP, arq.CS_REQUEST, 1,
                 id='ws8eoc-onair-0920-1115-454'),
    pytest.param(5, 1, 6, arq.CS_SPEED_UP, arq.CS_SPEED_UP, 2, id='ws8eoc-onair-0923-2244-903'),
    pytest.param(1, 2, 1, arq.CS_CYCLE_TOG, arq.CS_ACK, 1, id='ws8eoc-onair-0920-1115-648'),
])
def test_unread_gear_answers_at_observed_levels(
        tmp_path, sl, seq, unread_sl, word, expected, copies):
    s, tx, live, grid = receiving(tmp_path)
    a = s.host.arq
    a.cfg.repeat_gear = 0
    if seq == 2:
        arm_reply(tx, live, grid, 2)
        packet(s, tx, 1)
        assert emit_reply(tx, live) == arq.CS_REQUEST
    a.cfg.speed_up = 'auto'
    a.cfg.p3_speed_up_after = 1 if word == arq.CS_SPEED_UP else 1000
    a.cfg.long_cycle = word == arq.CS_CYCLE_TOG
    arm_reply(tx, live, grid, 3)
    packet(s, tx, seq, sl=sl, wants_long=word == arq.CS_CYCLE_TOG)
    words = [emit_reply(tx, live)]
    accepted = a.rx_progress
    for _ in range(copies):
        arm_reply(tx, live, grid, grid.next_slot(tx.slot))
        packet(s, tx, (seq + 1) % 4, sl=unread_sl, crc=False)
        words.append(emit_reply(tx, live))
    assert words == [word] + [expected] * copies
    assert a.rx_seq == seq and a.rx_progress == accepted


@pytest.mark.parametrize('seq', [1, 2])
def test_advanced_sender_keeps_unread_payload_after_gear(tmp_path, seq):
    from hfmodem.tests.shrike.p3_arq_support import control, pending, remaining

    s, tx, live, grid = receiving(tmp_path)
    receiver = s.host.arq
    receiver.cfg.repeat_gear = 0
    if seq == 2:
        arm_reply(tx, live, grid, 2)
        packet(s, tx, 1)
        assert emit_reply(tx, live) == arq.CS_REQUEST
    receiver.cfg.speed_up = 'auto'
    receiver.cfg.p3_speed_up_after = 1
    sender, wire = pending(1, seq, b'abc', b'unread payload' * 4)
    sender.arq.cfg.p3_max_try = 9
    arm_reply(tx, live, grid, 3)
    sl, sent_seq, payload = wire.frames[-1]
    packet(s, tx, sent_seq, sl=sl, payload=payload)
    first = emit_reply(tx, live)
    assert first == arq.CS_SPEED_UP
    control(sender, first + 1)
    assert sender.arq.tx_seq == (seq + 1) % 4
    assert wire.frames[-1][0] == 2
    retained = remaining(sender.arq)
    for slot in (4, 5):
        arm_reply(tx, live, grid, slot)
        control(sender, emit_reply(tx, live) + 1)
        assert sender.arq.tx_acked_bytes == 3
        assert sender.arq.tx_seq == (seq + 1) % 4
        assert remaining(sender.arq) == retained
        assert receiver.rx_seq == seq

    sl, sent_seq, payload = wire.frames[-1]
    receiver.cfg.speed_up = 'hold'
    arm_reply(tx, live, grid, 6)
    packet(s, tx, sent_seq, sl=sl, payload=payload)
    control(sender, emit_reply(tx, live) + 1)
    assert receiver.rx_seq == sent_seq
    assert sender.arq.tx_acked_bytes == 3 + len(payload)
    assert remaining(sender.arq) == retained[len(payload):]


def test_unread_after_gear_configuration():
    assert arq.ArqConfig().p3_unread_after_gear == 'parity'
    assert arq.ArqConfig(p3_unread_after_gear='repeat').p3_unread_after_gear == 'repeat'
    with pytest.raises(ValueError, match='p3_unread_after_gear'):
        arq.ArqConfig(p3_unread_after_gear='ignore')
