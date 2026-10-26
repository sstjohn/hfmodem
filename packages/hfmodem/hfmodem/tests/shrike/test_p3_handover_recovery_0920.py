# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VE3KPG cycle authorization and WS8EOC's established-SL3 handover stall."""
import pytest
from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.test_tx_gear_hold import sender, control
from hfmodem.tests.shrike.test_long_sl1 import linked, row


@pytest.mark.parametrize('answer', [arq.CS_ACK, arq.CS_SPEED_UP, arq.CS_CYCLE_TOG])
def test_changeover_answer_authorizes_only_its_command(answer):
    host, tx = linked(role=arq.IRS, long=True)
    a = host.arq
    a.on_host_data(b'abcdefghijklmnopqrstuvwxyz')
    a.on_host_breakin()
    a.on_rx_packet(1, b'abc', spec.status_byte(1), True,
                   protocol=spec.Protocol.PACTOR3, cycle_long=True)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    assert a._inflight.breakin and a.cycle_long
    pending = a._inflight.payload + bytes(a._outbuf)
    accepted = len(a._inflight.payload)
    a.on_rx_cs(answer)
    assert a.cycle_long == (answer == arq.CS_CYCLE_TOG)
    row('turn-answer', host, tx, pending[accepted:])


def stalled_handover():
    host, seam = sender(enabled=False)
    a = host.arq
    seam.send_long_packet = seam.send_packet
    a._outbuf.clear()
    a._buffer_raw = 0
    a._sl = 3
    a.cycle_long = True
    a._over_pending = True
    # 21:14: CS2 accepted sequence 1, then requested sequence 2 repeatedly.
    a._inflight = arq._Packet(0x19, b'', 3)
    a._next_seq = 2
    control(host, 2)
    return host, seam


def test_established_idle_ack_does_not_exempt_handover_from_recovery():
    host, seam = stalled_handover()
    a = host.arq
    assert a._inflight.status == 0x5a and a._inflight.trial_from is None
    for _ in range(18):
        control(host, 2)
    assert [sl for sl, _, _ in seam.frames] == [3]*6 + [2]*6 + [1]*7
    assert all(status == 0x5a and data == b'' for _, status, data in seam.frames)
    assert a.role == arq.ISS and a.cycle_long and a._over_pending
    assert a.tx_seq == 2 and a._buffer_raw == 0


def test_refused_handover_attempts_do_not_spend_recovery_budget():
    host, seam = stalled_handover()
    a = host.arq
    seam.refuse = True
    for _ in range(12):
        control(host, 2)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 1
    seam.refuse = False
    for _ in range(5):
        control(host, 2)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 6
    seam.refuse = True
    for _ in range(12):
        control(host, 2)
    assert a.speed_level == 2 and a._inflight.sent_sl == 3
    seam.refuse = False
    control(host, 2)
    assert a.speed_level == 2 and a._inflight.sent_at_level == 1
    control(host, 1)  # Accept sequence 2; never transfer role on this ACK alone.
    assert a.tx_seq == 3 and a.role == arq.ISS
    assert a._inflight.sent_at_level == 1


@pytest.mark.parametrize('answered', [True, False])
@pytest.mark.parametrize('long', [True, False])
def test_handover_downshift_renders_same_counter_and_preserves_queued_bytes(answered, long):
    host, tx = linked(sl=3, long=long)
    a = host.arq
    a.on_host_over()
    host.tick(elapsed_ticks=3 if long else 1, cycle_ticks=3 if long else 1)
    seq, status = a.tx_seq, a._inflight.status
    assert status & spec.STATUS_CHANGEOVER
    a.on_host_data(b'queued while handover is pending')
    pending = bytes(a._outbuf)
    a.cfg.min_sl = 2
    a.cfg.max_retries = 32  # Keep the independent silence timeout out of this test.
    for _ in range(6):
        a._on_nak(False, answered=answered)
    assert a.speed_level == 2 and a._inflight.sent_at_level == 1
    assert a.tx_seq == seq and a._inflight.status == status
    assert a.role == arq.ISS and a.cycle_long == long
    row('handover-downshift', host, tx, pending)
    for _ in range(7):
        a._on_nak(False, answered=answered)
    assert a.speed_level == 2 and a.tx_seq == seq
    assert bytes(a._outbuf) == pending
