# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Entry answers must preserve the receiver's speed decision.

VE3KPG onair-0920-1024 answered the SL1 entry with CS4. The host logged
SL2, then entry completion reset it to SL1 before rendering the next packet.
Exercise that ordering through PtcHost, including the first payload's trial
and byte accounting, rather than testing the speed setter in isolation.
"""
import pytest
import numpy as np

from hfmodem.shrike import arq, p3rx, spec
from hfmodem.tests.shrike.test_long_sl1 import linked
from hfmodem.tests.shrike.test_traffic_sl_hold import _cs, _granted
from hfmodem.tests.shrike.test_uninvited_entry import _uninvited


@pytest.mark.parametrize("traffic_sl", [None, 1, 3, 6])
@pytest.mark.parametrize("speed_up", ["auto", "hold"])
def test_cs4_answering_entry_overrides_the_traffic_preset(traffic_sl, speed_up):
    host, tx = _granted(traffic_sl=traffic_sl)
    a = host.arq
    a.cfg.long_cycle = False
    a.cfg.speed_up = speed_up
    queued = bytes(a._outbuf)
    seq = a.tx_seq
    accounted = a._buffer_raw
    assert queued and seq is not None

    host.on_rx_event(_cs(arq.CS_SPEED_UP))

    assert not a.entry_pending
    assert a.speed_level == 2
    assert tx.p3[-1] == ("packet", 2)
    assert a.tx_seq == (seq + 1) % arq.SEQ_MOD
    assert a._inflight.trial_from == 1
    assert a._buffer_raw == accounted
    assert a._inflight.payload + bytes(a._outbuf) == queued

    # A failed first trial must return to the speed of the entry, with the
    # new counter and every byte still pending. This is not another entry.
    repeat = arq.CS_ACK if a.tx_seq & 1 else arq.CS_REQUEST
    for _ in range(a.cfg.p3_max_try):
        host.on_rx_event(_cs(repeat))
    assert a.speed_level == 1
    assert tx.p3[-1] == ("packet", 1)
    assert a.tx_seq == (seq + 1) % arq.SEQ_MOD
    assert not a.entry_pending
    assert a._buffer_raw == accounted
    assert a._inflight.payload + bytes(a._outbuf) == queued


@pytest.mark.parametrize("entry_sl", [1, 3, 6])
def test_uninvited_entry_cs4_uses_actual_entry_speed(entry_sl):
    link = _uninvited(entry_sl=entry_sl)
    link.host.arq.cfg.traffic_sl = 1
    link.host.on_rx_event(_cs(arq.CS_SPEED_UP))
    assert not link.host._uninvited_entry
    assert link.host.arq.speed_level == min(entry_sl + 1, 6)
    link.host.tick()
    assert link.tx.p3[-1] == ("packet", min(entry_sl + 1, 6))


@pytest.mark.parametrize("entry_sl", [1, 3, 6])
def test_default_uninvited_entry_ack_retains_actual_speed(entry_sl):
    link = _uninvited(entry_sl=entry_sl)
    link.answer()
    assert not link.host._uninvited_entry
    assert link.host.arq.speed_level == entry_sl
    assert link.tx.p3[-1] == ("packet", entry_sl)


def test_entry_cs4_renders_decodable_sl2_with_pending_counter_and_payload():
    host, tx = linked()
    host.arq.entry_pending = True
    host.arq.cfg.traffic_sl = 1
    host.arq.cfg.long_cycle = False
    body = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3
    host.arq.on_host_data(body)
    host.tick()
    seq = host.arq.tx_seq
    host.on_rx_event(_cs(arq.CS_SPEED_UP))

    _, pcm = tx.frames[-1]
    packets = p3rx.decode_p3_packets(np.pad(pcm, (14400, 14400))).packets
    assert len(packets) == 1
    packet = packets[0]
    assert packet.sl == 2 and not packet.long_cycle
    assert packet.status & spec.STATUS_SEQ == (seq + 1) % arq.SEQ_MOD
    assert packet.payload == body[:23]
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == body


def test_default_entry_ack_keeps_acquired_speed():
    host, tx = _granted()
    seq = host.arq.tx_seq
    host.on_rx_event(_cs(arq.CS_REQUEST if seq & 1 else arq.CS_ACK))
    assert not host.arq.entry_pending
    assert tx.p3[-1] == ("packet", 1)


def test_default_entry_cs6_changes_length_without_raising_speed():
    host, tx = _granted()
    host.on_rx_event(_cs(arq.CS_CYCLE_TOG))
    assert not host.arq.entry_pending
    assert host.arq.cycle_long
    assert tx.p3[-1] == ("packet", 1)


def test_entry_repeat_does_not_complete_entry_or_change_speed():
    host, tx = _granted(traffic_sl=6)
    seq = host.arq.tx_seq
    host.on_rx_event(_cs(arq.CS_ACK if seq & 1 else arq.CS_REQUEST))
    assert host.arq.entry_pending
    assert host.arq.speed_level == 1
    assert tx.p3[-1] == ("entry", 1)
    assert host.arq.tx_seq == seq


def test_bare_cs4_while_receiving_does_not_complete_entry():
    host, _ = _granted()
    host.arq.role = arq.IRS
    host.on_rx_event(_cs(arq.CS_SPEED_UP))
    assert host.arq.entry_pending
    assert host.arq.speed_level == 1
    assert host.protocol == spec.Protocol.PACTOR3


def test_entry_cs4_respects_configured_maximum():
    host, tx = _granted(traffic_sl=6)
    host.arq.cfg.max_sl = 1
    host.on_rx_event(_cs(arq.CS_SPEED_UP))
    assert not host.arq.entry_pending
    assert tx.p3[-1] == ("packet", 1)
    assert host.arq._inflight.trial_from is None
