# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CS5 to our P3 changeover, read as the reference ISS reads it or as a NAK.

WS8EOC 40 m 2026-11-14 15:53Z (arm 3dc08d83): our first changeover drew an
unread answer, then CS5 x6 while we re-keyed it, then CS1, then CS5 on every
copy of counter 1 until the gateway went silent. The reference tape answers
every changeover packet CS5 and its new ISS goes straight on
(`unknowns.CHANGEOVER_ANSWER_CS5`).
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_cs6_alternation_0924 import feed, irs

P3 = spec.Protocol.PACTOR3
WS8EOC_1553 = [arq.CS_NAK] * 6 + [arq.CS_ACK] + [arq.CS_NAK] * 3


def keyed(mode):
    host, a, _ = irs()
    a.cfg.p3_breakin_cs5 = mode
    packets = []
    host.peer.send_packet = lambda sl, payload, status, breakin=False: packets.append(
        (payload, status & spec.STATUS_SEQ, breakin))
    return host, a, packets


def broken_in(mode):
    host, a, packets = keyed(mode)
    for seq in (2, 3, 0):
        feed(a, seq, b"x")
    a.on_host_data(b";FW: W9SSJ\r")
    a.on_rx_packet(1, b"", spec.status_byte(1) | spec.STATUS_CHANGEOVER, True, protocol=P3)
    a.on_cycle()
    a.on_cycle()                     # its first answer unread: the changeover again
    assert a.role == arq.ISS and a.unconfirmed_breakin
    assert packets == [(b";FW", 0, True)] * 2
    packets.clear()
    return a, packets


def replay(mode):
    a, packets = broken_in(mode)
    answered = []
    for cs in WS8EOC_1553:
        a.on_rx_cs(cs)
        answered.append(packets[-1])
    return answered


CHANGEOVER = (b";FW", 0, True)
DATA1 = (b": W9S", 1, False)
DATA2 = (b"SJ\r", 2, False)


def test_accept_sends_counter_1_after_the_first_cs5():
    assert replay("accept") == [DATA1] * 6 + [DATA2] * 4


def test_repeat_keys_the_changeover_until_the_cs1():
    assert replay("repeat") == [CHANGEOVER] * 6 + [DATA1] * 4


@pytest.mark.parametrize("mode", ["accept", "repeat"])
def test_a_cs5_to_a_data_packet_repeats_it_under_either_value(mode):
    a, packets = broken_in(mode)
    a.on_rx_cs(arq.CS_ACK)
    assert packets == [DATA1]
    for _ in range(3):
        a.on_rx_cs(arq.CS_NAK)
    assert packets == [DATA1] * 4 and a._sl == 1
    a.on_rx_cs(arq.CS_ACK)
    assert packets[-1] == DATA2


def test_the_breakin_cs5_flag():
    assert arq.ArqConfig().p3_breakin_cs5 == "repeat"
    assert onair.build_parser().parse_args([]).p3_breakin_cs5 == "repeat"
    assert onair.build_parser().parse_args(
        ["--p3-breakin-cs5", "accept"]).p3_breakin_cs5 == "accept"
    with pytest.raises(ValueError, match="p3_breakin_cs5"):
        arq.ArqConfig(p3_breakin_cs5="ack")
