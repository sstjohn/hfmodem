# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A counter one past the expected one is taken once the peer has moved on.

K4MSU 40 m 2026-11-14 15:18Z (arm ed4f8779): seq 2, 3, 0 long at SL1, then seq 1
at SL2, 111 bytes ending the banner `CMS via K4MSU >`. Seq 2 was lost to
fading, and K4MSU then repeated seq 3, 0 bytes, status 0x43: the changeover
invitation. We held on the gap and keyed CS2 on seq 1 against every copy.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_cs6_alternation_0924 import feed, irs

P3 = spec.Protocol.PACTOR3
BANNER = b"CMS via K4MSU >\r"


def k4msu(mode="accept"):
    host, a, sent = irs()
    a.cfg.p3_counter_gap = mode
    for seq in (2, 3, 0):
        feed(a, seq, b"x")
    feed(a, 1, BANNER)
    assert a.rx_seq == 1
    host.log_lines.clear()
    sent.clear()
    a.on_host_data(b";FW: W9SSJ\r")
    return host, a, sent


def invite(a):
    a.on_rx_packet(2, b"", 0x43, True, protocol=P3)


def test_the_invitation_past_a_lost_counter_is_acked_and_answered():
    host, a, sent = k4msu()
    invite(a)
    assert a.rx_seq == 3 and a.counters_lost == 1 and a._breakin_armed
    assert any("P3 seq 2 lost" in line for line in host.log_lines)
    # The slot's answer is our changeover, which acknowledges 3 in its place.
    a.on_cycle()
    assert sent == [] and a.role == arq.ISS
    assert a._inflight.breakin and a._inflight.payload == b";FW"


def test_a_plain_packet_past_a_gap_is_asked_for_once_then_taken():
    host, a, sent = k4msu()
    feed(a, 3, b"more")
    held = list(sent)
    assert a.rx_seq == 1 and len(held) == 1
    feed(a, 3, b"more")
    assert a.rx_seq == 3 and a.counters_lost == 1
    assert sent == held + [host._counter_cs_for(arq.CS_ACK)]
    assert bytes(host.channel(host.ptchn).rx).endswith(BANNER + b"more")


def test_the_hold_knob_keeps_the_arms_endless_repeat_request():
    host, a, sent = k4msu("hold")
    for _ in range(5):
        invite(a)
    assert len(sent) == 5 and len(set(sent)) == 1 and a.rx_seq == 1
    assert sum("2 was never read" in line for line in host.log_lines) == 5
    assert a.counters_lost == 0


def test_a_gap_filled_late_leaves_no_request_standing():
    # SEQ_MOD is 4, so a request left from an earlier gap at 3 would take
    # the next gap at 3 unasked and fail a greeting as lost.
    host, a, sent = irs()
    for seq in (2, 3, 0, 1):
        feed(a, seq, b"x")
    feed(a, 3, b"gap")
    assert a.rx_seq == 1 and len(sent) == 5
    feed(a, 2, b"y")
    for seq in (3, 0, 1):
        feed(a, seq, b"z")
    assert a.rx_seq == 1
    sent.clear()
    feed(a, 3, b"newgap")
    assert a.rx_seq == 1 and a.counters_lost == 0
    assert sent == [host._counter_cs_for(arq.CS_REQUEST)]


def test_the_counter_gap_flag():
    assert onair.build_parser().parse_args([]).p3_counter_gap == "accept"
    assert onair.build_parser().parse_args(
        ["--p3-counter-gap", "hold"]).p3_counter_gap == "hold"
    with pytest.raises(ValueError):
        arq.ArqConfig(p3_counter_gap="drop")
