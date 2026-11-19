# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A stint handed back after unread changeovers ends when the peer keys nothing.

WS8EOC 2026-11-14 05:42Z (arm d1fcd58a), after TX112: the gateway's last REQ,
then nothing at all. TX113 to TX115 went unanswered, the turn was handed back,
and CS1 ACKs TX116 to TX153 went into silence, 37 of them with the CRC-free
lifetime at 3/96 when they began and 40/96 at the operator's interrupt.
"""
import pytest

from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.test_p3_breakin_d5 import taken
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import opportunity

ACKS_INTO_SILENCE = 37


def handed_back(mode):
    a, io = taken()
    a.cfg.p3_resumed_silence = mode
    token = a.begin_receive_opportunity()
    a.on_rx_cs(arq.CS_REQUEST)
    a.qualify_peer(token, a._link_generation, "peer control")
    a.settle_peer_presence()
    a.on_cycle()
    a.finish_receive_opportunity(token)
    for _ in range(a.cfg.p3_breakin_tries):
        opportunity(a)
    assert a.role == arq.IRS and a._p3_crc_wait_ticks == a.cfg.p3_breakin_tries
    acks = 0
    while (a.state == arq.State.CONNECTED and not a._qrt_pending
           and acks < ACKS_INTO_SILENCE):
        opportunity(a)
        acks += 1
    return a, acks


def test_a_handed_back_stint_into_silence_ends_on_the_retry_budget():
    a, acks = handed_back("budget")
    assert a._qrt_pending and acks == a.cfg.max_retries + 1


def test_the_lifetime_knob_keeps_the_arms_acks_into_silence():
    a, acks = handed_back("lifetime")
    assert acks == ACKS_INTO_SILENCE and not a._qrt_pending
    assert a._p3_crc_wait_ticks < arq.P3_CRC_WAIT_TICKS


def test_a_peer_heard_again_lifts_the_budget():
    a, _ = handed_back("lifetime")
    assert a._resumed_unread
    opportunity(a, source="current CRC")
    assert not a._resumed_unread and a._silent_cycles == 0


def test_the_resumed_silence_flag():
    assert onair.build_parser().parse_args([]).p3_resumed_silence == "unread"
    assert onair.build_parser().parse_args(
        ["--p3-resumed-silence", "lifetime"]).p3_resumed_silence == "lifetime"
    with pytest.raises(ValueError):
        arq.ArqConfig(p3_resumed_silence="never")


def test_a_changeover_never_answered_keeps_the_lifetime():
    # The peer never took the receiving role, so it may still be the ISS
    # keying packets we cannot read: WS8EOC 2026-09-13 21:43 CDT
    # (captures/onair-0913-2143) went on keying packets for twelve cycles.
    # The old bound; `test_p3_both_irs_1116` pins what replaced it.
    a, io = taken()
    a.cfg.p3_handback_silence = "lifetime"
    for _ in range(a.cfg.p3_breakin_tries):
        opportunity(a)
    assert a.role == arq.IRS and not a._resumed_unread
    for _ in range(ACKS_INTO_SILENCE):
        opportunity(a)
    assert a.state == arq.State.CONNECTED and not a._qrt_pending
