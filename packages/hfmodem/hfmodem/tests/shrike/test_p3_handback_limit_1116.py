# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A changeover the peer never reads ends the link instead of looping.

WS8EOC 40 m 2026-11-16 21:57Z (arm 4e11cdb7): the gateway keyed one empty SL1
0x41 invite, counter 1, every cycle from TX[32] on. Our break-in bursts of
three (TX[32]-[34], [37]-[39], ...) went unread, 0 of 24, and each hand-back
ACKed the repeat and broke in again behind the next copy: eight hand-backs on
counter 1. The gateway went silent at about TX[70]; that hand-back had seen no
repeat request, so CS2 ACKs TX[71]-[88] went into nothing at crc_ticks up to
20/96 until the operator interrupted.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_breakin_d5 import Seam
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import opportunity

INVITE = 0x41                   # SL1, counter 1, changeover bit
CHANGEOVER = b"\x010W"          # TX[32]'s three bytes
ACKS_INTO_SILENCE = 18          # TX[71]-[88]
P3 = spec.Protocol.PACTOR3


def invited(**cfg):
    io = Seam()
    a = arq.PactorArq(io, arq.ArqConfig(**cfg))
    a.role = arq.IRS
    a._enter_connected()
    a.breakin_bytes_override = 3
    a.on_host_data(CHANGEOVER + b"pending")
    return a, io


def plain(a, heard=False):
    a.on_cycle()


def air(a, heard=False):
    opportunity(a, source="current CRC" if heard else None)


def burst(a, io, cycle=plain):
    """A copy of the invite arms the break-in, which keys `p3_breakin_tries`
    times unread; what the slot after that keys is returned."""
    a.on_rx_packet(1, b"", INVITE, True, protocol=P3)
    assert a._breakin_armed
    start = len(io.events)
    cycle(a, heard=True)
    for _ in range(a.cfg.p3_breakin_tries - 1):
        cycle(a)
    assert io.events[start:] == [("packet", CHANGEOVER, True)] * a.cfg.p3_breakin_tries
    cycle(a)
    return io.events[start + a.cfg.p3_breakin_tries:]


def handed_back(a, io, cycle=plain):
    assert burst(a, io, cycle) == [("cs", arq.CS_ACK)]
    assert a.role == arq.IRS and a.state == arq.State.CONNECTED
    # TX[36]: the first repeat of the invite after the hand-back is ACKed...
    a.on_rx_packet(1, b"", INVITE, True, protocol=P3)
    assert io.events[-1] == ("cs", arq.CS_ACK) and not a._breakin_armed
    keyed = len(io.events)
    cycle(a, heard=True)
    assert len(io.events) == keyed      # ...as that cycle's answer


def test_the_second_unread_burst_on_one_invite_ends_the_link_on_a_qrt():
    a, io = invited()
    logged = []
    io.log = logged.append
    handed_back(a, io)
    assert a._handbacks == 1
    after = burst(a, io)
    assert a._qrt_pending and a.state == arq.State.DISCONNECTING
    assert "changeover never read after 2 bursts -> QRT" in logged
    assert any(l.startswith("!! the peer kept counter 1 through 2 bursts")
               for l in logged)
    # The slot TX[40] took for the hand-back's ACK keys the goodbye instead.
    assert len(after) == 1 and after[0][0] == "packet"
    assert a._inflight is not None and a._inflight.status & spec.STATUS_QRT
    assert a.transfer_unconfirmed is not None


def test_zero_keeps_the_old_unbounded_hand_back():
    a, io = invited(p3_handback_limit=0)
    for _ in range(8):                  # the arm's eight hand-backs
        handed_back(a, io)
    assert not a._qrt_pending and a.state == arq.State.CONNECTED


def test_a_gateway_that_reads_the_second_burst_keeps_the_link():
    a, io = invited()
    handed_back(a, io)
    a.on_rx_packet(1, b"", INVITE, True, protocol=P3)
    a.on_cycle()
    a.on_cycle()
    assert a.unconfirmed_breakin
    a.on_rx_cs(arq.CS_ACK)
    assert a.role == arq.ISS and not a.unconfirmed_breakin
    assert a._handbacks == 0 and not a._qrt_pending


def test_a_new_counter_between_hand_backs_is_progress():
    # K4MSU 11-14 05:22Z, WS8EOC 05:42Z and 15:53Z: one hand-back, then the
    # gateway's next counter, and a changeover read later in the link.
    a, io = invited()
    handed_back(a, io)
    a.on_rx_packet(1, b"", 0x42, True, protocol=P3)
    assert a._handbacks == 0 and a._breakin_armed
    for _ in range(a.cfg.p3_breakin_tries + 1):
        a.on_cycle()
    assert io.events[-1] == ("cs", arq.CS_ACK)
    assert a._handbacks == 1 and not a._qrt_pending


def silent_after_hand_back(mode):
    a, io = invited(p3_resumed_silence=mode, p3_handback_limit=0,
                    p3_handback_silence="lifetime")
    for _ in range(7):
        handed_back(a, io, air)
    # The eighth hand-back, TX[67]-[70]; the gateway keyed nothing after it.
    assert burst(a, io, air) == [("cs", arq.CS_ACK)]
    acks = 0
    while (a.state == arq.State.CONNECTED and not a._qrt_pending
           and acks < ACKS_INTO_SILENCE):
        opportunity(a)
        acks += 1
    return a, acks


def test_silence_after_a_repeated_req_less_hand_back_ends_on_the_budget():
    a, acks = silent_after_hand_back("unread")
    assert a._qrt_pending and acks == a.cfg.max_retries + 1


@pytest.mark.parametrize("mode", ["budget", "lifetime"])
def test_the_older_rules_keep_the_arms_acks_into_silence(mode):
    a, acks = silent_after_hand_back(mode)
    assert acks == ACKS_INTO_SILENCE and not a._qrt_pending
    assert a._p3_crc_wait_ticks < arq.P3_CRC_WAIT_TICKS


def test_the_flags_reach_the_arq():
    args = onair.build_parser().parse_args([])
    assert args.p3_handback_limit == arq.ArqConfig.p3_handback_limit == 2
    assert args.p3_resumed_silence == arq.ArqConfig.p3_resumed_silence == "unread"
    args = onair.build_parser().parse_args(
        ["--p3-handback-limit", "0", "--p3-resumed-silence", "budget"])
    assert args.p3_handback_limit == 0 and args.p3_resumed_silence == "budget"
    with pytest.raises(SystemExit):
        onair.build_parser().parse_args(["--p3-handback-limit", "-1"])
    with pytest.raises(ValueError):
        arq.ArqConfig(p3_handback_limit=-1)
