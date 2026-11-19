# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Both ends receiving after a changeover only the peer read.

WS8EOC 40 m 2026-11-16 22:28:57Z (arm 6e3e5514), tree a91cdfe3f. The gateway
proposed `FC EM XEA6J7D7R24E 620 513 0` / `F> CA`, then keyed an empty SL2
counter-1 packet, status 0x01 (log line 1128). Our `FS +` was queued, so
TX[57] broke in behind it (line 1132), as TX[35] had behind the empty 0x02
that followed `FS Y` and been read at once. The capture holds the gateway's
answer: CS1 at 0 bit errors, -14 Hz, starting 139 ms past TX[57]'s audio end
(stream sample 6615640), ending about 1.5 ms after the 351 ms ISS read window
closed (RX end 6625792), and the same CS1 every 1.25 s to the end of the tape.
We read none of it: TX[58]-[59] re-keyed the changeover, TX[60] handed back
(line 1180), and CS2 ACKs of the gateway's old counter 1 went out from TX[60]
past TX[100] while the gateway, now the IRS, waited for our counter 1.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_breakin_d5 import Seam
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import opportunity

P3 = spec.Protocol.PACTOR3
REPLY = b"FS +\r"
ACKS_INTO_SILENCE = 40          # TX[60]-[100]


def broke_in(**cfg):
    """TX[57]: the changeover keyed behind the gateway's empty counter 1."""
    io = Seam()
    a = arq.PactorArq(io, arq.ArqConfig(**cfg))
    a.role = arq.IRS
    a._enter_connected()
    a.breakin_bytes_override = 3
    logged = []
    io.log = logged.append
    a.on_rx_packet(2, b"24E 620 513 0\rF> CA\r", 0x00, True, protocol=P3)
    a.on_cycle()
    a.on_host_data(REPLY)
    a.on_host_breakin()
    a.on_rx_packet(2, b"", 0x01, True, protocol=P3)
    io.events.clear()
    opportunity(a, source="current CRC")
    assert io.events == [("packet", b"FS ", True)]
    assert a.role == arq.ISS and a.unconfirmed_breakin
    return a, io, logged


def handed_back(**cfg):
    """TX[58]-[59] unread, then TX[60]: the hand-back's ACK of counter 1."""
    a, io, logged = broke_in(**cfg)
    for _ in range(a.cfg.p3_breakin_tries):
        opportunity(a)
    assert io.events[-1] == ("cs", arq.CS_ACK)
    assert a.role == arq.IRS and not a._resumed_unread
    return a, io, logged


def test_the_hand_back_says_the_changeover_went_unanswered():
    _, _, logged = handed_back()
    assert "no answer read to our changeover -> IRS; bytes unacknowledged, ACK first" \
        in logged
    assert not any("peer repeated its previous turn" in line for line in logged)


def test_silence_behind_an_unread_changeover_resumes_our_turn():
    a, io, logged = handed_back()
    for _ in range(a.cfg.p3_breakin_tries):     # TX[61]-[63]
        opportunity(a)
        assert io.events[-1][0] == "cs" and a.role == arq.IRS
    keyed = len(io.events)
    opportunity(a)                              # TX[64]
    assert a.role == arq.ISS and not a.unconfirmed_breakin
    assert io.events[keyed:] == [("packet", b"+\r", False)]
    assert a._inflight.seq == 1 and a.tx_acked_bytes == 3
    assert any(line.startswith("nothing read for 3 cycles behind our unread "
                               "changeover -> taking it as read") for line in logged)


def test_the_resumed_turn_is_bounded_by_the_retry_budget():
    a, io, _ = handed_back()
    for _ in range(a.cfg.p3_breakin_tries + 1):
        opportunity(a)
    sent = 0
    while a.state == arq.State.CONNECTED and not a._qrt_pending and sent < 2 * ACKS_INTO_SILENCE:
        opportunity(a)
        sent += 1
    assert a._qrt_pending and sent <= a.cfg.max_retries + 1


def test_a_peer_packet_behind_the_hand_back_keeps_us_its_receiver():
    a, io, _ = handed_back()
    opportunity(a)
    a.on_rx_packet(2, b"", 0x01, True, protocol=P3)
    for _ in range(a.cfg.p3_breakin_tries + 1):
        opportunity(a)
    assert a._handback_changeover is None
    assert ("packet", b"+\r", False) not in io.events


def test_unchecked_windows_are_not_silence():
    # The 0913-2143 cede: the peer kept keying packets nobody here checked.
    a, io, _ = handed_back()
    for _ in range(ACKS_INTO_SILENCE):
        token = a.begin_receive_opportunity()
        a.on_cycle()
        a.finish_receive_opportunity(token)
    assert a.role == arq.IRS and not a._qrt_pending
    assert io.events[-1][0] == "cs"


def test_qrt_ends_the_stint_on_the_budget():
    a, io, _ = handed_back(p3_handback_silence="qrt")
    acks = 0
    while a.state == arq.State.CONNECTED and not a._qrt_pending and acks < ACKS_INTO_SILENCE:
        opportunity(a)
        acks += 1
    assert a._qrt_pending and acks == a.cfg.max_retries + 1
    assert a.role == arq.IRS


def test_lifetime_keeps_the_arms_acks_into_silence():
    a, io, _ = handed_back(p3_handback_silence="lifetime")
    for _ in range(ACKS_INTO_SILENCE):
        opportunity(a)
    assert a.state == arq.State.CONNECTED and not a._qrt_pending
    assert a.role == arq.IRS and io.events[-1][0] == "cs"


def test_the_flag_reaches_the_arq():
    assert onair.build_parser().parse_args([]).p3_handback_silence \
        == arq.ArqConfig.p3_handback_silence == "resume"
    assert onair.build_parser().parse_args(
        ["--p3-handback-silence", "lifetime"]).p3_handback_silence == "lifetime"
    with pytest.raises(ValueError):
        arq.ArqConfig(p3_handback_silence="never")
