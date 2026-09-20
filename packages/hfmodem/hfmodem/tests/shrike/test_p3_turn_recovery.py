# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Unaccepted local CS3: retain bytes, deduplicate RMS, physically ACK first."""
import pytest

from hfmodem.shrike import arq, pactor1, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_no_p3_fallback import Recorder, cs


class Sink(arq.ArqIO):
    def __init__(self):
        self.sent = []
        self.delivered = []
        self.buffers = []
        self.refuse = False

    def send_packet(self, sl, payload, status, breakin=False):
        self.sent.append(("packet", bytes(payload[:3]), status, breakin))
        return arq.REFUSED if self.refuse else min(3, len(payload))

    def send_cs(self, index):
        self.sent.append(("cs", index))
        return arq.REFUSED if self.refuse else None

    def defer_rx_close(self):
        return True

    def deliver(self, data):
        self.delivered.append(data)

    def buffer(self, size):
        self.buffers.append(size)


def packet(a, seq=0, **kw):
    a.on_rx_packet(1, b"RMS", seq, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3, **kw)


def pending(seq=0):
    io = Sink()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    packet(a, seq)
    a.on_cs_emitted(arq.CS_ACK)
    a.on_cycle()
    a.on_host_data(b"ABCDEF")
    a.on_host_breakin()
    packet(a, seq)
    a.on_cycle()
    assert a.unconfirmed_breakin and a._inflight.payload == b"ABC"
    assert bytes(a._outbuf) == b"DEF"
    return a, io


@pytest.mark.parametrize("seq", range(4))
def test_repeat_requeues_once_and_preserves_receive_stream(seq):
    a, io = pending(seq)
    delivered, buffers = list(io.delivered), list(io.buffers)
    decoder = a._rx_decoder
    packet(a, seq, repeated_stint=True)
    packet(a, seq, repeated_stint=True)
    assert a.role == arq.IRS and not a.unconfirmed_breakin
    assert bytes(a._outbuf) == b"ABCDEF" and a._buffer_raw == 6
    assert io.buffers == buffers and io.delivered == delivered
    assert a._expected_seq == (seq + 1) % 4 and a._rx_seen
    assert a._rx_decoder is decoder
    assert io.sent[-1] == ("cs", arq.CS_ACK)


@pytest.mark.parametrize("physical_ack", [0, 1])
def test_application_waits_for_actual_ack_then_fresh_packet(physical_ack):
    a, io = pending()
    packet(a, repeated_stint=True)
    for _ in range(3):
        a.on_host_breakin()
        assert not a.taking_link
        a.on_cycle()
        assert a.role == arq.IRS
    # Quiet REQUESTs have replaced the un-emitted ACK: they earn no credit.
    a.on_cs_emitted(physical_ack)
    assert not a.taking_link
    packet(a)
    a.on_cycle()
    a.on_cs_emitted(physical_ack)
    assert a.taking_link
    a.on_cycle()
    assert a.role == arq.IRS  # No fresh decoded packet to license the turn.
    packet(a)
    a.on_cycle()
    assert a.role == arq.ISS and a._inflight.payload == b"ABC"


def test_refused_ack_and_reclaim_cannot_bypass_barrier():
    a, io = pending()
    io.refuse = True
    packet(a, repeated_stint=True)
    a._peer_receiving = arq.RECLAIM_CODEWORDS
    a.on_host_breakin()
    a.on_cs_emitted(0)  # No outstanding successful/queued ACK.
    assert not a.taking_link
    a.on_cycle()
    assert a.role == arq.IRS and a._turn_ack_owed


def test_local_goodbye_remains_bounded_while_ack_is_unplaceable():
    a, io = pending()
    a.on_host_disconnect()
    io.refuse = True
    packet(a, repeated_stint=True)
    assert a._qrt_pending and not a.taking_link
    a.on_cycle(elapsed_ticks=arq.GOODBYE_PLACE_TICKS)
    assert a.state == arq.State.DISCONNECTED and a.goodbye_unplaceable


def test_remote_goodbye_outranks_application_after_recovery():
    a, io = pending()
    packet(a, repeated_stint=True)
    a.on_host_breakin()
    a.on_rx_packet(1, b"", spec.STATUS_QRT | 1, True,
                   protocol=spec.Protocol.PACTOR3)
    assert a._rx_close_pending and io.sent[-1] == ("cs", arq.CS_ACK)
    a.on_cs_emitted(1)
    assert a.state == arq.State.DISCONNECTED
    assert not a._turn_ack_owed and not a._turn_ack_pending


def test_genuine_reversal_still_settles_local_bytes():
    a, io = pending()
    packet(a, repeated_stint=False)
    assert a.role == arq.IRS and a._buffer_raw == 3
    assert bytes(a._outbuf) == b"DEF" and not a._turn_ack_owed


def test_real_ack_ends_unconfirmed_turn_before_later_peer_reversal():
    a, io = pending()
    a.on_rx_cs(arq.CS_ACK)
    assert not a.unconfirmed_breakin and a._buffer_raw == 3
    # A stale classification hint is not allowed to undo a confirmed turn.
    packet(a, repeated_stint=True)
    assert a.role == arq.IRS and not a._turn_ack_owed


@pytest.mark.parametrize("case", ["bad_crc", "not_breakin", "p1"])
def test_unqualified_recovery_flag_cannot_requeue(case):
    a, io = pending()
    a.on_rx_packet(1, b"RMS", 0, case != "bad_crc",
                   breakin=case != "not_breakin", repeated_stint=True,
                   protocol=spec.Protocol.PACTOR1 if case == "p1"
                   else spec.Protocol.PACTOR3)
    assert bytes(a._outbuf) == b"DEF" and not a._turn_ack_owed


@pytest.mark.parametrize("refused_retry", [False, True])
def test_bare_p3_head_does_not_settle_or_rotate_pending_turn(refused_retry):
    tx = Recorder()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.arq, io = pending()
    host.protocol = spec.Protocol.PACTOR3
    if refused_retry:
        io.refuse = True
        host.arq.on_cycle()  # Consume the successful turn's answered cycle.
        host.arq.on_cycle()  # The next retransmission reaches the refusing seam.
        assert host.arq._refused_burst
    before = list(io.sent)
    host.on_rx_event(cs(pactor1.CS_CHANGEOVER))
    assert host.arq.unconfirmed_breakin
    assert host.arq._buffer_raw == 6 and bytes(host.arq._outbuf) == b"DEF"
    host.arq.on_cycle()
    assert io.sent == before
    packet(host.arq, repeated_stint=True)
    assert host.arq.role == arq.IRS and host.arq._buffer_raw == 6


def test_synchronous_recovery_ack_cannot_share_a_cycle_with_local_qrt():
    a, io = pending()
    io.defer_rx_close = lambda: False
    a.on_host_disconnect()
    packet(a, repeated_stint=True)
    assert not a._turn_ack_owed
    before = list(io.sent)
    a.on_cycle()
    assert io.sent == before and a.role == arq.IRS


def test_bare_heads_cannot_extend_goodbye_elapsed_budget():
    a, io = pending()
    a.on_host_disconnect()
    a.on_unconfirmed_breakin_head()
    a.on_cycle(elapsed_ticks=arq.GOODBYE_PLACE_TICKS)
    assert a.state == arq.State.DISCONNECTED


def yielded_p3():
    tx = Recorder()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.protocol = spec.Protocol.PACTOR3
    a = host.arq
    a._enter_connected()
    a.cfg.speed_up_after = 100          # keep the ladder out of the codeword
    a.role = arq.ISS
    a._yield_link()
    return host, tx, a


def keyed(tx, mark):
    return [e[3] for e in tx.emissions[mark:] if e[1] == "cs"]


def peer_holding_a_changeover_packet():
    """A symmetric station whose counter-0 changeover packet is on the air."""
    tx = Recorder()
    host = PtcHost(peer=tx, mycall="WS8EOC")
    host.protocol = spec.Protocol.PACTOR3
    a = host.arq
    a._enter_connected()
    a.role = arq.IRS
    a.on_rx_packet(1, b"hi", 1, True, protocol=spec.Protocol.PACTOR3)
    a.on_cs_emitted(arq.CS_REQUEST)
    a.on_cycle()
    a.on_host_data(b"//WL2K greeting")
    a.on_host_breakin()
    a.on_rx_packet(1, b"bye", 2, True, protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    assert a.role == arq.ISS and a.tx_seq == 0
    assert tx.emissions[-1][1:] == ("breakin", b"//W", 0)
    return host, tx


def test_yielded_p3_irs_asks_for_the_changeover_packet_until_it_decodes():
    host, tx, a = yielded_p3()
    assert a.rx_seq == 3 and not a._rx_seen
    mark = len(tx.emissions)
    a.on_cycle()                                    # nothing decoded yet
    a.on_cycle()                                    # ...still nothing
    assert keyed(tx, mark) == [arq.CS_REQUEST, arq.CS_REQUEST]
    mark = len(tx.emissions)
    packet(a)                                       # the changeover packet, counter 0
    assert a._rx_seen and a._expected_seq == 1
    a.on_cycle()                                    # the answer is already keyed
    a.on_rx_packet(1, b"more", 1, True, protocol=spec.Protocol.PACTOR3)
    assert keyed(tx, mark) == [arq.CS_ACK, arq.CS_REQUEST]


@pytest.mark.parametrize("decoded", [False, True])
def test_waiting_codeword_cannot_release_the_peers_changeover_field(decoded):
    """What the codeword we key does at a symmetric peer, end to end.

    The peer grades it against the counter it has on the air, so CS1 in the
    cycle before the changeover packet decodes is an acknowledgement of a field
    we never read -- and the greeting behind it is never sent again.
    """
    host, tx, a = yielded_p3()
    mark = len(tx.emissions)
    if decoded:
        packet(a)
    a.on_cycle()
    word, = keyed(tx, mark)

    peer, peer_tx = peer_holding_a_changeover_packet()
    mark = len(peer_tx.emissions)
    peer.on_rx_event(cs(word))
    peer.arq.on_cycle()
    repeated = peer_tx.emissions[mark:] == [(spec.Protocol.PACTOR3,
                                             "breakin", b"//W", 0)]
    assert repeated is not decoded
    assert (peer.arq.tx_seq == 1) is decoded


def test_yielded_p1_irs_still_asks_for_the_changeover_packet_with_cs2():
    tx = Recorder()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    a = host.arq
    a._enter_connected()
    a.role = arq.ISS
    a._yield_link()
    assert a.rx_seq == 3
    mark = len(tx.emissions)
    a.on_cycle()
    assert [(e[0], e[3]) for e in tx.emissions[mark:]] == [
        (spec.Protocol.PACTOR1, pactor1.CS_ACK_B)]


def test_new_contact_clears_recovery_barrier():
    a, io = pending()
    packet(a, repeated_stint=True)
    a.on_host_abort()
    a.role = arq.IRS
    a._enter_connected()
    assert not a._turn_ack_owed and not a._turn_ack_pending
