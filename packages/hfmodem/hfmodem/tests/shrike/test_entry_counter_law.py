# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The packet counter runs straight through an answered entry packet.

VE3KPG granted PACTOR-3 on 2026-09-13, took the speed-level-1 entry packet
`status=0x1a` -- counter 2 -- and answered it CS1. This station then keyed one
data packet a cycle for 35 cycles and the peer answered CS1 to every one of
them, which is the codeword that acknowledges an EVEN counter: the peer stayed
on the entry and never accepted the packet behind it
(`working/pactor3-header-0913/`).

The counter our first post-entry packet carried is not printed anywhere in that
transcript. Reconstructed through these seams it is 3 -- the entry's counter
plus one, which is what pactor3.md's "stop-and-wait cannot skip a counter"
requires and what the DL6MAA reference keys -- and the scenes below are what
holds it there. `_the_old_hypothesis` is the counterfactual the transcript was
first read as: a first packet numbered 1 deadlocks a spec-law peer on CS1
exactly as the air did, which is why the number is worth a regression.

Run:  python -m pytest hfmodem/tests/shrike/test_entry_counter_law.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import arq, onair, pactor1, rxfront, spec
from hfmodem.shrike.ptc import GRANT_ENTRY_SL, PtcHost

MESSAGE = b"the traffic the entry packet was keyed for"

# The counter the PACTOR-1 phase leaves behind it: one data packet at counter 1,
# so the entry packet is 2 -- measured, DL6MAA and every VE3KPG arm.
ENTRY_SEQ = 2


class Peer:
    """An IRS that follows the acknowledgement law and nothing else.

    CS1 answers an even counter and CS2 an odd one (pactor3.md §14), and a
    counter that is not the one it is waiting for leaves its codeword where it
    was -- which is how a real peer says "send that one again" without having a
    word for it.
    """

    def __init__(self, expecting: int = ENTRY_SEQ) -> None:
        self.expecting = expecting
        self.word = arq.CS_REQUEST if expecting & 1 else arq.CS_ACK
        self.accepted: list[int] = []

    def read(self, status: int) -> int:
        seq = status & spec.STATUS_SEQ
        if seq == self.expecting:
            self.accepted.append(status)
            self.word = arq.CS_REQUEST if seq & 1 else arq.CS_ACK
            self.expecting = (seq + 1) % arq.SEQ_MOD
        return self.word


class Keyed:
    """Every PACTOR-3 burst this station puts on the air, with its status byte."""

    def __init__(self) -> None:
        self.p3: list[tuple[str, int, int]] = []

    def attach(self, host) -> None: ...
    def pump(self) -> None: ...
    def cycle(self) -> None: ...
    def connect_burst(self, mycall: str, dxcall: str) -> None: ...
    def send_p1_packet(self, payload, baud, seq, **kw) -> int: return len(payload)
    def send_p1_breakin(self, payload, baud, seq, **kw) -> int: return len(payload)
    def send_p1_cs(self, index: int) -> None: ...
    def send_cs(self, index: int) -> None: ...

    @property
    def last(self) -> tuple[str, int, int]:
        return self.p3[-1]

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        self.p3.append(("packet", sl, status))
        return len(payload)

    def send_entry_packet(self, sl, payload, status, acquire=False) -> int:
        self.p3.append(("entry", sl, status))
        return len(payload)


def _cs(index: int):
    return rxfront.Event(0.1, "cs", "control", protocol=spec.Protocol.PACTOR3,
                         cs=index, sense=0)


def _granted(*, payload: bytes = MESSAGE) -> tuple[PtcHost, Keyed]:
    """A granted link with its entry packet on the air."""
    tx = Keyed()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.arq.cfg.traffic_sl = 1
    host.p1_grant_only = True
    host.arq.on_host_connect("W9SSJ", "VE3KPG")
    host.on_rx_event(rxfront.Event(0.1, "cs", "control",
                                   protocol=spec.Protocol.PACTOR1,
                                   cs=pactor1.CS_SPEED, sense=0))
    host.tick()
    if payload:
        host.arq.on_host_data(payload)
    host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                   protocol=spec.Protocol.PACTOR1,
                                   spare=pactor1.CS_59A, sense=0))
    host.tick()
    assert host.arq.entry_pending
    assert tx.p3 == [("entry", GRANT_ENTRY_SL, 0x1a)]
    return host, tx


def _turn(host: PtcHost, peer: Peer, tx: Keyed) -> None:
    """One cycle: the peer reads what we keyed and answers it."""
    host.on_rx_event(_cs(peer.read(tx.last[2])))
    host.tick()


def _seqs(tx: Keyed) -> list[int]:
    return [status & spec.STATUS_SEQ for _, _, status in tx.p3]


def test_the_entry_packet_carries_the_counter_the_pactor1_phase_reached():
    _, tx = _granted()
    assert tx.last[2] & spec.STATUS_SEQ == ENTRY_SEQ


def test_the_first_packet_after_an_answered_entry_continues_the_counter():
    host, tx = _granted()
    peer = Peer()
    _turn(host, peer, tx)
    assert not host.arq.entry_pending
    kind, _, status = tx.last
    assert kind == "packet"
    assert status & spec.STATUS_SEQ == (ENTRY_SEQ + 1) % arq.SEQ_MOD
    assert peer.accepted == [0x1a]


def test_the_peer_advances_us_once_the_phase_is_right():
    """CS1 answered the entry, and the CS2 behind it settles the packet at 3."""
    host, tx = _granted()
    peer = Peer()
    for _ in range(5):
        _turn(host, peer, tx)
    assert _seqs(tx) == [2, 3, 0, 1, 2, 3]
    assert [s & spec.STATUS_SEQ for s in peer.accepted] == [2, 3, 0, 1, 2]


def test_the_old_hypothesis_deadlocks_the_peer_on_cs1():
    """The counterfactual, and the reason the number above is asserted.

    A first packet numbered 1 is a counter the peer is not waiting for, so its
    codeword never moves off the entry's CS1 -- which is what 35 cycles of the
    2026-09-13 transcript look like from this side.
    """
    host, tx = _granted()
    peer = Peer()
    _turn(host, peer, tx)
    host.arq.requeue_inflight()
    host.arq._next_seq = 1
    host.tick()
    for _ in range(8):
        _turn(host, peer, tx)
    assert set(_seqs(tx)[2:]) == {1}
    assert peer.accepted == [0x1a]
    assert peer.word == arq.CS_ACK


def test_an_unread_announcement_does_not_ride_the_packet_behind_the_entry():
    """The counter is the entry's plus one whatever the PACTOR-1 phase was owed.

    `on_rx_grant` settles the packet the grant arrives behind and carries nothing
    into PACTOR-3, so the field at counter 3 is idle fill -- which is what both
    reference callers key there.
    """
    tx = Keyed()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.arq.cfg.traffic_sl = 1
    host.p1_grant_only = True
    host.arq.on_host_connect("W9SSJ", "VE3KPG")
    for _ in range(2):          # the same codeword twice: a repeat request
        host.on_rx_event(rxfront.Event(0.1, "cs", "control",
                                       protocol=spec.Protocol.PACTOR1,
                                       cs=pactor1.CS_ACK_A, sense=0))
        host.tick()
    host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                   protocol=spec.Protocol.PACTOR1,
                                   spare=pactor1.CS_59A, sense=0))
    host.tick()
    assert tx.last == ("entry", GRANT_ENTRY_SL, 0x1a)
    peer = Peer()
    _turn(host, peer, tx)
    kind, _, status = tx.last
    assert kind == "packet"
    assert status & spec.STATUS_SEQ == (ENTRY_SEQ + 1) % arq.SEQ_MOD
    assert host.arq._inflight.payload == b""
    assert not host.arq._outbuf


def test_the_changeover_request_rides_the_corrected_packet():
    """--over, on a link whose buffer drains into the first post-entry packet."""
    host, tx = _granted(payload=b"over")
    peer = Peer()
    host.arq.on_host_over()
    _turn(host, peer, tx)
    kind, _, status = tx.last
    assert kind == "packet"
    assert status & spec.STATUS_SEQ == (ENTRY_SEQ + 1) % arq.SEQ_MOD
    assert status & spec.STATUS_CHANGEOVER
    _turn(host, peer, tx)
    assert peer.accepted[-1] & spec.STATUS_CHANGEOVER


# -- what the session says it sent --------------------------------------------

def test_the_counter_account_records_the_pactor3_train(capsys, tmp_path):
    """Through the real transmitter, which is what the summary reads.

    The account was fed from the PACTOR-1 send path alone, so a session whose
    whole link ran in PACTOR-3 reported the counters of the two packets it left
    PACTOR-1 with: `packet counters sent: #1`, and a verdict that our counter
    never advanced over a train that was numbered 1, 2, 3, 0.
    """
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path)
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.arq.cfg.traffic_sl = 3
    host.p1_grant_only = True
    host.arq.on_host_connect("W9SSJ", "VE3KPG")
    host.on_rx_event(rxfront.Event(0.1, "cs", "control",
                                   protocol=spec.Protocol.PACTOR1,
                                   cs=pactor1.CS_SPEED, sense=0))
    host.tick()
    host.arq.on_host_data(MESSAGE)
    host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                   protocol=spec.Protocol.PACTOR1,
                                   spare=pactor1.CS_59A, sense=0))
    host.tick()
    peer = Peer()
    for _ in range(3):
        host.on_rx_event(_cs(peer.read(tx.seq_sent[-1])))
        host.tick()
    assert tx.seq_sent == [1, ENTRY_SEQ, 3, 0, 1]
    assert tx.p1_seq == [1]
    capsys.readouterr()
    onair._summary(cs_log=[], tx_slots=[], cycles=1, keyed=[], cycle_s=1.25,
                   seq_sent=tx.seq_sent, evidence=onair._ConnectEvidence(),
                   ended="test", breakin_at=tx.breakin_at)
    assert "packet counters sent: #1, #2, #3, #0, #1" in capsys.readouterr().out


# -- the grid's side of the same cycles ---------------------------------------

def _iss_grid() -> onair._MasterGrid:
    grid = onair._MasterGrid(anchor=0, slot_n=onair.FS * 5 // 4,
                             offset_n=0, packet_n=onair.FS,
                             cs_n=onair.FS // 10, d_max_n=onair.FS // 8)
    grid.protocol = spec.Protocol.PACTOR3
    grid.sending = True
    grid.acquired = True
    return grid


def test_a_decoded_p3_codeword_credits_the_sending_grid():
    grid = _iss_grid()
    for cycle in range(6):
        grid.note_peer_codeword(cycle * grid.slot_n + 1000, onair.P3_CS_N,
                                "ACK", "VE3KPG", protocol=spec.Protocol.PACTOR3)
        line = grid.update([], np.zeros(0, np.float32), 0, linked=True)
        assert grid.blind == 0
        assert "NO CONTROL SIGNAL" not in line
        assert "ACK we decoded" in line


def test_the_count_climbs_again_once_the_codewords_stop():
    grid = _iss_grid()
    grid.note_peer_codeword(1000, onair.P3_CS_N, "ACK", "VE3KPG",
                            protocol=spec.Protocol.PACTOR3)
    grid.update([], np.zeros(0, np.float32), 0, linked=True)
    lines = [grid.update([], np.zeros(0, np.float32), 0, linked=True)
             for _ in range(3)]
    assert grid.blind == 3
    assert all("NO CONTROL SIGNAL" in line for line in lines)


def test_a_pactor1_codeword_does_not_credit_an_upgraded_grid():
    """The clock the count hangs on is the one the credit has to come from."""
    grid = _iss_grid()
    grid.note_peer_codeword(1000, grid.p1_cs_n, "ACK", "VE3KPG",
                            protocol=spec.Protocol.PACTOR1)
    line = grid.update([], np.zeros(0, np.float32), 0, linked=True)
    assert grid.blind == 1
    assert "NO CONTROL SIGNAL" in line
