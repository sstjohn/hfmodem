# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-1 QRT packet in SCS's form: the called station's address, reversed.

The recorded packets are gateway QRTs from our own sessions, decoded CRC-valid:
VE3KPG at 100 Bd from onair-0918-1411 and at
200 Bd from onair-0917-0900, WS8EOC at 200 Bd from onair-0917-0845. All seven
recorded carry status 0x83, and all seven follow the same banner's counter-2
packet, so the 3 is the next counter and not a constant: hf-pactor's QRT takes
the running counter, and a counter held at 3 answers a peer reading the
alternation and a peer reading the counter with different codewords.
"""
from __future__ import annotations

import numpy as np

import pytest

from hfmodem.shrike import pactor1, rxfront, spec
from hfmodem.shrike.arq import ISS, State
from hfmodem.shrike.ptc import PtcHost

VE3KPG_100 = bytes.fromhex("0f 47 50 4b 33 45 56 55 83 10 7c")
VE3KPG_200_FIELD = bytes.fromhex(
    "00 ff 00 3f 30 00 33 cf 30 0f 0f 33 30 3c 33 33 33 00 00 00")
WS8EOC_200 = bytes.fromhex(
    "00 ff 00 0f 30 ff 30 33 30 c0 0f 0f 33 3f 33 33 33 00 00 00 83 b7 f1")


class _Keyed:
    def __init__(self) -> None:
        self.packets: list[tuple[bytes, int, int, bool]] = []

    def send_p1_packet(self, payload, baud, seq, *, qrt=False, **_):
        self.packets.append((payload, baud, seq, qrt))

    send_p1_breakin = send_p1_packet

    def __getattr__(self, _name):
        return lambda *a, **kw: None


def _cs(host: PtcHost, index: int) -> None:
    host.on_rx_event(rxfront.Event(0.0, "cs", "", protocol="PACTOR-1", cs=index))


def _packet(host: PtcHost, counter: int, field: bytes = b"x", **status) -> None:
    host.on_rx_event(rxfront.Event(0.0, "packet", "", protocol="PACTOR-1",
                                   packet=(0, spec.status_byte(counter, **status),
                                           field, True)))


def _qrt(peer: _Keyed) -> tuple[bytes, int, int, bool]:
    return next(p for p in peer.packets if p[3])


def _follows_its_data(peer: _Keyed) -> bool:
    at = next(i for i, p in enumerate(peer.packets) if p[3])
    return peer.packets[at][2] == (peer.packets[at - 1][2] + 1) % 4


class _Wire:
    def __init__(self) -> None:
        self.out: list = []
        self.packets: list[tuple[int, bool]] = []

    def send_p1_packet(self, payload, baud, seq, *, qrt=False,
                       changeover_request=False, **_):
        self.packets.append((seq, qrt))
        self.out.append(rxfront.Event(0.0, "packet", "", protocol="PACTOR-1", packet=(
            0, spec.status_byte(seq, changeover_request=changeover_request, qrt=qrt),
            payload, True)))

    def send_p1_cs(self, index):
        self.out.append(rxfront.Event(0.0, "cs", "", protocol="PACTOR-1", cs=index))

    def __getattr__(self, _name):
        return lambda *a, **kw: None


def _deliver(wire: _Wire, host: PtcHost) -> None:
    for ev in wire.out:
        host.on_rx_event(ev)
    wire.out.clear()


def test_recorded_packets_reproduce_from_their_calls() -> None:
    assert pactor1.data_packet(pactor1.qrt_field("VE3KPG"), 100,
                               3, qrt=True)[1:] == VE3KPG_100
    assert pactor1.data_packet(pactor1.qrt_field("WS8EOC", 200), 200,
                               3, qrt=True)[1:] == WS8EOC_200
    assert pactor1.qrt_field("VE3KPG", 200) == VE3KPG_200_FIELD


def test_the_address_reads_back_and_ordinary_fields_do_not() -> None:
    assert pactor1.qrt_address(VE3KPG_100[:8]) == "VE3KPG"
    assert pactor1.qrt_address(WS8EOC_200[:20]) == "WS8EOC"
    assert pactor1.qrt_address(pactor1.qrt_field("DL1ABCDE")) == "DL1ABCD"
    assert pactor1.qrt_address(b"bye now!") is None
    assert pactor1.qrt_address(b"\x1e" * 20) is None


def test_the_caller_signs_off_with_the_peers_call_after_its_data() -> None:
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "VE3KPG")
    _cs(host, pactor1.CS_ACK_A)
    host.arq.on_cycle()
    host.arq.on_host_data(b"73")
    host.arq.on_host_disconnect()
    for cs in (pactor1.CS_ACK_B, pactor1.CS_ACK_A, pactor1.CS_ACK_B):
        _cs(host, cs)
        host.arq.on_cycle()
    payload, baud, seq, _ = _qrt(peer)
    assert any(p[0] == b"73" and not p[3] for p in peer.packets)
    assert payload == pactor1.qrt_field("VE3KPG", baud)
    assert _follows_its_data(peer)
    assert host.arq.state == State.DISCONNECTED


def test_the_answerer_signs_off_with_its_own_call() -> None:
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("VE3KPG", "W9SSJ")
    _packet(host, 1)
    host.arq.on_cycle()
    host.arq.on_host_data(b"hi")
    _packet(host, 2, changeover_request=True)
    host.arq.on_cycle()
    _packet(host, 3)
    host.arq.on_cycle()
    host.arq.on_cycle()
    assert host.arq.role == ISS
    host.arq.on_host_disconnect()
    for cs in (pactor1.CS_ACK_A, pactor1.CS_ACK_B, pactor1.CS_ACK_A):
        _cs(host, cs)
        host.arq.on_cycle()
    payload, baud, seq, _ = _qrt(peer)
    assert payload == pactor1.qrt_field("W9SSJ", baud)
    assert _follows_its_data(peer)


@pytest.mark.parametrize("data", [1, 25, 45, 65])
def test_our_own_answerer_acknowledges_the_goodbye(data: int) -> None:
    caller_wire, called_wire = _Wire(), _Wire()
    caller = PtcHost(peer=caller_wire, mycall="W9SSJ")
    called = PtcHost(peer=called_wire, mycall="VE3KPG")
    called.arq.on_host_listen(True)
    called.arq.on_rx_connect("W9SSJ", "VE3KPG")
    caller.arq.on_host_connect("W9SSJ", "VE3KPG")
    _deliver(called_wire, caller)
    caller.arq.on_host_data(b"x" * data)
    caller.arq.on_host_disconnect()
    for _ in range(8):
        if not caller_wire.out:
            caller.arq.on_cycle()
        _deliver(caller_wire, called)
        called.arq.on_cycle()
        _deliver(called_wire, caller)
    assert caller.arq.goodbye_acked, caller_wire.packets
    assert sum(qrt for _, qrt in caller_wire.packets) == 1


def test_a_received_qrt_closes_the_link_and_delivers_no_address() -> None:
    audio = pactor1.packet_signal(pactor1.qrt_field("W9SSJ"), 100,
                                  packet_count=3, qrt=True,
                                  lead_s=0.1, tail_s=0.2)
    ev = rxfront.decode_expected_p1_packet(np.asarray(audio))
    assert ev is not None
    _, status, field, crc_ok = ev.packet
    assert crc_ok and status == 0x83 and pactor1.qrt_address(field) == "W9SSJ"

    host = PtcHost(peer=_Keyed(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("VE3KPG", "W9SSJ")
    _packet(host, 1, b"hello")
    host.arq.on_cycle()
    host.on_rx_event(rxfront.Event(0.0, "packet", "", protocol="PACTOR-1",
                                   packet=(0, status, field, True)))
    host.arq.on_cycle()
    assert host.arq.state != State.CONNECTED
    assert host.rcvd_total == len(b"hello")
