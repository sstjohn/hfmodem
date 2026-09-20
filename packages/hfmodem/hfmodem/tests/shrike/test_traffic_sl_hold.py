# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where the traffic runs once a peer has answered the entry packet.

A grant opens the PACTOR-3 phase at speed level 1 (`ptc.GRANT_ENTRY_SL`) and the
peer's answer is the only evidence there is that it acquired the waveform. What
that evidence is ABOUT is the level it was keyed at: VE3KPG read the speed-level-1
entry packet twice on 2026-09-13, and both times the speed-level-3 traffic that
followed drew the same codeword 32 and 17 times over -- never accepted, then
silence. `ArqConfig.traffic_sl` is what the arm sets to hold the traffic where the
peer was last read, and the climb off it belongs to the peer's CS4.

Run:  python -m pytest hfmodem/tests/shrike/test_traffic_sl_hold.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import arq, pactor1, rxfront, spec
from hfmodem.shrike.ptc import GRANT_ENTRY_SL, PtcHost

MESSAGE = b"the traffic the entry packet was keyed for"


class Keyed:
    """Every PACTOR-3 packet this station puts on the air, with its level."""

    def __init__(self) -> None:
        self.p3: list[tuple[str, int]] = []

    def attach(self, host) -> None: ...
    def pump(self) -> None: ...
    def cycle(self) -> None: ...
    def connect_burst(self, mycall: str, dxcall: str) -> None: ...
    def send_p1_packet(self, payload, baud, seq, **kw) -> int: return len(payload)
    def send_p1_breakin(self, payload, baud, seq, **kw) -> int: return len(payload)
    def send_p1_cs(self, index: int) -> None: ...
    def send_cs(self, index: int) -> None: ...

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        self.p3.append(("packet", sl))
        return len(payload)

    def send_entry_packet(self, sl, payload, status, acquire=False) -> int:
        self.p3.append(("entry", sl))
        return len(payload)


def _cs(index: int, protocol=spec.Protocol.PACTOR3):
    return rxfront.Event(0.1, "cs", "control", protocol=protocol, cs=index,
                         sense=0)


def _granted(traffic_sl: int | None = None) -> tuple[PtcHost, Keyed]:
    """A granted link with its entry packet on the air and traffic behind it."""
    tx = Keyed()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    if traffic_sl is not None:
        host.arq.cfg.traffic_sl = traffic_sl
    host.p1_grant_only = True
    host.arq.on_host_connect("W9SSJ", "VE3KPG")
    host.on_rx_event(_cs(pactor1.CS_SPEED, spec.Protocol.PACTOR1))
    host.tick()
    host.arq.on_host_data(MESSAGE)
    host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                   protocol=spec.Protocol.PACTOR1,
                                   spare=pactor1.CS_59A, sense=0))
    host.tick()
    assert host.protocol is spec.Protocol.PACTOR3
    assert host.arq.entry_pending
    assert tx.p3 == [("entry", GRANT_ENTRY_SL)]
    return host, tx


def _answer(host: PtcHost, cs: int | None = None) -> None:
    """The peer acknowledges what is in flight; its codeword follows our seq."""
    seq = host.arq.tx_seq
    assert seq is not None
    if cs is None:
        cs = arq.CS_REQUEST if seq & 1 else arq.CS_ACK
    host.on_rx_event(_cs(cs))
    host.tick()


def test_the_default_arm_runs_traffic_at_three():
    host, tx = _granted()
    _answer(host)
    assert host.arq.speed_level == host.arq.cfg.traffic_sl == 3
    assert tx.p3[1:] == [("packet", 3)]


def test_p3_traffic_sl_one_holds_the_level_the_peer_read():
    host, tx = _granted(traffic_sl=1)
    _answer(host)
    assert host.arq.speed_level == 1
    assert tx.p3[1:] == [("packet", 1)], tx.p3
    _answer(host)
    assert host.arq.speed_level == 1
    assert {sl for kind, sl in tx.p3 if kind == "packet"} == {1}


def test_the_peer_still_climbs_the_held_link():
    """CS4 is the receiving station's own gear command, and holding the traffic
    at the level the peer read does not take it away."""
    host, tx = _granted(traffic_sl=1)
    _answer(host)
    for expected in (2, 3, 4):
        _answer(host, arq.CS_SPEED_UP)
        assert host.arq.speed_level == expected
    assert [sl for kind, sl in tx.p3 if kind == "packet"] == [1, 2, 3, 4]


@pytest.mark.parametrize("sl", [1, 2, 3, 4, 5, 6])
def test_every_level_the_flag_accepts_is_the_level_that_flies(sl: int):
    host, tx = _granted(traffic_sl=sl)
    _answer(host)
    assert host.arq.speed_level == sl
    assert tx.p3[1] == ("packet", sl)
