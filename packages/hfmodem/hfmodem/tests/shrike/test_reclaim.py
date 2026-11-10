# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The reclaim counts codewords with no packet between them, which is what
`arq.RECLAIM_CODEWORDS` says it counts.

A bare codeword from the station we are receiving from is that station holding
the receiving role too -- the strand `_take_link` exists to end. Three of them
running is the evidence; three of them ANYWHERE is not, and the difference is a
whole receive phase. `on_rx_cs` incremented `_peer_receiving` and nothing on the
receive path ever put it back, so the count was monotonic from the moment the
link came up: three phantoms spread across an inbound message reclaimed the
channel from a gateway that was mid-transmission, and the message died there.

The phantoms are real rather than hypothetical. `onair._read_codeword_at_bursts`
runs `cs_anchored` at spurious onsets, whose measured false-accept rate on quiet
audio is 3.6 per 1000 twelve-bit reads, and a Winlink message at 100 Bd is
several hundred cycles with several onsets in each. One to three phantoms over
that span is the ordinary case, not the unlucky one.

A data packet is the refutation: the peer sending one is the peer holding the
SENDING role, which is the reading under which nothing is stranded at all --
the same thing CS3 already says in `on_rx_cs`.

Run: python -m pytest hfmodem/tests/shrike/test_reclaim.py
"""
from __future__ import annotations

from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, IRS, ISS, P1_SPEED_LEVEL,
                                RECLAIM_CODEWORDS, ArqConfig, ArqIO, PactorArq)

CFG = ArqConfig(max_retries=8)

#: One inbound Winlink proposal block's worth of packets, distinct so a
#: duplicate would show up in the delivered stream.
MESSAGE = [f"FC EM ABC{i:02d} 100 50 0\r".encode() for i in range(24)]

#: Cycles the phantom codeword lands in, spread the length of the message.
PHANTOMS = (4, 11, 19)


class _Air(ArqIO):
    def __init__(self):
        self.packets: list[tuple[int, bool]] = []
        self.cs: list[int] = []
        self.data = bytearray()
        self.lines: list[str] = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((status, breakin))

    def send_cs(self, cs_index):
        self.cs.append(cs_index)

    def send_p1_cs(self, index):
        self.cs.append(index)

    def upgrade(self, payload_waiting):
        return False

    def deliver(self, blob):
        self.data += blob

    def log(self, msg):
        self.lines.append(msg)


def _receiving() -> tuple[_Air, PactorArq]:
    """A linked IRS, reached the way a Winlink RMS puts us there: it broke in."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KB5LZK")
    a.on_rx_cs(CS_ACK)
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS
    return air, a


def _inbound(phantoms: tuple[int, ...]) -> tuple[_Air, PactorArq]:
    air, a = _receiving()
    for n, payload in enumerate(MESSAGE):
        if n in phantoms:
            a.on_rx_cs(CS_ACK)
            a.on_cycle()
        a.on_rx_packet(P1_SPEED_LEVEL, payload, (n + 1) % 4, True)
        a.on_cycle()
    return air, a


def test_phantom_codewords_spread_across_a_message_reclaim_nothing():
    air, a = _inbound(PHANTOMS)
    assert len(PHANTOMS) >= RECLAIM_CODEWORDS, "the control is not testing anything"
    assert a.role == IRS, "the link was taken from a gateway that was sending"
    assert not any(brk for _, brk in air.packets), air.packets
    assert bytes(air.data) == b"".join(MESSAGE)


def test_three_running_still_take_the_link_back():
    air, a = _receiving()
    for _ in range(RECLAIM_CODEWORDS):
        a.on_rx_cs(CS_ACK)
        a.on_cycle()
    assert a.role == ISS, "the strand no longer ends"
    assert air.packets and air.packets[-1][1], "no changeover packet went out"
    assert any("the peer is receiving too" in m for m in air.lines), air.lines[-3:]


def test_a_packet_between_them_is_what_breaks_the_run():
    """Two, a packet, two: four codewords and no strand."""
    air, a = _receiving()
    for _ in range(RECLAIM_CODEWORDS - 1):
        a.on_rx_cs(CS_ACK)
        a.on_cycle()
    a.on_rx_packet(P1_SPEED_LEVEL, MESSAGE[0], 1, True)
    a.on_cycle()
    for _ in range(RECLAIM_CODEWORDS - 1):
        a.on_rx_cs(CS_ACK)
        a.on_cycle()
    assert a.role == IRS
    assert bytes(air.data) == MESSAGE[0]
