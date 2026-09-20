# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The last packet of the stint we take, which our changeover packet answers.

`_take_link` flips the role on the cycle tick, and the peer cannot know that for
another cycle: the only thing that tells it is the CS3-headed packet we key from
the new role. A peer with one more packet already cut sends it into a station
that has just stopped being its IRS, and `on_rx_packet` dropped it at the
role guard -- the same line that drops the stranded ISS's traffic. Then the peer
read our changeover packet, which lands in that packet's acknowledgement slot and
IS its acknowledgement (`_yield_link(acked=True)` is this module's own side of
the rule), settled it, and never sent it again. Acknowledged bytes, once per
turnaround, and a Winlink exchange turns the link around every few lines.

Both shapes on file are one packet from the same banner. KB5LZK walked
`RMS Tri`/`mode 1.4`/`.2.0 KB5`/`LZK at t`/`he Arkan` through on 2026-08-22
(`working/night-17-kb5lzk-pactor-40m-2026-08-22.log:284-508`) with no changeover
in the middle of it; put one there -- which is what an attached app does the
moment it has a line to send (`ptc.PtcHost.app_turns`) -- and the counter-0
packet behind it is the one that goes. VE1YZ's 2026-09-02 session is the same
five packets under the same numbering.

The counter is what separates the tail from the strand, and it separates them
exactly: `_take_link` leaves the receive stream where the ended stint left it, so
the tail carries the number this end is still waiting for. A repeat of it does
not, a renumbered station does not, and a codeword from the peer closes the
stream outright -- a station keying codewords is a station receiving.

Run: python -m pytest hfmodem/tests/shrike/test_stinttail.py
"""
from __future__ import annotations

from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, IRS, ISS, P1_SPEED_LEVEL,
                                ArqConfig, ArqIO, PactorArq)

CFG = ArqConfig(max_retries=4)

BANNER = (b"RMS Tri", b"mode 1.4", b".2.0 KB5", b"LZK at t", b"he Arkan")


class _Air(ArqIO):
    def __init__(self):
        self.packets: list[tuple[int, bool]] = []
        self.cs: list[int] = []
        self.data = bytearray()

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


def _greeted() -> tuple[_Air, PactorArq]:
    """The gateway has the channel and has keyed three of the five packets."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KB5LZK")
    a.on_rx_cs(CS_ACK)
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS
    for i, field in enumerate(BANNER[:3]):
        a.on_rx_packet(P1_SPEED_LEVEL, field, i, True, breakin=i == 0)
        a.on_cycle()
    assert bytes(air.data) == b"".join(BANNER[:3]), bytes(air.data)
    return air, a


def _we_take_it(a: PactorArq, *, goodbye: bool = False) -> None:
    """The fourth packet is the one the changeover rides behind: it decodes, its
    acknowledgement is withheld for the CS3, and the tick keys it."""
    a.on_host_data(b"FS Y\r")
    (a.on_host_disconnect if goodbye else a.on_host_breakin)()
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[3], 3, True)
    a.on_cycle()
    assert a.role == ISS


def test_the_field_our_changeover_packet_acknowledges_reaches_the_host():
    air, a = _greeted()
    _we_take_it(a)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    assert bytes(air.data) == b"".join(BANNER), bytes(air.data)


def test_the_tail_is_delivered_once():
    """The counter is the identity inside a stint and stays it across the role:
    a peer that keys the packet again has not read our changeover packet, and the
    answer to that is not a second delivery."""
    air, a = _greeted()
    _we_take_it(a)
    for _ in range(3):
        a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
        a.on_cycle()
    assert bytes(air.data) == b"".join(BANNER), bytes(air.data)


def test_the_tail_is_not_answered_with_a_codeword():
    """We are the sending station. The changeover packet was the acknowledgement
    and the ISS keys nothing else -- a codeword here is a second transmission in
    a cycle that has already had ours."""
    air, a = _greeted()
    keyed = len(air.cs)
    _we_take_it(a)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    assert air.cs[keyed:] == [], air.cs[keyed:]


def test_the_peer_may_replay_the_tail_when_it_takes_the_link_back():
    """8f0c615's guard, over a packet that now HAS been delivered. The peer never
    heard it settle, so it opens its next stint with it -- acknowledged, and not
    delivered a second time."""
    air, a = _greeted()
    _we_take_it(a)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True, breakin=True)
    assert a.role == IRS
    assert bytes(air.data) == b"".join(BANNER), bytes(air.data)
    assert air.cs[-1] == CS_ACK, air.cs[-1]


def test_new_bytes_after_the_tail_are_delivered():
    air, a = _greeted()
    _we_take_it(a)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    a.on_rx_packet(P1_SPEED_LEVEL, b"sas Divi", 1, True, breakin=True)
    assert bytes(air.data) == b"".join(BANNER) + b"sas Divi", bytes(air.data)


def test_a_renumbered_packet_is_still_the_strand():
    """The station that has taken a changeover of its own counts from zero, and
    zero is not the number this end is waiting for. Nothing is delivered and the
    finding `_on_nak`'s yield spends is still recorded."""
    air, a = _greeted()
    _we_take_it(a)
    a.on_rx_packet(P1_SPEED_LEVEL, b"sion of ", 3, True)
    assert bytes(air.data) == b"".join(BANNER[:4]), bytes(air.data)
    assert a._peer_asked_for_channel


def test_a_codeword_from_the_peer_closes_the_stint():
    """A station keying codewords is a station receiving, so nothing of its own
    is still in the air behind them."""
    air, a = _greeted()
    _we_take_it(a)
    a.on_rx_cs(CS_ACK)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    assert bytes(air.data) == b"".join(BANNER[:4]), bytes(air.data)


def test_the_goodbye_takes_the_link_and_still_reads_the_tail():
    """`on_host_disconnect` breaks in because QRT rides a packet, which is the
    same changeover and the same open cycle behind it. The peer's last field is
    the one a mail session ends on."""
    air, a = _greeted()
    _we_take_it(a, goodbye=True)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    assert bytes(air.data) == b"".join(BANNER), bytes(air.data)


def test_the_second_changeover_starts_the_stream_again():
    """b1483e4's reset, unmoved. The tail advanced the expectation; the next
    stint numbers from zero regardless, and its first packet is taken at
    whatever counter it carries."""
    air, a = _greeted()
    _we_take_it(a)
    a.on_rx_packet(P1_SPEED_LEVEL, BANNER[4], 0, True)
    assert a._expected_seq == 1
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS and a._expected_seq == 0 and not a._rx_seen
    a.on_rx_packet(P1_SPEED_LEVEL, b"sas Divi", 2, True)
    assert bytes(air.data) == b"".join(BANNER) + b"sas Divi", bytes(air.data)
