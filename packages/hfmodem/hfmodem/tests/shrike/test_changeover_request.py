# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Status-byte bit 6, from the air to the role machine.

An ISS that has run out of things to say does not go quiet and does not hand the
channel over by itself: it keeps sending idle packets with bit 6 standing until
the other end's CS3-headed changeover packet arrives. Every bit-6 packet this
station has ever decoded is that -- 19 of them over ten sessions, three
gateways and two bands, all `0x41`, all with an empty field, and in all ten
sessions AFTER the peer had already taken the link with a changeover of its own
(its CS3 head, or the whole packet, decoded in an earlier cycle). So bit 6 is
never a polite request that escalates to a changeover; it is what the station
holding the link says while it waits for one.

Three things follow, and all three are asserted here:

  * a repeat of it is one standing request, not a second station claiming the
    channel. Graded as the stranded-ISS signature it hands the link back to the
    station asking us to take it, which asks again -- WS8EOC 3596500 kHz on
    2026-08-29 ran that loop twice and keyed sixteen empty 0.96 s packets over a
    gateway that was waiting for one.
  * a changeover packet's OWN status byte cannot carry it. That packet is the
    peer taking the channel, and arming a break-in off it keys one into the
    burst we just yielded to.
  * bit 7 outranks it, and outranks a break-in the host asked for. A changeover
    is something to do with a link; QRT is the end of one, so a peer that asks
    for the channel and then signs off is answered with an acknowledgement and
    not with a changeover packet keyed at a station that has gone.

The first case is read back off rendered audio rather than off a status byte
this file builds, so what is asserted is what a receiver hears.

Run: python -m pytest hfmodem/tests/shrike/test_changeover_request.py
"""
from __future__ import annotations

from hfmodem.shrike import p1rx, pactor1, spec
from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, IRS, ISS, ArqConfig, ArqIO,
                                P1_SPEED_LEVEL, PactorArq, State)

CFG = ArqConfig(max_retries=4)

#: The byte itself: counter 1, empty field, bit 6 up. Counter 1 because "das
#: erste normale Datenpaket mit Head=AA (HEX) und Paketzaehler=1" -- a station
#: that has just taken the link and has nothing to send is on its first packet.
ASKING = 0x41

#: The goodbye, which rides a data packet the same way: counter 2, bit 7 up.
SIGNING_OFF = 0x82


class _Air(ArqIO):
    def __init__(self):
        self.packets: list[tuple[int, bool]] = []
        self.cs: list[int] = []
        self.lines: list[str] = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((status, breakin))

    def send_cs(self, cs_index):
        self.cs.append(cs_index)

    def send_p1_cs(self, index):
        self.cs.append(index)

    def upgrade(self, payload_waiting):
        return False

    def log(self, msg):
        self.lines.append(msg)


def _receiving() -> tuple[_Air, PactorArq]:
    """The arm's own route into the receiving role: the peer broke in."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KI0BK")
    a.on_rx_cs(CS_ACK)
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS
    return air, a


def test_the_byte_a_gateway_sends_reads_as_a_request():
    """0x41 off the air, through the FSK front end and the frame gates."""
    audio = pactor1.packet_signal(b"", 200, packet_count=1,
                                  changeover_request=True)
    got = p1rx.decode_p1_packets(audio)
    assert len(got) == 1, got
    p = got[0]
    assert p.status == ASKING, hex(p.status)
    assert p.payload == b""
    assert p.packet_count == 1
    assert p.changeover_request and not p.qrt


def test_the_qrt_bit_is_read_too():
    audio = pactor1.packet_signal(b"73", 100, packet_count=3, qrt=True)
    got = p1rx.decode_p1_packets(audio)
    assert len(got) == 1, got
    assert got[0].qrt and not got[0].changeover_request


def test_the_data_type_stops_below_bit_4():
    """PACTOR-1's Datenmodus is bits 2-3; bit 4 is "noch nicht belegt".

    This asserted the opposite, on this synthetic render and nothing else, and
    the recordings refuse it. Both PACTOR-1 stations the corpus can check
    announce with bit 4 set -- the capability declaration a gateway answers with
    an upgrade grant -- and under a three-bit read each was reported as a PMC
    mode and its field handed to that decompressor: DL6MAA's 0x35 is plain
    Huffman and says `1dl6maa`, W4DNA's 0x31 is ASCII and says `1w4dna`, while
    three bits make them PMC German swapped and PMC German and produce garbage
    from both. `test_p1data.py` holds that against the audio; this is the
    property alone, and bit 4 is set here so it can still fail.
    """
    audio = pactor1.packet_signal(b"", 100, packet_count=1, bits45=1)
    got = p1rx.decode_p1_packets(audio)
    assert len(got) == 1, got
    assert got[0].status & 0x10, "the render did not set bit 4"
    assert got[0].data_type == spec.DataType.ASCII_8BIT


def test_a_standing_request_is_answered_once_and_not_re_graded():
    """The peer asks; we take the link; it asks again because it did not hear us.

    The second ask must not read as the peer claiming the channel. Yielding on
    it puts the link back where it started and the whole exchange repeats.
    """
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, b"", ASKING, True)
    a.on_cycle()
    assert a.role == ISS, "the changeover request was not acted on"
    assert air.packets and air.packets[-1][1], "no changeover packet went out"

    for _ in range(CFG.max_retries + 2):
        if a.state not in (State.CONNECTED, State.DISCONNECTING):
            break
        a.on_rx_packet(P1_SPEED_LEVEL, b"", ASKING, True)
        a.on_cycle()
    assert a.role == ISS, "we handed the channel back to the station asking for it"
    assert not any("sent a data packet of its own" in m for m in air.lines), \
        air.lines[-4:]
    assert any("still asking us to send" in m for m in air.lines), air.lines[-4:]


def test_a_packet_with_bit_6_clear_is_still_the_strand():
    """The other reading has to keep working: a peer sending ordinary data while
    we hold the sending role is the lost turnaround the yield exists for."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KI0BK")
    a.on_rx_cs(CS_ACK)
    a.on_host_data(b"hello")
    for _ in range(CFG.max_retries + 4):
        if a.role == IRS:
            break
        a.on_rx_packet(1, b"de KI0BK", 0, True)
        a.on_cycle()
    assert a.role == IRS, "the strand no longer yields"
    assert any("sent a data packet of its own" in m for m in air.lines), \
        air.lines[-4:]


def test_a_changeover_packets_own_bit_6_arms_nothing():
    """The peer takes the channel and, in the same frame, appears to ask for it
    back. Acting on that keys a break-in into the burst we just yielded to."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KI0BK")
    a.on_rx_cs(CS_ACK)
    a.on_host_data(b"hello")
    a.on_cycle()
    a.on_rx_packet(P1_SPEED_LEVEL, b"", ASKING, True, breakin=True)
    a.on_cycle()
    assert a.role == IRS, "we broke in on the packet that took the link from us"
    assert not any(brk for _, brk in air.packets), air.packets


def test_the_air_and_the_reader_agree_on_the_asking_byte():
    """The status byte the transmitter builds for a changeover request is the
    one the corpus of decoded sessions holds, bit for bit."""
    assert pactor1.status_byte(1, changeover_request=True) == ASKING
    assert ASKING & spec.STATUS_CHANGEOVER
    assert not ASKING & spec.STATUS_QRT


def test_a_goodbye_outranks_a_break_in_this_end_has_not_keyed_yet():
    """Bit 7 on the packet after bit 6, and the break-in is already armed.

    A changeover is something to do with a link; QRT is the end of one. The
    break-in return sat above the QRT test, so a peer that asked for the channel
    and then said goodbye before our changeover packet reached the air was
    answered with the changeover -- keyed at a station that had signed off -- and
    this end, with no QRT of its own pending, then held the channel with idle
    packets until the retry budget ran out. The link ended on a budget instead of
    on the goodbye it was given.
    """
    air, a = _receiving()
    a.on_rx_packet(P1_SPEED_LEVEL, b"", ASKING, True)
    assert a._breakin_pending and a._breakin_armed
    a.on_rx_packet(P1_SPEED_LEVEL, b"73 SK", SIGNING_OFF, True)
    assert a.state in (State.DISCONNECTED, State.LISTENING), a.state
    assert air.cs and air.cs[-1] == CS_ACK, air.cs
    a.on_cycle()
    assert not any(brk for _, brk in air.packets), air.packets


def test_a_goodbye_outranks_a_break_in_the_host_asked_for():
    """The same window, reached the way a mail session reaches it: the B2F layer
    owes a line, `app_turns` arms the break-in, and the gateway's next packet is
    its QRT."""
    air, a = _receiving()
    a.on_host_data(b"FF\r")
    a.on_host_breakin()
    assert a._breakin_pending
    a.on_rx_packet(P1_SPEED_LEVEL, b"73 SK", SIGNING_OFF, True)
    assert a.state in (State.DISCONNECTED, State.LISTENING), a.state
    a.on_cycle()
    assert not any(brk for _, brk in air.packets), air.packets


def test_a_packet_without_the_goodbye_still_arms_the_break_in():
    """The other side of the same branch: an ordinary packet with a break-in
    pending is answered by the changeover packet and by nothing else."""
    air, a = _receiving()
    a.on_host_data(b"FF\r")
    a.on_host_breakin()
    a.on_rx_packet(P1_SPEED_LEVEL, b"de KB5LZK", 1, True)
    assert a._breakin_armed and not air.cs
    a.on_cycle()
    assert a.role == ISS
    assert air.packets and air.packets[-1][1], air.packets
