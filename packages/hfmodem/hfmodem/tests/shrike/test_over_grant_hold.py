# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The handover request waits for the waveform the announcement is asking for.

VE3KPG answered both halves of this on September 13. Announcing without bit 6,
it read our PACTOR-3 entry after twenty keyings and then held CS1 over thirty-two
empty packets -- a peer on our packet clock, acknowledging. With `--over` armed,
the very first PACTOR-1 packet of the session went out 0x71: bits 4-5 asking for
PACTOR-3 and bit 6 offering the channel in the same byte. The gateway took the
cheaper of the two, reversed the grid on packet #1, and no entry packet was ever
keyed.

So bit 6 is held back, and it is held rather than dropped: it rides the first
PACTOR-3 packet after the peer has read an entry and the buffer has drained.

WHICH ARMS IT BITES ON is the narrow part. That arm flew `--p1-grant-only`: the
grant was the only door into PACTOR-3 and the announcement was the whole of the
ask, so there was something to lose. Status bits 4-5 by themselves are the
default on every arm not carrying mail, and an ordinary PACTOR-1 link that held
bit 6 on them would have no way left to end an over -- an ISS with a drained
buffer holds the channel until the peer breaks in, and bit 6 is the invitation.
So those arms hand over as before, and what holds the request there is the entry
packet itself, from the grant until the peer has read it.

Run:  python -m pytest hfmodem/tests/shrike/test_over_grant_hold.py
"""
from __future__ import annotations

from hfmodem.shrike import arq, pactor1, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p3_offer import Keyed, cs_event

QUEUED = b"pending application bytes"


class _Air(Keyed):
    """...with the flags each seam was actually handed kept alongside it."""

    p1_status_bits45 = 0

    def __init__(self, bits45: int = 0):
        super().__init__()
        self.p1_status_bits45 = bits45
        self.p1_flags: list[dict] = []
        self.p3_status: list[tuple[str, int]] = []

    def send_p1_packet(self, payload, baud, seq, **kw):
        self.p1_flags.append(kw)
        super().send_p1_packet(payload, baud, seq, **kw)

    def send_packet(self, sl, payload, status, breakin=False):
        self.p3_status.append(("data", status))
        super().send_packet(sl, payload, status, breakin)

    def send_entry_packet(self, sl, payload, status, acquire=False):
        self.p3_status.append(("entry", status))
        super().send_packet(sl, payload, status)

    @property
    def over_keyed(self) -> list[bool]:
        return [bool(kw["changeover_request"]) for kw in self.p1_flags]


def _grant() -> rxfront.Event:
    """The peer commanding PACTOR-3 -- `0x59A` in the answer slot."""
    return rxfront.Event(0.1, "unassigned", "0x59A", protocol=Protocol.PACTOR1,
                         spare=pactor1.CS_59A, sense=0)


def _linked(*, grant_only: bool = False, bits45: int = 0):
    """A connected ISS at 100 Bd, before its first data packet is keyed."""
    air = _Air(bits45)
    host = PtcHost(peer=air, mycall="W9SSJ")
    host.p1_grant_only = grant_only
    host.arq.on_host_connect("W9SSJ", "VE3KPG")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    return host, air


def test_the_announcement_packet_does_not_offer_the_channel():
    """0x71 is the byte that lost the entry: bits 4-5 and bit 6 together."""
    host, air = _linked(grant_only=True)
    host.arq.on_host_over()
    host.tick()
    assert air.over_keyed == [False]
    assert not host.arq._inflight.status & spec.STATUS_CHANGEOVER, \
        hex(host.arq._inflight.status)
    # HELD, NOT DROPPED: the host asked for the turn and still has it coming.
    assert host.arq._over_pending


def test_an_announcing_arm_with_another_door_still_offers_the_turn():
    """Bits 4-5 alone are not the ask -- they are the default on every arm.

    A `--replay` session, or any upgrade arm the operator never typed a door
    flag for, announces at 3 and can still reach PACTOR-3 uninvited. Holding
    bit 6 there takes away the only move an ISS with a drained buffer has: the
    peer never asked for the channel, so nobody breaks in and the link dies
    holding it (`test_breakin.the_changeover_the_peer_never_takes`).
    """
    host, air = _linked(bits45=3)
    host.arq.on_host_over()
    for _ in range(3):
        host.tick()
        host.on_rx_event(cs_event(pactor1.CS_ACK_A))
    assert air.over_keyed and all(air.over_keyed), air.p1_flags


def test_an_announcing_arm_holds_it_from_the_grant_to_the_entry():
    """...and once a grant HAS been drawn, the entry packet holds it anyway."""
    host, air = _linked(bits45=3)
    host.arq.on_host_over()
    host.tick()
    assert air.over_keyed == [True]
    host.arq.on_host_data(QUEUED)
    host.on_rx_event(_grant())
    host.tick()
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    assert [kind for kind, _ in air.p3_status] == ["entry"]
    assert not any(st & spec.STATUS_CHANGEOVER for _, st in air.p3_status), \
        [hex(st) for _, st in air.p3_status]

    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    host.tick()
    kind, status = air.p3_status[-1]
    assert kind == "data" and status & spec.STATUS_CHANGEOVER, (kind, hex(status))


def test_a_grant_only_arm_that_has_spent_its_grant_hands_over_again():
    """The door is a latch, and a closed door is nothing left to ask for.

    `_grant_taken` survives a fallback that ruled PACTOR-3 out, so from here the
    arm announces into a link that can no longer act on a grant. Holding the
    request on that is the same deadlock by another route.
    """
    host, air = _linked(grant_only=True, bits45=3)
    host.arq.on_host_data(QUEUED)
    host.on_rx_event(_grant())
    host.tick()
    assert host.protocol is Protocol.PACTOR3 and host._grant_taken
    host.fall_back("the peer never read the entry")
    assert host.protocol is Protocol.PACTOR1 and host._grant_taken

    air.p1_flags.clear()
    host.arq.on_host_over()
    host.arq._outbuf.clear()
    host.tick()
    assert air.over_keyed == [True], air.p1_flags


def test_a_plain_pactor1_arm_still_hands_the_channel_over():
    """The mail arm: nothing is being asked for, so bit 6 goes out drained."""
    host, air = _linked()
    host.arq.on_host_over()
    host.tick()
    assert air.over_keyed == [True]
    assert host.arq._inflight.status & spec.STATUS_CHANGEOVER


def test_the_request_rides_the_first_read_pactor3_packet():
    """The grant, the entry, the peer's acknowledgement, and only then bit 6."""
    host, air = _linked(grant_only=True, bits45=3)
    host.arq.on_host_over()
    host.tick()                      # the PACTOR-1 announcement, bit 6 withheld
    host.arq.on_host_data(QUEUED)
    host.on_rx_event(_grant())
    host.tick()
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    assert not any(air.over_keyed), air.p1_flags
    # (d) An entry packet cannot carry it: the peer has not acquired the waveform
    # and cannot be asked to take a channel it has not read us on.
    assert [kind for kind, _ in air.p3_status] == ["entry"] * len(air.p3_status)
    assert not any(st & spec.STATUS_CHANGEOVER for _, st in air.p3_status), \
        [hex(st) for _, st in air.p3_status]

    host.on_rx_event(cs_event(arq.CS_ACK, Protocol.PACTOR3))
    assert not host.arq.entry_pending, "the entry was never answered"
    host.tick()
    kind, status = air.p3_status[-1]
    assert kind == "data" and status & spec.STATUS_CHANGEOVER, (kind, hex(status))
    assert not host.arq._outbuf


def test_the_goodbye_outranks_it_on_either_arm():
    """Bit 7 is the end of a link; bit 6 is something to do with one."""
    for grant_only in (False, True):
        host, air = _linked(grant_only=grant_only)
        host.tick()
        host.arq.on_host_over()
        host.arq.on_host_disconnect()
        host.on_rx_event(cs_event(pactor1.CS_ACK_A))
        host.tick()
        status = host.arq._inflight.status
        assert status & spec.STATUS_QRT, hex(status)
        assert not status & spec.STATUS_CHANGEOVER, hex(status)
        assert air.over_keyed[-1] is False, air.p1_flags
