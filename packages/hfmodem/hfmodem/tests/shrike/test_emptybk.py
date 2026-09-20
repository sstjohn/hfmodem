# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The gateway's EMPTY changeover packet, and the counter-1 packet behind it.

Five transcripts run the same three cycles, four of them before `443c3c1` and
one after: the gateway keys a CS3-headed packet carrying counter 0 and no field
at all, then a zero-byte `status=0x41 (cnt=1 BK)` inviting us back, and the link
turns around. WS8EOC on 2026-08-29
(`working/onair-0829-2317/pactor-night-12-ws8eoc-80m-bits0.log:146-164`,
`working/onair-0829-2319/pactor-night-13-ws8eoc-80m-bits0.log:95-177`), VE1YZ on
2026-09-02 (`working/onair-0902-2347/pactor-night-10-ve1yz-mail2.log:159-174`),
and KB5LZK on 2026-09-03, the first arm to fly the counter rule
(`working/onair-0903-1051/pactor-day-05-kb5lzk-mail.log:142-158`): `TX[14] P1
CS1` against the counter-0 packet, and our own changeover packet -- never a
codeword -- against the counter-1 one.

AN EMPTY FIELD IS STILL A PACKET, which is the half of the rule these arms turn
on. `arq._accept_field` takes it into the receive stream whatever it carries, so
the counter behind it is new and the codeword moves; a stream that stalled on the
empty packet would hold CS1 against every counter-1 packet the gateway ever
sent -- the banner's failure in the other direction, and the same 45 cycles of
"send that again".

Run: python -m hfmodem.tests.shrike.test_emptybk
"""
from __future__ import annotations

import sys

from hfmodem.shrike import pactor1, rxfront, spec
from hfmodem.shrike.arq import CS_ACK, CS_BREAKIN, IRS, ISS, State
from hfmodem.shrike.ptc import PtcHost

NAMES = {pactor1.CS_ACK_A: "CS1", pactor1.CS_ACK_B: "CS2",
         pactor1.CS_CHANGEOVER: "CS3", pactor1.CS_SPEED: "CS4"}
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


class _Keyed:
    """A peer that records what we put on the air, by shape."""

    def __init__(self) -> None:
        self.cs: list[str] = []
        self.breakins: list[int] = []
        self.packets: list[int] = []

    def send_p1_cs(self, index: int) -> None:
        self.cs.append(NAMES.get(index, str(index)))

    def send_p1_breakin(self, payload, baud, seq, qrt=False):
        self.breakins.append(seq)

    def send_p1_packet(self, payload, baud, seq, **kw):
        self.packets.append(seq)

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append(status & spec.STATUS_SEQ)

    def __getattr__(self, _name):
        return lambda *a, **kw: None


def _packet(counter: int, field: bytes, breakin: bool = False, **status):
    return rxfront.Event(0.0, "packet", "", protocol="PACTOR-1", breakin=breakin,
                         packet=(0, spec.status_byte(counter, **status), field,
                                 True))


def _the_gateway_takes_the_channel() -> tuple[PtcHost, _Keyed]:
    """Our call answered, and the gateway's empty break-in packet behind it."""
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(rxfront.Event(0.0, "cs", "CS1", protocol="PACTOR-1",
                                   cs=pactor1.CS_ACK_A))
    host.arq.on_rx_cs(CS_BREAKIN)
    host.on_rx_event(_packet(0, b"", breakin=True))
    host.arq.on_cycle()
    return host, peer


def the_empty_changeover_packet_is_acknowledged() -> None:
    host, peer = _the_gateway_takes_the_channel()
    check("the gateway's break-in makes this station the receiving one",
          host.arq.role == IRS and host.arq.state == State.CONNECTED,
          f"{host.arq.role} / {host.arq.state}")
    check("its empty counter-0 packet is acknowledged CS1", peer.cs == ["CS1"],
          " ".join(peer.cs) or "nothing keyed")
    check("...with nothing for the host in it", host.rcvd_total == 0,
          f"{host.rcvd_total} bytes")

    for _ in range(4):
        host.on_rx_event(_packet(0, b"", breakin=True))
        host.arq.on_cycle()
    check("...and repeats of it hold that CS1", peer.cs == ["CS1"] * 5,
          " ".join(peer.cs))


def the_counter_1_packet_behind_it_toggles() -> None:
    """The empty field advanced the stream, so the packet behind it is new."""
    host, peer = _the_gateway_takes_the_channel()
    host.on_rx_event(_packet(1, b""))
    host.arq.on_cycle()
    check("the counter-1 packet behind the empty one draws CS2",
          peer.cs == ["CS1", "CS2"], " ".join(peer.cs) or "nothing keyed")

    for _ in range(3):
        host.on_rx_event(_packet(1, b""))
        host.arq.on_cycle()
    check("...and its repeats hold CS2 rather than walking back to CS1",
          peer.cs == ["CS1"] + ["CS2"] * 4, " ".join(peer.cs))


def the_invitation_hands_the_link_back() -> None:
    """`status=0x41` is the gateway asking us to send, and it gets the packet.

    A codeword here would be a second transmission in a cycle that already has
    ours -- the changeover packet lands in the acknowledgement slot and IS the
    acknowledgement.
    """
    host, peer = _the_gateway_takes_the_channel()
    host.on_rx_event(_packet(1, b"", changeover_request=True))
    host.arq.on_cycle()
    check("the counter-1 invitation is answered by our changeover packet",
          peer.cs == ["CS1"] and peer.breakins == [0],
          f"{' '.join(peer.cs)} / breakins {peer.breakins}")
    check("...which makes this station the sending one", host.arq.role == ISS,
          str(host.arq.role))

    host.arq.on_host_data(b"FS Y\r")
    for _ in range(3):
        host.arq.on_cycle()
    check("...and holds that packet until something acknowledges it",
          peer.breakins == [0, 0, 0], f"breakins {peer.breakins}")

    host.arq.on_rx_cs(CS_ACK)
    host.arq.on_cycle()
    check("...then numbers our stint from the changeover packet's own zero",
          peer.packets == [1], f"packets {peer.packets}")



def the_second_changeover_answers_the_same_way() -> None:
    """VE1YZ's position, where the free-running toggle came up CS2.

    The gateway takes the channel, we take it back on its invitation, and it
    takes it back again -- and the counter-0 packet opening that third stint is
    acknowledged by the number it carries, not by how many packets the link has
    seen (`working/onair-0902-2342/pactor-night-09-ve1yz-mail.log:323-360`).
    """
    host, peer = _the_gateway_takes_the_channel()
    host.on_rx_event(_packet(1, b"", changeover_request=True))
    host.arq.on_cycle()
    host.arq.on_cycle()
    mark = len(peer.cs)

    host.arq.on_rx_cs(CS_BREAKIN)
    host.arq.on_cycle()
    check("the cycle waiting for the gateway's next changeover packet asks for it",
          peer.cs[mark:] == ["CS2"], " ".join(peer.cs[mark:]) or "nothing keyed")

    host.on_rx_event(_packet(0, b"RMS Tri", breakin=True))
    host.arq.on_cycle()
    check("...and its counter-0 packet is acknowledged CS1 a second time",
          peer.cs[mark:] == ["CS2", "CS1"], " ".join(peer.cs[mark:]))
    check("...carrying its field to the host", host.rcvd_total == 7,
          f"{host.rcvd_total} bytes")


STAGES = (the_empty_changeover_packet_is_acknowledged,
          the_counter_1_packet_behind_it_toggles,
          the_invitation_hands_the_link_back,
          the_second_changeover_answers_the_same_way)


def main() -> int:
    global ok
    ok = True
    for stage in STAGES:
        stage()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
