# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS's codeword against a gateway that is repeating one packet.

PACTOR-1 acknowledges by ALTERNATION and by nothing else: "Wiederholung des
gleichen CS bedeutet 'REQUEST'". So the codeword the receiving station keys is a
function of the RECEIVE STREAM, not of the cycle -- one toggle per packet the
stream accepts, held on the air until the next one arrives. A repeat of a packet
already accepted is the sending station saying it did not read the codeword, and
the answer to that is the SAME codeword again, which is the only way it can be
delivered.

What the stream carries is the peer's packet counter, in bits 0-1 of every
status byte, and `ptc.PtcHost._p1_cs_for` reads the codeword off it: even -> CS1,
odd -> CS2, held while the counter stands. A toggle stepped on every
acknowledgement instead made the codeword walk, because `arq.on_rx_packet`
acknowledges a duplicate: six repeats of one packet drew CS1 CS2 CS1 CS2 CS1 CS2,
three of which land on whatever the peer last received and read as a Request by
the rule above. Half of every run of acknowledgements was spent asking for a
packet already in hand. `test_p1counter` carries what the same toggle then cost
at a changeover, which is the counter's other half.

Measured, WS8EOC on 2026-08-09 and 2026-08-15: `captures/onair-0809-2202` holds
fifty consecutive cycles of packet #1 answered by that alternating train and a
counter that never moved, and `captures/onair-0815-2123` holds thirteen. The
codeword train is not why those sessions stalled -- the gateway was reading
almost none of them -- but it is what doubles the cost of every one it does read,
and it is wrong under §4.1 whatever the channel is doing.

Run: python -m hfmodem.tests.shrike.test_p1ack
"""
from __future__ import annotations

import sys

from hfmodem.shrike import pactor1, rxfront, spec
from hfmodem.shrike.arq import CS_BREAKIN, IRS, ISS, State
from hfmodem.shrike.ptc import PtcHost

NAMES = {pactor1.CS_ACK_A: "CS1", pactor1.CS_ACK_B: "CS2",
         pactor1.CS_CHANGEOVER: "CS3", pactor1.CS_SPEED: "CS4"}
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


class _Keyed:
    """A peer that records the codewords, and nothing else."""

    def __init__(self) -> None:
        self.cs: list[str] = []

    def send_p1_cs(self, index: int) -> None:
        self.cs.append(NAMES.get(index, str(index)))

    def __getattr__(self, _name):
        return lambda *a, **kw: None


def _packet(counter: int, breakin: bool = False, field: bytes = b"mode 1.4",
            **status):
    return rxfront.Event(0.0, "packet", "", protocol="PACTOR-1", breakin=breakin,
                         packet=(0, spec.status_byte(counter, **status), field,
                                 True))


def receiving(counters, *, breakin_first: bool = True) -> list[str]:
    """One IRS cycle per counter, driven through the event seam the radio uses."""
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    if breakin_first:
        host.arq.on_rx_cs(CS_BREAKIN)                 # the gateway takes the link
    for counter in counters:
        host.on_rx_event(_packet(counter))
        host.arq.on_cycle()
    return peer.cs


def the_role_the_break_in_leaves() -> None:
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.arq.on_rx_cs(CS_BREAKIN)
    check("a break-in makes this station the receiving one",
          host.arq.role == IRS and host.arq.state == State.CONNECTED,
          f"{host.arq.role} / {host.arq.state}")


def a_repeated_packet_draws_the_same_codeword() -> None:
    train = receiving([0] * 6)
    check("six repeats of packet #0 draw one codeword, six times",
          len(set(train)) == 1, " ".join(train))
    check("...and it is the one that acknowledged the packet, not its alternate",
          train == [train[0]] * 6, " ".join(train))


def a_new_packet_toggles() -> None:
    train = receiving([0, 1, 2, 3])
    check("four packets in sequence draw four alternating codewords",
          train == ["CS1", "CS2", "CS1", "CS2"], " ".join(train))


def the_toggle_survives_the_repeats_between() -> None:
    train = receiving([0, 0, 0, 1, 1, 2])
    check("a run of repeats holds the codeword and the next packet still toggles it",
          train == ["CS1", "CS1", "CS1", "CS2", "CS2", "CS1"], " ".join(train))


def the_counter_wraps_without_holding() -> None:
    # mod 4, so #0 comes round again -- and the second one is a different packet
    # carrying the same counter, four cycles of new information later.
    train = receiving([0, 1, 2, 3, 0])
    check("a counter that wraps is a new packet and toggles",
          train == ["CS1", "CS2", "CS1", "CS2", "CS1"], " ".join(train))


def a_changeover_restarts_the_numbering() -> None:
    """A called station that hands the link over and takes it back again.

    `arq._reset_rx_seq` puts the receive counter back to zero as the link changes
    hands, and the codeword goes back with it -- even where that repeats the one
    the stint before ended on. Acceptance is a CHANGE of codeword WITHIN a stint
    (§5) and only within one: across a changeover the packet is new because
    nothing has been fed under the new numbering, not because the counter moved,
    and a toggle that insisted on a change there was answering the peer's
    counter-0 packet with the request to send it again.

    The train after the changeover carries a field of its own because the filler
    every other stage uses would be a byte-exact repeat of the last field
    delivered before it, which `arq.PactorArq._is_replay` reads as the deposed
    ISS putting back a packet it never heard acknowledged.
    """
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("WS8EOC", "W9SSJ")           # answered CS1
    for counter in (1, 2):
        host.on_rx_event(_packet(counter))
        host.arq.on_cycle()
    check("a called station answers CS1, then its caller's counters 1 and 2",
          peer.cs == ["CS1", "CS2", "CS1"], " ".join(peer.cs))

    host.arq.on_host_data(b"login please")
    host.on_rx_event(_packet(3, changeover_request=True))
    host.arq.on_cycle()
    host.on_rx_event(_packet(0))                        # arms the break-in
    host.arq.on_cycle()
    host.arq.on_cycle()                                 # ...which takes the link
    check("the changeover invitation makes this station the sending one",
          host.arq.role == ISS, str(host.arq.role))
    host.arq.on_rx_cs(CS_BREAKIN)
    check("...and the peer's break-in hands it straight back",
          host.arq.role == IRS, str(host.arq.role))

    mark = len(peer.cs)
    delivered = host.rcvd_total
    for counter in (0, 0, 1):
        host.on_rx_event(_packet(counter, field=b";PQ: 8213" if counter else b"RMS Tri "))
        host.arq.on_cycle()
    train = peer.cs[mark:]
    check("the peer's new stint is answered on its own numbering",
          train == ["CS1", "CS1", "CS2"] and host.rcvd_total > delivered,
          f"{' '.join(train)}, rcvd {delivered} -> {host.rcvd_total}")


def the_connect_answer_is_still_cs1() -> None:
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("WS8EOC", "W9SSJ")
    check("the called station still answers a connect with CS1",
          peer.cs[:1] == ["CS1"], " ".join(peer.cs) or "nothing keyed")


def an_irs_waiting_for_a_changeover_packet_asks_for_it() -> None:
    """The codeword owed before anything has been accepted under a new numbering.

    A caller whose peer takes the channel and whose changeover packet does not
    decode is the IRS of a stream with nothing in it, and it owes a codeword
    every cycle regardless. What a real station keys there is CS2 -- "eine
    Wiederholung des BK-Paketes wird mittels CS2 angefordert" -- and hf-pactor's
    `tx_rx_100` is that loop verbatim: `send_cs(2)` on every cycle it waits, and
    the CS1 branch only once `receive_packet` returns. Its mirror `rx_tx_100`
    takes an exact CS1 and nothing else as the acknowledgement of the packet it
    is repeating, so a CS1 keyed while waiting is not merely early: it is the
    codeword that says the packet arrived.

    `arq.rx_seq` says exactly this on its own, with no seed to keep. The
    changeover put the expectation back to counter 0, so the last accepted
    counter reads 3, and 3 is odd.
    """
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.arq.on_rx_cs(CS_BREAKIN)                 # the gateway takes the link
    for _ in range(4):                            # ...and nothing it sends decodes
        host.arq.on_cycle()
    check("an IRS waiting for a changeover packet asks for it, every cycle",
          peer.cs == ["CS2"] * 4, " ".join(peer.cs) or "nothing keyed")
    n = len(peer.cs)
    host.on_rx_event(_packet(0, field=b"RMS Tri "))
    host.arq.on_cycle()
    check("...and the counter-0 packet that finally decodes is acknowledged CS1",
          peer.cs[n:] == ["CS1"], " ".join(peer.cs[n:]) or "nothing keyed")


STAGES = (the_role_the_break_in_leaves,
          a_repeated_packet_draws_the_same_codeword,
          a_new_packet_toggles,
          the_toggle_survives_the_repeats_between,
          the_counter_wraps_without_holding,
          a_changeover_restarts_the_numbering,
          the_connect_answer_is_still_cs1,
          an_irs_waiting_for_a_changeover_packet_asks_for_it)


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
