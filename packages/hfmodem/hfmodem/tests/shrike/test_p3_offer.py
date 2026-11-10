# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the upgrade offer costs the peer, and how long it is allowed to stand.

`test_p3_upgrade.py` replays a stranger's PACTOR-1 -> PACTOR-3 session and asks
whether this station follows it. This asks the other half, which no recording can
answer: what OUR station puts on the air when it decides to upgrade, and what it
does with the first thing the peer says back.

MEASURED, and it is the same shape in every archived session that upgraded: the
link reached PACTOR-3, keyed exactly ONE speed-level-3 packet with an EMPTY
field, and was back in PACTOR-1 a cycle later, with the target ruled out for the
rest of the link. Two faults compounding, and they are independent:

  * the offer was taken on a DRAINED buffer -- `arq._on_ack` runs after the
    acknowledged packet's payload has left it -- so the peer's first look at a
    waveform it must acquire from nothing carried 59 bytes of 0x1E fill;
  * the PACTOR-1 codeword that arrived the next cycle was read as the peer's
    verdict on that packet, when the IRS builds its codeword from the cycle
    BEFORE and it was the answer to the PACTOR-1 packet ahead of the upgrade.

Run:  python -m hfmodem.tests.shrike.test_p3_offer
"""
from __future__ import annotations

import sys
from pathlib import Path

from hfmodem.shrike import arq, onair, pactor1, rxfront
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

MESSAGE = b"the manual's own prose, at four times the rate " * 4

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))


class Keyed:
    """Everything the station puts on the air, in the protocol it went out in."""

    def __init__(self) -> None:
        self.packets: list[tuple[Protocol, bytes]] = []
        self.cs: list[tuple[Protocol, int]] = []
        self.host: PtcHost | None = None

    def attach(self, host: PtcHost) -> None:
        self.host = host

    def pump(self) -> None: ...
    def cycle(self) -> None: ...
    def connect_burst(self, mycall: str, dxcall: str) -> None: ...

    def _packet(self, payload: bytes) -> None:
        self.packets.append((self.host.protocol, bytes(payload)))

    def send_p1_packet(self, payload, baud, seq, **kw) -> None:
        self._packet(payload)

    def send_p1_breakin(self, payload, baud, seq, **kw) -> None:
        self._packet(payload)

    def send_packet(self, sl, payload, status, breakin=False) -> None:
        self._packet(payload)

    def send_p1_cs(self, index: int) -> None:
        self.cs.append((Protocol.PACTOR1, index))

    def send_cs(self, index: int) -> None:
        self.cs.append((Protocol.PACTOR3, index))

    @property
    def p3(self) -> list[bytes]:
        return [pl for proto, pl in self.packets if proto is Protocol.PACTOR3]


def cs_event(cs: int, protocol: str = "PACTOR-1"):
    return rxfront.Event(0.1, "cs", f"CS{cs + 1}", protocol=protocol,
                         cs=cs, sense=0)


def calling_station() -> tuple[PtcHost, Keyed]:
    """A linked ISS at 100 Bd with its first packet -- the callsign announcement
    `_answer_link_setup` queues -- already on the air and unacknowledged."""
    keyed = Keyed()
    host = PtcHost(peer=keyed, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))     # CS4: the link runs 100 Bd
    host.tick()
    return host, keyed


def acknowledge(host: PtcHost, cs: int = pactor1.CS_ACK_A) -> None:
    """The peer's acknowledgement: in PACTOR-1 it is the CHANGE of codeword, and
    the connect answer seeded the reference at CS4."""
    host.on_rx_event(cs_event(cs))


def upgraded_station() -> tuple[PtcHost, Keyed]:
    """...with the link just taken to PACTOR-3 and the first packet of it keyed,
    which is the cycle every archived session got exactly one of."""
    host, keyed = calling_station()
    host.arq.on_host_data(MESSAGE)
    acknowledge(host)
    return host, keyed


def main() -> int:
    print("\nthe offer waits for something to carry:")
    # THE DEFECT, reproduced from the FSM's own seam. The announcement is 7 bytes
    # and the first packet takes all of it, so the acknowledgement that follows
    # finds a drained buffer -- which is every session this station has run.
    host, keyed = calling_station()
    acknowledge(host)
    check("an acknowledgement on a drained buffer does not upgrade",
          host.protocol is Protocol.PACTOR1, str(host.protocol))
    host.tick()
    check("...so the idle packet that fills the slot is still PACTOR-1",
          keyed.p3 == [], f"{len(keyed.p3)} PACTOR-3 packets keyed")

    # ...and with traffic behind it the offer is taken in the same cycle, so the
    # peer's first look at the new waveform is the traffic itself.
    host, keyed = calling_station()
    host.arq.on_host_data(MESSAGE)
    acknowledge(host)
    check("an acknowledgement with the buffer loaded upgrades",
          host.protocol is Protocol.PACTOR3, str(host.protocol))
    check("...and the first PACTOR-3 packet carries the host's bytes",
          keyed.p3 == [MESSAGE[:len(keyed.p3[0])]] if keyed.p3 else False,
          f"{[len(p) for p in keyed.p3]} byte fields")
    check("...a whole speed-level-3 field of them, not an idle frame",
          bool(keyed.p3) and len(keyed.p3[0]) == 59,
          f"{len(keyed.p3[0]) if keyed.p3 else 0}B")
    check("the link is up and this station is still the sender",
          host.arq.state == State.CONNECTED and host.arq._inflight is not None)

    # The seam on its own, because the FSM is what reports the fact and this is
    # the only thing that reads it.
    idle = PtcHost(peer=None, mycall="W9SSJ")
    check("`upgrade` refuses an empty buffer and says so by its answer",
          not idle.upgrade(payload_waiting=False)
          and idle.protocol is Protocol.PACTOR1)
    check("...and takes the same link up with one byte behind it",
          idle.upgrade(payload_waiting=True)
          and idle.protocol is Protocol.PACTOR3)

    print("\nthe offer stands long enough to be read:")
    # THE CODEWORD THAT ENDED EVERY UPGRADE THIS STATION HAS MADE, and it is the
    # one the log shows: a CS1 at zero bit errors, at anchor, one cycle behind
    # the speed-level-3 packet -- so it is the answer to the PACTOR-1 packet
    # ahead of that one, and a verdict on nothing.
    host, keyed = upgraded_station()
    first = bytes(host.arq._inflight.payload)
    # The upgrade and its packet happen INSIDE the acknowledgement's own cycle --
    # that is the slot the peer is timing us against -- so the first tick closes
    # that cycle and the count starts on the next one.
    for _ in range(arq.UPGRADE_SILENCE_CYCLES):
        acknowledge(host)
        host.tick()
    check("a PACTOR-1 codeword inside the window rules nothing out",
          host.protocol is Protocol.PACTOR3 and not host._ruled_out,
          f"{host.protocol}, ruled out {sorted(host._ruled_out)}")
    check("...and the PACTOR-3 packet repeats into it, one look per cycle",
          keyed.p3 == [first] * arq.UPGRADE_SILENCE_CYCLES,
          f"{len(keyed.p3)} keyed, fields {[len(p) for p in keyed.p3]}")
    check("...and nothing retired it: the peer has not acknowledged it",
          host.arq._inflight is not None
          and host.arq._inflight.payload == first)

    # ...and the patience is bounded by the constant that was written for it.
    acknowledge(host)
    host.tick()
    check("the window ends, and PACTOR-3 is ruled out for the rest of the link",
          host.protocol is Protocol.PACTOR1
          and Protocol.PACTOR3 in host._ruled_out,
          f"{host.protocol}, ruled out {sorted(host._ruled_out)}")
    check("...with the field re-chunked for PACTOR-1, not lost with the target",
          keyed.packets[-1] == (Protocol.PACTOR1, MESSAGE[:8]),
          f"{keyed.packets[-1]}")
    check("...so a PACTOR-1-only peer costs four cycles of deafness, once",
          not host.upgrade(payload_waiting=True)
          and len(keyed.p3) == arq.UPGRADE_SILENCE_CYCLES,
          f"{len(keyed.p3)} PACTOR-3 packets keyed in the whole link")

    # THE CONTRADICTION IS DEFERRED, NOT REMOVED. It is what makes an upgrade
    # into the 83 PACTOR-2-only channels recoverable, and it works from the
    # moment the far end has been heard in the protocol we moved to.
    host, keyed = upgraded_station()
    acknowledge(host)
    host.tick()
    host.on_rx_event(cs_event(arq.CS_ACK, protocol="PACTOR-3"))
    check("a PACTOR-3 codeword answers the upgrade",
          not host.arq.upgrade_unanswered and host.protocol is Protocol.PACTOR3)
    acknowledge(host)
    check("...and a PACTOR-1 answer after that contradicts it, as it always did",
          host.protocol is Protocol.PACTOR1
          and Protocol.PACTOR3 in host._ruled_out,
          f"{host.protocol}, ruled out {sorted(host._ruled_out)}")

    # A break-in is the peer taking the channel, which is not a statement about
    # what it can read -- and our own changeover packets are PACTOR-1 too.
    host, keyed = upgraded_station()
    host.on_rx_event(cs_event(pactor1.CS_CHANGEOVER))
    check("a PACTOR-1 break-in inside the window is followed down",
          host.protocol is Protocol.PACTOR1, str(host.protocol))
    check("...and rules nothing out, so the link may climb again",
          not host._ruled_out and host.arq.role == "irs",
          f"ruled out {sorted(host._ruled_out)}, role {host.arq.role}")

    print("\nand the flag that holds it still for a rig test:")
    # ONE VARIABLE. Everything above is the link deciding for itself, on evidence
    # that arrives while the session is running; an operator with a radio and a
    # gateway wants the PACTOR-3 on the air and nothing else moving.
    held, keyed = calling_station()
    held.stay_in_pactor3 = True
    acknowledge(held)
    check("--pactor3-only takes the offer on a drained buffer",
          held.protocol is Protocol.PACTOR3, str(held.protocol))
    for _ in range(arq.UPGRADE_SILENCE_CYCLES + 2):
        acknowledge(held)
        held.tick()
    check("...and neither the peer's PACTOR-1 nor the count that follows it "
          "brings the link back down",
          held.protocol is Protocol.PACTOR3 and not held._ruled_out,
          f"{held.protocol}, ruled out {sorted(held._ruled_out)}")
    held.on_rx_event(cs_event(pactor1.CS_CHANGEOVER))
    check("...while a break-in is still followed, because that is the peer "
          "taking the channel and not a verdict on what it can read",
          held.protocol is Protocol.PACTOR1 and not held._ruled_out,
          f"{held.protocol}, ruled out {sorted(held._ruled_out)}")

    # The two flags are one decision and the session has to act on both. Read off
    # the source for the reason `tests/core/test_occupied.py` gives for its
    # mirror: shrike's parser cannot be run without a radio behind it.
    src = Path(onair.__file__).read_text()
    check("the session parses the flag and hands it to the link layer",
          "--pactor3-only" in src and "stay_in_pactor3 = args.pactor3_only" in src)
    check("...and refuses to be given both at once",
          "add_mutually_exclusive_group" in src)

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
