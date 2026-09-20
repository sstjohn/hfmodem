# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS's codeword across a changeover, against the peer's packet counter.

PACTOR-1's alternation is the peer's counter and nothing else. "Der Paketzaehler
wird auf 0 gesetzt" at a changeover, and the packet carrying that zero is
acknowledged CS1: "bei fehlerfrei empfangenem Paket antwortet er mit CS1 ... eine
Wiederholung des BK-Paketes wird mittels CS2 angefordert" (Level-1 description,
Senderichtungswechsel). hf-pactor's `tx_rx_100` is the same shape in code -- CS2
in every cycle it waits for the changeover packet, CS1 on the cycle it decodes --
and its `rx_tx_100`, the station on the other side of that exchange, takes an
exact CS1 and nothing else as the acknowledgement of its own counter-0 packet.

Keyed off a free-running count of acceptances instead, the codeword that answers
the counter-0 packet is a coin flip, and both faces are on file against the same
seven bytes. KB5LZK's `RMS Tri` break-in drew CS1 on 2026-08-22
(`working/night-17-kb5lzk-pactor-40m-2026-08-22.log:284-330`) and the gateway
walked its whole banner through. The byte-identical packet drew CS2 from VE1YZ on
2026-09-02 (`working/onair-0902-2342/pactor-night-09-ve1yz-mail.log:323-360`),
which is the request to send it again, and the gateway sent those seven bytes 45
more times while we held CS2 for 96 cycles. That is the mail session that stalled.

Run: python -m hfmodem.tests.shrike.test_p1counter
"""
from __future__ import annotations

import sys

import pytest

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


def _packet(counter: int, field: bytes, breakin: bool = False, **status):
    return rxfront.Event(0.0, "packet", "", protocol="PACTOR-1", breakin=breakin,
                         packet=(0, spec.status_byte(counter, **status), field,
                                 True))


def _called_by_a_gateway(answer: int = pactor1.CS_ACK_A) -> tuple[PtcHost, _Keyed]:
    """A call answered, and the gateway taking the channel to greet us.

    The path a Winlink RMS arrives on, and the one both logs were flying: we
    call, the answer names the rate, and the gateway's CS3 head is the next
    thing on the air.

    THE ANSWER NAMES THE RATE and the two sessions replayed here were answered
    differently -- VE1YZ CS1, so 200 Bd, and KB5LZK CS4, so 100. The alternation
    below is the peer's counter and is the same at either rate, but the rate is
    not decoration: an IRS at 200 Bd that reads nothing for a run of cycles asks
    the peer to halve it (`ptc._p1_cs_for`), and a 100 Bd session replayed at 200
    would draw a codeword its own log has no place for.
    """
    peer = _Keyed()
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(rxfront.Event(0.0, "cs", NAMES[answer], protocol="PACTOR-1",
                                   cs=answer))
    host.arq.on_rx_cs(CS_BREAKIN)
    return host, peer


@pytest.mark.parametrize("previous", (pactor1.CS_ACK_A, pactor1.CS_ACK_B))
@pytest.mark.parametrize("request_first", (False, True))
def test_our_breakin_is_acked_by_cs1_regardless_of_the_previous_turn(previous,
                                                                  request_first):
    host, _ = _called_by_a_gateway()
    host.stay_in_pactor1 = True
    host._last_rx_cs = previous
    host.arq.on_host_data(b"login bytes")
    host.arq.on_host_breakin()
    host.arq._breakin_armed = True
    host.arq.on_cycle()
    packet = host.arq._inflight
    assert packet is not None and packet.breakin and host.arq.tx_seq == 0
    # CS2 requests the BK again even if it differs from the old turn's CS.
    if request_first:
        host.on_rx_event(rxfront.Event(1.0, "cs", "CS2", protocol="PACTOR-1",
                                       cs=pactor1.CS_ACK_B))
        assert host.arq._inflight is packet
    host.on_rx_event(rxfront.Event(2.0, "cs", "CS1", protocol="PACTOR-1",
                                   cs=pactor1.CS_ACK_A))
    assert host.arq._inflight is None
    # The normal alternation resumes, seeded by the accepted BK's CS1.
    host.arq.on_cycle()
    next_packet = host.arq._inflight
    assert next_packet is not None and not next_packet.breakin
    host.on_rx_event(rxfront.Event(3.0, "cs", "CS1", protocol="PACTOR-1",
                                   cs=pactor1.CS_ACK_A))
    assert host.arq._inflight is next_packet
    host.on_rx_event(rxfront.Event(4.0, "cs", "CS2", protocol="PACTOR-1",
                                   cs=pactor1.CS_ACK_B))
    assert host.arq._inflight is None


def the_night_the_mail_stalled() -> None:
    """VE1YZ, 2026-09-02, from the CS3 head to the counter-1 packet."""
    host, peer = _called_by_a_gateway()
    check("the gateway's CS3 makes this station the receiving one",
          host.arq.role == IRS and host.arq.state == State.CONNECTED,
          f"{host.arq.role} / {host.arq.state}")

    host.arq.on_cycle()
    check("the cycle between the CS3 head and the changeover packet asks for it",
          peer.cs == ["CS2"], " ".join(peer.cs) or "nothing keyed")

    host.on_rx_event(_packet(0, b"RMS Tri", breakin=True))
    host.arq.on_cycle()
    check("...and the counter-0 packet that decodes is acknowledged CS1",
          peer.cs[1:] == ["CS1"], " ".join(peer.cs[1:]) or "nothing keyed")
    check("...delivering the seven bytes it carried",
          host.rcvd_total == 7, f"{host.rcvd_total} bytes")

    for _ in range(45):
        host.on_rx_event(_packet(0, b"RMS Tri", breakin=True))
        host.arq.on_cycle()
    check("...and 45 repeats of it hold that CS1 rather than asking again",
          peer.cs[1:] == ["CS1"] * 46, " ".join(sorted(set(peer.cs[1:]))))
    check("...with the seven bytes delivered once",
          host.rcvd_total == 7, f"{host.rcvd_total} bytes")

    mark = len(peer.cs)
    host.on_rx_event(_packet(1, b"mode 1.4"))
    host.arq.on_cycle()
    check("...and the counter-1 packet behind them draws CS2",
          peer.cs[mark:] == ["CS2"], " ".join(peer.cs[mark:]) or "nothing keyed")


def _kb5lzk(extra: int) -> tuple[PtcHost, _Keyed, int]:
    """The 2026-08-22 session up to the second changeover, `extra` packets long.

    The gateway takes the channel with an empty break-in packet, holds it for a
    few cycles, and we take it back behind the next one that decodes. `extra` is
    how many packets it carried in between -- the quantity the codeword after the
    NEXT changeover must not be a function of.

    At 100 Bd, which is the session: "peer answered (CS4) -> link at 100 Bd" at
    line 127, and every packet in its 42 rate lines is a 100 Bd one.
    """
    host, peer = _called_by_a_gateway(pactor1.CS_SPEED)
    host.on_rx_event(_packet(0, b"", breakin=True))
    host.arq.on_cycle()
    host.on_rx_event(_packet(1, b""))
    for _ in range(3):
        host.arq.on_cycle()
    for i in range(extra):
        host.on_rx_event(_packet((2 + i) % 4, b""))
        host.arq.on_cycle()
    host.arq.on_host_data(b"login please")
    host.arq.on_host_breakin()
    host.on_rx_event(_packet((2 + extra) % 4, b""))
    host.arq.on_cycle()
    host.arq.on_cycle()
    host.arq.on_rx_cs(CS_BREAKIN)
    return host, peer, len(peer.cs)


def the_banner_that_crossed() -> None:
    """KB5LZK, 2026-08-22, end to end: two of the gateway's stints and ours.

    `working/night-17-kb5lzk-pactor-40m-2026-08-22.log:211-508`. Counter 0 held
    for four cycles, then 1, 2, 3 held for three, 0 held for five, 1 held for
    two, and the twenty codewords we keyed against them.

    Flown TWICE, over a first stint one packet longer the second time. It is the
    same gateway, the same banner and the same codewords, and the length of what
    came before a changeover is not one of the things the alternation after it
    can depend on -- "der Paketzaehler wird auf 0 gesetzt". A count of
    acceptances depends on exactly that, which is how the same seven bytes drew
    CS1 here and CS2 from VE1YZ eleven days later.
    """
    want = " ".join(["CS1"] * 5 + ["CS2"] + ["CS1"] + ["CS2"] * 4
                    + ["CS1"] * 6 + ["CS2"] * 3)
    banner = [(0, b"RMS Tri", 4), (1, b"mode 1.4", 0), (2, b".2.0 KB5", 0),
              (3, b"LZK at t", 3), (0, b"he Arkan", 5), (1, b"sas Divi", 2)]
    for extra in (0, 1):
        host, peer, mark = _kb5lzk(extra)
        check(f"the gateway's first stint runs CS1 and a held CS2 ({extra} carried)",
              peer.cs[:4] == ["CS1", "CS2", "CS2", "CS2"], " ".join(peer.cs[:4]))
        for counter, field, held in banner:
            host.on_rx_event(_packet(counter, field,
                                     breakin=counter == 0 and not host.rcvd_total))
            host.arq.on_cycle()
            for _ in range(held):
                host.arq.on_cycle()
        keyed = " ".join(peer.cs[mark:])
        check(f"...and the banner behind it draws the log's twenty codewords "
              f"({extra} carried)", keyed == want, f"{keyed} (want {want})")
        check("...with every field of it reaching the host",
              host.rcvd_total == 47, f"{host.rcvd_total} bytes")


def both_directions_restart_the_parity() -> None:
    """Two changeovers in one link, and the counter is what each of them resets.

    The peer greets us, we take the channel to answer, and it takes it back --
    which is a mail session's whole shape. Both of its stints open at counter 0,
    so both are answered CS1 -- behind the CS2 the cycle before the changeover
    packet decodes carries -- whatever the codeword the stint before ended on
    was. "Der Paketzaehler wird auf 0 gesetzt", and the alternation is that
    counter.
    """
    host, peer = _called_by_a_gateway()
    for counter, field in ((0, b"RMS Tri"), (1, b"mode 1.4"), (2, b".2.0 KB5")):
        host.on_rx_event(_packet(counter, field, breakin=counter == 0))
        host.arq.on_cycle()
    check("the gateway's first stint runs CS1 CS2 CS1", peer.cs == ["CS1", "CS2", "CS1"],
          " ".join(peer.cs))

    host.arq.on_host_data(b"login please")
    host.on_rx_event(_packet(3, b"LZK at t", changeover_request=True))
    host.arq.on_cycle()
    host.on_rx_event(_packet(0, b"he Arkan"))
    host.arq.on_cycle()
    host.arq.on_cycle()
    check("...the invitation in its status byte hands us the channel",
          host.arq.role == ISS, str(host.arq.role))

    check("...and its last packet, which our changeover packet acknowledged, "
          "reaches the host too", host.rcvd_total == 39, f"{host.rcvd_total} bytes")

    mark = len(peer.cs)
    host.arq.on_rx_cs(CS_BREAKIN)
    check("...which the gateway then takes back", host.arq.role == IRS,
          str(host.arq.role))
    host.arq.on_cycle()
    for counter, field in ((0, b"sas Divi"), (1, b"sion of ")):
        host.on_rx_event(_packet(counter, field, breakin=counter == 0))
        host.arq.on_cycle()
    second = peer.cs[mark:]
    check("...and the second stint opens on the parity the first one did",
          second == ["CS2", "CS1", "CS2"], " ".join(second))
    check("...and carries the second stint's fields to the host",
          host.rcvd_total == 55, f"{host.rcvd_total} bytes")


STAGES = (the_night_the_mail_stalled,
          the_banner_that_crossed,
          both_directions_restart_the_parity)


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
