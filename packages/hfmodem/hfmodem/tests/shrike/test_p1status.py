# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""When status bits 4-5 go on the air, and what counts as a packet.

`--p1-status-bits45` sets bits 4-5 of every data packet's status byte;
`--p1-status-from K` holds them back until the Kth packet. The distinction is
the whole experiment: in 10.36 hours of corpus, 44 real PACTOR-1 data packets,
bits 4-5 appear in exactly two -- both the FIRST packet of a link, both from
stations that went on to PACTOR-3 -- and no station on record has ever set them
mid-link. So nothing tells us whether a peer reads them once at connect or on
every packet, and a link that is already healthy when they appear is the only
arm that separates the two.

K COUNTS INFORMATION, NOT TRANSMISSIONS. A repeat is the same packet: `arq`
builds a status byte once in `_start_next_packet` and re-sends the packet in
flight unchanged until it is acknowledged, so a station that repeats packet #1
five times has put one packet on the air. The air says the same thing -- "bei
jedem Paket, das neue Information enthaelt, wird das Bitmuster invertiert", and
the header is the mod-4 counter's low bit -- which is what `RadioTx._p1_nth`
reads. Counting bursts instead would fire the switch inside a retry train, on a
cycle carrying bytes the peer had already been offered, and the arm would be
measuring the wrong packet.

The status bytes here are read back off the rendered audio, so what is asserted
is what a receiver would hear rather than what the transmitter believes it sent.

Run:  python -m hfmodem.tests.shrike.test_p1status
"""
from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterable
from pathlib import Path

from hfmodem.shrike import onair, p1rx

BITS45 = 0x30

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


class _Tx(onair.RadioTx):
    """The transmitter with the rig and the grid taken out.

    `_tx` is where a rendered burst leaves the renderer, and everything past it
    -- keying, levels, the cycle raster -- is another test's business.
    """

    def __init__(self, bits45: int, k: int):
        super().__init__(None, transmit=False, outdir=Path("."))
        self.p1_status_bits45, self.p1_status_from = bits45, k
        self.keyed: list = []

    def _tx(self, audio, what: str, **kwargs) -> None:
        self.keyed.append(audio)


def _statuses(bits45: int, k: int, counters: Iterable[int],
              breakin_at: frozenset[int] = frozenset()) -> tuple[list[int], str]:
    """The status byte of each ordinary packet keyed, as a receiver reads it."""
    tx = _Tx(bits45, k)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        for i, count in enumerate(counters):
            if i in breakin_at:
                tx.send_p1_breakin(b"abcdefg", 100, count)
            else:
                tx.send_p1_packet(b"the quick", 100, count)
    out = []
    for i, audio in enumerate(tx.keyed):
        if i in breakin_at:
            continue
        packets = p1rx.decode_p1_packets(audio)
        out.append(packets[0].status if packets else -1)
    return out, log.getvalue()


def the_renderer_default_keys_nothing_in_bits_4_5() -> None:
    """A bare `RadioTx` is neutral -- the byte-exact behaviour `test_p1.py`
    gates, and the value both JN36lf mid-link packets carry."""
    st, log = _statuses(onair.RadioTx.p1_status_bits45,
                        onair.RadioTx.p1_status_from, (1, 2, 3, 0))
    check("four packets go out with bits 4-5 clear",
          [s & BITS45 for s in st] == [0, 0, 0, 0], f"{[hex(s) for s in st]}")
    check("...and the log says nothing about them", "bits 4-5" not in log)


def the_shipped_arm_announces_the_upgrade() -> None:
    """`onair.P1_STATUS_ANNOUNCE` is what an arm flies unless told otherwise:
    both bits, from the first packet -- W4DNA's 0x31 and the value every
    granted session on record announced at. 0 is the opt-out and declares a
    PACTOR-1 ceiling."""
    check("the shipped announcement is both bits", onair.P1_STATUS_ANNOUNCE == 3)
    st, log = _statuses(onair.P1_STATUS_ANNOUNCE,
                        onair.RadioTx.p1_status_from, (1, 2))
    check("the first packet and every one after carry them",
          all(s & BITS45 == BITS45 for s in st), f"{[hex(s) for s in st]}")
    check("the log names the byte a receiver reads", "0x31" in log, log.strip())


def bit_5_alone_is_the_announcement_that_matches_our_traffic() -> None:
    """0x21: the coherent reading, and the value the air refuted.

    Bit 5 is "suggests switching to data mode". Bit 4 is the top of the 3-bit
    data type, so 3 declares PMC German on an ASCII payload -- yet 3 is what
    every grant on record was drawn at, and 0x21 flew twice (08-22, 08-28) and
    drew CS1 stalls both times. The renderer still has to key 2 correctly for
    the arm that re-tests it.
    """
    st, log = _statuses(2, 1, (1, 2, 3))
    check("every packet carries bit 5 and not bit 4",
          [s & BITS45 for s in st] == [0x20, 0x20, 0x20], f"{[hex(s) for s in st]}")
    check("the log names the byte a receiver reads", "0x21" in log, log.strip())


def k_of_1_is_every_packet() -> None:
    st, _ = _statuses(3, 1, (1, 2, 3))
    check("the bits ride the first packet and every one after",
          all(s & BITS45 == BITS45 for s in st), f"{[hex(s) for s in st]}")


def k_of_3_switches_on_the_third_packet() -> None:
    st, log = _statuses(3, 3, (1, 2, 3, 0))
    check("the first two packets go out clear",
          [s & BITS45 for s in st[:2]] == [0, 0], f"{[hex(s) for s in st]}")
    check("the third and everything behind it carry the bits",
          all(s & BITS45 == BITS45 for s in st[2:]), f"{[hex(s) for s in st]}")
    check("the log names the packet it switched on and the byte it sent",
          "packet 3 of the session" in log and "0x33" in log, log.strip())
    check("and says it once", log.count("bits 4-5 set") == 1)


def a_repeat_is_the_same_packet() -> None:
    """Three bursts of packet #2 are one packet, and K=3 is not reached inside
    the retry train."""
    st, log = _statuses(3, 3, (1, 2, 2, 2, 3))
    check("every burst of the repeated packet stays clear",
          [s & BITS45 for s in st] == [0, 0, 0, 0, BITS45],
          f"{[hex(s) for s in st]}")
    check("the switch lands on the third packet, the fifth burst",
          "packet 3 of the session" in log, log.strip())


def a_break_in_is_a_packet_the_counter_cannot_show() -> None:
    """The changeover resets the counter to 0, so it can repeat the value we
    last keyed and still be new information."""
    st, _ = _statuses(3, 3, (0, 0, 1), breakin_at=frozenset({1}))
    check("the packet after the break-in is the third",
          [s & BITS45 for s in st] == [0, BITS45], f"{[hex(s) for s in st]}")


STAGES = (the_renderer_default_keys_nothing_in_bits_4_5,
          the_shipped_arm_announces_the_upgrade,
          bit_5_alone_is_the_announcement_that_matches_our_traffic,
          k_of_1_is_every_packet,
          k_of_3_switches_on_the_third_packet,
          a_repeat_is_the_same_packet,
          a_break_in_is_a_packet_the_counter_cannot_show)


def main() -> int:
    global ok
    ok = True
    for stage in STAGES:
        print(stage.__name__.replace("_", " "))
        stage()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
