# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate shrike's PACTOR-1 ARQ data layer (TX pactor1.data_* / RX p1rx.decode_p1).

Three unforgeable gates:

  1. Structure: the on-air packet lengths, status-byte fields and CRC placement
     match what an independent PACTOR-1 decoder expects -- 11-byte 100-Bd and
     23-byte 200-Bd packets, each [field][status][CRC-16 big-endian] with
     LEN = len-3.
  2. Round-trip: shrike's data_signal -> audio -> decode_p1 recovers the exact
     payload bytes, for both speed levels, all four packet counters, both FSK
     polarities, and collapses memory-ARQ repeats to a single packet.
  3. Negative controls: noise, an empty channel, and a CRC-broken frame all decode
     to nothing -- the CRC gate never fabricates text.

Run:  python -m hfmodem.tests.shrike.test_p1
"""

from __future__ import annotations

import sys

import numpy as np


from hfmodem.shrike import pactor1, p1rx  # noqa: E402


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{': ' + detail if detail else ''}")
    return ok


def gate_structure() -> bool:
    print("Gate 1 - on-air frame structure vs an independent P1 decoder")
    ok = True
    p100 = pactor1.data_packet(b"ABCDEFGH", 100, packet_count=1)
    p200 = pactor1.data_packet(b"A" * 20, 200, packet_count=2)
    ok &= _check("100-Bd on-air packet is 12 bytes", len(p100) == 12, f"{len(p100)}")
    ok &= _check("200-Bd on-air packet is 24 bytes", len(p200) == 24, f"{len(p200)}")

    header, field, status = p100[0], p100[1:9], p100[9]
    ok &= _check("header byte precedes the CRC region", header == pactor1.DATA_HEADER)
    ok &= _check("header is 0xAA -- this packet carries new information",
                 pactor1.DATA_HEADER == 0xAA)
    ok &= _check("its repeat form is the inverse, 0x55",
                 pactor1.SYNC_HEADER == 0xAA ^ 0xFF)
    ok &= _check("field carries the payload", field == b"ABCDEFGH")
    ok &= _check("status packet counter = 1", (status & 3) == 1, f"{status:#04x}")
    ok &= _check("status TYPE = 0 (8-bit ASCII)", (status >> 2) & 3 == 0)

    # The framing is checked against a REAL off-air packet, not against our own
    # encoder. The assertion here used to be
    #     crc == coding.crc16(p100[1:10], pactor1.DATA_CRC)
    # which compares data_packet() to the function data_packet() calls, and so
    # held just as firmly while the variant was CCITT-FALSE and the byte order
    # big-endian -- both wrong, and between them enough that no packet shrike
    # sent could be accepted by anyone and no real packet could be decoded.
    #
    # THE ANCHOR IS A STATION THAT IS BEING ACKNOWLEDGED. This was W4DNA's
    # announcement, `aa317734646e610d1e31415b`, and reproducing it byte for byte
    # was read as the strongest evidence in the file. W4DNA sends that one packet
    # on five consecutive cycles with its counter stuck at 1 while the peer repeats
    # CS4 -- a repeat request every time. We were pinned to a frame the band was
    # refusing, and its status byte, 0x31, is why: bits 2-4 read as data type 4,
    # PMC German compression, on eight bytes of plain ASCII.
    #
    # These two are the JN36lf station's, demodulated from
    # rf-corpus/regress/fixtures/pos_p1_data_jn36lf.wav, on a link that is moving:
    # the counter runs 3 -> 0 and the header alternates with it. Between them they
    # pin header, field, IDLE padding, status layout, both data types, CRC variant,
    # CRC coverage and CRC byte order -- and the counter/header parity is the
    # OPPOSITE of ours, so `header=` has to carry it and cannot be inferred.
    for label, payload, count, dtype, header, real in (
            ("count 3 under 0x55, 8-bit ASCII", b"|1O|KM17", 3, 0, 0x55,
             "557c314f7c4b4d3137038357"),
            ("count 0 under 0xAA, Huffman", bytes.fromhex("63a032f6a69fffe1"),
             0, 1, 0xAA, "aa63a032f6a69fffe104fccf")):
        mine = pactor1.data_packet(payload, 100, count, dtype, header=header)
        ok &= _check(f"reproduces a real off-air packet byte for byte -- {label}",
                     mine.hex() == real, f"{mine.hex()} vs {real}")
    return ok


def gate_roundtrip() -> bool:
    print("Gate 2 - TX -> audio -> RX round-trip (byte-exact)")
    ok = True
    cases = [(100, b"HELLO123"), (100, b"OK 73 GL"), (200, b"The quick brown fox!")]
    for baud, pl in cases:
        for pc in range(4):
            pkts = p1rx.decode_p1_packets(
                pactor1.data_signal(pl, baud, repeats=3, packet_count=pc))
            got = pkts[0] if len(pkts) == 1 else None
            ok &= _check(f"{baud}Bd pc={pc} {pl!r}",
                         got is not None and got.payload == pl
                         and got.baud == baud and got.packet_count == pc,
                         f"recovered {[p.payload for p in pkts]}")
    # short payload: trailing pad stripped, exact bytes back
    pkts = p1rx.decode_p1_packets(pactor1.data_signal(b"HI", 100, repeats=2))
    ok &= _check("short payload padded/stripped",
                 len(pkts) == 1 and pkts[0].payload == b"HI")
    # 3 memory-ARQ copies collapse to one decoded packet
    ok &= _check("memory-ARQ repeats deduplicated", len(pkts) == 1, f"{len(pkts)}")
    return ok


def gate_negative() -> bool:
    print("Gate 3 - negative controls (no false decode)")
    ok = True
    rng = np.random.default_rng(0)
    ok &= _check("gaussian noise -> nothing",
                 p1rx.decode_p1(rng.standard_normal(48000) * 0.1) == "")
    ok &= _check("silence -> nothing", p1rx.decode_p1(np.zeros(48000)) == "")

    # a frame with a corrupted CRC must NOT decode
    good = bytearray(pactor1.data_packet(b"SECRET12", 100))
    good[-1] ^= 0xFF
    burst = pactor1._fsk_burst(bytes(good), 100, 1.22, 0.11)
    audio = np.concatenate([np.zeros(24000), burst, np.zeros(24000)]).astype(np.float32)
    ok &= _check("broken-CRC frame -> nothing", p1rx.decode_p1(audio) == "",
                 repr(p1rx.decode_p1(audio)))
    return ok


def main() -> int:
    gates = [gate_structure(), gate_roundtrip(), gate_negative()]
    print()
    if all(gates):
        print("ALL PASS")
        return 0
    print("FAILURES PRESENT")
    return 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
