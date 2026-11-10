# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A frame keyed into the rise: the head arrives under the noise, the CRC does not.

A peer keys up into its own first byte, and for the twenty to thirty milliseconds
its carrier takes to come up the receiver has nothing. At 100 Bd that is two or
three of the head's eight symbols; at 200 Bd it is four to six, so the header byte
is gone and the burst detector's idea of where the frame starts is that far late as
well. Nothing else about the frame is touched: the CRC covers [field][status] and
NOT the head, so the copy verifies completely.

That cost a Winlink gateway. On 2026-08-18 WS8EOC held a link at 200 Bd and the
nineteen full cycles of `captures/onair-0818-2235` -- every one of them a frame
this decoder can read -- yielded three, nine straight misses ended the session, and
no mail moved. So each arm below is planted on one of the two things that
threw them away:

  * the search reached one symbol either side of the burst onset, and the origin
    can be six symbols ahead of it (RISE_SYMBOLS);
  * the head had to read exactly, including the symbols that arrived before the
    carrier did (PREFIX_FLOOR).

And two negatives, because the head is the whole of what stands between a 16-bit
checksum and a coincidence: a head symbol that IS up must still be right, and a
buffer with no frame in it must still yield nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, rxfront

FS = p1rx.FS
HOLD = Path(__file__).resolve().parents[5] / "captures" / "onair-0818-2235"

PAYLOAD = b"RISETEST"
SIGMA = 0.02
"""Noise against `packet_signal`'s 0.11 amplitude: an easy channel, so what these
arms measure is the gate and not the SNR."""


def _keyed_into_the_rise(baud: int, symbols: int, seed: int = 1,
                         header: int | None = None) -> np.ndarray:
    """One packet whose first `symbols` symbols arrive under the noise floor."""
    lead = int(0.05 * FS)
    burst = pactor1.packet_signal(PAYLOAD, baud, header=header,
                                  lead_s=0.05, tail_s=0.05)
    x = np.asarray(burst, dtype=float).copy()
    x[:lead + symbols * (FS // baud)] = 0.0
    return x + np.random.default_rng(seed).normal(0, SIGMA, x.size)


@pytest.mark.parametrize("baud,symbols", [(200, 6), (200, 4), (100, 3)])
def test_a_frame_keyed_into_the_rise_still_reads(baud, symbols):
    pkts = p1rx.decode_p1_packets(_keyed_into_the_rise(baud, symbols))
    assert [p.payload for p in pkts] == [PAYLOAD], \
        f"{baud} Bd, {symbols} symbols into the rise: {pkts}"


@pytest.mark.parametrize("symbols", [0, 4])
def test_a_head_that_is_up_must_still_be_right(symbols):
    """A CRC-valid frame under a head that is neither 0x55 nor 0xAA is refused.

    The CRC never covered the head, so this frame checksums perfectly; the four
    symbols still standing at `symbols == 4` are what refuses it.
    """
    x = _keyed_into_the_rise(200, symbols, header=0x3C)
    assert not p1rx.decode_p1_packets(x), \
        f"a head read from {8 - symbols} live symbols must be charged for"


def test_the_gate_still_refuses_noise():
    rng = np.random.default_rng(7)
    for k in range(60):
        seg = rng.standard_normal(int(1.25 * FS)) * 0.1
        assert not p1rx.decode_p1_packets(seg), f"noise window {k} claimed a frame"
        assert not p1rx.decode_p1_packets(seg, breakin=True), \
            f"noise window {k} claimed a changeover"


def test_the_ws8eoc_hold_cycles():
    """The nineteen cycles the defect was measured on, read end to end.

    Every one of them carries a 200 Bd frame -- the changeover packet the gateway
    was offering, or the data packet with the break-in bit set that alternates with
    it -- and thirteen carry bit errors in the CRC-protected region that no gate
    can undo. Six is what an exhaustive alignment search with the CRC and the eye
    reaches on this audio, so it is the ceiling and not a ratchet; three is what
    the session got.
    """
    holds = [p for p in sorted(HOLD.glob("hold_*.wav"))
             if p1rx.load_wav(str(p)).size / FS > 0.8]
    if len(holds) != 19:
        pytest.skip(f"the 19 hold cycles are not at {HOLD} -- they live in the "
                    "capture record, which does not cross the publication boundary")
    memory = p1rx.PacketMemory()
    read = [p.name for p in holds
            if rxfront.decode_expected_p1_packet(p1rx.load_wav(str(p)),
                                                 memory) is not None]
    assert len(read) >= 6, f"only {len(read)} of 19 cycles read: {read}"
