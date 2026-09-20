# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A whole PACTOR-2 link on one raster, graded in both directions.

Every other PACTOR-2 test grades one thing at a time: a field through the coding
chain, a burst through the receiver, a codeword through the correlator. What none
of them touches is the GEOMETRY OF A CYCLE -- that the answer sits where
`pactor2.cs_slot` says it does, that it does not land on the packet it answers,
that the carrier swap carries both directions with it, and that a speed change
mid-link leaves the raster where it was. That geometry is what a session layer
will be built on, and it is the part a link stalls on when it is wrong.

So this renders the link a station would key: a PACTOR-1 connect, the capability
announcement, then twelve PACTOR-2 cycles walking speed levels 1, 2 and 3 with a
synthetic peer acknowledging CS1/CS2 alternately in the answer slot and asking
for the next level with CS4 at each rung. Our packets are graded through
`p2rx.decode_expected_burst` -- one cycle's window, no grid fitted over the
recording -- and the peer's codewords through `p2rx.control_signal_at` at the
instant the grid predicts, which is exactly what the two live readers will do.

WHAT THIS CANNOT SAY. The codeword keying is a hypothesis
(`pactor2.control_signal`), so a green run here is a round trip and not interop.
The same link was put to an independent monitor with the codewords at two
candidate instants and with none at all, and it read the three identically: a
receive-only monitor never answers, so nothing outside this package can grade a
reverse channel. What the monitor did settle is the other half -- our packets
still read at every level with a peer transmitting in the answer slot.
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import p2rx, pactor1, pactor2, spec

FS = spec.SAMPLE_RATE
CYCLE = spec.CYCLE_SHORT_S
AMP = 0.11
PHASE0 = 1.22
P1_ANSWER_S = 0.96 + 0.085          # packet plus the turnaround, test_p1_oracle
LEVELS = (0, 1, 2)
PER_LEVEL = 4
TEXT = {0: b"P2S", 1: b"P2 SL2 CYCLE", 2: b"P2 SL3 SHRIKE TWO SIDED CYCLE"}
CS_ACK = (0, 1)
CS_SPEED_UP = 3


def _cycles():
    """`(cycle index, path, field, codeword index)` for the whole PACTOR-2 run.

    The counter runs on from the PACTOR-1 announcement, the acknowledgement
    alternates on its parity -- pactor3.md s7's rule, an even counter answered
    CS1 -- and the last cycle of every rung but the top asks for the next one.
    """
    count = 2
    for cycle, (level, i) in enumerate(
            (l, i) for l in LEVELS for i in range(PER_LEVEL)):
        path = pactor2.PATHS[level]
        info = (b"%s %d" % (TEXT[level], i)).ljust(
            path.crc_bytes - 3, bytes([spec.IDLE]))[:path.crc_bytes - 3]
        field = pactor2.build_field(
            info + bytes([spec.status_byte(count % 4, data_type=0)]), path)
        rung_end = i == PER_LEVEL - 1 and level != LEVELS[-1]
        yield cycle, path, field, CS_SPEED_UP if rung_end else CS_ACK[count % 2]
        count += 1


def preamble() -> list[tuple[float, np.ndarray]]:
    """The two PACTOR-1 cycles a PACTOR-2 link opens on: connect, then announce.

    Every PACTOR-2 link starts in FSK, so this is the audio in front of the
    first PACTOR-2 burst on the air. `tests/shrike/test_p2_oracle.py` opens every
    arm with it so that what an outside decoder is offered is a link rather than
    loose bursts -- and because a monitor reading the announcement out of it is
    what says the monitor was listening at all. Hence a function rather than four
    lines inside `render`.
    """
    return [(0.0, pactor1._dualrate_frame("W1AW", AMP, PHASE0)),
            (P1_ANSWER_S, pactor1.control_signal(pactor1.CS_ACK_A, amp=AMP)),
            (CYCLE, pactor1._fsk_burst(
                pactor1.data_packet(b"1W9SSJ\r", 100, 1, bits45=3),
                100, PHASE0, AMP)),
            (CYCLE + P1_ANSWER_S,
             pactor1.control_signal(pactor1.CS_ACK_A, amp=AMP))]


def render() -> np.ndarray:
    """The link as audio, both stations on one 1.25 s raster."""
    items = preamble()
    lead = pactor2.pulse_lead(FS) / FS
    for cycle, path, field, cs in _cycles():
        t = (2 + cycle) * CYCLE
        swapped = bool(cycle & 1)
        items.append((t - lead, AMP * pactor2.data_burst(
            field, path, swapped=swapped, fs=FS)))
        items.append((t + pactor2.cs_slot(path) - lead,
                      AMP * pactor2.control_signal(cs, swapped=swapped, fs=FS)))

    span = max(int(t * FS) + x.size for t, x in items)
    out = np.zeros(span + FS)
    for t, x in items:
        at = int(round(t * FS))
        out[at:at + x.size] += x
    return out


def test_both_directions_read_off_one_raster():
    audio = render()
    fields = codewords = 0
    for cycle, path, field, cs in _cycles():
        t = (2 + cycle) * CYCLE
        swapped = bool(cycle & 1)
        window = audio[int((t - 0.2) * FS):int((t + 1.15) * FS)]
        got = p2rx.decode_expected_burst(window, FS)
        fields += got is not None and got[2] == field and got[1] is path
        answer = p2rx.control_signal_at(
            audio[int(t * FS):int((t + 1.3) * FS)],
            int(round(pactor2.cs_slot(path) * FS)), FS, swapped=swapped)
        codewords += answer == (cs, 0)
    assert fields == 12, f"{fields} of 12 fields read byte-exact"
    assert codewords == 12, f"{codewords} of 12 codewords read at zero errors"


def test_the_answer_does_not_land_on_the_packet_it_answers():
    """The slot has to clear the packet, ramps included, or both ends key over
    each other -- and a shaped burst is longer than its own pulse count."""
    for paths in (pactor2.PATHS, pactor2.PATHS_LONG):
        for path in paths:
            burst = pactor2.data_burst(
                bytes(path.crc_bytes), path, fs=FS).size / FS
            gap = pactor2.cs_slot(path) - (burst - pactor2.pulse_lead(FS) / FS)
            assert 0.0 < gap < spec.CS_WINDOW_S, \
                f"{path.name}: {gap * 1000:.0f} ms between packet and answer"
