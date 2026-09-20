# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Memory ARQ: soft combining across repeats of one unacknowledged PACTOR-1 packet.

A PACTOR-1 packet has no FEC: one bit error destroys it, and repetition is the
mode's only error-correction mechanism. `p1rx.PacketMemory` sums the per-bit
mark/space softs of consecutive failed copies, and the capability is worthless
unless each of these holds at once, so each is planted as its own arm:

  * copies that each FAIL alone must decode summed -- the gain itself, with the
    single-shot failures asserted rather than presumed;
  * the FSK shift inverts EVERY cycle, so a raw sum of consecutive copies adds
    mark to space and cancels. The un-inversion is established from the data,
    and the arm that sums WITHOUT it must go dark rather than soft;
  * copies of two DIFFERENT fields must sum to nothing -- the grouping guard.
    The header and CRC gates are what stand between a mixed sum and a wrong
    field delivered, and this arm fails if anyone relaxes them;
  * copies of different GEOMETRY (baud) must never share a sum -- a peer can
    drop 200 -> 100 Bd on a retransmission and the layouts are not combinable;
  * the real off-air repeat pair must combine: the jn36lf capture carries one
    packet transmitted twice with the shift inverted between the copies, and
    the second copy fails single-shot off the air as recorded.

Noise sigma and seeds are pinned; every decode below is deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, rxfront
from hfmodem.tests.kestrel import corpora

A = b"MEMARQ12"
B = b"OTHERFLD"
SIGMA = 0.55
"""Noise sigma against `packet_signal`'s 0.11-amplitude render: past the
single-shot cliff (0 of 12 seeded copies decode alone at this sigma) and inside
combining's reach."""

SEED = 5

_cache: dict = {}


def _window(payload: bytes, seed: int, k: int) -> np.ndarray:
    """One ARQ cycle's receive window: copy `k` of `payload`, in seeded noise.

    `invert` alternates with the cycle, which is the shift inversion the air
    carries -- consecutive copies arrive with MARK and SPACE exchanged."""
    key = (payload, seed, k)
    if key not in _cache:
        burst = pactor1.packet_signal(payload, 100, packet_count=1,
                                      invert=bool(k & 1), lead_s=0.1, tail_s=0.15)
        rng = np.random.default_rng(seed * 1000 + k)
        _cache[key] = burst + rng.normal(0, SIGMA, burst.size)
    return _cache[key]


def _failing_softs(payload: bytes, seed: int, k: int) -> np.ndarray:
    x = _window(payload, seed, k)
    assert not p1rx.decode_p1_packets(x), \
        f"premise broken: seed {seed} copy {k} decodes single-shot"
    d = p1rx.packet_softs(x, baud=100)
    assert d is not None
    return d


def test_copies_that_fail_alone_decode_summed():
    mem = p1rx.PacketMemory()
    assert mem.add(_failing_softs(A, SEED, 0), 100) is None   # one copy: nothing to sum
    got = mem.add(_failing_softs(A, SEED, 1), 100)
    assert got is not None and got.payload == A, \
        "two marginal copies, one per shift sense, must sum to the packet"
    # the delivered sense is the LAST copy's on-air shift -- what a grid taking
    # its phase from the packet needs -- and copy 1 went out inverted
    assert got.inverted is True
    # a delivered packet clears the memory: the next lone copy stands alone
    assert mem.add(_failing_softs(A, SEED, 0), 100) is None


def test_summing_without_uninverting_the_shift_is_refused():
    """The trap itself: consecutive copies arrive in opposite shift senses.

    The same two copies that combine through `PacketMemory` -- which reads each
    copy's sense off its correlation against the accumulator -- must decode to
    NOTHING when summed raw, because a raw sum adds mark to space. If this arm
    ever decodes, the un-inversion has stopped being load-bearing and something
    else is delivering the field.
    """
    d0 = _failing_softs(A, SEED, 0)
    d1 = _failing_softs(A, SEED, 1)
    assert p1rx.PacketMemory._decode(d0 + d1, 100, False) is None, \
        "a sum that never un-inverted the shift must not decode"
    mem = p1rx.PacketMemory()
    mem.add(d0, 100)
    got = mem.add(d1, 100)
    assert got is not None and got.payload == A


def test_copies_of_different_fields_sum_to_nothing():
    mem = p1rx.PacketMemory()
    mem.add(_failing_softs(A, SEED, 0), 100)
    got = mem.add(_failing_softs(B, SEED + 50, 1), 100)
    assert got is None, f"a mixed sum must decode to nothing, got {got!r}"


def test_bauds_never_share_a_sum():
    """Geometry is the baud, and each baud accumulates in its own lane.

    A 200 Bd soft read of the same window must neither reset the 100 Bd lane
    nor join its sum: after interleaving one, the two 100 Bd copies still
    combine and the 200 Bd lane still holds too few to try.
    """
    mem = p1rx.PacketMemory()
    assert mem.add(_failing_softs(A, SEED, 0), 100) is None
    d200 = p1rx.packet_softs(_window(A, SEED, 0), baud=200)
    if d200 is not None:
        assert mem.add(d200, 200) is None
    got = mem.add(_failing_softs(A, SEED, 1), 100)
    assert got is not None and got.payload == A


CORPUS = corpora.REGRESS_FIXTURES
JN36LF = CORPUS / "pos_p1_data_jn36lf.wav"
"""25 s off 14110 kHz holding the corpus's one real memory-ARQ repeat: the
Huffman packet (hdr 0xAA, count 0, field 63a032f6a69fffe1) at t=15.25 and again
at t=16.23 with the shift inverted. The repeat carries exactly two hard bit
errors off the air, both on low-confidence softs, and does not decode alone."""


def test_the_real_offair_repeat_combines():
    """Both copies real audio; only the first copy's noise is synthetic.

    The recorded repeat at t=16.23 fails single-shot AS CAPTURED -- that is the
    premise, asserted -- and the first copy is degraded with seeded noise until
    it fails too. The memory, fed both through the same entry point a session
    uses, returns the field byte-exact with the inter-copy shift inversion
    un-done from the data.
    """
    if not JN36LF.exists():
        pytest.skip(f"off-air capture absent: {JN36LF}")
    audio = p1rx.load_wav(str(JN36LF))
    fs = p1rx.FS
    want = bytes.fromhex("63a032f6a69fffe1")
    win_a = audio[int(15.0 * fs):int(16.25 * fs)]
    win_b = audio[int(16.0 * fs):int(17.25 * fs)]
    assert not p1rx.decode_p1_packets(win_b), \
        "premise broken: the off-air repeat now decodes single-shot"
    rng = np.random.default_rng(3)
    noisy = win_a + rng.normal(0, 2.6 * float(np.sqrt(np.mean(win_a ** 2))),
                               win_a.size)
    assert not p1rx.decode_p1_packets(noisy), \
        "premise broken: the degraded first copy decodes single-shot"
    mem = p1rx.PacketMemory()
    assert rxfront.decode_expected_p1_packet(noisy, memory=mem) is None
    ev = rxfront.decode_expected_p1_packet(win_b, memory=mem)
    assert ev is not None and ev.packet[2] == want, \
        "the real repeat pair must combine byte-exact"
