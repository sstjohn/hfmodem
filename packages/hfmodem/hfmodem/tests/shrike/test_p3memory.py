# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Memory ARQ: soft combining across repeats of one unacknowledged PACTOR-III field.

An unacked field is transmitted again, the same bytes under the same mod-4
counter, until acked; `p3rx.FieldMemory` sums the channel-order softs of
consecutive failed header-anchored frames in front of the trellis. The
capability is worthless unless each of these holds at once, so each is planted
as its own arm, on one set of noisy renders:

  * copies that each FAIL alone must decode summed -- the gain itself, with the
    single-shot failures asserted rather than presumed. The copies alternate
    the carrier swap, as consecutive ARQ cycles do, so the pass also proves the
    per-copy header unwinds the swap before the sum;
  * a swapped-cycle copy READ IN THE HOME arrangement must NOT combine -- the
    swap mapping is load-bearing, and a memory that skips it goes dark here
    rather than soft;
  * copies of two DIFFERENT fields must sum to nothing -- the grouping guard.
    The CRC is what stands between a mixed sum and a delivered wrong field;
  * copies of different GEOMETRY (a peer dropping speed level between repeats)
    must never share a sum -- the key check resets instead of mixing layouts;
  * the production path must carry it: `decode_headed` with a memory turns a
    recording whose every copy fails single-shot into a delivered field, and
    the memoryless pass over the same audio must deliver nothing, or the gain
    is not the memory's.

Noise SNR and seeds are pinned; every decode below is deterministic. The
punctured levels are where combining pays -- measured on rendered repeats,
speed level 6 at -2 dB decodes 0 of 8 recordings alone and 8 of 8 with the
memory -- so the arms run at speed levels 5 and 6.
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import p3frame, p3rx, placement, spec

FS = 48000
SPS = FS // 100
ROW0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
        + p3frame.DATA_OFFSET * SPS)
SNR_DB = -2.0
"""Past the single-shot cliff for the rate-8/9 levels (0 of 8 seeded recordings
decode alone at speed level 6 here) and inside combining's reach."""

PAY6 = bytes((0x41 + i % 26) for i in range(placement.SPEED_PATHS[6].crc_bytes - 3))
PAY6B = bytes((0x61 + i % 26) for i in range(placement.SPEED_PATHS[6].crc_bytes - 3))
PAY5 = bytes((0x41 + i % 26) for i in range(placement.SPEED_PATHS[5].crc_bytes - 3))

_cache: dict = {}


def _copy(sl: int, payload: bytes, seed: int, k: int) -> np.ndarray:
    """Copy `k` of one field in seeded noise, the swap alternating with `k`."""
    key = (sl, payload, seed, k)
    if key not in _cache:
        path = placement.SPEED_PATHS[sl]
        info = payload + bytes([spec.status_byte(1)])
        pkt = placement.data_packet(info, path, swapped=bool(k & 1))
        audio = np.concatenate([np.zeros(FS), pkt, np.zeros(FS)])
        sigma = float(np.sqrt(np.mean(pkt ** 2))) / 10 ** (SNR_DB / 20)
        _cache[key] = audio + np.random.default_rng(seed * 100 + k).normal(
            0, sigma, audio.size)
    return _cache[key]


def _header(sl: int, swapped: bool) -> p3frame.PacketHeader:
    return p3frame.PacketHeader(
        FS, p3frame.variable_header(sl, swapped=swapped), 1.0)


def _cells(audio: np.ndarray, sl: int, swapped: bool):
    """(cells for FieldMemory.add, path) -- the copy read in `swapped`."""
    path = p3rx.path_for(sl, _header(sl, swapped))
    return {off: p3rx._cells(audio, ROW0 + off, path, 0.0, fs=FS)
            for off in (-p3rx.CONFIRM_STEP, 0, p3rx.CONFIRM_STEP)}, path


def _failing(sl: int, payload: bytes, seed: int, k: int):
    audio = _copy(sl, payload, seed, k)
    assert p3rx.decode_at(audio, ROW0, sl, fs=FS,
                          header=_header(sl, bool(k & 1))) is None, \
        f"premise broken: SL{sl} seed {seed} copy {k} decodes single-shot"
    return audio


def test_copies_that_fail_alone_decode_summed():
    mem = p3rx.FieldMemory()
    cells0, path = _cells(_failing(6, PAY6, 0, 0), 6, False)
    assert mem.add(cells0, path) is None            # one copy: nothing to sum
    cells1, path1 = _cells(_failing(6, PAY6, 0, 1), 6, True)
    field = mem.add(cells1, path1)
    assert field is not None and field[:path.crc_bytes - 2][:-1] == PAY6, \
        "two marginal copies, one per arrangement, must sum to the field"
    # a delivered field clears the memory: the next lone copy stands alone
    cells2, _ = _cells(_failing(6, PAY6, 2, 0), 6, False)
    assert mem.add(cells2, path) is None


def test_the_carrier_swap_mapping_is_load_bearing():
    """The same swapped-cycle copy that combines above, read at home instead."""
    mem = p3rx.FieldMemory()
    cells0, path = _cells(_failing(6, PAY6, 0, 0), 6, False)
    mem.add(cells0, path)
    wrong, _ = _cells(_copy(6, PAY6, 0, 1), 6, False)     # rendered swapped
    assert mem.add(wrong, path) is None, \
        "a swapped-cycle copy read in the home arrangement must not combine"


def test_copies_of_different_fields_sum_to_nothing():
    mem = p3rx.FieldMemory()
    cells0, path = _cells(_failing(6, PAY6, 0, 0), 6, False)
    mem.add(cells0, path)
    cellsb, pathb = _cells(_failing(6, PAY6B, 7, 1), 6, True)
    got = mem.add(cellsb, pathb)
    assert got is None, f"a mixed sum must decode to nothing, got {got!r}"


def test_a_speed_level_change_resets_the_memory():
    """Copies with different geometry are not combinable and must not be tried.

    A peer that drops the speed level on a retransmission reframes the field;
    grouping is on geometry, so the SL6 copy leaves rather than joining an SL5
    sum it cannot belong to.
    """
    mem = p3rx.FieldMemory()
    cells6, path6 = _cells(_failing(6, PAY6, 0, 0), 6, False)
    mem.add(cells6, path6)
    cells5, path5 = _cells(_copy(5, PAY5, 3, 0), 5, False)
    assert mem.add(cells5, path5) is None, \
        "a cross-geometry add must reset, never combine"


CYCLE = int(spec.CYCLE_SHORT_S * FS)


def test_the_headed_path_carries_the_memory():
    """`decode_headed` end to end: acquisition, grouping, combining, delivery.

    Two consecutive cycles of one speed level 5 field at -4 dB, the swap
    alternating. Memoryless, the same audio delivers nothing -- asserted, so
    the with-memory delivery below cannot come from a copy that was never
    marginal. With a memory, the second cycle's add returns the field.
    """
    path = placement.SPEED_PATHS[5]
    info = PAY5 + bytes([spec.status_byte(1)])
    cycles = []
    for k in range(2):
        pkt = placement.data_packet(info, path, swapped=bool(k & 1))
        cyc = np.zeros(CYCLE)
        cyc[:min(pkt.size, CYCLE)] += pkt[:CYCLE]
        cycles.append(cyc)
    audio = np.concatenate([np.zeros(FS // 2), *cycles, np.zeros(FS // 2)])
    ref = placement.data_packet(info, path)
    sigma = float(np.sqrt(np.mean(ref ** 2))) / 10 ** (-4.0 / 20)
    audio = audio + np.random.default_rng(0).normal(0, sigma, audio.size)

    def sweep(memory):
        packets = []
        for lo, hi in p3rx.bursts(audio, FS):
            scan, _ = p3rx.decode_headed(audio, lo, hi, fs=FS, memory=memory)
            packets.extend(scan.packets)
        return packets

    assert sweep(None) == [], "premise broken: a copy decodes single-shot"
    got = sweep(p3rx.FieldMemory())
    assert [p.payload for p in got] == [PAY5], \
        f"the memory must deliver the field once, got {got!r}"


def test_a_combined_accept_must_survive_moving_the_clock():
    """The confirmation rule applies to the sum as it does to every accept.

    `FieldMemory.add` decodes the combined softs at the centre clock and at an
    eighth of a symbol either side, and delivers only when a neighbour agrees
    -- the same `confirmed` semantics the single-shot path uses, on the same
    per-copy demodulations. Starve the neighbours and the delivery must
    vanish, whatever the centre says.
    """
    mem = p3rx.FieldMemory()
    cells0, path = _cells(_failing(6, PAY6, 0, 0), 6, False)
    cells1, path1 = _cells(_failing(6, PAY6, 0, 1), 6, True)
    blind0 = {off: (c if off == 0 else np.zeros_like(c))
              for off, c in cells0.items()}
    blind1 = {off: (c if off == 0 else np.zeros_like(c))
              for off, c in cells1.items()}
    mem.add(blind0, path)
    assert mem.add(blind1, path1) is None, \
        "an accept with no agreeing neighbour must not be delivered"
