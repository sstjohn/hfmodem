# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 188-110A mod: its receiver against bursts built as the recordings were
measured to be (``mod110.synthesize``), and against what it must not take for
one. The recordings themselves are not kept here."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import mil110, mod110 as M, phy, sim

BITS = np.random.default_rng(110).integers(0, 2, 6432).astype(np.uint8)


def _trailing_ones(b) -> int:
    """The data's own last ones, which the end marker (a run of ones) takes
    with it: nothing tells them apart."""
    b = np.asarray(b)
    return int(b.size - (np.flatnonzero(b == 0)[-1] + 1)) if (b == 0).any() else b.size


def _same(got, want) -> bool:
    n = want.size - _trailing_ones(want)
    return got.size >= n and bool((got[:n] == want[:n]).all())


def _air(audio, **ch):
    a = np.concatenate([np.zeros(phy.FS), audio, np.zeros(2 * phy.FS)])
    return sim.channel(a, **ch)


def test_the_sync_phase_is_the_randomiser():
    """Sixteen frames of zero data: every symbol the randomiser's, which
    repeats every 160."""
    assert M.SYNC.size == 640
    assert (M.SYNC[:160] == mil110._SCR).all() and (M.SYNC[160:320] == M.SYNC[:160]).all()


@pytest.mark.parametrize("repeat, end, seven", [(1, False, False), (2, True, True)])
def test_both_formats_come_back_bit_for_bit(repeat, end, seven):
    a = M.synthesize(BITS, repeat=repeat, end_marker=end, seven=seven, seed=1)
    r = M.receive(_air(a, snr_db=15, freq_hz=7.0, seed=2))
    assert r.repeat == repeat and r.end_marker == end
    assert _same(r.payload, BITS)
    if end:                                     # the marker is not data
        assert r.payload.size == BITS.size - _trailing_ones(BITS)


def test_the_poor_channel():
    """At 10 dB on 2 ms / 1 Hz, near what the recordings ran at: raw bits,
    so the channel's errors pass through, but few, and none lost."""
    r = M.receive(_air(M.synthesize(BITS, seed=3), snr_db=10, preset="poor",
                       freq_hz=-11.0, seed=4))
    n = min(r.payload.size, BITS.size)
    assert n >= BITS.size - M.UNKNOWN
    assert np.mean(r.payload[:n] != BITS[:n]) < 0.02


def test_the_burst_ends_where_the_signal_does():
    """No end marker: the probes going to noise ends it, within a few frames."""
    a = _air(M.synthesize(BITS, seed=5), snr_db=12, seed=6)
    r = M.receive(a)
    frames = -(-BITS.size // M.UNKNOWN)
    assert abs(r.frames - frames) <= 4


def test_an_ordinary_188_110_transmission_is_not_taken_for_one():
    """A 188-110 data phase sending zeros is the sync phase exactly; without
    the tone header before it, it is not this waveform."""
    tx = mil110.transmit(bytes(400), "600S")
    assert M.receive(_air(tx, snr_db=30, seed=7)) is None


def test_two_bursts_one_after_the_other():
    a = np.concatenate([M.synthesize(BITS[:3000], seed=8), np.zeros(phy.FS),
                        M.synthesize(BITS[3000:], repeat=2, seed=9)])
    y = _air(a, snr_db=18, seed=10)
    r1 = M.receive(y)
    r2 = M.receive(y, start=r1.end_sample)
    assert _same(r1.payload, BITS[:3000])
    assert r2.repeat == 2 and _same(r2.payload, BITS[3000:])


def test_the_columns_are_the_message_in_order():
    """A message written down six columns and sent a row at a time comes
    back, read down the columns, in its order; what it leaves empty is zero."""
    msg = np.random.default_rng(6).integers(0, 256, 300, dtype=np.uint8)
    col_bits = np.zeros(M.BLOCK_BITS, dtype=np.uint8)
    col_bits[: msg.size * 8] = np.unpackbits(msg)
    sent = col_bits.reshape(M.COLUMNS, -1).T.ravel()          # row by row
    r = M.receive(_air(M.synthesize(sent, end_marker=False, seed=11), snr_db=25, seed=12))
    cols = b"".join(r.columns)
    assert r.blocks == 1 and cols[: msg.size] == msg.tobytes()
    assert cols[msg.size:].count(0) > len(cols[msg.size:]) - 4


def _framed(k: int, seed: int) -> np.ndarray:
    """k blocks as the recordings have them: random payload down six columns,
    each column's marker byte four times at row 1056 - 32k of the last block,
    sent row by row."""
    rng = np.random.default_rng(seed)
    rows = k * M.BLOCK_BITS // M.COLUMNS
    cols = rng.integers(0, 2, (M.COLUMNS, rows)).astype(np.uint8)
    at = rows - M.BLOCK_BITS // M.COLUMNS + 1056 - 32 * k
    for c in range(M.COLUMNS):
        cols[c, at:at + 32] = np.unpackbits(np.full(4, M.COLUMN_MARKERS[c], dtype=np.uint8))
    return cols.T.ravel()


@pytest.mark.parametrize("k", [1, 3])
def test_the_markers_say_where_the_burst_ends(k):
    """The signal going on past the burst (here, more random symbols) does
    not lengthen it: the column markers count its blocks."""
    sent = _framed(k, seed=20 + k)
    more = np.random.default_rng(5).integers(0, 2, 3000).astype(np.uint8)
    r = M.receive(_air(M.synthesize(np.r_[sent, more], end_marker=False, seed=k),
                       snr_db=12, seed=k))
    assert r.framed == k and r.payload.size == k * M.BLOCK_BITS
    assert np.mean(r.payload != sent) < 0.01
