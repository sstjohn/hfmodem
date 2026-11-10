# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Codeword <-> OFDM-cell bit-stream layout: both interleave disciplines.

Low gears (striped): every codeword is bit-striped across the whole frame,
spanning all carriers and the frame's full duration -- maximum frequency and
time diversity, the same striping as M2's FrameCodec; a repeat gear tiles
the stream and the receiver sums LLRs across the copies.

High gears (grouped): each codeword is confined to one of the N_GROUPS
carrier groups, so a notched or faded group kills only its own codewords --
the selective-ACK salvage unit. Assignment is a deterministic greedy
balance, recomputed identically at both ends from the header alone: each
codeword goes to the live group that would finish transmitting earliest,
ties to the lowest group index.
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.frame.codec import pn9
from hfmodem.sabir.phy.modem import N_GROUPS, carrier_groups


def _group_slots(phy, n_syms: int, loading):
    """Per group: (flat bit-slot indices, slot symbol rows), chronological."""
    s_idx, r_idx, cb, off = phy.cell_map(n_syms, loading=loading)
    grp = carrier_groups(phy.gear.n_carriers)[r_idx]
    out = []
    for g in range(N_GROUPS):
        sel = np.nonzero(grp == g)[0]
        if sel.size == 0:
            out.append((np.empty(0, np.int64), np.empty(0, np.int64)))
            continue
        b = int(cb[sel[0]])                 # loading is uniform within a group
        slots = (off[sel][:, None] + np.arange(b)).ravel()
        out.append((slots, np.repeat(s_idx[sel], b)))
    return out


def assign(phy, n_cw: int, cw_bits: int, loading) -> list[int]:
    """Codeword -> carrier-group assignment (grouped gears)."""
    probe = 4 * 3                           # whole lattice periods
    rate = np.array([s.size / probe
                     for s, _ in _group_slots(phy, probe, loading)])
    if not rate.any():
        raise ValueError("no live carrier groups")
    load = np.zeros(N_GROUPS)
    out = []
    for _ in range(n_cw):
        t = np.where(rate > 0, (load + cw_bits) / np.maximum(rate, 1e-9),
                     np.inf)
        g = int(np.argmin(t))
        out.append(g)
        load[g] += cw_bits
    return out


def _n_syms_grouped(phy, need: np.ndarray, loading) -> int:
    rows = phy.n_symbols_for(int(need.sum()), loading=loading)
    while True:
        per = _group_slots(phy, rows, loading)
        worst = 0
        for g in range(N_GROUPS):
            n = int(need[g])
            if not n:
                continue
            slots, sym = per[g]
            if slots.size < n:
                worst = -1
                break
            worst = max(worst, int(sym[n - 1]) + 1)
        if worst > 0:
            return worst
        rows *= 2


def assemble(phy, coded: np.ndarray, grouped: bool, repeat: int = 1,
             loading=None) -> tuple[np.ndarray, int]:
    """Coded codewords (n_cw, n) -> (flat bit stream for transmit, n_syms)."""
    coded = np.atleast_2d(np.asarray(coded, dtype=np.int64))
    n_cw, n = coded.shape
    if not grouped:
        stream = np.tile(coded.T.ravel(), repeat)
        return stream, phy.n_symbols_for(stream.size, loading=loading)
    if repeat != 1:
        raise ValueError("grouped layout does not tile repeats")
    groups = assign(phy, n_cw, n, loading)
    need = np.bincount(groups, minlength=N_GROUPS) * n
    n_syms = _n_syms_grouped(phy, need, loading)
    per = _group_slots(phy, n_syms, loading)
    cap = sum(int(s.size) for s, _ in per)
    bits = np.empty(cap, dtype=np.int64)
    fill = pn9(cap)
    for g in range(N_GROUPS):
        slots, _ = per[g]
        seq = coded[[j for j, gg in enumerate(groups) if gg == g]].ravel()
        bits[slots[: seq.size]] = seq
        bits[slots[seq.size:]] = fill[slots[seq.size:]]
    return bits, n_syms


def extract(phy, llr: np.ndarray, n_cw: int, n: int, grouped: bool,
            repeat: int = 1, loading=None,
            n_syms: int | None = None) -> np.ndarray:
    """Flat received LLR stream -> per-codeword LLR rows (n_cw, n)."""
    if not grouped:
        stream = n_cw * n
        return llr[: repeat * stream].reshape(repeat, n, n_cw).sum(axis=0).T
    groups = assign(phy, n_cw, n, loading)
    per = _group_slots(phy, n_syms, loading)
    out = np.empty((n_cw, n))
    used = np.zeros(N_GROUPS, dtype=np.int64)
    for j, g in enumerate(groups):
        out[j] = llr[per[g][0][used[g] : used[g] + n]]
        used[g] += n
    return out
