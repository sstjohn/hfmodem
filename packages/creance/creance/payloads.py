# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

# Adapted from an earlier payload generator by the same author
# (AGPL-3.0-only).
"""Deterministic test payloads: zeros, ones, counter, PRBS 7/9/15/23/31.

All generators return exactly n bytes and are fully deterministic, so any
payload can be regenerated from its transcript label for later analysis.
"""

from __future__ import annotations

import re
from functools import partial
from typing import Callable, Dict


def zeros(n: int) -> bytes:
    return bytes(n)


def ones(n: int) -> bytes:
    return b"\xff" * n


def counter(n: int, start: int = 0, step: int = 1) -> bytes:
    """Incrementing byte counter (mod 256): every position gets a unique,
    ordered value — good for spotting reordering and block boundaries."""
    return bytes((start + i * step) & 0xFF for i in range(n))


# Maximal-length LFSR sequences (ITU-T O.150 polynomials), the standard
# stimulus for whitening/BER characterization. Taps are the 1-based bit
# positions feeding the XOR.
_PRBS_TAPS = {
    7: (7, 6),
    9: (9, 5),
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}


def prbs(n: int, order: int = 9, seed: int | None = None) -> bytes:
    """n bytes of a maximal-length PRBS, clocked one bit per output bit,
    MSB-first per byte. seed defaults to all-ones and must be non-zero."""
    if order not in _PRBS_TAPS:
        raise ValueError(f"unsupported PRBS order {order!r} (have {sorted(_PRBS_TAPS)})")
    taps = _PRBS_TAPS[order]
    mask = (1 << order) - 1
    state = (seed if seed is not None else mask) & mask
    if state == 0:
        raise ValueError("PRBS seed must be non-zero")

    out = bytearray(n)
    for i in range(n):
        b = 0
        for _ in range(8):
            fb = 0
            for tap in taps:
                fb ^= (state >> (tap - 1)) & 1
            b = ((b << 1) | fb) & 0xFF
            state = ((state << 1) | fb) & mask
        out[i] = b
    return bytes(out)


GENERATORS: Dict[str, Callable[[int], bytes]] = {
    "zeros": zeros,
    "ones": ones,
    "counter": counter,
    **{f"prbs{k}": partial(prbs, order=k) for k in _PRBS_TAPS},
}


_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(kib|mib|k|m)?b?$")
_SIZE_MULT = {None: 1, "k": 1024, "kib": 1024, "m": 1 << 20, "mib": 1 << 20}


def size_bytes(v) -> int:
    """Transfer size in bytes, from an int or a written form: 4096, 10k, 10kb,
    100KiB, 1M. One grammar for every source of a size — the CLI flag, a
    campaign plan's params, an API caller — so none of them can accept a
    spelling that dies later, mid-session, at the payload builder."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    m = _SIZE_RE.match(str(v).strip().lower())
    if not m:
        raise ValueError(f"bad size {v!r} (use 4096, 10k, 1M)")
    return int(float(m.group(1)) * _SIZE_MULT[m.group(2)])


def build(name: str, n: int) -> bytes:
    try:
        gen = GENERATORS[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown payload {name!r} (have {', '.join(GENERATORS)})") from None
    return gen(n)
