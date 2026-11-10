# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The Reed-Solomon code on every ARDOP carrier block.

ARDOP fixes the code and its byte order; everything else here is ours. The field
is GF(256) built on x^8+x^4+x^3+x^2+1 (0x11D) with α = 2, and the code is the
length-255 one whose generator has the roots α^1 … α^r. A block is ``d`` data
bytes then ``r`` parity bytes, ``n = d + r`` of them, correcting up to ``r // 2``
byte errors.

Read the block as a polynomial whose coefficient of x^i is ``block[i]``: it is a
codeword exactly when it vanishes at all r roots. That one statement fixes the
wire layout, so it is the whole encoder: with ``D(x)`` the data and ``g(x)`` the
generator, ``x^255 ≡ 1 (mod g)`` makes

    P(x) = D(x) · x^(255 - d)  mod g(x)

the parity whose coefficients, constant term first, follow the data. The
positions ``n … 254`` that a short block leaves unused are the shortening: always
zero, never sent.

Decoding is the textbook sequence over the syndromes ``S_j = block(α^j)``:
Berlekamp-Massey for the error locator, a search of all 255 positions for its
roots, and Forney's formula for the values. A locator longer than ``r // 2``, or
one with fewer roots than its degree, means the block is past the code's reach.
So does a correction that lands in the shortening: that position is known to be
zero, so a decoder that wants to change it has found the wrong codeword.
"""

from __future__ import annotations

from functools import lru_cache

FIELD_POLY = 0x11D
ORDER = 255                     # nonzero field elements, and the full code length

_EXP = [0] * (2 * ORDER)        # doubled so a sum of two logs never needs reducing
_LOG = [0] * 256
_x = 1
for _i in range(ORDER):
    _EXP[_i] = _EXP[_i + ORDER] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= FIELD_POLY
del _x, _i


def _mul(a: int, b: int) -> int:
    return _EXP[_LOG[a] + _LOG[b]] if a and b else 0


def _div(a: int, b: int) -> int:
    return _EXP[_LOG[a] + ORDER - _LOG[b]] if a else 0


@lru_cache(maxsize=None)
def _generator(r: int) -> tuple[int, ...]:
    """g(x) = (x - α)(x - α²)…(x - α^r), highest power first; monic."""
    g = [1]
    for j in range(1, r + 1):
        root = _EXP[j]
        g = [hi ^ _mul(lo, root) for hi, lo in zip(g + [0], [0] + g)]
    return tuple(g)


def rs_parity(data: bytes, r: int) -> bytes:
    """The ``r`` parity bytes ARDOP sends after ``data``."""
    d = len(data)
    if r <= 0 or r % 2 or d + r > ORDER:
        raise ValueError(f"invalid RS parameters: d={d}, r={r}")
    taps = [(j, _LOG[c]) for j, c in enumerate(_generator(r)) if j and c]
    # D(x)·x^(255-d) highest power first is the data reversed, then zeros; divide
    # by g in place and the last r slots are the remainder.
    work = list(reversed(data)) + [0] * (ORDER - d)
    for i in range(ORDER - r):
        if work[i]:
            lead = _LOG[work[i]]
            for j, tap in taps:
                work[i + j] ^= _EXP[tap + lead]
    return bytes(reversed(work[ORDER - r:]))


def _syndromes(block: bytes, r: int) -> list[int]:
    out = []
    for j in range(1, r + 1):
        x, acc = _EXP[j], 0
        for byte in reversed(block):
            acc = _mul(acc, x) ^ byte
        out.append(acc)
    return out


def _locator(synd: list[int]) -> tuple[list[int], int]:
    """Berlekamp-Massey: the shortest connection polynomial Λ (constant term
    first, Λ0 = 1) that generates ``synd``, and its length L."""
    lam, last = [1], [1]
    length, gap, last_delta = 0, 1, 1
    for k, s in enumerate(synd):
        delta = s
        for i in range(1, min(length, len(lam) - 1) + 1):
            delta ^= _mul(lam[i], synd[k - i])
        if not delta:
            gap += 1
            continue
        scale = _div(delta, last_delta)
        grown = lam + [0] * max(0, len(last) + gap - len(lam))
        for i, c in enumerate(last):
            grown[i + gap] ^= _mul(scale, c)
        if 2 * length <= k:
            last, last_delta = lam, delta
            length, gap = k + 1 - length, 1
        else:
            gap += 1
        lam = grown
    return lam, length


def _poly_at(coeffs: list[int], x: int) -> int:
    """Evaluate a constant-term-first polynomial at x."""
    acc = 0
    for c in reversed(coeffs):
        acc = _mul(acc, x) ^ c
    return acc


def rs_correct(block: bytes, r: int) -> tuple[bytes, bool]:
    """Correct up to ``r // 2`` byte errors in ``data ‖ parity``.

    Returns ``(data, ok)``. ``ok`` is True when the block is, or was corrected
    to, a codeword. When the locator is unusable the received data comes back
    unchanged; when the only correction on offer reaches into the shortening the
    data comes back with that correction applied and ``ok`` False, since the
    decode was found and then disowned rather than never found.
    """
    n = len(block)
    if r <= 0 or r % 2 or not r <= n <= ORDER:
        raise ValueError(f"invalid RS parameters: len={n}, r={r}")
    d = n - r
    synd = _syndromes(block, r)
    if not any(synd):
        return bytes(block[:d]), True

    lam, length = _locator(synd)
    if length > r // 2:
        return bytes(block[:d]), False

    positions = [p for p in range(ORDER) if not _poly_at(lam, _EXP[ORDER - p])]
    if len(positions) != length:
        return bytes(block[:d]), False

    # Forney, first root α^1: e = Ω(X⁻¹) / Λ'(X⁻¹) with Ω = S·Λ mod x^r. In
    # characteristic 2 the formal derivative keeps only the odd-power terms.
    omega = [0] * r
    for i, s in enumerate(synd):
        for j, c in enumerate(lam[:r - i]):
            omega[i + j] ^= _mul(s, c)
    dlam = [c if i % 2 else 0 for i, c in enumerate(lam)][1:]

    fixed = bytearray(block)
    phantom = False
    for p in positions:
        inv = _EXP[ORDER - p]
        value = _div(_poly_at(omega, inv), _poly_at(dlam, inv))
        if not value:
            continue
        if p < n:
            fixed[p] ^= value
        else:
            phantom = True
    return bytes(fixed[:d]), not phantom
