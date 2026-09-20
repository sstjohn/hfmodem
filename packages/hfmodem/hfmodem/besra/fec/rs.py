# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Shortened Reed-Solomon over GF(2^8), byte-exact with ardopcf's rockliff codec.

The field is GF(2^8) with primitive polynomial 0x11D (x^8+x^4+x^3+x^2+1) and
primitive element alpha = 2; the generator has consecutive roots alpha^1..alpha^r
(fcr = 1). A carrier block carries ``d`` data bytes plus ``r`` parity bytes,
correcting ``t = r//2`` byte errors as a shortened RS(d+r, d) drawn from the
length-255 field.

Wire layout of the parity ardopcf appends (what the vectors pin down): the ``d``
data bytes sit right-justified in the length-(255-r) message region, left-padded
with zeros; the whole padded region is clocked through the systematic LFSR from
the highest-degree coefficient down; the ``r`` remainder bytes are then emitted
constant-term-first (b0..b_{r-1}) directly after the data. Because data[0] lands
at the *low* end of that region, the first transmitted data byte is the
lowest-order message coefficient -- the reversal a naive MSB-first encoder misses.

Transliterated structure-for-structure from Simon Rockliff's Reed-Solomon listing
as ardopcf carries it in ``lib/rockliff/rrs.c``: the ``alpha_to``/``index_of``
field tables, the ``elp[u][i]`` Berlekamp iteration, the Chien search over
``reg[]``/``root[]``/``loc[]``, and the Forney magnitudes in ``z[]``. That listing
was written by Simon Rockliff (University of Adelaide) in 1989 and 1991 and
modified by Peter LaRue in 2024 for use in ardopcf; ardopcf's MIT licence states
expressly that it does not cover its ``lib/`` directory, and the listing carries
its author's own terms:

    Notice
    --------
    This program may be freely modified and/or given to whoever wants it.
    A condition of such distribution is that the author's contribution be
    acknowledged by his name being left in the comments heading the program,
    however no responsibility is accepted for any financial or other loss which
    may result from some unforseen errors or malfunctioning of the program
    during use.
            Simon Rockliff, 26th June 1991

Rockliff's notice above and ardopcf's own MIT notice both travel with this
distribution in ``NOTICE``.
"""

from __future__ import annotations

NN = 255
MM = 8
PRIM_POLY_BITS = (1, 0, 1, 1, 1, 0, 0, 0, 1)  # 0x11D, low bit = x^0


def _generate_gf() -> tuple[list[int], list[int]]:
    alpha_to = [0] * (NN + 1)
    index_of = [0] * (NN + 1)
    mask = 1
    for i in range(MM):
        alpha_to[i] = mask
        index_of[mask] = i
        if PRIM_POLY_BITS[i]:
            alpha_to[MM] ^= mask
        mask <<= 1
    index_of[alpha_to[MM]] = MM
    mask >>= 1
    for i in range(MM + 1, NN):
        prev = alpha_to[i - 1]
        if prev >= mask:
            alpha_to[i] = alpha_to[MM] ^ ((prev ^ mask) << 1)
        else:
            alpha_to[i] = prev << 1
        index_of[alpha_to[i]] = i
    index_of[0] = -1
    return alpha_to, index_of


ALPHA_TO, INDEX_OF = _generate_gf()

_GEN_CACHE: dict[int, list[int]] = {}


def _generator(r: int) -> list[int]:
    """g(x) = prod_{i=1..r} (x - alpha^i), coefficients returned in index form."""
    gg = _GEN_CACHE.get(r)
    if gg is not None:
        return gg
    gg = [0] * (r + 1)
    gg[0] = 2
    gg[1] = 1
    for i in range(2, r + 1):
        gg[i] = 1
        for j in range(i - 1, 0, -1):
            if gg[j] != 0:
                gg[j] = gg[j - 1] ^ ALPHA_TO[(INDEX_OF[gg[j]] + i) % NN]
            else:
                gg[j] = gg[j - 1]
        gg[0] = ALPHA_TO[(INDEX_OF[gg[0]] + i) % NN]
    gg = [INDEX_OF[c] for c in gg]
    _GEN_CACHE[r] = gg
    return gg


def _encode(padded: list[int], r: int) -> list[int]:
    gg = _generator(r)
    kk = NN - r
    bb = [0] * r
    for i in range(kk - 1, -1, -1):
        feedback = INDEX_OF[padded[i] ^ bb[r - 1]]
        if feedback != -1:
            for j in range(r - 1, 0, -1):
                if gg[j] != -1:
                    bb[j] = bb[j - 1] ^ ALPHA_TO[(gg[j] + feedback) % NN]
                else:
                    bb[j] = bb[j - 1]
            bb[0] = ALPHA_TO[(gg[0] + feedback) % NN]
        else:
            for j in range(r - 1, 0, -1):
                bb[j] = bb[j - 1]
            bb[0] = 0
    return bb


def rs_parity(data: bytes, r: int) -> bytes:
    """Return the ``r`` Reed-Solomon parity bytes ardopcf appends to ``data``."""
    d = len(data)
    if r <= 0 or r % 2 or d + r > NN:
        raise ValueError(f"invalid RS parameters: d={d}, r={r}")
    padded = [0] * NN
    padded[NN - r - d:NN - r] = data
    return bytes(_encode(padded, r))


def _decode(rcvd: list[int], r: int) -> tuple[list[int], int]:
    """Berlekamp-Massey / Chien / Forney. ``rcvd`` is length-255 index form.

    Returns (codeword in polynomial form, retval) with retval 0 = clean,
    1 = errors corrected (maybe), -1 = uncorrectable.
    """
    tt = r // 2

    def to_poly(v: list[int]) -> list[int]:
        return [ALPHA_TO[x] if x != -1 else 0 for x in v]

    s = [0] * (r + 1)
    syn_error = False
    for i in range(1, r + 1):
        acc = 0
        for j in range(NN):
            if rcvd[j] != -1:
                acc ^= ALPHA_TO[(rcvd[j] + i * j) % NN]
        if acc != 0:
            syn_error = True
        s[i] = INDEX_OF[acc]

    if not syn_error:
        return to_poly(rcvd), 0

    elp = [[0] * (r + 2) for _ in range(r + 3)]
    dd = [0] * (r + 3)
    ll = [0] * (r + 3)
    u_lu = [0] * (r + 3)

    dd[0] = 0
    dd[1] = s[1]
    elp[0][0] = 0
    elp[1][0] = 1
    for i in range(1, r):
        elp[0][i] = -1
        elp[1][i] = 0
    u_lu[0] = -1
    u = 0
    while True:
        u += 1
        if dd[u] == -1:
            ll[u + 1] = ll[u]
            for i in range(ll[u] + 1):
                elp[u + 1][i] = elp[u][i]
                elp[u][i] = INDEX_OF[elp[u][i]]
        else:
            q = u - 1
            while dd[q] == -1 and q > 0:
                q -= 1
            if q > 0:
                j = q
                while True:
                    j -= 1
                    if dd[j] != -1 and u_lu[q] < u_lu[j]:
                        q = j
                    if j <= 0:
                        break
            ll[u + 1] = max(ll[u], ll[q] + u - q)
            for i in range(r):
                elp[u + 1][i] = 0
            for i in range(ll[q] + 1):
                if elp[q][i] != -1:
                    elp[u + 1][i + u - q] = ALPHA_TO[
                        (dd[u] + NN - dd[q] + elp[q][i]) % NN]
            for i in range(ll[u] + 1):
                elp[u + 1][i] ^= elp[u][i]
                elp[u][i] = INDEX_OF[elp[u][i]]
        u_lu[u + 1] = u - ll[u + 1]

        if u < r:
            dd[u + 1] = ALPHA_TO[s[u + 1]] if s[u + 1] != -1 else 0
            for i in range(1, ll[u + 1] + 1):
                if s[u + 1 - i] != -1 and elp[u + 1][i] != 0:
                    dd[u + 1] ^= ALPHA_TO[
                        (s[u + 1 - i] + INDEX_OF[elp[u + 1][i]]) % NN]
            dd[u + 1] = INDEX_OF[dd[u + 1]]

        if not (u < r and ll[u + 1] <= tt):
            break
    u += 1

    if ll[u] > tt:
        return to_poly(rcvd), -1

    for i in range(ll[u] + 1):
        elp[u][i] = INDEX_OF[elp[u][i]]

    reg = [0] * (tt + 1)
    for i in range(1, ll[u] + 1):
        reg[i] = elp[u][i]
    root = [0] * (NN + 1)
    loc = [0] * (NN + 1)
    count = 0
    for i in range(1, NN + 1):
        q = 1
        for j in range(1, ll[u] + 1):
            if reg[j] != -1:
                reg[j] = (reg[j] + j) % NN
                q ^= ALPHA_TO[reg[j]]
        if q == 0:
            root[count] = i
            loc[count] = NN - i
            count += 1

    if count != ll[u]:
        return to_poly(rcvd), -1

    z = [0] * (tt + 1)
    for i in range(1, ll[u] + 1):
        si, ei = s[i], elp[u][i]
        if si != -1 and ei != -1:
            z[i] = ALPHA_TO[si] ^ ALPHA_TO[ei]
        elif si != -1:
            z[i] = ALPHA_TO[si]
        elif ei != -1:
            z[i] = ALPHA_TO[ei]
        else:
            z[i] = 0
        for j in range(1, i):
            if s[j] != -1 and elp[u][i - j] != -1:
                z[i] ^= ALPHA_TO[(elp[u][i - j] + s[j]) % NN]
        z[i] = INDEX_OF[z[i]]

    out = to_poly(rcvd)
    err = [0] * NN
    for i in range(ll[u]):
        loc_i = loc[i]
        err[loc_i] = 1
        for j in range(1, ll[u] + 1):
            if z[j] != -1:
                err[loc_i] ^= ALPHA_TO[(z[j] + j * root[i]) % NN]
        if err[loc_i] != 0:
            err[loc_i] = INDEX_OF[err[loc_i]]
            denom = 0
            for j in range(ll[u]):
                if j != i:
                    denom += INDEX_OF[1 ^ ALPHA_TO[(loc[j] + root[i]) % NN]]
            denom %= NN
            err[loc_i] = ALPHA_TO[(err[loc_i] - denom + NN) % NN]
            out[loc_i] ^= err[loc_i]
    return out, 1


def rs_correct(block: bytes, r: int) -> tuple[bytes, bool]:
    """Correct up to ``t = r//2`` byte errors in ``[data ‖ parity]``.

    Returns (corrected_data, ok) where ``ok`` is True when the block decoded to
    a valid codeword (no errors or a successful correction) and False when the
    error pattern was beyond the code's reach.
    """
    combined = len(block)
    if r <= 0 or r % 2 or combined > NN or combined < r:
        raise ValueError(f"invalid RS parameters: len={combined}, r={r}")
    d = combined - r

    padded = [0] * NN
    padded[NN - combined:] = block
    idx = [INDEX_OF[x] for x in padded]

    out, retval = _decode(idx, r)
    data = bytes(out[NN - combined:NN - combined + d])

    if retval == -1:
        return data, False
    if retval == 1:
        # Non-zero values in the shortening pad mean the "correction" landed on
        # phantom symbols: the decode was a false positive (ardopcf rrs.c heuristic).
        if any(out[i] != 0 for i in range(NN - combined)):
            return data, False
    return data, True
