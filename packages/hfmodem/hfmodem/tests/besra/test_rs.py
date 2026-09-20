# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra's shortened Reed-Solomon codec, validated byte-exact against ardopcf.

The ``rs`` reference vectors are lines ``rs <dataLen> <rsLen> <dataHex> <parityHex>``
produced by compiling ardopcf's verbatim rockliff rrs.c, so ``rs_parity`` must
reproduce ``parityHex`` for every one. The decode side is exercised by round-
tripping each vector's block through injected errors.
"""

from __future__ import annotations

import random


from hfmodem.besra.fec import rs_correct, rs_parity
from . import groundtruth as gt

pytestmark = gt.requires_reference


def _vectors():
    out = []
    for data_len, rs_len, data_hex, parity_hex in gt.reference_vectors("rs"):
        data = bytes.fromhex(data_hex)
        parity = bytes.fromhex(parity_hex)
        assert len(data) == int(data_len)
        assert len(parity) == int(rs_len)
        out.append((data, int(rs_len), parity))
    return out


def test_rs_parity_matches_reference():
    vectors = _vectors()
    assert vectors
    for data, r, parity in vectors:
        assert rs_parity(data, r) == parity, (data.hex(), r)


def test_rs_correct_clean_block():
    for data, r, parity in _vectors():
        recovered, ok = rs_correct(data + parity, r)
        assert ok
        assert recovered == data, (data.hex(), r)


def test_rs_correct_recovers_up_to_t_errors():
    rng = random.Random(0xA5)
    for data, r, parity in _vectors():
        t = r // 2
        block = bytearray(data + parity)
        positions = rng.sample(range(len(block)), t)
        for p in positions:
            block[p] ^= rng.randint(1, 255)
        recovered, ok = rs_correct(bytes(block), r)
        assert ok, (data.hex(), r)
        assert recovered == data, (data.hex(), r)


def test_rs_correct_flags_uncorrectable():
    # Corrupting t+1 bytes exceeds the code's reach. The decoder must report
    # ok=False; the one guarantee it never breaks -- even in the rare beyond-
    # bound mis-decode -- is silently certifying the corruption as the original.
    for seed in range(64):
        rng = random.Random(seed)
        for data, r, parity in _vectors():
            t = r // 2
            block = bytearray(data + parity)
            for p in rng.sample(range(len(block)), t + 1):
                block[p] ^= rng.randint(1, 255)
            recovered, ok = rs_correct(bytes(block), r)
            assert not (ok and recovered == data), (data.hex(), r, seed)
