# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Wire-level outputs held to a digest of what the modem produced before its
Reed-Solomon codec, Packed6 fields and PSK symbol packer were rewritten.

Each test sweeps a seeded grid much wider than the reference vectors — every
parity length the frame catalog uses, error counts from none to three times the
code's reach, arbitrary words, malformed fields — and hashes every result. The
digests were recorded from the earlier implementation, so a change to any one
outcome, including which beyond-reach blocks happen to decode, fails here.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from hfmodem.besra.fec import rs_correct, rs_parity
from hfmodem.besra.frame import callsign
from hfmodem.besra.frame import frame as F
from hfmodem.besra.phy import modulator


def _digest(items) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(repr(item).encode())
    return h.hexdigest()


def _attempt(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:
        return ("ValueError", str(exc))


def _rs_outcomes():
    rng = random.Random(0x5EED)
    parities = sorted({fd.r for fd in F.FRAMES.values() if fd.r} | {2, 6, 50})
    for r in parities:
        t = r // 2
        for d in sorted({1, 12, 19, 67, 111, 131, 203, 255 - r}):
            if d > 255 - r:
                continue
            n = d + r
            for _ in range(3):
                data = bytes(rng.randrange(256) for _ in range(d))
                parity = rs_parity(data, r)
                yield r, data, parity
                for errors in sorted({0, 1, t, t + 1, t + 3, 3 * t}):
                    block = bytearray(data + parity)
                    for p in rng.sample(range(n), min(errors, n)):
                        block[p] ^= rng.randrange(1, 256)
                    yield r, bytes(block), rs_correct(bytes(block), r)
                word = bytes(rng.randrange(256) for _ in range(n))
                yield r, word, rs_correct(word, r)


def test_reed_solomon_outcomes_match_the_record():
    outcomes = list(_rs_outcomes())
    corrected = [o for o in outcomes if isinstance(o[2], tuple)]
    assert sum(ok for _, _, (_, ok) in corrected) > 0
    assert sum(not ok for _, _, (_, ok) in corrected) > 0
    assert _digest(outcomes) == RS_RECORD


def test_reed_solomon_rejects_the_same_shapes():
    for fn, args in ((rs_parity, (b"x", 0)), (rs_parity, (b"x", 3)),
                     (rs_parity, (bytes(250), 8)), (rs_correct, (b"x", 2)),
                     (rs_correct, (bytes(256), 8)), (rs_correct, (bytes(10), 5))):
        with pytest.raises(ValueError):
            fn(*args)
    assert rs_parity(b"", 4) == bytes(4)


def _packed6_outcomes():
    rng = random.Random(0xCA11)
    alphabet = [chr(c) for c in range(128)] + ["é", "☃"]
    for _ in range(1500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(11)))
        yield "encode", text, callsign.packed6_encode(text)
    for _ in range(1500):
        raw = bytes(rng.randrange(256) for _ in range(6))
        yield ("decode", raw, callsign.packed6_decode(raw),
               _attempt(callsign.unpack_callsign, raw), callsign.unpack_grid(raw))
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/abc-! "
    calls = ["W9SSJ", "K1ABC-15", "K1ABC-0", "K1ABC-z", "N0CALL-16", "K1ABC-",
             "K1ABC-1-2", "", "-5", "K1ABC-1A", "ABCDEFGH", "A"]
    calls += ["".join(rng.choice(chars) for _ in range(rng.randrange(1, 12)))
              for _ in range(1500)]
    for call in calls:
        packed = _attempt(callsign.pack_callsign, call)
        back = _attempt(callsign.unpack_callsign, packed) if isinstance(packed, bytes) else None
        yield "call", call, packed, back


def test_packed6_fields_match_the_record():
    assert _digest(_packed6_outcomes()) == PACKED6_RECORD


def _symbol_outcomes():
    rng = random.Random(0x5B5)
    for mod in (F.Mod.PSK4, F.Mod.PSK8, F.Mod.QAM16):
        for n in (0, 1, 2, 3, 5, 19, 67, 131, 195):
            block = bytes(rng.randrange(256) for _ in range(n))
            yield mod.name, block, modulator._carrier_symbols(block, mod)


def test_psk_symbols_match_the_record():
    assert _digest(_symbol_outcomes()) == SYMBOL_RECORD


@pytest.mark.parametrize("width, weight, whole", [
    (200, [0.7389, -1.0, 0.7389], [0, 0, 0]),
    (500, [0.10601, -0.59383, 1.0, -1.0, 1.0, -0.59383, 0.10601], [0, 0, 1, 1, 1, 0, 0]),
    (1000, [0.377] + [-1.0, 1.0] * 4 + [-1.0, 0.377], [0] + [1] * 9 + [0]),
    (2000, [0.371] + [1.0, -1.0] * 9 + [1.0, 0.371], [0] + [1] * 19 + [0]),
])
def test_filter_bins_carry_their_weights(width, weight, whole):
    f = modulator._TxFilter(width)
    np.testing.assert_array_equal(f._weight, weight)
    np.testing.assert_array_equal(f._whole, np.array(whole, dtype=bool))
    assert f._coef.size == len(weight)


def test_the_200_hz_filter_middle_bin_is_negative_off_centre_too():
    # Its middle bin is weighted -1 wherever the filter sits, not by parity.
    np.testing.assert_array_equal(modulator._TxFilter(200, centre=1600)._weight,
                                  [0.7389, -1.0, 0.7389])


RS_RECORD = "afda89664cc5d22b962bca714af5b7b1b40ef8d3a4e48a7b0e107332738bcea5"
PACKED6_RECORD = "131e3bcee562d103e33b740055571d5b330b0375f0a6da287e1f095c1424f4c2"
SYMBOL_RECORD = "bfac2cf29c23df08ac67ff74edad056ea88fe49c489067e25b3d402029dbac55"
