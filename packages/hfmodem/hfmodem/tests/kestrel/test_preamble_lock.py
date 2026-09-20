# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Locating a handshake burst: tolerant of fading, intolerant of noise.

Two properties that pull in opposite directions, both measured rather than assumed:

  * A single deep fade on ONE preamble symbol used to discard the whole burst — on
    every one of the 8 and 10 preamble positions. That is the wrong failure mode
    for a channel whose defining behaviour is fading, and it is backwards from
    tolerating fading in the payload, which is the part that actually identifies
    the station. The gate now allows one lost tone.
  * Relaxing a matched filter is exactly how false locks appear, so the cost is
    pinned here too.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_KINDS = [(VF.CONNECT_RESPONSE, "connect-response"), (VF.CR, "connect-request")]
_IDS = [k[1] for k in _KINDS]


def _fade_one_symbol(burst: np.ndarray, sym: int, depth_db: float) -> np.ndarray:
    y = burst.copy()
    s = sym * MK.HOP
    y[s:min(s + MK.STRIDE, len(y))] *= 10 ** (-depth_db / 20)
    return y


@pytest.mark.parametrize("kind,name", _KINDS, ids=_IDS)
def test_one_faded_preamble_symbol_does_not_lose_the_burst(kind, name):
    rng = np.random.default_rng(3)
    base = MK.synth_burst("NS0A", kind)
    noise_sigma = np.sqrt((base ** 2).mean())          # 0 dB SNR
    for sym in range(len(kind.preamble)):
        faded = _fade_one_symbol(base, sym, 34.0)
        rx = np.concatenate([np.zeros(9000), faded, np.zeros(9000)])
        rx = rx + rng.normal(0.0, noise_sigma, len(rx))
        assert MK.lock_preamble(rx, kind) is not None, (
            f"a 34 dB fade on preamble symbol {sym} of {len(kind.preamble)} lost the "
            f"whole {name} — the gate is demanding a perfect preamble again")


@pytest.mark.parametrize("kind,name", _KINDS, ids=_IDS)
def test_noise_does_not_produce_a_lock(kind, name):
    """The cost side of the relaxation above."""
    false_locks = sum(
        MK.lock_preamble(np.random.default_rng(i).standard_normal(48000), kind) is not None
        for i in range(120))
    assert false_locks == 0, f"{false_locks}/120 locks on pure noise for {name}"


@pytest.mark.parametrize("kind,name", _KINDS, ids=_IDS)
def test_lock_lands_near_the_true_start(kind, name):
    """A lock that is merely 'somewhere in the burst' is not good enough: the payload
    is only correct over part of the preamble-match plateau."""
    rng = np.random.default_rng(11)
    for pad in (0, 137, 4096, 20000):
        rx = np.concatenate([rng.normal(0, 1e-3, pad),
                             MK.synth_burst("NS0A", kind),
                             rng.normal(0, 1e-3, 5000)])
        at = MK.lock_preamble(rx, kind)
        assert at is not None, f"no lock at pad={pad}"
        assert abs(at - pad) < MK.STRIDE // 2, (
            f"lock at {at} for a burst starting at {pad} — off by "
            f"{abs(at - pad)} samples, more than half a symbol")
        n = len(kind.preamble) + kind.n_payload
        m, N = VF.payload_match(MK.demod_tones(rx[at:], n)[len(kind.preamble):],
                                "NS0A", kind)
        assert m == N, f"payload {m}/{N} at pad={pad}: lock is off-centre"
