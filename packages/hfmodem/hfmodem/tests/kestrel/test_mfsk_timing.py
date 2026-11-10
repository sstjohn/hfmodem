# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The per-symbol advance, pinned to the measurement it came from.

The advance is **2048 samples @48 kHz**, measured symbol start to symbol start over
85 handshake bursts in five recordings, 44 distinct tone sequences and 3 gateway
callsigns: 2048.00 ± 0.01 statistical, ± 0.05 systematic [spec 04 §4.4.1a]. It was
previously *derived* as `stride − WOLA = 2074 − 32 = 2042`, a subtraction nobody had
measured, and the whole suite was blind to the error: a 41-symbol connect-request
ran 246 samples short, its last symbol displaced 11.7 % of a symbol, and every
recogniser test still passed. Hence this file, which fails when the constant moves.

Two of the three tests below need no recording, because the advance leaves fixed
points in the arithmetic:

  * it is the exact reciprocal of the 23.4375 Hz tone spacing, so a symbol spans a
    whole number of cycles of every tone — orthogonal MFSK;
  * and therefore every symbol starts at the same phase, so a global oscillator and
    a per-symbol phase reset emit identical samples. That equivalence is what hid
    the wrong advance: kestrel synthesised globally, and the error only showed as
    0.3 dB of lost matched-filter output. Synthesising the same wrong advance with a
    per-symbol reset takes whole-burst correlation against the real KC9GHZ
    connect-request from 0.96 to 0.41.

The third measures the advance again, from real gateway audio.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import fftconvolve

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_MEASURED_ADVANCE = 2048        # [spec 04 §4.4.1a]


def test_the_advance_is_the_measured_one():
    assert MK.HOP == _MEASURED_ADVANCE, (
        f"per-symbol advance is {MK.HOP}; it was measured off air at "
        f"{_MEASURED_ADVANCE} [spec 04 §4.4.1a]. It is a measurement, not a "
        "quantity to derive from the stride and the cross-fade.")
    assert MK.HOP * MK.carrier_to_hz(1) / MK.FS == 1.0, (
        "the advance is no longer the reciprocal of the tone spacing, so adjacent "
        "tones are not orthogonal over a symbol")
    assert MK.STRIDE == MK.HOP + MK.WOLA_N, (
        "a symbol extends one advance plus the cross-fade it overlaps its "
        "successor by")


def test_the_phase_convention_is_not_a_choice():
    """At the measured advance, global phase and a per-symbol reset are one signal.

    A synthesiser that resets phase each symbol is the natural way to write this,
    and at advance 2042 it correlates 0.41 against the real connect-request where
    the global convention manages 0.96 — so this equivalence is the property that
    kept the wrong advance survivable, and losing it means the advance has drifted
    off a multiple of the FFT length.
    """
    tones = VF.handshake_tones("NS0A", VF.CR)
    reset = np.zeros((len(tones) - 1) * MK.HOP + MK.STRIDE)
    n = np.arange(MK.STRIDE)
    for k, c in enumerate(tones):
        a = k * MK.HOP
        reset[a:a + MK.STRIDE] += 0.5 * MK._W * np.sin(2 * np.pi * c * n / MK.NFFT)
    assert np.allclose(reset, MK.synth_tones(tones), atol=1e-9)


# The advance, measured off the recordings rather than assumed: a sliding complex
# DFT at each of the burst's own tones gives the symbol-start phase at every
# candidate position, and the true advance is the one that makes those phases
# constant across the burst (a wrong advance rotates symbol k by 2*pi*c_k*k*delta
# /2048, which scatters because the tones differ). Maximised over start offset and
# over the rig's frequency offset, which is a phase ramp linear in the symbol index.
_OFFAIR = [
    ("NS0A_2300", "NS0A", VF.CONNECT_RESPONSE, (9.0, 11.5), None, 0.97),
    # KC9GHZ's window holds 39 of the connect-request's 41 symbols; the tail is
    # outside the recording (projection 0.01, 0.16) and carries no phase.
    ("KC9GHZ_2300", "KC9GHZ", VF.CR, (4.5, 7.0), 39, 0.99),
]


def _phase_coherence(audio, tones, anchor, advance, slack=64):
    x = np.asarray(audio, float)
    bins = {c: fftconvolve(x, np.conj(np.exp(1j * 2 * np.pi * c * np.arange(MK.NFFT)
                                             / MK.NFFT))[::-1], mode="valid")
            for c in set(tones)}
    starts = np.arange(max(0, anchor - slack), anchor + slack + 1)
    idx = starts[:, None] + np.arange(len(tones))[None, :] * advance
    inside = (idx >= 0) & (idx < len(next(iter(bins.values()))))
    z = np.stack([bins[c][np.clip(idx[:, k], 0, len(bins[c]) - 1)]
                  for k, c in enumerate(tones)], axis=1)
    z = np.where(inside, z / (np.abs(z) + 1e-30), 0)      # phase only: fade-blind
    df = np.arange(-8.0, 8.01, 0.25)[:, None]
    ramp = np.exp(-2j * np.pi * df * np.arange(len(tones))[None, :] * advance / MK.FS)
    return float(np.abs(np.einsum("sk,dk->ds", z, ramp)).max() / len(tones))


@corpora.requires_gateway_session
@pytest.mark.parametrize("session,call,kind,span,keep,floor", _OFFAIR,
                         ids=[c[0] for c in _OFFAIR])
def test_real_gateway_bursts_advance_by_the_measured_amount(session, call, kind,
                                                            span, keep, floor):
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    fs, x = wavfile.read(str(path))
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    audio = x[int(span[0] * fs):int(span[1] * fs)] / (np.abs(x).max() or 1.0)

    tones = VF.handshake_tones(call, kind)[:keep]
    anchor = MK.lock_preamble(audio, kind)
    assert anchor is not None, "no burst located in a window known to hold one"

    scores = {a: _phase_coherence(audio, tones, anchor, a) for a in range(2040, 2057)}
    best = max(scores, key=scores.get)
    assert best == _MEASURED_ADVANCE, (
        f"{session}'s burst advances by {best} samples/symbol, not "
        f"{_MEASURED_ADVANCE}: {dict(sorted(scores.items()))}")
    assert scores[best] >= floor, (
        f"symbol-start phases at the measured advance are only {scores[best]:.3f} "
        f"coherent (was {floor:.2f} when measured) — the estimator or the "
        "recording changed, not the constant")
    # Immediate neighbours are close on a short burst — 23 symbols of phase resolve
    # ±1 sample by 0.018 — so the margin is required against everything further out.
    assert scores[best] - max(v for a, v in scores.items()
                              if abs(a - best) >= 2) > 0.05, (
        f"the advance is no longer resolved to ±1 sample: {dict(sorted(scores.items()))}")
    assert scores[2042] < 0.75, (
        f"the superseded advance 2042 scores {scores[2042]:.3f}, close enough to the "
        f"measured {best} that this test no longer discriminates them")
