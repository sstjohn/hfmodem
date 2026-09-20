#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""
Unit tests for the audio null-test classifier.

We cannot run a real loopback in CI, so we *inject* each failure mode into a
clean reference and assert the classifier names it correctly:

    bit-exact  : integer-delayed copy, byte-identical  -> PASS / "bit-exact"
    SRC        : 48k -> 44.1k -> 48k round trip         -> FAIL / "src"
    dropped    : a block of samples deleted mid-stream  -> FAIL / "dropped"
    drift      : capture clock runs at +N ppm           -> FAIL / "drift"
    gain       : level scaled off unity                 -> FAIL (not transparent)
    sine floor : bit-exact -> -inf ; SRC -> raised floor

Run:  python -m pytest test_nulltest.py -v      (or)   python test_nulltest.py
Requires only numpy + scipy.
"""

import nulltest as nt
import numpy as np
import pytest

FS = 48000
DUR = 8.0  # seconds -- long enough for a meaningful lag track


def ref_prn(seed=1234):
    fmt = nt.AudioFormat(FS, 1, "float32")
    return nt.generate_reference("prn", DUR, fmt, seed=seed)


def resample_to_len(x, new_len):
    """Fractional resample via linear interpolation to an arbitrary length."""
    old_idx = np.arange(len(x))
    new_idx = np.linspace(0, len(x) - 1, new_len)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


# --------------------------------------------------------------------------- #

def test_bit_exact_pass():
    ref = ref_prn()
    lead = 1000
    cap = np.concatenate([np.zeros(lead, np.float32), ref]).astype(np.float32)
    v = nt.classify(cap, ref, FS, mode="prn")
    assert v.passed, v.summary()
    assert v.mode == "bit-exact"
    assert v.global_lag == lead
    assert v.residual_db == float("-inf")
    assert v.bit_exact_hash_match


def test_src_roundtrip_fail():
    ref = ref_prn()
    # 48000 -> 44100 -> 48000: a hidden 44.1k stage. Same nominal length back,
    # so lag stays ~constant but the samples are interpolated (imaging).
    down = resample_to_len(ref, int(len(ref) * 44100 / 48000))
    up = resample_to_len(down, len(ref))
    v = nt.classify(up, ref, FS, mode="prn")
    assert not v.passed, v.summary()
    assert v.mode == "src", v.summary()
    assert v.residual_db > -60.0  # clearly nonzero residual
    assert abs(v.lag_slope_ppm) < nt.DRIFT_PPM_THRESH  # no ramp


def test_dropped_samples_fail():
    ref = ref_prn()
    d = int(len(ref) * 0.6)   # drop point at 60%
    k = 500                   # samples removed
    cap = np.concatenate([ref[:d], ref[d + k:]]).astype(np.float32)
    v = nt.classify(cap, ref, FS, mode="prn", max_shift=4000)
    assert not v.passed, v.summary()
    assert v.mode == "dropped", v.summary()
    assert v.max_lag_step > nt.STEP_THRESH_SAMPLES


def test_clock_drift_fail():
    ref = ref_prn()
    ppm = 150.0
    # Capture clock runs fast: it samples the same signal at (1+ppm) rate, so
    # its stream is slightly shorter in reference time -> lag ramps linearly.
    new_len = int(round(len(ref) / (1.0 + ppm * 1e-6)))
    cap = resample_to_len(ref, new_len)
    v = nt.classify(cap, ref, FS, mode="prn", max_shift=4000)
    assert not v.passed, v.summary()
    assert v.mode == "drift", v.summary()
    assert abs(v.lag_slope_ppm) > nt.DRIFT_PPM_THRESH
    assert v.lag_r2 > nt.DRIFT_R2_THRESH
    # slope sign/magnitude should be in the right ballpark
    assert 50.0 < abs(v.lag_slope_ppm) < 400.0, v.lag_slope_ppm


def test_gain_change_fails_transparency():
    ref = ref_prn()
    cap = (ref * 0.5).astype(np.float32)  # -6 dB volume, otherwise perfect
    v = nt.classify(cap, ref, FS, mode="prn")
    assert not v.passed, v.summary()
    # residual after removing best-fit gain is ~ -inf: it's purely a level change
    assert v.residual_after_gain_db < -60.0
    assert abs(20 * np.log10(abs(v.gain))) > 3.0  # ~ -6 dB detected


def test_sine_noise_floor_bit_exact_is_minus_inf():
    fmt = nt.AudioFormat(FS, 1, "float32")
    ref = nt.generate_reference("sine", 4.0, fmt, freq=2900.0)
    v = nt.classify(ref.copy(), ref, FS, mode="sine", freq=2900.0)
    assert v.passed, v.summary()
    assert v.noise_floor_db == float("-inf")


def test_sine_noise_floor_src_is_raised():
    fmt = nt.AudioFormat(FS, 1, "float32")
    ref = nt.generate_reference("sine", 4.0, fmt, freq=2900.0)
    down = resample_to_len(ref, int(len(ref) * 44100 / 48000))
    up = resample_to_len(down, len(ref))
    v = nt.classify(up, ref, FS, mode="sine", freq=2900.0)
    assert not v.passed, v.summary()
    assert v.noise_floor_db is not None
    assert v.noise_floor_db > -120.0  # SRC raises the floor well above bit-exact


def test_wav_roundtrip_float32():
    import os
    import tempfile
    fmt = nt.AudioFormat(FS, 1, "float32")
    ref = nt.generate_reference("prn", 0.5, fmt)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.wav")
        nt.write_wav(p, ref, fmt)
        back, bfmt = nt.read_wav(p)
        assert bfmt.samplerate == FS
        assert bfmt.subtype == "float32"
        assert np.array_equal(back.astype(np.float32), ref)


def test_wav_roundtrip_int24_bitexact_classifies():
    """A 24-bit PCM round trip should still classify bit-exact against itself."""
    import os
    import tempfile
    fmt = nt.AudioFormat(FS, 1, "int24")
    ref = nt.generate_reference("prn", 0.5, fmt)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.wav")
        nt.write_wav(p, ref, fmt)
        back, _ = nt.read_wav(p)
    # quantized reference vs itself -> bit-exact
    v = nt.classify(back.copy(), back, FS, mode="prn")
    assert v.passed and v.mode == "bit-exact", v.summary()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
