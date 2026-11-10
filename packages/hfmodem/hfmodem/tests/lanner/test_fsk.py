# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The two-tone core: modulate, discriminate, find the symbol grid, sample it."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.lanner import fsk

F_HI, F_LO = 2125.0, 1955.0


@pytest.mark.parametrize('fs', [6000, 48000])
@pytest.mark.parametrize('offset,idle', [(0, True), (-9, True), (9, True),
                                      (-50, False), (50, False), (-170, False)])
def test_acquired_mark_idle_resolves_neighbours_and_tuning_error(fs, offset, idle):
    t = np.arange(fs // 4) / fs
    audio = .5 * np.sin(2 * np.pi * (F_HI + offset) * t)
    audio += np.random.default_rng(19).normal(0, .001, len(audio))
    assert fsk.mark_idle(audio, fs, mark_hz=F_HI, space_hz=F_LO, power=.05) == idle


@pytest.mark.parametrize('fs', [6000, 48000])
@pytest.mark.parametrize('other_tone', [F_LO, F_HI + 170, F_HI + 200])
def test_mark_idle_rejects_keying_even_when_the_other_tone_is_outside_the_pair(fs, other_tone):
    units = [True, True, False, False] * 3
    audio = fsk.modulate(units, baud=45.45, fs=fs, f_hi=F_HI, f_lo=other_tone)
    assert not fsk.mark_idle(.7 * audio[:fs // 4], fs,
                             mark_hz=F_HI, space_hz=F_LO, power=.05)


@pytest.mark.parametrize('fs,centre,complex_input,centres', [
    (4000, 900, False, [897.5, 829.5, 975.0, 1066.5]),
    (4000, -24, True, [-26.5, -94.5, 51.0, 142.5]),
    (6000, 900, False, [897.5, 829.5, 975.0, 738.5]),
    (6000, -24, True, [-26.5, -94.5, 51.0, -185.5]),
    (6000, 2240, False, [2237.5, 2169.5, 2315.0, 2078.5]),
    (8000, 900, False, [897.5, 829.5, 975.0, 738.5]),
    (8000, -24, True, [-26.5, -94.5, 51.0, -185.5]),
    (8000, 2240, False, [2237.5, 2169.5, 2315.0, 2078.5]),
    (12001.135, 900, False, [897.5, 829.5, 975.0, 738.5]),
    (12001.135, -24, True, [-26.5, -94.5, 50.5, -185.5]),
    (12001.135, 2240, False, [2237.5, 2169.5, 2315.0, 2078.5]),
    (48000, 900, False, [897.5, 829.5, 975.0, 738.5]),
    (48000, -24, True, [-26.5, -94.5, 51.0, -185.5]),
    (48000, 2240, False, [2237.5, 2169.5, 2315.0, 2078.5]),
])
@pytest.mark.parametrize('keyed', [True, False])
def test_tone_search_baseband_conversion_preserves_candidates(fs, centre, complex_input, centres, keyed):
    from scipy.signal import hilbert
    from hfmodem.lanner import rtty

    prototype_centre = 900 if complex_input else centre
    samples = rtty.modulate('THE QUICK BROWN FOX 12345\n', fs=fs,
                            mark_hz=prototype_centre + 85, space_hz=prototype_centre - 85)
    if complex_input:
        # The one-second windows have almost equal in-band energy. A complex64
        # Hilbert transform can change which wins across FFT implementations;
        # construct the reference fixture in double precision before mixing.
        samples = hilbert(samples.astype(np.float64)) * np.exp(
            2j * np.pi * (centre - 900) * np.arange(len(samples)) / fs)
    pairs = fsk.tone_pairs(samples, fs, shift=170, near=centre, span=265,
                           tol=1, max_s=3, keyed=keyed)
    assert [hz for hz, _ in pairs] == (centres[:1] if keyed else centres)
    assert pairs[0][1] > 30


def _sample(soft, *, baud, start, fs=fsk.FS):
    """The soft values at symbol centres start + k * fs / baud, rounded."""
    soft = np.asarray(soft, dtype=np.float64)
    period = fs / baud
    k = np.arange(int((soft.size - 1 - start) / period) + 1)
    idx = np.round(start + k * period).astype(np.int64)
    return soft[idx[(idx >= 0) & (idx < soft.size)]]


def _pipeline(units, baud):
    audio = fsk.modulate(units, baud=baud, f_hi=F_HI, f_lo=F_LO)
    soft = fsk.discriminate(audio, f_hi=F_HI, f_lo=F_LO, baud=baud)
    start = fsk.symbol_phase(soft, baud=baud)
    return (_sample(soft, baud=baud, start=start) > 0)


def test_round_trip_at_100_bd():
    rng = np.random.default_rng(1)
    units = rng.random(400) > 0.5
    assert np.array_equal(_pipeline(units, 100.0), units)


def test_round_trip_at_45_45_bd_does_not_drift():
    rng = np.random.default_rng(2)
    units = rng.random(2000) > 0.5
    assert np.array_equal(_pipeline(units, 45.45), units)


def test_discriminate_is_amplitude_independent():
    rng = np.random.default_rng(3)
    units = rng.random(200) > 0.5
    audio = fsk.modulate(units, baud=100.0, f_hi=F_HI, f_lo=F_LO)
    a = fsk.discriminate(audio, f_hi=F_HI, f_lo=F_LO, baud=100.0)
    b = fsk.discriminate(audio * 0.01, f_hi=F_HI, f_lo=F_LO, baud=100.0)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_continuous_phase():
    rng = np.random.default_rng(4)
    units = rng.random(200) > 0.5
    audio = fsk.modulate(units, baud=100.0, f_hi=F_HI, f_lo=F_LO)
    n = audio.size
    h = np.ones(n)
    h[1:n // 2] *= 2
    h[n // 2 + 1:] = 0
    analytic = np.fft.ifft(np.fft.fft(audio.astype(np.float64)) * h)
    inst_freq = np.diff(np.unwrap(np.angle(analytic))) * fsk.FS / (2 * np.pi)
    assert np.abs(np.diff(inst_freq))[5:-5].max() <= (F_HI - F_LO) * 1.5


def _noise(rng, n, signal, snr_db):
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / fsk.FS) > 3000] = 0
    noise = np.fft.irfft(spec, n)
    return noise * np.sqrt(np.mean(signal ** 2) / np.mean(noise ** 2) / 10 ** (snr_db / 10))


def test_track_follows_a_drifting_clock():
    rng = np.random.default_rng(21)
    units = rng.random(60000) > 0.5
    audio = fsk.modulate(units, baud=100.0, f_hi=F_HI, f_lo=F_LO, fs=fsk.FS * (1 + 40e-6))
    soft = fsk.discriminate(audio, f_hi=F_HI, f_lo=F_LO, baud=100.0)
    got = (soft > 0)[np.rint(fsk.track(soft, baud=100.0)).astype(np.int64)]
    assert got.size == units.size
    assert np.array_equal(got, units)
    solo = _sample(soft, baud=100.0, start=fsk.symbol_phase(soft, baud=100.0)) > 0
    assert not np.array_equal(solo[:units.size], units)


@pytest.mark.slow
def test_track_survives_a_deep_fade():
    rng = np.random.default_rng(22)
    units = rng.random(60000) > 0.5
    audio = fsk.modulate(units, baud=100.0, f_hi=F_HI, f_lo=F_LO, fs=fsk.FS * (1 + 40e-6)).astype(np.float64)
    noise = _noise(rng, audio.size, audio, 15)
    mid = audio.size // 2
    edge = round(1.5 * fsk.FS / 2)
    audio[mid - edge:mid + edge] *= 0.01
    soft = fsk.discriminate(audio + noise, f_hi=F_HI, f_lo=F_LO, baud=100.0)
    centres = fsk.track(soft, baud=100.0)
    got = (soft > 0)[np.rint(centres).astype(np.int64)]
    assert got.size == units.size
    errors = np.flatnonzero(got != units)
    assert errors.size > 0
    boundaries = np.rint(np.arange(units.size + 1) * fsk.FS * (1 + 40e-6) / 100)
    expected = (boundaries[:-1] + boundaries[1:]) / 2
    assert np.max(np.abs(centres - expected)) < 0.2 * fsk.FS / 100
    assert np.all((expected[errors] >= mid - edge - fsk.FS / 100)
                  & (expected[errors] < mid + edge + fsk.FS / 100))


def test_track_bounds_memory():
    import tracemalloc

    # Allocate the input before tracing: only track's extra memory counts.
    pulse = np.maximum(0, 1 - np.abs(np.arange(120) - 60) / 60)
    soft = np.tile(pulse, 20 * 60 * 100)
    tracemalloc.start()
    try:
        centres = fsk.track(soft, baud=100.0, fs=12000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 50_000_000
    np.testing.assert_array_equal(centres, 60 + np.arange(120000) * 120)


def test_tone_pair_iq_and_real_reject_a_stronger_single_carrier():
    fs = 4000
    rng = np.random.default_rng(71)
    for real, midpoint in ((False, -24.0), (True, 900.0)):
        t = np.arange(round(4.3 * fs)) / fs
        pair = (np.exp(2j * np.pi * (midpoint - 85) * t)
                + np.exp(2j * np.pi * (midpoint + 85) * t))
        carrier = 4 * np.exp(2j * np.pi * (midpoint + 240) * t)
        noise = rng.normal(size=t.size) + 1j * rng.normal(size=t.size)
        signal = pair + carrier + 0.2 * noise
        if real:
            signal, noise = signal.real, noise.real
        got, signal_db = fsk.tone_pair(signal, fs, shift=170, near=midpoint)
        _, noise_db = fsk.tone_pair(noise, fs, shift=170, near=midpoint)
        assert abs(got - midpoint) <= 1
        assert noise_db < 15
        assert signal_db > noise_db + 20


def test_tone_pair_frequency_is_independent_of_recording_length():
    fs, midpoint = 4000, 37.4
    found = []
    for seconds in (2.3, 5.7):
        t = np.arange(round(seconds * fs)) / fs
        z = (np.exp(2j * np.pi * (midpoint - 85) * t)
             + np.exp(2j * np.pi * (midpoint + 85) * t))
        found.append(fsk.tone_pair(z, fs, shift=170)[0])
    assert all(abs(got - midpoint) <= 1 for got in found)
    assert abs(found[0] - found[1]) <= 1


def test_tone_pair_selects_loudest_in_band_window_and_caps_estimate():
    fs = 2000
    t = np.arange(2 * fs) / fs
    pair = np.exp(2j * np.pi * -65 * t) + np.exp(2j * np.pi * 105 * t)
    # The earlier window has far more total energy, all outside the search band.
    outside = 20 * np.exp(2j * np.pi * 800 * t)
    z = np.concatenate((outside, pair))
    got, db = fsk.tone_pair(z, fs, shift=170, max_s=2)
    assert abs(got - 20) <= 1
    assert db > 40


def test_moving_average_centres_windows_and_extends_edges():
    x = np.array([0, 2, 4, 8, 16], dtype=float)
    np.testing.assert_allclose(fsk.moving_average(x, 3), [2, 2, 14 / 3, 28 / 3, 28 / 3])
    np.testing.assert_allclose(fsk.moving_average(x, 2), [1, 3, 6, 12, 12])
    np.testing.assert_array_equal(fsk.moving_average(x, 1), x)
    np.testing.assert_allclose(fsk.moving_average(x * (1 + 2j), 5), np.full(5, 6 + 12j))


@pytest.mark.parametrize("size", [0, 1, 59, 6001, 18000, 24000])
def test_silent_track_retains_elapsed_symbols(size):
    with np.errstate(all="raise"):
        centres = fsk.track(np.zeros(size), baud=100, fs=6000)
    np.testing.assert_array_equal(centres, np.arange(0, size, 60, dtype=float))


def test_track_interpolates_across_zero_quality_windows():
    pulse = np.maximum(0, 1 - np.abs(np.arange(60) - 30) / 30)
    soft = np.tile(pulse, 900)
    soft[3 * 6000:6 * 6000] = 0
    with np.errstate(all="raise"):
        centres = fsk.track(soft, baud=100, fs=6000)
    np.testing.assert_array_equal(centres, 30 + np.arange(900) * 60)


@pytest.mark.parametrize("fs,baud", [(6000, 100), (6001.3, 100), (6000, 45.45)])
def test_short_track_bounds_every_rounded_centre(fs, baud):
    # An end-of-symbol peak makes the acquired phase exceed a partial tail.
    period = fs / baud
    for size in [1, round(period) - 1, round(fs) + 1, round(1.8 * fs)]:
        soft = np.zeros(size)
        peaks = np.rint(period - 1 + np.arange(size // int(period)) * period).astype(int)
        soft[peaks[peaks < size]] = 1
        centres = fsk.track(soft, baud=baud, fs=fs)
        indices = np.rint(centres).astype(int)
        assert np.all((indices >= 0) & (indices < size))
        assert np.isfinite(centres).all()


@pytest.mark.parametrize("baud", [45.45, 100])
def test_keyed_pair_accepts_one_transmitter(baud):
    units = np.random.default_rng(41).random(300) > 0.5
    samples = fsk.modulate(units, baud=baud, f_hi=1585, f_lo=1415, fs=6000)
    midpoint, ratio = fsk.tone_pair(samples, 6000, shift=170, near=1500,
                                    span=265, tol=1, keyed=True)
    assert midpoint == pytest.approx(1500, abs=10)
    assert ratio > 15


def test_keyed_pair_rejects_independent_tones_and_wrong_shift_sidebands():
    from hfmodem.lanner import rtty

    samples = rtty.modulate("WRONG SHIFT\n", mark_hz=1712.5, space_hz=1287.5, fs=6000)
    _, ratio = fsk.tone_pair(samples, 6000, shift=170, near=1500,
                            span=265, tol=1, keyed=True)
    assert ratio < 15
    one = rtty.modulate("ONE 12\n" * 4, mark_hz=1385, space_hz=1215, fs=6000)
    two = rtty.modulate("TWO 34\n" * 4, mark_hz=1725, space_hz=1555, fs=6000)
    mixed = np.zeros(max(len(one), len(two)))
    mixed[:len(one)] += one
    mixed[:len(two)] += two
    assert not fsk.keyed_pair(mixed, 6000, f_hi=1555, f_lo=1385)


@pytest.mark.parametrize("fs,baud", [(6000, 100.0), (6001.135, 100.0), (6000, 45.45)])
def test_track_centres_follow_modulated_symbol_boundaries(fs, baud):
    units = np.random.default_rng(62).random(round(5.7 * baud)) > 0.5
    audio = fsk.modulate(units, baud=baud, f_hi=F_HI, f_lo=F_LO, fs=fs)
    soft = fsk.discriminate(audio, f_hi=F_HI, f_lo=F_LO, baud=baud, fs=fs)
    centres = fsk.track(soft, baud=baud, fs=fs)
    boundaries = np.rint(np.arange(units.size + 1) * fs / baud)
    expected = (boundaries[:-1] + boundaries[1:]) / 2
    assert centres.size == units.size
    np.testing.assert_allclose(centres, expected, atol=0.2 * fs / baud, rtol=0)
    np.testing.assert_array_equal(soft[np.rint(centres).astype(int)] > 0, units)


@pytest.mark.parametrize("size", range(30))
def test_keyed_pair_short_input_is_not_keyed(size):
    assert not fsk.keyed_pair(np.zeros(size), 6000, f_hi=1585, f_lo=1415)


@pytest.mark.parametrize("rate", [6000, 48000])
def test_keyed_pair_accepts_readable_rtty_at_zero_db_snr(rate):
    from hfmodem.lanner import rtty

    text = "RYRYRY THE QUICK BROWN FOX 0123456789\n"
    audio = rtty.modulate(text, mark_hz=1585, space_hz=1415, fs=rate).astype(float)
    spectrum = np.fft.rfft(np.random.default_rng(7).standard_normal(audio.size))
    spectrum[np.fft.rfftfreq(audio.size, 1 / rate) > 3000] = 0
    noise = np.fft.irfft(spectrum, audio.size)
    noise *= np.sqrt(np.mean(audio**2) / np.mean(noise**2))
    noisy = audio + noise
    assert rtty.demodulate(noisy, mark_hz=1585, space_hz=1415, fs=rate) == text
    assert fsk.keyed_pair(noisy, rate, f_hi=1585, f_lo=1415)
    midpoint, ratio = fsk.tone_pair(noisy, rate, shift=170, near=1500,
                                    span=265, tol=1, keyed=True)
    assert midpoint == pytest.approx(1500, abs=10)
    assert ratio > 15
    assert rtty.demodulate(noisy, mark_hz=midpoint + 85, space_hz=midpoint - 85,
                           fs=rate) == text


@pytest.mark.parametrize('gain', [.2, .3])
def test_multi_pair_search_enumerates_keyed_pairs_and_respects_its_bound(gain):
    from hfmodem.lanner import rtty

    one = rtty.modulate('ONE 12\n', mark_hz=1385, space_hz=1215, fs=6000)
    two = rtty.modulate('TWO 34\n', mark_hz=1645, space_hz=1475, fs=6000)
    samples = np.zeros(max(len(one), len(two)))
    samples[:len(one)] += gain * one
    samples[:len(two)] += .8 * two
    pairs = fsk.tone_pairs(samples, 6000, shift=170, near=1500, span=265,
                           tol=1, keyed=True, baud=45.45)
    assert sorted(mid for mid, _ in pairs) == pytest.approx([1300, 1560], abs=10)
    assert all(ratio > 15 for _, ratio in pairs)
    strongest = fsk.tone_pairs(samples, 6000, shift=170, near=1500, span=265,
                               tol=1, keyed=True, baud=45.45, limit=1)
    assert len(strongest) == 1
    assert strongest[0][0] == pytest.approx(1560, abs=10)


def test_multi_pair_search_rejects_keyed_sidebands_of_a_single_transmitter():
    from hfmodem.lanner import rtty

    samples = rtty.modulate('FIRST\n', fs=6000)
    pairs = fsk.tone_pairs(samples, 6000, shift=170, near=1500, span=265,
                           tol=1, keyed=True)
    assert len(pairs) == 1
    assert pairs[0][0] == pytest.approx(1500, abs=10)


@pytest.mark.parametrize("midpoint", [3100, 4000, 5200])
def test_keyed_validation_preserves_high_frequency_rtty(midpoint):
    from hfmodem.lanner import rtty

    audio = rtty.modulate("KEYED PAIR TEST 12345", mark_hz=midpoint + 85,
                          space_hz=midpoint - 85, fs=48000)
    assert fsk.keyed_pair(audio, 48000, f_hi=midpoint + 85, f_lo=midpoint - 85)
    centre, ratio = fsk.tone_pair(audio, 48000, shift=170, near=midpoint,
                                  span=30, tol=1, keyed=True)
    assert centre == pytest.approx(midpoint, abs=10)
    assert ratio > 15
    pairs = fsk.tone_pairs(audio, 48000, shift=170, near=midpoint,
                           span=30, tol=1, keyed=True)
    assert pairs[0][0] == pytest.approx(midpoint, abs=10)
    assert rtty.demodulate(audio, mark_hz=centre + 85, space_hz=centre - 85) == "KEYED PAIR TEST 12345"



def test_framing_rescues_interfered_rtty_with_real_stop_evidence():
    from hfmodem.lanner import rtty

    rate = 6000
    samples = np.zeros(6 * rate)
    for midpoint, text in [(1380, 'SECOND 9876543210\n' * 3),
                           (1500, 'THIRD 0123456789\n' * 3),
                           (1740, 'FOURTH 1357924680\n' * 3),
                           (1620, 'REPLACEMENT 2468\n' * 3)]:
        sent = rtty.modulate(text, mark_hz=midpoint + 85, space_hz=midpoint - 85, fs=rate)
        samples += .25 * sent[:len(samples)]
    cut = samples[3 * rate:]
    assert fsk.framed_pair(cut, rate, f_hi=1705, f_lo=1535)
    assert fsk.keyed_pair(cut, rate, f_hi=1705, f_lo=1535, baud=rtty.BAUD)
    assert fsk.fit_pair(cut, rate, midpoint=1626) == pytest.approx(1620, abs=2)


def test_framing_counts_positive_stops_under_a_constant_space_carrier():
    from hfmodem.lanner import rtty

    rate = 6000
    audio = rtty.modulate('RYRYRYRYRYRY', fs=rate)
    t = np.arange(len(audio)) / rate
    audio += .4 * np.sin(2 * np.pi * rtty.SPACE_HZ * t)
    # The interferer reduces mark confidence without destroying the stop's
    # sign. Carrier classification uses the sign's one-half binomial null.
    assert fsk.framed_pair(audio, rate, f_hi=rtty.MARK_HZ, f_lo=rtty.SPACE_HZ)


@pytest.mark.parametrize('complex_input', [False, True])
def test_discriminator_preserves_the_quadrature_of_a_translated_pair(complex_input):
    from scipy.signal import hilbert

    units = np.random.default_rng(87).random(200) > .5
    audio = fsk.modulate(units, baud=100, f_hi=1785, f_lo=1615, fs=6000)
    if complex_input:
        audio = hilbert(audio) * np.exp(-2j * np.pi * 1700 * np.arange(len(audio)) / 6000)
    soft = fsk.discriminate(audio, baud=100, f_hi=85 if complex_input else 1785,
                            f_lo=-85 if complex_input else 1615, fs=6000)
    np.testing.assert_array_equal(soft[30::60] > 0, units)


@pytest.mark.parametrize('baud,shift', [(50, 425), (75, 850), (75, 170)])
def test_acquisition_reads_other_rates_when_told_them(baud, shift):
    from hfmodem.lanner import rtty

    rate = 6000
    mid = 1800
    keyed = rtty.modulate('THE QUICK BROWN FOX 0123456789\n' * 2, baud=baud,
                          mark_hz=mid + shift / 2, space_hz=mid - shift / 2, fs=rate)[:3 * rate]
    assert fsk.framed_pair(keyed, rate, f_hi=mid + shift / 2, f_lo=mid - shift / 2, baud=baud)
    if baud == 75:
        assert not fsk.framed_pair(keyed, rate, f_hi=mid + shift / 2, f_lo=mid - shift / 2)
    idle = .5 * np.sin(2 * np.pi * (mid + shift / 2) * np.arange(rate // 3) / rate)
    assert fsk.mark_idle(idle, rate, mark_hz=mid + shift / 2, space_hz=mid - shift / 2,
                         power=.01, baud=baud)
