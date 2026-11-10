# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Missing PCM cannot establish a quieter RF baseline or bridge persistence."""
import numpy as np
import pytest

from hfmodem.shrike import onair, rxfront

FS = rxfront.FS
D_MAX_N = onair._d_max_n(1.25, .030)
SAMPLES = round(.243 * FS)


@pytest.mark.parametrize('kind', ['zero', 'dc', 'gap', 'nan', 'inf'])
def test_invalid_measurement_preserves_floor_and_breaks_persistence(kind):
    rng = np.random.default_rng(7)
    quiet = [rng.normal(0, .02, SAMPLES) for _ in range(5)]
    loud = quiet[0] + rng.normal(0, .10, SAMPLES)
    band = onair._AnswerBand()
    for audio in quiet:
        band.sight(audio, 0, D_MAX_N, read=False, answered=True)
    first = band.sight(loud, 0, D_MAX_N, read=False, answered=True)
    assert first is not None and not first.occupied
    floor, count = band.floor, band.windows
    bad = quiet[0].copy()
    if kind in ('zero', 'dc'):
        bad[:] = 0 if kind == 'zero' else .025
    elif kind == 'gap':
        # Ten ms is two power-estimation frames. This is an exactly flat ADC
        # hold, not a ten-ms RF burst: even the lowest admitted tone (300 Hz)
        # changes over three cycles here. A short burst surrounded by digital
        # zero also cannot tell this estimator what the RF noise floor was.
        bad[round(.100 * FS):round(.110 * FS)] = 0
    else:
        bad[round(.100 * FS)] = np.nan if kind == 'nan' else np.inf
    assert band.sight(bad, 0, D_MAX_N, read=False, answered=True) is None
    assert band.floor == floor and band.windows == count
    assert band.last is None
    after_gap = band.sight(loud, 0, D_MAX_N, read=False, answered=True)
    assert after_gap is not None and not after_gap.occupied
    repeated = band.sight(loud, 0, D_MAX_N, read=False, answered=True)
    assert repeated is not None and repeated.occupied


@pytest.mark.parametrize('scale', [1., 1e-6])
def test_valid_stochastic_input_remains_measurable_at_low_gain(scale):
    audio = np.random.default_rng(4).normal(0, .02 * scale, SAMPLES)
    assert rxfront.quiet_level_db(audio, .055, .220) is not None


def test_digital_silence_cannot_seed_a_fictitious_rf_floor():
    band = onair._AnswerBand()
    assert band.sight(np.zeros(SAMPLES), 0, D_MAX_N,
                      read=False, answered=True) is None
    assert band.floor is None and band.windows == 0


def test_invalid_padding_outside_the_measurement_span_does_not_veto_rf():
    audio = np.random.default_rng(4).normal(0, .02, SAMPLES)
    audio[:round(.04 * FS)] = 0
    assert rxfront.quiet_level_db(audio, .055, .220) is not None
