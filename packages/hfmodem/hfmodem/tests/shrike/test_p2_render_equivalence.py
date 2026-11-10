# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The faster pulse renderer preserves the independently checked waveform."""
import numpy as np
import pytest

from hfmodem.shrike import pactor2


def _dense_render(walks, tones, fs, leads):
    """Prior renderer: explicit zero-stuffed impulses and direct convolution."""
    length = round(fs / pactor2.SYMBOL_RATE)
    pulse = pactor2.tx_pulse(fs)
    n_sym = max(len(w) for w in walks) + 1
    out = np.zeros(n_sym * length + pulse.size - 1 + max(leads))
    for steps, tone, start in zip(walks, tones, leads):
        phases = np.concatenate([[0.0], np.cumsum(steps)])
        train = np.zeros(len(phases) * length, dtype=complex)
        train[::length] = np.exp(1j * phases)
        baseband = np.convolve(train, pulse)
        time = np.arange(baseband.size) + start
        out[start:start + baseband.size] += (
            baseband * np.exp(2j * np.pi * tone * time / fs)).real
    return out / (np.abs(out).max() + 1e-30)


@pytest.mark.parametrize("path", pactor2.PATHS + pactor2.PATHS_LONG,
                         ids=lambda path: path.name)
@pytest.mark.parametrize("swapped", (False, True))
def test_every_speed_and_cycle_keeps_its_samples(path, swapped, monkeypatch):
    info = np.random.default_rng(908).bytes(path.crc_bytes - 2)
    field = pactor2.build_field(info, path)
    actual = pactor2.data_burst(field, path, swapped=swapped)
    monkeypatch.setattr(pactor2, "_render", _dense_render)
    expected = pactor2.data_burst(field, path, swapped=swapped)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
