# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where our PACTOR-III bursts actually sit in frequency, and how to ask.

The 40 m monitor of 2026-09-15 read our SL1 entry and data carriers 12.5 Hz below
what they were commanded at, by the squared-baseband line that
`dbpsk-offset-needs-the-squared-line` validated; an independent KiwiSDR read the
same render exact. The renders are exact. Squaring cancels the modulation only for
a control signal's {0, pi} steps -- `placement.BIT_PHASE`'s 135/315 diagonals
double to one angle, 270 degrees per symbol, which is a -12.5 Hz line that no
payload moves. See `docs/protocols/pactor/pactor3.md` section 9.
"""
import numpy as np
import pytest
from scipy.signal import fftconvolve, firwin

from hfmodem.shrike import modem, placement, spec

FS = spec.SAMPLE_RATE
SPS = FS // 100
TONES = (5, 12)


def squared_line_hz(audio, carrier_hz, *, search=180.0, pad=1 << 21):
    """The estimator the 40 m analysis used: squared baseband, peak, halved."""
    t = np.arange(len(audio)) / FS
    h = firwin(int(6 * FS / 150.0) | 1, 150.0, fs=FS)
    z = np.convolve(audio * np.exp(-2j * np.pi * carrier_hz * t), h, mode='same')
    y = z * z * np.hanning(len(z))
    mag = np.fft.fftshift(np.abs(np.fft.fft(y, pad)))
    freq = np.fft.fftshift(np.fft.fftfreq(pad, 1 / FS))
    band = (freq > -2 * search) & (freq < 2 * search)
    f, m = freq[band], mag[band]
    k = int(np.argmax(m))
    a, b, c = np.log(m[k - 1]), np.log(m[k]), np.log(m[k + 1])
    return carrier_hz + (f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])) / 2


def demod_hz(audio, carrier_hz, step_hz=0.0):
    """Symbol-timed residual rotation, data removed at `step_hz`'s constellation.

    Unambiguous over +-25 Hz at 100 Bd; a burst further off is seeded.
    """
    t = np.arange(len(audio)) / FS
    y = fftconvolve(audio * np.exp(-2j * np.pi * carrier_hz * t),
                    modem.matched_pulse(SPS), mode='same')
    phase = int(np.argmax([abs(y[k::SPS]).sum() for k in range(SPS)]))
    s = y[phase::SPS]
    lit = np.flatnonzero(abs(s) > .25 * np.median(np.sort(abs(s))[-20:]))
    s = s[lit[0]:lit[-1] + 1]
    d = s[1:] * np.conj(s[:-1]) * np.exp(-1j * step_hz)
    return carrier_hz + np.angle(np.sum(d * np.where(d.real >= 0, 1., -1.))) * 100 / (2 * np.pi)


def _two_carrier(steps, n_sym=81):
    bits = np.random.default_rng(11).integers(0, 2, n_sym)
    syms = np.exp(1j * np.cumsum(np.r_[0.0, [steps[b] for b in bits]]))
    return modem.modulate_tones({cn: syms for cn in TONES},
                                placement.protocol_config())


@pytest.mark.parametrize('burst', ['entry', 'data'])
def test_a_keyed_packet_sits_on_its_commanded_carriers(burst):
    if burst == 'entry':
        audio = placement.link_packet(1, b"", 0x1a, swapped=False,
                                      flush=placement.ENTRY_FLUSH)
    else:
        path = placement.SPEED_PATHS[1]
        audio = placement.data_packet(
            placement.field_info(b"1w9ss", path.crc_bytes - 3, 0x03), path,
            cfg=placement.protocol_config())
    for cn in TONES:
        nominal = spec.channel_freq_hz(cn)
        assert demod_hz(audio, nominal, .75 * np.pi) == pytest.approx(nominal, abs=1.0)


def test_a_keyed_control_sits_on_its_commanded_carriers():
    audio = placement.control_signal(1)
    for cn in TONES:
        nominal = spec.channel_freq_hz(cn)
        assert demod_hz(audio, nominal) == pytest.approx(nominal, abs=1.0)
        assert squared_line_hz(audio, nominal) == pytest.approx(nominal, abs=1.0)


def test_the_squared_line_reads_the_diagonals_as_minus_twelve_and_a_half_hz():
    """The synthetic that settles it: exactly on frequency, read 12.5 Hz low."""
    on_frequency = _two_carrier(placement.BIT_PHASE)
    textbook = _two_carrier({0: 0.0, 1: np.pi})
    for cn in TONES:
        nominal = spec.channel_freq_hz(cn)
        assert demod_hz(on_frequency, nominal, .75 * np.pi) == pytest.approx(nominal, abs=.5)
        assert squared_line_hz(on_frequency, nominal) == pytest.approx(nominal - 12.5, abs=.5)
        assert squared_line_hz(textbook, nominal) == pytest.approx(nominal, abs=.5)
