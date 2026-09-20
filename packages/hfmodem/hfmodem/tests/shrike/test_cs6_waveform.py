# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Controls against actual SCS modulation, including independent negative controls."""
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from hfmodem.shrike import modem, onair, p3frame, p3rx, placement, rx

FIXTURE = Path(__file__).parent / 'fixtures/cs6-waveform/scs-cs6.wav'
FS = 8000


def _tone(x, hz):
    return signal.fftconvolve(
        x * np.exp(-2j * np.pi * hz * np.arange(len(x)) / FS),
        signal.firwin(161, 180, fs=FS), mode='same')


def _fit(raw):
    rate, observed = wavfile.read(FIXTURE)
    assert rate == FS
    x = onair._trim_silence(raw)
    generated = signal.resample_poly(np.pad(x, (960, 960)), 1, 6)
    fits = []
    for hz in (1080, 1920):
        obs, model = _tone(observed, hz - 4.5), _tone(generated, hz)
        corr = signal.correlate(obs, model, mode='valid')
        energy = signal.convolve(abs(obs)**2, np.ones(len(model)), mode='valid')
        fits.append(abs(corr)**2 / (np.maximum(energy, 1e-20) * sum(abs(model)**2)))
    return float(np.sqrt(sum(fits) / 2).max())


def _old_control(*, stagger=False, runout=False):
    # Literal independently read CS6 word; reproduce the missing-shape hypotheses.
    bits = np.array([(0x4B4AD >> i) & 1 for i in range(20)], np.uint8)
    symbols = modem.differential_encode(bits, 1)
    if runout:
        symbols = np.r_[symbols, symbols[-1]]
    return modem.modulate_tones(
        {5: symbols, 12: symbols}, modem.ModConfig(matched_pulse=True),
        delay={5: 240} if stagger else None)


@pytest.mark.skipif(not FIXTURE.exists(), reason='independent SCS CS6 recording absent')
def test_control_reproduces_scs_stagger_and_runout():
    # The recorded CS6 has channel 12 leading. Use the measured physical order,
    # not its command number, to request that arrangement.
    actual = _fit(placement.control_signal(5, swapped=True))
    no_stagger = _fit(_old_control(runout=True))
    no_runout = _fit(_old_control(stagger=True))
    assert actual > .89
    assert actual > no_stagger + .10
    assert actual > no_runout + .015


@pytest.mark.skipif(not FIXTURE.exists(), reason='independent SCS CS6 recording absent')
def test_control_flag_shapes_land_in_the_measured_fit_bands():
    """`--p3-control-tail` and `--p3-control-stagger`, graded on real modulation.

    Bands from 27 real control bursts across three emitters. The recorded CS6
    has channel 12 leading, so `swapped=True` is its arrangement and `False` is
    the full-symbol error the wrong starting foot would key -- the one case that
    is worse than keying both tones on one clock.
    """
    fits = {(tail, swapped): _fit(placement.control_burst(5, tail=tail,
                                                          swapped=swapped))
            for tail in (False, True) for swapped in (None, False, True)}
    assert .741 <= fits[False, None] <= .763
    assert .760 <= fits[True, None] <= .784
    assert .866 <= fits[False, True] <= .887
    assert .887 <= fits[True, True] <= .908
    assert fits[True, False] < .65 and fits[False, False] < .65


def test_control_pulse_center_survives_trim_and_carrier_swap():
    for cs in range(6):
        for swapped in (False, True):
            raw = placement.control_signal(cs, swapped=swapped)
            leading = int(np.flatnonzero(abs(raw) > .02 * max(abs(raw)))[0])
            # Physical pulse center, independent of arbitrary renderer padding.
            expected = int(np.argmax(modem.matched_pulse(480))) - leading
            assert placement.control_pulse_lead(cs, swapped=swapped) == expected
            assert 0 < expected < 1000


def test_transmit_header_tracks_sequence_while_carrier_order_alternates():
    # Independent SCS short/long frames show VH bit0 follows sequence parity;
    # the morning retries retain seq1/VH1 while their physical order alternates.
    for sl, long_cycle in ((1, False), (3, False), (3, True)):
        for status in (0x20, 0x21, 0x22, 0x23, 0x20):
            for swapped in (False, True):
                audio = placement.link_packet(sl, b'x', status,
                                             long_cycle=long_cycle, swapped=swapped)
                cfg = placement.protocol_config()
                row0 = (cfg.pulse().size - 1)//2 + p3frame.DATA_OFFSET * 480
                path = (placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS)[sl]
                z = {cn: rx._baseband(audio, cn, 48000, rx._pulse(480))
                     for cn in path.tones}
                head = p3rx.header_of(z, [row0], path)
                assert head is not None
                assert head.vh & 1 == status & 1
                assert head.long_cycle == long_cycle
                assert head.carrier_swapped == swapped
