# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The head of a PACTOR-3 control signal, ours against SCS's.

`witness-v23-40` measured our control reaching the air 36.6 ms later in the
peer's cycle than an SCS station's does, and read 26 dB of that off the first
10 ms of the rendered file: "the phase-reference symbol, slot 0, is rendered
26 dB below the body, where an SCS emission is at full body power in that
symbol."

The first 10 ms of the file is not the phase-reference symbol. `modulate_tones`
convolves the symbol train with a 31-tap matched pulse resampled to 480
samples/symbol, so the output carries about 1.9 symbols of the pulse's own
leading tail in front of symbol 0, and `pulse_lead` is exactly how far in that
is -- 17.75 ms for `historical_control_signal`, 14.10 ms for `control_signal`.
Measured from the phase reference instead of from the first sample, the
reference symbol is at body in both renderers and so is SCS's, and the two
leading edges cross 50 % of body within about 2 ms of each other.

So there is nothing to fix in the renderer, and this file is the guard on that:
the SCS template it is measured against is in the tree
(`fixtures/cs6-waveform/scs-cs6.wav`, a crop of `PIII_Complete_1`), so the
comparison can be re-run rather than re-argued. What actually puts our energy
14-18 ms late is `--p3-control-placement audio-start`, which lands the file's
FIRST SAMPLE on the reply boundary and therefore the phase reference a
`pulse_lead` past it; `pulse-center` is the same waveform placed the way
pactor3.md §7 specifies, and profile B already flies it.

The witness convention throughout: channel 5 and channel 12 power through 140 Hz
Hann filters, 2 ms smoothing, normalised to each burst's own body.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from hfmodem.shrike import onair, placement, rx, spec

FIXTURE = Path(__file__).parent / "fixtures/cs6-waveform/scs-cs6.wav"
#: The crop's own rate, and the -4.5 Hz the recording sits off nominal.
SCS_FS, SCS_OFFSET_HZ = 8000, 4.5
#: Where CS6's phase reference sits in the crop: 17.42 s of `PIII_Complete_1`
#: against the crop's 17.15 s start, and the reader agrees -- `decode_control_
#: signal` returns CS6 at zero bit errors for starts of 264.2 to 273.2 ms.
SCS_REF_S = 0.2666

RENDERERS = (("historical", placement.historical_control_signal),
             ("control_signal", placement.control_signal))


def _body_power(x: np.ndarray, fs: int, tones: tuple[float, ...]) -> np.ndarray:
    n = int(round(fs * 4 / 140.0)) | 1
    h = signal.firwin(n, 70.0, fs=fs, window="hann")
    total = np.zeros(len(x))
    for hz in tones:
        bb = signal.fftconvolve(
            x * np.exp(-2j * np.pi * hz * np.arange(len(x)) / fs), h, mode="same")
        total += np.abs(bb) ** 2
    k = max(1, round(0.002 * fs))
    return np.convolve(total, np.ones(k) / k, mode="same")


def _edge(x: np.ndarray, fs: int, tones: tuple[float, ...],
          ref: int) -> tuple[float, float]:
    """(50 % onset in ms from the phase reference, reference symbol dB re body)."""
    p = _body_power(x, fs, tones)
    body = float(np.median(p[ref + round(0.040 * fs):ref + round(0.190 * fs)]))
    lo = max(0, ref - round(0.040 * fs))
    rising = np.flatnonzero(p[lo:ref + round(0.020 * fs)] >= 0.5 * body)
    onset = (lo + int(rising[0]) - ref) / fs * 1e3
    slot = 10 * np.log10(float(np.mean(p[ref:ref + round(0.010 * fs)])) / body)
    return onset, slot


def _scs() -> tuple[float, float]:
    rate, obs = wavfile.read(FIXTURE)
    assert rate == SCS_FS
    return _edge(np.asarray(obs, float), rate,
                 (1080 - SCS_OFFSET_HZ, 1920 - SCS_OFFSET_HZ),
                 round(SCS_REF_S * rate))


def _ours(render, cs: int) -> tuple[float, float]:
    raw = np.asarray(render(cs))
    # What `_tx` keys, not what the renderer returns: the 2 % trim takes the
    # bottom of the ramp off before the burst reaches the card.
    return _edge(onair._trim_silence(raw), onair.FS, (1080.0, 1920.0),
                 placement.pulse_lead(raw))


@pytest.mark.skipif(not FIXTURE.exists(), reason="SCS CS6 recording absent")
def test_the_scs_reference_symbol_is_at_body_and_its_edge_crosses_just_before_it():
    onset, slot = _scs()
    assert slot == pytest.approx(0.0, abs=1.5)
    assert -8.0 < onset < 0.0


@pytest.mark.skipif(not FIXTURE.exists(), reason="SCS CS6 recording absent")
def test_our_control_rises_where_an_scs_control_rises():
    scs_onset, scs_slot = _scs()
    for name, render in RENDERERS:
        for cs in range(len(spec.CONTROL_SIGNALS)):
            onset, slot = _ours(render, cs)
            assert abs(onset - scs_onset) < 4.0, f"{name} CS{cs + 1} onset {onset}"
            assert abs(slot - scs_slot) < 4.0, f"{name} CS{cs + 1} slot {slot}"


def test_the_reference_symbol_is_never_the_26_db_slot_the_witness_read():
    # That slot is the pulse's leading tail, and `pulse_lead` is its whole
    # width. Reading the file's first 10 ms as the phase-reference symbol is
    # what charged the placement's 14-18 ms to the waveform.
    for name, render in RENDERERS:
        raw = np.asarray(render(0))
        keyed = onair._trim_silence(raw)
        lead = placement.pulse_lead(raw)
        assert lead > round(0.010 * onair.FS), name
        p = _body_power(keyed, onair.FS, (1080.0, 1920.0))
        body = float(np.median(p[lead + 1920:lead + 9120]))
        head = 10 * np.log10(float(np.mean(p[:480])) / body)
        assert head < -10.0, f"{name} head {head}"
        assert _ours(render, 0)[1] > -4.0, name


def test_the_rendered_control_reads_back_at_zero_bit_errors():
    # The read-back reference is the project's own: `decode_control_signal` takes
    # the phase-reference sample, which is the sample the placement aims by.
    for name, render in RENDERERS:
        for cs in range(len(spec.CONTROL_SIGNALS)):
            raw = np.asarray(render(cs))
            start = placement.pulse_lead(raw)
            got = rx.decode_control_signal(onair._trim_silence(raw), start)
            assert got == (cs, 0), f"{name} CS{cs + 1} read back {got}"


def test_the_measured_renderer_reads_back_through_its_carrier_swap():
    for cs in range(len(spec.CONTROL_SIGNALS)):
        for swapped in (False, True):
            raw = np.asarray(placement.control_signal(cs, swapped=swapped))
            start = placement.control_pulse_lead(cs, swapped=swapped)
            assert rx.decode_control_signal(
                onair._trim_silence(raw), start) == (cs, 0)
