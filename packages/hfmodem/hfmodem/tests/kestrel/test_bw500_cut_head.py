# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A BW500 over that lost its head to our own changeover, and one that carries a
carrier offset, both decode: the alignment lock is differential and reaches
back over the preamble.

K5FIT's greeting of 2026-09-11 was a strong, clean level-4 frame that the coherent
one-column onset lock could not read (see the fixture's sidecar); it is the
recorded case. The synthetic one is the same failure with nothing else in it:
a rendered frame, +0.12 Hz, 20 dB, and no head cut at all.
"""
from pathlib import Path
import wave

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.tx import varahf500_tx as TX
from hfmodem.kestrel.vara import vara_ofdm as OF

FIXTURE = Path(__file__).with_name('fixtures') / 'k5fit_20260911_cut_head_greeting.wav'
GREETING = b'RMS Trimode 1.4.3.0\r\nW9SSJ has 120 daily mi'
LIVE_EDGE = 12000            # 74.73 s: the rig's return to receive, 0.25 s into the fixture

requires_fixture = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason='the recorded K5FIT greeting requires the source checkout '
           '(fixtures/k5fit_20260911_cut_head_greeting.wav)')


def _fixture():
    with wave.open(str(FIXTURE)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 48000)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float) / 32768


def _greeting(audio):
    res = RX.decode_stream(audio)
    assert res.complete and len(res.data_frames) == 1, [(f.crc_ok, f.self_consistency) for f in res.frames]
    f = res.data_frames[0]
    assert f.crc_ok and f.level == RX.BASE_LEVEL and f.c0 == 9
    assert bytes(f.payload) == GREETING and f.marker == 0x99
    return f


@requires_fixture
def test_recorded_k5fit_greeting_decodes_with_three_preamble_columns_gone():
    f = _greeting(_fixture())
    assert all(o < -2 * RX.H for o in f.onset), f.onset      # column 0 lies before the detected start


@requires_fixture
def test_the_same_greeting_decodes_from_a_bracket_cut_at_the_un_mute():
    _greeting(_fixture()[LIVE_EDGE:])


def test_a_rendered_over_survives_a_tenth_of_a_hertz():
    body = bytes(range(43))
    frame = TX.build_frame(body, 0x80)
    burst = TX.synth_burst(frame, onset=OF.ONSET_500, ncol=OF._NCOL_500)
    x = np.concatenate([np.zeros(6000), burst, np.zeros(6000)])
    t = np.arange(len(x)) / RX.FS
    rng = np.random.default_rng(11)
    x = np.real(hilbert(x) * np.exp(2j * np.pi * 0.12 * t))
    x += rng.standard_normal(len(x)) * np.sqrt((x ** 2).mean() / 100 * 24000 / 700)
    res = RX.decode_stream(x)
    assert res.complete and len(res.data_frames) == 1
    assert bytes(res.data_frames[0].payload) == body
