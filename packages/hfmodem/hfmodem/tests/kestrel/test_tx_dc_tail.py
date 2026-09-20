# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The codec's DC settling must not delay the post-TX receive cursor."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from . import corpora
from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK

kc = corpora.harness('kestrel_connect')
FIXTURES = Path(__file__).with_name('fixtures') / 'tx-dc-tail'


@pytest.mark.parametrize('row', json.loads((FIXTURES / 'provenance.json').read_text()))
def test_native_control_tails_do_not_wait_for_the_callback_boundary(row):
    fs, x = wavfile.read(FIXTURES / row['file'])
    assert fs == kc.FS
    x = x.astype(float) / 32768
    marker = kc._tx_end_in_capture(x, row['earliest'])
    assert row['marker_bounds'][0] <= marker <= row['marker_bounds'][1]
    assert marker < row['previous_marker'] - int(.04 * fs)


@pytest.mark.parametrize('gain', [.005, .05, .3])
@pytest.mark.parametrize('delay', [.02, .08, .12])
def test_monitored_end_survives_dc_settling_without_skipping_the_reply(gain, delay):
    fs = kc.FS
    wave = MK.synth_burst('KC9GHZ', VF.CR) * gain
    lead = np.zeros(round(delay * fs))
    end = len(lead) + len(wave)
    # The radio's DC block settles after the last actual audio sample. It is
    # still above the old absolute mute threshold when this callback arrives.
    tail = gain * np.exp(-np.arange(round(.06 * fs)) / (fs * .025))
    x = np.concatenate((lead, wave, tail))
    marker = kc._tx_end_in_capture(x, len(wave))
    assert end <= marker <= end + round(.006 * fs)
    # An answer opening 110 ms after our waveform remains ahead of the cursor.
    cursor = marker + round((kc.TX_IDLE_HOLD_S + kc.RX_ECHO_GUARD_S) * fs)
    assert cursor < end + round(.110 * fs)


def test_incomplete_capture_and_a_single_quiet_frame_do_not_move_cursor_back():
    fs = kc.FS
    wave = .1 * np.sin(2 * np.pi * 1500 * np.arange(fs) / fs)
    wave[-96:] = 0  # Only 2 ms, less than the required sustained end.
    assert kc._tx_ac_end(wave, len(wave) - 4800) is None
    assert kc._tx_ac_end(wave, len(wave) + 4096) is None
    assert kc._tx_ac_end(np.zeros(fs), fs // 2) is None


def test_keyup_mute_does_not_win_over_the_end_of_transmitted_audio():
    fs = kc.FS
    tone = .1 * np.sin(2 * np.pi * 1500 * np.arange(fs) / fs)
    x = np.concatenate((tone, np.zeros(2400), tone, np.zeros(960)))
    end = len(x) - 960
    assert end <= kc._tx_ac_end(x, fs - 960) <= end + 96
