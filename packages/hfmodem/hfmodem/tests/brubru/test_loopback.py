# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The in-memory medium retains samples through scheduler delays."""
import importlib.util
from pathlib import Path
import time

import numpy as np
import pytest

from hfmodem.core.audio import AudioError, BurstInvalidated

AIR = Path(__file__).resolve().parents[5] / 'tools/compatibility/pat_rpr_air.py'
if not AIR.exists():
    pytest.skip(f'{AIR} is not in this tree: the simulated medium ships from the repo, '
                'not the package', allow_module_level=True)


@pytest.fixture
def audio():
    spec = importlib.util.spec_from_file_location('rpr_loopback_fixture', AIR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.Audio()
    try:
        yield result
    finally:
        result._stream = None
        result.close()


def test_scheduler_delay_retains_every_simulated_output_sample(audio):
    waveform = np.linspace(-.5, .5, 960, dtype=np.float32)
    start = audio.reserve_burst(waveform, 0, 48000)
    # A delayed Python fill has not lost samples to an independently running DAC.
    audio._callback_at = time.monotonic() - .05
    audio.check_callbacks()
    chunks = []
    while audio.samples < start + len(waveform):
        out = np.zeros((480, 1), np.float32)
        audio._fill(out, 480, audio.samples)
        chunks.append(out[:, 0].copy())
        audio.samples += 480
    np.testing.assert_array_equal(np.concatenate(chunks)[start:start+len(waveform)], waveform)
    audio.wait_until_dac(start+len(waveform)-1, healthy=True)


def test_stopped_simulated_clock_still_hits_transmit_deadline(audio):
    audio._callback_at = time.monotonic() - .05
    with pytest.raises(BurstInvalidated, match='audio wait deadline expired'):
        audio.wait_until_dac(48000, healthy=True, deadline=time.monotonic()-1)


@pytest.mark.parametrize('failure', [False, True])
def test_simulated_audio_health_failure_is_not_ignored(audio, failure):
    if failure:
        audio.failure = RuntimeError('injected medium failure')
    else:
        audio._stream.active = False
    with pytest.raises(AudioError):
        audio.check_callbacks()
