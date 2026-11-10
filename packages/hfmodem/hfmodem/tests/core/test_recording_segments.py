# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Native-rate capacities and optional unhashed owned segment handoff."""
import hashlib
import wave

import numpy as np
import pytest

from hfmodem.core import capture_identity as ci, wav


@pytest.mark.parametrize('rate', [8000, 12000, 44100, 48000, 96000])
def test_segment_capacity_is_one_native_hour(tmp_path, rate):
    out = wav.SegmentedWriter(tmp_path / 'receive.wav', rate=rate, closed=lambda p: None)
    try:
        assert out.segment_samples == wav.SEGMENT_SAMPLES * rate // 48000
    finally:
        out.close()


@pytest.mark.parametrize('rate', [8, 80, 800])
def test_scaled_test_capacity_rotates_exact_pcm(tmp_path, monkeypatch, rate):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    capacity = 60 * rate
    pcm = (np.arange(capacity + 3) % 30000).astype('<i2')
    closed = []
    out = wav.SegmentedWriter(tmp_path / 'receive.wav', rate=rate, closed=closed.append)
    try:
        assert out.segment_samples == capacity
        out.writeframes(pcm.tobytes())
    finally:
        out.close()
    assert len(closed) == 2
    chunks = []
    for path, count in zip(closed, [capacity, 3], strict=True):
        with wave.open(str(path)) as src:
            assert (src.getframerate(), src.getnframes()) == (rate, count)
            chunks.append(np.frombuffer(src.readframes(count), '<i2'))
    np.testing.assert_array_equal(np.concatenate(chunks), pcm)


@pytest.mark.parametrize('defer', [False, True])
def test_owned_handoff_can_defer_hashing(tmp_path, monkeypatch, defer):
    capture = ci.allocate({'capture': tmp_path / 'capture'}, {'modem': 'capture'})
    owner = ci.Recording.reserve(capture.claim('modem'), 'receive.wav',
                                 clock_id='rx', continuous=True)
    hashed, closed = [], []
    digest = wav._sha256
    monkeypatch.setattr(wav, '_sha256', lambda p: hashed.append(p) or digest(p))
    kwargs = {'defer_finalize': True} if defer else {}
    out = wav.SegmentedWriter(owner.base, rate=48000, ownership=owner,
                              closed=closed.append, **kwargs)
    out.writeframes(np.array([1, -2, 3], '<i2').tobytes())
    out.close()
    assert closed == out.paths
    side = ci.audio_metadata(closed[0])
    if defer:
        assert hashed == []
        assert side['capture_state'] == 'storing'
        assert not side.get('sha256')
    else:
        assert hashed == closed
        assert side['capture_state'] == 'finalized'
        assert side['sha256'] == hashlib.sha256(closed[0].read_bytes()).hexdigest()
