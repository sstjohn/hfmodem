# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.tests import evidence
from hfmodem.tests.kestrel.test_bw500_pairing import _build, _receiver


@pytest.mark.parametrize('on,off', [(38.124481, 43.524857), (48.872281, 54.264766),
                                   (59.614009, 65.004737)])
def test_recorded_weak_l1_retry_completes_with_session_ownership(on, off):
    root = evidence.require_input(evidence.WORKING / 'varac-on-air/pat-gate/cnl-b2a0.6-2')
    record = json.loads(next(root.glob('*-fetch.json')).read_text())
    fs, audio = wavfile.read(evidence.require_input(record['cables']['b2a']), mmap=True)
    assert fs == rx.FS
    x = audio[int((on - .4) * fs):int((off + .5) * fs)].astype(float)
    if audio.dtype == np.int16:
        x /= 32768
    expected = bytes.fromhex('04f95cde64ecbcfdafb901bf')
    assert not rx.burst_spans(x)
    hs, _ = _receiver('initiator', 4, run=6)
    for read in (phy._decode_base(x, '500', expect_pair=hs._peer_pairs_500),
                 phy.decode_level(x, '500', 1, expect_pair=hs._peer_pairs_500),
                 phy.decode_level(x, '500', 6, expect_pair=hs._peer_pairs_500)):
        assert read is not None and read.frames == (expected[:-2],)
    assert hs._peer_data_over_500(x, len(x) / fs) == [expected[:-2]]
    assert phy._decode_base(x, '500', expect_pair=True) is None
    assert phy._decode_base(x, '500') is None


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('kind', ['single', 'pair', 'truncated', 'close'])
@pytest.mark.parametrize('seed', range(3))
def test_batch_and_random_callbacks_obey_stock_pair_ownership(level, kind, seed):
    hs, _ = _receiver('responder', level, run=0 if kind == 'single' else 6)
    frames = [_build(level, 0x82 if kind == 'close' else 0x95)]
    if kind in ('pair', 'truncated'):
        frames.append(_build(level, value=73))
    x = tx.synth_burst(frames, onset=12000, level=level)
    if kind == 'truncated':
        rec = rx.INDEX_RECORDS.get(level)
        slot = rec.ncols * rec.dw if rec else rx.NSYM * rx.H
        x = x[:-slot // 2]
    else:
        x = np.pad(x, (0, 12000))
    batch = rx.decode_stream(x, expect_pair=hs._peer_pairs_500)
    receiver, overs = locator.Receiver(), []
    rng = np.random.default_rng(seed)
    at = 0
    while at < len(x):
        stop = at + int(rng.integers(1, 5001))
        overs.extend(receiver.feed(x[at:stop], may_pair=lambda lv, _: hs._peer_pairs_500(lv)))
        at = stop
    overs.extend(receiver.feed(np.empty(0), may_pair=lambda lv, _: hs._peer_pairs_500(lv), eof=True))
    assert len(overs) == 1
    stream = overs[0].result
    assert batch.complete == stream.complete == (kind != 'truncated')
    assert [f.frame_bytes for f in batch.frames] == [f.frame_bytes for f in stream.frames]
    if kind != 'truncated':
        assert [f.frame_bytes for f in batch.frames] == frames
    else:
        assert batch.frames[0].crc_ok
        assert phy._decode_base(x, '500', expect_pair=hs._peer_pairs_500) is None


@pytest.mark.parametrize('expect_pair', [False, True])
def test_noise_never_reaches_phy_as_a_completed_over(expect_pair):
    for seed in range(3):
        x = np.random.default_rng(seed).normal(size=6 * rx.FS)
        assert phy._decode_base(x, '500', expect_pair=expect_pair) is None


def test_predicted_pair_cannot_use_the_unknown_owner_silent_tail_fallback(monkeypatch):
    frame = _build(4)
    x = np.pad(tx.synth_burst(frame, onset=24000), (0, 24000))
    monkeypatch.setattr(rx, 'burst_spans', lambda _: [])
    assert rx.decode_stream(x).complete
    assert rx.decode_stream(x, expect_pair=False).complete
    predicted = rx.decode_stream(x, expect_pair=True)
    assert predicted.frames[0].crc_ok and predicted.frames[0].frames_in_burst == 2
    assert not predicted.observation_complete and not predicted.complete


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_noise_merged_span_cannot_override_known_single_ownership(monkeypatch, level):
    frames = [_build(level)]
    x = np.pad(tx.synth_burst(frames, onset=12000, level=level), (0, 6 * rx.FS))
    monkeypatch.setattr(rx, 'burst_spans', lambda _: [(12000, len(x))])
    result = rx.decode_stream(x, expect_pair=False)
    assert result.complete
    assert [f.frame_bytes for f in result.frames] == frames
