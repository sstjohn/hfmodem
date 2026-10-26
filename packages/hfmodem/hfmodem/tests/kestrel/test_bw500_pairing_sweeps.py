# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.tests.kestrel.test_bw500_pairing import _build, _feed, _receiver


def _noise(signal, power, snr, seed):
    noise = np.random.default_rng(seed).standard_normal(len(signal))
    return signal + noise * np.sqrt(power / (np.mean(noise ** 2) * 2500 / 24000)
                                    / 10 ** (snr / 10))


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('snr', [10, 0])
@pytest.mark.parametrize('depth', [0, 6, 10, 15, 25])
@pytest.mark.parametrize('seed', range(3))
def test_faded_first_frame_cannot_ack_half_a_pair(level, snr, depth, seed):
    frames = [_build(level, value=v) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=24000, level=level)
    end = len(signal)
    rec = rx.INDEX_RECORDS.get(level)
    split = 24000 + ((rec.lead + rec.ncols) * rec.dw if rec else (9 + rx.NSYM) * rx.H)
    power = np.mean(signal[24000:72000] ** 2)
    signal[24000:split] *= 10 ** (-depth / 20)
    x = _noise(np.pad(signal, (0, 2 * rx.FS)), power, snr, seed)
    hs, io = _receiver('responder' if seed % 2 else 'initiator', level, 6)
    _feed(hs, io, x)
    assert all(k >= end for k in io.keyed_at)
    assert not io.host or io.host == [bytes([v]) * rx._PAYLOAD[level] for v in (31, 73)]
    if depth == 0:
        assert len(io.host) == 2


@pytest.mark.parametrize('level', [1, 2, 3])
@pytest.mark.parametrize('snr', range(-15, -7))
def test_threshold_pair_cannot_complete_as_single(level, snr):
    frames = [_build(level, value=v) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=24000, level=level)
    end = len(signal)
    power = np.mean(signal[24000:72000] ** 2)
    signal = np.pad(signal, (0, 4 * rx.FS))
    for seed in range(100):
        x = _noise(signal, power, snr, seed)
        receiver = locator.Receiver()
        found = []
        for j in range(0, len(x), locator.BLOCK):
            found.extend(receiver.feed(x[j:j + locator.BLOCK], may_pair=True))
        found.extend(receiver.feed(np.empty(0), may_pair=True, eof=True))
        for over in found:
            if any(f.crc_ok for f in over.frames):
                assert over.decision >= end, (seed, over)
                assert not over.result.complete or [f.frame_bytes for f in over.frames] == frames
        if seed < 3:
            hs, io = _receiver('responder' if seed % 2 else 'initiator', level, 6)
            _feed(hs, io, x)
            assert all(k >= end for k in io.keyed_at), seed
            if io.host and len(io.host) < 2:
                assert hs._pair_loss_500 and hs._owed_block, seed
