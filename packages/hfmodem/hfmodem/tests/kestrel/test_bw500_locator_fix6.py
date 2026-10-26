# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import firwin, lfilter

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.tests.kestrel.test_bw500_session import _station
from hfmodem.tests.kestrel.test_bw500_terminal_window import _waiting, _partial_result


def _single(level, kind, snr):
    payload = bytes(range(rx._PAYLOAD[level]))
    frame = tx.build_frame(payload, 0x95) if level == 4 else tx.build_l3_frame(payload, 0x95, level)
    signal = tx.synth_burst([frame], onset=24000, level=level)
    end = len(signal)
    signal = np.pad(signal, (0, 12 * rx.FS))
    noise = np.random.default_rng(1).standard_normal(len(signal))
    if kind == 'ssb':
        noise = lfilter(firwin(513, [300, 2700], pass_zero=False, fs=rx.FS), 1, noise)
    power = np.mean(noise ** 2) * (2500 / 24000 if kind == 'white' else 1)
    noise *= np.sqrt(np.mean(signal[24000:72000] ** 2) / power / 10 ** (snr / 10))
    return signal + noise, end, frame, payload


@pytest.mark.parametrize('kind,snr', [('ssb', -3), ('white', 0), ('white', -3), ('white', -6)])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_noisy_level4_nonlast_single_delivers(kind, snr, route):
    x, end, frame, payload = _single(4, kind, snr)
    assert not phy.over_is_last(frame[:-2], 'AAAA1')
    station, io = _station()
    if route == 'stream':
        for j in range(0, len(x), 1024):
            io.fed = j + len(x[j:j + 1024])
            station.on_rx_stream(x[j:j + 1024])
            if io.keys:
                break
    else:
        bracket = x[:end + rx.FS]
        read = phy._decode_base(bracket, '500')
        assert read is not None
        assert read.frames == (frame[:-2],)
        station.on_rx_audio(bracket)
    assert io.host == [payload]
    assert io.keys == 1


@pytest.mark.parametrize('site', ['pending', 'completed', 'control'])
def test_callback_recovers_after_invariant_failure(monkeypatch, site):
    station, io = _station()
    station.on_rx_stream(np.zeros(1024), sample_start=0)
    generation = station._front_generation
    old = station._base_rx
    if site == 'pending':
        candidates = [locator.Candidate(3, 100000 + j * 20000, 24, 1024, 100000 + j * 20000)
                      for j in range(5)]
        monkeypatch.setattr(old.grouper, 'feed', lambda *a, **kw: candidates)
    elif site == 'completed':
        monkeypatch.setattr(old, 'feed', lambda *a, **kw: [object()] * 5)
    else:
        def overflow(*args, **kwargs):
            raise RuntimeError('hard control template registry overflow')
        monkeypatch.setattr(station._control_schedule, 'feed', overflow)
    station.on_rx_stream(np.zeros(1024), sample_start=1024)
    assert station._front_generation > generation
    assert station._base_ready == []
    assert station._base_rx is None
    assert any('overflow' in msg for msg in io.msgs)
    station.on_rx_stream(np.zeros(1024), sample_start=2048)
    assert station._base_rx is not old
    assert station._front_heard == 1024
    assert not io.keys and not io.host


def test_truncated_located_peer_over_reaches_unread_recovery():
    station, io = _waiting()
    result, _ = _partial_result()
    frame = result.frames[0]
    candidate = locator.Candidate(4, frame.start, .7, 1000, frame.start)
    station._base_ready = [locator.Over(candidate, generation=station._front_generation,
        frames=[frame], status='truncated', end=1000, decision=1200)]
    station._front_heard = 1200
    station._base_rx = SimpleNamespace(raw=SimpleNamespace(origin=0, samples=1200, raw=np.zeros(1200)))
    assert station._take_located_500()
    assert station._owed_recovery and station._owed_block
    assert io.payloads == []
    assert station._resend_decoded is None
