# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-only owners retain levels and decoding under every policy."""
import importlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.tests.station.recording_stage4_support import (
    assert_idle, flush_all,
)
from hfmodem.tests.station.recording_support import pcm


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_console_recorder_policy_native_clock_and_hourly_segments(stage4, monkeypatch, policy):
    module = importlib.import_module('consolelib.monitor')
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(module, 'FS', 8)
    recorder = module.Recorder(stage4.root)
    try:
        for first in range(19, 51, 4):
            recorder(first, (pcm(first, first + 4) * 32768).astype('<i2').tobytes())
            flush_all(stage4)
    finally:
        recorder.close()
    session = assert_idle(stage4, policy, 32)
    lane = next(iter(session.lanes.values()))
    assert lane.rate == 8 and lane.origin == 19 and lane.position == 51
    if policy == 'everything':
        assert len(session.index['segments']) == 4
        assert lane.row['gaps'] == []


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_live_monitor_keeps_decode_levels_and_continuous_lane(stage4, monkeypatch, capsys, policy):
    from hfmodem.besra import monitor, radio
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(monitor, 'CARD_RATE_HZ', 8)
    monkeypatch.setattr(monitor.time, 'time', stage4.clock)
    decoded = []

    class Rolling:
        def __init__(self, demod, report):
            self.pending = []
            self.funnel = NS(frames=0)

        def push(self, block):
            self.pending.append(block.copy())

        def pump(self):
            data = np.concatenate(self.pending) if self.pending else np.empty(0, np.float32)
            self.pending.clear()
            decoded.extend(data)
            self.funnel.frames += bool(len(data))
            return data

        def flush(self):
            pass

    monkeypatch.setattr(radio, 'RollingDecoder', Rolling)
    monitor = monitor.LiveMonitor(NS(), stage4.root / 'monitor.wav', source='synthetic')
    try:
        for _ in range(4):
            monitor.push(np.array([.25, -.25, .5, -.5], np.float32))
            monitor.pump()
            flush_all(stage4)
    finally:
        monitor.close()
    assert len(decoded) == monitor.level.n == 16
    assert monitor.level.peak == .5
    assert monitor.level.rms == pytest.approx(np.sqrt(.15625))
    assert 'peak' in capsys.readouterr().out
    assert_idle(stage4, policy, 16)


def test_live_monitor_records_raw_blocks_before_decoder_backlog(stage4, monkeypatch):
    from hfmodem.besra import monitor
    from hfmodem.tests.station.recording_support import flush
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything')
    monkeypatch.setattr(monitor, 'CARD_RATE_HZ', 8)
    owner = monitor.LiveMonitor(monitor.Demodulator(), stage4.root / 'monitor.wav')
    decoded = []
    pending = []
    monkeypatch.setattr(owner._rolling, 'push', lambda block: pending.append(block))
    def pump():
        audio = np.concatenate(pending) if pending else np.empty(0, np.float32)
        decoded.extend(audio)
        pending.clear()
        return audio
    monkeypatch.setattr(owner._rolling, 'pump', pump)
    monkeypatch.setattr(owner._rolling, 'flush', lambda: None)
    for n in range(6):
        owner.push(pcm(4*n, 4*n+4))
        flush(owner.session)
    owner.pump()
    owner.close()
    assert len(decoded) == owner.level.n == 24
    assert owner.lane.gaps == []
    assert sum(d['samples'] for d in owner.session.index['segments']) == 24
