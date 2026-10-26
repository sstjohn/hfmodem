# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import threading
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.audio import ReplayAudio
from hfmodem.core.regulatory import centred
from hfmodem.core.rig import RigError
from hfmodem.core.virtualrig import VirtualRig
from hfmodem.station import process
from hfmodem.station.arbiter import TxArbiter, TxRequest
from hfmodem.tests.station.test_process import configuration


@pytest.fixture(autouse=True)
def no_hardware_or_claim(monkeypatch):
    def refuse(*a, **kw):
        pytest.fail('replay reached a hardware or claim factory')
    for name in ('Rig', 'YaesuCat', 'RigctldCat', 'RtsPtt', 'StationAudio'):
        monkeypatch.setattr(process, name, refuse)
    monkeypatch.setattr(process.claim, 'acquire', refuse)
    monkeypatch.setattr(process.devices, 'find_device', refuse)


@pytest.fixture
def replay():
    audio = ReplayAudio(np.zeros(48000, np.float32), blocksize=128)
    rig = VirtualRig(audio, centre_hz=7101500)
    rig.arm()
    arb = TxArbiter(audio, rig, drive=1.0)
    yield audio, rig, arb
    arb.close()
    rig.close()


@pytest.mark.parametrize('reserved', [False, True])
def test_finite_burst_keys_on_replay_clock(replay, monkeypatch, reserved):
    audio, rig, arb = replay
    keys = []
    original = rig.key
    def key(*a, **kw):
        token = original(*a, **kw)
        keys.append((token, audio.sample_now()))
        return token
    monkeypatch.setattr(rig, 'key', key)
    request = TxRequest(priority=0, seq=1, protocol='rpr', audio=np.ones(480), rate=48000,
                        emission=centred(7100000, 500), at=1000, settle_s=.01,
                        lead_s=0.01 if reserved else None)
    if reserved:
        result = arb.submit_result(request)
        assert result.outcome == "sent", result
        end = result.end
    else:
        end = arb.submit(request)
    token, clock = keys[0]
    assert token.card_index == clock == (1000 if reserved else 520)
    expected = 1480 if reserved else 1000
    assert audio.transmitted[0][0] == expected
    assert end == expected + 480 == audio.samples
    assert rig.released_low and not rig.keyed and arb.sent == 1


def test_streaming_lease_completes_and_stays_keyed_until_end(replay):
    audio, rig, arb = replay
    audio.wait_until(321)
    with arb.lease(protocol='rpr', emission=centred(7100000, 500),
                   settle_s=.01, capacity=1024) as lease:
        assert lease.key_index == 321
        stream = lease.stream()
        wave = np.arange(480, dtype=np.float32)
        stream.write(wave)
        audio.wait_until(1281)
        assert lease.complete(stream.written) == 1281
        assert rig.keyed
        assert lease.end() == 1281
    assert rig.released_low and not rig.keyed
    np.testing.assert_array_equal(np.concatenate([b for _, b in audio.transmitted]), wave)
    assert audio.transmitted[0][0] == 801


@pytest.mark.parametrize('receive', [False, True])
def test_replay_own_release_reown_never_claims_or_constructs_devices(tmp_path, monkeypatch, receive):
    path = tmp_path / 'replay.wav'
    wavfile.write(path, 48000, np.zeros(1000, np.int16))
    cfg = configuration(cat='yaesu')
    cfg = replace(cfg, audio=replace(cfg.audio, input=f'replay:{path}'))
    def refuse(*a, **kw):
        pytest.fail('replay reached a hardware or claim factory')
    for name in ('Rig', 'YaesuCat', 'RigctldCat', 'RtsPtt', 'StationAudio'):
        monkeypatch.setattr(process, name, refuse)
    monkeypatch.setattr(process.claim, 'acquire', refuse)
    monkeypatch.setattr(process.devices, 'find_device', refuse)
    station = process.Station(cfg)
    monkeypatch.setattr(station, 'preflight_reference', refuse)
    monkeypatch.setattr(station, 'preflight_measure', refuse)
    monkeypatch.setattr(station, '_poll', lambda: station._poll_stop.wait())
    for _ in range(2):
        try:
            assert station.own(receive=receive)['state'] == ('owning-receive' if receive else 'owning')
            assert station.claim is None
            assert isinstance(station.audio, ReplayAudio)
            assert isinstance(station.rig, VirtualRig)
            assert station.rig.cat is station.rig.ptt is None
            assert (station.arbiter is None) == receive
        finally:
            assert station.release()['state'] == 'observing'
        assert station.audio is station.rig is station.claim is None


def test_open_rig_reuses_resolved_replay_before_driver(tmp_path, monkeypatch):
    path = tmp_path / 'replay.wav'
    wavfile.write(path, 48000, np.zeros(100, np.int16))
    cfg = configuration(cat='yaesu')
    cfg = replace(cfg, audio=replace(cfg.audio, input=f'replay:{path}'))
    monkeypatch.setattr(process, 'YaesuCat', lambda *a, **kw: pytest.fail('driver'))
    station = process.Station(cfg)
    try:
        station.open_audio()
        assert isinstance(station.open_rig(), VirtualRig)
        assert isinstance(station.audio, ReplayAudio)
        assert station.claim is None
    finally:
        station.shutdown()


def test_virtual_rig_interlocks_and_replay_clock(replay):
    audio, rig, _ = replay
    rig.close()
    with pytest.raises(RigError, match='arm'):
        rig.key(None, why='test', open=True)
    rig.arm()
    audio.wait_until(123)
    token = rig.key(None, why='test', open=True, sample_now=lambda: 999)
    assert token.card_index == 123 and token.card_deadline is None
    with pytest.raises(RigError, match='already keyed'):
        rig.key(None, why='test', open=True)
    rig.unkey(token)
    rig.transmit = False
    with pytest.raises(RigError, match='disabled'):
        rig.key(None, why='test', open=True)
    rig.transmit = True
    rig.panic_unkey('test retirement')
    assert rig.must_retire and rig.released_low
    with pytest.raises(RigError, match='retired'):
        rig.key(None, why='test', open=True)


@pytest.mark.parametrize('channels', [(7102000,), (7102000, 10145500, 14109000)])
def test_ale_own_receive_tunes_and_releases_replay(monkeypatch, channels):
    from hfmodem.tests.bateleur.ale.test_station import _cfg
    station = process.Station(_cfg(channels=channels), replay=np.zeros(1000))
    monkeypatch.setattr(station, '_poll', lambda: station._poll_stop.wait())
    try:
        assert station.own(receive=True)['state'] == 'owning-receive'
        rig = station.rig
        assert rig.dial() == station.lanes['ale'].dials[0] == channels[0]
        assert station.snapshot()['rig']['source'] == 'virtual'
        assert station.lanes['ale'].transmit(np.ones(100)) is None
        assert station.release()['state'] == 'observing'
        assert rig.dial() == station.cfg.rig.dial_hz
    finally:
        station.shutdown()


def test_station_built_rig_keys_through_lane_and_identification(monkeypatch):
    from hfmodem.core.config import ProtocolConfig
    cfg = configuration(protocols={'bateleur': ProtocolConfig(enabled=True, host='bateleur', port=0)})
    cfg = replace(cfg, station=replace(cfg.station, id_mode='cw'),
                  rig=replace(cfg.rig, centre_hz=7102000))
    station = process.Station(cfg, replay=np.zeros(1000))
    monkeypatch.setattr(station, '_poll', lambda: station._poll_stop.wait())
    try:
        station.own()
        assert station.rig.dial() == cfg.rig.dial_hz
        lane = station.lanes['bateleur']
        assert lane.transmit(np.ones(480), why='replay test') is not None, lane.refusals
        assert station.arbiter.sent == 1 and station.audio.transmitted
        assert station.rig.released_low
        station.arbiter.identity.end()
        assert station.arbiter.identity.due
        station._identify_if_due()
        assert not station.arbiter.identity.due
        assert lane.transmit(np.ones(480)) is not None, lane.refusals
        assert station.arbiter.sent == 3
    finally:
        station.shutdown()


def test_burst_pending_at_tape_end_completes_before_shutdown(monkeypatch):
    from hfmodem.core.config import ProtocolConfig
    cfg = configuration(protocols={'bateleur': ProtocolConfig(enabled=True, host='bateleur', port=0)})
    station = process.Station(cfg, replay=np.zeros(4096))
    armed = threading.Event()
    build_lanes = station.build_lanes

    # Hold the poll back until the burst is armed, so the tape runs out under it.
    def gated_build_lanes():
        build_lanes()
        audio = station.audio
        pump, arm_burst = audio.pump, audio.arm_burst
        def gated_pump(blocks=1):
            armed.wait(10)
            return pump(blocks)
        def signalling_arm_burst(*a, **kw):
            at = arm_burst(*a, **kw)
            armed.set()
            return at
        monkeypatch.setattr(audio, 'pump', gated_pump)
        monkeypatch.setattr(audio, 'arm_burst', signalling_arm_burst)
    monkeypatch.setattr(station, 'build_lanes', gated_build_lanes)
    stopped_with = []
    stop = station._stop.set
    def recording_stop():
        if not stopped_with:
            stopped_with.append(list(station.audio.transmitted))
        stop()
    monkeypatch.setattr(station._stop, 'set', recording_stop)

    results = []
    def send():
        results.append(station.arbiter.submit_result(TxRequest(
            priority=0, seq=1, protocol='bateleur', audio=np.ones(48000), rate=48000,
            emission=centred(station.rig.dial(), 500), at=0, settle_s=.01)))
    try:
        station.own()
        sender = threading.Thread(target=send)
        sender.start()
        assert station._stop.wait(10)
    finally:
        station.shutdown()
    sender.join(10)
    assert results[0].outcome == 'sent', results
    [(at, burst)] = stopped_with[0]
    assert at < 4096 and len(burst) == 48000


def test_virtual_rig_stale_unkey_preserves_current_key(replay):
    _, rig, _ = replay
    old = rig.key(None, why='old', duration_s=1)
    rig.unkey(old)
    current = rig.key(None, why='current', duration_s=1)
    rig.unkey(old)
    assert rig.keyed and rig.line_state == 'high' and not rig.released_low
    rig.unkey(current)
    assert not rig.keyed and rig.released_low


def test_virtual_rig_refuses_retune_while_keyed(replay):
    _, rig, _ = replay
    token = rig.key(None, why='keyed', duration_s=1)
    with pytest.raises(RigError, match='retune while keyed'):
        rig.tune(14101500)
    assert rig.dial() == 7100000
    rig.unkey(token)
    rig.tune(14101500)
    assert rig.dial() == 14100000


@pytest.mark.parametrize('duration,settle,expected', [(1, .1, 3.1), (12, .5, 15.5), (29, 1, 30)])
def test_virtual_key_deadline_includes_shared_slack(replay, duration, settle, expected):
    audio, rig, _ = replay
    audio.wait_until(123)
    token = rig.key(None, why='deadline', duration_s=duration, settle_s=settle)
    assert token.deadline == expected
    assert token.card_deadline == 123 + int(expected * 48000)
