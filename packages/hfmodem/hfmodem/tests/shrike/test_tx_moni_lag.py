# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

import hashlib
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import pactor1

TOOL = Path(__file__).resolve().parents[5] / 'tools' / 'tx_moni_lag.py'
if not (TOOL).exists():
    pytest.skip('tools/tx_moni_lag.py is not in this tree', allow_module_level=True)
spec = importlib.util.spec_from_file_location('tx_moni_lag', TOOL)
moni = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moni)


def fixture(invert=False, lag=317):
    event = dict(tx_id=1, label='connect->WS8EOC', sample_rate=48000,
                 waveform_samples=46080, output_enqueue_sample=15000,
                 driver_offset_samples=756, ptt_start_sample=13000)
    ref = moni.references(event, 'p1-connect')[int(invert)]
    assert len(ref) == event['waveform_samples']
    a = np.random.default_rng(83).normal(0, .0001, 70000)
    start = event['output_enqueue_sample'] + event['driver_offset_samples'] + lag
    a[start:start + len(ref)] += ref * .5
    return a, event


@pytest.mark.parametrize('invert,lag', [(False, 317), (True, 961), (False, -113)])
def test_known_lag_within_one_sample(invert, lag):
    a, event = fixture(invert, lag)
    result = moni.analyse(a, [event], 'p1-connect')
    assert not result['rejected'], result
    assert len(result['matches']) == 1
    match = result['matches'][0]
    assert abs(match['residual_samples'] - lag) <= 1
    assert match['polarity'] == int(invert)
    assert match['uncertainty_samples'] >= 1
    assert not result['rf_calibration']


def test_missing_fields_are_named_without_inventing_d():
    a, event = fixture()
    del event['output_enqueue_sample']
    del event['driver_offset_samples']
    event['audio_start_sample'] = 16000
    result = moni.analyse(a, [event], 'p1-connect')
    assert not result['matches']
    assert result['median_samples'] is None
    assert result['rejected'][0]['missing_fields'] == [
        'output_enqueue_sample', 'driver_offset_samples']
    assert 'exact sample-clock' in result['rejected'][0]['reason']


def test_quiet_signal_is_rejected():
    a, event = fixture()
    ref = moni.references(event, 'p1-connect')[0]
    start = event['ptt_start_sample'] - 4800
    a[start:start + 4800] = ref[:4800] * .5
    result = moni.analyse(a, [event], 'p1-connect')
    assert not result['matches']
    assert 'quiet comparison' in result['rejected'][0]['reason']


def test_noise_is_rejected():
    _, event = fixture()
    a = np.random.default_rng(52).normal(0, .01, 70000)
    result = moni.analyse(a, [event], 'p1-connect')
    assert not result['matches']
    assert 'low correlation' in result['rejected'][0]['reason']


def write_wav(path, a):
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
        w.writeframes((a * 32767).astype('<i2').tobytes())


def test_cli_exact_render_text_and_json(tmp_path, capsys):
    a, event = fixture()
    ref = moni.references(event, 'p1-connect')[0]
    capture, render = tmp_path / 'stream.wav', tmp_path / 'render.wav'
    write_wav(capture, a)
    write_wav(render, ref)
    events, output = tmp_path / 'arm.log', tmp_path / 'report.json'
    events.write_text('tx_event ' + json.dumps(event) + '\n')
    assert moni.main(['--capture', str(capture), '--events', str(events),
                      '--render', str(render), '--json-out', str(output)]) == 0
    text = capsys.readouterr().out
    assert 'residual=317 samples' in text
    report = json.loads(output.read_text())
    assert report['median_samples'] == 317
    assert report['spread_samples'] == 0
    del event['output_enqueue_sample']
    events.write_text('tx_event ' + json.dumps(event) + '\n')
    assert moni.main(['--capture', str(capture), '--events', str(events),
                      '--render', str(render), '--json-out', str(output)]) == 1
    assert 'output_enqueue_sample' in capsys.readouterr().out
    assert json.loads(output.read_text())['median_samples'] is None


def test_matched_helper_agrees_at_the_peak():
    from hfmodem.tests.shrike.test_txhead import _matched

    a, event = fixture()
    at = event['output_enqueue_sample'] + event['driver_offset_samples']
    ref = moni.references(event, 'p1-connect')[0]
    filtered = moni.in_band(a)
    lag, score = _matched(filtered, at, ref)
    got = moni.match(filtered, at, ref, 2880)
    assert got['observed_sample'] == at + lag
    assert got['score'] == pytest.approx(score, abs=1e-9)


def test_p1_render_requires_known_burst():
    with pytest.raises(ValueError, match='exact trimmed render'):
        moni.references(dict(label='p4 short'), 'p1-connect')


def test_p1_default_render_is_exact():
    from hfmodem.shrike.onair import _trim_silence

    event = dict(label='connect->K4MSU')
    for invert, got in enumerate(moni.references(event, 'p1-connect')):
        np.testing.assert_array_equal(
            got, _trim_silence(pactor1.connect_signal('K4MSU', invert=bool(invert))))


def test_two_unique_copies_are_rejected(tmp_path):
    ref = moni.in_band(np.random.default_rng(51).normal(0, .05, 1000))
    a, event = fixture()
    a[:] = np.random.default_rng(83).normal(0, .0001, len(a))
    expected = event['output_enqueue_sample'] + event['driver_offset_samples']
    for offset in (-1800, 1800):
        a[expected + offset:expected + offset + len(ref)] += ref
    event['waveform_samples'] = len(ref)
    render = tmp_path / 'render.wav'
    write_wav(render, ref)
    report = moni.analyse(a, [event], str(render))
    assert not report['matches']
    assert 'non-unique peak' in report['rejected'][0]['reason']


def test_different_recording_clock_is_rejected(tmp_path, capsys):
    events = tmp_path / 'events.json'
    events.write_text(json.dumps([dict(recording=str(tmp_path / 'other.wav'))]))
    assert moni.main(['--capture', str(tmp_path / 'stream.wav'),
                      '--events', str(events), '--render', 'p1-connect']) == 2
    assert 'event recording differs' in capsys.readouterr().err


def callback_stream(monkeypatch):
    import sys
    from types import SimpleNamespace
    from hfmodem.shrike import onair

    clock = SimpleNamespace(now=0.0)
    outputs = []
    status = SimpleNamespace(input_overflow=False, input_underflow=False,
                             output_underflow=False, priming_output=False)

    class Device:
        latency = (0.0, 0.0)
        blocksize = 128

        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            self.n = 0
            self.offset = 756

        def pump(self):
            out = np.zeros((128, 1), np.float32)
            adc = self.n / 48000
            self.callback(np.zeros_like(out), out, 128,
                          SimpleNamespace(inputBufferAdcTime=adc,
                                          outputBufferDacTime=adc + self.offset / 48000,
                                          currentTime=clock.now), status)
            outputs.append((self.n, out[:, 0].copy()))
            self.n += 128

        def start(self):
            self.pump()

    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(Stream=Device))
    monkeypatch.setattr(onair, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(onair.threading, 'Thread',
                        lambda **kw: SimpleNamespace(start=lambda: None))
    live = onair._LiveInput('fake', 'fake', tx_latency_n=144)
    live._breathing.close()

    class Done:
        def clear(self):
            pass

        def set(self):
            pass

        def wait(self, limit):
            while live._tx is not None:
                live._stream.pump()
            return True

    live._tx_done = Done()
    live._sleep_until = lambda t: setattr(clock, 'now', t)
    live.flush_to = lambda n: setattr(live, 'pos', n)
    live.wait_until = lambda n: 0.0
    return live, clock, outputs


def emission_trace(monkeypatch, bursts):
    live, clock, outputs = callback_stream(monkeypatch)
    trace = []
    for audio, at in bursts:
        keys = []
        first, end = live.transmit(
            audio, at=at, settle=.04,
            key=lambda state: keys.append((state, round(clock.now * 48000))))
        trace.append((live._tx_at, first, end, keys))
    pcm = np.concatenate([a for _, a in outputs])
    return trace, hashlib.sha256(pcm.tobytes()).hexdigest()


def test_recorded_p1_connect_emission_and_ptt_pin(monkeypatch):
    from hfmodem.shrike import session

    path = Path(__file__).with_name('fixtures') / 'n9vc-0925-kiwi-connect.wav'
    audio = session.load_wav(str(path), 48000)[36000:82080].astype(np.float32)
    trace = emission_trace(monkeypatch, [(audio, None), (audio, 120000)])
    assert trace[1] == '994c2d324a8f68b13e152062576a87a6a0f19b6a4c256de013f572d5ce882905'
    assert hashlib.sha256(json.dumps(trace[0]).encode()).hexdigest() == (
        '1d20cea3a5e3a8cf87d60d44616dfed519a664287c1733818e45fb722f8e3d37')


def test_recorded_p4_arm_emission_and_ptt_pin(monkeypatch):
    from hfmodem.tests.shrike import p4_after_handover_helpers as recorded
    from hfmodem.tests.shrike.test_p4_link_0927 import Tx

    bursts = []
    emit = Tx._tx

    def remember(tx, audio, what):
        emit(tx, audio, what)
        bursts.append((audio.copy(), tx.tx_audio_start))

    monkeypatch.setattr(Tx, '_tx', remember)
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / 'captures/onair-0930-2028/p4-probe.json').exists()), None)
    if root is None:
        pytest.skip('local recorded P4 arm 0930-2028 absent')
    monkeypatch.setattr(recorded, 'ROOT', root)
    recorded.replay('2028', end=40.3)
    assert bursts
    trace = emission_trace(monkeypatch, bursts)
    assert trace[1] == 'f2a336f82ef358d31d17ff5a6c5d316625a4214f341508704dfef5ab7f9e4e41'
    assert hashlib.sha256(json.dumps(trace[0]).encode()).hexdigest() == (
        '9587799f9c95038e1a1e98fe323abac2ace42ec6911a0a47df4077ca022da6c1')


def test_enqueued_arm_log_to_synthetic_capture(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from hfmodem.shrike import onair

    live, clock, _ = callback_stream(monkeypatch)
    tx = onair.RadioTx(transmit=True, outdir=tmp_path, settle=.04,
                      rig=SimpleNamespace(ptt=lambda state: None,
                                          key_failure=lambda: None))
    tx.live = live
    events = []
    for offset in (900, 612):
        clock.now = (live._stream.n + 13000) / 48000
        live._stream.offset = offset
        tx._tx(pactor1.connect_signal('WS8EOC'), 'connect->WS8EOC')
        log = capsys.readouterr().out
        event, = [json.loads(line[len('tx_event '):]) for line in log.splitlines()
                  if line.startswith('tx_event ') and '"completed"' in line]
        assert event['output_enqueue_sample'] == live._tx_at
        assert event['driver_offset_samples'] == offset
        assert live._lat == 756
        events.append(event)
        with (tmp_path / 'arm.log').open('a') as f:
            f.write(log)
    capture = np.random.default_rng(83).normal(0, .0001, live._stream.n + 5000)
    for event in events:
        ref = moni.references(event, 'p1-connect')[0]
        start = event['output_enqueue_sample'] + event['driver_offset_samples'] + 317
        capture[start:start + len(ref)] += ref * .5
    write_wav(tmp_path / 'stream.wav', capture)
    output = tmp_path / 'report.json'
    assert moni.main(['--capture', str(tmp_path / 'stream.wav'),
                      '--events', str(tmp_path / 'arm.log'), '--render', 'p1-connect',
                      '--json-out', str(output)]) == 0
    report = json.loads(output.read_text())
    assert not report['rejected']
    assert len(report['matches']) == 2
    assert report['median_samples'] == 317
    assert report['spread_samples'] == 0
