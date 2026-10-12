# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, p4robusttx, p4turn, pactor1
from hfmodem.tests.shrike.test_p1_robust_0926 import FS, PACKET_N, SLOT_N

FX = Path(__file__).parent / 'fixtures/p4-breakin'


def peer_wave(kind):
    if kind == 'bare':
        return p4robusttx.robust_word('Y27'), 0
    meta = json.loads((FX / 've3kpg-audio.json').read_text())
    wave = onair.session.load_wav(str(FX / 've3kpg-audio.wav'), FS)
    first = round(meta['first_chip'] * FS / meta['rate'])
    if kind == 'bad':
        wave = wave.copy()
        for at in range(first + round(.30 * FS), len(wave), round(.05 * FS)):
            wave[at:at + round(.025 * FS)] *= -1
    return wave, first


def run_arm(tmp_path, monkeypatch, route, kind, link=True, recording=None):
    wave, first = peer_wave(kind)
    audio = np.zeros(24 * SLOT_N, np.float32)
    requests = 3 if route == 'hold' else 10 if route == 'stall-goodbye' else 6
    for k in range(requests + 1):
        cs = onair._trim_silence(np.asarray(
            pactor1.control_signal(pactor1.CS_ACK_A, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(.090 * FS)
        audio[at:at + len(cs)] += cs
    y = (requests + 1 + (3 if route != 'hold' else 0)) * SLOT_N + round(1.02 * FS)
    for at in (y, y + SLOT_N):
        start = at - first
        audio[start:start + len(wave)] += wave
    wav = recording or tmp_path / 'peer.wav'
    if recording is None:
        onair.session.write_wav(str(wav), audio)
    keyed, machines, receivers = [], [], []
    key, drive, machine_type = onair.RadioTx._tx, onair._run_p4_link, p4turn.P4Turn

    def tx(self, audio, what, *args, **kwargs):
        before = len(self.slots_used)
        key(self, audio, what, *args, **kwargs)
        if len(self.slots_used) > before:
            keyed.append((self.tx_audio_start, what))

    def machine(*args, **kwargs):
        result = machine_type(*args, **kwargs)
        result.decisions = []
        decide = result.decide
        def timed(now):
            result.decisions.append(now)
            return decide(now)
        result.decide = timed
        machines.append(result)
        return result

    def handoff(tx, live, raster):
        receiver = tx.sessrx.p4
        receivers.append((receiver, list(receiver.observations), live.pos))
        tx.settle = .074
        result = drive(tx, live, raster)
        assert tx.sessrx.p4 is receiver
        return result

    monkeypatch.setattr(onair.RadioTx, '_tx', tx)
    monkeypatch.setattr(onair, '_run_p4_link', handoff)
    monkeypatch.setattr(p4turn, 'P4Turn', machine)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', [
        'shrike.onair', '--replay', str(wav), '--max-cycles', '30' if recording else '3',
        '--mycall', 'W9SSJ', '--dxcall', 'VE3KPG', '--dial', '7101500',
        '--outdir', str(tmp_path / 'out'), '--after-connect', 'p1-robust',
        '--p1-status-bits45', '2', '--p1-robust-stall', '12' if route == 'hold' else '4',
        '--p1-robust-stall-then', 'silent' if route == 'stall' else 'goodbye',
        '--p4-entry-timeout', '25',
        *(['--p4-link', '--p4-data', 'none', '--p4-link-timeout', '3'] if link else [])])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        onair.main()
    return y, keyed, machines, receivers, output.getvalue()


@pytest.mark.parametrize('route', ['hold', 'stall', 'stall-goodbye'])
@pytest.mark.parametrize('kind', ['bare', 'good', 'bad'])
def test_peer_breakin_reaches_link_before_a_p1_key(tmp_path, monkeypatch, route, kind):
    y, keyed, machines, receivers, log = run_arm(tmp_path, monkeypatch, route, kind)
    assert machines, log
    assert all(at < y for at, what in keyed if what.startswith('P1 ')), keyed
    machine, = machines
    receiver, observations, handed_at = receivers[0]
    assert observations and handed_at < y + round(.846 * FS)
    assert any(o.word == 'Y27' for o in observations)
    events = [e for e in machine.events if e.observation.word == 'Y27']
    assert any(e.observation.provisional for e in events)
    assert sum(e.outcome.value == 'new' and abs(e.observation.stream_sample-y) < FS//100
               for e in events) == 1
    assert abs(machine.decisions[0] - (y + round(.846 * FS))) <= round(.002 * FS)
    assert any('200Bd' in what for _, what in keyed)
    p4 = [(at, what) for at, what in keyed if what.startswith('P4 LINK')]
    if kind == 'bad':
        assert p4 == [], keyed
        assert machine.unread == 2
    else:
        assert p4, keyed
        assert abs(p4[0][0] - (y + round(.920 * FS))) <= round(.002 * FS)
        if kind == 'bare':
            assert machine.grants >= 1
            assert machine.last.kind.value == "body" and machine.last.status == 0x40
        else:
            assert 'word X27' in p4[0][1] and machine.grants == 0
            assert machine.last.kind.value == 'word' and machine.last.word == 'X27'
        if kind == 'good':
            assert any(e.observation.frame is not None for e in machine.events)


@pytest.mark.parametrize('route', ['hold', 'stall'])
def test_without_link_keeps_the_existing_hold_and_listen(tmp_path, monkeypatch, route):
    y, keyed, machines, receivers, log = run_arm(tmp_path, monkeypatch, route, 'good', False)
    assert not machines and not receivers
    assert not any(what.startswith('P4 LINK') for _, what in keyed)
    if route == 'hold':
        assert any(at >= y for at, what in keyed if what.startswith('P1 '))
    else:
        assert 'capture ended during the post-stall listen' in log


def test_local_0927_breakin_leaves_the_p1_hold(monkeypatch):
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    recording = Path(__file__).resolve().parents[5] / 'captures/onair-0927-1003/stream.flac'
    if not recording.exists():
        pytest.skip('local September 27 flight capture absent')
    audio = onair.session.load_wav(str(recording), FS)
    live = Stream(audio)
    live.pos = round(32.46 * FS)
    tx = Tx(live)
    tx.slot, tx.tx_end = 27, round(32.16 * FS)
    tx.p4_probe.emissions.clear()
    tx.p4_probe.requested = False
    host = tx.p4_link_host
    host.p4_after_first_ack = True
    tx.sessrx.host = host
    receiver = tx.sessrx.p4
    raster = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N,
                                cs_n=12000, d_max_n=4800)
    start = round(32.16 * FS)
    heard = onair._p4_hold_word(tx, host, live, raster, audio[start:live.pos], start, start)
    if heard is None:
        assert tx.p4_hold_audio.pending and not tx.slots_used
        live.pos = round(32.56 * FS)
        heard = onair._p4_hold_word(tx, host, live, raster,
                                   audio[start:live.pos], start, start)
    assert heard is not None
    word = next(o for o in receiver.observations if o.word == 'Y27')
    assert abs(word.stream_sample / FS - 32.25) < .03
    tx.breakin_due = tx.listening = True
    drive = onair._run_p4_link
    def handoff(tx, live, raster):
        assert not tx.breakin_due and not tx.listening
        return drive(tx, live, raster)
    monkeypatch.setattr(onair, '_run_p4_link', handoff)
    onair._run_p4_probe(tx, live, raster, heard=heard)
    assert tx.sessrx.p4 is receiver
    assert any(r['phase'] == 'link' and r['word'] == 'Y27' for r in tx.p4_probe.heard)
