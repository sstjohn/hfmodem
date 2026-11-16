# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import sys
from collections import Counter

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import live, onair, pactor1, rxfront
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.tests import evidence


RESTRICTED = dict(p3_headers=False, p3_packets=False, p2_markers=False,
                  p3_envelope=False, p1_marker_veto=True)


def _parsed(monkeypatch, *flags):
    found = []
    monkeypatch.setattr(sys, 'argv', ['shrike.onair', '--mycall', 'W9SSJ', *flags])
    monkeypatch.setattr(onair, 'run', lambda args: found.append(args) or 0)
    onair.main()
    return found[0]


@pytest.mark.parametrize('flags,restricted,p1_only', [
    ([], False, False), (['--pactor1-only'], False, True),
    (['--decode-pactor1-only'], True, True),
    (['--decode-pactor1-only', '--pactor1-only'], True, True),
])
def test_flag_defaults_and_implication(monkeypatch, flags, restricted, p1_only):
    args = _parsed(monkeypatch, *flags)
    assert args.decode_pactor1_only is restricted
    assert args.pactor1_only is p1_only


@pytest.mark.parametrize('flags', [
    ['--pactor2-only'], ['--pactor3-only'], ['--offer-pactor2'],
    ['--p3-uninvited'], ['--decline-grant'], ['--p1-grant-only'],
    ['--p2-mail'], ['--p3-mail'], ['--p4-entry'],
    ['--p3-timing-trial', 'A'], ['--p3-entry-timing-trial', 'A'],
    ['--connect', 'robust', '--after-robust', 'p2'],
    ['--connect', 'robust', '--after-robust', 'p3'],
])
def test_incompatible_policies(monkeypatch, flags):
    with pytest.raises(SystemExit):
        args = _parsed(monkeypatch, '--decode-pactor1-only', *flags)
        onair._timing_trial_defaults(args)


@pytest.mark.parametrize('restricted', [False, True])
@pytest.mark.parametrize('entry', ['push', 'flush'])
def test_rolling_scope_and_event_timing(monkeypatch, restricted, entry):
    calls, delivered, upgrades = [], [], []
    enabled = [False]
    event = rxfront.Event(.1, 'cs', 'CS4', protocol='PACTOR-1', cs=3)

    def decode(audio, **kwargs):
        calls.append(kwargs)
        yield event

    def receive(ev):
        delivered.append(ev)
        enabled[0] = True

    def upgrade(*args, **kwargs):
        upgrades.append(kwargs)
        return iter(())

    monkeypatch.setattr(rxfront, 'decode_events', decode)
    monkeypatch.setattr(rxfront, 'p3_packet_events', upgrade)
    receiver = live.RollingRx(receive, window_s=1., keep_s=.5,
                              p3_packets=lambda: enabled[0],
                              decode_pactor1_only=restricted)
    receiver.t0 = 2.
    audio = np.zeros(rxfront.FS // 2, np.float32)
    if entry == 'push':
        receiver.push(audio)
    else:
        receiver.hold(audio)
        assert not calls
        receiver.flush()
    assert len(calls) == 1
    for name in ('p3_headers', 'p2_markers'):
        assert calls[0][name] is not restricted
    assert calls[0]['p3_packets'] is False
    assert calls[0]['p3_envelope'] is False
    assert calls[0]['p1_marker_veto'] is restricted
    assert len(upgrades) == int(not restricted)
    receiver._decode(audio)
    assert len(delivered) == 1
    assert delivered[0].t == pytest.approx(2.1)


@pytest.mark.parametrize('restricted', [False, True])
@pytest.mark.parametrize('protocol', [onair.Protocol.PACTOR1, onair.Protocol.PACTOR2,
                                      onair.Protocol.PACTOR3])
def test_session_reader_entries(monkeypatch, restricted, protocol):
    host = onair.PtcHost(mycall='W9SSJ')
    host.decode_pactor1_only = restricted
    host.arq.state = onair.State.CONNECTED
    host.protocol = protocol
    host.arq.p4_admitted = True
    receiver = onair._SessionRx(host)
    receiver.p4.expected = True
    calls = Counter()

    def spy(name, result=None):
        def read(*args, **kwargs):
            calls[name] += 1
            return result
        return read

    for name in ('_p1_cs', '_p2_cs', '_p3_cs', '_p1_packet', '_p2_packet', '_p3_packet'):
        monkeypatch.setattr(receiver, name, spy(name))
    monkeypatch.setattr(receiver.p4, 'read_window', spy('p4', []))
    audio = np.zeros(rxfront.FS, np.float32)
    receiver._p4_read(audio, 0, whole=True)
    receiver.control_signal(audio, 0, 0)
    onair._scan_frame(receiver, audio, 0)
    receiver.p4.expected = False
    receiver.p4.admitted = False
    host.arq.p4_admitted = False
    receiver.control_signal(audio, 0, 0)
    receiver.deep_scan(audio)
    receiver.upgrade_scan(audio)
    assert calls['p4'] == (0 if restricted else 3)
    if restricted:
        assert calls['_p1_cs'] == 2
        assert calls['_p1_packet'] == 2
        assert not any(calls[name] for name in ('_p2_cs', '_p3_cs', '_p2_packet', '_p3_packet'))
    else:
        assert calls['_p2_packet'] and calls['_p3_packet'] and calls['_p1_packet']
        if protocol != onair.Protocol.PACTOR1:
            assert calls['_p2_cs' if protocol == onair.Protocol.PACTOR2 else '_p3_cs']


@pytest.mark.parametrize('flags,restricted', [
    ([], False), (['--decode-pactor1-only'], True),
])
def test_called_replay_admission_and_readers(monkeypatch, tmp_path, capsys, flags, restricted):
    path = tmp_path / 'caller.wav'
    audio = np.concatenate([pactor1.connect_signal('W9SSJ'),
                            np.zeros(12 * rxfront.FS, np.float32)])
    wavfile.write(path, rxfront.FS, audio.astype(np.float32))
    calls, receivers, decode_flags = Counter(), [], []
    original_init = onair._SessionRx.__init__
    original_decode = rxfront.decode_events

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        receivers.append(self)

    def decode(audio, **kwargs):
        decode_flags.append(kwargs)
        yield from original_decode(audio, **kwargs)

    def spy(name, result=None):
        def read(*args, **kwargs):
            calls[name] += 1
            return result
        return read

    monkeypatch.setattr(onair._SessionRx, '__init__', init)
    for name in ('_p2_packet', '_p3_packet', '_p2_cs', '_p3_cs'):
        monkeypatch.setattr(onair._SessionRx, name, spy(name))
    monkeypatch.setattr(P4Receiver, 'read_window', spy('p4', []))
    monkeypatch.setattr(onair, '_p4_hold_word', spy('p4_hold'))
    monkeypatch.setattr(rxfront, 'decode_events', decode)
    monkeypatch.setattr(sys, 'argv', [
        'shrike.onair', '--mycall', 'W9SSJ', '--answer', '3', '--hold', '2',
        '--max-cycles', '1', '--replay', str(path), '--outdir', str(tmp_path / 'out'),
        *flags])
    assert onair.main() == 0
    assert 'CALLED:' in capsys.readouterr().out
    assert len(receivers) == 1
    receiver = receivers[0]
    assert receiver.host.arq.p4_admitted is not restricted
    assert receiver.p4.admitted is not restricted
    assert receiver.host.stay_in_pactor1 is bool(flags)
    assert decode_flags
    assert all(kw['p3_headers'] is not restricted and kw['p2_markers'] is not restricted
               for kw in decode_flags)
    if restricted:
        assert not calls
        assert all(not kw['p3_packets'] and not kw['p3_envelope'] for kw in decode_flags)
    else:
        assert calls['p4'] and calls['p4_hold']
        assert calls['_p2_packet'] and calls['_p3_packet']


@pytest.mark.parametrize('filename', [
    'pos_pactor1_local.wav', 'pos_p1cs_ws8eoc.wav', 'pos_p1_twosided_14110.wav',
    'pos_p2_sl3_hb9ak.wav', 'watch_pactor3_maryland.wav', 'neg_noise_chatham.wav',
])
def test_recorded_p1_decisions_identical(filename):
    path = evidence.require_input(evidence.CORPUS / 'regress' / 'fixtures' / filename)
    audio = rxfront.load_wav(path)

    def p1(events):
        return [ev for ev in events if ev.protocol == 'PACTOR-1'
                or ev.kind in ('connect', 'callb', 'fsk', 'p1reply', 'unassigned')]

    ordinary = p1(rxfront.decode_events(audio, p3_envelope=False))
    restricted = list(rxfront.decode_events(audio, **RESTRICTED))
    assert p1(restricted) == ordinary
    assert all(ev.protocol not in ('PACTOR-2', 'PACTOR-3', 'PACTOR-4') for ev in restricted)
    if filename.startswith('pos_p1') or filename == 'pos_pactor1_local.wav':
        assert ordinary


def test_marker_veto_survives_without_p2_reporting(monkeypatch):
    from hfmodem.shrike import p1rx, pactor2

    monkeypatch.setattr(rxfront, '_p2_markers', lambda audio: [
        rxfront._P2Frame(0., 2., pactor2.PATHS[0], .99)])
    monkeypatch.setattr(rxfront, '_fsk_present', lambda sp: True)
    monkeypatch.setattr(rxfront, '_p1_cs_bursts', lambda audio: ())
    monkeypatch.setattr(rxfront, 'call_b_frames', lambda *args: ())
    monkeypatch.setattr(p1rx, 'decode_connect', lambda *args, **kwargs: None)
    monkeypatch.setattr(rxfront, '_p2_present',
                        lambda *args: pytest.fail('P2 shape reader invoked'))
    audio = np.zeros(rxfront.FS // 2, np.float32)
    ordinary = list(rxfront.decode_events(audio, p3_headers=False, p3_packets=False))
    assert [ev.protocol for ev in ordinary] == ['PACTOR-2']
    assert list(rxfront.decode_events(audio, **RESTRICTED)) == []
    without_veto = dict(RESTRICTED, p1_marker_veto=False)
    assert [ev.kind for ev in rxfront.decode_events(audio, **without_veto)] == ['fsk']


@pytest.mark.parametrize('restricted', [False, True])
def test_regrid_p4_hold_scope(monkeypatch, tmp_path, restricted):
    from hfmodem.tests.shrike.test_p4_regrid_handoff_0929 import recovery_scene

    stream, raster, tx, host, receiver, audio, _ = recovery_scene(tmp_path, monkeypatch, None)
    host.decode_pactor1_only = restricted
    calls = []
    retain = onair._p4_retain_hold

    def retained(*args, **kwargs):
        calls.append('retain')
        return retain(*args, **kwargs)

    def read(*args, **kwargs):
        calls.append('read')
        return None

    monkeypatch.setattr(onair, '_p4_retain_hold', retained)
    monkeypatch.setattr(onair, '_p4_read_hold', read)
    onair._regrid(stream, raster, tx, host, receiver, 1, audio, 0, round(.04 * onair.FS))
    assert bool(calls) is not restricted
