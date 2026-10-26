# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pinned CQ capture replay, continuous acquisition and owner-thread host wiring."""
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest
import subprocess
from scipy.io import wavfile

from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.kestrel.rx.cq import build_frame, parse_frame, HEADER, TRAINING
from hfmodem.kestrel.rx.cq_stream import CQDetector
from hfmodem.kestrel.tx.cq import column_bins, synth_cq, synth_frame
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_compression import IO

FIXTURES = Path(__file__).with_name('fixtures')
CASES = json.loads((FIXTURES / 'cq-replay-cases.json').read_text())
BENCH = json.loads((FIXTURES / 'cq-bench-tones.json').read_text())
SOURCE = Path(os.environ.get('HFMODEM_CQ_CORPUS', RECORD))
_HASHES = {}


def recording(case):
    path = require_input(SOURCE / case['source'])
    if path not in _HASHES:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        _HASHES[path] = h.hexdigest()
    assert _HASHES[path] == case['sha256']
    if 'start_sample' in case:
        fs, data = wavfile.read(path, mmap=True)
        assert fs == case['sample_rate']
        x = data[case['start_sample']:case['stop_sample']]
    else:
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1',
            '-ss', str(case['start_s']), '-i', str(path), '-t', str(case['duration_s']),
            '-f', 'f32le', '-acodec', 'pcm_f32le', '-threads', '1', 'pipe:1'])
        data = np.frombuffer(raw, '<f4').reshape(-1, 2)
        x = data[:, 0] + 1j * data[:, 1]
    return x


def replay(detector, x, seed=13):
    rng = np.random.default_rng(seed)
    results = []
    i = 0
    while i < len(x):
        n = int(rng.integers(1, 4097))
        results.extend(detector.feed(x[i:i+n]))
        i += n
    return results


@pytest.mark.parametrize('case', [c for c in CASES if c['expected_frame']], ids=lambda c: c['id'])
def test_capture_replay(case):
    x = recording(case)
    rate = case['sample_rate']
    rng = np.random.default_rng(28)
    leading = rng.normal(0, .0001, round(rate * .31))
    tail = np.zeros(round(rate * .5))
    if np.iscomplexobj(x):
        leading = leading.astype(complex)
        tail = tail.astype(complex)
    d = CQDetector(rate)
    results = replay(d, np.r_[leading, x, tail])
    assert len(results) == 1, (case['id'], results, d.turbo_attempts)
    result = results[0]
    assert result.frame.hex() == case['expected_frame']
    assert f'CQFRAME {result.call} {result.bandwidth}' == case['expected_notification']
    session, commands, modem = session_mock()
    session._dispatch('MYCALL W9SSJ')
    session._dispatch('LISTEN CQ')
    modem.poll()
    station = modem.station
    commands.clear()
    if np.iscomplexobj(x):
        positions = np.arange(int(len(x)*48000/rate))*rate/48000
        audio = np.interp(positions, np.arange(len(x)), x)
        audio = (audio * np.exp(2j*np.pi*1500*np.arange(len(audio))/48000)).real
    else:
        audio = x
    audio = np.r_[np.zeros(14880), audio, np.zeros(24000)]
    cursor = 0
    while cursor < len(audio):
        chunk = int(rng.integers(128, 4097))
        station.on_rx_stream(audio[cursor:cursor+chunk])
        cursor += chunk
    station.on_rx_audio(audio)
    assert station.state == A.VaraState.LISTENING
    assert station.pending_bytes == 0 and not modem.connected
    assert len(commands) == 3 and commands[-1] == case['expected_notification']
    assert not session._data.output
    session, commands, _ = session_mock()
    session._dispatch('LISTEN CQ')
    commands.clear()
    session.modem_cqframe(result.call, str(result.bandwidth), result.sn)
    assert commands == ['BITRATE (1)  18 bps RX', f'SN {result.sn:.1f}', case['expected_notification']]
    session, commands, _ = session_mock()
    if 'receive_commands' in case:
        for command in case['receive_commands']:
            session._dispatch(command)
        commands.clear()
        session.modem_cqframe(result.call, str(result.bandwidth), result.sn)
        assert bool(commands) == bool(case['original_stock_notification'])
    elif 'start_sample' in case:
        session._dispatch('LISTEN OFF' if 'listenoff' in case['id'] else
                          'LISTEN ON' if 'listenon' in case['id'] else
                          'LISTEN CQ' if 'listencq' in case['id'] else 'CHAT ON')
        commands.clear()
        session.modem_cqframe(result.call, str(result.bandwidth), result.sn)
        assert bool(commands) == (not ('listenoff' in case['id'] or 'listenon' in case['id']))


@pytest.mark.parametrize('case', BENCH, ids=lambda c: c['id'])
def test_bench_reencoding(case):
    _, call, bw = case['command'].split()
    frame = build_frame(call, int(bw))
    assert frame.hex() == case['frame_hex']
    np.testing.assert_array_equal(HEADER, case['segments'][0]['bins'])
    np.testing.assert_array_equal(np.r_[TRAINING, column_bins(frame)], case['segments'][1]['bins'])
    rendered = synth_frame(frame)[32*2048+1536:].reshape(94, 2048)
    np.testing.assert_array_equal(abs(np.fft.rfft(rendered, axis=1)).argmax(1),
                                  case['segments'][1]['bins'])


def test_open_case_ledger_has_no_expected_payloads():
    assert [(c['id'], c['expected_frame']) for c in CASES if not c['expected_frame']] == [
        ('air-01', None), ('air-14', None)]


@pytest.mark.parametrize('suffix', range(16))
def test_suffix_and_raw_bits(suffix):
    frame = build_frame('AB5CC' + (f'-{suffix}' if suffix else ''), 500)
    result = parse_frame(frame)
    assert result.call == 'AB5CC' + (f'-{suffix}' if suffix else '')
    from hfmodem.kestrel.coding.crc import crc16_genibus
    raw = frame[:4] + bytes([frame[4] | 7]) + frame[5:6]
    raw += crc16_genibus(raw).to_bytes(2, 'big')
    assert parse_frame(raw).frame == raw


@pytest.mark.parametrize('call,bw', [('W9SSJ/P', 500), ('ABCDEFGH', 500), ('W9SSJ-16', 500),
                                     ('W9SSJ-0', 500), ('W9SSJ-99', 500), ('W9_SSJ', 500)])
def test_unsupported_fields(call, bw):
    with pytest.raises(ValueError):
        build_frame(call, bw)


@pytest.mark.parametrize('hz,gap,sign', [(-49, .026667, 1), (17, .053333, -1), (61, .032, 1)])
def test_repeated_cycles_frequency_timing_and_orientation(hz, gap, sign):
    from hfmodem.kestrel.rx.cq import CQ
    body = np.r_[TRAINING, column_bins(build_frame('KJ7MKX-15', 2300))]
    tones = np.r_[HEADER, body]
    z = np.exp(2j*np.pi*(sign*(tones[:, None]-64)*FS_BIN + hz)*np.arange(512)/12000)
    iq = np.r_[z[:32].ravel(), np.zeros(round(gap*12000)), z[32:].ravel()]
    rng = np.random.default_rng(46)
    x = np.r_[np.zeros(563), iq, np.zeros(12000), iq, np.zeros(8000)]
    x += .08*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))
    d = CQDetector(12000)
    got = replay(d, x)
    assert len(got) == 2
    assert all(r.call == 'KJ7MKX-15' and r.bandwidth == 2300 for r in got)
    assert got[1].sample > got[0].sample + 50000
    assert CQ.columns == 92


FS_BIN = 12000/512


AIR = json.loads((FIXTURES / 'varac-cq-air-1023' / 'provenance.json').read_text())


@pytest.mark.parametrize('name', AIR)
def test_weak_and_late_cqs_on_air(name):
    from hfmodem.core.wav import source
    from hfmodem.tests.evidence import CORPUS
    prov = AIR[name]
    path = require_input(CORPUS / 'varac-cq-air-1023' / f'{name}.flac')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prov['flac_sha256']
    rate, x = wavfile.read(source(path))
    assert rate == prov['rate'] and len(x) == prov['samples'][1] - prov['samples'][0]
    got = replay(CQDetector(rate), np.r_[x / 32768, np.zeros(rate)])
    assert [[r.call, r.bandwidth] if hasattr(r, 'call') else ['beacon', r.text] for r in got] == prov['expected']


def test_noise_arq_other_mode_and_corrupt_crc():
    rng = np.random.default_rng(84)
    noise = rng.normal(0, .01, 48000*10)
    arq = M.synth_burst('W9SSJ', F.connect_request('500'))
    frequencies = np.repeat(rng.choice([1415, 1585], 300), 960)
    other = np.sin(2*np.pi*np.cumsum(frequencies)/48000)
    bad = bytearray(build_frame('W9SSJ', 500)); bad[-1] ^= 1
    d = CQDetector()
    assert replay(d, np.r_[noise, arq, np.zeros(48000), other,
                              synth_frame(bytes(bad)), np.zeros(48000)]) == []
    assert d.turbo_attempts == 1


class MockSocket:
    def __init__(self): self.output = []
    def sendall(self, data): self.output.append(data)


def session_mock():
    modem = VaraHostModem(IO(), bandwidth='500')
    session = HostSession(MockSocket(), MockSocket(), modem)
    commands = []
    session._send_cmd = commands.append
    modem.start(session)
    modem.poll()
    assert commands == ['REGISTERED HFMODEM']
    commands.clear()
    return session, commands, modem


@pytest.mark.parametrize('toggles,enabled', [(['LISTEN CQ'], True), (['CHAT ON'], True),
    (['LISTEN ON'], False), (['LISTEN OFF'], False),
    (['CHAT ON', 'LISTEN OFF'], False), (['CHAT ON', 'LISTEN ON'], False),
    (['CHAT ON', 'CHAT OFF'], False), (['LISTEN CQ', 'CHAT OFF'], False)])
def test_listen_mode_suppression(toggles, enabled):
    session, commands, modem = session_mock()
    for cmd in toggles: session._dispatch(cmd)
    modem.poll()
    commands.clear()
    session.modem_cqframe('AB5CC-15', '500', -2.15)
    assert commands == (['BITRATE (1)  18 bps RX', 'SN -2.1', 'CQFRAME AB5CC-15 500'] if enabled else [])


def test_host_owner_order_and_single_frame():
    session, commands, modem = session_mock()
    tx = []
    modem._transport.tx = lambda x: tx.append((threading.get_ident(), x))
    thread = threading.Thread(target=session._dispatch, args=('CQFRAME W9SSJ-9 500',))
    thread.start(); thread.join()
    assert commands == ['OK'] and not tx
    modem.poll()
    assert commands[:6] == ['OK', 'PENDING', 'PTT ON', 'BITRATE (1)  18 bps TX', 'PTT OFF', 'CANCELPENDING']
    assert len(tx) == 1 and tx[0][0] == threading.get_ident()
    got = replay(CQDetector(), np.r_[tx[0][1], np.zeros(24000)])
    assert len(got) == 1 and got[0].call == 'W9SSJ-9'


@pytest.mark.parametrize('cmd', ['CQFRAME', 'CQFRAME W9SSJ', 'CQFRAME W9SSJ  500', 'CQFRAME  W9SSJ 500',
                                'CQFRAME W9SSJ/P 500', 'CQFRAME W9SSJ 500 EXTRA'])
def test_invalid_command_edges_reply_wrong(cmd):
    session, commands, modem = session_mock()
    session._dispatch(cmd)
    modem.poll()
    assert commands == ['WRONG']


@pytest.mark.parametrize('state', [A.VaraState.CONNECTING, A.VaraState.CONNECTED, A.VaraState.DISCONNECTING])
def test_cq_cannot_key_during_arq(state):
    session, commands, modem = session_mock()
    modem.station.state = state
    session._dispatch('CQFRAME W9SSJ 500')
    modem.poll()
    assert commands == ['WRONG']


@pytest.mark.parametrize('cancel', ['abort', 'stop'])
def test_cancel_pending_cq(cancel):
    session, commands, modem = session_mock()
    modem.send_cq('W9SSJ', 500)
    getattr(modem, cancel)()
    modem.poll()
    assert ('PTT ON' in commands) == (cancel == 'abort')


def test_tx_error_unkeys():
    session, commands, modem = session_mock()
    def fail(x): raise RuntimeError('mock transport failed')
    modem._transport.tx = fail
    modem.send_cq('W9SSJ', 500)
    with pytest.raises(RuntimeError): modem.poll()
    assert commands[-2:] == ['PTT OFF', 'CANCELPENDING']


def test_continuous_station_cq_has_no_arq_effects():
    session, commands, modem = session_mock()
    session._dispatch('MYCALL W9SSJ')
    session._dispatch('LISTEN CQ')
    modem.poll()
    station = modem.station
    commands.clear()
    x = np.r_[np.zeros(8000), synth_cq('KJ7MKX-9', 500), np.zeros(24000)]
    rng = np.random.default_rng(411)
    i = 0
    while i < len(x):
        n = int(rng.integers(128, 4097))
        station.on_rx_stream(x[i:i+n]); i += n
    assert station.state == A.VaraState.LISTENING
    assert station.pending_bytes == 0 and not modem.connected
    assert len(commands) == 3 and commands[-1] == 'CQFRAME KJ7MKX-9 500'
    assert not session._data.output


def test_duplicate_pending_cq_keeps_one_transmission():
    session, commands, modem = session_mock()
    tx = []
    modem._transport.tx = tx.append
    session._dispatch('CQFRAME W9SSJ 500')
    session._dispatch('CQFRAME W9SSJ-9 500')
    modem.poll()
    assert len(tx) == 1
    assert commands.count('PENDING') == 1
    assert commands[:5] == ['OK', 'PENDING', 'PTT ON', 'WRONG', 'BITRATE (1)  18 bps TX']


def test_discontinuity_resets_partial_cq_and_bounds_memory():
    d = CQDetector()
    x = synth_cq('W9SSJ', 500)
    assert replay(d, x[:100000]) == []
    d.reset()
    assert replay(d, np.r_[x[100000:], np.zeros(48000)]) == []
    rng = np.random.default_rng(5)
    assert replay(d, rng.normal(size=48000*25)) == []
    assert len(d._buffer) <= 8*12000
    assert len(d._spectra) <= 124 and not d._candidates


def test_wire_messages_are_cr_terminated():
    modem = VaraHostModem(IO())
    cmd, data = MockSocket(), MockSocket()
    session = HostSession(cmd, data, modem)
    session._dispatch('LISTEN CQ')
    session.modem_cqframe('W9SSJ-9', '500', -1.3)
    assert b''.join(cmd.output) == b'OK\rBITRATE (1)  18 bps RX\rSN -1.3\rCQFRAME W9SSJ-9 500\r'


@pytest.mark.parametrize('state,role', [(A.VaraState.CONNECTING, 'responder'),
                                      (A.VaraState.CONNECTED, 'responder'),
                                      (A.VaraState.CONNECTED, 'initiator')])
def test_cq_does_not_advance_or_retry_an_arq_session(state, role):
    session, commands, modem = session_mock()
    session._dispatch('LISTEN CQ')
    modem.poll()
    hs = modem.station
    hs.state, hs.role = state, role
    hs.caller, hs.called = 'W9SSJ', 'WI9YZA'
    if state == A.VaraState.CONNECTING:
        hs.step = A._R_RESP_SENT
    else:
        hs._connected(confirm=False)
    before = hs.state, hs.step, hs.progress, hs.pending_bytes
    commands.clear()
    audio = np.r_[np.zeros(4800), synth_cq('AB5CC-15', 500), np.zeros(24000)]
    for i in range(0, len(audio), 1024): hs.on_rx_stream(audio[i:i+1024])
    hs.on_rx_audio(audio)
    assert (hs.state, hs.step, hs.progress, hs.pending_bytes) == before
    assert len(commands) == 3 and commands[-1] == 'CQFRAME AB5CC-15 500'
    assert not session._data.output


@pytest.mark.parametrize('body', [bytes.fromhex('5e44d3280000'), bytes.fromhex('fe44d3280010'),
                                 bytes.fromhex('5e44d3280050'),
                                 bytes.fromhex('000000000010'), bytes.fromhex('5e0003280010')])
def test_crc_alone_does_not_admit_unsupported_fields(body):
    from hfmodem.kestrel.coding.crc import crc16_genibus
    frame = body + crc16_genibus(body).to_bytes(2, 'big')
    assert parse_frame(frame) is None


def test_busy_channel_keeps_stub_reply_and_does_not_key():
    session, commands, modem = session_mock()
    modem._channel_busy = True
    session._dispatch('CQFRAME W9SSJ 500')
    modem.poll()
    assert commands == ['OK']


PINS = json.loads((FIXTURES / 'cq-fix1-host.json').read_text())


@pytest.mark.parametrize('pin', PINS, ids=lambda p: p['finding'])
def test_fix1_transcript_pin(pin):
    path = require_input(SOURCE / pin['source'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin['sha256']
    events = [json.loads(line) for line in path.read_text().splitlines()]
    at = pin['line'] - 1
    assert events[at:at + len(pin['events'])] == pin['events']
    command = pin['events'][0]['text']
    session, commands, modem = session_mock()
    finding = pin['finding']
    if finding == 'duplicate':
        session._dispatch('CQFRAME W9SSJ-1 500')
    elif finding == 'link-pending':
        session._dispatch('CONNECT W9SSJ WI9YZA')
    elif finding in ('link-initiator', 'link-responder'):
        modem.station.state = A.VaraState.CONNECTED
        modem.station.role = 'initiator' if finding.endswith('initiator') else 'responder'
    elif finding.startswith('abort-'):
        session._dispatch('CQFRAME W9SSJ-3 500')
    commands.clear()
    session._dispatch(command)
    if finding == 'duplicate':
        assert commands == []
        modem.poll()
        assert commands[:4] == ['PENDING', 'PTT ON', 'WRONG', 'BITRATE (1)  18 bps TX']
    else:
        assert commands == [pin['expected_reply']]
    if finding in ('unsupported-bandwidth', 'zero-bandwidth', 'text-bandwidth'):
        modem.poll()
        assert commands == ['OK']
    if finding in ('listen-off', 'chat-off', 'listen-on', 'listen-cq'):
        assert any(e['who'] == 'B' and e['text'].startswith('CQFRAME ')
                   for e in pin['events']) == (finding == 'listen-cq')
        assert any(e['who'] == 'B' and e['text'].endswith('bps RX')
                   for e in pin['events']) == (finding == 'listen-cq')


@pytest.mark.parametrize('case', [c for c in CASES if 'command' in c], ids=lambda c: c['id'])
def test_fix1_capture_bytes(case):
    _, call, bw = case['command'].split()
    if call.endswith('-16'):
        with pytest.raises(ValueError):
            build_frame(call, int(bw))
        _, call, bw = case['expected_notification'].split()
    assert build_frame(call, int(bw)).hex() == case['expected_frame']
    parsed = parse_frame(bytes.fromhex(case['expected_frame']))
    assert f'CQFRAME {parsed.call} {parsed.bandwidth}' == case['expected_notification']


@pytest.mark.parametrize('bandwidth', ['500', '2300', '2750'])
@pytest.mark.parametrize('argument', [500, 2300, 2750])
def test_cq_argument_independent_of_modem_bandwidth(bandwidth, argument):
    session, commands, modem = session_mock()
    tx = []
    modem._transport.tx = tx.append
    session._dispatch(f'BW{bandwidth}')
    session._dispatch(f'CQFRAME KA1ABCD-1 {argument}')
    modem.poll()
    assert len(tx) == 1
    np.testing.assert_array_equal(tx[0], synth_cq('KA1ABCD-1', argument))


@pytest.mark.parametrize('offset', [24000, 96000])
def test_abort_during_audio_finishes_before_ptt_off(offset):
    session, commands, modem = session_mock()
    expected = synth_cq('W9SSJ-3', 500)
    entered, release = threading.Event(), threading.Event()
    sent = []
    failures = []
    def tx(audio):
        sent.extend(audio[:offset])
        entered.set()
        assert release.wait(10)
        sent.extend(audio[offset:])
    def owner():
        try:
            modem.poll()
        except BaseException as exc:
            failures.append(exc)
    modem._transport.tx = tx
    session._dispatch('CQFRAME W9SSJ-3 500')
    worker = threading.Thread(target=owner)
    worker.start()
    try:
        assert entered.wait(10)
        session._dispatch('ABORT')
        session._dispatch('CQFRAME W9SSJ-4 500')
        assert commands[-2:] == ['OK', 'WRONG']
        assert 'PTT OFF' not in commands and modem._cq_pending
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not failures
    np.testing.assert_array_equal(sent, expected)
    assert commands[-2:] == ['PTT OFF', 'CANCELPENDING']
    assert not modem._cq_pending
    session._dispatch('CQFRAME W9SSJ-4 500')
    assert commands[-1] == 'OK'


def test_incoming_pending_cq_is_wrong():
    session, commands, modem = session_mock()
    modem._incoming_pending_at = 1.0
    session._dispatch('CQFRAME W9SSJ 500')
    assert commands == ['WRONG']


def test_cq_admission_recovers_after_transmit_failure():
    session, commands, modem = session_mock()
    def fail(audio):
        raise RuntimeError('mock transport failed')
    modem._transport.tx = fail
    session._dispatch('CQFRAME W9SSJ 500')
    with pytest.raises(RuntimeError):
        modem.poll()
    session._dispatch('CQFRAME W9SSJ 500')
    assert commands[-1] == 'OK'


def test_abort_before_cq_keeps_admission_until_audio_ends():
    session, commands, modem = session_mock()
    session._dispatch('ABORT')
    session._dispatch('CQFRAME W9SSJ 500')
    def tx(audio):
        assert modem._cq_pending
        session._dispatch('CQFRAME W9SSJ-1 500')
        assert commands[-1] == 'WRONG'
    modem._transport.tx = tx
    modem.poll()
    assert commands[-2:] == ['PTT OFF', 'CANCELPENDING']
    assert not modem._cq_pending


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_cq_allowed_after_link_disconnected(role):
    session, commands, modem = session_mock()
    modem.station.state = A.VaraState.CONNECTED
    modem.station.role = role
    modem._connected = True
    session.modem_connected('W9SSJ', 'WI9YZA', '500')
    session._dispatch('CQFRAME W9SSJ 500')
    assert commands[-1] == 'WRONG'
    session._dispatch('ABORT')
    modem.poll()
    assert 'DISCONNECTED' in commands
    commands.clear()
    session._dispatch('CQFRAME W9SSJ 2750')
    modem.poll()
    assert commands[:6] == ['OK', 'PENDING', 'PTT ON', 'BITRATE (1)  18 bps TX',
                            'PTT OFF', 'CANCELPENDING']


MODES_FIX2 = json.loads((FIXTURES / 'cq-fix2-modes.json').read_text())
LADDER_FIX2 = json.loads((FIXTURES / 'cq-fix2-ladder.json').read_text())


@pytest.mark.parametrize('case', MODES_FIX2['cases'], ids=lambda c: ';'.join(c['commands']))
def test_fix2_stock_receive_modes(case):
    path = require_input(SOURCE / MODES_FIX2['source'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODES_FIX2['sha256']
    events = [json.loads(line) for line in path.read_text().splitlines()]
    window = events[case['start_line'] - 1:case['stop_line']]
    assert [e['text'] for e in window if e['who'] == 'A' and e['kind'] == 'cmd_tx'] == case['commands']
    assert [e['text'] for e in window if e['who'] == 'A' and e['kind'] == 'cmd_rx'
            and (e['text'].startswith(('SN ', 'CQFRAME ')) or e['text'].endswith('bps RX'))] == case['reports']
    session, commands, modem = session_mock()
    for command in case['commands']:
        session._dispatch(command)
    modem.poll()
    commands.clear()
    session.modem_cqframe('WI9YZA', '500', -1.2)
    session.modem_cqframe('WI9YZA-7', '2750', 1.2)
    assert commands == case['reports']


@pytest.mark.parametrize('call,bw', [('W9SSJ', 500), ('WI9YZA-7', 2750), ('W9SSJ-9', 2300)])
@pytest.mark.parametrize('amplitude', [.25, .5])
def test_fix2_body_level(call, bw, amplitude):
    body_start = 32 * 2048 + 1536
    new = synth_cq(call, bw, amplitude=amplitude)
    old = amplitude * np.sin(2 * np.pi * np.r_[TRAINING, column_bins(build_frame(call, bw))][:, None]
                            * np.arange(2048) / 2048)
    ratio = np.sqrt(np.mean(new[body_start:] ** 2) / np.mean(old ** 2))
    assert ratio == pytest.approx(.270 / .353, abs=1e-12)
    assert 20 * np.log10(ratio) == pytest.approx(-2.328, abs=.001)
    np.testing.assert_array_equal(new[:32 * 2048 + 32], M.synth_tones(HEADER, amplitude=amplitude))
    assert not new[32 * 2048 + 32:body_start].any()
    assert np.sqrt(np.mean(new[body_start:] ** 2)) == pytest.approx(.270 * amplitude / .5, abs=.0005)


@pytest.fixture(scope='module')
def fix2_noise():
    n = 48000 * 30
    rng = np.random.default_rng(LADDER_FIX2['noise_seed'])
    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    spectrum[np.fft.rfftfreq(n, 1 / 48000) > LADDER_FIX2['noise_cutoff_hz']] = 0
    noise = np.fft.irfft(spectrum, n)
    return (noise / np.sqrt(np.mean(noise ** 2))).astype(np.float32)


@pytest.mark.parametrize('row', LADDER_FIX2['rows'], ids=lambda r: f"{r['noise_rms']}-{r['sender']}-{r['start_sample']}")
def test_fix2_ladder_reconstruction(row, fix2_noise):
    path = require_input(SOURCE / LADDER_FIX2['source'])
    if path not in _HASHES:
        _HASHES[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _HASHES[path] == LADDER_FIX2['sha256']
    transcript = require_input(SOURCE / LADDER_FIX2['transcript'])
    assert hashlib.sha256(transcript.read_bytes()).hexdigest() == LADDER_FIX2['transcript_sha256']
    event = json.loads(transcript.read_text().splitlines()[row['stock_sn_line'] - 1])
    assert event['who'] == 'B' and event['text'] == f"SN {row['stock_sn']:.1f}"
    fs, x = wavfile.read(path, mmap=True)
    assert fs == 48000
    x = x[row['start_sample']:row['stop_sample']]
    positions = (np.arange(len(x)) + row['noise_pos']) % len(fix2_noise)
    mixed = x + np.clip(fix2_noise[positions] * row['noise_rms'], -.98, .98)
    got = CQDetector().feed(mixed)
    assert len(got) == 1 and got[0].call == 'W9SSJ'
    # In-band reference bins replace the old flank PSD estimate. On this
    # unfiltered white-noise ladder they should agree within sampling error;
    # exact equality would freeze the estimator that fails after filtering.
    assert got[0].sn == pytest.approx(row['quality'] - 10 * np.log10(12000 / 2900), abs=1.5)
    if row['noise_rms'] >= .2:
        assert abs(got[0].sn - row['stock_sn']) < 2
    else:
        assert got[0].sn > row['stock_sn']


def test_fix2_cq_sn_matches_stock_where_noise_limits():
    rows = LADDER_FIX2['rows']
    ours = np.array([r['quality'] for r in rows]) - 10 * np.log10(12000 / 2900)
    stock = np.array([r['stock_sn'] for r in rows])
    rungs = np.array([r['noise_rms'] for r in rows])
    for rung in (.2, .3, .45, .6, .8, .98):
        at = rungs == rung
        assert abs(ours[at].mean() - stock[at].mean()) < 1


def test_fix2_duplicate_wrong_after_ptt_before_audio():
    session, commands, modem = session_mock()
    def tx(audio):
        assert commands[:5] == ['OK', 'PENDING', 'PTT ON', 'WRONG', 'BITRATE (1)  18 bps TX']
        session._dispatch('CQFRAME W9SSJ-2 500')
        assert commands[-1] == 'WRONG'
    modem._transport.tx = tx
    session._dispatch('CQFRAME W9SSJ 500')
    session._dispatch('CQFRAME W9SSJ-1 500')
    assert commands == ['OK']
    modem.poll()
    assert commands[-2:] == ['PTT OFF', 'CANCELPENDING']


def test_fix2_duplicate_busy_cq_still_replies_without_ptt():
    session, commands, modem = session_mock()
    modem._channel_busy = True
    session._dispatch('CQFRAME W9SSJ 500')
    session._dispatch('CQFRAME W9SSJ-1 500')
    modem.poll()
    assert commands == ['OK', 'WRONG']
    assert not modem._cq_pending and not modem._cq_rejections


@pytest.mark.parametrize('pin', json.loads((FIXTURES / 'cq-fix2-order.json').read_text()),
                         ids=lambda p: Path(p['source']).parent.name)
def test_fix2_duplicate_stock_order_pin(pin):
    path = require_input(SOURCE / pin['source'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin['sha256']
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[pin['line'] - 1:pin['line'] - 1 + len(pin['events'])] == pin['events']
    commands_in = [e['text'] for e in pin['events'] if e['who'] == 'A' and e['kind'] == 'cmd_tx']
    replies = [e['text'] for e in pin['events'] if e['who'] == 'A' and e['kind'] == 'cmd_rx'
               and e['text'] != 'BUSY ON']
    assert replies == ['OK', 'PENDING', 'PTT ON', 'WRONG', 'BITRATE (1)  18 bps TX']
    session, commands, modem = session_mock()
    for command in commands_in:
        session._dispatch(command)
    modem.poll()
    assert commands[:len(replies)] == replies


def test_fix2_duplicate_admission_does_not_wait_for_owner_key():
    session, commands, modem = session_mock()
    entered, release = threading.Event(), threading.Event()
    failures = []
    def key(on):
        if on:
            entered.set()
            assert release.wait(10)
    def owner():
        try:
            modem.poll()
        except BaseException as exc:
            failures.append(exc)
    modem._transport.key = key
    session._dispatch('CQFRAME W9SSJ 500')
    worker = threading.Thread(target=owner)
    duplicate = threading.Thread(target=session._dispatch, args=('CQFRAME W9SSJ-1 500',))
    worker.start()
    try:
        assert entered.wait(10)
        duplicate.start()
        duplicate.join(2)
        assert not duplicate.is_alive()
        assert commands == ['OK', 'PENDING']
    finally:
        release.set()
        worker.join(10)
        if duplicate.ident is not None:
            duplicate.join(10)
    assert not worker.is_alive() and not failures
    assert commands[:5] == ['OK', 'PENDING', 'PTT ON', 'WRONG', 'BITRATE (1)  18 bps TX']


def test_fix2_command_reader_keeps_replies_in_command_order():
    session, commands, modem = session_mock()
    deferred = threading.Event()
    class Reader(MockSocket):
        def __init__(self):
            super().__init__()
            self.input = b'CQFRAME W9SSJ 500\rCQFRAME W9SSJ-1 500\rVERSION\r'
        def recv(self, size):
            data, self.input = self.input, b''
            return data
    session._cmd = Reader()
    original = modem.defer_cq_rejection
    def defer(reply):
        result = original(reply)
        deferred.set()
        return result
    modem.defer_cq_rejection = defer
    reader = threading.Thread(target=session._cmd_reader)
    reader.start()
    try:
        assert deferred.wait(10)
        assert commands == ['OK']
        modem.poll()
        reader.join(10)
        assert not reader.is_alive()
        version = next(c for c in commands if c.startswith('VERSION '))
        assert commands.index(version) > commands.index('WRONG')
    finally:
        session._cq_reply_ready.set()
        reader.join(10)
