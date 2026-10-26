# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock-proven CHAT setup recovery, separate from final DATA acknowledgement."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M


class IO(A.VaraIO):
    def __init__(self):
        self.sent, self.connections, self.logs, self.host = [], [], [], []
        self.accept = True
    def tx(self, x): self.sent.append(np.asarray(x))
    def tx_went_out(self): return self.accept
    def connected(self, *args): self.connections.append(args)
    def log(self, text): self.logs.append(text)
    def data(self, payload): self.host.append(bytes(payload))


def awaiting(*, refused=False):
    io = IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True, high_speed=True)
    hs.originate('K0EPI', 'W9SSJ')
    io.accept = not refused
    kind = F.connect_response('500', 1)
    hs.on_rx_tones(F.handshake_tones('K0EPI', kind), F.connect_response('500'))
    io.sent.clear()
    return hs, io


def idle(call='W9SSJ', kind=F.SESSION_TURN_IDLE_RESPONDER):
    return M.synth_burst(call, F.for_bw(kind, '500'))


def feed(hs, x, route):
    if route == 'bracket': hs.on_rx_audio(x)
    else:
        for at in range(0, len(x), 960): hs.on_rx_stream(x[at:at+960])


@pytest.mark.parametrize('route', ['bracket', 'stream'])
@pytest.mark.parametrize('native', [False, True])
def test_setup_idle_connects_once_without_crediting_queued_payload(route, native):
    hs, io = awaiting()
    hs.send(b'queued opening')
    pending = hs.payload_ledger.pending
    if native:
        rate, x = wavfile.read(Path(__file__).with_name('fixtures')/'k0epi-1024-idle.wav')
        assert rate == M.FS
    else:
        x = np.r_[np.zeros(960), idle(), np.zeros(9600)]
    feed(hs, x, route)
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.connections) == 1
    assert hs.turn == A._TURN_PEER
    assert hs.payload_ledger.pending == pending
    assert hs.payload_ledger.confirmed == 0
    assert hs._tx_pending is None and hs._txq
    assert len(io.sent) == 1
    np.testing.assert_allclose(io.sent[0], M.synth_burst('K0EPI', F.for_bw(F.SESSION_KEEPALIVE_B,'500')))
    hs._peer_setup_turn_idle(x)
    assert len(io.connections) == 1


def test_stream_waits_for_complete_idle_and_tail_before_keying():
    hs, io = awaiting()
    x = idle()
    feed(hs, x[:-M.HOP], 'stream')
    assert hs.state == A.VaraState.CONNECTING and io.sent == []
    feed(hs, np.r_[x[-M.HOP:], np.zeros(9600)], 'stream')
    assert hs.state == A.VaraState.CONNECTED and len(io.sent) == 1


@pytest.mark.parametrize('route', ['bracket', 'stream'])
@pytest.mark.parametrize('case', ['wrong-call', 'release', 'silence', 'truncated', 'refused', 'unsolicited', 'chat-off'])
def test_non_acknowledging_audio_or_state_cannot_promote_link(route, case):
    hs, io = awaiting(refused=case == 'refused')
    if case == 'unsolicited': hs.step, hs._linksetup_tx = A._I_CR_SENT, 0
    if case == 'chat-off': hs.chat = False
    x = idle('W1AW' if case == 'wrong-call' else 'W9SSJ',
             F.SESSION_TURN_RELEASE_RESPONDER if case == 'release' else F.SESSION_TURN_IDLE_RESPONDER)
    if case == 'silence': x = np.zeros(len(x))
    if case == 'truncated': x = x[:-3*M.HOP]
    feed(hs, np.r_[np.zeros(960),x,np.zeros(9600)], route)
    assert hs.state == A.VaraState.CONNECTING, io.logs
    assert not io.connections
    assert not io.sent, io.logs


def test_old_bracket_or_contradictory_tail_cannot_complete_setup():
    for corrupt in (False, True):
        hs, io = awaiting()
        if corrupt:
            tones = F.handshake_tones('W9SSJ', F.for_bw(F.SESSION_TURN_IDLE_RESPONDER,'500'))
            tones[-1] += 4
            x = np.r_[M.synth_tones(tones), np.zeros(9600)]
        else:
            x = np.r_[idle(), np.zeros(M.FS)]
        hs.on_rx_audio(x)
        assert hs.state == A.VaraState.CONNECTING
        assert io.connections == [] and io.sent == []


def test_refused_keepalive_does_not_undo_positive_connection_or_credit_data():
    hs, io = awaiting()
    hs.send(b'queued opening')
    pending = hs.payload_ledger.pending
    io.accept = False
    hs.on_rx_audio(np.r_[idle(), np.zeros(9600)])
    assert hs.state == A.VaraState.CONNECTED and len(io.connections) == 1
    assert hs.turn == A._TURN_PEER
    assert hs.payload_ledger.pending == pending and hs.payload_ledger.confirmed == 0
    assert hs._tx_pending is None and hs._txq


@pytest.mark.parametrize('phase', [0, 960, 2400, 4700])
@pytest.mark.parametrize('block', [960, 4096])
@pytest.mark.parametrize('native', [False, True])
def test_bounded_idle_scan_handles_callback_phases(phase, block, native):
    hs, io = awaiting()
    if native:
        _, burst = wavfile.read(Path(__file__).with_name('fixtures')/'k0epi-1024-idle.wav')
    else:
        burst = idle()
    x = np.r_[np.zeros(phase), burst, np.zeros(12000)]
    for at in range(0, len(x), block):
        hs.on_rx_stream(x[at:at+block])
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.connections) == 1 and len(io.sent) == 1


def test_expensive_idle_match_is_bounded_by_span_and_stride(monkeypatch):
    hs, io = awaiting()
    calls = []
    monkeypatch.setattr(hs, '_peer_turn_idle_frame', lambda x: calls.append(len(x)) or False)
    feed(hs, np.zeros(3*M.FS), 'stream')
    assert 1 <= len(calls) <= 16
    assert min(calls) >= A._span(F.for_bw(F.SESSION_TURN_IDLE_RESPONDER,'500')) + int(A._ANSWER_HELD_S*M.FS)
    hs._reset_stream()
    assert hs._setup_idle_scan_samples == 0


def test_native_idle_recovery_receives_next_level6_payload(monkeypatch):
    from types import SimpleNamespace
    from hfmodem.tests.kestrel.test_bw500_grant_reader_1021 import _stock_over, TEXT
    clock = SimpleNamespace(now=1000.)
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    hs, io = awaiting()
    rate, native = wavfile.read(Path(__file__).with_name('fixtures')/'k0epi-1024-idle.wav')
    for at in range(0, len(native), 960):
        clock.now += 960/M.FS
        hs.on_rx_stream(native[at:at+960])
    assert hs.state == A.VaraState.CONNECTED
    assert hs._expect_level == 6 and hs._lv_armed
    x = np.r_[np.zeros(4800), _stock_over(), np.zeros(M.FS)]
    for at in range(0, len(x), 1024):
        clock.now += 1024/M.FS
        hs.on_rx_stream(x[at:at+1024])
    assert b''.join(io.host) == TEXT, io.logs
    assert any('over at level 6 (asked 6)' in line for line in io.logs)
    assert not any('will not decode' in line for line in io.logs)
