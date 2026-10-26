# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A named long grant answers a close ask even in the peer's DATA turn.

The Xara 0.1.1 BW500 caller-close sequence exposed this route: its c14/f0
GO_AHEAD is the existing 32-symbol SESSION_DRAINED_RESPONDER waveform.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M


class Clock:
    t = 1000.

    @classmethod
    def monotonic(cls):
        return cls.t


class IO(A.VaraIO):
    def __init__(self):
        self.sent, self.logs = [], []

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))

    def log(self, message):
        self.logs.append(message)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    Clock.t = 1000.
    monkeypatch.setattr(A, 'time', Clock)


def caller():
    io = IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw='500')
    h.role, h.caller, h.called = 'initiator', 'W9SSJ', 'W1AW'
    h.state, h.step, h.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_OURS
    h._start_compression('OFF')
    assert h._release_turn()
    return h, io


def grant():
    return M.synth_burst('W1AW', F.for_bw(F.SESSION_DRAINED_RESPONDER, '500'))


@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('queued', [False, True])
def test_close_ask_accepts_named_long_grant_before_any_queued_data(route, queued):
    h, io = caller()
    if queued:
        h._queue_host_payload(b'data still queued when the host closes')
    pending = h.payload_ledger.pending
    h.disconnect(acquire=True)
    assert h.turn == A._TURN_PEER  # Keep the holder's poll/retry cadence.
    before = len(io.sent)
    # With no prior band history the generic scanner can wake half a second
    # after the grant; a real tracked band fall wakes it earlier.
    x = np.r_[np.zeros(int(.07 * M.FS)), grant(), np.zeros(M.FS // 2)]
    if route == 'stream':
        for at in range(0, len(x), 480):
            block = x[at:at+480]
            Clock.t += len(block) / M.FS
            h.on_rx_stream(block)
            if h.state == A.VaraState.DISCONNECTED:
                break
    else:
        h.on_rx_audio(x)
    assert h.state == A.VaraState.DISCONNECTED, io.logs
    assert len(io.sent) == before + 1  # The close, with no intervening DATA.
    assert np.array_equal(io.sent[-1], M.synth_burst(
        'W1AW', F.for_bw(F.SESSION_DISCONNECT_REQ, '500')))
    assert h._tx_pending is None and h.payload_ledger.pending == pending


def test_unsolicited_long_grant_does_not_take_empty_peer_turn():
    h, io = caller()
    before = len(io.sent)
    x = np.r_[grant(), np.zeros(M.FS // 4)]
    for at in range(0, len(x), 480):
        Clock.t += 480 / M.FS
        h.on_rx_stream(x[at:at+480])
    h.on_rx_audio(x)
    assert h.state == A.VaraState.CONNECTED and h.turn == A._TURN_PEER
    assert len(io.sent) == before


def test_close_ask_does_not_take_an_incomplete_long_grant():
    h, io = caller()
    h.disconnect(acquire=True)
    before = len(io.sent)
    x = grant()[:M.FS // 2]
    for at in range(0, len(x), 480):
        Clock.t += 480 / M.FS
        h.on_rx_stream(x[at:at+480])
    assert h.state == A.VaraState.CONNECTED and len(io.sent) == before
