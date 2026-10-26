# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Control ownership failures found by the two-application audio test."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M


class IO(A.VaraIO):
    def __init__(self):
        self.sent = []

    def tx(self, samples):
        self.sent.append(samples)

    def log(self, text):
        pass


def station(bw, role):
    io = IO()
    hs = A.VaraStationHandshake(['W9SSJ' if role == 'initiator' else 'W1AW'],
                                io, bw=bw, high_speed=True, chat=True)
    hs.caller, hs.called, hs.role = 'W9SSJ', 'W1AW', role
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_OURS
    hs._chat_profile = role == 'responder'
    return hs, io


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_streamed_grant_bracket_cannot_acknowledge_new_data(bw):
    hs, io = station(bw, 'initiator')
    hs.send(b'Not yet heard by the peer')
    pending, debt, sent = hs._tx_pending, hs.pending_bytes, len(io.sent)
    assert pending is not None and debt > 0
    # The stream already consumed this grant and keyed the outstanding over.
    # Its segmenter's bracket arrives afterwards, in the new sending state.
    hs._stream_owns_answer = True
    hs.on_rx_audio(M.synth_tone_pairs(F.turn_grant(hs.called, bw)))
    assert hs._tx_pending is pending and hs.pending_bytes == debt
    assert hs.payload_ledger.confirmed == 0
    assert len(io.sent) == sent


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('frame', [0, 1])
def test_empty_answerer_grants_complete_traffic_request_not_poll(bw, frame, monkeypatch):
    now = [1000.]
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    hs, io = station(bw, 'responder')
    hs._hold_released_turn()
    # The caller's role-specific frame 0 requests traffic/close; frame 1 has
    # the same preamble but must not acquire the holder's empty turn.
    audio = np.pad(M.synth_tone_pairs(F.control_bursts(hs.caller, bw, frame)[0]), (24000, 24000))
    for at in range(0, len(audio), 1024):
        block = audio[at:at + 1024]
        now[0] += len(block) / M.FS
        hs.on_rx_stream(block)
    assert hs.turn == (A._TURN_PEER if frame == 0 else A._TURN_OURS)
    assert bool(io.sent) == (frame == 0)
    assert hs.pending_bytes == hs.payload_ledger.confirmed == 0
