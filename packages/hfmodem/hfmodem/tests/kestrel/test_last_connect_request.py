# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.kestrel.vara.vara_arq import VaraState, VaraStationHandshake
from hfmodem.tests.kestrel.test_session_retry import runtime
from hfmodem.tests.kestrel.test_vara_connect import _IO


def final_request(mode):
    owner, modem, audio, clock = runtime(mode=mode)
    owner.step()
    deadline = owner._attempt_end
    while owner._retry_at < deadline and clock.now < deadline:
        clock.now = owner._retry_at
        owner.step()
    return owner, modem, audio, clock, deadline


def receive(hs, burst):
    samples = np.pad(burst, (24000, 12000))
    for at in range(0, len(samples), 1024):
        hs.on_rx_stream(samples[at:at + 1024])
    hs.on_rx_audio(samples)
    hs.on_rx_stream(np.empty(0), flush=True)


@pytest.mark.parametrize('mode', [None, 'WINLINK', 'P2P'])
def test_failure_waits_out_the_final_request_reply_window(mode):
    owner, modem, audio, clock, deadline = final_request(mode)
    last_end = audio.edges[-1][1]
    reply_window = audio.edges[-2][1] - audio.edges[-3][1]
    next_keyings = []
    modem.observer.modem_disconnected = lambda: next_keyings.append(clock.now)
    clock.now = max(clock.now, deadline)
    owner.step()
    assert modem.station.state == VaraState.CONNECTING
    assert not next_keyings
    clock.now = owner._retry_at - .001
    owner.step()
    assert not next_keyings
    clock.now = owner._retry_at
    count = len(audio.bursts)
    owner.step()
    assert modem.station.state != VaraState.CONNECTING
    assert len(audio.bursts) == count
    assert len(next_keyings) == 1
    assert next_keyings[0] - last_end >= reply_window - 1e-9


@pytest.mark.parametrize('mode', [None, 'WINLINK', 'P2P'])
@pytest.mark.parametrize('buffered', [False, True])
def test_reply_to_final_request_connects_after_attempt_deadline(mode, buffered):
    owner, modem, audio, clock, deadline = final_request(mode)
    clock.now = max(clock.now, deadline)
    owner.step()
    assert modem.station.connect_request_pending
    clock.now = owner._retry_at if buffered else owner._retry_at - .1
    replies = []
    peer = VaraStationHandshake(['WI9YZA'], _IO(replies), bw='500')
    peer.listen(True)
    receive(peer, audio.bursts[-1])
    assert replies

    def next_reply(timeout, hs=None):
        if replies:
            samples = replies.pop(0)
            if buffered and hs.connect_request_pending:
                for at in range(0, len(samples), 1024):
                    hs.on_rx_stream(samples[at:at + 1024])
                assert hs.connect_request_pending
            else:
                receive(hs, samples)

    audio.next_rx_burst = next_reply
    owner.step()
    assert modem.station.connect_setup_pending
    receive(peer, audio.bursts[-1])
    while replies:
        owner.step()
    assert modem.connected
    assert peer.state == VaraState.CONNECTED
    assert modem.observer.closes == 0


@pytest.mark.parametrize('mode', [None, 'WINLINK', 'P2P'])
def test_refused_final_request_does_not_extend_the_attempt(mode):
    owner, modem, audio, clock = runtime(mode=mode)
    owner.step()
    deadline = owner._attempt_end
    while owner._retry_at < deadline - 8:
        clock.now = owner._retry_at
        owner.step()
    audio.refused = True
    clock.now = owner._retry_at
    owner.step()
    clock.now = deadline
    owner.step()
    assert modem.station.state != VaraState.CONNECTING
    assert modem.observer.closes == 1
