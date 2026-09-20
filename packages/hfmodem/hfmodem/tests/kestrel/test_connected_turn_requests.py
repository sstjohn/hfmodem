# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recorded requests behind a pending final over must reach the held link."""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name("fixtures") / "kc9-connected-asks"


def audio(name):
    path = FIXTURES / (name + ".wav")
    if not path.exists():
        pytest.skip(f"KC9GHZ connected turn-request recording absent: {path}")
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768


def pending():
    hs, io = _connected(bw="2750")
    hs.turn = VA._TURN_OURS
    hs._txq = [b"p" * 89]  # Full intermediate over: a turn request is not its ACK.
    hs._tx_data_over()
    hs._since_progress = 7
    return hs, io


@pytest.mark.parametrize("chunk", [512, 4096, 4800])
def test_five_recorded_requests_are_named_once_without_acknowledging_data(chunk):
    hs, io = pending()
    original = hs._tx_pending
    for name in ["first-pair", "second-pair", "last"]:
        x = audio(name)
        for at in range(0, len(x), chunk):
            hs._stream_connect_ask(x[at:at + chunk])
    asks = [m for m in io.msgs if "unacknowledged block(s)" in m]
    assert len(asks) == 5, asks
    assert all("1 unacknowledged block(s), 0 queued" in m for m in asks)
    # Each is also logged where it is named on the stream, so a second request is
    # no longer invisible behind the release it does not draw.
    named = [m for m in io.msgs if "on the receive stream" in m]
    assert len(named) == 5 and all("already connected" in m for m in named)
    assert hs._tx_pending == original
    assert hs._since_progress == 7
    assert hs.turn == VA._TURN_OURS
    assert len(io.sent) == 1


def test_connected_stream_owns_the_request_before_a_late_bracket():
    hs, io = pending()
    x = audio("first-pair")
    for at in range(0, len(x), 4096):
        hs.on_rx_stream(x[at:at + 4096])
    asks = [m for m in io.msgs if "unacknowledged block(s)" in m]
    assert len(asks) == 2
    # Same first request, arriving later from an independently bounded gate.
    hs.on_rx_audio(x[int(.06 * MK.FS):int(1.66 * MK.FS)])
    assert len([m for m in io.msgs if "unacknowledged block(s)" in m]) == 2
    assert len(io.sent) == 1
    assert hs._since_progress == 7


def test_empty_queue_answers_the_recorded_request_after_its_tail():
    hs, io = _connected(bw="2750")
    hs.turn = VA._TURN_OURS
    x = audio("first-pair")[:int(1.7 * MK.FS)]
    answered_at = None
    for at in range(0, len(x), 512):
        hs.on_rx_stream(x[at:at + 512])
        if io.sent:
            answered_at = min(at + 512, len(x)) / MK.FS + 89.3
            break
    assert answered_at is not None
    assert 90.838 + MK.HOP / MK.FS <= answered_at <= 91.05
    assert hs.turn == VA._TURN_PEER
    assert len(io.sent) == 1
    assert any("tx session-turn-release" in m for m in io.msgs)


def test_a_delayed_poll_does_not_answer_a_stale_request():
    hs, io = _connected(bw="2750")
    hs.turn = VA._TURN_OURS
    # One already-ended request followed by a second of noise arrives in one
    # delayed poll. Searching earlier subwindows would transmit too late.
    x = np.concatenate([audio("first-pair")[:int(1.7 * MK.FS)], audio("noise")[:MK.FS]])
    hs.on_rx_stream(x)
    assert not io.sent


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_noise_and_another_call_cannot_release_the_turn(bw):
    hs, io = _connected(bw=bw)
    hs.turn = VA._TURN_OURS
    other = MK.synth_burst("KB3AC-10", VF.for_bw(VF.SESSION_TURN_REQUEST_RESPONDER, bw))
    x = np.concatenate([audio("noise"), other, audio("noise")])
    for at in range(0, len(x), 4096):
        hs.on_rx_stream(x[at:at + 4096])
    assert not io.sent
    assert not any(m.startswith("rx turn-request") for m in io.msgs)


def test_bracket_only_ignored_request_is_not_progress():
    hs, io = pending()
    x = audio("first-pair")[int(.06 * MK.FS):int(1.66 * MK.FS)]
    hs.on_rx_audio(x)
    assert hs._since_progress == 7
    assert hs._tx_pending is not None
    assert len(io.sent) == 1


def test_a_declined_release_does_not_claim_keying_or_progress(monkeypatch):
    hs, io = _connected(bw="2750")
    hs.turn = VA._TURN_OURS
    hs._since_progress = 7
    monkeypatch.setattr(hs, "_send_burst", lambda kind: False)
    assert not hs._took_turn_request()
    assert hs.turn == VA._TURN_OURS
    assert hs._since_progress == 7
    assert hs._release_owed
    assert not io.sent


def test_a_repeated_request_releases_again_without_resetting_progress():
    hs, io = _connected(bw="2750")
    hs.turn = VA._TURN_OURS
    assert hs._took_turn_request()
    assert hs._released
    hs._since_progress = 7
    progress = hs.progress
    assert hs._took_turn_request()
    assert len(io.sent) == 2
    assert hs.progress == progress
    assert hs._since_progress == 7
    assert hs._released
