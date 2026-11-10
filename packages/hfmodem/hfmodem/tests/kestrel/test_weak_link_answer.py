# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A weak caller-keyed BREAK can confirm setup, with the connected floor intact."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_turn_request_confirms_connect import _ConnectIO


def awaiting(bw):
    io = _ConnectIO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", "K0SI", "W9SSJ"
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_LINKSETUP_SENT
    return hs, io


def weak_answer(bw, seed, call="W9SSJ", *, kind=VF.SESSION_TURN_REQUEST_RESPONDER,
                lead=2 * MK.HOP, tail=3 * MK.HOP):
    kind = VF.for_bw(kind, bw)
    signal = MK.synth_burst(call, kind)
    signal *= .05 * 10 ** (-10 / 20) / np.sqrt(np.mean(signal ** 2))
    # Full-band white noise: 0.05 RMS in 250–2750 Hz, density over 0–24 kHz.
    noise = np.random.default_rng(seed).normal(
        0, .05 * np.sqrt(24000 / 2500), len(signal) + lead + tail)
    audio = noise.copy()
    end = lead + len(signal)
    audio[lead:end] += signal
    return audio, noise, end


@pytest.mark.parametrize("bw,seed,fitted_count", [("2300", 1, 20), ("2750", 2, 20)])
def test_weak_break_confirms_link_only_after_its_last_symbol(bw, seed, fitted_count):
    audio, _, end = weak_answer(bw, seed)
    hs, io = awaiting(bw)
    fitted, _, matched = hs._peer_frame(audio, VF.SESSION_TURN_REQUEST_RESPONDER,
                                       hs.caller)
    assert 19 <= sum(t >= 0 for t in fitted) <= 23
    assert sum(t >= 0 for t in fitted) == fitted_count
    assert matched
    hs, io = awaiting(bw)
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
        if at + 960 < end:
            assert hs.state is VA.VaraState.CONNECTING
            assert io.keys == 0
    assert hs.state is VA.VaraState.CONNECTED, io.msgs
    assert io.keys == len(io.sent) == 1
    assert io.up == [("W9SSJ", "K0SI", bw)]
    assert f"tx {VF.SESSION_TURN_RELEASE.name}" in " ".join(io.msgs)


@pytest.mark.parametrize("bw,seed", [("2300", 1), ("2750", 2)])
@pytest.mark.parametrize("negative", ["wrong_call", "noise", "connected"])
def test_weak_break_negative_controls(bw, seed, negative):
    audio, noise, _ = weak_answer(bw, seed)
    hs, io = awaiting(bw)
    if negative == "connected":
        fitted, _, matched = hs._peer_frame(audio, VF.SESSION_TURN_REQUEST_RESPONDER,
                                           hs.caller)
        assert matched and 19 <= sum(t >= 0 for t in fitted) <= 23
        hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
        assert not hs._peer_wants_turn(audio)
        assert VA._ASK_MIN_HEARD == 24
        return
    if negative == "wrong_call":
        audio, same_noise, _ = weak_answer(bw, seed, "K7ABC")
        np.testing.assert_array_equal(same_noise, noise)
    else:
        audio = noise
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
    assert hs.state is VA.VaraState.CONNECTING
    assert io.keys == 0 and not io.up


@pytest.mark.parametrize("bw,seed", [("2300", 1), ("2750", 2)])
@pytest.mark.parametrize("context", ["released", "holding", "pending", "wrong_call"])
def test_weak_connected_break_only_repeats_an_unread_release(bw, seed, context, monkeypatch):
    from .test_idle_poll import _Clock, _feed, _station

    _Clock.t = 1000.
    monkeypatch.setattr(VA, "time", _Clock)
    hs, io = _station(bw)
    hs.called, hs.turn = "K0SI", VA._TURN_OURS
    if context in ("released", "wrong_call"):
        assert hs._took_turn_request()
        hs.send(bytes(range(256)))
        assert hs._txq and hs._tx_pending is None and hs._released
    elif context == "pending":
        hs.send(bytes(range(256)))
        assert hs._tx_pending is not None
        hs._released, hs.turn = True, VA._TURN_PEER
    else:
        hs._released = True
        hs._txq = [b"still ours"]
    audio, _, end = weak_answer(bw, seed)
    fitted, _, matched = hs._peer_frame(audio, VF.SESSION_TURN_REQUEST_RESPONDER,
                                       hs.caller)
    assert matched and sum(t >= 0 for t in fitted) == 20
    if context == "wrong_call":
        audio, _, end = weak_answer(bw, seed, "K7ABC")
    assert hs._peer_wants_turn(audio) == (context == "released")
    sent, turn = len(io.sent), hs.turn
    pending, queue = hs._tx_pending, list(hs._txq)
    for at in range(0, len(audio), 960):
        _feed(hs, audio[at:at + 960])
        if at + 960 < end:
            assert len(io.sent) == sent
    assert hs._tx_pending == pending and hs._txq == queue
    if context == "released":
        assert len(io.sent) == sent + 1, io.msgs
        np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(
            hs.called, VF.for_bw(VF.SESSION_TURN_RELEASE, bw)))
        assert hs.turn == VA._TURN_PEER and hs._released
    else:
        assert len(io.sent) == sent and hs.turn == turn, io.msgs
