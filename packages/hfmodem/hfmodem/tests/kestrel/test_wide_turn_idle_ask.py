# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A wide CHAT OFF caller waits for, and asks behind, the release's idle answer."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_idle_poll import _Clock, _feed, _station, clock  # noqa: F401
from .test_weak_link_answer import weak_answer


def released(bw, *, chat=False):
    hs, io = _station(bw)
    hs.called, hs.chat, hs.turn = "K0SI", chat, VA._TURN_OURS
    assert hs._took_turn_request()
    assert hs._released and hs.turn == VA._TURN_PEER
    assert len(io.sent) == 1
    return hs, io


def idle(bw, *, weak=False, call="W9SSJ"):
    lead = int(.1 * MK.FS)
    if weak:
        audio, _, end = weak_answer(bw, 1, call, kind=VF.SESSION_TURN_IDLE_RESPONDER,
                                    lead=lead, tail=int(.22 * MK.FS))
        return audio, end
    signal = MK.synth_burst(call, VF.for_bw(VF.SESSION_TURN_IDLE_RESPONDER, bw))
    audio = np.zeros(lead + len(signal) + int(.22 * MK.FS))
    audio[lead:lead + len(signal)] += signal
    return audio, lead + len(signal)


def assert_ask(hs, io):
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(
        hs.caller, VF.for_bw(VF.SESSION_TURN_REQUEST, hs.bw)))
    assert hs.turn == VA._TURN_ASKED and hs._tx_pending is None


@pytest.mark.parametrize("bw", ["2300", "2750"])
def test_a_write_after_release_waits_for_peer_before_asking_once(bw):
    hs, io = released(bw)
    at = _Clock.t
    hs.send(bytes(range(256)))
    assert hs._ask_owed and len(io.sent) == 1
    _feed(hs, np.zeros(int((VA._TURN_IDLE_WAIT_S - .02) * MK.FS)))
    assert len(io.sent) == 1, io.msgs
    _feed(hs, np.zeros(int(.04 * MK.FS)))
    assert len(io.sent) == 1 and hs._ask_owed, io.msgs
    audio, end = idle(bw)
    at = _Clock.t
    _feed(hs, audio)
    assert len(io.sent) == 2, io.msgs
    behind = io.at[-1] - at - end / MK.FS
    assert VA._ANSWER_HELD_S <= behind <= (
        VA._ANSWER_HELD_S + VA._TURNAROUND_STEP / MK.FS + 960 / MK.FS)
    assert_ask(hs, io)
    _feed(hs, np.zeros(MK.FS))
    assert len(io.sent) == 2 and not hs._ask_owed


@pytest.mark.parametrize("bw", ["2300", "2750"])
@pytest.mark.parametrize("weak", [False, True], ids=["clean", "snr-10"])
def test_the_wide_idle_draws_one_ask_behind_its_last_symbol(bw, weak):
    hs, io = released(bw)
    at = _Clock.t
    hs.send(bytes(range(256)))
    audio, end = idle(bw, weak=weak)
    read = hs._peer_poll_held(audio)
    assert read is not None and read[1] == VF.SESSION_TURN_IDLE_RESPONDER.name
    assert read[0] == pytest.approx((len(audio) - end) / MK.FS, abs=.01)
    for start in range(0, len(audio), 960):
        _feed(hs, audio[start:start + 960])
        if _Clock.t - at < end / MK.FS + VA._ANSWER_HELD_S:
            assert len(io.sent) == 1, io.msgs
    assert len(io.sent) == 2, io.msgs
    behind = io.at[-1] - at - end / MK.FS
    assert VA._ANSWER_HELD_S <= behind <= (
        VA._ANSWER_HELD_S + VA._TURNAROUND_STEP / MK.FS + 960 / MK.FS)
    assert io.at[-1] - at < VA._TURN_IDLE_WAIT_S
    assert_ask(hs, io)
    _feed(hs, np.zeros(MK.FS))
    assert len(io.sent) == 2


@pytest.mark.parametrize("bw", ["2300", "2750"])
@pytest.mark.parametrize("negative", ["wrong_call", "empty", "chat"])
def test_only_our_idle_with_a_queue_draws_an_ask(bw, negative):
    hs, io = released(bw, chat=negative == "chat")
    if negative == "chat":
        _Clock.t += VA._RELEASE_ASK_GAP_S + .01
    if negative != "empty":
        hs.send(bytes(range(256)))
    sent = len(io.sent)
    audio, _ = idle(bw, call="K7ABC" if negative == "wrong_call" else hs.caller)
    _feed(hs, audio)
    assert len(io.sent) == sent, io.msgs


@pytest.mark.parametrize("bw", ["2300", "2750"])
def test_the_wide_chat_off_held_poll_reader_does_not_read_frame_1(bw):
    hs, _ = released(bw)
    audio = np.r_[np.zeros(int(.1 * MK.FS)), MK.synth_tone_pairs(
        VF.control_bursts(hs.caller, bw, 1)[1]), np.zeros(int(.22 * MK.FS))]
    assert hs._peer_poll_held(audio) is None
