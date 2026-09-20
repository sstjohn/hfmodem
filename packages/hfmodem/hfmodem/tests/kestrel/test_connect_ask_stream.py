# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""K0SI's caller-keyed requests must reach the connecting state without a gate.

The three retained PCM crops are exact samples, declared in fixtures/ manifest.
No acknowledgement preamble is present. The existing bracket recognizer reads
these whole crops; live segmentation either omitted them or delivered fragments.
"""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_turn_request_confirms_connect import _ConnectIO

FIXTURES = Path(__file__).with_name("fixtures") / "k0si-0909"


def awaiting():
    io = _ConnectIO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", "K0SI", "W9SSJ"
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_LINKSETUP_SENT
    return hs, io


def feed(hs, audio, chunk=960):
    for at in range(0, len(audio), chunk):
        hs.on_rx_stream(audio[at:at + chunk])
        if hs.state == VA.VaraState.CONNECTED:
            return min(at + chunk, len(audio))
    return None


@pytest.mark.parametrize("attempt", [1, 2, 3])
@pytest.mark.parametrize("chunk", [960, 4096, 24000])
@pytest.mark.skipif(not FIXTURES.is_dir(), reason="K0SI PCM fixtures require the source checkout")
def test_exact_recorded_request_connects_without_a_bracket(attempt, chunk):
    with wave.open(str(FIXTURES / f"attempt{attempt}-ask.wav")) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), "<i2") / 32768.0
    hs, io = awaiting()
    assert hs._peer_wants_turn(audio), "The existing tone evidence must suffice"
    assert VA._ack_plateau(audio) == 0
    # The caller's request starts ~0.100 s into the retained crop, lasts 1.366 s.
    accepted = feed(hs, audio, chunk)
    assert accepted is not None
    assert accepted >= int(1.46 * MK.FS), "Answered while the peer was still keying"
    assert io.keys == 1
    assert io.up == [("W9SSJ", "K0SI", "2300")]
    assert VF.SESSION_CONFIRM.name not in " ".join(io.msgs)


@pytest.mark.parametrize("call,kind", [
    ("K7ABC", VF.SESSION_TURN_REQUEST_RESPONDER),
    ("W9SSJ", VF.SESSION_TURN_RELEASE_RESPONDER),
    ("W9SSJ", VF.SESSION_TURN_REQUEST),
    ("K0SI", VF.SESSION_RESPONDER_IDLE),
])
def test_wrong_station_and_wrong_control_do_not_connect(call, kind):
    hs, io = awaiting()
    audio = np.r_[np.zeros(MK.FS), MK.synth_burst(call, kind), np.zeros(MK.FS)]
    assert feed(hs, audio) is None
    assert io.keys == 0


def test_partial_request_waits_for_its_tail_and_does_not_replay_after_tx():
    hs, io = awaiting()
    request = MK.synth_burst("W9SSJ", VF.SESSION_TURN_REQUEST_RESPONDER)
    assert feed(hs, request[:-MK.HOP]) is None
    assert io.keys == 0
    tail = np.r_[request[-MK.HOP:], np.zeros(MK.FS // 4)]
    assert feed(hs, tail) is not None
    assert io.keys == 1
    assert len(hs._connect_ask_buf) == 0


def test_a_stale_request_in_a_delayed_poll_is_not_answered():
    hs, io = awaiting()
    request = MK.synth_burst("W9SSJ", VF.SESSION_TURN_REQUEST_RESPONDER)
    hs.on_rx_stream(np.r_[request, np.zeros(2 * MK.FS)])
    assert hs.state == VA.VaraState.CONNECTING
    assert io.keys == 0


def test_stream_request_requires_a_preceding_link_setup():
    hs, io = awaiting()
    hs.step = VA._I_CR_SENT
    request = MK.synth_burst("W9SSJ", VF.SESSION_TURN_REQUEST_RESPONDER)
    assert feed(hs, np.r_[request, np.zeros(MK.FS // 4)]) is None
    assert io.keys == 0


def test_transmit_breaks_the_request_search_and_noise_stays_bounded():
    hs, io = awaiting()
    request = MK.synth_burst("W9SSJ", VF.SESSION_TURN_REQUEST_RESPONDER)
    feed(hs, request[:MK.FS])
    hs._key(True)
    hs._key(False)
    before = io.keys
    assert feed(hs, np.r_[request[MK.FS:], np.zeros(MK.FS // 4)]) is None
    rng = np.random.default_rng(909)
    assert feed(hs, rng.normal(0, .1, 8 * MK.FS)) is None
    assert io.keys == before
    assert len(hs._connect_ask_buf) <= VA._SESSION_NEED + 6 * MK.HOP


def test_our_own_wideband_setup_cannot_confirm_the_connect():
    from hfmodem.kestrel.vara import vara_ofdm as OF

    hs, io = awaiting()
    audio = np.r_[OF.link_setup_tx("W9SSJ", bw="2300"), np.zeros(MK.FS // 4)]
    assert feed(hs, audio) is None
    assert io.keys == 0


def test_stock_vara_measured_tones_reach_the_stream_route():
    # Literal tones measured from stock VARA, independent of our generator.
    tones = [74, 90, 59, 94, 83, 42, 51, 88, 37, 96, 39, 86, 55, 54, 79, 96,
             31, 88, 59, 86, 75, 68, 45, 64, 31, 50, 49, 36, 67, 36, 43, 46]
    hs, io = awaiting()
    assert feed(hs, np.r_[MK.synth_tones(tones), np.zeros(MK.FS // 4)]) is not None
    assert io.keys == 1
