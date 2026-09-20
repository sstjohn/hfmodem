# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Peer-gap ACKs must not leave a stale generic-idle deadline.

N5TW's September 10 recording holds a complete closing greeting frame omitted
from live delivery, then a keepalive. These regressions cover the independently
identified clock/state defect; they do not claim physical RF overlap in that
recording or reconstruct the live decoder's exact buffer history.
"""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA

PROMPT = Path(__file__).with_name("fixtures") / "n5tw_20260910_final_prompt.wav"


class _IO(VA.VaraIO):
    def __init__(self):
        self.sent = []
        self.received = []

    def tx(self, samples):
        self.sent.append(samples)

    def key(self, on):
        pass

    def log(self, message):
        pass

    def data(self, payload):
        self.received.append(bytes(payload))


def _waiting():
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2750")
    hs.caller, hs.called = "W9SSJ", "N5TW"
    hs.role = "initiator"
    hs.state = VA.VaraState.CONNECTED
    hs.turn = VA._TURN_PEER
    hs._answer_owed = VA._OWED_OVER
    hs._reack_frame = VA.OVER_CONTINUE_GENERATED
    return hs, io


def test_positive_reack_notifies_the_mail_cadence_clock():
    hs, io = _waiting()
    before = (hs.progress, hs.idle_keyed)
    assert hs._reack()
    assert len(io.sent) == 1
    assert hs.progress == before[0], "Repeating an ACK is not payload progress"
    assert hs.idle_keyed == before[1] + 1, "Mail must observe this keying"


def test_served_peer_gap_cannot_key_generic_idle_and_erase_partial_data():
    hs, io = _waiting()
    # The frame is not whole, so the deferred-complete-frame ACK guard cannot
    # protect it. A timer tick must preserve whatever receive prefix exists.
    prefix = np.arange(480, dtype=float) / 480
    hs._ov_buf = prefix.copy()
    hs._held_answer = None
    hs._keyed_on_peer_burst = True
    hs.idle_keepalive()
    assert io.sent == [], "Peer-gap ACK already served this timer interval"
    np.testing.assert_array_equal(hs._ov_buf, prefix)
    assert hs._answer_owed == VA._OWED_OVER


@pytest.mark.skipif(not PROMPT.exists(),
                    reason="the recorded N5TW closing prompt requires the source checkout "
                           "(fixtures/n5tw_20260910_final_prompt.wav)")
def test_retained_n5tw_prompt_survives_idle_tick_while_window_end_is_pending():
    """The frame is on tape, but stream lookahead has not consumed it yet.

    The timer position comes from the recorded keepalive onset; the 20 ms
    polling phase is a controlled reproducer, not a recovered live schedule.
    The pre-fix idle method keys and clears this intact frame before delivery.
    """
    with wave.open(str(PROMPT)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 48000)
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
    hs, io = _waiting()
    hs.step = VA._I_CONNECTED
    hs._peer_over = 3
    hs._peer_delivery_open = True
    hs._last_window = (bytes.fromhex(
        "6c696d6974656420746f203130206d696e757465732e0d7b534649203d20313039204f6e"
        "20323032362d30392d31302030343a3030205554437d0d0a5b574c324b2d352e302d4232"
        "465749484a4d245d0d3b50513a2037313181"),)
    hs._last_window_complete = True
    hs._reack_frame = VA.OVER_CONTINUE_CAPTURED
    hs._reacks = 2
    hs._keyed_on_peer_burst = True
    hs.idle_keyed = 2
    original_start, recorded_keepalive = 3030272, 3257440
    ticked = False
    for offset in range(0, len(audio), 960):
        chunk = audio[offset:offset + 960]
        hs.on_rx_stream(chunk)
        newest = original_start + offset + len(chunk)
        if not ticked and newest >= recorded_keepalive:
            assert not io.received, "This phase must exercise an unconsumed frame"
            retained = hs._ov_buf.copy()
            hs.idle_keepalive()
            assert not io.sent, "Owed DATA cannot be replaced by a generic keepalive"
            np.testing.assert_array_equal(hs._ov_buf, retained)
            ticked = True
        if io.received:
            break
    assert ticked
    assert io.received == [b"54681\rCMS via N5TW >\r"]
