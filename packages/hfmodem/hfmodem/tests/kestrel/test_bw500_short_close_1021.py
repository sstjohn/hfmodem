# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A VARA HF answerer closing in its own turn with the 16-symbol close.

2026-10-21, two runs: kestrel called, WI9YZA's last over drew our frame 0, it
released (289/391), we released back (61/391), and its host closed. WI9YZA
keyed 289/61, 16 symbols keyed to itself, at 168.45 s (168.51 s on the second run), then its
CW ident; kestrel read neither and closed on its stall check 73 s later.
"""

import numpy as np
import pytest

from hfmodem.core import cwid
from hfmodem.tests.kestrel import held
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_idle_poll import _Clock, _feed, _station, clock  # noqa: F401

#: The close at 168.45 s, demodulated: 16 of 16.
PEER_CLOSE = [62, 76, 50, 58, 70, 50, 52, 62, 50, 64, 66, 70, 60, 68, 64, 52]
RECORDING = held.recording("close-5a")


def _close(called, head_cut=0):
    frame = MK.synth_burst(called, VF.for_bw(VF.SESSION_DISCONNECT_REQ_RESPONDER_SHORT, "500"))
    envelope = np.concatenate([np.full(round(seconds * 5 / 7 * MK.FS), int(on))
                               for on, seconds in cwid.keying(called)])
    morse = .5 * envelope * np.sin(2 * np.pi * 1500 * np.arange(len(envelope)) / MK.FS)
    return np.concatenate([np.zeros(4800), frame[head_cut:], np.zeros(1700), morse,
                           np.zeros(24000)])


def test_the_frame_is_peers():
    kind = VF.for_bw(VF.SESSION_DISCONNECT_REQ_RESPONDER_SHORT, "500")
    assert list(VF.handshake_tones("WI9YZA", kind)) == PEER_CLOSE


def _state(name):
    hs, io = _station("500")
    hs.caller, hs.called = "W9SSJ", "WI9YZA"
    hs.chat = True
    hs.high_speed = hs._link_high_speed = True
    if name == "released":
        hs.turn, hs._released = VA._TURN_PEER, True
        hs._lv_owed = True
        hs._key(False)
    elif name == "holding":
        hs.turn, hs._poll_at = VA._TURN_OURS, _Clock.t + 30
    elif name == "asked":
        hs.turn = VA._TURN_ASKED
        hs._txq = [b"x" * 10]
    elif name == "recovering":
        hs.turn = VA._TURN_PEER
        hs._answer_owed, hs._owed_block, hs._owed_recovery = VA._OWED_OVER, True, True
    elif name == "over-pending":
        hs.turn = VA._TURN_OURS
        body = VA._phy.vara_body(bytes(43), hs.caller, body_len=44, tail=0x81)
        assert VA._phy.vara_payload(body, caller=hs.caller, body_len=44) == bytes(43)
        hs._tx_pending = (body, 1, 4)
    return hs, io


@pytest.mark.parametrize("name", ["released", "holding", "asked", "recovering", "over-pending"])
@pytest.mark.parametrize("head_cut", [0, 2048])
def test_the_short_close_closes_the_link_in_every_caller_state(name, head_cut):
    hs, io = _state(name)
    _feed(hs, _close("WI9YZA", head_cut))
    assert hs.state is VA.VaraState.DISCONNECTED, io.msgs[-4:]
    assert any("session-disconnect-request-responder-short" in m for m in io.msgs)
    assert not io.sent, "a close is answered with nothing"


@pytest.mark.skipif(not RECORDING.exists(), reason="recording unavailable")
def test_the_recorded_close_closes_the_link():
    from scipy.io import wavfile
    fs, x = wavfile.read(RECORDING)
    hs, io = _state("released")
    _feed(hs, np.asarray(x[int(167.5 * fs):int(176 * fs)], float))
    assert hs.state is VA.VaraState.DISCONNECTED, io.msgs[-4:]
    assert not io.sent, io.msgs
