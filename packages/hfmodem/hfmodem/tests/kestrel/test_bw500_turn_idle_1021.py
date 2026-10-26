# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Our turn-request against stock's 1078/63, bench 1, 2026-10-21
(pat-gate/tip3-b2a0.6-1).

Stock asked 1078/1 at 86.18 s and our release answered it, ending 88.40. Pat
wrote its last 2 bytes, and we asked at 88.53, on top of stock's 1078/63 from
88.48: the progress the 1078/1 counted had cleared the release's answer wait.
Stock keyed 1078/63 every 3.17 s and our asks rode our own clock, each into one,
until the one at 112.08 landed 0.17 s behind it and drew the grant at 113.56,
25.9 s after the release. A stock caller with data asks 0.10 s behind 1078/63's
end and nowhere else.
"""

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_answer_window_1021 import _band_noise, _burst, _caller
from hfmodem.tests.kestrel.test_idle_poll import CALLED, CALLER, _Clock, _station, clock  # noqa: F401

BLOCK = 960
CABLE = (evidence.WORKING / "varac-on-air/pat-gate/tip3-b2a0.6-1/"
         "cables/pat-tip3-b2a0.6-1__b2a.wav")
#: The last symbol of each 1078/63 stock keyed behind our release, b2a cable,
#: up to the one our live ask drew the grant behind.
IDLE_ENDS = [89.871, 93.031, 96.182, 99.339, 102.486, 105.629, 108.762, 111.913]


def _frame(kind, call=CALLER, seed=1):
    return _burst([(t, t) for t in VF.handshake_tones(call, VF.for_bw(kind, "500"))],
                  seed=seed)


def _transmitting(io):
    """The transport's tx returns when the burst has gone out."""
    tx = io.tx

    def keyed(samples):
        tx(samples)
        _Clock.t += len(samples) / MK.FS
    io.tx = keyed


def _feed(hs, io, x):
    """Feed until something is keyed; the samples heard by then, or None."""
    sent = len(io.sent)
    for at in range(0, len(x), BLOCK):
        _Clock.t += BLOCK / MK.FS
        hs.on_rx_stream(x[at:at + BLOCK])
        if len(io.sent) > sent:
            return at + BLOCK
    return None


def _released_on_the_ask():
    """Stock's 1078/1 to an empty queue, our release, and Pat's write behind it."""
    hs, io = _caller()
    _transmitting(io)
    hs.turn = VA._TURN_OURS
    ask = _frame(VF.SESSION_TURN_REQUEST_RESPONDER)
    assert _feed(hs, io, np.r_[_band_noise(0.5, 2), ask, _band_noise(0.6, 3)]) is not None
    assert io.msgs[-2] == f"tx session-turn-release keyed-by={CALLED}", io.msgs
    released = _Clock.t
    hs.send(b"ok")
    assert len(io.sent) == 1, "asked on top of stock's answer to our release"
    return hs, io, released


def test_the_ask_goes_behind_the_turn_idle_and_draws_the_grant():
    hs, io, released = _released_on_the_ask()
    lead, idle = _band_noise(0.08, 4), _frame(VF.SESSION_TURN_IDLE_RESPONDER, seed=5)
    heard = _feed(hs, io, np.r_[lead, idle, _band_noise(1.0, 6)])
    assert heard is not None and io.msgs[-1] == f"tx session-turn-request keyed-by={CALLER}"
    behind = (heard - len(lead) - len(idle)) / MK.FS
    assert VA._ANSWER_HELD_S - 0.03 <= behind <= 0.25, behind
    sent = len(io.sent)
    hs.idle_keepalive()
    assert len(io.sent) == sent, "repeated on our own clock behind a drawn ask"
    granted = _Clock.t + 0.15
    grant = _frame(VF.SESSION_DRAINED_RESPONDER, CALLED, seed=7)
    assert _feed(hs, io, np.r_[_band_noise(0.15, 8), grant, _band_noise(0.5, 9)]) is not None
    assert hs.turn == VA._TURN_OURS and io.msgs[-1].startswith("tx DATA over"), io.msgs[-3:]
    assert granted - released <= 3.5


def test_a_repeated_turn_request_draws_the_release_again_with_a_queue():
    hs, io, _ = _released_on_the_ask()
    ask = _frame(VF.SESSION_TURN_REQUEST_RESPONDER, seed=10)
    assert _feed(hs, io, np.r_[_band_noise(0.08, 11), ask, _band_noise(0.6, 12)]) is not None
    assert io.msgs[-2] == f"tx session-turn-release keyed-by={CALLED}", io.msgs[-4:]
    assert hs.turn == VA._TURN_PEER and hs._txq


@pytest.mark.skipif(not CABLE.exists(), reason="bench recording not kept here")
def test_the_bench_asks_go_behind_each_turn_idle():
    """From stock's 1078/1 at 86.18 with nothing queued, Pat writing behind
    our release; our keying is not on the b2a cable, and is skipped as the
    transport does."""
    from scipy.io import wavfile
    _, x = wavfile.read(CABLE)
    x = np.asarray(x, float)
    hs, io = _station("500")
    _transmitting(io)
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs._track_band(x[int(76 * MK.FS):int(80 * MK.FS)])
    hs.turn = VA._TURN_OURS
    wrote, skip = False, 0.0
    for at in range(int(86.1 * MK.FS), int(112.5 * MK.FS), BLOCK):
        t = (at + BLOCK) / MK.FS
        if t < skip:
            continue
        _Clock.t, sent = t, len(io.sent)
        hs.on_rx_stream(x[at:at + BLOCK])
        if io.sent and not wrote:
            wrote = True
            hs.send(b"ok")
        if len(io.sent) > sent:
            skip = _Clock.t + 0.1
    assert io.msgs[1] == "tx session-turn-release keyed-by=W1AW", io.msgs[:3]
    asks = io.at[1:]
    assert len(asks) == len(IDLE_ENDS), io.msgs
    for t, end in zip(asks, IDLE_ENDS):
        assert 0.05 <= t - end <= 0.25, (end, t - end)
