# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Five overlaps on bench 1 at b4a58721f, relay noise 0.6, 2026-10-21
(pat-gate/tip4-b2a0.6-*).

Four asks went 0.45-1.07 s into stock's 1078/63, each the first ask behind a
release that answered stock's 1078/1. The turn-idle reader drew none of them:
the release's answer wait took the band for stock's 0.48 s poll. Twice the band
dipped inside the 1.38 s frame and read as the poll's end. Twice it never read
lit under a floor stock's overs had raised, and the ask went when the 1.0 s wait
ran out. The asks the reader did draw went 0.16-0.22 s behind 1078/63's end.

The fifth followed a release that stock answered with a 4.95 s DATA over. The
over's head carries the control burst's preamble, the plateau named a poll
there, and the keepalive went 0.64 s into the over.
"""

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_answer_window_1021 import _band_noise, _burst, _caller
from hfmodem.tests.kestrel.test_bw500_turn_idle_1021 import (_feed, _frame, _released_on_the_ask,
                                                             _transmitting)
from hfmodem.tests.kestrel.test_idle_poll import CALLER, _Clock, _over, _station, clock  # noqa: F401

BLOCK = 960
CABLES = (evidence.WORKING / "varac-on-air/pat-gate")
#: (session, cable second just before stock's 1078/1, our keying in the 12 s
#: before it, the end of the 1078/63 stock answered our release with, by its PTT)
ASKS = [("tip4-b2a0.6-1", 70.8, [(57.77, 61.59), (62.90, 63.58), (65.11, 68.92),
                                 (70.21, 70.89)], 74.66),
        ("tip4-b2a0.6-2", 34.8, [(22.05, 22.73), (27.37, 27.84), (28.83, 32.93),
                                 (34.20, 34.88)], 38.69),
        ("tip4-b2a0.6-2", 72.8, [(59.75, 63.57), (64.84, 65.53), (67.12, 70.94),
                                 (72.21, 72.90)], 76.75),
        ("tip4-b2a0.6-3", 68.9, [(60.60, 61.07), (62.85, 66.96), (68.27, 68.96)], 72.77)]


def _cable(session):
    return CABLES / session / "cables" / f"pat-{session}__b2a.wav"


def _replay(session, start, ours, stop, write):
    """Our turn held from ``start``, the band's last 12 s heard as we heard them,
    our keying skipped; Pat writes 0.3 s behind our release when ``write``."""
    from scipy.io import wavfile
    _, x = wavfile.read(_cable(session))
    x = np.asarray(x, float)
    heard = np.ones(len(x), bool)
    for a, b in ours:
        heard[int(a * MK.FS):int((b + 0.1) * MK.FS)] = False
    lo, hi = int((start - 12) * MK.FS), int(start * MK.FS)
    hs, io = _station("500")
    _transmitting(io)
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs._track_band(x[lo:hi][heard[lo:hi]])
    hs.turn = VA._TURN_OURS
    _Clock.t, released, keyed = start, None, []
    for at in range(hi, int(stop * MK.FS), BLOCK):
        t = (at + BLOCK) / MK.FS
        if t < _Clock.t:
            continue
        _Clock.t, sent = t, len(io.sent)
        hs.on_rx_stream(x[at:at + BLOCK])
        if released is None and io.sent:
            released = _Clock.t
        elif write and released is not None and t >= released + 0.3:
            write = False
            hs.send(b"ok")
        if len(io.sent) > sent:
            keyed.append(io.at[-1])
    assert io.msgs[1] == "tx session-turn-release keyed-by=W1AW", io.msgs[:3]
    return keyed[1:], io


@pytest.mark.skipif(not CABLES.exists(), reason="bench recordings not kept here")
@pytest.mark.parametrize("session, start, ours, end", ASKS)
def test_the_ask_behind_our_release_goes_behind_stocks_1078_63(session, start, ours, end):
    asks, io = _replay(session, start, ours, end + 0.5, write=True)
    assert asks and 0.05 <= asks[0] - end <= 0.25, (asks, io.msgs[-3:])
    assert io.msgs[-1] == "tx session-turn-request keyed-by=W9SSJ"


@pytest.mark.skipif(not CABLES.exists(), reason="bench recordings not kept here")
def test_the_head_of_stocks_over_draws_nothing():
    """tip4-b2a0.6-3: stock's 1078/1 at 81.85, our release, its over 84.19-89.14."""
    keyed, io = _replay("tip4-b2a0.6-3", 81.7, [(70.59, 71.31), (72.45, 73.82),
                                                 (75.64, 79.75), (81.07, 81.76)],
                        89.14, write=False)
    assert not keyed, io.msgs[-3:]


def test_the_head_of_an_over_is_not_a_poll():
    """The preamble the plateau reads, and an over behind it."""
    hs, io = _caller()
    hs.turn = VA._TURN_OURS
    assert hs._release_turn()
    head = _burst(VF.control_bursts(CALLER, "500", 1)[1][:4], seed=2)
    over = _over(b"x" * 40, 6, 1)
    noise = _band_noise(len(over) / MK.FS, 3)
    over = over * np.sqrt(np.mean(noise ** 2) * 2400 / 700 * 2 / np.mean(over ** 2)) + noise
    assert _feed(hs, io, np.r_[_band_noise(0.08, 1), head, over]) is None, io.msgs[-2:]


def test_a_turn_idle_with_the_band_still_lit_behind_it_draws_no_ask():
    """1078/63 keyed straight into more keying: its tail fits, and its end is not."""
    hs, io, _ = _released_on_the_ask()
    idle = _frame(VF.SESSION_TURN_IDLE_RESPONDER, seed=5)
    more = _burst([(t, t) for t in VF.handshake_tones(CALLER, VF.for_bw(
        VF.SESSION_TURN_IDLE_RESPONDER, "500"))[::-1]], seed=6)
    assert _feed(hs, io, np.r_[_band_noise(0.08, 4), idle, more]) is None, io.msgs[-2:]
