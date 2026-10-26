# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Pat at BW500 with noise on kestrel's input on the locator tree, 2026-10-21
(pat-gate/cnloc-b2a*, bn2-b2a0.6-*).

- Nine turn-requests still went 0.29-0.32 s into stock's 0.48 s answer to our
  release. Pat's write came after the 0.295 s gap, while our receive stream was
  still behind our own transmission, so the band had nothing in it to read and
  the ask went at once. Stock answers our 17-symbol release every time (2026-10-18
  pairs: frame 1, 0.08 s on), so the ask now waits for that answer, or for longer
  than it could take.
- Our releases behind stock's acknowledgement came 0.02-0.17 s after it: the
  bracket route keyed them as its gate closed, ahead of the stream's hold.
- bn2-b2a0.6-1 keyed its close 0.84 s into stock's 1.40 s query and lost it.
"""

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.tests import evidence
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_answer_window_1021 import _band_noise, _burst, _caller
from hfmodem.tests.kestrel.test_idle_poll import CALLER, _Clock, _station, clock  # noqa: F401

CABLES = (evidence.WORKING / "varac-on-air/pat-gate")
BLOCK = 960


def _cable(session):
    return CABLES / session / "cables" / f"pat-{session}__b2a.wav"


def _wav(session):
    from scipy.io import wavfile
    fs, x = wavfile.read(_cable(session))
    return np.asarray(x, float)


#: (session, the live turn-request's start, the end of stock's answer on the b2a
#: band envelope), cable seconds.
ASKS = [("cnloc-b2a0.3-1", 24.43, 24.62), ("cnloc-b2a0.3-1", 51.30, 51.52),
        ("cnloc-b2a0.3-2", 24.39, 24.61), ("cnloc-b2a0.3-2", 51.34, 51.58),
        ("cnloc-b2a0.45-1", 34.77, 34.94), ("cnloc-b2a0.45-1", 61.84, 62.03),
        ("cnloc-b2a0.45-2", 30.47, 30.66), ("cnloc-b2a0.45-2", 57.50, 57.74)]


@pytest.mark.skipif(not CABLES.exists(), reason="bench recordings not kept here")
@pytest.mark.parametrize("session, live, end", ASKS)
def test_the_ask_behind_our_release_waits_for_stocks_answer(session, live, end):
    """Released at the live request's start less its 0.365 s; the stream starts
    0.35 s behind the release, as ours did behind its own keying, and Pat writes
    at 0.33 s."""
    x = _wav(session)
    hs, io = _station("500")
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs._track_band(x[int((live - 5.4) * MK.FS):int((live - 1.9) * MK.FS)])
    released = live - 0.365
    _Clock.t = released
    hs.turn = VA._TURN_OURS
    assert hs._release_turn()
    keyed, written = len(io.sent), False
    for at in range(int((released + 0.35) * MK.FS), int((released + 3) * MK.FS), BLOCK):
        _Clock.t = (at + BLOCK) / MK.FS
        if not written and _Clock.t >= released + 0.33:
            written = True
            hs.send(b"x" * 6)
        hs.on_rx_stream(x[at:at + BLOCK])
        if len(io.sent) > keyed:
            break
    assert len(io.sent) > keyed and 0.05 <= _Clock.t - end <= 0.3, (_Clock.t - end, io.msgs[-2:])


def test_with_no_answer_the_ask_goes_after_the_wait():
    hs, io = _caller()
    hs.turn = VA._TURN_OURS
    assert hs._release_turn()
    released, keyed = _Clock.t, len(io.sent)
    hs.send(b"x" * 6)
    x = _band_noise(2.0, 7)
    for at in range(0, len(x), BLOCK):
        _Clock.t += BLOCK / MK.FS
        hs.on_rx_stream(x[at:at + BLOCK])
        if len(io.sent) > keyed:
            break
    assert len(io.sent) > keyed
    assert VA._RELEASE_ANSWER_WAIT_S <= _Clock.t - released <= VA._RELEASE_ANSWER_WAIT_S + 0.05


def test_the_bracket_leaves_our_release_to_the_streams_hold():
    hs, io = _caller()
    hs.turn = VA._TURN_OURS
    hs._tx_pending = (phy.vara_body(b"x" * 20, CALLER), 1, 4)
    burst = _burst(VF.control_bursts(CALLER, "500", 1)[1], snr_db=6.0)
    lead, tail = _band_noise(0.3, 4), _band_noise(1.0, 5)
    x = np.concatenate([lead, burst, tail])
    end, keyed, bracketed = len(lead) + len(burst), None, False
    for at in range(0, len(x), BLOCK):
        _Clock.t += BLOCK / MK.FS
        hs.on_rx_stream(x[at:at + BLOCK])
        if not bracketed and at + BLOCK >= end + 0.03 * MK.FS:
            bracketed = True
            hs.on_rx_audio(x[len(lead) - 2400:at + BLOCK])
        if io.sent:
            keyed = at + BLOCK
            break
    assert keyed is not None and io.msgs[-1].startswith("turn released"), io.msgs[-3:]
    assert -0.03 <= (keyed - end) / MK.FS - VA._ANSWER_HELD_S <= 0.03 + BLOCK / MK.FS


@pytest.mark.skipif(not _cable("bn2-b2a0.6-1").exists(), reason="bench recording not kept here")
def test_the_close_waits_out_stocks_query():
    """bn2-b2a0.6-1: the close went at 279.49 s, 0.84 s into stock's query,
    which ends at 280.05 s by stock's PTT."""
    x = _wav("bn2-b2a0.6-1")
    hs, io = _station("500")
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs.turn = VA._TURN_PEER
    hs._track_band(x[int(270 * MK.FS):int(273 * MK.FS)])
    _Clock.t = 278.4
    hs._key(False)
    closed = False
    for at in range(int(278.4 * MK.FS), int(284 * MK.FS), BLOCK):
        _Clock.t = (at + BLOCK) / MK.FS
        hs.on_rx_stream(x[at:at + BLOCK])
        if not closed and _Clock.t >= 279.49:
            closed = True
            hs.disconnect()
        if io.sent:
            break
    assert any(m.startswith("disconnecting") for m in io.msgs) and hs.state is VA.VaraState.DISCONNECTED
    assert 0.05 <= _Clock.t - 280.05 <= 0.3, _Clock.t - 280.05
