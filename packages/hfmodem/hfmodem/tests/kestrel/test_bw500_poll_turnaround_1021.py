# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Our turn-requests against stock's repeating poll, bench 1 on e277ef175,
2026-10-21 (pat-gate/tip2-b2a0.6).

Behind our release at 126.21 s stock answered at 127.01 and polled every
2.25 s, 0.49 s each. We asked at 128.00, 137.13 and 148.65, 0.49-0.91 s behind
a poll's end, and each ask ran 0.12, 0.25 and 0.52 s under the next poll, which
stock keys on its own clock. Its listening gap is 1.76 s and the ask 1.37 s, so
only an ask keyed within 0.39 s of a poll's end reaches it whole. The repeats
rode the 10 s idle cadence because nothing drew them: an outstanding ask read
stock's poll only in the bracket route, and the bench log holds none of its
lines. Given the turn back, we named the 167.49 poll off its preamble 0.20 s
in and asked 0.43 s into it.
"""

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_answer_window_1021 import _band_noise, _burst, _caller
from hfmodem.tests.kestrel.test_idle_poll import CALLER, _Clock, _station, clock  # noqa: F401

BLOCK = 960
POLL_S = 2.25
CABLE = (evidence.WORKING / "varac-on-air/pat-gate/tip2-b2a0.6/"
         "cables/pat-tip2-b2a0.6__b2a.wav")
#: Stock's PTT on the cables behind our release: 127.01 then every 2.25 s.
POLL_ENDS = [127.51, 129.75, 131.99, 134.25, 136.49, 138.74, 140.99, 143.24, 145.50,
             147.74, 149.99, 152.24, 154.49, 156.74, 158.99, 161.24, 163.50, 165.75, 167.99]


def _drive(hs, io, x, start):
    """Feed ``x`` from cable second ``start``, skipping our own keying as the
    transport does; the cable second of every burst keyed."""
    keyed, skip = [], 0
    for at in range(0, len(x), BLOCK):
        _Clock.t = start + (at + BLOCK) / MK.FS
        if _Clock.t < skip:
            continue
        sent = len(io.sent)
        hs.on_rx_stream(x[at:at + BLOCK])
        if len(io.sent) > sent:
            keyed.append(_Clock.t)
            skip = _Clock.t + len(io.sent[-1]) / MK.FS + 0.1
    return keyed


def _released_with_a_write(hs, at):
    _Clock.t = at
    hs.turn = VA._TURN_OURS
    assert hs._release_turn()
    hs.send(b"ok")


def test_each_poll_draws_the_ask_into_its_turnaround():
    hs, io = _caller()
    _released_with_a_write(hs, 0.0)
    poll = _burst(VF.control_bursts(CALLER, "500", 1)[1], snr_db=-3.0)
    gap = int(POLL_S * MK.FS) - len(poll)
    x = np.concatenate([_band_noise(0.8, 2)]
                       + [np.r_[poll, _band_noise(gap / MK.FS, 10 + k)] for k in range(4)])
    ends = [0.8 + k * POLL_S + len(poll) / MK.FS for k in range(4)]
    keyed = _drive(hs, io, x, 0.0)
    assert len(keyed) == 4, io.msgs
    for t, end in zip(keyed, ends):
        assert VA._ANSWER_HELD_S - 0.03 <= t - end <= 0.25, (t - end, io.msgs)
    assert hs.turn == VA._TURN_ASKED
    sent = len(io.sent)
    hs.idle_keepalive()
    assert len(io.sent) == sent, "repeated on our own clock behind a poll-drawn ask"


@pytest.mark.skipif(not CABLE.exists(), reason="bench recording not kept here")
def test_the_bench_repeats_go_into_stocks_turnarounds():
    """From our first ask, 128.00-129.37: the 129.25 poll's head is under it."""
    from scipy.io import wavfile
    _, x = wavfile.read(CABLE)
    hs, io = _station("500")
    hs.called, hs.high_speed, hs._link_high_speed = "W1AW", True, True
    hs._track_band(np.asarray(x[int(120 * MK.FS):int(124 * MK.FS)], float))
    _released_with_a_write(hs, 126.21)
    _Clock.t = 128.0
    hs._ask_owed = False
    assert hs._tx_turn_request()
    start = 129.47
    keyed = _drive(hs, io, np.asarray(x[int(start * MK.FS):int(168.2 * MK.FS)], float), start)
    assert len(keyed) == len(POLL_ENDS) - 2, io.msgs
    for t, end in zip(keyed, POLL_ENDS[2:]):
        assert 0.05 <= t - end <= 0.25, (end, t - end)
