# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""When the answerer queries an over of its own, and for how long.

Asymmetric simulated air, -3 dB at the answerer and -17 dB at the caller,
2026-10-28: the caller's continue reached the answerer clean, 0.9-1.3 s behind
its unkey, where the turnaround scan steps every half second, and the query
went out before the scan that held it.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed
from .test_answer_query_recovery import answerer_sending, CALLED


@pytest.mark.parametrize('lead', [0.55, 0.7, 0.85])
def test_a_continue_between_scans_is_read_before_the_query(lead):
    hs, io = answerer_sending(chat=True)
    first, sent = hs._tx_pending, len(io.sent)
    burst = M.synth_tone_pairs(F.over_continue_frames(CALLED, '500', (6,))[6][0])
    feed(hs, np.concatenate([np.zeros(int(lead * M.FS)), burst]), 1024)
    # The audio stops short of the next scan, and the query falls due there.
    feed(hs, np.zeros(int(1.3 * M.FS) - int(lead * M.FS) - len(burst)), 1024)
    _Clock.t = hs._pending_answer_at + A._ANSWER_QUERY_S
    hs.query_intermediate_answer()
    assert hs._tx_pending is not first, io.msgs
    assert not any('queried' in m for m in io.msgs), io.msgs
    assert len(io.sent) == sent + 1


def test_a_repeated_release_behind_our_over_is_not_a_new_turn():
    """The caller re-released when it missed our ask's answer, and that copy
    trailed the over it started; taken as a new turn it closed the link with
    the over unanswered (simulated air, -17 dB one way, 2026-10-28)."""
    hs, io = answerer_sending(chat=True)
    first, sent = hs._tx_pending, len(io.sent)
    feed(hs, np.concatenate([np.zeros(4800), M.synth_burst(CALLED, F.for_bw(F.SESSION_TURN_RELEASE, '500')),
                             np.zeros(M.FS)]), 1024)
    assert hs.state == A.VaraState.CONNECTED, io.msgs
    assert hs._tx_pending is first and len(io.sent) == sent


def test_the_query_budget_outlasts_the_callers_own_recovery():
    """The caller, deaf to our queries, re-keys its answer on its own tick, 10 s
    behind its unkey and on stock's 12.06 s idle cadence at the latest."""
    hs, io = answerer_sending(chat=True)
    hs._progressed()
    first, start = hs._tx_pending, _Clock.t
    while _Clock.t - start < A._IDLE_POLL_S + 1.0:
        feed(hs, np.zeros(M.FS // 4), 1024)
        hs.query_intermediate_answer()
        hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED, io.msgs
    burst = M.synth_tone_pairs(F.over_continue_frames(CALLED, '500', (0,))[0][0])
    feed(hs, np.concatenate([np.zeros(4800), burst, np.zeros(M.FS)]), 1024)
    assert hs._tx_pending is not first, io.msgs
