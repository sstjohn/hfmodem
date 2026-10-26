# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Pat at BW500 with noise on kestrel's input, bench 1, 2026-10-21
(pat-gate/g4b-b2a0.6-1, -0.6-2).

The inbound message was lost 2 of 2 at 0.6 RMS. Four overs of stock's message
were read, the fifth was not, and stock idled (288/745) behind it every 3.1 s.
We answered each idle with 60/1117, three of the four clear of stock's keying,
drew only more idles and closed the link; stock's own level-1 resend began
0.17 s after our close. Behind a read over stock's caller keys 61/541 there
(stock pairs tk and tb-tf).

The requests went 0.40-0.45 s behind stock's idles, where its own receiver
keys them 0.20-0.24 s behind and ours, on the cadence path, drew its resend at
0.18-0.20 s.

Pat's reply was written 0.0-0.1 s after our answer to stock's last over, and
we keyed the turn-request at once: stock's 288/807 handover, 0.29 s behind our
answer, went out under it, and the turn came 11.7 s later on our retry. Stock's
caller waits for that handover (A's control f0, then B's release or grant).
"""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_idle_poll import (CALLED, CALLER, _Clock, _feed, _noise,
                                                  _station, clock)  # noqa: F401


def _caller():
    hs, io = _station("500")
    hs.high_speed = hs._link_high_speed = True
    hs.turn = VA._TURN_PEER
    return hs, io


def _keyed(io, name):
    return sum(m.startswith(f"tx {name} keyed") for m in io.msgs)


def test_an_over_lost_behind_read_ones_draws_stocks_short_missing_nak():
    hs, io = _caller()
    hs._note_close_level([phy.vara_body(b"x" * 89, CALLER)])
    hs._answer_owed, hs._owed_block, hs._owed_recovery = VA._OWED_OVER, True, True
    idle = MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, "500"))
    _feed(hs, np.concatenate([_noise(0.3), idle, _noise(0.6)]))
    assert _keyed(io, "session-over-nak-short") == 1, io.msgs[-3:]
    assert not _keyed(io, "session-over-nak")


def test_the_greetings_missing_nak_stays_the_long_form():
    hs, io = _caller()
    hs._answer_owed, hs._owed_block, hs._owed_recovery = VA._OWED_OVER, True, True
    idle = MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, "500"))
    _feed(hs, np.concatenate([_noise(0.3), idle, _noise(0.6)]))
    assert _keyed(io, "session-over-nak") == 1, io.msgs[-3:]


def test_a_reply_written_behind_our_last_answer_waits_for_stocks_handover():
    hs, io = _caller()
    hs._answer_owed = VA._OWED_RELEASE
    hs._key(False)
    hs.send(bytes(70))
    assert not _keyed(io, "session-turn-request"), "asked across stock's handover"
    handover = MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "500"))
    _feed(hs, np.concatenate([_noise(0.29), handover, _noise(0.5)]))
    assert any(m.startswith("tx DATA over #1") for m in io.msgs), io.msgs[-4:]
    assert not _keyed(io, "session-turn-request")


def test_the_request_for_an_unread_over_lands_in_stocks_window():
    """r1, over #10: the requests were named 0.29-0.32 s behind each idle and
    reached the air 0.40-0.45 s behind stock's key-down; stock took none. The
    ones that drew its resend in the same session went 0.18-0.20 s behind."""
    hs, io = _caller()
    hs._note_close_level([phy.vara_body(b"x" * 89, CALLER)])
    hs._answer_owed, hs._owed_block, hs._owed_recovery = VA._OWED_OVER, True, True
    idle = MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, "500"))
    x = np.concatenate([_noise(0.3), idle, _noise(0.6)])
    end = int(0.3 * MK.FS) + len(idle)
    for at in range(0, len(x), 960):
        _Clock.t += 960 / MK.FS
        hs.on_rx_stream(x[at:at + 960])
        if io.sent:
            break
    assert io.sent, io.msgs[-3:]
    held = (at + 960 - end) / MK.FS
    assert -0.03 <= held - VA._ANSWER_HELD_S <= 0.03 + 960 / MK.FS, held


def test_stocks_query_behind_our_nak_is_answered_in_its_window_not_its_turnaround():
    """r1 and r2's greetings: our 60/1117 went 0.07 and 0.12 s behind stock's
    289/331, inside its turnaround, and stock queried again. Its own receivers
    answer that query 0.195-0.235 s behind it."""
    hs, io = _caller()
    hs._expect_level, hs._lv_owed, hs._lv_prompt = 6, True, True
    hs._irs_nak(phy.LevelRead(6, (), None, 1))
    keyed = len(io.sent)
    query = MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_RESPONDER_IDLE_SHORT, "500"))
    x = np.concatenate([_noise(0.1), query, _noise(0.6)])
    end = int(0.1 * MK.FS) + len(query)
    for at in range(0, len(x), 960):
        _Clock.t += 960 / MK.FS
        hs.on_rx_stream(x[at:at + 960])
        if len(io.sent) > keyed:
            break
    assert _keyed(io, "session-over-nak") == 1, io.msgs[-3:]
    held = (at + 960 - end) / MK.FS
    assert -0.03 <= held - VA._ANSWER_HELD_S <= 0.03 + 960 / MK.FS, held
