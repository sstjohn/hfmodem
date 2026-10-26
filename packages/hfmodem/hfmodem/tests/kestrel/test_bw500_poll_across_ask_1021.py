# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Kestrel as BW500 caller, chat on: stock's poll across our ask is not its grant.

Bench gate 5a, the QSY contact, 2026-10-21 (working/varac-on-air/bench3/gate5).
Stock WI9YZA answered over #1 and we released at 42.36. The host wrote `<TL>`
and we keyed the 32-symbol turn-request at 42.64-44.07. Stock, holding the turn
with nothing queued, keyed its frame-1 poll at 43.93, under the request's tail,
and again every 3.0 s (46.91, 49.92, 52.90) because nothing answered it. We took
the first poll as the grant ("peer answered with unread"), keyed over #2 into it,
took the 49.92 poll as its answer and reported BUFFER 0. Stock's log counts
RX 17 bytes, the two earlier messages only.

Every stock BW500 grant to a caller's ask on the 2026-10-18 pairs is answer
frame 13 keyed to the called station; the poll and the over answers are its
control frames keyed to the caller. On those pairs a CHAT ON caller asks with
the 17-symbol 851 request 0.29-0.30 s behind its release, which ends 0.55 s
before stock's first poll would; a CHAT OFF caller asks with the 32-symbol one.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

CALLER, CALLED = "W9SSJ", "WI9YZA"
TAIL = np.zeros(int(0.13 * MK.FS))

#: Stock A's request at 19.775 s of chat-w9-on-callerclose__a2b.wav (sha256
#: 6ad07b2e..., working/vara-answer/bench/phase-census.txt), demodulated.
STOCK_SHORT_ASK = [62, 67, 72, 66, 62, 74, 52, 66, 62, 74, 76, 66, 66, 52, 60, 58, 70]
#: Gate 5a, from our release's end: the host's `<TL>` write and stock's poll.
WRITE_S, POLL_S = 0.048, 1.568


class _Clock:
    t = 1000.0

    @classmethod
    def monotonic(cls):
        return cls.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    _Clock.t = 1000.0
    monkeypatch.setattr(VA, "time", _Clock)


class _IO(VA.VaraIO):
    def __init__(self):
        self.msgs, self.sent = [], []

    def key(self, on): ...
    def tx(self, samples): self.sent.append((_Clock.t, np.asarray(samples, float)))
    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): ...
    def log(self, msg): self.msgs.append(msg)

    def keyed(self, prefix):
        return sum(m.startswith(prefix) for m in self.msgs)


def _released(chat=True):
    io = _IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw="500", chat=chat)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn, hs._released = VA._TURN_PEER, True
    return hs, io


def _bracket(pairs):
    return np.concatenate([MK.synth_tone_pairs(pairs), TAIL])


def test_stocks_poll_across_our_ask_holds_the_data_and_asks_again():
    hs, io = _released()
    hs.send(b"<TL>\r\n")
    assert io.keyed("tx session-turn-request") == 1

    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 1)[1]))
    assert not io.keyed("tx DATA over"), "the poll was taken as the grant"
    assert hs.turn == VA._TURN_ASKED and hs._tx_pending is None
    assert io.keyed("tx session-turn-request") == 2, "no ask in the poll's turnaround"
    assert hs.payload_ledger.pending == 6

    hs.on_rx_audio(_bracket(VF.turn_grant(CALLED, "500")))
    assert io.keyed("tx DATA over #1 (6 payload bytes") == 1
    assert hs.payload_ledger.pending == 6, "credited before stock answered the over"

    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 1)[1]))
    assert hs.payload_ledger.pending == 0


def _ask_behind_release(chat):
    """Gate 5a from our release on: release, `<TL>` 0.048 s later, then the
    receive stream until the ask goes out. Returns (keyed at, burst, io)."""
    hs, io = _released(chat)
    hs.turn = VA._TURN_OURS
    assert hs._release_turn()
    released = _Clock.t
    _Clock.t += WRITE_S
    hs.send(b"<TL>\r\n")
    noise = np.random.default_rng(5).standard_normal(int(2 * MK.FS)) * 1e-4
    for at in range(0, len(noise), 960):
        if io.keyed("tx session-turn-request"):
            break
        _Clock.t += 960 / MK.FS
        hs.on_rx_stream(noise[at:at + 960])
    when, burst = io.sent[-1]
    return when - released, burst, io


def test_a_chat_caller_asks_with_stocks_short_request_clear_of_its_poll():
    after, burst, io = _ask_behind_release(chat=True)
    assert io.keyed("tx session-turn-request-short keyed-by=W9SSJ") == 1
    kind = VF.for_bw(VF.SESSION_TURN_REQUEST_SHORT, "500")
    assert list(MK.demod_tones(burst, len(STOCK_SHORT_ASK), band=MK.band_for("500"))) \
        == STOCK_SHORT_ASK == list(VF.handshake_tones(CALLER, kind))
    assert 0.29 <= after <= 0.32
    assert after + len(burst) / MK.FS < POLL_S, "the ask still runs into stock's poll"


def test_a_plain_caller_keeps_the_flown_32_symbol_request():
    after, burst, io = _ask_behind_release(chat=False)
    assert io.keyed("tx session-turn-request keyed-by=W9SSJ") == 1
    assert len(burst) == len(MK.synth_burst(CALLER, VF.for_bw(VF.SESSION_TURN_REQUEST, "500")))


#: Stock A's handovers in chat-w9-on-callerclose__a2b.wav, demodulated: the f20
#: release at 18.755 s (and 26.155 s) behind B's frame-1 answer to its last over,
#: and the f13 release at 34.495 s behind B's frame-0 ask. The same sequence
#: holds on every CHAT ON pair of 2026-10-18, both callsign pairs.
STOCK_F20 = [62, 67, 68, 52, 50, 50, 62, 60, 56, 76, 56, 50, 76, 72, 50, 54, 54]
STOCK_F13 = [62, 67, 52, 74, 50, 52, 74, 66, 76, 50, 54, 50, 52, 72, 52, 52, 76]


def _tones(burst):
    return list(MK.demod_tones(burst, 17, band=MK.band_for("500")))


def _answered_last_over(chat, frame):
    hs, io = _released(chat)
    hs.turn = VA._TURN_OURS
    hs.send(b"hi from A\r")
    assert io.keyed("tx DATA over #1") == 1
    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", frame)[1]))
    assert hs.payload_ledger.pending == 0 and hs.turn == VA._TURN_PEER
    return hs, io


def test_a_chat_caller_hands_an_empty_turn_back_with_stocks_f20():
    hs, io = _answered_last_over(chat=True, frame=1)
    assert io.keyed("tx session-turn-idle-short keyed-by=WI9YZA") == 1
    assert _tones(io.sent[-1][1]) == STOCK_F20


def test_a_chat_caller_answers_the_peers_ask_with_stocks_f13():
    hs, io = _answered_last_over(chat=True, frame=1)
    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 0)[1]))
    assert io.keyed("tx session-turn-release keyed-by=WI9YZA") == 1
    assert _tones(io.sent[-1][1]) == STOCK_F13
    assert hs.turn == VA._TURN_PEER


def test_a_plain_caller_keeps_the_flown_f13_release():
    hs, io = _answered_last_over(chat=False, frame=1)
    assert io.keyed("tx session-turn-release keyed-by=WI9YZA") == 1
    assert not io.keyed("tx session-turn-idle-short")


def test_the_stream_reads_the_answers_frame_before_choosing_the_release():
    hs, io = _released(chat=True)
    hs.turn = VA._TURN_OURS
    hs.send(b"hi from A\r")
    rng = np.random.default_rng(7)
    x = np.concatenate([np.zeros(int(0.1 * MK.FS)), _bracket(VF.control_bursts(CALLER, "500", 1)[1]),
                        np.zeros(int(0.5 * MK.FS))])
    x = x + rng.standard_normal(len(x)) * 1e-4
    for at in range(0, len(x), 960):
        _Clock.t += 960 / MK.FS
        hs.on_rx_stream(x[at:at + 960])
    assert io.keyed("tx session-turn-idle-short keyed-by=WI9YZA") == 1
    assert not io.keyed("tx session-turn-release")


#: Stock A behind its delivery closing at level 12, chat-clean__a2b.wav (sha256
#: 278aa815...): its answer at 57.075 s to B's frame 1, and its grant at
#: 58.225 s to B's frame 0. The same two frames, 11/11, on erase-a-data,
#: erase-b-data and erase-b-ack; clean-overlap's B answers frame 0 at once and
#: draws the grant (57.180 s).
STOCK_IDLE_ANSWER = [(64, 67), (56, 74), (64, 69), (68, 78), (62, 72), (60, 74),
                     (56, 68), (60, 74), (50, 64), (58, 68), (60, 64)]
STOCK_CALLER_GRANT = [(64, 67), (56, 74), (64, 69), (68, 78), (56, 72), (56, 64),
                      (54, 72), (50, 68), (58, 68), (60, 68), (60, 70)]


def _pairs(burst):
    return [tuple(sorted(p)) for p in MK.demod_tone_pairs(burst, 11)]


def _closed_at_level_12(chat=True):
    hs, io = _released(chat)
    hs.turn = VA._TURN_OURS
    hs._tx_pending = (phy.vara_body(b"x" * 20, CALLER), 9, 12)
    return hs, io


def test_a_high_speed_close_answered_with_frame_1_uses_the_complete_idle_confirmation():
    hs, io = _closed_at_level_12()
    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 1)[1]))
    assert io.keyed("tx session-turn-idle-short") == 1
    assert np.array_equal(io.sent[-1][1], MK.synth_burst(
        CALLED, VF.for_bw(VF.SESSION_TURN_IDLE_SHORT, "500")))
    assert hs.turn == VA._TURN_PEER and not hs._holding_idle()

    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 0)[1]))
    assert io.keyed("tx session-turn-release") == 1
    assert np.array_equal(io.sent[-1][1], MK.synth_burst(
        CALLED, VF.for_bw(VF.SESSION_TURN_RELEASE, "500")))
    assert hs.turn == VA._TURN_PEER


def test_a_high_speed_close_answered_with_frame_0_draws_stocks_grant():
    hs, io = _closed_at_level_12()
    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 0)[1]))
    assert io.keyed("tx caller-turn-grant") == 1 and not io.keyed("tx session-turn")
    assert _pairs(io.sent[-1][1]) == STOCK_CALLER_GRANT
    assert hs.turn == VA._TURN_PEER


def test_a_plain_high_speed_close_keeps_the_flown_release():
    hs, io = _closed_at_level_12(chat=False)
    hs.on_rx_audio(_bracket(VF.control_bursts(CALLER, "500", 1)[1]))
    assert io.keyed("tx session-turn-release keyed-by=WI9YZA") == 1
