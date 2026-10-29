# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An answer that went unheard is asked after and said again, as stock says it.

W9SSJ-1 to W9SSJ, 7105 kHz BW500 CHAT, 2026-10-28 17:00Z: the answerer's first
over drew our 32-symbol answer, which it did not read; it queried four times
with its eight-symbol frame 11, every query arrived clean here, nothing read
them, and it closed. A stock caller answers that query with the 32-symbol frame
at the step its continue asked: 60/311 from VE6SCL, 60/497 from stock behind a
lost continue, 60/1117 where the over itself was lost (2026-10-24).
"""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF
from .test_idle_poll import clock as clock, _Clock as _Clock, _IO as _IO
from .test_responder_control_stream import feed
from .test_setup_query_recovery import setup, padded

CALLER, CALLED = 'W9SSJ-1', 'W9SSJ'


def caller_owing(monkeypatch, *, high_speed=True, quality=-2.0):
    """A caller that read the answerer's first over, a full one at level 4,
    and answered it."""
    io = _IO()
    hs = A.VaraStationHandshake([CALLER], io, bw='500')
    hs.role, hs.called, hs.caller = 'initiator', CALLED, CALLER
    hs.state, hs.step = A.VaraState.CONNECTED, A._I_CONNECTED
    hs.turn = A._TURN_PEER
    hs._link_high_speed = high_speed
    key = phy.level_key('500', 4)
    size = phy.body_size('500', key)
    body = phy.vara_body(bytes(range(size - 1)), CALLER, tail=0x95, body_len=size)

    def over(samples):
        hs._rx_read = (4, quality)
        hs._peer_over_state = (4, 0x95)
        return [body]

    monkeypatch.setattr(hs, '_peer_data_over', over)
    assert hs._answer_data_over([])
    assert hs._answer_owed == A._OWED_OVER and len(io.sent) == 1
    return hs, io


def answer_query(called=CALLED):
    pairs = F.answer_query(called, '500')
    return np.concatenate([np.zeros(4800), M.synth_tone_pairs(pairs), np.zeros(3 * M.FS)])


def keyed(io, called, kind):
    want = M.synth_burst(called, F.for_bw(kind, '500'))
    return len(io.sent[-1]) == len(want) and np.allclose(io.sent[-1], want)


def caller_answer(frame):
    return replace(F.SESSION_OVER_RESPONSE, preadv=1 + 62 * frame)


@pytest.mark.parametrize('high_speed', [True, False])
def test_the_answerers_query_is_answered(monkeypatch, high_speed):
    hs, io = caller_owing(monkeypatch, high_speed=high_speed)
    feed(hs, answer_query())
    assert len(io.sent) == 2, io.msgs
    assert any('answering again' in m for m in io.msgs)
    assert hs._answer_owed == A._OWED_OVER and hs.state == A.VaraState.CONNECTED


def test_a_query_behind_an_unread_window_draws_the_nak(monkeypatch):
    hs, io = caller_owing(monkeypatch)
    hs._owed_block = True
    feed(hs, answer_query())
    assert len(io.sent) == 2, io.msgs
    assert any(keyed(io, CALLED, kind) for kind in (F.SESSION_OVER_NAK, F.SESSION_OVER_NAK_SHORT))


def test_another_stations_query_is_not_answered(monkeypatch):
    hs, io = caller_owing(monkeypatch)
    feed(hs, answer_query('N0ZZZ'))
    assert len(io.sent) == 1, io.msgs


def test_with_nothing_owed_a_query_is_not_answered(monkeypatch):
    hs, io = caller_owing(monkeypatch)
    hs._owe_nothing()
    feed(hs, answer_query())
    assert len(io.sent) == 1, io.msgs


@pytest.mark.parametrize('high_speed', [True, False])
def test_the_query_draws_the_32_symbol_frame_at_the_continues_step(monkeypatch, high_speed):
    hs, io = caller_owing(monkeypatch, high_speed=high_speed)
    step = hs._continue_step if high_speed else 0
    feed(hs, answer_query())
    assert keyed(io, CALLED, caller_answer(step)), io.msgs


def responder_owing(*, marked):
    hs, io = setup(caller=CALLER, marked=marked, ask=2)
    feed(hs, padded(M.synth_burst(CALLER, F.for_bw(F.SESSION_TURN_REQUEST, '500'))))
    level = phy.level_key('500', hs._expect_level)
    size = phy.body_size('500', level)
    body = phy.vara_body(bytes(range(size - 1)), CALLER, tail=0x95, body_len=size)
    feed(hs, padded(OF.data_over_tx(body, bw='500', level=level)))
    assert hs._answer_owed == A._OWED_OVER, io.msgs
    return hs, io


@pytest.mark.parametrize('marked', [True, False])
@pytest.mark.parametrize('query', [F.SESSION_ANSWER_QUERIES_SHORT[0],
                                   F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]])
def test_the_answerer_answers_a_query_with_the_32_symbol_frame(marked, query):
    hs, io = responder_owing(marked=marked)
    step = hs._continue_step if hs._link_high_speed else 0
    feed(hs, padded(M.synth_burst(CALLED, F.for_bw(query, '500'))))
    assert any('answering again' in m for m in io.msgs), io.msgs
    assert keyed(io, CALLED, A._QUERY_ANSWERS[step]), io.msgs


def answerer_sending(*, high_speed=True, chat=False):
    io = _IO()
    hs = A.VaraStationHandshake([CALLED], io, bw='500')
    hs.role, hs.called, hs.caller = 'responder', CALLED, CALLER
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_OURS
    hs.chat = hs._chat_profile = chat
    hs._link_high_speed = high_speed
    hs._lead = A.VB6Rnd(F.lead_state(CALLER, high_speed, '500'))
    hs.send(bytes(range(33, 233)))
    assert hs._tx_pending is not None and not phy.over_is_last(hs._tx_pending[0], CALLER)
    return hs, io


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('step', [0, 4, 5, 6, 7, 8])
def test_the_callers_32_symbol_answer_to_our_first_over_is_read(step, chat):
    """A stock caller answers a callee ISS's first over with 60/(1 + 62*step),
    55 of 57 ladders  [ACK-LEVEL-REQUEST.md], and so does ours."""
    hs, io = answerer_sending(chat=chat)
    first = hs._tx_pending
    feed(hs, padded(M.synth_burst(CALLED, F.for_bw(caller_answer(step), '500'))))
    assert hs.payload_ledger.confirmed == len(first[0]) - 1, io.msgs
    assert hs._tx_pending is not first


def test_the_callers_32_symbol_answer_is_read_on_a_weak_path():
    """Simulated air at -17 dB to the answerer, the noise the pair harness
    adds: all 31 payload tones exact read 2 seeds of 20."""
    read = 0
    for seed in range(10):
        hs, io = answerer_sending(chat=True)
        first = hs._tx_pending
        x = M.synth_burst(CALLED, F.for_bw(caller_answer(6), '500'))
        noise = np.random.default_rng(seed).normal(0, np.sqrt(np.mean(x ** 2) * 10 ** 1.7),
                                                   len(x) + 72000)
        feed(hs, np.pad(x, (24000, 48000)) + noise)
        read += hs._tx_pending is not first
    assert read >= 7


def test_a_free_link_takes_no_unsolicited_32_symbol_answer():
    hs, io = answerer_sending(high_speed=False)
    feed(hs, padded(M.synth_burst(CALLED, F.for_bw(caller_answer(0), '500'))))
    assert hs.payload_ledger.confirmed == 0, io.msgs


@pytest.mark.parametrize('step', [0, 6])
def test_the_32_symbol_answer_to_a_low_record_query_is_read(step):
    """Behind a record-0 over both the low-record reader and the query's own
    read the 288 stream, and the two must fit the receive stream's templates."""
    io = _IO()
    hs = A.VaraStationHandshake([CALLER], io, bw='500')
    hs.role, hs.called, hs.caller = 'initiator', CALLED, CALLER
    hs.state, hs.step = A.VaraState.CONNECTED, A._I_CONNECTED
    hs.turn, hs._link_high_speed = A._TURN_OURS, True
    hs._rec500 = phy.level_key('500', 1)
    hs.send(bytes(range(33, 93)))
    pending = hs._tx_pending
    assert pending is not None and pending[2] == phy.level_key('500', 1), io.msgs
    feed(hs, np.zeros(int(1.5 * M.FS)))
    assert hs.query_intermediate_answer(), io.msgs
    # Late enough that the query's own reader has looked and let it go.
    feed(hs, np.zeros(int(1.4 * M.FS)))
    feed(hs, padded(M.synth_burst(CALLED, F.for_bw(A._QUERY_ANSWERS[step], '500'))))
    assert not any('invariant failure' in m for m in io.msgs), io.msgs
    assert hs._tx_pending is not pending, io.msgs
