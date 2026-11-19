# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The answer to our changeover is read from the window it lies in.

WS8EOC 40 m 2026-11-16 22:28:57Z (arm 6e3e5514): TX[57], our changeover,
ended at stream sample 6608948. The gateway's CS1 started at 6615668, 140 ms
later, and its 21 symbols ended 44 samples before the ISS read window closed
at 6625792 (hold 46). The cold search wanted a symbol more behind each
hypothesis than the window held, so it read nothing; TX[58] keyed the
changeover again and drew the same CS1 at 6675714 (hold 47, window
6668934-6685824), also unread, and the link was handed back.
"""
import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.shrike import arq, onair, p3acquire
from hfmodem.tests import evidence
from hfmodem.tests.shrike.onair_support import _Session

CAPTURE = (evidence.OPERATOR_DATA / 'captures'
           / 'onair-20261116T222858304078Z-6e3e5514-c9b0-4ae3-8f77-303f828e311d')
HOLD46 = (6608948, 6625792, 6615668)
HOLD47 = (6668934, 6685824, 6675714)


def window(lo, hi, _):
    evidence.require_input(next(CAPTURE.glob('stream-*.flac'), CAPTURE / 'stream.wav'))
    return wav.read(CAPTURE / 'stream.wav', start=lo, stop=hi).astype(np.float32)


@pytest.mark.parametrize('hold', [HOLD46, HOLD47], ids=['hold46', 'hold47'])
def test_the_word_reach_reads_the_cs1_the_old_reach_missed(hold):
    audio = window(*hold)
    assert p3acquire.control_signal(audio) is None
    got = p3acquire.control_signal(audio, reach_s=p3acquire.WORD_REACH_S)
    assert got is not None and got.event.cs == arq.CS_ACK
    assert hold[0] + got.event.start == hold[2]
    assert '0 bit errors' in got.event.text and abs(got.offset_hz + 14.1) < 1


def changed_over(reach):
    s = _Session(role=arq.ISS)
    s.rx.p3_changeover_answer_reach = reach
    s.host.arq._inflight = arq._Packet(0x00, b"FS ", 1, breakin=True)
    assert s.host.arq.unconfirmed_breakin
    return s


def answer(s, hold):
    s.rx.new_cycle()
    lo, hi, _ = hold
    audio = window(*hold)
    return s.rx._acquire_answer(audio, lo, audio.size - onair.P3_CS_N)


def test_tx57s_answer_is_read_by_hold_47():
    s = changed_over("word")
    assert answer(s, HOLD46) is None                    # a candidate, uncorroborated
    assert s.rx._p3_head_candidate[1] == HOLD46[2]
    got = answer(s, HOLD47)
    assert got is not None and got.cs == arq.CS_ACK
    assert s.rx._p3_answer_at == HOLD47[2]


def test_the_symbol_reach_keeps_the_old_miss():
    s = changed_over("symbol")
    assert answer(s, HOLD46) is None and answer(s, HOLD47) is None
    assert s.rx._p3_head_candidate is None


def test_ordinary_iss_reads_keep_the_old_reach():
    s = changed_over("word")
    s.host.arq._inflight = arq._Packet(0x01, b"+\r", 1)
    assert not s.host.arq.unconfirmed_breakin
    assert answer(s, HOLD46) is None and s.rx._p3_head_candidate is None


def test_the_flag():
    assert onair.build_parser().parse_args([]).p3_changeover_answer_reach == "word"
    assert onair.build_parser().parse_args(
        ["--p3-changeover-answer-reach", "symbol"]).p3_changeover_answer_reach == "symbol"
