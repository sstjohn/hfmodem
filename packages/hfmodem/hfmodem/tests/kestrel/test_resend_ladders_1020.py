# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock ladders and off-level refusals captured on 2026-10-20."""
import time

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA
from .test_data_over_gate import _connected


CHAINS = [
    ('500', 13, 'MMM', (9, 7, 5)),
    ('500', 12, 'MMM', (9, 7, 5)),
    ('500', 8, 'MMM', (6, 4, 1)),
    ('500', 4, 'MMM', (1, 1, 1)),
    ('500', 8, 'MM', (6, 4)),
    ('500', 7, 'M', (5,)),
    ('500', 13, 'NNN', (9, 8, 7)),
    ('500', 13, 'N', (9,)),
    ('500', 8, 'N', (6,)),
    ('500', 7, 'N', (6,)),
    ('2300', 16, 'MMM', (13, 10, 7)),
    ('2300', 8, 'MMM', (6, 4, 2)),
    ('2300', 8, 'MM', (6, 4)),
    ('2300', 4, 'MMM', (2, 1, 1)),
    ('2300', 16, 'NNN', (13, 12, 10)),
    ('2300', 8, 'NN', (6, 5)),
    ('2750', 17, 'MMM', (14, 11, 8)),
    ('2750', 8, 'MMM', (6, 4, 2)),
    ('2750', 4, 'MMM', (2, 1, 1)),
    ('2750', 17, 'NNN', (14, 13, 11)),
    ('2750', 8, 'N', (6,)),
    ('2750', 6, 'N', (5,)),
    ('2750', 7, 'MN', (5, 4)),
]


@pytest.fixture
def render(monkeypatch):
    monkeypatch.setattr(VA.OF, 'data_over_tx', lambda *a, **kw: np.zeros(64))


def sender(bw, start):
    hs, io = _connected(bw=bw)
    hs._link_high_speed = True
    hs._start_at(start)
    hs.turn = VA._TURN_OURS
    hs.send(bytes(i % 251 for i in range(12000)))
    return hs, io


def tx_nak(hs, kind):
    if kind == 'M':
        hs._retry_data_over()
        hs._took_responder_nak()
    else:
        hs._pending_answer_at = time.monotonic()
        hs._matched_data_nak = (hs._tx_pending, hs._pending_answer_at,
                               'session-data-pair-nak')
        hs._took_data_nak()
    return phy.host_level(hs.bw, hs._tx_pending[2])


def receiver(bw, start):
    hs, io = _connected(bw=bw, role='responder')
    hs._link_high_speed = True
    hs._expect_level = start
    hs._lv_owed = hs._lv_prompt = True
    return hs, io


def rx_nak(hs, kind):
    if kind == 'M':
        assert hs._nak_owed_over()
    else:
        hs._irs_nak(phy.LevelRead(hs._expect_level, (), None, 1))
    return hs._expect_level


@pytest.mark.parametrize('bw,start,kinds,levels', CHAINS)
def test_captured_sender_ladder(bw, start, kinds, levels, render):
    hs, io = sender(bw, start)
    before = phy.vara_payload(hs._tx_pending[0], caller=hs.caller,
                              body_len=len(hs._tx_pending[0])) + b''.join(hs._txq)
    got = []
    for kind in kinds:
        got.append(tx_nak(hs, kind))
        body = hs._tx_pending[0]
        assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) + b''.join(hs._txq) == before
    assert tuple(got) == levels, io.msgs


@pytest.mark.parametrize('bw,start,kinds,levels', CHAINS)
def test_captured_receiver_ladder(bw, start, kinds, levels):
    hs, io = receiver(bw, start)
    assert tuple(rx_nak(hs, kind) for kind in kinds) == levels, io.msgs


TOLERANCE = [
    ('500', 13, 'N', (8, 9, 10, 12), 9, 7),
    ('500', 13, 'M', (8, 9, 10, 11), 9, 7),
    ('500', 8, 'N', (5, 6, 7), 6, 4),
    ('500', 8, 'M', (5, 6, 7), 6, 4),
    ('500', 4, 'M', (1, 2), 1, 1),
    ('2300', 16, 'N', (12, 13, 14, 15), 13, 10),
    ('2300', 16, 'M', (12, 13, 14), 13, 10),
    ('2300', 8, 'N', (5, 6, 7), 6, 4),
    ('2300', 8, 'M', (5, 6, 7), 6, 4),
]


@pytest.mark.parametrize('bw,start,kind,keyed,expected,next_level', [
    (bw, start, kind, keyed, expected, next_level)
    for bw, start, kind, levels, expected, next_level in TOLERANCE for keyed in levels
])
def test_tolerance_reads_only_its_own_level(bw, start, kind, keyed, expected, next_level):
    hs, io = receiver(bw, start)
    rx_nak(hs, kind)
    assert hs._read_levels() == ((expected,) if expected > 4 else ())
    body = phy.vara_body(b'pin', hs.caller, level=4,
                         body_len=phy.body_size(bw, phy.level_key(bw, keyed)))
    if keyed > 4:
        hs._level_read = phy.LevelRead(keyed, (body,), None, 1)
        got = hs._peer_data_over(np.zeros(0))
        hs._level_read = None
    else:
        audio = VA.OF.data_over_tx(body, bw=bw, level=phy.level_key(bw, keyed))
        got = hs._peer_data_over(np.concatenate([np.zeros(4800), audio, np.zeros(4800)]))
    assert bool(got) == (keyed == expected), io.msgs
    if keyed != expected:
        assert rx_nak(hs, 'M') == next_level


@pytest.mark.parametrize('bw,start,kind,keyed,expected,next_level', TOLERANCE)
def test_sender_continues_from_peer_expectation_after_off_level_resend(
        bw, start, kind, keyed, expected, next_level, render):
    for wrong in keyed:
        if wrong == expected:
            continue
        hs, io = sender(bw, start)
        tx_nak(hs, kind)
        body, over, _ = hs._tx_pending
        delivery = phy.vara_payload(body, caller=hs.caller, body_len=len(body)) + b''.join(hs._txq)
        key = phy.level_key(bw, wrong)
        take = phy.body_size(bw, key) - 1
        hs._tx_pending = (phy.vara_body(delivery[:take], hs.caller, tail=0x95,
                                       body_len=take + 1), over, key)
        hs._txq = [delivery[i:i + take] for i in range(take, len(delivery), take)]
        if not hs._txq or len(hs._txq[-1]) == take:
            hs._txq.append(b'')
        hs._tx_recovery_levels = [key] * len(hs._txq)
        assert tx_nak(hs, 'M') == next_level, io.msgs


@pytest.mark.parametrize('bw,start,expected', [('500', 13, 9), ('2300', 16, 13), ('2750', 17, 14)])
@pytest.mark.parametrize('kind', ['M', 'N'])
def test_refused_retry_does_not_spend_a_ladder_step(bw, start, expected, kind, render):
    hs, io = sender(bw, start)
    pending, queue, chain = hs._tx_pending, list(hs._txq), hs._tx_resend
    io.tx_went_out = lambda: False
    tx_nak(hs, kind)
    assert (hs._tx_pending, hs._txq, hs._tx_resend) == (pending, queue, chain)
    assert hs._tx_retries == 0
    io.tx_went_out = lambda: True
    assert tx_nak(hs, kind) == expected


@pytest.mark.parametrize('bw,start,expected', [('500', 13, 9), ('2300', 16, 13), ('2750', 17, 14)])
@pytest.mark.parametrize('kind', ['M', 'N'])
def test_refused_receiver_nak_and_silence_do_not_spend_a_step(bw, start, expected, kind):
    hs, io = receiver(bw, start)
    io.tx_went_out = lambda: False
    if kind == 'M':
        assert not hs._nak_owed_over()
    else:
        rx_nak(hs, kind)
    assert hs._expect_level == start and hs._rx_resend is None
    io.tx_went_out = lambda: True
    assert rx_nak(hs, kind) == expected
    chain = hs._rx_resend
    for _ in range(3):
        hs._fall_expected()
    assert hs._expect_level == expected and hs._rx_resend == chain


@pytest.mark.parametrize('bw,start,expected,next_first', [
    ('500', 13, 9, 7), ('2300', 16, 13, 10), ('2750', 17, 14, 11)])
def test_continue_starts_a_new_chain(bw, start, expected, next_first, render):
    hs, _ = sender(bw, start)
    assert tx_nak(hs, 'N') == expected
    hs._continue_frame = (hs._tx_pending, 0)
    hs._took_over_continue()
    assert tx_nak(hs, 'N') == next_first


@pytest.mark.parametrize('bw,start,expected', [('500', 13, 9), ('2300', 16, 13), ('2750', 17, 14)])
def test_missing_short_final_resends_at_the_same_ladder(bw, start, expected, render):
    hs, io = _connected(bw=bw)
    hs._link_high_speed = True
    hs._start_at(start)
    hs.turn = VA._TURN_OURS
    hs.send(b'last bytes')
    assert tx_nak(hs, 'M') == expected, io.msgs
    body = hs._tx_pending[0]
    assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) == b'last bytes'


@pytest.mark.parametrize('bw,start,wrong', [('500', 13, 10), ('2300', 16, 12), ('2750', 17, 15)])
def test_expected_reader_ignores_an_off_level_decode(bw, start, wrong, monkeypatch):
    hs, io = receiver(bw, start)
    rx_nak(hs, 'N')
    expected, keyed = hs._expect_level, len(io.sent)
    asked = []

    def decode(samples, bw, level, *args, **kwargs):
        assert kwargs.get('expect_pair') == (hs._peer_pairs_500 if bw == '500' else None)
        asked.append(level)
        return phy.LevelRead(wrong, (b'wrong',), None, 1)

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    hs._lv_armed = True
    assert not hs._read_expected_level(np.zeros(VA._EXPECT_WINDOW[bw]))
    assert asked == [expected]
    assert hs._expect_level == expected and len(io.sent) == keyed and io.host == []


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_low_level_crc_failure_cancels_partial_ack_and_keeps_retry_bound(bw):
    hs, _ = receiver(bw, 4)
    hs._held_answer = (b'prefix',)
    assert hs._undecoded_over()
    assert hs._held_answer is None and hs._expect_level == 3
    assert hs._undecoded == 1
    closed = []
    hs.disconnect = lambda: closed.append(True)
    hs._undecoded = VA._OVER_NAK_MAX
    assert hs._undecoded_over()
    assert closed == [True]
