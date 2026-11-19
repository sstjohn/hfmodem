# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Climb-aware VARA retries, lower retry admission and recovery pacing."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA
from .test_resend_ladders_1020 import (
    clock as clock, render as render, receiver, rx_nak, sender, tx_nak,
)


CLIMBS = [
    ('2300', (4, 8, 12, 14, 15, 16), 'NNN', (13, 12, 10)),
    ('2300', (4, 8, 12, 14, 15, 16), 'MMM', (13, 10, 7)),
    ('2750', (4, 8, 12, 15, 16, 17), 'NNN', (14, 13, 11)),
    ('2750', (4, 8, 12, 15, 16, 17), 'MMM', (14, 11, 8)),
    *[(bw, (4, 8), kind, (6,))
      for bw in ('2300', '2750') for kind in ('N', 'M')],
]


def _continue(hs, level):
    previous = phy.host_level(hs.bw, hs._tx_pending[2])
    hs._continue_frame = (hs._tx_pending, VA._STEP_FRAME[level - previous])
    hs._took_over_continue()
    assert phy.host_level(hs.bw, hs._tx_pending[2]) == level


def _body(hs, level, payload=b'pin'):
    return phy.vara_body(payload, hs.caller, level=4,
                         body_len=phy.body_size(hs.bw, phy.level_key(hs.bw, level)))


def _accept(hs, level, index):
    """Use the decoded-body entry to track climbs independently of PHY costs."""
    size = phy.body_size(hs.bw, phy.level_key(hs.bw, level))
    body = phy.vara_body(bytes([index]) * (size - 1), hs.caller,
                         body_len=size, tail=0x95)
    hs._rx_read = (level, None)
    assert hs._accept_data_over(np.zeros(0), [body], False,
        (hs._peer_run_500, hs._peer_paired_500, hs._peer_over_state))


@pytest.mark.parametrize('bw,climb,kinds,expected', CLIMBS)
def test_bench_sender_climbs(bw, climb, kinds, expected, render):
    hs, _ = sender(bw, climb[0])
    for level in climb[1:]:
        _continue(hs, level)
    assert hs._tx_climb == (climb[-1] - 4, climb[-1])
    assert tuple(tx_nak(hs, kind) for kind in kinds) == expected
    assert hs._tx_climb == (0, expected[-1])


@pytest.mark.parametrize('bw,climb,kinds,expected', CLIMBS)
def test_bench_receiver_climbs(bw, climb, kinds, expected):
    hs, _ = receiver(bw, 4)
    for index, level in enumerate(climb):
        _accept(hs, level, index)
        assert hs._rx_climb == (level - 4, level)
    hs._expect_level = climb[-1]
    hs._lv_owed = hs._lv_prompt = True
    hs._rx_window.event('G')
    assert tuple(rx_nak(hs, kind) for kind in kinds) == expected
    assert hs._rx_climb == (0, None)


@pytest.mark.parametrize('role', ['sender', 'receiver'])
@pytest.mark.parametrize('kind,expected', [('N', 13), ('M', 12)])
def test_short_climb_after_a_down_step(role, kind, expected, render):
    if role == 'sender':
        hs, _ = sender('2300', 13)
        _continue(hs, 12)
        assert hs._tx_climb == (0, 12)
        _continue(hs, 14)
        assert hs._tx_climb == (2, 14)
        assert tx_nak(hs, kind) == expected
    else:
        hs, _ = receiver('2300', 13)
        for index, level in enumerate((13, 12, 14)):
            _accept(hs, level, index)
        assert hs._rx_climb == (2, 14)
        hs._expect_level = 14
        hs._lv_owed = hs._lv_prompt = True
        hs._rx_window.event('G')
        assert rx_nak(hs, kind) == expected


@pytest.mark.parametrize('bw,level,run,missing,expected', [
    ('2300', 12, 0, False, 10), ('2300', 16, 0, False, 15),
    ('2300', 16, 4, False, 13), ('2300', 16, 6, False, 13),
    ('2300', 14, 2, True, 12), ('2750', 17, 0, False, 16),
    ('2750', 17, 4, False, 14), ('2750', 17, 6, True, 14),
    ('2750', 10, 0, False, 8),
])
def test_fall_and_single_pass_over(bw, level, run, missing, expected):
    assert VA._fall(bw, level, run, missing) == expected


@pytest.mark.parametrize('bw,start,level', [
    ('2300', 16, 13), ('2300', 16, 12),
    ('2750', 17, 14), ('2750', 17, 13),
])
def test_clean_lower_retry_is_read(bw, start, level):
    hs, _ = receiver(bw, start)
    rx_nak(hs, 'N')
    body = _body(hs, level)
    hs._level_read = phy.LevelRead(level, (body,), None, 1)
    assert hs._peer_data_over(np.zeros(0)) == [body]
    assert hs._rx_read[0] == level
    assert hs._rx_resend is None
    assert not hs._lower_retry(level)


@pytest.mark.parametrize('bw,start,level,frames,located,high_speed', [
    (bw, start, level, frames, located, high_speed)
    for bw, start in [('2300', 16), ('2750', 17)]
    for level, frames, located, high_speed in [
        (12, False, 1, True), (12, True, 2, True),
        (start, True, 1, True), (start + 1, True, 1, True),
        (12, True, 1, False),
    ]
] + [('500', 13, 10, True, 1, True)])
def test_off_prediction_retry_must_be_clean_complete_and_eligible(
        bw, start, level, frames, located, high_speed):
    hs, io = receiver(bw, start)
    hs._link_high_speed = high_speed
    rx_nak(hs, 'N')
    # Earlier requested levels are eligible independently of decoder candidates.
    hs._rx_retry_rates.discard(start)
    body = _body(hs, min(level, phy.TOP[bw]))
    chain = hs._rx_resend
    hs._level_read = phy.LevelRead(level, (body,) if frames else (), None, located)
    assert not hs._peer_data_over(np.zeros(0))
    assert io.host == [] and hs._rx_resend == chain


@pytest.mark.parametrize('level', [2, 3])
def test_clean_base_retry_below_a_base_prediction(level):
    hs, _ = receiver('2300', 8)
    assert tuple(rx_nak(hs, 'M') for _ in range(2)) == (6, 4)
    assert hs._read_levels() == () and not hs._lv_armed
    body = _body(hs, level)
    audio = VA.OF.data_over_tx(body, bw=hs.bw, level=phy.level_key(hs.bw, level))
    assert hs._peer_data_over(np.concatenate([
        np.zeros(4800), audio, np.zeros(4800)])) == [body]
    assert hs._rx_read[0] == level and hs._rx_resend is None


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_base_first_failure_refuses_a_clean_lower_retry(bw):
    hs, io = receiver(bw, 4)
    rx_nak(hs, 'N')
    assert hs._rx_resend == (4, (False,), 0)
    body = _body(hs, 1)
    assert not hs._read_matches(phy.LevelRead(1, (body,), None, 1))
    chain = hs._rx_resend
    audio = VA.OF.data_over_tx(body, bw=bw, level=phy.level_key(bw, 1))
    assert not hs._peer_data_over(np.concatenate([
        np.zeros(4800), audio, np.zeros(4800)]))
    assert io.host == [] and hs._rx_resend == chain


@pytest.mark.parametrize('learned,candidates,repeat', [
    (None, (13, 12), 12), (True, (12, 13), 13)])
def test_missed_nak_retains_both_previous_ladder_candidates(learned, candidates, repeat):
    hs, _ = receiver('2300', 16)
    hs._rx_climb, hs._peer_steep = (12, 16), learned
    rx_nak(hs, 'N')
    assert hs._read_levels() == candidates
    rx_nak(hs, 'M')
    assert hs._read_levels() == (10,)
    body = _body(hs, repeat)
    hs._level_read = phy.LevelRead(repeat, (body,), None, 1)
    assert hs._peer_data_over(np.zeros(0)) == [body]
    assert hs._rx_read[0] == repeat and hs._rx_resend is None


@pytest.mark.parametrize('partial', [False, True])
def test_base_lower_retry_requires_every_located_frame_clean(partial, monkeypatch):
    hs, io = receiver('2300', 8)
    assert tuple(rx_nak(hs, 'M') for _ in range(2)) == (6, 4)
    body = _body(hs, 3)
    audio = VA.OF.data_over_tx(body, bw=hs.bw, level=phy.level_key(hs.bw, 3))
    window_reads = hs._window_reads
    closed = []

    def reads(samples, found):
        clean = window_reads(samples, found)
        assert clean and all(item[3].crc_ok for item in clean)
        lv, hits, of, frame, start, end = clean[-1]
        failed = (lv, hits, of, None, end, end + 1)
        return clean + [failed] if partial else [failed]

    monkeypatch.setattr(hs, '_window_reads', reads)
    monkeypatch.setattr(hs, '_note_close_level', lambda bodies: closed.append(bodies))
    chain = hs._rx_resend
    assert not hs._peer_data_over(np.concatenate([
        np.zeros(4800), audio, np.zeros(4800)]))
    assert closed == [] and io.host == [] and hs._rx_resend == chain


@pytest.mark.parametrize('bw,start', [('2300', 16), ('2750', 17)])
def test_failed_level_retains_the_existing_earlier_request_exception(bw, start):
    hs, _ = receiver(bw, start)
    rx_nak(hs, 'N')
    read = phy.LevelRead(start, (_body(hs, start),), None, 1)
    assert not hs._lower_retry(start)
    assert start in hs._rx_retry_rates and hs._read_matches(read)


def test_decoder_tries_a_second_lower_candidate(monkeypatch):
    hs, io = receiver('2300', 16)
    assert rx_nak(hs, 'N') == 13
    assert hs._read_levels() == (13, 12)
    body, asked = _body(hs, 12), []

    def decode(samples, bw, level, *args, **kwargs):
        asked.append(level)
        return None if level == 13 else phy.LevelRead(12, (body,), None, 1)

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    hs._lv_armed = True
    assert hs._read_expected_level(np.zeros(VA._EXPECT_WINDOW[hs.bw]))
    assert asked == [13, 12]
    assert io.host == [b'pin']
    assert hs._rx_read[0] == 12


def test_recovered_retry_holds_the_failed_level_for_four_answers():
    hs, _ = receiver('2300', 16)
    rx_nak(hs, 'N')
    hs._rx_read = (13, 30.0)
    hs._note_close_level([_body(hs, 13)])
    answers, overs = [], []
    for index, level in enumerate((13, 14, 15, 15, 15)):
        _accept(hs, level, index)
        overs.append(hs._peer_over)
        hs._rx_read, hs._peer_hint = (level, 30.0), None
        frame = hs._irs_step()
        assert hs._irs_step() == frame
        answers.append(level + VA._LEVEL_STEP[frame])
    assert overs == list(range(overs[0], overs[0] + 5))
    assert answers == [14, 15, 15, 15, 16]
    assert hs._rx_hold is None


@pytest.mark.parametrize('recovered', [10, 13])
def test_first_recovery_answer_caps_the_climb(recovered):
    hs, _ = receiver('2300', 16)
    rx_nak(hs, 'N')
    hs._rx_read, hs._peer_hint = (recovered, 30.0), None
    if recovered == 10:
        assert VA._LEVEL_STEP[VA._irs_frame(10, 30.0, phy.TOP['2300'], '2300')] > 1
    assert VA._LEVEL_STEP[hs._irs_step()] == 1
    assert hs._rx_resend is None
    assert hs._rx_hold == (16, hs._peer_over)


@pytest.mark.parametrize('quality,closing,expected', [(0.0, False, -1),
    (None, False, 0), (30.0, True, 0)])
def test_recovery_does_not_change_down_hold_or_close(quality, closing, expected):
    hs, _ = receiver('2300', 16)
    rx_nak(hs, 'N')
    hs._rx_read, hs._peer_hint = (13, quality), None
    assert VA._LEVEL_STEP[hs._irs_step(closing=closing)] == expected


@pytest.mark.parametrize('bw,high_speed,start', [('500', True, 13), ('2300', False, 16)])
def test_recovery_hold_only_applies_to_wide_high_speed(bw, high_speed, start):
    hs, _ = receiver(bw, start)
    hs._link_high_speed = high_speed
    rx_nak(hs, 'N')
    hs._rx_read, hs._peer_hint = (10, 30.0), None
    hs._note_close_level([_body(hs, 10)])
    assert hs._rx_hold is None
    assert hs._irs_step() == VA._irs_frame(10, 30.0, phy.RX_CEILING[bw], bw)
    assert hs._rx_hold is None


def test_missing_retry_re_meets_the_491_ladder(render):
    hs, _ = sender('2300', 16)
    assert hs._tx_climb == (12, 16)
    assert tx_nak(hs, 'N') == 13
    assert tx_nak(hs, 'M') == 10
    assert VA._resend_level('2300', 12, (True,), 0) == 10


@pytest.mark.parametrize('entry', ['originate', 'listen', 'respond', 'reconnect', 'relisten'])
def test_link_start_resets_climbs_and_hold(entry):
    hs, _ = receiver('2300', 16)
    hs._tx_climb, hs._rx_climb, hs._rx_hold = (12, 16), (8, 12), (16, 5)
    hs._peer_steep = True
    if entry in ('reconnect', 'relisten'):
        hs.state = VA.VaraState.DISCONNECTED
    if entry in ('originate', 'reconnect'):
        assert hs.originate('KC9GHZ')
    elif entry in ('listen', 'relisten'):
        hs.state = VA.VaraState.LISTENING
    else:
        hs.state = VA.VaraState.LISTENING
        hs._tx_climb, hs._rx_climb, hs._rx_hold = (12, 16), (8, 12), (16, 5)
        hs._peer_steep = True
        hs.state = VA.VaraState.CONNECTING
    assert hs._tx_climb == hs._rx_climb == (0, 4)
    assert hs._rx_hold is None
    assert hs._peer_steep is None


def test_delivery_end_keeps_climb_and_hold_but_ends_retry_window():
    hs, _ = receiver('2300', 16)
    rx_nak(hs, 'N')
    hs._rx_climb = (2, 13)
    body = _body(hs, 13)
    hs._note_close_level([body])
    assert hs._rx_resend is None and not hs._lower_retry(12)
    assert hs._rx_climb == (2, 13)
    assert hs._rx_hold == (16, None)
    hs._rx_read, hs._peer_hint = (13, 30.0), None
    hs._irs_step()
    hold = hs._rx_hold
    hs._note_close_level([body])
    assert hs._rx_climb == (2, 13) and hs._rx_hold == hold


@pytest.mark.parametrize('start,kinds,expected', [(4, 'MMM', (2, 1, 1)),
    (16, 'NNN', (13, 12, 10)), (16, 'MMM', (13, 10, 7))])
def test_speed_limited_receiver_keeps_table(start, kinds, expected):
    hs, _ = receiver('2300', start)
    hs._link_high_speed = False
    assert tuple(rx_nak(hs, kind) for kind in kinds) == expected
    assert hs._rx_resend[2] is None
    assert hs._read_levels() == ((expected[-1],) if expected[-1] > 4 else ())


@pytest.mark.parametrize('bw,level,quality,ceiling,frame', [
    ('2300', 4, -6.0, 16, 5), ('2300', 4, -1.0, 16, 4),
    ('2300', 4, -2.0, 16, 6), ('2300', 3, -4.0, 16, 6),
    ('2300', 4, None, 4, 0), ('500', 3, -6.0, 13, 5),
    ('500', 2, -3.0, 13, 6), ('500', 1, -4.0, 13, 0),
])
def test_index_steps_are_unchanged(bw, level, quality, ceiling, frame):
    assert VA._index_step(level, quality, ceiling, bw) == frame


@pytest.mark.parametrize('kind', ['N', 'M'])
def test_refused_keying_keeps_the_sender_climb(kind, render):
    hs, io = sender('2300', 16)
    climb, chain = hs._tx_climb, hs._tx_resend
    io.tx_went_out = lambda: False
    tx_nak(hs, kind)
    assert hs._tx_climb == climb and hs._tx_resend == chain


@pytest.mark.parametrize('climb,level,expected', [
    ((0, 4), 16, (12, 16)), ((12, 16), 16, (12, 16)),
    ((12, 16), 12, (0, 12)), ((0, 12), 14, (2, 14)),
    ((0, None), 13, (0, 13)),
])
def test_climb_tracks_up_down_and_repeated_reads(climb, level, expected):
    assert VA._climbed(climb, level) == expected


RETRY_CANDIDATES = [
    ('2300', 16, 'N', 13, 12),
    ('2300', 16, 'M', 13, 12),
    ('2300', 16, 'NM', 10, 10),
    ('2300', 16, 'MM', 10, 10),
    ('2300', 16, 'MN', 12, 10),
    ('2300', 16, 'NN', 12, 10),
    ('2300', 16, 'MMM', 7, 7),
    ('2750', 17, 'N', 14, 13),
    ('2750', 17, 'M', 14, 13),
    ('2750', 17, 'NM', 11, 11),
    ('2750', 17, 'MM', 11, 11),
    ('2750', 17, 'MN', 13, 11),
    ('2750', 17, 'NN', 13, 11),
    ('2750', 17, 'MMM', 8, 8),
    ('2750', 16, 'N', 13, 11),
    ('2750', 16, 'M', 13, 11),
    ('2750', 16, 'NM', 11, 8),
    ('2750', 16, 'MM', 11, 8),
    ('2750', 16, 'MN', 11, 10),
    ('2750', 16, 'NN', 11, 10),
    ('2750', 16, 'MMM', 8, 6),
]


@pytest.mark.parametrize('bw,start,kinds,normal,steep,level', [
    (bw, start, kinds, normal, steep, level)
    for bw, start, kinds, normal, steep in RETRY_CANDIDATES
    for level in dict.fromkeys((normal, steep, start - 1))
])
@pytest.mark.parametrize('learned', [None, True])
def test_expected_reader_decodes_known_retry_ladders(
        bw, start, kinds, normal, steep, level, learned, monkeypatch):
    hs, io = receiver(bw, start)
    hs._peer_steep = learned
    for kind in kinds:
        rx_nak(hs, kind)
    prediction = steep if learned else normal
    candidates = tuple(level for level in dict.fromkeys((prediction, normal, steep))
                       if level > 4)
    assert hs._expect_level == prediction
    assert hs._read_levels() == candidates
    body, asked = _body(hs, level), []

    def decode(samples, bw, candidate, *args, **kwargs):
        asked.append(candidate)
        return phy.LevelRead(level, (body,), None, 1) if candidate == level else None

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    hs._lv_armed = True
    taken = hs._read_expected_level(np.zeros(hs._expected_window_samples()))
    assert asked[0] == prediction
    if level in candidates:
        assert taken
        assert asked == list(candidates[:candidates.index(level) + 1])
        assert io.host == [b'pin'] and hs._rx_read[0] == level
    else:
        assert not taken and asked == list(candidates)
        assert level not in asked and io.host == []
        chain = hs._rx_resend
        hs._level_read = phy.LevelRead(level, (body,), None, 1)
        assert not hs._peer_data_over(np.zeros(0))
        assert io.host == [] and hs._rx_resend == chain


def test_base_prediction_does_not_arm_expected_reader(monkeypatch):
    hs, io = receiver('2300', 8)
    assert tuple(rx_nak(hs, 'M') for _ in range(2)) == (6, 4)
    assert hs._read_levels() == () and not hs._lv_armed
    body, asked = _body(hs, 6), []

    def decode(samples, bw, level, *args, **kwargs):
        asked.append(level)
        return phy.LevelRead(6, (body,), None, 1) if level == 6 else None

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    assert not hs._read_expected_level(np.zeros(hs._expected_window_samples()))
    assert asked == [] and io.host == []


@pytest.mark.parametrize('bw,high_speed,start', [
    ('500', True, 13), ('2300', False, 16), ('2750', False, 17)])
def test_recovery_ladder_candidates_exclude_narrow_and_speed_limited_links(
        bw, high_speed, start, monkeypatch):
    hs, _ = receiver(bw, start)
    hs._link_high_speed = high_speed
    rx_nak(hs, 'N')
    expect, asked = hs._expect_level, []
    assert hs._read_levels() == (expect,)

    def decode(samples, bw, level, *args, **kwargs):
        asked.append(level)
        return None

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    hs._lv_armed = True
    assert not hs._read_expected_level(np.zeros(hs._expected_window_samples()))
    assert asked == [expect]


@pytest.mark.parametrize('bw,start,kinds', [
    ('2300', 16, 'N'), ('2750', 17, 'M'), ('2750', 16, 'MM')])
def test_expected_window_holds_the_longest_candidate(bw, start, kinds, monkeypatch):
    hs, _ = receiver(bw, start)
    for kind in kinds:
        rx_nak(hs, kind)
    candidates = hs._read_levels()
    longest = candidates[-1]
    floor = VA._EXPECT_WINDOW[bw]
    calls = []

    def burst(bw, level, frames):
        calls.append((level, frames))
        return floor + 100 if level == longest else 1

    monkeypatch.setattr(phy, 'burst_samples', burst)
    monkeypatch.setattr(hs, '_receive_frame_count', lambda level: 2 if level == longest else 1)
    assert hs._expected_window_samples() == floor + 100 + VA._EXPECT_LEAD + VA._EXPECT_SLACK
    assert set(calls) == {(level, 2 if level == longest else 1)
                          for level in (hs._expect_level, *candidates)}


def test_unarmed_recovery_reader_leaves_base_scan_open(monkeypatch):
    hs, io = receiver('2300', 8)
    assert tuple(rx_nak(hs, 'M') for _ in range(2)) == (6, 4)
    assert not hs._lv_armed and hs._read_levels() == ()
    asked = []

    def decode(samples, bw, level, *args, **kwargs):
        asked.append(level)
        return None

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    body = _body(hs, 2)
    audio = VA.OF.data_over_tx(body, bw=hs.bw, level=phy.level_key(hs.bw, 2))
    hs._stream_over(np.concatenate([np.zeros(4800), audio, np.zeros(3 * VA.MK.FS)]))
    assert asked == []
    assert io.host == [b'pin'] and hs._rx_read[0] == 2, io.msgs


def test_unpredicted_failed_candidate_cannot_issue_a_nak():
    hs, io = receiver('2300', 8)
    assert tuple(rx_nak(hs, 'M') for _ in range(2)) == (6, 4)
    chain, sent = hs._rx_resend, len(io.sent)
    hs._irs_nak(phy.LevelRead(6, (), None, 1))
    assert hs._expect_level == 4 and hs._rx_resend == chain
    assert len(io.sent) == sent


def _stock_retry_read(bw, start, failures, steep, keyed):
    """A stock receiver reads only its own model's predicted retry."""
    level = start
    for index, missing in enumerate(failures):
        run = start - 4 if index == 0 else 0
        fall = 4 if steep and run >= 12 else 3 if run >= 6 else 2 if missing else 1
        level = max(1, level - fall)
        if level > 1 and level in VA._PASSED_OVER[bw]:
            level -= 1
    return keyed == level


@pytest.mark.parametrize('bw,start', [('2300', 16), ('2750', 17)])
@pytest.mark.parametrize('first', ['N', 'M'])
@pytest.mark.parametrize('model,learned,lost', [
    (False, None, 0), (True, None, 1), (True, True, 0), (False, False, 0)])
def test_sender_recovers_against_stock_receiver_models(bw, start, first, model, learned, lost, render):
    hs, _ = sender(bw, start)
    assert hs._peer_steep is None
    hs._peer_steep = learned
    failures = (first == 'M',)
    keyed = tx_nak(hs, first)
    read = _stock_retry_read(bw, start, failures, model, keyed)
    assert read == (lost == 0)
    if not read:
        failures += (True,)
        keyed = tx_nak(hs, 'M')
        assert _stock_retry_read(bw, start, failures, model, keyed)
    assert hs._tx_requested == keyed


@pytest.mark.parametrize('run,retry,learned,admitted', [
    (12, 12, True, True), (12, 13, False, True), (12, 11, None, False),
    (11, 12, None, False)])
@pytest.mark.parametrize('entry', ['close', 'step'])
def test_recovered_first_retry_learns_peer_model(run, retry, learned, admitted, entry, render):
    hs, _ = receiver('2300', 16)
    hs._rx_climb = (run, 16)
    rx_nak(hs, 'N')
    if entry == 'close':
        body = _body(hs, retry)
        hs._level_read = phy.LevelRead(retry, (body,), None, 1)
        assert hs._peer_data_over(np.zeros(0)) == ([body] if admitted else [])
    else:
        hs._rx_read = (retry, 30.0)
        hs._irs_step()
    assert hs._peer_steep is learned
    if learned is True:
        hs._start_at(16)
        hs.turn = VA._TURN_OURS
        hs.send(bytes(12000))
        assert hs._tx_climb == (12, 16)
        assert tx_nak(hs, 'N') == hs._tx_requested == 12


@pytest.mark.parametrize('bw,start,normal,steep', [
    ('2300', 16, (13, 12, 10), (12, 10, 9)),
    ('2750', 17, (14, 13, 11), (13, 11, 10))])
@pytest.mark.parametrize('learned', [None, False, True])
def test_receiver_prediction_uses_learned_model(bw, start, normal, steep, learned):
    hs, _ = receiver(bw, start)
    hs._peer_steep = learned
    assert tuple(rx_nak(hs, 'N') for _ in range(3)) == (steep if learned else normal)


@pytest.mark.parametrize('bw,start,normal,steep', [
    ('2300', 16,
     ((13, 10), (13, 12), (13, 10), (13, 12), (13, 10, 7), (13, 10, 7)),
     ((12, 10), (12, 10), (12, 10), (12, 10), (12, 10, 7), (12, 10, 7))),
    ('2750', 17,
     ((14, 11), (14, 13), (14, 11), (14, 13), (14, 11, 8), (14, 11, 8)),
     ((13, 11), (13, 11), (13, 11), (13, 11), (13, 11, 8), (13, 11, 8)))])
@pytest.mark.parametrize('learned', [None, False, True])
def test_sender_model_chains_preserve_bench_default(bw, start, normal, steep, learned, render):
    for kinds, expected in zip(('NM', 'NN', 'MM', 'MN', 'NMM', 'MMM'),
                               steep if learned else normal):
        hs, _ = sender(bw, start)
        hs._peer_steep = learned
        assert tuple(tx_nak(hs, kind) for kind in kinds) == expected
        assert hs._tx_requested == expected[-1]


@pytest.mark.parametrize('kinds,expected', [
    ('NM', (13, 11)), ('MM', (13, 11)), ('NMM', (13, 11, 8)),
    ('MN', (13, 11)), ('NNM', (13, 11, 8))])
def test_unlearned_bench_chain_at_2750_level_16(kinds, expected, render):
    hs, _ = sender('2750', 16)
    assert tuple(tx_nak(hs, kind) for kind in kinds) == expected
    assert hs._tx_requested == expected[-1]


@pytest.mark.parametrize('bw,start,run,expected', [
    ('2300', 16, 11, 13), ('2300', 16, 12, 12),
    ('2750', 16, 12, 11), ('2750', 17, 13, 13)])
def test_steep_fall_threshold(bw, start, run, expected):
    assert VA._fall(bw, start, run, False, steep=True) == expected


@pytest.mark.parametrize('bw,high_speed,failures,run', [
    ('500', True, (False,), 12), ('2300', False, (False,), 12),
    ('2300', True, (False, True), 12), ('2300', True, (False,), None)])
def test_ineligible_recovery_does_not_learn(bw, high_speed, failures, run):
    hs, _ = receiver(bw, 16 if bw != '500' else 13)
    hs._link_high_speed = high_speed
    hs._rx_resend = (16 if bw != '500' else 13, failures, run)
    hs._rx_read = (12, None)
    hs._note_recovery()
    assert hs._peer_steep is None


@pytest.mark.parametrize('later_clean', [True, False])
def test_a_lost_candidate_read_does_not_end_the_search(later_clean, monkeypatch):
    hs, _ = receiver('2300', 16)
    rx_nak(hs, 'N')
    assert hs._read_levels() == (13, 12)
    lost = phy.LevelRead(13, (), None, 1)
    clean = phy.LevelRead(12, (_body(hs, 12),), None, 1)
    asked, taken = [], []

    def decode(samples, bw, level, *args, **kwargs):
        asked.append(level)
        return lost if level == 13 else clean if later_clean else replace(lost, level=12)

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(hs, '_track_emission', lambda: None)
    monkeypatch.setattr(hs, '_take_expected_read', lambda read, *a, **k: taken.append(read) or True)
    hs._lv_armed = True
    assert hs._read_expected_level(np.zeros(hs._expected_window_samples()))
    assert asked == [13, 12]
    assert taken == [clean if later_clean else lost]


@pytest.mark.parametrize('bw,start,level', [
    ('2300', 16, 5), ('2300', 16, 6), ('2300', 16, 7),
    ('2300', 16, 8), ('2300', 16, 9), ('2300', 16, 10),
    ('2300', 16, 11), ('2300', 16, 14), ('2300', 16, 15),
    ('2750', 17, 5), ('2750', 17, 6), ('2750', 17, 7),
    ('2750', 17, 8), ('2750', 17, 9), ('2750', 17, 10),
    ('2750', 17, 11), ('2750', 17, 12), ('2750', 17, 15), ('2750', 17, 16),
])
def test_clean_off_candidate_level_read_is_refused(bw, start, level):
    hs, io = receiver(bw, start)
    rx_nak(hs, 'N')
    assert level > 4 and level < start
    assert level not in hs._read_levels() and level not in hs._rx_retry_rates
    chain = hs._rx_resend
    hs._level_read = phy.LevelRead(level, (_body(hs, level),), None, 1)
    assert not hs._peer_data_over(np.zeros(0))
    assert io.host == [] and hs._rx_resend == chain


@pytest.mark.parametrize('first,second,admitted_count', [(12, 13, 2), (13, 12, 1)])
def test_first_unambiguous_model_is_learned_once(first, second, admitted_count):
    hs, io = receiver('2300', 16)
    for retry in (first, second):
        hs._rx_climb, hs._expect_level = (12, 16), 16
        hs._advance_resend(missing=False)
        hs._rx_read = (retry, 30.0)
        hs._note_close_level([_body(hs, retry)])
    assert hs._peer_steep is (first == 12)
    learned = [msg for msg in io.msgs if 'peer retry model learned:' in msg]
    assert len(learned) == 1
    assert ('steep' if first == 12 else 'bench (4.9.0)') in learned[0]
    assert 'from level 16' in learned[0]
    admitted = [msg for msg in io.msgs if 'retry admitted at level' in msg]
    assert len(admitted) == admitted_count


@pytest.mark.parametrize('start,description', [(16, 'levels 13, 12'), (15, 'level 12')])
def test_recovery_report_names_every_candidate(start, description):
    hs, io = receiver('2300', start)
    rx_nak(hs, 'N')
    hs._recovery_report('requested')
    assert f'waiting for a valid retry at {description} or' in io.msgs[-1]
