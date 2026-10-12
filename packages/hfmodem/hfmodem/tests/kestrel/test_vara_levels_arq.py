# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The high speed capable gear-shift, against a fake peer: the ISS obeys the step
frames 0/6/4/7/8/5, falls by our table, doubles at the top; the IRS asks by
the decoder's quality within RX_CEILING; the confirm starts the callee at
4/5/6 (docs/protocols/vara/evidence/ACK-LEVEL-REQUEST.md, GEARSHIFT-OBSERVED.md).
"""
import time
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.rx import varahf2300 as RX
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import _fall, _finish_level, _irs_frame
from hfmodem.tests import evidence
from .test_data_over_gate import _connected, _on_air


@pytest.fixture
def ceilings(monkeypatch):
    def raise_to(tx=None, rx=None):
        for table, value in ((phy.TX_CEILING, tx), (phy.RX_CEILING, rx)):
            if value is not None:
                monkeypatch.setitem(table, '2300', value)
                monkeypatch.setitem(table, '500', value)
    return raise_to


@pytest.fixture
def overs(monkeypatch):
    """What every DATA over keyed carries: its level and its frame count."""
    keyed = []

    def data_over_tx(bodies, over=0, level=None, bw='2300', lead=None):
        bodies = [bodies] if isinstance(bodies, bytes) else list(bodies)
        keyed.append((phy.host_level(bw, level), len(bodies)))
        return np.zeros(64)
    monkeypatch.setattr(VA.OF, 'data_over_tx', data_over_tx)
    return keyed


def sending(bw, data=bytes(20000)):
    hs, io = _connected(bw=bw)
    hs._link_high_speed = True
    hs.turn = VA._TURN_OURS
    hs.send(data)
    return hs, io


def delivered(hs, frame):
    hs._continue_frame = (hs._tx_pending, frame)
    hs._took_over_continue()
    return phy.host_level(hs.bw, hs._tx_pending[2])


def test_frame_bytes_carry_the_measured_double_loads():
    assert [2 * (phy.FRAME_BYTES[bw][phy.TOP[bw]] - 3) for bw in ('500', '2300', '2750')] \
        == [1420, 6508, 7838]
    assert phy.TOP == {'500': 13, '2300': 16, '2750': 17}


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_level_keys_round_trip_and_size_bodies(bw):
    for host in range(1, phy.TOP[bw] + 1):
        assert phy.host_level(bw, phy.level_key(bw, host)) == host
    assert phy.level_key(bw, 4) == phy.base_level(bw)
    assert phy.body_size(bw, phy.level_key(bw, 10)) == phy.FRAME_BYTES[bw][10] - 2


@pytest.mark.parametrize('bw', ['500', '2300'])
@pytest.mark.parametrize('frame,step', [(0, 0), (6, 1), (4, 2), (7, 3), (8, 4), (5, -1)])
def test_the_iss_keys_the_level_every_frame_asks(bw, frame, step, ceilings, overs):
    ceilings(tx=phy.TOP[bw])
    hs, io = sending(bw)
    hs._relevel(phy.level_key(bw, 8), 'test')
    delivered(hs, 0)
    assert delivered(hs, frame) == 8 + step
    assert overs[-1] == (8 + step, 1)
    assert not any('ceiling' in m for m in io.msgs)


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_the_iss_clamps_to_the_level_space(bw, ceilings, overs):
    ceilings(tx=phy.TOP[bw])
    hs, _ = sending(bw)
    assert delivered(hs, 5) == 4
    hs._relevel(phy.level_key(bw, phy.TOP[bw] - 1), 'test')
    delivered(hs, 0)
    assert delivered(hs, 8) == phy.TOP[bw]


@pytest.mark.parametrize('bw,ceiling', [('2300', 4), ('2300', 6), ('500', 4)])
def test_a_request_over_the_tx_ceiling_keys_the_ceiling_and_says_so(bw, ceiling,
                                                                   ceilings, overs):
    ceilings(tx=ceiling)
    hs, io = sending(bw)
    assert delivered(hs, 8) == ceiling
    assert f'peer asked level 8, keying {ceiling} (ceiling)' in io.msgs


def test_frame_eight_is_read_as_a_continue():
    hs, _ = _connected(bw='2300')
    tails = hs._peer_continue_tails()
    assert set(tails) >= {0, 4, 5, 6, 7, 8}
    assert tails[8] == VF.over_continue_frames(hs.called, '2300', (8,))[8][1]


def test_a_speed_limited_link_keeps_its_validated_continue_floors():
    hs, _ = _connected(bw='2300')
    assert {f: hs._continue_floor(f) for f in (0, 4, 5, 6, 7, 8, 10)} == {
        0: 1.5, 4: 2.0, 5: 1.5, 6: 2.0, 7: 1.5, 8: 2.0, 10: 1.5}
    hs._link_high_speed = True
    assert {f: hs._continue_floor(f) for f in (0, 4, 5, 6, 7, 8)} == {
        0: 1.5, 4: 2.0, 5: 1.5, 6: 2.0, 7: 2.0, 8: 2.0}


@pytest.mark.parametrize('level,last_good,jumped,unanswered,to', [
    (10, 9, False, False, 9),     # one step up, one down
    (13, 9, True, False, 9),      # a jump falls back to what delivered
    (10, 9, False, True, 8),      # an unanswered poll, two down
    (9, 9, False, False, 8),      # a second failure in a row, one more
    (4, 4, False, True, 4),       # never below 4
    (4, 4, False, False, 4)])
def test_the_fall_table(level, last_good, jumped, unanswered, to):
    assert _fall(level, last_good, jumped, unanswered) == to


@pytest.mark.parametrize('level,unanswered,to', [(4, False, 3), (4, True, 2), (2, True, 1)])
def test_the_fall_goes_under_4_to_its_floor(level, unanswered, to):
    # K9KDJ 05:12Z re-sent 5 -> 3, 4 -> 2, 4 -> 1 and 3 -> 1 behind our NAKs
    assert _fall(level, 4, False, unanswered, floor=1) == to


def _nak(hs):
    hs._pending_answer_at = time.monotonic()
    hs._matched_data_nak = (hs._tx_pending, hs._pending_answer_at, 'session-data-pair-nak')
    hs._took_data_nak()
    return phy.host_level(hs.bw, hs._tx_pending[2])


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_a_nak_falls_by_the_table(bw, ceilings, overs):
    ceilings(tx=phy.TOP[bw])
    hs, _ = sending(bw)
    delivered(hs, 6)                     # 4 -> 5, a single step
    assert delivered(hs, 8) == 9         # 5 -> 9, a jump
    assert _nak(hs) == 5                 # back to the level that last delivered
    assert _nak(hs) == 4                 # a second failure: one more


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_a_jumped_level_that_has_delivered_falls_one(bw, ceilings, overs):
    ceilings(tx=phy.TOP[bw])
    hs, _ = sending(bw)
    delivered(hs, 6)                     # 4 -> 5
    assert delivered(hs, 8) == 9         # 5 -> 9, a jump
    assert delivered(hs, 0) == 9         # 9 delivered
    assert _nak(hs) == 8


def test_a_naked_over_a_single_step_reached_falls_one(ceilings, overs):
    ceilings(tx=16)
    hs, _ = sending('2300')
    delivered(hs, 6)
    delivered(hs, 6)                     # 6, reached by a single step
    assert hs._data_nak_level() == phy.level_key('2300', 5)


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_six_delivered_at_the_top_then_doubles(bw, ceilings, overs):
    top = phy.TOP[bw]
    ceilings(tx=top)
    load = phy.FRAME_BYTES[bw][top] - 3
    hs, io = sending(bw, bytes(12 * load + 5))
    hs._relevel(phy.level_key(bw, top), 'test')
    delivered(hs, 0)                     # the base over keyed before the move
    for _ in range(VA._SINGLES_BEFORE_PAIR - 1):
        delivered(hs, 0)
        assert overs[-1] == (top, 1)
    delivered(hs, 0)
    assert overs[-1] == (top, 2)
    assert 'in two frames' in [m for m in io.msgs if m.startswith('tx DATA')][-1]


def test_no_double_below_the_top(ceilings, overs):
    ceilings(tx=15)
    hs, _ = sending('2300', bytes(40000))
    hs._relevel(phy.level_key('2300', 15), 'test')
    for _ in range(8):
        delivered(hs, 0)
    assert {n for _, n in overs} == {1}


@pytest.mark.parametrize('bw,level,quality,frame', [
    ('2300', 6, None, 0),       # no reading holds
    ('2300', 4, None, 6),       # but a clean base frame clears level 5
    ('2300', 7, None, 0),
    ('2300', 6, -0.1, 5),       # under its own row
    ('2300', 6, 4.9, 0),
    ('2300', 6, 5.0, 6),        # clears 7 by the margin
    ('2300', 6, 7.5, 4),        # 8
    ('2300', 6, 9.0, 7),        # 9
    ('2300', 6, 30.0, 8),       # 10, four at most
    ('2300', 12, 30.0, 4),      # 14: never a jump onto 15 or 16
    ('2300', 13, 30.0, 6),
    ('2300', 14, 30.0, 6),
    ('2300', 15, 24.0, 6),      # 16 by a single step
    ('2300', 13, 12.9, 5),
    ('2300', 13, 14.9, 0),
    ('2300', 5, 1.9, 0),        # arm16 at KC9GHZ: 6 asked off 1.9 dB, unread at 1.2
    ('2300', 5, 3.0, 6),
    ('2300', 3, None, 6),       # a clean read below the base climbs, as at 4
    ('2300', 2, 30.0, 6),
    ('500', 7, 30.0, 8),        # 11
    ('500', 9, 30.0, 4),        # 11, not 13
    ('500', 6, 10.9, 0),
    ('2750', 9, 11.5, 4),       # 10 and 11 share 8 dB
    ('2750', 13, 30.0, 4)])     # 15, not 17
def test_the_irs_steps_by_the_rows_its_snr_clears(bw, level, quality, frame):
    assert _irs_frame(level, quality, phy.TOP[bw], None, bw) == frame


def test_the_required_snr_rows_follow_the_row_shape():
    from hfmodem.kestrel.rx import varaofdm, varatone
    rows = VA._REQUIRED_SNR
    assert [rows['2300'][lv] for lv in range(5, 17)] == [-2, 0, 2, 4, 6, 8, 10, 12, 13, 15, 18, 21]
    shape = {}
    for (bw, lv), r in [*varatone.ROWS.items(), *varaofdm.ROWS.items()]:
        key = (r.n_info, r.coded) if (bw, lv) in varatone.ROWS else (
            'ofdm', lv - varaofdm._FAMILIES[bw][0])
        shape.setdefault(key, set()).add(rows[str(bw)][lv])
    assert all(len(v) == 1 for v in shape.values()), shape
    assert set(rows) == set(phy.TOP)
    assert all(list(rows[bw]) == list(range(5, phy.TOP[bw] + 1)) for bw in rows)


def test_the_irs_never_asks_over_its_ceiling():
    assert _irs_frame(4, None, 4, None, '2300') == 0
    assert _irs_frame(8, 30.0, 8, None, '2300') == 0
    assert _irs_frame(9, 30.0, 8, None, '2300') == 5
    assert _irs_frame(6, 30.0, 8, None, '2300') == 4


@pytest.mark.parametrize('level,remaining,to', [
    (10, 400, 9),       # 425 at 9 still carries it in one frame
    (10, 100, 9),       # one level down a frame, however far it could go
    (10, 1100, 10),     # 9 would take three frames where 10 takes two
    (4, 10, 4)])
def test_the_end_of_a_delivery_asks_the_lowest_level_in_as_many_frames(level, remaining, to):
    assert _finish_level(level, remaining, '2300') == to
    assert _irs_frame(level, 30.0, 16, remaining, '2300') == (5 if to < level else 0)


def test_our_continue_keys_the_policy_frame_within_the_rx_ceiling(ceilings):
    hs, _ = _connected(bw='2300')
    hs._link_high_speed = True
    hs._rx_read = (4, 30.0)
    ceilings(rx=4)
    assert hs._continue_tail() == VF.over_continue_frames(hs.called, '2300', (0,))[0][0]
    ceilings(rx=16)
    assert hs._continue_tail() == VF.over_continue_frames(hs.called, '2300', (4,))[4][0]
    assert hs._expect_level == 6
    hs._rx_read = (4, None)
    assert hs._continue_tail() == VF.over_continue_frames(hs.called, '2300', (6,))[6][0]
    assert hs._expect_level == 5
    hs._rx_read = (7, None)
    assert hs._continue_tail() == VF.over_continue_frames(hs.called, '2300', (0,))[0][0]
    assert hs._expect_level == 7


def test_the_first_answer_is_the_32_symbol_frame_carrying_the_step(ceilings):
    ceilings(rx=16)
    hs, io = _connected(bw='2300')
    hs._link_high_speed, hs._peer_over, hs._rx_read = True, 1, (7, 30.0)
    sent = []
    hs._send_burst = lambda kind: sent.append(kind) or True
    hs._tx_over_response(last=False)
    assert sent == [replace(VF.SESSION_OVER_RESPONSE, preadv=1 + 62 * 8)]


def _keyed_tones(io, called, kinds):
    burst = io.sent[-1]
    return next(k for k in kinds
                if np.allclose(burst, MK.synth_burst(called, VF.for_bw(k, '2300'))))


@pytest.mark.parametrize('heard,rx,start', [(None, 16, 4), (30.0, 4, 4), (16.0, 16, 5),
                                            (30.0, 16, 6), (30.0, 5, 5)])
def test_the_caller_confirm_names_the_callee_start(heard, rx, start, ceilings):
    ceilings(rx=rx)
    hs, io = _connected(bw='2300')
    hs._link_high_speed, hs._heard_db = True, heard
    hs._tx_session_confirm()
    assert _keyed_tones(io, hs.called, VA._CONFIRM_START) == next(
        k for k, lv in VA._CONFIRM_START.items() if lv == start)
    assert [k.preadv for k in VA._CONFIRM_START] == [91, 121, 151]


def test_a_speed_limited_confirm_is_frame_three():
    hs, io = _connected(bw='2300')
    hs._heard_db = 30.0
    hs._tx_session_confirm()
    assert _keyed_tones(io, hs.called, VA._CONFIRM_START) == VF.SESSION_CONFIRM


@pytest.mark.parametrize('tx,frame,keyed', [(16, 4, 5), (16, 5, 6), (16, 3, 4), (4, 5, 4)])
def test_the_callee_starts_where_the_confirm_says(tx, frame, keyed, ceilings, overs):
    ceilings(tx=tx)
    hs, io = _connected(bw='2300', role='responder')
    hs._link_high_speed = True
    kind = next(k for k, lv in VA._CONFIRM_START.items() if lv == frame + 1)
    hs._send_burst = lambda kind: True
    hs.on_rx_tones(VF.handshake_tones(hs.called, kind), kind)
    assert phy.host_level('2300', hs._rec500) == keyed
    assert (f'keying {keyed} (ceiling)' in io.msgs[-1]) == (keyed < frame + 1)
    hs.turn = VA._TURN_OURS
    hs.send(bytes(5000))
    assert overs[-1][0] == keyed


def test_high_speed_link_setup_bytes():
    body = VF.link_setup_frame('W9SSJ', '2300', high_speed=True)
    assert body[7] == 0x81 and VF.link_setup_marks(body) == (True, 6)
    assert VF.is_link_setup(body)
    assert VF.link_setup_marks(VF.link_setup_frame('W9SSJ', '2300')) == (False, 4)


@pytest.mark.parametrize('ours,theirs', [(True, True), (True, False), (False, True),
                                         (False, False)])
def test_the_callee_joins_the_level_space_the_caller_marks(ours, theirs):
    from hfmodem.kestrel.vara import vara_ofdm as OF
    hs, io = _connected(bw='2300', role='responder')
    hs.state, hs.high_speed = VA.VaraState.CONNECTING, ours
    x = OF.link_setup_tx('W9SSJ', bw='2300', level=4,
                         ask=6 if theirs else 4, high_speed=theirs,
                         lead=TX.VB6Rnd(VF.lead_state('W9SSJ', False, '2300')) if theirs else None)
    assert hs._rx_link_setup(np.concatenate([np.zeros(4800), x, np.zeros(4800)]))
    assert hs._link_high_speed is (ours and theirs)
    assert (hs._lead is not None) is (ours and theirs)


def test_the_station_flag_keys_the_high_speed_setup(monkeypatch):
    frames = []
    monkeypatch.setattr(VA.OF, 'link_setup_tx',
                        lambda *a, **kw: frames.append(kw) or np.zeros(64))
    for high_speed in (False, True):
        hs, io = _connected(bw='2300')
        hs.high_speed = high_speed
        hs._tx_link_setup()
        assert hs._link_high_speed is high_speed
    assert [(kw['high_speed'], kw['ask']) for kw in frames] == [(False, 4), (True, 6)]


def test_the_expected_level_is_what_the_decoder_is_asked_for(monkeypatch, clock):
    asked, answered = [], []
    read = phy.LevelRead(6, (bytes(130) + b'\x95',), 14.0, 1)
    monkeypatch.setattr(VA._phy, 'decode_level',
                        lambda x, bw, level, *_: asked.append((len(x), bw, level)) or read)
    hs, _ = _connected(bw='2300')
    hs._link_high_speed, hs._expect_level, hs._lv_owed = True, 6, True
    hs._answer_data_over = lambda x, hold=False: answered.append(hs._peer_data_over(x))
    hs._key(False)
    assert not hs._read_expected_level(np.zeros(VA._EXPECT_FIRST - 1))
    assert hs._read_expected_level(np.zeros(VA._EXPECT_WINDOW['2300']))
    assert asked == [(VA._EXPECT_WINDOW['2300'], '2300', 6)]
    assert answered == [[bytes(130) + b'\x95']] and hs._rx_read == (6, 14.0)


def test_level_four_is_never_an_expected_read():
    hs, _ = _connected(bw='2300')
    hs._link_high_speed = True
    hs._key(False)
    assert not hs._lv_armed
    with pytest.raises(LookupError):
        phy.decode_level(np.zeros(10), '2300', 17)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_every_level_to_the_ceilings_has_a_receiver_and_a_renderer(bw):
    assert (phy.RX_CEILING[bw], phy.TX_CEILING[bw]) == {
        '500': (13, 13), '2300': (16, 16), '2750': (17, 17)}[bw]
    for lv in range(5, max(phy.RX_CEILING[bw], phy.TX_CEILING[bw]) + 1):
        rx, tx = phy._family(bw, lv)
        assert (int(bw), lv) in rx.ROWS and tx.__name__.endswith('_tx')


def test_frame_bytes_are_the_measured_rows():
    # payload + control + CRC, off the KPA reports' decodes
    assert phy.FRAME_BYTES == {
        '500': {4: 46, 5: 93, 6: 135, 7: 197, 8: 263, 9: 315, 10: 395, 11: 474,
                12: 594, 13: 713},
        '2300': {4: 92, 5: 135, 6: 176, 7: 259, 8: 344, 9: 428, 10: 899, 11: 1199,
                 12: 1439, 13: 1799, 14: 2159, 15: 2713, 16: 3257},
        '2750': {4: 92, 5: 135, 6: 176, 7: 259, 8: 344, 9: 428, 10: 558, 11: 1083,
                 12: 1444, 13: 1733, 14: 2167, 15: 2600, 16: 3268, 17: 3922}}


@pytest.mark.parametrize('caller,responder,bw,state', [
    ('AAAA1', False, '2300', 4107106), ('AAAA1', False, '2750', 12338801),
    ('W9SSJ', False, '2300', 2767643), ('W9SSJ', True, '2300', 14503988),
    ('W9SSJ', True, '2750', 3653944), ('N0CALL', False, '2300', 15762747),
    ('N0CALL', True, '2750', 2596114)])
def test_the_lead_stream_starts_at_the_measured_e1(caller, responder, bw, state):
    assert VF.lead_state(caller, responder, bw) == state


def _advanced(state, n):
    rnd = TX.VB6Rnd(state)
    for _ in range(n):
        rnd.draw()
    return rnd


def test_the_l4_preamble_off_the_session_stream_is_the_numbered_one_for_aaaa1():
    frame = TX.build_frame(bytes(90))
    e1 = VF.lead_state('AAAA1', False, '2300')
    assert np.array_equal(TX.synth_frame(frame, 3, TX.VB6Rnd(e1)), TX.synth_frame(frame, 3, 0))
    assert np.array_equal(TX.synth_frame(frame, 3, _advanced(e1, 13)),
                          TX.synth_frame(frame, 3, 1))


def test_a_high_speed_capable_session_draws_every_lead_off_its_own_stream(monkeypatch):
    states = []

    def render(*a, lead=None, **kw):
        states.append(lead.state)
        for _ in range(12):
            lead.draw()
        return np.zeros(64)
    monkeypatch.setattr(VA.OF, 'link_setup_tx', render)
    monkeypatch.setattr(VA.OF, 'data_over_tx', render)
    hs, _ = _connected(bw='2300')
    hs.high_speed = True
    hs._tx_link_setup()
    hs.turn = VA._TURN_OURS
    hs.send(bytes(300))
    delivered(hs, 0)
    hs._pending_answer_at = time.monotonic()
    hs._matched_data_nak = (hs._tx_pending, hs._pending_answer_at, 'session-data-pair-nak')
    hs._took_data_nak()
    e1 = VF.lead_state(hs.caller, False, '2300')
    assert states == [e1] + [_advanced(e1, 12 * k + 1).state for k in (1, 2, 3)]


def _lead_states(monkeypatch):
    states = []

    def render(*a, lead=None, **kw):
        states.append(lead.state)
        for _ in range(12):
            lead.draw()
        return np.zeros(64)
    monkeypatch.setattr(VA.OF, 'link_setup_tx', render)
    monkeypatch.setattr(VA.OF, 'data_over_tx', render)
    return states


def _second_delivery(hs, first: bytes, second: bytes):
    """Two host writes, each one short over, the first acknowledged by the
    peer's control burst before the turn comes back for the second."""
    hs.turn = VA._TURN_OURS
    hs.send(first)
    hs._took_control_burst()
    hs.turn = VA._TURN_OURS
    hs.send(second)


def test_the_callers_second_delivery_draws_one_more_like_arm4_at_kb8ay(monkeypatch):
    # working/vara-levels-air-0927/arm4.console: link-setup, the 47-byte login,
    # its control-burst answer, the release, then the 5-byte over #2 that KB8AY
    # never read at E1+25. Stock W9SSJ on the 2026-09-26 bench keyed its second
    # delivery one draw past the corpus rule, at both bandwidths and both roles.
    states = _lead_states(monkeypatch)
    hs, _ = _connected(bw='2300')
    hs.high_speed = True
    hs._tx_link_setup()
    hs._tx_session_confirm()
    _second_delivery(hs, bytes(47), bytes(5))
    e1 = VF.lead_state(hs.caller, False, '2300')
    assert states == [e1, _advanced(e1, 13).state, _advanced(e1, 26).state]


def test_the_responders_second_delivery_draws_one_more_like_stock_b(monkeypatch):
    # Stock WI9YZA answering W9SSJ, 2026-09-26 bench: its greeting at E1+1 and its
    # reply, one turn later, at E1+38 = 1 + 36 (one L6 over) + 1.
    states = _lead_states(monkeypatch)
    hs, _ = _connected(bw='2300', role='responder')
    hs._link_high_speed = True
    hs._lead = TX.VB6Rnd(VF.lead_state(hs.caller, True, '2300'))
    _second_delivery(hs, bytes(50), bytes(4))
    e1 = VF.lead_state(hs.caller, True, '2300')
    assert states == [_advanced(e1, 1).state, _advanced(e1, 14).state]


def _peer_over(hs, payload: bytes, host: int, field: int, close: int | None = None):
    """One of the peer's overs at ``host``: full with ``field`` behind the
    payload, or, given ``close``, short and ending ``<close> <field>``."""
    key = phy.level_key(hs.bw, host)
    body = phy.vara_body(payload, hs.caller, tail=field, body_len=phy.body_size(hs.bw, key))
    if close is not None:
        body = body[:-2] + bytes([close, field])
    return VA.OF.data_over_tx(body, level=key, bw=hs.bw, lead=TX.VB6Rnd(4107106))


def _peer_delivery(hs, payload: bytes, close: int, field: int = 0x82):
    """KB8AY's deliveries of arm12, 2026-09-26: a full over at level 4, which
    our continue steps up on, then the short over at the level asked that
    closes it naming ``close``. Its control burst then hands us the turn."""
    hs._answer_data_over(_on_air(_peer_over(hs, payload, 4, 0x81)))
    assert hs._expect_level > 4
    _hear(hs, _peer_over(hs, payload[:7], hs._expect_level, field, close))
    hs.turn = VA._TURN_OURS


def _high_speed_caller():
    hs, io = _connected(bw='2300')
    hs.high_speed = True
    hs._tx_link_setup()
    hs._tx_session_confirm()
    return hs, io


@pytest.mark.parametrize('close, field, host, body', [(0x05, 0x82, 5, 133),
                                                      (0x03, 0x42, 3, 48)])
def test_the_peers_close_names_the_level_of_our_next_delivery(close, field, host, body,
                                                              clock):
    # arm12: the greeting closed 04 82 and our 81-byte over at level 4 was read;
    # the reply closed 05 82 and our next over, keyed at 4 again, drew three
    # NAKs and nothing else. arm4's reply closed 03 42 with the same result.
    hs, _ = _high_speed_caller()
    _peer_delivery(hs, bytes(range(89)), 0x04)
    hs.send(bytes(81))
    assert hs._tx_pending[2] == phy.level_key('2300', 4)
    hs._took_control_burst()
    _peer_delivery(hs, bytes(range(89, 0, -1)), close, field)
    hs.send(bytes(2351))
    keyed, _, level = hs._tx_pending
    assert (phy.host_level('2300', level), len(keyed)) == (host, body)


def test_our_close_names_the_level_our_continue_asked(clock):
    # arm12: we asked KB8AY +1 on its greeting's full over, closed our own
    # delivery 04, and its reply came at level 4.
    hs, _ = _high_speed_caller()
    _peer_delivery(hs, bytes(range(89)), 0x04)
    hs.send(bytes(81))
    assert hs._tx_pending[0][-2:] == bytes([hs._expect_level, 0x82]) and hs._expect_level > 4


def _asked_frames(hs, io):
    """The continue frame each burst we put on the air keyed."""
    bursts = {f: MK.synth_tone_pairs(pair[0]) for f, pair in
              VF.over_continue_frames(hs.called, hs.bw, tuple(VA._LEVEL_STEP)).items()}
    return [next(f for f, x in bursts.items() if np.array_equal(x, sent))
            for sent in io.sent]


def _record2_delivery():
    """KB8AY on 40 m, 2026-09-26: its greeting behind us, its next delivery
    keyed at record 2 and read clean there."""
    hs, io = _high_speed_caller()
    hs._peer_over, io.sent = 2, []
    return hs, io


def test_the_last_full_over_below_the_base_draws_a_step(clock):
    # arm20 20:23Z: the over announcing its last full one raised KeyError: 3
    hs, io = _record2_delivery()
    hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(47)), 3, 0x99)))
    hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(47, 0, -1)), 3, 0x81)))
    assert _asked_frames(hs, io) == [6, 6] and hs._expect_level == 4


def test_clean_overs_below_the_base_climb_one_at_a_time(clock):
    # arm18 20:17Z: 22 overs at record 2, each read clean and each held there
    hs, io = _record2_delivery()
    for i in range(3):
        hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(i, i + 47)), 3, 0x99)))
        assert hs._expect_level == 4
    assert _asked_frames(hs, io) == [6, 6, 6]


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_every_level_a_high_speed_capable_link_reads_has_a_load(bw):
    assert all(_finish_level(lv, 10, bw) <= lv for lv in range(1, phy.TOP[bw] + 1))


def test_a_speed_limited_session_keeps_the_numbered_preamble(overs, monkeypatch):
    leads = []
    monkeypatch.setattr(VA.OF, 'data_over_tx',
                        lambda *a, lead=None, **kw: leads.append(lead) or np.zeros(64))
    hs, _ = _connected(bw='2300')
    hs.turn = VA._TURN_OURS
    hs.send(bytes(300))
    assert leads == [None]


def _bodies(bw, host, n=1, seed=0):
    size = phy.body_size(bw, phy.level_key(bw, host))
    rng = np.random.default_rng(seed)
    return [bytes(rng.integers(0, 256, size - 1, dtype=np.uint8)) + bytes([0x99 - 4 * k])
            for k in range(n)]


@pytest.mark.parametrize('bw,host,expect,frames,draws', [
    ('2300', 5, 5, 1, 27), ('2300', 9, 8, 1, 90), ('2300', 10, 12, 1, 196),
    ('2300', 16, 16, 2, 490), ('500', 6, 6, 1, 27), ('500', 13, 12, 2, 110),
    ('2750', 10, 10, 1, 117), ('2750', 17, 17, 1, 295)])
def test_a_family_over_renders_off_the_stream_and_reads_back_at_its_level(
        bw, host, expect, frames, draws):
    from hfmodem.kestrel.vara import vara_ofdm as OF
    bodies = _bodies(bw, host, frames)
    lead = TX.VB6Rnd(4107106)
    x = OF.data_over_tx(bodies, level=phy.level_key(bw, host), bw=bw, lead=lead)
    assert lead.state == _advanced(4107106, draws).state
    read = phy.decode_level(np.concatenate([np.zeros(9000), x, np.zeros(9000)]), bw, expect)
    assert read.level == host and read.frames == tuple(bodies)
    assert read.control == bodies[-1][-1] and read.payload == bodies[-1][:-1]
    assert read.quality > 20


def test_a_short_buffer_is_nothing_yet():
    assert phy.decode_level(np.zeros(48000), '2300', 10) is None


def test_bw500_level_four_draws_its_18_lead_values_off_the_stream():
    from hfmodem.kestrel.rx import varahf500 as RX500
    from hfmodem.kestrel.vara import vara_ofdm as OF
    body = bytes(range(43)) + b'\x95'
    lead = TX.VB6Rnd(4107106)
    x = OF.data_over_tx(body, level=4, bw='500', lead=lead)
    assert lead.state == _advanced(4107106, 18).state
    assert not np.array_equal(x, OF.data_over_tx(body, level=4, bw='500'))
    frames = RX500.decode_stream(np.concatenate([np.zeros(5000), x, np.zeros(5000)])).frames
    assert [(f.crc_ok, f.payload + bytes([f.marker])) for f in frames] == [(True, body)]


@pytest.mark.parametrize('bw,host', [('2300', 8), ('2300', 12), ('2750', 10), ('2750', 17),
                                     ('500', 5), ('500', 9)])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_the_arqs_close_above_four_is_the_taped_short_frame(bw, host, role, monkeypatch):
    from hfmodem.kestrel.rx import varatone
    from hfmodem.kestrel.tx import varaofdm_tx, varatone_tx
    keyed = []
    monkeypatch.setattr(VA.OF, 'data_over_tx',
                        lambda bodies, **kw: keyed.append(bodies) or np.zeros(64))
    hs, _ = _connected(bw=bw, role=role)
    hs.caller = 'N0CALL'
    hs._link_high_speed = True
    hs._expect_level = varaofdm_tx.SHORT_TAIL[int(bw)]
    hs._relevel(phy.level_key(bw, host), 'test')
    hs.turn = VA._TURN_OURS
    hs.send(b'a short close')
    body = keyed[-1][0]
    frame = (varatone_tx.build_frame(b'a short close', 0x82, varatone.ROWS[int(bw), host],
                                     caller='N0CALL')
             if (int(bw), host) in varatone.ROWS else
             varaofdm_tx.build_frame(b'a short close', 0x82, int(bw), host, caller='N0CALL'))
    assert body == frame[:-2]
    assert body[-2:] == bytes([0x07 if bw == '500' else 0x06, 0x82])


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_level_four_reads_wherever_the_over_sits_in_the_buffer(bw):
    from hfmodem.kestrel.vara import vara_ofdm as OF
    body = bytes(range(89)) + b'\x95'
    x = OF.data_over_tx(body, level=phy.base_level(bw), bw=bw, lead=TX.VB6Rnd(4107106))
    read = phy.decode_level(np.concatenate([np.zeros(4801), x, np.zeros(9000)]), bw, 4)
    assert (read.level, read.frames) == (4, (body,))


class _Clock:
    """A monotonic clock that runs with the audio fed  [see _hear]."""
    t = 1000.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.t


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(VA, 'time', _Clock)
    return _Clock


def _listening(bw, expect, last, run=0):
    """A high speed capable station that has just answered, due the peer's next over at
    ``expect`` behind one read clean at ``last``, with ``run`` singles before it."""
    hs, io = _connected(bw=bw)
    hs._link_high_speed, hs._expect_level, hs._rx_read = True, expect, (last, 30.0)
    hs._lv_owed = hs._lv_prompt = True
    hs._peer_run_500 = run
    answered = []
    hs._answer_data_over = lambda x, hold=False: answered.append(
        [b for b in hs._level_read.frames]) or True
    hs._key(False)
    return hs, io, answered


def _hear(hs, x, lead=0.1, tail=3.0):
    """``x`` keyed ``lead`` s behind our unkey, in quarter-second polls. The
    samples heard when the over was taken, or None."""
    audio = np.concatenate([np.zeros(int(lead * MK.FS)), x, np.zeros(int(tail * MK.FS))])
    step = MK.FS // 4
    for i in range(0, len(audio), step):
        _Clock.t += step / MK.FS
        if hs._read_expected_level(audio[i:i + step]):
            return i + step
    return None


def _double(bw, host, crc0=True):
    from hfmodem.kestrel.tx import varaofdm_tx
    bodies = _bodies(bw, host, 2)
    frames = [b + (crc16_genibus(b) ^ (0 if k or crc0 else 1)).to_bytes(2, 'big')
              for k, b in enumerate(bodies)]
    return bodies, varaofdm_tx.render(frames, int(bw), host, TX.VB6Rnd(4107106))


def _nak_keyed(hs, io):
    return np.array_equal(io.sent[-1], MK.synth_tone_pairs(
        VF.over_continue_frames(hs.called, hs.bw, (1,))[1][0]))


@pytest.mark.parametrize('lead', [0.1, 0.3, 0.85, 2.0])
@pytest.mark.parametrize('bw', ['2300', '2750', '500'])
def test_a_double_is_read_whole_before_it_is_answered(bw, lead, clock):
    top = phy.TOP[bw]
    bodies, x = _double(bw, top)
    hs, _, answered = _listening(bw, top, top, run=VA._SINGLES_BEFORE_PAIR)
    _hear(hs, x, lead)
    assert answered == [bodies]


@pytest.mark.parametrize('bw', ['2300', '500'])
def test_a_double_nobody_predicted_is_read_whole(bw, clock):
    top = phy.TOP[bw]
    bodies, x = _double(bw, top)
    hs, _, answered = _listening(bw, top, top)
    _hear(hs, x)
    assert answered == [bodies]


def test_an_unpredicted_double_whose_first_frame_fails_is_naked_behind_its_second(clock):
    _, x = _double('2300', 16, crc0=False)
    hs, io, answered = _listening('2300', 16, 16)
    taken = _hear(hs, x)
    assert answered == [] and _nak_keyed(hs, io)
    assert taken >= int(0.1 * MK.FS) + len(x)


def test_a_double_whose_first_frame_fails_draws_a_nak_and_nothing_is_delivered(clock):
    bodies, x = _double('2300', 16, crc0=False)
    hs, io, answered = _listening('2300', 16, 16, run=VA._SINGLES_BEFORE_PAIR)
    _hear(hs, x)
    assert answered == [] and _nak_keyed(hs, io)
    assert hs._expect_level == 15 and hs._lv_armed


def _single(bw, host, crc=True, seed=0):
    from hfmodem.kestrel.vara import vara_ofdm as OF
    (body,) = _bodies(bw, host, seed=seed)
    if crc:
        return body, OF.data_over_tx(body, level=phy.level_key(bw, host), bw=bw,
                                     lead=TX.VB6Rnd(4107106))
    from hfmodem.kestrel.tx import varaofdm_tx
    frame = body + (crc16_genibus(body) ^ 1).to_bytes(2, 'big')
    return body, varaofdm_tx.render([frame], int(bw), host, TX.VB6Rnd(4107106))


def test_an_over_that_will_not_read_draws_a_nak_and_its_retry_is_read(clock):
    _, bad = _single('2300', 12, crc=False)
    hs, io, answered = _listening('2300', 12, 11)
    _hear(hs, bad)
    assert answered == [] and _nak_keyed(hs, io)
    assert hs._expect_level == 11 and hs._lv_armed
    body, retry = _single('2300', 11, seed=1)
    _hear(hs, retry)
    assert answered == [[body]]


def test_a_resend_that_will_not_read_either_draws_the_16_symbol_data_nak(clock):
    # the 2026-09-26 bench, W9SSJ -> WI9YZA at BW2300, twice: a failed over
    # draws the 8-symbol NAK, the resend fails and draws 61/31 0.12 s behind
    # its unkey, and the resend one level under that reads
    _, bad = _single('2300', 12, crc=False)
    hs, io, answered = _listening('2300', 12, 11)
    _hear(hs, bad)
    assert _nak_keyed(hs, io) and hs._expect_level == 11
    _, bad_again = _single('2300', 11, crc=False, seed=1)
    _hear(hs, bad_again)
    assert _burst_named(io, hs.called, VF.SESSION_DATA_NAK_CALLER_SHORT), io.msgs[-2:]
    assert hs._expect_level == 10 and hs._lv_armed and answered == []
    body, retry = _single('2300', 10, seed=2)
    _hear(hs, retry)
    assert answered == [[body]]


def test_a_window_that_holds_nothing_stays_armed_where_the_retry_falls(clock):
    hs, io, answered = _listening('2300', 12, 12)
    _hear(hs, np.zeros(0), tail=VA._EXPECT_WINDOW['2300'] / MK.FS + 0.5)
    assert hs._lv_armed and hs._expect_level == 10 and not io.sent
    body, retry = _single('2300', 10, seed=2)
    _hear(hs, retry, lead=2.0)
    assert answered == [[body]]


def test_a_double_last_keeps_the_next_over_due_a_double(clock):
    hs, _, _ = _listening('2300', 16, 16, run=VA._SINGLES_BEFORE_PAIR)
    bodies = _bodies('2300', 16, 2)
    hs._level_read = phy.LevelRead(16, tuple(bodies), 25.0, 2)
    hs._peer_data_over(np.zeros(0))
    assert hs._peer_run_500 == 0 and hs._peer_doubles()
    hs._overs_hint = 0
    assert not hs._peer_doubles()


@pytest.mark.parametrize('hint', [0, 1])
def test_a_single_behind_the_last_full_overs_is_answered_as_it_ends(hint, clock):
    body, x = _single('2300', 16)
    hs, _, answered = _listening('2300', 16, 16, run=VA._SINGLES_BEFORE_PAIR)
    hs._overs_hint = hint
    assert not hs._peer_doubles()
    taken = _hear(hs, x)
    assert answered == [[body]]
    assert taken < int(0.1 * MK.FS) + len(x) + MK.FS


def test_a_deliverys_first_over_is_waited_for_where_it_was_asked(clock):
    hs, io, answered = _listening('2300', 12, 12)
    hs._lv_prompt = False
    hs._key(False)
    _hear(hs, np.zeros(0), tail=2 * VA._EXPECT_WINDOW['2300'] / MK.FS)
    assert hs._lv_armed and hs._expect_level == 12 and not io.msgs
    body, x = _single('2300', 12, seed=3)
    _hear(hs, x, lead=1.0)
    assert answered == [[body]]


@pytest.mark.parametrize('cost,most', [(0.0, 15), (1.5, 5)])
def test_a_missed_window_costs_a_bounded_number_of_decodes(cost, most, clock, monkeypatch):
    calls = []

    def decode(x, bw, level, *_):
        calls.append(len(x))
        clock.t += cost
    monkeypatch.setattr(VA._phy, 'decode_level', decode)
    hs, _, _ = _listening('2300', 12, 12)
    _hear(hs, np.zeros(0), tail=VA._EXPECT_WINDOW['2300'] / MK.FS)
    assert 0 < len(calls) <= most
    clock.t += 10.0
    hs._read_expected_level(np.zeros(0))
    n = len(calls)
    for _ in range(4):
        clock.t += 1.0
        hs._read_expected_level(np.zeros(100))
    assert len(calls) == n


def test_a_deliverys_first_over_that_will_not_read_is_naked_as_it_ends(clock):
    _, bad = _single('2300', 12, crc=False)
    hs, io, answered = _listening('2300', 12, 12)
    hs._lv_prompt = False
    hs._key(False)
    taken = _hear(hs, bad, lead=1.0)
    assert answered == [] and _nak_keyed(hs, io)
    assert taken <= int(1.0 * MK.FS) + len(bad) + MK.FS


@pytest.mark.parametrize('bw,expect', [('500', 5), ('500', 6)])
def test_a_level_four_over_asked_at_five_or_six_is_read_at_four(bw, expect, clock):
    from hfmodem.kestrel.vara import vara_ofdm as OF
    size = phy.body_size(bw)
    body = bytes(range(size - 1)) + b'\x95'
    x = OF.data_over_tx(body, level=phy.base_level(bw), bw=bw, lead=TX.VB6Rnd(4107106))
    hs, io, answered = _listening(bw, expect, 4)
    _hear(hs, x)
    assert answered == [[body]] and not io.sent


@pytest.mark.parametrize('bw,host', [('2300', 3), ('2300', 1), ('2750', 3)])
def test_a_base_over_under_four_asked_at_six_is_read_not_naked(bw, host, clock):
    # NS0A's CRC-clean record-2 resend, 2026-09-27 07:24Z, drew our 61/31
    body, x = _single(bw, host)
    hs, io, answered = _listening(bw, 6, 5)
    _hear(hs, x)
    assert answered == [[body]] and not io.sent


def _behind_our_over_nak(offset_hz):
    """N0LCR-1, 40 m 2026-09-27 21:03Z: its level-6 over read, the next one not,
    and our over-NAK keyed in its cadence; its carriers ``offset_hz`` off ours."""
    hs, io = _connected(bw='2300')
    hs._link_high_speed, hs.turn = True, VA._TURN_PEER
    hs._rx_read, hs._expect_level, hs._lv_owed = (6, 20.0), 5, True
    total = offset_hz / MK.carrier_to_hz(1)
    hs._peer_shift = round(total)
    hs._peer_offset = total - hs._peer_shift
    assert hs._nak_owed_over()
    return hs, io


def _streamed(hs, io, x, lead=0.2, tail=3.0):
    audio = np.concatenate([np.zeros(int(lead * MK.FS)), x, np.zeros(int(tail * MK.FS))])
    sent, said = len(io.sent), len(io.msgs)
    for i in range(0, len(audio), 960):
        _Clock.t += 960 / MK.FS
        hs.on_rx_stream(audio[i:i + 960])
    return io.sent[sent:], io.msgs[said:]


def _moved(x, hz):
    from scipy.signal import hilbert
    return (hilbert(x) * np.exp(2j * np.pi * hz * np.arange(len(x)) / MK.FS)).real


def test_a_level_1_resend_29_hz_off_behind_our_over_nak_is_read(clock):
    # N0LCR-1 resent at 3 and then at 1 seventeen times, each clean off the tape
    # 29 Hz up and none read live: we NAKed every one
    body = b' 90119783\x8d'
    x = _single_at('2300', 1, body)
    hs, io = _behind_our_over_nak(-29.0)
    keyed, said = _streamed(hs, io, _moved(x, -29.0))
    assert any(b'90119783' in p for p in io.host), said
    assert len(keyed) == 1 and not any('NAK' in m for m in said), said


@pytest.mark.parametrize('host', [3, 2, 1])
def test_an_over_under_four_asked_at_four_is_read_at_its_record(host):
    body, x = _single('2300', host)
    read = phy.decode_level(np.concatenate([np.zeros(4801), x, np.zeros(9000)]), '2300', 4)
    assert (read.level, read.frames) == (host, (body,))


def _level_1_the_tone_rows_locate():
    """A level-1 over in noise that the level-7 to 9 tone rows find and cannot
    read, as they found N0LCR-1's at 84 s."""
    body = b' 90119783\x8d'
    x = _single_at('2300', 1, body)
    return body, _in_noise(x, -10, 1)


def _single_at(bw, host, body):
    from hfmodem.kestrel.vara import vara_ofdm as OF
    return OF.data_over_tx(body, level=phy.level_key(bw, host), bw=bw, lead=TX.VB6Rnd(4107106))


@pytest.mark.parametrize('expect', [7, 8, 9])
def test_a_base_over_a_tone_row_finds_and_cannot_read_is_read_at_its_record(expect):
    body, y = _level_1_the_tone_rows_locate()
    read = phy.decode_level(y, '2300', expect)
    assert (read.level, read.frames) == (1, (body,))


@pytest.mark.parametrize('expect,host', [(4, 3), (7, 1)])
def test_a_resend_under_four_behind_our_nak_is_read_whatever_was_asked(expect, host, clock):
    # a stock ISS resends 1-4 under the failed over at a level of its choosing:
    # bench 7 -> 6, 10 -> 7; N0LCR-1 on air 6 -> 3 -> 1
    body, x = _single('2300', host) if host == 3 else _level_1_the_tone_rows_locate()
    hs, io = _connected(bw='2300')
    hs._link_high_speed, hs.turn = True, VA._TURN_PEER
    hs._rx_read, hs._expect_level, hs._rx_failed = (expect + 1, 20.0), expect, True
    hs._lv_owed = hs._lv_prompt = True
    hs._lv_above = (expect, expect + 1)
    hs._key(False)
    keyed, said = _streamed(hs, io, x)
    assert any(body[:-1] in p for p in io.host), said
    assert len(keyed) == 1 and not any('NAK' in m for m in said), said


NS0A_TAPE = evidence.LOGS / 'onair' / '20260927T072508Z-W9SSJ-NS0A.flac'


@pytest.mark.skipif(not NS0A_TAPE.exists(), reason='on-air tape not kept here')
def test_ns0a_record_2_resend_asked_at_six_is_read_whole_not_naked_arriving(clock):
    # 54.37-61.78 s: behind our NAK, NS0A's greeting again at record 2, CRC-clean
    from hfmodem.core import wav
    x = wav.read(str(NS0A_TAPE))
    hs, io, answered = _listening('2300', 6, 5)
    _hear(hs, (x[:, 0] if x.ndim > 1 else x)[2609536:2965504], lead=0)
    assert answered and answered[0][0].startswith(b'RMS Trimode') and not io.sent, io.msgs


@pytest.mark.parametrize('first,failed,late,high_speed', [
    (4, False, False, False), (6, False, False, True),
    (4, True, False, True),     # behind our NAK or a window that fell
    (4, False, True, True)])    # after the first window
def test_a_callee_that_ignores_our_confirm_is_free(first, failed, late, high_speed,
                                                   ceilings, clock):
    ceilings(rx=16)
    hs, io = _connected(bw='2300')
    hs.high_speed = hs._link_high_speed = True
    hs._heard_db = 30.0
    hs._tx_session_confirm()
    assert hs._confirm_start == 6
    clock.t += (VA._EXPECT_WINDOW['2300'] / MK.FS + 1) if late else 4.0
    hs._rx_read, hs._rx_failed = (first, 30.0), failed
    hs._peer_data_over = lambda x: [bytes(90)]
    hs._answer_data_over(np.zeros(64))
    assert hs._link_high_speed is high_speed
    assert any('a free station' in m for m in io.msgs) is not high_speed


# (bw, level, noise over the burst's r.m.s., dB): the most noise under which
# the span locator still finds each row's burst a decibel over the floor. At
# BW2300 L5, the most robust row, that meets the row's own decoding edge, and
# one seed in four reads clean there.
_EDGE = [('2300', 5, 14), ('2300', 9, 14), ('2750', 10, 14), ('500', 6, 19)]


def _in_noise(x, db, seed, lead=1.0, tail=2.0):
    y = np.concatenate([np.zeros(int(lead * MK.FS)), x, np.zeros(int(tail * MK.FS))])
    return y + np.random.default_rng(seed).normal(0, 10 ** (db / 20) * np.std(x), len(y))


@pytest.mark.parametrize('bw,host,db', _EDGE)
def test_a_tone_over_the_noise_ruined_is_found_whole_and_not_before(bw, host, db):
    _, x = _single(bw, host)
    lost = 0
    for seed in range(4):
        y = _in_noise(x, db, seed + 10)
        end = int(MK.FS) + len(x)
        read = phy.decode_level(y[:end + MK.FS], bw, host)
        assert read is not None, seed
        lost += read.lost
        assert phy.decode_level(y[:end - 2000], bw, host) is None
        assert phy.decode_level(np.random.default_rng(seed).normal(0, 1, end), bw, host) is None
    assert lost >= 3


@pytest.mark.parametrize('bw,host,db', _EDGE)
def test_a_tone_over_the_noise_ruined_draws_a_nak_as_it_ends(bw, host, db, clock):
    _, x = _single(bw, host)
    hs, io, answered = _listening(bw, host, host)
    y = _in_noise(x, db, 10, lead=0.3)
    taken = _hear(hs, y, lead=0, tail=0)
    assert answered == [] and _nak_keyed(hs, io)
    # the floor behind a 0.3 s lead is a few symbols, and the verdict waits
    # for enough of it behind the burst
    assert taken <= int(0.3 * MK.FS) + len(x) + 2 * MK.FS


def _floor(seconds, ramp_db=0.0, seed=1):
    """-31 dBFS across the rig's passband, sliding ``ramp_db`` end to end."""
    n = int(seconds * MK.FS)
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / MK.FS)
    spec[(f < 300) | (f > 2700)] = 0
    x = np.fft.irfft(spec, n)
    x *= 10 ** (-31 / 20) / np.std(x)
    return x * 10 ** (np.linspace(0, ramp_db, n) / 20)


def _keyed_at(x, burst, t, db):
    a = int(t * MK.FS)
    x[a:a + len(burst)] += burst * 10 ** ((-31 + db) / 20) / np.std(burst)


def test_a_drifting_floor_with_a_control_burst_in_it_is_not_a_located_over(clock):
    hs, io, answered = _listening('2300', 5, 5)
    hs._rx_failed = True
    x = _floor(5.6, ramp_db=-1.5)
    _keyed_at(x, MK.synth_tone_pairs(VF.nak(hs.called, '2300')[0]), 0.05, 10)
    _keyed_at(x, MK.synth_burst(hs.called, VF.SESSION_CONFIRM), 3.1, 6)
    _hear(hs, x, lead=0, tail=0)
    assert not io.sent and hs._expect_level == 5 and hs._lv_armed
    body, retry = _single('2300', 5, seed=4)
    _hear(hs, retry, lead=0.2)
    assert answered == [[body]]


def _burst_named(io, called, kind):
    want = MK.synth_burst(called, VF.for_bw(kind, '2300'))
    return len(io.sent[-1]) == len(want) and np.allclose(io.sent[-1], want)


def _band_noise(n, rms, seed, lo=0.0, hi=2900.0):
    rng = np.random.default_rng(seed)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / MK.FS)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x * rms / np.sqrt(np.mean(x * x))


def test_a_high_speed_capable_peer_idling_over_an_unread_over_draws_the_nak_not_the_ack(clock):
    # KB8AY 22:07Z, 2026-09-26: its level-5 close sat under a 2156-2461 Hz
    # interferer; behind its 683 idle we re-keyed the frame that had answered
    # its base over, it took that as the close's ack, and 40 bytes were lost
    hs, io = _high_speed_caller()
    hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(89)), 4, 0x41)))
    step = VA._STEP_FRAME[hs._expect_level - 4]
    assert _burst_named(io, hs.called, replace(VF.SESSION_OVER_RESPONSE, preadv=1 + 62 * step))
    assert hs._expect_level > 4 and hs._answer_owed == VA._OWED_OVER
    x = _peer_over(hs, bytes(range(40)), hs._expect_level, 0x82, close=6)
    buried = x + _band_noise(len(x), 10 ** (15 / 20) * np.sqrt(np.mean(x * x)), 3, 2156, 2461)
    assert phy.decode_level(np.concatenate([np.zeros(4800), buried]), '2300', 5) is None
    assert _hear(hs, buried) is None
    assert io.host == [bytes(range(89))]
    hs.on_rx_audio(MK.synth_burst(hs.called, VF.SESSION_RESPONDER_IDLE))
    assert _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT), io.msgs[-3:]
    ack = MK.synth_burst(hs.called, replace(VF.SESSION_OVER_RESPONSE, preadv=1 + 62 * step))
    assert not any(len(s) == len(ack) and np.allclose(s, ack) for s in io.sent[3:])
    _hear(hs, _peer_over(hs, b'', 6, 0x82, close=6))
    assert io.host == [bytes(range(89))] and hs._lv_owed


def test_a_high_speed_capable_peers_poll_behind_an_unread_over_draws_the_short_over_nak(clock):
    # the 2026-09-26 bench, three sessions: a level-10 over buried under noise
    # drew nothing, the responder polled with frame 12 1.3 s after its unkey,
    # the caller keyed 61/541 and the resend came lower
    hs, io = _high_speed_caller()
    hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(89)), 4, 0x41)))
    assert hs._expect_level > 4 and hs._answer_owed == VA._OWED_OVER
    x = _peer_over(hs, bytes(range(40)), hs._expect_level, 0x82, close=6)
    buried = x + _band_noise(len(x), 10 ** (15 / 20) * np.sqrt(np.mean(x * x)), 3, 2156, 2461)
    assert _hear(hs, buried) is None
    level, sent = hs._expect_level, len(io.sent)
    poll = MK.synth_tone_pairs(VF.over_continue_frames(hs.called, '2300', (12,))[12][1])
    x = np.concatenate([np.zeros(MK.FS // 2), poll, np.zeros(2 * MK.FS)])
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])
    assert len(io.sent) == sent + 1, io.msgs[-3:]
    assert _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT), io.msgs[-3:]
    assert hs._expect_level == max(4, level - 2) and hs._rx_failed


def test_a_link_up_on_the_turn_request_expects_the_greeting_the_setup_asked(clock):
    # KB8AY 22:07Z: the link came up on its turn-request, no confirm was keyed,
    # and its level-6 greeting, keyed at the level our link-setup asked, was
    # never looked for
    hs, io = _connected(bw='2300')
    hs.high_speed = True
    hs._tx_link_setup()
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_LINKSETUP_SENT
    ask = MK.synth_burst(hs.caller, VF.SESSION_TURN_REQUEST_RESPONDER)
    hs.on_rx_audio(np.concatenate([np.zeros(2 * MK.HOP), ask, np.zeros(3 * MK.HOP)]))
    assert hs.state is VA.VaraState.CONNECTED and hs._expect_level == 6
    assert VF.SESSION_CONFIRM.name not in ' '.join(io.msgs)
    _hear(hs, _peer_over(hs, bytes(range(40)), 6, 0x82, close=6))
    assert io.host == [bytes(range(40))]


@pytest.mark.parametrize('snr,frames', [(-6, (0, 5)), (-3, (0,)), (2, (4,))])
def test_a_base_over_reads_its_channel_snr_and_the_irs_steps_by_it(snr, frames):
    # the 2026-09-26 bench: record 3 reads 8/8 at -6 dB and 0/8 at -9, the
    # level-5 tone row 6/6 at -4 and 0/6 at -6, level 6 4/4 at -2 and 2/4 at -4
    body = phy.vara_body(bytes(range(89)), 'W9SSJ', tail=0x99)
    x = VA.OF.data_over_tx(body, level=3, bw='2300', lead=TX.VB6Rnd(4107106))
    y = np.concatenate([np.zeros(MK.FS // 2), x, np.zeros(MK.FS // 2)])
    for seed in (1, 2):
        rms = np.sqrt(np.mean(x * x)) / 10 ** (snr / 20)
        read = phy.decode_level(y + _band_noise(len(y), rms, seed), '2300', 4)
        assert read is not None and read.frames == (body,)
        assert abs(read.quality - snr) <= 1.5, (seed, read.quality)
        assert _irs_frame(4, read.quality, 16, None, '2300') in frames, (seed, read.quality)


def _live(hs, audio):
    """``audio`` from our unkey in 20 ms polls, on a clock that the audio and
    every decode move alike, as on a station whose poll loop runs the decoder.
    The audio position the over was taken at, or None."""
    t0, fed = _Clock.t, 0
    while fed < len(audio):
        _Clock.t += 0.02
        upto = min(len(audio), int((_Clock.t - t0) * MK.FS))
        taken = hs._read_expected_level(audio[fed:upto])
        fed = upto
        if taken:
            return fed
    return None


@pytest.mark.parametrize('cost', [0.5, 1.5])
@pytest.mark.parametrize('lead', [0.1, 0.3, 0.5])
def test_a_high_speed_capable_over_is_answered_as_it_ends_whatever_a_decode_costs(lead, cost, clock,
                                                                          monkeypatch):
    # bench r1, 2026-09-27 01:36Z: WI9YZA's level-10 over was read 0.93 s
    # after its unkey and our continue keyed at 1.33 s, into its poll
    decode = phy.decode_level

    def costly(*a):
        clock.t += cost
        return decode(*a)
    monkeypatch.setattr(VA._phy, 'decode_level', costly)
    body, x = _single('2300', 10)
    hs, _, answered = _listening('2300', 10, 10)
    a = int(lead * MK.FS)
    taken = _live(hs, np.concatenate([np.zeros(a), x, np.zeros(int(1.5 * MK.FS))]))
    assert answered == [[body]]
    assert taken <= a + len(x) + int(0.3 * MK.FS)


def test_a_bw500_over_a_few_db_over_the_band_behind_our_mute_is_read_as_it_ends(clock):
    # K5FIT, 2026-09-27 04:12Z: the block behind each unkey held our mute
    # residue, 30-50 dB under the band, and its overs fell 1-3 dB across the
    # passband at their end
    body, x = _single('500', 6)
    hs, _, answered = _listening('500', 6, 6)
    a = int(0.1 * MK.FS)
    audio = np.concatenate([np.zeros(a), x, np.zeros(MK.FS)])
    audio += _band_noise(len(audio), 10 ** (3 / 20) * np.sqrt(np.mean(x * x)), 1, 300, 2700)
    audio[:VA._EXPECT_STEP] *= 0.01
    taken = _live(hs, audio)
    assert answered == [[body]]
    assert taken <= a + len(x) + MK.FS // 2


class _CostClock:
    """Time spent asleep plus what the decoders cost, charged per second of
    audio each is handed: an L6 tone read, one that came out clean, a base
    read and a search under the base record, at what each costs a core on
    this receiver."""
    PER_S = {'tone': 0.07, 'clean': 0.008, 'base': 0.013, 'below': 0.03}

    def __init__(self, monkeypatch):
        self.t, self.spent = 0.0, 0.0
        for mod, name, kind in ((phy._tone, 'decode_burst', 'tone'),
                                (phy, '_decode_base', 'base'), (phy, '_base_below', 'below')):
            monkeypatch.setattr(mod, name, self._charged(getattr(mod, name), kind))

    def _charged(self, f, kind):
        def charged(x, *a, **k):
            out = f(x, *a, **k)
            cost = self.PER_S['clean' if getattr(out, 'ok', False) else kind] * len(x) / MK.FS
            self.t, self.spent = self.t + cost, self.spent + cost
            return out
        return charged

    def monotonic(self) -> float:
        return self.t


def _in_real_time(hs, clock, audio, block=MK.FS // 10):
    """``audio`` arriving in ``block``s as ``clock`` passes it: the audio fed
    and the clock when the over was taken, and the most audio ever waiting."""
    fed, lag = 0, 0.0
    while fed < len(audio):
        ready = int(clock.t * MK.FS + 0.5) // block * block
        if ready <= fed:
            clock.t = (fed + block) / MK.FS
            continue
        lag = max(lag, (ready - fed) / MK.FS)
        taken = hs._read_expected_level(audio[fed:fed + block])
        fed += block
        if taken:
            break
    return fed, clock.t, lag


def test_a_level_6_over_behind_a_keyed_band_is_read_in_real_time(monkeypatch):
    # KB2PCN-8, 20 m 2026-09-27 17:32Z: behind our release the level-6 reader
    # ran a core flat out 2-3 s behind the band, and our NAK keyed 5.9 s late
    body, x = _single('2300', 6)
    rms = np.sqrt(np.mean(x * x))
    qrm = _band_noise(20 * MK.FS, rms, 2, 250, 2750)
    qrm *= np.arange(len(qrm)) % int(0.6 * MK.FS) < int(0.2 * MK.FS)
    audio = np.concatenate([qrm, x, np.zeros(2 * MK.FS)])
    audio += _band_noise(len(audio), rms / 2, 1, 300, 2700)
    end = (len(qrm) + len(x)) / MK.FS
    clock = _CostClock(monkeypatch)
    monkeypatch.setattr(VA, 'time', clock)
    hs, io, answered = _listening('2300', 6, 6)
    hs._lv_prompt, hs._lv_quiet = False, 0
    fed, at, lag = _in_real_time(hs, clock, audio)
    assert answered == [[body]] and not io.sent, (fed / MK.FS, io.msgs)
    assert at <= end + 0.5 and lag < 1.0 and clock.spent <= 0.5 * fed / MK.FS, \
        (at - end, lag, clock.spent)


class _SearchClock:
    """Time spent asleep plus what the index-law search costs, charged by the
    work each piece is handed at what it costs a core on this receiver: a
    column transformed whole, a sample of the running transform, an equalised
    band reading, a turbo decode."""
    PER = {'_band_mag': 2.1e-9, '_hop_band': 13e-9, '_onset_columns': 12.5e-9,
           'frame_at': 0.045}
    WORK = {'_band_mag': lambda x, level, ncol: ncol * RX.RECORDS[level].dw50,
            '_hop_band': lambda x, *a: len(x),
            '_onset_columns': lambda rows, level: rows.size,
            'frame_at': lambda mag, level: 1}

    def __init__(self, monkeypatch):
        self.t, self.spent = 0.0, 0.0
        for name, rate in self.PER.items():
            if hasattr(RX, name):
                monkeypatch.setattr(RX, name, self._charged(getattr(RX, name), name, rate))

    def _charged(self, f, name, rate):
        def charged(*a):
            cost = rate * self.WORK[name](*a)
            self.t, self.spent = self.t + cost, self.spent + cost
            return f(*a)
        return charged

    def monotonic(self) -> float:
        return self.t


def test_a_busy_band_costs_the_base_search_a_fraction_of_real_time(monkeypatch):
    # KB2PCN-8, 20 m 2026-09-27 17:32Z: expecting level 4, the base search ran
    # 75-90% of a core scanning the same audio every half second
    body, x = _single('2300', 4)
    rms = np.sqrt(np.mean(x * x))
    qrm = _band_noise(12 * MK.FS, rms, 2, 250, 2750)
    qrm *= np.arange(len(qrm)) % int(0.6 * MK.FS) < int(0.2 * MK.FS)
    audio = np.concatenate([qrm, x, np.zeros(2 * MK.FS)])
    audio += _band_noise(len(audio), rms / 2, 1, 300, 2700)
    end = (len(qrm) + len(x)) / MK.FS
    clock = _SearchClock(monkeypatch)
    monkeypatch.setattr(VA, 'time', clock)
    hs, io, _ = _listening('2300', 4, 4)
    read = []
    hs._answer_data_over = lambda buf, hold=False: read.append(clock.t) or True
    fed, block = 0, MK.FS // 50
    while fed < len(audio) and not read:
        if clock.t * MK.FS < fed + block:
            clock.t = (fed + block) / MK.FS
        hs.on_rx_stream(audio[fed:fed + block])
        fed += block
    assert read and read[0] <= end + 0.6, (read, end)
    assert clock.spent <= 0.1 * fed / MK.FS, (clock.spent, fed / MK.FS)


def test_a_high_speed_capable_bw500_over_behind_our_over_nak_is_read_where_our_continue_asked(clock):
    # K5FIT, 2026-09-27 04:12Z: behind our over-NAK every unkey looked for
    # record 0, and of its overs at levels 2-4 one was lost and one answered
    # 1.5 s late
    hs, io = _connected(bw='500')
    hs._link_high_speed = True
    hs._send_burst(VF.SESSION_OVER_NAK)
    assert hs._next_500 == 1
    full = phy.body_size('500', phy.level_key('500', 3)) - 1
    assert hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(full)), 3, 0x95)))
    assert hs._expect_level == 4 and hs._expect_500 == 4
    full = phy.body_size('500', 4) - 1
    x = _peer_over(hs, bytes(range(1, full + 1)), 4, 0x95)
    a = int(0.2 * MK.FS)
    audio = np.concatenate([np.zeros(a), x, np.zeros(2 * MK.FS)])
    audio += _band_noise(len(audio), np.sqrt(np.mean(x * x)), 7, 300, 2700)
    for i in range(0, len(audio), 960):
        clock.t += 0.02
        hs._stream_over_500(audio[i:i + 960])
        if len(io.host) == 2:
            break
    assert io.host[-1] == bytes(range(1, full + 1))
    assert i + 960 <= a + len(x) + int(0.3 * MK.FS)


@pytest.mark.parametrize('snr_db', [None, -6.0])
def test_the_idle_tick_keys_nothing_while_the_expected_over_may_be_on_the_air(snr_db, clock):
    # bench r4b, 2026-09-27 01:50Z: the tick keyed the short over-NAK 2.3 s
    # into WI9YZA's level-4 over, 0.04 s behind our continue
    hs, io = _high_speed_caller()
    hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(89)), 4, 0x41)))
    hs._expect_level = 4
    assert hs._answer_owed == VA._OWED_OVER and hs._lv_prompt
    x = _peer_over(hs, bytes(range(89, 0, -1)), 4, 0x81)
    if snr_db is not None:
        x = x + _band_noise(len(x), 10 ** (-snr_db / 20) * np.sqrt(np.mean(x * x)), 5)
    x = np.concatenate([np.zeros(int(0.04 * MK.FS)), x, np.zeros(MK.FS)])
    sent, window = len(io.sent), VA._EXPECT_WINDOW['2300'] / MK.FS
    for i in range(0, len(x), MK.FS // 10):
        clock.t += 0.1
        hs.on_rx_stream(x[i:i + MK.FS // 10])
        if abs(clock.t - hs._unkeyed_at - 2.3) < 0.05:
            hs.idle_keepalive()
            assert len(io.sent) == sent, io.msgs[-2:]
    if snr_db is None:
        assert len(io.host) == 2 and len(io.sent) == sent + 1
    clock.t, sent = hs._unkeyed_at + window + 0.1, len(io.sent)
    hs.idle_keepalive()
    assert len(io.sent) == sent + 1 and _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT)


def test_the_expected_reader_arms_at_the_close_level_when_we_release_the_turn(clock):
    # bench r5, 2026-09-27 01:53Z, and every run before it: WI9YZA's first
    # level-6 over behind our release went unread, its idle drew a NAK, and
    # it resent at level 4 — about 14 s per handover
    hs, io = _high_speed_caller()
    _peer_delivery(hs, bytes(range(89)), 0x06)
    hs.send(bytes(45))
    hs._took_control_burst()
    assert hs.turn == VA._TURN_PEER and hs._expect_level == 6 and hs._lv_armed
    assert _hear(hs, _peer_over(hs, b'roster', 6, 0x82, close=6)) is not None
    assert io.host[-1] == b'roster'


def _k9kdj_banner(hs):
    """K9KDJ on 80 m, 2026-09-27 05:11Z: its banner read at levels 1 and 2,
    each full and each answered +1."""
    hs._answer_data_over(_on_air(_peer_over(hs, b' has 1400', 1, 0x81)))
    hs._answer_data_over(_on_air(_peer_over(hs, b' daily minutes remaini', 2, 0x81)))
    assert b''.join(hs.io.host) == b' has 1400 daily minutes remaini'


def _stream(hs, clock, x):
    for i in range(0, len(x), MK.FS // 10):
        clock.t += 0.1
        hs.on_rx_stream(x[i:i + MK.FS // 10])


def test_a_high_speed_capable_peer_idling_before_an_over_could_follow_our_answer_hears_it_again(clock):
    # K9KDJ keyed 745 1.5-2.6 s behind four of our continues and no over
    # ahead of any; each drew our short over-NAK, it re-sent the over we had
    # delivered, lower, and fell from 5 to 1
    hs, io = _record2_delivery()
    _k9kdj_banner(hs)
    sent = len(io.sent)
    assert _asked_frames(hs, io)[-2:] == [6, 6]
    idle = MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, '2300'))
    _stream(hs, clock, np.concatenate([np.zeros(int(1.6 * MK.FS)), idle, np.zeros(MK.FS)]))
    assert len(io.sent) == sent + 1, io.msgs[-3:]
    assert not _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT), io.msgs[-3:]
    assert _asked_frames(hs, io)[-3:] == [6, 6, 6] and hs._expect_level == 3


@pytest.mark.parametrize('resend, host', [
    (((b' daily mi', 1), (b'nutes remaining with K', 2)),
     b' has 1400 daily minutes remaining with K'),
    (((b'9KDJ (EM67HW)\r{SFI = 1', 2),),
     b' has 1400 daily minutes remaini9KDJ (EM67HW)\r{SFI = 1'),
    (((b' daily mi', 1), (b'nutes', None)),
     b' has 1400 daily minutes remaini daily minutes')])
def test_an_over_our_nak_draws_again_repacked_reaches_the_host_once(resend, host, clock):
    # K9KDJ 05:12Z: behind our NAK it re-sent the level-2 over we had
    # delivered as a level-1 and a level-2 over, and the host had it twice;
    # new bytes, and a delivery closing inside the replay, all reach it
    hs, io = _record2_delivery()
    _k9kdj_banner(hs)
    clock.t = hs._unkeyed_at + VA._EXPECT_WINDOW['2300'] / MK.FS + 0.1
    hs.idle_keepalive()
    assert _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT), io.msgs[-2:]
    for payload, level in resend:
        over = (_peer_over(hs, payload, level, 0x81) if level
                else _peer_over(hs, payload, 2, 0x82, close=3))
        hs._answer_data_over(_on_air(over))
    assert b''.join(io.host) == host


@pytest.mark.parametrize('bw', ['500', '2300'])
def test_a_naked_high_speed_capable_over_goes_again_a_level_lower_each_time(bw, ceilings, overs):
    # K9KDJ 05:13Z: our 89-byte over at level 4 drew its 289/31 NAK and three
    # responder NAKs, and each retry went out at level 4 again
    ceilings(tx=phy.TOP[bw])
    hs, _ = sending(bw, bytes(89) + bytes(549))
    assert _nak(hs) == 3 and overs[-1][0] == 3
    assert _nak(hs) == 2 and overs[-1][0] == 2
    delivered(hs, 5)
    assert overs[-1][0] == 2


def test_a_peer_keying_under_4_is_expected_to_resend_under_4(clock):
    # K9KDJ 05:12Z: its level-2 over answered +1, our NAK, and the resend
    # came at level 1
    hs, io = _record2_delivery()
    _k9kdj_banner(hs)
    clock.t = hs._unkeyed_at + VA._EXPECT_WINDOW['2300'] / MK.FS + 0.1
    hs.idle_keepalive()
    assert _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT)
    assert hs._expect_level == 1


def _released_greeting(bw='2300', hs=None, io=None):
    """KC9GHZ 05:32Z and K7EK-10 05:38Z: the link up on the callee's
    turn-request, and the turn released to it for its level-6 greeting."""
    if hs is None:
        hs, io = _connected(bw=bw)
    hs.high_speed = True
    hs._tx_link_setup()
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_LINKSETUP_SENT
    ask = MK.synth_burst(hs.caller, VF.for_bw(VF.SESSION_TURN_REQUEST_RESPONDER, bw))
    hs.on_rx_audio(np.concatenate([np.zeros(2 * MK.HOP), ask, np.zeros(3 * MK.HOP)]))
    hs._release_turn()
    assert hs._expect_level == 6 and hs._lv_armed
    return hs, io


def _fsk_qrm(seed, seconds=3.8):
    """An FSK pair 18 dB over the band where the peer's over sits, which a
    level decode locates and cannot read."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, int(seconds * 45.45))
    f = np.repeat(np.where(bits, 1000.0, 830.0), int(MK.FS / 45.45))
    tone = np.sin(2 * np.pi * np.cumsum(f) / MK.FS) * 10 ** (18 / 20) * 0.42
    x = _band_noise(int((seconds + 1.5) * MK.FS), 1.0, seed)
    x[int(0.1 * MK.FS):int(0.1 * MK.FS) + len(tone)] += tone
    return x


def _over_idle(hs, clock):
    sent = len(hs.io.sent)
    idle = MK.synth_burst(hs.called, VF.SESSION_RESPONDER_OVER_IDLE)
    _stream(hs, clock, np.concatenate([np.zeros(MK.FS // 2), idle, np.zeros(2 * MK.FS)]))
    return hs.io.sent[sent:]


def test_a_greeting_resent_at_the_level_our_nak_refused_is_read(clock):
    # KC9GHZ 05:32Z: its level-6 greeting drew our NAK, the level-5 resend
    # our 61/31, and the greeting then came clean at 5 where we looked at 4
    hs, io = _released_greeting()
    _hear(hs, _fsk_qrm(1))
    assert _nak_keyed(hs, io) and hs._expect_level == 5
    _hear(hs, _fsk_qrm(1))
    assert _burst_named(io, hs.called, VF.SESSION_DATA_NAK_CALLER_SHORT)
    assert hs._expect_level == 4
    _hear(hs, _peer_over(hs, bytes(range(40)), 5, 0x82, close=6))
    assert io.host == [bytes(range(40))]


@pytest.mark.parametrize('naks', [1, 2])
def test_a_high_speed_peer_idling_over_an_over_we_could_not_read_draws_the_over_nak(naks,
                                                                                   clock):
    # KC9GHZ 05:32Z (two NAKs) and K7EK-10 05:38Z (one, then a window that
    # held nothing): its 745 drew keepalive-a, and the over stayed lost
    hs, io = _released_greeting()
    for _ in range(naks):
        _hear(hs, _fsk_qrm(1))
    _hear(hs, np.zeros(0), tail=VA._EXPECT_WINDOW['2300'] / MK.FS + 0.5)
    assert hs._answer_owed is None and hs._lv_owed and io.host == []
    sent = _over_idle(hs, clock)
    assert len(sent) == 1 and _burst_named(io, hs.called, VF.SESSION_OVER_NAK_SHORT), io.msgs[-2:]


def test_a_queried_over_naked_three_times_goes_again_a_level_lower_each_time(ceilings, overs):
    # K9KDJ 06:10Z: its close named 4, and our 89-byte over there drew 288/63
    # behind our first query and its NAK behind the next two; every retry
    # went out at 4 until the budget closed the link
    ceilings(tx=16)
    hs, io = sending('2300', bytes(89) + bytes(513))
    assert overs[-1] == (4, 1)
    hs._retry_data_over()
    hs._matched_data_nak = (hs._tx_pending, hs._pending_answer_at,
                            VF.for_bw(VF.SESSION_DATA_NAK_QUERY, '2300'))
    hs._took_data_nak()
    for _ in range(2):
        hs._retry_data_over()
        hs._took_responder_nak()
    assert [lv for lv, _ in overs[1:]] == [3, 2, 1], io.msgs[-6:]


def _idles(hs, clock, kind, n, gap, before=()):
    """``n`` of the peer's ``kind`` idles in band noise 6 dB under them, each
    ``gap`` s behind our last unkey or its last idle, the first behind
    ``before``. What each drew."""
    idle = MK.synth_burst(hs.called, VF.for_bw(kind, hs.bw))
    rms = np.sqrt(np.mean(idle * idle)) / 2
    drew = []
    for k in range(n):
        sent = len(hs.io.sent)
        x = np.concatenate([() if k else before, np.zeros(int(gap * MK.FS)), idle,
                            np.zeros(MK.FS // 5)])
        _stream(hs, clock, x + _band_noise(len(x), rms, len(hs.io.sent)))
        drew.append(hs.io.sent[sent:])
    return drew


def _is(sent, hs, kind):
    want = MK.synth_burst(hs.called, VF.for_bw(kind, hs.bw))
    return len(sent) == 1 and len(sent[0]) == len(want) and np.allclose(sent[0], want)


def _aa5af_waits(hs, clock):
    """AA5AF, 40 m, 2026-09-27 08:21Z: up on its turn-request, it idled
    85.7 s on the CMS before its greeting, 683 twice and then 745. What
    each drew."""
    return (_idles(hs, clock, VF.SESSION_RESPONDER_IDLE, 1, 8.6)
            + _idles(hs, clock, VF.SESSION_RESPONDER_IDLE, 1, 1.9)
            + _idles(hs, clock, VF.SESSION_RESPONDER_OVER_IDLE, 6, 1.9))


def test_a_high_speed_peer_idling_before_its_greeting_draws_what_stock_keys(clock):
    # AA5AF: each idle drew our over-NAK, and the greeting came at level 1
    # where our link-setup had asked 6
    hs, io = _released_greeting('500')
    drew = _aa5af_waits(hs, clock)
    assert [_is(s, hs, VF.SESSION_OVER_NAK) for s in drew[:2]] == [True] * 2, io.msgs[-4:]
    assert [_is(s, hs, VF.SESSION_KEEPALIVE_A) for s in drew[2:]] == [True] * 6, io.msgs[-4:]
    assert hs._expect_level == 6 and not hs._rx_failed


def test_a_high_speed_peer_idling_behind_an_over_length_emission_draws_the_over_nak(clock):
    # a greeting due at the base, where no reader named what came
    hs, io = _released_greeting('500')
    hs._expect_level, hs._lv_armed = 4, False
    rms = np.sqrt(np.mean(MK.synth_burst(hs.called, VF.for_bw(
        VF.SESSION_RESPONDER_OVER_IDLE, '500')) ** 2)) / 2
    over = np.concatenate([_band_noise(int(8.6 * MK.FS), rms, 1),
                           _band_noise(int(4.3 * MK.FS), rms, 2, 1150, 1850)
                           + _band_noise(int(4.3 * MK.FS), rms, 3)])
    drew = _idles(hs, clock, VF.SESSION_RESPONDER_OVER_IDLE, 1, 0.5, over)
    assert _is(drew[0], hs, VF.SESSION_OVER_NAK) and hs._rx_failed, io.msgs[-3:]


def test_a_high_speed_peer_that_cannot_read_our_continue_hears_it_up_a_ladder(clock):
    # AA5AF: its 745s came 2.0 s behind the 8-symbol continue that answered
    # its level-2 over; we re-keyed that burst four times, then keyed
    # keepalive-a until the link closed. It read our 32-symbol frames
    hs, io = _released_greeting('500')
    _aa5af_waits(hs, clock)
    for n, level in enumerate((1, 2)):
        full = phy.body_size('500', phy.level_key('500', level)) - 1
        hs._answer_data_over(_on_air(_peer_over(hs, bytes(range(n, full + n)), level, 0x95)))
    assert len(io.host) == 2
    continued = io.sent[-1]
    drew = _idles(hs, clock, VF.SESSION_RESPONDER_OVER_IDLE, 5, 2.0)
    assert [len(s) for s in drew] == [1] * 5, io.msgs[-6:]
    assert np.array_equal(drew[0][0], continued), io.msgs[-8:]
    assert _is(drew[1], hs, replace(VF.SESSION_OVER_RESPONSE, preadv=373)), io.msgs[-6:]
    assert all(_is(s, hs, VF.SESSION_OVER_NAK) for s in drew[2:]), io.msgs[-4:]


@pytest.mark.parametrize('host, payload', [(1, 9), (2, 22)])
def test_a_greeting_at_the_level_the_connect_response_named_is_read_under_our_ask(
        host, payload, clock):
    # AA5AF, 04:19Z and 08:21Z: its connect-response named level 1 and its
    # greeting came at 1 whatever our link-setup asked; read as a level-4
    # over, our answer asked it up to 5
    hs, io = _connected(bw='500')
    hs._setup_level = 1
    hs, io = _released_greeting('500', hs, io)
    _aa5af_waits(hs, clock)
    x = _peer_over(hs, bytes(range(payload)), host, 0x95)
    x = np.concatenate([np.zeros(int(0.05 * MK.FS)), x, np.zeros(2 * MK.FS)])
    sent = len(io.sent)
    _stream(hs, clock, x + _band_noise(len(x), np.sqrt(np.mean(x * x)) / 2, 1))
    assert io.host == [bytes(range(payload))], io.msgs[-4:]
    assert len(io.sent) == sent + 1, io.msgs[-4:]
    assert _is(io.sent[-1:], hs, replace(VF.SESSION_OVER_RESPONSE, preadv=373)), io.msgs[-4:]
    assert hs._expect_level == host + 1


AA5AF_TAPE = evidence.LOGS / 'onair' / '20260927T082226Z-W9SSJ-AA5AF.flac'


@pytest.mark.skipif(not AA5AF_TAPE.exists(), reason='on-air tape not kept here')
def test_aa5af_level_1_greeting_expected_at_5_is_read_whole_not_naked_arriving(clock):
    # 85.86-91.3 s: behind our over-NAK, AA5AF's greeting at level 1
    from hfmodem.core import wav
    x = wav.read(str(AA5AF_TAPE))
    hs, io, answered = _listening('500', 5, 1)
    hs._rx_failed = True
    _live(hs, (x[:, 0] if x.ndim > 1 else x)[4121280:4464000])
    assert answered and hs._level_read is None and not io.sent, io.msgs
