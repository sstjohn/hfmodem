# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Fit-byte pins against recorded VARA HF links; no audio or devices."""
import json
from pathlib import Path

import pytest

from hfmodem.kestrel.arq import phy, rxwindow as RW
from hfmodem.kestrel.vara import vara_arq as VA
from .test_data_over_gate import _connected


DECODE = [
    (0x99, 6, False, True), (0x95, 5, False, True),
    (0x91, 4, False, True), (0x8d, 3, False, True),
    (0x89, 2, False, True), (0x85, 1, False, True),
    (0x81, 0, False, True), (0x80, 0, False, False),
    (0x82, 0, True, False), (0x19, 6, False, True),
    (0x51, 4, False, True), (0x59, 6, False, True),
    (0x00, 0, False, False), (0x02, 0, True, False),
    (0x40, 0, False, False), (0x42, 0, True, False),
]


@pytest.mark.parametrize('field,hint,last,full', DECODE)
@pytest.mark.parametrize('compressed', [False, True])
def test_u1_control_byte(field, hint, last, full, compressed):
    field |= 0x20 if compressed else 0
    assert phy.fit_hint(field) == hint
    assert phy.over_is_last(bytes([0, field]), 'W9SSJ') == last
    assert bool(field & 1) == full
    assert bool(field & 0x20) == compressed


@pytest.mark.parametrize('level,rest,field', [
    (4, 5911, 0x99), (12, 4134, 0x91), (14, 1978, 0x89),
    (14, 1478, 0x81), (12, 2734, 0x99), (14, 3628, 0x85),
    (15, 5468, 0x8d), (16, 2214, 0x81), (16, 6398, 0x85),
    (16, 3144, 0x89), (16, 3254, 0x81),
])
def test_u2_q3_writer(level, rest, field):
    """ks-q3-fit-6000/32200, 2026-10-31: strict fit, one frame first."""
    assert 0x81 | phy.fit_code('2300', level, rest, 1, 16) << 2 == field


def cap(bw, level):
    return phy.body_size(bw, phy.level_key(bw, level)) - 1


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_u2_all_capacity_boundaries(bw):
    top = phy.TOP[bw]
    for level in range(1, top + 1):
        choices = [(k, offset, k * cap(bw, lv),
                    0 if offset == -1 else here if offset == 0 else offset + 2)
                   for k, here in ((1, 2), (2, 1))
                   for offset in range(-1, 5)
                   if 1 <= (lv := level + offset) <= top]
        for _, _, boundary, _ in choices:
            for rest in (boundary - 1, boundary, boundary + 1):
                fits = sorted(c for c in choices if rest < c[2])
                want = fits[0][3] if fits else 6
                assert phy.fit_code(bw, level, rest, 1, top) == want, (bw, level, rest)
        if level > 1:
            assert cap(bw, level - 1) < cap(bw, level)


PEER = json.loads((Path(__file__).with_name('fixtures') / 'fit-hint-peer.json').read_text())


@pytest.mark.parametrize('row', PEER, ids=lambda r: f"{r['tape']}-L{r['level']}-{r['rest']}")
def test_u2_peer_fact_table(row):
    """VARA HF counter and ladder fetches, 2026-09-26."""
    code = phy.fit_code(row['bw'], row['level'], row['rest'], 1, phy.TOP[row['bw']])
    assert code == row['code']
    if row['code'] == (row['field'] >> 2) & 7:
        assert code == phy.fit_hint(row['field'])


@pytest.mark.parametrize('bw,rest,announcement,fit', [
    ('2300', 40, 0x81, 0), ('2300', 60, 0x89, 2),
    ('500', 20, 0x81, 0), ('500', 40, 0x89, 2),
])
def test_u2_close_pairings_and_ladder_exception(bw, rest, announcement, fit):
    """2026-09-09 fetch; 2026-10-31 hold runs."""
    assert phy.fit_code(bw, 4, rest, 1, phy.TOP[bw]) == fit
    assert VA._frame_field(0, announcement == 0x89) == announcement


def test_u2_september_nine_delivery():
    """524-byte fetch, 2026-09-09: first hint also uses the fit scale."""
    rest = [435, 346, 257, 168, 79]
    observed = [0x95, 0x99, 0x95, 0x91, 0x89, 0x82]
    assert [phy.fit_hint(f) for f in observed] == [5, 6, 5, 4, 2, 0]
    assert [0x81 | phy.fit_code('2300', 4, n, 1, 16) << 2 for n in rest] == [
        0x95, 0x95, 0x99, 0x91, 0x89]


def station(bw='2300', level=12, high=True):
    hs, io = _connected(bw=bw)
    hs._link_high_speed = high
    hs._rx_read = level, 40.0
    hs._expect_level = level
    hs._next_500 = min(level, 4)
    return hs, io


@pytest.mark.parametrize('hint,frame', [(0, 5), (1, 0), (2, 0), (3, 6),
                                        (4, 4), (5, 7), (6, 8), (7, 8)])
def test_u3_hint_caps_snr(hint, frame):
    hs, _ = station(level=6)
    hs._peer_hint = hint
    assert hs._irs_step() == frame
    assert hs._expect_level == 6 + VA._LEVEL_STEP[frame]


@pytest.mark.parametrize('level,field,frame', [(14, 0x81, 5), (12, 0x8d, 6),
    (12, 0x99, 4), (14, 0x85, 0), (16, 0x81, 5)])
def test_u3_q3_steps(level, field, frame):
    hs, _ = station(level=level)
    hs._peer_hint = phy.fit_hint(field)
    assert hs._irs_step() == frame


@pytest.mark.parametrize('hint', range(8))
def test_u3_weak_snr_wins(hint):
    hs, _ = station()
    hs._rx_read = 12, -20.0
    hs._peer_hint = hint
    assert hs._irs_step() == 5


def test_u3_floor_closing_and_unknown():
    hs, _ = station(level=1)
    hs._peer_hint = 0
    assert hs._irs_step() == 0
    hs._rx_read = 12, 40.0
    assert hs._irs_step(closing=True) == 0
    hs._peer_hint = None
    assert hs._irs_step() == 4


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_u3_limited_wide_keeps_hold(bw):
    hs, _ = station(bw, level=4, high=False)
    hs._peer_hint = 0
    hs._continue_tail()
    assert hs._continue_step == 0


def window(hs, number, field=0x99, level=None, repeat=False, hold=False, pair=False):
    level = hs._rx_read[0] if level is None else level
    body = number.to_bytes(2, 'big') + bytes([field])
    bodies = [b'head\x99', body] if pair else [body]
    hs._note_peer_window(level, bodies, repeat, hold)
    hs._rx_read = level, 40.0
    hs._rx_window.A = RW.Answer(number, hs._rx_window.alt,
        'final' if field & 2 else 'continue', original=VA.OVER_CONTINUE_GENERATED)
    return bodies


def key(hs, frame=0, sent=True, nak=False):
    hs._continue_step = frame
    hs.io.tx_went_out = lambda: sent
    aired = hs._transmit_audio([], packet_kind='data-nak' if nak else 'over-continue')
    if aired:
        if nak and hs._rx_window.A is not None and hs._rx_window.A.kind == 'nak':
            hs._holds_nak_double = hs._rx_window.A.retry_double
            hs._holds_nak_level = hs._rx_window.A.retry_level
        hs._peer_answer_keyed(1 if nak else frame)
        if hs._rx_window.A is not None:
            hs._rx_window.A.sent = True
            if hs._rx_window.A.kind == 'final':
                hs._reset_peer_holds(confirmed=True)
    return aired


def six(hs):
    for n in range(6):
        window(hs, n)
        key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)


@pytest.mark.parametrize('bw,level,verdict', [
    ('2300', 2, 'CERTAIN'), ('2300', 3, 'CERTAIN'), ('2300', 7, 'CERTAIN'),
    ('2300', 10, 'CERTAIN'), ('2300', 16, 'CERTAIN'),
    ('2750', 4, 'CERTAIN'), ('500', 13, 'CERTAIN'), ('500', 6, 'POSSIBLE')])
def test_u4_six_keyed_holds(bw, level, verdict):
    """ks-hold2/3/7/10 and ss-n32200/35000, 2026-10-31."""
    hs, _ = station(bw, level)
    six(hs)
    assert hs._peer_double_verdict() == verdict
    assert hs._peer_doubles() == (verdict == 'CERTAIN')
    assert hs._peer_may_double()
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)


@pytest.mark.parametrize('event', ['step', 'partial', 'nak', 'incomplete',
                                   'compressed', 'close', 'turn', 'link'])
def test_u4_reset_table(event):
    hs, _ = station(level=10)
    six(hs)
    if event == 'step':
        hs._expect_level = 11
        key(hs, 6)
    elif event == 'partial':
        hs._expect_level = 9
        key(hs, 10)
    elif event == 'nak':
        key(hs, nak=True)
        hs._expect_level = 7
    elif event == 'incomplete':
        hs._deliver(window(hs, 7, hold=True), hold=True)
        hs._finish_delivery_window(complete=False)
    elif event == 'compressed':
        window(hs, 7, 0xb9)
        key(hs)
    elif event == 'close':
        window(hs, 7, 0x82)
        key(hs)
    elif event == 'turn':
        hs.turn = VA._TURN_OURS
    else:
        hs.state = VA.VaraState.DISCONNECTED
    both = event in ('compressed', 'close', 'turn', 'link')
    # DESIGN §3, second amendment: an aired NAK alone proves no reset.
    epoch_holds = 6 if event in ('nak', 'incomplete') else 0
    assert (hs._holds_epoch, hs._holds_any) == (epoch_holds, 0 if both else 6)
    assert hs._peer_double_verdict() == ('NONE' if both else 'CERTAIN' if event in ('nak', 'incomplete') else 'POSSIBLE')


@pytest.mark.parametrize('event', ['step', 'partial', 'nak'])
@pytest.mark.parametrize('arrival', ['obeyed', 'obeyed_repeat', 'repeat', 'disobeyed'])
def test_u4_peer_confirms_reset(event, arrival, quiet_answers):
    hs, io = station(level=10)
    for letter in b'ABCDEF':
        read_window(hs, letter, 10)
    key(hs, 6 if event == 'step' else 10 if event == 'partial' else 1, nak=event == 'nak')
    hs._expect_level = 11 if event == 'step' else 9
    level = hs._expect_level if arrival.startswith('obeyed') else 10
    value = 5 if arrival == 'repeat' else 7
    body = value.to_bytes(2, 'big') + b'\x99'
    hs._level_read = phy.LevelRead(level, (body,), 40.0, 1)
    assert hs._answer_data_over([])
    hs._level_read = None

    assert hs._holds_any == 1
    key(hs)
    assert hs._holds_epoch == 1
    assert hs._holds_any == 1
    assert b''.join(io.host) == b''.join(bytes([c]) * cap('2300', 10) for c in b'ABCDEF') + value.to_bytes(2, 'big')


@pytest.mark.parametrize('pair', [False, True])
def test_u4_h2_resets_epoch_but_query_does_not_count_again(pair):
    """ss-n29200 and ks-hold3-1176-1 (double's last h2), 2026-10-31."""
    hs, _ = station(level=3)
    for n in range(5):
        window(hs, n)
        key(hs)
    window(hs, 5, 0x89, pair=pair)
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (0, 6)
    assert hs._peer_double_verdict() == 'POSSIBLE'
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (0, 6)
    for n in range(6, 12):
        window(hs, n)
        key(hs)
    assert hs._peer_double_verdict() == 'CERTAIN'


@pytest.mark.parametrize('bw,level', [('2300', 16), ('2750', 17), ('500', 13)])
@pytest.mark.parametrize('route', ['query', 'ladder'])
def test_u4_refused_first_answer_then_reanswer_counts_once(bw, level, route, quiet_answers):
    """Xara's sender bumps stays on each ACK_STAY it hears; the peer counted an
    erased sixth top hold heard only as query re-answer 288/1, then doubled
    (b4-sgl11-35000-1, 2026-10-31).
    """
    hs, io = station(bw, level)
    hs.role = 'responder'
    reanswer = hs._reanswer_over if route == 'query' else lambda: hs._reack(query_start=hs._unkeyed_at + .1, recognized=True)
    for n, letter in enumerate(b'ABCDEF'):
        io.tx_went_out = lambda: False
        read_window(hs, letter, level)
        assert (hs._holds_epoch, hs._holds_any) == (n, n)
        io.tx_went_out = lambda: True
        reanswer()
        assert (hs._holds_epoch, hs._holds_any) == (n + 1, n + 1)
        reanswer()
        assert (hs._holds_epoch, hs._holds_any) == (n + 1, n + 1)
    assert hs._peer_double_verdict() == 'CERTAIN'


@pytest.mark.parametrize('count', [0, 5, 6])
@pytest.mark.parametrize('answer', ['hold', 'step', 'partial', 'nak'])
def test_u4_refused_answer_never_keyed_preserves_counts(count, answer):
    """Xara counts heard ACK_STAY; the peer counted the recovered sixth hold
    (b4-sgl11-35000-1, 2026-10-31): a refused write changes neither count.
    """
    hs, _ = station(level=10)
    for n in range(count):
        window(hs, n)
        key(hs)
    window(hs, 7)
    answered = hs._hold_answered
    for _ in range(2):
        key(hs, {'hold': 0, 'step': 6, 'partial': 10, 'nak': 1}[answer],
            sent=False, nak=answer == 'nak')
        assert (hs._holds_epoch, hs._holds_any) == (count, count)
        assert hs._hold_answered is answered
    window(hs, 8)
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (count + 1, count + 1)


@pytest.mark.parametrize('route', ['query', 'ladder'])
def test_u4_refused_reanswer_of_counted_window_preserves_counts(route, quiet_answers):
    """Xara counts heard ACK_STAY; the peer's sixth hold query recovery caused
    its double (b4-sgl11-35000-1, 2026-10-31); one window counts once.
    """
    hs, io = station(level=16)
    hs.role = 'responder'
    for letter in b'ABCDEF':
        read_window(hs, letter, 16)
    reanswer = hs._reanswer_over if route == 'query' else lambda: hs._reack(query_start=hs._unkeyed_at + .1, recognized=True)
    io.tx_went_out = lambda: False
    reanswer()
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._peer_double_verdict() == 'CERTAIN'
    io.tx_went_out = lambda: True
    reanswer()
    reanswer()
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._peer_double_verdict() == 'CERTAIN'


def test_u4_nak_at_floor_replayed_payload_counts_in_new_epoch():
    """DESIGN §3 decision B: DATA after an aired NAK resets even at the floor."""
    hs, _ = station(level=1)
    six(hs)
    key(hs, nak=True)
    window(hs, 5, repeat=True)
    assert (hs._holds_epoch, hs._holds_any) == (0, 0)
    assert hs._peer_double_verdict() == 'NONE'
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)


@pytest.mark.parametrize('sent', [False, True])
def test_u4_unheard_nak_queries_do_not_confirm_reset(sent):
    """DESIGN §3, second amendment: unheard/refused NAKs draw queries, no DATA."""
    hs, _ = station(level=1)
    six(hs)
    for _ in range(3):
        key(hs, nak=True, sent=sent)
        assert (hs._holds_epoch, hs._holds_any) == (6, 6)
        assert hs._peer_double_verdict() == 'CERTAIN'
    if not sent:
        window(hs, 5, repeat=True)
        assert (hs._holds_epoch, hs._holds_any) == (6, 6)


@pytest.mark.parametrize('route', ['r5c', 'r5d-bracket', 'r5d-stream', 'r5d-timeout'])
@pytest.mark.parametrize('streamed', [False, True])
def test_r5c_r5d_data_after_aired_nak_confirms_floor_reset(route, streamed, quiet_answers):
    """DESIGN §3: reset both counts; an incomplete double still permits a pair."""
    hs, _ = station(level=1)
    six(hs)
    if route == 'r5d-bracket':
        hs._undecoded_over()
    elif route == 'r5d-stream':
        hs._irs_nak(phy.LevelRead(1, (), None, 1))
    elif route == 'r5d-timeout':
        hs._deliver(window(hs, 7, hold=True), hold=True)
        hs._held_answer = False, False
        hs._release_held_answer(VA._ANSWER_HOLD_MAX)
    else:
        key(hs, nak=True)
    counts = hs._holds_epoch, hs._holds_any
    hs._nak_unheard = True
    assert hs._nak_owed_over()
    assert (hs._holds_epoch, hs._holds_any) == counts
    window(hs, 5, repeat=True, hold=streamed)
    assert (hs._holds_epoch, hs._holds_any) == (0, 0)
    assert hs._holds_retry_double == route.startswith('r5d')
    assert hs._peer_double_verdict() == ('POSSIBLE' if route.startswith('r5d') else 'NONE')
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)
    window(hs, 8)
    assert hs._peer_double_verdict() == 'NONE'


@pytest.mark.parametrize('field,verdict', [(0xb9, 'NONE'), (0x89, 'POSSIBLE')])
def test_u4b_bw500_high_speed_top(field, verdict):
    """ss-text-60k-1 and ss-n29200, 2026-10-31; BW500's high speed capable top is L13."""
    hs, _ = station('500', 13)
    for n in range(6):
        window(hs, n, field if n == 5 or field == 0xb9 else 0x99)
        key(hs)
    assert hs._peer_double_verdict() == verdict
    assert hs._peer_run_500 == 0


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('field,verdict', [(0x99, 'CERTAIN'), (0x89, 'NONE'), (0xb9, 'NONE')])
def test_u4_bw500_limited_partition(level, field, verdict):
    hs, _ = station('500', level, high=False)
    hs._peer_over_state = level, field
    hs._peer_run_500 = 6
    assert hs._peer_double_verdict() == verdict
    hs._note_peer_500(level, [b'new\xb9'])
    assert hs._peer_run_500 == 0 and not hs._peer_paired_500


@pytest.mark.parametrize('n', [0, 8, 45, 47])
def test_u5_nonfinal_head_payload(n):
    size = 48
    payload = b'x' * n
    body = (payload + b'\x81' if n == 47 else
            (payload + b'\x14\xf6').ljust(size - 1, b'\0') + b'\x80')
    assert phy.vara_payload(body, caller='W9SSJ', body_len=size) == payload
    assert phy.vara_payload(phy.vara_body(b'', 'W9SSJ', body_len=size),
                            caller='W9SSJ', body_len=size) == b''


@pytest.mark.parametrize('compressed', [False, True])
def test_u5_xara_bare_delimiter(compressed):
    body = b'x' * 46 + b'\x14' + bytes([0xa0 if compressed else 0x80])
    assert phy.vara_payload(body, caller='W9SSJ', body_len=48) == b'x' * 46


def test_u5_full_literal_delimiter():
    body = b'x' * 46 + b'\x14\x81'
    assert phy.vara_payload(body, caller='W9SSJ', body_len=48) == body[:-1]


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('rest', [30, 600])
def test_c7_held_retry_writes_fit(bw, rest, monkeypatch):
    """ks-q3-fit-6000/32200, 2026-10-31: a held retry describes fit, not count."""
    hs, _ = station(bw, level=2, high=False)
    hs.turn = VA._TURN_OURS
    level = phy.level_key(bw, 2)
    load = cap(bw, 2)
    hs._txq = [b'x' * load] + [b'y' * min(load, rest - i) for i in range(0, rest, load)]
    if len(hs._txq[-1]) == load:
        hs._txq.append(b'')
    hs._tx_recovery_levels = [level] * len(hs._txq)
    monkeypatch.setattr(hs, '_render_over', lambda *args: [])
    hs._tx_data_over()
    field = hs._tx_pending[0][-1]
    assert field == (0x8d if rest == 30 else 0x99)
    assert (field >> 2) & 7 >= (3 if rest == 600 else 2)


def test_u2_limited_writer_uses_bandwidth_top():
    """ks-lim-hold2-1500-1/3, 2026-10-31: L4/432 announces L7 (h5)."""
    assert phy.fit_code('2300', 4, 432, 1, 16) == 5


@pytest.mark.parametrize('bw,host,rest,field', [
    ('2300', 14, 3628, 0x85), ('2300', 14, 1478, 0x81),
    ('2750', 4, 432, 0x95), ('500', 4, 100, 0x91),
])
def test_u2_adaptive_writer_uses_payload_behind_head(bw, host, rest, field, monkeypatch):
    hs, _ = station(bw, host)
    hs.turn = VA._TURN_OURS
    level = phy.level_key(bw, host)
    hs._relevel(level, 'test')
    load = cap(bw, host)
    hs._txq = [b'x' * load] + [b'y' * min(load, rest - i) for i in range(0, rest, load)]
    if len(hs._txq[-1]) == load:
        hs._txq.append(b'')
    if level != phy.base_level(bw):
        hs._tx_recovery_levels = [level] * len(hs._txq)
    hs._txq += [b'z' * load, b'later delivery']
    monkeypatch.setattr(hs, '_render_over', lambda *args: [])
    hs._tx_data_over()
    assert hs._tx_pending[0][-1] == field


@pytest.mark.parametrize('bw,host,high,rest,field', [
    ('2300', 16, True, 1000, 0x81), ('2750', 17, True, 1000, 0x81),
    ('500', 4, False, 40, 0x89),
])
def test_u2_pair_second_frame_excludes_both_payloads(bw, host, high, rest, field):
    hs, _ = station(bw, host, high)
    level = phy.level_key(bw, host)
    hs._relevel(level, 'test')
    hs._singles = 6
    hs._txq = [b'a' * cap(bw, host), b'b' * cap(bw, host), b'c' * rest]
    assert hs._pair(level)[-1] == field


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_c7_nak_repacketization_fit_not_countdown(bw, monkeypatch):
    """ks-q3-fit-6000-1, 2026-10-31: held retry may climb, on fit arithmetic."""
    from .test_nak_repacketization import pending, unconfirmed
    monkeypatch.setattr(VA.VaraStationHandshake, '_render_over', lambda *args: [])
    monkeypatch.setattr(VA.MK, 'synth_burst', lambda *args: [])
    hs, _, data = pending(bw, 94)
    hs._took_responder_nak()
    assert hs._tx_pending[0][-1] == 0x91
    assert unconfirmed(hs) == data


def test_u4_incomplete_positive_reanswer_cannot_count_a_hold():
    hs, _ = station(level=10)
    six(hs)
    hs._deliver(window(hs, 7, hold=True), hold=True)
    hs._finish_delivery_window(complete=False)
    hs._reanswer_over(query_start=hs._unkeyed_at + .1)
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)


def test_u4_turn_reset_has_no_window_to_reanswer():
    hs, _ = station(level=10)
    six(hs)
    hs.turn = VA._TURN_OURS
    key(hs)
    assert (hs._holds_epoch, hs._holds_any) == (0, 0)


@pytest.mark.parametrize('field,hint,compressed', [(0x19, 6, False), (0x85, 1, False),
                                                  (0xb8, 6, True), (0x82, 0, False)])
def test_u5_window_extracts_last_frame_including_first(field, hint, compressed):
    """ss-n35000-1 / ss-text-60k-1, 2026-10-31: last frame governs the answer."""
    hs, _ = station()
    assert not hs._peer_delivery_open
    hs._note_peer_window(12, [b'head\x85', bytes([0, field])], False, False)
    assert (hs._peer_hint, hs._peer_compressed) == (hint, compressed)


@pytest.mark.parametrize('level', [2, 3, 7, 10, 16])
def test_u4_real_answer_route_counts_keyed_windows(level, monkeypatch):
    """ks-hold2/3/7/10 and ss-n35000-1, 2026-10-31: holds at every row."""
    hs, io = station(level=level)
    monkeypatch.setattr(VA.MK, 'synth_burst', lambda *args: [])
    monkeypatch.setattr(VA.MK, 'synth_tone_pairs', lambda *args: [])
    def hold_step():
        hs._expect_level = level
        return 0
    monkeypatch.setattr(hs, '_irs_step', hold_step)
    for n in range(6):
        body = bytes([65 + n]) * cap('2300', level) + b'\x99'
        hs._level_read = phy.LevelRead(level, (body,), 40.0, 1)
        hs._answer_data_over([])
        hs._level_read = None
    assert len(io.sent) == 6
    assert hs._peer_doubles()
    hs._reanswer_over()
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)


def test_u4b_compression_through_high_speed_bw500_answer_route(monkeypatch):
    """ss-text-60k-1, 2026-10-31: even nine compressed top holds stay single."""
    hs, _ = station('500', 13)
    monkeypatch.setattr(VA.MK, 'synth_burst', lambda *args: [])
    monkeypatch.setattr(VA.MK, 'synth_tone_pairs', lambda *args: [])
    for n in range(9):
        body = bytes([65 + n]) * cap('500', 13) + b'\xb9'
        hs._level_read = phy.LevelRead(13, (body,), 40.0, 1)
        hs._answer_data_over([])
        hs._level_read = None
    assert not hs._peer_doubles()
    assert hs._peer_double_verdict() == 'NONE'


@pytest.mark.parametrize('level', [2, 3, 4])
def test_u3_limited_bw500_keeps_its_answer_step(level):
    hs, _ = station('500', level, high=False)
    hs._peer_over_state = level, 0x81
    hs._peer_hint = 0
    hs._continue_tail()
    assert hs._continue_step == 5
    assert hs._next_500 == level - 1


@pytest.mark.parametrize('route', ['bracket', 'stream', 'unread', 'timeout'])
def test_u4_incomplete_routes_preserve_upper_bound(route, monkeypatch):
    hs, _ = station(level=10)
    six(hs)
    monkeypatch.setattr(VA.MK, 'synth_burst', lambda *args: [])
    monkeypatch.setattr(VA.MK, 'synth_tone_pairs', lambda *args: [])
    hs.io.tx_went_out = lambda: False
    if route == 'bracket':
        hs._undecoded_over()
    elif route == 'stream':
        hs._irs_nak(phy.LevelRead(10, (), None, 1))
    elif route == 'unread':
        hs._owe_unread_over()
    else:
        hs._deliver(window(hs, 7, hold=True), hold=True)
        hs._held_answer = False, False
        hs._release_held_answer(VA._ANSWER_HOLD_MAX)
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._peer_double_verdict() == 'CERTAIN'


def test_u4_replayed_payload_is_suppressed_but_its_hold_counts_in_new_epoch(monkeypatch):
    """DESIGN §3, second amendment: replay after our NAK resets both counts."""
    hs, io = station(level=1)
    monkeypatch.setattr(VA.MK, 'synth_burst', lambda *args: [])
    monkeypatch.setattr(VA.MK, 'synth_tone_pairs', lambda *args: [])
    def hold_step():
        hs._expect_level = 1
        return 0
    monkeypatch.setattr(hs, '_irs_step', hold_step)
    for n in range(6):
        body = bytes([65 + n]) * cap('2300', 1) + b'\x99'
        hs._level_read = phy.LevelRead(1, (body,), 40.0, 1)
        hs._answer_data_over([])
    hs._rx_query(query_start=hs._unkeyed_at + 60)
    assert hs._rx_window.A.sent and hs._rx_window.A.kind == 'nak'
    assert (hs._rx_window.T.start, hs._rx_window.T.frontier) == (hs._rx_window.K.start, hs._rx_window.S)
    hs._answer_data_over([])
    hs._level_read = None
    assert len(io.host) == 6
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)
    hs._reanswer_over()
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)


def test_u2_explicit_low_ladder_announcement_wins_over_fit(monkeypatch):
    """ks-lim-hold2-1500-1/3, 2026-10-31: fit and a forced ladder may disagree."""
    hs = VA.VaraStationHandshake(['W9SSJ'], VA.VaraIO(), bw='2750', tx_level=1)
    hs.caller, hs.called = 'W9SSJ', 'W1AW'
    hs.send(b'x' * 196)
    monkeypatch.setattr(hs, '_render_over', lambda *args: [])
    hs._tx_data_over()
    assert hs._tx_pending[2] == phy.level_key('2750', 4)
    assert hs._tx_pending[0][-1] == 0x81
    assert phy.fit_code('2750', 4, 196 - 89, 1, 17) == 3


@pytest.mark.parametrize('field', [0x19, 0x59, 0x99, 0xd9])
def test_u4_reply_bits_do_not_create_another_counting_window(field, quiet_answers):
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new."""
    hs, io = station(level=10)
    body = b'F' * cap('2300', 10) + bytes([field])
    for letter in b'ABCDE':
        read_window(hs, letter, 10)
    hs._level_read = phy.LevelRead(10, (body,), 40.0, 1)
    assert hs._answer_data_over([])
    assert hs._answer_data_over([])
    assert hs._rx_window.state == 'OPEN' and hs._held_answer is not None
    hs._rx_query(query_start=hs._query_audio_at())
    hs._level_read = None
    assert b''.join(io.host) == b''.join(bytes([c]) * cap('2300', 10) for c in b'ABCDEF')
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)


@pytest.fixture
def quiet_answers(monkeypatch):
    monkeypatch.setattr(VA.MK, 'synth_burst', lambda *args: [])
    monkeypatch.setattr(VA.MK, 'synth_tone_pairs', lambda *args: [])
    monkeypatch.setattr(VA, '_irs_frame', lambda *args: 0)


def read_window(hs, letter, level):
    body = bytes([letter]) * cap(hs.bw, level) + b'\x99'
    hs._level_read = phy.LevelRead(level, (body,), 40.0, 1)
    try:
        assert hs._answer_data_over([])
    finally:
        hs._level_read = None


@pytest.mark.parametrize('level,stray', [(2, 3), (10, 8), (16, 14)])
def test_u4_equal_payload_new_window_occurrences(level, stray, quiet_answers):
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new.

    Behind the NAK the peer keys a level nobody asked for: a clean retry at
    the NAKed level is the retry since 15d6f5d39."""
    hs, io = station(level=level)
    outcomes = []
    for letter in b'ABACDE':
        for duplicate in (False, True):
            if duplicate:
                hs._reanswer_over()
            lv = level if hs._rx_resend is None else stray
            hs._level_read = phy.LevelRead(lv, (bytes([letter]) * cap('2300', lv) + b'\x99',), 40.0, 1)
            outcomes.append(hs._answer_data_over([]))
            hs._level_read = None
    assert outcomes == [True] * 7 + [False] * 5
    assert len(io.host) == 6
    assert b''.join(io.host) == b''.join(bytes([c]) * cap('2300', level) for c in b'AABBAA')
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._peer_double_verdict() == 'POSSIBLE'
    assert hs._rx_window.A.kind == 'nak'


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_u4_limited_wide_recovery_to_l2_six_holds(bw, quiet_answers):
    """ks-lim-hold2-1500-1/3, 2026-10-31: hold the recovered record."""
    hs, io = station(bw, level=4, high=False)
    hs.role = 'responder'
    assert hs._undecoded_over()
    assert io.sent
    for letter in b'ABCDEF':
        read_window(hs, letter, 2)
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._expect_level == 2
    assert hs._peer_double_verdict() == 'CERTAIN'


@pytest.mark.parametrize('bw,start', [('2300', 8), ('2750', 8), ('500', 8)])
def test_u4_nak_reanswer_preserves_rejected_geometry(bw, start, quiet_answers):
    hs, _ = station(bw, level=start)
    six(hs)
    hs._irs_nak(phy.LevelRead(start, (), None, 1))
    assert hs._expect_level == 6
    assert hs._holds_asked_from == start
    hs._nak_unheard = True
    assert hs._nak_owed_over()
    assert hs._expect_level == 6
    assert hs._holds_any == 6
    read_window(hs, ord('R'), 6)
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)
    assert hs._holds_asked_from is None
    assert hs._peer_double_verdict() == 'NONE'


def test_u4_turn_yield_counts_first_answer_in_new_turn(quiet_answers):
    hs, _ = station(level=10)
    six(hs)
    hs.turn = VA._TURN_OURS
    for n in range(VA._TURN_YIELD_AFTER):
        read_window(hs, 65 + n, 10)
    assert hs.turn == VA._TURN_PEER
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)
    hs._reanswer_over()
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)
    for letter in b'UVWXY':
        read_window(hs, letter, 10)
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._peer_double_verdict() == 'CERTAIN'


@pytest.mark.parametrize('streamed', [False, True])
def test_u4_repeated_pair_retains_window_occurrence(streamed, quiet_answers):
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new."""
    hs, io = station(level=10)
    pair = [b'A' * cap('2300', 10) + b'\x99',
            b'B' * cap('2300', 10) + b'\x99']
    for occurrence, field in enumerate((0x99, 0x19, 0x59, 0x99), 1):
        bodies = [b[:-1] + bytes([field]) for b in pair]
        reads = [(bodies[:1], True), (bodies[1:], False)] if streamed else [(bodies, False)]
        for frames, hold in reads:
            hs._level_read = phy.LevelRead(10, tuple(frames), 40.0, 1)
            assert hs._answer_data_over([], hold=hold)
        hs._level_read = None
        assert (hs._holds_epoch, hs._holds_any) == (occurrence, occurrence)
        hs._reanswer_over()
        assert (hs._holds_epoch, hs._holds_any) == (occurrence, occurrence)
    assert b''.join(io.host) == (b'A' * cap('2300', 10) + b'B' * cap('2300', 10)) * 4


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('idle', ['early', 'pair'])
def test_u4_idle_completed_single_retransmission_counts_once(bw, streamed, idle,
                                                            quiet_answers):
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new."""
    hs, io = station(bw, level=10)
    for letter in b'ABCD':
        read_window(hs, letter, 10)

    def complete_idle():
        hs._held_idle_early = idle == 'early'
        hs._idle_pair_seen = idle == 'pair'
        hs._release_held_answer(1)

    body = b'E' * cap(bw, 10) + b'\x99'
    hs._level_read = phy.LevelRead(10, (body,), 40.0, 1)
    assert hs._answer_data_over([], hold=True)
    assert (hs._holds_epoch, hs._holds_any) == (4, 4)
    complete_idle()
    assert (hs._holds_epoch, hs._holds_any) == (5, 5)
    for occurrence, field in enumerate((0x99, 0x19), 6):
        hs._level_read = phy.LevelRead(10, (body[:-1] + bytes([field]),), 40.0, 1)
        assert hs._answer_data_over([], hold=streamed)
        if streamed or occurrence == 7:
            complete_idle()
        assert (hs._holds_epoch, hs._holds_any) == (6, 6)
        assert hs._rx_window.A.kind == ('continue' if occurrence == 6 else 'nak')
        assert hs._peer_double_verdict() == ('CERTAIN' if occurrence == 6 else 'POSSIBLE')
        hs._reanswer_over()
        assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    # Level 10 was the NAKed over's and is a retry since 15d6f5d39; 8 nobody asked for.
    hs._level_read = phy.LevelRead(8, (b'F' * cap(bw, 8) + b'\x99',), 40.0, 1)
    assert not hs._answer_data_over([])
    hs._level_read = None
    assert b''.join(io.host) == b''.join(bytes([c]) * cap(bw, 10) for c in b'ABCDEE')
    assert (hs._holds_epoch, hs._holds_any) == (6, 6)
    assert hs._peer_double_verdict() == 'POSSIBLE'


@pytest.mark.parametrize('placement', ['frontier', 'start', 'cursor', 'nested-start',
                                        'nested-cursor', 'nested-both', 'nested-neither'])
def test_after_nak_any_read_window_resets_both_counts(placement):
    """DESIGN §3 decision B: every placement resets counts; false positives are R5."""
    hs, _ = station(level=7)
    six(hs)
    key(hs, nak=True)
    body = b'P'*cap('2300', 7)+b'\x99'
    payload = hs._retry_payload(body)
    prefix = b'prefix' if placement in ('cursor', 'nested-cursor') else b''
    hs._rx_window.R = prefix + payload + b'Q'
    if placement != 'frontier':
        hs._rx_window.T = RW.Replay(0, hs._rx_window.S,
            cursor=len(prefix) if 'cursor' in placement else None,
            alternative=len(payload) if placement == 'nested-both' else None)
    hs._note_peer_window(7, [body], False, False)
    assert (hs._holds_epoch, hs._holds_any) == (0, 0)
    assert hs._peer_double_verdict() == 'NONE'
    assert not hs._holds_nak_sent
    hs._peer_answer_keyed(0)
    assert (hs._holds_epoch, hs._holds_any) == (1, 1)


@pytest.mark.parametrize('bw,level,high', [('2300', 7, True), ('2750', 4, True),
                                         ('500', 13, True), ('500', 3, False)])
@pytest.mark.parametrize('shape', ['expected-pair', 'short-head', 'located-pair'])
def test_incomplete_double_nak_keeps_retry_possible_after_count_reset(bw, level, high, shape):
    hs, io = station(bw, level, high)
    if shape == 'expected-pair':
        if bw == '500' and not high:
            hs._peer_over_state = (3, 0x99)
            hs._peer_paired_500 = True
        else:
            six(hs)
        assert hs._peer_double_verdict() == 'CERTAIN'
    body = b'A'*cap(bw, level)+b'\x99'
    hs._deliver([b'head\x14\xf6\x80' if shape == 'short-head' else body], hold=True)
    if shape == 'located-pair':
        hs._deliver([None], hold=True)
    hs._finish_delivery_window(complete=False)
    if bw == '500' and not high:
        head = b'head' if shape == 'short-head' else b'A'*cap(bw, level)
        assert hs._rx_window.A.kind == 'frame10' and hs._rx_window.A.sent
        assert not hs._rx_window.A.retry_double
        assert hs._rx_window.R == head and io.host == [head]
        assert hs._owed_block and hs._expect_level == level - 1
        assert hs._peer_double_verdict() == 'NONE'
        return
    assert hs._rx_window.A.retry_double
    key(hs, nak=True, sent=False)
    assert not hs._holds_nak_sent
    key(hs, nak=True)
    key(hs, nak=True)
    hs._peer_over_state = None
    hs._note_peer_window(level, [body], False, True)
    assert (hs._holds_epoch, hs._holds_any) == (0, 0)
    assert hs._peer_double_verdict() == 'POSSIBLE'
    assert not hs._holds_nak_sent
    if bw != '500' or high:
        hs._holds_epoch = hs._holds_any = 6
        assert hs._peer_double_verdict() == 'POSSIBLE'
        hs._holds_epoch = hs._holds_any = 0
    hs._rx_window.event('T')
    hs._note_peer_window(level, [body], False, False)
    assert hs._peer_double_verdict() == 'NONE'


@pytest.mark.parametrize('reset', ['compressed', 'close', 'turn', 'link'])
def test_incomplete_double_retry_evidence_clears_at_delivery_reset(reset):
    hs, _ = station(level=7)
    six(hs)
    hs._deliver([b'head\x99'], hold=True)
    hs._finish_delivery_window(complete=False)
    key(hs, nak=True)
    if reset == 'compressed':
        window(hs, 9, 0xb9)
        key(hs)
    elif reset == 'close':
        window(hs, 9, 0x82)
        key(hs)
    elif reset == 'turn':
        hs.turn = VA._TURN_OURS
    else:
        hs.state = VA.VaraState.DISCONNECTED
    assert hs._peer_double_verdict() == 'NONE'
    assert not hs._holds_nak_double and not hs._holds_retry_double


@pytest.mark.parametrize('level,retry,envelope,want', [
    (7, 6, 'double', 'NONE'), (7, 7, 'single', 'POSSIBLE'),
    (1, 1, 'single', 'NONE'), (1, 1, 'double', 'POSSIBLE'),
    (1, 1, 'inconclusive', 'POSSIBLE'),
])
def test_incomplete_double_verdict_uses_level_then_envelope(level, retry, envelope, want):
    hs, _ = station(level=level)
    hs._note_incomplete_double(pair=True, level=level)
    key(hs, nak=True, sent=False)
    assert hs._peer_double_verdict(level=retry, envelope=envelope) == 'NONE'
    key(hs, nak=True)
    assert hs._peer_double_verdict(level=retry, envelope=envelope) == want
    hs._holds_envelope = envelope
    hs._rx_read = retry, None
    hs._note_peer_window(retry, [b'head\x99'], False, True)
    assert hs._peer_double_verdict() == want


@pytest.mark.parametrize('level,single,double', [
    (2, 239616, 473088), (3, 241664, 475136),
    (4, 209408, 411648), (7, 187152, 368400), (10, 176256, 347904),
])
def test_envelope_row_extents_match_step1(level, single, double):
    assert phy.envelope_lengths('2300', level) == (single, double)


@pytest.mark.parametrize('frames,outcome', [(1, 'single'), (2, 'double'), (1.5, 'inconclusive')])
def test_envelope_classifier_continuous_support_and_below_detection(frames, outcome):
    import numpy as np
    single, double = phy.envelope_lengths('2300', 1)
    n = round(single + (frames-1)*(double-single))
    carrier = np.sin(2*np.pi*1500*np.arange(n)/48000)
    assert phy.burst_extent(np.r_[np.zeros(12000), carrier, np.zeros(12000)], '2300', 1) == outcome
    if frames == 1:
        companion = 1e-8*np.sin(2*np.pi*1500*np.arange(double-single)/48000)
        assert phy.burst_extent(np.r_[np.zeros(12000), carrier, companion, np.zeros(12000)], '2300', 1) == 'single'
    assert phy.burst_extent(carrier, '2300', 1) == 'inconclusive'


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('statuses', [('read',), ('read', 'read'), ('read', 'X'),
                                      ('read', 'erased'), ('read', 'invisible'),
                                      ('erased', 'read'), ('invisible', 'read')])
def test_channel_envelope_matches_transmitted_index_audio(level, statuses):
    import numpy as np
    from hfmodem.kestrel.rx import varahf2300 as rx
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    from .test_rxwindow import ModelPeer, ModelChannel
    peer = ModelPeer('reference', 'initiator', 'OFF', b'A'*2000, level=level)
    peer.stays = 6 if len(statuses) == 2 else 0
    attempt = peer.key_data()
    r = rx.RECORDS[level-1]
    parts = []
    for i, status in enumerate(statuses):
        frame = bytearray(tx.build_frame(b'A'*(r.frame_bytes-2), level-1))
        if status == 'X':
            frame[-1] ^= 1
        audio = tx.synth_frame(bytes(frame), level-1, over=0 if i == 0 else None)
        if i:
            audio = audio[-r.ncols*r.dw50:]
        if status == 'erased':
            audio *= 0
        elif status == 'invisible':
            audio *= 1e-8
        parts.append(audio)
    signal = np.concatenate([np.zeros(12000), *parts, np.zeros(12000)])
    assert phy.burst_extent(signal, '2300', level) == ModelChannel.envelope(attempt, statuses)
