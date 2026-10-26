# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import collections
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.vara import vara_arq as arq, vara_frames as vf, vara_mfsk as mk
from hfmodem.tests.kestrel.test_bw500_recovery import _station

FIXTURES = Path(__file__).with_name('fixtures') / 'stock-pairing'


def _receiver(role, level, run=0, paired=False):
    hs, io = _station('W1AW')
    hs.role = role
    hs._peer_over_state = (level, 0x99)
    hs._peer_run_500, hs._peer_paired_500 = run, paired
    hs._next_500 = level
    hs._peer_over = max(1, run)
    hs._peer_delivery_open = True
    hs._answer_owed = arq._OWED_OVER
    return hs, io


def _feed(hs, io, signal, during=None):
    for j in range(0, len(signal), locator.BLOCK):
        block = signal[j:j + locator.BLOCK]
        io.fed = j + len(block)
        hs.on_rx_stream(block)
        if during is not None:
            during(hs, io)


def _build(level, marker=0x95, value=31):
    payload = bytes([value]) * rx._PAYLOAD[level]
    return tx.build_frame(payload, marker) if level == 4 else tx.build_l3_frame(payload, marker, level)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('outcome', ['ack_lost', 'unread'])
def test_stock_over_table_has_zero_prediction_mismatches(role, outcome):
    rows = [json.loads(line) for line in (FIXTURES / 's2s-overs.jsonl').read_text().splitlines()]
    files = collections.defaultdict(list)
    for row in rows:
        if not row['setup'] and row['level'] in rx._PAYLOAD:
            files[row['file']].append(row)
    counts = collections.Counter()
    for rows in files.values():
        for who in 'AB':
            hs, _ = _receiver(role, 4)
            last_t = last_end = retry = None
            mine = [r for r in rows if r['who'] == who]
            for i, row in enumerate(mine):
                level, buffer = row['level'], row['buffer']
                other = last_t is not None and (row['t0'] - last_end > 2.5 or any(
                    r['who'] != who and last_t < r['t0'] < row['t0'] for r in rows))
                if other:
                    hs._peer_over_state = None
                    hs._peer_run_500, hs._peer_paired_500 = 0, False
                    hs._last_window = ()
                    retry = None
                expect = hs._peer_pairs_500(level)
                marker = 0x82 if buffer is not None and buffer <= rx._PAYLOAD[level] else 0x99
                partner = expect and not (marker & 2)
                assert partner == (row['frames'] == 2), (row, outcome, hs._peer_run_500)
                counts['pair' if partner else 'close' if expect else 'single'] += 1
                bodies = ([b[:-1] + bytes([marker]) for b in retry] if retry else
                          [i.to_bytes(2, 'big') + bytes([k, marker if k == 0 else 0x95])
                           for k in range(row['frames'])])
                nxt = mine[i + 1]['buffer'] if i + 1 < len(mine) else None
                delivered = not (buffer is not None and buffer > 0 and nxt == buffer)
                if not delivered and outcome == 'unread':
                    hs._owe_unread_over()
                else:
                    hs._note_peer_500(level, bodies)
                    hs._last_window = tuple(bodies)
                    if marker & 2:
                        hs._tx_over_response(True)
                retry = None if delivered else bodies
                last_t, last_end = row['t0'], row['t0'] + row['dur']
    assert counts == {'pair': 60, 'close': 10, 'single': 417}


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_six_delivered_singles_then_pairs_until_level_change_duplicate_or_close(role, level):
    hs, _ = _receiver(role, level)
    for i in range(6):
        assert not hs._peer_pairs_500(level)
        hs._note_peer_500(level, [bytes([i, 0x99])])
    assert hs._peer_pairs_500(level)
    assert not hs._peer_pairs_500(level % 4 + 1)
    hs._note_peer_500(level, [b'a\x95', b'b\x81'])
    hs._overs_hint = 0
    assert hs._peer_pairs_500(level)
    hs._last_window = (b'a\x95', b'b\x81')
    hs._note_peer_500(level, list(hs._last_window))
    assert not hs._peer_pairs_500(level)
    assert hs._peer_run_500 == 1
    hs._note_peer_500(level, [b'c\x82'])
    hs._tx_over_response(True)
    assert not hs._peer_pairs_500(level)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('marker,run', [(0x95, 0), (0x95, 5), (0x82, 6), (0x83, 6)])
def test_single_is_answered_at_stock_gap(role, level, marker, run):
    hs, io = _receiver(role, level, run)
    signal = tx.synth_burst(_build(level, marker), onset=24000, level=level,
                            ncol=rx.NSYM + 9)
    end = len(signal)
    _feed(hs, io, np.pad(signal, (0, rx.FS)))
    assert len(io.keyed_at) == 1, io.msgs
    gap = (io.keyed_at[0] - end) / rx.FS
    assert locator.ANSWER_GAP[level] - .025 <= gap <= locator.ANSWER_GAP[level] + .06


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_erased_second_frame_gets_frame_ten_only_after_the_pair(role, level):
    hs, io = _receiver(role, level, 6)
    frames = [_build(level, value=v) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=24000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    end = len(signal)
    split = 24000 + ((rec.lead + rec.ncols) * rec.dw if rec else (9 + rx.NSYM) * rx.H)
    signal[split:] = 0
    checked = []

    def during(hs, io):
        if not checked and io.fed >= split + rx.FS // 2:
            checked.append(True)
            assert not hs._a_gap()
            hs._lv_prompt = hs._keyed_on_peer_burst = False
            hs.idle_keepalive()
            assert not io.keyed_at

    _feed(hs, io, np.pad(signal, (0, rx.FS)), during)
    assert checked
    assert len(io.keyed_at) == 1, io.msgs
    assert end <= io.keyed_at[0] <= end + rx.FS // 3
    assert io.host == [bytes([31]) * rx._PAYLOAD[level]]
    want = vf.over_continue_frames(hs.called, '500', (10,))[10][role == 'responder']
    heard = mk.demod_tone_pairs(io.sent[0], len(want), mk.band_for('500'))
    assert tuple(tuple(sorted(p)) for p in heard) == want
    assert hs._next_500 == max(1, level - 1)
    assert not hs._peer_pairs_500(level)
    assert not hs._unread_over_500()
    hs._took_stall_answer()
    assert hs._owed_block and hs._pair_loss_500
    assert hs._reack_over()
    np.testing.assert_array_equal(io.sent[-1], io.sent[0])


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('t', [209.797, 222.219])
def test_cables_short_remainder_80_waits_for_82_close(role, t):
    rows = [json.loads(line) for line in (FIXTURES / 'o011601.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['t'] == t)
    frames = [bytes.fromhex(body) for body in row['body']]
    assert not phy.over_is_last(frames[0][:-2], 'W9SSJ')
    assert phy.over_is_last(frames[1][:-2], 'W9SSJ')
    hs, io = _receiver(role, 4, paired=True)
    hs._overs_hint = 0
    signal = tx.synth_burst(frames, onset=24000)
    end = len(signal)
    _feed(hs, io, np.pad(signal, (0, rx.FS)))
    assert len(io.keyed_at) == 1, io.msgs
    assert end <= io.keyed_at[0] <= end + rx.FS // 5
    assert b''.join(io.host) == phy.vara_payload(frames[0][:-2], caller='W9SSJ', body_len=44)
    assert not hs._peer_delivery_open


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('damage', ['noise', 'head_fade', 'erased_tail'])
def test_predicted_partner_never_becomes_a_complete_single(level, damage):
    frames = [_build(level, value=v) for v in (31, 73)]
    signal = tx.synth_burst(frames, onset=24000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    split = 24000 + ((rec.lead + rec.ncols) * rec.dw if rec else (9 + rx.NSYM) * rx.H)
    power = np.mean(signal[24000:72000] ** 2)
    if damage == 'head_fade':
        signal[split:split + rx.FS] *= .001
    if damage == 'erased_tail':
        signal[split - 16 * (rec.dw if rec else rx.H):] = 0
    signal = np.pad(signal, (0, rx.FS))
    noise = np.random.default_rng(7).normal(size=len(signal))
    signal += noise * np.sqrt(power / (np.mean(noise ** 2) * 2500 / 24000) / 10 ** (3 / 10))
    receiver = locator.Receiver()
    found = receiver.feed(signal, may_pair=True, eof=True)
    good = [o for o in found if o.frames and o.frames[0].crc_ok]
    assert good
    assert all(not o.result.complete or [f.frame_bytes for f in o.frames] == frames for o in good)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_refused_frame_ten_keeps_debt_without_a_positive_fallback(role):
    hs, io = _receiver(role, 3, 6)
    frame_bytes = _build(3)
    frame = rx.FrameResult(0, (0,), 0, 1., True, frame_bytes, level=3, frames_in_burst=2)
    io.tx_went_out = lambda: False
    hs._answer_pair_loss_500(frame, frame_bytes[:-2])
    assert len(io.sent) == 1
    assert hs._owed_block and hs._pair_loss_500
    assert not hs._reack_over()
    assert len(io.sent) == 2
    np.testing.assert_array_equal(io.sent[0], io.sent[1])
    assert hs._reacks == 0


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_duplicate_read_resets_pair_run_without_redelivery(role):
    hs, io = _receiver(role, 3, 5)
    frame_bytes = _build(3)
    frame = rx.FrameResult(0, (0,), 0, 1., True, frame_bytes, level=3)
    hs._last_window = (frame_bytes[:-2],)
    hs._resend_decoded = rx.DecodeResult([frame], [frame])
    assert hs._answer_data_over(np.zeros(100))
    assert hs._peer_run_500 == 1
    assert not hs._peer_pairs_500(3)
    assert not io.host


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('outcome', ['ack_lost', 'unread'])
@pytest.mark.parametrize('markers', [(0x99, 0x95), (0x95, 0x99)])
def test_fetch_013206_retry_starts_six_singles(role, outcome, markers):
    hs, io = _receiver(role, 1)
    sequence = [('23.06', 1), ('29.56', 2), ('40.58', 2), ('46.95', 3),
                ('53.35', 4), ('59.73', 5), ('66.17', 6), ('72.52', 7)]
    for t, value in sequence:
        assert not hs._peer_pairs_500(1), t
        if t == '29.56' and outcome == 'unread':
            hs._owe_unread_over()
            assert (hs._peer_run_500, hs._peer_paired_500) == (0, False)
            continue
        # The transcript pins timing, BUFFER and frame counts, not decoded
        # control fields. An ACK-lost replay keeps its field; advancing the
        # field at unchanged geometry denotes new data even with equal bytes
        # (test_receive_turn_duplicates). An unread over has no delivered
        # predecessor and may be retried with either field.
        marker = markers[1] if t == '40.58' and outcome == 'unread' else markers[0]
        frame = rx.FrameResult(0, (0,), 0, 1., True, _build(1, marker, value), level=1)
        hs._resend_decoded = rx.DecodeResult([frame], [frame])
        assert hs._answer_data_over(np.zeros(100))
        if t == '40.58':
            assert (hs._peer_run_500, hs._peer_paired_500) == (1, False)
    assert hs._peer_pairs_500(1), '78.95'
    assert io.host == [bytes([v]) * rx._PAYLOAD[1] for v in range(1, 8)]


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('path', ['_undecoded_over', '_owe_unread_over', '_irs_nak', '_nak_owed_over'])
def test_unread_pair_resets_run_before_a_delivered_retry(role, path):
    hs, _ = _receiver(role, 3, 6, paired=True)
    if path == '_irs_nak':
        hs._irs_nak(phy.LevelRead(3, (), located=2))
    else:
        assert getattr(hs, path)()
    assert (hs._peer_run_500, hs._peer_paired_500) == (0, False)
    hs._note_peer_500(3, [b'retry\x99'])
    assert (hs._peer_run_500, hs._peer_paired_500) == (1, False)


@pytest.mark.parametrize('level', [5, 6, 7, 8])
def test_high_speed_read_does_not_leave_a_base_pair_prediction(level):
    hs, _ = _receiver('responder', 4, 6, paired=True)
    hs._level_read = phy.LevelRead(level, (b'new\x99',), located=1)
    assert hs._peer_data_over(np.zeros(100)) == [b'new\x99']
    assert hs._rx_read[0] == level
    assert hs._peer_over_state is None
    assert not hs._peer_pairs_500(4)
    hs._note_peer_500(4, [b'base\x99'])
    assert hs._peer_run_500 == 1


@pytest.mark.parametrize('marker', range(0x40, 0xa0))
def test_close_uses_field_kind_even_with_a_short_payload(marker):
    body = phy.vara_body(b'end', 'W9SSJ', body_len=44)
    assert phy.over_is_last(body[:-1] + bytes([marker]), 'W9SSJ') == bool(marker & 2)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [3, 4])
@pytest.mark.parametrize('snr', [-14, -18])
def test_unread_first_frame_never_keys_into_expected_partner(role, level, snr):
    hs, io = _receiver(role, level, 6)
    signal = tx.synth_burst([_build(level, value=v) for v in (31, 73)], onset=24000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    split = 24000 + ((rec.lead + rec.ncols) * rec.dw if rec else (9 + rx.NSYM) * rx.H)
    end = len(signal)
    noise = np.random.default_rng(3).normal(size=split - 24000)
    power = np.mean(signal[24000:split] ** 2)
    signal[24000:split] += noise * np.sqrt(power / (np.mean(noise ** 2) * 2500 / 24000)
                                          / 10 ** (snr / 10))
    _feed(hs, io, np.pad(signal, (0, 2 * rx.FS)))
    assert all(k >= end for k in io.keyed_at)
    assert not io.host


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_all_overs_has_zero_sender_rule_prediction_mismatches(role):
    rows = [json.loads(line) for line in (FIXTURES / 'all-overs.jsonl').read_text().splitlines()]
    files = collections.defaultdict(list)
    for r in rows:
        if not r['setup'] and r['level'] in (1, 2, 3, 4) and not 4.1 < r['dur'] < 4.25:
            files[r['file'], r['who']].append(r)
    counts = collections.Counter()
    for rows in files.values():
        hs, _ = _receiver(role, 4)
        run, paired, previous, retry = 0, False, None, None
        for i, r in enumerate(rows):
            lv, b = r['level'], r['buffer']
            below = {4: 34, 3: 22, 2: 9, 1: 0}[lv]
            if previous != lv:
                run, paired = 0, False
            state = run >= 6 or paired
            expected = state and b is not None and (b >= rx._PAYLOAD[lv] or paired and b < below)
            short_pair = paired and b is not None and b < below
            marker = 0x80 if short_pair else 0x82 if state and b is not None and b < rx._PAYLOAD[lv] else 0x99
            partner = hs._peer_pairs_500(lv) and not marker & 2
            assert partner == expected == (r['frames'] == 2), (r, run, paired)
            counts['pair' if partner else 'single'] += 1
            nxt = rows[i + 1]['buffer'] if i + 1 < len(rows) else None
            delivered = not (b is not None and nxt == b and b > 0)
            rekey = i + 1 < len(rows) and rows[i + 1]['t0'] - (r['t0'] + r['dur']) < 5
            closes = nxt == 0 and not rekey
            bodies = retry[:r['frames']] if retry else [str((i, k)).encode() + bytes([marker if k == 0 else 0x99])
                               for k in range(r['frames'])]
            if closes or nxt == 0 or short_pair:
                bodies[-1] = bodies[-1][:-1] + b'\x82'
            if not delivered:
                hs._owe_unread_over()
            else:
                hs._note_peer_500(lv, bodies)
                hs._last_window = tuple(bodies)
            if closes:
                hs._tx_over_response(True)
            retry = None if delivered else bodies
            if closes or not delivered:
                run, paired = 0, False
            elif r['frames'] == 2:
                run, paired = 0, True
            else:
                run, paired = run + 1, False
            previous = lv
    assert counts == {'pair': 256, 'single': 1145}


@pytest.mark.parametrize('sent', [False, True])
def test_nak_at_end_of_ack_ladder_resets_pair_run_only_when_sent(sent):
    hs, io = _receiver('responder', 3, 6, paired=True)
    hs._link_high_speed = hs._owed_block = False
    hs._reack_frame = arq.OVER_CONTINUE_ANSWERS[0]
    hs._reacks = len(arq.OVER_CONTINUE_ANSWERS)
    io.tx_went_out = lambda: sent
    hs._reack_over()
    assert (hs._peer_run_500, hs._peer_paired_500) == ((0, False) if sent else (6, True))


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_unread_over_ask_resets_pair_run(role):
    hs, _ = _receiver(role, 3, 6, paired=True)
    hs._owed_block = True
    assert hs._reack_over()
    assert (hs._peer_run_500, hs._peer_paired_500) == (0, False)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_fetch_085621_pairs_survive_polls_and_late_answers(role):
    rows = [json.loads(line) for line in (FIXTURES / 'all-overs.jsonl').read_text().splitlines()]
    rows = [r for r in rows if '085621Z' in r['file'] and not r['setup']]
    hs, io = _receiver(role, 1)
    paired_at = []
    for i, r in enumerate(rows):
        if r['level'] != 1 or r['t0'] > 130:
            continue
        assert hs._peer_pairs_500(1) == (r['frames'] == 2), r
        frames = [rx.FrameResult(0, (0,), 0, 1., True, _build(1, value=i * 2 + k), level=1)
                  for k in range(r['frames'])]
        hs._resend_decoded = rx.DecodeResult(frames, frames)
        assert hs._answer_data_over(np.zeros(100))
        before = hs._peer_run_500, hs._peer_paired_500
        assert hs._tx_control_burst()
        assert (hs._peer_run_500, hs._peer_paired_500) == before
        for _ in range(2):
            assert hs._reack_over()
            assert (hs._peer_run_500, hs._peer_paired_500) == before
        if r['frames'] == 2:
            paired_at.append(r['t0'])
    assert paired_at == [73.07, 91.52, 110.55, 129.56]
    assert io.host


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('retry_answer', [False, True])
def test_011601_rekeyed_close_at_222_43_keeps_pair_until_final_answer(role, retry_answer):
    rows = [json.loads(line) for line in (FIXTURES / 'o011601.jsonl').read_text().splitlines()]
    hs, io = _receiver(role, 4, paired=True)
    for t in (209.797, 222.219):
        assert hs._peer_pairs_500(4)
        row = next(r for r in rows if r['t'] == t)
        frames = [bytes.fromhex(body) for body in row['body']]
        io.tx_went_out = lambda: False
        signal = tx.synth_burst(frames, onset=24000)
        keys = len(io.keyed_at)
        _feed(hs, io, np.pad(signal, (0, rx.FS)))
        assert len(io.keyed_at) == keys + 1
        assert io.keyed_at[-1] >= len(signal)
        assert hs._peer_pairs_500(4) and hs._peer_delivery_open
        hs.idle_keepalive()
        assert not hs._reack_release()
        assert hs._peer_pairs_500(4)
    first = next(r for r in rows if r['t'] == 209.797)
    assert b''.join(io.host) == phy.vara_payload(bytes.fromhex(first['body'][0])[:-2],
                                               caller=hs.caller, body_len=44)
    delivered = list(io.host)
    io.tx_went_out = lambda: True
    if retry_answer:
        assert hs._reack_release()
    else:
        hs._tx_over_response(True)
    assert not hs._peer_pairs_500(4)
    assert not hs._peer_delivery_open
    assert io.host == delivered


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [2, 3, 4])
@pytest.mark.parametrize('high_speed', [False, True])
def test_last_full_81_asks_frame_five_for_close_alone(role, level, high_speed):
    hs, io = _receiver(role, level, paired=True)
    hs._link_high_speed = high_speed
    frame = rx.FrameResult(0, (0,), 0, 1., True, _build(level, 0x81), level=level)
    hs._resend_decoded = rx.DecodeResult([frame], [frame])
    assert hs._answer_data_over(np.zeros(100))
    want = vf.over_continue_frames(hs.called, '500', (5,))[5][role == 'responder']
    heard = mk.demod_tone_pairs(io.sent[-1], len(want), mk.band_for('500'))
    assert tuple(tuple(sorted(p)) for p in heard) == want
    assert hs._next_500 == level - 1
    assert not hs._peer_pairs_500(level - 1)
    close = rx.FrameResult(0, (0,), 0, 1., True, _build(level - 1, 0x82), level=level - 1)
    hs._resend_decoded = rx.DecodeResult([close], [close])
    assert hs._answer_data_over(np.zeros(100))
    assert not hs._peer_delivery_open


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('marker', [0x89, 0x49])
def test_last_full_89_or_49_holds_level_and_close_has_no_partner(role, marker):
    hs, _ = _receiver(role, 4, paired=True)
    hs._note_peer_500(4, [b'full' + bytes([marker])])
    assert hs._answer_step() == 0
    frame = rx.FrameResult(0, (0,), 0, 1., True, _build(4, 0x42), level=4)
    assert phy.over_is_last(frame.frame_bytes[:-2], hs.caller)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_exact_payload_remainder_is_full_plus_empty_close(role, level):
    hs, io = _receiver(role, level, 6)
    body = phy.vara_body(b'', hs.caller, body_len=rx._PAYLOAD[level] + 1)
    close = tx.build_frame(body[:-1], 0x82) if level == 4 else tx.build_l3_frame(body[:-1], 0x82, level)
    frames = [_build(level, 0x49), close]
    signal = tx.synth_burst(frames, onset=24000, level=level)
    end = len(signal)
    _feed(hs, io, np.pad(signal, (0, rx.FS)))
    assert len(io.keyed_at) == 1
    assert io.keyed_at[0] >= end
    assert io.host == [bytes([31]) * rx._PAYLOAD[level]]
    assert not hs._peer_pairs_500(level)
