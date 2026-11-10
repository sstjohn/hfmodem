# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Generated continuation windows at the peer index, tone and PSK rows."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.rx import varaofdm as ofdm, varatone as tone
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.kestrel.vara import vara_arq as A, vara_mfsk as M
from .receive_window_helpers import (
    station, feed, query, assert_missing_nak, arm_replay, full_body, index_window,
    psk_window, tone_window, rx_evidence,  # noqa: F401
)
from .test_idle_poll import clock, _Clock, _feed  # noqa: F401


def _payload(size, identity):
    return bytes((17 * i + identity) % 251 for i in range(size))


def _body(h, level, payload, field):
    size = phy.body_size(h.bw, phy.level_key(h.bw, level))
    if field & 3 == 0 and len(payload) < size - 1:
        body = bytearray(size)
        body[:len(payload)] = payload
        trailer = bytes([0x14, crc16_genibus(h.caller.encode()) >> 8])
        room = min(size - 1 - len(payload), len(trailer))
        body[len(payload):len(payload) + room] = trailer[:room]
    else:
        body = bytearray(phy.vara_body(payload, h.caller, body_len=size, tail=field,
                                      level=6 if level > 4 else 4))
    body[-1] = field
    assert phy.vara_payload(body, caller=h.caller, body_len=size) == payload
    return bytes(body)


def _wave(bw, level, bodies):
    if level <= 4:
        return index_window(bw, level, bodies)
    if (int(bw), level) in tone.ROWS:
        return tone_window(bw, level, bodies)
    return psk_window(bw, level, bodies)


def _listening(monkeypatch, bw, level, role, record_property):
    h, io = station(bw, role)
    h._link_high_speed = True
    h._expect_level, h._rx_read = level, (level, 30.)
    h._lv_owed = h._lv_prompt = True
    h._key(False)
    real = h._irs_step
    forced, ledger = [0], []

    def step(closing=False):
        real(closing=closing)
        frame = forced[0]
        h._expect_level = min(max(h._rx_read[0] + A._LEVEL_STEP[frame], 1), phy.TOP[bw])
        h._next_500 = min(h._expect_level, 4)
        return frame

    monkeypatch.setattr(h, '_irs_step', step)
    record_property('layout', 'DOUBLE-LAYOUT: one shared lead, independent bodies, zero gap')
    return h, io, forced, ledger


def _emission(h, io, ledger, level, payloads, fields, *, final=False):
    bodies = tuple(_body(h, level, p, field) for p, field in zip(payloads, fields))
    audio = _wave(h.bw, level, bodies)
    read = phy.decode_level(audio, h.bw, level)
    assert read is not None and read.frames == bodies and read.located == len(bodies)
    before, host_before, began = len(io.sent), b''.join(io.host), _Clock.t
    verdict = h._peer_double_verdict(level=level)
    if len(bodies) == 2:
        cut = phy.burst_samples(h.bw, level)
        _feed(h, audio[:cut])
        assert len(io.sent) == before and b''.join(io.host) == host_before, io.msgs
        feed(h, audio[cut:])
    else:
        feed(h, audio)
    assert b''.join(io.host) == host_before + b''.join(payloads), io.msgs
    assert len(io.sent) == before + 1, io.msgs
    assert io.at[-1] >= began + len(audio) / M.FS - .05
    answer = h._rx_window.A
    assert answer.kind == ('final' if final else 'continue') and answer.sent
    if not final:
        assert A._LEVEL_STEP[answer.step] == h._expect_level - level
    ledger.append({'level': level, 'payloads': [p.hex() for p in payloads],
                   'fields': list(fields), 'read': len(bodies), 'verdict': verdict,
                   'answer': answer.kind, 'step': answer.step,
                   'expected_level': h._expect_level,
                   'next_verdict': h._peer_double_verdict()})


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_r11_six_l7_holds_then_full_tone_pairs_deliver_6000_bytes(
        monkeypatch, record_property, role):
    h, io, forced, ledger = _listening(monkeypatch, '2300', 7, role, record_property)
    assert phy.body_size('2300', phy.level_key('2300', 7)) - 1 == 256
    expected = []
    for i in range(6):
        p = _payload(256, i)
        expected.append(p)
        _emission(h, io, ledger, 7, (p,), (0x99,))
        assert h._holds_epoch == i + 1
    assert h._peer_double_verdict() == 'CERTAIN'
    for i in range(8):
        pair = (_payload(256, 20 + 2 * i), _payload(256, 21 + 2 * i))
        expected.extend(pair)
        _emission(h, io, ledger, 7, pair, (0x99, 0x99))
    pair = (_payload(256, 80), _payload(112, 81))
    expected.extend(pair)
    _emission(h, io, ledger, 7, pair, (0x81, 0x82), final=True)
    assert b''.join(io.host) == b''.join(expected) and len(b''.join(io.host)) == 6000
    assert not h._peer_delivery_open and h._held_answer is None
    record_property('emissions', ledger)


@pytest.mark.parametrize('bw,level', [('2300', 10), ('2750', 11)])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_r1_psk_pairs_and_short_head_empty_close_deliver_once(
        monkeypatch, record_property, bw, level, role):
    approach_level = 6 if bw == '2300' else 7
    h, io, forced, ledger = _listening(monkeypatch, bw, approach_level, role, record_property)
    approach = _payload(phy.body_size(bw, phy.level_key(bw, approach_level)) - 1, 90)
    forced[0] = 8
    _emission(h, io, ledger, approach_level, (approach,), (0x99,))
    assert h._expect_level == level
    forced[0] = 0
    cap = phy.body_size(bw, phy.level_key(bw, level)) - 1
    expected = [approach]
    for i in range(6):
        p = _payload(cap, 100 + i)
        expected.append(p)
        _emission(h, io, ledger, level, (p,), (0x99,))
    assert h._peer_double_verdict() == 'CERTAIN'
    for i, fields in enumerate(((0x91, 0x99), (0x8d, 0x81))):
        pair = (_payload(cap, 120 + 2 * i), _payload(cap, 121 + 2 * i))
        expected.extend(pair)
        _emission(h, io, ledger, level, pair, fields)
    close = _payload(67, 140)
    expected.append(close)
    _emission(h, io, ledger, level, (close, b''), (0x80, 0x82), final=True)
    assert b''.join(io.host) == b''.join(expected)
    if bw == '2300':
        assert len(b''.join(io.host)) == 9200
    assert not h._peer_delivery_open and h._held_answer is None
    record_property('emissions', ledger)


@pytest.mark.parametrize('bw,level', [('2300', 10), ('2750', 11)])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_r4_certain_psk_head_with_erased_companion_late_query_naks(
        monkeypatch, record_property, bw, level, role):
    h, io, forced, ledger = _listening(monkeypatch, bw, level, role, record_property)
    assert phy.burst_samples(bw, level, 2) - phy.burst_samples(bw, level) == 149 * ofdm.SYM
    cap = phy.body_size(bw, phy.level_key(bw, level)) - 1
    for i in range(6):
        _emission(h, io, ledger, level, (_payload(cap, i),), (0x99,))
    assert h._peer_double_verdict() == 'CERTAIN'
    committed, before = b''.join(io.host), len(io.sent)
    head = _wave(bw, level, (_body(h, level, _payload(cap, 40), 0x99),))
    feed(h, np.r_[head, np.zeros(149 * ofdm.SYM)])
    assert b''.join(io.host) == committed
    assert len(io.sent) == before or h._rx_window.A.kind == 'nak'
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert b''.join(io.host) == committed and h._rx_window.A.kind == 'nak'
    ledger.append({'level': level, 'payloads': [_payload(cap, 40).hex(), 'erased'],
                   'read': 1, 'answer': 'nak', 'host_hex': committed.hex()})
    record_property('emissions', ledger)


@pytest.mark.parametrize('level,total,closing', [(2, 800, (0x80, 0x82)),
                                               (3, 1200, (0x81, 0x82))])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_c1_c2_index_holds_pairs_and_split_close_deliver_exact_bytes(
        monkeypatch, record_property, level, total, closing, role):
    h, io, forced, ledger = _listening(monkeypatch, '2300', 6, role, record_property)
    cap = phy.body_size('2300', phy.level_key('2300', level)) - 1
    expected = [_payload(173, 200)]
    forced[0] = 5
    _emission(h, io, ledger, 6, (expected[0],), (0x99,))
    if level == 2:
        committed = b''.join(io.host)
        unread = _wave('2300', 5, (_body(h, 5, _payload(132, 201), 0x51),))
        # C1's L5 was unread: its query NAK asks L3. A reversed waveform
        # instead triggers a direct CRC NAK to L4 before the query arrives.
        before = len(io.sent)
        feed(h, np.zeros_like(unread))
        assert len(io.sent) == before and b''.join(io.host) == committed
        query(h, delay=12.)
        assert_missing_nak(h, io, before)
        assert h._rx_window.A.kind == 'nak' and b''.join(io.host) == committed
        ledger.append({'level': 5, 'payloads': [_payload(132, 201).hex()],
                       'read': 0, 'answer': 'nak', 'expected_level': h._expect_level})
        assert h._expect_level == 3
        p = _payload(47, 210)
        expected.append(p)
        _emission(h, io, ledger, 3, (p,), (0x59,))
    else:
        for approach_level, size in ((5, 132), (4, 89)):
            p = _payload(size, 210 + approach_level)
            expected.append(p)
            _emission(h, io, ledger, approach_level, (p,), (0x99,))
    forced[0] = 0
    for i in range(6):
        p = _payload(cap, 10 + i)
        expected.append(p)
        _emission(h, io, ledger, level, (p,), (0x99,))
    assert h._peer_double_verdict() == 'CERTAIN'
    for i in range(10 if level == 2 else 5):
        pair = (_payload(cap, 30 + 2 * i), _payload(cap, 31 + 2 * i))
        expected.extend(pair)
        _emission(h, io, ledger, level, pair, (0x99, 0x99))
    pair = ((_payload(8, 60), b'') if level == 2 else
            (_payload(cap, 60), _payload(7, 61)))
    expected.extend(pair)
    _emission(h, io, ledger, level, pair, closing, final=True)
    assert b''.join(io.host) == b''.join(expected) and len(b''.join(io.host)) == total
    record_property('emissions', ledger)


@pytest.mark.parametrize('bw,level', [('2300', 1), ('2300', 2), ('2300', 3),
                                    ('2300', 4), ('2300', 7), ('2300', 10),
                                    ('2750', 11), ('500', 7)])
def test_shared_lead_double_has_two_locations_and_companion_end(bw, level):
    size = phy.body_size(bw, phy.level_key(bw, level))
    bodies = tuple(full_body(_payload(size - 1, i), 'W9SSJ', size, 0x99) for i in (1, 2))
    audio = _wave(bw, level, bodies)
    read = phy.decode_level(audio, bw, level)
    assert read.frames == bodies and read.located == 2 and not read.lost
    assert phy.burst_samples(bw, level) < read.end <= len(audio) + ofdm.SYM


@pytest.mark.parametrize('level', [2, 3, 7])
def test_cropped_double_with_tail_anchor_returns_both_frames(monkeypatch, level):
    bw = '2300'
    size = phy.body_size(bw, phy.level_key(bw, level))
    bodies = tuple(full_body(_payload(size - 1, i), 'W9SSJ', size, 0x99) for i in (1, 2))
    audio = _wave(bw, level, bodies)
    anchors = []
    if level == 7:
        rec = tone.ROWS[int(bw), level]
        span = (rec.nsym - tone.C0) * tone.H
        crop = 4 * tone.H
        start = len(rec.taps) // 2 - crop
        assert -tone.C0 * tone.H <= start < 0
        audio = audio[crop:]
        real = tone._read

        def tail_read(x, record, df=None, *, n0=None):
            if n0 is None:
                n0 = start + span
                anchors.append(n0)
            return real(x, record, 0.0 if df is None else df, n0=n0)

        tail = real(tone.Audio(audio), rec, 0.0, n0=start + span)
        assert tuple(f[:-2] for f in tone._decode(tail).frames) == bodies[1:]
        monkeypatch.setattr(tone, '_read', tail_read)
    else:
        rx = A._RX2300
        key = phy.level_key(bw, level)
        rec = rx.RECORDS[key]
        span = rec.ncols * rec.dw50
        crop = (rec.lead + 1) * rec.dw50
        start = -rec.dw50
        assert -rec.lead * rec.dw50 <= start < 0
        audio = audio[crop:]
        real = rx.decode_over

        def tail_read(x, s, e, *args, **kwargs):
            if kwargs.get('level', rx.BASE_LEVEL) != key:
                return real(x, s, e, *args, **kwargs)
            assert s == 0 and e == len(audio)
            frame = real(x, start + span, e, *args, **kwargs)
            assert frame.crc_ok and bytes(frame.payload) == bodies[1]
            anchors.append(frame.end - span)
            return frame

        monkeypatch.setattr(rx, 'decode_over', tail_read)
    read = phy.decode_level(audio, bw, level)
    assert anchors == [start + span]
    assert read is not None and read.frames == bodies and read.located == 2 and not read.lost
    assert span < read.end <= len(audio)


def test_h2k_possible_cached_head_partial_companion_fall_naks(monkeypatch):
    h, io, head, expired = _h2f_held_head(monkeypatch, prompt=True)
    h._holds_any = A._SINGLES_BEFORE_PAIR
    assert h._peer_double_verdict() == 'POSSIBLE'
    partial = round(.4 * M.FS)
    assert A._EDGE_BLOCKS * A._EXPECT_STEP < partial < (
        phy.burst_samples(h.bw, h._expect_level, 2) - head)
    _h2f_poll(h, np.full(head - 960, 4.))
    h._lv_found = True
    _h2f_poll(h, np.full(960, 4.))
    assert h._lv_pending is not None and h._rx_window.W is not None
    _h2f_poll(h, np.full(partial, 4.))
    assert not io.host and not io.sent
    deadline = h._lv_pending_deadline
    assert deadline > h._lv_heard + M.FS // 2
    _h2f_poll(h, np.zeros(M.FS // 2))
    assert not io.host, io.msgs
    assert len(io.sent) == 1 and h._rx_window.A.kind == 'nak', io.msgs
    assert h._rx_window.W is None and len(expired) == 1
    assert expired[0] < deadline
    assert h._lv_pending is None and h._lv_pending_deadline is None
    assert h._lv_pending_activity is None


@pytest.mark.parametrize('level', [7, 10])
def test_h2k_possible_quiet_single_answers_promptly(monkeypatch, record_property, level):
    h, io = station('2300', 'responder')
    h._link_high_speed = True
    h._expect_level, h._rx_read = level, (level, 30.)
    h._holds_any = A._SINGLES_BEFORE_PAIR
    h._lv_owed = h._lv_prompt = True
    h._key(False)
    assert h._peer_double_verdict() == 'POSSIBLE'
    size = phy.body_size(h.bw, phy.level_key(h.bw, level))
    payload = _payload(size - 1, 72)
    body = _body(h, level, payload, 0x99)
    audio = phy.render_level((body,), h.bw, level, VB6Rnd(4107106))
    read = phy.decode_level(audio, h.bw, level)
    assert read.frames == (body,) and read.located == 1
    _feed(h, np.zeros(M.FS // 2))
    began = _Clock.t
    _feed(h, audio)
    for _ in range(10):
        if io.sent:
            break
        _feed(h, np.zeros(960))
    assert b''.join(io.host) == payload, io.msgs
    assert len(io.sent) == 1 and h._rx_window.A.kind == 'continue', io.msgs
    latency = io.at[0] - began - read.end / M.FS
    record_property('head_end_samples', read.end)
    record_property('answer_latency_s', latency)
    assert 0 <= latency <= {7: .110, 10: .127}[level]
    assert h._rx_window.W is None
    assert h._lv_pending is None and h._lv_pending_deadline is None


def test_h2g_decreasing_continuation_fall_ends_incomplete(monkeypatch):
    h, io, head, expired = _h2f_held_head(monkeypatch, prompt=True)
    double = phy.burst_samples(h.bw, h._expect_level, 2)
    _h2f_poll(h, np.r_[np.full(head, 4.), np.full(double - head, 3.),
                      np.full(M.FS, 2.)])
    deadline = h._lv_pending_deadline
    assert deadline > h._lv_heard + M.FS // 2
    _h2f_poll(h, np.zeros(M.FS // 2))
    assert not io.host, io.msgs
    assert h._rx_window.W is None and h._rx_window.A.kind == 'nak', io.msgs
    assert len(io.sent) == 1 and len(expired) == 1
    assert expired[0] < deadline
    assert h._lv_pending is None and h._lv_pending_deadline is None


def _h2g_single(monkeypatch, *, start, delay=0):
    h, io = station('2300', 'responder')
    level = 7
    h._link_high_speed = True
    h._expect_level, h._rx_read = level, (level, 30.)
    h._lv_owed = h._lv_prompt = True
    h._key(False)
    size = phy.body_size(h.bw, phy.level_key(h.bw, level))
    payload = _payload(size - 1, 71)
    body = _body(h, level, payload, 0x99)
    end = start + phy.burst_samples(h.bw, level)
    reads = []

    def decode(audio, *args, **kwargs):
        if not reads and h._lv_heard >= end + delay:
            reads.append(h._lv_heard)
            return phy.LevelRead(level, (body,), 30., 1,
                                 end=end - h._lv_heard + len(audio))
        return None

    monkeypatch.setattr(phy, 'decode_level', decode)
    return h, io, payload, end, reads


def test_h2g_retained_earlier_emission_cannot_supply_head_floor(monkeypatch):
    head = phy.burst_samples('2300', 7)
    h, io, payload, end, reads = _h2g_single(
        monkeypatch, start=2 * head, delay=M.FS // 2)
    _h2f_poll(h, np.r_[np.full(head, 1.8), np.full(head, .8)])
    assert h._lv_fell is not None and h._lv_rose is None
    _h2f_poll(h, np.r_[np.full(head, 1.5), np.full(M.FS, 1.1)])
    assert len(reads) == 1
    assert reads[0] >= end + M.FS // 2
    assert not io.host, io.msgs
    assert not io.sent or h._rx_window.A.kind == 'nak', io.msgs


def test_h2g_weak_clean_single_answers_at_its_end(monkeypatch):
    lead = M.FS // 2
    h, io, payload, end, reads = _h2g_single(monkeypatch, start=lead)
    deadline = h._expected_window_samples()
    _h2f_poll(h, np.r_[np.ones(lead), np.full(end - lead, 1.5)])
    assert h._lv_fell is None and h._lv_rose * A._EXPECT_STEP <= lead
    assert not io.host and not io.sent
    _h2f_poll(h, np.ones(M.FS // 2))
    assert b''.join(io.host) == payload, io.msgs
    assert len(io.sent) == 1 and h._rx_window.A.kind == 'continue', io.msgs
    assert h._lv_pending is None and h._lv_pending_deadline is None
    assert reads[0] < end + M.FS // 2 < deadline - M.FS


@pytest.mark.parametrize('event', ['turn', 'disconnect'])
def test_h2g_lifecycle_clears_held_head_and_deadline(monkeypatch, event):
    h, io, head, expired = _h2f_held_head(monkeypatch, prompt=True)
    _h2f_poll(h, np.full(head + M.FS, 4.))
    assert h._lv_pending is not None and h._lv_pending_deadline is not None
    if event == 'turn':
        h.turn = A._TURN_OURS
    else:
        h.state = A.VaraState.DISCONNECTED
    assert h._rx_window.W is None
    assert h._lv_pending is None
    assert h._lv_pending_deadline is None
    assert not io.host and not io.sent and not expired


@pytest.mark.parametrize('level', [2, 7, 10])
def test_failed_companion_is_located_and_retains_the_head(monkeypatch, level):
    size = phy.body_size('2300', phy.level_key('2300', level))
    bodies = tuple(full_body(_payload(size - 1, i), 'W9SSJ', size, 0x99) for i in (1, 2))
    audio = _wave('2300', level, bodies)
    if level <= 4:
        rx = A._RX2300
        real = rx.frame_at

        def read(mag, record):
            frame = real(mag, record)
            return replace(frame, crc_ok=False) if bytes(frame.payload) == bodies[1] else frame

        monkeypatch.setattr(rx, 'frame_at', read)
    else:
        rx = tone if level == 7 else ofdm
        real = rx.decode_frame

        def read(*args, **kwargs):
            frame = real(*args, **kwargs)
            return frame[:-1] + bytes([frame[-1] ^ 1]) if frame[:-2] == bodies[1] else frame

        monkeypatch.setattr(rx, 'decode_frame', read)
    result = phy.decode_level(audio, '2300', level)
    assert result.located == 2 and result.frames == bodies[:1] and result.lost
    assert result.end > phy.burst_samples('2300', level)


@pytest.mark.parametrize('tail_location', ['validated', 'unlocated'])
def test_h2e_failed_tone_companion_with_small_fresh_retains_head(monkeypatch, tail_location):
    level, bw = 7, '2300'
    size = phy.body_size(bw, phy.level_key(bw, level))
    bodies = tuple(full_body(_payload(size - 1, i), 'W9SSJ', size, 0x99) for i in (1, 2))
    audio = _wave(bw, level, bodies)
    real = tone.decode_frame

    def read(*args, **kwargs):
        frame = real(*args, **kwargs)
        return frame[:-1] + bytes([frame[-1] ^ 1]) if frame[:-2] == bodies[1] else frame

    monkeypatch.setattr(tone, 'decode_frame', read)
    reads = []
    if tail_location == 'unlocated':
        audio = np.r_[audio, np.zeros(M.FS // 3)]
        decode = tone.decode_burst

        def burst(x, *args, **kwargs):
            reads.append(len(x))
            result = decode(x, *args, **kwargs)
            if len(x) < len(audio):
                return replace(result, frames=(), lock=0., located=0, end=None,
                               quality=replace(result.quality, snr_db=0.))
            return result

        monkeypatch.setattr(tone, 'decode_burst', burst)
    result = phy.decode_level(audio, bw, level, fresh=A._EXPECT_STEP)
    assert result is not None
    assert result.frames == bodies[:1] and result.located == 2 and result.lost
    assert phy.burst_samples(bw, level) < result.end <= len(audio)
    if tail_location == 'unlocated':
        assert len(reads) == 2 and reads[0] < reads[1] == len(audio)


def test_h2e_unpredicted_bw500_double_without_leading_silence_waits_for_companion(
        monkeypatch, record_property):
    bw, level = '500', phy.TOP['500']
    h, io, forced, ledger = _listening(monkeypatch, bw, level, 'responder', record_property)
    size = phy.body_size(bw, phy.level_key(bw, level))
    payloads = tuple(_payload(size - 1, i) for i in (1, 2))
    bodies = tuple(_body(h, level, p, 0x99) for p in payloads)
    audio = phy.render_level(bodies, bw, level, VB6Rnd(4107106))
    cut = phy.burst_samples(bw, level) + round(.4 * M.FS)
    assert h._peer_double_verdict() == 'NONE'
    assert phy.second_frame(audio[:cut], bw, level) is None
    assert phy.decode_level(audio[:cut], bw, level).frames == bodies[:1]
    began = _Clock.t
    _feed(h, audio[:cut])
    assert not io.sent and not io.host, io.msgs
    feed(h, audio[cut:])
    assert b''.join(io.host) == b''.join(payloads), io.msgs
    assert len(io.sent) == 1 and h._rx_window.A.kind == 'continue'
    assert io.at[-1] >= began + len(audio) / M.FS


@pytest.mark.parametrize('bw,level', [('2300', 7), ('500', 13)])
def test_h2e_none_head_under_sustained_keying_expires_incomplete(
        monkeypatch, record_property, bw, level):
    h, io, forced, ledger = _listening(monkeypatch, bw, level, 'responder', record_property)
    size = phy.body_size(bw, phy.level_key(bw, level))
    body = _body(h, level, _payload(size - 1, 1), 0x99)
    head = (_wave(bw, level, (body,)) if bw == '2300' else
            phy.render_level((body,), bw, level, VB6Rnd(4107106)))
    window = h._expected_window_samples()
    n = window + 2 * M.FS
    interference = np.random.default_rng(42).normal(size=n)
    interference *= 8 * np.sqrt(np.mean(head ** 2)) * np.exp(np.arange(n) / (4 * M.FS))
    expired = []
    nak = h._irs_nak

    def incomplete(read):
        expired.append(h._lv_heard)
        assert h._lv_heard >= window
        assert read.frames == (body,)
        assert h._rx_window.W.frames[0].body == body
        return nak(read)

    monkeypatch.setattr(h, '_irs_nak', incomplete)
    assert h._peer_double_verdict() == 'NONE'
    _feed(h, np.r_[head, interference])
    assert not io.host, io.msgs
    assert h._rx_window.W is None and h._rx_window.A is not None, io.msgs
    assert h._rx_window.A.kind == 'nak', io.msgs
    assert len(expired) == 1


def _h2f_held_head(monkeypatch, *, prompt):
    bw, level = '2300', 7
    h, io = station(bw, 'responder')
    h._link_high_speed = True
    h._expect_level, h._rx_read = level, (level, 30.)
    h._lv_owed = True
    if prompt:
        h._lv_prompt = True
    else:
        assert not h._lv_prompt
    h._key(False)
    assert h._lv_quiet is None if prompt else h._lv_quiet == 0
    size = phy.body_size(bw, phy.level_key(bw, level))
    body = _body(h, level, _payload(size - 1, 1), 0x99)
    head = phy.burst_samples(bw, level)
    reads, expired = [], []
    nak = h._irs_nak

    def decode(audio, *args, **kwargs):
        if not reads and h._lv_heard >= head:
            reads.append(h._lv_heard)
            return phy.LevelRead(level, (body,), 30., 1,
                                 end=head - h._lv_heard + len(audio))
        return None

    def incomplete(read):
        expired.append(h._lv_heard)
        assert read.frames == (body,)
        assert h._rx_window.W.frames[0].body == body
        return nak(read)

    monkeypatch.setattr(phy, 'decode_level', decode)
    monkeypatch.setattr(h, '_irs_nak', incomplete)
    assert h._peer_double_verdict() == 'NONE'
    return h, io, head, expired


def _h2f_poll(h, powers):
    audio = np.sqrt(2 * powers) * np.sin(2 * np.pi * 1500 * np.arange(len(powers)) / M.FS)
    for at in range(0, len(audio), 960):
        block = audio[at:at + 960]
        _Clock.t += len(block) / M.FS
        h._read_expected_level(block)


def test_h2f_retained_none_head_with_decreasing_activity_has_no_positive_answer(monkeypatch):
    h, io, head, expired = _h2f_held_head(monkeypatch, prompt=True)
    double = phy.burst_samples(h.bw, h._expect_level, 2)
    _h2f_poll(h, np.r_[np.full(head, 4.), np.full(double - head, 3.)])
    assert h._lv_pending is not None and h._rx_window.W is not None
    assert not io.sent and not io.host, io.msgs
    assert h._lv_fell is None
    _h2f_poll(h, np.full(M.FS, 2.))
    assert not io.host, io.msgs
    assert not io.sent, io.msgs
    assert h._rx_window.W is not None and not expired
    assert not phy.emission_ended(h._lv_buf, h.bw, h._expect_level, head_end=head)


def test_h2f_unprompted_held_head_expires_within_its_window(monkeypatch):
    h, io, head, expired = _h2f_held_head(monkeypatch, prompt=False)
    window = h._expected_window_samples()
    _h2f_poll(h, np.full(head + M.FS, 4.))
    assert h._lv_pending is not None and h._rx_window.W is not None
    assert h._lv_quiet == 0 and not io.sent and not io.host
    _h2f_poll(h, np.full(window + 960 - h._lv_heard, 4.))
    assert len(expired) == 1, (expired, h._lv_heard, h._lv_quiet, io.msgs)
    assert window <= expired[0] <= window + 960
    assert not io.host and len(io.sent) == 1, io.msgs
    assert h._rx_window.W is None and h._rx_window.A.kind == 'nak'
    assert h._lv_pending is None
    assert h._lv_pending_deadline is None


def test_h2f_later_rises_do_not_extend_held_head_expiry(monkeypatch):
    h, io, head, expired = _h2f_held_head(monkeypatch, prompt=True)
    window = h._expected_window_samples()
    _h2f_poll(h, np.full(head + M.FS, 4.))
    assert h._lv_pending is not None and h._rx_window.W is not None
    first = h._lv_first
    n = window + 960 - h._lv_heard
    powers = 16. * 4. ** (np.arange(n) // M.FS)
    changed = []
    track = h._track_emission

    def tracking():
        track()
        if h._lv_first is not None and h._lv_first != first:
            changed.append(h._lv_first)

    monkeypatch.setattr(h, '_track_emission', tracking)
    _h2f_poll(h, powers)
    assert changed and min(changed) > first
    assert len(expired) == 1, (expired, h._lv_heard, h._lv_first, io.msgs)
    assert window <= expired[0] <= window + 960
    assert not io.host and len(io.sent) == 1, io.msgs
    assert h._rx_window.W is None and h._rx_window.A.kind == 'nak'
    assert h._lv_pending is None
    assert h._lv_pending_deadline is None


@pytest.mark.parametrize('level', [2, 7, 10])
@pytest.mark.parametrize('following', ['noise', 'separate-over'])
def test_single_does_not_acquire_a_companion_from_noise_or_the_next_over(level, following):
    size = phy.body_size('2300', phy.level_key('2300', level))
    bodies = tuple(full_body(_payload(size - 1, i), 'W9SSJ', size, 0x99) for i in (1, 2))
    head = _wave('2300', level, bodies[:1])
    if following == 'noise':
        tail = np.random.default_rng(4).normal(0, .001, len(head))
    else:
        tail = np.r_[np.zeros(M.FS), _wave('2300', level, bodies[1:])]
    read = phy.decode_level(np.r_[head, tail], '2300', level)
    assert read is not None and read.located == 1 and len(read.frames) == 1
    assert read.frames[0] in bodies


@pytest.mark.parametrize('level', [7, 10])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_h2_sixth_hold_allows_a_found_companion_with_the_head(
        monkeypatch, record_property, level, role):
    h, io, forced, ledger = _listening(monkeypatch, '2300', level, role, record_property)
    cap = phy.body_size('2300', phy.level_key('2300', level)) - 1
    for i in range(6):
        _emission(h, io, ledger, level, (_payload(cap, i),),
                  (0x89 if i == 5 else 0x99,))
    assert h._peer_double_verdict() == 'POSSIBLE'
    assert h._receive_frame_count(level) == 2
    _emission(h, io, ledger, level, (_payload(cap, 80), _payload(cap, 81)), (0x99, 0x99))
    record_property('emissions', ledger)


@pytest.mark.parametrize('bw,level', [('2300', 2), ('2300', 3), ('2300', 7),
                                    ('2300', 10), ('2750', 2), ('2750', 3),
                                    ('2750', 7), ('2750', 11)])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_erased_single_close_after_h2_pair_query_between_single_and_double_bounds_naks(
        monkeypatch, record_property, bw, level, role):
    h, io, forced, ledger = _listening(monkeypatch, bw, level, role, record_property)
    cap = phy.body_size(bw, phy.level_key(bw, level)) - 1
    for i in range(6):
        _emission(h, io, ledger, level, (_payload(cap, i),), (0x99,))
    _emission(h, io, ledger, level, (_payload(cap, 80), _payload(cap, 81)), (0x99, 0x89))
    assert h._peer_double_verdict() == 'POSSIBLE' and h._receive_frame_count(level) == 2
    saved = h._rx_window.A
    assert saved.kind == 'continue' and saved.sent
    committed, before, unkeyed = b''.join(io.host), len(io.sent), h._unkeyed_at
    single, double = phy.envelope_lengths(bw, level)
    close = _wave(bw, level, (_body(h, level, _payload(7, 90), 0x82),))
    feed(h, np.zeros(len(close)))
    assert b''.join(io.host) == committed and len(io.sent) == before
    start = unkeyed + (single + double) / (2 * M.FS)
    assert unkeyed + single / M.FS < start < unkeyed + double / M.FS
    assert _Clock.t < start
    _feed(h, np.zeros(max(0, round((start - _Clock.t) * M.FS))))
    assert h._rx_window.W is None and h._rx_window.A is saved
    _Clock.t = start + A._span(h._idle_kind) / M.FS
    h._reanswer_over(query_start=start)
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert h._rx_window.A is not saved and b''.join(io.host) == committed
    ledger.append({'level': level, 'erased_close_samples': len(close),
                   'single_bound': unkeyed + single / M.FS,
                   'query_start': start, 'double_bound': unkeyed + double / M.FS,
                   'phase': None, 'answer': 'nak', 'host_hex': committed.hex()})
    record_property('emissions', ledger)


@pytest.mark.parametrize('bw,level', [('2300', 7), ('2300', 10), ('2750', 11)])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_answered_double_late_query_arms_replay_and_reads_both_frames_once(
        monkeypatch, record_property, bw, level, role):
    h, io, forced, ledger = _listening(monkeypatch, bw, level, role, record_property)
    cap = phy.body_size(bw, phy.level_key(bw, level)) - 1
    for i in range(6):
        _emission(h, io, ledger, level, (_payload(cap, i),), (0x99,))
    pair = (_payload(cap, 80), _payload(cap, 81))
    forced[0] = 7
    _emission(h, io, ledger, level, pair, (0x99, 0x99))
    assert h._expect_level == level + 3
    committed = b''.join(io.host)
    bodies = tuple(_body(h, level, p, 0x99) for p in pair)
    arm_replay(h, io)
    assert h._expect_level == level
    assert h._rx_window.T is not None and h._receive_frame_count(level) == 2
    assert h._rx_window.W is None and h._owed_block
    read_bodies, verdicts = [], []
    real = h._deliver

    def deliver(bodies, *args, **kwargs):
        read_bodies.extend(bodies)
        verdicts.append(h._peer_double_verdict())
        assert h._receive_frame_count(level) == 2
        return real(bodies, *args, **kwargs)

    monkeypatch.setattr(h, '_deliver', deliver)
    forced[0] = 0
    before = len(io.sent)
    audio = _wave(bw, level, bodies)
    feed(h, audio)
    assert tuple(read_bodies) == bodies and verdicts and set(verdicts) == {'NONE'}
    assert len(io.sent) == before + 1, io.msgs
    assert h._rx_window.A.kind == 'continue' and h._rx_window.A.sent
    assert b''.join(io.host) == committed
    assert h._rx_window.T is None and h._rx_window.W is None and not h._owed_block
    ledger.append({'level': level, 'replay_read': [b.hex() for b in read_bodies],
                   'answer': 'continue', 'host_hex': committed.hex(), 'replay_resolved': True})
    record_property('emissions', ledger)


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_every_row_reserves_one_lead_and_two_body_spans(bw):
    for level in range(1, phy.TOP[bw] + 1):
        single, double = phy.envelope_lengths(bw, level)
        assert (single, double) == (phy.burst_samples(bw, level), phy.burst_samples(bw, level, 2))
        assert single < double <= phy.LONGEST_BURST[bw]
        if (int(bw), level) in tone.ROWS:
            r = tone.ROWS[int(bw), level]
            assert double - single == (r.nsym - tone.C0) * tone.H
            assert single >= tone.H * (r.nsym - 1) + len(r.taps) + max(r.stagger)
        elif (int(bw), level) in ofdm.ROWS:
            r = ofdm.ROWS[int(bw), level]
            assert double - single == (ofdm.FRAME if r.qam else 149) * ofdm.SYM
    assert phy.LONGEST_BASE[bw] == max(phy.burst_samples(bw, level, 2) for level in range(1, 5))
