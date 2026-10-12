# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WC9P-10 and K5FIT, 2026-09-27: high speed capable links their closes had taken under
level 4, and answers to our overs on the air and unread.

WC9P-10 continued our record-1 over at frame 4 under the continue's floor, then
keyed 288/249 to every query behind it; K5FIT NAKed our record-2 final over with
two of the eight pairs faded under the clearance. NS0A's query answer at frame 6
stepped our next over to level 5, and we closed on its silence unqueried.
K5FIT at 15:18Z NAKed our level-6 over behind all three queries, faded under
the hard bar twice, and the retry the third drew inherited a spent budget.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_over_gate import _connected

FS = MK.FS
NOISE = 0.02
# Against NOISE: the frame-4 continue scores 1.88 (WC9P-10's 1.85), the query
# answer reads 29 of 31 tones, the NAK scores 2.8 (K5FIT's 2.75).
CONT_GAIN, ANSWER_GAIN, NAK_GAIN = 0.006, 0.008, 0.04


def station(monkeypatch, called, close_level, data):
    """A high speed capable caller whose peer's close named ``close_level``, with
    ``data`` keyed as its first over."""
    clock = [1000.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(VA.OF, 'data_over_tx', lambda *a, **k: np.zeros(64))
    hs, io = _connected(bw='2300')
    hs.called, hs._link_high_speed, hs.turn = called, True, VA._TURN_OURS
    close = phy.vara_body(b'x' * 5, hs.caller, tail=0x42, body_len=phy.body_size(
        '2300', phy.level_key('2300', close_level)))
    hs._note_close_level([close[:-2] + bytes([close_level, 0x42])])
    hs.send(data)
    assert phy.host_level('2300', hs._tx_pending[2]) == close_level
    return hs, io, clock


def air(*parts, seed, span=2.5):
    """``span`` seconds behind our last sample: our RX-mute residue, then
    ``parts`` as (seconds, waveform), in band noise."""
    rng = np.random.default_rng(seed)
    span = int(span * FS)
    y = rng.standard_normal(span) * NOISE
    y[:1152] *= 1e-2
    for t, x in parts:
        y[int(t * FS):int(t * FS) + len(x)] += x
    return y


def drive(hs, io, clock, audio):
    """Feed 20 ms blocks, keying whatever query the driver would, until the
    pending over moves or a query keys."""
    old, start, keys = hs._tx_pending, clock[0], io.keys
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
        clock[0] = start + (at + 960) / FS
        if hs._tx_pending is not old:
            return 'stream'
        for due, go in ((hs.intermediate_query_due_in, hs.query_intermediate_answer),
                        (hs.data_retry_due_in, hs.retry_data_over)):
            d = due()
            if d is not None and d <= 0:
                go()
        if io.keys > keys:
            return 'query'
    return None


def faded_continue(hs, frame, gain):
    return MK.synth_tone_pairs(hs._peer_continue_tails()[frame]) * gain


def test_wc9p_the_query_answer_at_frame_4_retires_the_over_at_level_4(monkeypatch):
    hs, io, clock = station(monkeypatch, 'WC9P-10', 2, bytes(605))
    old = hs._tx_pending
    cont = faded_continue(hs, 4, CONT_GAIN)
    x = air((0.183, cont), seed=1)
    mag = VA._band_power(x, hs._band, hs._grid)
    assert 1.6 < VA._frame_score(mag, hs._peer_continue_tails()[4], hs._band) < 2.0
    assert drive(hs, io, clock, x) == 'query', io.msgs
    answer = MK.synth_burst(hs.called, VF.SESSION_RETRY_QUERY_ANSWER) * ANSWER_GAIN
    assert drive(hs, io, clock, air((0.17, answer), seed=2, span=4.8)) == 'stream', io.msgs
    assert hs._tx_pending[1] == old[1] + 1
    assert phy.host_level('2300', hs._tx_pending[2]) == 4
    assert any('(288/249) — acknowledged' in m for m in io.msgs), io.msgs
    assert not any(m.startswith('intermediate query 2/') for m in io.msgs), io.msgs


def test_ns0a_an_unanswered_over_at_level_5_is_queried_not_closed(monkeypatch):
    hs, io, clock = station(monkeypatch, 'NS0A', 4, bytes(601))
    assert drive(hs, io, clock, air(seed=1)) == 'query', io.msgs
    answer = MK.synth_burst(hs.called, VF.SESSION_RETRY_QUERY_ANSWER_B) * ANSWER_GAIN
    assert drive(hs, io, clock, air((0.17, answer), seed=2, span=4.8)) == 'stream', io.msgs
    assert phy.host_level('2300', hs._tx_pending[2]) == 5 and hs._txq
    assert drive(hs, io, clock, air(seed=3, span=4.8)) == 'query', io.msgs
    assert not any('closing without blind' in m for m in io.msgs), io.msgs
    assert any(m.startswith('intermediate query 1/') and 'record 4' in m for m in io.msgs), io.msgs


def test_k5fit_a_nak_with_two_faded_pairs_resends_before_the_first_query(monkeypatch):
    hs, io, clock = station(monkeypatch, 'K5FIT', 3, bytes(32))
    assert phy.over_is_last(hs._tx_pending[0], hs.caller)
    nak = MK.synth_tone_pairs(VF.nak(hs.called, hs.bw)[1]) * NAK_GAIN
    nak[2 * MK.HOP:3 * MK.HOP] *= 0.1
    nak[5 * MK.HOP:6 * MK.HOP] *= 0.1
    x = air((0.205, nak), seed=3)
    assert hs._peer_nak_at(x[:int(1.5 * FS)]) is None
    assert drive(hs, io, clock, x) == 'stream', io.msgs
    assert hs._tx_retries == 1
    assert any(m.startswith('rx session-data-pair-nak') for m in io.msgs), io.msgs
    assert not any(m.startswith('final-answer query') for m in io.msgs), io.msgs


@pytest.mark.parametrize('frame', [0, 5, 6])
def test_a_continue_is_not_read_as_the_nak(monkeypatch, frame):
    hs, io, clock = station(monkeypatch, 'K5FIT', 3, bytes(605))
    x = air((0.2, faded_continue(hs, frame, 1.0)), seed=4)
    assert hs._held_nak_at(x) is None


def k5fit_1518(monkeypatch):
    """K5FIT 15:18Z: over #2 at level 4 continued at frame 4, over #3 at level 6
    with two blocks queued behind it."""
    hs, io, clock = station(monkeypatch, 'K5FIT', 4, bytes(89 + 173 + 301))
    hs._over = 2
    hs._tx_pending = (hs._tx_pending[0], 2, hs._tx_pending[2])
    hs._continue_frame = (hs._tx_pending, 4)
    hs._took_over_continue()
    assert hs._tx_pending[1:] == (3, 5) and hs._txq, io.msgs
    return hs, io, clock


def over_nak(hs, kind, gain=ANSWER_GAIN):
    return MK.synth_burst(hs.called, VF.for_bw(kind, hs.bw)) * gain


def test_k5fit_a_retry_owns_its_own_queries(monkeypatch):
    hs, io, clock = k5fit_1518(monkeypatch)
    for n, span in ((1, 2.5), (2, 4.8), (3, 4.8)):
        assert drive(hs, io, clock, air(seed=10 + n, span=span)) == 'query', io.msgs
        assert io.msgs[-1].startswith(f'intermediate query {n}/3 (60/745)'), io.msgs
    nak = over_nak(hs, VF.SESSION_OVER_NAK_RESPONDER)
    assert drive(hs, io, clock, air((0.158, nak), seed=14, span=4.8)) == 'stream', io.msgs
    assert hs._tx_pending[1:] == (3, 3) and hs._tx_retries == 1, io.msgs
    assert drive(hs, io, clock, air(seed=15)) == 'query', io.msgs
    assert io.msgs[-1].startswith('intermediate query 1/3 (60/683)'), io.msgs
    assert not any('closing without blind' in m for m in io.msgs), io.msgs


def faded_short_over_nak(hs):
    """289/541 with five symbols under the noise and five taken by a carrier
    two bins up: K5FIT's first copy, 6 of 15 tones exact and 5 wrong."""
    kind = VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER_SHORT, hs.bw)
    tones = VF.handshake_tones(hs.called, kind)
    true = MK.synth_tone_pairs([(t, t) for t in tones]) * 0.03
    other = MK.synth_tone_pairs([(t + 2, t + 2) for t in tones]) * 0.03
    x = true.copy()
    for k in (1, 4, 7, 10, 13):
        x[k * MK.HOP:(k + 1) * MK.HOP] *= 0.02
    for k in (2, 5, 8, 11, 14):
        s = slice(k * MK.HOP, (k + 1) * MK.HOP)
        x[s] = true[s] * 0.5 + other[s]
    return kind, np.asarray(tones), x


def test_k5fit_a_faded_short_over_nak_behind_the_first_query_is_read(monkeypatch):
    hs, io, clock = k5fit_1518(monkeypatch)
    assert drive(hs, io, clock, air(seed=11)) == 'query', io.msgs
    kind, tones, nak = faded_short_over_nak(hs)
    x = air((0.171, nak), seed=0, span=4.8)
    seg = x[:int(1.9 * FS)]
    heard, _ = VA._payload_fit(seg, kind, tones, None, hs._band)
    assert (heard[1:] == tones[1:]).sum() == 6
    assert ((heard[1:] >= 0) & (heard[1:] != tones[1:])).sum() >= 4
    score, _ = VA._frame_fit(VA._band_power(seg, hs._band, hs._grid),
                             [(t, t) for t in tones], hs._band)
    assert 2.2 < score < 2.8
    assert hs._short_over_nak_at(seg, None) is None
    assert drive(hs, io, clock, x) == 'stream', io.msgs
    assert any(m.startswith('rx responder NAK from K5FIT') for m in io.msgs), io.msgs
    assert hs._tx_pending[1:] == (3, 3) and hs._tx_retries == 1, io.msgs
    assert not any(m.startswith('intermediate query 2/') for m in io.msgs), io.msgs


def last_over_at_record_6(monkeypatch):
    """The stock-vs-ours bench, 2026-09-27: our last full over at level 7 unread,
    and the peer silent until our final-answer query."""
    hs, io, clock = station(monkeypatch, 'WI9YZA', 7, bytes(255))
    assert hs._tx_pending[2] == 6 and hs._final_ack_candidate()
    return hs, io, clock


@pytest.mark.parametrize('kind', [VF.SESSION_OVER_NAK_RESPONDER_SHORT,
                                  VF.SESSION_OVER_NAK_RESPONDER])
def test_missing_data_behind_the_final_query_resends_two_records_down(monkeypatch, kind):
    hs, io, clock = last_over_at_record_6(monkeypatch)
    assert drive(hs, io, clock, air(seed=21)) == 'query', io.msgs
    assert io.msgs[-1].startswith('final-answer query 1/'), io.msgs
    x = air((0.17, over_nak(hs, kind)), seed=22, span=4.8)
    assert drive(hs, io, clock, x) == 'stream', io.msgs
    assert hs._tx_pending[2] == 4 and hs._tx_retries == 1, io.msgs
    assert not any(m.startswith('final-answer query 2/') for m in io.msgs), io.msgs


def test_a_pair_nak_resends_one_record_down(monkeypatch):
    hs, io, clock = last_over_at_record_6(monkeypatch)
    nak = MK.synth_tone_pairs(VF.nak(hs.called, hs.bw)[1]) * NAK_GAIN
    assert drive(hs, io, clock, air((0.205, nak), seed=23)) == 'stream', io.msgs
    assert hs._tx_pending[2] == 5 and hs._tx_retries == 1, io.msgs
