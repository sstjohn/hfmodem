# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p4frame, p4shorttx, session
from hfmodem.shrike.p4hold import BreakinHold
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver

ROOT = Path(__file__).resolve().parents[5]
FS = 48000


def fields(status=1):
    frame = p4frame.decode_field(p4frame.encode_field(
        b'first copy'.ljust(40, b'\x1e'), p4frame.Status.from_byte(status)), data_bytes=40)
    return frame, 2*(1.-2*p4shorttx.coded_bits(p4frame.encode_field(frame.data, frame.status)).astype(float))


def test_fresh_normal_crc_fits_thirty_ms_even_after_a_slow_full_decode():
    hold = BreakinHold(FS)
    hold.decode_times = [.067]
    hold.now = lambda: 0.
    frame, scores = fields()
    assert hold.normal(FS, 0, 0., .7, scores, None, deadline=.030) == frame
    event = next(e for e in hold.events if e['event'] == 'normal-decode' and e['reason'] == 'accepted')
    assert event['iterations'] == 1
    hold.reset('our key')
    assert hold.normal(FS+60000, 0, 0., .7, scores, None, deadline=.030) == frame


@pytest.mark.parametrize('at,status', [(37.247, 1), (39.747, 2), (42.247, 2), (139.749, 2)])
def test_ws0939_fresh_copy_decodes_inside_its_recorded_allowance(at, status):
    path = ROOT/'captures/onair-0929-0939/stream.flac'
    if not path.exists():
        pytest.skip('local WS8EOC 0939 recording absent')
    audio = session.load_wav(str(path), FS)
    at = round(at*FS)
    start = at-round(.05*FS)
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=2.51)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, 2.51)
    rx.peer_onset = at-60000
    rx.acked_counter = 1
    rx.hold.decode_times = [.067]
    rx.normal_deadline = lambda onset: rx.hold.now()+.030
    obs = rx._normal(audio[start:at+round(.846*FS)], start)
    assert obs is not None and obs.frame is not None
    assert obs.frame.status.byte == status
    assert len(rx.hold.members) == 1
    assert any(e.get('iterations') == 1 and e.get('reason') == 'accepted' for e in rx.hold.events)


def test_fast_miss_leaves_full_decode_for_a_later_budget(monkeypatch):
    from hfmodem.shrike import p4short
    hold = BreakinHold(FS)
    clock = [0.]
    hold.now = lambda: clock[0]
    frame, scores = fields()
    decode = p4short.decode_scores
    calls = []

    def timed(scores, *, iterations=8):
        calls.append(iterations)
        clock[0] += .008 if iterations == 1 else .055
        return (None, None) if iterations == 1 else decode(scores, iterations=iterations)

    monkeypatch.setattr(p4short, 'decode_scores', timed)
    assert hold.normal(FS, 0, 0., .7, scores, None, deadline=.050) is None
    assert calls == [1]
    assert hold._normal_decode(.050) is None
    assert calls == [1]
    assert hold.refine(deadline=.100) == frame
    assert calls == [1, 8]
    assert hold.refine(deadline=.100) == frame
    assert calls == [1, 8]


@pytest.mark.parametrize('status,polarity', [(2, 1), (1, -1)])
def test_fast_crc_keeps_counter_and_training_polarity_checks(status, polarity):
    hold = BreakinHold(FS)
    hold.now = lambda: 0.
    _, scores = fields(status)
    assert hold.normal(FS, 0, 0., .7, scores*polarity, None, deadline=.030) is None
    reason = 'status' if polarity == 1 else 'polarity'
    assert any(e.get('iterations') == 1 and e.get('reason') == reason for e in hold.events)


def test_fast_decode_requires_a_unique_crc_polarity(monkeypatch):
    from hfmodem.shrike import p4decode, p4short
    frame, scores = fields()
    calls = []
    decode = p4decode.turbo_decode

    def ambiguous(*a, **kw):
        calls.append(kw['iterations'])
        return decode(*a, **kw)

    monkeypatch.setattr(p4decode, 'turbo_decode', ambiguous)
    monkeypatch.setattr(p4frame, 'decode_field', lambda *a, **kw: frame)
    assert p4short.decode_scores(scores, iterations=1) == (None, None)
    assert calls == [1]


def test_driver_answers_first_normal_copy_with_a_costed_fast_decode(monkeypatch):
    from hfmodem.shrike import onair, p4breakintx, p4short
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    audio = np.zeros(6*FS)
    at = round(.923*FS)
    greeting = p4breakintx.changeover_packet(b'hello', 0)
    audio[at:at+len(greeting)] = greeting
    wave = p4shorttx.short_packet(b'first copy', 1)
    starts = [at+60000*k for k in (1, 2)]
    for start in starts:
        head = start+round(308*FS/28800)
        audio[head:head+len(wave)] = wave
    stream = Stream(audio)
    wall = [0]
    stream.sample_now = lambda: max(stream.pos, wall[0])
    stream.clamp_late = lambda instant: max(0, stream.sample_now()-instant)
    take = stream.take_until

    def taking(end):
        part = take(end)
        wall[0] = max(wall[0], stream.pos)
        return part

    stream.take_until = taking
    monkeypatch.setattr(onair.time, 'monotonic', lambda: wall[0]/FS)
    tx = Tx(stream)
    tx.settle = .044
    tx.sessrx.p4.enable_hold()
    tx.sessrx.p4.hold.decode_times = [.080]
    tx.sessrx.p4.hold.now = lambda: wall[0]/FS
    key = tx._tx

    def flushing(audio, what):
        assert stream.sample_now() <= tx.raster.boundary(tx.slot)
        key(audio, what)
        stream.pos = tx.tx_end
        wall[0] = max(wall[0], stream.pos)

    tx._tx = flushing
    decode = p4short.decode_scores
    calls = []

    def timed(scores, *, iterations=8):
        calls.append(iterations)
        wall[0] += round((.008 if iterations == 1 else .080)*FS)
        return decode(scores) if iterations == 8 else decode(scores, iterations=iterations)

    monkeypatch.setattr(p4short, 'decode_scores', timed)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    requests = {e['at'] for e in tx.p4_probe.link_events
                if e['event'] == 'Keyed' and not e.get('current_slot_accepted', True)}
    words = [e for e in tx.p4_probe.emissions[1:] if e['audio_start'] not in requests]
    assert [e['word'] for e in words] == ['X27', 'V27', 'V27']
    assert [e['audio_start'] for e in words[1:]] == pytest.approx(
        [start+44160 for start in starts], abs=48)
    assert calls == [1, 1]


@pytest.mark.parametrize('rejection', ['status', 'distance'])
def test_rejected_fast_crc_still_allows_full_decode(monkeypatch, rejection):
    from hfmodem.shrike import p4short
    hold = BreakinHold(FS)
    hold.now = lambda: 0.
    frame, scores = fields()
    rejected = (fields(status=2)[0] if rejection == 'status' else
                p4frame.decode_field(p4frame.encode_field(bytes(range(40)), frame.status), data_bytes=40))
    calls = []

    def decoding(scores, *, iterations=8):
        calls.append(iterations)
        return (rejected if iterations == 1 else frame), 1

    monkeypatch.setattr(p4short, 'decode_scores', decoding)
    assert hold.normal(FS, 0, 0., .7, scores, None, deadline=.100) == frame
    assert calls == [1, 8]
    events = [e for e in hold.events if e.get('event') == 'normal-decode']
    assert [e['reason'] for e in events] == [rejection, 'accepted']


def test_slow_fast_stage_does_not_block_a_full_decode_that_fits():
    hold = BreakinHold(FS)
    hold.quick_decode_times = [.080]
    hold.now = lambda: 0.
    frame, scores = fields()
    assert hold.normal(FS, 0, 0., .7, scores, None, deadline=.060) == frame
    events = [e for e in hold.events if e.get('event') == 'normal-decode']
    assert [(e['iterations'], e['reason']) for e in events] == [(8, 'accepted')]
