# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Real entry ACKs missed live: early P3 timing, fine CFO, then P1-grid release."""
import hashlib
import json
from functools import cache
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, pactor1, placement, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_granted_entry_retry import granted

ROOT = Path(__file__).with_name('fixtures') / 'p3-entry-cs'
METADATA = ROOT / 'metadata.json'


@cache
def rows():
    if not METADATA.exists():
        pytest.skip(f'entry-CS recordings absent: {METADATA}')
    return {r['file']: r for r in json.loads(METADATA.read_text())}


def recorded(name):
    row = rows()[name]
    with wave.open(str(ROOT / name)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (48000, 1, 2)
        raw = wav.readframes(wav.getnframes())
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    return row, np.frombuffer(raw, dtype='<i2').astype(float) / 32768


def entry_session():
    sess = _Session(entry_pending=True)
    sess.host.arq.on_host_data(b'bench payload')
    sess.host.arq._next_seq = 2
    sess.host.arq._start_next_packet()
    assert sess.host.arq.tx_seq == 2
    return sess


@pytest.mark.parametrize('station,windows,cycles', [
    ('ve3kpg-arm01', [2, 4, 5], [1, 2, 1]),
    ('kb5lzk-arm09', [9, 10, 11], [1, 1, 1]),
])
def test_recorded_ack_confirms_entry_and_survives_old_grid_release(station, windows, cycles):
    sess = entry_session()
    answers = []
    for n, elapsed in zip(windows, cycles):
        row, x = recorded(f'{station}-hold{n:02}.wav')
        for _ in range(elapsed):
            sess.rx.new_cycle()
        answers.append(sess.rx.control_signal(x, row['start_stream_sample'],
                                               row['anchor_stream_sample']))
        if len(answers) == 1:
            assert sess.host.arq.entry_pending
        else:
            assert not sess.host.arq.entry_pending
            assert sess.host.protocol == spec.Protocol.PACTOR3
        # Same physical response cannot advance the state twice in one cycle.
        before = len(sess.events)
        assert sess.rx.control_signal(x, row['start_stream_sample'],
                                      row['anchor_stream_sample']) is None
        assert len(sess.events) == before
    assert answers == [None, arq.CS_ACK, arq.CS_ACK]
    assert sess.host.arq.role == arq.ISS  # CS1 is not a CS3 role change.
    assert not sess.rx._p3_changeover_pending
    assert sess.rx._p3_row0 is None  # A bare ACK cannot invent a DATA clock.
    assert sess.host.arq.tx_seq == 3  # Repeated CS1 does not ACK odd packet 3.
    # ...and what it is asking for again is the traffic. The counter-3 packet
    # used to go out with an empty field, and the request for it cost nothing.
    assert sess.host.arq._inflight.payload == b'bench payload'
    assert sess.host.arq._inflight.repeats == 1


def test_nonfollowing_ve3_arm_does_not_confirm_entry():
    sess = entry_session()
    for n in [7, 12, 15]:
        row, x = recorded(f've3kpg-arm14-hold{n:02}.wav')
        sess.rx.new_cycle()
        assert sess.rx.control_signal(x, row['start_stream_sample'],
                                      row['anchor_stream_sample']) is None
    assert sess.host.arq.entry_pending
    assert not any(ev.protocol == spec.Protocol.PACTOR3 for ev in sess.events)


@pytest.mark.parametrize('cs', range(6))
@pytest.mark.parametrize('offset', [-65, -20, 0, 55])
def test_all_control_words_acquire_at_receive_offsets(cs, offset):
    x = np.pad(placement.control_signal(cs), (4800, 4800))
    got = p3acquire.control_signal(p3acquire.compensate(x, -offset))
    assert got is not None and got.event.cs == cs


@pytest.mark.parametrize('seed', range(12))
def test_noise_is_not_an_entry_control(seed):
    x = np.random.default_rng(seed).normal(0, .1, 22080)
    assert p3acquire.control_signal(x) is None


@pytest.mark.parametrize('cs', range(5))
@pytest.mark.parametrize('inverted', [False, True])
def test_p1_grants_and_controls_are_not_p3(cs, inverted):
    x = np.pad(pactor1.control_signal(cs, invert=inverted), (4800, 4800))
    assert p3acquire.control_signal(x) is None


def test_same_recorded_window_in_later_cycle_cannot_corroborate_itself():
    sess = entry_session()
    row, x = recorded('kb5lzk-arm09-hold09.wav')
    for _ in range(3):
        sess.rx.new_cycle()
        assert sess.rx.control_signal(x, row['start_stream_sample'],
                                      row['anchor_stream_sample']) is None
    assert sess.host.arq.entry_pending


def test_recorded_control_stops_the_real_granted_entry_silence_budget():
    host, _ = granted()
    rx = onair._SessionRx(host, tag='RECORDED')
    # KB5LZK's first unrecognized P3 cycle preceded the two strong recordings.
    host.tick()
    assert host.arq._unanswered_upgrade == 1
    for n in [9, 10]:
        row, x = recorded(f'kb5lzk-arm09-hold{n:02}.wav')
        rx.new_cycle()
        rx.control_signal(x, row['start_stream_sample'], row['anchor_stream_sample'])
        host.tick()
    assert host.protocol == spec.Protocol.PACTOR3
    assert not host.arq.entry_pending
    assert host.arq._unanswered_upgrade is None


def test_confirmed_response_is_not_delivered_again_from_old_audio():
    sess = entry_session()
    for n in [9, 10]:
        row, x = recorded(f'kb5lzk-arm09-hold{n:02}.wav')
        sess.rx.new_cycle()
        sess.rx.control_signal(x, row['start_stream_sample'], row['anchor_stream_sample'])
    count = len(sess.events)
    sess.rx.new_cycle()
    assert sess.rx.control_signal(x, row['start_stream_sample'], row['anchor_stream_sample']) is None
    assert len(sess.events) == count


@pytest.mark.parametrize('change', ['role', 'protocol', 'disconnect'])
def test_acquired_answer_clock_does_not_outlive_its_sending_turn(change):
    sess = entry_session()
    sess.rx._p3_answer_at = 48000
    sess.rx._p3_head_candidate = (1, 48000, -65)
    if change == 'role':
        sess.host.arq.role = arq.IRS
    elif change == 'protocol':
        sess.host.protocol = spec.Protocol.PACTOR1
    else:
        sess.host.arq.state = arq.State.DISCONNECTED
    sess.rx.new_cycle()
    assert sess.rx._p3_answer_at is None
    assert sess.rx._p3_head_candidate is None


@pytest.mark.parametrize('trim', [1, 12])
def test_recropping_one_physical_response_does_not_make_two_cycles(trim):
    sess = entry_session()
    row, x = recorded('kb5lzk-arm09-hold09.wav')
    sess.rx.new_cycle()
    assert sess.rx.control_signal(x, row['start_stream_sample'], row['anchor_stream_sample']) is None
    sess.rx.new_cycle()
    assert sess.rx.control_signal(x[trim:], row['start_stream_sample'] + trim,
                                  row['anchor_stream_sample']) is None
    assert sess.host.arq.entry_pending


@pytest.mark.parametrize('long_cycle', [False, True])
def test_answer_projection_uses_current_short_or_long_cycle(long_cycle):
    sess = entry_session()
    sess.host.arq.entry_pending = False
    sess.host.arq.cycle_long = long_cycle
    period = spec.CYCLE_LONG_S if long_cycle else spec.CYCLE_SHORT_S
    origin = round((1 + period) * 48000)
    x = np.pad(placement.control_signal(arq.CS_ACK), (4800, 4800))
    # Establish the shaped waveform's measured phase, including its filter head.
    first = sess.rx._p3_cs(x, 4800, seg_start=48000)
    assert first is not None
    previous = sess.rx._p3_answer_at
    # Old P1 prediction is 150 ms late and outside the tracked reader's bracket.
    ev = sess.rx._p3_cs(x, round(.25 * 48000), seg_start=origin)
    assert ev is not None and ev.cs == arq.CS_ACK
    assert sess.rx._p3_answer_at == previous + round(period * 48000)


@pytest.mark.parametrize('read_complete_body', [True, False])
def test_complete_changeover_crc_confirms_first_acquired_cycle(monkeypatch,
                                                              read_complete_body):
    """Bound the head search without discarding an already-captured CRC body."""
    sess = entry_session()
    origin = 10 * 48000
    audio = np.pad(placement.changeover_packet(b'RMS', 0), (4800, 4800))
    audio = p3acquire.compensate(audio, 65)  # Peer RX offset -65 Hz.
    if not read_complete_body:
        # Negative control: every event reader gets only the acquired head,
        # even though control_signal was handed the complete received packet.
        # This is the behavior of the bounded-search-only implementation.
        make_event = onair.rxfront._cs_event

        def head_only(samples, ci, errors, at, t, tag):
            return make_event(samples[:at + round(.23 * 48000)],
                              ci, errors, at, t, tag)

        monkeypatch.setattr(onair.rxfront, '_cs_event', head_only)
    sess.rx.new_cycle()
    # The old P1 anchor is late enough that its tracked bracket misses the
    # actual head. No deep_scan is invoked to cover for this production seam.
    result = sess.rx.control_signal(audio, origin, origin + round(.22 * 48000))
    if not read_complete_body:
        assert result is None
        assert sess.host.arq.entry_pending
        assert sess.host.arq.role == arq.ISS
        assert not sess.packets
        return
    assert result == arq.CS_BREAKIN
    assert not sess.host.arq.entry_pending
    assert sess.host.arq.role == arq.IRS
    assert [ev.packet[2] for ev in sess.packets] == [b'RMS']
    assert bytes(sess.host.channel(sess.host.ptchn).rx) == b'RMS'
    assert not sess.rx._p3_changeover_pending
