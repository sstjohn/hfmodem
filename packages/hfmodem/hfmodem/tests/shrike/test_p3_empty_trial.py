# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, p3acquire, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_tx_gear_hold import control, sender

FIXTURE = Path(__file__).with_name('fixtures') / 'ws8eoc-empty-1002'


def acceptance():
    host, seam = sender(enabled=False)
    a = host.arq
    a.dxcall = 'WS8EOC'
    a.cfg.p3_max_try = 2
    a._inflight = arq._Packet(0, b'FS ', 1, breakin=True)
    a._outbuf = bytearray(b'+\r')
    a._buffer_raw = 5
    a._next_seq = 1
    return host, seam


def deferred():
    host, seam = acceptance()
    control(host, 4)
    assert host.arq._buffer_raw == 2 and host.arq.acked == 1
    assert seam.frames == [(2, 1, b'+\r')]
    control(host, 4)
    assert host.arq._buffer_raw == 0 and host.arq.acked == 2
    assert host.arq._inflight is None
    assert host.arq._p3_next_trial_from == 1
    return host, seam


def test_recorded_empty_next_packet_falls_back_without_changing_the_counter():
    host, seam = deferred()
    a = host.arq
    a.on_cycle()
    assert a._inflight.trial_from == 1 and not a._inflight.idle_trial
    a.on_host_over()
    control(host, 2)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 2
    control(host, 2)
    assert seam.frames == [(2, 1, b'+\r')] + [(3, 0x1a, b'')]*2 + [(1, 0x1a, b'')]
    assert a._inflight.seq == 2 and a._inflight.trial_from is None
    assert not a._inflight.payload and not a._outbuf and a._buffer_raw == 0
    assert a.role == arq.ISS and a.state == arq.State.CONNECTED
    control(host, 1)
    assert a.role == arq.ISS and a._inflight.seq == 3
    assert a._inflight.trial_from is None and a.acked == 3
    control(host, 3)
    assert a.role == arq.IRS


def test_refused_deferred_trial_does_not_spend_maxtry():
    host, seam = deferred()
    a = host.arq
    seam.refuse = True
    a.on_cycle()
    assert a._inflight.trial_from == 1 and a._inflight.sent_at_level == 0
    for _ in range(4):
        control(host, 2)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 0
    seam.refuse = False
    for i in range(2):
        control(host, 2)
        assert a.speed_level == 3 and a._inflight.sent_at_level == i + 1
    control(host, 2)
    assert a.speed_level == 1 and a._inflight.seq == 2
    assert [sl for sl, _, _ in seam.frames] == [2, 3, 3, 1]


def test_onair_1002_2057_controls_end_the_sl3_loop():
    meta = json.loads((FIXTURE / 'controls.json').read_text())
    path = evidence.TREE / 'captures' / meta['recording']
    if not path.is_file():
        path = evidence.CAPTURES / meta['recording']
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == meta['sha256']
    audio = rxfront.load_wav(str(path))
    host, seam = acceptance()
    levels = []
    for i, row in enumerate(meta['controls']):
        lo = row['phase_sample'] - meta['before_samples']
        cut = p3acquire.compensate(audio[lo:lo + meta['window_samples']].astype(float),
                                   meta['offset_hz'])
        ev = rxfront.SyncedRx().control_signal_at(cut, meta['before_samples'])
        assert ev is not None and ev.cs + 1 == row['cs']
        assert (lo + ev.start) / meta['sample_rate'] == pytest.approx(row['onset_s'], abs=1/48000)
        ev.start = row['onset_s']
        host.on_rx_event(ev)
        a = host.arq
        if a._inflight is None:
            assert i == 1 and a._p3_next_trial_from == 1
            a.on_cycle()
            assert a._inflight.trial_from == 1
        if a._buffer_raw == 0:
            a.on_host_over()
        levels.append(a.speed_level)
        assert a.acked == min(i + 1, 2)
        if i >= 1:
            assert (a._inflight.seq, a._inflight.status, a._inflight.payload) == (2, 0x1a, b'')
            assert not a._outbuf and a._buffer_raw == 0
    assert len(levels) == 72 and levels == [2, 3, 3] + [1]*69
    assert seam.frames == [(2, 1, b'+\r')] + [(3, 0x1a, b'')]*2 + [(1, 0x1a, b'')]*69
    assert a.role == arq.ISS and a.state == arq.State.CONNECTED
    assert a._inflight.trial_from is None


@pytest.mark.parametrize('long_cycle', [False, True])
@pytest.mark.parametrize('minimum', [1, 2])
@pytest.mark.parametrize('queued', [b'', b'new host bytes'])
def test_deferred_trial_uses_the_plainly_established_level(long_cycle, minimum, queued):
    host, seam = deferred()
    a = host.arq
    a.cycle_long = long_cycle
    a.cfg.min_sl = minimum
    a.on_host_data(queued)
    for _ in range(3 if long_cycle else 1):
        a.on_cycle()
    seq, status = a._inflight.seq, a._inflight.status
    for _ in range(2):
        control(host, 2)
    assert a.speed_level == minimum and a.cycle_long == long_cycle
    assert (a._inflight.seq, a._inflight.status) == (seq, status)
    assert a._inflight.payload + bytes(a._outbuf) == queued
    assert a._buffer_raw == len(queued) and a._p3_next_trial_from is None


@pytest.mark.parametrize('transition', ['_take_link', '_give_link', '_enter_connected',
                                        '_finish_disconnected', '_reset_upgrade'])
def test_deferred_trial_cannot_leak_across_a_reset(transition):
    host, _ = deferred()
    getattr(host.arq, transition)()
    assert host.arq._p3_next_trial_from is None


def test_protocol_level_reset_clears_the_deferred_trial():
    host, _ = deferred()
    host.arq.speed_level = 1
    assert host.arq._p3_next_trial_from is None


@pytest.mark.parametrize('excluded', ['p1', 'p2', 'entry', 'breakin', 'qrt'])
def test_deferred_trial_tags_only_ordinary_p3_packets(excluded):
    host, _ = deferred()
    a = host.arq
    if excluded in ('p1', 'p2'):
        host.protocol = getattr(spec.Protocol, 'PACTOR' + excluded[-1])
    elif excluded == 'entry':
        a.entry_pending = True
    elif excluded == 'qrt':
        a._qrt_pending = True
    a._start_next_packet(breakin=excluded == 'breakin')
    assert a._inflight.trial_from is None and a._p3_next_trial_from is None


@pytest.mark.parametrize('defer_qrt', [False, True])
def test_speed_up_does_not_tag_the_next_qrt(defer_qrt):
    host, _ = acceptance()
    control(host, 4)
    if not defer_qrt:
        host.arq._qrt_pending = True
    control(host, 4)
    if defer_qrt:
        host.arq._qrt_pending = True
        host.arq.on_cycle()
    assert host.arq._inflight.status & spec.STATUS_QRT
    assert host.arq._inflight.trial_from is None


def test_cs5_steps_from_the_commanded_level_after_an_empty_trial_return():
    host, _ = deferred()
    a = host.arq
    a.on_cycle()
    control(host, 2)
    control(host, 2)
    assert a.speed_level == 1 and a._peer_sl == 3
    control(host, 5)
    assert a.speed_level == 2 and a._peer_sl == 2
    control(host, 5)
    assert a.speed_level == 1 and a._peer_sl == 1
    assert (a._inflight.seq, a._inflight.status, a._inflight.payload) == (2, 0x1a, b'')


@pytest.mark.parametrize('immediate', [False, True])
def test_empty_trial_returns_to_plainly_acknowledged_sl2(immediate):
    host, _ = acceptance()
    a = host.arq
    control(host, 4)
    control(host, 2)
    assert a._established_sl == 2 and a._inflight is None
    a.on_cycle()
    if immediate:
        a.on_host_over()
    control(host, 4)
    if not immediate:
        assert a._inflight is None
        a.on_cycle()
    assert a._inflight.trial_from == 2 and a._inflight.payload == b''
    seq, status = a._inflight.seq, a._inflight.status
    control(host, 1)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 2
    control(host, 1)
    assert a.speed_level == 2 and a._inflight.trial_from is None
    assert (a._inflight.seq, a._inflight.status) == (seq, status)
    assert a._buffer_raw == 0 and not a._outbuf and a.role == arq.ISS
