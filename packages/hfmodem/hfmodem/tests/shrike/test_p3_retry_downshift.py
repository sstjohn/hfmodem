# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bound WS8EOC's recorded SL3 repeat train without settling unacked bytes."""
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, p3acquire, rxfront, spec
from hfmodem.tests.shrike.test_tx_gear_hold import BODY, control, sender

FIXTURE = Path(__file__).with_name('fixtures') / 'ws8eoc-speed-0920'


def at_sl3():
    host, seam = sender(enabled=False)
    control(host, 4)  # Changeover accepted; first ordinary packet at SL2.
    control(host, 4)  # SL2 accepted; next packet at SL3, counter 2.
    return host, seam


def assert_pending(host):
    a = host.arq
    assert a.tx_seq == 2
    assert a._buffer_raw == len(BODY) - 23
    assert a._inflight.payload + bytes(a._outbuf) == BODY[23:]


def test_recorded_ws8eoc_controls_bound_trial_then_hold_established_speed():
    if not (FIXTURE / 'controls.json').is_file():
        pytest.skip(f'optional recording metadata is not included: {FIXTURE}')
    meta = json.loads((FIXTURE / 'controls.json').read_text())
    path = FIXTURE / 'controls.wav'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
    audio = rxfront.load_wav(str(path))
    host, seam = sender(enabled=False)
    n = meta['window_samples']
    for i, row in enumerate(meta['controls']):
        cut = p3acquire.compensate(audio[i*n:(i+1)*n], meta['offset_hz'])
        ev = rxfront.SyncedRx().control_signal_at(cut, meta['local_phase'])
        assert ev is not None and ev.cs + 1 == row['cs']
        host.on_rx_event(ev)
        if i >= 1:
            assert_pending(host)
    assert [sl for sl, _, _ in seam.frames] == [2, 3, 3, 2, 2, 2, 2, 2]
    assert [status & 3 for _, status, _ in seam.frames] == [1] + [2]*7
    assert [len(data) for _, _, data in seam.frames] == [23, 59, 59, 23, 23, 23, 23, 23]
    assert host.arq.state == arq.State.CONNECTED


@pytest.mark.parametrize('answered', [True, False])
def test_refused_slots_do_not_spend_either_levels_transmit_budget(answered):
    host, seam = at_sl3()
    a = host.arq
    seam.refuse = True
    for _ in range(4):
        a._on_nak(False, answered=answered)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 1
    seam.refuse = False
    a._on_nak(False, answered=answered)
    assert a.speed_level == 3 and a._inflight.sent_at_level == 2
    seam.refuse = True
    for _ in range(4):
        a._on_nak(False, answered=answered)
    assert a.speed_level == 2 and a._inflight.sent_sl == 3
    assert len(a._inflight.payload) == 59
    assert_pending(host)
    seam.refuse = False
    a._on_nak(False, answered=answered)
    assert a._inflight.sent_sl == 2 and a._inflight.sent_at_level == 1
    assert len(a._inflight.payload) == 23
    assert_pending(host)


def test_ack_resets_budget_and_settles_only_the_smaller_transmitted_field():
    host, seam = at_sl3()
    for _ in range(3):
        control(host, 2)
    assert host.arq.speed_level == 2
    control(host, 1)  # ACK for counter 2; the next packet is counter 3.
    assert host.arq.tx_seq == 3 and host.arq._inflight.sent_at_level == 1
    assert host.arq._buffer_raw == len(BODY) - 46
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == BODY[46:]
    for _ in range(2):
        control(host, 1)  # Request for counter 3.
    assert seam.frames[-1][0] == 2
    control(host, 1)
    assert seam.frames[-1][0] == 2  # ACK established SL2: no trial remains.


@pytest.mark.parametrize("repeats", [0, 1])
def test_explicit_cs5_drops_immediately_but_never_twice(repeats):
    host, seam = at_sl3()
    for _ in range(repeats):
        control(host, 2)
    control(host, 5)
    assert seam.frames[-1][0] == 2 and host.arq._inflight.sent_at_level == 1
    control(host, 5)
    assert seam.frames[-1][0] == 1
    assert_pending(host)


@pytest.mark.parametrize('long_cycle', [False, True])
def test_real_renderer_downshift_preserves_clock_counter_and_queued_suffix(long_cycle):
    from hfmodem.tests.shrike.test_long_sl1 import PAYLOAD, linked, row

    host, tx = linked(sl=1, long=long_cycle)
    a = host.arq
    a.on_host_data(PAYLOAD)
    ticks = 3 if long_cycle else 1
    host.tick(elapsed_ticks=ticks, cycle_ticks=ticks)
    accepted = len(a._inflight.payload)
    a.on_rx_cs(arq.CS_SPEED_UP)
    seq = a.tx_seq
    for _ in range(2):
        a.on_rx_cs(arq.CS_REQUEST)
    assert a.speed_level == 1 and a.tx_seq == seq
    assert a.cycle_long == long_cycle
    assert a._buffer_raw == len(PAYLOAD) - accepted
    row('trial-fallback', host, tx, PAYLOAD[accepted:])
    assert len(a._inflight.payload) == (36 if long_cycle else 5)


@pytest.mark.parametrize('minimum', [1, 2])
def test_repeated_requests_respect_configured_minimum(minimum):
    host, seam = at_sl3()
    host.arq.cfg.min_sl = minimum
    for _ in range(26):
        control(host, 2)
    assert host.arq.speed_level == 2
    assert min(sl for sl, _, _ in seam.frames) == 2
    assert_pending(host)


@pytest.mark.parametrize('excluded', ['p2', 'entry', 'breakin', 'idle'])
def test_policy_only_changes_ordinary_loaded_p3_packets(excluded):
    host, _ = at_sl3()
    a = host.arq
    if excluded == 'p2':
        host.protocol = spec.Protocol.PACTOR2
    elif excluded == 'idle':
        a._inflight.payload = b''
    else:
        setattr(a._inflight, excluded, True)
    for _ in range(6):
        a._on_nak(False)
    assert a.speed_level == 3


@pytest.mark.parametrize('attempts', [1, 2, 9])
def test_maxtry_counts_total_keyed_attempts(attempts):
    host, seam = at_sl3()
    host.arq.cfg.p3_max_try = attempts
    for _ in range(attempts - 1):
        control(host, 2)
    assert [sl for sl, _, _ in seam.frames] == [2] + [3]*attempts
    control(host, 2)
    assert host.arq.speed_level == 2
    for _ in range(10):
        control(host, 2)
    assert host.arq.speed_level == 2 and host.arq._inflight.trial_from is None
    assert_pending(host)


def test_acknowledged_trial_establishes_speed_and_next_cs4_starts_new_trial():
    host, seam = at_sl3()
    control(host, 1)  # ACK the first SL3 packet, counter 2.
    assert host.arq.tx_seq == 3 and host.arq._inflight.trial_from is None
    for _ in range(8):
        control(host, 1)  # Request repeat of established counter 3 at SL3.
    assert host.arq.speed_level == 3
    control(host, 4)
    assert host.arq.speed_level == 4 and host.arq._inflight.trial_from == 3
    control(host, 2)
    control(host, 2)
    assert host.arq.speed_level == 3 and host.arq.tx_seq == 0


def test_refused_first_trial_packet_has_not_spent_an_attempt():
    host, seam = sender(enabled=False)
    control(host, 4)
    seam.refuse = True
    control(host, 4)
    for _ in range(6):
        control(host, 2)
    assert host.arq.speed_level == 3 and host.arq._inflight.sent_at_level == 0
    seam.refuse = False
    control(host, 2)
    control(host, 2)
    assert host.arq.speed_level == 3 and host.arq._inflight.sent_at_level == 2
    control(host, 2)
    assert host.arq.speed_level == 2
    assert_pending(host)
