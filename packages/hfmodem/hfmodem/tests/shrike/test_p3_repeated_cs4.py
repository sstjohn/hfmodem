# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.p3_arq_support import control, pending, remaining
from hfmodem.tests.evidence import require_input

RECORDINGS = json.loads(require_input(Path(__file__).with_name('fixtures') / 'p3_repeated_cs4_0924.json',
                                       allow_module_level=True).read_text())


def from_recording(recording):
    packets = [e for e in recording['events'] if e['kind'] == 'packet']
    first = packets[0]
    payloads = [bytes.fromhex(e['payload_hex']) for e in packets]
    host, seam = pending(first['sl'], first['seq'], payloads[0],
                         b''.join(payloads[1:]), long_cycle=first['cycle'] == 'long')
    return host, seam, packets, payloads


def test_recorded_sl6_tail_is_resent_and_delivered_without_loss():
    recording = RECORDINGS[0]
    host, seam, packets, payloads = from_recording(recording)
    a = host.arq
    words = [e['cs'] for e in recording['events'] if e['kind'] == 'control']
    assert words == [4, 4, 1, 5, 5, 5, 5, 5, 5, 5, 3]
    control(host, words[0])
    assert seam.frames[-1] == (6, packets[1]['seq'], payloads[1])
    tail = seam.frames[-1]
    control(host, words[1])
    assert seam.frames[-1] == tail
    assert len(tail[2]) == 124
    for word in words[2:-1]:
        control(host, word)
        assert a.tx_acked_bytes == len(payloads[0])
        assert remaining(a) == payloads[1]
    received = payloads[0]
    for _ in range(40):
        if not remaining(a):
            break
        received += a._inflight.payload
        control(host, 1 + a.tx_seq % 2)
    assert received == b''.join(payloads)
    assert not remaining(a)
    assert a.tx_acked_bytes == len(received)


def test_recorded_long_sl2_cs4_keeps_the_lower_level_acknowledgement():
    host, seam, packets, payloads = from_recording(RECORDINGS[1])
    words = [e['cs'] for e in RECORDINGS[1]['events'] if e['kind'] == 'control']
    assert words == [4, 4] + [1] * 10
    assert [len(p) for p in payloads[:3]] == [36, 116, 94]
    control(host, words[0])
    assert seam.frames[-1] == (2, 3, payloads[1])
    control(host, words[1])
    assert seam.frames[-1] == (3, 0, payloads[2])
    assert host.arq.tx_acked_bytes == 36 + 116
    assert remaining(host.arq) == payloads[2]


@pytest.mark.parametrize('seq', range(4))
@pytest.mark.parametrize('policy', ['current', 'stock'])
def test_sl6_repeated_cs4_preserves_counter_and_bytes(seq, policy):
    host, seam = pending(5, seq, b'accepted', bytes(range(256)) * 2)
    a = host.arq
    a.cfg.p3_speed_policy = policy
    control(host, 4)
    frame = seam.frames[-1]
    queued = remaining(a)
    control(host, 4)
    assert a.tx_seq == (seq + 1) % 4
    assert a.tx_acked_bytes == len(b'accepted')
    assert remaining(a) == queued
    assert seam.frames[-1][1] == frame[1]
    assert queued.startswith(seam.frames[-1][2])
    assert a.speed_level == (5 if policy == 'stock' else 6)


def test_ack_mode_restores_the_historical_sl6_advance():
    host, seam, packets, payloads = from_recording(RECORDINGS[0])
    a = host.arq
    a.cfg.p3_repeated_cs4 = 'ack'
    control(host, 4)
    control(host, 4)
    assert a._inflight is None
    assert not remaining(a)
    assert a.tx_acked_bytes == len(payloads[0]) + 124


@pytest.mark.parametrize('sl', range(1, 5))
def test_consecutive_cs4_below_sl6_can_acknowledge_and_climb(sl):
    host, seam = pending(sl, 0, b'accepted', bytes(range(256)) * 4)
    control(host, 4)
    second = seam.frames[-1]
    control(host, 4)
    assert seam.frames[-1][:2] == (sl + 2, 2)
    assert host.arq.tx_acked_bytes == len(b'accepted') + len(second[2])


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2])
def test_top_cs4_policy_does_not_change_other_protocols(protocol):
    host, seam = pending(1, 0, b'accepted', b'pending')
    host.protocol = protocol
    host.arq.cfg.max_sl = 1
    host.arq.on_rx_cs(arq.CS_SPEED_UP)
    assert host.arq.tx_acked_bytes == len(b'accepted')


def test_repeated_cs4_configuration_defaults_to_repeat_and_rejects_unknown_values():
    assert arq.ArqConfig().p3_repeated_cs4 == 'repeat'
    assert arq.ArqConfig(p3_repeated_cs4='ack').p3_repeated_cs4 == 'ack'
    with pytest.raises(ValueError, match='p3_repeated_cs4'):
        arq.ArqConfig(p3_repeated_cs4='ignore')
