# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.p3_arq_support import BODY, control, sender


def opening(*, option='sl3', policy='current', top=6, queued=BODY, long_cycle=False):
    host, seam = sender(enabled=False, payload=b'abc' + queued)
    a = host.arq
    a.cfg.p3_cs4_turn_open = option
    a.cfg.p3_speed_policy = policy
    a.cfg.max_sl = top
    a.cycle_long = long_cycle
    return host, seam


@pytest.mark.parametrize('policy', ['current', 'stock'])
@pytest.mark.parametrize('long_cycle', [False, True])
@pytest.mark.parametrize('top,target,fallback', [(6, 3, 2), (2, 2, 1)])
def test_cs4_opening_packet_uses_target_before_payload_is_cut(policy, long_cycle, top, target, fallback):
    host, seam = opening(policy=policy, long_cycle=long_cycle, top=top)
    a = host.arq
    wire = a._inflight.payload + bytes(a._outbuf)
    control(host, 4)
    packet = a._inflight
    assert seam.frames[-1] == (target, packet.status, wire[3:3 + a.ladder.payload(target, False)])
    assert packet.seq == 1 and packet.sl == target and packet.trial_from == fallback
    assert not a.cycle_long and a._established_sl == 1 and a._peer_sl == target
    assert a.tx_acked_bytes == 3
    assert a.tx_acked_bytes + len(packet.payload) + len(a._outbuf) == len(wire)


@pytest.mark.parametrize('policy', ['current', 'stock'])
@pytest.mark.parametrize('queue', [b'', b'xy'])
def test_delayed_or_small_opening_obeys_the_speed_policy(policy, queue):
    host, seam = opening(policy=policy, queued=queue)
    a = host.arq
    control(host, 4)
    if not queue:
        assert a._inflight is None
        a.on_host_data(BODY)
        host.tick()
    target, fallback = (3, 2) if policy == 'current' else (1, None)
    packet = a._inflight
    assert packet.sl == seam.frames[-1][0] == target and packet.seq == 1
    assert packet.trial_from == fallback
    assert len(packet.payload) == min(len(queue or BODY), a.ladder.payload(target, False))
    assert a._p3_open_fallback is None and a._p3_next_trial_from is None


@pytest.mark.parametrize('policy', ['current', 'stock'])
@pytest.mark.parametrize('delayed', [False, True])
@pytest.mark.parametrize('top', [2, 6])
def test_opening_maxtry_returns_one_rung_and_conserves_the_wire(policy, delayed, top):
    host, seam = opening(policy=policy, queued=b'' if delayed else BODY, top=top)
    a = host.arq
    control(host, 4)
    if delayed:
        a.on_host_data(BODY)
        host.tick()
    target = 1 if delayed and policy == 'stock' else min(3, top)
    assert a._inflight.sl == target
    if target == 1:
        assert a._inflight.trial_from is None
        return
    seq = a.tx_seq
    for _ in range(a.p3_max_try):
        control(host, 1)
    assert a._inflight.sl == seam.frames[-1][0] == target - 1 and a.tx_seq == seq == 1
    assert a._inflight.payload + bytes(a._outbuf) == BODY
    assert a._buffer_raw + a.tx_acked_bytes == len(BODY) + 3
    assert len(a._inflight.payload) == a.ladder.payload(target - 1, False)
    delivered = bytearray(b'abc')
    while a._inflight is not None and a._inflight.payload:
        packet = a._inflight
        delivered.extend(packet.payload)
        control(host, 1 + (packet.seq & 1))
    assert delivered == b'abc' + BODY
    assert not a._outbuf and a._buffer_raw == 0
    assert a.tx_acked_bytes == len(delivered)


@pytest.mark.parametrize('policy', ['current', 'stock'])
@pytest.mark.parametrize('delayed', [False, True])
def test_next_preserves_the_existing_opening(policy, delayed):
    host, seam = opening(option='next', policy=policy, queued=b'' if delayed else BODY)
    a = host.arq
    control(host, 4)
    if delayed:
        a.on_host_data(BODY)
        host.tick()
    target = 1 if delayed and policy == 'stock' else 2
    packet = a._inflight
    assert packet.sl == seam.frames[-1][0] == target
    assert packet.seq == 1 and packet.trial_from == (1 if target == 2 else None)
    assert packet.payload == BODY[:a.ladder.payload(target, False)]


@pytest.mark.parametrize('option', ['next', 'sl3'])
@pytest.mark.parametrize('top', [1, 2, 6])
def test_cs4_at_the_configured_top_repeats_without_credit(option, top):
    host, seam = opening(option=option, top=top)
    a = host.arq
    a._sl = a._peer_sl = top
    a._inflight = arq._Packet(1, b'field', top)
    credit = a.tx_acked_bytes
    control(host, 4)
    assert seam.frames[-1] == (top, 1, b'field')
    assert a.tx_seq == 1 and a.tx_acked_bytes == credit


@pytest.mark.parametrize('option', ['next', 'sl3'])
def test_second_cs4_keeps_the_mail_confirmation_candidate(option):
    host, seam = opening(option=option)
    host.p3_mail_active = True
    control(host, 4)
    packet = host.arq._inflight
    assert host.arq._mail_cs4_candidate is packet
    before = seam.frames[-1]
    credit = host.arq.tx_acked_bytes
    control(host, 4)
    assert host.arq._mail_cs4_wait == 0 and host.arq._inflight is packet
    assert seam.frames[-1] == before and host.arq.tx_acked_bytes == credit


@pytest.mark.parametrize('clear', ['take', 'give', 'disconnect'])
def test_delayed_fallback_cannot_escape_its_turn(clear):
    host, seam = opening(queued=b'')
    a = host.arq
    control(host, 4)
    assert a._p3_next_trial_from == a._p3_open_fallback == 2
    if clear == 'take':
        a._take_link()
    elif clear == 'give':
        a._give_link()
    else:
        a._finish_disconnected()
    assert a._p3_open_fallback is None and a._p3_next_trial_from is None


def test_turn_open_config_and_parser_defaults():
    assert arq.ArqConfig().p3_cs4_turn_open == 'next'
    assert arq.ArqConfig(p3_cs4_turn_open='sl3').p3_cs4_turn_open == 'sl3'
    with pytest.raises(ValueError, match='p3_cs4_turn_open'):
        arq.ArqConfig(p3_cs4_turn_open='sl6')
    parser = onair.build_parser()
    assert parser.parse_args([]).p3_cs4_turn_open == 'next'
    assert parser.parse_args(['--p3-cs4-turn-open', 'sl3']).p3_cs4_turn_open == 'sl3'


@pytest.mark.parametrize('option,target', [('next', 2), ('sl3', 3)])
def test_turn_open_through_the_packet_renderer(tmp_path, option, target, capsys):
    import json
    from hfmodem.tests.shrike.p3_reply_support import setup

    s, tx, live, grid = setup(tmp_path, 'B')
    tx.timing_trial = None
    a = s.host.arq
    a.entry_pending = False
    a.cfg.p3_cs4_turn_open = option
    a.on_host_data(b'abc' + BODY)
    a._inflight = arq._Packet(0, bytes(a._outbuf[:3]), 1, breakin=True)
    del a._outbuf[:3]
    a._next_seq = 1
    tx.aim(grid, 1)
    live.now = live.pos = tx.boundary - 5000
    a.on_rx_cs(arq.CS_SPEED_UP)
    assert not tx.refused
    packet = a._inflight
    assert packet.sl == target and packet.seq == 1 and packet.trial_from == target - 1
    assert packet.payload == BODY[:a.ladder.payload(target, False)]
    assert tx.slots_used == [0, 1]
    assert tx.tx_audio_start == 540000
    events = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('tx_event ')]
    waveform = [event for event in events if event['event'] == 'waveform'][-1]
    assert waveform['speed_level'] == target and waveform['status_byte'] & 3 == 1
    assert waveform['payload_bytes'] == len(packet.payload) and not waveform['long_cycle']


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2])
def test_turn_open_option_leaves_other_protocols_unchanged(protocol):
    def run(option):
        host, seam = opening(option=option)
        host.protocol = protocol
        host.arq.on_rx_cs(arq.CS_SPEED_UP)
        packet = host.arq._inflight
        return seam.frames, packet.sl, packet.status, packet.payload, packet.trial_from
    assert run('sl3') == run('next')


@pytest.mark.parametrize('parity', [False, True])
def test_turn_open_option_does_not_jump_on_a_midturn_cs4(parity):
    host, seam = opening()
    a = host.arq
    a.cfg.p3_cs4_parity = parity
    a._inflight = arq._Packet(1, b'abc', 1)
    a._next_seq = 2
    control(host, 4)
    assert a._inflight.sl == 2 and a._inflight.trial_from == 1
    assert seam.frames[-1][0] == 2


@pytest.mark.parametrize('level', [3, 4, 5])
def test_turn_open_at_sl3_or_above_uses_next_level(level):
    host, seam = opening()
    a = host.arq
    a._sl = a._peer_sl = a._established_sl = level
    a._inflight = arq._Packet(0, b'abc', level, breakin=True)
    control(host, 4)
    assert a._inflight.sl == level + 1
    assert a._inflight.trial_from == level
    assert a._p3_open_fallback is None
    assert seam.frames[-1][0] == level + 1
