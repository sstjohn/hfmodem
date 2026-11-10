# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS 1.7 R/@R and AX.25 H-bit forwarding, without host delivery credit."""
from collections import deque

import pytest

from hfmodem.ax25.connected import Connection, State
from hfmodem.ax25.frame import Frame, address, parse
from hfmodem.brubru.host.runtime import PacketRuntime
from .test_connected_hosts import Link, wait
from .test_tracker import setup


def routed(path=('NOCALL',), *, flags=(), info=bytes(range(256))):
    calls = ('TARGET', 'SOURCE', *path)
    return Frame(tuple(address(call, last=i == len(calls)-1,
                               flag=i == 0 or i >= 2 and i-2 in flags)
                       for i, call in enumerate(calls)), 3, 0xf0, info).encode()


def test_required_express_r1_repeats_only_the_next_hop_and_changes_only_its_h_bit():
    host, engine, tx, _ = setup()
    host.command('R1')
    assert host.command('R') == '1'
    raw = routed(('FIRST', 'NOCALL', 'LAST'), flags=(0,))
    engine.on_frame(raw)
    engine.tick()
    expected = bytearray(raw)
    expected[27] |= 0x80
    assert tx == [bytes(expected)]
    assert engine.state is State.DISCONNECTED
    assert engine.accepted == engine.delivered == engine.discarded == 0
    assert not host.events and not host.rx


def test_alias_disabled_identity_path_order_and_previously_repeated_frames():
    host, engine, tx, _ = setup()
    host.command('R1')
    assert host.command('@R') == 'SCSTNC'
    host.command('@R RELAY-3')
    for raw in (routed(('FIRST', 'NOCALL')), routed(flags=(0,)),
                routed(('NOCALL', 'LAST'), flags=(1,)),
                routed(('OTHER',)), b'bad', routed(info=b'x'*257)):
        engine.on_frame(raw)
    engine.tick()
    assert not tx
    raw = routed(('RELAY-3',))
    engine.on_frame(raw)
    engine.tick()
    assert len(tx) == 1 and parse(tx[0]).addresses[2][6] & 128
    host.command('IXXXXX')
    engine.on_frame(raw)
    engine.tick()
    host.command('INOCALL')
    host.command('Y0')
    engine.on_frame(raw)
    engine.tick()
    assert len(tx) == 1


def test_refused_output_is_bounded_and_retained_without_false_credit():
    host, engine, tx, _ = setup()
    engine.transmit = lambda raw: False
    host.command('R1')
    raw = routed(info=b'pending')
    engine.on_frame(raw)
    engine.on_frame(raw)  # same frame already waiting, not a second admission
    for i in range(40):
        engine.on_frame(routed(info=bytes([i])))
    engine.tick()
    assert len(engine.repeats) == 32
    assert engine.repeats[0][20] & 128
    assert engine.delivered == engine.accepted == 0 and not tx
    engine.transmit = lambda raw: tx.append(raw) or True
    engine.tick()
    assert len(tx) == 1 and len(engine.repeats) == 31
    host.command('R0')
    engine.tick()
    assert not engine.repeats and len(tx) == 1
    host.command('R1')
    engine.on_frame(raw)
    host.attachment_lost()
    assert not engine.repeat_enabled and not engine.repeats


def test_listen_inhibition_discards_queued_forwarding_before_transmit():
    host, engine, tx, _ = setup()
    host.command('R1')
    engine.on_frame(routed())
    engine.listen = False
    engine.tick()
    assert not tx and not engine.repeats


def test_forwarded_frames_do_not_starve_contact_or_change_peer_acknowledgements():
    host, engine, tx, _ = setup()
    host.command('R1')
    engine.on_frame(routed())
    engine.connect('REMOTE')
    engine.tick()
    assert parse(tx[0]).control == 0x3f  # establishment before unrelated forwarding
    engine.tick()
    assert parse(tx[1]).control == 3
    assert engine._deadline == 2*engine.rtt + engine._last_activity
    assert engine.accepted == engine.delivered == 0
    with pytest.raises(ValueError):
        host.command('@R CHANGED')
    assert host.command('@R') == 'SCSTNC'


def test_runtime_uses_owned_lane_completion_and_detach_retires_forwarding():
    link = Link()
    runtime = PacketRuntime(link, mycall='NOCALL')
    runtime.start()
    try:
        attachment = runtime.attach('tracker')
        assert not runtime.engine.repeat_enabled
        assert b'INVALID' not in runtime.feed(attachment, b'\x1bR1\r\x1bY1\r')
        runtime.send_frame(routed())
        wait(lambda: len(link.requests) == 1)
        assert runtime.active and not runtime.engine.active
        assert runtime.engine.accepted == runtime.engine.delivered == 0
        runtime.send_frame(routed(info=b'queued'))
        wait(lambda: bool(runtime.engine.repeats))
        runtime.detach(attachment)
        assert link.requests[0].state.value == 'cancelled'
        assert not runtime.engine.repeats and not runtime.engine.repeat_enabled
        link.complete()
        wait(lambda: runtime._transmission is None)
        replacement = runtime.attach('tracker')
        runtime.feed(replacement, b'\x1bY1\r')
        runtime.send_frame(routed())
        runtime._call(lambda: None)
        assert len(link.requests) == 1 and not runtime.engine.repeat_enabled
    finally:
        runtime.stop()


def test_complete_contact_binary_retransmission_and_close_through_explicit_repeater():
    host, relay, _, _ = setup()
    host.command('R1')
    air, received, credit = deque(), [], []
    now = [0.]

    def send(owner, frame):
        air.append((owner, frame))
        return True

    a = Connection('CALLER', lambda raw: send('a', raw), clock=lambda: now[0], confirmed=credit.append)
    b = Connection('TARGET', lambda raw: send('b', raw), clock=lambda: now[0], receive=received.append)
    relay.transmit = lambda raw: send('relay', raw)
    relay.clock = lambda: now[0]
    nodes = {'a': a, 'b': b, 'relay': relay}
    dropped = [False]

    def step():
        for node in nodes.values():
            node.tick()
        while air:
            owner, raw = air.popleft()
            frame = parse(raw)
            # Lose one end-to-end data acknowledgement after it was repeated.
            if (owner == 'relay' and not dropped[0] and b.vr
                    and frame.control & 3 == 1 and frame.addresses[1][:6] == address('TARGET')[:6]):
                dropped[0] = True
                continue
            for key, node in nodes.items():
                if key != owner:
                    node.on_frame(raw)
        now[0] += .1

    a.connect('TARGET', ('NOCALL',))
    for _ in range(20):
        step()
    assert a.state is b.state is State.CONNECTED
    original = bytes(range(256))*2 + b'last'
    a.write(original)
    for _ in range(500):
        step()
        if a.delivered == len(original):
            break
    assert dropped[0] and b''.join(received) == b''.join(credit) == original
    assert relay.accepted == relay.delivered == relay.discarded == 0
    assert not host.events and not host.rx
    a.disconnect()
    for _ in range(30):
        step()
    assert a.state is b.state is State.DISCONNECTED
