# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from collections import deque

import pytest

from hfmodem.ax25.connected import Connection, State, callsign
from hfmodem.ax25.frame import Frame, address, parse


def raw(src, dst, control, info=b'', *, command=True, via=()):
    calls = (dst, src, *via)
    return Frame(tuple(address(c, last=i == len(calls) - 1,
                               flag=command if i == 0 else not command if i == 1 else True)
                       for i, c in enumerate(calls)), control,
                 0xF0 if control & 1 == 0 else None, info).encode()


class Air:
    def __init__(self):
        self.now = 0.
        self.queue = deque()
        self.a_rx, self.b_rx, self.credit = [], [], []
        self.events = []
        self.a = Connection('W9SSJ', lambda f: self.send('b', f), clock=lambda: self.now,
                            receive=self.a_rx.append, confirmed=self.credit.append,
                            state_changed=lambda *e: self.events.append(e))
        self.b = Connection('N0CALL-1', lambda f: self.send('a', f), clock=lambda: self.now,
                            receive=self.b_rx.append)

    def send(self, side, frame):
        self.queue.append((side, frame))
        return True

    def step(self, *, drop=lambda _: False, copies=1):
        self.a.tick()
        self.b.tick()
        while self.queue:
            side, frame = self.queue.popleft()
            if not drop(frame):
                for _ in range(copies):
                    getattr(self, side).on_frame(frame)
        self.now += .1

    def connected(self):
        self.a.connect('N0CALL-1')
        for _ in range(3):
            self.step()
        assert self.a.state is self.b.state is State.CONNECTED


def test_sabm_wire_and_peer_ua_required():
    air = Air()
    air.a.connect('N0CALL-1')
    air.a.tick()
    frame = parse(air.queue[0][1])
    assert frame.control == 0x3F
    assert callsign(frame.addresses[0]) == 'N0CALL-1'
    assert frame.addresses[0][6] & 128 and not frame.addresses[1][6] & 128
    assert air.a.state is State.CONNECTING
    air.a.on_frame(raw('N0CALL-1', 'W9SSJ', 0x63, command=False))
    assert air.a.state is State.CONNECTING     # missing final
    air.a.on_frame(raw('N0CALL-1', 'W9SSJ', 0x73, command=False))
    assert air.a.state is State.CONNECTED


def test_binary_window_wrap_repeated_rf_copies_and_exact_credit():
    air = Air()
    air.connected()
    payload = bytes(range(256)) * 7 + b'final'
    air.a.write(payload)
    for _ in range(300):
        air.step(copies=3)
        if not air.a.pending_bytes:
            break
    assert b''.join(air.b_rx) == payload
    assert b''.join(air.credit) == payload
    assert air.a.accepted == air.a.delivered == len(payload)
    assert air.a.discarded == 0


def test_lost_ack_retries_without_duplicate_host_delivery():
    air = Air()
    air.connected()
    payload = b'a short final field'
    air.a.write(payload)
    for _ in range(120):
        air.step(drop=lambda f: parse(f).control & 3 == 1)
    assert air.a.pending_bytes == len(payload)
    assert b''.join(air.b_rx) == payload
    for _ in range(120):
        air.step()
        if not air.a.pending_bytes:
            break
    assert b''.join(air.b_rx) == b''.join(air.credit) == payload


def test_graceful_close_waits_for_last_ack_and_refuses_late_write():
    air = Air()
    air.connected()
    air.a.write(b'last')
    air.a.disconnect()
    assert air.a.state is State.CONNECTED
    with pytest.raises(ValueError):
        air.a.write(b'late')
    for _ in range(20):
        air.step()
    assert air.a.state is air.b.state is State.DISCONNECTED
    assert b''.join(air.b_rx) == b''.join(air.credit) == b'last'


def test_abort_retains_originals_and_does_not_credit():
    air = Air()
    air.connected()
    payload = bytes(range(256))
    air.a.write(payload)
    air.a.tick()
    assert len(air.a.offered) == len(air.a.pending) == 1
    air.a.abort()
    assert air.a.last_unconfirmed == payload
    assert air.a.accepted == air.a.discarded == 256
    assert air.a.delivered == 0 and not air.credit
    air.a.on_frame(raw('N0CALL-1', 'W9SSJ', 0x41, command=False))
    assert air.a.delivered == 0


def test_rnr_stops_window_and_rr_resumes():
    air = Air()
    air.connected()
    air.b.set_busy(True)
    air.step()
    air.a.write(b'x' * 300)
    air.step()
    assert air.a.peer_busy and len(air.a.pending) == 3
    air.b.set_busy(False)
    for _ in range(100):
        air.step()
    assert b''.join(air.b_rx) == b'x' * 300
    assert not air.a.pending_bytes


def test_rej_restarts_unacknowledged_window_after_lost_i():
    air = Air()
    air.connected()
    air.a.ipoll = 0
    air.a.write(b'a' * 128 + b'b' * 128 + b'c' * 15)
    air.step(drop=lambda f: parse(f).info == b'a' * 128)
    for _ in range(100):
        air.step()
    assert b''.join(air.b_rx) == b'a' * 128 + b'b' * 128 + b'c' * 15
    assert not air.a.pending_bytes


def test_invalid_ack_never_credits_payload():
    air = Air()
    air.connected()
    air.a.write(b'x')
    air.a.tick()
    air.a.on_frame(raw('N0CALL-1', 'W9SSJ', 0xE1, command=False))
    assert air.a.state is State.DISCONNECTED
    assert air.a.last_unconfirmed == b'x'
    assert not air.credit


def test_refused_transmission_does_not_start_timer_or_take_payload():
    now, frames = [0.], []
    c = Connection('W9SSJ', lambda f: frames.append(f) and True, clock=lambda: now[0])
    c.connect('N0CALL')
    for _ in range(100):
        now[0] += 10
        c.tick()
    assert c.state is State.CONNECTING
    assert c.retries == 0 and c._deadline is None


def test_retry_exhaustion_and_next_contact_has_fresh_sequence():
    air = Air()
    air.a.retry_limit = 1
    air.a.rtt = .1
    air.a.connect('N0CALL-1')
    for _ in range(20):
        air.step(drop=lambda _: True)
    assert air.a.state is State.DISCONNECTED
    air.connected()
    assert air.a.vs == air.a.vr == 0


def test_via_addressing_does_not_operate_as_repeater():
    air = Air()
    air.a.connect('N0CALL-1', ('N0VIA',))
    air.a.tick()
    frame = parse(air.queue[0][1])
    assert callsign(frame.addresses[2]) == 'N0VIA'
    assert not frame.addresses[2][6] & 128
    air.b.on_frame(frame.encode())
    assert air.b.state is State.DISCONNECTED
    addresses = (*frame.addresses[:2], frame.addresses[2][:6] + bytes([frame.addresses[2][6] | 128]))
    air.b.on_frame(Frame(addresses, frame.control, None, b'').encode())
    assert air.b.state is State.CONNECTED
    air.b.tick()
    reply = parse(air.queue[-1][1])
    assert callsign(reply.addresses[2]) == 'N0VIA'


def test_listen_off_and_extended_sabme_refused():
    air = Air()
    air.b.listen = False
    air.b.on_frame(raw('W9SSJ', 'N0CALL-1', 0x3F))
    air.b.tick()
    assert parse(air.queue[-1][1]).control == 0x1F
    assert air.b.state is State.DISCONNECTED
    air.b.listen = True
    air.b.on_frame(raw('W9SSJ', 'N0CALL-1', 0x7F))
    air.b.tick()
    assert parse(air.queue[-1][1]).control == 0x1F


def test_duplicate_sabm_recovers_lost_ua_while_host_event_remains_unread():
    tx, events = [], []
    c = Connection('W9SSJ', lambda f: tx.append(f) or True,
                   state_changed=lambda *args: events.append(args),
                   can_accept=lambda: not events)
    setup = raw('N0CALL', 'W9SSJ', 0x3F)
    c.on_frame(setup)
    c.tick()
    c.on_frame(setup)
    c.tick()
    assert len(events) == 1
    assert [parse(f).control for f in tx] == [0x73, 0x73]
    assert c.state is State.CONNECTED


def test_peer_reset_keeps_retirement_provenance_after_new_link_setup():
    c = Connection('W9SSJ', lambda _: True)
    setup = raw('N0CALL', 'W9SSJ', 0x3F)
    c.on_frame(setup)
    c.write(b'unconfirmed')
    c.on_frame(setup)
    assert c.state is State.DISCONNECTED
    c.on_frame(setup)       # a new host boundary may accept a later setup
    assert c.state is State.CONNECTED
    assert c.retired[-1]['unconfirmed'] == b'unconfirmed'
    assert c.retired[-1]['confirmed'] == 0
