# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Pre-release design counterexamples: delivery, addressing, negotiation, integrity."""
import numpy as np
import pytest
from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqFsm, ArqConfig, SessionState
from hfmodem.sabir import compress

class IO:
    def __init__(self): self.controls=[]; self.datas=[]; self.delivered=bytearray()
    def send_control(self,c): self.controls.append(c); return 4.5
    def send_data(self,*args): self.datas.append(args); return 8.
    def deliver(self,data): self.delivered.extend(data)
    def connected(self,*args): pass
    def disconnected(self): pass
    def log(self,*args): pass
    def state_changed(self,*args): pass

def endpoint(name, **cfg):
    io=IO()
    return io, ArqFsm(io,ArqConfig(callsign=name, **cfg),clock=lambda:0.)

def receive(dst,args):
    seq,gear,present,ncw,nibbles,coded,offset=args
    c=wire.Control(wire.DATA,dst.session,seq,gear,wire.cw_mask(present),wire.data_aux(1,ncw,nibbles),offset)
    dst.on_data(wire.Control.unpack(c.pack()),(1.-2.*coded.astype(float))*30,None)

def pair(**cfg):
    ai,a=endpoint('ALICE', **cfg); bi,b=endpoint('BOB')
    b.on_host_listen(True); a.on_host_connect('BOB')
    b.on_control(wire.Control.unpack(ai.controls[-1].pack()))
    a.on_control(wire.Control.unpack(bi.controls[-1].pack()))
    return ai,a,bi,b

@pytest.mark.parametrize('size', [5, 1000, 8000])
def test_lost_acks_rebuild_and_resegment_without_duplicate_delivery(size):
    ai,a,bi,b=pair(start_rung=4)
    payload=np.random.default_rng(size).bytes(size)
    a.on_host_data(payload)
    receive(b,ai.datas[-1])  # receiver commits, all feedback erased
    for _ in range(2):
        a.on_timer(a.next_deadline())
        receive(b,ai.datas[-1])
    assert a.stats['rebuilds'] == 1
    for _ in range(100):
        if a._frame is None: break
        a.on_control(bi.controls[-1])
        if a._frame is not None: receive(b,ai.datas[-1])
    assert a._frame is None and bi.delivered == payload


def test_explicit_destination_and_fresh_session():
    ai,a=endpoint('ALICE'); _,b=endpoint('NODE450'); b.on_host_listen(True)
    a.on_host_connect('BOB'); offer=ai.controls[-1]
    b.on_control(offer)
    assert b.state == SessionState.LISTENING
    _,other=endpoint('ALICE'); other.on_host_connect('BOB')
    assert other.session != a.session
    assert offer.destination == 'BOB'



def test_integrity_is_required_before_any_negotiation():
    raw=compress.pack(b'hello')
    assert list(compress.Unpacker().feed(raw)) == [b'hello']
    with pytest.raises(ValueError): compress.pack(b'hello', integrity=False)
    with pytest.raises(ValueError): list(compress.Unpacker().feed(b'\0\0\0\5hello'))


def test_wide_counters_and_offset_roundtrip():
    c=wire.Control(wire.DATA,2**63+1,seq=2**31+1,gear=65535|wire.HANDOVER,
                   mask=wire.cw_mask([0]),aux=wire.data_aux(1,1,(0,)*8),offset=2**63+9)
    assert wire.Control.unpack(c.pack()) == c


def test_previous_session_disc_and_ack_cannot_affect_current_session():
    ai,a,bi,b=pair()
    a._last_session = a.session ^ 123
    a.on_host_data(b'current')
    frame = a._frame
    a.on_control(wire.Control(wire.ACK,a._last_session,seq=frame.seq,
                              mask=wire.cw_mask(range(len(frame.chunks)))))
    assert a._frame is frame
    a.on_control(wire.Control.ident(wire.DISC,a._last_session,'BOB'))
    assert a.state == SessionState.CONNECTED and a._frame is frame




def test_changed_reoffer_cannot_renegotiate_active_session():
    from dataclasses import replace
    ai,a,bi,b=pair()
    original=b.peer_capabilities
    offer=ai.controls[0]
    modified=replace(offer, mask=wire.capabilities(range(3, 2 + 1)).to_bytes(4,'big')+offer.mask[4:])
    b.on_control(modified)
    assert b.peer_capabilities == original


def test_counter_exhaustion_closes_without_reusing_generation():
    ai,a,bi,b=pair()
    a._next_seq=2**32
    a.on_host_data(b'no wrap')
    assert a.state != SessionState.CONNECTED and not ai.datas


def test_partial_codeword_ack_does_not_report_stream_commit():
    ai,a,bi,b=pair(start_rung=1)
    a.on_host_data(bytes(200))
    f=a._frame
    a.on_control(wire.Control(wire.ACK,a.session,seq=f.seq,mask=wire.cw_mask([1])))
    assert a.tx_pending == 200


def test_retired_connect_is_not_accepted_again():
    ai,a,bi,b=pair()
    original=ai.controls[0]
    b.on_control(wire.Control.ident(wire.DISC,b.session,'ALICE'))
    assert b.state == SessionState.LISTENING
    b.on_control(original)
    assert b.state == SessionState.LISTENING






def test_lost_connect_ack_recovers_without_resetting_receiver_stream():
    ai, a = endpoint('ALICE')
    bi, b = endpoint('BOB')
    b.on_host_listen(True)
    a.on_host_connect('BOB')
    b.on_control(ai.controls[-1])
    assert b.state == SessionState.CONNECTED
    # Erase the first CONNECT_ACK. The caller retries its identical offer.
    a.on_timer(a.next_deadline())
    b.on_control(ai.controls[-1])
    a.on_control(bi.controls[-1])
    a.on_host_data(b'after handshake loss')
    receive(b, ai.datas[-1])
    a.on_control(bi.controls[-1])
    assert bi.delivered == b'after handshake loss' and a.tx_pending == 0


def test_lost_disconnect_ack_is_answered_after_receiver_teardown():
    ai, a, bi, b = pair()
    a.on_host_disconnect()
    b.on_control(ai.controls[-1])
    assert b.state == SessionState.LISTENING
    # Erase DISC_ACK, then deliver the repeated DISC and courtesy reply.
    a.on_timer(a.next_deadline())
    b.on_control(ai.controls[-1])
    assert bi.controls[-1].type == wire.DISC_ACK
    a.on_control(bi.controls[-1])
    assert a.state == SessionState.DISCONNECTED


def test_receiver_disconnect_waits_for_turn_and_completes():
    ai, a, bi, b = pair()
    b.on_host_disconnect()
    assert b.state == SessionState.CONNECTED and b._disc_pending
    assert bi.controls[-1].type != wire.DISC
    b.on_timer(b.next_deadline())
    assert bi.controls[-1].type == wire.TURN_REQ
    a.on_control(bi.controls[-1])
    assert ai.controls[-1].type == wire.TURN
    b.on_control(ai.controls[-1])
    assert bi.controls[-1].type == wire.DISC
    a.on_control(bi.controls[-1])
    b.on_control(ai.controls[-1])
    assert a.state == SessionState.DISCONNECTED and b.state == SessionState.LISTENING


def test_dropped_turn_grant_recovers_queued_reverse_traffic():
    ai, a, bi, b = pair()
    b.on_host_data(b'reverse')
    b.on_timer(b.next_deadline())
    a.on_control(bi.controls[-1])
    assert ai.controls[-1].type == wire.TURN
    # Erase the grant; requester retries while original sender awaits DATA.
    b.on_timer(b.next_deadline())
    a.on_control(bi.controls[-1])
    a.on_timer(a.next_deadline())
    assert ai.controls[-1].type == wire.TURN
    b.on_control(ai.controls[-1])
    receive(a, bi.datas[-1])
    b.on_control(ai.controls[-1])
    assert ai.delivered == b'reverse' and b.tx_pending == 0


def test_unanswered_turn_requests_fail_with_bounded_retries():
    from hfmodem.sabir.arq.fsm import DISC_LINK_FAILED
    _, _, bi, b = pair()
    b.on_host_data(b'cannot be sent')
    for _ in range(b.cfg.max_turn_reqs + 1):
        deadline = b.next_deadline()
        assert deadline is not None
        b.on_timer(deadline)
    assert b.state == SessionState.LISTENING
    assert b.disc_reason == DISC_LINK_FAILED and b.tx_pending == 0
    assert sum(c.type == wire.TURN_REQ for c in bi.controls) == b.cfg.max_turn_reqs


@pytest.mark.parametrize('gear, offset', [(65535, 0), (4, 1)])
def test_invalid_data_does_not_cancel_pending_turn_request(gear, offset):
    _, _, _, b = pair()
    b.on_host_data(b'waiting')
    deadline = b.next_deadline()
    bad = wire.Control(wire.DATA, b.session, seq=1, gear=gear,
                       mask=wire.cw_mask([0]), aux=wire.data_aux(1, 1, (0,) * 8),
                       offset=offset)
    b.on_data(bad, None, None)
    assert b.next_deadline() == deadline


def test_data_codeword_padding_and_capacity_are_checked():
    from hfmodem.sabir.frame.codec import FrameCodec, crc16
    codec = FrameCodec()
    with pytest.raises(ValueError):
        codec.encode_cw(bytes(codec.data_bytes + 1))
    body = b'\x01X' + b'\x01' + bytes(codec.info_bytes - 5)
    block = body + crc16(body).to_bytes(2, 'big')
    info = np.unpackbits(np.frombuffer(block, dtype=np.uint8))
    coded = (codec.code.encode(info) ^ codec.pn)[codec.perm]
    decoded, _ = codec.decode_cws((1.0 - 2.0 * coded) * 30)
    assert decoded == [None]


def test_session_success_requires_teardown_not_only_delivery():
    from hfmodem.sabir.sim.m3 import run_pair
    session = run_pair(b'exact but still connected', None, 30, seed=3, max_exchanges=2)
    assert session.byte_exact and not session.ok
    assert 'CONNECTED' in session.terminal_states or 'DISCONNECTING' in session.terminal_states


def test_asymmetric_sample_exchange_and_overhead_accounting():
    from hfmodem.sabir.sim.m3 import run_pair
    payload = np.random.default_rng(40).bytes(512)
    session = run_pair(payload, ('good', None), (14.0, -4.0), seed=41,
                       cfg_kw={'fast_ctrl': False},
                       drop_bursts=frozenset({('b', 2), ('b', 3)}))
    assert session.ok and session.delivered == payload
    assert session.stats_a['timeouts'] >= 2 and session.stats_a['rebuilds'] >= 1
    accounted = sum(session.airtime.values()) + session.timing['turnaround_s'] + session.timing['timer_idle_s']
    assert session.wall_s == pytest.approx(accounted)
    assert session.timing['timer_idle_s'] > 0


def test_idle_sessions_expire_without_transmitting_forever():
    from hfmodem.sabir.arq.fsm import DISC_LINK_FAILED
    _, a, _, b = pair(inactivity_timeout_s=180.)
    for f in (a, b):
        assert f.next_deadline() == 180.
        f.on_timer(179.)
        assert f.state == SessionState.CONNECTED
        f.on_timer(180.)
        assert f.state != SessionState.CONNECTED and f.disc_reason == DISC_LINK_FAILED
        assert f.next_deadline() is None


def test_noise_unrelated_stale_and_invalid_data_do_not_renew_idle_lease():
    _, a, _, b = pair()
    now = [20.]
    a._clock = b._clock = lambda: now[0]
    deadline = b.next_deadline()
    b.on_control(wire.Control.ident(wire.ID, b.session ^ 1, 'ALICE'))
    b.on_control(wire.Control.ident(wire.ID, b.session, 'OTHER'))
    b.on_control(wire.Control(wire.ACK, b.session, seq=99))
    b.on_data(wire.Control(wire.DATA, b.session, seq=1, gear=65535,
                          aux=wire.data_aux(1, 1, (0,) * 8)), None, None)
    assert b.next_deadline() == deadline
    b.on_control(wire.Control.ident(wire.ID, b.session, 'ALICE'))
    assert b.next_deadline() == 200.
    b._peer_generation = 5
    now[0] = 40.
    stale = wire.Control(wire.DATA, b.session, seq=4, gear=4,
                         aux=wire.data_aux(1, 1, (0,) * 8))
    assert not b.on_data_header(stale, 500.)
    assert b.next_deadline() == 200.


def test_valid_long_body_lease_and_local_transmit_hold():
    _, a, _, b = pair()
    header = wire.Control(wire.DATA, b.session, seq=1, gear=4,
                          mask=wire.cw_mask([0]), aux=wire.data_aux(1, 1, (0,) * 8))
    assert b.on_data_header(header, 240.)
    assert b.next_deadline() > 240.
    b.on_timer(180.)
    assert b.state == SessionState.CONNECTED
    a._tx_busy_until = 250.
    a.on_timer(180.)
    assert a.state == SessionState.CONNECTED
    a.on_timer(250.)
    assert a.state == SessionState.DISCONNECTED


@pytest.mark.parametrize('direction', ['a', 'b'])
def test_complete_one_direction_outage_eventually_closes_both_ends(direction):
    from hfmodem.sabir.sim.m3 import run_pair
    erased = frozenset((direction, i) for i in range(2, 100))
    session = run_pair(b'outage test', None, 30., seed=808,
                       drop_bursts=erased, max_exchanges=100)
    assert session.terminal_states == ('DISCONNECTED', 'LISTENING')
    assert session.wall_s < 1000.


@pytest.mark.parametrize('timeout', [0., -1., float('inf'), float('nan')])
def test_inactivity_timeout_is_positive_and_finite(timeout):
    with pytest.raises(ValueError):
        endpoint('ALICE', inactivity_timeout_s=timeout)
