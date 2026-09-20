# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Observed P3 cycles, emitted cycle commands, and bounded unplaced teardown."""
import pytest

from hfmodem.shrike import arq, spec
from hfmodem.shrike.ptc import PtcHost, SimPeer


class IO(arq.ArqIO):
    def __init__(self, *, deferred=True):
        self.deferred = deferred
        self.refuse_cs = self.refuse_packet = False
        self.controls, self.packets, self.logs = [], [], []
        self.delivered = bytearray()
        self.closed = 0

    def send_cs(self, cs_index):
        self.controls.append(cs_index)
        return arq.REFUSED if self.refuse_cs else None

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((sl, bytes(payload), status, breakin))
        return arq.REFUSED if self.refuse_packet else len(payload)

    def defer_rx_close(self):
        return self.deferred

    def deliver(self, payload):
        self.delivered.extend(payload)

    def disconnected(self):
        self.closed += 1

    def log(self, text):
        self.logs.append(text)


def linked(*, role=arq.IRS, deferred=True, long_cycle=True):
    io = IO(deferred=deferred)
    fsm = arq.PactorArq(io, arq.ArqConfig(long_cycle=long_cycle,
                                        speed_up_after=1000))
    fsm._enter_connected()
    fsm.role = role
    return fsm, io


def packet(fsm, *, long=False, request=True, seq=1, data=b" Trim", crc=True,
           sl=3):
    # Exercise command bookkeeping at SL3 by default; SL1 can now grant too.
    status = spec.status_byte(seq, data_type=spec.DataType.ASCII_8BIT,
                              long_cycle_request=request)
    fsm.on_rx_packet(sl, data, status, crc, protocol=spec.Protocol.PACTOR3,
                     cycle_long=long)


def test_floor_rung_can_grant_long_but_waits_for_its_emission():
    fsm, io = linked()
    packet(fsm, sl=1)
    assert io.controls == [arq.CS_CYCLE_TOG]
    assert fsm.cycle_request is True
    assert not fsm.cycle_command_emitted and not fsm.cycle_long


def test_queued_cs6_does_not_change_observed_cycle():
    fsm, io = linked()
    packet(fsm)
    assert io.controls == [arq.CS_CYCLE_TOG]
    assert fsm.cycle_request is True
    assert not fsm.cycle_command_emitted and not fsm.cycle_long
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    assert fsm.cycle_command_emitted and not fsm.cycle_long


@pytest.mark.parametrize("refused", [False, True])
def test_lost_or_refused_cs6_yields_to_normal_ack_on_short_repeats(refused):
    fsm, io = linked(deferred=False)
    io.refuse_cs = refused
    for n in range(35):
        packet(fsm)
        assert fsm.cycle_request is (True if n == 0 else None)
        assert not fsm.cycle_long
        assert fsm.cycle_command_emitted is (not refused and n == 0)
        fsm.on_cycle()
        # No LONG_TICKS gating: this short slot consumes the answered flag.
        assert not fsm._rx_this_cycle and fsm._subtick == 0
    assert io.controls == [arq.CS_CYCLE_TOG] + [arq.CS_ACK] * 34
    assert bytes(io.delivered) == b" Trim" and fsm.rx_progress == 1


@pytest.mark.parametrize('old_long', [False, True])
def test_repeat_ack_allows_new_data_and_a_later_cycle_transition(old_long):
    fsm, io = linked(deferred=False)
    fsm.observe_peer_cycle(old_long)
    packet(fsm, long=old_long, request=not old_long)
    packet(fsm, long=old_long, request=not old_long)
    assert io.controls == [arq.CS_CYCLE_TOG, arq.CS_ACK]
    assert fsm.cycle_request is None and not fsm.cycle_command_emitted
    # The peer advances after the ACK. A new packet can negotiate normally.
    packet(fsm, long=old_long, request=not old_long, seq=2, data=b'ode')
    assert io.controls[-1] == arq.CS_CYCLE_TOG
    packet(fsm, long=not old_long, request=not old_long, seq=3, data=b' OK')
    assert io.controls[-1] == arq.CS_ACK
    assert fsm.cycle_long is (not old_long)
    assert bytes(io.delivered) == b' Trimode OK'
    assert fsm.rx_progress == 3


def test_old_cycle_header_ends_wait_for_command_but_preserves_request():
    fsm, _ = linked()
    packet(fsm)
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    fsm.observe_peer_cycle(False)
    assert not fsm.cycle_command_emitted and fsm.cycle_request is True
    assert not fsm.cycle_long


def test_real_long_arrival_confirms_transition_and_duplicates_deliver_once():
    fsm, io = linked()
    packet(fsm)
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    packet(fsm, long=True, seq=2, data=b"ode")
    packet(fsm, long=True, seq=2, data=b"ode")
    assert fsm.cycle_long and fsm.cycle_request is None
    assert not fsm.cycle_command_emitted
    assert io.controls == [arq.CS_CYCLE_TOG, arq.CS_ACK, arq.CS_ACK]
    assert bytes(io.delivered) == b" Trimode" and fsm.rx_progress == 2


def test_long_to_short_waits_for_short_header_even_when_long_policy_disabled():
    fsm, io = linked(long_cycle=False)
    fsm.observe_peer_cycle(True)
    packet(fsm, long=True, request=False)
    assert fsm.cycle_long and fsm.cycle_request is False
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    assert fsm.cycle_long and fsm.cycle_command_emitted
    packet(fsm, long=False, request=False, seq=2, data=b"ode")
    assert not fsm.cycle_long and fsm.cycle_request is None
    assert not fsm.cycle_command_emitted
    assert io.controls == [arq.CS_CYCLE_TOG, arq.CS_ACK]


def test_invalid_header_cannot_confirm_or_cancel_a_command():
    fsm, _ = linked()
    packet(fsm)
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    packet(fsm, long=True, seq=2, crc=False)
    assert not fsm.cycle_long and fsm.cycle_request is True
    assert fsm.cycle_command_emitted


def test_silence_retains_possible_long_packet_protection():
    fsm, _ = linked()
    packet(fsm)
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    for _ in range(arq.LONG_TICKS):
        fsm.on_cycle()
    # The driver must still guard the possible long frame. No decoded header
    # has proved that the command was lost or that short-slot keying is safe.
    assert fsm.cycle_command_emitted and fsm.cycle_request is True
    assert not fsm.cycle_long


def test_role_change_and_new_contact_discard_old_command():
    fsm, _ = linked()
    packet(fsm)
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)
    fsm._take_link()
    assert fsm.cycle_request is None and not fsm.cycle_command_emitted
    fsm.on_cs_emitted(arq.CS_CYCLE_TOG)  # stale callback has no pending command
    assert not fsm.cycle_command_emitted
    fsm.on_host_abort()
    fsm._enter_connected()
    assert fsm.cycle_request is None and not fsm.cycle_long


@pytest.mark.parametrize("long", [False, True])
@pytest.mark.parametrize("role", [arq.IRS, arq.ISS])
def test_unplaceable_goodbye_expires_without_spending_refused_retries(long, role):
    fsm, io = linked(role=role)
    fsm.cycle_long = long
    io.refuse_packet = True
    fsm.on_host_disconnect()
    for _ in range(arq.GOODBYE_PLACE_TICKS - 1):
        fsm.on_host_disconnect()  # must not restart the decision clock
        fsm.on_cycle()
        assert fsm.state in (arq.State.CONNECTED, arq.State.DISCONNECTING)
        assert not fsm.said_goodbye
        if fsm._inflight is not None:
            assert fsm._inflight.retries == 0
    fsm.on_cycle()
    assert fsm.state == arq.State.DISCONNECTED and io.closed == 1
    assert fsm.goodbye_unplaceable and not fsm.said_goodbye
    assert io.packets and all(p[2] & spec.STATUS_QRT for p in io.packets)
    assert "goodbye could not be placed -> link down" in io.logs


def test_peer_traffic_during_refused_teardown_does_not_renew_deadline():
    fsm, io = linked()
    io.refuse_packet = True
    fsm.on_host_disconnect()
    for n in range(arq.GOODBYE_PLACE_TICKS):
        packet(fsm, request=False, seq=n % 4, data=b"a")
        fsm.on_cycle()
    assert fsm.state == arq.State.DISCONNECTED and fsm.goodbye_unplaceable
    assert not fsm.said_goodbye


def test_late_safe_placement_uses_emitted_timeout_then_can_be_acknowledged():
    fsm, io = linked()
    io.refuse_packet = True
    fsm.on_host_disconnect()
    for _ in range(arq.GOODBYE_PLACE_TICKS - 2):
        fsm.on_cycle()
    io.refuse_packet = False
    fsm.on_cycle()
    assert fsm.said_goodbye and not fsm.goodbye_unplaceable
    assert fsm.state == arq.State.DISCONNECTING
    fsm.on_rx_cs(arq.CS_ACK)
    assert fsm.goodbye_acked and fsm.state == arq.State.DISCONNECTED


def test_successful_final_physical_cs2_keeps_terminal_ack_completion():
    fsm, io = linked()
    fsm.on_rx_packet(1, b"", spec.status_byte(1, qrt=True), True)
    assert fsm._rx_close_pending and fsm.state == arq.State.DISCONNECTING
    fsm.on_cs_emitted(1)  # physical CS2, not logical REQUEST
    assert fsm.state == arq.State.DISCONNECTED and io.closed == 1


def test_normal_changeover_does_not_start_terminal_deadline():
    fsm, io = linked()
    io.refuse_packet = True
    fsm.on_host_breakin()
    for _ in range(arq.GOODBYE_PLACE_TICKS + 2):
        packet(fsm, request=False)
        fsm.on_cycle()
    assert fsm.state == arq.State.CONNECTED and not fsm.goodbye_unplaceable


@pytest.mark.parametrize("request_long", [False, True])
@pytest.mark.parametrize("elapsed", [2, 3, 7])
def test_elapsed_slots_do_not_replace_current_packet_answer(request_long, elapsed):
    fsm, io = linked()
    packet(fsm, request=request_long)
    expected = arq.CS_CYCLE_TOG if request_long else arq.CS_ACK
    fsm.on_cycle(elapsed_ticks=elapsed)
    assert io.controls == [expected]
    assert not fsm._rx_this_cycle
    assert fsm._silent_cycles == 0


@pytest.mark.parametrize("long", [False, True])
def test_elapsed_slots_allow_only_one_present_iss_emission(long):
    fsm, io = linked(role=arq.ISS)
    fsm.cycle_long = long
    fsm.on_cycle(elapsed_ticks=7, cycle_ticks=arq.LONG_TICKS if long else 1)
    assert len(io.packets) == 1
    assert fsm._inflight.retries == 0
    fsm.on_cycle(elapsed_ticks=7, cycle_ticks=arq.LONG_TICKS if long else 1)
    assert len(io.packets) == 2
    assert fsm._inflight.retries == 1


def test_elapsed_time_does_not_fake_a_long_cycle_opportunity():
    fsm, io = linked(role=arq.ISS)
    fsm.cycle_long = True
    fsm.on_cycle(elapsed_ticks=9)
    assert not io.packets and fsm._subtick == 1
    fsm.on_cycle(elapsed_ticks=1)
    assert not io.packets and fsm._subtick == 2
    fsm.on_cycle(elapsed_ticks=1)
    assert len(io.packets) == 1 and fsm._subtick == 0


@pytest.mark.parametrize("long", [False, True])
def test_skipped_time_expires_unplaced_qrt_without_replaying_refusals(long):
    fsm, io = linked(role=arq.ISS)
    fsm.cycle_long = long
    io.refuse_packet = True
    fsm.on_host_disconnect()
    fsm.on_cycle(elapsed_ticks=arq.GOODBYE_PLACE_TICKS - 1,
                 cycle_ticks=arq.LONG_TICKS if long else 1)
    assert len(io.packets) == 1 and fsm._inflight.retries == 0
    assert not fsm.said_goodbye
    fsm.on_cycle()
    assert fsm.state == arq.State.DISCONNECTED and fsm.goodbye_unplaceable
    assert len(io.packets) == 1


@pytest.mark.parametrize("long", [False, True])
def test_emitted_goodbye_timeout_charges_skips_despite_fresh_rx_phase(long):
    fsm, io = linked(role=arq.ISS)
    fsm.cycle_long = long
    period = arq.LONG_TICKS if long else 1
    fsm.on_host_disconnect()
    fsm.on_cycle(cycle_ticks=period)
    assert fsm.said_goodbye and len(io.packets) == 1
    # A decoded peer event resets the action gate, but cannot renew teardown.
    fsm._peer_answered()
    fsm.on_cycle(elapsed_ticks=(arq.GOODBYE_CYCLES + 1) * period)
    assert fsm.state == arq.State.DISCONNECTED
    assert len(io.packets) == 1 and not fsm.goodbye_unplaceable


def test_final_ack_timeout_accumulates_fractional_long_cycles():
    fsm, io = linked()
    fsm.on_rx_packet(1, b"", spec.status_byte(1, qrt=True), True,
                     protocol=spec.Protocol.PACTOR3, cycle_long=True)
    assert fsm._rx_close_pending and fsm.cycle_long
    fsm.on_cycle(elapsed_ticks=2)
    assert fsm._rx_close_cycles == 0
    fsm.on_cycle(elapsed_ticks=arq.GOODBYE_CYCLES * arq.LONG_TICKS)
    assert fsm.state == arq.State.DISCONNECTING
    assert fsm._rx_close_cycles == arq.GOODBYE_CYCLES
    fsm.on_cycle()
    assert fsm.state == arq.State.DISCONNECTED
    assert len(io.controls) == 1 and io.closed == 1


@pytest.mark.parametrize("name", ["elapsed_ticks", "cycle_ticks"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_tick_counts_fail_before_mutating_state(name, value):
    fsm, io = linked(role=arq.ISS)
    with pytest.raises(ValueError, match=name):
        fsm.on_cycle(**{name: value})
    assert not io.packets and fsm._subtick == 0


def test_host_tick_forwards_time_and_pumps_peer_only_once():
    class Peer:
        pumped = cycled = 0

        def pump(self):
            self.pumped += 1

        def cycle(self):
            self.cycled += 1

    fsm, io = linked(role=arq.ISS)
    fsm.cycle_long = True
    host = object.__new__(PtcHost)
    host.arq, host.peer = fsm, Peer()
    host.tick(elapsed_ticks=9, cycle_ticks=arq.LONG_TICKS)
    assert len(io.packets) == 1
    assert host.peer.pumped == host.peer.cycled == 1


@pytest.mark.parametrize("near_sends", [False, True])
def test_simpeer_confirms_actual_long_headers_in_both_directions(near_sends, monkeypatch):
    peer = SimPeer()
    host = PtcHost(peer)
    host.protocol = spec.Protocol.PACTOR3
    host.arq._enter_connected()
    peer._far._enter_connected()
    sender, receiver = ((host.arq, peer._far) if near_sends
                        else (peer._far, host.arq))
    sender.role, receiver.role = arq.ISS, arq.IRS
    observed = []
    receive = receiver.on_rx_packet

    def record_cycle(*args, **kwargs):
        receive(*args, **kwargs)
        observed.append(receiver.cycle_long)

    monkeypatch.setattr(receiver, "on_rx_packet", record_cycle)
    payload = b"A sustained payload across the cycle change. " * 20
    sender.on_host_data(payload)
    for _ in range(40):
        host.tick()
        if not sender._outbuf and sender._inflight is None:
            break
    assert not sender._outbuf and sender._inflight is None
    received = peer.received if near_sends else host.channel(host.ptchn).rx
    assert bytes(received) == payload
    # The loaded middle uses long fields; the drained tail commands short -- on
    # the idle packets behind the last loaded one, because status bit 5 drops on
    # the packet that empties the buffer (the reference's own rule) and the IRS
    # confirms a length change on the peer's NEXT header.
    assert True in observed
    for _ in range(4):
        host.tick()
    assert not sender.cycle_long and not receiver.cycle_long
    assert receiver.cycle_request is None


@pytest.mark.parametrize("old_long", [False, True])
def test_p2_keeps_command_time_cycle_latch_without_pending_p3_state(old_long):
    fsm, io = linked(deferred=False)
    fsm.ladder = arq.P2_LADDER
    fsm.cycle_long = old_long
    target = not old_long
    status = spec.status_byte(1, data_type=spec.DataType.ASCII_8BIT,
                              long_cycle_request=target)
    # P2's production reader supplies no observed-cycle metadata. Repeated
    # status requests must receive one toggle followed by an ordinary ACK.
    fsm.on_rx_packet(1, b"abc", status, True, protocol=spec.Protocol.PACTOR2)
    assert fsm.cycle_long is target
    assert fsm.cycle_request is None and not fsm.cycle_command_emitted
    fsm.on_rx_packet(1, b"abc", status, True, protocol=spec.Protocol.PACTOR2)
    assert io.controls == [arq.CS_CYCLE_TOG, arq.CS_ACK]
    assert bytes(io.delivered) == b"abc"
    assert fsm.cycle_long is target
