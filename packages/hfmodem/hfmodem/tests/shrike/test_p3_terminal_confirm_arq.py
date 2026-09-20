"""Opted-in QRT completion requires an emitted marker and the opposite ACK.

Exercises PtcHost's real wire-counter mapping, ARQ lifecycle and transmit-seam
refusals. The marker waveform itself is covered by test_p3_terminal_marker.
"""
import pytest

from hfmodem.shrike import arq, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_iss_login_advance import ScriptedPeer


class Peer(ScriptedPeer):
    def __init__(self):
        super().__init__()
        self.marker_calls = 0
        self.markers_emitted = 0
        self.marker_bits = []
        self.refuse_marker = False

    def send_p3_terminal(self, header_bit):
        self.marker_calls += 1
        self.marker_bits.append(header_bit)
        if self.refuse_marker:
            return arq.REFUSED
        self.markers_emitted += 1
        return 0


def wire(host, cs):
    host.on_rx_event(rxfront.Event(0, 'cs', 'specified wire control',
                                   protocol=host.protocol, cs=cs))


def closing(*, enabled=True, protocol=spec.Protocol.PACTOR3, role=arq.IRS,
            counter=None):
    peer = Peer()
    host = PtcHost(peer, mycall='W9SSJ')
    host.protocol = protocol
    host.p3_qrt_confirm = enabled
    host.arq.role = role
    host.arq._enter_connected()
    if counter is not None:
        host.arq._next_seq = counter
    host.arq.on_host_disconnect()
    host.tick()
    assert host.arq.state == arq.State.DISCONNECTING
    assert host.arq._inflight.status & spec.STATUS_QRT
    return host, peer


def qrt_answer(host):
    # In P3 the wire ACK is the parity of the packet actually awaiting it.
    wire(host, host.arq.tx_seq & 1)


@pytest.mark.parametrize('role', [arq.IRS, arq.ISS])
def test_qrt_ack_retains_p3_until_emitted_marker_gets_opposite_ack(role):
    host, peer = closing(role=role)
    a = host.arq
    previous_ack = a.tx_seq & 1
    marker_ack = previous_ack ^ 1
    qrt_answer(host)
    assert a.state == arq.State.DISCONNECTING
    assert a.terminal_confirm_pending and a.terminal_pending
    assert not a.terminal_confirm_emitted and not a.goodbye_acked
    assert a._inflight is None and a.tx_seq == marker_ack
    assert a.speed_level == 1 and not a.cycle_long
    assert host.protocol == spec.Protocol.PACTOR3
    assert peer.marker_calls == 0, 'the QRT callback does not transmit another burst'

    wire(host, marker_ack)
    assert a.terminal_confirm_pending, 'no control can ACK an unkeyed marker'
    host.tick()
    assert peer.markers_emitted == 1 and a.terminal_confirm_emitted
    assert peer.marker_bits == [marker_ack]
    wire(host, previous_ack)
    assert a.terminal_confirm_pending and not a.goodbye_acked
    assert peer.marker_calls == 1, 'a repeated ACK does not synchronously retransmit'
    wire(host, marker_ack)
    assert a.state == arq.State.DISCONNECTED and a.goodbye_acked
    assert host.protocol == spec.Protocol.PACTOR1
    assert not a.terminal_confirm_pending and not a.terminal_confirm_emitted
    assert a.tx_seq is None


@pytest.mark.parametrize('counter', [1, 3])
def test_odd_qrt_repeated_cs2_cannot_confirm_emitted_marker(counter):
    host, peer = closing(role=arq.ISS, counter=counter)
    assert host.arq.tx_seq == counter
    wire(host, 1)  # The odd QRT's physical CS2 acknowledgement.
    assert host.arq.terminal_confirm_pending and host.arq.tx_seq == 0
    wire(host, 0)
    assert not host.arq.goodbye_acked, 'CS1 before emission proves nothing'
    host.tick()
    assert peer.marker_bits == [0] and host.arq.terminal_confirm_emitted
    wire(host, 1)
    assert host.arq.terminal_confirm_pending and not host.arq.goodbye_acked
    wire(host, 0)
    assert host.arq.state == arq.State.DISCONNECTED and host.arq.goodbye_acked


def test_refused_marker_cannot_be_acknowledged_then_can_retry_successfully():
    host, peer = closing()
    qrt_answer(host)
    peer.refuse_marker = True
    host.tick()
    wire(host, 1)
    assert host.arq.terminal_confirm_pending
    assert not host.arq.terminal_confirm_emitted
    peer.refuse_marker = False
    host.tick()
    assert peer.marker_calls == 2 and peer.markers_emitted == 1
    wire(host, 1)
    assert host.arq.goodbye_acked and host.arq.state == arq.State.DISCONNECTED


@pytest.mark.parametrize('refused', [False, True])
def test_repeated_cs1_cannot_extend_terminal_confirmation_budget(refused):
    host, peer = closing()
    qrt_answer(host)
    peer.refuse_marker = refused
    for _ in range(arq.GOODBYE_CYCLES):
        wire(host, 0)
        host.tick()
        assert host.arq.terminal_confirm_pending
    wire(host, 0)
    host.tick()
    assert host.arq.state == arq.State.DISCONNECTED
    assert not host.arq.goodbye_acked
    assert peer.marker_calls == arq.GOODBYE_CYCLES
    assert peer.markers_emitted == (0 if refused else arq.GOODBYE_CYCLES)


def test_skipped_slots_age_stage_without_replaying_marker_transmissions():
    host, peer = closing()
    qrt_answer(host)
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES)
    assert peer.marker_calls == 1 and host.arq.terminal_confirm_pending
    host.tick()
    assert host.arq.state == arq.State.DISCONNECTED
    assert peer.marker_calls == 1 and not host.arq.goodbye_acked


def test_terminal_stage_has_its_own_short_cycle_budget_after_long_qrt():
    host, peer = closing()
    host.arq.speed_level = 5
    host.arq.cycle_long = True
    host.arq._goodbye_cycles = arq.GOODBYE_CYCLES
    qrt_answer(host)
    assert host.arq.speed_level == 1 and not host.arq.cycle_long
    host.tick()
    assert host.arq.terminal_confirm_pending and peer.markers_emitted == 1
    # Historical short slots consume the terminal budget without emitting.
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES)
    assert host.arq.state == arq.State.DISCONNECTED
    assert peer.markers_emitted == 1 and not host.arq.goodbye_acked


@pytest.mark.parametrize('cs', [arq.CS_BREAKIN, arq.CS_SPEED_UP, arq.CS_NAK,
                               arq.CS_CYCLE_TOG])
def test_other_controls_cannot_settle_marker_or_change_its_geometry(cs):
    host, peer = closing()
    qrt_answer(host)
    host.tick()
    wire(host, cs)
    assert host.arq.terminal_confirm_pending
    assert host.arq.role == arq.ISS and host.arq.speed_level == 1
    assert not host.arq.cycle_long and not host.arq.goodbye_acked


def test_packet_and_abort_cannot_leak_terminal_state_into_next_link():
    host, peer = closing()
    qrt_answer(host)
    host.arq.on_rx_packet(1, b'', 0x40, True, breakin=True,
                          protocol=spec.Protocol.PACTOR3)
    assert host.arq.terminal_confirm_pending and host.arq.role == arq.ISS
    host.arq.on_host_abort()
    assert not host.arq.terminal_confirm_pending
    host.arq.role = arq.ISS
    host.arq._enter_connected()
    assert not host.arq.terminal_confirm_emitted
    assert not host.arq.goodbye_acked and host.arq.tx_seq is None


def test_disabled_experiment_preserves_normal_qrt_ack_close():
    host, peer = closing(enabled=False)
    qrt_answer(host)
    assert host.arq.state == arq.State.DISCONNECTED and host.arq.goodbye_acked
    assert peer.marker_calls == 0


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2])
def test_enabled_experiment_does_not_change_other_protocols(protocol):
    host, peer = closing(protocol=protocol, role=arq.ISS)
    # P1 has its own wire alternation; this test isolates the shared QRT action.
    host.arq.on_rx_cs(arq.CS_ACK)
    assert host.arq.state == arq.State.DISCONNECTED and host.arq.goodbye_acked
    assert peer.marker_calls == 0


def test_base_io_cannot_report_an_unimplemented_marker_as_emitted():
    assert arq.ArqIO().send_p3_terminal(1) == arq.REFUSED
