# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Scanner promises against real ARQ events and independent host framing."""
import pytest

from hfhost import ptc
from hfmodem.shrike.arq import State
from hfmodem.core.hostmode import REQUEST
from hfmodem.shrike.ptc import PtcHost, SimPeer


class Wire:
    def __init__(self, host):
        self.host = host
        self.bit = 0
        self.decoder = ptc.Deframer()
        host.feed(b'PTC 31\rJHOST4\r')

    def command(self, text):
        self.bit ^= 1
        self.frame = ptc.Packet(31, True, text.encode(), counter=self.bit).encode()
        return self.decoder.feed(self.host.feed(self.frame))[0]

    def retry(self):
        return self.decoder.feed(self.host.feed(self.frame))[0]


class HeldPeer(SimPeer):
    def __init__(self):
        super().__init__()
        self.acquired = self.released = 0

    def begin_scan_hold(self):
        if not super().begin_scan_hold():
            return False
        self.acquired += 1
        return True

    def end_scan_hold(self):
        self.released += 1


def ready():
    peer = HeldPeer()
    host = PtcHost(peer, mycall='N0CALL')
    host.set_listen(True)
    wire = Wire(host)
    return host, peer, wire


def test_scan_hold_prevents_outgoing_and_incoming_until_explicit_release():
    host, peer, wire = ready()
    assert wire.command('%W0').text == '1'
    assert host.arq.scan_held and host.arq.state is State.DISCONNECTED
    assert wire.command('C N0DX').code == ptc.FAIL
    assert host.arq.last_connect is None and not peer._to_far
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect('N0DX', 'N0CALL')
    assert host.arq.state is State.DISCONNECTED and not peer._to_far
    host.tick()
    assert not peer._to_far and host.sent_total == 0
    assert wire.command('%W1').text == ''
    assert peer.acquired == peer.released == 1
    assert host.arq.state is State.LISTENING
    assert wire.command('C N0DX').code == ptc.OK
    assert host.arq.state is State.CONNECTING
    assert wire.command('%W0').text == '0'
    assert host.wait_until is None and not host.arq.scan_held


def test_queued_call_wins_before_scan_hold_and_is_not_silently_retired():
    host, peer, wire = ready()
    peer._far.on_host_connect('N0DX', host.mycall)
    assert peer._to_near
    assert wire.command('%W0').text == '0'
    assert not host.arq.scan_held and host.wait_until is None
    assert peer.acquired == 0
    peer.pump()
    assert host.arq.state is State.CONNECTED


def test_scan_hold_wins_before_a_later_call_without_transmitting_an_answer():
    host, peer, wire = ready()
    assert wire.command('%W0').text == '1'
    peer._far.on_host_connect('N0DX', host.mycall)
    peer.pump()
    assert host.arq.state is State.DISCONNECTED
    assert peer._far.state is State.CONNECTING
    assert not peer._to_far
    wire.command('%W1')
    peer._far.on_cycle()
    peer.pump()
    assert host.arq.state is State.CONNECTED


def test_renewal_listen_expiry_and_attachment_loss_keep_the_producer_in_step(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hfmodem.shrike.ptc.time.monotonic', lambda: clock[0])
    host, peer, wire = ready()
    assert wire.command('%W0').text == '1'
    assert host.wait_until == 110.
    clock[0] = 105.
    assert wire.command('%W0').text == '1'
    assert host.wait_until == 115. and peer.acquired == 1
    wire.command('%L1')
    assert host.arq.scan_held and host.wait_until == 115.
    clock[0] = 116.
    host.service_host_output()
    assert not host.arq.scan_held and host.wait_until is None
    assert host.arq.state is State.LISTENING and peer.released == 1
    wire.command('%W0')
    host.attachment_lost()
    assert peer.acquired == peer.released == 2
    assert not host.arq.scan_held and host.wait_until is None


def test_unimplemented_or_unreleasable_producer_hold_never_returns_permission():
    class UnqualifiedPeer:
        def attach(self, host):
            pass

        def begin_scan_hold(self):
            pytest.fail('must verify release support before acquiring')

    host = PtcHost(UnqualifiedPeer())
    host.set_listen(True)
    wire = Wire(host)
    assert wire.command('%W0').text == '0'
    assert host.arq.state is State.LISTENING
    assert host.wait_until is None


def test_cached_scan_permission_cannot_outlive_its_hold(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hfmodem.shrike.ptc.time.monotonic', lambda: clock[0])
    host, peer, wire = ready()
    granted = wire.command('%W0')
    clock[0] = 105.
    assert wire.retry() == granted
    assert host.wait_until == 110. and peer.acquired == 1
    clock[0] = 111.
    # A bare CRC retransmission request must not replay the expired "1".
    assert host.feed(REQUEST) == b''
    assert peer.released == 1 and not host.arq.scan_held
    # Retrying the full command takes a new hold rather than resurrecting one.
    assert wire.retry().text == '1'
    assert host.wait_until == 121. and peer.acquired == 2


def test_expired_cached_permission_cannot_mask_new_receive_work(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hfmodem.shrike.ptc.time.monotonic', lambda: clock[0])
    host, peer, wire = ready()
    assert wire.command('%W0').text == '1'
    clock[0] = 111.
    peer._far.on_host_connect('N0DX', host.mycall)
    assert wire.retry().text == '0'
    assert not host.arq.scan_held and peer.released == 1


def test_expiring_permission_preserves_a_partially_received_crc_packet(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('hfmodem.shrike.ptc.time.monotonic', lambda: clock[0])
    host, peer, wire = ready()
    assert wire.command('%W0').text == '1'
    split = len(wire.frame)//2
    clock[0] = 109.
    assert host.feed(wire.frame[:split]) == b''
    clock[0] = 111.
    reply = wire.decoder.feed(host.feed(wire.frame[split:]))
    assert len(reply) == 1 and reply[0].text == '1'
    assert host.wait_until == 121. and peer.acquired == 2


def test_release_failure_keeps_calls_inhibited():
    host, peer, wire = ready()
    assert wire.command('%W0').text == '1'

    def failed_release():
        raise OSError('producer hold could not be released')

    peer.end_scan_hold = failed_release
    with pytest.raises(OSError, match='could not be released'):
        host.release_wait()
    assert host.arq.scan_held and host.wait_until is not None
    with pytest.raises(ValueError, match='SCANNER HOLD'):
        host.arq.on_host_connect('N0CALL', 'N0DX')
    assert not peer._to_far
