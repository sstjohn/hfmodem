"""Repeated cycle grants must not consume the following unconfirmed frame."""
import pytest
from hfmodem.shrike import arq
from hfmodem.tests.shrike.test_long_sl1 import linked, PAYLOAD, row


@pytest.mark.parametrize('long', [False, True])
def test_repeated_cs6_preserves_granted_cycle_counter_and_bytes(long):
    host, tx = linked(long=long)
    a = host.arq
    a.on_host_data(PAYLOAD * 4)
    host.tick(elapsed_ticks=3 if long else 1, cycle_ticks=3 if long else 1)
    a.on_rx_cs(arq.CS_CYCLE_TOG)
    seq, count = a.tx_seq, a._buffer_raw
    pending = a._inflight.payload + bytes(a._outbuf)
    for _ in range(3):
        a.on_rx_cs(arq.CS_CYCLE_TOG)
        assert a.cycle_long == (not long)
        assert a.tx_seq == seq and a._buffer_raw == count
        assert a._inflight.payload + bytes(a._outbuf) == pending
    row('held-cycle-command', host, tx, pending)
    a.on_rx_cs(arq.CS_ACK)
    assert a.tx_seq != seq
    a.on_rx_cs(arq.CS_CYCLE_TOG)
    assert a.cycle_long == long  # A distinct acknowledgment permits a new grant.


def test_refused_first_long_frame_is_not_acknowledged_by_repeated_cs6():
    host, tx = linked()
    a = host.arq
    a.on_host_data(PAYLOAD)
    host.tick()
    send = host.send_packet
    host.send_packet = lambda *args, **kwargs: arq.REFUSED
    a.on_rx_cs(arq.CS_CYCLE_TOG)
    pending = a._inflight.payload + bytes(a._outbuf)
    for _ in range(3):
        a.on_rx_cs(arq.CS_CYCLE_TOG)
    assert a.tx_seq == 2 and a.cycle_long
    assert a._inflight.sent_at_level == 0
    assert a._inflight.payload + bytes(a._outbuf) == pending
    host.send_packet = send
    a.on_rx_cs(arq.CS_CYCLE_TOG)
    assert a._inflight.sent_at_level == 1
    row('first-long-emission', host, tx, pending)
