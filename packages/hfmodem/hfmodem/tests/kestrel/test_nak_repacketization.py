# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A solicited NAK may change record size, never the unconfirmed byte stream."""
import pytest
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA
from .test_data_over_gate import _connected


def pending(bw='2300', n=393):
    hs,io=_connected(bw=bw);hs.turn=VA._TURN_OURS
    data=bytes(i%251 for i in range(n));hs.send(data)
    hs._retry_data_over()
    return hs,io,data


def unconfirmed(hs):
    b=hs._tx_pending[0]
    return phy.vara_payload(b,caller=hs.caller,body_len=len(b))+b''.join(hs._txq)


@pytest.mark.parametrize('bw',['2300','2750'])
@pytest.mark.parametrize('n',[178,200,393,445])
def test_lower_retry_keeps_every_byte_and_separate_host_delivery(bw,n):
    hs,io,data=pending(bw,n)
    hs.send(b'next host write')
    old_over=hs._tx_pending[1]
    hs._took_responder_nak()
    assert hs._tx_pending[2]==phy.base_level(bw)-2
    assert hs._tx_pending[1]==old_over
    assert len(hs._tx_pending[0])==23
    assert unconfirmed(hs)==data+b'next host write'
    assert hs._txq[-1]==b'next host write'
    assert len(hs._txq[-2])<89
    assert hs._tx_retries==1
    assert hs._intermediate_query_for is None


def test_refused_lower_retry_does_not_rewrite_pending_or_queue():
    hs,io,_=pending();old=hs._tx_pending;queue=list(hs._txq)
    io.tx_went_out=lambda:False
    hs._took_responder_nak()
    assert hs._tx_pending==old and hs._txq==queue and hs._tx_retries==0


@pytest.mark.parametrize('state',['unsolicited','short_nak','explicit_ladder'])
def test_unmeasured_feedback_does_not_repacketize(state):
    hs,io,_=pending();old=hs._tx_pending
    if state=='unsolicited':hs._intermediate_query_for=None
    if state=='explicit_ladder':hs.tx_level=2
    (hs._took_nak if state=='short_nak' else hs._took_responder_nak)()
    assert hs._tx_pending==old


@pytest.mark.parametrize('bw',['2300','2750'])
def test_short_remainder_holds_lower_record_then_next_delivery_reenters_base(bw):
    hs,io,data=pending(bw,94)
    hs.send(b'next host write')
    hs._took_responder_nak()
    delivered=bytearray()
    while hs._tx_recovery_levels:
        body,_,level=hs._tx_pending
        assert level==phy.base_level(bw)-2
        delivered.extend(phy.vara_payload(body,caller=hs.caller,body_len=len(body)))
        assert phy.over_field(body)==0x89
        hs._took_over_continue()
    body,_,level=hs._tx_pending
    assert level==phy.base_level(bw)-2
    assert phy.over_is_last(body,hs.caller)
    delivered.extend(phy.vara_payload(body,caller=hs.caller,body_len=len(body)))
    assert bytes(delivered)==data
    hs._took_control_burst()
    assert hs._tx_pending[2]==phy.base_level(bw)
    assert unconfirmed(hs)==b'next host write'


@pytest.mark.parametrize('n', [0, 9, 17, 21, 22, 23, 44, 66, 79, 88])
def test_queried_short_final_nak_preserves_its_delivery_at_lower_record(n):
    hs, io, data = pending('2300', n)
    if n == 0:
        hs.send(b'')  # Empty host writes do not queue a DATA frame.
        assert hs._tx_pending is None
        return
    assert hs._final_query_fresh()
    old = hs._tx_pending
    hs._took_responder_nak()
    assert hs._tx_pending[1:] == (old[1], 1)
    assert unconfirmed(hs) == data
    assert hs._tx_retries == 1
    # Short retry needs no redundant final; full retry owes a closing frame.
    if n < 22:
        assert not hs._txq
        assert phy.over_is_last(hs._tx_pending[0], hs.caller)
    else:
        assert hs._txq and len(hs._txq[-1]) < 22
        assert hs._tx_recovery_levels == [1] * len(hs._txq)


def test_refused_short_final_lower_retry_preserves_pending_and_query_budget():
    hs, io, _ = pending('2300', 79)
    old = hs._tx_pending
    io.tx_went_out = lambda: False
    hs._took_responder_nak()
    assert hs._tx_pending == old and not hs._txq
    assert hs._tx_retries == 0 and hs._final_query_attempts == 1
    assert not hs._tx_recovery_levels


@pytest.mark.parametrize('state', ['unsolicited', 'expired', 'other_band', 'lower_record'])
def test_short_final_repacketization_needs_qualified_query_and_record(state):
    hs, io, _ = pending('2750' if state == 'other_band' else '2300', 79)
    if state == 'unsolicited': hs._final_query_for = None
    if state == 'expired': hs._final_query_at -= 10
    if state == 'lower_record': hs._tx_pending = (phy.vara_body(b'hi', hs.caller, body_len=48), 1, 2)
    old = hs._tx_pending
    hs._took_responder_nak()
    assert hs._tx_pending == old and not hs._txq
