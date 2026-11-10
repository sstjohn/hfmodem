# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Annex A-C edge cases and the import review's byte-channel sequences."""
import calendar

import pytest

from hfmodem.bateleur.s5066 import cas as C, dpdu as D, dts as T, primitives as P, sis as S
from hfmodem.bateleur.s5066.node import Node, Timing


class Upper(T.Upper):
    def __init__(self):
        self.received = []
        self.confirmed = []
        self.rejected = []
        self.lost = []

    def cpdu_received(self, src, dest, data, **kw):
        self.received.append((data, kw))

    def cpdu_confirmed(self, handle):
        self.confirmed.append(handle)

    def cpdu_rejected(self, handle, reason):
        self.rejected.append((handle, reason))

    def connection_lost(self, remote):
        self.lost.append(remote)


def transfer(**kw):
    up = Upper()
    d = T.DataTransfer(1, up, **kw)
    d.connection_made(2)
    return d, up


def frame(seq, **kw):
    return D.DPDU(D.DATA_ONLY, 1, 2, seq=seq, **kw)


def bind(node, mode=P.MODE_ARQ, confirm=P.CONFIRM_NONE, rank=0):
    got = []
    att = node.attach(got.append)
    att.handle(P.prim(P.BIND_REQUEST, sap=6, rank=rank,
                      service=P.ServiceType(mode, confirm)))
    return att, got


def submit(att, **kw):
    args = dict(dest=2, dest_sap=6, u_pdu=b'payload', ttl=0,
                mode=P.ServiceType(P.MODE_DEFAULT))
    args.update(kw)
    att.handle(P.prim(P.UNIDATA_REQUEST, **args))


def test_expedited_receiver_first_missing_edge():
    d, up = transfer()
    for s in (0, 2):
        d.received(D.DPDU(D.EXP_DATA_ONLY, 1, 2, seq=s, start=s == 0,
                          end=s == 2, data=b'x'), 0)
    p, = d.compose(0, 1000)
    assert (p.rx_lwe, p.acks) == (1, b'\x01')
    d.received(D.DPDU(D.EXP_DATA_ONLY, 1, 2, seq=1, data=b'y'), 1)
    p, = d.compose(1, 1000)
    assert p.rx_lwe == 3 and not p.acks
    assert up.received[0][0] == b'xyx'


def test_expedited_sender_does_not_confirm_missing_last_frame():
    d, up = transfer(segment=1)
    d.submit(T.CPDU(b'abc', 2, expedited=True, handle='msg'))
    d.compose(0, 1000)
    d.received(D.DPDU(D.EXP_ACK_ONLY, 1, 2, rx_lwe=2), 1)
    assert not up.confirmed
    assert list(d.links[2].exp.frames) == [2]


def test_management_ack_uses_senders_id_and_combined_request():
    d, _ = transfer()
    d.request_rate(2, 2400, 'short')
    d.compose(0, 1000)
    d.received(D.DPDU(D.MANAGEMENT, 1, 2, ack=True, valid=True, frame_id=91,
                      eow=D.eow_drc_request(600, 'short', 0)), 1)
    assert d.links[2].mgmt.out is None
    assert d.links[2].drc['role'] == 'slave'
    assert d.links[2].drc['rate'] == 600


@pytest.mark.parametrize('in_order', [False, True])
@pytest.mark.parametrize('length', [2, 300])
def test_arq_reassembly_crosses_windows_and_wrap(in_order, length):
    d, up = transfer()
    for i in range(length):
        d.received(frame(i % 256, start=i == 0, end=i == length - 1,
                         in_order=in_order, data=bytes([i % 251])), i)
    assert [x[0] for x in up.received] == [bytes(i % 251 for i in range(length))]
    assert not d.links[2].arq.held


@pytest.mark.parametrize('case', ['uwe', 'lwe', 'fsn', 'ack_edge', 'ack_map'])
def test_sync_failure_starts_reset(case):
    d, _ = transfer()
    a = d.links[2].arq
    if case == 'uwe':
        a.rx_uwe = 10
        p = frame(5, tx_uwe=True)
    elif case == 'lwe':
        p = frame(50, tx_lwe=True)
    elif case == 'fsn':
        a.rx_uwe = 100
        p = frame(200)
    else:
        p = D.DPDU(D.ACK_ONLY, 1, 2, rx_lwe=1 if case == 'ack_edge' else 0,
                   acks=b'\x01' if case == 'ack_map' else b'')
    d.received(p, 0)
    assert d.links[2].reset_out is not None
    assert any(p.full_reset for p in d.compose(0, 1000))


def test_reset_command_releases_handles_and_waits_for_ack():
    d, up = transfer(segment=1)
    d.submit(T.CPDU(b'ab', 2, handle='msg'))
    d.compose(0, 1000)
    d.received(D.DPDU(D.RESET_WIN_RESYNC, 1, 2, reset_win_cmnd=True,
                      new_rx_lwe=2, reset_id=7), 1)
    assert not d.links[2].arq.unacked
    assert up.rejected == [('msg', T.Reject.RESET)]
    d.received(D.DPDU(D.RESET_WIN_RESYNC, 1, 2, reset_tx_win_rqst=True), 2)
    sent = d.compose(2, 1000)
    command = next(p for p in sent if p.reset_win_cmnd)
    assert d.links[2].reset_out is not None
    d.received(D.DPDU(D.RESET_WIN_RESYNC, 1, 2, reset_ack=True,
                      reset_id=command.reset_id), 3)
    assert d.links[2].reset_out is None


def test_invalid_reset_edge_preserves_unconfirmed_handles():
    d, up = transfer()
    d.submit(T.CPDU(b'x', 2, handle='msg'))
    d.compose(0, 1000)
    d.received(D.DPDU(D.RESET_WIN_RESYNC, 1, 2, reset_win_cmnd=True,
                      new_rx_lwe=100), 1)
    assert d.links[2].reset_out is not None
    assert d.links[2].arq.tx_lwe == 0
    assert not up.confirmed


def test_crc_valid_unknown_type_draws_warning():
    raw = bytearray(D.encode(D.DPDU(D.ACK_ONLY, 1, 2)))
    raw[2] = 9 << 4
    raw[-2:] = D.crc16(raw[2:-2]).to_bytes(2, 'little')
    d, _ = transfer()
    for p in D.Parser().push(raw):
        d.received(p, 0)
    p, = d.compose(0, 1000)
    assert (p.type, p.rx_type, p.reason) == (D.WARNING, 9, 0)


def test_unconnected_management_is_acknowledged():
    d = T.DataTransfer(1)
    d.received(D.DPDU(D.MANAGEMENT, 1, 2, frame_id=99, valid=True), 0)
    assert any(p.ack for p in d.compose(0, 1000))
    assert not d.connected(2)


def test_management_precedes_other_types():
    d, _ = transfer()
    d.request_rate(2, 600, 'short')
    d._warn(2, 9, 0)
    d.submit(T.CPDU(b'x', 2, arq=False, expedited=True))
    assert d.compose(0, 1000)[0].type == D.MANAGEMENT


def test_unknown_eow_response_is_not_echoed_or_repeated():
    d, _ = transfer()
    for i, kind in enumerate((9, 3, 9)):
        d.received(D.DPDU(D.MANAGEMENT, 1, 2, valid=True, frame_id=i,
                          eow=kind << 8), i)
    responses = [x for x in d.links[2].mgmt.queue if x[0] >> 8 == 3]
    assert len(responses) == 1


def test_no_progress_acks_have_a_retry_limit():
    d, up = transfer(max_retries=3)
    d.submit(T.CPDU(b'x', 2, handle='msg'))
    for i in range(12):
        d.tick(i * 10)
        if not d.connected(2):
            break
        d.compose(i * 10, 1000)
        d.transmitted(i * 10 + 1, i * 10 + 5)
        d.received(D.DPDU(D.ACK_ONLY, 1, 2, rx_lwe=0), i * 10 + 2)
    assert up.lost == [2]
    assert len(up.rejected) == 1


def test_relink_clears_stale_receive_state():
    n = Node(1, Timing(1200))
    n.cas.links[2] = C.PhysicalLink(2, False, False, 'made', first_data=True)
    n.dts.connection_made(2)
    n.dts.links[2].arq.rx_lwe = 40
    n.cas.cpdu_received(2, 1, bytes([C.LINK_REQUEST << 4]), expedited=True, arq=False)
    assert n.dts.links[2].arq.rx_lwe == 0


def test_drc_ack_without_response_expires():
    d, up = transfer()
    d.request_rate(2, 2400, 'short')
    d.compose(0, 1000)
    d.transmitted(1, 5)
    d.received(D.DPDU(D.MANAGEMENT, 1, 2, ack=True), 2)
    d.tick(1200)
    assert up.lost == [2]


@pytest.mark.parametrize('complete', [False, True])
def test_drc_slave_sends_empty_data_on_return(complete):
    d, _ = transfer()
    link = d.links[2]
    link.drc = dict(role='slave', phase='confirming', rate=600, interleave='short', equal=False)
    if complete:
        link.mgmt.out = (3, D.eow_drc_response(D.RESP_CONFIRM), b'', 1)
        p = D.DPDU(D.MANAGEMENT, 1, 2, ack=True, frame_id=3)
    else:
        p = D.DPDU(D.MANAGEMENT, 1, 2, valid=True,
                   eow=D.eow_drc_response(D.RESP_CANCEL))
    d.received(p, 1)
    assert any(p.type in D.DATA_TYPES and not p.data for p in d.compose(1, 1000))


def test_drc_equal_parameters_are_checked():
    d, _ = transfer()
    d.request_rate(2, 2400, 'short')
    d.compose(0, 1000)
    d.can_receive = lambda rate, il: False
    d.received(D.DPDU(D.MANAGEMENT, 1, 2, ack=True, valid=True,
                      eow=D.eow_drc_response(D.RESP_ACCEPT, 1)), 1)
    assert 2 not in d.tx_rate
    assert any(D.eow_fields(x[0])[1] >> 5 == D.RESP_CANCEL for x in d.links[2].mgmt.queue)


@pytest.mark.parametrize('arq,expedited', [(True, False), (True, True), (False, False)])
def test_small_key_budget_preserves_whole_dpdus(arq, expedited):
    n = Node(1, Timing(75, lead=4, tail=4), max_tx_s=30)
    n.dts.connection_made(2)
    n.dts.submit(T.CPDU(bytes(200), 2, arq=arq, expedited=expedited))
    data, duration = n.poll(0)
    assert duration <= 30
    parser = D.Parser()
    assert parser.push(data) and not parser.buf
    assert len(data) <= n.timing.most_bytes(30)


def test_tiny_budget_does_not_spend_ack():
    d, _ = transfer()
    d.links[2].arq.owe_ack = True
    assert d.compose(0, 1) == []
    assert d.links[2].arq.owe_ack


def test_nonarq_window_includes_later_transmissions_and_repeats():
    n = Node(1, Timing(1200, lead=.6, tail=.6), max_tx_s=4)
    n.dts.submit(T.CPDU(bytes(1000), 2, arq=False, repeats=1))
    heard = []
    now = 0
    while n.dts._nonarq_out:
        result = n.poll(now)
        if result:
            data, secs = result
            offset = 0
            for p in D.Parser().push(data):
                start = now + n.timing.at(offset)
                heard.append((start, p.window))
                offset += D.size(p)
            end = now + secs
            now = end + n.turnaround_s
        else:
            now += 1
        assert now < 100
    assert all(start + window / 2 >= end for start, window in heard)
    assert all(window > 0 for _, window in heard)


def test_nonarq_memory_and_hold_time_are_bounded():
    d, up = transfer(deliver_with_errors=True)
    for i in range(2000):
        d.received(D.DPDU(D.NON_ARQ, 1, 2 + i, cpdu_id=i % 4096,
                          cpdu_size=65535, window=65535, data=b'x'), 0)
    assert not d._reassembly
    for i in range(100):
        d.received(D.DPDU(D.NON_ARQ, 1, 2, cpdu_id=i, cpdu_size=2048,
                          window=65535, data=b'x'), 0)
    assert len(d._reassembly) <= 8
    d.tick(601)
    assert not d._reassembly


def test_nonarq_in_order_waits_for_earlier_partial():
    d, up = transfer()
    d.received(D.DPDU(D.NON_ARQ, 1, 2, in_order=True, cpdu_id=0, cpdu_size=2, data=b'a', window=10), 0)
    d.received(D.DPDU(D.NON_ARQ, 1, 2, in_order=True, cpdu_id=1, cpdu_size=1, data=b'c', window=10), 0)
    assert not up.received
    d.received(D.DPDU(D.NON_ARQ, 1, 2, in_order=True, cpdu_id=0, cpdu_size=2, offset=1, data=b'b', window=8), 1)
    assert [x[0] for x in up.received] == [b'ab', b'c']


def test_heard_history_is_bounded():
    n = Node(1, Timing(1200))
    for _ in range(2000):
        n.rx_bytes(0, D.encode(D.DPDU(D.WARNING, 1, 2)))
    assert len(n.heard) <= 256


def test_ttl_wire_limit_is_fifteen_days():
    p = P.prim(P.UNIDATA_REQUEST, dest=2, dest_sap=6, u_pdu=b'x', ttl=30 * 86400)
    assert P.decode_body(P.encode_body(p))['ttl'] == 15 * 86400


@pytest.mark.parametrize('date,days', [((2026, 9, 1), 9), ((2026, 12, 29), 5),
                                       ((2027, 1, 1), 15)])
def test_ttd_future_day_survives_wrap(date, days):
    now = calendar.timegm((*date, 12, 0, 0))
    assert not S._ttd_passed(*S._ttd_field(now + days * 86400), now)


def test_nonarq_confirm_bit_does_not_provoke_reply():
    n = Node(1, Timing(1200))
    att, got = bind(n, P.MODE_NON_ARQ)
    n.sis.spdu_received(2, 1, b'\x00\x66\x80hello', expedited=False, arq=False)
    assert got[-1].type == P.UNIDATA_INDICATION
    assert not n.dts.pending(0)
    submit(att, mode=P.ServiceType(P.MODE_NON_ARQ, P.CONFIRM_CLIENT))
    n.sis.tick(0)
    assert not n.dts._nonarq_out[-1][0].data[3] & 0x80


def test_hard_link_requires_physical_link():
    n = Node(1, Timing(1200))
    bind(n)
    n.sis.spdu_received(2, 1, b'\x30\x66', expedited=True, arq=False)
    assert n.sis.hard is None


def test_invalid_primitive_preserves_neighbours():
    good = P.encode(P.prim(P.KEEP_ALIVE))
    bad = b'\x90\xeb\x00\x00\x01\xff'
    assert [p.type for p in P.Framer().push(good + bad + good)] == [P.KEEP_ALIVE] * 2


def test_reason_codes_on_hard_link_timeout_and_break():
    n = Node(1, Timing(1200))
    _, got = bind(n)
    n.sis.hard = S.HardLink(2, 0, 0, 0, 6, 6, True, 'terminating', 1)
    n.sis.tick(2)
    assert got[-1]['reason'] == 3
    n.sis.hard = S.HardLink(2, 0, 0, 0, 6, 6, True, 'established')
    n.sis.link_broken(2, 'lost')
    assert got[-1]['reason'] == 5


def test_errored_delivery_selected_by_bound_service():
    n = Node(1, Timing(1200))
    _, got = bind(n, P.MODE_NON_ARQ_ERRORS)
    n.dts.received(D.DPDU(D.NON_ARQ, 1, 2, cpdu_size=9, data=b'\x00\x00\x66\x00hello',
                          data_ok=False, window=2), 0)
    n.dts.tick(3)
    assert not [p for p in got if p.type == P.UNIDATA_INDICATION]
    # A good C/S header followed by a bad segment can be delivered to this SAP.
    n.dts.received(D.DPDU(D.NON_ARQ, 1, 2, cpdu_id=1, cpdu_size=9,
                          data=b'\x00\x00\x66\x00', window=2), 4)
    n.dts.received(D.DPDU(D.NON_ARQ, 1, 2, cpdu_id=1, cpdu_size=9, offset=4,
                          data=b'hello', data_ok=False, window=2), 4)
    n.dts.tick(7)
    assert got[-1].type == P.UNIDATA_INDICATION
    assert got[-1]['errored'] == [(0, 5)]


def test_local_refusal_preserves_pending_bytes_and_retry_count():
    n = Node(1, Timing(1200))
    n.dts.connection_made(2)
    n.dts.submit(T.CPDU(b'payload', 2, handle='msg'))
    for now in range(10):
        assert n.poll(now, defer=True)
        n.sent(False)
        assert n.dts.links[2].deadline is None
        assert n.dts.links[2].silences == 0
    data, _ = n.poll(20)
    assert b'payload' in data
    assert n.transmissions == 1


def test_local_refusal_does_not_confirm_nonarq():
    n = Node(1, Timing(1200))
    att, got = bind(n, P.MODE_NON_ARQ, P.CONFIRM_NODE)
    submit(att)
    n.poll(0, defer=True)
    assert not [p for p in got if p.type == P.UNIDATA_REQUEST_CONFIRM]
    n.sent(False)
    assert n.dts._nonarq_out


def test_expedited_eviction_detaches_and_purges():
    n = Node(1, Timing(1200))
    n.sis.max_expedited = 1
    att, got = bind(n, confirm=P.CONFIRM_NODE)
    p = P.prim(P.EXPEDITED_UNIDATA_REQUEST, dest=2, dest_sap=6, u_pdu=b'x', ttl=0,
               mode=P.ServiceType(P.MODE_DEFAULT))
    att.handle(p)
    att.handle(p)
    assert att.client is None and not n.sis.queue and not n.sis.awaiting
    assert got[-1].type == P.UNBIND_INDICATION
    att.handle(p)
    assert not n.sis.queue


def test_expedited_limit_counts_outstanding_only():
    n = Node(1, Timing(1200))
    n.sis.max_expedited = 1
    att, got = bind(n, P.MODE_NON_ARQ)
    for now in range(3):
        att.handle(P.prim(P.EXPEDITED_UNIDATA_REQUEST, dest=2, dest_sap=6, u_pdu=b'x',
                          ttl=0, mode=P.ServiceType(P.MODE_DEFAULT)))
        n.poll(now * 20)
    assert att.client is not None
    assert not [p for p in got if p.type == P.UNBIND_INDICATION]


def test_late_confirmation_cannot_reach_new_binder():
    n = Node(1, Timing(1200))
    att, _ = bind(n, confirm=P.CONFIRM_NODE)
    submit(att)
    old = n.sis.queue[0]
    att.close()
    _, new = bind(n, confirm=P.CONFIRM_NODE)
    n.sis._report(old, True)
    assert new[-1].type == P.BIND_ACCEPTED


def test_higher_rank_preempts_sap():
    n = Node(1, Timing(1200))
    old, got = bind(n, rank=1)
    new, received = bind(n, rank=2)
    assert old.client is None and new.client is not None
    assert got[-1]['reason'] == P.UNBIND_PREEMPTED
    assert received[-1].type == P.BIND_ACCEPTED


def test_sis_submission_has_a_per_client_bound():
    n = Node(1, Timing(1200))
    att, got = bind(n, confirm=P.CONFIRM_NODE)
    for _ in range(1000):
        submit(att)
    assert len(n.sis.queue) <= 64
    assert got[-1].type == P.UNIDATA_REQUEST_REJECTED


def test_unconfirmed_data_does_not_relink_without_limit():
    n = Node(1, Timing(1200))
    att, _ = bind(n)
    submit(att)
    n.sis.link_failed(2, 'no answer')
    assert not n.sis.queue


def test_drc_refuses_mode_that_cannot_fit_existing_frame():
    from hfmodem.bateleur.modem import Mode
    from hfmodem.bateleur.s5066.onair import ModeTiming
    n = Node(1, ModeTiming(Mode('110', '1200S')), max_tx_s=10)
    n.dts.connection_made(2)
    n.dts.submit(T.CPDU(bytes(200), 2))
    n.poll(0)
    assert not n.dts.can_receive(75, 'long')


def test_full_reset_repetition_preserves_new_data():
    d, up = transfer()
    p = D.DPDU(D.RESET_WIN_RESYNC, 1, 2, full_reset=True, reset_id=7)
    d.received(p, 0)
    d.received(frame(0, start=True, end=True, data=b'x'), 1)
    d.received(p, 2)
    d.received(frame(0, start=True, end=True, data=b'x'), 3)
    assert len(up.received) == 1


@pytest.mark.parametrize('kind', range(9, 15))
def test_unknown_type_needs_only_generic_crc_header(kind):
    head = bytes([kind << 4, 0, 0, (1 << 5) | 6, 0x12])
    raw = D.SYNC + head + D.crc16(head).to_bytes(2, 'little')
    d, _ = transfer()
    for p in D.Parser().push(raw):
        d.received(p, 0)
    assert d.compose(0, 100)[0].reason == 0
    assert not D.Parser().push(raw[:-1] + bytes([raw[-1] ^ 1]))


def test_master_changes_rate_after_accepted_ack_is_sent():
    n = Node(1, Timing(1200))
    n.dts.connection_made(2)
    n.dts.request_rate(2, 2400, 'short')
    n.poll(0)
    n.dts.received(D.DPDU(D.MANAGEMENT, 1, 2, valid=True, ack=True,
                          eow=D.eow_drc_response(D.RESP_ACCEPT)), 10)
    assert n.timing.rate == 1200
    n.poll(10, defer=True)
    assert n.timing.rate == 1200
    n.sent(False)
    assert n.timing.rate == 1200
    n.poll(12)
    assert n.timing.rate == 2400


def test_break_handshake_finishes_before_next_soft_link():
    n = Node(1, Timing(1200))
    att, _ = bind(n)
    submit(att, dest=3)
    n.cas.links[2] = C.PhysicalLink(2, False, True, 'breaking')
    n.sis.tick(0)
    assert 3 not in n.cas.links


def test_advertised_nonarq_deadline_is_not_extended_after_expiry():
    n = Node(1, Timing(1200), max_tx_s=2)
    n.dts.submit(T.CPDU(bytes(1000), 2, arq=False))
    n.poll(0)
    assert n.dts._nonarq_out
    n.tick(100)
    assert not n.dts._nonarq_out
    assert n.poll(100) is None


def test_reset_command_during_segmentation_rejects_once():
    d, up = transfer(segment=1)
    d.submit(T.CPDU(b'abc', 2, handle='msg'))
    d.compose(0, 17)
    d.received(D.DPDU(D.RESET_WIN_RESYNC, 1, 2, reset_win_cmnd=True,
                      new_rx_lwe=1), 1)
    d.compose(1, 1000)
    assert up.rejected == [('msg', T.Reject.RESET)]
    assert not d.links[2].arq.unacked


def test_expedited_ack_cannot_confirm_unsent_frames():
    d, up = transfer(segment=1)
    d.submit(T.CPDU(b'abc', 2, expedited=True, handle='msg'))
    d.compose(0, 17)
    d.received(D.DPDU(D.EXP_ACK_ONLY, 1, 2, rx_lwe=3), 1)
    assert not up.confirmed
    assert d.links[2].reset_out is not None


def test_invalid_bind_does_not_evict_existing_client():
    n = Node(1, Timing(1200))
    old, _ = bind(n, rank=1)
    new, got = bind(n, mode=P.MODE_DEFAULT, rank=2)
    assert old.client is not None and new.client is None
    assert got[-1].type == P.BIND_REJECTED and got[-1]['reason'] == P.BIND_NO_RESOURCES


def test_unbind_removes_queued_dts_requests():
    n = Node(1, Timing(1200))
    att, _ = bind(n, P.MODE_NON_ARQ)
    submit(att)
    n.sis.tick(0)
    assert n.dts._nonarq_out
    att.close()
    assert not n.dts._nonarq_out


def test_75_long_modem_renders_one_whole_dpdu_within_key_limit():
    from hfmodem.bateleur.modem import Mode
    from hfmodem.bateleur.s5066.onair import ModeTiming
    from hfmodem.bateleur.spec import FS
    mode = Mode('110', '75L')
    n = Node(1, ModeTiming(mode), max_tx_s=30)
    n.dts.connection_made(2)
    n.dts.submit(T.CPDU(bytes(200), 2))
    data, duration = n.poll(0)
    assert duration <= 30 and len(data) <= mode.most_bytes(30)
    assert mode.render(data).size / FS <= 30
    parser = D.Parser()
    assert parser.push(data) and not parser.buf


def test_drc_defaults_to_three_retransmissions():
    d, up = transfer()
    d.request_rate(2, 2400, 'short')
    attempts = 0
    for i in range(6):
        now = i * 10
        d.tick(now)
        if not d.connected(2):
            break
        attempts += bool(d.compose(now, 1000))
        d.transmitted(now + 1, now + 5)
    assert attempts == 4 and up.lost == [2]


def test_drc_cancellation_waits_for_its_ack():
    d, _ = transfer()
    d.request_rate(2, 2400, 'short')
    d.compose(0, 1000)
    d.received(D.DPDU(D.MANAGEMENT, 1, 2, ack=True, valid=True,
                      eow=D.eow_drc_response(D.RESP_REFUSE)), 1)
    assert d.links[2].drc is not None
    d.compose(1, 1000)
    d.received(D.DPDU(D.MANAGEMENT, 1, 2, ack=True), 2)
    assert d.links[2].drc is None


def test_any_order_placeholder_waits_for_an_earlier_gap():
    d, up = transfer()
    for seq in (0, 2, 3, 4, 5, 1):
        d.received(frame(seq, start=seq in (0, 4), end=seq in (3, 5),
                         data=bytes([seq]), in_order=False), seq)
    assert [data for data, _ in up.received] == [b'\x04\x05', b'\x00\x01\x02\x03']


def test_modem_exception_releases_prepared_turn():
    from hfmodem.bateleur.modem import BateleurModem, Mode
    from hfmodem.bateleur.s5066.onair import OnAir
    from hfmodem.bateleur.spec import FS

    def failed_send(audio, rate):
        raise RuntimeError('test send failure')

    modem = BateleurModem(failed_send, card_rate=FS, max_key_s=30)
    on = OnAir(modem, 1, Mode('110', '1200S'), clock=lambda: 0)
    modem.set_mode(Mode('110', '1200S'))
    on.node.dts.connection_made(2)
    on.node.dts.submit(T.CPDU(b'x', 2))
    with pytest.raises(RuntimeError, match='test send failure'):
        on.step()
    assert on.node.dts.links[2].deadline is None
    assert on.node.poll(2) is not None


def test_refused_drc_does_not_start_peer_timeout():
    n = Node(1, Timing(1200))
    n.dts.connection_made(2)
    n.dts.request_rate(2, 2400, 'short')
    for now in (0, 60, 120, 180):
        assert n.poll(now, defer=True)
        n.sent(False)
    assert n.dts.connected(2)
    assert n.dts.links[2].drc['until'] is None


def test_answer_deadline_uses_actual_transmission_end():
    n = Node(1, Timing(1200))
    n.dts.connection_made(2)
    n.dts.submit(T.CPDU(b'x', 2))
    n.poll(0, defer=True)
    n.sent(True, at=100)
    assert n.dts.links[2].deadline == 100 + n.answer_s


def test_errored_control_cannot_establish_hard_link():
    n = Node(1, Timing(1200))
    bind(n, P.MODE_NON_ARQ_ERRORS)
    n.cas.links[2] = C.PhysicalLink(2, False, False, 'made')
    n.dts.connection_made(2)
    n.sis.spdu_received(2, 1, b'\x30\x66', expedited=True, arq=False,
                        errors={'errored': [(1, 1)], 'missing': []})
    assert n.sis.hard is None


def test_cpdu_over_mtu_rejected_before_segmentation():
    d, up = transfer()
    assert not d.submit(T.CPDU(bytes(2055), 2, handle='large'))
    assert up.rejected == [('large', T.Reject.SIZE)]
    assert not d.links[2].arq.pending
