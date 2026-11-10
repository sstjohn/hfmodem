# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A solicited NAK may change record size, never the unconfirmed byte stream."""
import pytest
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as F, vara_mfsk as M
from .test_data_over_gate import _connected
from .test_idle_poll import clock, _Clock  # noqa: F401
from .test_limited_responder_query_recovery import on_air
from .test_responder_control_stream import feed
from .test_wide_answer_query import pending as answerer_pending, final_pending, receiver


def pending(bw='2300', n=393):
    hs,io=_connected(bw=bw);hs.turn=VA._TURN_OURS
    data=bytes(i%251 for i in range(n));hs.send(data)
    hs._retry_data_over()
    return hs,io,data


def unconfirmed(hs):
    b=hs._tx_pending[0]
    return phy.vara_payload(b,caller=hs.caller,body_len=len(b))+b''.join(hs._txq)


@pytest.mark.parametrize('bw,records', [('2300', (1, 0, 0)), ('2750', (101, 100, 100))])
def test_free_wide_missing_data_ladder_holds_only_this_delivery(bw, records):
    hs, io, data = pending(bw, 393)
    assert not hs._link_high_speed
    base = hs._rec500
    hs.send(b'next host write')
    for record in records:
        hs._retry_data_over()
        hs._took_responder_nak()
        assert hs._tx_pending[2] == record, io.msgs
        assert hs._tx_pending[1] == 1
        assert unconfirmed(hs) == data + b'next host write'
        assert hs._rec500 == base
    delivered = bytearray(phy.vara_payload(hs._tx_pending[0], caller=hs.caller,
                                         body_len=len(hs._tx_pending[0])))
    hs._continue_frame = (hs._tx_pending, 0)
    hs._took_over_continue()
    body, over, level = hs._tx_pending
    assert over == 2 and level == records[-1]
    while not phy.over_is_last(hs._tx_pending[0], hs.caller):
        body, _, level = hs._tx_pending
        take = phy.body_size(bw, level) - 1
        rest = len(data) - len(delivered) - take
        assert level == records[-1]
        # ks-q3-fit-32200-1, 2026-10-31: held retries announce payload fit.
        assert phy.over_field(body) == 0x81 | phy.fit_code(bw, phy.host_level(bw, level),
                                                        rest, 1, phy.TOP[bw]) << 2
        delivered.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    body, _, level = hs._tx_pending
    assert level == records[-1]
    delivered.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
    assert bytes(delivered) == data
    hs._took_control_burst()
    hs._turn_granted((), said="session-drained-responder")
    assert hs._tx_pending[2] == phy.base_level(bw)
    assert unconfirmed(hs) == b'next host write'
    assert hs._rec500 == base


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('n', [256, 1024])
@pytest.mark.parametrize('frame', [0, 4])
def test_free_wide_retry_count_allows_peer_to_continue_up(bw, n, frame):
    hs, io, data = pending(bw, n)
    base = phy.base_level(bw)
    assert hs.role == 'initiator' and not hs._link_high_speed
    assert hs._tx_pending[1:] == (1, base)
    hs._took_responder_nak()
    body, over, level = hs._tx_pending
    assert (over, level) == (1, base - 2), io.msgs
    take = phy.body_size(bw, level) - 1
    # ks-q3-fit-6000-1, 2026-10-31: one-frame tests precede two-frame tests.
    assert phy.over_field(body) == 0x81 | phy.fit_code(bw, phy.host_level(bw, level),
                                                    n - take, 1, phy.TOP[bw]) << 2
    assert unconfirmed(hs) == data
    delivered = bytearray(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
    hs._continue_frame = (hs._tx_pending, frame)
    hs._took_over_continue()
    expected = base if frame == 4 else base - 2
    assert hs._tx_pending[1:] == (2, expected), io.msgs
    assert hs._rec500 == base
    while True:
        body, _, level = hs._tx_pending
        delivered.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        if phy.over_is_last(body, hs.caller):
            break
        assert level == expected
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    assert bytes(delivered) == data
    hs._took_control_burst()
    assert hs._tx_pending is None and not hs._txq
    assert hs.payload_ledger.confirmed == n and hs.payload_ledger.pending == 0


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_free_wide_held_over_climbs_on_a_frame_6_query_answer(bw):
    # Xara kit, -10 dB BW2300, 2026-11-01: the held overs behind a confirmed
    # lowered retry carry the countdown, and the peer answers their query 288/373.
    hs, io, data = pending(bw, 393)
    base = phy.base_level(bw)
    hs._took_responder_nak()
    assert hs._tx_pending[2] == base - 2
    hs._retry_data_over()
    hs._matched_intermediate_answer = (
        hs._tx_pending, VA.VF.for_bw(VA.VF.SESSION_INTERMEDIATE_QUERY_ANSWER, bw))
    hs._took_intermediate_query_answer()
    held = hs._tx_pending
    assert held[1:] == (2, base - 2) and hs._tx_retries == 0, io.msgs
    assert phy.over_field(held[0]) != 0x89
    confirmed = hs.payload_ledger.confirmed
    hs._retry_data_over()
    assert any(m.startswith('intermediate query 1/') and 'over #2' in m for m in io.msgs), io.msgs
    answer = M.synth_burst(hs.called, VA.VF.for_bw(VA.VF.SESSION_RETRY_QUERY_ANSWER_B, bw))
    feed(hs, on_air(answer))
    assert any('(288/373) — acknowledged over #2' in m for m in io.msgs), io.msgs
    assert hs.payload_ledger.confirmed == confirmed + len(held[0]) - 1
    assert hs._tx_pending[1:] == (3, base - 1), io.msgs
    assert not any('closing without blind' in m for m in io.msgs), io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('naks', [1, 2])
@pytest.mark.parametrize('frame,kind', [(4, F.SESSION_RETRY_QUERY_ANSWER),
                                      (6, F.SESSION_RETRY_QUERY_ANSWER_B)])
def test_held_query_answer_steps_from_its_record_and_preserves_deliveries(bw, naks, frame, kind):
    hs, io, data = pending(bw, 393)
    hs.send(b'next host write')
    for _ in range(naks):
        hs._retry_data_over()
        hs._took_responder_nak()
    hs._continue_frame = (hs._tx_pending, 0)
    hs._took_over_continue()
    held = hs._tx_pending
    base = phy.base_level(bw)
    assert held[2] == base - naks - 1 and hs._tx_retries == 0
    confirmed = hs.payload_ledger.confirmed
    hs._retry_data_over()
    feed(hs, on_air(M.synth_burst(hs.called, F.for_bw(kind, bw))))
    expected = held[2] + (2 if frame == 4 else 1)
    assert hs._tx_pending[1:] == (held[1] + 1, expected), io.msgs
    confirmed += len(held[0]) - 1
    assert hs.payload_ledger.confirmed == confirmed
    sent = bytearray()
    while True:
        body, _, level = hs._tx_pending
        assert level == expected
        sent.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        if phy.over_is_last(body, hs.caller):
            break
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    assert bytes(sent) == data[confirmed:]
    assert hs._txq == [b'next host write']
    hs._took_control_burst()
    assert hs.payload_ledger.confirmed == len(data)
    hs._turn_granted((), said="session-drained-responder")
    body, _, level = hs._tx_pending
    assert level == expected
    assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) == b'next host write'


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('n', [178, 393, 445])
def test_retry_query_answer_returns_a_long_held_remainder_to_base_whole(bw, n):
    hs, io, data = pending(bw, n)
    hs.send(b'next host write')
    hs._retry_data_over()
    hs._took_responder_nak()
    assert len(hs._tx_recovery_levels) > 1
    hs._retry_data_over()
    hs._matched_intermediate_answer = (
        hs._tx_pending, VA.VF.for_bw(VA.VF.SESSION_RETRY_QUERY_ANSWER, bw))
    hs._took_intermediate_query_answer()
    assert hs._tx_pending[1] == 2 and hs._tx_pending[2] == phy.base_level(bw)
    assert not hs._tx_recovery_levels
    sent = bytearray()
    while True:
        body = hs._tx_pending[0]
        sent.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        if phy.over_is_last(body, hs.caller):
            break
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    assert bytes(sent) == data[22:]
    assert hs._txq == [b'next host write']


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
def test_short_remainder_holds_lower_record_then_next_delivery_reenters_base(bw, monkeypatch):
    hs,io,data=pending(bw,94)
    hs.send(b'next host write')
    hs._took_responder_nak()
    delivered=bytearray()
    while hs._tx_recovery_levels:
        body,_,level=hs._tx_pending
        assert level==phy.base_level(bw)-2
        delivered.extend(phy.vara_payload(body,caller=hs.caller,body_len=len(body)))
        # ks-q3-fit-6000-1, 2026-10-31: the short remainder uses the same fit rule.
        assert phy.over_field(body) == 0x81 | phy.fit_code(bw, phy.host_level(bw, level),
            len(data) - len(delivered), 1, phy.TOP[bw]) << 2
        hs._took_over_continue()
    body,_,level=hs._tx_pending
    assert level==phy.base_level(bw)-2
    assert phy.over_is_last(body,hs.caller)
    delivered.extend(phy.vara_payload(body,caller=hs.caller,body_len=len(body)))
    assert bytes(delivered)==data
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    kinds = []
    io.tx_metadata = lambda **metadata: kinds.append(metadata["packet_kind"])
    hs._took_control_burst()
    assert kinds == ["session-turn-release"]
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert hs._txq == [b'next host write'] and hs._ask_owed
    now[0] += .294
    hs.idle_keepalive()
    assert kinds == ["session-turn-release"]
    now[0] += .002
    hs.idle_keepalive()
    assert kinds == ["session-turn-release"]
    hs._took_poll()
    assert kinds == ["session-turn-release", "session-turn-request"]
    assert hs._tx_pending is None and hs.turn == VA._TURN_ASKED
    assert not hs._ask_owed
    hs._turn_granted((), said="session-drained-responder")
    assert kinds[-1] == "data"
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


@pytest.mark.parametrize('bw,records', [('2300', (1, 0, 0)), ('2750', (101, 100, 100)),
                                        ('500', (0, 0, 0))])
@pytest.mark.parametrize('n', [393, 9])
@pytest.mark.parametrize('long_nak', [False, True], ids=['short-nak', 'long-nak'])
@pytest.mark.parametrize('high_speed', [False, True], ids=['speed-limited', 'high-speed'])
def test_responder_missing_data_follows_caller_ladder(bw, records, n, long_nak,
                                                     high_speed, clock):  # noqa: F811
    data = bytes(i % 251 for i in range(n)) if n == 393 else b'last over'
    hs, io = answerer_pending(bw, data) if n == 393 else final_pending(bw)
    hs._link_high_speed = high_speed
    caller, _ = receiver(hs)
    caller._expect_level = 4
    base = hs._rec500
    over = hs._tx_pending[1]
    keyed = []
    io.tx_metadata = lambda **metadata: keyed.append(metadata)
    kind = F.SESSION_OVER_NAK if long_nak else F.SESSION_OVER_NAK_SHORT
    reply = on_air(M.synth_burst(hs.called, F.for_bw(kind, bw)))
    observed, predicted = [], []
    for k, record in enumerate(records, 1):
        caller._advance_resend(missing=True)
        expected = phy.level_key(bw, VA._resend_level(bw, 4, (True,) * k))
        assert phy.level_key(bw, caller._expect_level) == expected == record
        if high_speed:
            predicted.append(hs._data_nak_level(missing=True))
        _Clock.t = hs._pending_answer_at + VA._ANSWER_QUERY_S
        assert hs._query_answerer_over(), io.msgs
        assert hs._caller_over_nak_fresh()
        feed(hs, reply)
        observed.append(hs._tx_pending[2])
        assert keyed[-1]['packet_kind'] == 'data' and keyed[-1]['retry']
        assert keyed[-1]['speed_level'] == hs._tx_pending[2]
        assert hs._tx_pending[1] == over and hs._tx_retries == k
        assert hs._answer_query_for is None
        assert unconfirmed(hs) == data
        assert hs.payload_ledger.confirmed == 0 and hs.payload_ledger.pending == n
        if bw != '500' and not high_speed:
            assert hs._rec500 == base
    assert tuple(observed) == records, io.msgs
    if high_speed:
        assert tuple(predicted) == records
