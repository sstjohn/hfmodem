# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Full turn rollback and journal ownership; all source fields are synthetic."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.shrike.p4turn import State, CloseContext, DataEpisode, HeldWordYield
from hfmodem.shrike.p4credit import SourceReceipt
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, FS, CYCLE, setup, packet, legacy


def turn_state(m):
    def value(v):
        if isinstance(v, dict): return {k: value(x) for k, x in v.items()}
        if isinstance(v, (list, set, tuple)): return type(v)(value(x) for x in v)
        if isinstance(v, CloseContext): return id(v), dict(vars(v))
        return v
    return {n: value(v) for n, v in vars(m).items()
            if n not in ('adapter', 'outbox', '_watch_admitted', '_watch_recovery') and not callable(v)}


def adapter_state(m):
    return deepcopy({n: getattr(m.adapter, n) for n in
        ('last', 'last_credit', 'result', '_receipts', '_active', '_changeover_receipt',
         '_source_kind', '_source_counter', '_source_width', '_covered', '_resegmented')})


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('callback', [False, True])
def test_f8_full_write_set_nested_close_and_final_ack(kind, callback):
    m, a = setup(b'TEST\r', wall_clock=lambda: 10.)
    m.update_application_progress(100.)
    m.observe(packet(), now=AT+158400)
    pending, held = m.pending[AT], m._held_answer
    legacy(m)
    # A genuine credit object in synthetic initial context makes receipt
    # restoration non-vacuous without inventing a recorded BODY observation.
    prefix = b'old receipt\r'
    receipt = SourceReceipt(m._receive_epoch, (Kind.BODY, AT-60000),
                            (0x40, prefix.ljust(31, b'\x1e')))
    m.adapter.candidate(receipt, prefix, 'changeover')
    m.adapter.arm(receipt)
    m._receipt = receipt
    episode = DataEpisode(1, KEY, Kind.BODY, 0, b'old')
    yielded = HeldWordYield(episode, KEY+44880, KEY, None)
    m._data_episode, m._held_word_yield = episode, yielded
    m.close_context = CloseContext(None, False, b'TEST\r', b'', KEY, 10., KEY+30*FS,
                                   held_word_yield=yielded, progress_at=AT)
    ctx = m.close_context
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.sync_peer_watch(m.peer_watch)
    width = 40 if kind == Kind.LONG else 22
    field = (b'X'*(width-6)+b'\x1cB7 ').ljust(width, b'\x1e')
    obs = packet(AT+CYCLE, kind, 0x42, field).observation
    with rx.scoped_clock(rx.peer_watch.clock, watch_generation=rx.peer_watch.generation):
        reg = rx.register(obs)
    permit = [callback]
    delivered, supervisor, entered = [], [], []
    def deliver(data):
        delivered.append(data)
        permit[0] = False
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *a: supervisor.append(a))
    defer = m._defer
    def tracked(*a):
        entered.append((a[0], a[1] is pending))
        return defer(*a)
    m._defer = tracked
    before, host_before = turn_state(m), adapter_state(m)
    balances = bytes(a._outbuf), m.outbox.inflight, a.tx_acked_bytes
    with m.adapter.admission(lambda: permit[0]): m.observe(reg, now=AT+CYCLE+158400)
    assert entered == [(AT, True)]
    assert turn_state(m) == before and adapter_state(m) == host_before
    assert m.pending[AT] is pending and m._held_answer is held and m.close_context is ctx
    assert m._receipt is receipt
    assert ctx.held_word_yield is yielded and m._data_episode is episode
    assert (bytes(a._outbuf), m.outbox.inflight, a.tx_acked_bytes) == balances
    journal = m.adapter._delivery_journal
    if callback:
        assert journal['token'] == ('accept', 0x42, field)
        assert journal['done'] == [('deliver', (b'X'*(width-6),))]
        # An unrelated source cannot bypass the completed effect's journal.
        unrelated = replace(reg, observation=replace(obs, frame=packet(AT+CYCLE, kind, 0x42,
                          b'unrelated'.ljust(width, b'\x1e')).observation.frame))
        permit[0] = True
        with m.adapter.admission(lambda: permit[0]): m.observe(unrelated, now=AT+CYCLE+158400)
        assert m.adapter._delivery_journal is journal and turn_state(m) == before
    else:
        assert journal is None and delivered == []
    permit[0] = True
    m.adapter.host.deliver = delivered.append
    with m.adapter.admission(lambda: permit[0]): m.observe(reg, now=AT+CYCLE+158400)
    assert delivered == [b'X'*(width-6)] and supervisor == [('B', b'7')]
    assert m.adapter._delivery_journal is None and m.state == State.IRS_ANSWER
    assert len([e for e in m.events if e is reg]) == 1
    due = KEY+CYCLE
    p = m.decide(due-m.lead)
    identity, epoch, target = p.ack_identity, p.receive_epoch, p.accepted_target
    assert target == obs.stream_sample and m.can_key(p)
    m.refused('deadline')
    assert m._deferred is not None
    p = m.decide(due+CYCLE-m.lead)
    assert (p.ack_identity, p.receive_epoch, p.accepted_target) == (identity, epoch, target)
    assert m.can_key(p)
    sent = m.keyed()
    assert sent.state == State.IRS_ANSWER and sent.accepted_target == target
    assert a.tx_acked_bytes == 0 and not m.outbox.inflight
    assert m.application_deadline == 100.
    print(dict(kind=kind, callback=callback, defer_entries=entered,
               final_key=(sent.stream_sample, sent.accepted_target, sent.receive_epoch),
               deliveries=delivered, supervisors=supervisor))


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
def test_f8_displaced_dropped_refinement_rollback(kind):
    m, a = setup()
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    rx = P4Receiver(FS, admitted=True, chirp=True)
    from hfmodem.shrike.p4receiver import Clock, Expect
    rx.clock = Clock(Expect.PACKET, AT+CYCLE, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    width = 40 if kind == Kind.LONG else 22
    full = packet(AT+CYCLE+1, kind, 0x42, b'refined\r'.ljust(width, b'\x1e')).observation
    header = rx.register(replace(full, stream_sample=AT+CYCLE, frame=None, provisional=True))
    m.observe(header, now=AT+CYCLE+40000)
    m.decide(KEY+CYCLE-m.lead)
    assert AT+CYCLE in m._decided and AT+CYCLE in m._dropped
    with rx.scoped_clock(rx.peer_watch.clock, watch_generation=rx.peer_watch.generation):
        reg = rx.register(full)
    assert reg.replaced is header.observation
    before = turn_state(m)
    migration = []
    def deny():
        migration.append((full.stream_sample in m.pending, full.stream_sample in m._decided,
                          full.stream_sample in m._dropped, AT+CYCLE in m.pending))
        return False
    with m.adapter.admission(deny): m.observe(reg, now=AT+CYCLE+158400)
    assert migration == [(True, True, True, False)]
    assert turn_state(m) == before and not delivered
    with m.adapter.admission(lambda: True): m.observe(reg, now=AT+CYCLE+158400)
    assert delivered == [b'refined\r'] and a.tx_acked_bytes == 0
    m.observe(rx.register(full), now=AT+CYCLE+158400)
    assert delivered == [b'refined\r']


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('outstanding', [False, True])
def test_f8_callback_time_fq_termination_and_exact_journal_retry(kind, failure, outstanding):
    from hfmodem.winlink import B2FSession, MailClient
    from hfmodem.winlink.session import _THEIR_TURN
    from hfmodem.shrike.onair import _p4_mail_progress
    m, a = setup(b'TEST\r' if outstanding else b'')
    m.observe(packet(), now=AT+158400)
    pending = m.pending[AT]
    if outstanding:
        m.decide(KEY-m.lead)
        m.keyed()
        m.silence(KEY+60000)
        captured = m._next
        assert captured is not None and captured.word == 'Y27' and m.can_key(captured)
        m.allow_breakin = False
    else:
        legacy(m)
        captured = m.last
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.session._state = _THEIR_TURN
    delivered, supervisors = [], []
    permit = [True]
    def deliver(data):
        delivered.append(data)
        assert m.state == State.IRS_ANSWER
        client.on_link_data(data)
        if failure: client.session._fail('callback terminal failure')
        _p4_mail_progress(client, m, AT+CYCLE+158400, drained=True)
        permit[0] = False
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *a: supervisors.append(a))
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.sync_peer_watch(m.peer_watch)
    width = 40 if kind == Kind.LONG else 22
    field = (b'FQ\r'+b'\x1cB7 ').ljust(width, b'\x1e')
    with rx.scoped_clock(rx.peer_watch.clock, watch_generation=rx.peer_watch.generation):
        reg = rx.register(packet(AT+CYCLE, kind, 0x42, field).observation)
    with m.adapter.admission(lambda: permit[0]): m.observe(reg, now=AT+CYCLE+158400)
    assert delivered == [b'FQ\r'] and m.app_done and m.peer_watch is None
    assert m._next is not captured
    assert m._next is None or m._next.peer_watch_generation is None
    assert m.adapter._delivery_journal['done'] == [('deliver', (b'FQ\r',))]
    if failure:
        assert m._closing_failure and not m.pending and not m.can_key(captured)
        return
    assert m.state == State.IRS_ANSWER and m.pending[AT] is pending
    permit[0] = True
    with m.adapter.admission(lambda: permit[0]): m.observe(reg, now=AT+CYCLE+158400)
    assert delivered == [b'FQ\r'] and supervisors == [('B', b'7')]
    assert m.adapter._delivery_journal is None and m.peer_watch is None
    _p4_mail_progress(client, m, KEY+CYCLE-m.lead, drained=True)
    p = m.decide(KEY+CYCLE-m.lead)
    assert p.accepted_target == AT+CYCLE and p.state == State.IRS_ANSWER and m.can_key(p)
    m.refused('deadline')
    p = m.decide(KEY+2*CYCLE-m.lead)
    assert p.accepted_target == AT+CYCLE and m.can_key(p)
    m.keyed()
    assert a.tx_acked_bytes == 0 and m.peer_watch is None
