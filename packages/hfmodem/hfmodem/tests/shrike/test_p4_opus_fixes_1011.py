# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""F1–F8 regression controls; semantic inputs are explicitly synthetic."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE, FS, setup, packet, legacy
from hfmodem.tests.shrike.test_p4turn import word


@pytest.mark.parametrize('breakin', ['body', 'y27'])
@pytest.mark.parametrize('eligibility', ['empty', 'gated', 'disabled', 'eligible'])
def test_f1_unassociated_empty_body_preserves_long_answer(breakin, eligibility):
    m, a = setup(breakin=breakin)
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    if eligibility != 'empty': a.on_host_data(b'TEST\r')
    m.allow_breakin = eligibility != 'gated'
    m.sends_data = eligibility != 'disabled'
    epoch = m._receive_epoch
    # Accepted unassociated checked BODY, through the physical registry.
    rx = P4Receiver(FS, admitted=True)
    rx.clock = Clock(Expect.PACKET, KEY+60000, .330, 3.75, -.19)
    head = rx.register(word('Y27', KEY+60000, extent=.245).observation)
    m.observe(head, now=KEY+60000+12000)
    r = rx.register(packet(KEY+60000, Kind.BODY, 0x40).observation)
    assert r.outcome == Registered.ENRICHED and r.responds_to is None
    m.observe(r, now=KEY+60000+38640)
    p = m.decide(KEY+60000+44160-m.lead)
    assert p.kind == Kind.WORD
    assert (p.word, p.state, p.stream_sample) == (
        ('Y27', State.IRS_BREAKIN, KEY+CYCLE) if eligibility == 'eligible'
        else ('X27', State.IRS_ANSWER, KEY+CYCLE))
    m.keyed()
    m.observe(word('V27', KEY+CYCLE+44880, extent=.245), now=KEY+CYCLE+58000)
    assert not m.outbox.inflight and a.tx_acked_bytes == 0
    assert m._receive_epoch == epoch and m.state != State.ISS_HOLD
    assert not any(e['event'] == 'turn' for e in m.link_events)


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('recovery', ['continuation', 'exhaustion', 'gated', 'empty', 'disabled'])
def test_f3_captured_request_is_canceled_by_recovery(managed, recovery):
    m, a = setup(b'TEST\r')
    m.managed_application = managed
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    m.silence(KEY+60000)
    captured = m._next
    assert captured is not None and captured.word == 'Y27' and m.can_key(captured)
    if recovery == 'continuation':
        m.observe(packet(AT+CYCLE, Kind.LONG, 0x42,
                         peer_watch_generation=m.peer_watch.generation), now=AT+CYCLE+158400)
    elif recovery == 'exhaustion':
        for at in (KEY+CYCLE, KEY+2*CYCLE, KEY+3*CYCLE):
            m.decide(at-m.lead)
            m.keyed(at)
            m.silence(at+60000)
    else:
        if recovery == 'gated': m.allow_breakin = False
        if recovery == 'disabled': m.sends_data = False
        if recovery == 'empty': a._outbuf.clear()
        assert not m.can_key(captured)
        m.silence(KEY+60000)
    assert not m.can_key(captured)
    live = m.decide((8911870 if recovery == 'exhaustion' else KEY+CYCLE)-m.lead)
    assert live is not captured and live.word == 'X27' and live.state == State.IRS_ANSWER
    assert m.can_key(live) and a.tx_acked_bytes == 0


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('status', [0x42, 0x62])
@pytest.mark.parametrize('variant', ['crc', 'provisional', 'weak_header', 'weak_frame', 'good'])
def test_f4_coincident_primary_keeps_ordinary_receive(kind, status, variant):
    m, a = setup()
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    # Failed frames have no protocol bits: seed the actual P2/P3 peer context.
    m.observe(packet(AT, kind, status), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.clock = Clock(Expect.PACKET, AT+CYCLE, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    width = 40 if kind == Kind.LONG else 22
    o = packet(AT+CYCLE, kind, status, b'primary\r'.ljust(width, b'\x1e')).observation
    if variant in ('crc', 'provisional', 'weak_header'):
        o = replace(o, frame=None, provisional=variant == 'provisional', weak=variant == 'weak_header')
    if variant == 'weak_frame': o = replace(o, weak=True)
    r = rx.register(o)
    assert r.peer_watch_generation is None, 'primary work must keep ordinary association authority'
    m.observe(r, now=o.stream_sample+158400)
    expected_pending = kind == Kind.LONG or variant not in ('weak_header', 'weak_frame')
    assert (o.stream_sample in m.pending) == expected_pending
    if expected_pending:
        assert m.pending[o.stream_sample]['body'] is r.observation
    assert delivered == ([b'primary\r'] if variant == 'good'
                         or kind == Kind.LONG and variant == 'weak_frame' else [])
    before = m.unread
    m.decide(o.stream_sample+163200-m.lead)
    if variant in ('crc', 'provisional', 'weak_header') and expected_pending:
        assert m.unread == before+1 and o.stream_sample in m._dropped
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('terminal', [False, True])
def test_f5_revoked_header_refinement_retains_primary_source(kind, terminal):
    m, a = setup()
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.clock = Clock(Expect.PACKET, AT+CYCLE, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    g = rx.peer_watch.generation
    o = packet(AT+CYCLE, kind, 0x42, b'novel\r'.ljust(22 if kind == Kind.CHIRP else 40, b'\x1e')).observation
    with rx.scoped_clock(rx.peer_watch.clock, watch_generation=g):
        header = rx.register(replace(o, frame=None, provisional=True))
    m.observe(header, now=o.stream_sample+24000)
    if terminal: m.finish_application(o.stream_sample+24000)
    else: m.revoke_peer_watch()
    rx.sync_peer_watch(None)
    completed = rx.register(o)
    assert completed.replaced is header.observation and completed.peer_watch_generation == g
    m.observe(completed, now=o.stream_sample+158400)
    assert delivered == [b'novel\r'] and a.tx_acked_bytes == 0
    answer = m.decide(o.stream_sample+163200-m.lead)
    assert answer.state == State.IRS_ANSWER and m.can_key(answer)
    m.observe(rx.register(o), now=o.stream_sample+158400)
    assert delivered == [b'novel\r']
    if terminal: assert m.peer_watch is None


def exhausted():
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    for at in (KEY, KEY+CYCLE, KEY+2*CYCLE, KEY+3*CYCLE):
        m.decide(at-m.lead)
        m.keyed(at)
        m.silence(at+60000)
    assert m.next_offer_at == 9691870
    m.keyed(8911870)
    return m, a


@pytest.mark.parametrize('stage', ['selected', 'refused', 'keyed'])
@pytest.mark.parametrize('expiry', ['sample', 'wall'])
def test_f6_exhausted_short_grant_cannot_cancel_bounded_close(stage, expiry):
    wall = [10.]
    m, a = setup(b'TEST\r', wall_clock=lambda: wall[0])
    m.managed_application = False
    m._body(FS, 0x40, State.ISS_GRANT)
    m.keyed()
    for _ in range(3):
        m._repeat()
        m.keyed()
    old_grant = m.last
    m._repeat()
    ctx = m.close_context
    limits = ctx.end, ctx.wall_started
    if stage == 'refused': m.refused('deadline')
    if stage == 'keyed': m.keyed()
    chosen = m._next
    m.observe(word('V27', old_grant.stream_sample+44880,
                   responds_to=old_grant.stream_sample, extent=.245),
              now=old_grant.stream_sample+58000)
    assert m._next is chosen and m.close_context is ctx
    assert (ctx.end, ctx.wall_started) == limits
    assert not m.outbox.inflight and a.tx_acked_bytes == 0
    assert not any(e['event'] == 'turn' for e in m.link_events)
    wall[0] = ctx.wall_started+30 if expiry == 'wall' else wall[0]
    m.decide(ctx.end if expiry == 'sample' else old_grant.stream_sample+58000)
    assert m.state == State.DOWN and bytes(a._outbuf) == b'TEST\r'


@pytest.mark.parametrize('skip', [1, 3])
def test_f7_skipped_ack_renews_at_candidate_raster(skip):
    m, a = exhausted()
    for at in (9091870, 9271870, 9451870):
        m.silence(at-m.lead, answer_due=at)
        m.keyed(at)
    due = 9811870+(skip-1)*CYCLE
    m.reschedule_breakin(due-m.lead)
    p = m.decide(due-m.lead)
    assert p.word == 'Y27' and p.stream_sample == due and m.can_key(p)
    assert m.next_offer_at == 9691870 and m.peer_watch.attempts == 4
    m.refused('deadline')
    assert m.next_offer_at == 9691870 and m.peer_watch.attempts == 4
    p = m.decide(due+CYCLE-m.lead)
    assert p.word == 'Y27' and p.stream_sample == due+CYCLE
    m.keyed()
    assert m.peer_watch.attempts == 1 and m.next_offer_at is None and a.tx_acked_bytes == 0


@pytest.mark.parametrize('after', [1, 3])
@pytest.mark.parametrize('reason', ['deadline', 'peer protected span', 'admission'])
def test_f3_refusals_inside_actual_attempt_group(after, reason):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    due = KEY
    for i in range(after):
        m.decide(due-m.lead)
        m.keyed(due)
        m.silence(due+60000)
        due += CYCLE
    captured, watch = m._next, m.peer_watch
    m.refused(reason)
    assert m.peer_watch == watch and m.peer_watch.attempts == after
    assert m._next is not captured and m._next.stream_sample == due+CYCLE
    assert not m.can_key(captured)
    due += CYCLE
    for i in range(after, 4):
        p = m.decide(due-m.lead)
        assert p.word == 'Y27' and p.stream_sample == due
        m.keyed(due)
        m.silence(due+60000)
        assert m.peer_watch.attempts == i+1 and m.peer_watch.last_attempt == due
        due += CYCLE
    assert m._next.word == 'X27' and m._next.state == State.IRS_ANSWER
    assert m.next_offer_at == due-CYCLE+960000 and a.tx_acked_bytes == 0


def test_f7_skipped_ack_then_current_invitation_keeps_floor_until_key():
    m, a = exhausted()
    for due in (9091870,9271870,9451870):
        m.silence(due-m.lead, answer_due=due)
        m.keyed()
    rx = P4Receiver(FS, admitted=True)
    at = 9811870-163200
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    m.observe(rx.register(packet(at, Kind.LONG, 0x42).observation), now=at+158400)
    p = m.decide(9811870-m.lead)
    assert p.word == 'Y27' and p.stream_sample == 9811870 and m.can_key(p)
    assert m.next_offer_at == 9691870 and m.peer_watch.attempts == 4
    m.keyed()
    assert m.next_offer_at is None and m.peer_watch.attempts == 1 and a.tx_acked_bytes == 0


@pytest.mark.parametrize('at', [9691869, 9691870, 9691871])
def test_f3_strict_floor_survives_proposal_and_admission(at):
    m, a = exhausted()
    floor = m.next_offer_at
    m.reschedule_breakin(at-m.lead)
    p = m.decide(at-m.lead)
    assert p.stream_sample == 9811870 and p.word == 'Y27'
    assert m.next_offer_at == floor and m.peer_watch.attempts == 4
    early = replace(p, stream_sample=floor-1)
    m._next = early
    assert not m.can_key(early), 'exercise the floor guard with a current captured object'
    m._next = p
    assert m.can_key(p)
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('boundary', ['local', 'call', 'side', 'newer'])
def test_f5_obsolete_header_cannot_gain_new_receive_authority(kind, boundary):
    m, a = setup(b'TEST\r')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    m.observe(packet(), now=AT+158400)
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.clock = Clock(Expect.PACKET, AT+CYCLE, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    watch = rx.peer_watch
    width = 40 if kind == Kind.LONG else 22
    o = packet(AT+CYCLE, kind, 0x42, b'old\r'.ljust(width, b'\x1e')).observation
    with rx.scoped_clock(watch.clock, watch_generation=watch.generation):
        rx.register(replace(o, frame=None, provisional=True))
    m.revoke_peer_watch()
    rx.sync_peer_watch(None)
    if boundary == 'local':
        m._data(KEY)
        m.keyed()
    elif boundary == 'call':
        rx.new_call()
        m._close_receive_epoch('new call')
        m.peer_phase = None
        m._peer_generation += 1
    elif boundary == 'newer':
        m.observe(packet(AT+2*CYCLE, Kind.LONG, 0x42), now=AT+2*CYCLE+158400)
    if boundary == 'call':
        reg = replace(rx.register(o), peer_watch_generation=watch.generation)
    elif boundary == 'side':
        with rx.scoped_clock(watch.clock, watch_generation=watch.generation): reg = rx.register(o)
    else:
        reg = rx.register(o)
    before = (m.state, m._next, m.peer_watch, m.peer_phase, m.next_offer_at)
    m.observe(reg, now=AT+CYCLE+158400)
    assert (m.state, m._next, m.peer_watch, m.peer_phase, m.next_offer_at) == before
    assert not delivered and a.tx_acked_bytes == 0


def transaction(m):
    names = ('state', '_next', '_accepted_proposal', '_held_answer', '_deferred',
             '_decided', '_dropped', 'events', 'link_events', 'pending', '_seen',
             '_receive_epoch', '_receive_active', '_receive_boundary', '_root_epoch',
             '_observation_roots', '_accepted_roots', '_last_source_onset',
             'peer_watch', 'peer_phase', '_peer_consumed', '_offer_at', 'next_offer_at',
             '_waiting_ack', '_waiting_until', '_waiting_observation', '_deferred_ack',
             '_deferred_slot', '_held_word_yield', '_data_episode', '_peer_word', '_peer_word_at',
             '_train_heads', '_irs_silent_start', '_irs_silent_until', '_irs_silent_done',
             '_unsupported_copies', '_close_held_v27')
    def value(v):
        if isinstance(v, dict): return {k: value(x) for k, x in v.items()}
        if isinstance(v, (list, set, tuple)): return type(v)(value(x) for x in v)
        return v
    return {n: value(getattr(m, n)) for n in names}


@pytest.mark.parametrize('after_callback', [False, True])
def test_f8_denied_recovery_restores_deferred_transaction(after_callback):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    old_context = m.pending[AT]
    old_answer = old_context['answer']
    legacy(m)  # Key without deciding: the older accepted answer is still pending.
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.sync_peer_watch(m.peer_watch)
    source = packet(AT+CYCLE, Kind.LONG, 0x42,
                    (b'X'*34+b'\x1cB7 ').ljust(40, b'\x1e')).observation
    with rx.scoped_clock(rx.peer_watch.clock, watch_generation=rx.peer_watch.generation):
        reg = rx.register(source)
    permit = [after_callback]
    deliveries, supervisors, entered = [], [], []
    def deliver(data):
        deliveries.append(data)
        permit[0] = False
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *a: supervisors.append(a))
    original = m._defer
    def defer(*a):
        entered.append(a[0])
        return original(*a)
    m._defer = defer
    before = transaction(m)
    with m.adapter.admission(lambda: permit[0]):
        m.observe(reg, now=AT+CYCLE+158400)
    assert entered == [AT], 'the denial must follow real _defer branch entry'
    assert transaction(m) == before
    assert m.pending[AT] is old_context and m.pending[AT]['answer'] is old_answer
    assert a.tx_acked_bytes == 0 and m.outbox.inflight == b''
    journal = m.adapter._delivery_journal
    assert (journal is not None) == after_callback
    permit[0] = True
    m.adapter.host.deliver = deliveries.append
    with m.adapter.admission(lambda: permit[0]):
        m.observe(reg, now=AT+CYCLE+158400)
    assert deliveries == [b'X'*34] and supervisors == [('B', b'7')]
    assert m.adapter._delivery_journal is None and m.state == State.IRS_ANSWER
    assert len([e for e in m.events if e is reg]) == 1
    assert a.tx_acked_bytes == 0


# IDENTITY-SCOPE-AMENDMENT-1011: ordinary protocol-0 compatibility controls.
def ordinary_captured_request(*, managed=False):
    from hfmodem.tests.shrike.test_p4turn import machine
    from hfmodem.tests.shrike.test_p4_greeting_resegment_1010 import packet as ordinary_packet
    m, a = machine(role='irs', data='body', ladder='none', prekey_lead=.025,
                   managed_application=managed)
    reg = ordinary_packet(Kind.LONG, 10., 0x01, b'peer source', width=69)
    assert reg.observation.frame.status.protocol == 0
    m.observe(reg, now=10*FS)
    due = 10*FS+round(m.t[Kind.LONG]*FS)
    answer = m.decide(due-m.lead)
    assert answer.state == State.IRS_ANSWER and m.can_key(answer)
    m.keyed()
    a.on_host_data(b'our mail')
    m._break_in(due+CYCLE)
    captured = m.decide(due+CYCLE-m.lead)
    assert captured.word == 'Y27' and captured.peer_watch_generation is not None
    assert m.can_key(captured)
    return m, a, captured


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('replacement', ['equal', 'retimed'])
def test_identity_scope_ordinary_compatible_replacements(managed, replacement):
    m, a, captured = ordinary_captured_request(managed=managed)
    if replacement == 'equal':
        m._next = replace(captured)
    else:
        m.reschedule_breakin(captured.stream_sample-m.lead+1)
        assert m._next.stream_sample == captured.stream_sample+CYCLE
    current = m._next
    assert current is not captured
    assert replace(captured, stream_sample=current.stream_sample) == current
    assert m.can_key(current)
    assert m.can_key(captured) is (not managed)
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('field,value', [
    ('kind', Kind.BODY), ('word', 'V27'), ('value', 13), ('header', 'B'),
    ('payload', b'different render bytes'), ('status', 0x40),
    ('state', State.ISS_GRANT), ('index', 1), ('repeat', True),
    ('ack_identity', (1, b'peer source')), ('ack_reason', 'other answer'),
    ('retry_of', 123), ('peer_control', 456), ('responds_to', 789),
    ('chirp_slot', 1), ('receive_epoch', 999), ('source_root', ('other', 1)),
    ('purpose', 'other purpose'), ('accepted_target', 123), ('current_slot', 456),
    ('current_slot_accepted', False), ('peer_watch_generation', -1),
])
def test_identity_scope_every_nonplacement_field_must_match(field, value):
    from dataclasses import fields
    m, a, captured = ordinary_captured_request()
    compatible = replace(captured)
    m._next = compatible
    assert compatible is not captured and m.can_key(captured)
    assert getattr(compatible, field) != value
    m._next = replace(compatible, **{field: value})
    assert [f.name for f in fields(captured)
            if getattr(captured, f.name) != getattr(m._next, f.name)] == [field]
    watch, floor = m.peer_watch, m.next_offer_at
    assert not m.can_key(captured)
    assert m.peer_watch == watch and m.next_offer_at == floor
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('current', ['absent', 'held_answer'])
def test_identity_scope_same_generation_recovery_cancels_ordinary_capture(current):
    m, a, captured = ordinary_captured_request()
    generation = m.peer_watch.generation
    assert m._held_answer is not None
    m._recover_peer_turn(None if current == 'absent' else captured.stream_sample)
    assert m.peer_watch.generation == generation
    if current == 'absent':
        assert m._next is None
    else:
        assert m._next.state == State.IRS_ANSWER and m._next.word != 'Y27'
        assert m.can_key(m._next)
    assert not m.can_key(captured) and a.tx_acked_bytes == 0


def test_identity_scope_captured_floor_is_not_normalized():
    from unittest.mock import patch
    m, a, captured = ordinary_captured_request()
    m._next = replace(captured, stream_sample=captured.stream_sample+CYCLE)
    assert m.can_key(m._next) and m.can_key(captured)
    m.next_offer_at = captured.stream_sample+1
    current, watch = m._next, m.peer_watch
    with patch.object(m, '_wants_turn', side_effect=AssertionError('floor must reject first')):
        assert not m.can_key(captured)
    assert m._next is current and m.peer_watch == watch
    assert m.next_offer_at == captured.stream_sample+1 and a.tx_acked_bytes == 0


@pytest.mark.parametrize('cancellation', ['continuation', 'exhaustion', 'gated', 'empty',
                                         'disabled', 'revoked', 'done', 'failure'])
def test_identity_scope_protocol0_compatibility_does_not_bypass_cancellation(cancellation):
    from hfmodem.tests.shrike.test_p4_greeting_resegment_1010 import packet as ordinary_packet
    m, a, captured = ordinary_captured_request()
    m._next = replace(captured)
    assert m.can_key(captured)
    generation = m.peer_watch.generation
    if cancellation == 'continuation':
        m.keyed()
        m.silence(captured.stream_sample+60000)
        retry = m._next
        assert retry is not None and m.can_key(retry)
        reg = ordinary_packet(Kind.LONG, 13.75, 0x02, b'new peer source', width=69)
        m.observe(replace(reg, peer_watch_generation=generation),
                  now=round(13.75*FS)+158400)
        assert m.state == State.IRS_ANSWER and m.peer_watch.generation != generation
        assert not m.can_key(retry)
    elif cancellation == 'exhaustion':
        for i in range(4):
            due = captured.stream_sample+i*CYCLE
            p = m.decide(due-m.lead)
            assert p.word == 'Y27' and m.can_key(p)
            m.keyed(due)
            m.silence(due+60000)
        assert m.peer_watch.generation == generation and m.peer_watch.attempts == 4
        assert m._next.state == State.IRS_ANSWER and m.can_key(m._next)
    elif cancellation in ('gated', 'empty', 'disabled'):
        if cancellation == 'gated': m.allow_breakin = False
        if cancellation == 'empty': a._outbuf.clear()
        if cancellation == 'disabled': m.sends_data = False
        assert not m.can_key(captured)
        assert m._next.state == State.IRS_ANSWER and m.can_key(m._next)
    else:
        if cancellation == 'revoked': m.revoke_peer_watch()
        else: m.finish_application(captured.stream_sample,
                     failure='terminal failure' if cancellation == 'failure' else None)
        assert m.peer_watch is None
        m._next = replace(captured)  # Equal retained work cannot revive revoked authority.
    assert not m.can_key(captured) and a.tx_acked_bytes == 0
