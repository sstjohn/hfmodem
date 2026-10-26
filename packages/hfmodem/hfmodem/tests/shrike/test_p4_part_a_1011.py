# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Approved long-peer changeover: semantic controls, not recorded air events."""
from dataclasses import replace
import pytest
from hfmodem.shrike import p4frame
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.shrike.p4turn import State, Proposal
from hfmodem.tests.shrike.test_p4turn import FS, machine, word

AT = 8028670
KEY = 8191870
CYCLE = 180000


def packet(at=AT, kind=Kind.CHIRP, status=0x41, data=None, **kw):
    data = data if data is not None else b'\x1e' * (22 if kind == Kind.CHIRP else 31 if kind == Kind.BODY else 40)
    assert len(data) == {Kind.CHIRP: 22, Kind.BODY: 31, Kind.LONG: 40, Kind.NORMAL: 128}[kind]
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)), data_bytes=len(data))
    assert frame is not None and frame.data == data
    return RegisteredObservation(Observation(kind, FS, 0, at, at, 3.28, -.19, .9, frame=frame), Registered.NEW, **kw)


def setup(queued=b'', **kw):
    m, a = machine(role='irs', data='body', ladder='none', prekey_lead=.025,
                   managed_application=True, **kw)
    m.first_body = False
    if queued:
        a.on_host_data(queued)
    return m, a


@pytest.mark.parametrize('gated', [False, True])
@pytest.mark.parametrize('kind,status,policy,expected', [
    (Kind.CHIRP, 0x41, 'fixed-x27', 'X27'),
    (Kind.LONG, 0x42, 'fixed-x27', 'X27'),
    (Kind.LONG, 0x42, 'counter-even-t27', 'T27'),
    (Kind.LONG, 0x41, 'counter-odd-t27', 'T27')])
def test_no_eligible_long_invitation(gated, kind, status, policy, expected):
    m, a = setup(b'TEST\r' if gated else b'', long_ack=policy)
    m.allow_breakin = not gated
    # A historical cached empty grant must not win over this accepted answer.
    m._last_ack = ((status, b'\x1e'*(22 if kind == Kind.CHIRP else 40)),
                   Proposal(Kind.BODY, KEY, status=0x40, state=State.ISS_GRANT))
    for at in (AT, AT+CYCLE):
        m.observe(packet(at, kind, status), now=at+158400)
        p = m.decide(at+163200-m.lead)
        assert (p.kind, p.word, p.state, p.stream_sample) == (Kind.WORD, expected, State.IRS_ANSWER, at+163200)
        epoch = m._receive_epoch
        m.keyed()
        assert m._receive_epoch == epoch and m._receive_active == (epoch > 0)
    assert m.grants == 0 and not m.outbox.inflight and a.tx_acked_bytes == 0


@pytest.mark.parametrize('breakin', ['body', 'y27'])
@pytest.mark.parametrize('release', ['V27', 'BODY'])
def test_bare_long_request_and_distinct_release(breakin, release):
    m, a = setup(b'TEST\r', breakin=breakin)
    m.observe(packet(), now=AT+158400)
    p = m.decide(KEY-m.lead)
    assert (p.kind, p.word, p.state, p.stream_sample) == (Kind.WORD, 'Y27', State.IRS_BREAKIN, KEY)
    epoch = m._receive_epoch
    m.keyed()
    assert m._receive_epoch == epoch
    grant = KEY+44880
    r = (word('V27', grant, responds_to=KEY, extent=.245) if release == 'V27' else
         packet(grant, Kind.BODY, 0x40, responds_to=KEY))
    m.observe(r, now=grant+round((.245 if release == 'V27' else .805)*FS))
    due = 8371870 if release == 'V27' else 8280910
    p = m.decide(due-m.lead)
    assert (p.kind, p.status, p.stream_sample, p.payload) == (Kind.BODY, 0x40 if release == 'V27' else 0, due, b'TEST\r')
    assert m.peer_watch is None and a.tx_acked_bytes == 0
    m.keyed()
    m.observe(word('X27', due+44880, responds_to=due, extent=.245), now=due+58000)
    assert a.tx_acked_bytes == 5


def legacy(m):
    m._next = Proposal(Kind.BODY, KEY, status=0x40, state=State.ISS_GRANT)
    return m.keyed()


def test_legacy_emission_has_independent_watch():
    m, _ = setup()
    m.observe(packet(), now=AT+158400)
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    legacy(m)
    rx.note_emission(Emission(Kind.BODY, Role.ISS, KEY, .800))
    assert rx.clock.expects == Expect.WORD and rx.clock.next_due == 8236750
    assert rx.peer_watch.clock.next_due == 8208670
    generation = m.peer_watch.generation
    m.observe(packet(AT+CYCLE, Kind.LONG, 0x42, peer_watch_generation=generation), now=AT+CYCLE+158400)
    assert m.state == State.IRS_ANSWER and m.next_offer_at == KEY+960000
    assert m.decide(8371870-m.lead).word == 'X27'


def test_four_actual_keys_and_strict_cooldown():
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    p = m.decide(KEY-m.lead)
    m.refused('deadline')
    assert m.peer_watch.attempts == 0 and m._next.stream_sample == KEY+CYCLE
    # Fresh group gives the exact approved four-key pin.
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    for i in range(4):
        at = KEY+i*CYCLE
        p = m.decide(at-m.lead)
        assert p.stream_sample == at and p.word == 'Y27'
        m.keyed(at)
        m.silence(at+60000)
    p = m.decide(8911870-m.lead)
    assert (p.state, p.word, p.stream_sample) == (State.IRS_ANSWER, 'X27', 8911870)
    assert m.next_offer_at == 9691870 and m.peer_watch.attempts == 4
    m.keyed()
    for at in range(9091870, 9811871, CYCLE):
        m.silence(at-m.lead, answer_due=at)
        p = m.decide(at-m.lead)
        assert p.word == ('Y27' if at == 9811870 else 'X27')
        m.keyed()
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('mutation', ['stale', 'phase', 'weak', 'provisional', 'crc', 'duplicate', 'revoked'])
def test_watch_recovery_requires_all_qualifiers(mutation):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    legacy(m)
    gen = m.peer_watch.generation
    r = packet(AT+CYCLE, Kind.LONG, 0x42, peer_watch_generation=gen)
    if mutation == 'stale': r = replace(r, peer_watch_generation=gen-1)
    if mutation == 'phase': r = replace(r, observation=replace(r.observation, stream_sample=AT+CYCLE+1681))
    if mutation in ('weak', 'provisional'): r = replace(r, observation=replace(r.observation, **{mutation: True}))
    if mutation == 'crc': r = replace(r, observation=replace(r.observation, frame=None))
    if mutation == 'duplicate': r = replace(r, outcome=Registered.DUPLICATE)
    if mutation == 'revoked': m.revoke_peer_watch()
    m.observe(r, now=AT+CYCLE+158400)
    assert m.state == State.ISS_GRANT and a.tx_acked_bytes == 0


@pytest.mark.parametrize('delta', [-1680, 1680])
def test_exact_phase_boundary_and_adversarial_ack(delta):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    legacy(m)
    # Even a malicious outbound association cannot turn a watch into credit.
    r = packet(AT+CYCLE+delta, Kind.LONG, 0x42,
               peer_watch_generation=m.peer_watch.generation, responds_to=KEY)
    m._data_packet = replace(m.last, payload=b'TEST\r')
    m.observe(r, now=r.observation.stream_sample+158400)
    assert m.state == State.IRS_ANSWER and a.tx_acked_bytes == 0
    assert m.peer_phase.onset == AT+CYCLE+delta
    assert m.next_offer_at == KEY+960000


def test_recovery_source_denial_retains_retry_journal():
    from types import SimpleNamespace
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    legacy(m)
    original = m.peer_watch
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    permit = [True]
    def delivery(data):
        delivered.append(data)
        permit[0] = False
    m.adapter.host.deliver = delivery
    r = packet(AT+CYCLE, Kind.LONG, 0x42, data=(b'X'*34+b'\x1cB7 ').ljust(40, b'\x1e'),
               peer_watch_generation=original.generation)
    with m.adapter.admission(lambda: permit[0]):
        m.observe(r, now=AT+CYCLE+158400)
    assert m.state == State.ISS_GRANT and m.peer_watch == original
    assert m.adapter._delivery_journal is not None and m.next_offer_at is None
    permit[0] = True
    with m.adapter.admission(lambda: permit[0]):
        m.observe(r, now=AT+CYCLE+158400)
    assert m.state == State.IRS_ANSWER and delivered == [b'X'*34]
    assert m.adapter._delivery_journal is None and a.tx_acked_bytes == 0


@pytest.mark.parametrize('terminal', ['success', 'failure', 'down', 'close'])
def test_watch_revocation_preserves_success_answer(terminal):
    m, a = setup()
    m.observe(packet(), now=AT+158400)
    legacy(m)
    r = packet(AT+CYCLE, Kind.LONG, 0x42, peer_watch_generation=m.peer_watch.generation)
    m.observe(r, now=AT+CYCLE+158400)
    p = m.decide(8371870-m.lead)
    if terminal in ('success', 'failure'):
        m.finish_application(8371870-m.lead, failure='failed' if terminal == 'failure' else None)
    elif terminal == 'down': m.link_down('test')
    else: m.observe(packet(AT+2*CYCLE, Kind.LONG, 0x21), now=AT+2*CYCLE+158400)
    assert m.peer_watch is None
    if terminal == 'success':
        assert m._next is p and m.can_key(p)
    elif terminal in ('failure', 'down'):
        assert not m.can_key(p)
    assert a.tx_acked_bytes == 0


def test_rendered_peer_span_never_fits_body_on_long_response():
    m, _ = setup()
    m.observe(packet(), now=AT+158400)
    for offset in (0, CYCLE, 2*CYCLE):
        p = Proposal(Kind.BODY, KEY+offset, status=0x40, state=State.ISS_GRANT)
        assert not m.peer_span_fits(p, round(.800*FS), round(.1*FS))
        assert m.peer_span_fits(replace(p, kind=Kind.WORD, word='Y27'), round(.244*FS), round(.1*FS))


@pytest.mark.parametrize('ownership', ['pending', 'selected', 'deferred'])
@pytest.mark.parametrize('failure', [False, True])
def test_real_fq_recovery_keeps_answer_transaction(ownership, failure):
    from types import SimpleNamespace
    from hfmodem.winlink import B2FSession, MailClient
    from hfmodem.winlink.session import _THEIR_TURN
    from hfmodem.shrike.onair import _p4_mail_progress
    m, a = setup()
    m.observe(packet(), now=AT+158400)
    legacy(m)
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.session._state = _THEIR_TURN
    delivered = []
    def deliver(data):
        delivered.append(data)
        client.on_link_data(data)
        # Exercise the synchronous callback boundary, before answer installation.
        assert m.state == State.IRS_ANSWER
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *a: None)
    onset, due = AT+CYCLE, KEY+CYCLE
    r = packet(onset, Kind.LONG, 0x42, b'FQ\r'.ljust(40, b'\x1e'),
               peer_watch_generation=m.peer_watch.generation)
    m.observe(r, now=due if ownership == 'deferred' else onset+158400)
    if ownership == 'selected': m.decide(due-m.lead)
    owned = (dict(m.pending), m._deferred, m._accepted_proposal, m._next)
    if failure: client.session._fail('terminal failure')
    assert _p4_mail_progress(client, m, due-m.lead, drained=True)
    assert m.peer_watch is None and delivered == [b'FQ\r']
    if failure:
        assert not m.pending and not m._deferred and m._closing_failure
    else:
        assert (m.pending, m._deferred, m._accepted_proposal, m._next) == owned
        p = m.decide(due+(CYCLE if ownership == 'deferred' else 0)-m.lead)
        assert p.word == 'X27' and p.state == State.IRS_ANSWER and m.can_key(p)
        m.keyed()
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('offset', [-1, 0, 1])
@pytest.mark.parametrize('legacy_attempt', [False, True])
def test_actual_keys_anchor_one_budget_and_cooldown(offset, legacy_attempt):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    for i in range(4):
        due = KEY+i*CYCLE
        p = m.decide(due-m.lead)
        if legacy_attempt and i % 2 == 0:
            m._next = replace(p, kind=Kind.BODY, word=None, status=0x40, state=State.ISS_GRANT)
        m.keyed(due+offset)
        assert m.peer_watch.attempts == i+1
        m.silence(due+60000)
    assert m.next_offer_at == KEY+3*CYCLE+offset+960000
    assert m._next.state == State.IRS_ANSWER and m._next.stream_sample == KEY+4*CYCLE
    m.keyed()
    floor = m.next_offer_at
    for at in range(KEY+5*CYCLE, KEY+10*CYCLE, CYCLE):
        m.silence(at-m.lead, answer_due=at)
        p = m.decide(at-m.lead)
        assert p.word == ('Y27' if at >= floor else 'X27')
        if at >= floor:
            assert at-CYCLE < floor
            break
        m.keyed(at+offset)
        assert m.next_offer_at == floor
    assert a.tx_acked_bytes == 0


def test_continuations_do_not_renew_cooldown_or_application_deadline():
    wall = [0.]
    m, a = setup(b'TEST\r', wall_clock=lambda: wall[0])
    m.update_application_progress(20.)
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed(KEY+1)
    for i in (1, 2, 3):
        at = AT+i*CYCLE
        m.observe(packet(at, Kind.LONG, 0x42, peer_watch_generation=m.peer_watch.generation), now=at+158400)
        assert m.next_offer_at == KEY+1+960000 and m.application_deadline == 20.
        p = m.decide(at+163200-m.lead)
        assert p.word == 'X27'
        m.keyed()
    wall[0] = 20.
    m.reschedule_breakin(9811870-m.lead)
    p = m.decide(9811870-m.lead)
    assert m.app_done and m._closing_failure and m.peer_watch is None
    assert p is None or p.payload == b''
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('reason', ['unassociated', 'stale', 'provisional', 'weak', 'head'])
def test_long_release_negative(reason):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    watch = m.peer_watch
    r = word('V27' if reason != 'head' else 'Y27', 8236750,
             responds_to=None if reason == 'unassociated' else KEY,
             extent=.245)
    if reason == 'stale': r = replace(r, response_generation=17)
    if reason in ('provisional', 'weak'): r = replace(r, observation=replace(r.observation, **{reason: True}))
    m.observe(r, now=8236750+12000)
    assert m.peer_watch is watch and a.tx_acked_bytes == 0 and not m.outbox.inflight


def test_watch_source_replay_keeps_normal_long_ledger_until_real_ownership():
    from types import SimpleNamespace
    m, a = setup()
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    source = bytes(range(32, 127))+b'abcdefghijklmno\r'
    source = source.ljust(128, b'\x1e')
    m.observe(packet(AT, Kind.NORMAL, 1, source), now=AT+40000)
    m.decide(AT+44160-m.lead)
    m.keyed()
    m.observe(packet(AT+60000, Kind.LONG, 1, source[:40]), now=AT+60000+158400)
    assert delivered == [source.rstrip(b'\x1e')]
    epoch = m._receive_epoch
    due = AT+60000+163200
    m._next = Proposal(Kind.BODY, due, status=0x40, state=State.ISS_GRANT)
    m.keyed()
    assert m._receive_epoch == epoch and len(m.adapter._covered) > m.adapter._source_width
    r = packet(AT+60000+CYCLE, Kind.LONG, 2, source[40:80], peer_watch_generation=m.peer_watch.generation)
    m.observe(r, now=r.observation.stream_sample+158400)
    assert delivered == [source.rstrip(b'\x1e')] and a.tx_acked_bytes == 0


def test_receiver_watch_provenance_scoping_and_once_per_cycle():
    m, _ = setup()
    m.observe(packet(), now=AT+158400)
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    rx.sync_peer_watch(m.peer_watch)
    rx.note_emission(Emission(Kind.BODY, Role.ISS, KEY, .8))
    primary, watch = rx.clock, rx.peer_watch
    obs = packet(AT+CYCLE, Kind.LONG, 0x42).observation
    with rx.scoped_clock(watch.clock, watch_generation=watch.generation):
        r = rx.register(obs)
        assert r.peer_watch_generation == watch.generation and r.responds_to is None
        assert r.response_generation is None and r.response_owner is None
    assert rx.clock is primary and primary.next_due == 8236750
    rx.finish_peer_watch(watch, AT+CYCLE)
    assert rx.finish_cycle(AT+CYCLE) == 0
    rx.finish_peer_watch(watch, AT+2*CYCLE)
    assert rx.finish_cycle(AT+2*CYCLE) == 1
    generation = watch.generation
    rx.new_call()
    assert rx.peer_watch is None and rx.register(obs).peer_watch_generation is None
    # Revocation/replacement during feed wins a temporary clock restoration.
    rx.sync_peer_watch(m.peer_watch)
    with rx.scoped_clock(rx.peer_watch.clock, watch_generation=generation):
        replacement = Clock(Expect.PACKET, AT+2*CYCLE, .330, 3.75, -.19)
        rx.clock = replacement
    assert rx.clock is replacement


def test_managed_renewal_after_recovery_ack_and_live_invitation():
    m, _ = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    for i in range(1, 10):
        at, due = AT+i*CYCLE, KEY+i*CYCLE
        m.observe(packet(at, Kind.LONG, 0x42, peer_watch_generation=m.peer_watch.generation), now=at+158400)
        p = m.decide(due-m.lead)
        assert p.word == ('Y27' if due >= KEY+960000 else 'X27')
        m.keyed()
        if p.word == 'Y27':
            assert m.peer_watch.attempts == 1
            break
    else: pytest.fail('managed renewal never selected a legal request')


def test_terminal_request_keeps_long_protected_span_after_watch_revocation():
    m, _ = setup()
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    m.finish_application(KEY)
    assert m.peer_watch is None and m.peer_phase.kind == Kind.CHIRP
    m.reschedule_breakin(KEY+CYCLE-m.lead)
    p = m.decide(KEY+CYCLE-m.lead)
    assert p.word == 'Y27' and p.kind == Kind.WORD
    assert m.peer_span_fits(p, 11712, 4800)
    m.keyed()
    m.observe(word('V27', p.stream_sample+44880, responds_to=p.stream_sample, extent=.245), now=p.stream_sample+56880)
    assert m.peer_phase is None


@pytest.mark.parametrize('reason', ['unassociated', 'wrong', 'weak', 'provisional', 'crc'])
def test_checked_body_only_releases_actual_long_request(reason):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    watch = m.peer_watch
    r = packet(8236750, Kind.BODY, 0x40,
               responds_to=None if reason == 'unassociated' else KEY-180000 if reason == 'wrong' else KEY)
    if reason in ('weak', 'provisional'):
        r = replace(r, observation=replace(r.observation, **{reason: True}))
    if reason == 'crc': r = replace(r, observation=replace(r.observation, frame=None))
    m.observe(r, now=8236750+38640)
    p = m.decide(8280910-m.lead)
    assert m.peer_watch is watch and not m.outbox.inflight and a.tx_acked_bytes == 0
    assert p is None or p.kind == Kind.WORD and p.word == 'Y27' and p.stream_sample == KEY+CYCLE


def test_watch_plus_real_outbound_response_cannot_settle_payload():
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    generation = m.peer_watch.generation
    m.observe(word('V27', 8236750, responds_to=KEY, extent=.245), now=8248750)
    p = m.decide(8371870-m.lead)
    m.keyed()
    r = word('X27', p.stream_sample+44880, responds_to=p.stream_sample, extent=.245)
    assert m.outbox.inflight == b'TEST\r' and m._body_ack(r)
    poisoned = replace(r, peer_watch_generation=generation)
    assert not m._body_ack(poisoned)
    m.observe(poisoned, now=p.stream_sample+58000)
    assert a.tx_acked_bytes == 0 and m.outbox.inflight == b'TEST\r'
    malicious_packet = packet(AT+2*CYCLE, Kind.LONG, 0x42,
                               peer_watch_generation=generation, responds_to=p.stream_sample)
    assert not m._body_ack(malicious_packet)
    m.observe(malicious_packet, now=AT+2*CYCLE+158400)
    assert a.tx_acked_bytes == 0
    m.observe(r, now=p.stream_sample+58000)
    assert a.tx_acked_bytes == 5


@pytest.mark.parametrize('outcome', [Registered.NEW, Registered.CONFIRMED, Registered.ENRICHED])
def test_current_watch_complete_enrichment_recovers_once(outcome):
    m, _ = setup()
    m.observe(packet(), now=AT+158400)
    legacy(m)
    watch = m.peer_watch
    r = packet(AT+CYCLE, Kind.LONG, 0x42, peer_watch_generation=watch.generation)
    provisional = replace(r.observation, frame=None, provisional=True)
    m.observe(replace(r, observation=provisional), now=AT+CYCLE+40000)
    assert m.peer_watch is watch and m.state == State.ISS_GRANT
    r = replace(r, outcome=outcome, replaced=provisional if outcome != Registered.NEW else None)
    m.observe(r, now=AT+CYCLE+158400)
    assert m.state == State.IRS_ANSWER and m.peer_watch.generation != watch.generation
    current = m.peer_watch
    m.observe(r, now=AT+CYCLE+158400)
    assert m.peer_watch is current


def test_success_with_genuine_output_drains_only_after_release_and_ack():
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    m.finish_application(KEY)
    assert m.app_done and a.tx_acked_bytes == 0 and not m.outbox.inflight
    m.observe(word('V27', 8236750, responds_to=KEY, extent=.245), now=8248750)
    p = m.decide(8371870-m.lead)
    assert p.payload == b'TEST\r' and p.status == 0x40
    m.keyed()
    assert a.tx_acked_bytes == 0
    m.observe(word('X27', p.stream_sample+44880, responds_to=p.stream_sample, extent=.245), now=p.stream_sample+58000)
    assert a.tx_acked_bytes == 5 and not m.outbox.inflight


def test_sample_wall_and_postdecision_watch_ownership_bounds():
    wall = [0.]
    m, _ = setup(b'TEST\r', wall_clock=lambda: wall[0], session_end=20*FS+AT,
                 session_wall_end=30.)
    m.observe(packet(), now=AT+158400)
    p = m.decide(KEY-m.lead)
    m.update_application_progress(10.)
    wall[0] = 10.  # sample time frozen; admission still enforces wall expiry.
    assert not m.can_key(p) and m._closing_failure and m.peer_watch is None
    m, _ = setup()
    m.observe(packet(), now=AT+158400)
    legacy(m)
    rx = P4Receiver(FS, admitted=True)
    rx.clock = Clock(Expect.WORD, 8236750, .935, 1.25, 0.)
    rx.sync_peer_watch(m.peer_watch)
    watch = rx.peer_watch
    with rx.scoped_clock(watch.clock, watch_generation=watch.generation):
        r = rx.register(packet(AT+CYCLE, Kind.LONG, 0x42).observation)
        m.observe(r, now=AT+CYCLE+158400)
        rx.sync_peer_watch(m.peer_watch, receiving=True)
    assert rx.clock.expects == Expect.PACKET and rx.clock.next_due == AT+CYCLE
    assert rx.peer_watch.generation != watch.generation
    m.link_down('sample ceiling')
    rx.sync_peer_watch(m.peer_watch)
    assert rx.peer_watch is None


def test_sample_ceiling_and_refused_deferred_long_answer():
    m, a = setup(session_end=AT+10*FS)
    m.observe(packet(), now=AT+158400)
    p = m.decide(KEY-m.lead)
    m.refused('deadline')
    assert m._deferred[1]['due'] == KEY+CYCLE
    p = m.decide(KEY+CYCLE-m.lead)
    assert p.word == 'X27' and p.stream_sample == KEY+CYCLE and m.can_key(p)
    m.keyed()
    assert m.decide(AT+10*FS) is None and m.state == State.DOWN and m.peer_watch is None
    assert a.tx_acked_bytes == 0


def test_revoked_terminal_watch_does_not_release_peer_or_end_epoch():
    m, a = setup()
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    epoch = m._receive_epoch
    m.finish_application(KEY)
    m.reschedule_breakin(KEY+CYCLE-m.lead)
    p = m.decide(KEY+CYCLE-m.lead)
    m.keyed()
    assert m._receive_epoch == epoch and m._receive_boundary != p.stream_sample
    r = packet(p.stream_sample+44880, Kind.BODY, 0x40, responds_to=None)
    r = replace(r, observation=replace(r.observation, weak=True))
    m.observe(r, now=r.observation.stream_sample+38640)
    m.decide(r.observation.stream_sample+44160-m.lead)
    assert m.peer_phase is not None and m.peer_phase.kind == Kind.CHIRP
    assert a.tx_acked_bytes == 0


def test_previous_call_generation_and_nonmail_groups_stay_finite():
    old, _ = setup()
    old.observe(packet(), now=AT+158400)
    generation = old.peer_watch.generation
    m, a = machine(role='irs', data='body', ladder='none', prekey_lead=.025)
    a.on_host_data(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    for i in range(4):
        p = m.decide(KEY+i*CYCLE-m.lead)
        m.keyed()
        m.silence(p.stream_sample+60000)
    current = m.peer_watch
    m.observe(packet(AT+4*CYCLE, Kind.LONG, 0x42, peer_watch_generation=generation), now=AT+4*CYCLE+158400)
    assert m.peer_watch is current and m.next_offer_at == KEY+3*CYCLE+960000
    m.keyed()
    m.silence(9811870-m.lead, answer_due=9811870)
    assert m.decide(9811870-m.lead).word == 'X27' and m.peer_watch.attempts == 4
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
def test_unsupported_extended_protocol2_cannot_arm_or_recover(kind):
    m, a = setup(b'TEST\r')
    m.adapter.terminal = True
    m.observe(packet(), now=AT+158400)
    legacy(m)
    r = packet(AT+CYCLE, kind, 0xc0, peer_watch_generation=m.peer_watch.generation)
    m.observe(r, now=AT+CYCLE+158400)
    assert m.adapter.failure and m.app_done and m.peer_watch is None
    assert m.state == State.IRS_CLOSE and a.tx_acked_bytes == 0


def placeholder_request(kind, delta=0):
    """Semantic registry hypotheses; no RF decoder or invented association."""
    from types import SimpleNamespace
    m, a = setup(b'TEST\r')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    m.observe(packet(), now=AT+158400)
    request = m.decide(KEY-m.lead)
    m.keyed()
    rx = P4Receiver(FS, admitted=True, chirp=True, breakin=True)
    rx.enable_hold()
    rx.sync_peer_watch(m.peer_watch)
    rx.note_emission(Emission(request.kind, Role.IRS, KEY, .244))
    at = rx.peer_watch.clock.next_due+delta
    old = Observation(Kind.SHORT, FS, at-2400, 3678, at+1278,
                      .8, -.19, .27, provisional=True)
    data = b'peer\r'.ljust(22 if kind == Kind.CHIRP else 40, b'\x1e')
    incoming = replace(packet(at, kind, 2, data).observation,
                       origin=at-2400, first_chip=2400)
    return m, a, rx, old, incoming, delivered


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('delta', [-1680, 0, 1680])
def test_current_watch_crc_resolves_only_empty_short(kind, delta):
    m, a, rx, old, incoming, delivered = placeholder_request(kind, delta)
    watch = rx.peer_watch
    with rx.scoped_clock(watch.clock, watch_generation=watch.generation):
        initial = rx.register(old)
        assert initial.outcome == Registered.NEW
        m.observe(initial, now=incoming.stream_sample+40000)
        reg = rx.register(incoming)
        assert reg.outcome == Registered.CONFIRMED and reg.replaced is old
        assert reg.observation is incoming and rx.observations == [incoming]
        assert (incoming.origin, incoming.first_chip, incoming.stream_sample) == (
            incoming.stream_sample-2400, 2400, AT+CYCLE+delta)
        assert reg.peer_watch_generation == watch.generation == m.peer_watch.generation
        assert (reg.responds_to, reg.response_generation, reg.response_slot,
                reg.response_owner) == (None, None, None, None)
        assert len(rx.registrations) == 2 and rx.registrations[-1] is reg
        m.observe(reg, now=incoming.stream_sample+158400)
        assert m.state == State.IRS_ANSWER and delivered == [b'peer\r']
        assert m._held_answer.context['body'] is incoming
        assert m.next_offer_at == KEY+960000
        ledger = (bytes(m.adapter._covered), m.adapter._source_width)
        epoch = m._receive_epoch
        held, phase = m._held_answer, m.peer_phase
        again = rx.register(incoming)
        assert again.outcome == Registered.DUPLICATE and again.replaced is None
        m.observe(again, now=incoming.stream_sample+158400)
        assert m._held_answer is held and m.peer_phase is phase
        assert (bytes(m.adapter._covered), m.adapter._source_width) == ledger
        assert m._receive_epoch == epoch and delivered == [b'peer\r']
    answer = m.decide(incoming.stream_sample+163200-m.lead)
    assert answer.kind == Kind.WORD and answer.state == State.IRS_ANSWER and not answer.payload
    m.keyed()
    assert not m.outbox.inflight and bytes(a._outbuf) == b'TEST\r'
    assert a._buffer_raw == 5 and a.tx_acked_bytes == 0
    assert not [e for e in m.link_events if e['event'] == 'turn']


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('mutation,expected', [
    ('header', Registered.CONTRADICTED), ('value', Registered.CONTRADICTED),
    ('word', Registered.CONTRADICTED), ('frame', Registered.CONTRADICTED),
    ('placed', Registered.CONTRADICTED), ('nonprovisional', Registered.DUPLICATE),
    ('proven-word', Registered.DUPLICATE), ('conflicting-long', Registered.CONTRADICTED),
    ('weak', Registered.CONTRADICTED), ('unexpected', Registered.CONTRADICTED),
    ('provisional', Registered.DUPLICATE), ('no-crc', Registered.CONTRADICTED),
    ('absent', Registered.CONTRADICTED), ('unscoped', Registered.CONTRADICTED),
    ('revoked', Registered.CONTRADICTED),
    ('stale', Registered.CONTRADICTED), ('replaced', Registered.CONTRADICTED),
    ('early', Registered.CONTRADICTED), ('late', Registered.CONTRADICTED),
    ('wrong-cycle', Registered.CONTRADICTED), ('word-clock', Registered.CONTRADICTED)])
def test_empty_short_exception_retains_legacy_rejection(kind, mutation, expected):
    delta = -1681 if mutation == 'early' else 1681 if mutation == 'late' else 0
    m, a, rx, old, incoming, delivered = placeholder_request(kind, delta)
    if mutation in ('header', 'value', 'word', 'frame'):
        old = replace(old, **{mutation: {'header': 'A', 'value': 13,
            'word': 'Y27', 'frame': incoming.frame}[mutation]})
    if mutation == 'placed': old = replace(old, placed=True)
    if mutation == 'nonprovisional': old = replace(old, provisional=False)
    if mutation == 'proven-word': old = replace(old, kind=Kind.WORD, word='Y27', provisional=False)
    if mutation == 'conflicting-long':
        old = replace(packet(incoming.stream_sample, Kind.LONG, 3).observation)
        incoming = replace(incoming, kind=Kind.LONG)
    if mutation in ('weak', 'unexpected', 'provisional'):
        incoming = replace(incoming, **{mutation: True})
    if mutation == 'no-crc': incoming = replace(incoming, frame=None)
    watch = rx.peer_watch
    generation = watch.generation
    if mutation == 'absent': rx.sync_peer_watch(None)
    with rx.scoped_clock(watch.clock, watch_generation=(
            None if mutation == 'unscoped' else generation-1 if mutation == 'stale' else generation)):
        if mutation == 'revoked': rx.sync_peer_watch(None)
        if mutation == 'replaced':
            rx.sync_peer_watch(replace(m.peer_watch, generation=generation+1))
        if mutation == 'wrong-cycle': watch.clock.next_due += CYCLE
        if mutation == 'word-clock': watch.clock.expects = Expect.WORD
        initial = rx.register(old)
        assert initial.outcome == Registered.NEW
        before = (m.state, m.peer_watch, m._held_answer, m._receive_epoch,
                  dict(m.pending), bytes(m.adapter._covered))
        # Historical provenance alone must never authorize replacement.
        rx._peer_watch_evidence[old.stream_sample] = generation
        reg = rx.register(incoming)
        assert reg.outcome == expected
        assert reg.replaced is (None if expected == Registered.DUPLICATE else old)
        assert len(rx.observations) == 1
        m.observe(reg, now=incoming.stream_sample+158400)
        assert (m.state, m.peer_watch, m._held_answer, m._receive_epoch,
                m.pending, m.adapter._covered) == before
    assert delivered == [] and not m.outbox.inflight
    assert bytes(a._outbuf) == b'TEST\r' and a._buffer_raw == 5 and a.tx_acked_bytes == 0
    assert not [e for e in m.link_events if e['event'] == 'turn']


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('watched', [False, True])
def test_admitted_invitation_continuation_cannot_release_request(kind, watched):
    m, a = setup(b'TEST\r')
    m.observe(packet(), now=AT+158400)
    m.decide(KEY-m.lead)
    m.keyed()
    r = packet(AT+CYCLE, kind, 0x42,
               peer_watch_generation=m.peer_watch.generation if watched else None)
    m.observe(r, now=AT+CYCLE+158400)
    assert m._held_answer.context['accepted']
    p = m.decide(KEY+CYCLE-m.lead)
    assert p.kind == Kind.WORD and not p.payload
    assert p.state == State.IRS_ANSWER
    m.keyed()
    assert bytes(a._outbuf) == b'TEST\r' and a._buffer_raw == 5
    assert not m.outbox.inflight and a.tx_acked_bytes == 0
    assert not [e for e in m.link_events if e['event'] == 'turn']


def test_late_watched_continuation_keys_recovered_ack_at_next_legal_slot():
    # Finite pure semantic analogue of 1112's late read; no tape/cost/stop change.
    m, a = setup(b'TEST\r')
    onset, key = 1248445, 1411645
    m.observe(packet(onset, Kind.LONG, 2), now=onset+158400)
    for at in (key, key+CYCLE, key+2*CYCLE):
        p = m.decide(at-m.lead)
        assert p.word == 'Y27' and p.stream_sample == at
        m.keyed()
        if at < key+2*CYCLE: m.silence(at+60000)
    last = m.last
    r = packet(1608445, Kind.LONG, 2, peer_watch_generation=m.peer_watch.generation)
    m.observe(r, now=1786044)
    assert m.last is last and last.stream_sample == 1771645 and last.word == 'Y27'
    assert m.state == State.IRS_ANSWER and m._deferred[1]['due'] == 1951645
    held = m._held_answer
    p = m.decide(1951645-m.lead)
    assert (p.kind, p.word, p.state, p.stream_sample) == (
        Kind.WORD, 'X27', State.IRS_ANSWER, 1951645)
    assert p.ack_identity == held.answer.ack_identity and p.accepted_target == 1608445
    m.keyed()
    assert not m.outbox.inflight and bytes(a._outbuf) == b'TEST\r'
    assert a.tx_acked_bytes == 0
