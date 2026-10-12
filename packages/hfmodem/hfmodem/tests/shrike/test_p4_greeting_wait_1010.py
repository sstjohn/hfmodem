# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded hand-back refusals, followed by synthetic waiting opportunities."""
import re
from dataclasses import replace
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from hfmodem.tests.evidence import WORKING, require_input
from hfmodem.tests.shrike.test_p4turn import FS, machine, word
from hfmodem.tests.shrike.test_p4_greeting_resegment_1010 import packet, FULL
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import modem
from hfmodem.shrike.onair import _p4_mail_progress, _MasterGrid, _keyable_slot
from hfmodem.shrike.p3progress import MailProgress
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4turn import Proposal, State, counter_ack
from hfmodem.winlink import B2FSession, MailClient
from hfmodem.winlink.message import compose
from hfmodem.winlink.session import _WAIT_FS, _THEIR_TURN

CYCLE = round(1.25*FS)
ARMS = WORKING / 'pactor-level-air-0916/arms'
W8 = 'pactor-p4-bench-ws8eoc-40-force-20261010T224944Z-p4-mail--f100385d-463a-44de-b7d2-f135f673136f'
V3 = 'pactor-p4-bench-ve3kpg-40-force-20261010T230003Z-p4-mail--9e5567a3-e18f-4658-b53c-0e28b73f192b'


def waiting(clock, **kwargs):
    m, a = machine(role='irs', data='body', ladder='none', normal_ack='parity',
                   prekey_lead=.025, hold_cycles=16, wall_clock=lambda: clock[0],
                   session_end=round(600*FS), managed_application=True, **kwargs)
    return m, a


def source(m, arm, lines, clock=None):
    evidence = require_input(ARMS / arm / 'arm.log').read_text().splitlines()
    for line in lines:
        match = re.search(r'\[p4 frame\] (body|normal) at (\S+) status=(\S+) data=(\S+)',
                          evidence[line-1])
        assert match, (arm, line)
        kind, seconds, status, data = match.groups()
        if clock is not None:
            clock[0] = float(seconds)
        reg = packet(Kind.BODY if kind == 'body' else Kind.NORMAL,
                     float(seconds), int(status, 16), bytes.fromhex(data))
        at = reg.observation.stream_sample
        m.observe(reg, now=at)
        p = m.decide(at+round(m.t[reg.observation.kind]*FS)-m.lead)
        assert p.kind == Kind.WORD
        m.keyed()


def seed(m, keyed, onset, now, actual):
    m._next = Proposal(Kind.WORD, keyed, word='X27' if keyed == 992221 else 'V27',
                       state=State.IRS_ANSWER)
    m.keyed()
    m.observe(word('V27', round(onset*FS), extent=.245), now=round(now*FS))
    offer = m.decide(round(now*FS))
    assert offer.state == State.ISS_HANDBACK
    m.keyed(at=actual)
    assert m._hold_until == actual+4*CYCLE
    return actual


def speculative(m, provisional, confirmed):
    offer = m.last.stream_sample
    early = word('V27', round(provisional*FS), partial=True, extent=.245,
                 responds_to=offer)
    m.observe(early, now=round((provisional+.180)*FS))
    p = m.decide(offer+CYCLE-m.lead)
    assert p.kind == Kind.NORMAL and p.status == 0
    m.keyed()
    m.observe(word('V27', round(confirmed*FS), outcome=Registered.CONFIRMED,
                   replaced=early.observation, extent=.245, responds_to=offer),
              now=m.last.stream_sample+round(.050*FS))


def refusal(m, seconds=None, name='V27', offset=.930):
    at = m.last.stream_sample+round(offset*FS) if seconds is None else round(seconds*FS)
    m.observe(word(name, at, extent=.245, responds_to=m.last.stream_sample),
              now=at+round(.245*FS))
    p = m.decide(m.last.stream_sample+CYCLE-m.lead)
    return p


def idle(m, at):
    p = m.decide(at-m.lead)
    assert (p.kind, p.status, p.payload, p.state) == (Kind.NORMAL, 0, b'', State.ISS_HOLD)
    assert p.stream_sample == at
    assert m.close_context is None
    m.keyed()
    return p


def bridge(client, m, now):
    return _p4_mail_progress(client, m, now, drained=False)


def ws8(clock):
    m, a = waiting(clock)
    source(m, W8, [164])
    assert m._held_answer is not None
    history = m._packet_ack, dict(m._packet_words)
    at = seed(m, 992221, 21.601, 22., 1112221)
    assert (m._held_answer, m._last_ack, m._last_body_ack, m.acked_counter) == (None,)*4
    assert m._handoff_ticket == m._receive_boundary == at
    speculative(m, 24.102, 24.101)
    for i, seconds in enumerate((25.353, 26.601, 27.851), 2):
        refusal(m, seconds)
        idle(m, at+i*CYCLE)
    assert m._handoff_ticket == m._receive_boundary == at
    assert (m._packet_ack, m._packet_words) == history
    return m, a, at


def reoffer(m, offer):
    for i in range(5, 16):
        refusal(m)
        idle(m, offer+i*CYCLE)
    refusal(m)
    p = m.decide(offer+16*CYCLE-m.lead)
    assert (p.kind, p.status, p.payload, p.state) == (Kind.BODY, 0x40, b'', State.ISS_HANDBACK)
    assert p.stream_sample == offer+16*CYCLE
    m.keyed()
    assert m._handoff_ticket == m._receive_boundary == p.stream_sample
    return p


def test_peer_data_accepts_a_reoffer():
    m, _, at = ws8([16.001])
    offer = reoffer(m, at).stream_sample
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *args: None)
    epoch = m._receive_epoch
    m.observe(replace(packet(Kind.BODY, offer/FS+.935, 0, FULL[:40]), responds_to=offer),
              now=offer+round(.935*FS))
    assert m._receive_epoch == epoch+1 and m._handoff_ticket is None
    assert delivered == [FULL[:40]]
    answer = m.decide(offer+round((.935+m.t[Kind.BODY])*FS)-m.lead)
    assert answer.state == State.IRS_ANSWER and m.can_key(answer)
    m.keyed()
    assert m.state == State.IRS_ANSWER and m.next_offer_at is None


def test_ws8eoc_waits_past_refused_offer_and_reoffers():
    clock = [16.001]
    m, _, at = ws8(clock)
    assert 'handback=0/4' in m.waiting_status(round(24.106*FS))
    assert 'handback=1/4' in m.waiting_status(round(25.356*FS))
    assert 'handback=2/4' in m.waiting_status(round(26.606*FS))
    assert reoffer(m, at).stream_sample == 2072221
    refusal(m)
    idle(m, 2072221+CYCLE)
    second = 2072221
    for i in range(2, 16):
        refusal(m)
        idle(m, second+i*CYCLE)
    refusal(m)
    p = m.decide(second+16*CYCLE-m.lead)
    previous_schedule = m._offer_at, m.next_offer_at, m._hold_until
    m.refused('deadline')
    assert (m._offer_at, m.next_offer_at, m._hold_until) == previous_schedule
    m.keyed()
    assert m._offer_at == second+17*CYCLE
    assert m._hold_until == m._offer_at+4*CYCLE
    assert m.next_offer_at == m._offer_at+16*CYCLE
    control, a = waiting(clock)
    control.allow_breakin = True
    a.on_host_data(b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;')
    control._body(2011832, 0x40, State.IRS_BREAKIN)
    control.keyed()
    early = word('V27', round(42.843*FS), partial=True, extent=.245, responds_to=2011832)
    control.observe(early, now=round(43.*FS))
    control.observe(word('V27', round(42.843*FS), outcome=Registered.CONFIRMED,
                         replaced=early.observation, extent=.245, responds_to=2011832),
                    now=round(43.09*FS))
    p = control.decide(2071832-control.lead)
    assert (p.kind, p.status, p.state, p.stream_sample) == (Kind.BODY, 0x40, State.ISS_DATA, 2071832)
    control.keyed()
    assert control.outbox.inflight and any(e['event'] == 'turn' for e in control.link_events)
    unconfirmed = control.outbox.inflight
    control.finish_application(p.stream_sample)
    assert control.close_context is None and control.outbox.inflight == unconfirmed
    close = refusal(control, name='X27')
    assert close.state == State.ISS_CLOSE and not control.outbox.inflight


def test_ve3kpg_waits_and_closes_only_at_application_deadline():
    clock = [51.015]
    m, a = waiting(clock)
    client = MailClient(B2FSession('W9SSJ', target='VE3KPG'), a.on_host_data)
    client.progress = MailProgress(client, a, now=lambda: clock[0])
    m.adapter.host = SimpleNamespace(deliver=client.on_link_data, supervisor=lambda *args: None)
    source(m, V3, [393, 405, 419, 434, 449, 464, 479, 494, 510, 524], clock)
    assert not client.session.remote_sid and not a.tx_queued_bytes
    at = seed(m, 3032980, 64.116, 64.500, 3152980)
    speculative(m, 66.615, 66.615)
    for i, seconds in enumerate((67.865, 69.115, 70.365), 2):
        refusal(m, seconds)
        idle(m, at+i*CYCLE)
    assert m.last.stream_sample == 3392980
    clock[0] = 70.687
    assert not bridge(client, m, m.last.stream_sample)
    deadline = m.application_deadline
    assert deadline == pytest.approx(181.015)
    assert reoffer(m, at).stream_sample == 4112980
    clock[0] = 86.
    assert not bridge(client, m, m.last.stream_sample)
    assert m.application_deadline == deadline
    clock[0] = deadline-1e-6
    now = round(clock[0]*FS)
    assert not bridge(client, m, now)
    m.decide(now)
    assert not m.app_done and not client.session.failure
    clock[0] = deadline
    now = round(clock[0]*FS)
    assert bridge(client, m, now)
    assert 'without a SID' in client.session.failure
    p = m.decide(now)
    assert m.app_done and m._closing_failure and m.reason == client.session.failure
    assert p.state == State.ISS_CLOSE and p.status == 0x20
    assert p.stream_sample <= round(182.265*FS)
    assert m.close_context.end <= m.session_end
    end = m.close_context.end
    m.keyed()
    for _ in range(30):
        now = m.last.stream_sample+CYCLE
        m.silence(now-m.lead)
        p = m.decide(now-m.lead)
        if m.state == State.DOWN:
            break
        assert p.state == State.ISS_CLOSE and not p.payload
        m.keyed()
    assert m.state == State.DOWN and now <= end


@pytest.mark.parametrize('exit', ['z27', 'silence', 'expired', 'contradicted', 'x27', 'idle-x27'])
@pytest.mark.parametrize('terminal', ['done', 'failure', 'sample', 'wall'])
def test_all_handback_exit_paths_share_application_gate(exit, terminal):
    clock = [0.]
    m, a, at = ws8(clock)
    a.on_host_data(b'gated setup')
    m.allow_breakin = False
    reoffer(m, at)
    offer = m.last.stream_sample
    if exit == 'z27':
        refusal(m, name='Z27', offset=.935)
        assert m._handback_refusals == {offer}
        m.silence(offer+CYCLE-m.lead)
        repeat = m.decide(offer+CYCLE-m.lead)
        assert (repeat.kind, repeat.status, repeat.payload, repeat.state, repeat.repeat) == (
            Kind.BODY, 0x40, b'', State.ISS_HANDBACK, True)
        m.keyed()
        assert m.last == repeat and m.last.stream_sample == offer+CYCLE
        refusal(m, name='Z27', offset=.935)
        assert m._handback_refusals == {offer, offer+CYCLE}
        exit_idle = m.decide(offer+2*CYCLE-m.lead)
        assert (exit_idle.kind, exit_idle.payload, exit_idle.state) == (
            Kind.NORMAL, b'', State.ISS_HOLD)
        assert m.keyed() == exit_idle and m.state == State.ISS_HOLD
    elif exit == 'silence':
        for _ in range(4):
            now = m.last.stream_sample+CYCLE-m.lead
            m.silence(now)
            p = m.decide(now)
            assert p.state != State.ISS_CLOSE
            m.keyed()
    elif exit in ('expired', 'contradicted'):
        for _ in range(3):
            now = m.last.stream_sample+CYCLE-m.lead
            m.silence(now)
            assert m.decide(now).state == State.ISS_HANDBACK
            m.keyed()
        offer = m.last.stream_sample
        early = word('V27', offer+round(.935*FS), partial=True, responds_to=offer)
        m.observe(early, now=early.observation.stream_sample+round(.180*FS))
        m.decide(offer+CYCLE-m.lead)
        m.keyed()
        m.observe(word('V27' if exit == 'expired' else 'Z27', early.observation.stream_sample,
                       outcome=Registered.EXPIRED if exit == 'expired' else Registered.CONTRADICTED,
                       replaced=early.observation, responds_to=offer),
                  now=m.last.stream_sample+round(.050*FS))
    else:
        refusal(m, name='X27' if exit == 'x27' else 'V27')
        m.keyed()
        if exit == 'idle-x27':
            refusal(m, name='X27')
    now = m.last.stream_sample+CYCLE-m.lead
    p = m.decide(now)
    assert p.state in (State.ISS_HOLD, State.ISS_HANDBACK)
    assert m.close_context is None and a.tx_acked_bytes == 0
    assert a._buffer_raw == len(b'gated setup')
    if terminal in ('failure', 'wall'):
        m.state = State.ISS_HOLD
        m._idle(m.last.stream_sample+CYCLE)
        now = m.next_offer_at-m.lead
        p = m.decide(now)
        assert p.state == State.ISS_HANDBACK
    if terminal in ('sample', 'wall'):
        if terminal == 'sample':
            m.session_end = now
        else:
            m.session_wall_end = 1.
            clock[0] = 1.
        assert not m.can_key(p)
        assert m.state == State.DOWN
    else:
        m.finish_application(now, failure='test failure' if terminal == 'failure' else None)
        assert not m.can_key(p)
        close = m.decide(now)
        assert close.status == 0x20 and close.state == State.ISS_CLOSE
        assert a._buffer_raw == len(b'gated setup') and a.tx_acked_bytes == 0
        assert m.close_context.queued == b'gated setup'
        reply_at = m.last.stream_sample+round(.935*FS)
        m.observe(word('V27', reply_at, extent=.245, responds_to=m.last.stream_sample), now=now)
        assert m.decide(now).state == State.ISS_CLOSE
        m.update_application_progress(None)
        assert m.app_done


def test_p4_progress_bridge_never_runs_p3_disconnect_effects():
    clock = [16.001]
    m, _, at = ws8(clock)
    a, _ = modem()
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.progress = MailProgress(client, a, now=lambda: clock[0])
    for name in ('_cancel_p3_proposals', '_enter_receive_only', 'on_host_disconnect', 'retain_unconfirmed'):
        setattr(a, name, Mock())
    client.on_link_data(b'RMS Trimode\r')
    assert not bridge(client, m, at)
    assert m.application_deadline == 136.001
    snapshot = client.progress.check()
    with pytest.raises(AttributeError):
        snapshot.deadline = 0
    reoffer(m, at)
    for _ in range(4):
        refusal(m)
        m.keyed()
    reoffer(m, m.last.stream_sample-4*CYCLE)
    clock[0] = 100.
    m.adapter.host = SimpleNamespace(deliver=client.on_link_data, supervisor=lambda *args: None)
    m.observe(packet(Kind.NORMAL, 100., 1, FULL), now=100*FS)
    m.decide(100*FS+round(m.t[Kind.NORMAL]*FS)-m.lead)
    m.keyed()
    bridge(client, m, m.last.stream_sample)
    assert m.application_deadline == 220.
    for seconds, status, data in ((101.25, 1, FULL[:40]),
                                  (102.5, 0x82, b'\x01\x01\x01'+FULL[40:77])):
        clock[0] = seconds
        m.observe(packet(Kind.NORMAL, seconds, status, data), now=round(seconds*FS))
        m.decide(round((seconds+m.t[Kind.NORMAL])*FS)-m.lead)
        m.keyed()
        bridge(client, m, round(seconds*FS))
        assert m.application_deadline == 220.
    clock[0] = 136.001
    assert not bridge(client, m, round(clock[0]*FS))
    m.silence(round(clock[0]*FS))
    stale = m.decide(round(clock[0]*FS))
    clock[0] = 220.
    assert not m.can_key(stale)
    assert m.app_done and m._closing_failure
    assert bridge(client, m, round(clock[0]*FS))
    closing = m.state, m._next
    bridge(client, m, round(clock[0]*FS))
    assert (m.state, m._next) == closing and m.reason == client.session.failure
    assert len([line for line in client.session.log_lines if line.startswith('session failed')]) == 1
    assert 'without a SID' in client.session.failure
    for name in ('_cancel_p3_proposals', '_enter_receive_only', 'on_host_disconnect', 'retain_unconfirmed'):
        getattr(a, name).assert_not_called()
    unarmed, _ = waiting(clock)
    unarmed._idle(0)
    unarmed.keyed()
    unarmed.update_application_progress(None)
    assert unarmed.decide(round(500*FS)).state == State.ISS_HOLD
    unarmed.session_wall_end = 221.
    clock[0] = 221.
    assert not unarmed.can_key(unarmed._next) and unarmed.state == State.DOWN


def test_p4_settlement_arms_the_post_send_deadline():
    clock = [0.]
    m, a = waiting(clock)
    message = compose('W9SSJ', 'WS8EOC', 'test', 'body\r\n', mid='PROGRESS120')
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC', outbox=[message]), a.on_host_data)
    progress = client.progress = MailProgress(client, a, now=lambda: clock[0])
    client.session._state = _WAIT_FS
    proposal = bytearray()
    client.session._our_turn(proposal)
    client._push(bytes(proposal))
    assert progress.watermark == a.tx_queued_bytes == len(proposal) > 31
    m._body(2011832, 0x40, State.IRS_BREAKIN)
    m.keyed()
    early = word('V27', round(42.843*FS), partial=True, extent=.245, responds_to=2011832)
    m.observe(early, now=round(43.*FS))
    m.observe(word('V27', round(42.843*FS), outcome=Registered.CONFIRMED,
                   replaced=early.observation, extent=.245, responds_to=2011832),
              now=round(43.09*FS))
    assert m.decide(2071832-m.lead).state == State.ISS_DATA
    m.keyed()
    clock[0] = 50.
    assert refusal(m, name=counter_ack(m.last.status)).payload == bytes(proposal[31:])
    assert progress.check().deadline is None
    m.keyed()
    refusal(m, name=counter_ack(m.last.status))
    assert a.tx_acked_bytes == len(proposal) and not m.outbox.inflight
    assert progress.check() == (170., False, None)


@pytest.mark.parametrize('ownership', ['pending', 'selected', 'deferred'])
@pytest.mark.parametrize('failure', [False, True])
def test_final_fq_answer_survives_success_only(ownership, failure):
    clock = [10.]
    m, a = waiting(clock)
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.session._state = _THEIR_TURN
    delivered = []
    def deliver(data):
        delivered.append(data)
        client.on_link_data(data)
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *args: None)
    reg = packet(Kind.NORMAL, 10., 1, b'FQ\r', width=40)
    onset = reg.observation.stream_sample
    due = onset+round(m.t[Kind.NORMAL]*FS)
    now = due if ownership == 'deferred' else onset
    m.observe(reg, now=now)
    assert client.done and not client.session.failure and delivered == [b'FQ\r']
    assert not a._buffer_raw and not m.outbox.inflight
    held = m._held_answer.answer
    if ownership == 'selected':
        selected = m.decide(due-m.lead)
        assert m._accepted_proposal[0] is selected
        now = due-m.lead
    pending, deferred, context, next_answer = (
        dict(m.pending), m._deferred, m._accepted_proposal, m._next)
    epoch = m._receive_epoch
    if failure:
        client.session._fail('terminal receive failure')
    assert _p4_mail_progress(client, m, now, drained=True)
    assert m.app_done and delivered == [b'FQ\r']
    if failure:
        assert m._closing_failure and not m.pending and m._deferred is None
        assert m._accepted_proposal is None and not m.can_key(next_answer or held)
        assert m.decide(now).state == State.IRS_BREAKIN
        return
    assert not m._closing_failure and not client.session.failure
    assert m.pending == pending and m._deferred is deferred
    assert m._accepted_proposal is context and m._next is next_answer
    assert m._receive_epoch == epoch and m._held_answer.answer is held
    key_at = due+(CYCLE if ownership == 'deferred' else 0)
    answer = m.decide(key_at-m.lead)
    assert (answer.kind, answer.word, answer.state, answer.stream_sample) == (
        Kind.WORD, held.word, State.IRS_ANSWER, key_at)
    assert (answer.ack_identity, answer.receive_epoch, answer.source_root,
            answer.accepted_target) == (
        held.ack_identity, held.receive_epoch, held.source_root, onset)
    assert m.can_key(answer)
    assert m.keyed().stream_sample == key_at
    assert a.tx_acked_bytes == 0
    # A CRC-valid repeat still answers normally without delivering FQ twice.
    repeat_at = onset+(2*CYCLE if ownership == 'deferred' else CYCLE)
    m.observe(packet(Kind.NORMAL, repeat_at/FS, 1, b'FQ\r', width=40), now=repeat_at)
    repeat_due = repeat_at+round(m.t[Kind.NORMAL]*FS)
    repeat_answer = m.decide(repeat_due-m.lead)
    assert repeat_answer.word == held.word and m.can_key(repeat_answer)
    m.keyed()
    assert delivered == [b'FQ\r']
    # The existing application-done peer-control route requests the turn.
    peer_at = repeat_at+CYCLE
    control = replace(reg, observation=replace(reg.observation, kind=Kind.SHORT,
        stream_sample=peer_at, frame=None, value=59, header='A', placed=True))
    m.observe(control, now=peer_at)
    request = m.decide(peer_at)
    assert request.state == State.IRS_BREAKIN
    m.keyed()
    reply = refusal(m, name='V27', offset=.935)
    assert reply.state == State.ISS_CLOSE and reply.status == 0x20
    m.keyed()
    assert m.close_context.end <= m.session_end
    if ownership == 'selected':
        clock[0] = m.close_context.wall_started+30.
        assert m.decide(m.last.stream_sample+CYCLE) is None
    else:
        assert m.decide(m.close_context.end) is None
    assert m.state == State.DOWN and m.reason == 'close deadline'


@pytest.mark.parametrize('miss', ['decision', 'refusal', 'deferred-expired'])
def test_final_fq_missed_answer_uses_existing_one_cycle_deferral(miss):
    m, a = waiting([0.])
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.session._state = _THEIR_TURN
    deliver = Mock(wraps=client.on_link_data)
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *args: None)
    m.observe(packet(Kind.NORMAL, 10., 1, b'FQ\r', width=40), now=10*FS)
    due = 10*FS+round(m.t[Kind.NORMAL]*FS)
    assert _p4_mail_progress(client, m, due-m.lead, drained=True)
    selected = m.decide(due-m.lead)
    assert selected.state == State.IRS_ANSWER
    if miss == 'refusal':
        m.refused('deadline')
    else:
        assert m.decide(due) is None
    assert m._deferred[1]['due'] == due+CYCLE
    if miss == 'deferred-expired':
        assert m.decide(due+CYCLE+CYCLE//2) is None
        assert m._deferred is None and m.last is None
    else:
        answer = m.decide(due+CYCLE-m.lead)
        assert answer.stream_sample == due+CYCLE and m.can_key(answer)
        assert m.keyed().stream_sample == due+CYCLE
    deliver.assert_called_once_with(b'FQ\r')


@pytest.mark.parametrize('ceiling', ['unanswered', 'sample', 'wall'])
def test_final_fq_preserves_breakin_and_hard_bounds(ceiling):
    clock = [0.]
    m, a = waiting(clock)
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.session._state = _THEIR_TURN
    m.adapter.host = SimpleNamespace(deliver=client.on_link_data, supervisor=lambda *args: None)
    m.observe(packet(Kind.NORMAL, 10., 1, b'FQ\r', width=40), now=10*FS)
    due = 10*FS+round(m.t[Kind.NORMAL]*FS)
    _p4_mail_progress(client, m, due-m.lead, drained=True)
    assert m.decide(due-m.lead).state == State.IRS_ANSWER
    m.keyed()
    peer_at = 10*FS+CYCLE
    reg = packet(Kind.NORMAL, 10., 1, b'FQ\r', width=40)
    control = replace(reg, observation=replace(reg.observation, kind=Kind.SHORT,
        stream_sample=peer_at, frame=None, value=59, header='A', placed=True))
    m.observe(control, now=peer_at)
    request = m.decide(peer_at)
    assert request.state == State.IRS_BREAKIN
    if ceiling == 'sample':
        m.session_end = request.stream_sample
        assert not m.can_key(request)
    elif ceiling == 'wall':
        m.session_wall_end = 1.
        clock[0] = 1.
        assert not m.can_key(request)
    else:
        for i in range(4):
            assert m.can_key(request)
            m.keyed()
            assert m.breakins == i+1
            now = m.last.stream_sample+CYCLE-m.lead
            m.silence(now)
            request = m.decide(now)
        assert m.reason == 'break-in unanswered'
    assert m.state == State.DOWN and a.tx_acked_bytes == 0


@pytest.mark.parametrize('replacement', ['equal', 'retimed'])
def test_nonmail_captured_proposal_admission_survives_replacement(replacement):
    m, a = machine(role='irs', data='body', ladder='none', prekey_lead=.025)
    reg = packet(Kind.LONG, 10., 1, b'peer source', width=69)
    m.observe(reg, now=10*FS)
    due = 10*FS+round(m.t[Kind.LONG]*FS)
    m.decide(due-m.lead)
    m.keyed()
    a.on_host_data(b'our mail')
    first = due+round(3.750*FS)
    m._break_in(first)
    captured = m.decide(first-m.lead)
    assert m.can_key(captured)
    if replacement == 'equal':
        m._next = replace(captured)
        assert m._next == captured and m._next is not captured
        now = first-m.lead
    else:
        now = first-m.lead+1
        m.reschedule_breakin(now)
        assert m._next.stream_sample == first+round(3.750*FS)
    current = m.decide(now)
    assert current is not captured and m.can_key(current)
    assert m.can_key(captured)
    # Admission alone cannot emit the obsolete slot; use the driver's guard.
    live = SimpleNamespace(pos=now, clamp_late=lambda at: max(0, now+m.lead-at))
    def grid(p):
        return _MasterGrid(anchor=p.stream_sample, slot_n=CYCLE, offset_n=0,
                           packet_n=round(.8*FS), cs_n=round(.244*FS), d_max_n=round(.1*FS))
    if replacement == 'retimed':
        assert _keyable_slot(live, grid(captured), 0, 0, listen=False, shift=False) != 0
    assert _keyable_slot(live, grid(current), 0, 0, listen=False, shift=False) == 0
    assert m.keyed().stream_sample == current.stream_sample


def test_nonmail_identity_scope_preserves_sf8_admission():
    m, _ = machine(role='irs', prekey_lead=.025)
    eligible = [True]
    m.sf8_eligible = lambda obs: eligible[0]
    reg = packet(Kind.NORMAL, 10., 1, b'peer source', width=40)
    reg = replace(reg, observation=replace(reg.observation, sf8=True))
    m.observe(reg, now=10*FS)
    answer = m.decide(10*FS+round(m.t[Kind.NORMAL]*FS)-m.lead)
    assert m.can_key(answer)
    m._next = replace(answer)
    eligible[0] = False
    assert not m.can_key(answer) and not m.can_key(m._next)


@pytest.mark.parametrize('captured_state', [State.ISS_HOLD, State.ISS_HANDBACK])
@pytest.mark.parametrize('terminal', ['done', 'failure', 'expiry'])
def test_managed_mail_stale_idle_and_offer_still_refused(captured_state, terminal):
    clock = [0.]
    m, a, _ = ws8(clock)
    a.on_host_data(b'gated setup')
    m.allow_breakin = False
    now = (m.next_offer_at-m.lead if captured_state == State.ISS_HANDBACK
           else m.last.stream_sample+CYCLE-m.lead)
    captured = m.decide(now)
    assert captured.state == captured_state and m.can_key(captured)
    if terminal == 'expiry':
        m.update_application_progress(1.)
        sample = captured.stream_sample
        clock[0] = 1.  # Deadline crosses during capture; sample time stays fixed.
        assert not m.can_key(captured) and captured.stream_sample == sample
    else:
        m.finish_application(now, failure='terminal failure' if terminal == 'failure' else None)
    assert not m.can_key(captured)
    close = m.decide(now)
    assert close.state == State.ISS_CLOSE and close.status == 0x20 and m.can_key(close)
    assert not m.can_key(captured) and m._next is close
    assert m.close_context.queued == b'gated setup'
    assert a._buffer_raw == len(b'gated setup') and not m.outbox.inflight
    assert a.tx_acked_bytes == 0


def test_p4_non_greeting_progress_timeout_is_transport_neutral():
    clock = [0.]
    m, a = waiting(clock)
    m.state = State.ISS_HOLD
    m._idle(10*FS)
    m.keyed()
    client = MailClient(B2FSession('W9SSJ', target='WS8EOC'), a.on_host_data)
    client.session._state = _THEIR_TURN
    client.progress = MailProgress(client, a, now=lambda: clock[0])
    assert not bridge(client, m, 10*FS) and m.application_deadline == 120.
    stage = client.session.stage
    clock[0] = 120.
    assert bridge(client, m, 120*FS)
    assert client.session.failure == (
        f'application progress timeout while {stage}; transfer unconfirmed')
    close = m.decide(120*FS)
    assert m._closing_failure and close.state == State.ISS_CLOSE
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('gated', [False, True])
def test_empty_checked_grant_retains_actual_offer_schedule(gated):
    m, a, offer = ws8([0.])
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *args: None)
    if gated:
        a.on_host_data(b'gated setup')
    m.allow_breakin = False
    schedule = m._offer_at, m.next_offer_at
    idle_at = m.last.stream_sample
    counter = m.counter
    reg = replace(packet(Kind.BODY, idle_at/FS+.935, 0x40, b'', width=31),
                  responds_to=idle_at)
    m.observe(reg, now=reg.observation.stream_sample)
    assert m.counter == (counter+1)%4 and m._acked_idle == idle_at
    grant_due = reg.observation.stream_sample+round(m.t[Kind.BODY]*FS)
    grant = m.decide(grant_due-m.lead)
    assert (grant.kind, grant.status, grant.payload, grant.state) == (
        Kind.BODY, 0x40, b'', State.ISS_GRANT)
    assert m.can_key(grant)
    m.keyed()
    assert m.last.stream_sample == grant_due
    continuation = refusal(m, name='V27', offset=.935)
    assert (continuation.kind, continuation.payload, continuation.state) == (
        Kind.NORMAL, b'', State.ISS_HOLD)
    m.keyed()
    assert m._handback_attempted and m.state == State.ISS_HOLD
    assert (m._offer_at, m.next_offer_at) == schedule
    assert m._handback_pending is None and not delivered
    assert m.close_context is None and a.tx_acked_bytes == 0
    assert a._buffer_raw == (len(b'gated setup') if gated else 0)
    assert not m.outbox.inflight
    while m.last.stream_sample+CYCLE < schedule[1]:
        refusal(m)
        assert m.decide(m.last.stream_sample+CYCLE-m.lead).state == State.ISS_HOLD
        m.keyed()
    candidate = m.last.stream_sample+CYCLE
    refusal(m)
    reoffer = m.decide(candidate-m.lead)
    assert reoffer.state == State.ISS_HANDBACK
    assert reoffer.stream_sample == candidate >= offer+960000
    assert candidate-CYCLE < offer+960000
    assert m.keyed().stream_sample == candidate
    assert not delivered and a.tx_acked_bytes == 0 and not m.outbox.inflight


@pytest.mark.parametrize('kind,status,data,state', [
    (Kind.BODY, 0, b'fresh source', State.IRS_ANSWER),
    (Kind.NORMAL, 0, b'', State.IRS_ANSWER),
    (Kind.BODY, 0x20, b'', State.IRS_CLOSE),
])
def test_cooldown_peer_receive_and_close_clear_offer(kind, status, data, state):
    m, a, _ = ws8([0.])
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *args: None)
    idle_at = m.last.stream_sample
    reg = replace(packet(kind, idle_at/FS+.935, status, data,
                         width=40 if kind == Kind.NORMAL else 31), responds_to=idle_at)
    m.observe(reg, now=reg.observation.stream_sample)
    p = m.decide(reg.observation.stream_sample+round(m.t[kind]*FS)-m.lead)
    assert p.kind == Kind.WORD and p.state == state and m.can_key(p)
    m.keyed()
    assert m.state == state and m.next_offer_at is None and m._offer_at is None
    assert delivered == ([data] if data else []) and a.tx_acked_bytes == 0


@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_reoffer_never_precedes_actual_offer_plus_twenty_seconds(offset):
    m, _, offer = ws8([0.])
    deadline = offer+960000
    # Displace one actual idle; all later candidates use that actual raster.
    refusal(m)
    p = m.decide(offer+5*CYCLE-m.lead)
    assert p.stream_sample == offer+5*CYCLE and p.state == State.ISS_HOLD
    assert m.keyed(at=p.stream_sample+offset).stream_sample == offer+5*CYCLE+offset
    while True:
        candidate = m.last.stream_sample+CYCLE
        refusal(m)
        p = m.decide(candidate-m.lead)
        assert p.stream_sample == candidate
        if candidate >= deadline:
            assert p.state == State.ISS_HANDBACK and candidate-CYCLE < deadline
            break
        assert p.state == State.ISS_HOLD and m.next_offer_at == deadline
        if candidate == deadline-1:
            assert offset == -1
        assert m.keyed().stream_sample == candidate
    assert candidate == deadline+(CYCLE-1 if offset == -1 else offset)
    actual = m.keyed().stream_sample
    assert actual == candidate and actual >= deadline
    deadline = m.next_offer_at
    m.silence(actual+CYCLE-m.lead)
    repeat = m.decide(actual+CYCLE-m.lead)
    assert repeat.state == State.ISS_HANDBACK and repeat.repeat
    m.keyed()
    assert m._offer_at == actual and m.next_offer_at == deadline
    refusal(m)
    m.keyed()
    p = m.decide(deadline-m.lead)
    assert p.state == State.ISS_HANDBACK and p.stream_sample == deadline
    schedule = m._offer_at, m.next_offer_at, m._hold_until
    m.refused('deadline')
    assert (m._offer_at, m.next_offer_at, m._hold_until) == schedule
    # The unkeyed offer did not reset the interval; its next legal key does.
    assert m.keyed().stream_sample == deadline+CYCLE
    assert m.next_offer_at == deadline+17*CYCLE


def test_generic_hold_readout_only_describes_its_live_interval():
    m, a = machine(data='body', ladder='none', hold_cycles=16)
    m._data(10*FS)
    m.keyed()
    start, end = m._hold_started, m._hold_until
    assert m.waiting_status(start) == ['hold=0/16']
    assert m.waiting_status(start+3*CYCLE) == ['hold=3/16']
    # Reusing the live deadline retains its original origin and elapsed count.
    m._data(start+4*CYCLE)
    assert (m._hold_started, m._hold_until) == (start, end)
    assert m.waiting_status(start+4*CYCLE) == ['hold=4/16']
    a.on_host_data(b'eligible data')
    m._begin_data_turn(start+5*CYCLE)
    assert m._next.payload == b'eligible data'
    # Acquisition is a proposal; only the actual payload key ends this hold.
    assert (m._hold_started, m._hold_until) == (start, end)
    assert m.waiting_status(start+5*CYCLE) == ['hold=5/16']  # Direct helper.
    m.keyed()
    assert m._hold_until is None
    assert m.waiting_status(start+5*CYCLE) == []
    m._commit_data()
    new = start+6*CYCLE
    m._data(new)
    assert m.state == State.ISS_HOLD and m._hold_started == new
    assert m.waiting_status(new) == ['hold=0/16']
    assert m.waiting_status(new+2*CYCLE) == ['hold=2/16']
    deadline = m._hold_until
    assert m.waiting_status(deadline) == [] and m._hold_until == deadline
    m._handback_attempted = True
    assert m.waiting_status(new+2*CYCLE) == []
    m._handback_attempted = False
    m.state = State.ISS_HANDBACK
    assert m.waiting_status(new+2*CYCLE) == []


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('counted', [False, True], ids=['reset', 'counted'])
def test_acquisition_preserves_nonmail_hold_until_payload_key(managed, counted):
    m, a = machine(data='body', ladder='none', hold_cycles=16,
                   normal_ack='parity', managed_application=managed)
    m._data(10*FS)
    m.keyed()
    origin = m._hold_started
    # A full associated seed creates and keys the ordinary four-cycle offer.
    at = m.last.stream_sample+round(.930*FS)
    m.observe(word('V27', at, extent=.245, responds_to=m.last.stream_sample),
              now=at+round(.245*FS))
    offer = m.decide(at+round(.245*FS))
    assert offer.state == State.ISS_HANDBACK
    m.keyed()
    deadline = m._hold_until
    assert deadline == m.last.stream_sample+4*CYCLE
    assert m._hold_started == origin and m.next_offer_at is not None
    a.on_host_data(b'new bytes')
    if counted:
        at = m.last.stream_sample+round(.930*FS)
        m.observe(word('X27', at, extent=.245, responds_to=m.last.stream_sample),
                  now=at+round(.245*FS))
        proposal = m.decide(at+round(.245*FS))
    else:
        m._begin_data_turn(m.last.stream_sample+CYCLE)
        proposal = m.decide(m.last.stream_sample+CYCLE-m.lead)
    assert (proposal.kind, proposal.status, proposal.payload, proposal.state) == (
        Kind.NORMAL if counted else Kind.BODY, 1 if counted else 0x40,
        b'new bytes', State.ISS_DATA)
    assert m.counter == (1 if counted else 0)
    assert not m._handback_attempted
    assert (m._offer_at, m.next_offer_at, m._handback_pending) == (None,)*3
    assert (m._hold_started, m._hold_until) == (origin, None if managed else deadline)
    balance = (a._buffer_raw, a.tx_queued_bytes, a.tx_acked_bytes,
               bytes(a._outbuf), m.outbox.inflight, m.adapter.result)
    assert balance[:3] == (9, 9, 0) and m.outbox.inflight == b'new bytes'
    # Ordinary refusal retimes the same payload without progress or hold renewal.
    m.refused('deadline')
    retry = m.decide(proposal.stream_sample+CYCLE-m.lead)
    assert retry.stream_sample == proposal.stream_sample+CYCLE
    assert retry.payload == proposal.payload and retry.state == State.ISS_DATA
    assert (m._hold_started, m._hold_until) == (origin, None if managed else deadline)
    assert (a._buffer_raw, a.tx_queued_bytes, a.tx_acked_bytes,
            bytes(a._outbuf), m.outbox.inflight, m.adapter.result) == balance
    assert m.can_key(retry)
    actual = m.keyed(at=retry.stream_sample)
    assert actual.payload == b'new bytes' and m._hold_until is None
    assert m._hold_started == origin and m.waiting_status(actual.stream_sample) == (
        ['application_remaining=unarmed'] if managed else [])
    assert (a._buffer_raw, a.tx_queued_bytes, a.tx_acked_bytes,
            bytes(a._outbuf), m.outbox.inflight, m.adapter.result) == balance


@pytest.mark.parametrize('state', [State.ISS_HOLD, State.ISS_GRANT, State.ISS_HANDBACK])
@pytest.mark.parametrize('payload_pending', [False, True])
def test_nonmail_finite_hold_keeps_its_existing_close_predicates(state, payload_pending):
    m, a = machine(data='body', ladder='none', hold_cycles=16)
    m._data(10*FS)
    m.keyed()
    origin, deadline = m._hold_started, m._hold_until
    m.state = state
    assert m.decide(deadline-1) is None or m._next.state == State.ISS_HOLD
    assert m.close_context is None
    # A pending payload already exempts itself from the finite-hold close guard.
    if payload_pending:
        a.on_host_data(b'pending data')
        m._data(deadline+CYCLE)
    else:
        m._idle(deadline+CYCLE)
    pending = m._next
    proposal = m.decide(deadline)
    assert (m._hold_started, m._hold_until) == (origin, deadline)
    assert a.tx_acked_bytes == 0
    if payload_pending:
        assert proposal is pending and proposal.state == State.ISS_DATA
        assert m.close_context is None and m.outbox.inflight == b'pending data'
        assert m.can_key(proposal)
    else:
        assert (proposal.kind, proposal.status, proposal.state) == (
            Kind.BODY, 0x20, State.ISS_CLOSE)
        assert proposal.stream_sample == deadline+m.lead
        assert m.close_context is not None and a._buffer_raw == 0


@pytest.mark.parametrize('managed', [False, True])
def test_reset_ladder_acquisition_keeps_managed_cleanup_and_nonmail_hold(managed):
    m, a = machine(data='body', ladder='long', hold_cycles=16,
                   managed_application=managed)
    m._data(10*FS)
    m.keyed()
    origin, deadline = m._hold_started, m._hold_until
    a.on_host_data(b'ladder data')
    m._begin_data_turn(10*FS+CYCLE, word_yield=True)
    proposal = m._next
    assert (proposal.kind, proposal.state, proposal.index, proposal.payload) == (
        Kind.SHORT, State.ISS_LADDER, 0, b'')
    expected = origin, None if managed else deadline
    assert (m._hold_started, m._hold_until) == expected
    assert a._buffer_raw == len(b'ladder data') and not m.outbox.inflight
    m.refused('deadline')
    assert m._next.stream_sample == proposal.stream_sample+CYCLE
    assert (m._hold_started, m._hold_until) == expected
    assert m.can_key(m._next)
    m.keyed()
    assert (m._hold_started, m._hold_until) == expected  # SHORT is not payload.
    # Exercise the existing completion helper; ladder negotiation is unchanged.
    m._after_ladder(m.last.stream_sample+CYCLE)
    assert m._next.payload == b'ladder data'
    assert (m._hold_started, m._hold_until) == expected
    assert m.can_key(m._next)
    m.keyed()
    assert m._hold_until is None and m._hold_started == origin
    assert a._buffer_raw == len(b'ladder data') and a.tx_acked_bytes == 0
    assert m.outbox.inflight == b'ladder data'
