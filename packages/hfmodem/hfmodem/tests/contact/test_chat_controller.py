# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import FrozenInstanceError, fields, replace
from threading import Event, get_ident
from uuid import uuid4

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_controller import Controller, Disposition, TERMINAL
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.contact import varac as v


def request(action, *, controller_id='test', generation=0, actor='view-a', rid=None,
            control_id=None, control_generation=None):
    return s.Request(rid or str(uuid4()), controller_id, generation, actor,
                     control_id, control_generation, action)


class Port:
    def __init__(self):
        initial = s.initial_snapshot()
        self.state = replace(initial, tx=replace(initial.tx, physical_line=False))
        self.on_event = lambda *args: None
        self.calls = []
        self.threads = set()
        self.entered, self.release = Event(), Event()
        self.block = False
        self.poll_error = self.quit_error = None
        self.quit_count = 0
        self.events = []

    def note(self, method):
        self.threads.add(get_ident())
        self.calls.append(method)

    def project(self):
        self.note('project')
        return self.state

    def poll(self, timeout_s):
        self.note('poll')
        if self.poll_error:
            raise self.poll_error
        events, self.events = self.events, []
        for event in events:
            self.on_event(*event)

    def preflight(self, action):
        self.note('check')
        return s.Capability(action.kind, s.effect(action), 'allowed', None, None, None)

    def execute(self, req):
        self.note(req.action.kind)
        if self.block:
            self.entered.set()
            assert self.release.wait(5)
        return Disposition('done', s.Applied(1))

    def quit(self, *, unwinding=None):
        self.note('quit')
        self.quit_count += 1
        if self.quit_error:
            raise self.quit_error

    def legacy_line(self, line):
        self.note('legacy')


@pytest.fixture
def controlled():
    p = Port()
    c = Controller(p, controller_id='test')
    c.ready.result(2)
    yield c, p
    p.release.set()
    c.shutdown().result(2)
    assert c.join(2)


def sync(controller):
    controller.check(s.VerboseSet(False)).result(2)


def completed(controller, req):
    assert controller.submit(req).queued
    for _ in range(100):
        sync(controller)
        result = controller.outcome(req.id)
        if isinstance(result, s.RequestResult) and result.state.state in TERMINAL:
            return result
    pytest.fail('request did not terminate')


def test_immutable_snapshots_roundtrip_and_closed_schema(controlled):
    c, _ = controlled
    snap = c.snapshot()
    assert s.decode(s.encode(snap)) == snap
    assert s.decode(s.encode(c.projection())) == c.projection()
    with pytest.raises(FrozenInstanceError):
        snap.link.generation = 9
    assert not hasattr(snap, '__dict__')
    wire = s.to_wire(request(s.ChatSend('hello')))
    wire['action']['text'] = []
    with pytest.raises(ValueError):
        s.from_wire(wire)
    wire = s.to_wire(snap)
    wire['unexpected'] = 1
    with pytest.raises(ValueError):
        s.from_wire(wire)


def test_owner_only_access_and_poll_before_next_ordinary_action(controlled):
    c, p = controlled
    completed(c, request(s.VerboseSet(True)))
    completed(c, request(s.VerboseSet(False)))
    assert len(p.threads) == 1 and get_ident() not in p.threads
    first = p.calls.index('verbose.set')
    second = p.calls.index('verbose.set', first + 1)
    assert 'poll' in p.calls[first + 1:second]


def test_blocking_command_isolation_queue_pressure_and_priority(controlled):
    c, p = controlled
    p.block = True
    first = request(s.ChatSend('blocked'))
    assert c.submit(first).queued and p.entered.wait(2)
    snap = c.snapshot()
    assert snap is c.snapshot()
    assert isinstance(c.projection(), s.Projection)
    queued = [request(s.ChatSend(str(n))) for n in range(64)]
    assert all(c.submit(r).queued for r in queued)
    assert c.submit(request(s.ChatSend('overflow'))).reason == 'queue-full'
    stop = request(s.ChatCancelWait())
    assert c.submit(stop).queued
    p.block = False
    p.release.set()
    for _ in range(70):
        sync(c)
    actions = [name for name in p.calls if name in ('chat.send', 'chat.cancel_wait')]
    assert actions[:2] == ['chat.send', 'chat.cancel_wait']


def test_duplicate_ids_do_not_execute_twice_and_conflicts_refuse(controlled):
    c, p = controlled
    req = request(s.ChatSend('one'))
    result = completed(c, req)
    assert c.submit(req).queued
    assert c.outcome(req.id) is result
    assert not c.submit(replace(req, action=s.ChatSend('two'))).queued
    assert p.calls.count('chat.send') == 1
    assert c.submit(replace(req, id='old', controller_id='old')).reason == 'controller-stale'


def test_stale_generation_and_authority_are_rechecked_on_owner(controlled):
    c, p = controlled
    p.state = replace(p.state, link=replace(p.state.link, generation=5))
    sync(c)
    assert completed(c, request(s.ChatSend('old'))).state.reason_code == 'link-stale'
    p.state = replace(p.state, ownership=s.Ownership('station', 'a', 'lease', 8, None, 'ready', True))
    sync(c)
    req = request(s.ChatSend('stale lease'), generation=5, control_id='lease', control_generation=7)
    assert completed(c, req).state.reason_code == 'authority-stale'
    assert 'chat.send' not in p.calls


def test_read_bookkeeping_stability_and_projection_section_completeness(controlled):
    c, p = controlled
    before = c.snapshot()
    for _ in range(3):
        assert completed(c, request(s.ChatProjection())).state.state == 'done'
    after = c.snapshot()
    assert after.revision == before.revision
    assert after.requests == ()
    assert {f.name for f in fields(c.projection())} == {
        'controller_id', 'revision', 'through_seq', 'sampled_at', 'utc', 'live',
        'application', 'receive', 'capabilities', 'requests', 'omitted_sections'}
    assert not hasattr(c.projection(), 'presentation')
    draft = s.Draft('d', 2, 'other', 'vmail', 0, 'N0CALL', 's', None, None, ('line',))
    p.state = replace(p.state, application=replace(p.state.application, draft=draft,
        files=replace(p.state.application.files, received_bytes=17)))
    sync(c)
    assert c.projection().application.draft == draft
    assert c.projection().application.files.received_bytes == 17
    check = c.check(s.ChatSend('specific')).result(2)
    assert check.action == s.ChatSend('specific') and check.controller_id == c.controller_id
    assert 'chat.send' not in p.calls


def test_shutdown_once_error_chaining_and_no_daemon_abandonment():
    p = Port()
    c = Controller(p)
    c.ready.result(2)
    p.quit_error = ValueError('cleanup')
    p.poll_error = RuntimeError('initiating')
    for _ in range(100):
        if not c._thread.is_alive():
            break
        c._thread.join(.02)
    future = c.shutdown()
    assert c.shutdown() is future
    with pytest.raises(RuntimeError, match='initiating') as error:
        future.result(2)
    assert str(error.value.__cause__) == 'cleanup'
    assert p.quit_count == 1 and c.join(2) and not c._thread.daemon


def adapter(*, connected=False, **kwargs):
    desk = Desk(**kwargs)
    port = SessionAdapter(desk.session)
    events = []
    port.on_event = lambda *args: events.append(args)
    port._listening = True
    if connected:
        desk.flow.dial = 14106500
        desk.hear('CONNECTED W9SSJ N0XYZ 500')
    port.project()
    return desk, port, events


def execute(port, action, **kwargs):
    req = request(action, generation=port.session.client.link_generation, **kwargs)
    return req, port.execute(req)


def test_real_projection_sources_nonzero_monitor_offset_and_unknowns():
    desk, port, _ = adapter(connected=True)
    desk.session.band_setup = {'mode': 'USB', 'offset': 1200}
    port._receive = replace(port._receive, wide_offset=-1500.)
    desk.host.client.buffer = 31
    desk.host.client.buffer_sequence = 7
    desk.host.client._sent_after = 8
    desk.session._away_owed = desk.host.client.link_generation
    desk.session.hopping.readiness[20] = 'ready'
    snap = port.project()
    assert snap.identity.channel_hz == desk.flow.dial == 14106500
    assert snap.receive.wide_offset == -1500
    assert snap.identity.setup == (('mode', 'USB'), ('offset', '1200'))
    assert snap.tx.buffer_bytes == 31 and snap.tx.unreported and snap.tx.sent_after == 8
    assert snap.link.contact_number == desk.session.contact['n']
    assert snap.link.started_utc == desk.session.contact['started_utc']
    assert snap.policy.away_owed_generation == desk.session._away_owed
    assert snap.beacon.hop.readiness == ((20, 'ready'),)
    assert snap.tx.physical_line is None
    assert s.decode(s.encode(snap)) == snap


def test_real_preflight_pure_including_unverified_broadcast_line():
    desk, port, _ = adapter()
    baseline = desk.sent[:], desk.texts[:], desk.tunes[:], desk.armed[:]
    for a in (s.BroadcastSelect(None), s.BroadcastSend('hello'), s.BeaconOnce(False),
              s.ChatCall('W1AW', None, True), s.AwaySet('out'), s.AutoSet(True)):
        assert isinstance(port.preflight(a), s.Capability)
    assert (desk.sent, desk.texts, desk.tunes, desk.armed) == baseline


def test_real_send_admission_not_buffer_ack_and_literal_text():
    desk, port, events = adapter(connected=True)
    req, result = execute(port, s.ChatSend('[bold]hello'))
    assert result.state == 'done' and isinstance(result.result, s.WriteAccepted)
    assert desk.texts[-1].endswith('[bold]hello\n')
    desk.hear('BUFFER 0')
    desk.session.diagnostic('acknowledged', bytes=17)
    port._flush()
    assert any(isinstance(e[0], s.TxCredit) for e in events)
    assert all(e[0].status == 'queued' for e in events if isinstance(e[0], s.TxTextOutcome))
    assert port._results[req.id] == result


def test_draft_identity_revision_replacement_two_views_and_literal_period():
    desk, port, _ = adapter(connected=True)
    _, result = execute(port, s.VMailCompose('first'))
    draft = result.result.draft
    _, refused = execute(port, s.VMailCompose('other'), actor='view-b')
    assert refused.reason_code == 'draft-owned'
    _, appended = execute(port, s.DraftAppend(draft.id, 0, '.'))
    assert appended.result.draft.lines == ('.',) and not desk.texts
    _, stale = execute(port, s.VMailSend(draft.id, 0))
    assert stale.reason_code == 'draft-stale'
    _, owned = execute(port, s.DraftDrop(draft.id, 1), actor='view-b')
    assert owned.reason_code == 'draft-owned'
    _, replaced = execute(port, s.VMailCompose('replacement'))
    assert replaced.result.draft.id != draft.id
    _, stale = execute(port, s.DraftAppend(draft.id, 1, 'old'))
    assert stale.reason_code == 'draft-stale'


def test_interrupted_multiline_upload_keeps_draft_and_does_not_send():
    desk, port, _ = adapter(connected=True)
    desk.receive('<S:E>')
    _, composed = execute(port, s.EmailCompose('a@example.org', 'b@example.org', 'subject'))
    draft = composed.result.draft
    _, first = execute(port, s.DraftAppend(draft.id, 0, 'one'))
    assert first.result.draft.revision == 1
    _, refused = execute(port, s.DraftAppend(draft.id, 1, 'x' * 501))
    assert refused.state == 'refused'
    assert port.project().application.draft.lines == ('one',)
    assert not any('<SM>' in text for text in desk.texts)


def test_queued_bbs_cancellation_releases_reservation_without_write():
    desk, port, _ = adapter(connected=True)
    desk.flow.dial = v.CALLING_HZ[20]
    req, result = execute(port, s.BBSList())
    assert result.stage == 'queued'
    assert port.preflight(s.BBSList()).reservation.state == 'pending'
    execute(port, s.ChatCancelWait())
    assert port._results[req.id].state == 'cancelled'
    assert port.preflight(s.BBSList()).reservation.state == 'available'
    assert not any('<BLR>' in text for text in desk.texts)


@pytest.mark.parametrize('family,action,reply', [
    ('directory', s.BBSList(), '<BL:one.txt|2026-01-01|5>'),
    ('mail', s.VMailSend('unused', 0), '<SMR>'),
    ('term', s.BBSTerm('ansi16', 80, 24), 'BBS TERM OK ANSI16 80 24'),
])
def test_late_uncorrelated_reply_cannot_settle_reissued_service(family, action, reply):
    desk, port, events = adapter(connected=True)
    if family == 'mail':
        _, comp = execute(port, s.VMailCompose('hello'))
        action = s.VMailSend(comp.result.draft.id, 0)
    elif family == 'term':
        desk.host.client.enable_writer = lambda: None
        execute(port, s.BBSEnter())
    req, _ = execute(port, action)
    assert req.id in port._sent
    port._finish(req.id, 'unknown', code='cancelled-wait')
    before = port._results[req.id]
    assert port.preflight(action).reservation.state == 'spent'
    _, refused = execute(port, action)
    assert refused.state == 'refused' and refused.reason_code == 'reconnect-required'
    desk.receive(reply)
    port._flush()
    assert port._results[req.id] == before
    assert sum(isinstance(e[0], Disposition) and e[1] == req.id and e[0].state in TERMINAL for e in events) == 1


@pytest.mark.parametrize('second', [s.BeaconOnce(False), s.BeaconOnce(True)])
def test_one_shot_reservation_normal_to_normal_or_force(second):
    desk, port, events = adapter()
    req, result = execute(port, s.BeaconOnce(False))
    assert result.state == 'running'
    assert port.preflight(second).reason_code == 'beacon-pending'
    execute(port, s.BeaconConfigure(10))
    assert port._beacon_id == req.id and desk.session._beacon_now == 'now'
    execute(port, s.BeaconConfigure(None))
    assert port._results[req.id].state == 'cancelled'
    assert port.preflight(second).reservation.state == 'available'
    assert sum(isinstance(e[0], Disposition) and e[1] == req.id and e[0].state in TERMINAL for e in events) == 1


def test_one_shot_expiry_restore_legacy_replacement_and_start_failure():
    desk, port, _ = adapter()
    execute(port, s.BeaconConfigure(10))
    req, _ = execute(port, s.BeaconOnce(False))
    desk.session._beacon_expires = desk.host.client.clock() - 1
    desk.session._beacon_tick()
    assert req.id not in port._results and port._beacon_id == req.id
    port.legacy_line('/beacon force')
    assert port._results[req.id].state == 'cancelled' and port._beacon_id is None
    assert port.preflight(s.BeaconOnce(False)).reason_code == 'beacon-pending'
    desk.session._beacon_now = False
    req, _ = execute(port, s.BeaconOnce(True))
    desk.session._leave = lambda *args: (_ for _ in ()).throw(v.HostError('admission failure'))
    desk.session._beacon_tick()
    assert port._results[req.id].state == 'failed'


def test_swallowed_generator_exception_is_failed_and_false_call_is_not_success():
    desk, port, _ = adapter()
    def failure(*args, **kwargs):
        yield .1
        raise v.HostError('normal backend failure')
    desk.session._connecting = failure
    req, result = execute(port, s.ChatCall('W1AW', 2, False))
    assert result.state == 'running'
    desk.session._step()
    assert port._results[req.id].state == 'failed'
    def false(*args, **kwargs):
        yield .1
        return False
    desk.session._connecting = false
    req, _ = execute(port, s.ChatCall('W1AW', 2, False))
    desk.session._step()
    assert port._results[req.id].state == 'cancelled'


def test_controller_bbs_context_revision_and_loss_latch_require_reconnect(controlled):
    c, p = controlled
    p.state = replace(p.state, application=replace(p.state.application,
                      bbs=replace(p.state.application.bbs, remote_view=True)))
    sync(c)
    context = c.snapshot().presentation.bbs_context
    p.events.append((s.BBSPage('p', 'plain', 80, 24, (s.StyledSpan('Menu', None, None, False),)), None, 0))
    sync(c)
    revision = c.snapshot().presentation.bbs_revision
    req = request(s.BBSInput(context, revision, '1'))
    assert completed(c, req).state.state == 'done'
    assert completed(c, replace(req, id='second', actor='b')).state.reason_code == 'bbs-stale-view'
    c.max_records = 1
    p.events.append((s.BBSPage('p2', 'plain', 80, 24, (s.StyledSpan('Next', None, None, False),)), None, 0))
    sync(c)
    assert c.snapshot().presentation.bbs_reconnect_required
    assert completed(c, request(s.BBSEnter())).state.reason_code == 'bbs-reconnect-required'
    p.state = replace(p.state, application=replace(p.state.application,
                      bbs=replace(p.state.application.bbs, remote_view=False)))
    sync(c)
    p.state = replace(p.state, application=replace(p.state.application,
                      bbs=replace(p.state.application.bbs, remote_view=True)))
    sync(c)
    assert c.snapshot().presentation.bbs_reconnect_required
    p.state = replace(p.state, link=replace(p.state.link, generation=2, connected=False))
    sync(c)
    assert not c.snapshot().presentation.bbs_reconnect_required


def test_overflow_recovery_retains_tx_and_text_with_atomic_watermark(controlled):
    c, p = controlled
    sub = c.subscribe(0)
    p.events.extend((s.TextRecord(str(n), 'rx', 'message', 'W1AW', 'contact'), None, 0) for n in range(150))
    sync(c)
    assert any(isinstance(e.payload, s.Gap) for e in sub.drain())
    snap = c.snapshot()
    assert len(snap.presentation.records) == 150
    p.events.append((s.TxTextOutcome('149', 1, 'acknowledged', 7, 7, 0, 0, 'verified'), None, 0))
    sync(c)
    recovered = c.snapshot()
    assert recovered.presentation.records[-1].tx.status == 'acknowledged'
    assert recovered.presentation.records[-1].updated_seq <= recovered.through_seq
    assert recovered.through_seq > snap.through_seq
    assert not any(name.startswith('chat.') for name in p.calls)


def test_oversized_record_explicit_marker_and_live_state_invalidation(controlled):
    c, p = controlled
    sub = c.subscribe(0)
    p.events.append((s.TextRecord('huge', 'rx', '\\"' * 60000, None, 'contact'), None, 0))
    sync(c)
    record = c.snapshot().presentation.records[-1]
    assert len(s.encode(record)) <= 48 * 1024
    assert record.payload.text.endswith('[content omitted]')
    p.state = replace(p.state, identity=replace(p.state.identity, setup=(('huge', 'x' * 12000),)))
    c._last_publish = float('-inf')
    sync(c)
    changed = [e.payload for e in sub.drain() if isinstance(e.payload, s.StateChanged)]
    assert changed[-1].state is None


@pytest.mark.parametrize('incoming', ['none', 'pending-connected', 'cancelled-child', 'direct-connected'])
def test_real_client_two_cq_frames_do_not_manufacture_attempts(incoming):
    from hfmodem.tests.contact.test_varac import CQ_DONE
    desk, port, events = adapter()
    command = desk.flow.codec.cq_command('W9SSJ', 2, 500)
    desk.host.replies[command] = CQ_DONE
    desk.flow.opening_steps = lambda: iter(())
    req, result = execute(port, s.ChatCQ(2, False))
    assert result.state == 'running'
    for _ in range(1500):
        port.poll(.1)
        if desk.flow.state is v.State.WAITING:
            break
    assert desk.sent.count(command) == 2
    assert port._attempts == {} and port._frame is None
    if incoming == 'pending-connected':
        desk.host.lines('PENDING', 'CONNECTED W9SSJ W1AW 500')
    elif incoming == 'cancelled-child':
        desk.host.lines('PENDING', 'CANCELPENDING')
    elif incoming == 'direct-connected':
        desk.host.lines('CONNECTED W9SSJ W1AW 500')
    for _ in range(100):
        port.poll(.1)
        if req.id in port._results:
            break
    result = port._results[req.id]
    assert result.state == 'done'
    assert result.result.result == ('connected' if incoming.endswith('connected') else 'no-answer')
    if incoming == 'none':
        assert result.result.attempt_id is None and result.result.generation is None
    else:
        assert len(port._attempts) == 1
        assert result.result.attempt_id and result.result.generation == desk.host.client.link_generation
        assert port._attempts[result.result.attempt_id]['request_id'] == req.id
    assert sum(isinstance(e[0], Disposition) and e[1] == req.id and e[0].state in TERMINAL for e in events) == 1


def test_cq_busy_before_frame_retains_parent_without_attempt():
    desk, port, _ = adapter()
    desk.flow.occupied = lambda: True
    req, _ = execute(port, s.ChatCQ(2, False))
    for _ in range(100):
        port.poll(.1)
        if req.id in port._results:
            break
    assert port._results[req.id].result == s.ConnectionResult(None, None, 'busy')
    assert not any(line.startswith('CQFRAME ') for line in desk.sent)


@pytest.mark.parametrize('advanced,force', [(False, False), (True, False), (False, True), (True, True)])
def test_real_client_beacon_frame_then_genuine_incoming(advanced, force):
    from hfmodem.tests.contact.test_varac import CQ_DONE, BEACON
    desk, port, _ = adapter(kiss=advanced, loc='EN63BC' if advanced else None)
    command = 'BEACONFORCE ' + desk.session.beacon_text() if advanced and force else (
        'CQFRAMEFORCE W9SSJ-9 500' if force else BEACON)
    desk.host.replies[command] = CQ_DONE
    req, _ = execute(port, s.BeaconOnce(force))
    for _ in range(100):
        port.poll(.1)
        if req.id in port._results:
            break
    assert port._results[req.id].result == s.BeaconResult('sent')
    assert not port._attempts
    desk.host.lines('PENDING', 'CANCELPENDING')
    port.poll(.1)
    assert len(port._attempts) == 1
    attempt = next(iter(port._attempts.values()))
    assert attempt['request_id'] is None and attempt['ended']
    desk.host.lines('PENDING', 'CONNECTED W9SSJ W1AW 500')
    port.poll(.1)
    assert len(port._attempts) == 2 and port.project().link.connected


def test_outbound_connect_pending_does_not_rebind_and_success_waits_for_opening():
    desk, port, events = adapter(**{'CONNECT W9SSJ W1AW': ['OK', 'PENDING']})
    req, result = execute(port, s.ChatCall('W1AW', None, False))
    assert result.state == 'running'
    attempt_id = port._attempt
    generation = desk.host.client.link_generation
    desk.host.lines('PENDING', 'CONNECTED W9SSJ W1AW 500')
    port.poll(.1)
    result = port._results[req.id]
    assert result.state == 'done'
    assert result.result == s.ConnectionResult(attempt_id, generation, 'connected')
    assert len(port._attempts) == 1
    bound = [e for e in events if isinstance(e[0], Disposition) and e[0].state == 'running'
             and e[0].attempt_id == attempt_id]
    assert len(bound) == 1


def test_one_shot_off_during_tx_keeps_owner_until_actual_completion():
    from hfmodem.tests.contact.test_varac import BEACON
    desk, port, events = adapter(**{BEACON: ['OK', 'PENDING', 'PTT ON']})
    req, _ = execute(port, s.BeaconOnce(False))
    for _ in range(100):
        port.poll(.1)
        if desk.host.client.ptt:
            break
    assert desk.host.client.ptt and port._beacon_running
    execute(port, s.BeaconConfigure(None))
    assert req.id not in port._results and port._beacon_id == req.id
    execute(port, s.ChatCancelWait())
    assert req.id not in port._results
    desk.host.lines('PTT OFF', 'CANCELPENDING')
    for _ in range(10):
        port.poll(.1)
    assert port._results[req.id].result == s.BeaconResult('sent')
    assert sum(isinstance(e[0], Disposition) and e[1] == req.id and e[0].state in TERMINAL for e in events) == 1


def test_coverage_observed_even_when_follow_disabled_and_expires():
    from hfmodem.contact.slots import Said
    desk, port, _ = adapter()
    assert desk.session.following is None
    desk.session._slot_said(Said(2, 100., 'coverage', 'gap', duration_s=12.))
    coverage = port.project().receive.coverage
    assert coverage[0].lost_seconds_60s == 12 and coverage[0].share_60s == .2
    desk.session._slot_said(Said(3, 100., 'coverage', 'backlog'))
    assert port.project().receive.coverage[-1].share_60s is None
    desk.host.now += 61
    assert port.project().receive.coverage == ()


def test_unrelated_connected_replacement_cannot_inherit_service_or_draft():
    desk, port, _ = adapter(connected=True)
    req, _ = execute(port, s.BBSList())
    generation = desk.host.client.link_generation
    desk.hear('CONNECTED W9SSJ K1ABC 500')
    assert desk.host.client.link_generation > generation
    assert port._results[req.id].state == 'unknown'
    assert port._results[req.id].reason_code == 'link-replaced'
    assert port.project().link.peer == 'K1ABC'
    assert port.preflight(s.BBSList()).reservation.state == 'available'


def test_legacy_compose_cannot_be_adopted_by_a_replacement_link():
    desk, port, _ = adapter(connected=True)
    _, composed = execute(port, s.VMailCompose('old contact'))
    draft = composed.result.draft
    assert port.project().application.draft.id == draft.id
    desk.hear('CONNECTED W9SSJ K1ABC 500')
    port.poll(0)
    assert port.project().application.draft is None
    assert port.project().application.draft is None
    _, stale = execute(port, s.VMailSend(draft.id, draft.revision))
    assert stale.state == 'refused'
    assert not desk.texts


def test_cancelled_check_does_not_cancel_owner_or_shared_shutdown(controlled):
    c, p = controlled
    p.block = True
    assert c.submit(request(s.ChatSend('block'))).queued
    assert p.entered.wait(2)
    future = c.check(s.VerboseSet(True))
    assert future.cancel()
    p.block = False
    p.release.set()
    sync(c)
    assert c._thread.is_alive()
    completion = c.shutdown()
    assert not completion.cancel()
    completion.result(2)


@pytest.mark.parametrize('line,action', [
    ('/bbs list', s.BBSList()),
    ('/email send', s.EmailSend('missing', 0)),
    ('/vmail send', s.VMailSend('missing', 0)),
    ('/bbs term plain 80 24', s.BBSTerm('plain', 80, 24)),
    ('/bbs get absent.txt', s.BBSGet('absent.txt')),
    ('/bbs request absent.txt', s.BBSRequest('absent.txt')),
])
def test_refused_legacy_service_does_not_spend_next_link(line, action):
    desk, port, _ = adapter()
    port.legacy_line(line)
    assert not desk.texts
    assert port.preflight(action).reservation.state == 'available'
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    assert port.preflight(action).reservation.state == 'available'


@pytest.mark.parametrize('line,action', [
    ('Try /bbs list later', s.BBSList()),
    ('Try /email send later', s.EmailSend('missing', 0)),
])
def test_legacy_chat_mentions_do_not_spend_service(line, action):
    desk, port, _ = adapter(connected=True)
    port.legacy_line(line)
    assert line in desk.texts[-1]
    assert port.preflight(action).reservation.state == 'available'


@pytest.mark.parametrize('incoming', ['direct', 'pending', 'outbound'])
def test_new_link_generation_clears_legacy_service_reservations(incoming):
    desk, port, _ = adapter()
    desk.host.client.request('LISTEN ON')
    port._reservations['directory'] = s.ServiceReservation('directory', 'spent', None)
    assert port.preflight(s.BBSList()).reservation.state == 'spent'
    generation = desk.host.client.link_generation
    if incoming == 'pending':
        desk.hear('PENDING')
    elif incoming == 'outbound':
        desk.host.replies['CONNECT W9SSJ N0XYZ'] = ['OK', 'PENDING']
        execute(port, s.ChatCall('N0XYZ', None, False))
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    assert desk.host.client.link_generation > generation
    assert port.preflight(s.BBSList()).reservation.state == 'available'


@pytest.mark.parametrize('command,listening', [
    ('LISTEN ON', True), ('LISTEN CQ', True), ('LISTEN OFF', False),
])
def test_send_command_tracks_listen_state_without_changing_wire(command, listening):
    desk, _, _ = adapter()
    client = desk.host.client
    client.listening = not listening
    before = len(desk.host.command.sent)
    client.send_command(command)
    assert client.listening is listening
    assert desk.host.command.sent[before:] == [command.encode('ascii') + b'\r']


def test_poll_waits_for_reload_hook_despite_unrelated_error(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    desk, port, _ = adapter()
    desk.session.bbs.core = SimpleNamespace()
    desk.session.bbs.reload = Future
    req, result = execute(port, s.BBSReload())
    assert result.state == 'running'
    desk.session._error(v.HostError('unrelated earlier error'))
    def tick(timeout):
        desk.session.bbs.reload_future = None
    monkeypatch.setattr(desk.session, 'tick', tick)
    port.poll(0)
    assert req.id not in port._results
    desk.session.semantic('reload_finished', revision=23)
    assert port._results[req.id].result == s.ReloadResult(23)
    assert port._results[req.id].state == 'done'


def test_history_fill_encodes_bounded_records_per_line(monkeypatch):
    c = Controller(Port(), clock=lambda: 0., utc=lambda: 'now')
    c.ready.result(2)
    c.shutdown().result(2)
    assert c.join(2)
    c._owner = get_ident()
    original = s.to_wire
    work = 0
    def counted(value):
        nonlocal work
        if isinstance(value, s.DisplayRecord):
            work += 1
        return original(value)
    monkeypatch.setattr(s, 'to_wire', counted)
    for n in range(2001):
        work = 0
        c._event(s.TextRecord(str(n), 'rx', 'hello', None, 'contact'))
        assert work <= 6
    assert len(c.snapshot().presentation.records) == 2000


def test_idle_refresh_does_not_encode_snapshot_or_history(monkeypatch):
    c = Controller(Port(), clock=lambda: 0., utc=lambda: 'now')
    c.ready.result(2)
    c.shutdown().result(2)
    assert c.join(2)
    c._owner = get_ident()
    for n in range(20):
        c._event(s.TextRecord(str(n), 'rx', 'hello', None, 'contact'))
    original = s.encode
    encoded = []
    def counted(value):
        encoded.append(value)
        return original(value)
    monkeypatch.setattr(s, 'encode', counted)
    for _ in range(10):
        c._refresh()
    assert encoded == []


def test_line_command_does_not_repeat_typed_application_preflight(monkeypatch):
    desk, _, _ = adapter(connected=True)
    def unexpected():
        pytest.fail('typed application preflight reached from line command')
    monkeypatch.setattr(desk.session, 'check_application_input', unexpected)
    desk.session.command('hello')
    assert desk.texts[-1].endswith('hello\n')



def test_listen_off_does_not_attribute_incoming_connect_to_cq():
    desk, port, _ = adapter()
    req = request(s.ChatCQ(2, False))
    port._requests[req.id] = req
    desk.host.client.request('LISTEN CQ')
    desk.host.client.send_command('LISTEN OFF')
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    assert not port._attempts
    assert port._results[req.id].reason_code == 'link-replaced'


def test_incremental_byte_totals_match_wire_after_updates_and_eviction():
    p = Port()
    c = Controller(p, max_records=3, history_bytes=1200)
    c.ready.result(2)
    c.shutdown().result(2)
    assert c.join(2)
    c._owner = get_ident()
    for n in range(8):
        c._event(s.TextRecord(str(n), 'tx', 'é\\"\n' * 20, None, 'contact'))
        c._event(s.TxTextOutcome(str(n), n, 'partial', 100, 40, 0, 60, 'verified'))
        snap = c.snapshot()
        assert c._record_bytes == sum(len(s.encode(r)) for r in snap.presentation.records)
        assert c._event_bytes == sum(len(s.encode(e)) for e in c._events)
        assert c._snapshot_bytes(snap) == len(s.encode(snap))
        assert c._record_bytes <= c.history_bytes
    p.state = replace(p.state, application=replace(p.state.application,
        mailbox=(s.MailHeader(1, 'm', 'rx', 'W1AW', 'W9SSJ', 'é' * (3 * 1024 * 1024), 'now'),)))
    c._refresh()
    snap = c.snapshot()
    assert snap.application.mailbox == ()
    assert 'application.mailbox' in snap.omitted_sections
    assert c._snapshot_bytes(snap) == len(s.encode(snap)) <= 4 * 1024 * 1024
    p.state = replace(p.state, identity=replace(p.state.identity, mycall='x' * (4 * 1024 * 1024)))
    with pytest.raises(ValueError, match='required state exceeds snapshot budget'):
        c._refresh()
