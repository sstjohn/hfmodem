# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from importlib.util import find_spec
import socket
import sys
from types import SimpleNamespace

import pytest

from hfmodem.bbs.presentation import StyleParser
from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_controller import Controller, Disposition
from hfmodem.contact.chat_fake import FakeSession, ScenarioStep, SCENARIOS, patched

if find_spec('textual') is None:
    pytest.skip('install hfmodem[tui] to run the Textual prototype tests', allow_module_level=True)

from hfmodem.ui.textual_chat import (ActionPalette, ChatApp, Composer, Input, MailScreen,
                                     TabbedContent, events, parse_action, plain, styled)


def run(coroutine):
    asyncio.run(coroutine)


@asynccontextmanager
async def screen(*, size=(80, 24), scenario=(), connected=True, actor='view', **controller_options):
    fake = FakeSession(scenario=scenario)
    if connected:
        fake.state = patched(fake.state, (
            ('link.connected', True), ('link.state', 'established'), ('link.peer', 'W1AW'),
            ('link.flow_state', s.State.CONNECTED), ('identity.channel_hz', 14106500),
            ('operation.phase', s.Phase.READY_SLOT),
        ))
    controller = Controller(fake, clock=fake.clock, **controller_options)
    app = ChatApp(controller, actor=actor)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        yield app, pilot, fake


async def settle(pilot, fake, seconds=.25):
    app = pilot.app
    await asyncio.wrap_future(fake.advance(seconds))
    await asyncio.wrap_future(app.controller.check(s.ChatCancelWait()))
    app.collect_updates()
    await pilot.pause()
    cached = app.controller.projection()
    stamp = cached.revision, cached.through_seq
    if app._dirty and all(all(a >= b for a, b in zip(stamp, floor))
                          for floor in app._dirty.values()):
        next_read = max(app._last_projection + .2, app._next_recovery_at)
        await asyncio.wrap_future(fake.advance(max(0., next_read - fake.clock()) + .001))
        await asyncio.wrap_future(app.controller.check(s.ChatCancelWait()))
        app.collect_updates()
        await pilot.pause()
    if not app._dependent_stale() and not app.query_one(Composer).buffering:
        outcomes = [app.controller.outcome(request.id) for request in fake.trace]
        if outcomes and all(isinstance(result, s.RequestResult) and
                            result.state.state in ('done', 'failed', 'refused', 'cancelled', 'unknown')
                            for result in outcomes):
            for worker in tuple(app.workers):
                if worker.group == 'composer':
                    await worker.wait()


def test_enter_sends_exactly_one_line_and_typing_period_does_not_send():
    async def check():
        async with screen() as (app, pilot, fake):
            editor = app.query_one(Composer)
            editor.load_text('hello [bold] literal')
            await pilot.press('enter')
            await settle(pilot, fake)
            assert [r.action for r in fake.trace] == [s.ChatSend('hello [bold] literal')]
            assert editor.text == ''
            await pilot.press('.')
            await settle(pilot, fake)
            assert editor.text == '.' and len(fake.trace) == 1
            await pilot.press('enter')
            await settle(pilot, fake)
            assert fake.trace[-1].action == s.ChatSend('.')
    run(check())


def test_draft_lines_and_submitted_period_use_acknowledged_revision_once():
    async def check():
        async with screen() as (app, pilot, fake):
            await app.perform(s.VMailCompose('Subject'))
            await settle(pilot, fake)
            draft = app.view.application.draft
            editor = app.query_one(Composer)
            editor.load_text('Body')
            await pilot.press('enter')
            await settle(pilot, fake)
            assert fake.trace[-1].action == s.DraftAppend(draft.id, 0, 'Body')
            editor.load_text('.')
            await settle(pilot, fake)
            assert not any(isinstance(r.action, s.VMailSend) for r in fake.trace)
            await pilot.press('enter')
            await settle(pilot, fake, 1.)
            assert [r.action for r in fake.trace if isinstance(r.action, s.VMailSend)] == [s.VMailSend(draft.id, 1)]
    run(check())


def test_multiline_paste_period_is_literal_and_ctrl_s_uploads_in_order():
    async def check():
        async with screen() as (app, pilot, fake):
            await app.perform(s.VMailCompose('Subject'))
            await settle(pilot, fake)
            editor = app.query_one(Composer)
            editor.post_message(events.Paste('one\n.\ntwo'))
            await pilot.pause()
            assert editor.buffering and len(fake.trace) == 1
            await pilot.press('enter')
            assert editor.text == 'one\n.\ntwo\n'
            assert len(fake.trace) == 1
            await pilot.press('ctrl+s')
            for _ in range(6):
                await settle(pilot, fake)
            actions = [r.action for r in fake.trace]
            appends = [a for a in actions if isinstance(a, s.DraftAppend)]
            assert [a.line for a in appends] == ['one', '.', 'two', '']
            assert [a.expected_revision for a in appends] == [0, 1, 2, 3]
            assert sum(isinstance(a, s.VMailSend) for a in actions) == 1
            assert actions[-1].expected_revision == 4
    run(check())


def test_interrupted_upload_retains_text_and_never_sends_partial_body():
    async def check():
        async with screen() as (app, pilot, fake):
            await app.perform(s.VMailCompose('Subject'))
            await settle(pilot, fake)
            original = fake.execute
            def execute(request):
                if isinstance(request.action, s.DraftAppend) and request.action.expected_revision == 1:
                    return Disposition('refused', reason_code='invalid-body')
                return original(request)
            fake.execute = execute
            editor = app.query_one(Composer)
            editor.post_message(events.Paste('one\ntwo\nthree'))
            await pilot.pause()
            await pilot.press('ctrl+s')
            for _ in range(4):
                await settle(pilot, fake)
            assert editor.text == 'one\ntwo\nthree'
            assert not any(isinstance(r.action, s.VMailSend) for r in fake.trace)
            assert app.controller.projection().application.draft.lines == ('one',)
    run(check())


@pytest.mark.parametrize('size', [(120, 40), (80, 24), (60, 18), (39, 11)])
def test_layout_focus_resize_and_navigation_never_mutate(size):
    async def check():
        async with screen(size=size) as (app, pilot, fake):
            assert app.query_one('#status').region.height > 0
            assert app.query_one('Footer').region.height > 0
            assert app.screen.has_class('compact') == (size[0] < 40 or size[1] < 12)
            if size[0] >= 40:
                editor = app.query_one(Composer)
                editor.load_text('retained')
                await pilot.press('tab', 'shift+tab', 'ctrl+l')
                assert app.focused is editor
                app.query_one('#main-tabs', TabbedContent).active = 'bbs-tab'
                await pilot.resize_terminal(120, 40)
                await pilot.resize_terminal(*size)
                assert editor.text == 'retained'
                app.query_one('#filter', Input).value = 'W1AW'
            await pilot.pause()
            assert fake.trace == []
    run(check())


def test_palette_mail_diagnostics_escape_and_cancel_close_bindings():
    async def check():
        async with screen() as (app, pilot, fake):
            await pilot.press('f2')
            assert isinstance(app.screen, ActionPalette)
            await pilot.press('escape')
            await pilot.press('f4')
            assert isinstance(app.screen, MailScreen)
            await pilot.press('escape', 'f3', 'escape')
            assert fake.trace == []
            await pilot.press('ctrl+x')
            await settle(pilot, fake)
            assert fake.trace[-1].action == s.ChatCancelWait()
            await pilot.press('ctrl+b')
            await settle(pilot, fake, 1.)
            assert sum(isinstance(r.action, s.ChatClose) for r in fake.trace) == 1
    run(check())


def test_styled_bbs_and_plain_text_attack_fixtures_are_literal():
    payload = '\x1b]52;c;secret\x07[bold]<script>\x9b2J\x00\n\tSafe'
    assert plain(payload) == ']52;c;secret[bold]<script>2J\n\tSafe'
    document = StyleParser().feed('\x1b[31;42;1m' + payload + '\x1b[0mnext', final=True)
    spans = tuple(s.StyledSpan(span.text, span.style.foreground, span.style.background, span.style.bold)
                  for span in document.spans)
    page = s.BBSPage('page', 'ansi16', 80, 24, spans)
    rendered = styled(page)
    assert rendered.plain == document.text
    assert 'secret' not in rendered.plain and '[bold]<script>' in rendered.plain
    assert rendered.spans[0].style.color.triplet.hex == '#800000'
    assert rendered.spans[0].style.bgcolor.triplet.hex == '#008000'
    assert rendered.spans[0].style.bold is True
    assert not styled(replace(page, mode='plain')).spans


def test_shared_idle_fixture_renders_status_receive_and_no_actions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('hardware/network factory invoked')
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(
        InputStream=forbidden, OutputStream=forbidden, RawInputStream=forbidden,
        RawOutputStream=forbidden, Stream=forbidden))
    monkeypatch.setitem(sys.modules, 'serial', SimpleNamespace(Serial=forbidden))
    async def check():
        async with screen(scenario='idle-monitoring', connected=False) as (app, pilot, fake):
            await settle(pilot, fake, 100.)
            assert app.view.live.identity.mycall == 'W9SSJ'
            assert app.view.receive.heard
            assert app.view.receive.slots
            assert fake.trace == []
    try:
        loop.run_until_complete(check())
    finally:
        loop.close()


def test_link_change_orphans_local_text_and_recovery_has_no_effects():
    scenario = (ScenarioStep(1., 'poll', None,
                            (('link.generation', 1), ('link.connected', False), ('link.peer', None)),
                            (s.TextRecord('incoming', 'rx', 'retained incoming', 'W1AW', 'chat'),)),)
    async def check():
        async with screen(scenario=scenario, max_records=2) as (app, pilot, fake):
            editor = app.query_one(Composer)
            editor.load_text('unsent')
            await settle(pilot, fake, 1.)
            assert app._orphan
            assert editor.text == 'unsent'
            await pilot.press('enter')
            await settle(pilot, fake)
            assert fake.trace == []
            before = app.display_cursor
            app.recover()
            assert 'incoming' in app.records
            assert app.display_cursor >= before
            assert fake.trace == []
    run(check())


def test_projection_updates_do_not_download_history_or_advance_display_cursor():
    scenario = (ScenarioStep(1., 'poll', None, (
        ('application.files.received_bytes', 72), ('receive.snr', -7.),
        ('application.bbs.remote_view', False)), ()),)
    async def check():
        async with screen(scenario=scenario) as (app, pilot, fake):
            reads = app.history_reads
            await settle(pilot, fake, 1.)
            assert app.view.application.files.received_bytes == 72
            assert app.view.receive.snr == -7.
            assert app.history_reads == reads
            cursor = app.display_cursor
            old = replace(app.view, revision=0, through_seq=0,
                          receive=replace(app.view.receive, snr=99.))
            assert not app.apply_projection(old)
            assert app.view.receive.snr == -7.
            app.apply_projection(app.controller.projection())
            assert app.display_cursor == cursor
            assert fake.trace == []
    run(check())


@pytest.mark.parametrize('phase', [s.Phase.QSY, s.Phase.BBS, s.Phase.EMAIL,
                                   s.Phase.VMAIL_ACK, s.Phase.CONNECTING, s.Phase.CQ,
                                   s.Phase.BEACON, s.Phase.BROADCAST, s.Phase.CLOSING,
                                   s.Phase.SETTLING, s.Phase.BBS_SEND, s.Phase.FILE_DOWNLOAD,
                                   s.Phase.FILE_CAPTURE, s.Phase.OPENING])
def test_quit_during_each_wait_joins_owner_once(phase):
    scenario = (
        ScenarioStep(0., 'poll', None, (('operation.phase', phase),
                                       ('operation.active', True)), ()),
        ScenarioStep(0., 'action', s.ChatSend('waiting'), (),
                     (Disposition('running', stage='waiting'),)),
    )
    async def check():
        async with screen(scenario=scenario) as (app, pilot, fake):
            app.query_one(Composer).load_text('waiting')
            await pilot.press('enter')
            await pilot.pause()
            assert app.controller.outcome(fake.trace[0].id).state.state == 'running'
            await pilot.press('ctrl+q')
            await pilot.pause()
            assert app.controller.join(1)
            assert fake.shutdown_trace
            assert app.controller.outcome(fake.trace[0].id).state.state == 'unknown'
            trace = list(fake.shutdown_trace)
            app.controller.shutdown().result(timeout=1)
            assert fake.shutdown_trace == trace
    run(check())


def test_typed_palette_parses_local_preserving_actions():
    assert parse_action('/follow off') == s.FollowSet(None)
    assert parse_action('/away gone') == s.AwaySet('gone')
    assert parse_action('/bbs term ansi16 80 24') == s.BBSTerm('ansi16', 80, 24)
    assert parse_action('/vmail pending') == s.VMailPending()
    assert parse_action('/beacon hop status') == s.BeaconHopStatus()
    assert parse_action('/email status') == s.EmailStatus()
    assert parse_action('/vmail send two  spaces') == s.VMailCompose('two  spaces')
    assert parse_action('/bye two  spaces') == s.ChatClose('two  spaces')
    assert parse_action('/all two  spaces') == s.BroadcastSend('two  spaces')
    assert parse_action('/all clear') == s.BroadcastClear()
    assert parse_action('/all yes') == s.BroadcastConfirm(True)
    assert parse_action('/all no') == s.BroadcastConfirm(False)
    for command in ('/bad',):
        with pytest.raises(ValueError):
            parse_action(command)


def test_bbs_input_keeps_rendered_pair_while_new_output_arrives():
    page = s.BBSPage('first', 'plain', 80, 24, (s.StyledSpan('1. Read', None, None, False),))
    scenario = (
        ScenarioStep(1., 'poll', None, (), (page,)),
        ScenarioStep(2., 'poll', None, (), (replace(page, page_id='second'),)),
    )
    async def check():
        async with screen(scenario=scenario) as (app, pilot, fake):
            await app.perform(s.BBSEnter())
            await settle(pilot, fake, 1.)
            app.query_one('#main-tabs', TabbedContent).active = 'bbs-tab'
            field = app.query_one('#bbs-input', Input)
            field.focus()
            field.value = '1'
            await pilot.pause()
            old_pair = app._bbs_fence
            await settle(pilot, fake, 1.)
            assert old_pair[1] < app.view.live.bbs_revision
            before = len(fake.trace)
            field.focus()
            await pilot.press('enter')
            await settle(pilot, fake)
            assert len(fake.trace) == before
            assert field.value == '1'
            field.value = ''
            await pilot.pause()
            field.value = '2'
            await pilot.pause()
            field.focus()
            await pilot.press('enter')
            await settle(pilot, fake)
            assert isinstance(fake.trace[-1].action, s.BBSInput)
            assert fake.trace[-1].action.text == '2'
    run(check())


def test_gap_recovers_retained_text_outcome_and_bbs_loss_latch_without_actions():
    pages = tuple(s.BBSPage(f'page-{n}', 'plain', 80, 24,
                           (s.StyledSpan('menu ' + str(n), None, None, False),)) for n in range(4))
    text = s.TextRecord('incoming-retained', 'rx', 'last text', 'W1AW', 'chat')
    scenario = (ScenarioStep(1., 'poll', None, (), pages + (text,)),)
    async def check():
        async with screen(scenario=scenario, max_records=2) as (app, pilot, fake):
            await app.perform(s.BBSEnter())
            await settle(pilot, fake, .25)
            trace = list(fake.trace)
            sub, app._subscription = app._subscription, None
            sub.max_events = 2
            await asyncio.wrap_future(fake.advance(1.))
            app._subscription = sub
            await pilot.pause(.1)
            await settle(pilot, fake)
            assert app.history_reads >= 2
            assert 'incoming-retained' in app.records
            assert app.view.live.bbs_reconnect_required
            assert 'History unavailable' in str(app.query_one('#history-notice').render())
            assert app.query_one('#bbs-input', Input).disabled
            assert fake.trace == trace
    run(check())


def test_two_views_observe_shared_draft_and_autonomous_state_via_projection():
    scenario = (ScenarioStep(1., 'poll', None, (
        ('application.files.received_bytes', 81), ('receive.snr', -11.),
        ('application.bbs.remote_view', False)), ()),)
    async def check():
        async with screen(scenario=scenario) as (first, pilot, fake):
            second = ChatApp(first.controller, actor='observer')
            async with second.run_test(size=(80, 24)) as other:
                history = first.history_reads, second.history_reads
                await first.perform(s.VMailCompose('shared'))
                await settle(pilot, fake)
                await other.pause(.1)
                assert second.view.application.draft.subject == 'shared'
                await settle(pilot, fake, 1.)
                await other.pause(.1)
                assert second.view.application.files.received_bytes == 81
                assert second.view.receive.snr == -11.
                assert (first.history_reads, second.history_reads) == history
                assert len(fake.trace) == 1
    run(check())


def test_projection_coalescing_and_read_throttling_preserve_dirty_sections():
    async def check():
        async with screen() as (app, pilot, fake):
            app._last_projection = fake.clock()
            reads = app.projection_reads
            seq = app.display_cursor + 1
            for offset, section in enumerate(s.ProjectionSection):
                payload = s.StateChanged(app.view.revision + 1, seq + offset, None, (section,))
                event = s.EventEnvelope(seq + offset, app.view.controller_id, '', fake.clock(),
                                        app.view.live.link.generation, None, 'StateChanged', payload)
                app.post_message(ChatApp.Updates((event,)))
            await pilot.pause(.1)
            assert set(app._dirty) == set(s.ProjectionSection)
            assert app.projection_reads == reads
            await settle(pilot, fake)
            assert app.projection_reads == reads + 1
            assert app._dirty
            assert fake.trace == []
    run(check())


def test_palette_discards_check_reply_after_editor_parameters_change():
    from concurrent.futures import Future

    async def check():
        async with screen() as (app, pilot, fake):
            await pilot.press('f2')
            palette = app.screen
            pending = []
            def preview(action):
                future = Future()
                pending.append((action, future))
                return future
            app.controller.check = preview
            palette.query_one(Input).value = '/call W1AW 2'
            await pilot.pause()
            palette.preview()
            await pilot.pause()
            assert len(pending) == 1
            action, future = pending[0]
            version = palette._version
            palette.query_one(Input).value = '/call K9ABC 3'
            await pilot.pause()
            assert palette._version > version
            palette.preview()
            assert len(pending) == 1
            future.set_result(s.CapabilityCheck(app.view.controller_id, app.view.revision,
                app.view.through_seq, action,
                s.Capability(action.kind, s.Effect.X, 'blocked', None, 'old', 'Old peer refusal')))
            await pilot.pause()
            palette.preview()
            assert len(pending) == 1
            fake.clock.advance(.2)
            palette.preview()
            await pilot.pause()
            assert len(pending) == 2
            assert pending[1][0] == s.ChatCall('K9ABC', 3, True)
            assert 'Old peer refusal' not in str(palette.query_one('#preview').render())
            current, future = pending[1]
            future.set_result(s.CapabilityCheck(app.view.controller_id, app.view.revision,
                app.view.through_seq, current,
                s.Capability(current.kind, s.Effect.X, 'blocked', None, 'current', 'Current peer refusal')))
            await pilot.pause()
            assert 'Current peer refusal' in str(palette.query_one('#preview').render())
            assert fake.trace == []
    run(check())


def test_oversized_paste_is_rejected_before_retaining_body():
    async def check():
        async with screen() as (app, pilot, fake):
            editor = app.query_one(Composer)
            editor.load_text('retained')
            editor.post_message(events.Paste('é' * 32769))
            await pilot.pause()
            assert editor.text == 'retained'
            assert fake.trace == []
    run(check())


def test_repeated_exit_keeps_one_uncancelled_shutdown_completion():
    from threading import Event

    async def check():
        async with screen() as (app, pilot, fake):
            release = Event()
            original = fake.quit
            calls = []
            def quit_once(*, unwinding=None):
                calls.append(True)
                assert release.wait(5)
                original()
            fake.quit = quit_once
            app.action_ordered_exit()
            await pilot.pause(.05)
            completion = app.controller.shutdown()
            app.action_ordered_exit()
            assert not completion.cancelled()
            assert not completion.done()
            release.set()
            await asyncio.shield(asyncio.wrap_future(completion))
            await pilot.pause()
            assert len(calls) == 1
    run(check())


@pytest.mark.parametrize('active', [False, True])
def test_blank_operator_line_stops_only_an_active_wait(active):
    scenario = (ScenarioStep(0., 'poll', None, (('operation.active', active),), ()),)
    async def check():
        async with screen(scenario=scenario) as (app, pilot, fake):
            app.query_one(Composer).load_text('  ')
            await pilot.press('enter')
            await settle(pilot, fake)
            assert [r.action for r in fake.trace] == ([s.ChatCancelWait()] if active else [])
    run(check())


def test_projection_fences_reject_either_stale_component_and_merge_dirty_components():
    async def check():
        async with screen() as (app, pilot, fake):
            app._applied = {section: (5, 100) for section in s.ProjectionSection}
            for revision, sequence in ((6, 99), (4, 101)):
                incoming = replace(app.view, revision=revision, through_seq=sequence,
                                   receive=replace(app.view.receive, snr=99.))
                assert not app.apply_projection(incoming)
                assert app.view.receive.snr != 99.
            section = s.ProjectionSection.APPLICATION
            app._dirty[section] = (7, 80)
            event = s.EventEnvelope(101, app.view.controller_id, '', fake.clock(), 0, None,
                'StateChanged', s.StateChanged(6, 101, None, (section,)))
            app.on_chat_app_updates(ChatApp.Updates((event,)))
            assert app._dirty[section] == (7, 101)
            assert fake.trace == []
    run(check())


def test_live_history_counts_escaped_encoded_bytes_and_marks_eviction():
    fake = FakeSession()
    controller = Controller(fake, clock=fake.clock)
    controller.ready.result(timeout=2)
    app = ChatApp(controller)
    try:
        for sequence in range(100):
            payload = s.TextRecord(str(sequence), 'rx', '"' * 12000, 'W1AW', 'chat')
            app._retain(s.DisplayRecord(str(sequence), sequence + 1, sequence + 1,
                                        0, None, None, payload, None))
        assert sum(len(s.encode(record)) for record in app.records.values()) <= 2 * 1024 * 1024
        assert app._record_bytes == sum(len(s.encode(record)) for record in app.records.values())
        assert app.history_evicted_through is not None
        assert len(app.records) < 100
        assert fake.trace == []
    finally:
        controller.shutdown().result(timeout=2)
        assert controller.join(2)


def test_repeated_gaps_coalesce_and_recovery_reads_obey_encoded_byte_rate():
    async def check():
        async with screen() as (app, pilot, fake):
            await settle(pilot, fake, 1.)
            base = app.controller.snapshot()
            records = tuple(s.DisplayRecord(str(n), n + 1, n + 1, 0, None, None,
                s.TextRecord(str(n), 'rx', 'x' * 24000, 'W1AW', 'chat'), None) for n in range(40))
            snapshots = [replace(base, through_seq=100,
                                 presentation=replace(base.presentation, records=records))]
            app.controller.snapshot = lambda: snapshots[0]
            app.recover()
            size = len(s.encode(snapshots[0]))
            next_read = fake.clock() + size / (4 * 1024 * 1024)
            assert app._next_recovery_at == next_read
            reads = app.history_reads
            gaps = tuple(s.EventEnvelope(n, app.view.controller_id, '', fake.clock(), 0, None,
                                          'Gap', s.Gap(n - 1, n, 'slow view')) for n in range(101, 111))
            app.on_chat_app_updates(ChatApp.Updates(gaps))
            assert app.history_reads == reads
            assert app._recovery_pending
            assert app.query_one('#bbs-input', Input).disabled
            editor = app.query_one(Composer)
            editor.load_text('retained during recovery')
            await pilot.press('enter')
            assert fake.trace == [] and editor.text == 'retained during recovery'
            snapshots[0] = replace(snapshots[0], through_seq=500)
            await settle(pilot, fake, (next_read - fake.clock()) / 2)
            assert app.history_reads == reads
            await settle(pilot, fake, next_read - fake.clock() + .001)
            assert app.history_reads == reads + 1
            assert not app._recovery_pending
            assert fake.trace == []
    run(check())


def test_dirty_capabilities_disable_submission_and_keep_editor():
    async def check():
        async with screen() as (app, pilot, fake):
            app._dirty[s.ProjectionSection.CAPABILITIES] = (999, 999)
            editor = app.query_one(Composer)
            editor.load_text('held')
            await pilot.press('enter')
            await pilot.pause()
            assert fake.trace == [] and editor.text == 'held'
            assert await app.perform(s.ChatSend('held')) is None
            assert fake.trace == []
    run(check())


def test_late_state_event_already_covered_by_projection_does_not_disable_send():
    async def check():
        async with screen() as (app, pilot, fake):
            projection = replace(app.view, revision=10, through_seq=100)
            assert app.apply_projection(projection)
            event = s.EventEnvelope(100, app.view.controller_id, '', fake.clock(), 0, None,
                'StateChanged', s.StateChanged(10, 100, None, tuple(s.ProjectionSection)))
            app.on_chat_app_updates(ChatApp.Updates((event,)))
            assert app._dirty == {}
            assert not app._dependent_stale()
            assert app.display_cursor == 100
            assert fake.trace == []
    run(check())


def test_large_projection_and_snapshot_reads_share_one_encoded_byte_budget():
    fake = FakeSession()
    controller = Controller(fake, clock=fake.clock)
    controller.ready.result(timeout=2)
    app = ChatApp(controller)
    try:
        app.recover(initial=True)
        headers = tuple(s.MailHeader(n, str(n), 'in', 'W1AW', 'W9SSJ', 'x' * 2048,
                                     '2026-10-28T00:00:00Z') for n in range(1000))
        original = controller.projection()
        large = replace(original, application=replace(original.application, mailbox=headers))
        size = len(s.encode(large))
        assert 2 * 1024 * 1024 < size < 4 * 1024 * 1024
        reads = []
        def projection():
            reads.append((fake.clock(), size))
            return large
        controller.projection = projection
        fake.clock.advance(1.)
        for _ in range(21):
            app.refresh_projection()
            fake.clock.advance(.1)
        assert 2 <= app.projection_reads <= 4
        for (before, amount), (after, _) in zip(reads, reads[1:]):
            assert after - before >= amount / (4 * 1024 * 1024)
        fake.clock.advance(1.)
        assert app.refresh_projection()
        history = app.history_reads
        app.recover()
        assert app._recovery_pending and app.history_reads == history
        fake.clock.advance(size / (4 * 1024 * 1024) + .001)
        app._maybe_recover()
        assert not app._recovery_pending and app.history_reads == history + 1
        assert fake.trace == []
    finally:
        controller.shutdown().result(timeout=2)
        assert controller.join(2)


def test_terminal_already_in_applied_projection_does_not_redirty_inputs():
    async def check():
        async with screen() as (app, pilot, fake):
            original = app.controller.outcome
            def outcome(request_id):
                result = original(request_id)
                if isinstance(result, s.RequestResult) and result.state.state == 'done':
                    app.controller.check(s.ChatCancelWait()).result(timeout=2)
                    app.apply_projection(app.controller.projection())
                    app._last_projection = fake.clock()
                return result
            app.controller.outcome = outcome
            result = await app.perform(s.VMailCompose('covered'))
            assert result is not None
            assert app._request_covered(result.state)
            assert not app._dependent_stale()
            assert app._dirty == {}
    run(check())


@pytest.mark.parametrize('scenario', tuple(SCENARIOS))
def test_every_scenario_mounts_at_80_columns_with_visible_composer_and_status(scenario):
    async def check():
        async with screen(size=(80, 24), scenario=scenario, connected=False) as (app, pilot, fake):
            editor = app.query_one(Composer)
            status = app.query_one('#status')
            assert app.focused is editor
            assert app.screen.region.contains_region(editor.region)
            assert app.screen.region.contains_region(status.region)
            assert status.region.height >= 4
            assert app.view.live.identity.mycall in str(status.content)
            assert app.screen.has_class('narrow')
            assert not app.query_one('#side').display
            tabs = app.query_one('#main-tabs', TabbedContent)
            for name in ('bbs-tab', 'receive-tab', 'contact-tab'):
                tabs.active = name
                await pilot.pause()
                assert app.screen.region.contains_region(editor.region)
            assert fake.trace == []
    run(check())


def test_footer_shows_every_binding_without_horizontal_clipping_at_80_columns():
    async def check():
        async with screen(size=(80, 24)) as (app, pilot, fake):
            footer = app.query_one('Footer')
            keys = list(footer.query('FooterKey'))
            assert {key.action.removeprefix('app.') for key in keys} >= {
                'focus_next', 'focus_previous', 'commands', 'diagnostics', 'mail',
                'focus_composer', 'submit', 'cancel_wait', 'close_contact', 'ordered_exit', 'interrupt'}
            for key in keys:
                assert footer.region.contains_region(key.region)
                assert len(key.render().plain) <= key.region.width
            assert any('^q' in key.render().plain.lower() for key in keys)
            assert any('^c' in key.render().plain.lower() for key in keys)
            assert fake.trace == []
    run(check())


@pytest.mark.parametrize('binding', ('f2', 'f3', 'f4'))
def test_modal_controls_fit_24_rows_and_tab_reaches_every_control(binding):
    async def check():
        async with screen(size=(80, 24)) as (app, pilot, fake):
            editor = app.query_one(Composer)
            await pilot.press(binding)
            modal = app.screen
            expected = set(modal.focus_chain)
            visited = {modal.focused}
            for _ in range(len(expected) + 1):
                await pilot.press('tab')
                visited.add(modal.focused)
            assert expected <= visited
            for control in modal.query('Button, Input'):
                assert modal.region.contains_region(control.region)
                assert modal.query_one('.dialog').content_region.contains_region(control.region)
            await pilot.press('escape')
            assert app.screen is app.screen_stack[0]
            assert app.focused is editor
            assert fake.trace == []
    run(check())


def test_main_tab_order_reaches_bbs_input_and_wide_filter_without_actions():
    async def check():
        from hfmodem.contact.chat_fake import BBS_ATTACK
        scenario = (ScenarioStep(.5, 'poll', None, (), (BBS_ATTACK,)),)
        async with screen(size=(80, 24), scenario=scenario) as (app, pilot, fake):
            await app.perform(s.BBSEnter())
            await settle(pilot, fake, .75)
            await settle(pilot, fake)
            before = list(fake.trace)
            app.query_one('#main-tabs', TabbedContent).active = 'bbs-tab'
            await pilot.pause()
            bbs_input = app.query_one('#bbs-input', Input)
            assert not bbs_input.disabled, (app._dirty, app.view.live, app.view.application.bbs)
            expected = set(app.screen.focus_chain)
            visited = set()
            for _ in range(len(expected) + 1):
                await pilot.press('tab')
                visited.add(app.focused)
            assert bbs_input in expected
            assert expected <= visited
            await pilot.resize_terminal(100, 24)
            await pilot.pause()
            assert not app.screen.has_class('narrow')
            expected = set(app.screen.focus_chain)
            visited = set()
            for _ in range(len(expected) + 1):
                await pilot.press('tab')
                visited.add(app.focused)
            assert app.query_one('#filter', Input) in visited
            assert expected <= visited
            assert fake.trace == before
    run(check())


def test_textual_terminal_mode_delivers_ctrl_s_and_ctrl_q():
    import termios
    from textual.drivers.linux_driver import LinuxDriver
    flags = termios.IXON | termios.IXOFF | termios.ICRNL
    assert LinuxDriver._patch_iflag(flags) & (termios.IXON | termios.IXOFF) == 0


@pytest.mark.parametrize('binding', ('ctrl+q', 'ctrl+c'))
def test_each_exit_binding_closes_the_owner_once_at_80_columns(binding):
    async def check():
        async with screen(size=(80, 24)) as (app, pilot, fake):
            await pilot.press(binding)
            await pilot.pause()
            assert app.controller.join(1)
            assert fake.shutdown_trace
            before = list(fake.shutdown_trace)
            app.controller.shutdown().result(1)
            assert fake.shutdown_trace == before
    run(check())


def test_prototype_cli_lists_scenarios_without_starting_controller(monkeypatch, capsys):
    from hfmodem.contact.chat_fake import SCENARIOS
    from hfmodem.ui import textual_chat
    def forbidden(*args, **kwargs):
        raise AssertionError('controller constructed for --list')
    monkeypatch.setattr(textual_chat, 'Controller', forbidden)
    textual_chat.main(['--list'])
    assert capsys.readouterr().out.splitlines() == list(SCENARIOS)


@pytest.mark.parametrize('argv,scenario', [([], 'idle-monitoring'),
                                         (['--scenario', 'bbs'], 'bbs')])
def test_prototype_cli_constructs_only_the_selected_fake(monkeypatch, argv, scenario):
    from hfmodem.ui import textual_chat
    seen = []
    def controller(fake, **kwargs):
        assert isinstance(fake, FakeSession)
        seen.append(fake.scenario_name)
        return SimpleNamespace(clock=kwargs['clock'])
    monkeypatch.setattr(textual_chat, 'Controller', controller)
    monkeypatch.setattr(textual_chat, 'ChatApp', lambda owner: SimpleNamespace(run=lambda: None))
    textual_chat.main(argv)
    assert seen == [scenario]


def test_real_app_title_override_keeps_prototype_title_default():
    fake = FakeSession()
    controller = SimpleNamespace(clock=fake.clock, controller_id='title-test')
    assert 'offline prototype' in ChatApp(controller).title
    assert ChatApp(controller, title='Kestrel chat').title == 'Kestrel chat'


def test_broadcast_status_displays_outstanding_frames_without_delivery_claim():
    scenario = (ScenarioStep(0., 'poll', None,
                            (('broadcast.possibly_outstanding', 2), ('broadcast.result', 'written')), ()),)
    async def check():
        async with screen(scenario=scenario, connected=False) as (app, pilot, fake):
            await settle(pilot, fake)
            text = str(app.query_one('#status').render())
            assert '2 possibly outstanding; reception unconfirmed' in text
            assert fake.trace == []
    run(check())
