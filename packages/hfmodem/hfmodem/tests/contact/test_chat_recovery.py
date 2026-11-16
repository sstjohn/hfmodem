# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from contextlib import contextmanager
from dataclasses import replace

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_controller import Controller, Disposition
from hfmodem.tests.contact.test_chat_controller import Port, completed, request, sync


@contextmanager
def owner(port=None, **options):
    port = port or Port()
    controller = Controller(port, controller_id='test', **options)
    controller.ready.result(5)
    try:
        yield controller, port
    finally:
        port.release.set()
        completion = controller.shutdown()
        try:
            completion.result(10)
        finally:
            assert controller.join(10)


def page(ident, text='Choose 1'):
    return s.BBSPage(ident, 'plain', 80, 24, (s.StyledSpan(text, None, None, False),))


def remote(port, enabled=True):
    port.state = replace(port.state, link=replace(port.state.link, connected=True),
                         application=replace(port.state.application,
                             bbs=replace(port.state.application.bbs, remote_view=enabled)))


def emit(port, *payloads):
    port.events.extend((payload, None, port.state.link.generation) for payload in payloads)


def test_shutdown_freezes_queued_mutations_after_a_blocking_action():
    with owner() as (controller, port):
        port.block = True
        first = request(s.ChatSend('already executing'))
        queued = request(s.ChatSend('must not execute'))
        assert controller.submit(first).queued
        assert port.entered.wait(2)
        assert controller.submit(queued).queued
        completion = controller.shutdown()
        port.block = False
        port.release.set()
        completion.result(5)
        assert controller.join(5)
        assert port.calls.count('chat.send') == 1
        assert controller.outcome(queued.id).state.state == 'cancelled'
        assert not controller.submit(request(s.ChatSend('too late'))).queued


def test_cancel_cannot_consume_the_reserved_close_and_exit_slots():
    with owner() as (controller, port):
        port.block = True
        assert controller.submit(request(s.ChatSend('hold owner'))).queued
        assert port.entered.wait(2)
        cancel = request(s.ChatCancelWait())
        close = request(s.ChatClose())
        exit_request = request(s.ChatExit())
        assert controller.submit(cancel).queued
        assert not controller.submit(request(s.ChatCancelWait())).queued
        assert controller.submit(close).queued
        assert controller.submit(exit_request).queued
        assert not controller.submit(request(s.ChatClose())).queued
        assert controller.submit(request(s.ChatSend('queued after exit'))).queued
        port.block = False
        port.release.set()
        assert controller.join(5)
        actions = [call for call in port.calls if call.startswith('chat.')]
        assert actions == ['chat.send', 'chat.cancel_wait', 'chat.close', 'chat.exit']
        assert port.quit_count == 1


def test_check_selected_before_a_failing_poll_gets_a_terminal_exception():
    port = Port()
    controller = Controller(port, controller_id='test')
    try:
        controller.ready.result(2)
        port.block = True
        assert controller.submit(request(s.ChatSend('hold owner'))).queued
        assert port.entered.wait(2)
        check = controller.check(s.VerboseSet(True))
        port.poll_error = RuntimeError('owner poll failed')
        port.block = False
        port.release.set()
        with pytest.raises(RuntimeError):
            check.result(2)
        with pytest.raises(RuntimeError, match='owner poll failed'):
            controller.shutdown().result(2)
        assert controller.join(2)
    finally:
        port.release.set()
        controller.shutdown()
        assert controller.join(5)


def test_large_read_publishes_a_reference_and_keeps_the_full_journal_result():
    with owner() as (controller, port):
        emit(port, *(s.TextRecord(f'text-{n}', 'rx', 'x' * 30000, 'W1AW', 'contact') for n in range(5)))
        sync(controller)
        subscriber = controller.subscribe(controller.snapshot().through_seq)
        req = request(s.ChatSnapshot())
        result = completed(controller, req)
        events = subscriber.drain()
        assert isinstance(result.result, s.Snapshot)
        assert len(result.result.presentation.records) == 5
        assert any(isinstance(event.payload, s.ResultAvailable) and event.payload.request_id == req.id
                   for event in events)
        assert not any(isinstance(event.payload, (s.ReadResult, s.Gap)) for event in events)
        assert max(len(s.encode(event)) for event in events) <= 64 * 1024
        assert controller.outcome(req.id) is result


class CollectionPort(Port):
    def execute(self, req):
        self.note(req.action.kind)
        if isinstance(req.action, s.HeardList):
            return Disposition('done', s.HeardRows(self.state.receive.heard))
        return super().execute(req)


def test_omitted_collection_is_available_in_explicit_read_without_revision_churn():
    port = CollectionPort()
    row = s.HeardRow(1, 'W1AW', None, 500, -10., 0., False, '', '',
                     '\\"' * (1100 * 1024), 20, 14105000)
    port.state = replace(port.state, receive=replace(port.state.receive, heard=(row,)))
    with owner(port) as (controller, _):
        snapshot = controller.snapshot()
        assert snapshot.receive.heard == ()
        assert 'receive.heard' in snapshot.omitted_sections
        assert len(s.encode(snapshot)) <= 4 * 1024 * 1024
        revision = snapshot.revision
        sync(controller)
        assert controller.snapshot().revision == revision
        result = completed(controller, request(s.HeardList()))
        assert isinstance(result.result, s.HeardRows)
        assert result.result.rows == (row,)
        assert controller.snapshot().revision == revision


class ImmediatePagePort(Port):
    def __init__(self):
        super().__init__()
        self.entries = 0
        remote(self)

    def execute(self, req):
        self.note(req.action.kind)
        if isinstance(req.action, s.BBSEnter):
            self.entries += 1
            remote(self)
            self.on_event(page(f'entry-{self.entries}', f'Menu {self.entries}'), req.id,
                          self.state.link.generation)
            return Disposition('done', s.WriteAccepted(None, f'write-{self.entries}'))
        return Disposition('done', s.Applied(1))


def test_same_link_reentry_changes_context_and_fences_input_before_synchronous_page():
    with owner(ImmediatePagePort()) as (controller, port):
        emit(port, page('original'))
        sync(controller)
        old = controller.snapshot().presentation
        stale = request(s.BBSInput(old.bbs_context, old.bbs_revision, '1'), actor='second-view')
        assert completed(controller, request(s.BBSEnter())).state.state == 'done'
        current = controller.snapshot().presentation
        assert current.bbs_context != old.bbs_context
        assert current.records[-1].bbs_context == current.bbs_context
        assert current.records[-1].bbs_revision == current.bbs_revision
        assert completed(controller, stale).state.reason_code == 'bbs-stale-view'
        assert 'bbs.input' not in port.calls
        fresh = request(s.BBSInput(current.bbs_context, current.bbs_revision, '1'), actor='second-view')
        assert completed(controller, fresh).state.state == 'done'
        assert port.calls.count('bbs.input') == 1
        assert completed(controller, replace(fresh, id='third-view', actor='third-view')).state.reason_code == 'bbs-stale-view'


@pytest.mark.parametrize('context', ['paging', 'remote composition', 'game'])
def test_bbs_loss_latch_survives_delayed_output_same_link_reentry_and_gap(context):
    with owner(max_records=2) as (controller, port):
        remote(port)
        sync(controller)
        emit(port, page('first', context), page('second'), page('third'))
        sync(controller)
        lost = controller.snapshot().presentation
        assert not lost.bbs_complete and lost.bbs_reconnect_required
        assert lost.evicted_through is not None
        subscriber = controller.subscribe(0)
        assert completed(controller, request(s.BBSEnter())).state.reason_code == 'bbs-reconnect-required'
        emit(port, page('delayed', 'Unsolicited old output'))
        sync(controller)
        assert controller.snapshot().presentation.bbs_reconnect_required
        emit(port, *(page(f'delayed-{n}', context) for n in range(140)))
        sync(controller)
        assert any(isinstance(event.payload, s.Gap) for event in subscriber.drain())
        assert controller.snapshot().presentation.bbs_reconnect_required
        remote(port, False)
        sync(controller)
        remote(port, True)
        sync(controller)
        assert controller.snapshot().presentation.bbs_reconnect_required
        pair = controller.snapshot().presentation
        blocked = request(s.BBSInput(pair.bbs_context, pair.bbs_revision, '1'))
        assert completed(controller, blocked).state.reason_code == 'bbs-reconnect-required'
        subscriber.drain()
        assert 'bbs.enter' not in port.calls and 'bbs.input' not in port.calls
        assert completed(controller, request(s.ChatClose())).state.state == 'done'
        port.state = replace(port.state, link=replace(port.state.link, connected=False, generation=1),
            application=replace(port.state.application,
                                bbs=replace(port.state.application.bbs, remote_view=False)))
        sync(controller)
        port.state = replace(port.state, link=replace(port.state.link, connected=True, generation=2))
        remote(port)
        sync(controller)
        emit(port, page('fresh', 'Fresh link menu'))
        sync(controller)
        fresh = controller.snapshot().presentation
        assert fresh.bbs_complete and not fresh.bbs_reconnect_required
        assert completed(controller, request(s.BBSInput(fresh.bbs_context, fresh.bbs_revision, '1'),
                                             generation=2)).state.state == 'done'


def test_subscriber_gap_recovers_complete_bbs_context_without_latching_loss():
    with owner() as (controller, port):
        remote(port)
        sync(controller)
        subscriber = controller.subscribe(controller.snapshot().through_seq)
        emit(port, *(page(f'page-{n}', f'Page {n}') for n in range(150)))
        sync(controller)
        assert any(isinstance(event.payload, s.Gap) for event in subscriber.drain())
        recovered = controller.snapshot()
        assert len(recovered.presentation.records) == 150
        assert recovered.presentation.bbs_complete
        assert not recovered.presentation.bbs_reconnect_required
        assert all(record.updated_seq <= recovered.through_seq for record in recovered.presentation.records)
        assert 'bbs.input' not in port.calls and 'bbs.enter' not in port.calls


def test_oversized_bbs_output_has_an_explicit_marker_and_disables_input():
    with owner() as (controller, port):
        remote(port)
        sync(controller)
        emit(port, page('too-large', '\\"\n' * 20000))
        sync(controller)
        presentation = controller.snapshot().presentation
        record = presentation.records[-1]
        assert len(s.encode(record)) <= 48 * 1024
        assert record.payload.spans[-1].text.endswith('[content omitted]')
        assert presentation.bbs_reconnect_required and not presentation.bbs_complete
        assert completed(controller, request(s.BBSInput(presentation.bbs_context,
                          presentation.bbs_revision, '1'))).state.reason_code == 'bbs-reconnect-required'
        assert 'bbs.input' not in port.calls


def test_maximum_history_slow_subscriber_and_repeated_atomic_recovery_are_bounded():
    with owner() as (controller, port):
        subscriber = controller.subscribe(controller.snapshot().through_seq)
        for batch in range(3):
            emit(port, *(s.TextRecord(f'{batch}-{n}', 'rx', '\\"é\n' * 5000, None, 'contact')
                         for n in range(140)))
            sync(controller)
            assert subscriber._bytes <= 256 * 1024 and len(subscriber._items) <= 128
            assert any(isinstance(event.payload, s.Gap) for event in subscriber.drain())
            snapshot = controller.snapshot()
            records = snapshot.presentation.records
            assert len(records) <= 2000
            assert sum(len(s.encode(record)) for record in records) <= 2 * 1024 * 1024
            assert max(len(s.encode(record)) for record in records) <= 48 * 1024
            assert len(s.encode(snapshot)) <= 4 * 1024 * 1024
            assert snapshot.presentation.evicted_through is not None
            assert all(record.updated_seq <= snapshot.through_seq for record in records)
            subscriber.close()
            subscriber = controller.subscribe(snapshot.through_seq)
            last = records[-1]
            emit(port, s.TxTextOutcome(last.id, 7, 'partial', 100, 40, 0, 60, 'verified'),
                       s.TxTextOutcome('evicted-text', 8, 'unknown', 100, 0, 0, 100, 'gap'))
            sync(controller)
            recovered = controller.snapshot()
            assert recovered.presentation.records[-1].tx.acknowledged_bytes == 40
            assert snapshot.presentation.records[-1].tx is None
            assert not any(record.id == 'evicted-text' for record in recovered.presentation.records)
            assert all(event.seq > snapshot.through_seq for event in subscriber.drain())
        assert not any(call in s.ACTIONS for call in port.calls)


def test_tx_settlement_preserves_the_retained_history_byte_cap():
    with owner(history_bytes=1000) as (controller, port):
        emit(port, *(s.TextRecord(str(n), 'tx', 'x' * 100, None, 'contact') for n in range(2)))
        sync(controller)
        assert len(controller.snapshot().presentation.records) == 2
        emit(port, s.TxTextOutcome('0', 1, 'partial', 100, 40, 0, 60, 'verified'),
                   s.TxTextOutcome('1', 2, 'partial', 100, 40, 0, 60, 'verified'))
        sync(controller)
        presentation = controller.snapshot().presentation
        assert sum(len(s.encode(record)) for record in presentation.records) <= 1000
        assert len(presentation.records) == 1
        assert presentation.evicted_through is not None
