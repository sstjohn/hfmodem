# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact.data_writer import Completion
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.vara.ledger import PayloadLedger
from hfmodem.tests.contact.test_chat_controller import adapter, execute


class Writer:
    def __init__(self):
        self.writes = []
        self.done = []
        self.pending = False

    def submit(self, payload, generation):
        self.writes.append((bytes(payload), generation))
        self.pending = True
        return len(self.writes)

    def poll(self):
        done, self.done = self.done, []
        return done

    def finish(self, disposition='complete'):
        self.done.append(Completion(len(self.writes), self.writes[-1][1], disposition))
        self.pending = False

    def close(self, **kwargs):
        self.pending = False


def bound(*, writer=False):
    desk, port, events = adapter(connected=True)
    ledger = PayloadLedger()
    port.bind_delivery(ledger.epoch, desk.host.client.link_generation)
    port._delivery_evidence = lambda: (ledger.evidence(),)
    if writer:
        desk.host.client.data_writer = Writer()
    return desk, port, events, ledger


def latest(events):
    return {p.text_id: p for p, _, _ in events if isinstance(p, s.TxTextOutcome)}


def admit(desk, ledger, *, compressed=False):
    writer = desk.host.client.data_writer
    payload = writer.writes[-1][0] if writer else desk.host.data.sent[-1]
    ledger.queue(b'zip' if compressed else payload, payload, atomic=compressed)
    return payload


def test_tags_between_messages_partial_ack_and_nonprefix_discard():
    desk, port, events, ledger = bound()
    ids = []
    for action in (s.ChatSend('café'), s.ChatTag('NAME', 'Saul'), s.ChatSend('last')):
        _, result = execute(port, action)
        ids.append(result.result.text_id)
        admit(desk, ledger)
    sizes = [len(w['payload']) for w in port._delivery_writes]
    ledger.settle(sizes[0] + sizes[1] + 2)
    port.poll(0)
    outcomes = latest(events)
    assert outcomes[ids[0]].status == outcomes[ids[1]].status == 'acknowledged'
    assert outcomes[ids[2]].acknowledged_bytes == 2
    ledger.clear()
    port.poll(0)
    assert latest(events)[ids[2]].discarded_bytes == sizes[2] - 2
    _, result = execute(port, s.ChatSend('after clear'))
    admit(desk, ledger)
    ledger.settle(ledger.wire_pending)
    port.poll(0)
    assert latest(events)[result.result.text_id].status == 'acknowledged'
    assert latest(events)[ids[2]].status == 'discarded'
    assert sum(sizes) == sum(len(p) for p in desk.host.data.sent[:3])


def test_compressed_originals_require_whole_container_and_socket_completion():
    desk, port, events, ledger = bound(writer=True)
    _, result = execute(port, s.ChatSend('compress me ' * 30))
    payload = admit(desk, ledger, compressed=True)
    ledger.settle(2)
    port.poll(0)
    assert latest(events)[result.result.text_id].acknowledged_bytes == 0
    ledger.settle(1)
    port.poll(0)
    assert latest(events)[result.result.text_id].status == 'queued'
    desk.host.client.data_writer.finish()
    port.poll(0)
    outcome = latest(events)[result.result.text_id]
    assert outcome.status == 'acknowledged' and outcome.acknowledged_bytes == len(payload)


@pytest.mark.parametrize('disposition', ('unknown', 'cancelled'))
def test_writer_failure_or_cancel_cannot_claim_modem_delivery(disposition):
    desk, port, events, ledger = bound(writer=True)
    _, result = execute(port, s.ChatSend('failed writer'))
    admit(desk, ledger)
    ledger.settle(ledger.wire_pending)
    desk.host.client.data_writer.finish(disposition)
    port.poll(0)
    outcome = latest(events)[result.result.text_id]
    assert outcome.acknowledged_bytes == 0
    assert outcome.status == ('unknown' if disposition == 'unknown' else 'discarded')
    assert desk.host.client._stream_poisoned == (disposition == 'unknown')


def test_abort_retirement_and_delayed_old_epoch_do_not_credit_new_link():
    desk, port, events, ledger = bound()
    _, first = execute(port, s.ChatSend('old'))
    admit(desk, ledger)
    old_evidence = ledger.evidence()
    ledger.retire()
    port.poll(0)
    assert latest(events)[first.result.text_id].status == 'discarded'
    desk.hear('DISCONNECTED')
    desk.flow.dial = 14106500
    desk.hear('CONNECTED W9SSJ N0ABC 500')
    fresh = PayloadLedger()
    port.bind_delivery(fresh.epoch, desk.host.client.link_generation)
    _, second = execute(port, s.ChatSend('new'))
    admit(desk, fresh)
    port.observe_delivery(replace(old_evidence, sequence=999, acknowledged=((0, 10000),)))
    port._flush()
    assert latest(events)[second.result.text_id].status == 'queued'
    fresh.settle(fresh.wire_pending)
    port.observe_delivery(fresh.evidence())
    port._flush()
    assert latest(events)[second.result.text_id].status == 'acknowledged'


def test_missing_binding_and_evidence_gap_stay_unknown():
    desk, port, events = adapter(connected=True)
    _, result = execute(port, s.ChatSend('unmapped'))
    ledger = PayloadLedger()
    admit(desk, ledger)
    ledger.settle(ledger.wire_pending)
    port.observe_delivery(ledger.evidence())
    desk.session.diagnostic('acknowledged', bytes=1000)
    port._flush()
    assert latest(events)[result.result.text_id].attribution == 'unattributed'
    with pytest.raises(ValueError, match='fresh'):
        port.bind_delivery(ledger.epoch, desk.host.client.link_generation)
    desk, port, events, ledger = bound()
    _, result = execute(port, s.ChatSend('gap'))
    admit(desk, ledger)
    port.observe_delivery(replace(ledger.evidence(), complete=False))
    port._flush()
    assert latest(events)[result.result.text_id].status == 'unknown'


def test_runtime_evidence_survives_dropped_diagnostics_and_repeated_reads():
    ledger = PayloadLedger()
    ledger.queue(b'abcdef', b'abcdef')
    ledger.settle(2)
    ledger.clear()
    def drop(*args, **kwargs):
        raise RuntimeError('diagnostic queue lost')
    runtime = AudioHostRuntime(SimpleNamespace(diagnostic=drop))
    runtime.modem = SimpleNamespace(confirmed_total=2, discarded_total=4,
                                   payload_evidence_sequence=(ledger.epoch, ledger.sequence),
                                   payload_evidence=lambda: (ledger.evidence(),))
    with runtime.subscribe_payload_evidence() as evidence:
        runtime._report_payload_credit()
        captured = evidence()
        assert captured == evidence() == (ledger.evidence(),)
    assert captured[0].acknowledged == ((0, 2),) and captured[0].discarded == ((2, 6),)
    ledger.queue(b'new', b'new')
    assert captured[0].accepted == 6


@pytest.mark.parametrize('case', ('stale-zero', 'pending', 'failed', 'discarded', 'partial', 'full'))
def test_email_modem_ack_requires_full_framed_verified_coverage(case):
    desk, port, events, ledger = bound(writer=True)
    desk.receive('<S:E>')
    _, result = execute(port, s.EmailCompose('a@example.org', 'b@example.org', 'subject'))
    draft = result.result.draft
    desk.host.client.buffer = 0
    req, _ = execute(port, s.EmailSend(draft.id, draft.revision))
    payload = admit(desk, ledger)
    assert payload.endswith(b'>') and len(payload) > len('subject')
    if case in ('pending', 'failed', 'full'):
        ledger.settle(ledger.wire_pending)
    elif case == 'partial':
        ledger.settle(ledger.wire_pending - 1)
    elif case == 'discarded':
        ledger.clear()
    if case != 'pending':
        desk.host.client.data_writer.finish('unknown' if case == 'failed' else 'complete')
    desk.hear('BUFFER 0')
    desk.session.emit('email', event='modem_acknowledged', peer=desk.session.peer)
    port.poll(0)
    port.poll(0)
    acknowledgements = [rid for p, rid, _ in events if isinstance(p, s.ApplicationOutcome)
                        and p.result == s.MailResult('email', 'modem-ack')]
    assert acknowledgements == ([req.id] if case == 'full' else [])
    if case == 'failed':
        assert port._results[req.id].state == 'unknown'
    else:
        assert req.id not in port._results
    if desk.session.steps is not None:
        desk.session.steps.close()


def test_synchronous_partial_write_failure_is_unknown_without_legacy_sent_event():
    desk, port, events, ledger = bound()
    def fail(payload):
        raise OSError('partial sendall')
    desk.host.data.sendall = fail
    _, result = execute(port, s.ChatSend('possibly sent'))
    assert result.state == 'unknown'
    outcome, = latest(events).values()
    assert outcome.status == 'unknown' and outcome.unknown_bytes == outcome.framed_bytes
    assert not [e for e in desk.host.events if e.kind == 'sent']
    assert port._delivery_binding is None


def test_pending_writer_completion_after_link_retirement_never_settles_new_text():
    desk, port, events, ledger = bound(writer=True)
    _, result = execute(port, s.ChatSend('old pending write'))
    admit(desk, ledger)
    old_generation = desk.host.client.link_generation
    desk.host.client._retire_link_messages()
    port.project()
    writer = desk.host.client.data_writer
    writer.done.append(Completion(1, old_generation, 'complete'))
    ledger.settle(ledger.wire_pending)
    port.poll(0)
    outcome = latest(events)[result.result.text_id]
    assert outcome.status == 'unknown' and outcome.acknowledged_bytes == 0


def test_evicted_epoch_is_an_explicit_unknown_and_cannot_be_recovered_as_ack():
    desk, port, events, ledger = bound()
    _, result = execute(port, s.ChatSend('old evidence'))
    admit(desk, ledger)
    port._delivery_evidence = lambda: (PayloadLedger().evidence(),)
    port.poll(0)
    assert latest(events)[result.result.text_id].status == 'unknown'
    ledger.settle(ledger.wire_pending)
    port.observe_delivery(ledger.evidence())
    port._flush()
    assert latest(events)[result.result.text_id].acknowledged_bytes == 0


def test_verified_settlement_updates_controller_history_and_terminal_journal(tmp_path):
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.contact.chat_session import SessionAdapter
    from hfmodem.contact.varac import frame_message
    from hfmodem.tests.contact.test_chat_controller import completed, request, sync
    from hfmodem.tests.contact.test_varac import Desk
    ledger = PayloadLedger()
    captured = {'value': (ledger.evidence(),)}
    def factory():
        desk = Desk(mailbox=tmp_path)
        desk.flow.dial = 14106500
        desk.hear('CONNECTED W9SSJ N0XYZ 500')
        port = SessionAdapter(desk.session, delivery_evidence=lambda: captured['value'])
        port.bind_delivery(ledger.epoch, desk.host.client.link_generation)
        return port
    controller = Controller(factory, controller_id='test')
    controller.ready.result(2)
    try:
        req = request(s.ChatSend('café'), generation=controller.snapshot().link.generation)
        original = completed(controller, req)
        payload = frame_message('café\n')
        ledger.queue(payload, payload)
        ledger.settle(len(payload))
        captured['value'] = (ledger.evidence(),)
        sync(controller)
        result = controller.outcome(req.id)
        assert result.state == original.state
        outcome, = result.text_outcomes
        assert outcome.status == 'acknowledged' and outcome.framed_bytes == len(payload)
        record, = (r for r in controller.snapshot().presentation.records if r.id == outcome.text_id)
        assert record.tx == outcome
    finally:
        controller.shutdown().result(2)
        assert controller.join(2)


def test_clear_command_response_alone_cannot_settle_framed_bytes():
    desk, port, events, ledger = bound()
    _, sent = execute(port, s.ChatSend('clear this'))
    admit(desk, ledger)
    _, cleared = execute(port, s.BufferClear())
    assert cleared.state == 'done' and desk.sent[-1] == 'CLEANTXBUFFER'
    port.poll(0)
    assert latest(events)[sent.result.text_id].status == 'queued'
    ledger.clear()
    port.poll(0)
    assert latest(events)[sent.result.text_id].status == 'discarded'


def test_retained_writer_bytes_are_bounded_and_eviction_preserves_stream_offsets():
    desk, port, events, ledger = bound()
    first_id = None
    for n in range(2001):
        _, result = execute(port, s.ChatSend(str(n)))
        first_id = first_id or result.result.text_id
        admit(desk, ledger)
    assert len(port._delivery_writes) == 2000
    assert port._delivery_bytes == sum(len(w['payload']) for w in port._delivery_writes)
    assert latest(events)[first_id].status == 'unknown'
    ledger.settle(ledger.wire_pending)
    port.poll(0)
    assert latest(events)[first_id].acknowledged_bytes == 0
    assert latest(events)[result.result.text_id].status == 'acknowledged'


def test_retired_epoch_without_original_acceptance_keeps_write_unknown():
    desk, port, events, ledger = bound()
    _, result = execute(port, s.ChatSend('socket complete but host admission absent'))
    ledger.retire()
    port.poll(0)
    outcome = latest(events)[result.result.text_id]
    assert outcome.status == 'unknown' and outcome.unknown_bytes == outcome.framed_bytes
