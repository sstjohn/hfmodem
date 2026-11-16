# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real host grammar and controller adapter, entirely in-memory streams."""
import pytest
from threading import Event, Thread, get_ident

from hfmodem.contact import chat_state as s
from hfmodem.contact import varac as v
from hfmodem.contact.chat_controller import Controller
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.contact.data_writer import DataWriter
from hfmodem.kestrel.host.vara_modem import PayloadBoundary
from hfmodem.kestrel.vara.ledger import PayloadLedger
from hfmodem.station.chat import PrivateSession, ReceiveSink
from hfmodem.tests.contact.test_varac import Desk, Socket


class Fence:
    live = True
    def __call__(self): return self.live
    def check(self):
        if not self.live: raise v.HostError('authority revoked')


class Modem:
    supports_frame_force = True
    default_compression = 'TEXT'
    def __init__(self): self.frames = []
    def start(self, observer): self.observer = observer
    def set_listen(self, on): self.listening = on
    def abort(self): pass
    def stop(self): pass
    def disconnect(self): self.observer.modem_disconnected()
    def send_beacon(self, text, *, force=False):
        self.frames.append((text, force))
        self.observer.modem_pending()
        self.observer.modem_ptt(True)
        self.observer.modem_bitrate(2, 41, True)
        self.observer.modem_ptt(False)
        self.observer.modem_pending(cancel=True)


def bridge():
    desk = Desk(loc='EN63BC')
    modem, fence = Modem(), Fence()
    command = Socket(lambda payload: desk.host.command.incoming.append(payload))
    data = Socket(lambda payload: desk.host.data.incoming.append(payload))
    host = PrivateSession(command, data, modem, fence=fence)
    modem.start(host)
    desk.host.command.respond = lambda payload: host._dispatch(payload.decode().rstrip('\r'))
    desk.host.client.beacon_submit = host.beacon_submit
    return desk, host, modem, fence


def test_listen_off_keeps_cq_sn_bitrate_visible_through_controller():
    desk, host, modem, fence = bridge()
    desk.host.client.request('LISTEN OFF')
    assert not host.listen_cq and not host.listening
    host.modem_cqframe('K7ABC', '500', 6.2)
    port = SessionAdapter(desk.session)
    controller = Controller(lambda: port)
    try:
        controller.ready.result(timeout=2)
        for _ in range(20):
            controller.check(s.VerboseSet(False)).result(timeout=2)
            snapshot = controller.snapshot()
            if snapshot.receive.heard:
                break
        assert snapshot.receive.snr == 6.2
        assert snapshot.receive.bitrate == 'BITRATE (1)  18 bps RX'
        assert any(row.call == 'K7ABC' for row in snapshot.receive.heard)
        assert not snapshot.link.connected
        assert not port._attempts
    finally:
        controller.shutdown().result(timeout=2)
        assert controller.join(2)


@pytest.mark.parametrize('services,letter', [('V', 'A'), ('VEA', 'G')])
def test_injected_advanced_beacon_is_padded_without_a_kiss_socket(services, letter):
    desk, host, modem, fence = bridge()
    port = SessionAdapter(desk.session)
    desk.flow.services = services
    text = desk.session.beacon_text()
    assert text == f'20{letter}EN63BCW9SSJ'
    assert desk.host.client.kiss_socket is None
    assert desk.host.client.advanced_beacon_available
    desk.host.client.finish(desk.host.client.beacon_steps(text))
    assert modem.frames == [(text.ljust(v.BEACON_CHARS), False)]
    assert not port._attempts
    assert not desk.host.client.connected


def test_forced_advanced_beacon_uses_host_command_and_keeps_admission_distinct():
    desk, host, modem, fence = bridge()
    port = SessionAdapter(desk.session)
    text = desk.session.beacon_text()
    desk.host.client.finish(desk.host.client.forced_beacon_steps(text))
    assert modem.frames == [(text, True)]
    assert desk.host.command.sent[-1] == f'BEACONFORCE {text}\r'.encode()
    assert not port._attempts


def test_revoked_or_retired_submitter_cannot_enqueue_advanced_beacon():
    desk, host, modem, fence = bridge()
    text = desk.session.beacon_text()
    fence.live = False
    with pytest.raises(v.HostError, match='revoked'):
        host.beacon_submit(text.ljust(v.BEACON_CHARS))
    fence.live = True
    host.submitter_live = False
    with pytest.raises(v.HostError, match='retired'):
        host.beacon_submit(text.ljust(v.BEACON_CHARS))
    assert modem.frames == []


def test_beacon_submission_failure_never_falls_back_or_reports_a_sent_frame():
    desk, host, modem, fence = bridge()
    def fail(text):
        raise v.HostError('submission failed')
    modem.send_beacon = fail
    before = tuple(desk.host.command.sent)
    with pytest.raises(v.HostError, match='submission failed'):
        desk.host.client.finish(desk.host.client.beacon_steps(desk.session.beacon_text()))
    assert tuple(desk.host.command.sent) == before
    assert not modem.frames
    assert not any(event.kind == 'kiss-sent' for event in desk.host.events)


def test_receive_sink_copies_callbacks_and_decodes_only_on_controller_owner():
    from hfmodem.kestrel.broadcast import encode_ui
    desk, sink = Desk(), ReceiveSink()
    observed, callback = [], desk.host.client.event
    def event(value):
        observed.append((get_ident(), value.kind))
        callback(value)
    desk.host.client.event = event
    payload = bytearray(encode_ui('N0XYZ', 'CQ', b'hello'))
    def receive():
        sink.deliver('20AEN63BCN0XYZ')
        sink.deliver_frame(1, payload)
        payload[:] = b'changed'
    worker = Thread(target=receive)
    worker.start()
    worker.join(2)
    assert not worker.is_alive() and not observed
    sink.drain(desk.host.client)
    assert observed == [(get_ident(), 'kiss'), (get_ident(), 'broadcast')]
    assert any(row.call == 'N0XYZ' for row in desk.session.heard)
    for _ in range(129):
        sink.deliver('overflow')
    with pytest.raises(v.HostError, match='overflow'):
        sink.drain(desk.host.client)
    sink.close()


def test_boundary_bridge_overflow_refuses_mapping_instead_of_using_latest_epoch():
    desk, host, modem, fence = bridge()
    for n in range(129):
        host.modem_payload_boundary(PayloadBoundary(str(n), 0, 0, 0,
                                                    'CONNECTED W9SSJ K7ABC 500'))
    assert host.delivery_boundary('CONNECTED W9SSJ K7ABC 500') is None
    assert host.boundary_failed


def test_writer_rechecks_revocation_before_socket_submission_and_cancels_queue():
    entered, proceed = Event(), Event()
    fence = Fence()
    sent = []

    def admit():
        entered.set()
        assert proceed.wait(2)
        fence.check()

    stream = type('Stream', (), {'sendall': lambda self, payload: sent.append(payload),
                                'shutdown': lambda *args: None})()
    writer = DataWriter(stream, before_write=admit)
    try:
        writer.submit(b'first')
        assert entered.wait(2)
        writer.submit(b'second')
        fence.live = False
        proceed.set()
        writer._thread.join(2)
        assert not writer._thread.is_alive()
        assert [item.disposition for item in writer.poll()] == ['cancelled', 'cancelled']
        assert writer.poisoned and not sent
    finally:
        proceed.set()
        writer.close()


def test_client_close_bounds_compatibility_writer_wait_by_attachment_deadline():
    desk = Desk()
    waits = []
    desk.host.client.data_writer = type('Writer', (), {
        'close': lambda self, **kwargs: waits.append(kwargs['timeout'])})()
    desk.host.client.close(deadline=0)
    assert waits == [0]
    assert desk.host.client.closed


@pytest.mark.parametrize('offset,queued', [(0, 0), (4, 0), (0, 4)])
def test_production_connected_boundary_binds_only_a_fresh_stream(offset, queued):
    from hfmodem.tests.contact.test_chat_controller import execute
    from hfmodem.tests.contact.test_chat_delivery import latest
    desk = Desk()
    ledger = PayloadLedger()
    host = PrivateSession(Socket(), Socket(), Modem(), fence=Fence())
    port = SessionAdapter(desk.session, delivery_evidence=lambda: (ledger.evidence(),),
                          delivery_boundary=host.delivery_boundary)
    events = []
    port.on_event = lambda *args: events.append(args)
    desk.flow.dial = 14106500
    line = 'CONNECTED W9SSJ N0XYZ 500'
    host.modem_payload_boundary(PayloadBoundary(ledger.epoch, offset, 0, queued, line))
    desk.hear(line)
    _, result = execute(port, s.ChatSend('owned evidence'))
    payload = desk.host.data.sent[-1]
    ledger.queue(payload, payload)
    ledger.settle(len(payload))
    port.poll(0)
    outcome = latest(events)[result.result.text_id]
    if offset or queued:
        assert outcome.acknowledged_bytes == 0 and outcome.attribution == 'unattributed'
    else:
        assert outcome.status == 'acknowledged' and outcome.attribution == 'verified'


@pytest.mark.parametrize('failure', ['mismatch', 'overflow', 'callback'])
def test_bad_connected_boundary_unbinds_without_ending_line_console(failure):
    from hfmodem.tests.contact.test_chat_controller import execute
    desk, host, modem, fence = bridge()
    port = SessionAdapter(desk.session, delivery_boundary=host.delivery_boundary)
    desk.flow.dial = 14106500
    line = 'CONNECTED W9SSJ K7ABC 500'
    port.bind_delivery('old', desk.host.client.link_generation)
    if failure == 'callback':
        def broken(line): raise RuntimeError('bridge failed')
        port._delivery_boundary = broken
    else:
        for n in range(129 if failure == 'overflow' else 1):
            host.modem_payload_boundary(PayloadBoundary(str(n), 0, 0, 0,
                line if failure == 'overflow' else 'CONNECTED W9SSJ N0OTHER 500'))
    desk.hear(line)
    assert desk.host.client.connected
    assert port._delivery_binding is None
    _, result = execute(port, s.ChatSend('still connected'))
    assert result.state == 'done'
    assert desk.host.data.sent
