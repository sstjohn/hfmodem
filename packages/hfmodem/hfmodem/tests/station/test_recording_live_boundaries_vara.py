# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA core edges precede host delivery, including rejected application links."""
import threading
from types import SimpleNamespace as NS

import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.tests.station.recording_live_support import (
    LOCAL, PEER, OTHER, Clock, Observer, assert_edges,
)


class Transport(A.VaraIO):
    def __init__(self):
        self.deliveries = []

    def connected(self, *args):
        self.deliveries.append(args)

    def log(self, message):
        pass


@pytest.fixture(autouse=True)
def source_clock(monkeypatch):
    monkeypatch.setattr(A, 'time', NS(monotonic=Clock()))


def core(owner, monkeypatch, *, incoming=False, mfsk=False):
    io = Transport()
    hs = owner.bind(A.VaraStationHandshake([LOCAL], io, bw='500', mfsk_only=mfsk))
    hs.caller, hs.called = (PEER, LOCAL) if incoming else (LOCAL, PEER)
    hs.role = 'responder' if incoming else 'initiator'
    hs.state = A.VaraState.CONNECTING
    monkeypatch.setattr(hs, '_tx_connected_ack', lambda: True)
    monkeypatch.setattr(hs, '_send_burst', lambda *a, **kw: True)
    monkeypatch.setattr(hs, '_band_busy', lambda: False)
    return hs, io


@pytest.mark.parametrize('edge', ['outgoing', 'incoming-setup', 'incoming-mfsk'])
def test_three_connected_edges_stamp_before_application_delivery(live, monkeypatch, edge):
    owner = live()
    owner.advance(65)
    hs, io = core(owner, monkeypatch, incoming=edge != 'outgoing', mfsk=edge == 'incoming-mfsk')
    delivered = []
    io.connected = lambda *args: delivered.append((args, len(owner.starts)))
    if edge == 'outgoing':
        hs._connected(confirm=False)
    elif edge == 'incoming-setup':
        assert hs._take_link_setup(F.link_setup_frame(PEER, bw='500'))
    else:
        hs.state = A.VaraState.DISCONNECTED
        hs.listen()
        kind = F.connect_request('500')
        monkeypatch.setattr(hs, '_matches', lambda *a: True)
        monkeypatch.setattr(F, 'best_match', lambda *a: (LOCAL, 1, 1))
        monkeypatch.setattr(hs, '_connect_answer', lambda: kind)
        hs.on_rx_tones([], kind)
    assert delivered and delivered[0][1] == 1, 'boundary must precede io.connected'
    row = owner.one()
    assert row['positions'] == {'card': 520, 'dac': 325}
    assert row['peer'] == (A.CALLER_UNKNOWN if edge == 'incoming-mfsk' else PEER)
    assert row['key'][1] == 'vara'
    assert owner.stamps == [threading.get_ident()]


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_local_disconnect_ends_at_transmitted_close_without_progress(live, monkeypatch, role):
    owner = live()
    owner.advance(65)
    hs, _ = core(owner, monkeypatch, incoming=role == 'responder')
    hs._connected(confirm=False)
    owner.advance(67)
    hs.disconnect()
    hs.disconnect()
    owner.advance(72)
    assert hs.state == A.VaraState.DISCONNECTED
    row = assert_edges(owner, start={'card': 520, 'dac': 325},
                       end={'card': 536, 'dac': 335})
    assert row['lanes']['rx']['actual'] == [40, 576]
    assert row['lanes']['submitted-tx']['actual'] == [25, 360]


@pytest.mark.parametrize('stopping', [False, True])
def test_host_retirement_keeps_minimal_partial_even_when_delivery_is_stopped(live, stopping):
    owner = live('off')
    transport = Transport()
    owner.bind(transport)
    modem = owner.bind(VaraHostModem(transport, bandwidth='500', clock=Clock()))
    observer = Observer(owner)
    modem.observer = observer
    owner.bind(modem.station)
    modem.station.caller, modem.station.called = LOCAL, PEER
    modem.station.role = 'initiator'
    modem._stopping = stopping
    modem.station._connected(confirm=False)
    if not stopping:
        modem._stopping = True
    modem._poll()
    row = owner.one()
    assert row['state'] == 'partial'
    assert len(owner.starts) == len(owner.ends) == 1
    if stopping:
        assert not transport.deliveries
    else:
        assert observer.events[0][2] == 1
    assert list(owner.root.rglob('*.wav')) == []


def test_unsuccessful_attempt_and_broadcast_do_not_select_audio(live, monkeypatch):
    owner = live()
    hs, io = core(owner, monkeypatch)
    hs.listen()
    io.pending()
    io.data(b'unconnected payload')
    assert not owner.rows()
    owner.advance(65)
    hs.caller, hs.called, hs.role = LOCAL, PEER, 'initiator'
    hs._connected(confirm=False)
    assert owner.one()['peer'] == PEER


def test_recording_failure_cannot_cancel_connected_application_delivery(live, monkeypatch):
    reports = []
    owner = live('off', report=reports.append)
    hs, io = core(owner, monkeypatch)
    monkeypatch.setattr(owner.session, 'begin', lambda *a, **kw: (_ for _ in ()).throw(
        OSError('synthetic boundary storage failure')))
    hs._connected(confirm=False)
    assert io.deliveries == [(LOCAL, PEER, '500')]
    owner.rows()
    assert any('synthetic boundary storage failure' in str(r) for r in reports)


@pytest.mark.parametrize('end', ['peer-ident', 'peer-control', 'peer-mfsk'])
def test_peer_exit_paths_close_without_waiting_for_host_poll(live, monkeypatch, end):
    owner = live('off')
    hs, _ = core(owner, monkeypatch, incoming=True)
    hs._connected(confirm=False)
    if end == 'peer-ident':
        hs._took_peer_ident()
    elif end == 'peer-control':
        hs._answer_responder_control(F.SESSION_DISCONNECT_REQ)
    else:
        monkeypatch.setattr(hs, '_matches', lambda *a: True)
        hs.on_rx_tones([], F.SESSION_DISCONNECT_REQ)
    assert hs.state == A.VaraState.DISCONNECTED
    row = owner.one()
    assert row.get('end_positions') is not None
    assert len(owner.starts) == len(owner.ends) == 1


@pytest.mark.parametrize('application', ['mail', 'varac-rejection', 'varac-shutdown'])
def test_application_refusal_or_shutdown_cannot_erase_source_contact(
        live, monkeypatch, tools, application):
    import kestrel_connect
    from hfmodem.contact.varac import Event, Session

    owner = live('off')
    hs, io = core(owner, monkeypatch, incoming=True)
    seen = []
    if application == 'mail':
        seam = kestrel_connect._MailSeam()
        seam.client = NS(link_up=lambda: seen.append(len(owner.starts)))
        io.connected = seam.connected
    else:
        app = Session.__new__(Session)
        app.contact, app.intent = None, None
        app.verbose, app.auto = False, False
        app._closing = application == 'varac-shutdown'
        app._quitting = app._force_close = False
        app.client = NS(send_command=lambda command: seen.append((command, len(owner.starts))))
        app._farewell = lambda: iter(())
        app._run = lambda steps: seen.append(('farewell', len(owner.starts)))
        if application == 'varac-rejection':
            io.connected = lambda *args: app._begin()
        else:
            io.connected = lambda *args: app.handle(Event('host', f'CONNECTED {PEER} {LOCAL} 500'))
    hs._connected(confirm=False)
    hs._took_peer_ident()
    row = owner.one()
    assert row['peer'] == PEER and row.get('end_positions') is not None
    assert seen and all((event if isinstance(event, int) else event[1]) == 1 for event in seen)
    if application != 'mail':
        assert app.contact is None and app._force_close


def test_rejected_source_contact_publishes_minimal_arm(live, monkeypatch, tmp_path):
    from hfmodem.station.contactlog import ContactLog

    owner = live('off')
    journal = ContactLog(tmp_path / 'sessions' / 'vara-run', recording=owner.session)
    hs, _ = core(owner, monkeypatch)
    hs._connected(confirm=False)
    hs._disconnected('host retired')
    owner.finish()
    record, = journal.records.values()
    assert record['call'] == PEER
    assert record['recording']['key'][1] == 'vara'
    assert record['recording']['state'] == 'partial'
    assert len(list((tmp_path / 'arms').glob('*/provenance.json'))) == 1


def test_varac_metadata_uses_source_id_and_does_not_end_radio_link(live, monkeypatch, tmp_path):
    from hfmodem.station.contactlog import ContactLog

    owner = live('off')
    journal = ContactLog(tmp_path / 'sessions' / 'vara-run', recording=owner.session)
    hs, _ = core(owner, monkeypatch)
    hs._connected(confirm=False)
    cid = owner.one()['id']
    contact = dict(n=1, peer=PEER, dial_hz=7100000, bandwidth_hz=500,
                   how='outgoing', started_utc='2026-11-02T00:00:00+00:00')
    journal.update(contact, recording_id=cid)
    journal.update(contact | dict(ended_utc='2026-11-02T00:00:01+00:00'))
    assert owner.one().get('end_positions') is None
    hs._took_peer_ident()
    owner.finish()
    assert len(owner.starts) == len(owner.ends) == len(journal.records) == 1
    assert len(list((tmp_path / 'arms').glob('*/provenance.json'))) == 1


def test_inactivity_close_is_partial_at_the_source_edge(live, monkeypatch):
    owner = live('off')
    hs, _ = core(owner, monkeypatch)
    hs._connected(confirm=False)
    hs.chat = True
    hs._poll_at = -1
    hs._since_progress = A._MAX_WITHOUT_PROGRESS
    hs._idle_poll()
    row = owner.one()
    assert row['end_positions'] == {'card': 0, 'dac': 0}
    assert row['state'] == 'partial'


def test_late_application_metadata_preserves_published_source_result(live, monkeypatch, tmp_path):
    from hfmodem.station.contactlog import ContactLog

    owner = live('off')
    journal = ContactLog(tmp_path / 'sessions' / 'vara-run', recording=owner.session)
    hs, _ = core(owner, monkeypatch)
    hs._connected(confirm=False)
    hs._disconnected('host retired')
    selected, = owner.finish().values()
    contact = dict(n=1, peer=PEER, dial_hz=7100000, bandwidth_hz=500,
                   how='outgoing', started_utc='2026-11-02T00:00:00+00:00',
                   ended_utc='2026-11-02T00:00:01+00:00')
    row = journal.update(contact, recording_id=selected['id'])
    assert row['recording']['state'] == 'partial'
    assert row['recording']['end_positions'] == selected['end_positions']
    assert len(journal.records) == len(owner.starts) == len(owner.ends) == 1


@pytest.mark.parametrize('failure', ['begin', 'positions', 'end'])
def test_boundary_and_reporter_failure_preserve_protocol_continuation(live, monkeypatch, failure):
    called = threading.Event()
    reporting_threads = []

    def report(message):
        if 'synthetic' in message:
            reporting_threads.append(threading.get_ident())
            called.set()
            raise OSError('synthetic reporter failure')

    owner = live('off', report=report)
    hs, io = core(owner, monkeypatch)
    def broken(*args, **kwargs):
        raise OSError('synthetic boundary failure')
    if failure == 'begin':
        monkeypatch.setattr(owner.session, 'begin', broken)
    elif failure == 'positions':
        hs.recording_positions = broken
    confirmed = []
    monkeypatch.setattr(hs, '_tx_session_confirm', lambda: confirmed.append(True))
    hs._connected(confirm=True)
    assert confirmed == [True]
    assert io.deliveries == [(LOCAL, PEER, '500')]
    if failure == 'end':
        monkeypatch.setattr(owner.session, 'end', broken)
        hs._disconnected('peer close')
        assert hs.state is A.VaraState.DISCONNECTED
    assert called.wait(2)
    assert threading.get_ident() not in reporting_threads


def test_delayed_varac_connection_event_keeps_source_id_after_station_replacement(
        live, monkeypatch, tmp_path, tools):
    import varac_chat
    from hfmodem.contact.varac import Event
    from hfmodem.tests.contact.test_varac import Desk

    owner = live('off')
    io = Transport()
    host = varac_chat.Host(io)
    modem = owner.bind(VaraHostModem(io, bandwidth='500', clock=Clock()))
    wire = host.server_factory.session_class(NS(sendall=lambda data: None),
                                            NS(sendall=lambda data: None), modem)
    modem.observer = wire
    wire._command_depth = 1
    owner.bind(modem.station)
    modem.station.caller, modem.station.called = LOCAL, PEER
    modem.station.role = 'initiator'
    modem.station._connected(confirm=False)
    first = owner.one()['id']
    modem.station._disconnected('station replaced')
    old = modem.station
    modem.station = owner.bind(modem._new_station())
    modem.station.caller, modem.station.called = LOCAL, OTHER
    modem.station.role = 'initiator'
    modem.station._connected(confirm=False)
    queued = [value for kind, value in wire._events if str(value).startswith('CONNECTED ')]
    assert getattr(queued[0], 'recording_id', None) == first
    wire._finish_command()
    app = Desk(mailbox=tmp_path / 'mail', mycall=LOCAL)
    app.host.client.mycall = LOCAL
    item = Event('host', f'CONNECTED {LOCAL} {PEER} 500')
    stamp = getattr(host, 'stamp_event', lambda event: event)
    item = stamp(item)
    assert getattr(item, 'recording_id', None) == first
    app.host.client.connected = True
    app.session.handle(item)
    assert app.session.contact['recording_id'] == first
    app.session._end()
    records = [event for event in app.events if event['kind'] == 'contact']
    assert records and all(event['recording_id'] == first for event in records)
    assert old._recording_id == first
    second = next(cid for cid in owner.rows() if cid != first)
    successor = host.stamp_event(Event('host', f'CONNECTED {LOCAL} {OTHER} 500'))
    assert successor.recording_id == second


def test_varac_application_source_association_is_bound_once(live, monkeypatch, tmp_path):
    from hfmodem.station.contactlog import ContactLog

    owner = live('off')
    journal = ContactLog(tmp_path / 'sessions' / 'vara-run', recording=owner.session)
    hs, _ = core(owner, monkeypatch)
    hs._connected(confirm=False)
    first = owner.one()['id']
    hs._disconnected('station replaced')
    hs, _ = core(owner, monkeypatch)
    hs._connected(confirm=False)
    second = next(cid for cid in owner.rows() if cid != first)
    contact = dict(n=1, peer=PEER, dial_hz=7100000, bandwidth_hz=500,
                   how='outgoing', started_utc='2026-11-02T00:00:00+00:00')
    journal.update(contact, recording_id=first)
    record = journal.update(contact | dict(ended_utc='2026-11-02T00:00:01+00:00'),
                            recording_id=second)
    assert journal.source_ids[1] == first
    assert record['recording']['id'] == first
