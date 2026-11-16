# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Station-owned observation routing never consumes another session's credit."""
import queue
from types import SimpleNamespace

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.host.station_runtime import StationTransport
from hfmodem.station.diagnostics import Diagnostics
from hfmodem.station.process import Station
from hfmodem.tests.contact.test_varac import Desk


def test_single_station_tick_drains_and_copies_only_matching_authority():
    router = Diagnostics()
    first, second, public = [], [], []
    router.subscribe(('chat-a', 3), lambda kind, **fields: first.append((kind, fields)))
    router.subscribe(('chat-b', 4), lambda kind, **fields: second.append((kind, fields)))
    arbiter = SimpleNamespace(records=queue.SimpleQueue())
    rig = SimpleNamespace(meter_reports=queue.SimpleQueue())
    arbiter.records.put(dict(authority=('chat-a', 3), protocol='vara', outcome='released'))
    arbiter.records.put(dict(authority=('chat-b', 4), protocol='ident', outcome='released'))
    rig.meter_reports.put(dict(authority=('chat-a', 3), peak_watts=12., peak_alc=.2))
    rig.meter_reports.put(dict(peak_watts=99.))  # Never infer its owner from the current chat.
    router.put('acknowledged', authority=('chat-a', 3), bytes=5)
    item = Station.__new__(Station)
    item.diagnostics, item.arbiter, item.rig = router, arbiter, rig
    item.service = SimpleNamespace(events=SimpleNamespace(publish=lambda *args: public.append(args)))
    import threading
    item._stop = threading.Event()
    item._identify_if_due = lambda: None
    item.capture = None
    item.replay = None
    item._poll_stop = SimpleNamespace(wait=lambda delay: None)
    item.lanes = {}
    item._tick()
    assert [kind for kind, _ in first] == ['arbiter', 'tx_meters', 'acknowledged']
    assert [kind for kind, _ in second] == ['arbiter']
    assert first[1][1]['peak_watts'] == 12.
    assert len(public) == 4
    assert public[-1][1]['authority'] is None
    assert arbiter.records.empty() and rig.meter_reports.empty()
    item._tick()
    assert len(first) == 3 and len(second) == 1 and len(public) == 4


def test_bounded_runtime_queue_reports_loss_and_unsubscribe_stops_delivery():
    router = Diagnostics(capacity=1)
    notes = []
    leave = router.subscribe(('chat', 1), lambda kind, **fields: notes.append((kind, fields)))
    fields = dict(nested={'value': 1})
    router.put('first', authority=('chat', 1), **fields)
    fields['nested']['value'] = 99
    router.put('second', authority=('chat', 1))
    router.drain()
    assert notes == [('first', {'nested': {'value': 1}}), ('dropped', {'count': 1})]
    leave()
    router.put('late', authority=('chat', 1))
    router.drain()
    assert len(notes) == 2


def test_runtime_credit_uses_captured_modem_authority_and_station_route():
    router = Diagnostics()
    notes = []
    router.subscribe(('chat', 7), lambda kind, **fields: notes.append((kind, fields)))
    station = SimpleNamespace(arbiter=None, audio=SimpleNamespace(samples=0), diagnostics=router)
    transport = StationTransport(SimpleNamespace(station=station))
    runtime = AudioHostRuntime(transport)
    runtime.modem = SimpleNamespace(authority=('chat', 7), confirmed_total=5, discarded_total=2)
    runtime._report_payload_credit()
    assert notes == []
    router.drain()
    assert notes == [('acknowledged', {'bytes': 5}), ('discarded', {'bytes': 2})]
    runtime._report_payload_credit()
    router.drain()
    assert len(notes) == 2


def test_tagged_meter_and_arbiter_observations_reach_controller_notices():
    desk = Desk()
    port = SessionAdapter(desk.session)
    notes = []
    port.on_event = lambda payload, *rest: notes.append(payload)
    desk.session.diagnostic('tx_meters', peak_watts=12., peak_alc=.2)
    desk.session.diagnostic('arbiter', protocol='vara', outcome='released')
    port._flush()
    text = [note.text for note in notes if isinstance(note, s.Notice)]
    assert any('12 W' in line for line in text)
    assert any('vara: released' in line for line in text)


def test_attachment_drop_count_survives_producer_during_owner_reporting(monkeypatch):
    from hfmodem.station import chat
    attachment = chat.Attachment(SimpleNamespace(station=object(), fence=lambda: True))
    attachment._initializing = False
    attachment._cleaning = True
    attachment.runtime = SimpleNamespace(failure=None)
    attachment._diagnostics = queue.Queue(1)
    notes = []
    def diagnostic(kind, **fields):
        notes.append((kind, fields))
        if kind == 'dropped' and len(notes) == 2:
            import threading
            def produce():
                attachment._diagnostic('next')
                attachment._diagnostic('overflow')
            worker = threading.Thread(target=produce)
            worker.start()
            worker.join(1)
            assert not worker.is_alive()
    attachment.session = SimpleNamespace(diagnostic=diagnostic)
    monkeypatch.setattr(chat.select, 'select', lambda *args: ([], [], []))
    attachment._diagnostic('first')
    attachment._diagnostic('overflow')
    attachment._ready([], 0)
    attachment._ready([], 0)
    assert notes == [('first', {}), ('dropped', {'count': 1}),
                     ('next', {}), ('dropped', {'count': 1})]
