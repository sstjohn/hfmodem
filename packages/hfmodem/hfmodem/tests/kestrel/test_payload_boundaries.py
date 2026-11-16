# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owner notification boundaries use immutable ledger identity, not counters."""
from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor

import pytest

from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from .test_compression import IO
from .test_vara_host_modem import Observer


class Boundaries(Observer):
    def __init__(self):
        super().__init__()
        self.events = []

    def modem_payload_boundary(self, boundary):
        self.events.append(('boundary', boundary))

    def modem_connected(self, caller, called, bw):
        self.events.append(('connected', caller, called, bw))


def notify_connected(modem, bw='500'):
    modem.station.io.connected('W9SSJ', 'K7ABC', bw)


def test_boundary_precedes_exact_connected_notification_and_is_immutable():
    modem = VaraHostModem(IO())
    observer = Boundaries()
    modem.start(observer)
    epoch = modem.station.payload_ledger.epoch
    notify_connected(modem)
    assert observer.events[1] == ('connected', 'W9SSJ', 'K7ABC', '500')
    kind, boundary = observer.events[0]
    assert kind == 'boundary'
    assert (boundary.epoch, boundary.offset, boundary.sequence, boundary.queued_original) == (
        epoch, 0, 0, 0)
    assert boundary.line == 'CONNECTED W9SSJ K7ABC 500'
    with pytest.raises(FrozenInstanceError):
        boundary.offset = 5
    modem._replace_station()
    assert boundary.epoch == epoch != modem.station.payload_ledger.epoch


def test_boundary_exposes_preexisting_and_not_yet_applied_original_bytes():
    modem = VaraHostModem(IO())
    observer = Boundaries()
    modem.start(observer)
    modem.station.payload_ledger.queue(b'old', b'old')
    modem.transmit(b'queued')
    notify_connected(modem, None)
    boundary = observer.events[0][1]
    assert (boundary.offset, boundary.queued_original) == (3, 6)
    assert boundary.line == 'CONNECTED W9SSJ K7ABC'


def test_runtime_recovers_distinct_epoch_tokens_across_replacement_and_attachment():
    runtime = AudioHostRuntime(IO())
    with runtime.subscribe_payload_evidence() as read:
        modem = runtime.modem_factory()
        runtime.modem = runtime._new.get_nowait()
        modem.start(Boundaries())
        notify_connected(modem)
        first = runtime.payload_boundaries()
        modem._replace_station()
        notify_connected(modem)
        runtime._report_payload_credit()
        assert len(runtime.payload_boundaries()) == 2
        assert len(first) == 1  # Returned history cannot mutate underneath a reader.
        assert first[0].epoch != runtime.payload_boundaries()[1].epoch
        assert read()[-1].epoch == runtime.payload_boundaries()[1].epoch
        second = runtime.modem_factory()
        runtime.modem = runtime._new.get_nowait()
        second.start(Boundaries())
        notify_connected(second)
        runtime._report_payload_credit()
        assert len({b.epoch for b in runtime.payload_boundaries()}) == 3
        assert read()[-1].epoch == runtime.payload_boundaries()[-1].epoch
    assert runtime.payload_boundaries() == ()
    assert runtime.payload_evidence() == ()


def test_runtime_boundary_snapshots_can_be_read_off_owner_thread():
    runtime = AudioHostRuntime(IO())
    with runtime.subscribe_payload_evidence(), ThreadPoolExecutor(max_workers=1) as worker:
        modem = runtime.modem_factory()
        modem.start(Boundaries())
        for _ in range(10):
            notify_connected(modem)
            captured = worker.submit(runtime.payload_boundaries).result(timeout=2)
            assert isinstance(captured, tuple)
            assert all(b.epoch == modem.station.payload_ledger.epoch for b in captured)


def test_legacy_observer_needs_no_boundary_callback():
    modem = VaraHostModem(IO())
    modem.start(Observer())
    notify_connected(modem)
    assert modem.connected


@pytest.mark.parametrize('failure', ['overflow', 'mismatch'])
def test_reporting_session_bad_boundary_stays_unattributed(failure):
    import queue
    from hfmodem.kestrel.host.server import ReportingSession
    from hfmodem.kestrel.host.vara_modem import PayloadBoundary
    host = ReportingSession.__new__(ReportingSession)
    host.payload_boundaries = queue.Queue(1)
    host.boundary_failed = False
    line = 'CONNECTED W9SSJ K7ABC 500'
    wrong = 'CONNECTED W9SSJ N0OTHER 500'
    host.modem_payload_boundary(PayloadBoundary('old', 0, 0, 0,
                                               line if failure == 'overflow' else wrong))
    if failure == 'overflow':
        host.modem_payload_boundary(PayloadBoundary('new', 0, 0, 0, line))
    assert host.delivery_boundary(line) is None
    assert host.boundary_failed
    assert host.delivery_boundary(line) is None
