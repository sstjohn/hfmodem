# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""External pairs share chat's reservation through host and owner teardown."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from hfmodem.station.chat import AttachmentGate
from hfmodem.station.chat_attachment import serve_pair


class Modem:
    def __init__(self, *, acknowledge=True):
        self._retired_attachment = False
        self.acknowledge = acknowledge
        self.stopped = threading.Event()

    def stop(self):
        self._retired_attachment = self.acknowledge
        self.stopped.set()


class Session:
    def __init__(self, command, data, modem):
        self.modem = modem
        self.events = []

    def run(self): self.events.append('run')
    def close(self): self.events.append('close')
    def ensure_retired(self, **kwargs): self.events.append('ensure-retired')


def runtime(modem):
    return SimpleNamespace(attachments=AttachmentGate(), modem_factory=lambda: modem,
                           owner=SimpleNamespace(high_speed=False))


def assert_reserved(gate):
    with pytest.raises(Exception, match='attachment'):
        gate.reserve(object())


def test_external_reservation_spans_readers_and_audio_retirement():
    modem = Modem(acknowledge=False)
    owner = runtime(modem)
    sessions = []
    class CheckedSession(Session):
        def ensure_retired(self, **kwargs):
            assert_reserved(owner.attachments)
            super().ensure_retired()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(serve_pair, owner, None, None, CheckedSession,
                             publish=sessions.append, retirement_timeout=2)
        assert modem.stopped.wait(1)
        assert_reserved(owner.attachments)
        assert sessions[0].events == ['run', 'close', 'ensure-retired']
        modem._retired_attachment = True
        result.result(timeout=2)
    assert sessions[-1] is None
    token = object()
    owner.attachments.reserve(token)
    owner.attachments.release(token)


def test_chat_reservation_prevents_external_factory_creation():
    owner = runtime(Modem())
    owner.attachments.reserve(object())
    owner.modem_factory = lambda: pytest.fail('created a second attachment')
    with pytest.raises(Exception, match='attachment'):
        serve_pair(owner, None, None, Session)


def test_factory_failure_before_modem_releases_reservation():
    owner = runtime(Modem())
    def fail(): raise RuntimeError('factory failed')
    owner.modem_factory = fail
    with pytest.raises(RuntimeError, match='factory failed'):
        serve_pair(owner, None, None, Session)
    owner.attachments.reserve(object())


@pytest.mark.parametrize('failure', ['reader', 'owner'])
def test_uncertain_retirement_keeps_reservation(failure):
    owner = runtime(Modem(acknowledge=failure != 'owner'))
    class FailedSession(Session):
        def ensure_retired(self, **kwargs):
            if failure == 'reader': raise RuntimeError('reader did not retire')
    with pytest.raises(RuntimeError, match='did not'):
        serve_pair(owner, None, None, FailedSession, retirement_timeout=.01)
    assert owner.modem_factory().stopped.is_set()
    assert_reserved(owner.attachments)


def test_session_construction_failure_still_retires_published_modem():
    modem = Modem()
    owner = runtime(modem)
    def fail(*args): raise ValueError('session failed')
    with pytest.raises(ValueError, match='session failed'):
        serve_pair(owner, None, None, fail)
    assert modem.stopped.is_set()
    owner.attachments.reserve(object())


def test_host_and_connection_retirement_honor_an_expired_absolute_deadline():
    from hfmodem.core.hostio import Connection
    from hfmodem.kestrel.host.server import HostSession
    waits = []
    class Worker:
        def join(self, timeout): waits.append(timeout)
        def is_alive(self): return False
    conn = Connection.__new__(Connection)
    conn._writer = Worker()
    conn._delivery_thread = Worker()
    conn._opening_thread = Worker()
    host = HostSession.__new__(HostSession)
    host._threads = [Worker(), Worker()]
    host._cmd, host._data = conn, object()
    host.retirement_error = None
    host.ensure_retired(deadline=0)
    assert waits == [0.] * 5


@pytest.mark.parametrize('managed', [False, True])
def test_both_external_station_endpoints_use_shared_gate(monkeypatch, managed):
    from hfmodem.station import hosts
    from hfmodem.kestrel.host import server
    from hfmodem.core import managed_host
    class Server:
        session_class = Session
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.iamalive_interval = 60
            self._log = lambda *args: None
            self._kiss = None
    class Managed:
        def __init__(self, serve, **kwargs): self.serve = serve
    monkeypatch.setattr(server, 'VaraServer', Server)
    monkeypatch.setattr(server, 'HostSession', Session)
    monkeypatch.setattr(managed_host, 'ManagedHost', Managed)
    monkeypatch.setattr(hosts, '_managed_secret', lambda p: bytes(32))
    owner = runtime(Modem())
    station = SimpleNamespace(lanes={'vara': SimpleNamespace(runtime=owner)},
                               rig=object(), arbiter=object())
    config = SimpleNamespace(host='vara-managed' if managed else 'vara',
                              port=0, cmd_port=0, data_port=0, kiss_port=0, high_speed=False)
    endpoint = hosts._build_one('vara', config, station).server
    owner.attachments.reserve(object())
    with pytest.raises(Exception, match='attachment'):
        (endpoint.serve if managed else endpoint._serve_pair)(None, None)


@pytest.mark.parametrize('late', ['reader', 'owner'])
def test_late_retirement_recovers_on_next_attachment_without_waiting(late):
    modem = Modem(acknowledge=late != 'owner')
    owner = runtime(modem)
    ready = [False]
    deadlines = []
    class LateSession(Session):
        def ensure_retired(self, *, deadline):
            deadlines.append(deadline)
            if late == 'reader' and not ready[0]:
                raise RuntimeError('reader did not retire')
    with pytest.raises(RuntimeError, match='did not'):
        serve_pair(owner, None, None, LateSession, retirement_timeout=.001)
    assert_reserved(owner.attachments)
    ready[0] = True
    modem._retired_attachment = True
    import time
    before = time.monotonic()
    replacement = object()
    owner.attachments.reserve(replacement)
    assert before <= deadlines[-1] <= time.monotonic()
    owner.attachments.release(replacement)
    assert owner.attachments._owner is None
