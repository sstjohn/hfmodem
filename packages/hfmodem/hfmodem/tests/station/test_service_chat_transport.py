# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from hfmodem.station import wire
from hfmodem.station.client import Client, ClientError
from hfmodem.station.chat_transport import TokenBucket
from hfmodem.station.service import ReadBusy, Service
from hfmodem.station.sessions import Refused


class FakeChat:
    def __init__(self, service, station):
        self.service, self.station = service, station
        self.generation = 0
        self.reservations = []
        self.dispatched = threading.Event()
        self.completion = None
        self.reads = 0

    def validate(self, command):
        if command['kind'] not in wire.CHAT_READ_KINDS and command.get('control_id') != 'lease':
            raise Refused('not-controller')

    def reserve(self, command):
        if command['kind'] == 'chat.start':
            if command['args'].get('expected_lifecycle_generation') != self.generation:
                raise Refused('chat-lifecycle-stale')
            self.generation += 1
            self.reservations.append(command['id'])

    def lifecycle(self):
        return {'generation': self.generation}

    def live(self):
        return {'controller_id': None}

    def dispatch(self, command, completion):
        self.completion = completion
        self.dispatched.set()

    def pump(self):
        pass

    def read(self, path, query):
        self.reads += 1
        return {'path': path, 'text': '😀', 'query': query}

    def revoke(self):
        pass

    def close(self, deadline=None):
        pass


@pytest.fixture
def service(tmp_path, monkeypatch):
    module = ModuleType('hfmodem.station.chat')
    module.Chat = FakeChat
    monkeypatch.setitem(sys.modules, 'hfmodem.station.chat', module)
    config = tmp_path / 'station.toml'
    config.write_text('[station]\n')
    def authorize(command):
        raise AssertionError('chat must use explicit typed validation')
    sessions = SimpleNamespace(lock=threading.RLock(), authorize=authorize,
                               expire=lambda: None)
    stopped = threading.Event()
    station = SimpleNamespace(sessions=sessions, _stop=threading.Event(),
                              transitioning=False, stop_transmit=stopped.set)
    result = Service(config, station=station, port=0)
    result.stopped = stopped
    yield result
    result.close()


def command(kind='chat.start', id='start', **fields):
    return {'kind': kind, 'id': id, 'client': 'client', 'control_id': 'lease',
            'args': {'expected_lifecycle_generation': 0}, **fields}


def test_direct_envelope_size_cannot_bypass_http_limit(service):
    with pytest.raises(ValueError, match='4096'):
        service.submit(command(args={'text': '\x00' * 1000}))
    assert service.events.seq == 0 and service.commands.empty()


def test_read_commands_are_observer_accessible_mutation_requires_explicit_lease(service):
    read = command('chat.snapshot', control_id=None)
    ack = service.submit(read)
    assert service.events.after(ack['seq'] - 1)[0]['data']['state'] == 'accepted'
    denied = service.submit(command(id='denied', control_id=None))
    assert service.events.after(denied['seq'] - 1)[0]['data']['reason'] == 'not-controller'
    assert service.chat.reservations == []


def test_start_reservation_dedup_conflict_and_eviction_fence(service):
    envelope = command()
    first = service.submit(envelope)
    assert service.chat.generation == 1
    assert service.submit(dict(reversed(list(envelope.items())))) == first
    assert service.commands.qsize() == 1
    conflict = service.submit(command(args={'expected_lifecycle_generation': 5}))
    assert service.events.after(conflict['seq'] - 1)[0]['data']['reason'] == 'command-id-conflict'
    assert service.submit(envelope) == first
    assert service.chat.reservations == ['start']
    for n in range(1000):
        service._remember(str(n), {'id': str(n), 'seq': 1})
    assert not service._chat_payloads
    stale = service.submit(envelope)
    assert service.events.after(stale['seq'] - 1)[0]['data']['reason'] == 'chat-lifecycle-stale'
    assert service.chat.reservations == ['start']


def test_dispatch_handoff_does_not_block_stop_or_read_and_retains_terminal_first(service):
    service.submit(command())
    worker = threading.Thread(target=service._dispatch)
    worker.start()
    try:
        assert service.chat.dispatched.wait(1)
        service.submit(command('stop', id='stop'))
        assert service.stopped.is_set()
        assert json.loads(service.read_bulk('/chat/projection', {}))['path'] == '/chat/projection'
        original = service.events.publish
        def publish(kind, payload):
            if kind == 'command' and payload['state'] == 'done' and payload['id'] == 'start':
                assert service.results.lookup(service.incarnation, 'client', 'start')['status'] == 'found'
            return original(kind, payload)
        service.events.publish = publish
        service.chat.completion('done', result={'controller_id': 'new'})
        found = service.results.lookup(service.incarnation, 'client', 'start')
        assert found['result']['result'] == {'controller_id': 'new'}
    finally:
        service._closed.set()
        worker.join(1)
        assert not worker.is_alive()


def test_bulk_reads_share_concurrency_and_actual_utf8_budget(service):
    service._bulk_budget = TokenBucket.bulk(clock=lambda: 0)
    body = service.read_bulk('/chat/snapshot', {})
    assert body == json.dumps(json.loads(body), ensure_ascii=False, separators=(',', ':')).encode()
    assert service._bulk_budget.retry_after(service._bulk_budget.burst) > 0
    with pytest.raises(ReadBusy):
        service.read_bulk('/chat/result', {})
    assert service.chat.reads == 1
    assert service._bulk_lock.acquire(False)
    try:
        with pytest.raises(ReadBusy):
            service.read_bulk('/chat/projection', {})
    finally:
        service._bulk_lock.release()
    assert service.chat.reads == 1


def test_terminal_result_get_identity_and_subscriber_limit(service):
    service._result(command(), 'failed', reason='startup-failed')
    query = {'incarnation': [service.incarnation], 'client': ['client'], 'id': ['start']}
    assert json.loads(service.read_bulk('/commands/result', query))['result']['reason'] == 'startup-failed'
    assert [service._sse_slots.acquire(False) for _ in range(8)] == [True] * 8
    assert not service._sse_slots.acquire(False)
    for _ in range(8):
        service._sse_slots.release()


def test_client_resolves_gap_from_terminal_lookup_without_retry(monkeypatch):
    client = object.__new__(Client)
    client.client_id, client.timeout, client.snapshot = 'client', 2, None
    calls = []
    def request(method, path, body=None, **kwargs):
        calls.append((method, path))
        if method == 'POST':
            return {'id': 'start', 'seq': 1, 'incarnation': 'inc'}
        return {'status': 'found', 'result': {'id': 'start', 'client': 'client', 'state': 'done'}}
    monkeypatch.setattr(client, '_json', request)
    def events(*args, **kwargs):
        yield {'type': 'gap'}
    monkeypatch.setattr(client, 'events', events)
    assert client.command('chat.start', command_id='start', wait=True)['state'] == 'done'
    assert [method for method, _ in calls] == ['POST', 'GET']
    assert 'incarnation=inc' in calls[-1][1]
    monkeypatch.setattr(client, 'command_result', lambda *a, **k: {'status': 'expired'})
    with pytest.raises(ClientError, match='outcome unknown'):
        client.command('chat.start', command_id='start', wait=True)


def test_close_preserves_chat_failure_after_http_cleanup(service):
    cause = ValueError('original startup failure')
    failure = RuntimeError('retained chat attachment')
    failure.__cause__ = cause
    def close(deadline=None):
        raise failure
    service.chat.close = close
    called = []
    service._http = SimpleNamespace(shutdown=lambda: called.append('shutdown'),
                                     server_close=lambda: called.append('server_close'))
    try:
        with pytest.raises(RuntimeError) as error:
            service.close()
        assert error.value is failure and error.value.__cause__ is cause
        assert service.closing.is_set()
        assert called == ['shutdown', 'server_close']
    finally:
        service.chat.close = lambda deadline=None: None
        service._http = None


def test_lifecycle_replay_does_not_overwrite_or_resurrect_terminal_result(service):
    original = command()
    service.submit(original)
    service._result(original, 'done', result={'controller_id': 'old'})
    for n in range(1000):
        service._remember(str(n), {'id': str(n), 'seq': 1})
    service.submit(original)
    assert service.results.lookup(service.incarnation, 'client', 'start')['result']['state'] == 'done'
    for n in range(1000):
        service.results.remember(service.incarnation, 'client', str(n), {'state': 'done'})
    assert service.results.lookup(service.incarnation, 'client', 'start') == {'status': 'expired'}
    service._ids.pop(('client', 'start'), None)
    service._chat_payloads.pop(('client', 'start'), None)
    service.submit(original)
    assert service.results.lookup(service.incarnation, 'client', 'start') == {'status': 'expired'}


def test_pump_failure_publishes_gap_and_keeps_station_dispatch_alive(service):
    faulted = threading.Event()
    def pump():
        faulted.set()
        raise ValueError('escaped event exceeded budget')
    service.chat.pump = pump
    worker = threading.Thread(target=service._dispatch)
    worker.start()
    try:
        assert faulted.wait(1)
        service.submit(command('stop', id='stop'))
        assert service.stopped.is_set()
        assert worker.is_alive()
    finally:
        service._closed.set()
        worker.join(1)
    types = [event['type'] for event in service.events.after(0)]
    assert 'gap' in types and 'warning' in types


def test_client_waits_past_gap_before_terminal_result_exists(monkeypatch):
    client = object.__new__(Client)
    client.client_id, client.timeout, client.snapshot = 'client', 2, None
    posts = []
    def request(*args, **kwargs):
        posts.append(args)
        return {'id': 'pending', 'seq': 1, 'incarnation': 'inc'}
    monkeypatch.setattr(client, '_json', request)
    monkeypatch.setattr(client, 'command_result', lambda *a, **k: {'status': 'expired'})
    def events(*args, **kwargs):
        yield {'type': 'gap'}
        yield {'type': 'command', 'data': {'id': 'pending', 'state': 'running'}}
        yield {'type': 'command', 'data': {'id': 'pending', 'state': 'done'}}
    monkeypatch.setattr(client, 'events', events)
    assert client.command('release', command_id='pending', wait=True)['state'] == 'done'
    assert len(posts) == 1
