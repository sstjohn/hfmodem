# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import contextmanager
import errno
import http.client
import json
import os
import signal
import threading

import pytest

from hfmodem.station import client, wire
from hfmodem.station.service import BindConflict, Events, Service
from .test_process import fake_claim as fake_claim, station as station


def command(station, kind, id='one', **args):
    control = station.sessions.snapshot()['control'] or station.sessions.control('test')
    return {'id': id, 'client': 'test', 'kind': kind, 'args': args,
            'control_id': control['control_id'], 'generation': control['generation']}


def credentials(attached):
    control = attached.command('control', wait=True)['result']
    return {k: control[k] for k in ('control_id', 'generation')}


@contextmanager
def running(station):
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    try:
        service.bind()
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            pytest.skip(f'loopback bind blocked by sandbox: {exc}')
        raise
    try:
        yield service
    finally:
        service.close()
        if station.state != 'observing':
            station.release()


def test_event_replay_retention_and_gap():
    clock = [0]
    events = Events(capacity=2, age=10, clock=lambda: clock[0])
    for n in range(3):
        events.publish('tape', {'n': n})
    assert len(events.after(0)) == 3
    assert [e['seq'] for _, e in events.ring] == [2, 3]
    assert events.after(0)[0]['type'] == 'gap'
    clock[0] = 11
    events.publish('tape', {'n': 3})
    replay = events.after(2)
    assert [event['seq'] for event in replay] == [3, 4]
    gap = events.after(0)[0]
    assert gap['type'] == 'gap' and gap['data'] == {'from': 0, 'to': 4}
    assert gap['seq'] == 3
    assert [e['seq'] for e in events.after(gap['seq'])] == [4]
    assert events.seq == 4
    assert all(e['type'] != 'gap' for e in events.after(3))
    clock[0] = 21
    assert events.after(4) == [] and not events.ring
    with pytest.raises(ValueError):
        events.publish('invented', {})


def test_ack_names_first_event_and_duplicate_does_nothing(station):
    service = Service(station.station_file, station=station, port=0)
    first = service.submit(command(station, 'release'))
    assert first == service.submit(command(station, 'own'))
    assert service.commands.qsize() == 1
    event = service.events.after(first['seq'] - 1)[0]
    assert event['seq'] == first['seq'] and event['data']['state'] == 'accepted'


def test_queue_is_bounded_and_stop_bypasses_it(station, monkeypatch):
    service = Service(station.station_file, station=station, port=0)
    for i in range(64):
        service.submit(command(station, 'release', str(i)))
    ack = service.submit(command(station, 'release', 'overflow'))
    assert service.events.after(ack['seq'] - 1)[0]['data']['reason'] == 'queue-full'
    stopped = threading.Event()
    monkeypatch.setattr(station, 'stop_transmit', stopped.set)
    service.submit(command(station, 'stop', 'stop'))
    assert stopped.is_set() and service.commands.qsize() == 64


def test_transition_is_refused_before_queue(station):
    service = Service(station.station_file, station=station, port=0)
    with station._lifecycle:
        ack = service.submit(command(station, 'release'))
    assert service.commands.empty()
    assert service.events.after(ack['seq'] - 1)[0]['data']['reason'] == 'transitioning'


def test_signal_teardown(station):
    from hfmodem.tests.core.test_arbiter import Card, LineRig, req
    from hfmodem.station.arbiter import TxArbiter
    from .test_process import FakeLine

    station.own()
    card = Card()
    card.wait_gate = threading.Event()
    card.close = lambda: None
    rig = LineRig(card)
    rig.ptt = FakeLine()
    rig.close = rig.ptt.release
    station.audio, station.rig = card, rig
    station.arbiter = TxArbiter(card, rig)
    service = Service(station.station_file, station=station, port=0)
    results = []
    burst = threading.Thread(target=lambda: results.append(station.arbiter.submit_result(req())))
    burst.start()
    try:
        assert card.wait_entered.wait(2)
        assert rig.keyed
        for i in range(128):
            service.submit(command(station, 'release', str(i)))
        assert service.commands.qsize() == 64
        service.submit(command(station, 'stop', 'stop'))
        burst.join(3)
        assert not burst.is_alive() and not rig.keyed
        assert any(call[0] == 'unkey' and call[2] != threading.current_thread().name
                   for call in rig.calls)
        station._signal(signal.SIGTERM, None)
        assert station._signals == 1
        station.shutdown()
        assert station.state == 'observing' and station.claim is None
        assert rig.ptt.released_low is True
    finally:
        card.wait_gate.set()
        burst.join(3)
        if station.arbiter:
            station.arbiter.close()


def test_client_own_release_wait_against_service(station):
    with running(station) as service:
        attached = client.attach(station.station_file, port=service.port)
        assert attached.exit_code == wire.EXIT_ATTACHED
        assert attached.snapshot['service']['state'] == 'observing'
        control = credentials(attached)
        result = attached.own(receive=True, **control)
        assert result['state'] == 'done' and result['result']['state'] == 'owning-receive'
        assert attached.release(**control)['result']['state'] == 'observing'
        assert attached.own(**control)['result']['state'] == 'owning'
        assert attached.release(**control)['result']['state'] == 'observing'


def test_attach_4_and_server_3_4(station, tmp_path):
    with running(station) as service:
        other = tmp_path / 'other.toml'
        other.write_text('[station]\n')
        with pytest.raises(client.PortConflict) as error:
            client.attach(other, port=service.port)
        assert error.value.exit_code == 4
        with pytest.raises(BindConflict) as error:
            Service(station.station_file, port=service.port).bind()
        assert error.value.exit_code == 3
        with pytest.raises(BindConflict) as error:
            Service(other, port=service.port).bind()
        assert error.value.exit_code == 4


def test_attach_6_when_real_service_stays_starting(station, monkeypatch):
    with running(station) as service:
        service.station = None
        monkeypatch.setattr(client.time, 'sleep', lambda duration: None)
        with pytest.raises(client.LaunchFailed) as error:
            client.attach(station.station_file, port=service.port,
                          launch_timeout=.05, poll_interval=.001)
        assert error.value.exit_code == 6


def test_attach_launch_path_with_real_service(station, monkeypatch, tmp_path):
    service = Service(station.station_file, station=station, port=0)
    try:
        service.bind()
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            pytest.skip(f'loopback bind blocked by sandbox: {exc}')
        raise
    port = service.port
    service.close()
    launched = []
    replacement = Service(station.station_file, station=station, port=port)

    def launch(argv, **kwargs):
        launched.append((argv, kwargs))
        replacement.bind()
        return type('Child', (), {'poll': lambda self: None})()

    monkeypatch.setattr(client.shutil, 'which', lambda executable: '/fake/hfmodem')
    monkeypatch.setattr(client.subprocess, 'Popen', launch)
    try:
        attached = client.attach(station.station_file, port=port, data_root=tmp_path)
        assert attached.exit_code == 0
        assert launched[0][0][-1] == '--detach'
        assert launched[0][1]['start_new_session'] is True
    finally:
        replacement.close()


def test_http_validation_mount_and_cached_state(station, monkeypatch):
    with running(station) as service:
        service.mount('/extra', lambda handler: handler.reply(200, {'mounted': True}))
        connection = http.client.HTTPConnection('127.0.0.1', service.port, timeout=2)
        try:
            for headers, body, status in [
                ({'Content-Type': 'application/json', 'Origin': 'http://foreign'}, '{}', 403),
                ({'Content-Type': 'text/plain'}, '{}', 415),
                ({'Content-Type': 'application/json'}, 'x' * 4097, 413),
                ({'Content-Type': 'application/json'}, '[]', 400),
            ]:
                connection.request('POST', wire.COMMANDS, body=body, headers=headers)
                response = connection.getresponse()
                response.read()
                assert response.status == status
            connection.request('GET', '/extra/item')
            response = connection.getresponse()
            assert json.loads(response.read()) == {'mounted': True}
            assert response.getheader('Cache-Control') == 'no-store'
            monkeypatch.setattr(station, 'open_rig', lambda **kw: pytest.fail('HTTP opened rig'))
            assert client.connect(station.station_file, port=service.port).state()['rig']['retired'] is False
        finally:
            connection.close()


@pytest.fixture
def socketpair_http(monkeypatch):
    import socket
    from hfmodem.station import service as module
    servers = []

    class SocketpairServer:
        def __init__(self, address, handler):
            self.server_port = 8737
            self.handler = handler
            self.stopped = threading.Event()
            self.threads = []
            self.sockets = []
            servers.append(self)

        def serve_forever(self, **kwargs):
            self.stopped.wait()

        def shutdown(self):
            self.stopped.set()

        def server_close(self):
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
            for thread in self.threads:
                thread.join(2)
                assert not thread.is_alive()

        def connect(self, connection):
            left, right = socket.socketpair()
            left.settimeout(connection.timeout)
            connection.sock = left
            self.sockets.append(right)
            thread = threading.Thread(target=self.handler, args=(right, ('127.0.0.1', 1), self),
                                      daemon=True)
            self.threads.append(thread)
            thread.start()

    monkeypatch.setattr(module, 'ThreadingHTTPServer', SocketpairServer)
    monkeypatch.setattr(http.client.HTTPConnection, 'connect', lambda connection: servers[0].connect(connection))


def test_real_handlers_and_client_over_socketpair(station, socketpair_http):
    service = Service(station.station_file, station=station)
    station.service = service
    service.bind()
    try:
        attached = client.attach(station.station_file)
        assert attached.exit_code == 0
        assert attached.snapshot['service']['pid'] == os.getpid()
        control = credentials(attached)
        assert attached.own(receive=True, **control)['result']['state'] == 'owning-receive'
        assert attached.release(**control)['result']['state'] == 'observing'
        ack = attached.command('unknown', **control)
        with pytest.raises(client.CommandFailed):
            attached.command('unknown', command_id=ack['id'], wait=True, **control)
        with station._lifecycle:
            with pytest.raises(client.CommandFailed) as error:
                attached.release(**control)
        assert error.value.result['reason'] == 'transitioning'
        service.mount('/extra', lambda handler: handler.reply(200, {'mounted': True}))
        assert attached._json('GET', '/extra') == {'mounted': True}
        with pytest.raises(client.ClientError, match='HTTP 400'):
            attached._json('POST', wire.COMMANDS, [])
        assert attached.command('shutdown', wait=True, **control)['state'] == 'done'
        assert station._stop.is_set()
    finally:
        service.close()
        if station.state != 'observing':
            station.release()


def test_previous_incarnation_cursor_gets_gap_and_retained_events():
    events = Events()
    events.publish('tape', {'now': True})
    assert [(e['type'], e['seq']) for e in events.after(100)] == [('gap', 0), ('tape', 1)]
    assert events.seq == 1 and len(events.ring) == 1
    empty = Events()
    assert empty.after(100)[0]['type'] == 'gap'
    assert empty.after(0) == []


@pytest.mark.parametrize('host,expected', [
    ('localhost:8737', 200), ('127.0.0.1:8737', 200), ('[::1]:8737', 200),
    ('attacker.example:8737', 403), ('127.0.0.1:9999', 403), ('localhost', 403),
    ('127.1:8737', 403), ('localhost.evil:8737', 403), ('user@localhost:8737', 403),
])
def test_only_bound_loopback_host_is_accepted(station, socketpair_http, host, expected):
    with running(station) as service:
        conn = http.client.HTTPConnection('127.0.0.1', service.port, timeout=2)
        try:
            conn.request('GET', wire.STATE, headers={'Host': host})
            response = conn.getresponse()
            response.read()
            assert response.status == expected
            conn.request('POST', wire.COMMANDS, json.dumps(command(station, 'stop')),
                         headers={'Host': host, 'Origin': f'http://{host}',
                                  'Content-Type': 'application/json'})
            response = conn.getresponse()
            response.read()
            assert response.status == (202 if expected == 200 else 403)
        finally:
            conn.close()


def test_stale_reader_gap_does_not_break_another_command_wait(station, socketpair_http, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    original = station.own
    def own(**kw):
        entered.set()
        assert finish.wait(3)
        return original(**kw)
    monkeypatch.setattr(station, 'own', own)
    with running(station) as service:
        service.events = Events(capacity=10)
        for _ in range(12):
            service.events.publish('tape', {})
        attached = client.connect(station.station_file, port=service.port)
        control = credentials(attached)
        results, errors = [], []
        def command_wait():
            try:
                results.append(attached.own(receive=True, timeout=3, **control))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=command_wait)
        thread.start()
        stream = attached.events(0, timeout=2, duration=2)
        try:
            assert entered.wait(2)
            assert next(stream)['type'] == 'gap'
            assert next(stream)['type'] == 'tape'
            assert all(event['type'] != 'gap' for _, event in service.events.ring)
            finish.set()
            thread.join(3)
            assert not thread.is_alive() and not errors
            assert results[0]['state'] == 'done'
        finally:
            finish.set()
            stream.close()
            thread.join(3)


def test_shutdown_release_failure_is_terminal(station, socketpair_http, monkeypatch):
    station.own()
    monkeypatch.setattr(station.rig.ptt, 'release', lambda: None)
    with running(station) as service:
        attached = client.connect(station.station_file, port=service.port)
        control = credentials(attached)
        with pytest.raises(client.CommandFailed):
            attached.command('shutdown', wait=True, **control)
        service._worker.join(2)
        assert not service._worker.is_alive()
        assert any(isinstance(call, tuple) and call[0] == 'fatal'
                   and 'shutdown release failed' in call[1] for call in station.calls)
        monkeypatch.setattr(station.rig.ptt, 'release', lambda: setattr(station.rig.ptt, 'released_low', True))
