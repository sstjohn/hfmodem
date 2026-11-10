# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import contextmanager
import errno
from http.server import BaseHTTPRequestHandler
import http.client
import io
import json
from pathlib import Path
import re
import socket
import threading
from types import SimpleNamespace

import pytest

from hfmodem.station import client, wire


@pytest.fixture
def station(tmp_path):
    path = tmp_path / 'station.toml'
    path.write_text('[station]\ntransmit = false\n')
    return path


@contextmanager
def server(station, *, snapshot=None, stream=None, ack=None, status=200,
           content_type='application/json'):
    requests = []
    responses = []
    state = snapshot if snapshot is not None else {
        'schema': wire.SCHEMA, 'seq': 10,
        'service': {'name': wire.SERVICE_NAME, 'schema': wire.SCHEMA,
                    'incarnation': 'fake-incarnation', 'state': 'observing',
                    'claim_holder': ['legacy pid=123']},
        'station': {'station_file': str(station.resolve())}, 'rig': {'retired': False},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.command, self.path, dict(self.headers), None))
            is_events = self.path.startswith(wire.EVENTS)
            payload = (stream or '').encode() if is_events else json.dumps(state).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'text/event-stream' if is_events else content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.command, self.path, dict(self.headers), body))
            result = ack or {'id': body['id'], 'seq': 11}
            payload = json.dumps(result).encode()
            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    class MemorySocket:
        def __init__(self, incoming=b''):
            self.incoming = incoming
            self.outgoing = bytearray()

        def makefile(self, *args, **kwargs):
            return io.BytesIO(self.incoming)

        def sendall(self, data):
            self.outgoing.extend(data)

        def shutdown(self, how):
            pass

        def close(self):
            pass

    original = http.client.HTTPConnection

    class Connection(original):
        def connect(self):
            self.sock = MemorySocket()

        def getresponse(self):
            request = MemorySocket(bytes(self.sock.outgoing))
            Handler(request, ('127.0.0.1', 1), None)
            response = http.client.HTTPResponse(MemorySocket(bytes(request.outgoing)))
            response.begin()
            responses.append(response)
            return response

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(client.http.client, 'HTTPConnection', Connection)
        yield SimpleNamespace(port=wire.DEFAULT_PORT, requests=requests,
                              responses=responses, snapshot=state)


def test_connect_canonical_file_and_observing_claim_holder(station, tmp_path, monkeypatch):
    alias = tmp_path / 'alias.toml'
    alias.symlink_to(station)
    monkeypatch.setattr(client.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with server(station) as fake:
        attached = client.attach(alias, port=fake.port)
        assert attached.exit_code == 0
        assert attached.snapshot['service']['claim_holder'] == ['legacy pid=123']
        assert attached.snapshot['service']['incarnation'] == 'fake-incarnation'
        assert attached.snapshot['rig']['retired'] is False
        assert attached.state(full=True)['seq'] == 10
        assert fake.requests[-1][1] == '/state?rig=full'


@pytest.mark.parametrize('field,value', [
    ('name', 'foreign'), ('schema', 1), ('schema', True),
    ('state', 'bogus'), ('state', []), ('incarnation', ''),
])
def test_foreign_service_refuses_without_launch(station, monkeypatch, field, value):
    monkeypatch.setattr(client.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with server(station) as fake:
        fake.snapshot['service'][field] = value
        with pytest.raises(client.PortConflict) as error:
            client.attach(station, port=fake.port)
        assert error.value.exit_code == 4


def test_different_station_names_both_identities(station):
    with server(station) as fake:
        fake.snapshot['station']['station_file'] = '/another/station.toml'
        with pytest.raises(client.PortConflict) as error:
            client.connect(station, port=fake.port)
        assert str(station) in str(error.value)
        assert '/another/station.toml' in str(error.value)


@pytest.mark.parametrize('snapshot', [[], {}, {'service': None, 'station': None}])
def test_malformed_snapshot_is_conflict(station, snapshot):
    with server(station, snapshot=snapshot) as fake:
        with pytest.raises(client.PortConflict):
            client.connect(station, port=fake.port)


def test_http_error_is_conflict(station):
    with server(station, status=404) as fake:
        with pytest.raises(client.PortConflict):
            client.connect(station, port=fake.port)


def test_port_config_and_defaults(station):
    assert client.Client(station).port == 8737
    station.write_text('[station]\nport = 9012\n[rig]\nport = 4532\n')
    assert client.Client(station).port == 9012
    station.write_text('[station]\nport = true\n')
    with pytest.raises(ValueError):
        client.Client(station)


def test_sse_comments_multiline_gap_and_replay(station):
    stream = (': keepalive\n\nid: 11\nevent: gap\ndata: {"type":"gap",\n'
              'data: "data":{"from":1,"to":10}}\n\n'
              'id: 12\nevent: tape\ndata: {"seq":12,"type":"tape","data":{"text":"héllo"}}\n\n'
              'data: {"incomplete":true}\n')
    with server(station, stream=stream) as fake:
        attached = client.connect(station, port=fake.port)
        events = list(attached.events(1, control='id /?'))
        assert [event['type'] for event in events] == ['gap', 'tape']
        assert events[1]['data']['text'] == 'héllo'
        request = fake.requests[-1]
        assert request[1] == '/events?control=id+%2F%3F'
        assert request[2]['Last-Event-ID'] == '1'


def test_command_envelope_and_idempotency_id(station):
    with server(station) as fake:
        attached = client.connect(station, port=fake.port, client_id='console')
        ack = attached.command('stop', command_id='repeat', control_id='controller', generation=4)
        assert ack == {'id': 'repeat', 'seq': 11}
        request = fake.requests[-1]
        assert request[1] == '/commands'
        assert request[2]['Content-Type'] == 'application/json'
        assert request[3] == {'id': 'repeat', 'client': 'console', 'kind': 'stop', 'args': {},
                              'control_id': 'controller', 'generation': 4}
        attached.command('stop', command_id='repeat')
        assert fake.requests[-1][3]['id'] == 'repeat'
        before = len(fake.requests)
        with pytest.raises(ValueError, match='4096'):
            attached.command('fly', {'text': 'x' * 5000})
        assert len(fake.requests) == before


def command_event(state='done', **extra):
    result = {'type': 'command', 'data': {'id': 'own-id', 'state': state, **extra}}
    return 'data: ' + json.dumps(result) + '\n\n'


@pytest.mark.parametrize('verb', ['own', 'release'])
def test_ownership_commands_wait_for_matching_terminal_result(station, verb):
    stream = (': keepalive\n\n' + command_event('accepted')
              + command_event('running') + command_event(result={'ok': True}))
    with server(station, stream=stream) as fake:
        attached = client.connect(station, port=fake.port)
        result = getattr(attached, verb)(command_id='own-id')
        assert result['result'] == {'ok': True}
        post = next(request for request in fake.requests if request[0] == 'POST')
        assert post[3]['kind'] == verb
        assert fake.requests[-1][2]['Last-Event-ID'] == '10'


def test_receive_ownership(station):
    with server(station, stream=command_event()) as fake:
        client.connect(station, port=fake.port).own(receive=True, command_id='own-id')
        assert fake.requests[1][3]['args'] == {'receive': True}


def test_claim_refusal_maps_to_five_and_names_holder(station):
    with server(station, stream=command_event('refused', reason='claimed',
                                             claim_holder=['legacy pid=321'])) as fake:
        with pytest.raises(client.Claimed, match='legacy pid=321') as error:
            client.connect(station, port=fake.port).own(command_id='own-id')
        assert error.value.exit_code == 5


def test_other_refusal_preserves_detail(station):
    with server(station, stream=command_event('refused', reason='not-controller',
                                             detail='take control first')) as fake:
        with pytest.raises(client.CommandFailed, match='take control first'):
            client.connect(station, port=fake.port).own(command_id='own-id')


@pytest.mark.parametrize('stream', ['', 'data: {"type":"gap","data":{}}\n\n',
                                  command_event('done').replace('own-id', 'other-id')])
def test_missing_or_lost_command_result_fails(station, stream):
    with server(station, stream=stream) as fake:
        with pytest.raises(client.ClientError):
            client.connect(station, port=fake.port).release(command_id='own-id')


def launch_setup(monkeypatch, states, code=None):
    attempts = []
    launches = []
    snapshots = iter(states)

    def state(self, **kwargs):
        attempts.append(kwargs.get('timeout'))
        result = next(snapshots)
        if isinstance(result, Exception):
            raise result
        self.snapshot = result
        return result

    def popen(argv, **kwargs):
        launches.append((argv, kwargs))
        kwargs['stdout'].write(b'fake launch\n')
        return SimpleNamespace(poll=lambda: code)

    monkeypatch.setattr(client.Client, 'state', state)
    monkeypatch.setattr(client.subprocess, 'Popen', popen)
    monkeypatch.setattr(client.shutil, 'which', lambda name: '/fake/bin/hfmodem')
    return attempts, launches


def refused():
    return ConnectionRefusedError(errno.ECONNREFUSED, 'fake refusal')


def test_attach_allows_windows_refusal_before_launch_and_while_waiting(station, tmp_path, monkeypatch):
    attempts = []
    launches = []
    original_state = client.Client.state

    def connect(sock, address):
        attempts.append(sock.gettimeout())
        if sock.gettimeout() <= 2.03:
            raise TimeoutError('Windows is still retrying the SYN')
        raise refused()

    def state(self, **kwargs):
        if len(attempts) < 2:
            return original_state(self, **kwargs)
        return {'service': {'state': 'observing'}}

    def popen(*args, **kwargs):
        launches.append(args)
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(socket.socket, 'connect', connect)
    monkeypatch.setattr(client.Client, 'state', state)
    monkeypatch.setattr(client.subprocess, 'Popen', popen)
    monkeypatch.setattr(client.shutil, 'which', lambda name: '/fake/bin/hfmodem')
    attached = client.attach(station, data_root=tmp_path, timeout=0.1,
                             launch_timeout=1, poll_interval=0.001)
    assert attached.exit_code == 0
    assert len(launches) == 1
    assert len(attempts) == 2 and all(timeout > 2.03 for timeout in attempts)


@pytest.mark.parametrize('operation', ['state', 'connect'])
def test_client_allows_windows_refusal_with_short_timeout(station, monkeypatch, operation):
    def connect(sock, address):
        if sock.gettimeout() <= 2.03:
            raise TimeoutError('Windows is still retrying the SYN')
        raise refused()

    monkeypatch.setattr(socket.socket, 'connect', connect)
    if operation == 'state':
        with pytest.raises(ConnectionRefusedError):
            client.Client(station, timeout=0.1).state()
    else:
        with pytest.raises(client.ClientError) as error:
            client.connect(station, timeout=0.1)
        assert isinstance(error.value.__cause__, ConnectionRefusedError)


@pytest.mark.parametrize('host,timeout,connect_timeout', [
    ('127.0.0.1', 0.1, 3), ('127.0.0.2', 2, 3), ('::1', 2, 3),
    ('localhost', 2, 3), ('127.0.0.1', 10, 10), ('127.0.0.1', None, None),
    ('192.0.2.1', 0.1, 0.1), ('example.invalid', 2, 2),
])
def test_connect_allowance_preserves_response_timeout(monkeypatch, host, timeout, connect_timeout):
    attempts = []
    response_timeouts = []
    sock = SimpleNamespace(settimeout=response_timeouts.append)
    source = ('127.0.0.1', 0)

    def connect(address, **kwargs):
        attempts.append((address, kwargs))
        return sock

    monkeypatch.setattr(socket, 'create_connection', connect)
    assert wire.create_connection((host, 8737), timeout, source) is sock
    assert attempts == [((host, 8737), {'timeout': connect_timeout, 'source_address': source})]
    assert response_timeouts == [timeout]


def test_connect_time_counts_against_the_response_deadline(monkeypatch):
    clock = [0.0]
    intervals = []
    sock = SimpleNamespace()

    def connect():
        clock[0] += 2.5

    closed = []
    connection = SimpleNamespace(connect=connect, sock=sock, close=lambda: closed.append(True))
    timer = SimpleNamespace(start=lambda: None, cancel=lambda: None)
    monkeypatch.setattr(client, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(client.threading, 'Timer',
                        lambda seconds, callback: intervals.append(seconds) or timer)
    with client._bounded(connection, 3):
        assert intervals == [0.5]
    entered = []
    with pytest.raises(TimeoutError, match='deadline expired'):
        with client._bounded(connection, 1):
            entered.append(True)
    assert entered == [] and closed == [True] and intervals == [0.5]


@pytest.mark.parametrize('code', [None, 3])
def test_launch_and_bind_race_wait_until_observing(station, tmp_path, monkeypatch, code):
    starting = {'service': {'state': 'starting'}}
    observing = {'service': {'state': 'observing', 'claim_holder': ['legacy']}}
    attempts, launches = launch_setup(monkeypatch, [refused(), refused(), starting, observing], code)
    attached = client.attach(station, data_root=tmp_path, poll_interval=0.001, launch_timeout=1)
    argv, options = launches[0]
    assert argv == [str(Path('/fake/bin/hfmodem').resolve()),
                    'station', str(station.resolve()), '--detach']
    if client.sys.platform == 'win32':
        assert options['creationflags'] == 0x08000200
    else:
        assert options['start_new_session'] is True
    assert options['stdin'] == client.subprocess.DEVNULL
    assert options['stdout'] is options['stderr']
    assert attached.log_path.parent == tmp_path / 'station'
    assert attached.log_path.read_text() == 'fake launch\n'
    assert attached.exit_code == 0
    assert all(0 < value <= 1 for value in attempts[1:])


@pytest.mark.parametrize('own_venv', [True, False])
def test_windows_station_launch_keeps_path_selection_and_isolates_console(
        station, tmp_path, monkeypatch, own_venv):
    _, launches = launch_setup(monkeypatch, [refused(), {'service': {'state': 'observing'}}])
    executable = tmp_path / ('Scripts' if own_venv else 'foreign') / 'hfmodem.exe'
    monkeypatch.setattr(client.sys, 'platform', 'win32')
    monkeypatch.setattr(client.sys, 'executable', str(tmp_path / 'Scripts' / 'python.exe'))
    monkeypatch.setattr(client.sys, '_base_executable', 'base-python.exe')
    monkeypatch.setattr(client.subprocess, 'CREATE_NO_WINDOW', 0x08000000, raising=False)
    monkeypatch.setattr(client.subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200, raising=False)
    monkeypatch.setattr(client.shutil, 'which', lambda name: str(executable))
    client.attach(station, data_root=tmp_path)
    command, options = launches.pop()
    tail = ['station', str(station.resolve()), '--detach']
    assert command == (['base-python.exe', '-m', 'hfmodem', *tail] if own_venv
                       else [str(executable.resolve()), *tail])
    assert options['creationflags'] == 0x08000200
    assert 'start_new_session' not in options
    assert options['env']['HFMODEM_DATA_DIR'] == str(tmp_path)
    if own_venv:
        assert options['env']['__PYVENV_LAUNCHER__'] == client.sys.executable


@pytest.mark.parametrize('code', [0, 1, 5])
def test_launcher_exit_other_than_three_is_failure(station, tmp_path, monkeypatch, code):
    launch_setup(monkeypatch, [refused()], code)
    with pytest.raises(client.LaunchFailed, match=f'exited {code}.*log:') as error:
        client.attach(station, data_root=tmp_path)
    assert error.value.exit_code == 6


def test_launch_timeout(station, tmp_path, monkeypatch):
    launch_setup(monkeypatch, [refused()] * 100)
    with pytest.raises(client.LaunchFailed, match='timed out.*log:'):
        client.attach(station, data_root=tmp_path, launch_timeout=0.01, poll_interval=0.001)


def test_spawn_error_names_log(station, tmp_path, monkeypatch):
    launch_setup(monkeypatch, [refused()])
    def missing(*args, **kwargs):
        raise FileNotFoundError('hfmodem')
    monkeypatch.setattr(client.subprocess, 'Popen', missing)
    with pytest.raises(client.LaunchFailed, match='launch failed.*log:'):
        client.attach(station, data_root=tmp_path)


def test_conflicting_winner_is_never_relaunched(station, tmp_path, monkeypatch):
    _, launches = launch_setup(monkeypatch, [refused(), client.PortConflict('another station')], 3)
    with pytest.raises(client.PortConflict, match='another station'):
        client.attach(station, data_root=tmp_path)
    assert len(launches) == 1


def test_initial_timeout_is_not_permission_to_launch(station, monkeypatch):
    launch_setup(monkeypatch, [TimeoutError('silent listener')])
    monkeypatch.setattr(client.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(client.PortConflict):
        client.attach(station)


def test_silent_http_attempt_is_bounded(station, monkeypatch):
    stopped = threading.Event()

    class SilentSocket:
        def shutdown(self, how):
            stopped.set()

        def close(self):
            pass

    class SilentConnection:
        sock = SilentSocket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            assert stopped.wait(1), 'request deadline did not interrupt the socket'
            raise TimeoutError('deadline')

        def close(self):
            pass

    attached = client.Client(station, timeout=0.02)
    monkeypatch.setattr(attached, '_connection', lambda timeout: SilentConnection())
    with pytest.raises(TimeoutError):
        attached.state()


def test_read_deadline_survives_connection_detaching_socket(station, monkeypatch):
    stopped = threading.Event()

    class Socket:
        def shutdown(self, how):
            stopped.set()

        def close(self):
            pass

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            assert stopped.wait(1), 'read deadline lost the detached socket'
            return b'{}'

    class Connection:
        def connect(self):
            self.sock = Socket()

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            self.sock = None
            return Response()

        def close(self):
            pass

    attached = client.Client(station, timeout=0.02)
    monkeypatch.setattr(attached, '_connection', lambda timeout: Connection())
    with pytest.raises(TimeoutError):
        attached.state()


def test_command_deadline_is_not_extended_by_keepalives(station, monkeypatch):
    stopped = threading.Event()
    response_closed = threading.Event()

    class Socket:
        def shutdown(self, how):
            stopped.set()

        def close(self):
            pass

    class Response:
        status = 200

        def getheader(self, name, default):
            return 'text/event-stream'

        def __iter__(self):
            while not stopped.wait(0.001):
                yield b': keepalive\n'

        def close(self):
            response_closed.set()

    class Connection:
        def connect(self):
            self.sock = Socket()

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    attached = client.Client(station)
    monkeypatch.setattr(attached, '_connection', lambda timeout: Connection())
    monkeypatch.setattr(attached, '_json', lambda *a, **k: {'id': 'own-id', 'seq': 1})
    with pytest.raises(client.ClientError, match='timed out'):
        attached.own(command_id='own-id', timeout=0.02)
    assert response_closed.is_set()


def test_closed_sse_iterator_closes_response(station):
    with server(station, stream=command_event()) as fake:
        attached = client.connect(station, port=fake.port)
        stream = attached.events()
        assert next(stream)['type'] == 'command'
        stream.close()
        assert fake.responses[-1].isclosed()


@pytest.mark.parametrize('drop', [http.client.RemoteDisconnected('dropped'),
                                  ConnectionResetError(errno.ECONNRESET, 'reset')])
def test_state_preserves_transport_drop(station, monkeypatch, drop):
    attached = client.Client(station)

    def dropped(*args, **kwargs):
        raise drop

    monkeypatch.setattr(attached, '_json', dropped)
    with pytest.raises(type(drop)) as error:
        attached.state()
    assert error.value is drop


@pytest.mark.parametrize('drop', [http.client.RemoteDisconnected('dropped'),
                                  ConnectionResetError(errno.ECONNRESET, 'reset')])
def test_attach_poll_retries_drop_once(station, tmp_path, monkeypatch, drop):
    _, launches = launch_setup(monkeypatch, [refused(), drop,
                                           {'service': {'state': 'observing'}}])
    assert client.attach(station, data_root=tmp_path, poll_interval=0.001).exit_code == 0
    assert len(launches) == 1


def test_existing_starting_station_waits_without_spawn(station, monkeypatch):
    _, launches = launch_setup(monkeypatch, [
        {'service': {'state': 'starting'}},
        http.client.RemoteDisconnected('dropped'),
        {'service': {'state': 'observing'}},
    ])
    attached = client.attach(station, poll_interval=0.001)
    assert attached.snapshot['service']['state'] == 'observing'
    assert attached.log_path is None
    assert not launches


def test_existing_starting_station_has_startup_timeout(station, monkeypatch):
    _, launches = launch_setup(monkeypatch, [{'service': {'state': 'starting'}}] * 100)
    with pytest.raises(client.LaunchFailed, match='timed out'):
        client.attach(station, launch_timeout=0.01, poll_interval=0.001)
    assert not launches


@pytest.mark.parametrize('initial', [True, False])
def test_failed_station_is_attachable_with_retirement_diagnostics(station, tmp_path, monkeypatch, initial):
    with server(station) as fake:
        fake.snapshot['service']['state'] = 'failed'
        fake.snapshot['rig'] = {'retired': True, 'retired_why': 'line unconfirmed'}
        if not initial:
            original = client.Client.state
            calls = []

            def state(self, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise refused()
                return original(self, **kwargs)

            monkeypatch.setattr(client.Client, 'state', state)
            monkeypatch.setattr(client.shutil, 'which', lambda name: '/fake/bin/hfmodem')
            monkeypatch.setattr(client.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(poll=lambda: 3))
        else:
            monkeypatch.setattr(client.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
        attached = client.attach(station, data_root=tmp_path, port=fake.port)
        assert attached.exit_code == 0
        assert attached.snapshot['rig']['retired'] is True
        assert attached.snapshot['rig']['retired_why'] == 'line unconfirmed'


@pytest.mark.parametrize('rig', [None, {}, {'retired': 1}, {'retired': 'false'}])
def test_retirement_status_is_validated(station, rig):
    with server(station) as fake:
        fake.snapshot['rig'] = rig
        with pytest.raises(client.PortConflict, match='retirement'):
            client.attach(station, port=fake.port)


@pytest.mark.parametrize('port', [None, 8737])
def test_missing_station_file_has_client_error(tmp_path, port):
    missing = tmp_path / 'missing.toml'
    with pytest.raises(client.ClientError, match=re.escape(str(missing))) as error:
        client.attach(missing, port=port)
    assert error.value.exit_code == 1


def test_invalid_station_toml_has_client_error(station):
    station.write_text('[station')
    with pytest.raises(client.ClientError, match='cannot read station file'):
        client.Client(station)


def test_missing_executable_names_path_and_log(station, tmp_path, monkeypatch):
    _, launches = launch_setup(monkeypatch, [refused()])
    monkeypatch.setattr(client.shutil, 'which', lambda name: None)
    with pytest.raises(client.LaunchFailed, match='not found on PATH.*log:'):
        client.attach(station, data_root=tmp_path)
    assert not launches


def test_generic_command_errors_use_one(station):
    with server(station, stream=command_event('failed', detail='command failed')) as fake:
        with pytest.raises(client.CommandFailed) as error:
            client.connect(station, port=fake.port).release(command_id='own-id')
        assert error.value.exit_code == 1
    assert client.ClientError.exit_code == 1
    assert client.LaunchFailed.exit_code == 6


@pytest.mark.parametrize('seq', [0, -1, True, '11'])
def test_ack_sequence_must_name_an_event(station, seq):
    with server(station, ack={'id': 'own-id', 'seq': seq}) as fake:
        with pytest.raises(client.ClientError, match='acknowledgement'):
            client.connect(station, port=fake.port).release(command_id='own-id')


@pytest.mark.parametrize('state', ['done', 'failed', 'refused'])
@pytest.mark.parametrize('close_error', [TimeoutError('deadline'), http.client.HTTPException('close')])
def test_terminal_at_deadline_survives_stream_close(station, monkeypatch, state, close_error):
    now = [0]
    closed = []
    attached = client.Client(station)
    monkeypatch.setattr(client.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(attached, '_json', lambda *a, **k: {'id': 'own-id', 'seq': 11})

    class Stream:
        def __iter__(self):
            now[0] = 1  # Complete event delivered exactly at the deadline.
            yield {'seq': 11, 'type': 'command',
                   'data': {'id': 'own-id', 'state': state, 'detail': 'terminal'}}

        def close(self):
            closed.append(True)
            raise close_error

    def events(after, **kwargs):
        assert after == 10  # Terminal at ack.seq must be replayed.
        return Stream()

    monkeypatch.setattr(attached, 'events', events)
    if state == 'done':
        assert attached.release(command_id='own-id', timeout=1)['state'] == 'done'
    else:
        with pytest.raises(client.CommandFailed, match='terminal'):
            attached.release(command_id='own-id', timeout=1)
    assert closed == [True]


@pytest.mark.parametrize('source', ['argument', 'environment'])
def test_attach_passes_resolved_data_root_to_child(station, tmp_path, monkeypatch, source):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    _, launches = launch_setup(monkeypatch, [refused(), {'service': {'state': 'observing'}}])
    root = '~/records/../data'
    if source == 'environment':
        monkeypatch.setenv('HFMODEM_DATA_DIR', root)
        attached = client.attach(station)
    else:
        monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'unused'))
        attached = client.attach(station, data_root=root)
    expected = tmp_path / 'data'
    assert launches[0][1]['env']['HFMODEM_DATA_DIR'] == str(expected)
    assert attached.log_path.parent == expected / 'station'
