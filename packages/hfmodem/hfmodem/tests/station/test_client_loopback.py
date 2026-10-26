# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Real TCP/HTTP regressions; restricted runners explicitly skip bind failures."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import errno
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import sys
import threading
import time

import pytest

from hfmodem.station import client, wire


@pytest.fixture
def station(tmp_path):
    path = tmp_path / 'station.toml'
    path.write_text('[station]\ntransmit = false\n')
    return path


@contextmanager
def bound_socket():
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', 0))
        except OSError as exc:
            pytest.skip(f'loopback bind unavailable in this environment: {exc}')
        yield sock


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def begin_events(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()

    def event(self, event):
        self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
        self.wfile.flush()


@contextmanager
def server(handler):
    try:
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    except OSError as exc:
        pytest.skip(f'loopback bind unavailable in this environment: {exc}')
    thread = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive(), 'HTTP server did not stop'


def snapshot(station, state='observing'):
    return {'seq': 10,
            'service': {'name': wire.SERVICE_NAME, 'schema': wire.SCHEMA,
                        'incarnation': 'loopback', 'state': state},
            'station': {'station_file': str(station.resolve())},
            'rig': {'retired': False}}


def test_real_econnrefused(station):
    # macOS drops a SYN to a bound, non-listening socket; only a released port refuses.
    with bound_socket() as sock:
        port = sock.getsockname()[1]
    attached = client.Client(station, port=port)
    with pytest.raises(ConnectionRefusedError) as error:
        attached.state()
    assert error.value.errno == errno.ECONNREFUSED
    with pytest.raises(client.ClientError) as error:
        client.connect(station, port=attached.port)
    assert error.value.exit_code == 1


def test_incremental_sse_with_keepalives(station):
    finish = threading.Event()
    requests = []

    class Incremental(Handler):
        def do_GET(self):
            requests.append((self.path, self.headers.get('Last-Event-ID')))
            self.begin_events()
            self.wfile.write(b': keepalive\n\n')
            self.event({'seq': 11, 'type': 'tape', 'data': {'text': 'first'}})
            try:
                until = time.monotonic() + 3
                while not finish.wait(0.01) and time.monotonic() < until:
                    self.wfile.write(b': keepalive\n\n')
                    self.wfile.flush()
                self.event({'seq': 12, 'type': 'tape', 'data': {'text': 'second'}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    with server(Incremental) as port:
        stream = client.Client(station, port=port).events(10, timeout=2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(next, stream)
            try:
                assert pending.result(timeout=2)['data']['text'] == 'first'
                pending = pool.submit(next, stream)
                with pytest.raises(FutureTimeout):
                    pending.result(timeout=0.05)
                finish.set()
                assert pending.result(timeout=2)['data']['text'] == 'second'
                assert list(stream) == []
            finally:
                finish.set()
        stream.close()
    assert requests == [('/events', '10')]


def test_bounded_interrupts_real_blocked_recv(station):
    finish = threading.Event()
    reading = threading.Event()

    class Hung(Handler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '999')
            self.end_headers()
            self.wfile.write(b'{')
            self.wfile.flush()
            reading.set()
            finish.wait(3)

    with server(Hung) as port:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match='deadline expired'):
                with client._bounded(connection, 0.15):
                    connection.request('GET', '/state')
                    with connection.getresponse() as response:
                        response.read()
            assert reading.is_set()
            assert time.monotonic() - started < 1.5, 'waited for socket timeout instead of deadline'
        finally:
            finish.set()
            connection.close()


def test_drop_once_during_starting_then_observing(station, monkeypatch):
    calls = []

    class Drops(Handler):
        def do_GET(self):
            calls.append(self.path)
            if len(calls) == 2:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return  # Real RemoteDisconnected, not an HTTP error response.
            self.reply(snapshot(station, 'starting' if len(calls) == 1 else 'observing'))

    monkeypatch.setattr(client.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with server(Drops) as port:
        attached = client.attach(station, port=port, launch_timeout=2, poll_interval=0.01)
        assert attached.snapshot['service']['state'] == 'observing'
        assert attached.exit_code == 0
        assert attached.log_path is None
    assert calls == ['/state'] * 3


@pytest.mark.parametrize('state', ['done', 'refused'])
def test_terminal_event_at_ack_sequence_is_replayed(station, state):
    replay = []

    class Commands(Handler):
        def do_POST(self):
            envelope = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.reply({'id': envelope['id'], 'seq': 11}, status=202)

        def do_GET(self):
            after = int(self.headers['Last-Event-ID'])
            replay.append(after)
            self.begin_events()
            if after < 11:
                self.event({'seq': 11, 'type': 'command',
                            'data': {'id': 'cmd', 'state': state, 'detail': 'terminal at ack'}})

    with server(Commands) as port:
        attached = client.Client(station, port=port)
        if state == 'done':
            assert attached.release(command_id='cmd')['state'] == 'done'
        else:
            with pytest.raises(client.CommandFailed, match='terminal at ack'):
                attached.release(command_id='cmd')
    assert replay == [10]


# The executable is a real independent process: one binds/serves, one exits 3.
FAKE_HFMODEM = '''
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tomllib

assert sys.argv[1] == 'station' and sys.argv[3:] == ['--detach']
station = str(Path(sys.argv[2]).resolve())
with open(station, 'rb') as stream:
    port = tomllib.load(stream)['station']['port']

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        payload = json.dumps({
            'service': {'name': 'hfmodem station', 'schema': 2,
                        'incarnation': str(os.getpid()), 'state': 'observing'},
            'station': {'station_file': station}, 'rig': {'retired': False},
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

try:
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
except OSError as exc:
    if exc.errno == errno.EADDRINUSE:
        print('lost bind race', flush=True)
        sys.exit(3)
    raise
print('serving', flush=True)
server.serve_forever(poll_interval=0.01)
'''


def test_concurrent_attaches_launch_one_winner_and_exit_three_loser(station, tmp_path, monkeypatch):
    with bound_socket() as sock:
        port = sock.getsockname()[1]
    station.write_text(f'[station]\nport = {port}\ntransmit = false\n')
    executable = tmp_path / 'hfmodem'
    executable.write_text(f'#!{sys.executable}\n' + FAKE_HFMODEM)
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    original_state = client.Client.state
    original_popen = client.subprocess.Popen
    first_probe = threading.local()
    refused_together = threading.Barrier(2)
    children = []
    launches = []
    lock = threading.Lock()

    def state(self, **kwargs):
        try:
            return original_state(self, **kwargs)
        except ConnectionRefusedError:
            if not getattr(first_probe, 'done', False):
                first_probe.done = True
                refused_together.wait(timeout=3)
            raise

    def popen(argv, **kwargs):
        child = original_popen(argv, **kwargs)
        with lock:
            children.append(child)
            launches.append(argv)
        return child

    monkeypatch.setattr(client.Client, 'state', state)
    monkeypatch.setattr(client.subprocess, 'Popen', popen)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(client.attach, station, data_root=tmp_path,
                                   launch_timeout=5, poll_interval=0.01) for _ in range(2)]
            attached = [future.result(timeout=8) for future in futures]
        assert len(children) == 2
        assert all(item.exit_code == 0 for item in attached)
        assert len({item.snapshot['service']['incarnation'] for item in attached}) == 1
        until = time.monotonic() + 3
        while all(child.poll() is None for child in children) and time.monotonic() < until:
            time.sleep(0.01)
        assert sorted(child.poll() for child in children if child.poll() is not None) == [3]
        assert sum(child.poll() is None for child in children) == 1
        assert all(argv == [str(executable), 'station', str(station), '--detach'] for argv in launches)
        assert len({item.log_path for item in attached}) == 2
        logs = [item.log_path.read_text() for item in attached]
        assert any('lost bind race' in log for log in logs)
        assert any('serving' in log for log in logs)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except client.subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
