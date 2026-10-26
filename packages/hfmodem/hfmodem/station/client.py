# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from contextlib import contextmanager, nullcontext, suppress
import errno
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import threading
import tomllib
from urllib.parse import urlencode
import uuid

from . import wire


class ClientError(Exception):
    """Client/command failure (1); attach-specific failures use 4 or 6."""

    exit_code = wire.EXIT_CLIENT_ERROR


class PortConflict(ClientError):
    exit_code = wire.EXIT_PORT_CONFLICT


class Claimed(ClientError):
    exit_code = wire.EXIT_CLAIMED


class LaunchFailed(ClientError):
    exit_code = wire.EXIT_LAUNCH_FAILED


class CommandFailed(ClientError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.get('detail') or result.get('reason') or str(result))


@contextmanager
def _bounded(connection, seconds):
    started = time.monotonic()
    connection.connect()
    sock = connection.sock
    expired = threading.Event()

    def interrupt():
        expired.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    timer = threading.Timer(max(0, seconds - (time.monotonic() - started)), interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        if expired.is_set():
            raise TimeoutError('HTTP request deadline expired')


class Client:
    exit_code = wire.EXIT_ATTACHED

    def __init__(self, station_file, *, port=None, host='127.0.0.1', timeout=2,
                 client_id=None):
        self.station_file = str(Path(station_file).expanduser().resolve())
        try:
            with open(self.station_file, 'rb') as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ClientError(f'cannot read station file {self.station_file}: {exc}') from exc
        if port is None:
            port = config.get('station', {}).get('port', wire.DEFAULT_PORT)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError('station port must be an integer in 1..65535')
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_id = client_id or str(uuid.uuid4())
        self.snapshot = None
        self.log_path = None

    @property
    def url(self):
        return f'http://{self.host}:{self.port}'

    def _connection(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout if timeout is None else timeout)

    def _json(self, method, route, body=None, *, timeout=None):
        connection = self._connection(timeout)
        try:
            with _bounded(connection, self.timeout if timeout is None else timeout):
                headers = {'Accept': 'application/json'}
                if body is not None:
                    body = json.dumps(body).encode('utf-8')
                    headers['Content-Type'] = 'application/json'
                connection.request(method, route, body=body, headers=headers)
                with connection.getresponse() as response:
                    payload = response.read()
                if not 200 <= response.status < 300:
                    raise ClientError(f'{method} {route}: HTTP {response.status}: '
                                      f'{payload.decode("utf-8", errors="replace")}')
                result = json.loads(payload)
                if not isinstance(result, dict):
                    raise ValueError('expected a JSON object')
                return result
        finally:
            connection.close()

    def state(self, *, full=False, timeout=None):
        try:
            snapshot = self._json('GET', wire.STATE + ('?rig=full' if full else ''),
                                  timeout=timeout)
        except ConnectionError:
            # RemoteDisconnected is also an HTTPException. A dropped transport
            # is not evidence of a foreign service, including during startup.
            raise
        except (ClientError, ValueError, http.client.HTTPException) as exc:
            raise PortConflict(f'{self.url} does not identify station {self.station_file}: {exc}') from exc
        service = snapshot.get('service')
        station = snapshot.get('station')
        if not isinstance(service, dict) or not isinstance(station, dict):
            raise PortConflict(f'{self.url}: expected {self.station_file}; foreign service {snapshot!r}')
        if (service.get('name') != wire.SERVICE_NAME
                or type(service.get('schema')) is not int or service['schema'] != wire.SCHEMA
                or station.get('station_file') != self.station_file):
            raise PortConflict(f'{self.url}: expected {wire.SERVICE_NAME} schema {wire.SCHEMA} '
                               f'for {self.station_file}; found {service!r}, {station!r}')
        if (not isinstance(service.get('state'), str)
                or service['state'] not in wire.SERVICE_STATES
                or not isinstance(service.get('incarnation'), str)
                or not service['incarnation']):
            raise PortConflict(f'{self.url}: invalid station identity/state {service!r}')
        rig = snapshot.get('rig')
        if not isinstance(rig, dict) or type(rig.get('retired')) is not bool:
            raise PortConflict(f'{self.url}: invalid rig retirement status {rig!r}')
        self.snapshot = snapshot
        return snapshot

    def events(self, after=None, *, control=None, timeout=30, duration=None):
        query = urlencode({'control': control}) if control is not None else ''
        route = wire.EVENTS + ('?' + query if query else '')
        headers = {'Accept': 'text/event-stream'}
        if after is not None:
            headers['Last-Event-ID'] = str(after)
        connection = self._connection(timeout)
        response = None
        try:
            with _bounded(connection, duration) if duration is not None else nullcontext():
                connection.request('GET', route, headers=headers)
                response = connection.getresponse()
                if response.status != 200 or response.getheader('Content-Type', '').split(';')[0] != 'text/event-stream':
                    raise ClientError(f'GET {route}: expected event stream, HTTP {response.status}')
                data = []
                for raw in response:
                    line = raw.decode('utf-8').rstrip('\r\n')
                    if not line:
                        if data:
                            event = json.loads('\n'.join(data))
                            if not isinstance(event, dict):
                                raise ClientError('event must be a JSON object')
                            yield event
                        data = []
                    elif line.startswith('data:'):
                        data.append(line[5:].removeprefix(' '))
        finally:
            if response is not None:
                response.close()
            connection.close()

    def command(self, kind, args=None, *, command_id=None, control_id=None,
                generation=None, wait=False, timeout=20):
        envelope = {'id': command_id or str(uuid.uuid4()), 'client': self.client_id,
                    'kind': kind, 'args': {} if args is None else args}
        if control_id is not None:
            envelope['control_id'] = control_id
        if generation is not None:
            envelope['generation'] = generation
        body = json.dumps(envelope).encode('utf-8')
        if len(body) > 4096:
            raise ValueError('command exceeds 4096 bytes')
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        deadline = time.monotonic() + timeout
        try:
            ack = self._json('POST', wire.COMMANDS, envelope,
                             timeout=min(self.timeout, timeout))
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise ClientError(f'command {envelope["id"]}: {exc}') from exc
        if (ack.get('id') != envelope['id'] or type(ack.get('seq')) is not int
                or ack['seq'] < 1):
            raise ClientError(f'invalid command acknowledgement: {ack!r}')
        if not wait:
            return ack
        # U6b: ack.seq names the first command event (possibly terminal).
        # Last-Event-ID is exclusive, so include that event in the replay.
        stream = self.events(ack['seq'] - 1, control=control_id,
                             timeout=max(0.001, deadline - time.monotonic()),
                             duration=max(0.001, deadline - time.monotonic()))
        try:
            for event in stream:
                if event.get('type') == 'gap':
                    raise ClientError(f'command {envelope["id"]}: result lost from event ring')
                result = event.get('data', {})
                if (event.get('type') != 'command' or not isinstance(result, dict)
                        or result.get('id') != envelope['id']):
                    if time.monotonic() >= deadline:
                        break
                    continue
                if result.get('state') == 'done':
                    return result
                if result.get('state') in ('refused', 'failed'):
                    holder = result.get('claim_holder')
                    reason = result.get('reason')
                    if kind in ('own', 'fly', 'rtty', 'voice', 'hflink') and (holder or reason == wire.REFUSAL_CLAIMED):
                        raise Claimed(f'{result.get("detail", "rig claimed")}: {holder or reason}')
                    raise CommandFailed(result)
                if time.monotonic() >= deadline:
                    break
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            raise ClientError(f'command {envelope["id"]} timed out or disconnected: {exc}') from exc
        finally:
            # Closing a suspended events() exits _bounded, whose timer may
            # already have fired. Cleanup must not replace a terminal result.
            with suppress(OSError, http.client.HTTPException):
                stream.close()
        raise ClientError(f'command {envelope["id"]}: no terminal result before timeout or EOF')

    def release(self, **kwargs):
        return self.command('release', wait=True, **kwargs)

    def own(self, *, receive=False, **kwargs):
        return self.command('own', {'receive': receive}, wait=True, **kwargs)


def connect(station_file, **kwargs):
    client = Client(station_file, **kwargs)
    try:
        client.state()
    except OSError as exc:
        raise ClientError(f'{client.url}: cannot connect: {exc}') from exc
    return client


def attach(station_file, *, data_root=None, launch_timeout=20, poll_interval=0.1,
           **kwargs):
    """Attach without taking hardware, even to a failed or retired station.

    A starting service must leave startup before success. A failed service is
    still a live service: callers receive its state and rig.retired diagnostics.
    Only connection refusal authorizes launching the PATH-selected hfmodem.
    """
    if launch_timeout <= 0 or poll_interval <= 0:
        raise ValueError('launch_timeout and poll_interval must be positive')
    client = Client(station_file, **kwargs)
    child = None
    try:
        snapshot = client.state(timeout=min(2, client.timeout))
        if snapshot['service']['state'] != 'starting':
            return client
    except OSError as exc:
        if exc.errno != errno.ECONNREFUSED:
            raise PortConflict(f'{client.url}: cannot inspect holder: {exc}') from exc
        root = Path(data_root or os.environ.get('HFMODEM_DATA_DIR', 'operator-data')).expanduser().resolve()
        client.log_path = root / 'station' / f'station-{time.time_ns()}-{uuid.uuid4().hex}.log'
        try:
            # Resolve once, then execute the absolute path. PATH selection is
            # the §1.2 launcher contract, not an import of the modem package.
            executable = shutil.which('hfmodem')
            if executable is None:
                raise FileNotFoundError('hfmodem executable not found on PATH')
            client.log_path.parent.mkdir(parents=True, exist_ok=True)
            with client.log_path.open('ab') as log:
                child = subprocess.Popen([str(Path(executable).resolve()), 'station', client.station_file, '--detach'],
                                         start_new_session=True, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=log,
                                         env=dict(os.environ, HFMODEM_DATA_DIR=str(root)))
        except OSError as exc:
            raise LaunchFailed(f'launch failed: {exc}; log: {client.log_path}') from exc
    deadline = time.monotonic() + launch_timeout
    while time.monotonic() < deadline:
        code = child.poll() if child is not None else None
        if code is not None and code != wire.EXIT_ALREADY_RUNNING:
            raise LaunchFailed(f'station launcher exited {code}; log: {client.log_path}')
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            snapshot = client.state(timeout=min(2, client.timeout, remaining))
            if snapshot['service']['state'] != 'starting':
                return client
        except OSError as exc:
            if exc.errno != errno.ECONNREFUSED and not isinstance(exc, (ConnectionError, TimeoutError)):
                raise LaunchFailed(f'waiting for station: {exc}; log: {client.log_path}') from exc
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
    raise LaunchFailed(f'station launch timed out; log: {client.log_path}')
