# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import nullcontext
from datetime import datetime, timezone
import errno
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
import tomllib
from urllib.parse import parse_qs, urlsplit
import uuid

from hfmodem.core import claim
from hfmodem.core.operations import LeaseTimeout
from . import wire
from .sessions import Refused


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class Events:
    def __init__(self, *, capacity=10000, age=3600, clock=time.monotonic):
        self.capacity, self.age, self.clock = capacity, age, clock
        self.condition = threading.Condition(threading.RLock())
        self.ring = deque()
        self.seq = 0

    def publish(self, type, payload):
        if type not in wire.EVENT_TYPES:
            raise ValueError(f'unknown event type: {type}')
        with self.condition:
            self.seq += 1
            event = dict(seq=self.seq, utc=utc(), type=type, data=payload)
            self.ring.append((self.clock(), event))
            self._trim()
            self.condition.notify_all()
            return event

    def _trim(self):
        while self.ring and (len(self.ring) > self.capacity
                             or self.ring[0][0] <= self.clock() - self.age):
            self.ring.popleft()

    def after(self, seq):
        with self.condition:
            self._trim()
            oldest = self.ring[0][1]['seq'] if self.ring else self.seq + 1
            if seq < oldest - 1 or seq > self.seq:
                gap = dict(seq=oldest - 1, utc=utc(), type='gap',
                           data={'from': seq, 'to': oldest})
                return [gap, *(event for _, event in self.ring)]
            return [event for _, event in self.ring if event['seq'] > seq]


class BindConflict(RuntimeError):
    def __init__(self, detail, exit_code):
        super().__init__(detail)
        self.exit_code = exit_code


def endpoint(station_file):
    path = Path(station_file).expanduser().resolve()
    with path.open('rb') as stream:
        port = tomllib.load(stream).get('station', {}).get('port', wire.DEFAULT_PORT)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('station port must be an integer in 1..65535')
    return path, port


class Service:
    def __init__(self, station_file, *, station=None, port=None):
        self.station_file = str(Path(station_file).expanduser().resolve())
        self.port = endpoint(station_file)[1] if port is None else port
        self._station = None
        self.events = Events()
        self.incarnation = str(uuid.uuid4())
        self.started_utc = utc()
        self.commands = queue.Queue(64)
        self._ids = OrderedDict()
        self._submit_lock = threading.Lock()
        self._closed = threading.Event()
        self._mounts = {}
        self._shutdown_seq = None
        self._delivered = 0
        self._http = self._http_thread = self._worker = None
        self.station = station

    @property
    def station(self):
        return self._station

    @station.setter
    def station(self, station):
        if station is not None:
            station.sessions.publish = lambda type, payload: self.events.publish(type, payload)
            station.sessions.provenance = dict(holder_pid=os.getpid(),
                station_url=f'http://127.0.0.1:{self.port}', incarnation=self.incarnation)
            if self._http is not None:
                station.sessions.recover()
        self._station = station

    def mount(self, prefix, handler):
        if not prefix.startswith('/'):
            raise ValueError('mount prefix must be absolute')
        self._mounts[prefix] = handler

    def snapshot(self):
        with (self.station.sessions.lock if self.station is not None else nullcontext()), self.events.condition:
            state = self.station.snapshot() if self.station is not None else {
                'service': {'state': 'starting', 'ready': False, 'failed': []},
                'rig': {'source': 'none', 'retired': False}}
            state['service'].update(name=wire.SERVICE_NAME, schema=wire.SCHEMA, pid=os.getpid(),
                                    incarnation=self.incarnation, started_utc=self.started_utc)
            state.setdefault('station', {})['station_file'] = self.station_file
            state.update(seq=self.events.seq, utc=utc())
            return state

    def bind(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def log_message(self, *_):
                pass

            def reply(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_GET(self):
                self.handle_route('GET')

            def do_HEAD(self):
                self.handle_route('HEAD')

            def do_POST(self):
                self.handle_route('POST')

            def handle_route(self, method):
                try:
                    if self.headers.get_all('Host', []) not in [
                            [f'{host}:{service.port}'] for host in ('localhost', '127.0.0.1', '[::1]')]:
                        self.reply(403, {'error': 'expected loopback Host and bound port'})
                        return
                    route = urlsplit(self.path)
                    if method in ('GET', 'HEAD') and route.path == wire.STATE:
                        self.reply(200, service.snapshot())
                    elif method == 'GET' and route.path == wire.EVENTS:
                        after = self.headers.get('Last-Event-ID',
                                  parse_qs(route.query).get('after', [str(service.events.seq)])[0])
                        self.stream(int(after))
                    elif method == 'POST' and route.path == wire.COMMANDS:
                        origin = self.headers.get('Origin')
                        if origin is not None and urlsplit(origin).netloc != self.headers.get('Host'):
                            self.reply(403, {'error': 'origin mismatch'})
                            return
                        if self.headers.get_content_type() != 'application/json':
                            self.reply(415, {'error': 'expected application/json'})
                            return
                        size = int(self.headers.get('Content-Length', '0'))
                        if not 0 < size <= 4096 or self.headers.get('Transfer-Encoding'):
                            self.reply(413, {'error': 'command body must be 1..4096 bytes'})
                            return
                        self.reply(202, service.submit(json.loads(self.rfile.read(size))))
                    else:
                        for prefix in sorted(service._mounts, key=len, reverse=True):
                            if route.path == prefix or route.path.startswith(prefix.rstrip('/') + '/'):
                                service._mounts[prefix](self)
                                return
                        self.reply(404, {'error': 'no face mounted' if route.path == '/' else 'not found'})
                except (OSError, ConnectionError):
                    return
                except (ValueError, TypeError) as exc:
                    self.reply(400, {'error': str(exc)})
                except Exception as exc:
                    try:
                        self.reply(500, {'error': str(exc)})
                    except OSError:
                        pass

            def stream(self, after):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                while not service._closed.is_set():
                    with service.events.condition:
                        events = service.events.after(after)
                        if not events:
                            service.events.condition.wait(15)
                            events = service.events.after(after)
                    if events:
                        for event in events:
                            self.wfile.write((f"id: {event['seq']}\nevent: {event['type']}\n"
                                              f"data: {json.dumps(event)}\n\n").encode())
                            after = event['seq']
                        with service.events.condition:
                            service._delivered = max(service._delivered, after)
                            service.events.condition.notify_all()
                    else:
                        self.wfile.write(b': keepalive\n\n')
                    self.wfile.flush()

        try:
            self._http = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            code = wire.EXIT_PORT_CONFLICT
            found = 'unidentified service'
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
            timer = None
            started = time.monotonic()
            try:
                connection.connect()
                sock = connection.sock

                def interrupt():
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()

                timer = threading.Timer(max(0, 2 - (time.monotonic() - started)), interrupt)
                timer.daemon = True
                timer.start()
                connection.request('GET', wire.STATE)
                response = connection.getresponse()
                payload = json.loads(response.read(65536))
                found = repr(payload)
                identity = payload['service']
                if (response.status == 200 and identity['name'] == wire.SERVICE_NAME
                        and type(identity['schema']) is int and identity['schema'] == wire.SCHEMA
                        and identity['state'] in wire.SERVICE_STATES
                        and isinstance(identity['incarnation'], str) and identity['incarnation']
                        and type(payload['rig']['retired']) is bool
                        and payload['station']['station_file'] == self.station_file):
                    code = wire.EXIT_ALREADY_RUNNING
            except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
                pass
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join()
                connection.close()
            raise BindConflict(f'{self.station_file}: port {self.port}: {found}', code) from exc
        self.port = self._http.server_port
        if self.station is not None:
            try:
                self.station = self.station
            except BaseException:
                self._http.server_close()
                raise
        self._http_thread = threading.Thread(target=self._http.serve_forever,
                                            kwargs={'poll_interval': 0.05},
                                            name='station-http', daemon=True)
        self._http_thread.start()
        self._worker = threading.Thread(target=self._dispatch, name='station-commands', daemon=True)
        self._worker.start()

    def _result(self, command, state, **fields):
        return self.events.publish('command', dict(id=command['id'], state=state,
                                                   detail=fields.pop('detail', ''), **fields))

    def submit(self, command):
        if (not isinstance(command, dict)
                or any(not isinstance(command.get(k), str) or not command[k]
                       for k in ('id', 'client', 'kind'))
                or not isinstance(command.get('args', {}), dict)):
            raise ValueError('expected id, client, kind and object args')
        command = json.loads(json.dumps(command))
        station = self.station
        with self._submit_lock, (station.sessions.lock
                                if station is not None and command['kind'] != 'stop'
                                else nullcontext()):
            kind = command['kind']
            key = command['id']
            authorized = False
            try:
                if station is None:
                    raise Refused(wire.REFUSAL_STARTING)
                if self._closed.is_set() or station._stop.is_set():
                    raise Refused(wire.REFUSAL_SHUTTING_DOWN)
                if kind not in ('control', 'stop', 'shutdown'):
                    station.sessions.authorize(command)
                authorized = True
                if key in self._ids:
                    ack = self._ids[key]
                    if kind == 'control':
                        station.sessions.authorize(dict(kind='control', client=command['client'], **ack))
                    return dict(ack)
                if kind not in ('stop', 'control', 'heartbeat', 'ack'):
                    if station.transitioning:
                        raise Refused(wire.REFUSAL_TRANSITIONING)
                    if kind == 'fly':
                        station.sessions.check_holder()
                    if self.commands.full():
                        raise Refused(wire.REFUSAL_QUEUE_FULL)
                if kind in ('heartbeat', 'ack'):
                    heartbeat = station.sessions.heartbeat(command)
            except Refused as exc:
                event = self._refused(command, exc)
                ack = {'id': command['id'], 'seq': event['seq']}
                if authorized and exc.reason not in (wire.REFUSAL_NOT_CONTROLLER,
                                                     wire.REFUSAL_SUPERSEDED):
                    self._remember(key, ack)
                return ack
            event = self._result(command, 'accepted')
            ack = {'id': command['id'], 'seq': event['seq']}
            if kind == 'control':
                result = station.sessions.control(command['client'])
                ack.update(control_id=result['control_id'], generation=result['generation'])
                self._result(command, 'done', result=result)
            elif kind in ('heartbeat', 'ack'):
                self._result(command, 'done', result=heartbeat)
            elif kind == 'stop':
                station.stop_transmit()
                self._result(command, 'done', result={'stop_requested': True})
            else:
                self.commands.put_nowait(command)
            self._remember(key, ack)
            return ack

    def _remember(self, id, ack):
        self._ids[id] = dict(ack)
        if len(self._ids) > 1000:
            self._ids.popitem(last=False)

    def _refused(self, command, exc):
        return self._result(command, 'refused', reason=exc.reason, detail=str(exc),
                            **({'claim_holder': str(exc)} if exc.holder is not None else {}))

    def _dispatch(self):
        while not self._closed.is_set():
            if self.station is not None:
                self.station.sessions.expire()
            try:
                command = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                kind, args = command['kind'], command.get('args', {})
                with self.station.sessions.lock:
                    if kind == 'own' and self.station._stop.is_set():
                        raise Refused(wire.REFUSAL_SHUTTING_DOWN)
                    self.station.sessions.authorize(command)
                    self._result(command, 'running')
                if kind == 'own':
                    result = self.station.own(receive=args.get('receive', False))
                elif kind in ('release', 'shutdown'):
                    result = self.station.release()
                else:
                    self._result(command, 'refused', reason=wire.REFUSAL_UNKNOWN_COMMAND)
                    continue
                event = self._result(command, 'done', result=result,
                             detail='PTT proof and measurements key briefly without modulation'
                             if kind == 'own' and not self.station.receive else '')
                if kind == 'shutdown':
                    self._shutdown_seq = event['seq']
                    self.station._stop.set()
                    return
            except Refused as exc:
                self._refused(command, exc)
            except claim.ClaimError as exc:
                self._result(command, 'refused', reason=wire.REFUSAL_CLAIMED,
                             claim_holder=str(exc), detail=str(exc))
            except Exception as exc:
                self._result(command, 'refused' if str(exc) in (wire.REFUSAL_TRANSITIONING,
                                                             wire.REFUSAL_SHUTTING_DOWN)
                             else 'failed',
                             reason=str(exc), detail=str(exc))
                if command['kind'] == 'shutdown':
                    self.station.fatal_shutdown(f'shutdown release failed: {exc}')
                    return

    @property
    def closing(self):
        """Set once close begins, for work mounted on the service to stop by."""
        return self._closed

    def close(self, timeout=45):
        if self._shutdown_seq is not None:
            with self.events.condition:
                self.events.condition.wait_for(
                    lambda: self._delivered >= self._shutdown_seq, timeout=0.25)
        self._closed.set()
        with self.events.condition:
            self.events.condition.notify_all()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        for thread in (self._worker, self._http_thread):
            if thread is not None:
                thread.join(timeout)
                if thread.is_alive():
                    raise LeaseTimeout(thread.name)
