# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from collections import OrderedDict
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
from hfmodem.core.waits import Timer, join
from . import wire
from .chat_transport import Events, MiB, TerminalResults, TokenBucket, encode_event
from .sessions import Refused


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class ReadBusy(ValueError):
    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__('bulk read budget busy; retry later')


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
        self._chat_payloads = OrderedDict()
        self.results = TerminalResults()
        self.chat = None
        self._bulk_lock = threading.Lock()
        self._bulk_budget = TokenBucket.bulk()
        self._sse_slots = threading.BoundedSemaphore(8)
        self._chat_pump_failed = False
        self._failed_subscription = None
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
        if station is not self._station:
            if self.chat is not None:
                self.chat.revoke()
            if station is not None:
                from .chat import Chat
                self.chat = Chat(self, station)
            else:
                self.chat = None
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
        if self.chat is not None:
            state['chat_lifecycle'] = self.chat.lifecycle()
            state['chat'] = self.chat.live()
        return state

    def read_bulk(self, path, query):
        """Preflight the maximum response before copying recovery state."""
        if not self._bulk_lock.acquire(blocking=False):
            raise ReadBusy(0.1)
        try:
            retry_after = self._bulk_budget.retry_after(4 * MiB)
            if retry_after:
                raise ReadBusy(retry_after)
            def value(key):
                values = query.get(key, [])
                if len(values) != 1 or not values[0]:
                    raise ValueError(f'expected {key}')
                return values[0]
            if path == wire.COMMAND_RESULT:
                payload = self.results.lookup(value('incarnation'), value('client'), value('id'))
            elif path in wire.CHAT_READ_PATHS and self.chat is not None:
                payload = self.chat.read(path, query)
            else:
                raise ValueError('chat unavailable or unknown read path')
            encoded = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            if len(encoded) > 4 * MiB:
                raise ValueError('bulk response exceeds 4 MiB; use a result page')
            # Charge when output is ready. Charging before serialization lets
            # a slow capture refill its own reservation before the write.
            if not self._bulk_budget.consume(len(encoded)):
                raise ReadBusy(self._bulk_budget.retry_after(len(encoded)))
            return encoded
        finally:
            self._bulk_lock.release()

    def bind(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def log_message(self, *_):
                pass

            def reply(self, status, payload, *, retry_after=None):
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                if retry_after is not None:
                    self.send_header('Retry-After', str(max(1, int(retry_after + 0.999))))
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
                    elif method in ('GET', 'HEAD') and route.path in (*wire.CHAT_READ_PATHS, wire.COMMAND_RESULT):
                        self.reply(200, service.read_bulk(route.path, parse_qs(route.query)))
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
                except ReadBusy as exc:
                    self.reply(429, {'error': str(exc)}, retry_after=exc.retry_after)
                except Refused as exc:
                    self.reply(409, {'error': exc.reason, 'detail': str(exc)})
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
                if not service._sse_slots.acquire(blocking=False):
                    self.reply(429, {'error': 'subscriber limit'}, retry_after=1)
                    return
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    budget = TokenBucket.sse()
                    while not service._closed.is_set():
                        with service.events.condition:
                            events = service.events.after(after)
                            if not events:
                                service.events.condition.wait(15)
                                events = service.events.after(after)
                        body = b''.join(encode_event(event) for event in events) if events else b': keepalive\n\n'
                        while not budget.consume(len(body)):
                            if service._closed.wait(min(.25, budget.retry_after(len(body)))):
                                return
                        self.wfile.write(body)
                        self.wfile.flush()
                        if events:
                            after = events[-1]['seq']
                            with service.events.condition:
                                service._delivered = max(service._delivered, after)
                                service.events.condition.notify_all()
                finally:
                    service._sse_slots.release()

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

                timer = Timer(max(0, 2 - (time.monotonic() - started)), interrupt)
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

    def _result(self, command, state, *, retain=True, **fields):
        result = dict(id=command['id'], client=command['client'], state=state,
                      detail=fields.pop('detail', ''), **fields)
        if retain and state in ('done', 'failed', 'refused'):
            self.results.remember(self.incarnation, command['client'], command['id'], result)
        return self.events.publish('command', result)

    def submit(self, command):
        if (not isinstance(command, dict)
                or any(not isinstance(command.get(k), str) or not command[k]
                       for k in ('id', 'client', 'kind'))
                or not isinstance(command.get('args', {}), dict)):
            raise ValueError('expected id, client, kind and object args')
        encoded = json.dumps(command, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        if len(encoded) > 4096:
            raise ValueError('command exceeds 4096 bytes')
        command = json.loads(encoded)
        signature = json.dumps(command, sort_keys=True).encode('utf-8')
        station = self.station
        with self._submit_lock, (station.sessions.lock
                                if station is not None and command['kind'] != 'stop'
                                else nullcontext()):
            kind = command['kind']
            chat_kind = kind in wire.CHAT_COMMAND_KINDS
            key = (command['client'], command['id']) if chat_kind else command['id']
            authorized = False
            try:
                if station is None:
                    raise Refused(wire.REFUSAL_STARTING)
                if self._closed.is_set() or station._stop.is_set():
                    raise Refused(wire.REFUSAL_SHUTTING_DOWN)
                if not chat_kind and kind not in ('control', 'stop', 'shutdown'):
                    station.sessions.authorize(command)
                authorized = True
                if key in self._ids:
                    if chat_kind and self._chat_payloads.get(key) != signature:
                        raise Refused('command-id-conflict')
                    ack = self._ids[key]
                    if kind == 'control':
                        station.sessions.authorize(dict(kind='control', client=command['client'], **ack))
                    return dict(ack)
                if chat_kind:
                    self.chat.validate(command)
                if kind not in ('stop', 'control', 'heartbeat', 'ack'):
                    if station.transitioning:
                        raise Refused(wire.REFUSAL_TRANSITIONING)
                    if kind == 'fly':
                        station.sessions.check_holder()
                    if self.commands.full():
                        raise Refused(wire.REFUSAL_QUEUE_FULL)
                if chat_kind:
                    self.chat.reserve(command)
                if kind in ('heartbeat', 'ack'):
                    heartbeat = station.sessions.heartbeat(command)
            except Refused as exc:
                event = self._refused(command, exc)
                ack = {'id': command['id'], 'seq': event['seq'], 'incarnation': self.incarnation}
                if authorized and exc.reason not in (wire.REFUSAL_NOT_CONTROLLER,
                                                     wire.REFUSAL_SUPERSEDED, 'command-id-conflict'):
                    self._remember(key, ack)
                    if chat_kind:
                        self._chat_payloads[key] = signature
                return ack
            event = self._result(command, 'accepted')
            ack = {'id': command['id'], 'seq': event['seq'], 'incarnation': self.incarnation}
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
            if chat_kind:
                self._chat_payloads[key] = signature
            return ack

    def _remember(self, id, ack):
        self._ids[id] = dict(ack)
        if len(self._ids) > 1000:
            removed, _ = self._ids.popitem(last=False)
            self._chat_payloads.pop(removed, None)

    def _refused(self, command, exc):
        return self._result(command, 'refused', retain=exc.reason not in ('command-id-conflict', 'chat-lifecycle-stale'),
                            reason=exc.reason, detail=str(exc),
                            **({'claim_holder': str(exc)} if exc.holder is not None else {}))

    def _dispatch(self):
        while not self._closed.is_set():
            if self.station is not None:
                self.station.sessions.expire()
                subscription = getattr(self.chat, '_subscription', None)
                if not self._chat_pump_failed or subscription is not self._failed_subscription:
                    try:
                        self.chat.pump()
                    except Exception as error:
                        # The controller still owns its journal/snapshot. Stop
                        # publishing the failed batch, fence its attachment and
                        # signal recovery without killing station safety STOP.
                        self._chat_pump_failed = True
                        self._failed_subscription = subscription
                        self.chat.revoke()
                        self.events.publish('gap', {'from': self.events.seq,
                            'to': self.events.seq + 1, 'reason': 'chat-event-publication-failed'})
                        self.events.publish('warning', {'reason': 'chat-event-publication-failed',
                            'detail': str(error)[:512]})
            try:
                command = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                kind, args = command['kind'], command.get('args', {})
                with self.station.sessions.lock:
                    if kind == 'own' and self.station._stop.is_set():
                        raise Refused(wire.REFUSAL_SHUTTING_DOWN)
                    if kind in wire.CHAT_COMMAND_KINDS:
                        if kind != 'chat.start':
                            self.chat.validate(command)
                    else:
                        self.station.sessions.authorize(command)
                    self._result(command, 'running')
                if kind in wire.CHAT_COMMAND_KINDS:
                    self.chat.dispatch(command, lambda status, _command=command, **fields:
                                       self._result(_command, status, **fields))
                    continue
                if kind in ('release', 'shutdown'):
                    self.chat.revoke()
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
        chat_failure = None
        if self.chat is not None:
            try:
                self.chat.revoke()
                self.chat.close(deadline=time.monotonic() + timeout)
            except BaseException as exc:
                chat_failure = exc
        try:
            with self.events.condition:
                self.events.condition.notify_all()
            if self._http is not None:
                self._http.shutdown()
                self._http.server_close()
            for thread in (self._worker, self._http_thread):
                if thread is not None:
                    if not join(thread, time.monotonic() + timeout):
                        raise LeaseTimeout(thread.name)
        except BaseException:
            if chat_failure is None:
                raise
        if chat_failure is not None:
            raise chat_failure
