# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from hfmodem.tests import hermetic
from hfmodem.contact import chat_state as s
from hfmodem.station.client import Client, CommandFailed

ASSET = Path(__file__).parents[2] / 'ui/assets/station_chat.mjs'


def test_browser_station_protocol_is_pure_bounded_and_preserves_reservations():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js unavailable')
    script = '''
import assert from 'node:assert/strict';
import {supportsStationChat, stationCommand, stationStart, stationCommandResult, capabilityAllows}
 from ASSET;
for (const name of ['fetch', 'WebSocket', 'EventSource']) globalThis[name] = () => {throw Error('network');};
const state = {service: {schema: 2, incarnation: 'inc'}, chat_lifecycle:
 {incarnation: 'inc', generation: 3, state: 'idle', controller_id: null}};
assert.equal(supportsStationChat(state), true);
assert.equal(supportsStationChat({service: state.service}), false);
const start = stationStart(state, {id: 'start', client: 'browser', control_id: 'lease', generation: 2});
assert.deepEqual(start.args, {expected_incarnation: 'inc', expected_lifecycle_generation: 3, expected_controller_id: null});
const request = {_type: 'Request', id: 'one', controller_id: 'controller', link_generation: 1,
 actor: 'browser', control_id: 'lease', control_generation: 2,
 action: {_type: 'BeaconOnce', kind: 'beacon.once', force: false}};
const envelope = stationCommand(request, 'browser');
assert.equal(envelope.kind, 'beacon.once'); assert.equal(envelope.generation, 2);
assert.deepEqual(envelope.args, request);
request.action.force = true; assert.equal(envelope.args.action.force, false);
assert.throws(() => stationCommand({...request, actor: '\\x00'.repeat(1000)}, 'browser'), /4096/);
const capability = {_type: 'Capability', kind: 'beacon.once', effect: 'X', availability: 'blocked',
 reservation: {_type: 'ServiceReservation', family: 'beacon', state: 'pending', request_id: 'one'},
 reason_code: 'beacon-pending', reason: 'Wait'};
assert.equal(capabilityAllows(capability), false);
assert.equal(capabilityAllows({...capability, availability: 'allowed'}), false);
assert.equal(capabilityAllows({...capability, availability: 'check-on-execution',
 reservation: {...capability.reservation, state: 'available'}}), true);
const result = stationCommandResult({type: 'command', data: {id: 'start', state: 'failed', reason: 'chat-start-cancelled'}});
assert.equal(result.state, 'failed'); assert.equal(result.reason, 'chat-start-cancelled');
'''.replace('from ASSET;', f'from {json.dumps(ASSET.as_uri())};')
    hermetic.guard.admit([node, '--input-type=module', '--eval'], os.environ)
    result = subprocess.run([node, '--input-type=module', '--eval', script],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_python_station_serialization_and_cancellation(monkeypatch):
    client = object.__new__(Client)
    client.client_id, client.timeout = 'client', 2
    client.snapshot = {'service': {'schema': 2, 'incarnation': 'inc'},
                       'chat_lifecycle': {'incarnation': 'inc', 'generation': 3,
                                          'state': 'idle', 'controller_id': None}}
    calls = []
    def request(method, path, body=None, **kwargs):
        calls.append(body)
        return {'id': body['id'], 'seq': 1, 'incarnation': 'inc'}
    monkeypatch.setattr(client, '_json', request)
    request = s.Request('one', 'controller', 2, 'browser', 'lease', 4, s.BeaconOnce(False))
    client.chat_command(s.to_wire(request))
    assert calls[-1]['kind'] == 'beacon.once'
    assert calls[-1]['control_id'] == 'lease' and calls[-1]['generation'] == 4
    assert s.from_wire(calls[-1]['args']) == request
    client.chat_start(command_id='start', control_id='lease', generation=4)
    assert calls[-1]['args'] == {'expected_incarnation': 'inc',
                                'expected_lifecycle_generation': 3, 'expected_controller_id': None}
    def events(*args, **kwargs):
        yield {'type': 'command', 'data': {'id': 'start', 'state': 'failed',
                                           'reason': 'chat-start-cancelled'}}
    monkeypatch.setattr(client, 'events', events)
    with pytest.raises(CommandFailed) as error:
        client.chat_start(command_id='start', control_id='lease', generation=4, wait=True)
    assert error.value.result['state'] == 'failed'
    assert error.value.result['reason'] == 'chat-start-cancelled'
    reservation = s.ServiceReservation('beacon', 'pending', 'one')
    capability = s.Capability('beacon.once', s.Effect.X, 'blocked', reservation,
                              'beacon-pending', 'Wait for pending request')
    assert s.from_wire(s.to_wire(capability)) == capability


def test_portable_wire_chat_kinds_match_typed_actions():
    from hfmodem.station import wire
    assert wire.CHAT_COMMAND_KINDS == set(s.ACTIONS) | {'chat.start'}
    assert wire.CHAT_READ_KINDS == {kind for kind, (_, effect) in s.ACTIONS.items()
                                    if effect is s.Effect.R}


def run_station_transport_js(body):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js unavailable')
    from dataclasses import replace
    initial = s.initial_snapshot()
    initial = replace(initial, controller_id='fixture-controller', revision=1, through_seq=1,
                      link=replace(initial.link, generation=1))
    model = Path(__file__).resolve().parents[5] / 'tools/logbook_view/chat_mock.mjs'
    script = '''
import assert from 'node:assert/strict';
import {StationChatTransport} from ASSET;
import {ChatView, project} from MODEL;
import {readFileSync} from 'node:fs';
const deskSource = readFileSync(DESK, 'utf8').replace("'./station_chat.mjs'", ASSET_URI).replace("'./chat_mock.mjs'", MODEL_URI);
const {DeskChat, browserStationIO, mountDeskChat} = await import('data:text/javascript;base64,' + Buffer.from(deskSource).toString('base64'));
for (const name of ['fetch', 'WebSocket', 'EventSource']) globalThis[name] = () => {throw Error('network');};
let now = 0, serial = 0, current = SNAPSHOT;
const commands = [], reads = [], subscriptions = new Set();
const state = () => ({service: {schema: 2, incarnation: 'inc'}, chat_lifecycle:
 {incarnation: 'inc', generation: 1, state: 'ready', controller_id: current.controller_id}});
const io = {state: async () => state(), command: async envelope => {
 commands.push(structuredClone(envelope));
 return {id: envelope.id, client: envelope.client, state: 'done', result: {handoff: true}};
}, subscribe: handler => {subscriptions.add(handler); return () => subscriptions.delete(handler);},
 read: async path => {reads.push(path); now += .001;
  if (path.startsWith('/chat/projection')) return project(current);
  if (path.startsWith('/chat/snapshot')) return structuredClone(current);
  throw Error(`unexpected read ${path}`);
 }};
const options = {client: 'observer', clock: () => now,
 id: () => `request-${++serial}`, pause: async ms => {now += ms / 1000;}};
const transport = new StationChatTransport(io, options);
const view = new ChatView(transport, {clock: () => now, id: options.id});
const emit = (seq, payload) => { const inner = {_type: 'EventEnvelope', seq,
 controller_id: current.controller_id, observed_at: now, utc: '', link_generation: 1,
 request_id: null, kind: payload._type, payload};
 for (const handler of subscriptions) handler({seq: seq + 1000, type: 'chat.event', data: inner});
};
BODY
'''.replace('from ASSET;', f'from {json.dumps(ASSET.as_uri())};').replace(
        'from MODEL;', f'from {json.dumps(model.as_uri())};').replace(
        'SNAPSHOT', json.dumps(s.to_wire(initial))).replace('DESK', json.dumps(str(ASSET.parent / 'hfdesk_chat.mjs'))).replace(
        'ASSET_URI', json.dumps(json.dumps(ASSET.as_uri()))).replace(
        'MODEL_URI', json.dumps(json.dumps(model.as_uri()))).replace('BODY', body)
    hermetic.guard.admit([node, '--input-type=module', '--eval'], os.environ)
    result = subprocess.run([node, '--input-type=module', '--eval', script],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_injected_station_two_views_share_history_free_projection_and_observer_fence():
    run_station_transport_js('''
await view.mount();
const secondTransport = new StationChatTransport(io, {...options, client: 'second'});
const second = new ChatView(secondTransport, {clock: () => now});
await second.mount();
assert.equal(subscriptions.size, 2);
assert.equal(commands.length, 0);
current.revision = 2; current.through_seq = 2;
current.application.draft = {_type: 'Draft', id: 'shared', revision: 1, owner: 'holder',
 kind: 'vmail', generation: 1, peer: 'N0XYZ', subject: 'test', recipient: null, reply_to: null, lines: ['shared text']};
emit(2, {_type: 'StateChanged', revision: 2, through_seq: 2, state: null,
 changed_sections: ['application', 'capabilities']});
now = 1; await view.pump(); await second.pump();
assert.equal(view.state.application.draft.lines[0], 'shared text');
assert.deepEqual(view.state.application.draft, second.state.application.draft);
assert.equal(reads.filter(path => path === '/chat/snapshot').length, 2);
assert.equal(reads.filter(path => path.startsWith('/chat/projection')).length, 2);
await assert.rejects(view.action({_type: 'ChatSend', kind: 'chat.send', text: 'forbidden'}), /not-controller/);
assert.equal(commands.length, 0);
transport.close(); secondTransport.close(); assert.equal(subscriptions.size, 0);
''')


def test_station_gap_uses_controller_watermark_and_recovery_never_sends_commands():
    run_station_transport_js('''
await view.mount();
const handler = [...subscriptions][0];
handler({seq: 99999, type: 'gap', data: {from: 1, to: 100000}});
now = 2; await view.pump();
assert.equal(view.needRecovery, false);
assert.equal(view.displayCursor, current.through_seq);
assert.equal(commands.length, 0);
assert.equal(reads.filter(path => path === '/chat/snapshot').length, 2);
transport.close();
''')


def test_station_slow_snapshot_repeated_overflow_keeps_input_disabled_until_atomic_recovery():
    run_station_transport_js('''
const originalRead = io.read;
let finish;
io.read = async path => {
 if (path === '/chat/snapshot') {
   const captured = structuredClone(current);
   await new Promise(resolve => {finish = resolve;});
   return captured;
 }
 return originalRead(path);
};
const mounting = view.mount();
while (!finish) await Promise.resolve();
for (let seq = 2; seq <= 181; seq++) emit(seq, {_type: 'Notice', level: 'info', text: '\\x00'.repeat(500)});
assert(transport.subscribers.values().next().value.items.length <= 128);
assert(transport.subscribers.values().next().value.size <= 256 * 1024);
finish(); await mounting;
assert.equal(view.needRecovery, true); assert.equal(view.bbsEnabled(), false);
current.through_seq = 181; current.revision = 181;
now = 2; let finishing;
io.read = async path => {
 if (path === '/chat/snapshot') {
   const captured = structuredClone(current);
   await new Promise(resolve => {finishing = resolve;});
   return captured;
 }
 return originalRead(path);
};
const recovering = view.pump();
while (!finishing) await Promise.resolve();
for (let seq = 182; seq <= 361; seq++) emit(seq, {_type: 'Notice', level: 'info', text: '\\x00'.repeat(500)});
finishing(); await recovering;
assert.equal(view.needRecovery, true); assert.equal(view.bbsEnabled(), false);
io.read = originalRead; current.through_seq = 361; current.revision = 361;
now = 4; await view.pump();
assert.equal(view.needRecovery, false); assert.equal(view.displayCursor, 361);
assert.equal(commands.length, 0); transport.close();
''')


def test_station_projection_reads_coalesce_to_five_hz():
    run_station_transport_js('''
await view.mount();
const changed = seq => { current.revision = seq; current.through_seq = seq;
 emit(seq, {_type: 'StateChanged', revision: seq, through_seq: seq, state: null, changed_sections: ['application']}); };
now = 1; changed(2); await view.pump();
now = 1.05; changed(3); await view.pump();
now = 1.10; changed(4); await view.pump();
assert.equal(reads.filter(path => path.startsWith('/chat/projection')).length, 1);
now = 1.21; await view.pump();
assert.equal(reads.filter(path => path.startsWith('/chat/projection')).length, 2);
assert.equal(view.applied.get('application').revision, 4);
assert.equal(commands.length, 0); transport.close();
''')


def test_station_capability_check_discards_reply_after_editor_change():
    run_station_transport_js('''
await view.mount(); now = 1;
const originalRead = io.read;
let finish;
const action = {_type: 'ChatSend', kind: 'chat.send', text: 'reviewed'};
io.read = async path => {
 if (path.startsWith('/chat/result?request_id=')) return new Promise(resolve => {
  finish = () => resolve({_type: 'RequestResult', state: {state: 'done'}, result:
   {_type: 'CapabilityCheck', controller_id: current.controller_id, revision: current.revision,
    through_seq: current.through_seq, action, capability: {_type: 'Capability', kind: 'chat.send',
     availability: 'check-on-execution', effect: 'X', reservation: null, reason: null, reason_code: null}}});
 });
 return originalRead(path);
};
const checking = view.preview(action);
while (!finish) await Promise.resolve();
view.edit('changed editor'); finish(); await checking;
assert.equal(view.capabilityCheck, null);
assert.equal(commands.length, 1); assert.equal(commands[0].kind, 'chat.check');
assert.equal(commands[0].args.control_id, null);
assert.equal(commands[0].args.action.action.kind, 'chat.send'); transport.close();
''')


def test_hfdesk_native_mount_is_inert_and_existing_controller_attach_is_read_only():
    run_station_transport_js('''
const scheduled = [];
const desk = new DeskChat(io, {...options, schedule: (work, ms) => {
 const timer = {work, ms, closed: false}; scheduled.push(timer); return () => {timer.closed = true;};
}});
assert.equal(reads.length, 0); assert.equal(commands.length, 0); assert.equal(subscriptions.size, 0);
await desk.open();
assert.equal(desk.view.controllerId, current.controller_id); assert.equal(commands.length, 0);
assert.equal(desk.credentials, null); assert.equal(scheduled.filter(timer => timer.ms === 10000).length, 0);
desk.view.resize(80, 24); desk.view.setFilter('N0'); desk.view.setTab('BBS');
now = 2; await desk.view.reconnect(); assert.equal(commands.length, 0);
desk.close(); assert.equal(subscriptions.size, 0); assert(scheduled.every(timer => timer.closed));
class Node {constructor() {this.listeners = {}; this.children = [];}
 append(...nodes) {this.children.push(...nodes);} replaceChildren(...nodes) {this.children = nodes;}
 setAttribute() {} addEventListener(name, callback) {this.listeners[name] = callback;}}
const document = {createElement: () => new Node()}; const root = new Node();
const before = reads.length;
const mounted = mountDeskChat(document, root, io, options);
assert.equal(reads.length, before); assert.equal(commands.length, 0);
assert.equal(root.children[1].children[0].disabled, true);
mounted.close();
''')


def test_hfdesk_explicit_start_and_heartbeat_never_refresh_replaced_lease():
    run_station_transport_js('''
let generation = 3, controller = null;
io.state = async () => ({service: {schema: 2, incarnation: 'inc'}, chat_lifecycle:
 {incarnation: 'inc', generation, state: 'idle', controller_id: controller}});
const scheduled = [];
io.command = async envelope => {
 commands.push(structuredClone(envelope));
 if (envelope.kind === 'control') return {state: 'done', result: {control_id: 'exact-lease', generation: 7}};
 if (envelope.kind === 'heartbeat') return {state: 'refused', reason: 'superseded'};
 if (envelope.kind === 'chat.start') {generation++; return {state: 'failed', reason: 'chat-start-cancelled'};}
 throw Error('unexpected mutation');
};
const desk = new DeskChat(io, {...options, schedule: (work, ms) => {
 const timer = {work, ms, closed: false}; scheduled.push(timer); return () => {timer.closed = true;};
}});
await desk.open(); assert.equal(commands.length, 0);
await desk.acquire(); assert.deepEqual(commands.map(command => command.kind), ['control']);
await assert.rejects(desk.start(), /chat-start-cancelled/);
assert.deepEqual(commands.map(command => command.kind), ['control', 'chat.start']);
assert.equal(commands[1].args.expected_lifecycle_generation, 3);
assert.equal(commands[1].control_id, 'exact-lease'); assert.equal(commands[1].generation, 7);
const heartbeat = scheduled.find(timer => timer.ms === 10000);
await heartbeat.work();
assert.equal(desk.credentials, null); assert.equal(heartbeat.closed, true);
assert.equal(commands[2].kind, 'heartbeat'); assert.equal(commands[2].control_id, 'exact-lease');
await heartbeat.work(); assert.equal(commands.length, 3);
await assert.rejects(desk.start(), /Acquire control/); assert.equal(commands.length, 3);
desk.close();
''')


def test_hfdesk_browser_io_construction_does_not_open_network():
    run_station_transport_js('''
let requests = 0, streams = 0;
const injected = browserStationIO({fetch: async () => {requests++; throw Error('not requested');},
 EventSource: class {constructor() {streams++;}}});
assert.equal(requests, 0); assert.equal(streams, 0);
const desk = new DeskChat(injected, options);
assert.equal(requests, 0); assert.equal(streams, 0); desk.close();
''')


def test_hfdesk_static_chat_assets_use_existing_get_face_without_sockets(tools):
    from io import BytesIO
    from types import SimpleNamespace
    from desk.face import Handler
    for path in ('/station_chat.mjs', '/hfdesk_chat.mjs', '/chat_mock.mjs'):
        statuses, headers = [], {}
        request = SimpleNamespace(path=path, command='GET', wfile=BytesIO(),
                                  send_response=statuses.append,
                                  send_header=lambda key, value: headers.update({key: value}),
                                  end_headers=lambda: None)
        Handler(request, None).do_GET()
        assert statuses == [200]
        assert headers['Content-Type'] == 'text/javascript; charset=utf-8'
        assert b'SPDX-License-Identifier' in request.wfile.getvalue()
    index = (Path(__file__).resolve().parents[5] / 'tools/logbook_view/index.html').read_text()
    assert 'id="chatViewPanel" aria-label="Station chat" hidden' in index
    assert "await desk.open()" in index


def test_station_completed_read_waits_for_semantics_after_handoff_without_resending():
    run_station_transport_js('''
await view.mount(); now = 2;
const originalRead = io.read; let lookups = 0;
io.read = async path => {
 if (path.startsWith('/chat/result?request_id=')) {
  lookups++;
  if (lookups === 1) return {_type: 'NotFound', id: 'pending'};
  return {_type: 'RequestResult', state: {state: 'done'}, result: {_type: 'HelpText', topic: null, text: 'ready'}};
 }
 return originalRead(path);
};
const result = await view.completed({_type: 'ChatHelp', kind: 'chat.help', topic: null});
assert.equal(result.text, 'ready'); assert.equal(lookups, 2);
assert.equal(commands.length, 1); assert.equal(commands[0].kind, 'chat.help'); transport.close();
''')


def test_hfdesk_http_command_lookup_backoff_never_resends_start():
    run_station_transport_js('''
const requests = []; let lookups = 0;
const response = (status, value) => ({status, ok: status === 200 || status === 202,
 headers: {get: key => key === 'Retry-After' ? '1' : String(JSON.stringify(value).length)},
 text: async () => JSON.stringify(value)});
const injected = browserStationIO({pause: async () => {}, fetch: async (path, options) => {
 requests.push([path, options]);
 if (options.method === 'POST') return response(202, {id: 'start', incarnation: 'inc', seq: 1});
 lookups++;
 if (lookups === 1) return response(429, {error: 'busy'});
 return response(200, {status: 'found', result: {id: 'start', state: 'failed', reason: 'chat-start-cancelled'}});
}});
const result = await injected.command({id: 'start', client: 'operator', kind: 'chat.start', args: {}});
assert.equal(result.reason, 'chat-start-cancelled');
assert.equal(requests.filter(([, options]) => options.method === 'POST').length, 1);
assert.equal(lookups, 2);
''')
