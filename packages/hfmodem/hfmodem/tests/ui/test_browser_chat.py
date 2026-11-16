# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
import os
from pathlib import Path
import shutil
import subprocess
from dataclasses import replace

import pytest

from hfmodem.tests import hermetic
from hfmodem.contact import chat_state as s

REPO = Path(__file__).resolve().parents[5]
MODULE = REPO / 'tools/logbook_view/chat_mock.mjs'
NODE = shutil.which('node')
pytestmark = [
    pytest.mark.skipif(NODE is None, reason='install Node.js to run the offline browser tests'),
    pytest.mark.skipif(not MODULE.exists(), reason='tools/logbook_view is absent from the source drop'),
]


def _chat_fixture():
    return pytest.importorskip('logbook_view.chat_fixture',
                               reason='tools/logbook_view/chat_fixture.py is a development tool that does not ship')


def fixture():
    initial = s.initial_snapshot()
    initial = replace(initial, controller_id='fixture-controller', revision=1, through_seq=1,
                      identity=replace(initial.identity, mycall='W9SSJ', idcall='W9SSJ',
                                       channel_hz=14106500, bandwidth_hz=500),
                      link=replace(initial.link, generation=1, state='established',
                                   connected=True, peer='N0XYZ'))
    return {'snapshot': s.to_wire(initial)}


def run_js(body, bundle=None):
    script = f'''
import assert from 'node:assert/strict';
import {{readFileSync}} from 'node:fs';
import {{ChatView, MockTransport, EventBuffer, project, appendSpans, literal, mountChat}}
  from {json.dumps(MODULE.as_uri())};
const bundle = JSON.parse(readFileSync(0, 'utf8'));
const clone = value => JSON.parse(JSON.stringify(value));
let now = 0, serial = 0;
const transport = new MockTransport(bundle);
const view = new ChatView(transport, {{clock: () => now, id: () => `r${{++serial}}`}});
const event = (seq, payload) => ({{_type: 'EventEnvelope', seq,
  controller_id: bundle.snapshot.controller_id, utc: '', observed_at: now,
  link_generation: 1, request_id: null, kind: payload._type, payload}});
const stateEvent = (seq, revision, changed_sections, state=null) => event(seq,
  {{_type: 'StateChanged', revision, through_seq: seq, state, changed_sections}});
const record = (id, seq, text='literal <img onerror=attack>') => ({{_type: 'DisplayRecord',
  id, created_seq: seq, updated_seq: seq, link_generation: 1, bbs_context: null,
  bbs_revision: null, payload: {{_type: 'TextRecord', id, direction: 'rx', text,
    peer: 'N0XYZ', source: 'contact'}}, tx: null}});
const draft = (revision=0) => ({{_type: 'Draft', id: 'draft', revision, owner: 'browser',
  kind: 'vmail', generation: 1, peer: 'N0XYZ', subject: 'hello', recipient: null,
  reply_to: null, lines: []}});
const accepted = (value, state='done') => ({{_type: 'RequestResult', state: {{id: '',
  kind: 'draft.append', submitted_generation: 1, bound_generation: 1, attempt_id: null,
  state, stage: null, reason_code: null, detail: '', updated_at: 0}}, result: value,
  text_outcomes: []}});
{body}
'''
    hermetic.guard.admit([NODE, '--input-type=module', '--eval'], os.environ)
    result = subprocess.run([NODE, '--input-type=module', '--eval', script],
                            input=json.dumps(bundle or fixture()), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_browser_mount_filter_resize_and_reconnect_are_read_only():
    run_js(r'''
for (const name of ['fetch', 'WebSocket', 'EventSource']) globalThis[name] = () => {
  throw Error('network factory invoked');
};
await view.mount();
view.select({slot: 3}); view.setFilter('N0'); view.setTab('BBS');
view.edit('retained body'); view.resize(60, 18);
now = 1;
await view.reconnect();
assert.equal(view.editor, 'retained body');
assert.equal(view.tab, 'BBS');
assert.equal(transport.commands.length, 0);
assert.equal(transport.reads.snapshot, 2);
assert.equal(transport.reads.projection, 0);
''')


def test_two_browser_views_refresh_every_section_without_history_downloads():
    run_js(r'''
const second = new ChatView(transport, {actor: 'second', clock: () => now});
await view.mount(); await second.mount();
const projection = project(bundle.snapshot);
projection.revision = 2; projection.through_seq = 2;
projection.application.draft = draft(1);
projection.application.bbs.remote_view = false;
projection.application.files.received_bytes = 4096;
projection.receive.snr = -8;
projection.capabilities = [{kind: 'bbs.list', effect: 'X', availability: 'blocked',
  reservation: {family: 'directory', state: 'spent', request_id: 'list'},
  reason_code: 'reconnect-required', reason: 'Reconnect'}];
transport.deliver({projection, events: [stateEvent(2, 2,
  ['application', 'receive', 'capabilities'])]});
await view.pump(); await second.pump();
for (const current of [view, second]) {
  assert.equal(current.state.application.draft.revision, 1);
  assert.equal(current.state.application.bbs.remote_view, false);
  assert.equal(current.state.application.files.received_bytes, 4096);
  assert.equal(current.state.receive.snr, -8);
  assert.equal(current.state.capabilities[0].reservation.state, 'spent');
}
assert.equal(transport.reads.snapshot, 2);
assert.equal(transport.reads.projection, 2);
assert.equal(transport.commands.length, 0);
''')


def test_browser_coalesces_dirty_sections_and_rejects_stale_projection_sections():
    run_js(r'''
await view.mount();
view.event(stateEvent(2, 2, ['application', 'live']));
view.event(stateEvent(3, 3, ['receive']));
const projection = project(bundle.snapshot);
projection.revision = 2; projection.through_seq = 2;
projection.application.draft = draft(); projection.receive.snr = -10;
const cursor = view.displayCursor;
assert.equal(view.applyProjection(projection), true);
assert.equal(view.state.application.draft.id, 'draft');
assert.equal(view.state.receive.snr, null);
assert.equal(view.dirty.has('receive'), true);
assert.equal(view.displayCursor, cursor);
projection.revision = 3; projection.through_seq = 3;
transport.currentProjection = clone(projection);
await view.refresh();
assert.equal(view.dirty.size, 0);
assert.equal(transport.reads.projection, 1);
view.event(stateEvent(4, 4, ['receive']));
await view.refresh();
assert.equal(transport.reads.projection, 1);
now = .2;
transport.currentProjection = {...projection, revision: 4, through_seq: 4};
await view.refresh();
assert.equal(transport.reads.projection, 2);
assert.equal(view.applyProjection({...projection, controller_id: 'old'}), false);
assert.equal(view.applyProjection({...projection, revision: 0, through_seq: 0}), false);
view.event(stateEvent(5, 2, ['application'], {...projection.live, identity: {mycall: 'OLD'}}));
assert.equal(view.state.live.identity.mycall, 'W9SSJ');
''')


def test_browser_parameter_checks_fence_editor_controller_revision_and_throttle():
    run_js(r'''
await view.mount();
const action = {_type: 'ChatSend', kind: 'chat.send', text: 'one'};
let resolve;
transport.check = async () => {transport.reads.check++; return await new Promise(done => resolve=done);};
const pending = view.preview(action);
view.edit('two');
resolve({controller_id: view.controllerId, revision: 1, through_seq: 1, action,
  capability: {availability: 'allowed'}});
await pending;
assert.equal(view.capabilityCheck, null);
view.preview({...action, text: 'two'});
assert.equal(transport.reads.check, 1);
now = .2;
const next = view.flushCheck();
resolve({controller_id: 'different', revision: 2, through_seq: 2,
  action: {...action, text: 'two'}, capability: {availability: 'allowed'}});
await next;
assert.equal(view.capabilityCheck, null);
now = .4;
const last = view.preview(action);
resolve({controller_id: view.controllerId, revision: 1, through_seq: 1, action,
  capability: {availability: 'blocked', reason: 'not linked'}});
await last;
assert.equal(view.capabilityCheck.capability.reason, 'not linked');
assert.equal(transport.commands.length, 0);
''')


def test_browser_gap_reconciles_retained_text_and_tx_without_rf_or_history_recreation():
    run_js(r'''
await view.mount();
const snapshot = clone(bundle.snapshot);
snapshot.revision = 20; snapshot.through_seq = 201;
snapshot.presentation.evicted_through = 199;
const retained = record('retained', 200, 'RX remains');
retained.tx = {_type: 'TxTextOutcome', text_id: 'retained', write_id: 1,
  status: 'partial', framed_bytes: 20, acknowledged_bytes: 10, discarded_bytes: 0,
  unknown_bytes: 10, attribution: 'verified'};
snapshot.presentation.records = [retained];
transport.current = snapshot;
for (let seq=2; seq<=201; seq++) transport.deliver({events: [event(seq,
  {_type: 'DisplayChanged', record: record(`text-${seq}`, seq)})]});
assert.ok(view.subscription.items.length <= 128);
assert.ok(view.subscription.size <= 256*1024);
now = 1;
await view.pump();
assert.equal(view.needRecovery, false);
assert.equal(view.records.size, 1);
assert.equal(view.records.get('retained').tx.acknowledged_bytes, 10);
assert.equal(view.evictedThrough, 199);
view.event(event(202, {_type: 'TxTextOutcome', text_id: 'evicted', status: 'unknown'}));
assert.equal(view.records.has('evicted'), false);
assert.equal(transport.commands.length, 0);
assert.equal(transport.reads.snapshot, 2);
''')


def test_browser_repeated_overflow_has_bounded_recovery_and_keeps_bbs_disabled():
    run_js(r'''
await view.mount();
let resolve;
transport.snapshot = async () => {transport.reads.snapshot++; return await new Promise(done => resolve=done);};
now = 1;
view.needRecovery = true;
const pending = view.recover();
for (let seq=2; seq<400; seq++) view.recoveryBuffer.put(event(seq,
  {_type: 'DisplayChanged', record: record(`long-${seq}`, seq, 'a'.repeat(8000))}));
assert.ok(view.recoveryBuffer.size <= 256*1024);
assert.ok(view.recoveryBuffer.items.length <= 128);
resolve(bundle.snapshot); await pending;
assert.equal(view.needRecovery, true);
assert.equal(view.bbsEnabled(), false);
await view.recover();
assert.equal(transport.reads.snapshot, 2);
now = 2;
const again = view.recover();
resolve({...bundle.snapshot, through_seq: 400}); await again;
assert.equal(view.needRecovery, false);
assert.equal(transport.commands.length, 0);
''')


def test_browser_bbs_input_keeps_rendered_pair_across_new_output_and_recovery():
    bundle = fixture()
    snapshot = bundle['snapshot']
    snapshot['application']['bbs']['remote_view'] = True
    snapshot['presentation'].update(bbs_context='bbs-1', bbs_revision=1)
    page = s.DisplayRecord('page', 1, 1, 1, 'bbs-1', 1,
                          s.BBSPage('page', 'plain', 80, 24,
                                    (s.StyledSpan('Choose 1', None, None, False),)), None)
    snapshot['presentation']['records'] = [s.to_wire(page)]
    run_js(r'''
await view.mount(); view.setTab('BBS'); view.edit('1');
assert.equal(view.bbsEnabled(), true);
assert.deepEqual(view.bbsPair, ['bbs-1', 1]);
const changed = {...clone(bundle.snapshot.presentation.records[0]), id: 'next',
  created_seq: 2, updated_seq: 2, bbs_revision: 2};
view.event(event(2, {_type: 'DisplayChanged', record: changed}));
await assert.rejects(() => view.submit(), /BBS view changed/);
assert.equal(transport.commands.length, 0);
view.bbsPair = null; view.edit('1');
assert.deepEqual(view.bbsPair, ['bbs-1', 2]);
transport.expectations.push({action: {_type: 'BBSInput', kind: 'bbs.input',
  context_id: 'bbs-1', expected_revision: 2, text: '1'}});
await view.submit();
assert.equal(transport.commands.length, 1);
view.edit('old');
view.event(event(3, {_type: 'Gap', after_seq: 2, through_seq: 3, reason: 'test'}));
assert.equal(view.bbsEnabled(), false);
assert.equal(view.bbsPair, null);
const recovered = clone(bundle.snapshot);
recovered.through_seq = 4; recovered.revision = 4;
recovered.presentation.bbs_context = 'bbs-2';
transport.current = recovered;
now = 1; await view.recover();
assert.equal(view.editor, 'old');
assert.equal(view.bbsPair, null);
await assert.rejects(() => view.submit(), /BBS view changed/);
assert.equal(transport.commands.length, 1);
''', bundle)


def test_browser_multiline_upload_uses_acknowledged_revisions_and_literal_period():
    run_js(r'''
await view.mount(); view.state.application.draft = draft();
const append = (revision, line) => ({_type: 'DraftAppend', kind: 'draft.append',
  draft_id: 'draft', expected_revision: revision, line});
transport.expectations = [
  {action: append(0, 'first'), outcome: accepted({_type: 'DraftAccepted', draft: draft(1)})},
  {action: append(1, '.'), outcome: accepted({_type: 'DraftAccepted', draft: draft(2)})},
  {action: {_type: 'VMailSend', kind: 'vmail.send', draft_id: 'draft', expected_revision: 2}}
];
view.edit('first\n.', {paste: true});
assert.equal(transport.commands.length, 0);
const pending = view.submit({explicit: true});
assert.equal(await view.submit({explicit: true}), false);
assert.equal(await pending, true);
assert.deepEqual(transport.commands.map(request => request.action.kind),
  ['draft.append', 'draft.append', 'vmail.send']);
assert.equal(transport.commands[1].action.line, '.');
assert.equal(view.editor, '');
view.state.application.draft = draft();
transport.expectations = [
  {action: append(0, 'one'), outcome: accepted({_type: 'DraftAccepted', draft: draft(1)})},
  {action: append(1, 'two'), outcome: accepted(null, 'refused')}
];
view.edit('one\ntwo', {paste: true});
await assert.rejects(() => view.submit({explicit: true}));
assert.equal(view.editor, 'one\ntwo');
assert.equal(transport.commands.filter(request => request.action.kind === 'vmail.send').length, 1);
''')


def test_browser_literal_rendering_allows_only_fixed_span_styles():
    run_js(r'''
const document = {createElement: () => ({textContent: '', classes: [],
  classList: {add(value) {this.owner.classes.push(value);}}})};
const original = document.createElement;
document.createElement = () => {const value=original(); value.classList.owner=value; return value;};
const parent = {children: [], append(value) {this.children.push(value);}};
appendSpans(document, parent, {mode: 'ansi16', spans: [
  {text: '<script>attack()</script>\u001b]52;clipboard\u0007\u009b31m', foreground: 1, background: 99, bold: true},
  {text: 'normal\n\tline', foreground: 'red;url(bad)', background: -1, bold: false},
  {text: 'parser colors', foreground: 91, background: 44, bold: false}
]});
assert.equal(parent.children[0].textContent, '<script>attack()</script>]52;clipboard31m');
assert.deepEqual(parent.children[0].classes, ['fg-1', 'bold']);
assert.deepEqual(parent.children[1].classes, []);
assert.deepEqual(parent.children[2].classes, ['fg-9', 'bg-4']);
const plain = {children: [], append(value) {this.children.push(value);}};
appendSpans(document, plain, {mode: 'plain', spans: [{text: 'plain', foreground: 2, background: 3, bold: true}]});
assert.deepEqual(plain.children[0].classes, []);
assert.equal(literal('a\u0000b\n\tc'), 'ab\n\tc');
''')


def test_browser_consumes_the_shared_controller_scenarios():
    from hfmodem.contact.chat_fake import SCENARIOS
    capture = _chat_fixture().capture

    for name in SCENARIOS:
        bundle = capture(name)
        run_js(r'''
await view.mount();
for (const step of bundle.playback) {
  now = step.at;
  transport.advance(Math.max(0, now-transport.time));
  await view.pump();
  if (step.action) {
    await view.action(step.action);
    await view.pump();
  }
}
now += 1;
await view.pump();
for (const section of ['application', 'receive', 'capabilities'])
  assert.deepEqual(view.state[section], bundle.expected_projection[section], bundle.scenario + ':' + section);
assert.deepEqual(transport.commands.map(request => request.action), bundle.expected_actions);
assert.equal(transport.reads.snapshot, 1);
assert.equal(transport.expectations.length, 0);
''', bundle)


def test_browser_native_controls_preserve_focus_and_emit_only_explicit_actions():
    run_js(r'''
class Node {
  constructor(tag) {this.tagName=tag; this.children=[]; this.handlers={}; this.value='';
    this.textContent=''; this.classList={add: () => {}}; this.scrollTop=0;
    this.clientHeight=100; this.scrollHeight=100;}
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children=children;}
  setAttribute(name,value) {this[name]=value;}
  addEventListener(name,fn) {(this.handlers[name] ||= []).push(fn);}
  fire(name, extra={}) {for (const fn of this.handlers[name] || []) fn({preventDefault(){}, ...extra});}
  focus() {document.activeElement=this;}
  showModal() {this.open=true;}
  close() {this.open=false;}
}
const document={createElement: tag => new Node(tag)}, root=new Node('main');
const ui=mountChat(document,root,view);
await view.mount();
view.state.live.broadcast.possibly_outstanding=2;
view.changed();
assert.ok(root.children.some(node => node.textContent.includes('2 possibly outstanding; reception unconfirmed')));
root.fire('keydown',{ctrlKey:true,key:'l'});
assert.equal(document.activeElement,ui.composer);
ui.composer.value='hello'; ui.composer.fire('input');
view.resize(80,24); view.setTab('Heard'); view.setTab('Contact');
assert.equal(view.editor,'hello');
assert.equal(document.activeElement,ui.composer);
transport.expectations.push({action:{_type:'ChatSend',kind:'chat.send',text:'hello'}});
ui.composer.fire('keydown',{key:'Enter',shiftKey:false});
await new Promise(resolve=>setImmediate(resolve));
assert.equal(transport.commands.length,1);
assert.equal(transport.commands[0].action.text,'hello');
root.fire('keydown',{key:'F2'});
const dialog=root.children.find(node=>node.tagName==='dialog');
assert.equal(dialog.open,true);
dialog.close();
assert.equal(transport.commands.length,1);
root.fire('keydown',{key:'F4'});
assert.equal(view.tab,'Mail');
assert.equal(transport.commands.length,1);
''')


def test_browser_exit_is_idempotent_during_a_gap_and_waits_for_completion():
    run_js(r'''
await view.mount();
transport.expectations.push({action:{_type:'ChatExit',kind:'chat.exit'},
  outcome: accepted(null,'running')});
view.needRecovery=true;
await Promise.all([view.exit(),view.exit()]);
assert.equal(transport.commands.length,1);
assert.equal(view.closed,false);
view.needRecovery=false;
view.event(event(2,{_type:'Lifecycle',state:'closed',error:null,unkey:'low'}));
assert.equal(view.closed,true);
assert.equal(view.subscription.closed,true);
''')


def test_browser_encoded_caps_include_escaped_multibyte_content():
    run_js(r'''
const buffer=new EventBuffer();
for (let seq=2; seq<200; seq++) buffer.put(event(seq,{_type:'DisplayChanged',
  record:record(`escaped-${seq}`,seq,String.fromCharCode(92,34,10).repeat(7000)+'é'.repeat(7000))}));
assert.ok(buffer.overflow);
assert.ok(buffer.size<=256*1024);
assert.ok(buffer.items.length<=128);
await view.mount();
const giant=clone(bundle.snapshot);
giant.presentation.records=[record('large',2,'é'.repeat(2200000))];
transport.current=giant;
await assert.rejects(()=>transport.snapshot(),/byte budget/);
assert.equal(transport.commands.length,0);
''')


def test_browser_standalone_artifact_needs_no_server_and_escapes_script_boundaries():
    standalone = _chat_fixture().standalone

    bundle = fixture()
    bundle['snapshot']['identity']['mycall'] = '</script><script>attack()</script>'
    html = standalone(bundle)
    assert "from './chat_mock.mjs'" not in html
    assert 'class MockTransport' in html
    assert '<script>attack()' not in html
    assert r'\u003c/script>' in html
    assert html.count('</script>') == 1
    assert 'await view.mount();' in html


def test_browser_scenario_advance_does_not_synthesize_unsubmitted_actions():
    capture = _chat_fixture().capture

    run_js(r'''
await view.mount();
transport.advance(100); now=100;
await view.pump();
assert.equal(view.state.application.draft,null);
assert.equal(transport.commands.length,0);
const action=bundle.expectations[0].action;
await view.action(action);
now=101;
await view.pump();
assert.equal(view.state.application.draft.subject,'Greetings');
assert.equal(transport.commands.length,1);
''', capture('vmail-email'))


def test_browser_recovery_rejects_snapshots_behind_gap_or_applied_state():
    run_js(r'''
await view.mount();
view.event(event(2, {_type:'DisplayChanged',record:record('retained',2,'Keep this text')}));
const projection=project(bundle.snapshot);
projection.revision=6; projection.through_seq=14;
projection.application.draft=draft(3);
view.applyProjection(projection);
view.event(event(20,{_type:'Gap',after_seq:2,through_seq:25,reason:'missing output'}));
now=1; transport.time=1;
await view.recover();
assert.equal(view.needRecovery,true);
assert.equal(view.bbsEnabled(),false);
assert.equal(view.displayCursor,2);
assert.equal(view.records.get('retained').payload.text,'Keep this text');
assert.equal(view.state.application.draft.revision,3);
const stale=clone(bundle.snapshot);
stale.through_seq=26; stale.revision=5;
assert.equal(view.applySnapshot(stale),false);
assert.equal(view.displayCursor,2);
const fresh=clone(bundle.snapshot);
fresh.through_seq=26; fresh.revision=7;
fresh.presentation.records=[record('recovered',26,'Recovered text')];
transport.current=fresh;
now=2; transport.time=2;
await view.recover();
assert.equal(view.needRecovery,false);
assert.equal(view.displayCursor,26);
assert.equal(view.records.get('recovered').payload.text,'Recovered text');
assert.equal(view.applySnapshot(bundle.snapshot),false);
assert.equal(view.displayCursor,26);
assert.equal(transport.commands.length,0);
''')


def test_browser_gap_during_snapshot_read_fences_the_inflight_response():
    run_js(r'''
await view.mount();
view.event(event(2,{_type:'DisplayChanged',record:record('keep',2)}));
let resolve;
transport.snapshot=async()=>{transport.reads.snapshot++; return await new Promise(done=>resolve=done);};
view.needRecovery=true;
now=1;
const pending=view.recover();
transport.deliver({events:[event(30,{_type:'Gap',after_seq:2,through_seq:30,reason:'overflow'})]});
const old=clone(bundle.snapshot);
old.through_seq=10; old.revision=2;
resolve(old); await pending;
assert.equal(view.needRecovery,true);
assert.equal(view.recoveryFloor,30);
assert.equal(view.displayCursor,2);
assert.equal(view.records.has('keep'),true);
assert.equal(transport.commands.length,0);
''')


def test_browser_equal_watermark_state_event_does_not_redirty_a_projection():
    run_js(r'''
await view.mount();
const projection=project(bundle.snapshot);
projection.revision=6; projection.through_seq=14;
projection.live.identity.mycall='CURRENT';
view.applyProjection(projection);
assert.equal(view.displayCursor,1);
view.event(event(13,{_type:'DisplayChanged',record:record('unseen',13)}));
assert.equal(view.records.has('unseen'),true);
const old=clone(projection.live); old.identity.mycall='OLD';
view.event(stateEvent(14,6,['live','application'],old));
assert.equal(view.dirty.size,0);
assert.equal(view.state.live.identity.mycall,'CURRENT');
await view.pump();
assert.equal(transport.reads.projection,0);
''')


def test_browser_large_projections_and_recovery_share_encoded_byte_budget():
    run_js(r'''
await view.mount();
const projection=project(bundle.snapshot);
projection.revision=2; projection.through_seq=2;
projection.application.files.failure='é'.repeat(1950000);
transport.currentProjection=clone(projection);
view.event(stateEvent(2,2,['application']));
now=.2; transport.time=.2;
await view.refresh();
assert.equal(transport.reads.projection,1);
assert.equal(view.state.application.files.failure.length,1950000);
projection.revision=3; projection.through_seq=3;
transport.currentProjection=clone(projection);
view.event(stateEvent(3,3,['application']));
now=.4; transport.time=.4;
await view.refresh();
assert.equal(transport.reads.projection,1);
assert.equal(view.dirty.has('application'),true);
const recovery=clone(bundle.snapshot);
recovery.application=clone(projection.application);
recovery.revision=3; recovery.through_seq=3;
transport.current=recovery;
view.needRecovery=true;
now=1; transport.time=1;
await view.recover();
assert.equal(transport.reads.snapshot,1);
assert.equal(view.needRecovery,true);
now=1.1; transport.time=1.1;
await view.recover();
assert.equal(transport.reads.snapshot,2);
assert.equal(view.needRecovery,false);
assert.equal(view.dirty.size,0);
assert.equal(transport.commands.length,0);
''')


def test_browser_mock_bulk_budget_is_shared_between_views():
    run_js(r'''
const second=new ChatView(transport,{actor:'second',clock:()=>now});
await view.mount(); await second.mount();
const projection=project(bundle.snapshot);
projection.revision=2; projection.through_seq=2;
projection.application.files.failure='é'.repeat(1150000);
transport.currentProjection=clone(projection);
now=.2; transport.time=.2;
view.event(stateEvent(2,2,['application']));
second.event(stateEvent(2,2,['application']));
await view.refresh(); await second.refresh();
assert.equal(transport.reads.projection,1);
assert.equal(second.dirty.has('application'),true);
assert.match(second.error,/budget exhausted/);
now=1.2; transport.time=1.2;
await second.refresh();
assert.equal(transport.reads.projection,2);
assert.equal(second.dirty.size,0);
assert.equal(transport.commands.length,0);
''')


def test_browser_palette_submits_broadcast_clear_and_retune_yes_no():
    run_js(r'''
const {mountPalette} = await import(''' + json.dumps(MODULE.as_uri()) + r''');
class Node {
  constructor(tag) {this.tagName=tag; this.children=[]; this.handlers={}; this.value='0';}
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children=children;}
  setAttribute() {}
  addEventListener(name, fn) {this.handlers[name]=fn;}
  fire(name) {return this.handlers[name]({preventDefault(){}});}
  close() {}
}
const document={createElement: tag => new Node(tag)}, root=new Node('main');
const pending=[];
mountPalette(document, root, view, promise => pending.push(promise));
await view.mount();
const form=root.children[0].children[0], select=form.children[0], fields=form.children[1];
for (const [label, action] of [
 ['Acknowledge outstanding broadcasts', {_type:'BroadcastClear', kind:'broadcast.clear'}],
 ['Confirm broadcast retune', {_type:'BroadcastConfirm', kind:'broadcast.confirm', proceed:false}],
 ['Confirm broadcast retune', {_type:'BroadcastConfirm', kind:'broadcast.confirm', proceed:true}]
]) {
 select.value=String(select.children.findIndex(option => option.textContent===label));
 assert.notEqual(select.value, '-1');
 select.fire('change');
 if ('proceed' in action) fields.children[0].children[0].checked=action.proceed;
 transport.expectations.push({action});
 form.fire('submit');
 await pending.shift();
 assert.deepEqual(transport.commands.at(-1).action, action);
}
assert.equal(transport.commands.length, 3);
''')
