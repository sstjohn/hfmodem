// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {StationChatTransport, stationStart, supportsStationChat} from './station_chat.mjs';
import {ChatView, mountChat} from './chat_mock.mjs';

export class DeskChat {
  constructor(io, {client = globalThis.crypto.randomUUID(), id = () => globalThis.crypto.randomUUID(),
    schedule = (work, ms) => {const timer = setInterval(work, ms); return () => clearInterval(timer);},
    changed = () => {}, render = () => {}} = {}) {
    this.io = io; this.client = client; this.id = id; this.schedule = schedule;
    this.changed = changed; this.render = render; this.state = null; this.credentials = null;
    this.view = null; this.transport = null; this.cancelHeartbeat = null; this.cancelPump = null;
    this.busy = false; this.error = null; this.closed = false;
  }
  async open() {
    if (this.closed) throw Error('Chat view closed');
    this.state = await this.io.state();
    if (!supportsStationChat(this.state)) throw Error('Station chat unavailable');
    const controller = this.state.chat_lifecycle.controller_id;
    if (controller && controller !== this.view?.controllerId) {
      this.cancelPump?.(); this.transport?.close();
      this.transport = new StationChatTransport(this.io, {client: this.client,
        credentials: () => this.credentials, id: this.id});
      this.view = new ChatView(this.transport, {actor: this.client, id: this.id});
      this.render(this.view);
      await this.view.mount();
      this.cancelPump = this.schedule(() => {
        if (!this.pumping) this.pumping = this.view.pump().catch(error => {
          this.error = String(error); this.changed(this);
        }).finally(() => {this.pumping = null;});
      }, 100);
    }
    this.changed(this);
    return this.state;
  }
  async acquire() {
    if (this.busy) throw Error('Chat operation already pending');
    this.busy = true;
    try {
      const result = await this.io.command({id: this.id(), client: this.client, kind: 'control', args: {}});
      if (this.closed) throw Error('Chat view closed');
      const control = result.result || result;
      if (!control.control_id || !Number.isSafeInteger(control.generation)) throw Error('Control acquisition failed');
      this.credentials = {control_id: control.control_id, generation: control.generation};
      this.cancelHeartbeat?.();
      const lease = this.credentials;
      this.cancelHeartbeat = this.schedule(async () => {
        if (this.credentials !== lease || this.heartbeating) return;
        this.heartbeating = true;
        try {
          const result = await this.io.command({id: this.id(), client: this.client, kind: 'heartbeat',
            args: {}, control_id: lease.control_id, generation: lease.generation});
          if (result.state && result.state !== 'done') throw Error(result.reason || 'Lease heartbeat refused');
        } catch (error) {
          if (this.credentials === lease) {
            this.credentials = null; this.cancelHeartbeat?.(); this.cancelHeartbeat = null;
            this.error = `Control expired or replaced: ${error}`; this.changed(this);
          }
        } finally {this.heartbeating = false;}
      }, 10000);
      await this.open();
    } finally {this.busy = false; this.changed(this);}
  }
  async start() {
    if (this.busy || !this.credentials) throw Error('Acquire control explicitly before Start');
    this.busy = true;
    try {
      // A fresh explicit click captures a fresh generation. This envelope is
      // sent exactly once; failures never silently refresh it or resubmit.
      this.state = await this.io.state();
      if (this.closed) throw Error('Chat view closed');
      const envelope = stationStart(this.state, {id: this.id(), client: this.client, ...this.credentials});
      const result = await this.io.command(envelope);
      if (result.state && result.state !== 'done') throw Error(result.detail || result.reason || 'Start failed');
      await this.open();
    } finally {this.busy = false; this.changed(this);}
  }
  async stop() {
    return this.io.command({id: this.id(), client: this.client, kind: 'stop', args: {}});
  }
  close() {
    this.closed = true;
    this.cancelHeartbeat?.(); this.cancelPump?.(); this.transport?.close();
    this.credentials = null;
  }
}

export function mountDeskChat(document, root, io, options = {}) {
  const create = (tag, text) => {const node = document.createElement(tag); node.textContent = text || ''; return node;};
  const status = create('p', 'Open Chat to inspect station availability.'), controls = create('div'),
    acquire = create('button', 'Acquire Control'), start = create('button', 'Start Chat'),
    stop = create('button', 'Station STOP'),
    content = create('section'), error = create('p');
  error.setAttribute('role', 'alert'); controls.append(acquire, start, stop);
  root.replaceChildren(status, controls, content, error);
  const desk = new DeskChat(io, {...options, render: view => mountChat(document, content, view),
    changed: model => {
      const lifecycle = model.state?.chat_lifecycle;
      status.textContent = lifecycle ? `Chat ${lifecycle.state} · ${model.credentials ? 'Control held by this view' : 'Observer'}` : 'Station chat unavailable';
      acquire.disabled = model.closed || !supportsStationChat(model.state) || model.busy || !!model.credentials;
      start.disabled = model.busy || !model.credentials || !['idle', 'ready'].includes(lifecycle?.state);
      error.textContent = model.error || '';
      options.changed?.(model);
    }});
  const run = action => Promise.resolve().then(action).catch(value => {desk.error = String(value); desk.changed(desk);});
  acquire.addEventListener('click', () => run(() => desk.acquire()));
  start.addEventListener('click', () => run(() => desk.start()));
  stop.addEventListener('click', () => run(() => desk.stop()));
  acquire.disabled = start.disabled = true;
  return desk;
}

export function browserStationIO({fetch = globalThis.fetch?.bind(globalThis),
  EventSource = globalThis.EventSource, pause = ms => new Promise(resolve => setTimeout(resolve, ms)),
  requestTimeout = 5000} = {}) {
  let incarnation = null;
  async function read(path, options = {}) {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), requestTimeout);
    try {
    const response = await fetch(path, {cache: 'no-store', ...options, signal: abort.signal});
    if (!response.ok) {
      const error = Error(`HTTP ${response.status}: ${await response.text()}`);
      if (response.status === 429) error.retryAfter = Number(response.headers.get('Retry-After')) || 1;
      throw error;
    }
    const size = Number(response.headers.get('Content-Length'));
    if (size > 4 * 1024 * 1024) throw Error('Response exceeds bulk limit');
    const body = await response.text();
    if (new TextEncoder().encode(body).length > 4 * 1024 * 1024) throw Error('Response exceeds bulk limit');
    return JSON.parse(body);
    } finally {clearTimeout(timer);}
  }
  return {
    async state() {const value = await read('/state'); incarnation = value.service.incarnation; return value;},
    read,
    async command(envelope) {
      const body = JSON.stringify(envelope);
      if (new TextEncoder().encode(body).length > 4096) throw Error('Command exceeds 4096 bytes');
      const ack = await read('/commands', {method: 'POST', headers: {'Content-Type': 'application/json'}, body});
      const exactIncarnation = ack.incarnation || incarnation;
      const deadline = Date.now() + 20000;
      for (let attempt = 0; attempt < 400 && Date.now() < deadline; attempt++) {
        let result;
        try {
          result = await read('/commands/result?' + new URLSearchParams({
            incarnation: exactIncarnation, client: envelope.client, id: envelope.id}));
        } catch (error) {
          if (!error.retryAfter) throw error;
          await pause(error.retryAfter * 1000); continue;
        }
        if (result.status === 'found') return result.result;
        await pause(50);
      }
      throw Error('Command result unknown; review before retry');
    },
    subscribe(handler) {
      const stream = new EventSource('/events');
      for (const kind of ['chat.state', 'chat.event', 'chat.outcome', 'command', 'gap'])
        stream.addEventListener(kind, event => handler(JSON.parse(event.data)));
      stream.onerror = () => handler({type: 'gap', seq: 0, data: {reason: 'station stream disconnected'}});
      return () => stream.close();
    }
  };
}
