// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

// Pure protocol helpers. Importing this module never attaches a view or starts
// station work; the offline prototype retains its explicit MockTransport.
const clone = value => JSON.parse(JSON.stringify(value));
const encodedBytes = value => new TextEncoder().encode(JSON.stringify(value)).length;

export function supportsStationChat(snapshot) {
  const lifecycle = snapshot?.chat_lifecycle;
  return snapshot?.service?.schema === 2 &&
    typeof snapshot.service.incarnation === 'string' && !!snapshot.service.incarnation &&
    lifecycle?.incarnation === snapshot.service.incarnation &&
    Number.isSafeInteger(lifecycle.generation) && lifecycle.generation >= 0 &&
    ['idle', 'starting', 'ready', 'retiring', 'failed'].includes(lifecycle.state) &&
    (lifecycle.controller_id === null || typeof lifecycle.controller_id === 'string');
}

function bounded(envelope) {
  if (encodedBytes(envelope) > 4096) throw Error('command exceeds 4096 bytes');
  return envelope;
}

export function stationCommand(request, client) {
  if (request?._type !== 'Request' || !request.id || !request.controller_id ||
      !Number.isSafeInteger(request.link_generation) || !request.actor ||
      typeof request.action?.kind !== 'string' || !request.action.kind ||
      typeof client !== 'string' || !client) throw Error('typed Request and client required');
  const envelope = {id: request.id, client, kind: request.action.kind, args: clone(request)};
  if (request.control_id != null) envelope.control_id = request.control_id;
  if (request.control_generation != null) envelope.generation = request.control_generation;
  return bounded(envelope);
}

export function stationStart(snapshot, {id, client, control_id, generation}) {
  if (!supportsStationChat(snapshot)) throw Error('station chat capability unavailable');
  if (!id || !client || !control_id || !Number.isSafeInteger(generation) || generation < 1)
    throw Error('explicit live control credentials required');
  return bounded({id, client, kind: 'chat.start', control_id, generation,
    args: {expected_incarnation: snapshot.service.incarnation,
      expected_lifecycle_generation: snapshot.chat_lifecycle.generation,
      expected_controller_id: snapshot.chat_lifecycle.controller_id}});
}

export function stationCommandResult(event) {
  const result = event?.type === 'command' ? event.data : event;
  if (!result || !['accepted', 'running', 'done', 'refused', 'failed'].includes(result.state))
    throw Error('invalid station command result');
  // Startup cancellation is a failed station operation; its reason is retained.
  return clone(result);
}

export function capabilityAllows(capability) {
  return !!capability && ['allowed', 'check-on-execution'].includes(capability.availability) &&
    (capability.reservation == null || capability.reservation.state === 'available');
}

const terminalStation = new Set(['done', 'refused', 'failed']);
const readKinds = new Set(['chat.snapshot', 'chat.projection', 'chat.check', 'chat.outcome',
  'chat.help', 'heard.list', 'slots.list', 'files.list', 'vmail.list', 'vmail.read',
  'vmail.pending', 'broadcast.status', 'beacon.hop.status', 'email.status', 'pskreporter.status']);

export class StationEventBuffer {
  constructor(after = 0, close = () => {}) {
    this.after = after; this.items = []; this.size = 0; this.closed = false;
    this.overflow = false; this.onClose = close;
  }
  put(event, size = encodedBytes(event)) {
    if (this.closed) return;
    if (this.items.length >= 128 || this.size + size > 256 * 1024) {
      const first = this.items[0];
      const after = first?.payload?._type === 'Gap' ? first.payload.after_seq : first ? first.seq - 1 : this.after;
      event = {...event, kind: 'Gap', request_id: null,
        payload: {_type: 'Gap', after_seq: after, through_seq: event.seq, reason: 'station subscriber overflow'}};
      this.items = []; this.size = 0; this.overflow = true;
      size = encodedBytes(event);
    }
    if (size > 256 * 1024) { this.closed = true; this.onClose(); return; }
    this.items.push(clone(event)); this.size += size;
  }
  drain() {
    const result = this.items;
    this.items = []; this.size = 0;
    if (result.length) this.after = result.at(-1).seq;
    return result;
  }
  close() {
    if (this.closed) return;
    this.closed = true; this.items = []; this.size = 0; this.onClose();
  }
}

// IO is explicitly injected: state(), command(envelope), subscribe(callback),
// read(path). A ChatView supplies the shared projection/check/recovery rules.
// Construction and subscription never acquire control or start an attachment.
export class StationChatTransport {
  constructor(io, {client, credentials = null, id = () => globalThis.crypto.randomUUID(),
    clock = () => performance.now() / 1000, pause = ms => new Promise(resolve => setTimeout(resolve, ms)),
    timeout = 20} = {}) {
    for (const name of ['state', 'command', 'subscribe', 'read'])
      if (typeof io?.[name] !== 'function') throw Error(`injected ${name} required`);
    if (typeof client !== 'string' || !client) throw Error('station client required');
    this.io = io; this.client = client; this.credentials = credentials; this.id = id;
    this.clock = clock; this.pause = pause; this.timeout = timeout;
    this.identity = null; this.controllerId = null; this.linkGeneration = 0;
    this.lastInner = 0; this.subscribers = new Set(); this.detach = null;
    this.reading = false; this.tokens = 4 * 1024 * 1024; this.refilled = clock();
    this.outstanding = new Map(); this.closed = false;
  }
  subscribe(after = 0) {
    if (this.closed || this.subscribers.size >= 8) throw Error('station subscriber limit');
    const buffer = new StationEventBuffer(after, () => {
      this.subscribers.delete(buffer);
      if (!this.subscribers.size) { this.detach?.(); this.detach = null; }
    });
    this.subscribers.add(buffer);
    if (!this.detach) {
      const subscription = this.io.subscribe(event => this.receive(event));
      this.detach = typeof subscription === 'function' ? subscription : () => subscription?.close();
    }
    return buffer;
  }
  receive(outer) {
    if (outer.type === 'command') {
      const value = outer.data;
      if (value?.client === this.client && terminalStation.has(value.state) && this.outstanding.has(value.id))
        this.outstanding.set(value.id, clone(value));
      return;
    }
    let event;
    if (outer.type === 'gap') {
      event = {_type: 'EventEnvelope', controller_id: this.controllerId, seq: this.lastInner,
        kind: 'Gap', request_id: null, payload: {_type: 'Gap', after_seq: this.lastInner,
          through_seq: this.lastInner, reason: 'station replay gap'}};
    } else if (['chat.state', 'chat.event', 'chat.outcome'].includes(outer.type)) {
      event = outer.data;
      if (event?._type !== 'EventEnvelope' || !Number.isSafeInteger(event.seq)) return;
      if (this.controllerId && event.controller_id !== this.controllerId) {
        // An old controller cannot overwrite the attached view. Attachment to
        // a different controller requires an explicit new view/subscription.
        return;
      }
      this.lastInner = Math.max(this.lastInner, event.seq);
    } else return;
    const size = encodedBytes(outer) + new TextEncoder().encode(`id: ${outer.seq}\nevent: ${outer.type}\ndata: \n\n`).length;
    for (const buffer of this.subscribers) buffer.put(event, size);
  }
  async ensureIdentity() {
    if (!this.identity) {
      const state = await this.io.state();
      if (!supportsStationChat(state)) throw Error('station chat capability unavailable');
      this.identity = clone(state);
      this.controllerId = state.chat_lifecycle.controller_id;
    }
    return this.identity;
  }
  async read(path) {
    if (this.reading) throw Error('bulk read already pending');
    const now = this.clock(), cap = 4 * 1024 * 1024;
    this.tokens = Math.min(cap, this.tokens + Math.max(0, now - this.refilled) * cap);
    this.refilled = now;
    if (this.tokens < cap) throw Error('bulk read budget exhausted; retry after refill');
    this.reading = true;
    try {
      const result = await this.io.read(path), size = encodedBytes(result);
      if (size > cap) throw Error('bulk response exceeds 4 MiB');
      this.tokens -= size;
      return clone(result);
    } finally { this.reading = false; }
  }
  async snapshot() {
    await this.ensureIdentity();
    const value = await this.read('/chat/snapshot');
    if (value._type !== 'Snapshot' || value.controller_id !== this.controllerId)
      throw Error('snapshot controller mismatch');
    this.linkGeneration = value.link.generation;
    this.lastInner = Math.max(this.lastInner, value.through_seq);
    return value;
  }
  async projection() {
    await this.ensureIdentity();
    const value = await this.read('/chat/projection?' + new URLSearchParams({controller_id: this.controllerId}));
    if (value._type !== 'Projection' || value.controller_id !== this.controllerId)
      throw Error('projection controller mismatch');
    this.linkGeneration = value.live.link.generation;
    return value;
  }
  async outcome(id) {
    return this.read('/chat/result?' + new URLSearchParams({request_id: id}));
  }
  async page(read_id, cursor) {
    return this.read('/chat/result?' + new URLSearchParams({read_id, cursor}));
  }
  async waitOutcome(id) {
    const end = this.clock() + this.timeout;
    while (!this.closed && this.clock() < end) {
      const value = await this.outcome(id);
      if (value._type === 'ReadPage' || ['done', 'refused', 'failed', 'cancelled', 'unknown'].includes(value.state?.state))
        return value;
      await this.pause(50);
    }
    throw Error('semantic outcome unresolved; review before retry');
  }
  async command(request) {
    await this.ensureIdentity();
    if (this.outstanding.size >= 64) throw Error('station command queue full');
    const envelope = stationCommand(request, this.client);
    if (!readKinds.has(envelope.kind)) {
      const credentials = typeof this.credentials === 'function' ? this.credentials() : this.credentials;
      if (!credentials?.control_id || !Number.isSafeInteger(credentials.generation))
        return {id: request.id, queued: false, reason: 'not-controller'};
      if (request.control_id !== credentials.control_id || request.control_generation !== credentials.generation)
        return {id: request.id, queued: false, reason: 'authority-stale'};
      envelope.control_id = credentials.control_id; envelope.generation = credentials.generation;
    }
    this.outstanding.set(request.id, null);
    try {
      const ack = await this.io.command(envelope);
      let result = terminalStation.has(ack?.state) ? stationCommandResult(ack) : this.outstanding.get(request.id);
      const end = this.clock() + this.timeout;
      while (!result && !this.closed && this.clock() < end) {
        const lookup = await this.read('/commands/result?' + new URLSearchParams({
          incarnation: this.identity.service.incarnation, client: this.client, id: request.id}));
        if (lookup.status === 'found') result = stationCommandResult(lookup.result);
        else { await this.pause(50); result = this.outstanding.get(request.id); }
      }
      if (!result) throw Error('station command outcome unknown; do not retry automatically');
      if (result.id !== request.id || (result.client != null && result.client !== this.client))
        throw Error('station result identity mismatch');
      return {id: request.id, queued: result.state === 'done', reason: result.reason || result.detail};
    } finally { this.outstanding.delete(request.id); }
  }
  async check(action) {
    await this.ensureIdentity();
    const id = this.id();
    const receipt = await this.command({_type: 'Request', id, controller_id: this.controllerId,
      link_generation: this.linkGeneration, actor: this.client, control_id: null,
      control_generation: null, action: {_type: 'ChatCheck', kind: 'chat.check', action: clone(action)}});
    if (!receipt.queued) throw Error(receipt.reason);
    const end = this.clock() + this.timeout;
    while (!this.closed && this.clock() < end) {
      const outcome = await this.outcome(id);
      if (outcome._type !== 'NotFound' && terminalStation.has(outcome.state?.state)) {
        if (outcome.state.state !== 'done') throw Error(outcome.state.detail || 'check failed');
        return outcome.result;
      }
      await this.pause(50);
    }
    throw Error('capability check outcome unknown');
  }
  close() {
    this.closed = true;
    for (const buffer of [...this.subscribers]) buffer.close();
    this.detach?.(); this.detach = null; this.outstanding.clear();
  }
}
