// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

const sections = ['live', 'application', 'receive', 'capabilities', 'requests'];
const terminal = new Set(['done', 'refused', 'failed', 'cancelled', 'unknown']);
const copy = value => JSON.parse(JSON.stringify(value));
const bytes = value => new TextEncoder().encode(JSON.stringify(value)).length;
const ordered = value => Array.isArray(value) ? value.map(ordered) :
  value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(name => [name, ordered(value[name])])) : value;
const key = value => JSON.stringify(ordered(value));
const stamp = (revision, seq) => ({revision, seq});
const covers = (left, right) => left.revision >= right.revision && left.seq >= right.seq;
const zero = () => stamp(-1, -1);
const bulkBytes = 4 * 1024 * 1024;

class ByteBudget {
  constructor(clock) { this.clock = clock; this.at = clock(); this.tokens = bulkBytes; }
  take(count) {
    const now = this.clock();
    this.tokens = Math.min(bulkBytes, this.tokens + Math.max(0, now - this.at) * bulkBytes);
    this.at = Math.max(this.at, now);
    if (count > this.tokens) return false;
    this.tokens -= count;
    return true;
  }
  refund(count) { this.tokens = Math.min(bulkBytes, this.tokens + count); }
}

export function literal(text) {
  return String(text ?? '').replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, '');
}

export function live(snapshot) {
  const result = {};
  for (const field of ['sampled_at', 'identity', 'link', 'operation', 'tx', 'policy',
    'beacon', 'broadcast', 'ownership']) result[field] = snapshot[field];
  for (const field of ['bbs_context', 'bbs_revision', 'bbs_complete', 'bbs_reconnect_required'])
    result[field] = snapshot.presentation[field];
  return result;
}

export function project(snapshot) {
  return {_type: 'Projection', controller_id: snapshot.controller_id,
    revision: snapshot.revision, through_seq: snapshot.through_seq,
    sampled_at: snapshot.sampled_at, utc: snapshot.utc, live: live(snapshot),
    application: snapshot.application, receive: snapshot.receive,
    capabilities: snapshot.capabilities, requests: snapshot.requests,
    omitted_sections: snapshot.omitted_sections};
}

export class EventBuffer {
  constructor(maxEvents = 128, maxBytes = 256 * 1024) {
    this.maxEvents = maxEvents;
    this.maxBytes = maxBytes;
    this.items = [];
    this.size = 0;
    this.closed = false;
    this.overflow = false;
  }
  put(event) {
    if (this.closed) return;
    let value = copy(event), size = bytes(value);
    if (this.items.length >= this.maxEvents || this.size + size > this.maxBytes) {
      const after = this.items.length ? this.items[0].seq - 1 : event.seq - 1;
      this.items = [];
      this.size = 0;
      this.overflow = true;
      value = {...event, kind: 'Gap', request_id: null,
        payload: {_type: 'Gap', after_seq: after, through_seq: event.seq,
          reason: 'subscriber overflow'}};
      size = bytes(value);
    }
    this.items.push(value);
    this.size += size;
  }
  drain() {
    const items = this.items;
    this.items = [];
    this.size = 0;
    return items;
  }
  close() { this.closed = true; this.items = []; this.size = 0; }
}

export class MockTransport {
  constructor(bundle) {
    if (!bundle.snapshot || bundle.snapshot.schema !== 1) throw Error('chat schema 1 required');
    this.current = copy(bundle.snapshot);
    this.currentProjection = project(this.current);
    this.frames = copy(bundle.frames || []);
    this.expectations = copy(bundle.expectations || []);
    this.outcomes = new Map(Object.entries(bundle.outcomes || {}));
    this.checkResults = copy(bundle.checks || []);
    this.subscribers = new Set();
    this.commands = [];
    this.reads = {snapshot: 0, projection: 0, outcome: 0, check: 0};
    this.ids = new Map();
    this.time = 0;
    this.bulkBudget = new ByteBudget(() => this.time);
  }
  subscribe(after = 0) {
    const buffer = new EventBuffer();
    buffer.after = after;
    this.subscribers.add(buffer);
    return buffer;
  }
  async snapshot() {
    this.chargeRead(this.current, 'snapshot');
    this.reads.snapshot++;
    return copy(this.current);
  }
  async projection() {
    this.chargeRead(this.currentProjection, 'projection');
    this.reads.projection++;
    return copy(this.currentProjection);
  }
  readSize(kind) { return bytes(kind === 'snapshot' ? this.current : this.currentProjection); }
  chargeRead(value, name) {
    const count = bytes(value);
    if (count > bulkBytes) throw Error(`${name} exceeds byte budget`);
    if (!this.bulkBudget.take(count)) throw Error('bulk read budget exhausted; retry after refill');
  }
  async check(action) {
    this.reads.check++;
    const result = this.checkResults.shift();
    if (!result) return {controller_id: this.currentProjection.controller_id,
      revision: this.currentProjection.revision, through_seq: this.currentProjection.through_seq,
      action: copy(action), capability: {kind: action.kind, availability: 'blocked',
        reason_code: 'fixture-check-unavailable', reason: 'This fixture has no parameter check for this action'}};
    if (key(result.action) !== key(action)) throw Error('unexpected capability check');
    return copy(result);
  }
  async outcome(id) {
    this.reads.outcome++;
    return copy(this.outcomes.get(id) || {_type: 'NotFound', id});
  }
  async command(request) {
    if (bytes(request) > 4096) return {id: request.id, queued: false, reason: 'envelope-too-large'};
    if (this.ids.has(request.id)) return {id: request.id,
      queued: key(this.ids.get(request.id)) === key(request), reason: 'duplicate'};
    const expected = this.expectations[0];
    if (!expected || key(expected.action) !== key(request.action))
      throw Error(`unexpected action ${request.action.kind}`);
    if (expected.at != null && expected.at > this.time) throw Error('scenario has not reached this action');
    for (const field of ['controller_id', 'link_generation', 'actor', 'control_id', 'control_generation']) {
      if (field in expected && expected[field] !== request[field]) throw Error(`unexpected ${field}`);
    }
    this.expectations.shift();
    this.commands.push(copy(request));
    this.ids.set(request.id, copy(request));
    if (expected.outcome) {
      const outcome = copy(expected.outcome);
      outcome.state.id = request.id;
      this.outcomes.set(request.id, outcome);
    }
    if (expected.frame) this.deliver(expected.frame);
    this.advance(0);
    return {id: request.id, queued: expected.queued !== false, reason: expected.reason || null};
  }
  deliver(frame) {
    if (frame.snapshot) this.current = copy(frame.snapshot);
    if (frame.projection) this.currentProjection = copy(frame.projection);
    else if (frame.snapshot) this.currentProjection = project(this.current);
    for (const [id, value] of Object.entries(frame.outcomes || {})) this.outcomes.set(id, copy(value));
    for (const event of frame.events || []) {
      for (const subscriber of this.subscribers) {
        if (!subscriber.closed && event.seq > subscriber.after) subscriber.put(event);
      }
    }
  }
  advance(seconds) {
    this.time += seconds;
    while (this.frames.length && this.frames[0].at <= this.time &&
      (this.frames[0].after_actions || 0) <= this.commands.length) this.deliver(this.frames.shift());
  }
}

export class ChatView {
  constructor(transport, {actor = 'browser', clock = () => performance.now() / 1000,
    changed = () => {}, id = () => globalThis.crypto.randomUUID()} = {}) {
    this.transport = transport;
    this.actor = actor;
    this.clock = clock;
    this.changed = changed;
    this.id = id;
    this.controllerId = null;
    this.state = {};
    this.applied = new Map(sections.map(section => [section, zero()]));
    this.dirty = new Map();
    this.records = new Map();
    this.outcomes = new Map();
    this.displayCursor = 0;
    this.evictedThrough = null;
    this.recoveryBuffer = new EventBuffer();
    this.needRecovery = true;
    this.recovering = null;
    this.projecting = null;
    this.checking = null;
    this.lastProjection = -Infinity;
    this.lastRecovery = -Infinity;
    this.lastCheck = -Infinity;
    this.pendingCheck = null;
    this.capabilityCheck = null;
    this.editorVersion = 0;
    this.editor = '';
    this.multiline = false;
    this.orphan = false;
    this.submitting = false;
    this.bbsPair = null;
    this.tab = 'Contact';
    this.filter = '';
    this.selection = null;
    this.closed = false;
    this.paused = false;
    this.unread = 0;
    this.error = null;
    this.recoveryFloor = 0;
    this.bulkBudget = new ByteBudget(clock);
    this.bulkReading = false;
  }
  async mount() {
    this.subscription = this.transport.subscribe(this.displayCursor);
    await this.recover();
  }
  async reconnect() {
    this.subscription?.close();
    this.subscription = this.transport.subscribe(this.displayCursor);
    this.needRecovery = true;
    this.bbsPair = null;
    await this.pump();
  }
  select(value) { this.selection = value; this.changed(this); }
  setFilter(value) { this.filter = value; this.changed(this); }
  setTab(value) { this.tab = value; this.changed(this); }
  resize(width, height) { this.size = [width, height]; this.changed(this); }
  edit(text, {paste = false} = {}) {
    if (bytes(text) > 64 * 1024) throw Error('draft exceeds 64 KiB');
    this.editor = text;
    this.editorVersion++;
    this.capabilityCheck = null;
    if (paste && /[\r\n]/.test(text)) this.multiline = true;
    if (this.tab === 'BBS' && !this.bbsPair && this.bbsEnabled()) {
      this.bbsPair = [this.state.live.bbs_context, this.state.live.bbs_revision];
    }
    this.changed(this);
  }
  markDirty(section, revision, seq) {
    const previous = this.dirty.get(section) || zero();
    this.dirty.set(section, stamp(Math.max(previous.revision, revision), Math.max(previous.seq, seq)));
  }
  applyProjection(projection) {
    if (projection.controller_id !== this.controllerId) return false;
    const incoming = stamp(projection.revision, projection.through_seq);
    let applied = false;
    for (const section of sections) {
      if (!covers(incoming, this.applied.get(section)) ||
          !covers(incoming, this.dirty.get(section) || zero())) continue;
      if (section === 'live') this.observeLink(projection.live);
      this.state[section] = copy(projection[section]);
      this.applied.set(section, incoming);
      this.dirty.delete(section);
      applied = true;
    }
    this.omittedSections = copy(projection.omitted_sections);
    this.changed(this);
    return applied;
  }
  observeLink(value) {
    if (this.state.live && this.state.live.link.generation !== value.link.generation && this.editor) {
      this.orphan = true;
      this.bbsPair = null;
    }
    if (this.bbsPending && key(this.bbsPending.pair) !== key([value.bbs_context, value.bbs_revision]))
      this.bbsPending = null;
  }
  applySnapshot(snapshot) {
    if (this.controllerId && snapshot.controller_id !== this.controllerId) return false;
    const applied = [...this.applied.values()];
    if (snapshot.through_seq < Math.max(this.displayCursor, this.recoveryFloor, ...applied.map(value => value.seq)) ||
        snapshot.revision < Math.max(...applied.map(value => value.revision))) return false;
    this.controllerId = snapshot.controller_id;
    const projection = project(snapshot);
    this.applyProjection(projection);
    this.records = new Map(snapshot.presentation.records.map(record => [record.id, copy(record)]));
    this.evictedThrough = snapshot.presentation.evicted_through;
    this.displayCursor = snapshot.through_seq;
    this.changed(this);
    return true;
  }
  event(event) {
    if (event.controller_id !== this.controllerId) return;
    const payload = event.payload;
    if (payload._type === 'Gap') {
      this.recoveryFloor = Math.max(this.recoveryFloor, event.seq, payload.through_seq);
      this.needRecovery = true;
      this.bbsPair = null;
      this.capabilityCheck = null;
      return;
    }
    if (event.seq <= this.displayCursor) return;
    if (payload._type === 'StateChanged') {
      const incoming = stamp(payload.revision, event.seq);
      for (const section of payload.changed_sections) {
        if (!covers(this.applied.get(section), incoming)) this.markDirty(section, payload.revision, event.seq);
      }
      if (payload.state && !covers(this.applied.get('live'), incoming) && covers(incoming, this.applied.get('live')) &&
          covers(incoming, this.dirty.get('live') || zero())) {
        this.observeLink(payload.state);
        this.state.live = copy(payload.state);
        this.applied.set('live', incoming);
        this.dirty.delete('live');
      }
    } else if (payload._type === 'DisplayChanged') {
      const record = payload.record, previous = this.records.get(record.id);
      if (!previous || previous.updated_seq < record.updated_seq) {
        this.records.set(record.id, copy(record));
        if (this.paused && !previous) this.unread++;
      }
      if (record.bbs_context === this.state.live?.bbs_context && record.bbs_revision != null &&
          record.bbs_revision > this.state.live.bbs_revision) {
        this.state.live = {...this.state.live, bbs_revision: record.bbs_revision};
        this.bbsPending = null;
      }
      while (this.records.size > 2000 || bytes([...this.records.values()]) > 2 * 1024 * 1024) {
        const first = this.records.keys().next().value, removed = this.records.get(first);
        this.evictedThrough = Math.max(this.evictedThrough || 0, removed.created_seq);
        this.records.delete(first);
        if (removed.bbs_context && removed.bbs_context === this.state.live.bbs_context) {
          this.needRecovery = true;
          this.bbsPair = null;
        }
      }
    } else if (payload._type === 'RequestChanged') {
      const request = payload.request;
      this.outcomes.set(request.id, {...this.outcomes.get(request.id), state: copy(request)});
      if (this.bbsPending?.id === request.id && terminal.has(request.state) && request.state !== 'done')
        this.bbsPending = null;
      if (!isRead(request.kind)) {
        const existing = this.state.requests || [];
        this.state.requests = existing.filter(item => item.id !== request.id).concat(copy(request));
        this.applied.set('requests', stamp(this.applied.get('requests').revision, event.seq));
      }
    } else if (payload._type === 'Lifecycle') {
      if (payload.state === 'closed') {
        this.closed = true;
        this.subscription?.close();
      } else if (payload.state === 'failed') this.error = payload.error || 'Shutdown failed';
    } else if (payload._type === 'TxTextOutcome') {
      const record = this.records.get(payload.text_id);
      if (record && record.updated_seq < event.seq) {
        this.records.set(record.id, {...record, tx: copy(payload), updated_seq: event.seq});
      }
    }
    this.displayCursor = event.seq;
    this.changed(this);
  }
  async recover() {
    if (this.recovering || this.clock() - this.lastRecovery < 1) return this.recovering;
    if (!this.reserveBulk('snapshot')) return;
    this.lastRecovery = this.clock();
    this.needRecovery = true;
    this.recovering = (async () => {
      try {
        const snapshot = await this.readBulk(() => this.transport.snapshot());
        for (const event of this.subscription.drain()) this.bufferRecovery(event);
        if (this.recoveryBuffer.overflow) {
          this.recoveryBuffer = new EventBuffer();
          return;
        }
        if (!this.applySnapshot(snapshot)) throw Error('snapshot does not cover the current controller and recovery watermark');
        this.needRecovery = false;
        const buffered = this.recoveryBuffer.drain();
        for (const event of buffered) {
          if (event.seq <= snapshot.through_seq) continue;
          if (this.needRecovery) this.bufferRecovery(event);
          else this.event(event);
        }
        for (const request of this.state.requests || []) {
          if (!terminal.has(request.state)) this.outcomes.set(request.id, await this.transport.outcome(request.id));
        }
      } catch (error) {
        this.error = String(error);
        this.needRecovery = true;
      } finally {
        this.recovering = null;
        this.changed(this);
      }
    })();
    return this.recovering;
  }
  async refresh() {
    if (this.projecting || !this.dirty.size || this.clock() - this.lastProjection < .2) return;
    if (!this.reserveBulk('projection')) return;
    this.lastProjection = this.clock();
    this.projecting = (async () => {
      try { this.applyProjection(await this.readBulk(() => this.transport.projection())); }
      catch (error) { this.error = String(error); }
      finally { this.projecting = null; }
    })();
    return this.projecting;
  }
  async pump() {
    if (this.closed) return;
    for (const event of this.subscription?.drain() || []) {
      if (this.needRecovery || this.recovering) this.bufferRecovery(event);
      else this.event(event);
    }
    if (this.needRecovery) await this.recover();
    else await this.refresh();
    await this.flushCheck();
  }
  bufferRecovery(event) {
    if (event.payload._type === 'Gap')
      this.recoveryFloor = Math.max(this.recoveryFloor, event.seq, event.payload.through_seq);
    this.recoveryBuffer.put(event);
    if (this.recoveryBuffer.overflow) this.recoveryFloor = Math.max(this.recoveryFloor, event.seq);
  }
  reserveBulk(kind) {
    const count = this.transport.readSize ? this.transport.readSize(kind) : bulkBytes;
    if (count > bulkBytes) { this.error = 'bulk read exceeds byte budget'; return false; }
    if (this.bulkReading || !this.bulkBudget.take(count)) return false;
    this.bulkReserved = count;
    this.bulkReading = true;
    return true;
  }
  async readBulk(read) {
    let count = this.bulkReserved;
    try {
      const value = await read();
      count = bytes(value);
      if (count > bulkBytes) throw Error('bulk read exceeds byte budget');
      return value;
    } finally {
      this.bulkBudget.refund(this.bulkReserved - count);
      this.bulkReading = false;
    }
  }
  preview(action) {
    this.pendingCheck = copy(action);
    this.capabilityCheck = null;
    return this.flushCheck();
  }
  async flushCheck() {
    if (this.checking || !this.pendingCheck || this.clock() - this.lastCheck < .2) return;
    const action = this.pendingCheck, version = this.editorVersion;
    this.pendingCheck = null;
    this.lastCheck = this.clock();
    this.checking = (async () => {
      try {
        const result = await this.transport.check(action);
        const revision = Math.max(...[...this.applied.values()].map(value => value.revision));
        if (result.controller_id === this.controllerId && result.revision >= revision &&
            version === this.editorVersion && key(result.action) === key(action) && !this.pendingCheck)
          this.capabilityCheck = copy(result);
      } catch (error) { this.error = String(error); }
      finally { this.checking = null; this.changed(this); }
    })();
    return this.checking;
  }
  bbsEnabled() {
    const current = this.state.live;
    return !!current && !this.bbsPending && !this.needRecovery && !this.recovering && !this.dirty.has('live') &&
      !this.dirty.has('application') && this.state.application?.bbs.remote_view && current.bbs_complete &&
      !current.bbs_reconnect_required && [...this.records.values()].some(record =>
        record.bbs_context === current.bbs_context);
  }
  async action(action) {
    if (!this.controllerId || this.closed) throw Error('view is not attached');
    if (!isRead(action.kind) && !['chat.cancel_wait', 'chat.close', 'chat.exit'].includes(action.kind) &&
        (this.needRecovery || this.recovering || this.dirty.size))
      throw Error('state needs synchronization');
    const ownership = this.state.live.ownership;
    const request = {_type: 'Request', id: this.id(), controller_id: this.controllerId,
      link_generation: this.state.live.link.generation, actor: this.actor,
      control_id: ownership.control_id, control_generation: ownership.control_generation,
      action: copy(action)};
    if (bytes(request) > 4096) throw Error('request exceeds 4096-byte envelope');
    const receipt = await this.transport.command(request);
    if (!receipt.queued) throw Error(receipt.reason);
    return request.id;
  }
  async completed(action) {
    const id = await this.action(action), result = await (this.transport.waitOutcome ?
      this.transport.waitOutcome(id) : this.transport.outcome(id));
    this.outcomes.set(id, result);
    if (result.state?.state !== 'done') throw Error(result.state?.detail || 'outcome unresolved; review before retry');
    return result.result;
  }
  async submit({explicit = false} = {}) {
    if (this.submitting) return false;
    if (this.orphan) throw Error('link changed; review the retained draft before a new composition');
    if (this.multiline && !explicit) { this.edit(this.editor + '\n'); return false; }
    this.submitting = true;
    const original = this.editor;
    try {
      if (this.tab === 'BBS') {
        if (!this.bbsEnabled() || !this.bbsPair || key(this.bbsPair) !==
            key([this.state.live.bbs_context, this.state.live.bbs_revision])) throw Error('BBS view changed; review input');
        const id = await this.action({_type: 'BBSInput', kind: 'bbs.input', context_id: this.bbsPair[0],
          expected_revision: this.bbsPair[1], text: original});
        this.bbsPending = {id, pair: this.bbsPair};
        this.bbsPair = null;
      } else {
        let draft = this.state.application.draft;
        if (draft) {
          if (draft.owner !== this.actor) throw Error('draft belongs to another view');
          const finish = original.trim() === '.' && !this.multiline;
          for (const line of finish ? [] : original.replace(/\r\n/g, '\n').split('\n')) {
            const result = await this.completed({_type: 'DraftAppend', kind: 'draft.append',
              draft_id: draft.id, expected_revision: draft.revision, line});
            if (this.needRecovery || this.closed || result?._type !== 'DraftAccepted' ||
                !result.draft || result.draft.id !== draft.id) throw Error('upload interrupted; review retained text');
            draft = result.draft;
          }
          if (finish || this.multiline) await this.action({_type: draft.kind === 'email' ? 'EmailSend' : 'VMailSend',
            kind: draft.kind === 'email' ? 'email.send' : 'vmail.send', draft_id: draft.id,
            expected_revision: draft.revision});
        } else {
          await this.action({_type: 'ChatSend', kind: 'chat.send', text: original});
        }
      }
      if (this.editor === original) this.edit('');
      this.multiline = false;
      return true;
    } finally { this.submitting = false; }
  }
  async exit() {
    if (this.exitPending) return this.exitPending;
    this.exitPending = (async () => {
      const id = await this.action({_type: 'ChatExit', kind: 'chat.exit'});
      this.exitRequest = id;
      const outcome = await (this.transport.waitOutcome ? this.transport.waitOutcome(id) : this.transport.outcome(id));
      this.outcomes.set(id, outcome);
      if (outcome.state?.state === 'done') {
        this.closed = true;
        this.subscription?.close();
      } else if (terminal.has(outcome.state?.state)) this.error = outcome.state.detail || 'Exit failed';
      this.changed(this);
      return id;
    })();
    return this.exitPending;
  }
}

const readKinds = new Set(['chat.snapshot', 'chat.projection', 'chat.check', 'chat.outcome',
  'chat.help', 'heard.list', 'slots.list', 'files.list', 'vmail.list', 'vmail.read',
  'vmail.pending', 'broadcast.status', 'beacon.hop.status', 'email.status', 'pskreporter.status']);
export const isRead = kind => readKinds.has(kind);

const ansiColor = (value, background) => {
  if (!Number.isInteger(value)) return null;
  const base = background ? 40 : 30, bright = background ? 100 : 90;
  if (value >= base && value < base + 8) return value - base;
  if (value >= bright && value < bright + 8) return value - bright + 8;
  return value >= 0 && value < 16 ? value : null;
};

export function appendSpans(document, parent, page) {
  for (const item of page.spans) {
    const span = document.createElement('span');
    span.textContent = literal(item.text);
    if (page.mode === 'ansi16') {
      const foreground = ansiColor(item.foreground, false), background = ansiColor(item.background, true);
      if (foreground !== null) span.classList.add(`fg-${foreground}`);
      if (background !== null) span.classList.add(`bg-${background}`);
      if (item.bold) span.classList.add('bold');
    }
    parent.append(span);
  }
}

const paletteActions = [
  ['Call', 'ChatCall', 'chat.call', [['peer', 'Station'], ['slot', 'Slot (blank for CF)', 'optional-number'], ['auto_qsy', 'Automatic QSY', 'boolean']]],
  ['CQ', 'ChatCQ', 'chat.cq', [['slot', 'Slot (blank for automatic)', 'optional-number'], ['repeat', 'Repeat', 'boolean']]],
  ['Answer heard station', 'ChatAnswer', 'chat.answer', [['heard_number', 'Heard number', 'number']]],
  ['QSY', 'ChatQSY', 'chat.qsy', [['slot', 'Slot', 'number']]],
  ['Auto answer', 'AutoSet', 'auto.set', [['on', 'On', 'boolean']]],
  ['Away', 'AwaySet', 'away.set', [['text', 'Away message']]],
  ['Back', 'AwayBack', 'away.back', []],
  ['Follow', 'FollowSet', 'follow.set', [['target', 'Slot, cf, all, kHz or off', 'target']]],
  ['Band', 'BandSet', 'band.set', [['band_m', 'Band (metres)', 'number']]],
  ['Band ready', 'BandReady', 'band.ready', [['ready', 'Ready', 'boolean']]],
  ['Beacon schedule', 'BeaconConfigure', 'beacon.configure', [['minutes', 'Minutes (blank for off)', 'optional-number']]],
  ['Beacon once', 'BeaconOnce', 'beacon.once', [['force', 'Override detected occupancy', 'boolean']]],
  ['Select broadcast slot', 'BroadcastSelect', 'broadcast.select', [['slot', 'Slot (blank for CF)', 'optional-number']]],
  ['Send broadcast', 'BroadcastSend', 'broadcast.send', [['text', 'Broadcast text']]],
  ['Broadcast off', 'BroadcastOff', 'broadcast.off', []],
  ['Acknowledge outstanding broadcasts', 'BroadcastClear', 'broadcast.clear', []],
  ['Confirm broadcast retune', 'BroadcastConfirm', 'broadcast.confirm', [['proceed', 'Proceed with pending retune', 'boolean']]],
  ['Compose VMail', 'VMailCompose', 'vmail.compose', [['subject', 'Subject']]],
  ['Compose email', 'EmailCompose', 'email.compose', [['recipient', 'Recipient'], ['reply_to', 'Reply address'], ['subject', 'Subject']]],
  ['Collect VMail', 'VMailCollect', 'vmail.collect', [['peer', 'Relay station']]],
  ['Dismiss mail notice', 'VMailDismiss', 'vmail.dismiss', [['peer', 'Relay station']]],
  ['Enter BBS', 'BBSEnter', 'bbs.enter', []],
  ['BBS directory', 'BBSList', 'bbs.list', []],
  ['Download BBS file', 'BBSGet', 'bbs.get', [['selection', 'Directory number or filename']]],
  ['Request file offer', 'BBSRequest', 'bbs.request', [['selection', 'Directory number or filename']]],
  ['Capture opaque file', 'BBSCapture', 'bbs.capture', []],
  ['BBS terminal', 'BBSTerm', 'bbs.term', [['mode', 'plain or ansi16'], ['width', 'Width', 'number'], ['height', 'Height', 'number']]],
  ['Reload local BBS', 'BBSReload', 'bbs.reload', []],
  ['Remote heard inquiry', 'InquireHeard', 'inquire.heard', []],
  ['Clear transmit buffer', 'BufferClear', 'buffer.clear', []],
  ['Verbose diagnostics', 'VerboseSet', 'verbose.set', [['on', 'On', 'boolean']]],
  ['Disable PSKReporter', 'PSKReporterOff', 'pskreporter.off', []],
];

export function mountPalette(document, root, view, run) {
  const dialog = document.createElement('dialog'), form = document.createElement('form'),
    select = document.createElement('select'), fields = document.createElement('section'),
    reason = document.createElement('p'), confirm = document.createElement('button'),
    dismiss = document.createElement('button');
  dialog.setAttribute('aria-label', 'Typed chat actions');
  select.setAttribute('aria-label', 'Action');
  confirm.textContent = 'Apply action';
  confirm.type = 'submit';
  dismiss.textContent = 'Cancel';
  dismiss.type = 'button';
  dismiss.addEventListener('click', () => dialog.close());
  paletteActions.forEach(([label], index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = label;
    select.append(option);
  });
  let inputs = [];
  function redraw() {
    fields.replaceChildren();
    const entry = paletteActions[Number(select.value)];
    inputs = entry[3].map(([name, title, type]) => {
      const label = document.createElement('label'), input = document.createElement('input');
      label.textContent = title + ' ';
      input.type = type === 'boolean' ? 'checkbox' : type?.includes('number') ? 'number' : 'text';
      input.setAttribute('aria-label', title);
      label.append(input);
      fields.append(label);
      return {name, type, input};
    });
    const capability = view.state.capabilities?.find(item => item.kind === entry[2]);
    reason.textContent = literal(capability?.reason ||
      (capability?.availability === 'check-on-execution' ? 'Backend checks again when executed' : 'Explicit action; backend rechecks guards'));
    confirm.disabled = capability?.availability === 'blocked';
  }
  select.addEventListener('change', redraw);
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (confirm.disabled) return;
    const entry = paletteActions[Number(select.value)], action = {_type: entry[1], kind: entry[2]};
    for (const {name, type, input} of inputs) {
      const value = input.value;
      action[name] = type === 'boolean' ? input.checked :
        type === 'optional-number' && value === '' ? null :
        type?.includes('number') ? Number(value) :
        type === 'target' ? (value === 'off' ? null : /^\d+$/.test(value) ? Number(value) : value) : value;
    }
    confirm.disabled = true;
    run(view.action(action).then(() => dialog.close()).finally(() => {confirm.disabled = false;}));
  });
  form.append(select, fields, reason, confirm, dismiss);
  dialog.append(form);
  root.append(dialog);
  return {open() { redraw(); dialog.showModal(); select.focus(); }, dialog};
}

export function mountChat(document, root, view) {
  const create = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const status = create('header', 'Offline Kestrel chat'), tabs = create('nav'),
    content = create('section', null, 'chat-content'), composer = create('textarea'),
    ownership = create('p'), detail = create('p'), controls = create('footer'), error = create('p', '', 'error');
  ownership.setAttribute('role', 'status');
  ownership.setAttribute('aria-live', 'polite');
  ownership.setAttribute('aria-atomic', 'true');
  tabs.setAttribute('aria-label', 'Chat views and actions');
  content.setAttribute('aria-label', 'Contact conversation');
  content.setAttribute('tabindex', '0');
  error.setAttribute('role', 'alert');
  composer.setAttribute('aria-label', 'Operator line or reviewed draft');
  composer.rows = 3;
  const run = promise => Promise.resolve(promise).catch(value => { view.error = String(value); render(); });
  let palette;
  const tabButtons = new Map();
  for (const name of ['Contact', 'BBS', 'Heard', 'Slots', 'Mail', 'Diagnostics']) {
    const button = create('button', name);
    button.setAttribute('aria-pressed', String(view.tab === name));
    button.addEventListener('click', () => view.setTab(name));
    tabButtons.set(name, button);
    tabs.append(button);
  }
  const commands = create('button', 'Commands (F2)');
  commands.addEventListener('click', () => palette.open());
  tabs.append(commands);
  const filter = create('input');
  filter.setAttribute('aria-label', 'Filter heard and slots');
  filter.addEventListener('input', () => view.setFilter(filter.value));
  tabs.append(filter);
  for (const [name, action] of [['Send', () => view.submit({explicit: true})],
    ['Cancel wait', () => view.action({_type: 'ChatCancelWait', kind: 'chat.cancel_wait'})],
    ['Close contact', () => view.action({_type: 'ChatClose', kind: 'chat.close', text: '73'})],
    ['Exit', () => view.exit()]]) {
    const button = create('button', name);
    button.addEventListener('click', () => run(action()));
    controls.append(button);
  }
  composer.addEventListener('input', () => view.edit(composer.value));
  composer.addEventListener('paste', event => {
    const pasted = event.clipboardData.getData('text/plain');
    if (/[\r\n]/.test(pasted)) view.multiline = true;
  });
  composer.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !view.multiline) {
      event.preventDefault(); run(view.submit());
    }
  });
  root.addEventListener('keydown', event => {
    if (event.ctrlKey) {
      const actions = {l: () => composer.focus(), s: () => view.submit({explicit: true}),
        x: () => view.action({_type: 'ChatCancelWait', kind: 'chat.cancel_wait'}),
        b: () => view.action({_type: 'ChatClose', kind: 'chat.close', text: '73'}),
        q: () => view.exit(), c: () => view.exit()};
      const action = actions[event.key.toLowerCase()];
      if (action) { event.preventDefault(); run(action()); }
    } else if (event.key === 'F2') {
      event.preventDefault(); palette.open();
    } else if (event.key === 'F3' || event.key === 'F4') {
      event.preventDefault(); view.setTab(event.key === 'F3' ? 'Diagnostics' : 'Mail');
    }
  });
  const positions = new Map();
  let renderedTab = view.tab;
  content.addEventListener('scroll', () => {
    view.paused = content.scrollTop + content.clientHeight < content.scrollHeight - 2;
    if (!view.paused) view.unread = 0;
    positions.set(renderedTab, content.scrollTop);
  });
  function render() {
    const current = view.state.live;
    if (!current) return;
    status.textContent = literal(`${current.identity.mycall} · ${current.identity.channel_hz ?? 'unknown'} Hz · ` +
      `${current.link.peer || 'no peer'} · ${current.operation.phase} · ${current.operation.description}\n` +
      `${current.tx.description}\n${current.operation.next_action}`);
    const broadcast = current.broadcast;
    if (broadcast.selected_hz !== null || broadcast.possibly_outstanding) {
      status.textContent += literal(`\nBroadcast: ${broadcast.phase || broadcast.result || 'idle'}; ` +
        `${broadcast.possibly_outstanding} possibly outstanding; reception unconfirmed`);
    }
    const owner = current.ownership;
    ownership.textContent = literal(`Ownership: ${owner.mode} / ${owner.health} · ` +
      `owner ${owner.actor ?? 'none'} · ${owner.accepting ? 'accepting actions' : 'actions closed'} · ` +
      (owner.mode === 'fake' ? 'offline fake; no station lease' :
        `lease ${owner.control_id ?? 'none'} · generation ${owner.control_generation ?? 'none'} · ` +
        `expires ${owner.lease_expires_utc ?? 'unknown'}`));
    for (const [name, button] of tabButtons) button.setAttribute('aria-pressed', String(view.tab === name));
    content.setAttribute('aria-label', `${view.tab} conversation and records`);
    detail.textContent = literal(`${view.multiline ? 'Reviewed body' : 'Operator line'} · ` +
      `${current.link.peer || 'no peer'} · generation ${current.link.generation} · ` +
      `draft revision ${view.state.application.draft?.revision ?? 'none'}${view.orphan ? ' · orphan draft: review required' : ''}`);
    composer.disabled = view.closed || view.submitting || (view.tab === 'BBS' && !view.bbsEnabled());
    if (composer.value !== view.editor) composer.value = view.editor;
    error.textContent = literal(view.error || (view.needRecovery ? 'History gap: synchronizing; BBS input disabled' : ''));
    positions.set(renderedTab, content.scrollTop);
    const tabChanged = renderedTab !== view.tab;
    renderedTab = view.tab;
    content.replaceChildren();
    if (view.evictedThrough != null) content.append(create('p', `History unavailable through ${view.evictedThrough}`));
    if (view.tab === 'Contact' || view.tab === 'BBS') {
      for (const record of view.records.values()) {
        const payload = record.payload, bbs = payload._type === 'BBSPage';
        if ((view.tab === 'BBS') !== bbs) continue;
        const row = create('pre');
        if (bbs) appendSpans(document, row, payload);
        else row.textContent = literal(payload.text);
        if (record.tx) row.append(create('small', ` [${record.tx.status}]`));
        content.append(row);
      }
    } else if (view.tab === 'Heard' || view.tab === 'Slots') {
      for (const item of view.state.receive[view.tab === 'Heard' ? 'heard' : 'slots']) {
        const text = view.tab === 'Heard' ? `${item.n} ${item.call} · ${item.frequency_hz} Hz · ${item.sn ?? '?'} dB` :
          `${item.slot ?? 'CF'} · ${item.state} · ${item.calls.join(', ') || 'identity unknown'}`;
        if (!text.toLowerCase().includes(view.filter.toLowerCase())) continue;
        const button = create('button', literal(text));
        button.addEventListener('click', () => view.select(item));
        content.append(button);
      }
    } else if (view.tab === 'Mail') {
      for (const item of view.state.application.mailbox) content.append(create('p', literal(`${item.sender}: ${item.subject}`)));
      for (const item of view.state.application.notices) content.append(create('p', literal(`${item.source} has mail for ${item.destination}`)));
      content.append(create('pre', literal(view.state.application.draft?.lines.join('\n') || 'No backend draft')));
    } else {
      content.append(create('pre', literal(JSON.stringify({operation: current.operation,
        receive: view.state.receive, capabilities: view.state.capabilities}, null, 2))));
    }
    if (tabChanged || view.paused) content.scrollTop = positions.get(view.tab) || 0;
    else content.scrollTop = content.scrollHeight;
  }
  view.changed = render;
  root.replaceChildren(status, ownership, tabs, content, detail, composer, controls, error);
  palette = mountPalette(document, root, view, run);
  return {render, composer, content, status, ownership};
}
