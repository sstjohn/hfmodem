// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {loadPublicQsos} from './public-qsos.mjs';

const $ = id => document.getElementById(id);
const state = {peer:'', protocol:'', role:'', band:'', search:'', since:'', until:'', sort:'startedUtc', direction:-1};
let rows = [], selected = null, order = '', loaded = false;
const nodes = new Map(), modeNodes = new Map(), countNodes = new Map();
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
export function selectPublicRows(records, filters) {
  return records.filter(row => (!filters.peer || row.peer === filters.peer) &&
    (!filters.protocol || row.protocol === filters.protocol) && (!filters.role || row.role === filters.role) &&
    (!filters.band || row.band === filters.band) && (!filters.search || row.peer.toLowerCase().includes(filters.search.toLowerCase())) &&
    (!filters.since || row.date >= filters.since) && (!filters.until || row.date <= filters.until));
}
export function mailTotals(records) {
  return records.reduce((total, row) => ({exchanges:total.exchanges+1, sent:total.sent+row.mail.sent,
    received:total.received+row.mail.received, delivered:total.delivered+row.mail.delivered}),
    {exchanges:0, sent:0, received:0, delivered:0});
}
function inspectorLink(row, label) {
  const link = el('a', label, 'inspect-link');
  link.href = row.inspectorHref;
  link.setAttribute('aria-label', `${label} ${row.protocol} ${row.role.toLowerCase()} with ${row.peer}`);
  link.onclick = event => {
    if (parent !== window && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && event.button === 0) {
      event.preventDefault();
      parent.postMessage({kind:'inspect-arm', arm:row.route}, location.origin);
    }
  };
  return link;
}
function fact(list, name, value) { list.append(el('dt', name), el('dd', value)); }
function openRow(route) {
  selected = selected === route ? null : route;
  for (const [id, node] of nodes) {
    const open = id === selected;
    node.detail.hidden = !open; node.tr.classList.toggle('open', open);
    node.button.setAttribute('aria-expanded', String(open));
  }
}
function makeRow(row) {
  const tr = el('tr', null, 'row'); tr.dataset.route = row.route;
  tr.append(el('td', `${row.date} ${row.time}`, 't'));
  const call = el('td', null, 'call'), button = el('button', row.peer, 'entry-open');
  button.type = 'button'; button.id = `open-${row.route}`;
  button.setAttribute('aria-label', `${row.peer} ${row.protocol} ${row.role.toLowerCase()} details`);
  button.setAttribute('aria-expanded', 'false'); button.setAttribute('aria-controls', `detail-${row.route}`);
  button.onclick = () => openRow(row.route); call.append(button); tr.append(call);
  for (const text of [row.protocol, row.role, row.band, row.speedLabel, row.mailLabel]) tr.append(el('td', text));
  const replay = el('td'); replay.append(inspectorLink(row, 'Inspect')); tr.append(replay);
  const detail = el('section', null, 'record-detail'); detail.id = `detail-${row.route}`; detail.hidden = true;
  const heading = el('h2', `${row.peer} · ${row.protocol} ${row.role.toLowerCase()}`); heading.id = `title-${row.route}`;
  detail.setAttribute('aria-labelledby', heading.id); detail.append(heading);
  const facts = el('dl');
  fact(facts, 'Participants', `${row.local} ↔ ${row.peer}`);
  fact(facts, 'Started (UTC)', row.startedUtc);
  fact(facts, 'Dial frequency', `${(row.dialHz/1000).toFixed(3)} kHz USB`);
  fact(facts, 'Listed bandwidth', `${row.bandwidthHz} Hz`);
  fact(facts, 'Recording duration', `${row.durationSeconds.toFixed(2)} seconds`);
  fact(facts, 'Mail', row.mailLabel);
  fact(facts, 'Acknowledged transmit', row.speed.acknowledged || 'Not established');
  fact(facts, 'Decoded receive', row.speed.decoded || 'Not established');
  fact(facts, 'Recording sources', row.sources.map(source => `${({'rig rx':'Rig receiver input','rig tap':'Radio monitor tap','sdr witness':'Remote KiwiSDR witness'})[source.kind]||source.kind}${source.audioAvailable?'':' (audio unavailable)'}`).join(' · '));
  detail.append(facts);
  if (row.note) detail.append(el('p', row.note, 'qualification'));
  const evidence = el('details'); evidence.open = true; evidence.append(el('summary', 'Replay coverage and limitations'));
  const limits = el('ul'); limits.append(...row.limits.map(limit => el('li', limit))); evidence.append(limits); detail.append(evidence);
  const actions = el('p', null, 'actions'); actions.append(inspectorLink(row, 'Open recorded replay'));
  const disclosure = el('a', 'Data-loss incident and surviving evidence'); disclosure.href = row.dataLossHref;
  disclosure.target = '_blank'; disclosure.rel = 'noopener'; actions.append(disclosure); detail.append(actions);
  $('details').append(detail); nodes.set(row.route, {tr, detail, button}); return tr;
}
function render() {
  const shown = selectPublicRows(rows, state);
  const sorted = [...rows].sort((a,b) => {
    const x = state.sort === 'band' ? parseInt(a.band) : a[state.sort];
    const y = state.sort === 'band' ? parseInt(b.band) : b[state.sort];
    return (x < y ? -1 : x > y ? 1 : 0) * state.direction;
  });
  const nextOrder = sorted.map(row => row.route).join('|');
  if (nextOrder !== order) { for (const row of sorted) $('rows').append(nodes.get(row.route).tr); order = nextOrder; }
  const visible = new Set(shown.map(row => row.route));
  for (const [route, node] of nodes) {
    node.tr.hidden = !visible.has(route);
    node.detail.hidden = route !== selected || !visible.has(route);
  }
  $('empty').hidden = shown.length > 0 || !loaded;
  $('counts').textContent = `${shown.length} exchanges · ${new Set(shown.map(row => row.peer)).size} stations`;
  if (loaded) { const message = `${shown.length} of 6 approved exchanges match the current filters.`;
    if ($('loadStatus').textContent !== message) $('loadStatus').textContent = message; }
  for (const button of document.querySelectorAll('[data-sort]')) {
    button.closest('th').setAttribute('aria-sort', button.dataset.sort === state.sort ? state.direction === 1 ? 'ascending' : 'descending' : 'none');
  }
  const total = mailTotals(shown);
  for (const [key, node] of countNodes) node.textContent = total[key];
  for (const [protocol, cells] of modeNodes) {
    const subtotal = mailTotals(shown.filter(row => row.protocol === protocol));
    for (const [key, cell] of Object.entries(cells)) cell.textContent = subtotal[key];
  }
}
function initialize() {
  nodes.clear(); modeNodes.clear(); countNodes.clear(); order = ''; selected = null;
  $('rows').replaceChildren(); $('details').replaceChildren(); $('reportCards').replaceChildren(); $('reportModes').replaceChildren();
  for (const row of rows) $('rows').append(makeRow(row));
  for (const key of ['peer','protocol','band']) {
    const select = $(key); while (select.options.length > 1) select.remove(1);
    const values = [...new Set(rows.map(row => row[key]))].sort((a,b) => key === 'band' ? parseInt(a)-parseInt(b) : a.localeCompare(b));
    select.append(...values.map(value => new Option(value, value))); select.value = state[key];
  }
  for (const [key, label] of [['exchanges','Recorded exchanges'], ['sent','Messages sent'], ['received','Messages received'], ['delivered','Destination receipts']]) {
    const card = el('div', null, 'report-card'), count = el('b', '0');
    card.append(count, el('span', label)); $('reportCards').append(card); countNodes.set(key,count);
  }
  for (const protocol of [...new Set(rows.map(row => row.protocol))].sort()) {
    const tr = el('tr'); tr.append(el('th',protocol)); tr.firstChild.scope = 'row'; const cells = {};
    for (const key of ['exchanges','sent','received','delivered']) { cells[key] = el('td','0'); tr.append(cells[key]); }
    modeNodes.set(protocol,cells); $('reportModes').append(tr);
  }
}
async function load() {
  const retryHadFocus = document.activeElement === $('retry');
  $('retry').disabled = true; $('loadStatus').textContent = 'Loading the six approved recordings…';
  try { rows = await loadPublicQsos(); initialize(); loaded = true; $('retry').hidden = true; render(); if (retryHadFocus) $('main').focus(); }
  catch (error) { rows = []; loaded = false; $('rows').replaceChildren(); $('details').replaceChildren();
    $('loadStatus').textContent = `The approved recordings could not be loaded. ${error.message}`; $('retry').hidden = false;
  } finally { $('retry').disabled = false; }
}
if (typeof document !== 'undefined') {
for (const key of ['peer','protocol','role','band','search','since','until']) $(key).addEventListener('input', () => {state[key]=$(key).value;render();});
$('reset').onclick = () => {for (const key of ['peer','protocol','role','band','search','since','until']) {state[key]='';$(key).value='';}render();};
$('moreFilters').onclick = () => {const expanded=$('extraFilters').hidden;$('extraFilters').hidden=!expanded;$('moreFilters').setAttribute('aria-expanded',String(expanded));};
for (const button of document.querySelectorAll('[data-sort]')) button.onclick = () => {state.direction=state.sort===button.dataset.sort?-state.direction:1;state.sort=button.dataset.sort;render();};
for (const [id, view] of [['bookView','book'],['summaryView','summary']]) $(id).onclick = () => {
  $('bookViewPanel').hidden = view !== 'book'; $('summary').hidden = view !== 'summary';
  $('bookView').setAttribute('aria-pressed',String(view==='book')); $('summaryView').setAttribute('aria-pressed',String(view==='summary'));
};
$('retry').onclick = load;
load();
}
