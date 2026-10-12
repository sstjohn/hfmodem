// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
import {PUBLIC_QSO_ROUTES, publicQsoRows, loadPublicQsos} from './demo/public-qsos.mjs';

const indexUrl = new URL('./examples/logbook/entries.json', import.meta.url);
const entries = JSON.parse(readFileSync(indexUrl));
const expected = [
  ['vara-send', 'NS0A', 'Send'], ['vara-receive', 'N5TW', 'Receive'],
  ['pactor3-send', 'VE1YZ', 'Send'], ['pactor3-receive', 'WS8EOC', 'Receive'],
  ['ardop-send', 'WW2MI', 'Send'], ['ardop-receive', 'W6IDS', 'Receive'],
];
const manifestUrl = route => new URL(`./examples/${route}/data/manifest.json`, import.meta.url);
const manifests = Object.fromEntries(expected.map(([route]) =>
  [route, JSON.parse(readFileSync(manifestUrl(route)))]));
const clone = value => JSON.parse(JSON.stringify(value));

function changedManifest(route, patch) {
  const changed = clone(manifests);
  patch(changed[route]);
  return changed;
}

test('public roster is exactly the six approved examples and links to existing inspectors', () => {
  const rows = publicQsoRows(entries, manifests);
  assert.deepEqual(rows.map(row => [row.route, row.peer, row.role]), expected);
  assert.deepEqual(PUBLIC_QSO_ROUTES, expected.map(([route]) => route));
  assert.deepEqual(publicQsoRows([...entries].reverse(), manifests), rows);
  for (const row of rows) {
    const entry = entries.find(entry => entry.route === row.route), manifest = manifests[row.route];
    assert.equal(row.speedLabel, entry.speed);
    assert.equal(row.mailLabel, entry.mail);
    assert.deepEqual(row.mail, manifest.session.mail);
    assert.deepEqual(row.limits, manifest.limits);
    assert.deepEqual(row.sources.map(source => source.kind), manifest.sources.map(source => source.kind));
    assert(existsSync(new URL(`./demo/${row.inspectorHref}index.html`, import.meta.url)));
    assert(existsSync(new URL(`./demo/${row.manifestHref}`, import.meta.url)));
    assert(Object.isFrozen(row) && Object.isFrozen(row.mail) && Object.isFrozen(row.limits));
  }
  assert(Object.isFrozen(rows));
  const logbook = readFileSync(new URL('./examples/logbook/index.html', import.meta.url), 'utf8');
  assert.match(logbook, /id="data-loss"/);
});

test('actual speeds and ARDOP outcomes remain distinct from playback coverage', () => {
  const rows = Object.fromEntries(publicQsoRows(entries, manifests).map(row => [row.route, row]));
  assert.equal(rows['vara-send'].speed.acknowledged, 'L9');
  assert.equal(rows['vara-receive'].speed.decoded, 'L4');
  assert.equal(rows['pactor3-send'].speed.acknowledged, 'P3 SL6');
  assert.equal(rows['pactor3-receive'].speed.decoded, 'P3 SL3');
  assert.deepEqual(rows['ardop-send'].mail, {sent:1, received:0, delivered:1});
  assert.equal(rows['ardop-send'].speed.acknowledged, '4FSK.200.50S');
  assert.equal(rows['ardop-send'].speed.decoded, '4PSK.200.100');
  assert.match(rows['ardop-send'].note, /Subsequent receive failed/);
  assert.match(rows['ardop-send'].limits.join(' '), /END was not confirmed/);
  assert.deepEqual(rows['ardop-receive'].mail, {sent:0, received:3, delivered:0});
  assert.equal(rows['ardop-receive'].speed.acknowledged, null);
  assert(rows['ardop-receive'].sources.every(source => source.kind === 'sdr witness'));
  assert.match(rows['vara-receive'].limits.join(' '), /single data-loss incident/);
});

test('extra, 148-record, old three-record, duplicate and unknown rosters are rejected', () => {
  for (const invalid of [
    [...entries, entries[0]],
    Array.from({length:148}, (_, n) => ({...entries[n % 6], route:`snapshot-${n}`})),
    entries.slice(0, 3),
    [...entries.slice(0, 5), entries[0]],
    entries.map((entry, n) => n ? entry : {...entry, route:'unapproved-example'}),
  ]) assert.throws(() => publicQsoRows(invalid, manifests), /Public logbook:/);
});

test('mismatched identities, roles, dates, bands and missing manifest evidence fail closed', () => {
  for (const patch of [entry => entry.peer = 'N0TEST', entry => entry.role = 'Receive',
                       entry => entry.protocol = 'ARDOP']) {
    const invalid = clone(entries); patch(invalid[0]);
    assert.throws(() => publicQsoRows(invalid, manifests), /identity or role/);
  }
  assert.throws(() => publicQsoRows(entries, {...manifests, 'vara-send':undefined}), /manifest does not match/);
  for (const patch of [m => m.called = 'N0TEST', m => m.caller = 'N0TEST',
                       m => m.example_role = 'receive', m => m.qso_only = false]) {
    assert.throws(() => publicQsoRows(entries, changedManifest('vara-send', patch)), /manifest does not match/);
  }
  for (const [patch, message] of [
    [m => m.started_utc = '2026-09-28T21:21:05Z', /start time mismatch/],
    [m => m.session.band_m = 20, /band mismatch/],
    [m => m.session.mail.delivered = 2, /mail outcome/],
    [m => m.session.mail.sent = 0, /mail outcome/],
    [m => m.limits = [], /limitations are missing/],
  ]) assert.throws(() => publicQsoRows(entries, changedManifest('vara-send', patch)), message);
});

test('private references in displayed labels, speed evidence, sources and limitations are rejected', () => {
  const privateReferences = [
    ['', 'Users', 'operator', 'private'].join('/'),
    ['', 'private', 'tmp', 'recording'].join('/'),
    ['working', 'not-public'].join('/'),
    ['', 'file', 'recording'].join('/'),
    ['', 'api', 'station'].join('/'),
  ];
  for (const value of privateReferences) {
    for (const field of ['note', 'speed', 'mail']) {
      const invalid = clone(entries); invalid[0][field] = value;
      assert.throws(() => publicQsoRows(invalid, manifests), /private reference/);
    }
    for (const patch of [m => m.limits = [value], m => m.session.speed.decoded = value,
                         m => m.sources[0].kind = value]) {
      assert.throws(() => publicQsoRows(entries, changedManifest('vara-send', patch)), /private reference/);
    }
  }
});

test('projection excludes unrelated snapshot fields, exact coordinates, credentials and audio URLs', () => {
  const extraEntries = entries.map(entry => ({...entry, transcript:'not-for-display',
    latitude:1.234567, longitude:2.345678, password:'fixture-not-a-secret', snapshot:[]}));
  const extraManifests = clone(manifests);
  for (const manifest of Object.values(extraManifests)) manifest.privateExtra = 'not-for-display';
  const rows = publicQsoRows(extraEntries, extraManifests);
  for (const row of rows) {
    for (const field of ['transcript', 'latitude', 'longitude', 'password', 'snapshot', 'recording', 'privateExtra'])
      assert.equal(Object.hasOwn(row, field), false);
    assert(row.sources.every(source => !Object.hasOwn(source, 'url')));
  }
  assert(!JSON.stringify(rows).includes('fixture-not-a-secret'));
});

test('loader fetches only canonical public roster plus six selected manifests', async () => {
  const requests = [];
  const rows = await loadPublicQsos(async url => {
    requests.push(url.href);
    assert(url.href === indexUrl.href || expected.some(([route]) => url.href === manifestUrl(route).href));
    return {ok:true, status:200, json:async () => JSON.parse(readFileSync(url))};
  });
  assert.deepEqual(requests, [indexUrl.href, ...expected.map(([route]) => manifestUrl(route).href)]);
  assert.deepEqual(rows, publicQsoRows(entries, manifests));
});

test('loader never follows an unapproved roster and reports missing public files without fallback', async () => {
  const requests = [];
  await assert.rejects(loadPublicQsos(async url => {
    requests.push(url.href);
    return {ok:true, status:200, json:async () => Array.from({length:148}, () => entries[0])};
  }), /exactly six/);
  assert.deepEqual(requests, [indexUrl.href]);
  await assert.rejects(loadPublicQsos(async () => ({ok:false, status:404})), /could not load.*404/);
  await assert.rejects(loadPublicQsos(async url => url.href === indexUrl.href
    ? {ok:true, status:200, json:async () => entries}
    : {ok:false, status:404}), /could not load.*404/);
});
