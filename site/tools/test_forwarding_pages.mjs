// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdir, mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {checkForwardingPage, forwardTarget} from './forwarding_pages.mjs';

const forwarding = (target, canonical) => `<!doctype html><html><head>
<link rel="canonical" href="https://example.test/${canonical}">
<meta http-equiv="refresh" content="0; url=${target}">
<script>location.replace(${JSON.stringify(target)} + location.hash)</script></head><body></body></html>`;

async function site(files) {
 const root = await mkdtemp(join(tmpdir(), 'forward-'));
 for (const [path, text] of Object.entries(files)) { await mkdir(join(root, path, '..'), {recursive:true}); await writeFile(join(root, path), text); }
 const server = createServer(async (req, res) => {
  const path = req.url.endsWith('/') ? req.url + 'index.html' : req.url;
  const text = await readFile(join(root, path)).catch(() => null);
  res.writeHead(text ? 200 : 404).end(text);
 });
 await new Promise(done => server.listen(0, '127.0.0.1', done));
 return {root, origin:`http://127.0.0.1:${server.address().port}`, close:() => server.close()};
}

test('a forwarding page whose destination, canonical and script agree passes without a browser', async () => {
 const s = await site({'old/index.html':forwarding('../new/', 'new/'), 'new/index.html':'<html></html>'});
 try {
  const text = await readFile(join(s.root, 'old/index.html'), 'utf8');
  assert.equal(forwardTarget(text), '../new/');
  assert.deepEqual(await checkForwardingPage(s.root, 'old/index.html', text, s.origin), []);
  const metaOnly = text.replace(/<script>.*<\/script>/, '');
  assert.deepEqual(await checkForwardingPage(s.root, 'old/index.html', metaOnly, s.origin), []);
 } finally { s.close(); }
});

test('missing destination, wrong canonical and diverging script are each reported', async () => {
 const s = await site({'old/index.html':'x', 'a/index.html':'x'});
 try {
  const gone = forwarding('../gone/', 'gone/');
  assert.match((await checkForwardingPage(s.root, 'old/index.html', gone, s.origin)).join(), /not a built page/);
  const wrong = forwarding('../a/', 'elsewhere/');
  assert.match((await checkForwardingPage(s.root, 'old/index.html', wrong, s.origin)).join(), /canonical .* is not the destination/);
  const split = forwarding('../a/', 'a/').replace('location.replace("../a/"', 'location.replace("../b/"');
  assert.match((await checkForwardingPage(s.root, 'old/index.html', split, s.origin)).join(), /script forwards somewhere other/);
 } finally { s.close(); }
});

test('a meta-refresh-only forwarding page with no script passes', async () => {
 const s = await site({'old/index.html':'x', 'new/index.html':'<html></html>'});
 try {
  const page = `<!doctype html><html><head>
<link rel="canonical" href="https://example.test/new/">
<meta http-equiv="refresh" content="0; url=../new/"></head><body></body></html>`;
  assert.deepEqual(await checkForwardingPage(s.root, 'old/index.html', page, s.origin), []);
 } finally { s.close(); }
});
