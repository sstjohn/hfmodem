// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {scrollRegions} from '../js/accessibility.mjs';

function element(tag, textContent = '', attrs = {}) {
  if (tag === 'DIV' && !attrs.class) attrs.class = 'table-scroll';
  return {tagName:tag, textContent, scrollWidth:500, clientWidth:300,
    parentElement:null, previousElementSibling:null,
    getAttribute:name => attrs[name] ?? null,
    setAttribute:(name, value) => { attrs[name] = value; },
    matches:selector => selector.split(',').some(part => part.startsWith('.')
      ? (attrs.class || '').split(' ').includes(part.slice(1)) : part === tag.toLowerCase()),
    querySelector:() => null, querySelectorAll:() => []};
}
const documentFor = nodes => ({querySelectorAll:selector => nodes.filter(node => node.matches(selector))});

test('multiple scrolling tables under one heading have distinct contextual names', () => {
  const heading = element('H2', '  PACTOR   results ');
  const first = element('DIV'), second = element('DIV'), third = element('PRE');
  first.previousElementSibling = heading;
  second.previousElementSibling = first;
  third.previousElementSibling = second;
  const nodes = [first, second, third];
  scrollRegions(documentFor(nodes));
  const names = nodes.map(node => node.getAttribute('aria-label'));
  assert.equal(new Set(names).size, 3);
  for (const node of nodes) {
    assert.match(node.getAttribute('aria-label'), /^PACTOR results; scroll horizontally/);
    assert.equal(node.getAttribute('role'), 'region');
    assert.equal(node.tabIndex, 0);
  }
});

test('table captions name generated regions and explicit accessible names survive', () => {
  const captioned = element('DIV');
  captioned.querySelector = () => element('CAPTION', 'Receiver comparison');
  const labeled = element('PRE', '', {'aria-label':'Decoded control bytes'});
  const referenced = element('DIV', '', {'aria-labelledby':'result-heading'});
  scrollRegions(documentFor([captioned, labeled, referenced]));
  assert.match(captioned.getAttribute('aria-label'), /^Receiver comparison;/);
  assert.equal(labeled.getAttribute('aria-label'), 'Decoded control bytes');
  assert.equal(referenced.getAttribute('aria-labelledby'), 'result-heading');
  assert.equal(referenced.getAttribute('aria-label'), null);
});

test('resize keeps generated names stable and respects a later author label', () => {
  const node = element('PRE');
  const document = documentFor([node]);
  scrollRegions(document);
  const name = node.getAttribute('aria-label');
  scrollRegions(document);
  assert.equal(node.getAttribute('aria-label'), name);
  node.previousElementSibling = element('H2', 'Updated results');
  scrollRegions(document);
  assert.match(node.getAttribute('aria-label'), /^Updated results; scroll horizontally/);
  node.setAttribute('aria-label', 'Received packet bytes');
  scrollRegions(document);
  assert.equal(node.getAttribute('aria-label'), 'Received packet bytes');
});

test('regions without horizontal overflow are left alone', () => {
  const node = element('PRE');
  node.scrollWidth = node.clientWidth;
  scrollRegions(documentFor([node]));
  assert.equal(node.tabIndex, undefined);
  assert.equal(node.getAttribute('role'), null);
  assert.equal(node.getAttribute('aria-label'), null);
});


test('byte rows and bitbar diagrams expose keyboard scrolling as named regions', () => {
  const bytes = element('DIV', '', {class:'binline'});
  const bits = element('DIV', '', {class:'barwrap'});
  scrollRegions(documentFor([bytes, bits]));
  for (const node of [bytes, bits]) {
    assert.equal(node.tabIndex, 0);
    assert.equal(node.getAttribute('role'), 'region');
    assert.ok(node.getAttribute('aria-label'));
  }
  assert.notEqual(bytes.getAttribute('aria-label'), bits.getAttribute('aria-label'));
});
