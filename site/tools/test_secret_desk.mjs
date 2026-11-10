// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createCheatCode, installSecretDoor} from '../js/secret-desk.mjs';

function accepted(keys) {
  const code = createCheatCode();
  return keys.some((key, n) => code.push(key, n * 1000));
}

test('recognizes familiar cheats and generously remembered ones', () => {
  for (const keys of [
    ['ArrowUp','ArrowUp','ArrowDown','ArrowDown','ArrowLeft','ArrowRight','ArrowLeft','ArrowRight','b','a'],
    [...'uuddllrrab'], [...'uuddrrllBA'], [...'uuddlrrba'], [...'uuuddllrrab'],
    ['u','d','l','r','Enter'], ['u','u','d','d','l','r','l','r',' '],
    [...'uuddlrlrstart'], [...'uuddllrrbba'],
  ]) assert.ok(accepted(keys), keys.join(' '));
});

test('ordinary navigation, typing, incomplete codes do not open it', () => {
  for (const keys of [
    [...'about'], [...'uuddllrra'], [...'uuddlrlr'], [...'llllrrab'],
    ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'], [...'uuddxxllrrab'],
  ]) assert.equal(accepted(keys), false, keys.join(' '));
  const code = createCheatCode();
  for (const k of 'uuddllrr') assert.equal(code.push(k, 0), false);
  assert.equal(code.push('a', 46000), false);
  assert.equal(code.push('b', 47000), true);
});

test('normal typing, shortcuts, composition and held keys leave the door alone', () => {
  let handler, opened = 0, prevented = 0;
  const document = {
    addEventListener: (name, fn) => { assert.equal(name, 'keydown'); handler = fn; },
    removeEventListener: (name, fn) => { assert.equal(fn, handler); handler = null; },
  };
  const detach = installSecretDoor(document, () => opened++);
  const send = (key, extra = {}) => handler({key, timeStamp:0, composedPath:()=>[],
    preventDefault:()=>prevented++, ...extra});
  for (const extra of [
    {composedPath:()=>[{tagName:'INPUT'}]}, {composedPath:()=>[{tagName:'TEXTAREA'}]},
    {composedPath:()=>[{tagName:'SELECT'}]}, {composedPath:()=>[{isContentEditable:true}]},
    {composedPath:()=>[{getAttribute:()=> 'textbox'}]},
    {ctrlKey:true}, {metaKey:true}, {altKey:true}, {isComposing:true}, {repeat:true},
  ]) for (const key of 'uuddllrrab') send(key, extra);
  assert.equal(opened, 0);
  assert.equal(prevented, 0);
  for (const key of 'uuddllrrab') send(key);
  assert.equal(opened, 1);
  assert.equal(prevented, 1);
  for (const key of 'uuddlrlrba') send(key);
  assert.equal(opened, 1);
  detach();
  assert.equal(handler, null);
});

 test('letter shortcuts can be disabled and slow switch input has no deadline', () => {
  const code = createCheatCode();
  assert.equal([...'uuddllrrab'].some((k, n) => code.push(k, n * 600000)), true);
  let handler, count = 0;
  const doc = {addEventListener: (_, fn) => handler = fn};
  installSecretDoor(doc, () => count++, () => false);
  for (const key of 'uuddllrrab') handler({key});
  assert.equal(count, 0);
});
