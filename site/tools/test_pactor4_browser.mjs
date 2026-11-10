// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import assert from 'node:assert/strict';

export async function checkPactor4(page, baseURL) {
  await page.goto(new URL('pactor-4/', baseURL).href);
  await page.locator('main').waitFor();
  assert.equal(await page.locator('main svg').count(), 5);
  assert.equal(await page.locator('main details[open]').count(), 0, 'Numerical tables start collapsed');
  for (const [id, value, output, text] of [
    ['p4-waveform-mode', 'long', 'p4-waveform-summary', /157735.*3\.286/],
    ['p4-level', '10', 'p4-constellation-summary', /32 ideal points/],
    ['p4-layout', 'long', 'p4-layout-summary', /4968.*5768/],
    ['p4-rate', '5/6', 'p4-puncture-summary', /C1 indices 0, 10.*C2 indices 4, 14/],
    ['p4-c2-size', '1160', 'p4-c2-summary', /22 ordered correction swaps/]
  ]) {
    await page.locator('#'+id).focus();
    await page.locator('#'+id).selectOption(value);
    assert.equal(await page.evaluate(()=>document.activeElement.id), id, 'Changing the diagram preserves selector focus');
    assert.match(await page.locator('#'+output).textContent(), text);
  }
  assert.equal(await page.locator('#p4-constellation-values tr').count(), 32);
  assert.equal(await page.locator('#p4-c2-values tr').count(), 64);
  await page.setViewportSize({width:320, height:900});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth), 'Hidden guide reflows at 320px');
  await page.setViewportSize({width:1280, height:900});
  return ['waveform', 'constellation', 'frame layout', 'parity selection', 'C2 reconstruction', '320px reflow'];
}
