// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

// A cheat-code-shaped knock. No controller pedigree required.
// Accept arrows or U/D/L/R, left/right pairs in either order, AB or BA,
// and Enter/Space/S for Start. Extra direction presses and missing doubles
// are fine. Start is optional after both letter buttons.
export function createCheatCode() {
  let directions = '', buttons = new Set();
  const reset = () => { directions = ''; buttons.clear(); };
  function looksLikeCode() {
    const run = directions.match(/u{1,3}d{1,3}[lr]{2,6}$/)?.[0];
    return !!run && run.includes('l') && run.includes('r');
  }
  return {
    reset,
    push(key, now = Date.now()) {
      const k = key.toLowerCase();
      const direction = {arrowup:'u', arrowdown:'d', arrowleft:'l', arrowright:'r'}[k]
        || (/^[udlr]$/.test(k) ? k : null);
      if (direction) {
        if (buttons.size) directions = '';
        buttons.clear();
        directions = (directions + direction).slice(-12);
      } else if (k === 'a' || k === 'b') {
        if (looksLikeCode()) buttons.add(k);
        else reset();
        if (buttons.size === 2) { reset(); return true; }
      } else if (['enter', ' ', 'spacebar', 'start', 's'].includes(k)) {
        if (looksLikeCode()) { reset(); return true; }
      } else if (!['shift', 'tab', ',', '-'].includes(k)) {
        reset();
      }
      return false;
    },
  };
}

export function installSecretDoor(document, open, enabled = () => true) {
  const code = createCheatCode();
  let opened = false;
  const keydown = event => {
    if (!enabled()) { code.reset(); return; }
    const typing = event.composedPath().some(node => node?.isContentEditable
      || ['INPUT', 'TEXTAREA', 'SELECT'].includes(node?.tagName)
      || node?.getAttribute?.('role') === 'textbox');
    if (typing || event.ctrlKey || event.metaKey || event.altKey || event.isComposing) {
      code.reset();
      return;
    }
    if (opened || event.repeat) return;
    if (code.push(event.key, event.timeStamp)) {
      opened = true;
      event.preventDefault();
      open();
    }
  };
  document.addEventListener('keydown', keydown);
  return () => document.removeEventListener('keydown', keydown);
}

if (typeof document !== 'undefined') {
  const open = () => location.assign(new URL('./demo/', import.meta.url));
  const preference = document.getElementById('letter-shortcuts');
  let enabled = true;
  try { enabled = localStorage.getItem('hfmodem-letter-shortcuts') !== 'off'; } catch {}
  if (preference) {
    preference.checked = enabled;
    preference.addEventListener('change', () => {
      enabled = preference.checked;
      try { localStorage.setItem('hfmodem-letter-shortcuts', enabled ? 'on' : 'off'); } catch {}
    });
  }
  installSecretDoor(document, open, () => enabled);
  document.getElementById('shortcut-form')?.addEventListener('submit', event => {
    event.preventDefault();
    const code = createCheatCode();
    const value = document.getElementById('shortcut-sequence').value;
    if ([...value].some(key => code.push(key))) open();
    else document.getElementById('shortcut-status').textContent = 'Shortcut not recognized.';
  });
}
