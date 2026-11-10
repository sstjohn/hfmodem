// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
/* Shared presentation helpers. No modem or hardware access. */
(() => {
  const key = node => node.nodeType === 1 ? node.id && `id:${node.id}` ||
    ['data-peer','data-message','data-mail','data-call','data-exchange','data-command','data-event','data-entry','data-part'].map(a => node.hasAttribute(a) ? `${node.tagName}:${a}:${node.getAttribute(a)}` : '').find(Boolean) ||
    node.getAttribute('name') && `${node.tagName}:name:${node.getAttribute('name')}` : null;
  function patch(parent, incoming) {
    const prior = [...parent.childNodes], used = new Set();
    for (const [i, fresh] of [...incoming.childNodes].entries()) {
      const identity = key(fresh);
      const old = identity ? prior.find(n => !used.has(n) && key(n) === identity) : prior[i];
      let current = old && !used.has(old) && old.nodeType === fresh.nodeType && old.nodeName === fresh.nodeName && key(old) === identity ? old : fresh;
      if (current === old) {
        if (current.nodeType === 3) { if (current.data !== fresh.data) current.data = fresh.data; }
        else if (current.nodeType === 1) {
          for (const attr of [...current.attributes]) if (!fresh.hasAttribute(attr.name)) current.removeAttribute(attr.name);
          for (const attr of [...fresh.attributes]) if (current.getAttribute(attr.name) !== attr.value) current.setAttribute(attr.name, attr.value);
          patch(current, fresh);
          if (typeof fresh.onclick === 'function') current.onclick = fresh.onclick;
          if (['INPUT','TEXTAREA','SELECT'].includes(current.tagName) && current.value !== fresh.value) current.value = fresh.value;
          if (current.tagName === 'INPUT') current.checked = fresh.checked;
        }
      }
      used.add(current);
      if (parent.childNodes[i] !== current) parent.insertBefore(current, parent.childNodes[i] || null);
    }
    for (const old of prior) if (!used.has(old) && old.parentNode === parent) old.remove();
  }
  function render(host, html) {
    const incoming = host.cloneNode(false);
    incoming.innerHTML = html;
    // Keep submitted field errors through status updates while the value is invalid.
    for(const field of host.querySelectorAll('[aria-invalid="true"]')) {
      const replacement=[...incoming.querySelectorAll('input,select,textarea')].find(n=>field.id?n.id===field.id:n.name===field.name&&n.form?.id===field.form?.id);
      if(!replacement?.validity||replacement.validity.valid)continue;
      for(const id of (field.getAttribute('aria-describedby')||'').split(' ')) {
        const error=document.getElementById(id);
        if(!error?.classList.contains('field-error'))continue;
        replacement.setAttribute('aria-invalid','true');
        const described=new Set((replacement.getAttribute('aria-describedby')||'').split(' ').filter(Boolean));described.add(id);
        replacement.setAttribute('aria-describedby',[...described].join(' '));
        replacement.closest('label')?.after(error.cloneNode(true));
      }
    }
    patch(host, incoming);
  }
  function focus(id) { document.getElementById(id)?.focus(); }
  function renderNodes(host, nodes) { const incoming = host.cloneNode(false); incoming.replaceChildren(...nodes); patch(host,incoming); }
  function bindDialogs() {
    document.querySelectorAll('dialog').forEach(dialog => {
      let invoker;
      dialog.addEventListener('beforetoggle', event => { if (event.newState === 'open') invoker = document.activeElement; });
      dialog.addEventListener('close', () => { if (invoker?.isConnected && !invoker.disabled) invoker.focus(); });
      dialog.addEventListener('keydown', event => {
        if (event.key !== 'Tab') return;
        const controls = [...dialog.querySelectorAll('button,input,select,textarea,a[href],summary,[tabindex="0"]')].filter(e => !e.disabled && e.getClientRects().length);
        const target = event.shiftKey && document.activeElement === controls[0] ? controls.at(-1) : !event.shiftKey && document.activeElement === controls.at(-1) ? controls[0] : null;
        if (target) { event.preventDefault(); target.focus(); }
      });
    });
  }
  function bindErrors() {
    document.addEventListener('invalid', event => {
      const field = event.target;
      if (!field.form) return;
      field.setAttribute('aria-invalid','true');
      const id = field.form.id + '-' + (field.id || field.name) + '-error';
      let message = document.getElementById(id);
      if (!message) { message = document.createElement('p'); message.id = id; message.className = 'field-error'; field.closest('label')?.after(message); }
      const label = field.labels?.[0]?.childNodes[0]?.textContent.trim() || field.getAttribute('aria-label') || field.name;
      message.textContent = `${label}: ${field.validationMessage}`;
      const described = new Set((field.getAttribute('aria-describedby') || '').split(' ').filter(Boolean)); described.add(id);
      field.setAttribute('aria-describedby', [...described].join(' '));
    }, true);
    document.addEventListener('input', event => {
      const field = event.target;
      if (field.getAttribute('aria-invalid') !== 'true') return;
      if (field.validity.valid) { field.removeAttribute('aria-invalid'); const message = document.getElementById(field.form.id + '-' + (field.id || field.name) + '-error'); if (message) message.textContent = ''; }
    });
  }
  window.HFAccessibility = {render, renderNodes, focus};
  bindDialogs(); bindErrors();
})();
