// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Native scrolling regions remain usable without a pointing device.
const generatedLabels = new WeakMap();
const headings = 'h1,h2,h3,h4,h5,h6';
const text = node => node?.textContent.trim().replace(/\s+/g, ' ') || '';

function precedingHeading(node) {
  for (let current = node; current; current = current.parentElement) {
    for (let sibling = current.previousElementSibling; sibling; sibling = sibling.previousElementSibling) {
      if (sibling.matches(headings)) return text(sibling);
      const nested = sibling.querySelectorAll(headings);
      if (nested.length) return text(nested[nested.length - 1]);
    }
  }
  return '';
}

export function scrollRegions(document) {
  const nodes = document.querySelectorAll('.table-wrap,.table-scroll,.tbl-scroll,.tblwrap,.csbits,.spectrum-scroll,.scroll,.binline,.barwrap,pre');
  for (const [index, node] of [...nodes].entries()) {
    if (node.scrollWidth <= node.clientWidth) continue;
    node.tabIndex = 0;
    node.setAttribute('role', 'region');
    const label = node.getAttribute('aria-label');
    if (node.getAttribute('aria-labelledby') || (label && label !== generatedLabels.get(node))) continue;
    const context = text(node.querySelector('caption')) || precedingHeading(node)
      || (node.tagName === 'PRE' ? 'Code or decoded bytes' : 'Diagram or table');
    const generated = `${context}; scroll horizontally (${index + 1})`;
    node.setAttribute('aria-label', generated);
    generatedLabels.set(node, generated);
  }
}

if (typeof document !== 'undefined') {
  scrollRegions(document);
  window.addEventListener('resize', () => scrollRegions(document));
}
