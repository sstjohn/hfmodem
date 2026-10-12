// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Native scrolling regions remain usable without a pointing device.
function scrollRegions() { for (const node of document.querySelectorAll('.table-wrap,.table-scroll,.tbl-scroll,.tblwrap,.csbits,.spectrum-scroll,.scroll,pre')) {
  if (node.scrollWidth <= node.clientWidth) continue;
  node.tabIndex = 0;
  node.setAttribute('role', 'region');
  const caption = node.querySelector('caption')?.textContent;
  node.setAttribute('aria-label', caption || (node.tagName === 'PRE' ? 'Code or decoded bytes; scroll horizontally' : 'Diagram or table; scroll horizontally'));
} }

scrollRegions();
window.addEventListener('resize', scrollRegions);
