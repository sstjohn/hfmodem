// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Forwarding pages leave the browser at once (meta refresh plus location.replace), so
// there is no stable document to inject axe into. They are checked from the build
// output instead, and their destinations are audited as the routes they are.
import {readFile} from 'node:fs/promises';
import {relative, resolve} from 'node:path';

const refresh = /http-equiv="refresh" content="0; url=([^"]+)"/;
const unescape = text => text.replaceAll('&quot;','"').replaceAll('&#x27;',"'").replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&');

export const forwardTarget = text => { const match = refresh.exec(text); return match && unescape(match[1]); };

export async function checkForwardingPage(root, route, text, origin) {
 const problems = [];
 const target = forwardTarget(text);
 const base = new URL(route, origin + '/');
 const destination = new URL(target, base);
 const file = resolve(root, destination.pathname.slice(1), destination.pathname.endsWith('/') ? 'index.html' : '');
 if (relative(root, file).startsWith('..')) return [`destination ${destination.pathname} leaves the site`];
 const destinationText = await readFile(file, 'utf8').catch(() => null);
 if (destinationText === null) problems.push(`destination ${destination.pathname} is not a built page`);
 else if (forwardTarget(destinationText)) problems.push(`destination ${destination.pathname} forwards again`);
 const canonical = /<link rel="canonical" href="([^"]+)">/.exec(text);
 if (!canonical) problems.push('no canonical link');
 else if (new URL(unescape(canonical[1])).pathname !== destination.pathname)
  problems.push(`canonical ${canonical[1]} is not the destination ${destination.pathname}`);
 // Hand-written forwards rely on the meta refresh alone; the generated ones add a script that must agree.
 const script = /location\.replace\(("(?:[^"\\]|\\.)*")/.exec(text);
 if (script && JSON.parse(script[1]) !== target) problems.push('script forwards somewhere other than the meta refresh');
 const served = await fetch(base).then(r => r.status).catch(() => 0);
 if (served !== 200) problems.push(`served with status ${served}`);
 return problems;
}
