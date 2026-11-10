// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Read-only adapter for the curated public logbook. No station services or audio copies.
const approved = Object.freeze([
  ['vara-send', 'NS0A', 'VARA', 'VARA', 'send'],
  ['vara-receive', 'N5TW', 'VARA', 'VARA', 'receive'],
  ['pactor3-send', 'VE1YZ', 'PACTOR-3', 'PACTOR', 'send'],
  ['pactor3-receive', 'WS8EOC', 'PACTOR-3', 'PACTOR', 'receive'],
  ['ardop-send', 'WW2MI', 'ARDOP', 'ARDOP', 'send'],
  ['ardop-receive', 'W6IDS', 'ARDOP', 'ARDOP', 'receive'],
].map(Object.freeze));

export const PUBLIC_QSO_ROUTES = Object.freeze(approved.map(([route]) => route));

function check(condition, message) {
  if (!condition) throw new Error(`Public logbook: ${message}`);
}

function publicText(value, field, optional = false) {
  if (optional && value == null) return '';
  check(typeof value === 'string' && (optional || value.length > 0), `${field} is missing`);
  check(!/(?:\/Users\/|\/private\/|working\/|\/file\/|\/api\/)/.test(value), `${field} contains a private reference`);
  return value;
}

function roster(entries) {
  check(Array.isArray(entries) && entries.length === approved.length, 'expected exactly six approved examples');
  const byRoute = new Map(entries.map(entry => [entry?.route, entry]));
  check(byRoute.size === approved.length, 'duplicate example route');
  return approved.map(([route, peer, protocol, manifestProtocol, role]) => {
    const entry = byRoute.get(route);
    check(entry && entry.peer === peer && entry.protocol === protocol && entry.role?.toLowerCase() === role,
          `unexpected identity or role for ${route}`);
    return {entry, route, peer, protocol, manifestProtocol, role};
  });
}

// Pure projection also allows build checks to use the approved files without networking.
export function publicQsoRows(entries, manifests) {
  return Object.freeze(roster(entries).map(({entry, route, peer, protocol, manifestProtocol, role}) => {
    const manifest = manifests[route];
    check(manifest && manifest.protocol === manifestProtocol && manifest.call === peer &&
          manifest.called === peer && manifest.caller === 'W9SSJ' && manifest.example_role === role &&
          manifest.qso_only === true, `manifest does not match ${route}`);
    const started = publicText(manifest.started_utc, `${route} start time`);
    check(entry.date === started.slice(0, 10) && entry.time === started.slice(11, 16), `start time mismatch for ${route}`);
    const session = manifest.session;
    check(session && entry.band === `${session.band_m} m`, `band mismatch for ${route}`);
    check(Number.isInteger(session.dial_hz) && session.dial_hz > 0, `dial is missing for ${route}`);
    check(Number.isFinite(manifest.duration_s) && manifest.duration_s > 0, `duration is missing for ${route}`);
    const counts = {};
    for (const key of ['sent', 'received', 'delivered']) {
      check(Number.isInteger(session.mail?.[key]) && session.mail[key] >= 0, `${route} ${key} count is missing`);
      counts[key] = session.mail[key];
    }
    check(counts.delivered <= counts.sent && counts[role === 'send' ? 'sent' : 'received'] > 0,
          `mail outcome does not establish the selected role for ${route}`);
    const speed = {};
    for (const key of ['acknowledged', 'decoded', 'top']) {
      speed[key] = session.speed?.[key] == null ? null : publicText(session.speed[key], `${route} ${key} speed`);
    }
    check(Array.isArray(manifest.limits) && manifest.limits.length > 0, `limitations are missing for ${route}`);
    const limits = Object.freeze(manifest.limits.map(value => publicText(value, `${route} limitation`)));
    const sources = Object.freeze((manifest.sources || []).map(source => Object.freeze({
      kind: publicText(source.kind, `${route} source kind`),
      audioAvailable: source.audio_available === true,
    })));
    return Object.freeze({
      route, protocol, role: entry.role, local: manifest.caller, peer,
      band: entry.band, date: entry.date, time: entry.time, startedUtc: started,
      dialHz: session.dial_hz, bandwidthHz: manifest.bw_hz, durationSeconds: manifest.duration_s,
      speedLabel: publicText(entry.speed, `${route} speed label`), speed: Object.freeze(speed),
      mailLabel: publicText(entry.mail, `${route} mail label`), mail: Object.freeze(counts),
      note: publicText(entry.note, `${route} note`, true), sources, limits,
      inspectorHref: `../examples/${route}/`,
      manifestHref: `../examples/${route}/manifest.json`,
      evidenceHref: '../examples/logbook/',
      dataLossHref: '../examples/logbook/#data-loss',
    });
  }));
}

async function readJson(href, fetcher) {
  const response = await fetcher(new URL(href, import.meta.url));
  check(response.ok, `could not load ${href} (${response.status})`);
  return response.json();
}

export async function loadPublicQsos(fetcher = globalThis.fetch) {
  const entries = await readJson('../../examples/logbook/entries.json', fetcher);
  const selected = roster(entries); // Reject unexpected data before requesting any manifests.
  const pairs = await Promise.all(selected.map(async ({route}) =>
    [route, await readJson(`../../examples/${route}/manifest.json`, fetcher)]));
  return publicQsoRows(entries, Object.fromEntries(pairs));
}
