// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
/* Summaries use the same entries as the map; no additional log store. */
const LogSummary = (() => {
  const peer = e => (e.call || '').toUpperCase().replace(/-\d+$/, '');
  const outcome = e => e.reached.includes('qso') ? 'imported'
    : e.reached.includes('connect') ? 'connected' : !e.ended_utc ? 'pending' : 'failed';
  const radius = (attempts, scaled) => scaled ? Math.min(16, 3 + 1.6 * Math.sqrt(attempts)) : 5;
  function tally(entries) {
    const count = kind => entries.filter(e => outcome(e) === kind).length;
    const connected = count('connected'), failed = count('failed'), pending = count('pending');
    const successes = entries.filter(e => ['connected', 'imported'].includes(outcome(e)));
    const durations = successes.map(e => e.duration_s).filter(n => n != null && n > 0);
    const distances = successes.map(e => e.distance_km).filter(n => n != null);
    return {
      entries: entries.length, peers: new Set(entries.map(peer).filter(Boolean)).size,
      successfulPeers: new Set(successes.map(peer).filter(Boolean)).size,
      attempts: connected + failed + pending, connected, failed, pending, imported: count('imported'),
      answered: entries.filter(e => e.answered).length,
      rate: connected + failed ? 100 * connected / (connected + failed) : null,
      meanDuration: durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null,
      longest: distances.length ? Math.max(...distances) : null,
      sent: entries.reduce((n, e) => n + (e.mail?.sent || 0), 0),
      received: entries.reduce((n, e) => n + (e.mail?.received || 0), 0),
      delivered: entries.reduce((n, e) => n + (e.mail?.delivered || 0), 0),
    };
  }
  function groups(entries, key, compare = (a, b) => a.localeCompare(b, undefined, {numeric: true})) {
    const buckets = new Map();
    for (const e of entries) {
      const label = key(e);
      if (!buckets.has(label)) buckets.set(label, []);
      buckets.get(label).push(e);
    }
    return [...buckets].sort(([a], [b]) => compare(a, b)).map(([label, rows]) => ({label, ...tally(rows)}));
  }
  const clocks = new Map();
  function hourAt(e, timeZone) {
    const date = new Date(e.started_utc);
    if (!clocks.has(timeZone)) clocks.set(timeZone, new Intl.DateTimeFormat('en-GB', {
      timeZone, hour: '2-digit', hourCycle: 'h23',
    }));
    return Number.isNaN(date.getTime()) ? null : Number(clocks.get(timeZone).format(date));
  }
  function inHours(e, from, through, timeZone) {
    if (from === '' && through === '') return true;
    const hour = hourAt(e, timeZone), start = from === '' ? 0 : Number(from), end = through === '' ? 23 : Number(through);
    return hour != null && (start <= end ? hour >= start && hour <= end : hour >= start || hour <= end);
  }
  function build(entries, timeZone = 'UTC') {
    const clock = new Intl.DateTimeFormat('en-GB', {timeZone, hour: '2-digit', hourCycle: 'h23'});
    const hourly = Array.from({length: 24}, () => []);
    for (const e of entries) {
      const date = new Date(e.started_utc);
      if (!Number.isNaN(date.getTime())) hourly[Number(clock.format(date))].push(e);
    }
    return {
      total: tally(entries),
      peers: groups(entries, e => `${peer(e)} · ${e.band_m == null ? 'Unknown band' : e.band_m + ' m'}`).sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1) || (b.connected + b.failed) - (a.connected + a.failed) || a.label.localeCompare(b.label)),
      bands: groups(entries, e => e.band_m == null ? 'Unknown' : `${e.band_m} m`, (a, b) => parseInt(b) - parseInt(a) || a.localeCompare(b)),
      protocols: groups(entries, e => e.protocol.toUpperCase()),
      presets: groups(entries, e => `${e.protocol.toUpperCase()} · ${e.level || 'Unknown preset'}`),
      speeds: groups(entries, e => `${e.protocol.toUpperCase()} · ${e.speed?.top || 'No recorded speed'}`),
      outcomes: groups(entries, e => `${e.protocol.toUpperCase()} · ${e.reached.at(-1) || 'No milestone'}`),
      hours: hourly.map((rows, hour) => ({label: `${String(hour).padStart(2, '0')}:00`, ...tally(rows)})),
    };
  }
  const columns = ['label', 'entries', 'peers', 'successfulPeers', 'attempts', 'answered', 'connected', 'failed', 'pending', 'rate', 'imported', 'sent', 'received', 'delivered', 'meanDuration', 'longest'];
  function csv(report, timeZone) {
    const cell = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
    const rows = [['section', 'time_zone', ...columns], ['total', timeZone, 'All filtered entries', ...columns.slice(1).map(k => report.total[k])]];
    for (const section of ['peers', 'bands', 'protocols', 'presets', 'speeds', 'outcomes', 'hours'])
      for (const row of report[section]) rows.push([section, timeZone, ...columns.map(k => row[k])]);
    return rows.map(row => row.map(cell).join(',')).join('\r\n') + '\r\n';
  }
  return {peer, outcome, radius, hourAt, inHours, tally, build, csv};
})();
if (typeof module !== 'undefined') module.exports = LogSummary;
