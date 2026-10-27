# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Reacquire the homepage excerpt from retained audio. Run from the repository root.

The original export missed frequency-offset P3 controls and placed local PTT
intervals on an obsolete grid. Search windows come from session control times
and emitted-audio sample positions; they are hints, never synthetic frames.
Every replacement control must independently pass the acquisition checks.
"""
from pathlib import Path
import html
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/logbook_view'))
from vara_view import load_audio
from vara_signal import signal_view
from vara_packets import packet_bytes
from hfmodem.shrike import p3acquire, placement

EXAMPLE = ROOT / 'site/examples/pactor3-send'
# CS numbers are one-based here. Three independently decoded break-ins precede
# the local ACK; the session log only reported the first of those three.
RX = [(110.37, 1), (111.62, 6), (115.38, 1), (119.12, 4),
      (122.88, 1), (126.62, 1), (130.37, 1), (134.12, 3),
      (135.37, 3), (136.62, 3)]
TX = [(66, 6601525, 1), (67, 6661496, 6), (68, 6841405, 1),
      (69, 7021405, 1), (70, 7201375, 1), (71, 7381383, 1)]
NAMES = {1: 'ACK', 3: 'BREAK-IN', 4: 'SPEED-UP', 6: 'CYCLE-TOG'}


def acquire(audio, hint, shift, cs):
    lo = round((hint + shift - .12) * 48000)
    candidate = p3acquire.control_signal(
        audio[lo:lo + round(1.12 * 48000)], offsets=(-15., -10., 0.))
    if candidate is None or candidate.event.cs != cs - 1:
        raise ValueError(f'Control not independently recovered at {hint}: CS{cs}')
    if cs == 3 and not (candidate.event.packet and candidate.event.packet[3]):
        raise ValueError('Changeover body did not pass CRC')
    return candidate, round((lo + candidate.event.start) / 48000 - shift, 6)


def diagram(events):
    lo, hi = 110, 151
    x = lambda t: 140 + (t - lo) / (hi - lo) * 740
    svg = ['<svg viewBox="0 0 900 195" role="img" aria-labelledby="pactor-exchange-title pactor-exchange-desc">',
           '<title id="pactor-exchange-title">PACTOR-3 data, acknowledgements and changeover with VE1YZ</title>',
           '<desc id="pactor-exchange-desc">Seconds 110–151. Peer controls answer local data. Three CRC-valid CS3 break-in packets precede the local acknowledgement and peer SL1 data. CS1 is the acknowledgement codeword; repeated CS1 replies during the changeover request do not advance its sequence.</desc>']
    for y, label in [(48, 'W9SSJ · TX'), (121, 'VE1YZ · RX')]:
        svg += [f'<text x="12" y="{y+14}" fill="currentColor" font-size="13">{label}</text>',
                f'<path d="M140 {y+10} H880" stroke="currentColor" opacity=".18"/>']
    for e in events:
        if e.get('timestamp_only') or not (e['t1'] > lo and e['t0'] < hi):
            continue
        y = 48 if e['direction'] == 'tx' else 121
        a, b = max(lo, e['t0']), min(hi, e['t1'])
        title = html.escape(f"{e['t0']:.3f} s · {e['direction'].upper()} · {e['kind']} · {e['status']}")
        svg.append(f'<rect data-event="{e["id"]}" x="{x(a):.2f}" y="{y}" width="{max(2,x(b)-x(a)):.2f}" height="20" rx="2" fill="currentColor" opacity="{.8 if e["direction"]=="rx" else .45}"><title>{title}</title></rect>')
    for t in range(110, 151, 10):
        svg.append(f'<text x="{x(t):.2f}" y="181" text-anchor="middle" fill="currentColor" font-size="12">{t} s</text>')
    svg.append('</svg>')
    return ''.join(svg)


def repair():
    manifest = json.loads((EXAMPLE / 'manifest.json').read_text())
    sources = {s['id']: (load_audio(EXAMPLE / s['url'])[0], s['offset_s'])
               for s in manifest['sources']}
    events = json.loads((EXAMPLE / 'events.json').read_text())
    # Replace the faulty PTT records, preserving the independently decoded data.
    removed = {'ptt-59'} | {f'ptt-{n}' for n, _, _ in TX}
    events = [e for e in events if e['id'] not in removed and not e['id'].startswith('excerpt-cs-')]
    local = next(e for e in events if e['id'] == 'p3-00086')
    assert local['payloads'][0]['hex'] == '3a20573953' and local['payloads'][0]['status_byte'] == 0x21
    local.update(direction='tx', role='caller', preferred_source='source0', tx_confirmed=True,
                 attribution='Local TX59 payload and status match; also decoded from the local monitor. Original grid-based direction corrected.')
    recovered = []
    windows = [('rx', t, cs, f'rx-{i}') for i, (t, cs) in enumerate(RX)]
    windows += [('source0', sample / 48000, cs, f'tx-{n}') for n, sample, cs in TX]
    for source, hint, cs, key in windows:
        audio, shift = sources[source]
        c, t = acquire(audio, hint, shift, cs)
        ev = c.event
        e = dict(id=f'excerpt-cs-{key}', t0=t, t1=round(t + (placement.PACKET_S if cs == 3 else .21), 6),
                 kind=f'P3 CS{cs} {NAMES[cs]}', protocol='PACTOR-3',
                 control_number=cs, frame_type='data' if cs == 3 else 'control',
                 direction='rx' if source == 'rx' else 'tx', role='called' if source == 'rx' else 'caller',
                 status='CRC pass' if cs == 3 else 'decoded control · 0 bit errors',
                 detail=ev.text, origin='Frequency-compensated acquisition from retained audio',
                 source_id=source, preferred_source=source, symbols=None,
                 offset_hz=c.offset_hz, quality=c.quality,
                 timing_precision='Acquired control-symbol clock; nominal frame duration',
                 acquisition=dict(search_hint_s=hint, offsets_hz=[-15, -10, 0], source_sha256=next(s['sha256'] for s in manifest['sources'] if s['id']==source)))
        if cs == 3:
            e['payloads'] = [packet_bytes(ev.packet[2], 'PACTOR-3 changeover payload', status_byte=ev.packet[1])]
        if source != 'rx':
            e['tx_confirmed'] = True
        e['views'] = {}
        for sid, (samples, offset) in sources.items():
            name = f'symbols-{e["id"]}-{sid}.json'
            signal_view(samples, e['t0']+offset, e['t1']+offset, EXAMPLE/'data', name, sid)
            e['views'][sid] = dict(signal=name, context=False, frames=[], decoded=[])
        recovered.append(e)
    events = sorted(events + recovered, key=lambda e: e['t0'])
    (EXAMPLE/'events.json').write_text(json.dumps(events, separators=(',', ':'))+'\n')
    note = ('The homepage excerpt (110–151 s) was rechecked with frequency-compensated control acquisition: '
            '10 peer controls, including three CRC-valid break-ins, and five local controls recovered from audio. '
            'Local control timing and one data-frame direction corrected. The remainder retains the original export and may omit controls or contain grid-timing errors.')
    manifest['excerpt_review'] = dict(start_s=110, end_s=151, note=note, tool='site/tools/repair_pactor_excerpt.py')
    manifest['limits'] = [s for s in manifest['limits'] if not s.startswith(('The homepage excerpt', '225/225 TX intervals'))] + [note]
    manifest['timing'] = 'Original export used session-grid TX intervals; the 110–151 s excerpt now uses audio-acquired control clocks. See excerpt review and limits.'
    (EXAMPLE/'manifest.json').write_text(json.dumps(manifest, separators=(',', ':'))+'\n')
    page = ROOT/'site/index.html'
    s = page.read_text()
    pattern = r'(<figure class="cycle" id="evidence-pactor".*?)(<svg.*?</svg>)(.*?</figure>)'
    s = re.sub(pattern, lambda m: m[1]+diagram(events)+m[3], s, flags=re.S)
    start = s.index('<figure class="cycle" id="evidence-pactor"')
    end = s.index('</figure>', start)
    figure = re.sub(r'<figcaption><strong>Winlink mail with VE1YZ.</strong> One message sent, two received; acknowledged SL6 transmit. Replay includes receiver, local monitor and remote witness audio.</figcaption>', '<figcaption><strong>Winlink mail with VE1YZ.</strong> One message sent, two received; acknowledged SL6 transmit. Replay includes receiver, local monitor and remote witness audio.</figcaption>', s[start:end], flags=re.S)
    page.write_text(s[:start]+figure+s[end:])
    print(f'Recovered {len(recovered)} controls; updated replay and homepage.')


if __name__ == '__main__':
    repair()
