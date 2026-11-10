# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Draw the homepage exchange figures from their examples' events. Run from the repository root."""
from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'site/index.html'

FIGURES = {
    'pactor': dict(
        example='pactor3-send', window=(110, 151), peer='VE1YZ', levels=False,
        title='PACTOR-3 data, acknowledgements and changeover with VE1YZ',
        desc='Seconds 110–151. Peer controls answer local data. Three CRC-valid CS3 break-in packets '
             'precede the local acknowledgement and peer SL1 data. CS1 is the acknowledgement codeword; '
             'repeated CS1 replies during the changeover request do not advance its sequence.'),
    'vara': dict(
        example='vara-send', window=(60, 135), peer='NS0A', levels=True,
        title='VARA BW2300 connection and mail delivery to NS0A',
        desc='Seconds 60–135, from the connect response to the local disconnect request. '
             'Local transmissions on the upper row, NS0A on the lower. Labels give the speed level of '
             'each CRC-valid data frame, L4 to L9. One peer frame, possibly OFDM L12, failed CRC. '
             'Short control bursts fill the turns between data frames.'),
}


def events(name):
    return json.loads((ROOT / 'site/examples' / FIGURES[name]['example'] / 'events.json').read_text())


def diagram(name, events):
    spec = FIGURES[name]
    lo, hi = spec['window']
    x = lambda t: 140 + (t - lo) / (hi - lo) * 740
    svg = [f'<svg viewBox="0 0 900 195" role="img" aria-labelledby="{name}-exchange-title {name}-exchange-desc">',
           f'<title id="{name}-exchange-title">{spec["title"]}</title>',
           f'<desc id="{name}-exchange-desc">{spec["desc"]}</desc>']
    for y, label in [(48, 'W9SSJ · TX'), (121, f'{spec["peer"]} · RX')]:
        svg += [f'<text x="12" y="{y+14}" fill="currentColor" font-size="13">{label}</text>',
                f'<path d="M140 {y+10} H880" stroke="currentColor" opacity=".18"/>']
    for e in events:
        if e.get('timestamp_only') or not (e['t1'] > lo and e['t0'] < hi):
            continue
        y = 48 if e['direction'] == 'tx' else 121
        a, b = max(lo, e['t0']), min(hi, e['t1'])
        kind = e['kind'].removeprefix('tx ')
        title = html.escape(f"{e['t0']:.3f} s · {e['direction'].upper()} · {kind} · {e['status']}")
        svg.append(f'<rect data-event="{e["id"]}" x="{x(a):.2f}" y="{y}" width="{max(2,x(b)-x(a)):.2f}" height="20" rx="2" fill="currentColor" opacity="{.8 if e["direction"]=="rx" else .45}"><title>{title}</title></rect>')
        level = kind.split()[-1]
        if spec['levels'] and e['status'] == 'CRC pass' and re.fullmatch(r'L\d+', level):
            svg.append(f'<text x="{(x(a)+x(b))/2:.2f}" y="{y-6}" text-anchor="middle" fill="currentColor" font-size="11">{level}</text>')
    for t in range(lo, hi, 10):
        svg.append(f'<text x="{x(t):.2f}" y="181" text-anchor="middle" fill="currentColor" font-size="12">{t} s</text>')
    svg.append('</svg>')
    return ''.join(svg)


def place(page, name, svg):
    figure = rf'(<div id="evidence-{name}"><figure class="cycle">.*?)<svg.*?</svg>'
    page, count = re.subn(figure, lambda m: m[1] + svg, page, count=1, flags=re.S)
    if not count:
        raise ValueError(f'No evidence-{name} figure on the homepage')
    return page


if __name__ == '__main__':
    page = PAGE.read_text()
    for name in FIGURES:
        page = place(page, name, diagram(name, events(name)))
    PAGE.write_text(page)
