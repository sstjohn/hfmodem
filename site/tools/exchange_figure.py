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
        example='pactor3-long', window=(90, 135), peer='VE3KPG',
        title='PACTOR-3 long-cycle mail from W9SSJ to VE3KPG, SL1 to SL3',
        desc='Seconds 90–135. VE3KPG completes a long-cycle packet. W9SSJ breaks in and sends its message '
             'in long-cycle packets: six at SL1, two at SL2 and two at SL3. The last SL3 packet is empty. '
             'VE3KPG answers each packet with a control signal. Two of these signals tell W9SSJ to '
             'increase the speed level. Near 133 s VE3KPG breaks in and starts to send data. The figure does '
             'not show packets that failed CRC.'),
    'vara': dict(
        example='vara-send', window=(60, 135), peer='NS0A',
        title='VARA BW2300 connection and mail delivery to NS0A',
        desc='Seconds 60–135, from the connect response to the local disconnect request. '
             'Local transmissions on the upper row, NS0A on the lower. Labels give the speed level of '
             'each CRC-valid data frame, L4 to L9. One peer frame, possibly OFDM L12, failed CRC. '
             'Short control bursts fill the turns between data frames.'),
}


def events(name):
    return json.loads((ROOT / 'site/examples' / FIGURES[name]['example'] / 'events.json').read_text())


def drawn(e, lo, hi):
    """Instants and unattributed activity would draw as slivers on the peer's row."""
    return e['t1'] > e['t0'] and e['status'] != 'signal candidate' and e['t1'] > lo and e['t0'] < hi


def label(e, events):
    """The speed level of a data packet that carries bytes, or what a closing burst says."""
    if e['kind'].startswith('ID '):
        return 'ID'
    # A break-in's decode is an instant inside its keyed interval; its bytes name the B2 command.
    for inner in events:
        if inner['t1'] <= inner['t0'] and e['t0'] <= inner['t0'] <= e['t1'] and inner['direction'] == e['direction']:
            said = re.search(r"type=6 (QRT)|b'(F[FQ])\\r'", inner['detail'])
            if said:
                return said[1] or said[2]
    level = re.search(r'\b(S?L\d+)(?: data)?$', e['kind'])
    if level and e['status'] == 'CRC pass' and not e['detail'].startswith('0 payload bytes'):
        return level[1]


def diagram(name, events):
    spec = FIGURES[name]
    lo, hi = spec['window']
    x = lambda t: 140 + (t - lo) / (hi - lo) * 740
    svg = [f'<svg viewBox="0 0 900 195" role="img" aria-labelledby="{name}-exchange-title {name}-exchange-desc">',
           f'<title id="{name}-exchange-title">{spec["title"]}</title>',
           f'<desc id="{name}-exchange-desc">{spec["desc"]}</desc>']
    for y, row in [(48, 'W9SSJ · TX'), (121, f'{spec["peer"]} · RX')]:
        svg += [f'<text x="12" y="{y+14}" fill="currentColor" font-size="13">{row}</text>',
                f'<path d="M140 {y+10} H880" stroke="currentColor" opacity=".18"/>']
    for e in events:
        if not drawn(e, lo, hi):
            continue
        y = 48 if e['direction'] == 'tx' else 121
        a, b = max(lo, e['t0']), min(hi, e['t1'])
        kind = e['kind'].removeprefix('tx ')
        title = html.escape(f"{e['t0']:.3f} s · {e['direction'].upper()} · {kind} · {e['status']}")
        svg.append(f'<rect data-event="{e["id"]}" x="{x(a):.2f}" y="{y}" width="{max(2,x(b)-x(a)):.2f}" height="20" rx="2" fill="currentColor" opacity="{.8 if e["direction"]=="rx" else .45}"><title>{title}</title></rect>')
        text = label(e, events)
        if text:
            svg.append(f'<text x="{(x(a)+x(b))/2:.2f}" y="{y-6}" text-anchor="middle" fill="currentColor" font-size="11">{text}</text>')
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
