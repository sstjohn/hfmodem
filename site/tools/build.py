# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Assemble static routes and render reviewed documentation and source references.

Reference inputs are an explicit public inventory. Markdown is escaped before
presentation, source files have line anchors, and compact C2 resources retain
their original bytes.
"""
from pathlib import Path
import argparse
import html
import re
import shutil
import json
import posixpath
from urllib.parse import quote, unquote, urljoin, urlsplit
import references
import accessibility

ROOT = Path(__file__).resolve().parents[2]
SITE = 'https://' + (ROOT / 'site/CNAME').read_text().strip() + '/'
ROUTES = {'flock-compared.html': 'compared', 'pactor-explained.html': 'pactor',
          'vara-explained.html': 'vara', 'ardop-explained.html': 'ardop',
          'sabir-explained.html': 'sabir',
          'robust-packet-explained.html': 'robust-packet',
          'stanag-explained.html': 'stanag', 'rtty-explained.html': 'rtty',
          'pactor4-explained.html': 'pactor-4'}

# The visible mode switcher is shared by every guide.
# Source guides retain these same links for offline use.
MODE_GUIDES = (
    ('flock-compared.html', 'Comparison'), ('pactor-explained.html', 'PACTOR'),
    ('pactor4-explained.html', 'PACTOR-4'),
    ('vara-explained.html', 'VARA'), ('ardop-explained.html', 'ARDOP'),
    ('sabir-explained.html', 'sabir'), ('robust-packet-explained.html', 'Robust Packet'),
    ('stanag-explained.html', 'STANAG'), ('rtty-explained.html', 'RTTY / SITOR / NAVTEX'),
)


def mode_navigation(current):
    links = []
    for filename, label in MODE_GUIDES:
        selected = ' aria-current="page"' if filename == current else ''
        links.append(f'<a href="{filename}" data-route="{ROUTES[filename]}"{selected}>{label}</a>')
    return '<nav class="site-nav" aria-label="Modes">' + ''.join(links) + '</nav>'

DOCUMENTS = [('STATUS.md', 'status', 'Implementation evidence'),
             ('ACCESSIBILITY.md', 'accessibility', 'Accessibility'),
             ('README.md', 'docs', 'Documentation'),
             ('CONSOLE.md', 'console', 'Console and operator tools'),
             ('console-monitor.md', 'monitor', 'Passive monitoring'),
             ('console-hflink.md', 'hflink', 'ALE and selcall console')]

GOATCOUNTER = ('<script data-goatcounter="https://hfmodem.goatcounter.com/count" '
               'async src="//gc.zgo.at/count.js"></script>')


NOTICE = ('<aside class="release-notice" aria-label="Experimental prerelease warning">'
          '<p><strong>hfmodem is an experimental prerelease for developers only.</strong></p></aside>')

def reference_inventory(paths):
    """Only reviewed, existing repository files can become reference routes."""
    reviewed = set()
    for source in paths:
        path = Path(source)
        if (path.is_absolute() or '..' in path.parts or path.as_posix() != source
                or not (source.startswith(('docs/', 'packages/', 'tools/', 'evidence/'))
                        or source in {'README.md', 'ARCHITECTURE.md', 'SECURITY.md', 'RELATED-WORK.md'})
                or not (ROOT / path).resolve().is_relative_to(ROOT)
                or not (ROOT / path).is_file()):
            raise ValueError(f'Invalid public reference: {source}')
        reviewed.add(source)
    return reviewed


REFERENCE_FILES = reference_inventory(json.loads((ROOT / 'site/reference-files.json').read_text()))

# Released reference routes whose sources have since moved; each old route
# keeps a page that forwards to the new one.
MOVED = {f'docs/protocols/{name}': f'docs/implementation/{name}' for name in (
    'bateleur/ALE-HOST.md', 'bateleur/ALE.md', 'bateleur/HOST-API.md', 'bateleur/README.md',
    'bateleur/S5066.md', 'pactor/P3-ADAPTATION.md', 'pactor/console-traffic.md',
    'pactor/pactor2-mail.md', 'robust-packet/link.md', 'robust-packet/station.md',
    'sabir/ONAIR.md', 'sabir/RELEASE-NOTES.md', 'sitor/README.md',
    'vara/22-turnaround-capture.md', 'vara/24-lower-record-recovery.md',
    'vara/25-recovery-diagnostics.md', 'vara/27-nak-scan-deadline.md',
    'vara/28-short-final-nak-retry.md', 'vara/29-low-level-data-nacks.md')}


# Search titles and descriptions for the site's own pages, by output route.
METADATA = {
    'index.html': (
        'hfmodem: HF data modem for PACTOR, VARA, ARDOP and more',
        'An HF modem with PACTOR, VARA, ARDOP, Robust Packet, ALE, selcall, voice, CW and '
        'text-mode monitoring. Protocol guides, recorded exchanges and operator console.'),
    'pactor/index.html': (
        'PACTOR 1–4 in hfmodem: waveform, coding and ARQ',
        'How hfmodem’s shrike implements PACTOR-1, -2 and -3, with partial experimental '
        'PACTOR-4: FSK and multicarrier waveforms, coding and the ARQ cycle.'),
    'pactor-4/index.html': (
        'PACTOR-4 in hfmodem: chirp, spread and QAM packet layouts',
        'Partial, experimental PACTOR-4 in hfmodem: chirp, spread and single-carrier PSK '
        'and QAM packets, channel coding, training and symbol layouts.'),
    'vara/index.html': (
        'VARA in hfmodem: MFSK handshake, OFDM data and ARQ',
        'How hfmodem’s kestrel carries VARA at 500, 2300 and 2750 Hz: MFSK link entry, '
        'OFDM payload, turbo coding and session control.'),
    'ardop/index.html': (
        'ARDOP in hfmodem: adaptive FSK, PSK and QAM carriers',
        'How hfmodem’s besra implements ARDOP: 4FSK to 16QAM carriers, protected byte '
        'blocks and acknowledgements, with figures generated from the implementation.'),
    'sabir/index.html': (
        'sabir: an experimental half-duplex HF data protocol',
        'sabir pairs coherent OFDM data with four-tone FSK control: session negotiation, '
        'selective retransmission, turn-taking and the host interface.'),
    'robust-packet/index.html': (
        'Robust Packet in hfmodem: eight-carrier AX.25 for HF',
        'How hfmodem’s brubru carries AX.25 over eight shaped carriers at 200 and 600 '
        'bit/s, with error correction, interleaving and a KISS host interface.'),
    'stanag/index.html': (
        'STANAG 4285, 4529, 4539 and MIL-STD-188-110 in hfmodem',
        'Experimental STANAG and MIL-STD HF waveforms in hfmodem’s bateleur: frame and '
        'constellation geometry, coding and interleavers.'),
    'rtty/index.html': (
        'RTTY, SITOR-B and NAVTEX in hfmodem: framing and decoding',
        'Illustrated guide to ITA2 RTTY framing, SITOR-B seven-unit time diversity and '
        'NAVTEX bulletin decoding in hfmodem.'),
    'compared/index.html': (
        'HF data modes compared: PACTOR, VARA, ARDOP and sabir',
        'Carrier placement, timing and mode parameters for PACTOR, VARA, ARDOP and sabir '
        'side by side, with recorded off-air audio.'),
    'examples/index.html': (
        'Recorded HF contacts: Winlink and VarAC replays with packets',
        'Replays of real hfmodem contacts over PACTOR-3, VARA and ARDOP: listen to the '
        'audio and inspect each packet’s timing, signal and decoded bytes.'),
    'examples/logbook/index.html': (
        'Recorded Winlink mail logbook: PACTOR-3, VARA and ARDOP',
        'Selected Winlink mail exchanges with WS8EOC, VE1YZ, N5TW, NS0A, W6IDS and WW2MI, '
        'each linked to its recording and packet inspector.'),
    'examples/ardop-receive/index.html': (
        'ARDOP Winlink receive from W6IDS on 40 m: recorded replay',
        'Three Winlink messages received from W6IDS over ARDOP on 40 m, August 26, 2026. '
        'KiwiSDR witness audio with packet-by-packet inspection.'),
    'examples/ardop-send/index.html': (
        'ARDOP Winlink send to WW2MI on 40 m: recorded replay',
        'One Winlink message delivered to WW2MI over ARDOP on 40 m, August 23, 2026, '
        'confirmed by destination receipt. KiwiSDR witness audio and packets.'),
    'examples/pactor3-receive/index.html': (
        'PACTOR-3 Winlink receive from WS8EOC on 80 m: recorded replay',
        'A complete Winlink message received from WS8EOC over PACTOR-3 on 80 m, October 3, '
        '2026, reaching SL3. Station and remote witness audio, aligned.'),
    'examples/pactor3-send/index.html': (
        'PACTOR-3 Winlink mail with VE1YZ on 40 m: recorded replay',
        'One message sent to VE1YZ and two received over PACTOR-3 on 40 m, September 24, '
        '2026; local transmit reached SL6. Receiver, monitor and witness audio.'),
    'examples/vara-receive/index.html': (
        'VARA Winlink mail with N5TW on 20 m: recorded replay',
        'Two Winlink messages received from N5TW and one sent over VARA BW2300 on 20 m, '
        'September 23, 2026, at L4 both ways. Audio and packet inspection.'),
    'examples/vara-send/index.html': (
        'VARA Winlink send to NS0A on 40 m: recorded replay',
        'One Winlink message sent to NS0A over VARA BW2300 on 40 m, September 27, 2026, '
        'with destination receipt; the session reached L9 both ways.'),
    'console/index.html': (
        'hfmodem console: monitoring, Winlink, chat, ALE and logbook',
        'The hfmodem operator console: per-mode monitoring, PACTOR, VARA and ARDOP Winlink, '
        'VarAC chat, voice and CW, ALE, selcall and a replay logbook.'),
    'monitor/index.html': (
        'Passive HF monitoring with the hfmodem console',
        'Decode one HF protocol at a time with hfmodem console monitor. It listens on the '
        'rig’s current dial and never calls, answers or transmits.'),
    'hflink/index.html': (
        '2G ALE and HF selcall with the hfmodem console',
        'Run a 2G ALE station that scans, answers, links and exchanges text, or a '
        'held-channel HF selcall station, from the hfmodem console.'),
    'docs/index.html': (
        'hfmodem documentation: operator guides and protocol references',
        'Index of hfmodem operator guides, protocol references and recorded results, with '
        'each mode’s release status and implementation.'),
    'status/index.html': (
        'hfmodem implementation evidence: on-air and bench results',
        'What hfmodem has shown for each mode: RF reception, gateway transfers, independent '
        'bench decodes and simulations, each identified separately.'),
    'accessibility/index.html': (
        'hfmodem website accessibility: WCAG 2.2 AA',
        'How the hfmodem website targets WCAG 2.2 Level AA for screen readers, speech '
        'input, keyboards and switches, with optional audio throughout.'),
}


def page_route(path, home):
    return path.resolve().relative_to((path.parent / home).resolve()).as_posix()


def page_url(route):
    return SITE + quote(route.removesuffix('index.html'))


def describe(text, route):
    if route not in METADATA:
        return text
    title, description = (html.escape(value, quote=True) for value in METADATA[route])
    text = re.sub(r'<meta name="description"[^>]*>\n?', '', text, count=1)
    return re.sub(r'<title>[\s\S]*?</title>', lambda _: f'<title>{title}</title>\n'
                  f'<meta name="description" content="{description}">', text, count=1)


def canonical(path, text, home):
    # A forwarding page names its destination; every other page names itself.
    url = page_url(page_route(path, home))
    if forward := re.search(r'http-equiv="refresh" content="0; url=([^"]+)"', text):
        url = urljoin(url, html.unescape(forward[1]))
    return '<link rel="canonical" href="' + html.escape(url, quote=True) + '">'


def write_page(path, text, home='./', *, homepage=False, secret_door=True, addressed=True):
    # The warning also reaches guides opened directly rather than via the home page.
    # Replace any older notice with the current warning.
    text = re.sub(r'<aside class="release-notice"[^>]*>[\s\S]*?</aside>', '', text)
    text = re.sub(r'<p class="release-demonstration">[\s\S]*?</p>', '', text)
    notice = NOTICE
    if homepage:
        notice = notice.replace('</strong>', ' Do not rely on it for any purpose, '
                                'especially emergency communications.</strong>')
    if 'class="release-notice"' not in text:
        if re.search(r'<body\b[^>]*>', text):
            text = re.sub(r'(<body\b[^>]*>)', r'\1' + notice, text, count=1)
        elif '</head>' in text:
            text = text.replace('</head>', '</head><body>' + notice, 1)
        else:
            text = text.replace('<header', '<body>' + notice + '<header', 1)
    if 'href="release-notice.css"' not in text:
        text = text.replace('</title>', '</title>\n<link rel="stylesheet" href="' + home + 'release-notice.css">', 1)
    if secret_door:
        # Relative to this module, the hidden route works at any site mount path.
        script = '<script type="module" src="' + home + 'js/secret-desk.mjs"></script>'
        text = text.replace('</body>', script + '</body>') if '</body>' in text else text + script
    text = accessibility.enhance(text, home, shortcuts=secret_door)
    head = GOATCOUNTER
    if addressed:
        text = describe(text, page_route(path, home))
        head = canonical(path, text, home) + '\n' + head
    # Some guides omit the optional head/body tags; every page has a title.
    path.write_text(text.replace('</title>', '</title>\n' + head, 1))


# Shared attribution and license notice for the published pages.
MARK = '<!--colophon-->'
COLOPHON = ('<a class="brand" href="{home}">hf<span>modem</span></a>'
            '<p>© 2026 Saul St John (W9SSJ) · Code and software documentation: '
            'AGPL-3.0-only · Protocol descriptions and site: '
            '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></p>'
            '<nav aria-label="Project links">'
            '<a href="mailto:hfmodem@ssj.lol">hfmodem@ssj.lol</a> · '
            '<a href="{home}accessibility/">Accessibility</a> · '
            '<a href="https://github.com/sstjohn/hfmodem">Source ↗</a> · '
            '<a href="https://github.com/sstjohn/hfmodem/issues">Issues ↗</a> · '
            '<a href="https://github.com/sstjohn/hfmodem/blob/main/SECURITY.md">'
            'Security ↗</a></nav>')

MISSING = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Page not found — hfmodem</title>
<meta name="description" content="No page is published at this address on the hfmodem site.">
<meta name="robots" content="noindex">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/style.css"></head><body>
<header class="wrap masthead">
<a class="brand" href="/" aria-label="hfmodem home">hf<span>modem</span></a>
<nav aria-label="Main navigation"><a href="/#modes">Modes</a><a href="/#results">Results</a><a href="/#run">Winlink mail</a><a href="/#operate">Monitor &amp; call</a></nav>
</header>
<main class="wrap document">
<h1>Not found</h1>
<p>There’s nothing at this address. <a href="/">Back to hfmodem</a></p>
</main>
<footer class="wrap">{colophon}</footer></body></html>'''


def reference_route(source):
    source = Path(source)
    for filename, route, _ in DOCUMENTS:
        if source == Path('docs') / filename:
            return Path(route) / 'index.html'
    if source.parent == Path('docs/viz') and source.name in ROUTES:
        return Path(ROUTES[source.name]) / 'index.html'
    if source.name == 'README.md':
        return Path('reference') / source.parent / 'index.html'
    if source.suffix == '.z':
        return Path('reference') / source
    return Path('reference') / source.with_suffix(source.suffix + '.html' if source.suffix != '.md' else '.html')


# Public downloads linked by the evidence index. Keep the source audio single-copy.
EVIDENCE_DOWNLOADS = (
    'evidence/vara-mail-0918/provenance.json',
    'evidence/vara-mail-0918/audio/20260918T154839Z-W9SSJ-tap.wav',
    'evidence/pactor4-ve3kpg-0929/provenance.json',
    'evidence/pactor4-ve3kpg-0929/audio/receiver.flac',
)
RECORDING_FILES = ('audio/vara-ofdm-k5dat-0930.flac', 'vara-ofdm-k5dat-0930.json')


def link(target, source='docs/README.md', page='docs/index.html'):
    if urlsplit(target).scheme or target.startswith(('#', '//')):
        return target
    path, mark, fragment = target.partition('#')
    candidate = (ROOT / source).parent.joinpath(unquote(path)).resolve()
    if candidate.is_dir():
        candidate /= 'README.md'
    if not candidate.is_relative_to(ROOT):
        return None
    rel = candidate.relative_to(ROOT).as_posix()
    if rel in EVIDENCE_DOWNLOADS:
        return posixpath.relpath(rel, Path(page).parent.as_posix()) + (mark + fragment if mark else '')
    if (rel.startswith('site/examples/') or rel in {'site/recordings/' + name for name in RECORDING_FILES}) and candidate.is_file():
        dest = posixpath.relpath(Path(rel).relative_to('site').as_posix(), Path(page).parent.as_posix())
        return dest.removesuffix('index.html') + (mark + fragment if mark else '')
    if rel not in REFERENCE_FILES and not (candidate.parent == ROOT / 'docs/viz' and candidate.name in ROUTES):
        return None
    dest = reference_route(rel)
    result = posixpath.relpath(dest.as_posix(), Path(page).parent.as_posix())
    if dest.name == 'index.html':
        result = result.removesuffix('index.html')
    return result + (mark + fragment if mark else '')


def inline(text):
    # Tokenise before escaping so code and link contents are never parsed twice.
    parts = re.split(r'(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*)', text)
    out = []
    for part in parts:
        if part.startswith('`'):
            out.append('<code>' + html.escape(part[1:-1]) + '</code>')
        elif part.startswith('**'):
            out.append('<strong>' + html.escape(part[2:-2]) + '</strong>')
        elif m := re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', part):
            target = link(m[2])
            out.append(f'<a href="{html.escape(target, quote=True)}">{html.escape(m[1])}</a>' if target else html.escape(m[1]))
        else:
            out.append(html.escape(part))
    return ''.join(out)


def public_markdown(text):
    """Drop comment lines, such as licence headers, outside fenced code."""
    out = []
    fenced = False
    for line in text.splitlines():
        if re.match(r'\s*(```|~~~)', line):
            fenced = not fenced
        if fenced or not re.fullmatch(r'\s*<!--.*-->\s*', line):
            out.append(line)
    return '\n'.join(out)


def markdown(text):
    out = []
    # Dropped comment lines can leave several blank lines between visible blocks.
    blocks = []
    for part in re.split(r'(?m)(^```[^\n]*\n[\s\S]*?^```[ \t]*$)', text.strip()):
        if part.startswith('```'):
            blocks.append(part)
        else:
            blocks.extend(block for block in re.split(r'\n(?:[ \t]*\n)+', part.strip()) if block)
    for block in blocks:
        lines = block.splitlines()
        if block.startswith('```'):
            assert len(lines) >= 2 and lines[-1].strip() == '```', 'Unclosed fenced code'
            language = lines[0][3:].strip()
            assert re.fullmatch(r'[\w+-]*', language), 'Invalid code language'
            label = f' class="language-{language}"' if language else ''
            out.append('<pre><code' + label + '>' + html.escape('\n'.join(lines[1:-1])) + '</code></pre>')
        elif m := re.fullmatch(r'(#{1,3}) (.+)', block):
            level, title = len(m[1]), m[2]
            slug = re.sub(r'[^\w -]', '', title.lower()).replace(' ', '-')
            out.append(f'<h{level} id="{slug}">{inline(title)}</h{level}>')
        elif lines[0].startswith('|'):
            rows = [[inline(cell.strip()) for cell in row.strip('|').split('|')] for row in lines]
            assert re.fullmatch(r'[| :\-]+', lines[1]), 'Table needs a separator'
            out.append('<div class="table-scroll"><table><thead><tr>' + ''.join('<th scope="col">'+c+'</th>' for c in rows[0]) + '</tr></thead><tbody>')
            out.extend('<tr>'+''.join('<td>'+c+'</td>' for c in row)+'</tr>' for row in rows[2:])
            out.append('</tbody></table></div>')
        elif all(line.lstrip().startswith(('- ', '* ')) for line in lines):
            items = [inline(line.lstrip()[2:].strip()) for line in lines]
            out.append('<ul>' + ''.join('<li>' + item + '</li>' for item in items) + '</ul>')
        else:
            assert not re.match(r'\s*(?:[-*>] |\d+\. |```)', block), 'Unsupported Markdown block'
            out.append('<p>'+inline(' '.join(lines))+'</p>')
    return '\n'.join(out)


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ('style.css', 'release-notice.css', 'favicon.svg', 'CNAME', 'accessibility.css'):
        shutil.copyfile(ROOT / 'site' / name, output / name)
    shutil.copytree(ROOT / 'site/js', output / 'js', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.*', '__pycache__'))
    recordings = output / 'recordings'
    (recordings / 'audio').mkdir(parents=True, exist_ok=True)
    for name in RECORDING_FILES:
        shutil.copyfile(ROOT / 'site/recordings' / name, recordings / name)
    for source in EVIDENCE_DOWNLOADS:
        target = output / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    shutil.copyfile(ROOT / 'docs/viz/guide.css', output / 'guide.css')
    index = (ROOT / 'site/index.html').read_text()
    colophon = COLOPHON.format(home='./').replace(
        'Saul St John', '<a href="https://ssj.lol/" rel="author">Saul St John</a>', 1)
    write_page(output / 'index.html', index.replace(MARK, colophon), homepage=True)
    examples = output / 'examples'
    shutil.copytree(ROOT / 'site/examples', examples, dirs_exist_ok=True)
    for page in examples.rglob('*.html'):
        home = '../' * (len(page.relative_to(output).parts) - 1)
        write_page(page, page.read_text().replace(MARK, COLOPHON.format(home=home)), home)
    # Explicit reviewed public assets only; the moving frontend workspace
    # is never read by the site build.
    demo = output / 'demo'
    shutil.copytree(ROOT / 'site/demo', demo, dirs_exist_ok=True)
    for page in demo.rglob('*.html'):
        home = '../' * (len(page.relative_to(output).parts) - 1)
        write_page(page, page.read_text().replace(MARK, COLOPHON.format(home=home)),
                   home, secret_door=False)
    deep = COLOPHON.format(home='../')
    for filename, route in ROUTES.items():
        text = (ROOT / 'docs/viz' / filename).read_text()
        text = text.replace('</title>', '</title>\n<link rel="icon" href="../favicon.svg" type="image/svg+xml">', 1)
        text = text.replace('href="guide.css"', 'href="../guide.css"')
        text = re.sub(
            r'<div class="site-heading">.*?</div>',
            '<div class="site-heading"><a class="site-home" href="../" '
            'aria-label="hfmodem home">hfmodem</a>'
            '<nav class="site-global-nav" aria-label="Main navigation">'
            '<a href="../#modes">Modes</a><a href="../#results">Results</a>'
            '<a href="../#run">Winlink mail</a><a href="../#operate">Monitor &amp; call</a></nav></div>', text, count=1)
        text = text.replace('<a class="site-home" href="../README.md">Documentation</a>', '<a class="site-home" href="../">Home</a>')
        text, nav_count = re.subn(r'<nav class="site-nav"[^>]*>.*?</nav>',
                                 lambda _: mode_navigation(filename), text, count=1, flags=re.S)
        if nav_count != 1:
            raise ValueError(f'Missing shared mode navigation in {filename}')
        for sibling, destination in ROUTES.items():
            text = text.replace(f'href="{sibling}"', f'href="../{destination}/"')
        text = text.replace('href="/"', 'href="../"')
        text = text.replace('href="../README.md">Documentation', 'href="../docs/">Documentation')
        text = text.replace('href="../STATUS.md', 'href="../status/')
        text = re.sub(r'href="(../protocols/[^"]+)"',
                      lambda m: 'href="'+html.escape(link(m[1], 'docs/viz/'+filename, route+'/index.html') or '../docs/', quote=True)+'"', text)
        # New protocol guides point directly to reviewed implementation files.
        # A missing clearance is a build failure, not a silent docs-index detour.
        def guide_reference(match):
            target = link(match[1], 'docs/viz/'+filename, route+'/index.html')
            if target is None:
                raise ValueError(f'Unreviewed guide reference in {filename}: {match[1]}')
            return 'href="'+html.escape(target, quote=True)+'"'
        text = re.sub(r'href="((?:\.\./)+[^" ]+\.(?:py|md|json)(?:#[^" ]*)?)"', guide_reference, text)
        # Inside the page's own footer, and inside its wrapper where it has one,
        # so the colophon inherits that page's fine print and gutters.
        close = '</div></footer>' if '</div></footer>' in text else '</footer>'
        text = text.replace(close, f'<div>{deep}</div>{close}')
        (output / route).mkdir(exist_ok=True)
        write_page(output / route / 'index.html', text, '../')
    for filename, route, title in DOCUMENTS:
        body = references.markdown(public_markdown((ROOT / 'docs' / filename).read_text()),
                                   lambda target: link(target, 'docs/'+filename, route+'/index.html'))
        text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title} — hfmodem</title>
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="../style.css"></head><body>
<a class="skip" href="#main">Skip to content</a>
<header class="wrap masthead">
<a class="brand" href="../" aria-label="hfmodem home">hf<span>modem</span></a>
<nav aria-label="Main navigation"><a href="../#modes">Modes</a><a href="../#results">Results</a><a href="../#run">Winlink mail</a><a href="../#operate">Monitor &amp; call</a></nav>
</header>
<main id="main" class="wrap document">{body}</main>
<footer class="wrap">{deep}</footer></body></html>'''
        (output / route).mkdir(exist_ok=True)
        write_page(output / route / 'index.html', text, '../')
    # GitHub Pages serves this at every missing URL, at any depth, so its links
    # are root-absolute and it claims no address of its own.
    write_page(output / '404.html', MISSING.format(colophon=COLOPHON.format(home='/')),
               '/', addressed=False)
    primary = {('docs/'+name) for name, _, _ in DOCUMENTS} | {('docs/viz/'+name) for name in ROUTES}
    for source in sorted(REFERENCE_FILES - primary):
        page = reference_route(source)
        if Path(source).suffix == '.z':
            (output / page).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, output / page)
            continue
        title = Path(source).name
        raw = (ROOT / source).read_text()
        if source.endswith('.md'):
            body = references.markdown(public_markdown(raw), lambda target: link(target, source, page.as_posix()))
        else:
            body = '<h1>'+html.escape(title)+'</h1>'+references.listing(title, raw)
        # Every reference source ships unchanged in the public repository, so
        # only the site's own pages are indexed.
        write_reference(output, page, source, body, '<meta name="robots" content="noindex">')
    for old, new in MOVED.items():
        page = reference_route(old)
        target = html.escape(posixpath.relpath(reference_route(new).as_posix(), page.parent.as_posix())
                             .removesuffix('index.html'), quote=True)
        write_reference(output, page, new,
                        f'<p>This page moved to <a href="{target}">{html.escape(new)}</a>.</p>',
                        f'<meta http-equiv="refresh" content="0; url={target}">'
                        f'<script>location.replace({json.dumps(target)} + location.hash)</script>')
    write_sitemap(output)


def write_sitemap(output):
    """List each page that names itself canonical and allows indexing.

    No lastmod: release exports carry no per-page history, and a build time
    would claim every page changed on every deploy.
    """
    root = output.resolve()
    urls = []
    for page in sorted(root.rglob('*.html')):
        text = page.read_text()
        url = page_url(page.relative_to(root).as_posix())
        if (f'<link rel="canonical" href="{html.escape(url, quote=True)}">' in text
                and not re.search(r'<meta name="robots" content="[^"]*noindex', text)):
            urls.append(url)
    (output / 'sitemap.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + ''.join(f'<url><loc>{html.escape(url)}</loc></url>\n' for url in sorted(urls))
        + '</urlset>\n')
    (output / 'robots.txt').write_text(f'User-agent: *\nAllow: /\n\nSitemap: {SITE}sitemap.xml\n')


def write_reference(output, page, title, body, head=''):
    home = '../' * (len(page.parts)-1)
    text = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+html.escape(title)+' · hfmodem</title>'+head+
            '<link rel="stylesheet" href="'+home+'style.css"></head><body>'
            '<header class="wrap masthead"><a href="'+home+'">hfmodem</a>'
            '<a href="'+home+'docs/">Documentation</a></header>'
            '<main class="wrap document">'+body+'</main><footer class="wrap">'+
            COLOPHON.format(home=home)+'</footer></body></html>')
    (output / page).parent.mkdir(parents=True, exist_ok=True)
    write_page(output / page, text, home)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '_site')
    build(parser.parse_args().output)
