# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Assemble static routes and render the two short Markdown site documents.

Only headings, paragraphs and tables are used by these documents. Reject other
block syntax rather than silently publishing a broken rendering.
"""
from pathlib import Path
import argparse
import html
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {'flock-compared.html': 'compared', 'pactor-explained.html': 'pactor',
          'vara-explained.html': 'vara', 'ardop-explained.html': 'ardop',
          'sabir-explained.html': 'sabir'}

GOATCOUNTER = ('<script data-goatcounter="https://hfmodem.goatcounter.com/count" '
               'async src="//gc.zgo.at/count.js"></script>')


def write_page(path, text):
    # Some guides omit the optional head/body tags; every page has a title.
    path.write_text(text.replace('</title>', '</title>\n' + GOATCOUNTER, 1))


# Shared attribution and license notice for the published pages.
MARK = '<!--colophon-->'
COLOPHON = ('<a class="brand" href="{home}">hf<span>modem</span></a>'
            '<p>© 2026 Saul St John (W9SSJ) · Code: AGPL-3.0-only · '
            'Documentation, images and site: '
            '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></p>'
            '<nav aria-label="Project links">'
            '<a href="https://github.com/sstjohn/hfmodem">Source ↗</a> · '
            '<a href="https://github.com/sstjohn/hfmodem/issues">Issues ↗</a> · '
            '<a href="https://github.com/sstjohn/hfmodem/blob/main/SECURITY.md">'
            'Security ↗</a></nav>')


def link(target):
    if target == 'STATUS.md':
        return '../status/'
    if target.startswith('viz/') and target[4:] in ROUTES:
        return '../' + ROUTES[target[4:]] + '/'
    if target.startswith(('https:', '#')):
        return target
    path = (ROOT / 'docs' / target).resolve().relative_to(ROOT).as_posix()
    return 'https://github.com/sstjohn/hfmodem/blob/main/' + path


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
            out.append(f'<a href="{html.escape(link(m[2]), quote=True)}">{html.escape(m[1])}</a>')
        else:
            out.append(html.escape(part))
    return ''.join(out)


def markdown(text):
    out = []
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        if m := re.fullmatch(r'(#{1,3}) (.+)', block):
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
    for name in ('style.css', 'favicon.svg', 'CNAME'):
        shutil.copyfile(ROOT / 'site' / name, output / name)
    shutil.copyfile(ROOT / 'docs/viz/guide.css', output / 'guide.css')
    index = (ROOT / 'site/index.html').read_text()
    write_page(output / 'index.html', index.replace(MARK, COLOPHON.format(home='./')))
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
            '<a href="../#run">Run</a></nav></div>', text, count=1)
        text = text.replace('<a class="site-home" href="../README.md">Documentation</a>', '<a class="site-home" href="../">Home</a>')
        for sibling, destination in ROUTES.items():
            text = text.replace(f'href="{sibling}"', f'href="../{destination}/"')
        text = text.replace('href="/"', 'href="../"')
        text = text.replace('href="../README.md">Documentation', 'href="../docs/">Documentation')
        text = text.replace('href="../STATUS.md', 'href="../status/')
        text = text.replace('href="../protocols/',
                            'href="https://github.com/sstjohn/hfmodem/blob/main/docs/protocols/')
        # Inside the page's own footer, and inside its wrapper where it has one,
        # so the colophon inherits that page's fine print and gutters.
        close = '</div></footer>' if '</div></footer>' in text else '</footer>'
        text = text.replace(close, f'<div>{deep}</div>{close}')
        (output / route).mkdir(exist_ok=True)
        write_page(output / route / 'index.html', text)
    for filename, route, title in [('STATUS.md', 'status', 'Implementation evidence'), ('README.md', 'docs', 'Documentation')]:
        body = markdown((ROOT / 'docs' / filename).read_text())
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
<nav aria-label="Main navigation"><a href="../#modes">Modes</a><a href="../#results">Results</a><a href="../#run">Run</a></nav>
</header>
<main id="main" class="wrap document">{body}</main>
<footer class="wrap">{deep}</footer></body></html>'''
        (output / route).mkdir(exist_ok=True)
        write_page(output / route / 'index.html', text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '_site')
    build(parser.parse_args().output)
