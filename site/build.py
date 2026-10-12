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
from urllib.parse import unquote, urlsplit
import references
import accessibility

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {'flock-compared.html': 'compared', 'pactor-explained.html': 'pactor',
          'vara-explained.html': 'vara', 'ardop-explained.html': 'ardop',
          'sabir-explained.html': 'sabir',
          'robust-packet-explained.html': 'robust-packet',
          'stanag-explained.html': 'stanag', 'rtty-explained.html': 'rtty',
          'pactor4-explained.html': 'pactor-4'}

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
                or not (source.startswith(('docs/', 'packages/', 'tools/'))
                        or source in {'README.md', 'ARCHITECTURE.md', 'SECURITY.md', 'CHANGELOG.md'})
                or not (ROOT / path).resolve().is_relative_to(ROOT)
                or not (ROOT / path).is_file()):
            raise ValueError(f'Invalid public reference: {source}')
        reviewed.add(source)
    return reviewed


REFERENCE_FILES = reference_inventory(json.loads((ROOT / 'site/reference-files.json').read_text()))


def write_page(path, text, home='./', *, homepage=False, secret_door=True):
    # The warning also reaches guides opened directly rather than via the home page.
    # Replace any older notice, so only the homepage carries the demonstration.
    text = re.sub(r'<aside class="release-notice"[^>]*>[\s\S]*?</aside>', '', text)
    text = re.sub(r'<p class="release-demonstration">[\s\S]*?</p>', '', text)
    notice = NOTICE
    if homepage:
        notice = notice.replace('</strong>', ' Do not rely on it for any purpose, '
                                'especially emergency communications.</strong>')
        demonstration = ('<p>It has nevertheless demonstrated successful Winlink mail transfers '
                         'using <a href="examples/pactor3-send/">PACTOR-3</a> and full-speed '
                         '<a href="examples/vara-send/">VARA HF</a> protocols.</p>')
        notice = notice.replace('</aside>', demonstration + '</aside>')
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
        script = '<script type="module" src="' + home + 'secret-desk.mjs"></script>'
        text = text.replace('</body>', script + '</body>') if '</body>' in text else text + script
    text = accessibility.enhance(text, home, shortcuts=secret_door)
    # Some guides omit the optional head/body tags; every page has a title.
    path.write_text(text.replace('</title>', '</title>\n' + GOATCOUNTER, 1))


# Shared attribution and license notice for the published pages.
MARK = '<!--colophon-->'
COLOPHON = ('<a class="brand" href="{home}">hf<span>modem</span></a>'
            '<p>© 2026 Saul St John (W9SSJ) · Code: AGPL-3.0-only · '
            'Documentation, images and site: '
            '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></p>'
            '<nav aria-label="Project links">'
            '<a href="mailto:hfmodem@ssj.lol">hfmodem@ssj.lol</a> · '
            '<a href="{home}accessibility/">Accessibility</a> · '
            '<a href="https://github.com/sstjohn/hfmodem">Source ↗</a> · '
            '<a href="https://github.com/sstjohn/hfmodem/issues">Issues ↗</a> · '
            '<a href="https://github.com/sstjohn/hfmodem/blob/main/SECURITY.md">'
            'Security ↗</a></nav>')


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
    """Exclude explicitly marked repository-only sections from site documents."""
    start, end = '<!-- site:omit:start -->', '<!-- site:omit:end -->'
    out = []
    hidden = False
    for line in text.splitlines():
        if line == start:
            if hidden:
                raise ValueError('Nested site omission')
            hidden = True
        elif line == end:
            if not hidden:
                raise ValueError('Unmatched site omission end')
            hidden = False
        elif not hidden:
            out.append(line)
    if hidden:
        raise ValueError('Unclosed site omission')
    return '\n'.join(out)


def markdown(text):
    out = []
    # Omitting sections can leave several blank lines between visible blocks.
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
    for name in ('style.css', 'release-notice.css', 'favicon.svg', 'CNAME', 'secret-desk.mjs', 'accessibility.css', 'accessibility.mjs'):
        shutil.copyfile(ROOT / 'site' / name, output / name)
    recordings = output / 'recordings'
    recordings.mkdir(exist_ok=True)
    for name in ('vara-ofdm-k5dat-0930.flac', 'vara-ofdm-k5dat-0930.json'):
        shutil.copyfile(ROOT / 'site/recordings' / name, recordings / name)
    shutil.copyfile(ROOT / 'docs/viz/guide.css', output / 'guide.css')
    index = (ROOT / 'site/index.html').read_text()
    write_page(output / 'index.html', index.replace(MARK, COLOPHON.format(home='./')), homepage=True)
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
    primary = {('docs/'+name) for name, _, _ in DOCUMENTS} | {('docs/viz/'+name) for name in ROUTES}
    for source in sorted(REFERENCE_FILES - primary):
        page = reference_route(source)
        if Path(source).suffix == '.z':
            (output / page).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, output / page)
            continue
        home = '../' * (len(page.parts)-1)
        title = Path(source).name
        raw = (ROOT / source).read_text()
        if source.endswith('.md'):
            body = references.markdown(public_markdown(raw), lambda target: link(target, source, page.as_posix()))
        else:
            body = '<h1>'+html.escape(title)+'</h1><pre><code>'+''.join(
                f'<span id="L{number}">'+html.escape(line)+'\n</span>'
                for number, line in enumerate(raw.splitlines(), 1))+'</code></pre>'
        text = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>'+html.escape(title)+' · hfmodem</title>'
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
