# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Conservative Markdown presentation for the reviewed local reference inventory.

All text is escaped; only generated links, headings, lists and code are HTML.
Unknown constructs remain visible text instead of being executed or discarded.
"""
import html
import re
from pygments.lexers import get_lexer_for_filename
from pygments.token import Comment, Keyword, Name, Number, String
from pygments.util import ClassNotFound


def slug(text):
    text = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'[*`_]', '', text).lower()
    return re.sub(r'[^\w -]', '', text).replace(' ', '-')


def inline(text, resolve):
    parts = re.split(r'(`+[^`\n]*`+|!?\[[^]]+\]\([^)]+\)|\*\*[^*]+\*\*)', text)
    out = []
    for part in parts:
        if part.startswith('`'):
            out.append('<code>'+html.escape(part.strip('`'))+'</code>')
        elif part.startswith('**') and part.endswith('**'):
            out.append('<strong>'+html.escape(part[2:-2])+'</strong>')
        elif m := re.fullmatch(r'(!?)\[([^]]+)\]\(([^)]+)\)', part):
            target = resolve(m[3]);label = html.escape(m[2])
            if target is None:
                out.append(label+' <span class="caption">(not included in this release)</span>')
            elif m[1]:
                out.append(f'<a href="{html.escape(target, quote=True)}">{label or "Image"}</a>')
            else:
                out.append(f'<a href="{html.escape(target, quote=True)}">{label}</a>')
        else:
            out.append(emphasis(part))
    return ''.join(out)


# CommonMark's flanking rule, minus intraword emphasis so `a*b*c` stays arithmetic.
EMPHASIS = re.compile(r'(?<![\w*\\])\*(?![\s*])([^*\n]*?[^\s*\\])\*(?![\w*])')


def emphasis(text):
    parts = EMPHASIS.split(text)
    return ''.join(f'<em>{html.escape(p)}</em>' if i % 2 else html.escape(p)
                   for i, p in enumerate(parts))


def cells(line):
    # A literal pipe inside inline code is part of the cell.
    result=[];cell=[];quoted=False;escaped=False
    for char in line.strip().strip('|'):
        if char=='`' and not escaped:quoted=not quoted
        if char=='|' and not quoted and not escaped:
            result.append(''.join(cell).strip());cell=[]
        else:cell.append(char)
        escaped=char=='\\' and not escaped
    return result+[''.join(cell).strip()]


def markdown(text, resolve):
    lines=text.splitlines();out=[];i=0;seen={}
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if m:=re.match(r'^\s*(`{3,}|~{3,})(\w*)\s*$',line):
            fence,language=m.groups();i+=1;code=[]
            while i<len(lines) and not re.match(r'^\s*'+re.escape(fence)+r'\s*$',lines[i]):
                code.append(lines[i]);i+=1
            label=f' class="language-{language}"' if language else ''
            out.append('<pre><code'+label+'>'+html.escape('\n'.join(code))+'</code></pre>');i+=1;continue
        if m:=re.match(r'^(#{1,6})\s+(.+?)\s*#*$',line):
            level,title=len(m[1]),m[2];anchor=slug(title);count=seen.get(anchor,0);seen[anchor]=count+1
            if count:anchor+='-'+str(count)
            out.append(f'<h{level} id="{anchor}">{inline(title,resolve)}</h{level}>');i+=1;continue
        if m:=re.fullmatch(r'\s*<a\s+(?:id|name)=["\']([^"\']+)["\']\s*>\s*</a>\s*',line):
            out.append('<span id="'+html.escape(m[1],quote=True)+'"></span>');i+=1;continue
        if line.lstrip().startswith('|') and i+1<len(lines) and re.fullmatch(r'\s*[| :\-]+\s*',lines[i+1]):
            header=cells(line);i+=2;rows=[]
            while i<len(lines) and lines[i].lstrip().startswith('|'):
                rows.append(cells(lines[i]));i+=1
            out.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th scope="col">'+inline(c,resolve)+'</th>' for c in header)+'</tr></thead><tbody>')
            out.extend('<tr>'+''.join('<td>'+inline(c,resolve)+'</td>' for c in row)+'</tr>' for row in rows)
            out.append('</tbody></table></div>');continue
        if re.match(r'^\s*(?:[-*+] |\d+[.)] )',line):
            ordered=bool(re.match(r'^\s*\d+[.)] ',line));tag='ol' if ordered else 'ul';items=[]
            while i<len(lines) and re.match(r'^\s*(?:[-*+] |\d+[.)] )',lines[i]):
                item=re.sub(r'^\s*(?:[-*+] |\d+[.)] )','',lines[i]);i+=1
                while i<len(lines) and lines[i].startswith('  ') and lines[i].strip() and not re.match(r'^\s*(?:[-*+] |\d+[.)] )',lines[i]):
                    item+=' '+lines[i].strip();i+=1
                items.append(item)
            out.append('<'+tag+'>'+''.join('<li>'+inline(x,resolve)+'</li>' for x in items)+'</'+tag+'>');continue
        if re.match(r'^\s*(?:---+|\*\*\*+)\s*$',line):out.append('<hr>');i+=1;continue
        if line.startswith('    '):
            code=[]
            while i<len(lines) and (lines[i].startswith('    ') or not lines[i].strip()):code.append(lines[i][4:] if lines[i].startswith('    ') else '');i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code).rstrip())+'</code></pre>');continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(?:#|```|~~~|\s*[-*+] |\s*\d+[.)] |\s*<a |\|)',lines[i]):paragraph.append(lines[i]);i+=1
        quote=all(x.startswith('>') for x in paragraph)
        body=inline(' '.join(x.removeprefix('>').lstrip() if quote else x for x in paragraph),resolve)
        out.append(('<blockquote><p>' if quote else '<p>')+body+('</p></blockquote>' if quote else '</p>'))
    return '\n'.join(out)


# One class per colour keeps the largest listings small; self and cls stay
# plain because colouring them alone added a tenth to the largest listing.
TOKEN_CLASSES = ((Name.Builtin.Pseudo, None), (Comment, 'c'), (String, 's'), (Number, 'm'), (Keyword, 'k'),
                 (Name.Function, 'f'), (Name.Class, 'f'),
                 (Name.Builtin, 'b'), (Name.Decorator, 'b'))


def token_class(kind):
    return next((name for parent, name in TOKEN_CLASSES if kind in parent), None)


def listing(name, text):
    """Source coloured at build time, one anchored span per line."""
    try:
        tokens = get_lexer_for_filename(name, stripnl=False).get_tokens(text)
    except ClassNotFound:
        tokens = [(None, text)]
    lines, runs = [], []
    for kind, value in tokens:
        cls = token_class(kind)
        for index, part in enumerate(value.split('\n')):
            if index:
                lines.append(runs)
                runs = []
            if not part:
                continue
            if runs and runs[-1][0] == cls:
                runs[-1][1] += part
            else:
                runs.append([cls, part])
    if runs:
        lines.append(runs)
    return '<pre class="listing"><code>' + ''.join(
        f'<span id="L{number}">' + ''.join(
            f'<span class={cls}>{html.escape(part)}</span>' if cls else html.escape(part)
            for cls, part in runs) + '\n</span>'
        for number, runs in enumerate(lines, 1)) + '</code></pre>'
