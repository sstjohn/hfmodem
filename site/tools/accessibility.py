# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Shared, static navigation semantics for every generated site route."""
import re


def enhance(text, home, *, shortcuts=True):
    main = re.search(r'<main\b([^>]*)>', text)
    if not main:
        raise ValueError('Published prose page needs a main landmark')
    attrs = main[1]
    identifier = re.search(r'\bid="([^"]+)"', attrs)
    target = identifier[1] if identifier else 'main'
    if not identifier:
        attrs += ' id="main"'
    if 'tabindex=' not in attrs:
        attrs += ' tabindex="-1"'
    text = text[:main.start()] + '<main' + attrs + '>' + text[main.end():]
    # The bypass link comes before the shared notice, including on deep sources.
    text = re.sub(r'<a\b[^>]*class="skip"[^>]*>[\s\S]*?</a>', '', text)
    skip = f'<a class="skip" href="#{target}">Skip to content</a>'
    text = re.sub(r'(<body\b[^>]*>)', r'\1' + skip, text, count=1)
    assets = (f'<link rel="stylesheet" href="{home}accessibility.css">'
              f'<script type="module" src="{home}js/accessibility.mjs"></script>')
    # Last stylesheet wins even when a guide keeps its CSS inline.
    if '</head>' in text:
        text = text.replace('</head>', assets + '</head>', 1)
    else:
        text = re.sub(r'(<body\b[^>]*>)', assets + r'\1', text, count=1)
    if '</footer>' not in text:
        text = text.replace('</body>', '<footer></footer></body>', 1)
    # Also offer a non-timed text entry for users who cannot send key sequences.
    # No sequence or hidden destination is advertised here.
    policy = f'<a href="{home}accessibility/">Accessibility</a>'
    if f'href="{home}accessibility/"' not in text:
        text = text.replace('</footer>', policy + '</footer>', 1)
    if shortcuts:
        preferences = ('<details class="keyboard-preferences"><summary>Keyboard preferences</summary>'
                       '<label><input id="letter-shortcuts" type="checkbox" checked> '
                       'Allow letter-key shortcuts</label>'
                       '<form id="shortcut-form"><label for="shortcut-sequence">Shortcut sequence</label> '
                       '<input id="shortcut-sequence" name="sequence" type="text" autocomplete="off" '
                       'spellcheck="false"> <button type="submit">Run shortcut</button>'
                       '<p id="shortcut-status" role="status"></p></form></details>')
        text = text.replace('</footer>', preferences + '</footer>', 1)
    return text
