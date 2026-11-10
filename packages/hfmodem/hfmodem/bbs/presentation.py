# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Canonical mutable styled text, bounded control filtering and safe paging.

Only color and intensity survive parsing. The parser never returns an ESC in
text; renderers generate their own SGR sequences from the canonical styles.
"""
from dataclasses import dataclass, field
import unicodedata

from .model import Presentation


@dataclass(frozen=True)
class Style:
    foreground: int | None = None
    background: int | None = None
    bold: bool = False


@dataclass
class Span:
    text: str
    style: Style = field(default_factory=Style)


@dataclass
class StyledText:
    spans: list[Span] = field(default_factory=list)

    @property
    def text(self):
        return ''.join(span.text for span in self.spans)

    def append(self, text, style=None):
        if not text:
            return
        style = style or Style()
        if self.spans and self.spans[-1].style == style:
            self.spans[-1].text += text
        else:
            self.spans.append(Span(text, style))


def cell_width(char):
    if unicodedata.combining(char) or unicodedata.category(char) in ('Mn', 'Me', 'Cf'):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1


class StyleParser:
    """Incremental escape parser with constant-size escape state.

    Unsupported control strings are discarded through their terminator, even
    after the length bound is exceeded. CR is retained only to normalize CRLF.
    Standalone CR and all other raw active C0/C1 controls are discarded.
    """
    def __init__(self, max_escape=128, tab_width=8):
        self.max_escape = max_escape
        self.tab_width = tab_width
        self.style = Style()
        self.state = 'text'
        self.sequence = ''
        self.column = 0
        self.cr = False
        self.string_escape = False

    def feed(self, text, *, final=False):
        result = StyledText()
        for char in text:
            if self.state == 'string':
                if char in ('\x07', '\x9c') or self.string_escape and char == '\\':
                    self.state = 'text'
                    self.string_escape = False
                else:
                    self.string_escape = char == '\x1b'
                continue
            if self.state == 'escape':
                if char == '[':
                    self.state, self.sequence = 'csi', ''
                elif char in ']PX^_':
                    self.state, self.string_escape = 'string', False
                elif '\x20' <= char <= '\x2f':
                    self.state = 'escape_intermediate'
                else:
                    self.state = 'text'
                continue
            if self.state == 'escape_intermediate':
                if '\x30' <= char <= '\x7e':
                    self.state = 'text'
                continue
            if self.state in ('csi', 'csi_drop'):
                if '\x40' <= char <= '\x7e':
                    if self.state == 'csi' and char == 'm':
                        self._sgr(self.sequence)
                    self.state, self.sequence = 'text', ''
                elif self.state == 'csi':
                    self.sequence += char
                    if len(self.sequence) > self.max_escape:
                        self.state, self.sequence = 'csi_drop', ''
                continue
            if self.cr:
                self.cr = False
                if char == '\n':
                    result.append('\n', self.style)
                    self.column = 0
                    continue
            if char == '\r':
                self.cr = True
            elif char == '\x1b':
                self.state = 'escape'
            elif char == '\x9b':
                self.state, self.sequence = 'csi_drop', ''
            elif char in ('\x90', '\x9d', '\x98', '\x9e', '\x9f'):
                self.state, self.string_escape = 'string', False
            elif char == '\n':
                result.append(char, self.style)
                self.column = 0
            elif char == '\t':
                spaces = self.tab_width - self.column % self.tab_width
                result.append(' ' * spaces, self.style)
                self.column += spaces
            elif ord(char) >= 32 and not 127 <= ord(char) <= 159 and not 0xd800 <= ord(char) <= 0xdfff:
                result.append(char, self.style)
                self.column += cell_width(char)
        if final:
            self.state, self.sequence, self.cr = 'text', '', False
            self.string_escape = False
        return result

    def _sgr(self, sequence):
        if any(c not in '0123456789;' for c in sequence):
            return
        codes = [int(c) if c else 0 for c in sequence.split(';')]
        # Reject the whole unsupported SGR, rather than treating 38;5;1 as bold.
        allowed = lambda c: c in (0, 1, 22, 39, 49) or 30 <= c <= 37 or 40 <= c <= 47 or 90 <= c <= 97 or 100 <= c <= 107
        if not all(allowed(c) for c in codes):
            return
        fg, bg, bold = self.style.foreground, self.style.background, self.style.bold
        for c in codes:
            if c == 0:
                fg, bg, bold = None, None, False
            elif c in (1, 22):
                bold = c == 1
            elif c == 39:
                fg = None
            elif c == 49:
                bg = None
            elif 30 <= c <= 37 or 90 <= c <= 97:
                fg = c
            else:
                bg = c
        self.style = Style(fg, bg, bold)


def parse(text):
    return StyleParser().feed(text, final=True)


def _sgr(style):
    codes = [0]
    if style.bold:
        codes.append(1)
    if style.foreground is not None:
        codes.append(style.foreground)
    if style.background is not None:
        codes.append(style.background)
    return '\x1b[' + ';'.join(map(str, codes)) + 'm'


def pages(document, presentation=None, *, continuation=''):
    """Wrap visible cells and bound every self-contained UTF-8 wire page.

    Input span text is filtered again, so constructing mutable spans cannot
    bypass the terminal boundary. Long words are split at cell boundaries.
    """
    presentation = presentation or Presentation()
    if isinstance(document, str):
        document = parse(document)
    safe = StyledText()
    for span in document.spans:
        clean = parse(span.text)
        # Styles produced by untrusted callers must also be in the allowlist.
        style = span.style
        if (style.foreground not in (None, *range(30, 38), *range(90, 98))
                or style.background not in (None, *range(40, 48), *range(100, 108))):
            style = Style()
        safe.spans.extend(Span(part.text, style) for part in clean.spans)
    ansi = presentation.mode == 'ansi16' and presentation.ansi_capable
    reset = '\x1b[0m' if ansi else ''
    rendered, chunk, current, column, lines = [], '', None, 0, 1
    for span in safe.spans:
        for char in span.text:
            width = cell_width(char)
            wrap = char != '\n' and column + width > presentation.width
            prefix = '\n' if wrap else ''
            sequence = _sgr(span.style) if ansi and current != span.style else ''
            addition = prefix + sequence + char
            new_lines = lines + int(wrap or char == '\n')
            if chunk and (len((chunk + addition + reset + continuation).encode()) > presentation.page_bytes
                          or new_lines > presentation.height - 2 - bool(continuation)):
                rendered.append(chunk + reset + continuation)
                chunk, current, column, lines = '', None, 0, 1
                prefix = ''
                sequence = _sgr(span.style) if ansi else ''
                addition = sequence + char
            elif wrap:
                column, lines = 0, lines + 1
            if char == '\n':
                column, lines = 0, lines + 1
            else:
                column += width
            chunk += addition
            current = span.style
    if chunk:
        rendered.append(chunk + reset)
    return rendered


def plain(document):
    return parse(document).text if isinstance(document, str) else ''.join(parse(s.text).text for s in document.spans)


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds} sec' if seconds < 60 else f'{seconds // 60} min'


def menu(view, *, title='BBS', games=(), blocked='', expert=True):
    if expert:
        if view == 'doors':
            choices = '\n'.join(f'{n} {name}' for n, name in enumerate(games, 1))
            return f'Doors: {blocked or choices or "none"}\nDoor: ? help; Q back'
        return {'main': f'{title}\nMain: D doors; ? help',
                'bulletins': 'Bulletin: ? help; Q back', 'files': 'File: ? help; Q back',
                'messages': 'Messages: M areas; AREA name; READ id; POST subject; REPLY id; SYSOP subject; Q back',
                'preferences': 'Prefs: MENUS ON/OFF; P pause; L n; C color setup; Q back'}[view]
    if view == 'main':
        return f'{title}\n[D] Doors  [B] Bulletins  [F] Files  [N] News  [M] Messages\n[T] Time left  [P] Preferences  [?] Help  [G] Goodbye\nMENUS ON: classic; MENUS OFF: terse\nMain:'
    if view == 'doors':
        choices = '\n'.join(f'[{n}] {name}' for n, name in enumerate(games, 1))
        return f'Doors\n{blocked or choices or "No games installed and available."}\n[Q] Main menu  [T] Time left  [G] Goodbye\nDoor:'
    if view in ('files', 'bulletins'):
        return f'{view.title()}\nREAD N reads a text bulletin. GET N requests a supported transfer.\n[Q] Main menu  [G] Goodbye'
    if view == 'messages':
        return 'Messages\nM areas; AREA name; READ id; POST subject; REPLY id; SYSOP subject; Q back.'
    return 'Preferences\nMENUS ON: classic menus; MENUS OFF: terse (default)\n[E] Toggle terse mode  [P] Pause pages  [L N] Page length\n[C] Color setup (explicit per attachment)\n[Q] Main menu  [G] Goodbye'
