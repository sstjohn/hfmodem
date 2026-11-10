# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import re

import pytest

from hfmodem.bbs.model import Presentation
from hfmodem.bbs.presentation import Span, Style, StyleParser, StyledText, cell_width, pages, parse, plain


def test_fragmented_sgr_crlf_and_raw_controls():
    parser = StyleParser()
    first = parser.feed('A\r')
    second = parser.feed('\n\x1b[9')
    third = parser.feed('1;1mR\x1b[0m\tX\rY\b\x07\x85Z', final=True)
    assert first.text + second.text + third.text == 'A\nR       XYZ'
    assert third.spans[0].style == Style(91, None, True)
    assert '\x1b' not in third.text


@pytest.mark.parametrize('control', [
    '\x1b[2J', '\x1b[?25l', '\x1b]52;c;clipboard\x07',
    '\x1b]0;title\x1b\\', '\x1bPgarbage\x1b\\', '\x9b31m',
    '\x9dsecret\x9c', '\x1b[38;5;1m', '\x1b[' + '1' * 10000 + 'm',
])
def test_non_color_controls_and_large_sequences_never_escape(control):
    parser = StyleParser(max_escape=16)
    result = StyledText()
    for char in 'before' + control + 'after':
        result.spans.extend(parser.feed(char).spans)
        assert len(parser.sequence) <= 16
    result.spans.extend(parser.feed('', final=True).spans)
    assert result.text == 'beforeafter'
    assert plain(''.join(pages(result, Presentation('ansi16', ansi_capable=True)))) == 'beforeafter'


def test_incomplete_escape_is_discarded_and_parser_recovers_on_final():
    parser = StyleParser()
    assert parser.feed('text\x1b]secret', final=True).text == 'text'
    assert parser.feed('visible').text == 'visible'


def test_mutable_spans_safe_plain_and_ansi_same_words():
    doc = StyledText([Span('Red', Style(31)), Span(' room')])
    doc.spans[0].text += ' door\x1b]52;c;attack\x07'
    plain_pages = pages(doc)
    color_pages = pages(doc, Presentation('ansi16', ansi_capable=True))
    assert ''.join(plain_pages) == plain(''.join(color_pages)) == 'Red door room'
    assert color_pages[0].endswith('\x1b[0m')
    assert '\x1b[31m' not in color_pages[0]  # canonical renderer establishes reset too


def test_utf8_byte_bound_visible_cells_and_self_contained_pages():
    doc = parse('\x1b[1;94m' + '東e\u0301 ' * 120)
    config = Presentation('ansi16', width=20, height=8, page_bytes=80, ansi_capable=True)
    output = pages(doc, config)
    assert len(output) > 10
    for page in output:
        assert len(page.encode()) <= 80
        assert page.startswith('\x1b[') and page.endswith('\x1b[0m')
        assert all(sum(cell_width(c) for c in line) <= 20 for line in plain(page).splitlines())
        assert all(match.group() in ('\x1b[0m', '\x1b[0;1;94m')
                   for match in re.finditer(r'\x1b\[[^m]*m', page))


def test_unconfirmed_ansi_and_bad_dimensions_rejected():
    with pytest.raises(ValueError):
        Presentation('ansi16')
    with pytest.raises(ValueError):
        Presentation(width=1)


@pytest.mark.parametrize('byte_limit', [64, 600])
@pytest.mark.parametrize('mode', ['plain', 'ansi16'])
def test_pager_suffix_is_inside_wire_budget_and_outside_backend_style(byte_limit, mode):
    suffix = '\n[BBS MORE | R | Q]'
    rendered = pages('\x1b[31m' + '東abc ' * 200,
                     Presentation(mode, page_bytes=byte_limit, ansi_capable=mode == 'ansi16'),
                     continuation=suffix)
    assert len(rendered) > 1
    assert all(len(page.encode()) <= byte_limit for page in rendered)
    assert all(page.endswith(suffix) for page in rendered[:-1])
    if mode == 'ansi16':
        assert all(page.endswith('\x1b[0m' + suffix) for page in rendered[:-1])
    assert not rendered[-1].endswith(suffix)
