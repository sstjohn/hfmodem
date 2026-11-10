# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Static regression checks supplement the browser and assistive-technology audit."""
import tempfile
import unittest
import re
from pathlib import Path
from html.parser import HTMLParser
import build


class Semantics(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def contrast(first, second):
    def luminance(colour):
        channels = [int(colour[i:i+2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class Accessibility(unittest.TestCase):
    def test_code_token_colours_meet_aa_on_the_code_panel_in_both_themes(self):
        css = (build.ROOT / 'site/style.css').read_text()
        light, dark = re.search(r':root \{(.*?)\}', css, re.S)[1], re.search(
            r'@media\(prefers-color-scheme:dark\) \{\s*:root \{(.*?)\}', css, re.S)[1]
        rules = dict(re.findall(r'\.listing \.(\w) \{ color:var\(--([\w-]+)\); \}', css))
        self.assertEqual(set(rules), {'b', 'c', 'f', 'k', 'm', 's'})
        for theme in (light, dark):
            tokens = dict(re.findall(r'--([\w-]+):(#[0-9a-f]{6})', theme))
            for cls, token in rules.items():
                self.assertGreaterEqual(contrast(tokens[token], tokens['panel']), 4.5, (cls, token))

    def test_every_route_has_a_bypass_link_and_one_focusable_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            for page in root.rglob('*.html'):
                text = page.read_text()
                tags = Semantics(text).tags
                mains = [attrs for tag, attrs in tags if tag == 'main']
                self.assertEqual(len(mains), 1, page)
                self.assertEqual(mains[0]['tabindex'], '-1', page)
                skips = [attrs for tag, attrs in tags if tag == 'a' and attrs.get('class') == 'skip']
                self.assertEqual([x['href'] for x in skips], ['#'+mains[0]['id']], page)
                self.assertLess(text.index('class="skip"'), text.index('class="release-notice"'), page)
                self.assertIn('accessibility.css', text, page)
                self.assertIn('>Accessibility</a>', text, page)

    def test_each_inspector_offers_non_audio_non_pointer_routes(self):
        for route in ('vara-send','vara-receive','pactor3-send','pactor3-receive','ardop-send','ardop-receive'):
            text = (build.ROOT / 'site/examples' / route / 'index.html').read_text()
            tags = Semantics(text).tags
            controls = {attrs.get('id'): (tag, attrs) for tag, attrs in tags if attrs.get('id')}
            self.assertEqual(controls['clock'][1]['aria-live'], 'off')
            self.assertEqual(controls['seek'][0], 'input')
            self.assertEqual(controls['seek'][1]['aria-label'], 'Playback position')
            for anchor in ('transport','frame-reading','frames','signal-text'):
                self.assertIn(f'href="#{anchor}"', text)
            self.assertNotIn('role="listitem"', text)
            for name in ('waterfall','timeline','constellation'):
                self.assertEqual(controls[name][1]['role'], 'img')
            self.assertIn('not a complete transcript', text)
            self.assertIn('unidentified intervals are unknown', text)
            self.assertIn('id="eventsNext"', text)
