# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static-link checks: no browser script is needed to resolve site navigation."""
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
import build


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.hrefs = []
        self.assets = []
        self.ids = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.hrefs.append(attrs['href'])
        if tag == 'link' and 'href' in attrs:
            self.assets.append(attrs['href'])
        if tag in ('script', 'img') and 'src' in attrs:
            self.assets.append(attrs['src'])


class SiteBuild(unittest.TestCase):
    def test_every_local_link_and_fragment_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            build.build(root)
            pages = list(root.rglob('*.html'))
            self.assertEqual(len(pages), 8)
            for page in pages:
                parsed = Links(page.read_text())
                for href in parsed.hrefs + parsed.assets:
                    target = urlsplit(href)
                    if target.scheme or target.netloc:
                        continue
                    dest = (page.parent / target.path).resolve() if target.path else page
                    if dest.is_dir():
                        dest /= 'index.html'
                    self.assertTrue(dest.is_relative_to(root), (page, href))
                    self.assertTrue(dest.is_file(), (page, href))
                    if target.fragment:
                        self.assertIn(target.fragment, Links(dest.read_text()).ids, (page, href))

    def test_source_sibling_links_resolve(self):
        for filename in build.ROUTES:
            page = build.ROOT / 'docs/viz' / filename
            for href in Links(page.read_text()).hrefs:
                if href in build.ROUTES:
                    self.assertTrue((page.parent / href).is_file())

    def test_inline_code_is_escaped(self):
        self.assertEqual(build.inline('`<a>`'), '<code>&lt;a&gt;</code>')
