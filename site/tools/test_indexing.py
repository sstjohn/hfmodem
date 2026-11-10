# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Search engines see one canonical URL per page and a sitemap of exactly the indexable ones."""
import html
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin
import build

HOST = 'https://hfmodem.ssj.lol/'
# The hidden operating desk is deliberately unlisted, and reference pages
# mirror files the public repository already serves.
DESK = {'demo/index.html', 'demo/logbook.html', 'demo/console-tools.html'}
PRIMARY = ({'docs/' + name for name, _, _ in build.DOCUMENTS}
           | {'docs/viz/' + name for name in build.ROUTES})
REFERENCES = {build.reference_route(source).as_posix() for source in build.REFERENCE_FILES - PRIMARY
              if not source.endswith('.z')}
# GitHub Pages serves the not-found page at every missing URL; it names none.
MISSING = '404.html'
NOINDEX = DESK | REFERENCES | {MISSING}


class Indexing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        build.build(cls.root)
        cls.pages = {page.relative_to(cls.root).as_posix(): page.read_text()
                     for page in sorted(cls.root.rglob('*.html'))}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def url(route):
        return HOST + route.removesuffix('index.html')

    @staticmethod
    def forward(text):
        match = re.search(r'http-equiv="refresh" content="0; url=([^"]+)"', text)
        return match and match[1]

    def indexable(self):
        return {route for route, text in self.pages.items()
                if route not in NOINDEX and not self.forward(text)}

    def test_canonical_host_is_the_published_custom_domain(self):
        self.assertEqual(build.SITE, HOST)

    def test_every_page_has_exactly_one_canonical_in_its_head(self):
        for route, text in self.pages.items():
            links = re.findall(r'<link\b[^>]*\brel="canonical"[^>]*>', text)
            if route == MISSING:
                self.assertEqual(links, [])
                continue
            self.assertEqual(len(links), 1, route)
            href = re.fullmatch(r'<link rel="canonical" href="([^"]+)">', links[0])[1]
            head = re.split(r'<body\b|<header\b|<main\b', text, maxsplit=1)[0]
            self.assertIn(links[0], head, route)
            target = self.forward(text)
            self.assertEqual(href, urljoin(self.url(route), target) if target else self.url(route), route)

    def test_forwarding_pages_name_a_published_destination(self):
        destinations = {self.url(route) for route, text in self.pages.items() if not self.forward(text)}
        forwards = 0
        for route, text in self.pages.items():
            if self.forward(text):
                forwards += 1
                href = re.search(r'<link rel="canonical" href="([^"]+)">', text)[1]
                self.assertIn(href, destinations, route)
        self.assertEqual(forwards, len(build.MOVED) + 2)

    def test_only_the_hidden_desk_and_reference_mirrors_are_noindexed(self):
        noindexed = {route for route, text in self.pages.items()
                     if re.search(r'<meta\b[^>]*name="robots"', text, re.I)}
        self.assertEqual(noindexed, NOINDEX)
        self.assertGreater(len(REFERENCES), 190)
        self.assertTrue(all(route.startswith('reference/') for route in REFERENCES))
        self.assertTrue({'status/index.html', 'console/index.html', 'pactor/index.html'}.isdisjoint(NOINDEX))
        for route in NOINDEX:
            self.assertIn('<meta name="robots" content="noindex">', self.pages[route], route)

    def test_sitemap_lists_every_indexable_route_and_only_those(self):
        tree = ET.parse(self.root / 'sitemap.xml')
        namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        locs = [loc.text for loc in tree.findall('s:url/s:loc', namespace)]
        self.assertEqual(len(locs), len(set(locs)))
        self.assertEqual(set(locs), {self.url(route) for route in self.indexable()})
        self.assertIn(HOST, locs)
        self.assertIn(HOST + 'pactor/', locs)
        for loc in locs:
            self.assertTrue(loc.startswith(HOST), loc)
            route = loc.removeprefix(HOST)
            self.assertIn(route + 'index.html' if not route or route.endswith('/') else route, self.pages)

    def test_robots_allows_crawling_and_names_the_sitemap(self):
        robots = (self.root / 'robots.txt').read_text()
        self.assertIn(f'Sitemap: {HOST}sitemap.xml', robots.splitlines())
        self.assertNotRegex(robots, r'(?m)^Disallow:\s*/')

    def test_indexable_pages_have_distinct_titles(self):
        titles = Counter(re.search(r'<title>(.*?)</title>', self.pages[route], re.S)[1]
                         for route in self.indexable())
        self.assertEqual([title for title, count in titles.items() if count > 1], [])

    def test_every_indexable_page_has_its_own_title_and_description(self):
        self.assertEqual(set(build.METADATA), self.indexable())
        for route in self.indexable():
            text = self.pages[route]
            title = re.search(r'<title>(.*?)</title>', text, re.S)[1]
            descriptions = re.findall(r'<meta name="description" content="([^"]*)">', text)
            self.assertEqual(len(descriptions), 1, route)
            self.assertLessEqual(len(html.unescape(title)), 65, route)
            self.assertTrue(50 <= len(html.unescape(descriptions[0])) <= 160, route)
            head = re.split(r'<body\b|<header\b|<main\b', text, maxsplit=1)[0]
            self.assertIn(descriptions[0], head, route)

    def test_indexable_pages_link_to_directory_urls_not_index_html(self):
        for route in self.indexable():
            self.assertNotRegex(self.pages[route], r'href="[^"#]*index\.html', route)


if __name__ == '__main__':
    unittest.main()
