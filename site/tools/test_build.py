# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static-link checks: no browser script is needed to resolve site navigation."""
import tempfile
import unittest
import base64
import hashlib
import json
import re
import importlib.util
import tomllib
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
    def test_visible_guides_share_mode_navigation_and_current_page(self):
        expected_routes = ['compared', 'pactor', 'pactor-4', 'vara', 'ardop', 'sabir',
                           'robust-packet', 'stanag', 'rtty']
        expected_labels = ['Comparison', 'PACTOR', 'PACTOR-4', 'VARA', 'ARDOP', 'sabir',
                           'Robust Packet', 'STANAG', 'RTTY / SITOR / NAVTEX']

        class ModeLinks(HTMLParser):
            def __init__(self, text):
                super().__init__()
                self.links = []
                self.active = None
                self.feed(text)

            def handle_starttag(self, tag, attrs):
                if tag == 'a':
                    self.active = dict(attrs)
                    self.active['label'] = ''
                    self.links.append(self.active)

            def handle_data(self, data):
                if self.active is not None:
                    self.active['label'] += data

            def handle_endtag(self, tag):
                if tag == 'a':
                    self.active = None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            for filename, _ in build.MODE_GUIDES:
                current = build.ROUTES[filename]
                source = (build.ROOT / 'docs/viz' / filename).read_text()
                rendered = (root / current / 'index.html').read_text()
                for kind, text in [('standalone', source), ('site', rendered)]:
                    with self.subTest(page=current, kind=kind):
                        nav = re.search(r'<nav class="site-nav"[^>]*>.*?</nav>', text, re.S)
                        self.assertIsNotNone(nav)
                        links = ModeLinks(nav[0]).links
                        self.assertEqual([a['data-route'] for a in links], expected_routes)
                        self.assertEqual([a['label'] for a in links], expected_labels)
                        self.assertEqual([a['data-route'] for a in links
                                          if a.get('aria-current') == 'page'], [current])
                        self.assertTrue(all(a['href'] == ('../'+a['data-route']+'/'
                                                         if kind == 'site' else
                                                         next(f for f, r in build.ROUTES.items()
                                                              if r == a['data-route'])) for a in links))
                # The shared chrome rules must follow local diagram styles in
                # standalone files as well as the site, or page rules win again.
                self.assertGreater(source.index('href="guide.css"'), source.rfind('</style>'))

    def test_every_local_link_and_fragment_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            build.build(root)
            pages = list(root.rglob('*.html'))
            primary = {('docs/'+name) for name, _, _ in build.DOCUMENTS} | {('docs/viz/'+name) for name in build.ROUTES}
            rendered = {source for source in build.REFERENCE_FILES - primary if not source.endswith('.z')}
            self.assertEqual(len(pages), 31 + len(rendered) + len(build.MOVED))
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

    def test_recordings_the_comparison_figures_link_are_published(self):
        # The figure script builds these links at run time, out of reach of the
        # static link check above.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            page = (root / 'compared/index.html').read_text()
            self.assertIn('href="../recordings/${c.audio}"', page)
            figures = json.loads(re.search(r'^const FIG = (.*);$', page, re.M)[1])
            linked = [c['audio'] for c in figures['constellations'].values() if 'audio' in c]
            self.assertTrue(linked)
            for audio in linked:
                self.assertTrue((root / 'recordings' / audio).is_file(), audio)

    def test_released_routes_of_moved_sources_forward_to_the_new_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            build.build(root)
            for old, new in build.MOVED.items():
                self.assertNotIn(old, build.REFERENCE_FILES)
                self.assertIn(new, build.REFERENCE_FILES)
                page = root / build.reference_route(old)
                text = page.read_text()
                url = re.search(r'http-equiv="refresh" content="0; url=([^"]+)"', text)[1]
                self.assertIn(f'location.replace({json.dumps(url)} + location.hash)', text)
                self.assertIn(f'<a href="{url}">', text)
                dest = (page.parent / url).resolve()
                self.assertEqual(dest / 'index.html' if dest.is_dir() else dest,
                                 root / build.reference_route(new), old)

    def test_pactor4_protocol_and_prng_are_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            page = (root / 'pactor-4/index.html').read_text()
            self.assertNotIn('name="robots" content="noindex"', page)
            for text in ('id="prng"', '2147483563', '2147483399', '1160', 'unresolved', '20N'):
                self.assertIn(text, page)
            self.assertIn('href="../pactor-4/"', (root / 'pactor/index.html').read_text())
            reference = root / build.reference_route('docs/protocols/pactor/pactor4-prng.md')
            self.assertTrue(reference.is_file())
            self.assertIn('67108862', reference.read_text())

    def test_operator_guides_render_code_and_local_navigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            console = (root / 'console/index.html').read_text()
            self.assertIn('<pre><code class="language-sh">', console)
            self.assertIn('hfmodem winlink', console)
            self.assertIn('python3 -m pip install --pre hfmodem', console)
            self.assertIn('Start this mail call?', console)
            self.assertIn('hfmodem console setup --replace', console)
            self.assertIn('hfmodem winlink --inbox', console)
            self.assertIn('Send a text message and receive mail', console)
            self.assertIn('href="../monitor/"', console)
            self.assertIn('href="../hflink/"', console)
            self.assertNotIn('```', console)
            monitor = (root / 'monitor/index.html').read_text()
            self.assertIn('href="../hflink/"', monitor)
            index = (root / 'index.html').read_text()
            for route in ('console/', 'monitor/', 'hflink/'):
                self.assertIn(f'href="{route}"', index)

    def test_hidden_desk_is_packaged_without_a_visible_invitation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            room = (root / 'demo/index.html').read_text()
            self.assertIn('Every radio action is simulated', room)
            self.assertIn('Browser demo · no radio attached', room)
            self.assertNotIn('secret-desk.mjs', room)
            self.assertEqual((root / 'js/secret-desk.mjs').read_bytes(),
                             (build.ROOT / 'site/js/secret-desk.mjs').read_bytes())
            for page in root.rglob('*.html'):
                if page.is_relative_to(root / 'demo'):
                    continue
                self.assertEqual(page.read_text().count('secret-desk.mjs'), 1, page)
                for href in Links(page.read_text()).hrefs:
                    path = (page.parent / urlsplit(href).path).resolve()
                    self.assertFalse(path.is_relative_to(root / 'demo'), (page, href))
            # The finalized shell imports only reviewed text assets; recordings
            # and public evidence stay in their existing inspector routes.
            def files(tree):
                return {p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()}
            self.assertEqual(files(root / 'demo'),
                             {'index.html','README.md','desk.css','console-tools.html',
                              'console-tools.css','console-schema.json','favicon.svg',
                              'logbook.html','logbook-ui.css'})
            self.assertEqual(files(root / 'js/demo'),
                             {'public-qsos.mjs','desk.js','accessibility.js','signal-palette.js',
                              'embedded-theme.js','console-tools.js','logbook-desk.js',
                              'public-logbook.mjs'})

    def test_fenced_code_preserves_blank_lines_and_escapes_html(self):
        self.assertEqual(build.markdown('```text\n<a>\n\n&\n```'),
                         '<pre><code class="language-text">&lt;a&gt;\n\n&amp;</code></pre>')

    def test_release_warning_precedes_content_on_every_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            for page in root.rglob('*.html'):
                text = page.read_text()
                self.assertEqual(text.count('class="release-notice"'), 1, page)
                warning = text.index('class="release-notice"')
                for anchor in ('<header', '<main', '<h1'):
                    if anchor in text:
                        self.assertLess(warning, text.index(anchor), page)
                self.assertIn('aria-label="Experimental prerelease warning"', text)
                self.assertIn('experimental prerelease for developers only', text)
                self.assertNotIn('alpha release for developers only', text)
                notice = re.search(r'<aside class="release-notice"[^>]*>(.*?)</aside>', text, re.S)[1]
                if page == root / 'index.html':
                    self.assertIn('especially emergency communications.</strong></p>', notice)
                    self.assertNotIn('Recorded Winlink mail transfers used', notice)
                else:
                    self.assertNotIn('Do not rely on it', notice)
                    self.assertNotIn('Recorded Winlink mail transfers used', text)

    def test_reference_catalog_is_a_subset_of_the_public_manifest(self):
        manifest = build.ROOT / 'publish/manifest.toml'
        if not manifest.exists():
            self.skipTest('Publication policy is maintained in the private release tree')
        spec = importlib.util.spec_from_file_location('site_public_manifest', manifest.with_name('build.py'))
        publisher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(publisher)
        man = tomllib.loads(manifest.read_text())
        selected = publisher.expand(man)
        self.assertFalse(publisher.denied(man, selected))
        selected = {p.relative_to(build.ROOT).as_posix() for p in selected}
        self.assertFalse(build.REFERENCE_FILES - selected)

    def test_private_traversal_and_unlisted_source_cannot_be_published(self):
        private_doc = '/'.join(('working', 'not-for-publication.md'))
        for path in ('../' + private_doc, private_doc,
                     '/etc/passwd', 'docs/../' + private_doc):
            with self.subTest(path=path), self.assertRaises(ValueError):
                build.reference_inventory([path])
        self.assertIsNone(build.link('../../' + private_doc))
        self.assertIsNone(build.link('../publish/manifest.toml'))
        self.assertIsNone(build.link('../../../etc/passwd'))

    def test_constants_map_anchors_land_on_the_named_source_symbols(self):
        for mode in ('pactor', 'vara'):
            guide = build.ROOT / 'docs/protocols' / mode / 'implementation-constants.md'
            links = re.findall(r'\[([^]]+)\]\((../../../packages/[^)]+\.py)#L(\d+)\)', guide.read_text())
            self.assertGreater(len(links), 20, mode)
            for label, relative, number in links:
                source = (guide.parent / relative).resolve()
                lines = source.read_text().splitlines()
                symbol = label.split('.')[-1]
                line = lines[int(number)-1]
                self.assertRegex(line, r'\b'+re.escape(symbol)+r'\b', (guide, label, line))
                self.assertIn(source.relative_to(build.ROOT).as_posix(), build.REFERENCE_FILES)

    def test_generated_high_speed_geometry_and_existing_tables_agree(self):
        vara = (build.ROOT / 'docs/viz/vara-explained.html').read_text()
        pactor = (build.ROOT / 'docs/viz/pactor-explained.html').read_text()
        vf = json.loads(re.search(r'const FIG = (.*);', vara)[1])
        pf = json.loads(re.search(r'const FIG = (.*);', pactor)[1])
        self.assertEqual([len(vf['highspeed']['points'][str(p)]) for p in (1,2,3,4)], [4,8,16,32])
        self.assertEqual(len(vf['highspeed']['rows']), 21)
        self.assertEqual(vf['mfsk']['connected_ack_symbols'], len(vf['mfsk']['connected_ack']))
        self.assertNotIn('2074 samples', vara)
        self.assertNotIn('hop = 2042', vara)
        self.assertNotIn('genBurst("K7ABC","ACK")', vara)
        self.assertIn('data-fig="mfsk.stride"', vara)
        self.assertIn('data-fig="mfsk.hop"', vara)
        self.assertNotIn('p4', pf)
        hidden = (build.ROOT / 'docs/viz/pactor4-explained.html').read_text()
        p4 = json.loads(re.search(r'const GUIDE_FACTS = (.*?);', hidden)[1])
        self.assertEqual(len(p4['c2_sizes']), 13)
        self.assertEqual(p4['c2_1160_corrections'], 22)
        self.assertNotIn('Full-frame interleaver geometry remains unresolved', pactor)

    def test_runtime_permutation_downloads_preserve_the_reviewed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            resources = [source for source in build.REFERENCE_FILES if source.endswith('.z')]
            self.assertEqual(len(resources), 13)
            for source in resources:
                self.assertEqual((root / build.reference_route(source)).read_bytes(),
                                 (build.ROOT / source).read_bytes())

    def test_mode_maturity_labels_are_separate_from_rf_results(self):
        text = (build.ROOT / 'site/index.html').read_text()
        modes = text.split('<section id="modes"', 1)[1].split('</section>', 1)[0]
        results = text.split('<section id="results"', 1)[1].split('</section>', 1)[0]
        self.assertIn('PACTOR-1/2/3 and VARA are beta. ARDOP, voice and CW are alpha. PACTOR-4 and all other modes are experimental.', modes)
        for bird in ('shrike', 'kestrel'):
            self.assertIn(bird+' · beta', modes)
        for bird in ('besra', 'macagua + merlin'):
            self.assertIn(bird+' · alpha', modes)
        comparison = (build.ROOT / 'docs/viz/flock-compared.html').read_text()
        self.assertIn('shrike · beta · P4 experimental', comparison)
        self.assertIn('kestrel · beta', comparison)
        self.assertIn('besra · alpha', comparison)
        for bird in ('sabir', 'brubru', 'bateleur', 'lanner'):
            self.assertIn(bird+' · experimental', modes)
        for name in ('Robust Packet', 'STANAG', 'RTTY', 'SITOR', 'NAVTEX'):
            self.assertNotIn(name, results)

    def test_source_sibling_links_resolve(self):
        for filename in build.ROUTES:
            page = build.ROOT / 'docs/viz' / filename
            for href in Links(page.read_text()).hrefs:
                if href in build.ROUTES:
                    self.assertTrue((page.parent / href).is_file())

    def test_pactor4_support_limits_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            for route in ('index.html', 'docs/index.html', 'status/index.html', 'pactor/index.html'):
                text = (root / route).read_text()
                self.assertIn('PACTOR-4', text, route)
            for route in ('status/index.html', 'pactor/index.html'):
                text = (root / route).read_text()
                self.assertIn('No successful on-air PACTOR-4 mail transfer', text, route)
                self.assertIn('SL6–10', text, route)
                self.assertIn('offline', text, route)
                self.assertIn('unimplemented', text, route)

    def test_varac_recording_and_evidence_index(self):
        page = build.ROOT / 'site/examples/varac-receive'
        provenance = json.loads((page / 'provenance.json').read_text())
        audio = page / provenance['audio']
        self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), provenance['audio_sha256'])
        self.assertLess(audio.stat().st_size, 4_000_000)
        packed = int.from_bytes(audio.read_bytes()[18:26], 'big')
        self.assertEqual(packed >> 44, provenance['sample_rate'])
        self.assertEqual(packed & ((1 << 36) - 1), provenance['samples'])
        self.assertEqual([f['level'] for f in provenance['decoded_frames']], [6, 7, 6, 5])
        self.assertIsNone(provenance['rf_endpoints'])
        for frame in provenance['decoded_frames']:
            self.assertLess(frame['start_s'], frame['end_s'])
            self.assertLessEqual(frame['end_s'], provenance['duration_s'])
            self.assertIn(f"{frame['start_s']:.2f}–{frame['end_s']:.2f} s", (page / 'index.html').read_text())
        for source in build.EVIDENCE_DOWNLOADS:
            self.assertIsNotNone(build.link('../../' + source,
                                          'evidence/vara-mail-0918/README.md',
                                          'reference/evidence/vara-mail-0918/index.html'))
        self.assertIsNotNone(build.link('../site/recordings/vara-ofdm-k5dat-0930.json',
                                       'evidence/README.md', 'reference/evidence/index.html'))
        for name in ('index.html', 'provenance.json'):
            text = (page / name).read_text()
            for private in ('/Users/', '/file/', 'working/', '/api/'):
                self.assertNotIn(private, text)
        self.assertIn('href="varac-receive/"', (build.ROOT / 'site/examples/index.html').read_text())
        self.assertIn('evidence/README.md', build.REFERENCE_FILES)
        self.assertEqual(build.link('../site/examples/varac-receive/index.html',
                                    'evidence/README.md', 'reference/evidence/index.html'),
                         '../../examples/varac-receive/index.html')

    def test_recorded_examples_have_complete_data_and_recordings(self):
        def references(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('symbols', 'signal') and isinstance(item, str):
                        yield item
                    else:
                        yield from references(item)
            elif isinstance(value, list):
                for item in value:
                    yield from references(item)

        for route in ('pactor3-send', 'pactor3-receive', 'vara-send', 'vara-receive',
                      'ardop-send', 'ardop-receive', 'varac-bbs'):
            page = build.ROOT / 'site/examples' / route
            data = page / 'data'
            manifest = json.loads((page / 'manifest.json').read_text())
            role = route.rsplit('-', 1)[1]
            self.assertEqual(manifest['example_role'], role)
            if role != 'bbs':
                self.assertGreater(manifest['session']['mail']['sent' if role == 'send' else 'received'], 0)
            events = json.loads((page / manifest['events']).read_text())
            self.assertTrue(events)
            self.assertTrue(manifest['qso_only'])
            self.assertTrue(all(event['direction'] in ('tx', 'rx') for event in events))
            self.assertEqual(manifest['caller'], 'W9SSJ')
            self.assertEqual(manifest['called'], manifest['call'])
            self.assertTrue(all(event['role'] in ('caller', 'called') for event in events))
            self.assertTrue(any(event['status'].startswith('CRC pass') for event in events))
            verified = [event for event in events if event['status'] == 'CRC pass' and event.get('payloads')]
            self.assertTrue(any(event['direction'] == ('tx' if role == 'send' else 'rx') for event in verified))
            for event in verified:
                for payload in event['payloads']:
                    self.assertEqual(len(bytes.fromhex(payload['hex'])), payload['length'])
            for name in set(references(events)):
                item = json.loads((data / name).read_text())
                if 'map_base64' in item:
                    self.assertEqual(len(base64.b64decode(item['map_base64'])),
                                     item['rows'] * item['bins'])
            wf = manifest['waterfall']
            for tile in wf['tiles']:
                item = json.loads((data / tile['file']).read_text())
                self.assertEqual(len(base64.b64decode(item['raster_base64'])),
                                 tile['rows'] * wf['bins'])
            for source in manifest['sources']:
                if not source['audio_available']:
                    self.assertIsNone(source['url'])
                    continue
                audio = page / source['url']
                self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(),
                                 source['file_sha256'])
            for path in (page / 'index.html', page / 'manifest.json', page / 'events.json'):
                text = path.read_text()
                for private in ('/Users/', '/file/', '/api/inspect', 'working/', 'captures/'):
                    self.assertNotIn(private, text, (path, private))

    def test_replay_modules_share_an_explicit_asset_revision(self):
        script = (build.ROOT / 'site/js/examples/viewer.js').read_text()
        revision = re.search(r"from './link-state\.js\?v=([^']+)'", script).group(1)
        for page in (build.ROOT / 'site/examples').glob('*/index.html'):
            html = page.read_text()
            if 'id="workspace"' in html:
                self.assertIn(f'viewer.js?v={revision}', html, page)

    def test_preview_disables_cache_for_in_place_rebuilds(self):
        import io
        from preview import RecordingHandler
        handler = object.__new__(RecordingHandler)
        handler.request_version = 'HTTP/1.1'
        handler.wfile = io.BytesIO()
        handler.end_headers()
        headers = handler.wfile.getvalue()
        self.assertIn(b'Cache-Control: no-store', headers)
        self.assertIn(b'Accept-Ranges: bytes', headers)

    def test_bbs_transfer_provenance_and_evidence_navigation(self):
        page = build.ROOT / 'site/examples/varac-bbs'
        proof = json.loads((page / 'provenance.json').read_text())
        manifest = json.loads((page / 'manifest.json').read_text())
        events = json.loads((page / 'events.json').read_text())
        self.assertEqual(proof['last_sample'] - proof['first_sample'], 118 * 48000)
        self.assertEqual(hashlib.sha256((page / proof['audio']).read_bytes()).hexdigest(), proof['audio_sha256'])
        self.assertEqual(proof['download']['size'], 2516)
        self.assertEqual(proof['download']['sha256'], '44c36a9969d42021d608be4eee4b61db5ef7c9c8762e07405728d41ce45497b1')
        reply = next(e for e in events if e['id'] == proof['download']['completion_reply_event'])
        self.assertEqual(reply['direction'], 'tx')
        self.assertEqual(reply['status'], 'CRC pass')
        self.assertTrue(any(b'<SFOK>' in bytes.fromhex(p['hex']) for p in reply['payloads']))
        self.assertTrue(any(e.get('level') == 11 and e['direction'] == 'rx' and e['status'] == 'CRC pass' for e in events))
        self.assertIn('File completion', manifest['outcome_note'])
        self.assertIn('approximate', proof['timing_basis'])
        home = (build.ROOT / 'site/index.html').read_text()
        index = (build.ROOT / 'site/examples/index.html').read_text()
        self.assertNotIn('PACTOR-1 timing', home)
        self.assertNotIn('examples/varac-receive/', home)
        self.assertIn('examples/varac-bbs/', home)
        for heading in ('vara', 'pactor', 'ardop'):
            self.assertIn(f'id="{heading}"', index)
        self.assertIn('../../js/examples/viewer.js', (page / 'index.html').read_text())

    def test_curated_logbook_roles_speeds_and_legacy_routes(self):
        expected = {'vara-send': ('L9', 'NS0A', 9),
                    'vara-receive': ('L4', 'N5TW', 4),
                    'pactor3-send': ('P3 SL6', 'VE1YZ', 6),
                    'pactor3-receive': ('P3 SL3', 'WS8EOC', 3),
                    'ardop-send': ('4FSK.200.50S', 'WW2MI', None),
                    'ardop-receive': ('4PSK.200.100', 'W6IDS', None)}
        examples = build.ROOT / 'site/examples'
        entries = json.loads((examples / 'logbook/entries.json').read_text())
        self.assertEqual({row['route'] for row in entries}, set(expected))
        logbook = (examples / 'logbook/index.html').read_text()
        home = (build.ROOT / 'site/index.html').read_text()
        for route, (speed, peer, level) in expected.items():
            self.assertIn(f'href="../{route}/"', logbook)
            self.assertIn(f'href="{route}/"', (examples / 'index.html').read_text())
            m = json.loads((examples / route / 'manifest.json').read_text())
            role = route.rsplit('-', 1)[1]
            self.assertEqual(m['session']['speed']['acknowledged' if role == 'send' else 'decoded'], speed)
            self.assertEqual(m['call'], peer)
            events = json.loads((examples / route / 'events.json').read_text())
            self.assertTrue(any(e['status'] == 'CRC pass' and e.get('payloads')
                                and e['direction'] == ('tx' if role == 'send' else 'rx')
                                and (e['kind'].startswith(speed + '.') if level is None else
                                     e.get('level') == level or e['kind'] == f'Base data L{level}')
                                for e in events), route)
        self.assertIn('original ARDOP rig recordings were lost', logbook)
        self.assertIn('remote KiwiSDR recordings', logbook)
        proof = json.loads((examples / 'vara-receive/manifest.json').read_text())['alignment_proof']
        self.assertEqual(proof['anchor_count'], proof['available_anchor_count'])
        self.assertEqual(proof['anchor_count'], 16)
        self.assertEqual(len(proof['anchor_types']), 3)
        self.assertLess(proof['rival_anchor_count'], proof['anchor_count'])
        self.assertLess(proof['max_residual_s'], .05)
        for old, new in (('vara', 'vara-send'), ('pactor3', 'pactor3-send')):
            self.assertIn(f'url=../{new}/', (examples / old / 'index.html').read_text())
            self.assertFalse((examples / old / 'audio').exists())

    def test_ardop_sender_evidence_and_partial_send_outcome(self):
        for route, session in (('ardop-send', 243), ('ardop-receive', 13)):
            page = build.ROOT / 'site/examples' / route
            m = json.loads((page / 'manifest.json').read_text())
            events = json.loads((page / 'events.json').read_text())
            proof = m['alignment_proof']
            tx = proof['tx']
            self.assertGreater(tx['anchor_count'], 40)
            self.assertLess(tx['rival_anchor_count'], 10)
            self.assertLess(tx['max_residual_s'], .06)
            anchors = {a['event_id']: a for a in tx['anchors']}
            self.assertEqual(len(anchors), tx['anchor_count'])
            self.assertEqual(proof['session_id'], session)
            self.assertFalse(set(proof['peer_only_frame_kinds']) & set(proof['local_frame_kinds']))
            for e in events:
                if e['direction'] == 'tx':
                    anchor = anchors[e['id']]
                    self.assertEqual(e['kind'], anchor['kind'])
                    self.assertAlmostEqual(e['t0'], anchor['recording_s'])
                    self.assertLess(abs(e['t0'] - anchor['log_s'] - tx['offset_s']), .06)
                else:
                    self.assertEqual(e['status'], 'CRC pass')
                    self.assertIn(e['kind'], proof['peer_only_frame_kinds'])
                    self.assertTrue(e['payloads'])
                for payload in e.get('payloads', []):
                    self.assertEqual(payload['session_id'], session)
            self.assertEqual(m['sources'][0]['kind'], 'sdr witness')
            caption = (page / 'index.html').read_text()
            self.assertIn('original rig receiver recording was lost', caption)
            self.assertIn('href="../logbook/#data-loss"', caption)
            if route == 'ardop-send':
                self.assertIn('three offered messages were not fetched', caption)
                self.assertIn('END was not confirmed', m['comparison']['original_summary'])
                self.assertEqual(m['session']['mail']['received'], 0)

    def test_public_markdown_preserves_text_outside_omissions(self):
        source = ('# Public\n\n<!-- site:omit:start -->\n'
                  'Private section\n<!-- site:omit:end -->\n\nVisible tail')
        self.assertEqual(build.public_markdown(source), '# Public\n\n\nVisible tail')

    def test_public_markdown_drops_comment_lines_outside_code(self):
        source = ('<!-- SPDX-License-Identifier: AGPL-3.0-only -->\n# Public\n\n'
                  '```html\n<!-- kept -->\n```')
        self.assertEqual(build.public_markdown(source),
                         '# Public\n\n```html\n<!-- kept -->\n```')

    def test_no_page_shows_a_licence_header_outside_source_listings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.build(root)
            for page in root.rglob('*.html'):
                text = re.sub(r'<!--.*?-->|<pre\b.*?</pre>', '', page.read_text(), flags=re.S)
                self.assertNotIn('SPDX-License-Identifier', text, page.relative_to(root))

    def test_public_markdown_rejects_broken_boundaries(self):
        for source in ('<!-- site:omit:end -->', '<!-- site:omit:start -->',
                       '<!-- site:omit:start -->\n<!-- site:omit:start -->'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                build.public_markdown(source)

    def test_headings_render_after_omission_filtering(self):
        for blank in ('\n', '\n\n', '\n \n\t\n'):
            source = ('# Public\n' + blank + '<!-- site:omit:start -->\n'
                      'Private section\n<!-- site:omit:end -->\n' + blank +
                      '## Running hfmodem\n\nVisible tail')
            with self.subTest(blank=blank):
                self.assertEqual(
                    build.markdown(build.public_markdown(source)),
                    '<h1 id="public">Public</h1>\n'
                    '<h2 id="running-hfmodem">Running hfmodem</h2>\n'
                    '<p>Visible tail</p>')

    def test_inline_code_is_escaped(self):
        self.assertEqual(build.inline('`<a>`'), '<code>&lt;a&gt;</code>')
