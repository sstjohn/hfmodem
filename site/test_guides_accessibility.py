"""Keep interactive guide diagrams named, operable and quiet on initial load."""
from html.parser import HTMLParser
from pathlib import Path
import re
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.nodes = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))


class GuideAccessibilityTests(unittest.TestCase):
    def test_every_guide_has_a_main_and_named_canvases(self):
        for path in sorted((ROOT / "docs/viz").glob("*.html")):
            with self.subTest(guide=path.name):
                nodes = Elements(path.read_text()).nodes
                self.assertEqual(sum(tag == "main" for tag, _ in nodes), 1)
                for tag, attrs in nodes:
                    if tag == "canvas":
                        self.assertEqual(attrs.get("role"), "img")
                        self.assertTrue(attrs.get("aria-label"))
                labels = {attrs.get("for") for tag, attrs in nodes if tag == "label"}
                for tag, attrs in nodes:
                    if tag == "input":
                        self.assertIn(attrs["id"], labels)

    def test_animation_is_an_explicit_page_wide_choice(self):
        for name in ("pactor-explained.html", "vara-explained.html"):
            with self.subTest(guide=name):
                source = (ROOT / "docs/viz" / name).read_text()
                self.assertIn("let guideMotion=false", source)
                self.assertIn("wfPaused=true", source)
                self.assertIn("if(guideMotion&&!document.hidden)", source)
                self.assertIn("Play diagrams", source)
                self.assertIn("Pause diagrams", source)
                self.assertNotIn("location.reload()", source)
                # Static mode must render the complete encoder/ladder, not an empty reveal.
                self.assertIn("(!guideMotion||prefersReduce)?LADDER.length", source)

    def test_ardop_motion_and_announcements_are_opt_in(self):
        source = (ROOT / "docs/viz/ardop-explained.html").read_text()
        self.assertIn("running=false", source)
        self.assertNotIn("running=!reduced.matches", source)
        self.assertIn("exchangeRunning?'off':'polite'", source)

    @unittest.skipUnless(shutil.which("node"), "Node is required to parse the inline guide scripts")
    def test_inline_scripts_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in sorted((ROOT / "docs/viz").glob("*.html")):
                scripts = re.findall(r"<script(\s[^>]*)?>([\s\S]*?)</script>", path.read_text())
                for number, (attrs, script) in enumerate(scripts):
                    with self.subTest(guide=path.name, script=number):
                        if 'type="application/json"' in attrs:
                            json.loads(script)
                            continue
                        target = Path(directory) / "script.js"
                        target.write_text(script)
                        result = subprocess.run(["node", "--check", str(target)], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
