# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The visualization pages' figures still match the tree they were drawn from.

Five generated pages in `docs/viz/` — the cross-protocol comparison and one description
each of PACTOR-III, VARA HF, ARDOP and sabir — state measurements.
An off-air constellation, two received spectra, four rate ladders, a transmit
kernel, a carrier raster, a speed-level table: a figure that states a measurement
is a claim. `docs/viz/tools/build.py --check` regenerates each page's generated block
from the shipped recordings and the constants the modems import, and reports drift
rather than repairing it. Running it here is what stops a page ageing quietly while
the code moves underneath it.

That the check has to be run against all of them is the lesson of 2026-08-27. Only
`flock-compared` had a generator and a gate, and its every generated number came
back byte-identical; the two pages beside it, which nothing rebuilt, had spent
weeks drawing waveforms the tree had withdrawn — a narrowband caller-ID burst that
does not exist and a Zadoff-Chu spreading of the PACTOR-III data field that was
retracted with `p3frame`'s seed phasors. A figure a generator does not touch is a
figure that drifts.

The self-containment test is the other half, and it covers every page rather than
one. They have to render from a file:// URL with no network, and a strict content
policy would block a fetch even where there is one, so the moment anything in one
of them points outward it is broken for the reader it was written for.

Drawing and navigating fail differently, so they are held to different rules. A
`src` or an `@import` that cannot be reached leaves a hole where a figure was.
All eight guides must draw without network dependencies. The five original
guides also promise repository-only navigation; the three experimental guides
link to primary standards as well as the shipped implementation.
"""
from __future__ import annotations

import re
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
BUILD = REPO / "docs/viz/tools/build.py"
PAGES = sorted((REPO / "docs/viz").glob("*.html"))
# These original guides promise repository-only navigation. The three newer
# experimental guides also cite external primary standards; their figures still
# have to render without fetching any resources.
REPOSITORY_LINK_PAGES = [p for p in PAGES if p.name not in {
    "robust-packet-explained.html", "stanag-explained.html", "rtty-explained.html"}]

#: Drawing dependencies, rather than hyperlinks or SVG namespace identifiers.
#: Inline data URLs need no network; CSS gradients are not URLs.
FETCHED = re.compile(r"""src\s*=\s*["'](?!data:)|@import|url\(\s*["']?(?!data:)[^\s"')]|\bfetch\s*\(""")

#: A link out of the tree the page ships in. `#` stays in the document and a
#: relative path resolves beside the file, neither of which asks anything of the
#: network; a scheme — `https:`, `mailto:`, `data:` — or a protocol-relative `//`
#: names a host the distribution has not got.
DEPARTS = re.compile(r"""href\s*=\s*["']\s*(?:[A-Za-z][A-Za-z0-9+.-]*:|//)""")


def _hits(page: Path, rule: re.Pattern[str]) -> list[str]:
    return [f"{n}: {line.strip()[:90]}"
            for n, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1)
            if rule.search(line)]


def test_the_figures_match_the_tree_they_were_drawn_from():
    done = subprocess.run([sys.executable, str(BUILD), "--check"],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, (
        f"{done.stdout}{done.stderr}"
        "\na page no longer matches what the tree produces; "
        "run `python docs/viz/tools/build.py` and read the diff before committing it")


def test_comparison_constellations_preserve_recorded_decisions():
    """Regeneration must not turn measured exports into a synthetic point cloud."""
    page = (REPO / "docs/viz/flock-compared.html").read_text()
    figures = json.loads(re.search(r"^const FIG = (.*);$", page, re.M)[1])
    constellations = figures["constellations"]
    for key, route, filename in [
        ("pactor3", "pactor3-send", "symbols-source0-symbols-pactor-00071.json"),
        ("vara", "vara-send", "symbols-source0-symbols-00057.json"),
        ("ardop", "ardop-send", "symbols-ardop-604005-64.json"),
    ]:
        raw = (REPO / "site/examples" / route / "data" / filename).read_bytes()
        recorded = json.loads(raw)
        plotted = constellations[key]
        assert plotted["asset_sha256"] == hashlib.sha256(raw).hexdigest()
        assert plotted["points"] == [list(p) for p in zip(recorded["i"], recorded["q"])]
        assert plotted["ideal"] == recorded["ideal"]
        assert plotted["normalization"] == recorded["normalization"]
    assert "phase-only" in constellations["pactor3"]["normalization"]
    assert constellations["pactor3"]["payload_bytes"] == 0
    assert constellations["vara"]["payload_bytes"] == 425
    assert constellations["ardop"]["quality"] == 89
    ofdm = constellations["vara-ofdm"]
    provenance = json.loads((REPO / "site/recordings/vara-ofdm-k5dat-0930.json").read_text())
    assert ofdm["source_sha256"] == provenance["audio_file_sha256"]
    assert ofdm["pcm_s16le_sha256"] == provenance["pcm_s16le_sha256"]
    assert ofdm["packet_sha256"] == provenance["packet_sha256"]
    assert ofdm["point_count"] == 1584 and ofdm["carriers"] == 11
    assert ofdm["frame_bytes"] == 197 and ofdm["payload_bytes"] == 194
    assert ofdm["decoded_bytes"] == 197
    assert ofdm["bytes_label"] == "decoded frame bytes (including control/CRC)"


def test_hidden_pactor4_graphics_match_the_implementation():
    import numpy as np
    from hfmodem.shrike import p4c2, p4normal, p4shorttx, p4longtx, tablegen

    page = (REPO / "docs/viz/pactor4-explained.html").read_text()
    facts = json.loads(re.search(r"const GUIDE_FACTS = (.*?);", page)[1])
    data = json.loads(re.search(r'<script id="p4-geometry" type="application/json">(.*?)</script>', page, re.S)[1])
    for level, points in data["points"].items():
        assert points == tablegen.p4_constellation_points(int(level)).tolist()
    assert facts["c2_sizes"] == sorted(p4c2.TABLE_SHA256)
    for size, displayed in data["c2"].items():
        order = p4c2.permutation(int(size))
        assert displayed["head"] == order[:64].tolist()
        assert displayed["hash"] == hashlib.sha256(order.astype('<u2').tobytes()).hexdigest()
    for name, renderer in [("short", p4shorttx.short_packet), ("long", p4longtx.long_packet)]:
        layout = p4normal.layout(name, terminal_training=True)
        assert data["layouts"][name]["total_symbols"] == layout.total_symbols
        audio = renderer(b'PACTOR-4 waveform diagram', 0, fs=48000)
        shown = data["waveforms"][name]
        assert shown["samples"] == len(audio)
        assert shown["duration_s"] == len(audio)/48000
        assert abs(shown["rms"]-float(np.sqrt(np.mean(audio**2)))) < 1e-9
        edges = np.linspace(0, len(audio), 641).astype(int)
        np.testing.assert_allclose(shown["lo"], [audio[a:b].min() for a,b in zip(edges,edges[1:])], atol=5e-7, rtol=0)
        np.testing.assert_allclose(shown["hi"], [audio[a:b].max() for a,b in zip(edges,edges[1:])], atol=5e-7, rtol=0)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_the_pages_ask_the_network_for_nothing(page: Path):
    fetched = _hits(page, FETCHED)
    assert not fetched, "\n".join(fetched)


@pytest.mark.parametrize("page", REPOSITORY_LINK_PAGES, ids=lambda p: p.name)
def test_no_link_leaves_the_tree_the_pages_ship_in(page: Path):
    departs = _hits(page, DEPARTS)
    assert not departs, "\n".join(departs)
