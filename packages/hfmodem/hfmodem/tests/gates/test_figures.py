# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The visualization pages' figures still match the tree they were drawn from.

`docs/viz/` holds five pages — the cross-protocol comparison and one description
each of PACTOR-III, VARA HF, ARDOP and sabir — and every one of them states measurements.
An off-air constellation, two received spectra, four rate ladders, a transmit
kernel, a carrier raster, a speed-level table: a figure that states a measurement
is a claim. `docs/viz/build.py --check` regenerates each page's generated block
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
`src` or an `@import` that cannot be reached leaves a hole where a figure was, and
the reader has no way to know what was meant to be there. A link costs nothing
until it is followed, so it is asked only to stay inside the tree the page ships
in: a fragment, or a path resolved against the page itself.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
BUILD = REPO / "docs/viz/build.py"
PAGES = sorted((REPO / "docs/viz").glob("*.html"))

#: What a page would have to fetch to draw itself: a subresource, a stylesheet
#: import, an absolute or protocol-relative URL. `url(` is not here: the only ones
#: in the pages are `url(data:` if any, and the rule that would catch those catches
#: every CSS gradient as well.
FETCHED = re.compile(r"""src\s*=\s*["']|@import|https?://|//cdn\.""")

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
        "run `python docs/viz/build.py` and read the diff before committing it")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_the_pages_ask_the_network_for_nothing(page: Path):
    fetched = _hits(page, FETCHED)
    assert not fetched, "\n".join(fetched)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_link_leaves_the_tree_the_pages_ship_in(page: Path):
    departs = _hits(page, DEPARTS)
    assert not departs, "\n".join(departs)
