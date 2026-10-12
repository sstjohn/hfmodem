# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Publication markers in every protocol's documentation."""
from pathlib import Path

from .publication import MARKERS


def test_protocol_docs_have_no_publication_markers():
    root = Path(__file__).resolve().parents[5]
    hits = []
    for path in sorted((root / "docs" / "protocols").rglob("*.md")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if MARKERS.search(line):
                hits.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    assert not hits, "\n".join(hits)
