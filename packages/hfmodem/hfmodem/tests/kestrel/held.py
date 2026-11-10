# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recordings the distribution does not carry, by name.

A private map beside them names where each one is kept. Without the map a name
resolves to a path that does not exist, so a test's own absence check skips it.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from hfmodem.tests import evidence

from . import corpora


@cache
def _paths() -> dict[str, str]:
    path = corpora.HELD_RECORDINGS / "paths.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def recording(name: str) -> Path:
    """The recording kept under ``name``."""
    rel = _paths().get(name)
    return evidence.WORKING / rel if rel else corpora.HELD_RECORDINGS / name
