# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Integrity-checked external recordings for the A2 greeting tests."""
from __future__ import annotations

from pathlib import Path

from hfmodem.tests import evidence

FIXTURES = Path(__file__).with_name("fixtures") / "greetings-a2-0930"
MANIFEST = FIXTURES / "external.json"
DIRECTORY = evidence.CAPTURES / "test-evidence" / "greetings-a2-0930"


def samples(name: str):
    """Read A2 external samples through the common checked-evidence loader."""
    return evidence.checked_samples(DIRECTORY / name, MANIFEST)
