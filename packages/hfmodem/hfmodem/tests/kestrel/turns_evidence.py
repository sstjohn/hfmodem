# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Verified handover and turn-law substitutes; all decoding uses checked_samples.

Times in the session excerpts retain the lost recording's clock for B1. B2's
witnesses use their own clock, resampled at the measured 11999 Hz before slicing.
See fixtures/turns-b-0930/*.json for source offsets and byte integrity.
"""
from pathlib import Path

from hfmodem.tests import evidence

FIXTURES = Path(__file__).with_name("fixtures") / "turns-b-0930"
EXTERNAL = evidence.CAPTURES / "test-evidence" / "turns-b-0930"

UNASKED = (
    ("kb5lzk-0829-141915-session", "KB5LZK", 72.7113, 15, 0xdd833c),
    ("n5tw-0829-143002-session", "N5TW", 110.7300, 15, 0xfec4e1),
    ("kb5lzk-0829-144437-session", "KB5LZK", 98.7020, 12, None),
)
ACKS = (("ke8lva-0823-whole-witness", 52.15),
        ("kb3ac-0823-whole-witness", 64.169))


def samples(name, *, tracked=False):
    """PCM16 amplitudes at 48 kHz, verified on every read."""
    directory = FIXTURES if tracked else EXTERNAL
    manifest = FIXTURES / ("provenance.json" if tracked else "external.json")
    return evidence.checked_samples(directory / (name + ".flac"), manifest)
