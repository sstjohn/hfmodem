# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The surveyed monitor tapes, read through the shared evidence checker."""
from pathlib import Path

from hfmodem.tests import evidence

GROUP = "monitor-e-0930"
MANIFEST = Path(__file__).with_name("fixtures") / GROUP / "external.json"
DIRECTORY = evidence.CAPTURES / "test-evidence" / GROUP
K0SI = tuple(DIRECTORY / f"k0si-{stamp}-whole-e.flac" for stamp in (
    "20260918T000839Z", "20260918T005744Z", "20260918T041501Z",
    "20260918T041811Z", "20260918T042438Z", "20260918T050941Z",
    "20260918T174039Z", "20260918T184226Z", "20260918T233509Z",
    "20260919T214843Z", "20260919T224803Z",
))
OCCUPANT = DIRECTORY / "ke8lva-0903-occupant-e.flac"
HANDSHAKE = DIRECTORY / "ve3kpg-0903-handshake-e.flac"
WITNESS = DIRECTORY / "ve3kpg-0903-witness-e.flac"


def samples(path):
    return evidence.checked_samples(Path(path), MANIFEST)
