# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Verified receive substitutes for group D, using the common evidence loader."""
from __future__ import annotations

import json
from pathlib import Path

from hfmodem.tests import evidence

FIXTURES = Path(__file__).with_name("fixtures") / "receive-d-0930"
EXTERNAL = evidence.CAPTURES / "test-evidence" / "receive-d-0930"

MINT = "w6ids-0828-midframe-tap.flac"
PHANTOMS = ("w6ids-0826-1339-phantoms-tap.flac", "w6ids-0826-1524-phantoms-tap.flac")
PROPOSAL = "w6ids-0829-proposal-tap-273s.flac"
REPLAY = "w6ids-0829-replay-tap-273s.flac"
REGULAR = "w6ids-0829-six-requests-tap-192s.flac"
OFFSET = "n0lcr-0815-whole-txwitness.flac"
UNANSWERED = "kc9ghz-20260920T024755Z-whole-unanswered-tap.flac"
ANSWERED = "kc9ghz-20260918T043807Z-whole-answered-tap.flac"
EXTERNAL_NAMES = {PROPOSAL, REPLAY, REGULAR, OFFSET, UNANSWERED, ANSWERED}


def samples(name: str):
    external = name in EXTERNAL_NAMES
    path = (EXTERNAL if external else FIXTURES) / name
    manifest = FIXTURES / ("external.json" if external else "provenance.json")
    return evidence.checked_samples(path, manifest)


def card(name: str):
    return samples(name).astype("float32") / 32768.0


def checked_path(name: str) -> Path:
    """Check integrity before a shipped replay adapter opens the recording."""
    samples(name)
    return (EXTERNAL if name in EXTERNAL_NAMES else FIXTURES) / name


def replay_path(name: str, directory: Path, recording_name: str) -> Path:
    """Give a checked tap the session name the adapter and original log use.

    The alias changes no samples. Its sidecar retains the tap slice's own zero,
    rather than borrowing the lost receiver recording's clock.
    """
    target = checked_path(name)
    path = directory / recording_name
    path.symlink_to(target)
    rows = json.loads((FIXTURES / "external.json").read_text())["files"]
    row = next(r for r in rows if r["file"] == name)
    if "started_utc" in row:
        with path.with_suffix(".json").open("x") as stream:
            json.dump({"started_utc": row["started_utc"]}, stream)
    return path


def log(name: str) -> Path:
    """The small original transcripts are tracked alongside their provenance."""
    path = FIXTURES / name
    manifest = FIXTURES / "logs.json"
    evidence.checked_record(path, manifest)
    return path
