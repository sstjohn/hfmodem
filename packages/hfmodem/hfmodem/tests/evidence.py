# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where a run's evidence is, which in a worktree is not where its code is.

Every recording this suite is measured against lives outside the package — under
``working/``, ``logs/`` and ``captures/``, or in the shared corpus beside the
checkout. Of those four only ``working/`` is in the index: ``.gitignore`` names
``logs/`` and ``captures/``, and the corpus is not in the checkout at all. A
worktree is made from the tree, not from the untracked files in it, so it gets
the code and the working record and none of the audio, and every declaration
that resolved the other three against the worktree's own root pointed at
nothing.

That is silent by construction. The tests reading those paths carry a `skipif`,
which is the right answer for a run that has no way to get the material and the
wrong one for a worktree of the very checkout holding it: an agent reading
`91 passed, 3 skipped` cannot tell that the three were the WW2MI tests covering
the stream it had been sent to investigate.

So the evidence resolves here, once, against the checkout a worktree was made
from. Code and evidence part company deliberately: `conftest.py` puts this tree's
packages first on `sys.path` precisely so a worktree tests its OWN code, while
the recordings are not a thing a worktree has its own version of.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The checkout this run's code is in.
TREE = Path(__file__).resolve().parents[4]


def _origin() -> Path | None:
    """The checkout ``TREE`` was made a worktree of, or None if it is not one.

    A worktree's ``.git`` is a file naming its own git dir, and ``commondir``
    beside that names the ``.git`` the two share. A plain checkout has a
    directory there and an installed wheel has neither, and both answer for
    themselves.
    """
    dotgit = TREE / ".git"
    if not dotgit.is_file():
        return None
    gitdir = Path(dotgit.read_text().partition("gitdir:")[2].strip())
    common = gitdir / (gitdir / "commondir").read_text().strip()
    return common.resolve().parent


#: The checkout its evidence is in.
RECORD = _origin() or TREE

WORKING = RECORD / "working"
LOGS = RECORD / "logs"
CAPTURES = RECORD / "captures"

#: The shared off-air corpus, which is kept beside a checkout rather than in one.
#: One name across the tree; the per-modem names still work, because scripts and
#: notes use them.
CORPUS = Path(os.environ.get("HFMODEM_CORPUS")
              or os.environ.get("KESTREL_RF_CORPUS")
              or RECORD.parent / "rf-corpus")


def private_source_tree(*roots: Path) -> bool:
    """Whether code or evidence comes from a tree promising private records.

    The publication manifest belongs to development checkouts and never ships.
    Git, operator tools and incidental working directories can exist publicly.
    """
    return any((root / "publish" / "manifest.toml").is_file() for root in roots)


def _sweep_floor(tree: Path, record: Path | None = None) -> tuple[int, int]:
    roots = (tree,) if record is None else (tree, record)
    return (500, 20) if private_source_tree(*roots) else (250, 10)


#: Fewest brackets tried and words found before a sweep over every fixture
#: recording counts, for the private checkout or the public recording subset.
SWEEP_FLOOR = _sweep_floor(TREE, RECORD)


def require_input(path: Path | str, *, allow_module_level: bool = False) -> Path:
    """Require an optional recording or its replay support before using it.

    Call only for inputs that do not ship, never for tracked evidence that the
    distribution promises to include (use ``checked_record`` for that). This
    checks absence only: decode errors, damaged metadata and failed assertions
    with an input present remain failures. Module-scope callers must opt into
    skipping collection with ``allow_module_level=True``.
    """
    import pytest

    path = Path(path)
    if path.suffix in {".wav", ".flac"}:
        from hfmodem.core.wav import stored

        path = stored(path)
    try:
        path.stat()
    except FileNotFoundError as exc:
        if exc.filename != os.fspath(path):
            raise
        pytest.skip(f"unavailable recording input: {path}",
                    allow_module_level=allow_module_level)
    return path


def manifest_rows(manifest: Path):
    """Read a group manifest or an existing single-recording sidecar."""
    import json

    data = json.loads(manifest.read_text())
    return data["files"] if "files" in data else [data]


def checked_record(path: Path, manifest: Path):
    """Verify one audio or transcript file and return its manifest row.

    Only rows in ``external.json`` may skip missing files. Missing tracked
    files, manifests and ambiguous rows fail. A published tree may omit a whole
    group (both manifest and file).
    """
    import hashlib

    import pytest

    context = f"{path}; manifest: {manifest}"
    if not manifest.is_file():
        if not private_source_tree(TREE, RECORD) and not path.exists():
            pytest.skip(f"MISSING DISTRIBUTED EVIDENCE: {context}")
        raise AssertionError(f"Missing evidence manifest: {context}")
    rows = [r for r in manifest_rows(manifest) if r["file"] == path.name]
    assert len(rows) == 1, (
        f"Evidence manifest needs exactly one row for {context}; found {len(rows)}")
    row, = rows
    # Check before open: an optional path may traverse a self-pointing logs link.
    if not path.is_file():
        if manifest.name == "external.json":
            pytest.skip(f"MISSING EVIDENCE: {context}")
        raise AssertionError(f"Missing tracked evidence: {context}")
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if "rate" not in row or "bytes" in row:
            assert size == row["bytes"], (
                f"Evidence size mismatch: {context}; expected {row['bytes']}, got {size}")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == row["sha256"], (
        f"Evidence SHA-256 mismatch: {context}; expected {row['sha256']}, got {digest}")
    return row


def checked_samples(path: Path, manifest: Path):
    """Read checked PCM16 mono evidence in PCM units through the shared decoder.

    Published single-file sidecars use ``rate`` and ``samples`` instead of the
    group schema's sample metadata; both formats pin the complete file hash.
    """
    import numpy as np
    from scipy.io import wavfile

    from hfmodem.core.wav import source

    row = checked_record(path, manifest)
    context = f"{path}; manifest: {manifest}"
    sidecar = "rate" in row and "samples" in row
    sr, samples = wavfile.read(source(path))
    assert samples.ndim == 1 and samples.dtype == np.int16, (
        f"Evidence must be PCM16 mono: {context}")
    assert (sr, len(samples), 1, 16) == (
        row["rate"] if sidecar else row["sample_rate"],
        row["samples"] if sidecar else row["sample_count"],
        1 if sidecar else row["channels"], 16 if sidecar else row["bits_per_sample"]
    ), f"Evidence sample metadata mismatch: {context}"
    return samples.astype(float)
