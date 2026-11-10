# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import hashlib
import json

import pytest

from hfmodem.tests import evidence


def test_public_operator_tools_do_not_require_private_recording_counts(tmp_path):
    for name in ("tools", ".git", "working"):
        (tmp_path / name).mkdir()
    assert evidence._sweep_floor(tmp_path) == (250, 10)
    (tmp_path / "publish").mkdir()
    (tmp_path / "publish" / "manifest.toml").write_text("# development source")
    assert evidence._sweep_floor(tmp_path) == (500, 20)


@pytest.mark.parametrize("private_root", [None, "TREE", "RECORD"])
def test_absent_evidence_group_uses_source_contract_not_git(tmp_path, monkeypatch, private_root):
    for name in ("tools", ".git", "working"):
        (tmp_path / name).mkdir()
    origin = tmp_path / "origin"
    origin.mkdir()
    monkeypatch.setattr(evidence, "TREE", tmp_path)
    monkeypatch.setattr(evidence, "RECORD", origin)
    if private_root:
        marker = getattr(evidence, private_root) / "publish" / "manifest.toml"
        marker.parent.mkdir()
        marker.write_text("# development source")
    expected_floor = (500, 20) if private_root else (250, 10)
    assert evidence._sweep_floor(tmp_path, origin) == expected_floor
    outcome = AssertionError if private_root else pytest.skip.Exception
    message = "Missing evidence manifest" if private_root else "MISSING DISTRIBUTED EVIDENCE"
    with pytest.raises(outcome, match=message):
        evidence.checked_record(tmp_path / "clip.flac", tmp_path / "provenance.json")


@pytest.mark.parametrize("damage", ["orphan", "missing", "hash", "ambiguous"])
def test_present_public_evidence_keeps_integrity_failures(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(evidence, "TREE", tmp_path)
    monkeypatch.setattr(evidence, "RECORD", tmp_path)
    path = tmp_path / "clip.flac"
    data = b"pinned recording"
    row = dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest = tmp_path / "provenance.json"
    if damage != "orphan":
        manifest.write_text(json.dumps({"files": [row] * (2 if damage == "ambiguous" else 1)}))
    if damage != "missing":
        path.write_bytes(data if damage != "hash" else b"Pinned recording")
    message = {"orphan": "Missing evidence manifest", "missing": "Missing tracked evidence",
               "hash": "SHA-256 mismatch", "ambiguous": "exactly one row"}[damage]
    with pytest.raises(AssertionError, match=message):
        evidence.checked_record(path, manifest)
