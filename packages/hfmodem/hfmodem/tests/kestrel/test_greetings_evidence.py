# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Absent external evidence skips; damaged evidence cannot masquerade as absent."""
import hashlib
import io
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests.kestrel import greetings_evidence as E
from hfmodem.tests import evidence


@pytest.fixture
def recording(tmp_path, monkeypatch):
    stream = io.BytesIO()
    wavfile.write(stream, 48000, np.arange(64, dtype=np.int16))
    data = stream.getvalue()
    path = tmp_path / "recording.wav"
    manifest = tmp_path / "external.json"
    manifest.write_text(json.dumps({"files": [{
        "file": path.name, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sample_rate": 48000, "sample_count": 64,
        "channels": 1, "bits_per_sample": 16,
    }]}))
    monkeypatch.setattr(E, "DIRECTORY", tmp_path)
    monkeypatch.setattr(evidence, "CAPTURES", tmp_path)
    monkeypatch.setattr(E, "MANIFEST", manifest)
    return path, manifest, data


def test_missing_evidence_names_the_file_and_manifest(recording):
    path, manifest, _ = recording
    with pytest.raises(pytest.skip.Exception) as caught:
        E.samples(path.name)
    assert str(path) in str(caught.value)
    assert str(manifest) in str(caught.value)
    assert "MISSING EVIDENCE" in str(caught.value)


@pytest.mark.parametrize("damage", ["truncate", "same-size corruption"])
def test_present_corrupt_evidence_fails(recording, damage):
    path, manifest, data = recording
    path.write_bytes(data[:-1] if damage == "truncate" else b"!" + data[1:])
    with pytest.raises(AssertionError) as caught:
        E.samples(path.name)
    assert str(path) in str(caught.value)
    assert str(manifest) in str(caught.value)
    assert ("size mismatch" if damage == "truncate" else "SHA-256 mismatch") in str(caught.value)


def test_matching_evidence_is_available(recording):
    path, _, data = recording
    path.write_bytes(data)
    np.testing.assert_array_equal(E.samples(path.name), np.arange(64))


@pytest.mark.parametrize("copies", [0, 2], ids=["missing-row", "duplicate-row"])
def test_manifest_requires_one_row_naming_the_file(recording, copies):
    path, manifest, data = recording
    path.write_bytes(data)
    rows = json.loads(manifest.read_text())["files"] * copies
    manifest.write_text(json.dumps({"files": rows}))
    with pytest.raises(AssertionError, match="exactly one row") as caught:
        E.samples(path.name)
    assert str(path) in str(caught.value) and str(manifest) in str(caught.value)
