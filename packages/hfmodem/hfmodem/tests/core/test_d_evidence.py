# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The D fixtures obey the shared evidence contract in an ordinary checkout."""
import hashlib
import io
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests import d_evidence as E


@pytest.fixture
def recording(tmp_path, monkeypatch):
    directory = tmp_path / "external"
    directory.mkdir()
    monkeypatch.setattr(E, "FIXTURES", tmp_path)
    monkeypatch.setattr(E, "EXTERNAL", directory)
    stream = io.BytesIO()
    wavfile.write(stream, 48000, np.arange(64, dtype=np.int16))
    data = stream.getvalue()
    for manifest, name in [("external.json", E.PROPOSAL), ("provenance.json", E.MINT)]:
        (tmp_path / manifest).write_text(json.dumps({"files": [{
            "file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "sample_rate": 48000, "sample_count": 64, "channels": 1, "bits_per_sample": 16,
        }]}))
    return tmp_path, directory, data


def test_missing_external_skips_loudly(recording):
    fixtures, directory, _ = recording
    with pytest.raises(pytest.skip.Exception) as caught:
        E.samples(E.PROPOSAL)
    message = str(caught.value)
    assert "MISSING EVIDENCE" in message
    assert str(directory / E.PROPOSAL) in message
    assert str(fixtures / "external.json") in message


def test_missing_tracked_clip_fails(recording):
    try:
        with pytest.raises(AssertionError, match="Missing tracked evidence"):
            E.samples(E.MINT)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing tracked evidence must fail, not skip: {skipped}")


@pytest.mark.parametrize("external", [False, True])
def test_bad_hash_fails_even_when_size_is_right(recording, external):
    fixtures, directory, data = recording
    name = E.PROPOSAL if external else E.MINT
    path = (directory if external else fixtures) / name
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        E.samples(name)


def test_missing_manifest_fails_even_when_external_is_absent(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "publish").mkdir()
    (tmp_path / "publish" / "manifest.toml").write_text("# development source")
    monkeypatch.setattr(E.evidence, "TREE", tmp_path)
    monkeypatch.setattr(E, "FIXTURES", tmp_path)
    monkeypatch.setattr(E, "EXTERNAL", tmp_path)
    try:
        with pytest.raises(AssertionError, match="Missing evidence manifest"):
            E.samples(E.PROPOSAL)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing evidence manifest must fail, not skip: {skipped}")


@pytest.mark.parametrize("damage", ["missing", "sha256"])
def test_tracked_transcript_integrity(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(E, "FIXTURES", tmp_path)
    path = tmp_path / "session.log"
    data = b"received frame\n"
    (tmp_path / "logs.json").write_text(json.dumps({"files": [dict(
        file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())]}))
    if damage == "sha256":
        path.write_bytes(b"Received frame\n")
    message = "Missing tracked evidence" if damage == "missing" else "SHA-256 mismatch"
    try:
        with pytest.raises(AssertionError, match=message):
            E.log(path.name)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"tracked transcript integrity must fail, not skip: {skipped}")


def test_published_tree_can_omit_the_transcript_group(tmp_path, monkeypatch):
    monkeypatch.setattr(E.evidence, "RECORD", tmp_path)
    monkeypatch.setattr(E, "FIXTURES", tmp_path)
    monkeypatch.setattr(E.evidence, "TREE", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="MISSING DISTRIBUTED EVIDENCE"):
        E.log("session.log")
