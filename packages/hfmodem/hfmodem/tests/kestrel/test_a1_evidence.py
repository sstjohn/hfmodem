# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Absent external evidence skips; changed evidence must never turn green."""
import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests.kestrel import corpora
from hfmodem.tests import evidence


@pytest.fixture
def external(tmp_path, monkeypatch):
    path = tmp_path / "session.wav"
    samples = np.arange(64, dtype=np.int16)
    wavfile.write(path, 48000, samples)
    record = dict(file=path.name, bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  sample_rate=48000, sample_count=len(samples),
                  channels=1, bits_per_sample=16)
    manifest = tmp_path / "external.json"
    manifest.write_text(json.dumps({"files": [record]}))
    monkeypatch.setattr(corpora, "A1_EXTERNAL", tmp_path)
    monkeypatch.setattr(evidence, "CAPTURES", tmp_path)
    monkeypatch.setattr(corpora, "A1_MANIFEST", manifest)
    return path, manifest, samples


def test_missing_evidence_names_the_file_and_manifest(external):
    path, manifest, _ = external
    path.unlink()
    with pytest.raises(pytest.skip.Exception) as exc:
        corpora.a1_samples(path)
    assert str(path) in str(exc.value) and str(manifest) in str(exc.value)


def test_missing_tracked_clip_fails(external):
    path, manifest, _ = external
    tracked = manifest.with_name("provenance.json")
    tracked.write_text(manifest.read_text())
    path.unlink()
    try:
        with pytest.raises(AssertionError, match="Missing tracked evidence") as exc:
            corpora.evidence.checked_samples(path, tracked)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing tracked evidence must fail, not skip: {skipped}")
    assert str(path) in str(exc.value) and str(tracked) in str(exc.value)


def test_missing_manifest_fails_even_if_audio_is_missing(external, monkeypatch):
    path, manifest, _ = external
    (path.parent / ".git").mkdir()
    (path.parent / "publish").mkdir()
    (path.parent / "publish" / "manifest.toml").write_text("# development source")
    monkeypatch.setattr(evidence, "TREE", path.parent)
    path.unlink()
    manifest.unlink()
    try:
        with pytest.raises(AssertionError, match="Missing evidence manifest"):
            corpora.a1_samples(path)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing evidence manifest must fail, not skip: {skipped}")


@pytest.mark.parametrize("same_size", [False, True], ids=["size", "sha256"])
def test_changed_evidence_fails_before_decoding(external, monkeypatch, same_size):
    from hfmodem.core import wav

    path, manifest, _ = external
    raw = bytearray(path.read_bytes())
    if same_size:
        raw[-1] ^= 1
    else:
        raw.append(0)
    path.write_bytes(raw)

    def must_not_decode(*args):
        pytest.fail("unverified bytes reached the audio decoder")

    monkeypatch.setattr(wav, "source", must_not_decode)
    with pytest.raises(AssertionError, match="SHA-256" if same_size else "size") as exc:
        corpora.a1_samples(path)
    assert str(path) in str(exc.value) and str(manifest) in str(exc.value)


def test_verified_samples_are_rechecked_on_the_next_read(external):
    path, _, samples = external
    np.testing.assert_array_equal(corpora.a1_samples(path), samples)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(AssertionError, match="SHA-256"):
        corpora.a1_samples(path)


def test_wrong_sample_metadata_fails(external):
    path, manifest, _ = external
    data = json.loads(manifest.read_text())
    data["files"][0]["sample_count"] += 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="sample metadata"):
        corpora.a1_samples(path)


@pytest.mark.parametrize("copies", [0, 2], ids=["missing-row", "duplicate-row"])
def test_manifest_requires_one_row_naming_the_file(external, copies):
    path, manifest, _ = external
    data = json.loads(manifest.read_text())
    data["files"] *= copies
    manifest.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="exactly one row") as exc:
        corpora.a1_samples(path)
    assert str(path) in str(exc.value) and str(manifest) in str(exc.value)
