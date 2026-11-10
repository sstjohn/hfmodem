# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The shared loader distinguishes absent external evidence from broken checkouts."""
import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests import evidence


@pytest.fixture
def clip(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    (checkout / "publish").mkdir()
    (checkout / "publish" / "manifest.toml").write_text("# development source")
    monkeypatch.setattr(evidence, "TREE", checkout)
    path = tmp_path / "positive.wav"
    wavfile.write(path, 48000, np.arange(64, dtype=np.int16))
    manifest = tmp_path / "provenance.json"
    manifest.write_text(json.dumps({"files": [dict(
        file=path.name, bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sample_rate=48000, sample_count=64, channels=1, bits_per_sample=16)]}))
    return path, manifest


def test_missing_tracked_clip_fails(clip):
    path, manifest = clip
    path.unlink()
    try:
        with pytest.raises(AssertionError, match="Missing tracked evidence") as caught:
            evidence.checked_samples(path, manifest)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing tracked evidence must fail, not skip: {skipped}")
    assert str(path) in str(caught.value) and str(manifest) in str(caught.value)


def test_missing_external_clip_skips(clip):
    path, manifest = clip
    path.unlink()
    external_manifest = manifest.with_name("external.json")
    external_manifest.write_text(manifest.read_text())
    manifest = external_manifest
    with pytest.raises(pytest.skip.Exception, match="MISSING EVIDENCE") as caught:
        evidence.checked_samples(path, manifest)
    assert str(path) in str(caught.value) and str(manifest) in str(caught.value)


def test_absent_manifest_fails_even_when_external_audio_is_absent(clip):
    path, manifest = clip
    path.unlink()
    manifest.unlink()
    try:
        with pytest.raises(AssertionError, match="Missing evidence manifest"):
            evidence.checked_samples(path, manifest.with_name("external.json"))
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing evidence manifest must fail, not skip: {skipped}")


def test_published_tree_reports_an_entirely_omitted_evidence_group(clip, monkeypatch):
    path, manifest = clip
    path.unlink()
    manifest.unlink()
    monkeypatch.setattr(evidence, "TREE", path.parent)
    monkeypatch.setattr(evidence, "RECORD", path.parent)
    with pytest.raises(pytest.skip.Exception, match="MISSING DISTRIBUTED EVIDENCE"):
        evidence.checked_samples(path, manifest)


def test_same_size_corruption_fails(clip):
    path, manifest = clip
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        evidence.checked_samples(path, manifest)
