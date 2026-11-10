# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Storage failures must not hide behind a recording test's skip or xfail."""
import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests import evidence
from hfmodem.tests import monitor_evidence as E


@pytest.mark.parametrize("path", (*E.K0SI, E.OCCUPANT, E.HANDSHAKE, E.WITNESS),
                         ids=lambda path: path.stem)
def test_monitor_evidence_matches_its_manifest(path):
    E.samples(path)


@pytest.mark.parametrize("external", [True, False], ids=["external", "tracked"])
def test_missing_audio_obeys_its_storage_contract(tmp_path, external):
    path = tmp_path / "absent.flac"
    manifest = tmp_path / ("external.json" if external else "provenance.json")
    manifest.write_text(json.dumps({"files": [{"file": path.name}]}))
    outcome = pytest.skip.Exception if external else AssertionError
    with pytest.raises((AssertionError, pytest.skip.Exception)) as caught:
        evidence.checked_samples(path, manifest)
    assert isinstance(caught.value, outcome), str(caught.value)
    assert str(path) in str(caught.value) and str(manifest) in str(caught.value)
    assert ("MISSING EVIDENCE" if external else "Missing tracked evidence") in str(caught.value)


def test_a_missing_manifest_is_not_missing_external_audio(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "publish").mkdir()
    (tmp_path / "publish" / "manifest.toml").write_text("# development source")
    monkeypatch.setattr(evidence, "TREE", tmp_path)
    try:
        with pytest.raises(AssertionError, match="Missing evidence manifest"):
            evidence.checked_samples(tmp_path / "absent.flac", tmp_path / "external.json")
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing evidence manifest must fail, not skip: {skipped}")


def test_same_size_corruption_fails_before_the_decoder(tmp_path, monkeypatch):
    from hfmodem.core import wav

    path = tmp_path / "damaged.wav"
    wavfile.write(path, 48000, np.arange(64, dtype=np.int16))
    data = path.read_bytes()
    manifest = tmp_path / "external.json"
    manifest.write_text(json.dumps({"files": [dict(
        file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
        sample_rate=48000, sample_count=64, channels=1, bits_per_sample=16)]}))
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))

    def must_not_decode(*args):
        pytest.fail("corrupt evidence reached the decoder")

    monkeypatch.setattr(wav, "source", must_not_decode)
    with pytest.raises(AssertionError, match="SHA-256"):
        evidence.checked_samples(path, manifest)
