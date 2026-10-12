# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recordings rest as FLAC and are read as the WAV they were written as.

The claim is byte-exact, header included, and it is what makes the storage
invisible: a hash taken of the WAV before it was stored is still the hash of
what every reader is handed afterwards. The other half is that nothing is lost
on the way. A WAV FLAC cannot hold, a missing encoder, or a round trip that
does not come back identical leaves the WAV where it was.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import wave

import numpy as np
import pytest
from scipy.io import wavfile

from hfhost.audio import WavReplaySource
from hfmodem.core import wav
from hfmodem.shrike import session

pytestmark = pytest.mark.skipif(shutil.which("flac") is None, reason="no flac on PATH")

FS = 48000


@pytest.fixture(autouse=True)
def _flac_looked_up_on_this_path():
    wav._flac_exe.cache_clear()
    yield
    wav._flac_exe.cache_clear()


def _tap(path, seconds=2.0, seed=7):
    """A tap-shaped recording: 48 kHz mono int16, a noise floor and a keyed tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = 0.01 * rng.standard_normal(t.size) + 0.3 * np.sin(2 * np.pi * 1500 * t) * (t > 0.5)
    pcm = np.clip(np.round(x * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return pcm


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_gonewav_decodes_to_its_own_bytes_after_it_is_gone(tmp_path):
    path = tmp_path / "tap.wav"
    _tap(path)
    original = path.read_bytes()

    facts = wav.store(path)

    assert facts == {"wav": "tap.flac", "sha256": _sha(original)}
    assert not path.exists()
    assert not (tmp_path / "tap.flac.part").exists()
    assert (tmp_path / "tap.flac").stat().st_size < len(original)
    assert wav.source(path).getvalue() == original


@pytest.mark.parametrize("name", ["tap.wav", "tap.flac"])
def test_every_reader_sees_the_same_samples_whichever_form_is_on_disk(tmp_path, name):
    path = tmp_path / "tap.wav"
    pcm = _tap(path)
    before = wav.read(path)
    wav.store(path)

    assert np.array_equal(wav.read(tmp_path / name), before)
    assert np.array_equal(session.load_wav(str(tmp_path / name)), before)
    assert np.array_equal(wavfile.read(wav.source(tmp_path / name))[1], pcm)
    with wave.open(wav.source(tmp_path / name)) as w:
        assert np.array_equal(np.frombuffer(w.readframes(w.getnframes()), "<i2"), pcm)


def test_a_wav_on_disk_is_read_in_place(tmp_path):
    path = tmp_path / "tap.wav"
    _tap(path)
    assert wav.source(path) == str(path)
    assert wav.source(tmp_path / "tap.flac") == str(path)


def test_a_float_wav_stays_as_it_is(tmp_path):
    path = tmp_path / "offair.wav"
    wavfile.write(path, FS, np.linspace(-0.5, 0.5, FS, dtype=np.float32))
    original = path.read_bytes()

    facts = wav.store(path)

    assert facts["wav"] == "offair.wav" and facts["sha256"] == _sha(original)
    assert "integer PCM" in facts["kept"]
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_no_flac_keeps_the_wav(tmp_path, monkeypatch):
    path = tmp_path / "tap.wav"
    _tap(path)
    original = path.read_bytes()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    facts = wav.store(path)

    assert "no `flac`" in facts["kept"]
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_a_round_trip_that_does_not_come_back_keeps_the_wav(tmp_path, monkeypatch):
    """A flac whose decode is off by one byte, which only the hash can catch."""
    real = shutil.which("flac")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    fake = bin_ / "flac"
    fake.write_text(f"""#!/bin/sh
"{real}" "$@" || exit $?
case " $* " in *" -d "*)
    while [ "$1" != -o ]; do shift; done
    printf x >> "$2";;
esac
""")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_))
    path = tmp_path / "tap.wav"
    _tap(path)
    original = path.read_bytes()

    facts = wav.store(path)

    assert facts["kept"] == "the FLAC does not decode back to this WAV"
    assert path.read_bytes() == original
    assert not (tmp_path / "tap.flac").exists()
    assert not (tmp_path / "tap.flac.part").exists()


def _record(path, blocks):
    rec = wav.CaptureRecorder(path, rate=FS, source="KT USB")
    for b in blocks:
        rec.push(b)
    rec.drain()
    rec.facts["xruns"] = 0
    rec.close()
    return rec


def test_the_tap_sidecar_names_the_flac_and_the_wav_it_came_from(tmp_path):
    rng = np.random.default_rng(3)
    blocks = [(0.05 * rng.standard_normal(4800)).astype(np.float32) for _ in range(20)]
    path = tmp_path / "tap.wav"

    _record(path, blocks)

    side = json.loads((tmp_path / "tap.json").read_text())
    assert side["wav"] == "tap.flac"
    assert not path.exists()
    assert side["sha256"] == _sha(wav.source(path).getvalue())
    assert side["samples"] == 20 * 4800 and side["xruns"] == 0
    expect = np.clip(np.round(np.concatenate(blocks) * 32768.0), -32768, 32767) / 32768
    assert np.array_equal(wav.read(path), expect)


def test_a_tap_that_cannot_be_gonesays_why_in_its_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    path = tmp_path / "tap.wav"

    _record(path, [np.zeros(4800, np.float32)])

    side = json.loads((tmp_path / "tap.json").read_text())
    assert side["wav"] == "tap.wav" and "no `flac`" in side["kept"]
    assert side["sha256"] == _sha(path.read_bytes())


@pytest.mark.parametrize("name", ["tap.wav", "tap.flac"])
def test_a_gonetap_rehears_exactly_as_it_was_recorded(tmp_path, name):
    path = tmp_path / "tap.wav"
    _tap(path, seconds=3.3)
    heard = list(WavReplaySource(path, realtime=False).frames())
    wav.store(path)

    again = list(WavReplaySource(tmp_path / name, realtime=False).frames())

    assert [t0 for t0, _ in again] == [t0 for t0, _ in heard]
    assert b"".join(p for _, p in again) == b"".join(p for _, p in heard)


def test_rehear_finds_each_recording_once_in_whichever_form_it_rests(tmp_path):
    from hfmodem.tests.kestrel import corpora
    R = corpora.harness("rehear")
    B = corpora.harness("rehear.besra")
    gone = tmp_path / "20260920T010203Z-besra-7102000.wav"
    _tap(gone, seconds=0.1)
    wav.store(gone)
    both = tmp_path / "20260921T010203Z-besra-7102000.wav"
    _tap(both, seconds=0.1)
    shutil.copy(tmp_path / gone.with_suffix(".flac").name, both.with_suffix(".flac"))

    found = R.recordings([tmp_path])

    assert found == [gone.with_suffix(".flac"), both]
    assert all(B.claims(p) for p in found)
    assert R.recordings([gone]) == [gone.with_suffix(".flac")]
