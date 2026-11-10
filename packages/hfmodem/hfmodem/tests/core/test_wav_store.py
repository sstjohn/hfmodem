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
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfhost.audio import WavReplaySource
from hfmodem.core import wav
from hfmodem.shrike import session
from hfmodem.tests import hermetic

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


def test_a_wav_on_disk_is_read_in_place(tmp_path, monkeypatch):
    path = tmp_path / "tap.wav"
    _tap(path)
    for name in ("tap.wav", "tap.flac"):
        src = wav.source(tmp_path / name)
        assert not isinstance(src, io.BytesIO) and src.name == str(path)
        src.close()
    calls = []
    read = wavfile.read
    monkeypatch.setattr(wavfile, "read", lambda f, **kw: calls.append(kw) or read(f, **kw))
    wav.read(path)
    assert calls == [{"mmap": True}]


def test_rifx_recording_samples_are_preserved(tmp_path):
    path = tmp_path / 'big-endian.wav'
    pcm = np.array([-32768, -12345, 0, 12345, 32767], dtype='>i2')
    data = pcm.tobytes()
    path.write_bytes(
        b'RIFX' + struct.pack('>I', 36 + len(data)) + b'WAVEfmt '
        + struct.pack('>IHHIIHH', 16, 1, 1, FS, FS * 2, 2, 16)
        + b'data' + struct.pack('>I', len(data)) + data)
    assert np.array_equal(wav.read(path), pcm.astype(np.float64) / 32768)


def test_rf64_recording_samples_are_preserved(tmp_path):
    path = tmp_path / 'rf64.wav'
    pcm = np.array([-32768, -12345, 0, 12345, 32767], dtype='<i2')
    data = pcm.tobytes()
    path.write_bytes(
        b'RF64\xff\xff\xff\xffWAVEds64'
        + struct.pack('<IQQQI', 28, 72 + len(data), len(data), len(pcm), 0)
        + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, FS, FS * 2, 2, 16)
        + b'data\xff\xff\xff\xff' + data)
    assert np.array_equal(wav.read(path), pcm.astype(np.float64) / 32768)


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

    flac = wav._flac
    def admitted(*args):
        argv = [str(fake), '-s', '-f', *map(str, args)]
        hermetic.guard.admit(argv, os.environ)
        return flac(*args)
    monkeypatch.setattr(wav, '_flac', admitted)

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
    [segment] = wav.segments(path)
    assert side["wav"] == segment.name and segment.suffix == '.flac'
    assert not path.exists()
    assert side["sha256"] == _sha(wav.source(segment).getvalue())
    assert side["samples"] == 20 * 4800 and side["xruns"] == 0
    expect = np.clip(np.round(np.concatenate(blocks) * 32768.0), -32768, 32767) / 32768
    assert np.array_equal(wav.read(path), expect)


def test_a_tap_that_cannot_be_gonesays_why_in_its_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    path = tmp_path / "tap.wav"

    _record(path, [np.zeros(4800, np.float32)])

    side = json.loads((tmp_path / "tap.json").read_text())
    [segment] = wav.segments(path)
    assert side["wav"] == segment.name and segment.suffix == '.wav' and "no `flac`" in side["kept"]
    assert side["sha256"] == _sha(segment.read_bytes())


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


def test_flac_spawn_closes_handles_only_on_windows(monkeypatch):
    from types import SimpleNamespace
    import os
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)
    with monkeypatch.context() as patch:
        patch.setattr(wav, "_flac_exe", lambda: "/absolute/flac")
        patch.setattr(wav.subprocess, "run", run)
        wav._flac("--version")
    assert calls[0][0] == ["/absolute/flac", "-s", "-f", "--version"]
    assert calls[0][1]["close_fds"] is (os.name == "nt")



def test_a_long_capture_is_stored_one_segment_at_a_time(tmp_path, monkeypatch):
    minute = 60 * FS
    monkeypatch.setattr(wav, "SEGMENT_SAMPLES", 60 * 48000)
    rng = np.random.default_rng(5)
    blocks = [(0.05 * rng.standard_normal(minute // 2)).astype(np.float32) for _ in range(5)]
    path = tmp_path / "stream.wav"
    rec = wav.CaptureRecorder(path, rate=FS, source="KT USB")
    for b in blocks[:3]:
        rec.push(b)
    rec.drain()
    rec.drain()
    for b in blocks[3:]:
        rec.push(b)
    rec.drain()
    rec.close()

    segments = wav.segments(path)
    assert [p.suffix for p in segments] == [".flac"] * 3
    assert list(tmp_path.glob("*.wav")) == []
    side = json.loads((tmp_path / "stream.json").read_text())
    assert side["samples"] == 5 * minute // 2 and side["segment_samples"] == minute
    assert side["segments"] == [p.name for p in segments]
    expect = np.clip(np.round(np.concatenate(blocks) * 32768.0), -32768, 32767) / 32768
    assert np.array_equal(wav.read(path), expect)


def test_a_finished_capture_segment_is_compressed_before_close(tmp_path, monkeypatch):
    minute = 60 * FS
    monkeypatch.setattr(wav, "SEGMENT_SAMPLES", 60 * 48000)
    rec = wav.CaptureRecorder(tmp_path / "stream.wav", rate=FS, source="KT USB")
    try:
        rec.push(np.zeros(minute + 4800, np.float32))
        rec.drain()
        deadline = time.monotonic() + 30
        while (wav.segments(tmp_path / "stream.wav")[0].suffix != ".flac"
               and time.monotonic() < deadline):
            time.sleep(0.05)
        assert [p.suffix for p in wav.segments(tmp_path / "stream.wav")] == [".flac", ".wav"]
    finally:
        rec.close()


def test_a_detached_store_compresses_a_finished_segment(tmp_path):
    path = tmp_path / "receive.wav"
    wav.write(path, np.zeros(4800), FS)
    wav.store_detached(path, tmp_path / "receive.store.log")
    deadline = time.monotonic() + 20
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (tmp_path / "receive.flac").is_file() and not path.exists()


def test_rehear_folds_segments_into_one_logical_recording(tmp_path, monkeypatch):
    from hfmodem.tests.kestrel import corpora
    R = corpora.harness('rehear')
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    path = tmp_path / '20260920T010203Z-besra-7102000.wav'
    recorder = wav.CaptureRecorder(path, rate=8)
    recorder.push(np.zeros(1000, np.float32))
    recorder.close()
    parts = wav.segments(path)
    assert len(parts) == 3
    assert R.recordings([tmp_path]) == [path]
    assert R.recordings([path]) == [path]
    assert R.recordings([parts[1]]) == [path]
    assert R.recordings([tmp_path, path]) == [path]


def test_a_harness_name_found_only_outside_the_harness_skips(tmp_path, monkeypatch):
    """The distribution does not ship the `tools/rehear/` package these tests
    drive; a same-named module elsewhere on the path must not stand in for it."""
    from hfmodem.tests.kestrel import corpora
    shipped = tmp_path / 'logbook_view'
    shipped.mkdir()
    (shipped / 'rehear.py').write_text('')
    monkeypatch.syspath_prepend(str(shipped))
    monkeypatch.delitem(sys.modules, 'rehear', raising=False)
    monkeypatch.setattr(corpora, 'TOOLS', tmp_path / 'tools')
    monkeypatch.setattr(corpora, 'ROOT', tmp_path / 'corpus')
    with pytest.raises(pytest.skip.Exception):
        corpora.harness('rehear')


def test_writer_splits_exact_boundaries_and_uses_its_constructed_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav.time, 'time', lambda: 0)
    handed = []
    writer = wav.SegmentedWriter(tmp_path / 'stream.wav', rate=8, closed=handed.append)
    first = writer._wav
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 120 * 48000)
    writer.writeframes(bytes(2 * 480))
    assert writer.samples == 480 and writer._wav is None
    assert first._file is None and len(handed) == 1
    writer.writeframes(bytes(2 * 481))
    writer.close()
    writer.close()
    assert [p.name for p in handed] == ['stream-19700101T0000Z.wav',
        'stream-19700101T0001Z.wav', 'stream-19700101T0002Z.wav']
    counts = []
    for part in handed:
        with wave.open(str(part)) as src:
            counts.append(src.getnframes())
    assert counts == [480, 480, 1] and writer.samples == 961


def test_streaming_replay_reads_all_segments_on_one_timebase(tmp_path, monkeypatch):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * FS)
    path = tmp_path / 'stream.wav'
    out = wav.SegmentedWriter(path, rate=FS, closed=lambda _: None)
    out.writeframes(bytes(2 * (60 * FS + 3)))
    out.close()
    parts = wav.segments(path)
    wav.store(parts[0])
    frames = list(WavReplaySource(path, realtime=False).frames())
    assert sum(len(pcm) // 2 for _, pcm in frames) == 60 * FS + 3
    assert frames[-1][0] + len(frames[-1][1]) // 2 == 60 * FS + 3


def test_capture_drain_does_not_wait_for_storage_and_close_waits_for_every_store(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    stored_paths = []
    def hold(path, **kwargs):
        started.set()
        assert release.wait(3)
        stored_paths.append(path)
        return {'wav': path.name}
    monkeypatch.setattr(wav, 'store', hold)
    path = tmp_path / 'stream.wav'
    recorder = wav.CaptureRecorder(path, rate=8)
    recorder.push(np.zeros(481, np.float32))
    recorder.drain()
    assert started.wait(3)
    def close():
        try:
            recorder.close()
        finally:
            closed.set()
    thread = threading.Thread(target=close)
    thread.start()
    try:
        assert not closed.wait(.05)
    finally:
        release.set()
        thread.join(3)
    assert closed.is_set()
    assert stored_paths == wav.segments(path)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['segments'] == [p.name for p in stored_paths] and side['samples'] == 481


@pytest.mark.parametrize('damage', ['missing', 'short'])
@pytest.mark.parametrize('reader', ['segments', 'read', 'replay'])
def test_readers_refuse_to_close_a_segment_gap(tmp_path, monkeypatch, damage, reader):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    path = tmp_path / 'stream.wav'
    writer = wav.SegmentedWriter(path, rate=8, closed=lambda _: None)
    writer.writeframes(bytes(2 * 961))
    writer.close()
    middle = writer.paths[1]
    if damage == 'missing':
        middle.unlink()
    else:
        with wave.open(str(middle), 'wb') as out:
            out.setparams((1, 2, 8, 0, 'NONE', 'not compressed'))
            out.writeframes(bytes(2))
    with pytest.raises(ValueError, match='not yet completely written'):
        if reader == 'segments':
            wav.segments(path)
        elif reader == 'read':
            wav.read(path)
        else:
            list(WavReplaySource(path, realtime=False).frames())


@pytest.mark.parametrize('occupied', ['tap.wav', 'tap.flac', 'tap-20010101T0000Z.wav',
                                      'tap-20010101T0100Z.flac'])
@pytest.mark.parametrize('owned', [False, True])
def test_occupied_recording_family_is_refused(tmp_path, monkeypatch, occupied, owned):
    from hfmodem.core import capture_identity as ci
    if owned:
        capture = ci.allocate({'capture': tmp_path / 'capture'}, {'modem': 'capture'})
        writer = capture.claim('modem')
        root = writer.root
    else:
        root = tmp_path
    foreign = root / occupied
    foreign.write_bytes(b'foreign')
    with pytest.raises(FileExistsError):
        owner = ci.Recording.reserve(writer, 'tap.wav', clock_id='card', continuous=True) if owned else None
        wav.CaptureRecorder(root / 'tap.wav', rate=8, ownership=owner)
    assert foreign.read_bytes() == b'foreign'


@pytest.mark.parametrize('reader', ['read', 'cut', 'frames', 'replay'])
def test_reader_retries_a_wav_replaced_between_resolution_and_open(tmp_path, monkeypatch, reader):
    from hfmodem.station.contactlog import cut_audio, _audio_frames
    path = tmp_path / 'tap.wav'
    samples = _tap(path, seconds=.01)
    real_open = Path.open
    replaced = False
    def opening(candidate, *args, **kwargs):
        nonlocal replaced
        if candidate == path and not replaced:
            replaced = True
            wav.store(path)
        return real_open(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opening)
    if reader == 'read':
        assert np.array_equal(wav.read(path), samples / 32768)
    elif reader == 'frames':
        assert _audio_frames(path) == (FS, len(samples))
    elif reader == 'cut':
        run = subprocess.run
        def admitted(argv, **kwargs):
            if argv[0] == 'flac':
                hermetic.guard.admit(argv, kwargs.get('env', os.environ))
            return run(argv, **kwargs)
        monkeypatch.setattr(subprocess, 'run', admitted)
        target = tmp_path / 'clip.wav'
        cut_audio(path, target, 0, len(samples))
        assert np.array_equal(wav.read(target), samples / 32768)
    else:
        assert b''.join(raw for _, raw in WavReplaySource(path, realtime=False).frames()) == samples.tobytes()
    assert replaced


def test_failed_segment_creation_resumes_at_its_true_next_index(tmp_path, monkeypatch, capsys):
    from hfmodem.station.contactlog import cut_audio
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav.time, 'time', lambda: 0)
    path = tmp_path / 'receive.wav'
    handed = []
    writer = wav.SegmentedWriter(path, rate=8, closed=handed.append)
    opening = Path.open
    attempts = []
    def fail_middle(candidate, *args, **kwargs):
        if candidate.name == 'receive-19700101T0001Z.wav' and args == ('xb',):
            attempts.append(candidate)
            raise OSError('synthetic segment creation failure')
        return opening(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_middle)
    with pytest.raises(OSError):
        writer.writeframes(np.arange(600, dtype='<i2').tobytes())
    writer.writeframes(np.arange(600, 963, dtype='<i2').tobytes())
    writer.close()
    assert writer.samples == 963 and len(attempts) == 1
    assert [p.name for p in handed] == ['receive-19700101T0000Z.wav', 'receive-19700101T0002Z.wav']
    assert capsys.readouterr().err.count('synthetic segment creation failure') == 1
    with pytest.raises(ValueError, match='missing segment'):
        wav.segments(path)
    target = tmp_path / 'clip.wav'
    with pytest.raises(ValueError, match='not yet completely written'):
        cut_audio(path, target, 480, 483)
    cut_audio(path, target, 960, 963)
    assert np.array_equal(wav.read(target), np.arange(960, 963) / 32768)


@pytest.mark.parametrize('failure', ['levels', 'pcm'])
def test_capture_conversion_failure_abandons_the_segment(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav, 'store', lambda path, **kwargs: {'wav': path.name})
    recorder = wav.CaptureRecorder(tmp_path / 'stream.wav', rate=8)
    def fail(*args, **kwargs):
        raise MemoryError('synthetic conversion failure')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'abs' if failure == 'levels' else 'clip', fail)
        recorder.push(np.zeros(2, np.float32))
        with pytest.raises(MemoryError, match='synthetic conversion failure'):
            recorder.drain()
    recorder.push(np.ones(481, np.float32) / 2)
    recorder.drain()
    with pytest.raises(MemoryError, match='synthetic conversion failure'):
        recorder.close()
    assert recorder._out.samples == 483
    assert recorder._n == 483
    first, second = recorder._out.paths
    with wave.open(str(first)) as audio:
        assert audio.getnframes() == 0
    with wave.open(str(second)) as audio:
        assert audio.getnframes() == 3
        assert np.frombuffer(audio.readframes(3), '<i2').tolist() == [16384] * 3


@pytest.mark.parametrize('failure', [None, 'write', 'replace'])
def test_capture_sidecar_publication_keeps_the_previous_valid_recording(tmp_path, monkeypatch, failure):
    import io
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav, 'store', lambda path, **kwargs: {'wav': path.name})
    path = tmp_path / 'stream.wav'
    recorder = wav.CaptureRecorder(path, rate=8)
    recorder.push(np.zeros(481, np.float32))
    recorder.drain()
    side = path.with_suffix('.json')
    before = side.read_bytes()
    opening, replace = io.open, Path.replace
    observed = []

    class DuringWrite:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, text):
            if not observed:
                self.stream.write(text[:1])
                self.stream.flush()
                assert side.read_bytes() == before
                assert wav.segments(path) == recorder._out.paths
                observed.append('write')
                if failure == 'write':
                    raise OSError('synthetic sidecar write failure')
                return 1 + self.stream.write(text[1:])
            return self.stream.write(text)

    def during_open(*args, **kwargs):
        stream = opening(*args, **kwargs)
        return DuringWrite(stream) if stream.writable() else stream

    def during_replace(temporary, target):
        assert target == side and temporary.parent == side.parent
        assert side.read_bytes() == before
        assert json.loads(temporary.read_text())['samples'] == 481
        observed.append('replace')
        if failure == 'replace':
            raise OSError('synthetic sidecar replace failure')
        return replace(temporary, target)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(io, 'open', during_open)
            patch.setattr(Path, 'replace', during_replace)
            if failure:
                with pytest.raises(OSError, match=f'synthetic sidecar {failure} failure'):
                    recorder._write_sidecar(False)
                assert side.read_bytes() == before
            else:
                recorder._write_sidecar(False)
                assert json.loads(side.read_text())['samples'] == 481
        assert observed == (['write'] if failure == 'write' else ['write', 'replace'])
        assert not list(side.parent.glob(f'.{side.name}.*.partial'))
    finally:
        recorder.close()


@pytest.mark.parametrize('owned', [False, True])
def test_fractional_start_at_44100_reconstructs_the_same_segment_names(tmp_path, monkeypatch, owned):
    from hfmodem.core import capture_identity as ci
    from hfhost.recordings import indexed
    capture = ci.allocate({'capture': tmp_path / 'capture'}, {'modem': 'capture'}) if owned else None
    calls = []
    def now():
        calls.append(None)
        return 41.8
    monkeypatch.setattr(wav.time, 'time', now)
    monkeypatch.setattr(wav, 'store', lambda path, **kwargs: {'wav': path.name})
    owner = ci.Recording.reserve(capture.claim('modem'), 'stream.wav', clock_id='card',
                                 continuous=True) if owned else None
    path = owner.base if owned else tmp_path / 'stream.wav'
    recorder = wav.CaptureRecorder(path, rate=44100, ownership=owner)
    out = recorder._out
    out.queue_through(out.segment_samples + 1)
    out.samples = out.segment_samples
    out.writeframes(b'\0\0')
    recorder.close()
    assert calls == [None]
    assert out.started == 41
    if owned:
        assert owner.started == out.started
    assert json.loads(path.with_suffix('.json').read_text())['started_utc'] == '1970-01-01T00:00:41Z'
    assert [part.name for part in out.paths] == ['stream-19700101T0000Z.wav', 'stream-19700101T0100Z.wav']
    parts, capacity = indexed(path)
    assert parts == dict(enumerate(out.paths))
    assert capacity == out.segment_samples


def test_capture_close_waits_for_rollover_and_final_storage(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    entered, release, closing, closed = (threading.Event() for _ in range(4))
    stored = []
    def store(path, **kwargs):
        stored.append(path)
        return {'wav': path.name}
    monkeypatch.setattr(wav, 'store', store)
    recorder = wav.CaptureRecorder(tmp_path / 'stream.wav', rate=8)
    handoff = recorder._out._closed_segment
    def rollover(path):
        entered.set()
        assert release.wait(3)
        handoff(path)
    recorder._out._closed_segment = rollover
    recorder.push(np.zeros(481, np.float32))
    errors = []
    def run(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)
    drain = threading.Thread(target=lambda: run(recorder.drain))
    def close():
        closing.set()
        recorder.close()
        closed.set()
    shutdown = threading.Thread(target=lambda: run(close))
    drain.start()
    try:
        assert entered.wait(3)
        shutdown.start()
        assert closing.wait(3)
        assert not closed.wait(.05)
    finally:
        release.set()
        drain.join(3)
        shutdown.join(3)
    assert not errors
    assert closed.is_set() and not drain.is_alive() and not shutdown.is_alive()
    assert recorder._out._wav is None
    assert stored == recorder._out.paths and len(stored) == 2
    recorder.push(np.zeros(480, np.float32))
    recorder.drain()
    assert recorder._out.samples == 481 and len(recorder._out.paths) == 2
    assert json.loads(recorder.path.with_suffix('.json').read_text())['samples'] == 481


def test_timestamp_shaped_legacy_recordings_need_positive_family_evidence(tmp_path):
    from hfmodem.tests.kestrel import corpora
    from hfhost.recordings import base, segments
    rehear = corpora.harness('rehear')
    paths = [tmp_path / f'tap-20261030T{hour}00Z.wav' for hour in ('13', '14')]
    for path in paths:
        _tap(path, seconds=.01)
    # Old per-file metadata is not evidence of a segmented family.
    (tmp_path / 'tap.json').write_text(json.dumps({'samplerate': FS, 'started_utc': '2026-10-30T13:00:00Z'}))
    assert rehear.recordings([tmp_path]) == paths
    assert all(base(path) == path and segments(path) == [path] for path in paths)
    with pytest.raises(FileNotFoundError):
        wav.read(tmp_path / 'tap.wav')


def test_sidecar_does_not_claim_an_unlisted_timestamp_neighbor(tmp_path, monkeypatch):
    from hfhost.recordings import base
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav.time, 'time', lambda: 0)
    logical = tmp_path / 'tap.wav'
    writer = wav.SegmentedWriter(logical, rate=8, closed=lambda _: None)
    writer.writeframes(bytes(960))
    writer.close()
    neighbor = tmp_path / 'tap-19700101T0100Z.wav'
    _tap(neighbor, seconds=.01)
    assert base(neighbor) == neighbor
    assert wav.segments(logical) == writer.paths


def test_writer_publishes_family_evidence_at_construction_and_rollover(tmp_path, monkeypatch):
    from hfhost.recordings import base
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    monkeypatch.setattr(wav.time, 'time', lambda: 0)
    path = tmp_path / 'receive.wav'
    writer = wav.SegmentedWriter(path, rate=8, closed=lambda _: None)
    side = path.with_suffix('.json')
    initial = json.loads(side.read_text())
    assert initial['started_utc'] == '1970-01-01T00:00:00Z'
    assert initial['segments'] == [writer.paths[0].name]
    assert base(writer.paths[0]) == path
    writer.writeframes(bytes(962))
    during = json.loads(side.read_text())
    assert during['segments'] == [part.name for part in writer.paths]
    assert base(writer.paths[-1]) == path
    writer.close()
    final = json.loads(side.read_text())
    assert final['samples'] == 481 and final['segments'] == during['segments']


@pytest.mark.parametrize('reader', ['segments', 'read', 'read_pcm', 'pcm_frames', 'replay'])
@pytest.mark.parametrize('damaged_index', [0, 1])
def test_physical_truncation_never_shifts_later_segment_samples(tmp_path, monkeypatch, reader, damaged_index):
    monkeypatch.setattr(wav, 'SEGMENT_SAMPLES', 60 * 48000)
    path = tmp_path / 'stream.wav'
    writer = wav.SegmentedWriter(path, rate=8, closed=lambda _: None)
    writer.writeframes(np.arange(960, dtype='<i2').tobytes())
    writer.close()
    damaged = writer.paths[damaged_index]
    damaged.write_bytes(damaged.read_bytes()[:-4])
    with pytest.raises(ValueError, match='ended early'):
        if reader == 'pcm_frames':
            list(wav.pcm_frames(damaged, 127))
        elif reader == 'read_pcm':
            wav.read_pcm(damaged)
        elif reader == 'replay':
            list(WavReplaySource(path, realtime=False).frames())
        else:
            getattr(wav, reader)(path)


@pytest.mark.parametrize('reader', ['read', 'read_pcm', 'pcm_frames'])
def test_single_file_readers_reject_physical_truncation(tmp_path, reader):
    path = tmp_path / 'short.wav'
    with wave.open(str(path), 'wb') as out:
        out.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
        out.writeframes(bytes(16))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='ended early'):
        if reader == 'pcm_frames':
            list(wav.pcm_frames(path, 3))
        else:
            getattr(wav, reader)(path)


def test_float_recording_with_a_short_data_chunk_is_rejected(tmp_path):
    path = tmp_path / 'short.wav'
    wavfile.write(path, 48000, np.arange(4, dtype=np.float32))
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match='ended early'):
        wav.read(path)
