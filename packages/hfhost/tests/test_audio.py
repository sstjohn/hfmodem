# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import struct
import sys
import wave

import pytest

from hfhost import audio
from hfhost.audio import FS, FRAME_SAMPLES, WavReplaySource


def _write_wav(path, seconds, *, rate=FS, channels=1):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        # a quiet ramp, enough to be non-trivial without needing numpy
        frames = b"".join(
            struct.pack("<h", (i % 100 - 50) * 100) * channels
            for i in range(n))
        w.writeframes(frames)


def test_replay_yields_48k_mono_frames_on_a_common_clock(tmp_path):
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.5)
    src = WavReplaySource(wav, realtime=False)

    frames = list(src.frames())
    assert frames, "replay produced no audio"
    # timeline is contiguous in samples, first frame at 0
    assert frames[0][0] == 0
    total = 0
    for t0, pcm in frames:
        assert t0 == total
        assert len(pcm) % 2 == 0
        total += len(pcm) // 2
    assert total == int(0.5 * FS)
    # full frames are FRAME_SAMPLES; only the tail may be short
    assert all(len(p) == FRAME_SAMPLES * 2 for _, p in frames[:-1])


def test_replay_missing_file_is_a_clean_error(tmp_path):
    src = WavReplaySource(tmp_path / "nope.wav", realtime=False)
    with pytest.raises(audio.AudioError):
        list(src.frames())


@pytest.mark.realtime
def test_replay_realtime_paces_to_the_wall_clock(tmp_path):
    import time
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.3)
    src = WavReplaySource(wav, realtime=True, speed=10.0)   # 10x to keep it quick
    start = time.monotonic()
    list(src.frames())
    elapsed = time.monotonic() - start
    # ~0.03 s of paced replay; generously bounded, just proves it does pace
    assert 0.005 < elapsed < 0.5


def test_source_is_a_context_manager(tmp_path):
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.1)
    with WavReplaySource(wav, realtime=False) as src:
        assert list(src.frames())


# -- audio off another process -------------------------------------------------

def test_a_command_source_reads_s16le_off_its_stdout():
    """The station's capture runs in a process of its own — PortAudio needs a
    library hfhost may not import — so what this has to get right is the pipe,
    not the card."""
    n = 3 * FRAME_SAMPLES
    src = audio.CommandSource(
        [sys.executable, "-c",
         f"import sys; sys.stdout.buffer.write(bytes(2 * {n}))"])
    frames = list(src.frames())

    assert [t0 for t0, _ in frames] == [0, FRAME_SAMPLES, 2 * FRAME_SAMPLES]
    assert sum(len(p) for _, p in frames) == 2 * n


def test_a_command_that_produces_nothing_ends_the_stream_rather_than_hanging():
    src = audio.CommandSource([sys.executable, "-c", ""])
    assert list(src.frames()) == []
